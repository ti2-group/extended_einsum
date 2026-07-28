"""A native Cirkit implementation of an untied, two-factor Monarch sum layer.

This module intentionally does not use postponed evaluation of annotations. Cirkit 0.3
discovers custom compiler rules from their concrete runtime annotations.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchInnerLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer, Initializer, NormalInitializer
from cirkit.symbolic.layers import Layer, SumLayer
from cirkit.symbolic.parameters import MixingWeightParameter, Parameter, TensorParameter
from torch import Tensor


def make_monarch_parameter(
    shape: tuple[int, int, int],
    *,
    initializer: Initializer | None = None,
) -> Parameter:
    """Create one learnable raw-logit factor parameter.

    Normalization is deliberately performed by :class:`TorchMonarchSumLayer`, rather
    than encoded in this parameter graph. Consequently, the only trainable tensor in
    the graph is the canonical raw-logit tensor and its shape is preserved verbatim.
    """
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"Expected a positive rank-three factor shape, found {shape}")
    if initializer is None:
        initializer = NormalInitializer()
    return Parameter.from_input(TensorParameter(*shape, initializer=initializer))


def make_monarch_parameter_from_logits(logits: np.ndarray) -> Parameter:
    """Create a learnable factor initialized from canonical raw logits."""
    values = np.asarray(logits)
    if values.ndim != 3 or any(size <= 0 for size in values.shape):
        raise ValueError(f"Expected a positive rank-three raw-logit tensor, found shape {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    values = np.array(values, copy=True)
    shape = tuple(int(size) for size in values.shape)
    return make_monarch_parameter(
        shape,
        initializer=ConstantTensorInitializer(values),
    )


class MonarchSumLayer(SumLayer):
    """Symbolic arity-one Monarch sum layer with no dense weight parameter.

    Given ``H = p * q``, the input and output units are laid out as ``(i, j)``
    and ``(k, l)`` respectively. The raw factor logits have canonical layouts
    ``A[j, k, i]`` and ``B[k, l, j]``.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        p: int,
        q: int,
        factor_a: Parameter | None = None,
        factor_b: Parameter | None = None,
        initializer: Initializer | None = None,
    ) -> None:
        if p <= 0 or q <= 0:
            raise ValueError(f"Monarch factors must be positive, found p={p}, q={q}")
        if arity != 1:
            raise ValueError(f"MonarchSumLayer only supports arity one, found {arity}")
        num_units = p * q
        if num_input_units != num_units or num_output_units != num_units:
            raise ValueError(f"MonarchSumLayer requires num_input_units == num_output_units == p * q, found input={num_input_units}, output={num_output_units}, p*q={num_units}")

        # Bypass SumLayer.__init__: it would allocate a hidden H-by-H weight.
        Layer.__init__(self, num_input_units, num_output_units, arity=arity)
        self.p = p
        self.q = q

        factor_a_shape = (q, p, p)
        factor_b_shape = (p, q, q)
        if factor_a is None:
            factor_a = make_monarch_parameter(factor_a_shape, initializer=initializer)
        if factor_b is None:
            factor_b = make_monarch_parameter(factor_b_shape, initializer=initializer)
        if factor_a.shape != factor_a_shape:
            raise ValueError(f"Expected factor_a shape {factor_a_shape}, found {factor_a.shape}")
        if factor_b.shape != factor_b_shape:
            raise ValueError(f"Expected factor_b shape {factor_b_shape}, found {factor_b.shape}")
        self.factor_a = factor_a
        self.factor_b = factor_b

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "p": self.p,
            "q": self.q,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"factor_a": self.factor_a, "factor_b": self.factor_b}


class CompactMixingSumLayer(SumLayer):
    """Diagonal mixing over alternative partitions without a materialized matrix."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        mixing_logits: Parameter | None = None,
        initializer: Initializer | None = None,
    ) -> None:
        if arity <= 1:
            raise ValueError(f"CompactMixingSumLayer requires arity greater than one, found {arity}")
        if num_input_units != num_output_units:
            raise ValueError(f"CompactMixingSumLayer requires equal input/output widths, found input={num_input_units}, output={num_output_units}")
        Layer.__init__(self, num_input_units, num_output_units, arity=arity)
        if initializer is None:
            initializer = NormalInitializer()
        expected_shape = (num_output_units, arity)
        if mixing_logits is None:
            mixing_logits = Parameter.from_input(TensorParameter(*expected_shape, initializer=initializer))
        if mixing_logits.shape != expected_shape:
            raise ValueError(f"Expected mixing logits shape {expected_shape}, found {mixing_logits.shape}")
        self.mixing_logits = mixing_logits

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"mixing_logits": self.mixing_logits}


class TorchMonarchSumLayer(TorchInnerLayer):
    """Compiled Monarch layer using two semiring tensor contractions."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        p: int,
        q: int,
        factor_a: TorchParameter,
        factor_b: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ) -> None:
        if arity != 1:
            raise ValueError(f"TorchMonarchSumLayer only supports arity one, found {arity}")
        if p <= 0 or q <= 0:
            raise ValueError(f"Monarch factors must be positive, found p={p}, q={q}")
        num_units = p * q
        if num_input_units != num_units or num_output_units != num_units:
            raise ValueError(f"TorchMonarchSumLayer requires input and output units equal to p * q, found input={num_input_units}, output={num_output_units}, p*q={num_units}")
        super().__init__(
            num_input_units,
            num_output_units,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
        )
        self.p = p
        self.q = q
        self._validate_factor("factor_a", factor_a, (q, p, p))
        self._validate_factor("factor_b", factor_b, (p, q, q))
        self.factor_a = factor_a
        self.factor_b = factor_b

    def _validate_factor(
        self,
        name: str,
        factor: TorchParameter,
        expected_shape: tuple[int, int, int],
    ) -> None:
        if factor.num_folds != self.num_folds or factor.shape != expected_shape:
            raise ValueError(f"Expected {name} to have {self.num_folds} folds and shape {expected_shape}, found {factor.num_folds} folds and shape {factor.shape}")

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "p": self.p,
            "q": self.q,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"factor_a": self.factor_a, "factor_b": self.factor_b}

    def normalized_factors(self) -> tuple[Tensor, Tensor]:
        """Return the two last-axis-normalized factor tensors, including folds."""
        return (
            torch.softmax(self.factor_a(), dim=-1),
            torch.softmax(self.factor_b(), dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (fold, arity=1, batch, i*j) -> (fold, batch, j, i)
        x = x.squeeze(dim=1)
        x = x.reshape(x.shape[0], x.shape[1], self.p, self.q)
        x = x.permute(0, 1, 3, 2)
        factor_a, factor_b = self.normalized_factors()

        # u[f,b,j,k] = sum_i x[f,b,j,i] A[f,j,k,i]. Keeping the reduced
        # dimension is required by Cirkit's numerically stable LSE semiring.
        u = self.semiring.einsum(
            "fbji,fjki->fbjk",
            inputs=(x,),
            operands=(factor_a,),
            dim=-1,
            keepdim=True,
        )
        u = u.permute(0, 1, 3, 2)

        # y[f,b,k,l] = sum_j u[f,b,k,j] B[f,k,l,j].
        y = self.semiring.einsum(
            "fbkj,fklj->fbkl",
            inputs=(u,),
            operands=(factor_b,),
            dim=-1,
            keepdim=True,
        )
        return y.reshape(y.shape[0], y.shape[1], self.num_output_units)


class TorchCompactMixingSumLayer(TorchInnerLayer):
    """Compiled compact partition mixing with O(H * arity) storage and compute."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        mixing_logits: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ) -> None:
        if arity <= 1 or num_input_units != num_output_units:
            raise ValueError("TorchCompactMixingSumLayer requires arity > 1 and equal input/output widths")
        super().__init__(
            num_input_units,
            num_output_units,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
        )
        expected_shape = (num_output_units, arity)
        if mixing_logits.num_folds != num_folds or mixing_logits.shape != expected_shape:
            raise ValueError(f"Expected {num_folds} folds and mixing logits shape {expected_shape}, found {mixing_logits.num_folds} folds and {mixing_logits.shape}")
        self.mixing_logits = mixing_logits

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"mixing_logits": self.mixing_logits}

    def forward(self, x: Tensor) -> Tensor:
        # x: (fold, arity, batch, H), weights: (fold, H, arity)
        weights = torch.softmax(self.mixing_logits(), dim=-1)
        return self.semiring.einsum(
            "fabk,fka->fbk",
            inputs=(x,),
            operands=(weights,),
            dim=1,
            keepdim=False,
        )


def compile_monarch_sum_layer(
    compiler: TorchCompiler,
    layer: MonarchSumLayer,
) -> TorchMonarchSumLayer:
    """Compile a symbolic Monarch layer into its native Torch implementation."""
    return TorchMonarchSumLayer(
        layer.num_input_units,
        layer.num_output_units,
        arity=layer.arity,
        p=layer.p,
        q=layer.q,
        factor_a=compiler.compile_parameter(layer.factor_a),
        factor_b=compiler.compile_parameter(layer.factor_b),
        semiring=compiler.semiring,
    )


def compile_compact_mixing_sum_layer(
    compiler: TorchCompiler,
    layer: CompactMixingSumLayer,
) -> TorchCompactMixingSumLayer:
    return TorchCompactMixingSumLayer(
        layer.num_input_units,
        layer.num_output_units,
        arity=layer.arity,
        mixing_logits=compiler.compile_parameter(layer.mixing_logits),
        semiring=compiler.semiring,
    )


def register_monarch_compilation(context: PipelineContext) -> None:
    """Register the Monarch layer rule on a Cirkit pipeline context."""
    context.add_layer_compilation_rule(compile_monarch_sum_layer)
    context.add_layer_compilation_rule(compile_compact_mixing_sum_layer)


def replace_cp_sum_layers(circuit: Circuit, *, p: int, q: int) -> Circuit:
    """Replace eligible hidden dense CP sums by independently parameterized Monarch sums.

    Only exact ``SumLayer`` instances with arity one and ``p*q`` input/output units
    are replaced by Monarch layers. Compact multi-input mixing sums are also rewritten
    to avoid materializing Cirkit's effective diagonal ``H x (arity * H)`` matrix.
    Arity-one roots stay dense.
    """
    num_units = p * q
    output_layers = set(circuit.outputs)
    replacements: dict[Layer, Layer] = {}
    for layer in circuit.layers:
        if type(layer) is SumLayer and layer not in output_layers and layer.arity == 1 and layer.num_input_units == num_units and layer.num_output_units == num_units:
            replacements[layer] = MonarchSumLayer(
                num_units,
                num_units,
                p=p,
                q=q,
            )
        elif type(layer) is SumLayer and layer.arity > 1 and any(isinstance(node, MixingWeightParameter) for node in layer.params["weight"].nodes):
            replacements[layer] = CompactMixingSumLayer(
                layer.num_input_units,
                layer.num_output_units,
                arity=layer.arity,
            )

    def replaced(layer: Layer) -> Layer:
        return replacements.get(layer, layer)

    layers = [replaced(layer) for layer in circuit.layers]
    in_layers = {replaced(layer): [replaced(child) for child in circuit.layer_inputs(layer)] for layer in circuit.layers}
    outputs = [replaced(layer) for layer in circuit.outputs]
    return Circuit(layers, in_layers, outputs)


def materialize_monarch_matrix(
    factor_a_logits: Tensor,
    factor_b_logits: Tensor,
) -> Tensor:
    """Materialize the dense Monarch matrix for small correctness oracles only.

    Leading dimensions, such as Cirkit's fold dimension, are preserved. The result
    layout is ``(..., output=k*l, input=i*j)``.
    """
    if factor_a_logits.ndim < 3 or factor_b_logits.ndim < 3:
        raise ValueError("Monarch factor logits must have at least three dimensions")
    q, p_out, p_in = factor_a_logits.shape[-3:]
    p, q_out, q_in = factor_b_logits.shape[-3:]
    if p_out != p_in or p != p_out or q_out != q_in or q_out != q:
        raise ValueError(f"Expected factor shapes (..., q, p, p) and (..., p, q, q), found {tuple(factor_a_logits.shape)} and {tuple(factor_b_logits.shape)}")
    factor_a = torch.softmax(factor_a_logits, dim=-1)
    factor_b = torch.softmax(factor_b_logits, dim=-1)
    matrix = torch.einsum("...jki,...klj->...klij", factor_a, factor_b)
    return matrix.reshape(*matrix.shape[:-4], p * q, p * q)
