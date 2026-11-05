"""Input bindings, Cirkit input adapters and Torch references."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class CategoricalInput:
    """A virtual ``(fold, batch, units)`` log-valued input.

    ``pixels`` contains integer category IDs in ``(batch, variables)`` order.
    ``tables`` contains FP32 log probabilities in ``(banks, categories, units)``
    order, or unnormalized logits when ``logits=True``. Each folded row selects
    a pixel through ``variable_ids`` and a parameter bank through ``table_ids``
    (defaults to ``variable_ids``). Repeated and reordered IDs are supported.

    Triton reads evidence inside its consuming subtree. Custom circuits require
    normalized log-valued tables. ``materialize`` also supports logits and
    probability-valued output through Torch, including autograd.

    Inputs must be contiguous, on one device, and retain their shapes/dtypes.
    Category, variable and bank IDs must be in range. Compiled execution masks
    invalid addresses and returns NaNs for invalid evidence. Routing IDs and
    pixels may change between calls.
    """

    pixels: torch.Tensor
    tables: torch.Tensor
    variable_ids: torch.Tensor
    table_ids: torch.Tensor | None = None
    logits: bool = False
    log_values: bool = True

    def __post_init__(self):
        if self.table_ids is None:
            object.__setattr__(self, "table_ids", self.variable_ids)
        if self.pixels.ndim != 2 or self.tables.ndim != 3 or self.variable_ids.ndim != 1 or self.table_ids.shape != self.variable_ids.shape:
            raise ValueError("Categorical inputs require pixels[B,V], tables[P,C,U], and matching one-dimensional variable/table IDs.")
        if (
            self.tables.dtype != torch.float32
            or self.pixels.dtype not in (torch.uint8, torch.int32, torch.int64)
            or any(a.dtype not in (torch.int32, torch.int64) for a in (self.variable_ids, self.table_ids))
        ):
            raise ValueError("Categorical tables must be float32, pixels uint8/int32/int64, and routing IDs int32/int64.")
        if any(a.device != self.pixels.device or a.layout != torch.strided or not a.is_contiguous() for a in self.tensors):
            raise ValueError("Categorical tensors must be contiguous and on one device.")
        if any(a.numel() == 0 or a.numel() >= 2**31 for a in self.tensors):
            raise ValueError("Categorical tensors must be nonempty with fewer than 2**31 elements each.")

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.pixels, self.tables, self.variable_ids, self.table_ids

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.variable_ids.numel(), self.pixels.shape[0], self.tables.shape[2]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return self.tables.dtype

    @property
    def device(self):
        return self.pixels.device

    @property
    def layout(self):
        return torch.strided

    @property
    def requires_grad(self):
        return self.tables.requires_grad

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return all(a.is_contiguous() for a in self.tensors)

    def materialize(self, method: str = "advanced", *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Torch reference, including normalization when tables are logits."""
        table = self.tables.to(dtype=dtype) if dtype is not None else self.tables
        table = table.log_softmax(dim=1) if self.logits else table
        categories = self.pixels.index_select(1, self.variable_ids.long()).T.long()
        banks = self.table_ids.long()
        if method == "advanced":
            result = table[banks[:, None], categories].contiguous()
        elif method == "gather":
            selected = table.index_select(0, banks)
            result = torch.gather(selected, 1, categories[:, :, None].expand(-1, -1, table.shape[2]))
        elif method == "flattened":
            rows = (banks[:, None] * table.shape[1] + categories).reshape(-1)
            result = table.reshape(-1, table.shape[2]).index_select(0, rows).reshape(self.shape)
        else:
            raise ValueError("Categorical lookup must be advanced, gather, or flattened.")
        return result if self.log_values else result.exp()


@dataclass(frozen=True)
class InputSelection:
    backend: str
    reason: str
    options: dict


class CirkitInput:
    """Evaluate a Cirkit input from raw [batch,variable] data, with autograd.

    Parameters are read live from the layer on every call. Categorical logits
    are unnormalized log values, as in Cirkit. Constants/evidence broadcast
    rather than materializing repeated batch rows. Complex/unsupported inputs
    preserve Cirkit's Torch behavior. Compile again after moving the layer.
    """

    def __init__(self, layer, *, backend="auto", tile=None, num_warps=4, table_layout="native", backward_units=None, backward_categories=None, output_layout=None):
        from cirkit.backend.torch.layers.input import (
            TorchBinomialLayer,
            TorchCategoricalLayer,
            TorchConstantValueLayer,
            TorchEmbeddingLayer,
            TorchEvidenceLayer,
            TorchGaussianLayer,
            TorchPolynomialLayer,
        )

        if backend not in ("auto", "torch", "triton") or table_layout not in ("native", "units") or output_layout not in (None, "units", "batch"):
            raise ValueError("Unsupported input backend/layout")
        if tile is not None and (len(tile) != 2 or any(n < 1 or n & (n - 1) for n in tile)):
            raise ValueError("Input tiles must be two positive powers of two")
        if num_warps not in (1, 2, 4, 8, 16) or any(n is not None and (n < 16 or n & (n - 1)) for n in (backward_units, backward_categories)):
            raise ValueError("Invalid input warp count or backward tile")
        self.layer, self.backend, self.tile, self.warps = layer, backend, tile, num_warps
        self.layout, self.backward_units, self.backward_categories = table_layout, backward_units, backward_categories
        self.output_layout = output_layout
        self.f, self.u = layer.num_folds, layer.num_output_units
        self.inner = None
        if isinstance(layer, TorchEvidenceLayer):
            self.kind = "evidence"
            self.inner = CirkitInput(
                layer.layer,
                backend=backend,
                tile=tile,
                num_warps=num_warps,
                table_layout=table_layout,
                backward_units=backward_units,
                backward_categories=backward_categories,
                output_layout=output_layout,
            )
            self.evidence_routes = torch.arange(self.f, device=layer.observation().device)
        elif isinstance(layer, TorchConstantValueLayer):
            self.kind = "constant"
        elif isinstance(layer, TorchCategoricalLayer):
            self.kind = "categorical_logits" if layer.logits is not None else "categorical_probs"
        elif isinstance(layer, TorchEmbeddingLayer):
            self.kind = "embedding"
        elif isinstance(layer, TorchGaussianLayer):
            self.kind = "gaussian"
        elif isinstance(layer, TorchBinomialLayer):
            self.kind = "binomial_logits" if layer.logits is not None else "binomial_probs"
        elif isinstance(layer, TorchPolynomialLayer):
            self.kind = "polynomial"
        else:
            self.kind = "unsupported"
        self.output_layout = output_layout or ("units" if self.kind.startswith("binomial") else "batch")
        self.log = layer.semiring.__name__ == "LSESumSemiring"
        self.real_semiring = layer.semiring.__name__ in ("LSESumSemiring", "SumProductSemiring")
        # Cirkit's final parameter address-book gather may copy the entire
        # table even when its fold order is the identity. Read the live leaf
        # directly in that case. Any changed routing falls back to Cirkit.
        from cirkit.backend.torch.parameters.nodes import TorchTensorParameter

        self._direct_parameters = {}
        for name, parameter in layer.params.items():
            nodes = tuple(parameter.nodes)
            if len(nodes) == 1 and isinstance(nodes[0], TorchTensorParameter) and nodes[0].num_folds == parameter.num_folds:
                entry = tuple(parameter.address_book)[-1]
                indices = entry.in_fold_idx[0]
                if not indices.is_inference() and torch.equal(indices, torch.arange(parameter.num_folds, dtype=indices.dtype, device=indices.device)):
                    self._direct_parameters[name] = (parameter, nodes[0], indices, indices._version, parameter.address_book._entry_in_fold_idx_targets[-1][0])
        self.selection = InputSelection("torch", "Not yet evaluated", {})

    def reference(self, data, routes=None):
        if self.kind in ("constant", "evidence"):
            return self.layer(data if isinstance(data, int) else data.shape[0])
        scope = self.layer.scope_idx if routes is None else routes[:, None]
        folded = data[..., scope].permute(1, 0, 2)
        return self.layer(folded)

    def __call__(self, data):
        if self.kind == "constant":
            self.selection = InputSelection("torch-view", "Broadcast constant without batch-sized writes", {})
            value = self._read("value")
            value = self.layer.semiring.map_from(value, self.layer._source_semiring)
            batch = data if isinstance(data, int) else data.shape[0]
            return value.unsqueeze(1).expand(-1, batch, -1)
        if self.kind == "evidence":
            batch = data if isinstance(data, int) else data.shape[0]
            if self.inner.kind in ("polynomial", "unsupported", "constant", "evidence"):
                self.selection = InputSelection("torch", "Preserve wrapped Cirkit input behavior", {})
                return self.reference(data)
            observation = self._read("observation")
            y = self.inner._call(observation.T.contiguous(), self.evidence_routes)
            self.selection = self.inner.selection
            return y.expand(-1, batch, -1)
        return self._call(data, self.layer.scope_idx[:, 0].contiguous())

    def _read(self, name):
        parameter = self.layer.params[name]
        plan = self._direct_parameters.get(name)
        if plan is not None and parameter is plan[0] and getattr(parameter.address_book, plan[4], None) is plan[2] and plan[2]._version == plan[3]:
            return plan[1]()
        return parameter()

    def _parameters(self):
        if self.kind == "gaussian":
            return self._read("mean"), self._read("stddev"), self._read("log_partition") if "log_partition" in self.layer.params else None
        if self.kind in ("categorical_logits", "binomial_logits"):
            return self._read("logits"), None, None
        if self.kind in ("categorical_probs", "binomial_probs"):
            return self._read("probs"), None, None
        return self._read("weight" if self.kind == "embedding" else "coeff"), None, None

    def _call(self, data, routes):
        if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError("Expected nonempty raw data [batch,variables]")
        if self.kind == "unsupported":
            self.selection = InputSelection("torch", "Unsupported Cirkit input type", {})
            return self.reference(data, routes)
        if self.backend == "torch" or (self.backend == "auto" and (not self.real_semiring or data.device.type != "cuda")):
            self.selection = InputSelection("torch", "Cirkit reference for this dtype/device/semiring", {})
            return self.reference(data, routes)
        params = self._parameters()
        supported = (
            self.real_semiring
            and data.device.type == "cuda"
            and torch.cuda.get_device_capability(data.device)[0] >= 8
            and (not torch.is_grad_enabled() or data.shape[0] <= 512)
            and data.dtype in (torch.float32, torch.int64, torch.int32, torch.uint8)
            and all(p is None or (p.dtype == torch.float32 and p.device == data.device and p.numel() < 2**31 and all(s >= 0 for s in p.stride())) for p in params)
            and routes.device == data.device
            and self.f * data.shape[0] * self.u < 2**31
            and data.numel() < 2**31
            and not (self.kind == "polynomial" and data.shape[0] == 1)
        )
        if self.backend == "torch" or not supported:
            if self.backend == "triton" and not supported:
                raise ValueError("Custom inputs require real FP32 CUDA parameters and a supported batch shape")
            self.selection = InputSelection("torch", "Cirkit reference for this dtype/device/semiring", {})
            return self.reference(data, routes)
        from importlib.util import find_spec

        if not find_spec("triton"):
            if self.backend == "triton":
                raise ModuleNotFoundError("Install extended-einsum[cuda] for input kernels")
            return self.reference(data, routes)
        data = data.contiguous()
        if self.kind in ("categorical_logits", "categorical_probs", "embedding", "binomial_logits", "binomial_probs"):
            data = data.detach()
        p0, p1, p2 = params
        if self.layout == "units" and p0.ndim == 3:
            p0 = p0.transpose(1, 2).contiguous().transpose(1, 2)
        if data.shape[0] == 1:
            default_tile = (1, 32 if self.u <= 32 else 128)
        elif self.kind in ("categorical_logits", "categorical_probs", "embedding"):
            default_tile = (8, 32 if self.u <= 32 else 64)
        elif self.kind == "binomial_logits":
            default_tile = (16, 32) if self.u <= 32 else (8, 64)
        elif self.kind == "binomial_probs":
            default_tile = (32, 32) if self.u <= 32 else (16, 64)
        else:
            default_tile = (16, 32 if self.u <= 32 else 64)
        if self.kind in ("categorical_logits", "categorical_probs", "embedding"):
            backward_n = 32 if self.u <= 32 else 64
            backward_c = 64 if self.u <= 32 else 32
        else:
            backward_c = 32
            if (
                self.kind in ("binomial_probs", "polynomial")
                or (self.kind == "binomial_logits" and self.u <= 32)
                or (self.kind == "gaussian" and "log_partition" in self.layer.params and self.u <= 32)
            ):
                backward_n = 16
            else:
                backward_n = 64 if data.shape[0] == 1 and self.u >= 128 else 32
        options = dict(
            tile=self.tile or default_tile,
            num_warps=self.warps,
            table_layout=self.layout,
            output_layout=self.output_layout,
            backward_units=self.backward_units or backward_n,
            backward_categories=self.backward_categories or backward_c,
        )
        self.selection = InputSelection("triton", "Fused raw-variable selection and input evaluation", options)
        return _InputAutograd.apply(self, data, routes, p0, p1, p2)

    def _kwargs(self, data, p0, p1, p2):
        return dict(
            B=data.shape[0],
            V=data.shape[1],
            U=self.u,
            C=p0.shape[2] if p0.ndim == 3 else 1,
            KIND=self.kind,
            LOG=self.log,
            S0=p0.stride(0),
            S1=p0.stride(1),
            S2=p0.stride(2) if p0.ndim == 3 else 0,
            T0=p1.stride(0) if p1 is not None else 0,
            T1=p1.stride(1) if p1 is not None else 0,
            R0=p2.stride(0) if p2 is not None else 0,
            R1=p2.stride(1) if p2 is not None else 0,
            COUNT=getattr(self.layer, "total_count", 0),
            DEGREE=getattr(self.layer, "degree", 0),
            PART=p2 is not None,
        )


class _InputAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, program, data, routes, p0, p1, p2):
        import triton

        from extended_einsum.backends.triton_inputs import _input_forward

        opts = program._kwargs(data, p0, p1, p2)
        m, n = program.selection.options["tile"]
        result = (
            torch.empty((program.f, program.u, data.shape[0]), device=data.device, dtype=torch.float32).transpose(1, 2)
            if program.output_layout == "batch"
            else torch.empty((program.f, data.shape[0], program.u), device=data.device, dtype=torch.float32)
        )
        with torch.cuda.device(data.device):
            _input_forward[(triton.cdiv(program.u, n), triton.cdiv(data.shape[0], m), program.f)](
                data, routes, p0, p1, p2, result, **opts, M=m, N=n, BATCH_FAST=program.output_layout == "batch", num_warps=program.warps
            )
        ctx.program, ctx.opts = program, opts
        ctx.backward_tiles = (program.selection.options["backward_units"], program.selection.options["backward_categories"])
        ctx.save_for_backward(data, routes, p0, p1, p2, result)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        import triton

        from extended_einsum.backends.triton_inputs import _input_backward, _lookup_backward

        data, routes, p0, p1, p2, result = ctx.saved_tensors
        program, opts = ctx.program, ctx.opts
        strides = dict(GS0=grad.stride(0), GS1=grad.stride(1), GS2=grad.stride(2))
        grads = [torch.empty(p.shape, dtype=p.dtype, device=p.device) if p is not None and needed else None for p, needed in zip((p0, p1, p2), ctx.needs_input_grad[3:], strict=True)]
        want_x = ctx.needs_input_grad[1] and program.kind in ("gaussian", "polynomial")
        gx = torch.zeros_like(data) if want_x else None
        n, nc = ctx.backward_tiles
        with torch.cuda.device(data.device):
            if program.kind in ("categorical_logits", "categorical_probs", "embedding"):
                _lookup_backward[(triton.cdiv(program.u, n), triton.cdiv(opts["C"], nc), program.f)](
                    data, routes, p0, grad, grads[0], **{k: v for k, v in opts.items() if k not in ("T0", "T1", "R0", "R1", "COUNT", "DEGREE", "PART")}, **strides, NC=nc, N=n, K=32, num_warps=4
                )
            else:
                _input_backward[(triton.cdiv(program.u, n), program.f)](
                    data,
                    routes,
                    p0,
                    p1,
                    p2,
                    result,
                    grad,
                    *grads,
                    gx,
                    **opts,
                    **strides,
                    OS0=result.stride(0),
                    OS1=result.stride(1),
                    OS2=result.stride(2),
                    WANT0=grads[0] is not None,
                    WANT1=grads[1] is not None,
                    WANT2=grads[2] is not None,
                    WANTX=want_x,
                    M=triton.next_power_of_2(data.shape[0]),
                    N=n,
                    num_warps=8,
                )
        return None, gx, None, *grads


def compile_input(layer, **options):
    """Compile a Cirkit Torch input layer; accepts raw [batch,variable] data.

    Unlike CategoricalInput(logits=True).materialize(), Cirkit categorical
    logits are unnormalized and are deliberately not passed through softmax.
    Parameter computation remains owned by the original Cirkit layer.
    """
    return CirkitInput(layer, **options)
