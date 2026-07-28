from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import HadamardLayer, InputLayer, KroneckerLayer, SumLayer
from cirkit.symbolic.parameters import MixingWeightParameter, TensorParameter, mixing_weight_factory
from cirkit.symbolic.parameters import Parameter as CirkitParameter
from cirkit.templates import utils
from cirkit.templates.region_graph import QuadGraph, QuadTree

import extended_einsum.interface as xe
from demo.cirkit import input_shape_tuple, make_optimizer, torch_program_runner
from experiments.monarch.cirkit import (
    CompactMixingSumLayer,
    MonarchSumLayer,
    register_monarch_compilation,
    replace_cp_sum_layers,
)
from experiments.monarch.xe import transform as xe_monarch_transform
from extended_einsum.interface.tensor_expression import Parameter
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.preprocess import FoldSameShapedOperations, OptimizeContractionPaths

EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class CanonicalState:
    initialization_hash: str
    parameters: int


@dataclass
class TrainingSetup:
    step: Callable[[torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer
    initialization_hash: str
    parameters: int
    monarch_layers: int
    runtime_program: RichProgram | None


def resolve_factors(units: int, factors: tuple[int, int] | None) -> tuple[int, int]:
    if factors is not None:
        p, q = factors
        if p <= 0 or q <= 0 or p * q != units:
            raise ValueError(f"Monarch factors {factors} do not multiply to H={units}")
        return p, q
    p = math.isqrt(units)
    while p > 1 and units % p:
        p -= 1
    if p <= 1:
        raise ValueError(f"H={units} has no non-trivial Monarch factorization")
    return p, units // p


def build_symbolic_circuit(
    *,
    width: int,
    height: int,
    units: int,
    categories: int,
    region_graph: str,
    parameterization: str,
    factors: tuple[int, int] | None,
):
    if parameterization not in {"dense", "monarch"}:
        raise ValueError(f"unknown sum parameterization: {parameterization}")
    if region_graph == "quad-tree-2":
        region = QuadTree((1, height, width), num_patch_splits=2)
    elif region_graph == "quad-graph":
        region = QuadGraph((1, height, width))
    else:
        raise ValueError(f"unsupported region graph: {region_graph}")

    probability_factory = utils.parameterization_to_factory(
        utils.Parameterization(activation="softmax", initialization="normal")
    )
    input_factory = utils.name_to_input_layer_factory(
        "categorical",
        num_categories=categories,
        probs_factory=probability_factory,
    )
    circuit = region.build_circuit(
        input_factory=input_factory,
        sum_product="cp",
        sum_weight_factory=probability_factory,
        nary_sum_weight_factory=functools.partial(
            mixing_weight_factory,
            param_factory=probability_factory,
        ),
        num_input_units=units,
        num_sum_units=units,
        num_classes=1,
        factorize_multivariate=True,
    )
    if parameterization == "monarch":
        p, q = resolve_factors(units, factors)
        circuit = replace_cp_sum_layers(circuit, p=p, q=q)
    return circuit


def _raw_tensor_nodes(parameter: CirkitParameter) -> tuple[TensorParameter, ...]:
    return tuple(
        node
        for node in parameter.nodes
        if isinstance(node, TensorParameter) and node.learnable
    )


def canonicalize_parameters(symbolic_circuit, *, seed: int) -> CanonicalState:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    digest = hashlib.sha256()
    seen: set[int] = set()
    total = 0
    for layer_index, layer in enumerate(symbolic_circuit.layers):
        for parameter_name, parameter in sorted(layer.params.items()):
            for leaf_index, tensor_parameter in enumerate(_raw_tensor_nodes(parameter)):
                if id(tensor_parameter) in seen:
                    continue
                seen.add(id(tensor_parameter))
                values = torch.randn(
                    tensor_parameter.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                values_numpy = values.numpy().copy()
                tensor_parameter.initializer = ConstantTensorInitializer(values_numpy)
                digest.update(
                    (
                        f"{layer_index}:{type(layer).__name__}:"
                        f"{parameter_name}:{leaf_index}:{tensor_parameter.shape}"
                    ).encode()
                )
                digest.update(values_numpy.tobytes(order="C"))
                total += values.numel()
    return CanonicalState(digest.hexdigest(), total)


def _parameter_logits(
    parameter: CirkitParameter,
    *,
    output_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    raw_nodes = _raw_tensor_nodes(parameter)
    if len(raw_nodes) != 1:
        raise ValueError(
            f"expected exactly one trainable tensor, found {len(raw_nodes)}"
        )
    initializer = raw_nodes[0].initializer
    if not isinstance(initializer, ConstantTensorInitializer):
        raise ValueError("parameters must be canonicalized before XE lowering")
    value = torch.as_tensor(initializer.value, dtype=torch.float32).clone()
    return value.reshape(output_shape) if output_shape is not None else value


def _monarch_factors(symbolic_circuit) -> tuple[int, int] | None:
    layers = [
        layer
        for layer in symbolic_circuit.layers
        if isinstance(layer, MonarchSumLayer)
    ]
    if not layers:
        return None
    factors = {(layer.p, layer.q) for layer in layers}
    if len(factors) != 1:
        raise ValueError(f"mixed Monarch factors: {sorted(factors)}")
    return next(iter(factors))


def _scope_id(scope) -> int:
    if len(scope) != 1:
        raise ValueError(f"expected a singleton scope, found {scope!r}")
    return next(iter(scope))


def _symbols(count: int) -> str:
    if count > len(EINSUM_SYMBOLS):
        raise ValueError(f"cannot allocate {count} einsum symbols")
    return EINSUM_SYMBOLS[:count]


def _dense_sum_to_xe(layer: SumLayer, children):
    output_units = layer.num_output_units
    input_state_shape = tuple(children[0].shape[1:])
    if not all(tuple(child.shape[1:]) == input_state_shape for child in children):
        raise ValueError("sum-layer children have different state shapes")
    if math.prod(input_state_shape) != layer.num_input_units:
        raise ValueError(
            f"state shape {input_state_shape} does not match "
            f"{layer.num_input_units} input units"
        )

    is_mixing = any(
        isinstance(node, MixingWeightParameter)
        for node in layer.params["weight"].nodes
    )
    if is_mixing:
        stacked = xe.stack(children, axis=1)
        mixing_shape = (*input_state_shape, len(children))
        logits = Parameter(
            xe.array(
                _parameter_logits(
                    layer.params["weight"],
                    output_shape=mixing_shape,
                )
            )
        )
        weights = xe.softmax(logits, axis=len(mixing_shape) - 1)
        labels = _symbols(2 + len(input_state_shape))
        batch_label, arity_label = labels[:2]
        state_labels = labels[2:]
        return xe.einsum(
            (
                f"{batch_label}{arity_label}{state_labels},"
                f"{state_labels}{arity_label}->{batch_label}{state_labels}"
            ),
            stacked,
            weights,
        )

    output_state_shape = (
        input_state_shape
        if output_units == layer.num_input_units
        else (output_units,)
    )
    if len(children) == 1:
        data = children[0]
        data_labels = _symbols(1 + len(input_state_shape))
        output_labels = _symbols(
            1 + len(input_state_shape) + len(output_state_shape)
        )[1 + len(input_state_shape) :]
        weight_shape = (*output_state_shape, *input_state_shape)
        weight_labels = output_labels + data_labels[1:]
        result_labels = data_labels[0] + output_labels
    else:
        data = xe.stack(children, axis=1)
        labels = _symbols(2 + 2 * len(input_state_shape))
        batch_label, arity_label = labels[:2]
        input_labels = labels[2 : 2 + len(input_state_shape)]
        output_labels = labels[
            2 + len(input_state_shape) : 2 + 2 * len(input_state_shape)
        ]
        if output_state_shape == (output_units,):
            output_labels = output_labels[:1]
        data_labels = batch_label + arity_label + input_labels
        weight_shape = (*output_state_shape, len(children), *input_state_shape)
        weight_labels = output_labels + arity_label + input_labels
        result_labels = batch_label + output_labels

    logits = Parameter(
        xe.array(
            _parameter_logits(
                layer.params["weight"],
                output_shape=weight_shape,
            )
        )
    )
    input_axis_start = len(output_state_shape)
    input_axes = tuple(range(input_axis_start, len(weight_shape)))
    softmax_axis: int | tuple[int, ...] = (
        input_axes[0] if len(input_axes) == 1 else input_axes
    )
    weights = xe.softmax(logits, axis=softmax_axis)
    return xe.einsum(
        f"{data_labels},{weight_labels}->{result_labels}",
        data,
        weights,
    )


def to_xe_expression(
    symbolic_circuit,
    layer,
    data_by_scope,
    expression_by_layer=None,
):
    if expression_by_layer is None:
        expression_by_layer = {}
    if layer in expression_by_layer:
        return expression_by_layer[layer]
    children = [
        to_xe_expression(
            symbolic_circuit,
            child,
            data_by_scope,
            expression_by_layer,
        )
        for child in symbolic_circuit.layer_inputs(layer)
    ]

    if not children:
        result = xe.select(data_by_scope, _scope_id(layer.scope))
    elif isinstance(layer, HadamardLayer):
        labels = _symbols(len(children[0].shape))
        if not all(child.shape == children[0].shape for child in children):
            raise ValueError("Hadamard children have different shapes")
        result = xe.einsum(
            ",".join([labels] * len(children)) + "->" + labels,
            *children,
        )
    elif isinstance(layer, KroneckerLayer):
        labels = _symbols(len(children) + 1)
        result = xe.einsum(
            ",".join(f"a{label}" for label in labels[1:]) + "->" + labels,
            *children,
        )
    elif isinstance(layer, CompactMixingSumLayer):
        input_state_shape = tuple(children[0].shape[1:])
        stacked = xe.stack(children, axis=1)
        mixing_shape = (*input_state_shape, len(children))
        logits = Parameter(
            xe.array(
                _parameter_logits(
                    layer.mixing_logits,
                    output_shape=mixing_shape,
                )
            )
        )
        weights = xe.softmax(logits, axis=len(mixing_shape) - 1)
        labels = _symbols(2 + len(input_state_shape))
        batch_label, arity_label = labels[:2]
        state_labels = labels[2:]
        result = xe.einsum(
            (
                f"{batch_label}{arity_label}{state_labels},"
                f"{state_labels}{arity_label}->{batch_label}{state_labels}"
            ),
            stacked,
            weights,
        )
    elif isinstance(layer, MonarchSumLayer):
        if len(children) != 1:
            raise ValueError("Monarch layers require arity one")
        factor_a = Parameter(xe.array(_parameter_logits(layer.factor_a)))
        factor_b = Parameter(xe.array(_parameter_logits(layer.factor_b)))
        result = xe_monarch_transform(children[0], factor_a, factor_b)
    elif isinstance(layer, SumLayer):
        result = _dense_sum_to_xe(layer, children)
    else:
        raise NotImplementedError(f"unsupported Cirkit layer: {layer!r}")

    expression_by_layer[layer] = result
    return result


def translate_to_xe(symbolic_circuit, *, batch_size: int):
    input_layer = next(
        layer
        for layer in symbolic_circuit.layers
        if isinstance(layer, InputLayer)
    )
    state_shape = _monarch_factors(symbolic_circuit) or (
        input_layer.num_output_units,
    )
    data = xe.array(
        torch.empty(
            (symbolic_circuit.num_variables, batch_size, *state_shape),
            dtype=torch.float32,
        )
    )
    expression = to_xe_expression(
        symbolic_circuit,
        symbolic_circuit.layers[-1],
        data,
    )
    return xe.extract_program(xe.log(expression), stability_mode="logspace_max")


def _categorical_logits(
    symbolic_circuit,
    *,
    state_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    input_layers = sorted(
        (
            layer
            for layer in symbolic_circuit.layers
            if isinstance(layer, InputLayer)
        ),
        key=lambda layer: _scope_id(layer.scope),
    )
    logits = []
    for layer in input_layers:
        raw = _parameter_logits(layer.params["probs"])
        logits.append(
            raw.reshape(math.prod(state_shape), raw.shape[-1])
            .T.reshape(raw.shape[-1], *state_shape)
        )
    return torch.stack(logits).to(device)


def count_parameters(parameters) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def _metadata(
    symbolic_circuit,
    canonical: CanonicalState,
    parameters,
) -> tuple[int, int]:
    measured = count_parameters(parameters)
    if measured != canonical.parameters:
        raise RuntimeError(
            f"runtime exposes {measured} trainable scalars, "
            f"symbolic model has {canonical.parameters}"
        )
    monarch_layers = sum(
        isinstance(layer, MonarchSumLayer)
        for layer in symbolic_circuit.layers
    )
    return measured, monarch_layers


def setup_xe(
    *,
    width: int,
    height: int,
    units: int,
    categories: int,
    batch_size: int,
    region_graph: str,
    parameterization: str,
    factors: tuple[int, int] | None,
    seed: int,
    device: torch.device,
) -> TrainingSetup:
    symbolic = build_symbolic_circuit(
        width=width,
        height=height,
        units=units,
        categories=categories,
        region_graph=region_graph,
        parameterization=parameterization,
        factors=factors,
    )
    canonical = canonicalize_parameters(symbolic, seed=seed)
    program, inputs = translate_to_xe(symbolic, batch_size=batch_size)
    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=True,
    )
    runtime_program = OptimizeContractionPaths.apply(folded.program)
    gather_orders = folded.gather_index_orders
    index_input_ids = frozenset(
        range(
            runtime_program.n_inputs - len(gather_orders),
            runtime_program.n_inputs,
        )
    )
    run = torch_program_runner(
        runtime_program,
        device=device,
        index_input_ids=index_input_ids,
        shift_mode="xe",
    )

    variables = width * height
    state_shape = input_shape_tuple(inputs[0])[2:]
    input_order = folded.input_axis0_orders.get(0, tuple(range(variables)))
    input_order_tensor = torch.tensor(input_order, dtype=torch.long, device=device)
    categorical_logits = torch.nn.Parameter(
        _categorical_logits(
            symbolic,
            state_shape=state_shape,
            device=device,
        ).index_select(0, input_order_tensor)
    )

    used_input_ids = {
        argument
        for instruction in program.instructions
        for argument in instruction.argument_ssa_ids
        if argument < program.n_inputs
    }
    packed_sequence = tuple(
        input_id
        for stack_order in folded.parameter_stack_orders
        for input_id in stack_order
    )
    packed_ids = set(packed_sequence)
    if len(packed_sequence) != len(packed_ids):
        raise ValueError("an XE parameter appears in multiple folded stacks")
    retained_ids = tuple(
        input_id
        for input_id in range(program.n_inputs)
        if input_id in used_input_ids and input_id not in packed_ids
    )
    data_input_id = retained_ids.index(0)
    initialized = {
        input_id: inputs[input_id].backend_array.detach().clone().to(device)
        for input_id in sorted(program.parameter_indices)
    }
    parameters = {
        input_id: torch.nn.Parameter(tensor)
        for input_id, tensor in initialized.items()
        if input_id not in packed_ids
    }
    packed_parameters = [
        torch.nn.Parameter(
            torch.stack(
                [initialized[input_id] for input_id in stack_order],
                dim=0,
            )
        )
        for stack_order in folded.parameter_stack_orders
    ]
    gather_inputs = [
        torch.tensor(indices, dtype=torch.long, device=device)
        for indices in gather_orders
    ]
    constants = {
        input_id: inputs[input_id].backend_array.to(device)
        for input_id in range(program.n_inputs)
        if input_id != 0 and input_id not in program.parameter_indices
    }
    variable_range = torch.arange(variables, device=device)[:, None]

    def categorical_input(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.index_select(1, input_order_tensor)
        probabilities = F.softmax(categorical_logits, dim=1)
        return probabilities[variable_range, batch.T].contiguous()

    def original_input(input_id: int, data_input: torch.Tensor) -> torch.Tensor:
        if input_id == 0:
            return data_input
        if input_id in parameters:
            return parameters[input_id]
        return constants[input_id]

    def step(batch: torch.Tensor) -> torch.Tensor:
        data_input = categorical_input(batch)
        runtime_inputs = [
            original_input(input_id, data_input)
            for input_id in retained_ids
        ]
        runtime_inputs[data_input_id] = data_input
        runtime_inputs.extend(packed_parameters)
        runtime_inputs.extend(gather_inputs)
        if len(runtime_inputs) != runtime_program.n_inputs:
            raise RuntimeError(
                f"expected {runtime_program.n_inputs} XE inputs, "
                f"got {len(runtime_inputs)}"
            )
        return -run(runtime_inputs).mean()

    step = torch.compile(
        step,
        mode="reduce-overhead" if device.type == "cuda" else None,
    )
    trainable = (categorical_logits, *parameters.values(), *packed_parameters)
    optimizer = make_optimizer(trainable, device, 0.01)
    measured, monarch_layers = _metadata(symbolic, canonical, trainable)
    return TrainingSetup(
        step=step,
        optimizer=optimizer,
        initialization_hash=canonical.initialization_hash,
        parameters=measured,
        monarch_layers=monarch_layers,
        runtime_program=runtime_program,
    )


def setup_cirkit(
    *,
    width: int,
    height: int,
    units: int,
    categories: int,
    batch_size: int,
    region_graph: str,
    parameterization: str,
    factors: tuple[int, int] | None,
    seed: int,
    device: torch.device,
) -> TrainingSetup:
    symbolic = build_symbolic_circuit(
        width=width,
        height=height,
        units=units,
        categories=categories,
        region_graph=region_graph,
        parameterization=parameterization,
        factors=factors,
    )
    canonical = canonicalize_parameters(symbolic, seed=seed)
    context = PipelineContext(
        backend="torch",
        semiring="lse-sum",
        fold=True,
        optimize=True,
    )
    register_monarch_compilation(context)
    circuit = context.compile(symbolic).to(device)
    trainable = tuple(circuit.parameters())
    optimizer = make_optimizer(trainable, device, 0.01)

    def step(batch: torch.Tensor) -> torch.Tensor:
        return -circuit(batch).mean()

    step = torch.compile(
        step,
        mode="reduce-overhead" if device.type == "cuda" else None,
    )
    measured, monarch_layers = _metadata(symbolic, canonical, trainable)
    return TrainingSetup(
        step=step,
        optimizer=optimizer,
        initialization_hash=canonical.initialization_hash,
        parameters=measured,
        monarch_layers=monarch_layers,
        runtime_program=None,
    )
