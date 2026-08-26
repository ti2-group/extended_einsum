from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import csv
import gc
import itertools
import random
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_THIS_DIR):
    sys.path.pop(0)
_REPOSITORY_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import torch
import torch.nn.functional as F
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.layers import HadamardLayer, InputLayer, KroneckerLayer, SumLayer
from cirkit.templates import data_modalities, utils
from torch import optim
from torch.fx.experimental.proxy_tensor import make_fx

import extended_einsum.interface as xe
from extended_einsum.backend_translation import run_program, translate_to_backend_program
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.interface.tensor_expression import Parameter
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import StabilityMode
from extended_einsum.preprocess import (
    AnnotateShortSameIndexContractions,
    FoldSameShapedOperations,
    OptimizeContractionPaths,
)

WIDTH = 4
HEIGHT = 4
DEFAULT_UNITS = 64
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 1
DEFAULT_NUM_SAMPLES = 0
DEFAULT_PIXEL_VALUES = 256
EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CP_FLATTENED_LOOKUP_MIN_UNITS = 128
CSV_FIELDS = (
    "timestamp",
    "backend",
    "status",
    "error",
    "device",
    "device_name",
    "width",
    "height",
    "variables",
    "pixel_values",
    "units",
    "batch_size",
    "epoch",
    "epochs",
    "max_batches",
    "num_samples",
    "dataset",
    "region_graph",
    "sum_product_layer",
    "semiring",
    "backend_type",
    "torch_compile",
    "compile_scope",
    "fold_strategy",
    "gather_materialization",
    "optimizer",
    "optimizer_fused",
    "seed",
    "warmup_steps",
    "warmup_epochs",
    "setup_ms",
    "warmup_ms",
    "epoch_total_ms",
    "data_loading_ms",
    "forward_loss_ms",
    "backward_ms",
    "optimizer_step_ms",
    "zero_grad_ms",
    "metrics_logging_ms",
    "batches",
    "samples",
    "avg_nll",
    "samples_per_sec",
    "batches_per_sec",
    "avg_batch_ms",
    "memory_backend",
    "memory_scope",
    "baseline_allocated_memory_bytes",
    "baseline_reserved_memory_bytes",
    "peak_allocated_memory_bytes",
    "peak_allocated_memory_mib",
    "peak_incremental_allocated_memory_bytes",
    "peak_incremental_reserved_memory_bytes",
    "peak_memory_bytes",
    "peak_memory_mib",
    "peak_reserved_memory_bytes",
    "peak_reserved_memory_mib",
    "reserved_memory_bytes",
    "allocated_memory_bytes",
)


def resolve_categorical_lookup(
    categorical_lookup: str,
    *,
    sum_product_layer: str,
    num_units: int,
    region_graph: str,
    semiring: str,
) -> str:
    """Choose the compiler-friendly categorical lookup for a circuit shape."""

    choices = {"advanced", "auto", "gather", "flattened"}
    if categorical_lookup not in choices:
        raise ValueError(f"Unknown categorical lookup: {categorical_lookup}")
    if categorical_lookup != "auto":
        return categorical_lookup
    if (
        semiring == "scaled-max"
        and region_graph == "quad-graph"
        and sum_product_layer == "cp"
        and num_units == 64
    ):
        return "flattened"
    if sum_product_layer in {"cp-t", "tucker"}:
        return "gather"
    if sum_product_layer == "cp" and num_units >= CP_FLATTENED_LOOKUP_MIN_UNITS:
        return "flattened"
    return "advanced"


def make_symbolic_circuit(
    *,
    width: int,
    height: int,
    num_units: int,
    sum_product_layer: str,
    region_graph: str = "quad-tree-2",
):
    return data_modalities.image_data(
        (1, width, height),
        region_graph=region_graph,
        input_layer="categorical",
        num_input_units=num_units,
        sum_product_layer=sum_product_layer,
        num_sum_units=num_units,
        sum_weight_param=utils.Parameterization(
            activation="softmax",
            initialization="normal",
        ),
    )


def get_scope_id(scope: utils.Scope) -> int:
    if len(scope) != 1:
        raise ValueError(f"Expected a singleton scope, got {scope!r}")
    return next(iter(scope))


def generate_symbols(count: int) -> str:
    if count > len(EINSUM_SYMBOLS):
        raise ValueError(f"Cannot generate {count} unique einsum symbols")
    return EINSUM_SYMBOLS[:count]


def to_xe_expression(symbolic_circuit, layer, data_by_scope, expression_by_layer=None):
    if expression_by_layer is None:
        expression_by_layer = {}
    if layer in expression_by_layer:
        return expression_by_layer[layer]

    children = symbolic_circuit.layer_inputs(layer)
    child_nodes = [
        to_xe_expression(
            symbolic_circuit,
            child,
            data_by_scope,
            expression_by_layer,
        )
        for child in children
    ]

    if not children:
        scope_id = get_scope_id(layer.scope)
        result = xe.select(data_by_scope, scope_id)
    elif isinstance(layer, HadamardLayer):
        format_string = ",".join(["ab"] * len(child_nodes)) + "->ab"
        if not all(child.shape == child_nodes[0].shape for child in child_nodes):
            raise ValueError("Hadamard layer children must have the same shape")
        result = xe.einsum(format_string, *child_nodes)
    elif isinstance(layer, KroneckerLayer):
        child_indices = generate_symbols(len(child_nodes) + 1)
        batched_child_indices = [f"a{symbol}" for symbol in child_indices[1:]]
        format_string = ",".join(batched_child_indices) + "->" + "".join(child_indices)
        result = xe.einsum(format_string, *child_nodes)
    elif isinstance(layer, SumLayer):
        output_units = layer.params["weight"].shape[0]
        if len(children) > 1:
            if layer.num_input_units != output_units or not all(child.shape == child_nodes[0].shape for child in child_nodes):
                raise ValueError("Multi-input sum layers must be mixing layers over equally shaped children")
            stacked_children = xe.stack(child_nodes, axis=1)
            mixing_logits = Parameter(xe.array(torch.empty((output_units, len(children)), dtype=torch.float32)))
            mixing_weights = xe.softmax(mixing_logits, axis=1)
            result = xe.einsum("bhu,uh->bu", stacked_children, mixing_weights)
        else:
            child = child_nodes[0]
            child_indices = generate_symbols(len(child.shape))
            weight_shape = (output_units, *child.shape[1:])
            output_unit_index = generate_symbols(len(child.shape) + 1)[-1]
            weight_indices = output_unit_index + child_indices[1:]
            out_indices = child_indices[0] + output_unit_index
            format_string = f"{child_indices},{weight_indices}->{out_indices}"
            weight_logits = Parameter(xe.array(torch.empty(weight_shape, dtype=torch.float32)))
            weight_input_axes = tuple(range(1, len(weight_shape)))
            softmax_axis: int | tuple[int, ...] = weight_input_axes[0] if len(weight_input_axes) == 1 else weight_input_axes
            weights = xe.softmax(weight_logits, axis=softmax_axis)
            result = xe.einsum(format_string, child, weights)
    else:
        raise NotImplementedError(f"Unsupported Cirkit layer: {layer!r}")

    expression_by_layer[layer] = result
    return result


def translate_cirkit_to_xe(
    symbolic_circuit,
    *,
    batch_size: int,
    stability: StabilityMode,
) -> tuple[RichProgram, list[object]]:
    input_layer = next(layer for layer in symbolic_circuit.layers if isinstance(layer, InputLayer))
    data_by_scope = xe.array(
        torch.empty(
            (symbolic_circuit.num_variables, batch_size, input_layer.params["probs"].shape[0]),
            dtype=torch.float32,
        )
    )

    expression = to_xe_expression(
        symbolic_circuit,
        symbolic_circuit.layers[-1],
        data_by_scope,
    )
    expression = xe.log(expression)
    return xe.extract_program(expression, stability_mode=stability)


def preprocess_xe_program(
    program: RichProgram,
    *,
    optimize_stacking: bool,
) -> RichProgram:
    if not optimize_stacking:
        return program

    # Paper compiler pipeline (sec:compiler-optimizations): first apply
    # "IR-Level Folding" plus the consumer ordering of sec:memory-layout, then
    # merge nested einsums, choose pairwise paths (sec:contraction-path), and
    # mark short weighted reductions for backend fusion.
    folded = FoldSameShapedOperations.apply(program)
    path_optimized = OptimizeContractionPaths.apply(folded)
    return AnnotateShortSameIndexContractions.apply(path_optimized)


def input_shape(value: object) -> tuple[int, ...] | str:
    if isinstance(value, tuple) and all(isinstance(dimension, int) for dimension in value):
        return value
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return tuple(int(dimension) for dimension in shape)


def format_input_shapes(inputs, limit: int) -> str:
    shapes = [input_shape(value) for value in inputs]
    preview = ", ".join(str(shape) for shape in shapes[:limit])
    if len(shapes) > limit:
        preview = f"{preview}, ... (+{len(shapes) - limit} more)"
    return f"[{preview}]"


def print_program_summary(name: str, program: RichProgram, inputs: Sequence[object], *, shape_preview: int) -> None:
    op_counts = Counter(instruction.operator.name for instruction in program.instructions)
    input_shapes = inputs if len(inputs) == program.n_inputs else program.shapes[: program.n_inputs]
    print(f"{name}:")
    print(f"  inputs:       {program.n_inputs}")
    print(f"  instructions: {len(program.instructions)}")
    print(f"  output_ssa:   {program.output_ssa}")
    stability = getattr(program, "stability_mode", None)
    if stability is not None:
        print(f"  stability:    {stability}")
    print(f"  input_shapes: {format_input_shapes(input_shapes, shape_preview)}")
    print(f"  operators:    {dict(sorted(op_counts.items()))}")


def print_instructions(program: RichProgram, limit: int) -> None:
    if limit <= 0:
        return
    print(f"\nfirst {min(limit, len(program.instructions))} instruction(s):")
    for instruction in program.instructions[:limit]:
        print(f"  {instruction}")


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_backends(value: str) -> tuple[str, ...]:
    backends = tuple(part.strip() for part in value.split(",") if part.strip())
    if not backends:
        raise argparse.ArgumentTypeError("expected at least one backend")
    unknown = set(backends) - {"xe", "cirkit"}
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown backend(s): {sorted(unknown)}")
    return backends


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "mps"
    return "cpu"


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def start_timer(device: torch.device):
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def stop_timer(device: torch.device):
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def elapsed_timer_ms(start, end, device: torch.device) -> float:
    if device.type == "cuda":
        return start.elapsed_time(end)
    return (end - start) * 1000.0


def elapsed_wall_ms(start_time: float, device: torch.device) -> float:
    synchronize_device(device)
    return (time.perf_counter() - start_time) * 1000.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: torch.device) -> dict[str, int | float | str]:
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        return {
            "memory_backend": "cuda",
            "peak_allocated_memory_bytes": peak_allocated,
            "peak_allocated_memory_mib": peak_allocated / 1024**2,
            "peak_memory_bytes": peak_allocated,
            "peak_memory_mib": peak_allocated / 1024**2,
            "peak_reserved_memory_bytes": peak_reserved,
            "peak_reserved_memory_mib": peak_reserved / 1024**2,
            "reserved_memory_bytes": torch.cuda.memory_reserved(device),
            "allocated_memory_bytes": torch.cuda.memory_allocated(device),
        }
    if device.type == "mps" and hasattr(torch, "mps"):
        allocated = torch.mps.current_allocated_memory()
        return {
            "memory_backend": "mps_current",
            "peak_memory_bytes": "",
            "peak_memory_mib": "",
            "peak_reserved_memory_bytes": "",
            "peak_reserved_memory_mib": "",
            "reserved_memory_bytes": torch.mps.driver_allocated_memory(),
            "allocated_memory_bytes": allocated,
        }
    return {
        "memory_backend": "unavailable",
        "peak_memory_bytes": "",
        "peak_memory_mib": "",
        "peak_reserved_memory_bytes": "",
        "peak_reserved_memory_mib": "",
        "reserved_memory_bytes": "",
        "allocated_memory_bytes": "",
    }


def cleanup_device(device: torch.device) -> None:
    # Compiled graphs can own CUDA graph pools. Release those references before
    # collecting Python objects and emptying the allocator cache so the next
    # benchmark configuration starts from a clean, measurable baseline.
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_optimizer(params, device: torch.device, lr: float):
    kwargs = {"lr": lr}
    if device.type == "cuda":
        kwargs["fused"] = True
    return optim.Adam(params, **kwargs)


def load_train_images(
    *,
    dataset: str,
    device: torch.device,
    data_dir: str,
    num_samples: int,
    num_variables: int,
    pixel_values: int,
    download: bool = True,
) -> torch.Tensor:
    if dataset == "synthetic":
        return torch.randint(pixel_values, size=(num_samples, num_variables), device=device)

    if dataset != "mnist":
        raise ValueError(f"Unsupported dataset: {dataset}")
    if num_variables != 28 * 28:
        raise ValueError("MNIST training requires --width 28 --height 28")

    from torchvision import datasets

    mnist_train = datasets.MNIST(data_dir, train=True, download=download)
    train_images = mnist_train.data.reshape(-1, num_variables).long()
    if num_samples:
        train_images = train_images[:num_samples]
    return train_images.to(device)


def torch_program_runner(
    program: RichProgram,
    *,
    device: torch.device,
    index_input_ids: frozenset[int] = frozenset(),
    shift_mode: str = "xe",
    scale_interval: int = 3,
) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
    class DifferentiableShiftTorchBackendFunctions(TorchBackendFunctions):
        @staticmethod
        def stop_gradient(array: torch.Tensor) -> torch.Tensor:
            # Paper ablation "What drives the gains?": this deliberately
            # disables "Detached reference shifts" while leaving the forward
            # log-space/scaled formulas unchanged.
            return array

    if shift_mode not in {"differentiable", "xe"}:
        raise ValueError(f"Unknown log-space shift mode: {shift_mode}")
    backend_functions = {
        "differentiable": DifferentiableShiftTorchBackendFunctions,
        "xe": TorchBackendFunctions,
    }[shift_mode]()
    backend_program = translate_to_backend_program(
        program,
        backend_functions,
        scale_interval=scale_interval,
    )

    def run_flat(*inputs: torch.Tensor) -> torch.Tensor:
        return run_program(backend_program, inputs)

    example_inputs = [
        torch.empty(
            tuple(program.shapes[input_id]),
            dtype=torch.long if input_id in index_input_ids else torch.float32,
            device=device,
        )
        for input_id in range(program.n_inputs)
    ]
    graph_module = make_fx(run_flat, tracing_mode="fake")(*example_inputs)

    def run_graph(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return graph_module(*inputs)

    return run_graph


def setup_xe_training(
    *,
    width: int,
    height: int,
    num_units: int,
    batch_size: int,
    sum_product_layer: str,
    region_graph: str,
    device: torch.device,
    dataset: str,
    data_dir: str,
    num_samples: int,
    pixel_values: int,
    semiring: str,
    lr: float,
    folding_depth: str = "input",
    folding_split_by_routing: bool = False,
    folding_force_gathers: bool = False,
    categorical_lookup: str = "auto",
    optimize_group_order: bool = True,
    preorder_inputs: bool = True,
    shift_mode: str = "xe",
    optimize_contraction_paths: bool = True,
    scale_interval: int = 3,
    download: bool = True,
):
    symbolic_circuit = make_symbolic_circuit(
        width=width,
        height=height,
        num_units=num_units,
        sum_product_layer=sum_product_layer,
        region_graph=region_graph,
    )
    program, inputs = translate_cirkit_to_xe(
        symbolic_circuit,
        batch_size=batch_size,
        stability={
            "lse-sum": "logspace_max",
            "scaled-min": "scaled_min",
            "scaled-max": "scaled_max",
            "scaled-sum": "scaled_sum",
        }[semiring],
    )
    # Metadata records one-time parameter/input packing and any unavoidable
    # gather indices used by the rewritten program.
    if folding_depth == "input":
        folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
            program,
            optimize_group_order=optimize_group_order,
            split_by_routing=folding_split_by_routing,
            _gather_fragmented_batches=(
                True if folding_force_gathers else None
            ),
        )
    elif folding_depth == "output":
        folded = FoldSameShapedOperations.apply_with_metadata(
            program,
            optimize_group_order=optimize_group_order,
        )
    else:
        raise ValueError(f"Unknown folding depth: {folding_depth}")
    resolved_categorical_lookup = resolve_categorical_lookup(
        categorical_lookup,
        sum_product_layer=sum_product_layer,
        num_units=num_units,
        region_graph=region_graph,
        semiring=semiring,
    )
    path_optimized_program = (
        # Paper sec:contraction-path: merge nested einsums and lower them to the
        # optimizer-selected pairwise schedule.
        OptimizeContractionPaths.apply(folded.program) if optimize_contraction_paths else folded.program
    )
    runtime_program = AnnotateShortSameIndexContractions.apply(
        path_optimized_program
    )
    gather_index_orders = folded.gather_index_orders
    index_input_ids = frozenset(range(runtime_program.n_inputs - len(gather_index_orders), runtime_program.n_inputs))
    run = torch_program_runner(
        runtime_program,
        device=device,
        index_input_ids=index_input_ids,
        shift_mode=shift_mode,
        scale_interval=scale_interval,
    )

    num_variables = width * height
    input_shape = input_shape_tuple(inputs[0])
    categorical_units = input_shape[-1]
    data_axis_order = folded.input_axis0_orders.get(0, tuple(range(num_variables)))
    data_axis_order_tensor = torch.tensor(data_axis_order, dtype=torch.long, device=device)
    train_images = load_train_images(
        dataset=dataset,
        device=device,
        data_dir=data_dir,
        num_samples=num_samples,
        num_variables=num_variables,
        pixel_values=pixel_values,
        download=download,
    )
    if preorder_inputs:
        train_images = train_images.index_select(1, data_axis_order_tensor)
    initial_categorical_logits = torch.normal(
        mean=0.0,
        std=1.0,
        size=(num_variables, pixel_values, categorical_units),
        device=device,
    )
    if preorder_inputs:
        initial_categorical_logits = initial_categorical_logits.index_select(0, data_axis_order_tensor)
    categorical_logits = torch.nn.Parameter(initial_categorical_logits)
    used_input_ids = {argument for instruction in program.instructions for argument in instruction.argument_ssa_ids if argument < program.n_inputs}
    packed_parameter_input_sequence = tuple(input_id for stack_order in folded.parameter_stack_orders for input_id in stack_order)
    packed_parameter_input_ids = set(packed_parameter_input_sequence)
    if len(packed_parameter_input_sequence) != len(packed_parameter_input_ids):
        raise ValueError("A parameter input cannot occur in more than one packed stack.")
    retained_input_ids = tuple(input_id for input_id in range(program.n_inputs) if input_id in used_input_ids and input_id not in packed_parameter_input_ids)
    data_input_id = retained_input_ids.index(0)
    initialized_parameter_inputs = {
        input_id: torch.normal(
            mean=0.0,
            std=1.0,
            size=input_shape_tuple(inputs[input_id]),
            device=device,
        )
        for input_id in sorted(program.parameter_indices)
    }
    parameter_inputs = {input_id: torch.nn.Parameter(tensor) for input_id, tensor in initialized_parameter_inputs.items() if input_id not in packed_parameter_input_ids}
    packed_parameter_inputs = [torch.nn.Parameter(torch.stack([initialized_parameter_inputs[input_id] for input_id in stack_order], dim=0)) for stack_order in folded.parameter_stack_orders]
    gather_index_inputs = [torch.tensor(indices, dtype=torch.long, device=device) for indices in gather_index_orders]
    constant_inputs = {input_id: inputs[input_id].backend_array.to(device) for input_id in range(program.n_inputs) if input_id != 0 and input_id not in program.parameter_indices}
    pixel_range = torch.arange(num_variables, device=device)[:, None]
    flattened_pixel_offsets = pixel_values * pixel_range

    def categorical_input(batch: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(categorical_logits, dim=1)
        transposed_batch = batch.T
        if resolved_categorical_lookup == "advanced":
            data_input = probabilities[pixel_range, transposed_batch].contiguous()
        elif resolved_categorical_lookup == "gather":
            indices = transposed_batch.unsqueeze(-1).expand(-1, -1, categorical_units)
            data_input = torch.gather(probabilities, 1, indices)
        else:
            flattened_indices = (flattened_pixel_offsets + transposed_batch).reshape(-1)
            flattened_probabilities = probabilities.reshape(num_variables * pixel_values, categorical_units)
            selected = flattened_probabilities.index_select(0, flattened_indices)
            data_input = selected.reshape(num_variables, batch.shape[0], categorical_units)
        if not preorder_inputs:
            data_input = data_input.index_select(0, data_axis_order_tensor)
        return data_input

    def original_input_tensor(input_id: int, data_input: torch.Tensor) -> torch.Tensor:
        if input_id == 0:
            return data_input
        if input_id in parameter_inputs:
            return parameter_inputs[input_id]
        return constant_inputs[input_id]

    def step(batch: torch.Tensor) -> torch.Tensor:
        data_input = categorical_input(batch)
        runtime_tensors: list[torch.Tensor] = [original_input_tensor(input_id, data_input) for input_id in retained_input_ids]
        runtime_tensors[data_input_id] = data_input
        runtime_tensors.extend(packed_parameter_inputs)
        runtime_tensors.extend(gather_index_inputs)
        if len(runtime_tensors) != runtime_program.n_inputs:
            raise RuntimeError(f"Expected {runtime_program.n_inputs} runtime inputs, got {len(runtime_tensors)}")
        log_likelihoods = run(runtime_tensors)
        return -torch.mean(log_likelihoods)

    compile_mode = "reduce-overhead" if device.type == "cuda" else None
    step = torch.compile(step, mode=compile_mode)

    optimizer = make_optimizer((categorical_logits, *parameter_inputs.values(), *packed_parameter_inputs), device, lr)
    return step, optimizer, train_images, runtime_program


def input_shape_tuple(value: object) -> tuple[int, ...]:
    shape = input_shape(value)
    if isinstance(shape, str):
        raise TypeError(f"Expected a shaped input, got {shape}")
    return shape


def setup_cirkit_training(
    *,
    width: int,
    height: int,
    num_units: int,
    sum_product_layer: str,
    region_graph: str,
    device: torch.device,
    dataset: str,
    data_dir: str,
    num_samples: int,
    pixel_values: int,
    lr: float,
    semiring: str = "lse-sum",
    download: bool = True,
):
    symbolic_circuit = make_symbolic_circuit(
        width=width,
        height=height,
        num_units=num_units,
        sum_product_layer=sum_product_layer,
        region_graph=region_graph,
    )
    ctx = PipelineContext(
        backend="torch",
        semiring=semiring,
        fold=True,
        optimize=True,
    )
    circuit = ctx.compile(symbolic_circuit).to(device)

    train_images = load_train_images(
        dataset=dataset,
        device=device,
        data_dir=data_dir,
        num_samples=num_samples,
        num_variables=width * height,
        pixel_values=pixel_values,
        download=download,
    )
    optimizer = make_optimizer(circuit.parameters(), device, lr)

    def step(batch: torch.Tensor) -> torch.Tensor:
        log_likelihoods = circuit(batch)
        return -torch.mean(log_likelihoods)

    compile_mode = "reduce-overhead" if device.type == "cuda" else None
    step = torch.compile(step, mode=compile_mode)

    return step, optimizer, train_images, None


def run_warmup(
    step: Callable,
    optimizer,
    *,
    batch_size: int,
    num_variables: int,
    pixel_values: int,
    warmup_steps: int,
    device: torch.device,
) -> float:
    if warmup_steps <= 0:
        return 0.0
    start = time.perf_counter()
    for _ in range(warmup_steps):
        batch = torch.randint(pixel_values, size=(batch_size, num_variables), device=device)
        loss = step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return elapsed_wall_ms(start, device)


def run_warmup_epochs(
    step: Callable,
    optimizer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int | None,
    warmup_epochs: int,
    device: torch.device,
) -> float:
    if warmup_epochs <= 0:
        return 0.0
    start = time.perf_counter()
    for _ in range(warmup_epochs):
        run_epoch(
            step,
            optimizer,
            train_images,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
        )
    return elapsed_wall_ms(start, device)


def empty_epoch_stats() -> dict[str, float | int]:
    return {
        "data_loading_ms": 0.0,
        "forward_loss_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_step_ms": 0.0,
        "zero_grad_ms": 0.0,
        "metrics_logging_ms": 0.0,
        "batches": 0,
        "samples": 0,
    }


def run_epoch(
    step: Callable,
    optimizer,
    train_images: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, float | int]:
    stats = empty_epoch_stats()
    epoch_loss = 0.0
    epoch_samples = 0
    full_batches = train_images.shape[0] // batch_size
    num_batches = min(full_batches, max_batches) if max_batches else full_batches
    permutation = torch.randperm(train_images.shape[0], device=device)

    synchronize_device(device)
    epoch_start = time.perf_counter()
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        data_loading_start = start_timer(device)
        batch_indices = permutation[batch_start : batch_start + batch_size]
        batch = train_images[batch_indices]
        data_loading_end = stop_timer(device)

        forward_start = start_timer(device)
        loss = step(batch)
        forward_end = stop_timer(device)

        backward_start = start_timer(device)
        loss.backward()
        backward_end = stop_timer(device)

        optimizer_step_start = start_timer(device)
        optimizer.step()
        optimizer_step_end = stop_timer(device)

        zero_grad_start = start_timer(device)
        optimizer.zero_grad(set_to_none=True)
        zero_grad_end = stop_timer(device)

        synchronize_device(device)
        stats["data_loading_ms"] += elapsed_timer_ms(data_loading_start, data_loading_end, device)
        stats["forward_loss_ms"] += elapsed_timer_ms(forward_start, forward_end, device)
        stats["backward_ms"] += elapsed_timer_ms(backward_start, backward_end, device)
        stats["optimizer_step_ms"] += elapsed_timer_ms(optimizer_step_start, optimizer_step_end, device)
        stats["zero_grad_ms"] += elapsed_timer_ms(zero_grad_start, zero_grad_end, device)

        metrics_start = time.perf_counter()
        loss_value = loss.detach().item()
        epoch_loss += loss_value * batch_size
        epoch_samples += batch_size
        stats["batches"] += 1
        stats["samples"] += batch_size
        stats["metrics_logging_ms"] += elapsed_wall_ms(metrics_start, device)

    stats["epoch_total_ms"] = elapsed_wall_ms(epoch_start, device)
    stats["avg_nll"] = epoch_loss / epoch_samples if epoch_samples else float("nan")
    stats["samples_per_sec"] = 1000.0 * epoch_samples / stats["epoch_total_ms"] if stats["epoch_total_ms"] > 0.0 else 0.0
    stats["batches_per_sec"] = 1000.0 * stats["batches"] / stats["epoch_total_ms"] if stats["epoch_total_ms"] > 0.0 else 0.0
    stats["avg_batch_ms"] = stats["epoch_total_ms"] / stats["batches"] if stats["batches"] else 0.0
    return stats


def append_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_has_header = output_path.exists() and output_path.stat().st_size > 0
    fieldnames = CSV_FIELDS
    if file_has_header:
        with output_path.open(newline="") as existing_file:
            existing_header = next(csv.reader(existing_file), None)
        if existing_header:
            fieldnames = tuple(existing_header)
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_has_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_epoch_summary(row: dict) -> None:
    print(
        f"{row['backend']:>6} units={row['units']:<4} batch={row['batch_size']:<4} "
        f"epoch={row['epoch']}: {row['epoch_total_ms']:.2f} ms, "
        f"{row['avg_nll']:.4f} avg_nll, "
        f"{row['samples_per_sec']:.1f} samples/s"
    )


def display_backend(backend: str, args: argparse.Namespace) -> str:
    backend_name = backend if args.region_graph == "quad-tree-2" else f"{backend}-{args.region_graph}"
    if backend != "xe":
        return backend_name
    return "-".join((backend_name, args.semiring, "torch-compile"))


def base_row(
    backend: str,
    num_units: int,
    batch_size: int,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    is_cirkit = backend.startswith("cirkit")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "status": "",
        "error": "",
        "device": str(device),
        "device_name": get_device_name(device),
        "width": args.width,
        "height": args.height,
        "variables": args.width * args.height,
        "pixel_values": args.pixel_values,
        "units": num_units,
        "batch_size": batch_size,
        "epoch": epoch,
        "epochs": args.epochs,
        "max_batches": args.max_batches or "",
        "num_samples": args.num_samples,
        "dataset": args.dataset,
        "region_graph": args.region_graph,
        "sum_product_layer": args.sum_product_layer,
        "semiring": args.semiring,
        "backend_type": "cirkit" if is_cirkit else "xe",
        "torch_compile": True,
        "compile_scope": "whole-step",
        "fold_strategy": (
            "cirkit"
            if is_cirkit
            else "input-depth"
        ),
        "gather_materialization": not is_cirkit,
        "optimizer": "adam",
        "optimizer_fused": device.type == "cuda",
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "warmup_epochs": args.warmup_epochs,
    }


def run_training_config(
    *,
    backend: str,
    num_units: int,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    # Include model construction, compilation, and warmup in the VRAM high-water
    # mark. Resetting after warmup would hide CUDA graph pools allocated during
    # capture even though the run needs to keep that reservation.
    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        baseline_allocated = 0
        baseline_reserved = 0
    reset_peak_memory(device)
    setup_fn = setup_cirkit_training if backend == "cirkit" else setup_xe_training
    setup_start = time.perf_counter()
    setup_kwargs = {
        "width": args.width,
        "height": args.height,
        "num_units": num_units,
        "sum_product_layer": args.sum_product_layer,
        "region_graph": args.region_graph,
        "device": device,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "num_samples": args.num_samples,
        "pixel_values": args.pixel_values,
        "lr": args.lr,
    }
    if backend != "cirkit":
        setup_kwargs["batch_size"] = batch_size
        setup_kwargs["semiring"] = args.semiring
    step, optimizer, train_images, _program = setup_fn(**setup_kwargs)
    setup_ms = elapsed_wall_ms(setup_start, device)
    warmup_ms = run_warmup(
        step,
        optimizer,
        batch_size=batch_size,
        num_variables=args.width * args.height,
        pixel_values=args.pixel_values,
        warmup_steps=args.warmup_steps,
        device=device,
    )
    warmup_ms += run_warmup_epochs(
        step,
        optimizer,
        train_images,
        batch_size=batch_size,
        max_batches=args.max_batches,
        warmup_epochs=args.warmup_epochs,
        device=device,
    )

    rows = []
    for epoch in range(args.epochs):
        stats = run_epoch(
            step,
            optimizer,
            train_images,
            batch_size=batch_size,
            max_batches=args.max_batches,
            device=device,
        )
        row = base_row(display_backend(backend, args), num_units, batch_size, epoch, args, device)
        row.update(
            {
                "status": "ok",
                "setup_ms": setup_ms if epoch == 0 else 0.0,
                "warmup_ms": warmup_ms if epoch == 0 else 0.0,
            }
        )
        row.update(stats)
        memory = memory_snapshot(device)
        peak_allocated = memory.get("peak_allocated_memory_bytes", "")
        peak_reserved = memory.get("peak_reserved_memory_bytes", "")
        memory.update(
            {
                "memory_scope": "setup+warmup+measured-training",
                "baseline_allocated_memory_bytes": baseline_allocated,
                "baseline_reserved_memory_bytes": baseline_reserved,
                "peak_incremental_allocated_memory_bytes": (
                    max(0, int(peak_allocated) - baseline_allocated)
                    if peak_allocated != ""
                    else ""
                ),
                "peak_incremental_reserved_memory_bytes": (
                    max(0, int(peak_reserved) - baseline_reserved)
                    if peak_reserved != ""
                    else ""
                ),
            }
        )
        row.update(memory)
        rows.append(row)
    return rows


def run_training_sweep(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    device = get_device(args.device)
    print(f"device={device} ({get_device_name(device)})")
    print(f"writing results to {args.output}")
    print(f"region_graph={args.region_graph}")

    for backend, num_units, batch_size in itertools.product(args.backends, args.unit_sizes, args.batch_sizes):
        cleanup_device(device)
        set_seed(args.seed)
        print(f"running backend={display_backend(backend, args)} units={num_units} batch={batch_size}")
        try:
            rows = run_training_config(
                backend=backend,
                num_units=num_units,
                batch_size=batch_size,
                args=args,
                device=device,
            )
        except RuntimeError as error:
            if device.type == "cuda" and "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
            row = base_row(display_backend(backend, args), num_units, batch_size, -1, args, device)
            row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            append_row(args.output, row)
            print(f"failed: {row['error']}")
            if args.stop_on_error:
                raise
            continue
        except Exception as error:
            row = base_row(display_backend(backend, args), num_units, batch_size, -1, args, device)
            row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            append_row(args.output, row)
            print(f"failed: {row['error']}")
            if args.verbose_errors:
                traceback.print_exc()
            if args.stop_on_error:
                raise
            continue

        for row in rows:
            append_row(args.output, row)
            print_epoch_summary(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate or train a Cirkit symbolic image circuit with XE.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the MNIST training benchmark pipeline instead of only printing the translated XE program.",
    )
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument(
        "--units",
        "--unit-sizes",
        dest="unit_sizes",
        type=parse_positive_ints,
        default=(DEFAULT_UNITS,),
        metavar="N[,N...]",
        help="Unit count, or comma-separated counts for a training sweep.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-sizes",
        dest="batch_sizes",
        type=parse_positive_ints,
        default=(DEFAULT_BATCH_SIZE,),
        metavar="N[,N...]",
        help="Batch size, or comma-separated sizes for a training sweep.",
    )
    parser.add_argument("--sum-product-layer", choices=("cp", "cp-t", "tucker"), default="cp")
    parser.add_argument("--region-graph", choices=("quad-tree-2", "quad-graph"), default="quad-tree-2")
    parser.add_argument(
        "--semiring",
        choices=("scaled-min", "scaled-max", "scaled-sum", "lse-sum"),
        default="lse-sum",
        help="Numerical-stability mode to request from XE.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip the preprocessing compatibility hook.",
    )
    parser.add_argument(
        "--no-optimize-stacking",
        action="store_true",
        help="Disable folding and contraction-path optimization preprocessing.",
    )
    parser.add_argument(
        "--dump-instructions",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N final XE instructions.",
    )
    parser.add_argument(
        "--shape-preview",
        type=int,
        default=12,
        metavar="N",
        help="Print at most N input shapes per program summary.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/cirkit_train.csv"))
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument(
        "--dataset",
        choices=("mnist", "synthetic"),
        default="mnist",
        help="Training data source. MNIST is the default and uses torchvision.",
    )
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=("xe",),
        help="Comma-separated subset of: xe,cirkit.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap for quick training runs; full train split when omitted.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of training samples to load; 0 means all available samples.",
    )
    parser.add_argument("--pixel-values", type=int, default=DEFAULT_PIXEL_VALUES)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    args = parser.parse_args()
    if not args.train and (len(args.unit_sizes) != 1 or len(args.batch_sizes) != 1):
        parser.error("multiple --units or --batch-size values require --train")
    return args


def main() -> None:
    args = parse_args()
    if args.train:
        if args.dataset == "mnist" and args.width == WIDTH and args.height == HEIGHT:
            args.width = 28
            args.height = 28
        run_training_sweep(args)
        return

    num_units = args.unit_sizes[0]
    batch_size = args.batch_sizes[0]

    symbolic_circuit = make_symbolic_circuit(
        width=args.width,
        height=args.height,
        num_units=num_units,
        sum_product_layer=args.sum_product_layer,
        region_graph=args.region_graph,
    )
    program, inputs = translate_cirkit_to_xe(
        symbolic_circuit,
        batch_size=batch_size,
        stability={
            "lse-sum": "logspace_max",
            "scaled-min": "scaled_min",
            "scaled-max": "scaled_max",
            "scaled-sum": "scaled_sum",
        }[args.semiring],
    )

    print(
        f"symbolic circuit: layers={len(symbolic_circuit.layers)}, variables={symbolic_circuit.num_variables}, units={num_units}, "
        f"sum_product_layer={args.sum_product_layer}, region_graph={args.region_graph}"
    )
    print_program_summary("direct XE program", program, inputs, shape_preview=args.shape_preview)

    final_program = program
    if not args.no_preprocess:
        try:
            preprocessed = preprocess_xe_program(
                program,
                optimize_stacking=not args.no_optimize_stacking,
            )
        except NameError as error:
            print()
            print(f"preprocessing skipped: {error}")
            preprocessed_inputs = inputs
        else:
            final_program = preprocessed.program if hasattr(preprocessed, "program") else preprocessed
            preprocessed_inputs = preprocessed.inputs if hasattr(preprocessed, "inputs") else inputs
            print()
            print_program_summary(
                "preprocessed XE program",
                final_program,
                preprocessed_inputs,
                shape_preview=args.shape_preview,
            )
            batched_input_ids = getattr(preprocessed_inputs, "batched_input_ids", None)
            index_input_ids = getattr(preprocessed_inputs, "index_input_ids", None)
            dynamic_transforms = getattr(preprocessed, "dynamic_input_transforms", None)
            if batched_input_ids is not None:
                print(f"  batched_input_ids: {batched_input_ids}")
            if index_input_ids is not None:
                print(f"  index_input_ids:   {index_input_ids}")
            if dynamic_transforms is not None:
                print(f"  dynamic_transforms: {len(dynamic_transforms)}")

    print_instructions(final_program, args.dump_instructions)


if __name__ == "__main__":
    main()
