from __future__ import annotations

# This file is also executed directly from experiments/.
# ruff: noqa: E402,I001

import argparse
import csv
import math
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]
sys.path.insert(0, str(_ROOT))

import torch

import extended_einsum.interface as xe
from demo.cirkit import (
    cleanup_device,
    make_symbolic_circuit,
    run_epoch,
    set_seed,
    setup_cirkit_training,
    setup_xe_training,
    translate_cirkit_to_xe,
)
from extended_einsum.backend_translation import (
    run_program,
    translate_to_backend_program,
)
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.interface.tensor_expression import Parameter
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import StabilityMode
from extended_einsum.preprocess import (
    FoldSameShapedOperations,
    OptimizeContractionPaths,
)
from torch.fx.experimental.proxy_tensor import make_fx

RESULTS = _HERE / "results" / "correctness.csv"
MNIST_TRAINING_RESULTS = _HERE / "results" / "correctness_mnist_training.csv"
SEEDS = (0, 1, 2, 3, 4)
REGION_GRAPHS = ("quad-tree-2", "quad-graph")
LAYERS = ("cp", "tucker")
MNIST_TRAINING_VARIANTS = ("cirkit", "logspace", "scaled-max")
AGREEMENT_VARIANTS: dict[str, StabilityMode] = {
    "scaled-max": "scaled_max",
    "logspace-max": "logspace_max",
}
STRESS_VARIANTS: tuple[tuple[str, StabilityMode, torch.dtype], ...] = (
    ("unstable-fp32", "unstable", torch.float32),
    ("unstable-fp64", "unstable", torch.float64),
    ("scaled-max-fp32", "scaled_max", torch.float32),
    ("logspace-max-fp32", "logspace_max", torch.float32),
)
MNIST_UNITS = {"cp": 512, "tucker": 64}
CSV_FIELDS = (
    "suite",
    "status",
    "error",
    "seed",
    "region_graph",
    "layer",
    "width",
    "height",
    "units",
    "batch_size",
    "pixel_values",
    "depth",
    "variant",
    "reference",
    "storage_dtype",
    "reference_dtype",
    "matmul_precision",
    "reference_matmul_precision",
    "device",
    "device_name",
    "torch_version",
    "cuda_version",
    "autocast",
    "torch_compile",
    "tf32_permitted",
    "compiler_pipeline",
    "forward_relative_l2",
    "forward_max_absolute_error",
    "data_gradient_relative_l2",
    "parameter_gradient_relative_l2",
    "worst_parameter_gradient_relative_l2",
    "gradient_max_absolute_error",
    "forward_finite_fraction",
    "gradient_finite_fraction",
    "reference_forward_finite_fraction",
    "reference_gradient_finite_fraction",
    "parameter_tensors",
    "parameters",
)
MNIST_TRAINING_CSV_FIELDS = (
    "status",
    "error",
    "variant",
    "backend",
    "seed",
    "region_graph",
    "layer",
    "units",
    "batch_size",
    "epoch",
    "epochs",
    "max_batches",
    "batches",
    "samples",
    "avg_nll",
    "lr",
    "dataset",
    "pixel_values",
    "device",
    "device_name",
    "torch_version",
    "cuda_version",
    "matmul_precision",
    "torch_compile",
    "optimizer",
    "optimizer_fused",
    "parameter_initialization",
    "minibatch_pairing",
)


@dataclass(frozen=True)
class Evaluation:
    output: torch.Tensor
    gradients: dict[int, torch.Tensor]


@dataclass(frozen=True)
class ErrorMetrics:
    forward_relative_l2: float
    forward_max_absolute_error: float
    data_gradient_relative_l2: float
    parameter_gradient_relative_l2: float
    worst_parameter_gradient_relative_l2: float
    gradient_max_absolute_error: float
    forward_finite_fraction: float
    gradient_finite_fraction: float
    reference_forward_finite_fraction: float
    reference_gradient_finite_fraction: float


def parse_ints(value: str, *, positive: bool = False) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("expected comma-separated integers") from error
    minimum = 1 if positive else 0
    if not parsed or any(item < minimum for item in parsed):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"expected one or more {qualifier}integers")
    return parsed


def parse_names(
    value: str,
    *,
    choices: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(parsed) - set(choices)
    if not parsed or unknown:
        raise ValueError(f"expected comma-separated {label} from {', '.join(choices)}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Supplementary forward/gradient correctness validation and depth-underflow stress test."))
    parser.add_argument(
        "--suites",
        default="agreement,stress",
        help="agreement, stress, mnist, mnist-training, or a comma-separated combination",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--layers", default="cp,tucker")
    parser.add_argument("--region-graphs", default="quad-tree-2,quad-graph")
    parser.add_argument("--precisions", default="highest,high")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="run optimized agreement/MNIST candidates through the production FX/torch.compile path",
    )
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--agreement-width", type=int, default=8)
    parser.add_argument("--agreement-height", type=int, default=8)
    parser.add_argument("--agreement-units", type=int, default=8)
    parser.add_argument("--agreement-batch-size", type=int, default=8)
    parser.add_argument("--agreement-pixel-values", type=int, default=2)
    parser.add_argument(
        "--stress-depths",
        default="4,8,16,32,64,128,256,512,1024",
    )
    parser.add_argument("--stress-units", type=int, default=64)
    parser.add_argument("--stress-batch-size", type=int, default=4)
    parser.add_argument("--stress-factor-scale", type=float, default=0.01)
    parser.add_argument("--mnist-batch-size", type=int, default=8)
    parser.add_argument(
        "--mnist-training-output",
        type=Path,
        default=MNIST_TRAINING_RESULTS,
    )
    parser.add_argument("--mnist-training-seed", type=int, default=0)
    parser.add_argument("--mnist-training-epochs", type=int, default=20)
    parser.add_argument("--mnist-training-units", type=int, default=512)
    parser.add_argument("--mnist-training-batch-size", type=int, default=512)
    parser.add_argument(
        "--mnist-training-region-graph",
        choices=REGION_GRAPHS,
        default="quad-tree-2",
    )
    parser.add_argument(
        "--mnist-training-variants",
        default="cirkit,logspace,scaled-max",
    )
    parser.add_argument(
        "--mnist-training-max-batches",
        type=int,
        default=None,
        help="cap batches per epoch for a smoke test; omitted means the full training split",
    )
    parser.add_argument("--mnist-training-lr", type=float, default=0.01)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow torchvision to download MNIST if it is not present",
    )
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument(
        "--_mnist-training-single",
        choices=MNIST_TRAINING_VARIANTS,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    try:
        args.suites = parse_names(
            args.suites,
            choices=("agreement", "stress", "mnist", "mnist-training"),
            label="suites",
        )
        args.seeds = parse_ints(args.seeds)
        args.layers = parse_names(
            args.layers,
            choices=LAYERS,
            label="layers",
        )
        args.region_graphs = parse_names(
            args.region_graphs,
            choices=REGION_GRAPHS,
            label="region graphs",
        )
        args.precisions = parse_names(
            args.precisions,
            choices=("highest", "high"),
            label="matmul precisions",
        )
        args.stress_depths = parse_ints(args.stress_depths, positive=True)
        args.mnist_training_variants = parse_names(
            args.mnist_training_variants,
            choices=MNIST_TRAINING_VARIANTS,
            label="MNIST training variants",
        )
    except ValueError as error:
        parser.error(str(error))
    positive_values = (
        args.agreement_width,
        args.agreement_height,
        args.agreement_units,
        args.agreement_batch_size,
        args.agreement_pixel_values,
        args.stress_units,
        args.stress_batch_size,
        args.mnist_batch_size,
        args.mnist_training_epochs,
        args.mnist_training_units,
        args.mnist_training_batch_size,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("all dimensions, units, and batch sizes must be positive")
    if not 0.0 < args.stress_factor_scale < 1.0:
        parser.error("--stress-factor-scale must be strictly between zero and one")
    if args.mnist_training_seed < 0:
        parser.error("--mnist-training-seed must be non-negative")
    if (
        args.mnist_training_max_batches is not None
        and args.mnist_training_max_batches <= 0
    ):
        parser.error("--mnist-training-max-batches must be positive")
    if args.mnist_training_lr <= 0.0:
        parser.error("--mnist-training-lr must be positive")
    return args


def get_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    return device


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def append_mnist_training_rows(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as stream:
            header = tuple(next(csv.reader(stream), ()))
        if header != MNIST_TRAINING_CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the MNIST training schema; move or remove it"
            )
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=MNIST_TRAINING_CSV_FIELDS,
        )
        if write_header:
            writer.writeheader()
        writer.writerows(
            {
                field: row.get(field, "")
                for field in MNIST_TRAINING_CSV_FIELDS
            }
            for row in rows
        )


def completed_mnist_training_variants(
    path: Path,
    *,
    seed: int,
    region_graph: str,
    units: int,
    batch_size: int,
    epochs: int,
    max_batches: int | None,
    lr: float,
    device: torch.device,
) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MNIST_TRAINING_CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the MNIST training schema; move or remove it"
            )
        matching_epochs: dict[str, set[int]] = {}
        expected_max_batches = "" if max_batches is None else str(max_batches)
        for row in reader:
            if (
                row["status"] != "ok"
                or int(row["seed"]) != seed
                or row["region_graph"] != region_graph
                or row["layer"] != "cp"
                or int(row["units"]) != units
                or int(row["batch_size"]) != batch_size
                or int(row["epochs"]) != epochs
                or row["max_batches"] != expected_max_batches
                or float(row["lr"]) != lr
                or row["device"] != str(device)
            ):
                continue
            matching_epochs.setdefault(row["variant"], set()).add(
                int(row["epoch"])
            )
    expected_epochs = set(range(1, epochs + 1))
    return {
        variant
        for variant, observed_epochs in matching_epochs.items()
        if observed_epochs == expected_epochs
    }


def successful_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} does not use the correctness schema; move or remove it")
        return {row_key(row) for row in reader if row["status"] == "ok"}


def row_key(row: dict[str, object]) -> tuple[str, ...]:
    fields = (
        "suite",
        "seed",
        "region_graph",
        "layer",
        "width",
        "height",
        "units",
        "batch_size",
        "pixel_values",
        "depth",
        "variant",
        "matmul_precision",
        "device",
        "device_name",
        "torch_compile",
    )
    return tuple(str(row.get(field, "")) for field in fields)


def program_with_mode(
    program: RichProgram,
    mode: StabilityMode,
) -> RichProgram:
    return replace(program, stability_mode=mode)


def make_circuit_program(
    *,
    width: int,
    height: int,
    units: int,
    batch_size: int,
    layer: str,
    region_graph: str,
) -> RichProgram:
    circuit = make_symbolic_circuit(
        width=width,
        height=height,
        num_units=units,
        sum_product_layer=layer,
        region_graph=region_graph,
    )
    program, _inputs = translate_cirkit_to_xe(
        circuit,
        batch_size=batch_size,
        stability="unstable",
    )
    return program


def make_depth_program(
    *,
    depth: int,
    units: int,
    batch_size: int,
) -> RichProgram:
    current = xe.array(torch.empty((batch_size, units), dtype=torch.float32))
    for _ in range(depth):
        factor = xe.array(torch.empty((batch_size, units), dtype=torch.float32))
        logits = Parameter(xe.array(torch.empty((units, units), dtype=torch.float32)))
        weights = xe.softmax(logits, axis=1)
        current = xe.einsum("bu,bu->bu", current, factor)
        current = xe.einsum("bi,oi->bo", current, weights)
    expression = xe.log(xe.einsum("bu->b", current))
    # Tensor-expression extraction currently walks the expression recursively.
    # Each synthetic level adds several nodes, so deliberately deep stress
    # graphs can exceed Python's default recursion limit before evaluation.
    previous_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_recursion_limit, 8 * depth + 1_000))
    try:
        program, _inputs = xe.extract_program(
            expression,
            stability_mode="unstable",
        )
    finally:
        sys.setrecursionlimit(previous_recursion_limit)
    return program


def categorical_likelihoods(
    *,
    observations: torch.Tensor,
    units: int,
    pixel_values: int,
    generator: torch.Generator,
    chunk_variables: int = 16,
) -> torch.Tensor:
    batch_size, num_variables = observations.shape
    likelihoods = torch.empty(
        (num_variables, batch_size, units),
        dtype=torch.float32,
    )
    for start in range(0, num_variables, chunk_variables):
        stop = min(start + chunk_variables, num_variables)
        logits = torch.randn(
            (stop - start, pixel_values, units),
            generator=generator,
            dtype=torch.float32,
        )
        probabilities = torch.softmax(logits, dim=1)
        chunk_observations = observations[:, start:stop].T
        variable_indices = torch.arange(stop - start)[:, None]
        likelihoods[start:stop] = probabilities[
            variable_indices,
            chunk_observations,
        ]
    return likelihoods


def make_circuit_inputs(
    program: RichProgram,
    *,
    seed: int,
    pixel_values: int,
    observations: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_variables, batch_size, units = program.shapes[0]
    if observations is None:
        observations = torch.randint(
            pixel_values,
            (batch_size, num_variables),
            generator=generator,
        )
    if observations.shape != (batch_size, num_variables):
        raise ValueError("observations do not match the circuit batch and variable dimensions")
    values = [
        categorical_likelihoods(
            observations=observations,
            units=units,
            pixel_values=pixel_values,
            generator=generator,
        )
    ]
    for input_id in range(1, program.n_inputs):
        values.append(
            torch.randn(
                program.shapes[input_id],
                generator=generator,
                dtype=torch.float32,
            )
        )
    return values


def make_depth_inputs(
    program: RichProgram,
    *,
    seed: int,
    factor_scale: float,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values: list[torch.Tensor] = []
    for input_id in range(program.n_inputs):
        shape = program.shapes[input_id]
        if input_id in program.parameter_indices:
            value = torch.randn(
                shape,
                generator=generator,
                dtype=torch.float32,
            )
        else:
            value = factor_scale * (
                0.75
                + 0.5
                * torch.rand(
                    shape,
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        values.append(value)
    return values


def optimized_program_and_inputs(
    program: RichProgram,
    original_inputs: list[torch.Tensor],
    *,
    device: torch.device,
    optimize_contractions: bool,
) -> tuple[RichProgram, list[torch.Tensor]]:
    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=True,
    )
    runtime_program = OptimizeContractionPaths.apply(folded.program) if optimize_contractions else folded.program
    used_input_ids = {argument for instruction in program.instructions for argument in instruction.argument_ssa_ids if argument < program.n_inputs}
    packed_sequence = tuple(input_id for stack_order in folded.parameter_stack_orders for input_id in stack_order)
    packed_ids = set(packed_sequence)
    if len(packed_sequence) != len(packed_ids):
        raise ValueError("a parameter input occurs in more than one packed stack")
    retained_ids = tuple(input_id for input_id in range(program.n_inputs) if input_id in used_input_ids and input_id not in packed_ids)

    runtime_inputs: list[torch.Tensor] = []
    for input_id in retained_ids:
        value = original_inputs[input_id]
        order = folded.input_axis0_orders.get(input_id)
        if order is not None:
            indices = torch.tensor(order, dtype=torch.long, device=device)
            value = value.index_select(0, indices)
        runtime_inputs.append(value)
    runtime_inputs.extend(
        torch.stack(
            [original_inputs[input_id] for input_id in stack_order],
            dim=0,
        )
        for stack_order in folded.parameter_stack_orders
    )
    runtime_inputs.extend(torch.tensor(indices, dtype=torch.long, device=device) for indices in folded.gather_index_orders)
    if len(runtime_inputs) != runtime_program.n_inputs:
        raise RuntimeError(f"expected {runtime_program.n_inputs} runtime inputs, got {len(runtime_inputs)}")
    return runtime_program, runtime_inputs


def evaluate(
    program: RichProgram,
    canonical_inputs: list[torch.Tensor],
    *,
    mode: StabilityMode,
    dtype: torch.dtype,
    device: torch.device,
    matmul_precision: Literal["highest", "high"],
    optimize: bool,
    optimize_contractions: bool = True,
    torch_compile_run: bool = False,
) -> Evaluation:
    torch.set_float32_matmul_precision(matmul_precision)
    differentiable_ids = tuple(sorted({0, *program.parameter_indices}))
    original_inputs: list[torch.Tensor] = []
    for input_id, canonical in enumerate(canonical_inputs):
        value = canonical.to(device=device, dtype=dtype).detach()
        value.requires_grad_(input_id in differentiable_ids)
        original_inputs.append(value)

    mode_program = program_with_mode(program, mode)
    if optimize:
        runtime_program, runtime_inputs = optimized_program_and_inputs(
            mode_program,
            original_inputs,
            device=device,
            optimize_contractions=optimize_contractions,
        )
    else:
        runtime_program = mode_program
        runtime_inputs = original_inputs
    backend_program = translate_to_backend_program(
        runtime_program,
        TorchBackendFunctions(),
    )

    def run_flat(*inputs: torch.Tensor) -> torch.Tensor:
        return run_program(backend_program, inputs)

    if torch_compile_run:
        graph_module = make_fx(run_flat, tracing_mode="fake")(*runtime_inputs)
        compile_mode = "reduce-overhead" if device.type == "cuda" else None
        run_flat = torch.compile(graph_module, mode=compile_mode)
    output = run_flat(*runtime_inputs)
    loss = -torch.mean(output)
    gradient_values = torch.autograd.grad(
        loss,
        [original_inputs[input_id] for input_id in differentiable_ids],
        allow_unused=False,
    )
    gradients = {
        input_id: gradient.detach().to(device="cpu", dtype=torch.float64)
        for input_id, gradient in zip(
            differentiable_ids,
            gradient_values,
            strict=True,
        )
    }
    return Evaluation(
        output=output.detach().to(device="cpu", dtype=torch.float64),
        gradients=gradients,
    )


def finite_fraction(tensors: list[torch.Tensor]) -> float:
    total = sum(tensor.numel() for tensor in tensors)
    if total == 0:
        return 1.0
    finite = sum(int(torch.count_nonzero(torch.isfinite(tensor))) for tensor in tensors)
    return finite / total


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if not torch.all(torch.isfinite(candidate)) or not torch.all(torch.isfinite(reference)):
        return math.inf
    difference_norm = torch.linalg.vector_norm(candidate - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    denominator = max(float(reference_norm), torch.finfo(torch.float64).tiny)
    return float(difference_norm) / denominator


def relative_l2_many(
    candidate: list[torch.Tensor],
    reference: list[torch.Tensor],
) -> float:
    if len(candidate) != len(reference):
        raise ValueError("candidate and reference tensor counts differ")
    if any(not torch.all(torch.isfinite(tensor)) for tensor in (*candidate, *reference)):
        return math.inf
    difference_squared = sum(float(torch.sum(torch.square(left - right))) for left, right in zip(candidate, reference, strict=True))
    reference_squared = sum(float(torch.sum(torch.square(tensor))) for tensor in reference)
    return math.sqrt(difference_squared) / max(
        math.sqrt(reference_squared),
        torch.finfo(torch.float64).tiny,
    )


def maximum_absolute_error(
    candidate: list[torch.Tensor],
    reference: list[torch.Tensor],
) -> float:
    if any(not torch.all(torch.isfinite(tensor)) for tensor in (*candidate, *reference)):
        return math.inf
    return max(
        (float(torch.max(torch.abs(left - right))) for left, right in zip(candidate, reference, strict=True)),
        default=0.0,
    )


def compare(
    candidate: Evaluation,
    reference: Evaluation,
    *,
    parameter_ids: frozenset[int],
) -> ErrorMetrics:
    data_candidate = [candidate.gradients[0]]
    data_reference = [reference.gradients[0]]
    ordered_parameters = sorted(parameter_ids)
    parameter_candidate = [candidate.gradients[input_id] for input_id in ordered_parameters]
    parameter_reference = [reference.gradients[input_id] for input_id in ordered_parameters]
    all_candidate_gradients = [candidate.gradients[input_id] for input_id in sorted(candidate.gradients)]
    all_reference_gradients = [reference.gradients[input_id] for input_id in sorted(reference.gradients)]
    per_parameter_errors = [
        relative_l2(left, right)
        for left, right in zip(
            parameter_candidate,
            parameter_reference,
            strict=True,
        )
    ]
    return ErrorMetrics(
        forward_relative_l2=relative_l2(candidate.output, reference.output),
        forward_max_absolute_error=maximum_absolute_error(
            [candidate.output],
            [reference.output],
        ),
        data_gradient_relative_l2=relative_l2_many(
            data_candidate,
            data_reference,
        ),
        parameter_gradient_relative_l2=relative_l2_many(
            parameter_candidate,
            parameter_reference,
        ),
        worst_parameter_gradient_relative_l2=max(
            per_parameter_errors,
            default=0.0,
        ),
        gradient_max_absolute_error=maximum_absolute_error(
            all_candidate_gradients,
            all_reference_gradients,
        ),
        forward_finite_fraction=finite_fraction([candidate.output]),
        gradient_finite_fraction=finite_fraction(all_candidate_gradients),
        reference_forward_finite_fraction=finite_fraction([reference.output]),
        reference_gradient_finite_fraction=finite_fraction(all_reference_gradients),
    )


def metric_row(metrics: ErrorMetrics) -> dict[str, float]:
    return {field: getattr(metrics, field) for field in ErrorMetrics.__dataclass_fields__}


def base_row(
    *,
    suite: str,
    seed: int,
    program: RichProgram,
    device: torch.device,
    region_graph: str = "",
    layer: str = "",
    width: int | str = "",
    height: int | str = "",
    units: int,
    batch_size: int,
    pixel_values: int | str = "",
    depth: int | str = "",
    torch_compile_run: bool = False,
    matmul_precision: str = "",
    compiler_pipeline: str = "fold+consumer-order+contraction-path+stability",
) -> dict[str, object]:
    return {
        "suite": suite,
        "status": "",
        "error": "",
        "seed": seed,
        "region_graph": region_graph,
        "layer": layer,
        "width": width,
        "height": height,
        "units": units,
        "batch_size": batch_size,
        "pixel_values": pixel_values,
        "depth": depth,
        "device": str(device),
        "device_name": device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "autocast": False,
        "torch_compile": torch_compile_run,
        "tf32_permitted": device.type == "cuda" and matmul_precision == "high",
        "compiler_pipeline": compiler_pipeline,
        "parameter_tensors": len(program.parameter_indices),
        "parameters": sum(math.prod(program.shapes[input_id]) for input_id in program.parameter_indices),
    }


def record_comparison(
    output: Path,
    *,
    base: dict[str, object],
    variant: str,
    reference_name: str,
    storage_dtype: torch.dtype,
    reference_dtype: torch.dtype,
    matmul_precision: str,
    reference_matmul_precision: str,
    candidate: Evaluation,
    reference: Evaluation,
    parameter_ids: frozenset[int],
) -> None:
    metrics = compare(
        candidate,
        reference,
        parameter_ids=parameter_ids,
    )
    row = {
        **base,
        "status": "ok",
        "variant": variant,
        "reference": reference_name,
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
        "reference_dtype": str(reference_dtype).removeprefix("torch."),
        "matmul_precision": matmul_precision,
        "reference_matmul_precision": reference_matmul_precision,
        "tf32_permitted": (
            str(base["device"]).startswith("cuda")
            and storage_dtype == torch.float32
            and matmul_precision == "high"
        ),
        **metric_row(metrics),
    }
    append_row(output, row)
    print(
        f"ok {base['suite']:>9} {variant:>18} seed={base['seed']} forward={metrics.forward_relative_l2:.3e} parameter-grad={metrics.parameter_gradient_relative_l2:.3e}",
        flush=True,
    )


def record_failure(
    output: Path,
    *,
    base: dict[str, object],
    variant: str,
    reference_name: str,
    storage_dtype: torch.dtype,
    reference_dtype: torch.dtype,
    matmul_precision: str,
    error: Exception,
    verbose: bool,
) -> None:
    append_row(
        output,
        {
            **base,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "variant": variant,
            "reference": reference_name,
            "storage_dtype": str(storage_dtype).removeprefix("torch."),
            "reference_dtype": str(reference_dtype).removeprefix("torch."),
            "matmul_precision": matmul_precision,
            "reference_matmul_precision": "highest",
            "tf32_permitted": (
                str(base["device"]).startswith("cuda")
                and storage_dtype == torch.float32
                and matmul_precision == "high"
            ),
        },
    )
    print(
        f"failed {base['suite']} {variant}: {type(error).__name__}: {error}",
        flush=True,
    )
    if verbose:
        traceback.print_exc()


def run_agreement(args: argparse.Namespace, device: torch.device) -> None:
    completed = successful_keys(args.output)
    for seed in args.seeds:
        for layer in args.layers:
            for region_graph in args.region_graphs:
                program = make_circuit_program(
                    width=args.agreement_width,
                    height=args.agreement_height,
                    units=args.agreement_units,
                    batch_size=args.agreement_batch_size,
                    layer=layer,
                    region_graph=region_graph,
                )
                canonical_inputs = make_circuit_inputs(
                    program,
                    seed=seed,
                    pixel_values=args.agreement_pixel_values,
                )
                base = base_row(
                    suite="agreement",
                    seed=seed,
                    program=program,
                    device=device,
                    region_graph=region_graph,
                    layer=layer,
                    width=args.agreement_width,
                    height=args.agreement_height,
                    units=args.agreement_units,
                    batch_size=args.agreement_batch_size,
                    pixel_values=args.agreement_pixel_values,
                    torch_compile_run=args.torch_compile,
                )
                reference = evaluate(
                    program,
                    canonical_inputs,
                    mode="unstable",
                    dtype=torch.float64,
                    device=device,
                    matmul_precision="highest",
                    optimize=False,
                )
                for precision in args.precisions:
                    for variant, mode in AGREEMENT_VARIANTS.items():
                        prospective = {
                            **base,
                            "variant": variant,
                            "matmul_precision": precision,
                        }
                        if row_key(prospective) in completed:
                            continue
                        try:
                            candidate = evaluate(
                                program,
                                canonical_inputs,
                                mode=mode,
                                dtype=torch.float32,
                                device=device,
                                matmul_precision=precision,
                                optimize=True,
                                torch_compile_run=args.torch_compile,
                            )
                            record_comparison(
                                args.output,
                                base=base,
                                variant=variant,
                                reference_name="unstable-fp64-unoptimized",
                                storage_dtype=torch.float32,
                                reference_dtype=torch.float64,
                                matmul_precision=precision,
                                reference_matmul_precision="highest",
                                candidate=candidate,
                                reference=reference,
                                parameter_ids=program.parameter_indices,
                            )
                        except Exception as error:
                            record_failure(
                                args.output,
                                base=base,
                                variant=variant,
                                reference_name="unstable-fp64-unoptimized",
                                storage_dtype=torch.float32,
                                reference_dtype=torch.float64,
                                matmul_precision=precision,
                                error=error,
                                verbose=args.verbose_errors,
                            )
                del reference, canonical_inputs
                cleanup_device(device)


def run_stress(args: argparse.Namespace, device: torch.device) -> None:
    completed = successful_keys(args.output)
    for seed in args.seeds:
        for depth in args.stress_depths:
            program = make_depth_program(
                depth=depth,
                units=args.stress_units,
                batch_size=args.stress_batch_size,
            )
            canonical_inputs = make_depth_inputs(
                program,
                seed=seed,
                factor_scale=args.stress_factor_scale,
            )
            base = base_row(
                suite="stress",
                seed=seed,
                program=program,
                device=device,
                units=args.stress_units,
                batch_size=args.stress_batch_size,
                depth=depth,
                torch_compile_run=False,
                compiler_pipeline="fold+consumer-order+stability",
            )
            reference = evaluate(
                program,
                canonical_inputs,
                mode="logspace_max",
                dtype=torch.float64,
                device=device,
                matmul_precision="highest",
                optimize=False,
            )
            for precision in args.precisions:
                for variant, mode, dtype in STRESS_VARIANTS:
                    prospective = {
                        **base,
                        "variant": variant,
                        "matmul_precision": precision,
                    }
                    if row_key(prospective) in completed:
                        continue
                    try:
                        candidate = evaluate(
                            program,
                            canonical_inputs,
                            mode=mode,
                            dtype=dtype,
                            device=device,
                            matmul_precision=precision,
                            optimize=True,
                            optimize_contractions=False,
                        )
                        record_comparison(
                            args.output,
                            base=base,
                            variant=variant,
                            reference_name="logspace-fp64-unoptimized",
                            storage_dtype=dtype,
                            reference_dtype=torch.float64,
                            matmul_precision=precision,
                            reference_matmul_precision="highest",
                            candidate=candidate,
                            reference=reference,
                            parameter_ids=program.parameter_indices,
                        )
                    except Exception as error:
                        record_failure(
                            args.output,
                            base=base,
                            variant=variant,
                            reference_name="logspace-fp64-unoptimized",
                            storage_dtype=dtype,
                            reference_dtype=torch.float64,
                            matmul_precision=precision,
                            error=error,
                            verbose=args.verbose_errors,
                        )
            del reference, canonical_inputs
            cleanup_device(device)


def load_mnist_observations(
    *,
    data_dir: str,
    batch_size: int,
    seed: int,
    download: bool,
) -> torch.Tensor:
    from torchvision import datasets

    dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=download,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset.data), generator=generator)[:batch_size]
    return dataset.data[indices].reshape(batch_size, -1).long()


def run_mnist(args: argparse.Namespace, device: torch.device) -> None:
    completed = successful_keys(args.output)
    for seed in args.seeds:
        observations = load_mnist_observations(
            data_dir=args.data_dir,
            batch_size=args.mnist_batch_size,
            seed=seed,
            download=args.download,
        )
        for layer in args.layers:
            units = MNIST_UNITS[layer]
            for region_graph in args.region_graphs:
                program = make_circuit_program(
                    width=28,
                    height=28,
                    units=units,
                    batch_size=args.mnist_batch_size,
                    layer=layer,
                    region_graph=region_graph,
                )
                canonical_inputs = make_circuit_inputs(
                    program,
                    seed=seed,
                    pixel_values=256,
                    observations=observations,
                )
                base = base_row(
                    suite="mnist",
                    seed=seed,
                    program=program,
                    device=device,
                    region_graph=region_graph,
                    layer=layer,
                    width=28,
                    height=28,
                    units=units,
                    batch_size=args.mnist_batch_size,
                    pixel_values=256,
                    torch_compile_run=args.torch_compile,
                    matmul_precision="high",
                )
                prospective = {
                    **base,
                    "variant": "scaled-max",
                    "matmul_precision": "high",
                }
                if row_key(prospective) in completed:
                    continue
                try:
                    reference = evaluate(
                        program,
                        canonical_inputs,
                        mode="logspace_max",
                        dtype=torch.float32,
                        device=device,
                        matmul_precision="high",
                        optimize=True,
                        torch_compile_run=args.torch_compile,
                    )
                    candidate = evaluate(
                        program,
                        canonical_inputs,
                        mode="scaled_max",
                        dtype=torch.float32,
                        device=device,
                        matmul_precision="high",
                        optimize=True,
                        torch_compile_run=args.torch_compile,
                    )
                    record_comparison(
                        args.output,
                        base=base,
                        variant="scaled-max",
                        reference_name="logspace-fp32-optimized",
                        storage_dtype=torch.float32,
                        reference_dtype=torch.float32,
                        matmul_precision="high",
                        reference_matmul_precision="high",
                        candidate=candidate,
                        reference=reference,
                        parameter_ids=program.parameter_indices,
                    )
                except Exception as error:
                    record_failure(
                        args.output,
                        base=base,
                        variant="scaled-max",
                        reference_name="logspace-fp32-optimized",
                        storage_dtype=torch.float32,
                        reference_dtype=torch.float32,
                        matmul_precision="high",
                        error=error,
                        verbose=args.verbose_errors,
                    )
                del canonical_inputs
                cleanup_device(device)


def mnist_training_base_row(
    args: argparse.Namespace,
    device: torch.device,
    *,
    variant: str,
) -> dict[str, object]:
    is_cirkit = variant == "cirkit"
    return {
        "status": "",
        "error": "",
        "variant": variant,
        "backend": "cirkit" if is_cirkit else "extended-einsum",
        "seed": args.mnist_training_seed,
        "region_graph": args.mnist_training_region_graph,
        "layer": "cp",
        "units": args.mnist_training_units,
        "batch_size": args.mnist_training_batch_size,
        "epochs": args.mnist_training_epochs,
        "max_batches": args.mnist_training_max_batches or "",
        "lr": args.mnist_training_lr,
        "dataset": "mnist-train",
        "pixel_values": 256,
        "device": str(device),
        "device_name": device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "matmul_precision": "high",
        "torch_compile": True,
        "optimizer": "adam",
        "optimizer_fused": device.type == "cuda",
        "parameter_initialization": (
            "native-cirkit-seeded"
            if is_cirkit
            else "paired-xe-seeded"
        ),
        "minibatch_pairing": f"torch-seed-{1_000_000 + args.mnist_training_seed}",
    }


def run_mnist_training_variant(
    args: argparse.Namespace,
    device: torch.device,
    *,
    variant: str,
) -> None:
    torch.set_float32_matmul_precision("high")
    set_seed(args.mnist_training_seed)
    base = mnist_training_base_row(
        args,
        device,
        variant=variant,
    )
    common = {
        "width": 28,
        "height": 28,
        "num_units": args.mnist_training_units,
        "sum_product_layer": "cp",
        "region_graph": args.mnist_training_region_graph,
        "device": device,
        "dataset": "mnist",
        "data_dir": args.data_dir,
        "num_samples": 0,
        "pixel_values": 256,
        "lr": args.mnist_training_lr,
        "download": args.download,
    }
    try:
        if variant == "cirkit":
            step, optimizer, train_images, _program = setup_cirkit_training(
                **common,
                semiring="lse-sum",
            )
        else:
            semiring = {
                "logspace": "lse-sum",
                "scaled-max": "scaled-max",
            }[variant]
            step, optimizer, train_images, _program = setup_xe_training(
                **common,
                batch_size=args.mnist_training_batch_size,
                semiring=semiring,
                shift_mode="xe",
                optimize_group_order=True,
                preorder_inputs=True,
                optimize_contraction_paths=True,
            )

        # Setup consumes a backend-dependent amount of randomness. Reset here
        # so each variant sees the same permutation in each numbered epoch.
        set_seed(1_000_000 + args.mnist_training_seed)
        rows: list[dict[str, object]] = []
        for epoch in range(1, args.mnist_training_epochs + 1):
            stats = run_epoch(
                step,
                optimizer,
                train_images,
                batch_size=args.mnist_training_batch_size,
                max_batches=args.mnist_training_max_batches,
                device=device,
            )
            row = {
                **base,
                "status": "ok",
                "epoch": epoch,
                "batches": stats["batches"],
                "samples": stats["samples"],
                "avg_nll": stats["avg_nll"],
            }
            rows.append(row)
            print(
                f"ok mnist-training {variant:>10} "
                f"epoch={epoch:>2}/{args.mnist_training_epochs} "
                f"nll={float(stats['avg_nll']):.6f}",
                flush=True,
            )
        # Commit a complete trajectory together. A failed or interrupted run
        # cannot masquerade as a completed variant on resume.
        append_mnist_training_rows(
            args.mnist_training_output,
            rows,
        )
    except Exception as error:
        append_mnist_training_rows(
            args.mnist_training_output,
            [
                {
                    **base,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "epoch": 0,
                }
            ],
        )
        print(
            f"failed mnist-training {variant}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        if args.verbose_errors:
            traceback.print_exc()
    finally:
        cleanup_device(device)


def mnist_training_child_command(
    args: argparse.Namespace,
    *,
    variant: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--suites",
        "mnist-training",
        "--device",
        args.device,
        "--data-dir",
        args.data_dir,
        "--mnist-training-output",
        str(args.mnist_training_output),
        "--mnist-training-seed",
        str(args.mnist_training_seed),
        "--mnist-training-epochs",
        str(args.mnist_training_epochs),
        "--mnist-training-units",
        str(args.mnist_training_units),
        "--mnist-training-batch-size",
        str(args.mnist_training_batch_size),
        "--mnist-training-region-graph",
        args.mnist_training_region_graph,
        "--mnist-training-lr",
        str(args.mnist_training_lr),
        "--_mnist-training-single",
        variant,
    ]
    if args.mnist_training_max_batches is not None:
        command.extend(
            (
                "--mnist-training-max-batches",
                str(args.mnist_training_max_batches),
            )
        )
    if args.download:
        command.append("--download")
    if args.verbose_errors:
        command.append("--verbose-errors")
    return command


def run_mnist_training(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    completed = completed_mnist_training_variants(
        args.mnist_training_output,
        seed=args.mnist_training_seed,
        region_graph=args.mnist_training_region_graph,
        units=args.mnist_training_units,
        batch_size=args.mnist_training_batch_size,
        epochs=args.mnist_training_epochs,
        max_batches=args.mnist_training_max_batches,
        lr=args.mnist_training_lr,
        device=device,
    )
    remaining = [
        variant
        for variant in args.mnist_training_variants
        if variant not in completed
    ]
    print(
        f"MNIST training: {len(remaining)} remaining / "
        f"{len(args.mnist_training_variants)} requested; "
        f"output={args.mnist_training_output}",
        flush=True,
    )
    for index, variant in enumerate(remaining, start=1):
        print(
            f"[{index}/{len(remaining)}] mnist-training {variant}",
            flush=True,
        )
        subprocess.run(
            mnist_training_child_command(args, variant=variant),
            check=False,
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args._mnist_training_single:
        print(
            f"device={device} ({device_name(device)}); "
            f"output={args.mnist_training_output}",
            flush=True,
        )
        run_mnist_training_variant(
            args,
            device,
            variant=args._mnist_training_single,
        )
        return
    print(
        f"device={device} ({device_name(device)}); output={args.output}",
        flush=True,
    )
    if "agreement" in args.suites:
        run_agreement(args, device)
    if "stress" in args.suites:
        run_stress(args, device)
    if "mnist" in args.suites:
        run_mnist(args, device)
    if "mnist-training" in args.suites:
        run_mnist_training(args, device)


if __name__ == "__main__":
    main()
