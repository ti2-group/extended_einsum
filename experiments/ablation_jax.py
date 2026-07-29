from __future__ import annotations

import argparse
import csv
import gc
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import torch

from demo.cirkit import (
    input_shape_tuple,
    load_train_images,
    make_symbolic_circuit,
    translate_cirkit_to_xe,
)
from experiments.common import BATCHES_PER_EPOCH, MEASURED_EPOCHS, RESULTS_DIR, WARMUP_BATCHES, parse_ints, parse_layers
from extended_einsum.backend_translation import run_program, translate_to_backend_program
from extended_einsum.backends.jax import JaxBackendFunctions
from extended_einsum.preprocess import FoldSameShapedOperations, OptimizeContractionPaths

VARIANTS = {
    "xe": ("scaled-max", "xe", True),
    "logspace": ("lse-sum", "xe", True),
    "shift-gradients": ("scaled-max", "differentiable", True),
    "logspace-shift-gradients": ("lse-sum", "differentiable", True),
    "no-ordering": ("scaled-max", "xe", False),
}
ABLATION_GRID = {
    "cp": ((256, 128), (512, 512)),
    "tucker": ((256, 32), (512, 64)),
}
REGION_GRAPHS = ("quad-tree-2", "quad-graph")
CSV_FIELDS = (
    "backend",
    "variant",
    "status",
    "error",
    "device_name",
    "seed",
    "region_graph",
    "layer",
    "units",
    "batch_size",
    "warmup_batches",
    "measured_batches",
    "forward_ms_per_batch",
    "backward_estimate_ms_per_batch",
    "forward_backward_ms_per_batch",
    "optimizer_step_ms_per_batch",
    "train_step_ms_per_batch",
    "timing_note",
)


class DifferentiableShiftJaxBackendFunctions(JaxBackendFunctions):
    @staticmethod
    def stop_gradient(array: jax.Array) -> jax.Array:
        return array


@dataclass(frozen=True)
class Configuration:
    seed: int
    region_graph: str
    layer: str
    batch_size: int
    units: int
    variant: str

    @property
    def key(self) -> tuple[str, int, str, str, int, int]:
        return self.variant, self.seed, self.region_graph, self.layer, self.batch_size, self.units

    def child_arguments(self) -> list[str]:
        return [
            "--_single",
            str(self.seed),
            self.region_graph,
            self.layer,
            str(self.batch_size),
            str(self.units),
            self.variant,
        ]


def to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    backend_array = getattr(value, "backend_array", value)
    if isinstance(backend_array, torch.Tensor):
        return backend_array.detach().cpu().numpy()
    return np.asarray(backend_array)


def setup(configuration: Configuration):
    semiring, shift_mode, optimize_group_order = VARIANTS[configuration.variant]
    symbolic_circuit = make_symbolic_circuit(
        width=28,
        height=28,
        num_units=configuration.units,
        sum_product_layer=configuration.layer,
        region_graph=configuration.region_graph,
    )
    program, inputs = translate_cirkit_to_xe(
        symbolic_circuit,
        batch_size=configuration.batch_size,
        stability={
            "lse-sum": "logspace_max",
            "scaled-max": "scaled_max",
        }[semiring],
    )
    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=optimize_group_order,
    )
    runtime_program = OptimizeContractionPaths.apply(folded.program)
    backend_functions = DifferentiableShiftJaxBackendFunctions() if shift_mode == "differentiable" else JaxBackendFunctions()
    backend_program = translate_to_backend_program(runtime_program, backend_functions)

    num_variables = 28 * 28
    categorical_units = input_shape_tuple(inputs[0])[-1]
    data_axis_order = folded.input_axis0_orders.get(0, tuple(range(num_variables)))
    train_images_torch = load_train_images(
        dataset="mnist",
        device=torch.device("cpu"),
        data_dir="datasets",
        num_samples=0,
        num_variables=num_variables,
        pixel_values=256,
    )
    train_images = jax.device_put(np.asarray(train_images_torch[:, data_axis_order], dtype=np.int32))

    rng = np.random.default_rng(configuration.seed)
    categorical_logits = jax.device_put(rng.normal(size=(num_variables, 256, categorical_units)).astype(np.float32))
    categorical_logits = categorical_logits[np.asarray(data_axis_order)]
    initialized_parameter_inputs = {input_id: jax.device_put(rng.normal(size=input_shape_tuple(inputs[input_id])).astype(np.float32)) for input_id in sorted(program.parameter_indices)}

    used_input_ids = {argument for instruction in program.instructions for argument in instruction.argument_ssa_ids if argument < program.n_inputs}
    packed_parameter_input_sequence = tuple(input_id for stack_order in folded.parameter_stack_orders for input_id in stack_order)
    packed_parameter_input_ids = set(packed_parameter_input_sequence)
    retained_input_ids = tuple(input_id for input_id in range(program.n_inputs) if input_id in used_input_ids and input_id not in packed_parameter_input_ids)
    data_input_id = retained_input_ids.index(0)
    parameter_input_ids = tuple(input_id for input_id in retained_input_ids if input_id in program.parameter_indices)
    packed_parameter_inputs = tuple(
        jnp.stack(
            [initialized_parameter_inputs[input_id] for input_id in stack_order],
            axis=0,
        )
        for stack_order in folded.parameter_stack_orders
    )
    params = (
        categorical_logits,
        *(initialized_parameter_inputs[input_id] for input_id in parameter_input_ids),
        *packed_parameter_inputs,
    )
    parameter_positions = {input_id: position + 1 for position, input_id in enumerate(parameter_input_ids)}
    packed_offset = 1 + len(parameter_input_ids)
    constants = {input_id: jax.device_put(to_numpy(inputs[input_id])) for input_id in retained_input_ids if input_id != 0 and input_id not in program.parameter_indices}
    gather_indices = tuple(jax.device_put(np.asarray(indices, dtype=np.int32)) for indices in folded.gather_index_orders)
    pixel_range = jnp.arange(num_variables, dtype=jnp.int32)[:, None]

    def loss_fn(current_params, batch):
        logits = current_params[0]
        probabilities = jax.nn.softmax(logits, axis=1)
        data_input = probabilities[pixel_range, batch.T]
        runtime_tensors = []
        for input_id in retained_input_ids:
            if input_id == 0:
                runtime_tensors.append(data_input)
            elif input_id in parameter_positions:
                runtime_tensors.append(current_params[parameter_positions[input_id]])
            else:
                runtime_tensors.append(constants[input_id])
        runtime_tensors[data_input_id] = data_input
        runtime_tensors.extend(current_params[packed_offset:])
        runtime_tensors.extend(gather_indices)
        return -jnp.mean(run_program(backend_program, runtime_tensors))

    example_batch = train_images[: configuration.batch_size]
    forward = jax.jit(loss_fn).trace(params, example_batch).lower().compile()
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn)).trace(params, example_batch).lower().compile()

    zeros = jax.tree.map(jnp.zeros_like, params)
    optimizer_state = (jnp.asarray(0, dtype=jnp.int32), zeros, zeros)

    def adam_update(current_params, gradients, state):
        step, first_moment, second_moment = state
        next_step = step + 1
        next_first = jax.tree.map(
            lambda moment, gradient: 0.9 * moment + 0.1 * gradient,
            first_moment,
            gradients,
        )
        next_second = jax.tree.map(
            lambda moment, gradient: 0.999 * moment + 0.001 * jnp.square(gradient),
            second_moment,
            gradients,
        )
        correction1 = 1.0 - jnp.power(0.9, next_step)
        correction2 = 1.0 - jnp.power(0.999, next_step)
        next_params = jax.tree.map(
            lambda parameter, first, second: parameter - 0.01 * (first / correction1) / (jnp.sqrt(second / correction2) + 1e-8),
            current_params,
            next_first,
            next_second,
        )
        return next_params, (next_step, next_first, next_second)

    _, example_gradients = value_and_grad(params, example_batch)
    jax.block_until_ready(example_gradients)
    update = jax.jit(adam_update).trace(params, example_gradients, optimizer_state).lower().compile()
    return forward, value_and_grad, update, params, optimizer_state, train_images


def elapsed_ms(call, *arguments):
    start = time.perf_counter()
    result = call(*arguments)
    jax.block_until_ready(result)
    return result, (time.perf_counter() - start) * 1000.0


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def run_configuration(configuration: Configuration, output: Path, verbose_errors: bool) -> None:
    row = {
        "backend": "jax",
        "variant": configuration.variant,
        "status": "",
        "error": "",
        "device_name": str(jax.devices()[0]),
        "seed": configuration.seed,
        "region_graph": configuration.region_graph,
        "layer": configuration.layer,
        "units": configuration.units,
        "batch_size": configuration.batch_size,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": 0,
        "timing_note": "backward_estimate = compiled value_and_grad - separately compiled forward",
    }
    try:
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX GPU backend required for publication ablation; found {jax.default_backend()}")
        forward, value_and_grad, update, params, optimizer_state, images = setup(configuration)
        permutation_rng = np.random.default_rng(1_000_000 + configuration.seed)
        batch_count = len(images) // configuration.batch_size
        total_batches = WARMUP_BATCHES + MEASURED_EPOCHS * BATCHES_PER_EPOCH
        batch_indices = np.concatenate(
            [
                permutation_rng.permutation(len(images))[: batch_count * configuration.batch_size].reshape(batch_count, configuration.batch_size)
                for _ in range((total_batches + batch_count - 1) // batch_count)
            ],
            axis=0,
        )[:total_batches]

        def one_batch(indices):
            nonlocal params, optimizer_state
            batch = images[indices]
            _, forward_ms = elapsed_ms(forward, params, batch)
            (loss, gradients), forward_backward_ms = elapsed_ms(value_and_grad, params, batch)
            (params, optimizer_state), optimizer_ms = elapsed_ms(update, params, gradients, optimizer_state)
            jax.block_until_ready(loss)
            return forward_ms, forward_backward_ms, optimizer_ms

        for indices in batch_indices[:WARMUP_BATCHES]:
            one_batch(indices)

        measurements = []
        offset = WARMUP_BATCHES
        for _ in range(MEASURED_EPOCHS):
            epoch_rows = [one_batch(indices) for indices in batch_indices[offset : offset + BATCHES_PER_EPOCH]]
            offset += BATCHES_PER_EPOCH
            measurements.append(tuple(np.mean(epoch_rows, axis=0)))

        forward_ms = statistics.median(item[0] for item in measurements)
        forward_backward_ms = statistics.median(item[1] for item in measurements)
        optimizer_ms = statistics.median(item[2] for item in measurements)
        backward_estimate = max(0.0, forward_backward_ms - forward_ms)
        row.update(
            status="ok",
            measured_batches=MEASURED_EPOCHS * BATCHES_PER_EPOCH,
            forward_ms_per_batch=forward_ms,
            backward_estimate_ms_per_batch=backward_estimate,
            forward_backward_ms_per_batch=forward_backward_ms,
            optimizer_step_ms_per_batch=optimizer_ms,
            train_step_ms_per_batch=forward_backward_ms + optimizer_ms,
        )
        append_row(output, row)
        print(
            f"ok jax {configuration.variant:>28} {configuration.layer:>6} "
            f"{configuration.region_graph:>11} B={configuration.batch_size:<3} "
            f"U={configuration.units:<4} seed={configuration.seed}: "
            f"{forward_backward_ms:.3f} ms/batch",
            flush=True,
        )
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        append_row(output, row)
        print(f"failed {configuration}: {row['error']}", flush=True)
        if verbose_errors:
            traceback.print_exc()
        raise
    finally:
        gc.collect()
        jax.clear_caches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX publication ablation.")
    parser.add_argument("--layers", default="cp,tucker")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation_jax.csv")
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--_single", nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
    args.seeds = parse_ints(args.seeds)
    return args


def completed_keys(path: Path) -> set[tuple[str, int, str, str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} does not use the JAX ablation schema")
        return {
            (
                row["variant"],
                int(row["seed"]),
                row["region_graph"],
                row["layer"],
                int(row["batch_size"]),
                int(row["units"]),
            )
            for row in reader
            if row["status"] == "ok"
        }


def main() -> None:
    args = parse_args()
    if args._single:
        seed, graph, layer, batch, units, variant = args._single
        run_configuration(
            Configuration(int(seed), graph, layer, int(batch), int(units), variant),
            args.output,
            args.verbose_errors,
        )
        return

    blocks = [(seed, graph, layer, batch, units) for seed in args.seeds for layer in args.layers for graph in REGION_GRAPHS for batch, units in ABLATION_GRID[layer]]
    random.Random(20260731).shuffle(blocks)
    configurations = []
    variant_names = tuple(VARIANTS)
    for index, (seed, graph, layer, batch, units) in enumerate(blocks):
        order = variant_names if index % 2 == 0 else tuple(reversed(variant_names))
        configurations.extend(Configuration(seed, graph, layer, batch, units, variant) for variant in order)

    completed = completed_keys(args.output)
    remaining = [configuration for configuration in configurations if configuration.key not in completed]
    print(
        f"JAX ablation: {len(remaining)} remaining / {len(configurations)} total; output={args.output}",
        flush=True,
    )
    for index, configuration in enumerate(remaining, start=1):
        print(f"[{index}/{len(remaining)}] {configuration}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--_single",
                *configuration.child_arguments()[1:],
                *(["--verbose-errors"] if args.verbose_errors else []),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
