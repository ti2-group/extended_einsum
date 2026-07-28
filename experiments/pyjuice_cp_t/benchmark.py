from __future__ import annotations

# The script is executed directly from its subdirectory.
# ruff: noqa: E402,I001

import argparse
import csv
import gc
import importlib.metadata
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import pyjuice as juice
import torch

from demo.cirkit import (
    cleanup_device,
    get_device,
    get_device_name,
    set_seed,
    setup_cirkit_training,
    setup_xe_training,
)
from experiments.pyjuice_cp_t.model import (
    build_cp_t_quad_tree,
    expected_parameters,
    validate_structure,
)

PYJUICE_VERSION = "2.6.1"
WARMUP_BATCHES = 30
MEASURED_BATCHES = 90
SEEDS = (0, 1, 2, 3, 4)
SIZE_PAIRS = ((256, 64), (512, 64), (256, 128), (512, 512))
BACKENDS = ("pyjuice", "xe", "cirkit")
HEIGHT = WIDTH = 28
CATEGORIES = 256
PATCHES = HEIGHT * WIDTH - 1
HERE = Path(__file__).resolve().parent

CSV_FIELDS = (
    "backend",
    "status",
    "error",
    "pyjuice_version",
    "device_name",
    "seed",
    "batch_size",
    "units",
    "variables",
    "cp_t_patches",
    "parameters",
    "parameter_match",
    "backward_quantity",
    "warmup_batches",
    "measured_batches",
    "forward_ms_per_batch",
    "backward_ms_per_batch",
    "forward_backward_ms_per_batch",
    "forward_microseconds_per_patch_batch",
    "backward_microseconds_per_patch_batch",
    "peak_allocated_memory_bytes",
    "peak_reserved_memory_bytes",
)


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(optimizer: torch.optim.Optimizer) -> int:
    seen: set[int] = set()
    total = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def time_calls(
    forward,
    backward,
    reset_backward,
    batch: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    def run_once() -> tuple[float, float]:
        reset_backward()
        synchronize(device)
        start = time.perf_counter()
        loss = forward(batch)
        synchronize(device)
        forward_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        backward(loss)
        synchronize(device)
        backward_ms = (time.perf_counter() - start) * 1000.0
        return forward_ms, backward_ms

    for _ in range(WARMUP_BATCHES):
        run_once()
    values = [run_once() for _ in range(MEASURED_BATCHES)]
    return [item[0] for item in values], [item[1] for item in values]


def setup_pyjuice(batch_size: int, units: int, device: torch.device):
    root = build_cp_t_quad_tree(
        height=HEIGHT,
        width=WIDTH,
        units=units,
        categories=CATEGORIES,
    )
    metadata = validate_structure(
        root,
        height=HEIGHT,
        width=WIDTH,
        units=units,
        categories=CATEGORIES,
    )
    circuit = juice.compile(
        root,
        layer_sparsity_tol=0.0,
        device=device,
        verbose=False,
    ).to(device)
    circuit._optim_hyperparams["compute_param_flows"] = True
    circuit._optim_hyperparams["flows_memory"] = 1.0

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        return -circuit(
            inputs,
            record_cudagraph=device.type == "cuda",
            apply_cudagraph=True,
        ).mean()

    return (
        forward,
        lambda loss: loss.backward(),
        circuit.zero_param_flows,
        metadata["parameters"],
        "positive-em-parameter-flows",
    )


def setup_torch_backend(
    backend: str,
    batch_size: int,
    units: int,
    device: torch.device,
):
    common = {
        "width": WIDTH,
        "height": HEIGHT,
        "num_units": units,
        "sum_product_layer": "cp-t",
        "region_graph": "quad-tree-2",
        "device": device,
        "dataset": "synthetic",
        "data_dir": "datasets",
        "num_samples": batch_size,
        "pixel_values": CATEGORIES,
        "lr": 0.01,
    }
    if backend == "xe":
        step, optimizer, _images, _program = setup_xe_training(
            **common,
            batch_size=batch_size,
            semiring="scaled-max",
            shift_mode="xe",
            optimize_group_order=True,
            preorder_inputs=True,
            optimize_contraction_paths=True,
        )
    elif backend == "cirkit":
        step, optimizer, _images, _program = setup_cirkit_training(
            **common,
            semiring="lse-sum",
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    return (
        step,
        lambda loss: loss.backward(),
        lambda: optimizer.zero_grad(set_to_none=True),
        parameter_count(optimizer),
        "log-likelihood-logit-gradients",
    )


def run_one(
    backend: str,
    seed: int,
    batch_size: int,
    units: int,
    *,
    output: Path,
    device_arg: str,
    verbose_errors: bool,
) -> None:
    device = get_device(device_arg)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    cleanup_device(device)
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    row = {
        "backend": backend,
        "status": "",
        "error": "",
        "pyjuice_version": importlib.metadata.version("pyjuice"),
        "device_name": get_device_name(device),
        "seed": seed,
        "batch_size": batch_size,
        "units": units,
        "variables": HEIGHT * WIDTH,
        "cp_t_patches": PATCHES,
        "parameter_match": False,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": 0,
    }
    try:
        setup = setup_pyjuice if backend == "pyjuice" else (lambda batch, width, target: setup_torch_backend(backend, batch, width, target))
        forward, backward, reset_backward, parameters, backward_quantity = setup(batch_size, units, device)
        expected = expected_parameters(
            variables=HEIGHT * WIDTH,
            patches=PATCHES,
            units=units,
            categories=CATEGORIES,
        )
        if parameters != expected:
            raise ValueError(f"parameter mismatch: {parameters} != {expected}")
        set_seed(2_000_000 + seed)
        batch = torch.randint(
            CATEGORIES,
            size=(batch_size, HEIGHT * WIDTH),
            device=device,
        )
        forward_values, backward_values = time_calls(
            forward,
            backward,
            reset_backward,
            batch,
            device=device,
        )
        forward_ms = statistics.median(forward_values)
        backward_ms = statistics.median(backward_values)
        row.update(
            {
                "status": "ok",
                "parameters": parameters,
                "parameter_match": True,
                "backward_quantity": backward_quantity,
                "measured_batches": MEASURED_BATCHES,
                "forward_ms_per_batch": forward_ms,
                "backward_ms_per_batch": backward_ms,
                "forward_backward_ms_per_batch": forward_ms + backward_ms,
                "forward_microseconds_per_patch_batch": forward_ms * 1000.0 / PATCHES,
                "backward_microseconds_per_patch_batch": backward_ms * 1000.0 / PATCHES,
                "peak_allocated_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                "peak_reserved_memory_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
            }
        )
        append_row(output, row)
        print(
            f"ok {backend:>7} B={batch_size} U={units} seed={seed}: {forward_ms:.3f} + {backward_ms:.3f} ms",
            flush=True,
        )
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        append_row(output, row)
        print(f"failed {backend} B={batch_size} U={units} seed={seed}: {row['error']}")
        if verbose_errors:
            traceback.print_exc()
        raise
    finally:
        cleanup_device(device)
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact parameter-matched CP-T comparison.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "comparison.csv")
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--_single", nargs=4, help=argparse.SUPPRESS)
    return parser.parse_args()


def completed(path: Path) -> set[tuple[str, int, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} does not use the publication schema")
        return {(row["backend"], int(row["seed"]), int(row["batch_size"]), int(row["units"])) for row in reader if row["status"] == "ok"}


def main() -> None:
    args = parse_args()
    installed_version = importlib.metadata.version("pyjuice")
    if installed_version != PYJUICE_VERSION:
        raise RuntimeError(f"This publication experiment requires PyJuice {PYJUICE_VERSION}; found {installed_version}.")
    if args._single:
        backend, seed, batch, units = args._single
        run_one(
            backend,
            int(seed),
            int(batch),
            int(units),
            output=args.output,
            device_arg=args.device,
            verbose_errors=args.verbose_errors,
        )
        return

    blocks = [(seed, batch, units) for seed in SEEDS for batch, units in SIZE_PAIRS]
    random.Random(20260730).shuffle(blocks)
    configurations: list[tuple[str, int, int, int]] = []
    for block_index, (seed, batch, units) in enumerate(blocks):
        order = BACKENDS if block_index % 2 == 0 else tuple(reversed(BACKENDS))
        configurations.extend((backend, seed, batch, units) for backend in order)
    done = completed(args.output)
    remaining = [item for item in configurations if item not in done]
    print(f"CP-T comparison: {len(remaining)} remaining / {len(configurations)} total")
    for index, (backend, seed, batch, units) in enumerate(remaining, start=1):
        print(f"[{index}/{len(remaining)}] {backend} B={batch} U={units} seed={seed}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--device",
            args.device,
            "--output",
            str(args.output),
            "--_single",
            backend,
            str(seed),
            str(batch),
            str(units),
        ]
        if args.verbose_errors:
            command.append("--verbose-errors")
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
