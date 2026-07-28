from __future__ import annotations

# This file is also executed directly from experiments/monarch/.
# ruff: noqa: E402,I001

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

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIRECTORY.parents[1]
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]
sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader, Sampler

from demo.cirkit import (
    cleanup_device,
    elapsed_timer_ms,
    get_device,
    get_device_name,
    memory_snapshot,
    set_seed,
    start_timer,
    stop_timer,
    synchronize_device,
)
from experiments.common import parse_ints
from experiments.monarch.imagenet64 import ImageNet64Dataset
from experiments.monarch.model import setup_cirkit, setup_xe

WIDTH = HEIGHT = 64
VARIABLES = WIDTH * HEIGHT
CATEGORIES = 256
WARMUP_BATCHES = 30
MEASURED_EPOCHS = 3
BATCHES_PER_EPOCH = 30
MEASURED_BATCHES = MEASURED_EPOCHS * BATCHES_PER_EPOCH
SEEDS = (0, 1, 2, 3, 4)
BACKENDS = ("cirkit", "xe")
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Scale:
    region_graph: str
    parameterization: str
    batch_size: int
    units: int
    p: int = 0
    q: int = 0


# Conservative scales selected for full 64x64 grayscale inputs on a 48 GiB GPU.
# Monarch uses a larger width because its hidden sums use H(P+Q), not H^2.
SCALES = (
    Scale("quad-tree-2", "dense", 256, 128),
    Scale("quad-tree-2", "monarch", 256, 512, 16, 32),
    Scale("quad-graph", "dense", 256, 64),
    Scale("quad-graph", "monarch", 256, 256, 16, 16),
)

CSV_FIELDS = (
    "backend",
    "status",
    "error",
    "device_name",
    "seed",
    "region_graph",
    "parameterization",
    "fold_strategy",
    "dataset",
    "color_transform",
    "data_workers",
    "sample_limit",
    "width",
    "height",
    "variables",
    "categories",
    "units",
    "batch_size",
    "monarch_p",
    "monarch_q",
    "monarch_layers",
    "parameters",
    "initialization_hash",
    "warmup_batches",
    "measured_batches",
    "forward_loss_ms_per_batch",
    "backward_ms_per_batch",
    "forward_backward_ms_per_batch",
    "optimizer_step_ms_per_batch",
    "train_step_ms_per_batch",
    "peak_allocated_memory_bytes",
    "peak_reserved_memory_bytes",
)

RAW_CSV_FIELDS = (
    "backend",
    "device_name",
    "seed",
    "region_graph",
    "parameterization",
    "fold_strategy",
    "dataset",
    "color_transform",
    "data_workers",
    "dataset_examples",
    "width",
    "height",
    "variables",
    "categories",
    "units",
    "batch_size",
    "monarch_p",
    "monarch_q",
    "monarch_layers",
    "parameters",
    "initialization_hash",
    "measurement_block",
    "batch_in_block",
    "measured_batch",
    "samples",
    "data_loading_ms",
    "forward_loss_ms",
    "backward_ms",
    "optimizer_step_ms",
    "zero_grad_ms",
    "train_step_ms",
    "end_to_end_ms",
    "loss",
)


class TensorIndexSampler(Sampler[int]):
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices

    def __iter__(self):
        return (int(index) for index in self.indices)

    def __len__(self) -> int:
        return self.indices.numel()


@dataclass(frozen=True)
class Configuration:
    backend: str
    seed: int
    scale: Scale

    @property
    def key(self) -> tuple[str, int, str, str, int, int]:
        return (
            self.backend,
            self.seed,
            self.scale.region_graph,
            self.scale.parameterization,
            self.scale.batch_size,
            self.scale.units,
        )

    def child_arguments(self) -> list[str]:
        return [
            "--_single",
            self.backend,
            str(self.seed),
            self.scale.region_graph,
            self.scale.parameterization,
            str(self.scale.batch_size),
            str(self.scale.units),
            str(self.scale.p),
            str(self.scale.q),
        ]


def parse_names(
    value: str,
    *,
    allowed: frozenset[str],
    description: str,
) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or set(names) - allowed:
        raise ValueError(f"expected comma-separated {description}")
    return names


def parse_single(values: list[str]) -> Configuration:
    if len(values) != 8:
        raise ValueError(
            "internal configuration requires "
            "BACKEND SEED GRAPH PARAMETERIZATION BATCH UNITS P Q"
        )
    backend, seed, graph, parameterization, batch, units, p, q = values
    scale = Scale(
        region_graph=graph,
        parameterization=parameterization,
        batch_size=int(batch),
        units=int(units),
        p=int(p),
        q=int(q),
    )
    configuration = Configuration(backend, int(seed), scale)
    if (
        backend not in BACKENDS
        or configuration.seed < 0
        or graph not in {"quad-tree-2", "quad-graph"}
        or parameterization not in {"dense", "monarch"}
        or scale.batch_size <= 0
        or scale.units <= 0
        or (
            parameterization == "monarch"
            and (scale.p <= 0 or scale.q <= 0 or scale.p * scale.q != scale.units)
        )
        or (parameterization == "dense" and (scale.p or scale.q))
    ):
        raise ValueError("invalid internal Monarch configuration")
    return configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-image CP benchmark: Cirkit native folding versus XE "
            "input-depth folding, for dense and Monarch sums."
        )
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--graphs", default="quad-tree-2,quad-graph")
    parser.add_argument("--parameterizations", default="dense,monarch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Optional deterministic source-image subset; zero uses the full split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "performance.csv",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=HERE / "results" / "performance_batches.csv",
    )
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--_single", nargs=8, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.seeds = parse_ints(args.seeds)
        args.graphs = parse_names(
            args.graphs,
            allowed=frozenset({"quad-tree-2", "quad-graph"}),
            description="region graphs",
        )
        args.parameterizations = parse_names(
            args.parameterizations,
            allowed=frozenset({"dense", "monarch"}),
            description="parameterizations",
        )
        args.single = parse_single(args._single) if args._single else None
        if args.data_workers < 0 or args.sample_limit < 0:
            raise ValueError("data workers and sample limit must be non-negative")
    except ValueError as error:
        parser.error(str(error))
    return args


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def append_raw_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as input_file:
            header = tuple(next(csv.reader(input_file), ()))
        if header != RAW_CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the Monarch raw-batch schema. "
                "Move or remove it before running."
            )
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in RAW_CSV_FIELDS}
            for row in rows
        )


def completed_keys(
    path: Path,
    raw_path: Path,
) -> set[tuple[str, int, str, str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, int, str, str, int, int]] = set()
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the Monarch publication schema. "
                "Move or remove it before running."
            )
        for row in reader:
            if row["status"] == "ok":
                completed.add(
                    (
                        row["backend"],
                        int(row["seed"]),
                        row["region_graph"],
                        row["parameterization"],
                        int(row["batch_size"]),
                        int(row["units"]),
                    )
                )
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    raw_batches: dict[
        tuple[str, int, str, str, int, int],
        set[int],
    ] = {}
    with raw_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != RAW_CSV_FIELDS:
            raise ValueError(
                f"{raw_path} does not use the Monarch raw-batch schema. "
                "Move or remove it before running."
            )
        for row in reader:
            key = (
                row["backend"],
                int(row["seed"]),
                row["region_graph"],
                row["parameterization"],
                int(row["batch_size"]),
                int(row["units"]),
            )
            raw_batches.setdefault(key, set()).add(int(row["measured_batch"]))
    return {
        key
        for key in completed
        if len(raw_batches.get(key, set())) >= MEASURED_BATCHES
    }


def base_row(
    configuration: Configuration,
    *,
    device: torch.device,
    data_workers: int = 0,
    sample_limit: int = 0,
) -> dict[str, object]:
    scale = configuration.scale
    return {
        "backend": configuration.backend,
        "status": "",
        "error": "",
        "device_name": get_device_name(device),
        "seed": configuration.seed,
        "region_graph": scale.region_graph,
        "parameterization": scale.parameterization,
        "fold_strategy": (
            "cirkit-native"
            if configuration.backend == "cirkit"
            else "xe-input-depth"
        ),
        "dataset": "imagenet64",
        "color_transform": "grayscale",
        "data_workers": data_workers,
        "sample_limit": sample_limit,
        "width": WIDTH,
        "height": HEIGHT,
        "variables": VARIABLES,
        "categories": CATEGORIES,
        "units": scale.units,
        "batch_size": scale.batch_size,
        "monarch_p": scale.p or "",
        "monarch_q": scale.q or "",
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": 0,
    }


def run_block(
    training,
    dataset: ImageNet64Dataset,
    *,
    configuration: Configuration,
    device: torch.device,
    data_workers: int,
    epoch: int,
    batches: int,
    measurement_block: int | None,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    scale = configuration.scale
    dataset.set_epoch(epoch)
    indices = dataset.epoch_indices(
        seed=4_000_000 + configuration.seed,
        epoch=epoch,
    )
    required = scale.batch_size * batches
    if indices.numel() < required:
        raise ValueError(
            f"ImageNet64 selection has {indices.numel()} examples, "
            f"but {required} are required for one benchmark block"
        )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": scale.batch_size,
        "sampler": TensorIndexSampler(indices[:required]),
        "num_workers": data_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if data_workers:
        loader_kwargs["prefetch_factor"] = 2
    iterator = iter(DataLoader(**loader_kwargs))
    totals: dict[str, float | int] = {
        "data_loading_ms": 0.0,
        "forward_loss_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_step_ms": 0.0,
        "zero_grad_ms": 0.0,
        "end_to_end_ms": 0.0,
        "batches": 0,
    }
    raw_rows: list[dict[str, object]] = []

    for batch_index in range(batches):
        synchronize_device(device)
        batch_wall_start = time.perf_counter()
        data_start = time.perf_counter()
        batch = next(iterator).to(
            device,
            non_blocking=device.type == "cuda",
        )
        synchronize_device(device)
        data_loading_ms = (time.perf_counter() - data_start) * 1000.0

        forward_start = start_timer(device)
        loss = training.step(batch)
        forward_end = stop_timer(device)
        backward_start = start_timer(device)
        loss.backward()
        backward_end = stop_timer(device)
        optimizer_start = start_timer(device)
        training.optimizer.step()
        optimizer_end = stop_timer(device)
        zero_grad_start = start_timer(device)
        training.optimizer.zero_grad(set_to_none=True)
        zero_grad_end = stop_timer(device)
        synchronize_device(device)

        forward_ms = elapsed_timer_ms(forward_start, forward_end, device)
        backward_ms = elapsed_timer_ms(backward_start, backward_end, device)
        optimizer_ms = elapsed_timer_ms(optimizer_start, optimizer_end, device)
        zero_grad_ms = elapsed_timer_ms(
            zero_grad_start,
            zero_grad_end,
            device,
        )
        end_to_end_ms = (time.perf_counter() - batch_wall_start) * 1000.0
        values = {
            "data_loading_ms": data_loading_ms,
            "forward_loss_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_step_ms": optimizer_ms,
            "zero_grad_ms": zero_grad_ms,
            "end_to_end_ms": end_to_end_ms,
        }
        for field, value in values.items():
            totals[field] = float(totals[field]) + value
        totals["batches"] = int(totals["batches"]) + 1

        if measurement_block is not None:
            raw_rows.append(
                {
                    "backend": configuration.backend,
                    "device_name": get_device_name(device),
                    "seed": configuration.seed,
                    "region_graph": scale.region_graph,
                    "parameterization": scale.parameterization,
                    "fold_strategy": (
                        "cirkit-native"
                        if configuration.backend == "cirkit"
                        else "xe-input-depth"
                    ),
                    "dataset": "imagenet64",
                    "color_transform": "grayscale",
                    "data_workers": data_workers,
                    "dataset_examples": len(dataset),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "variables": VARIABLES,
                    "categories": CATEGORIES,
                    "units": scale.units,
                    "batch_size": scale.batch_size,
                    "monarch_p": scale.p or "",
                    "monarch_q": scale.q or "",
                    "monarch_layers": training.monarch_layers,
                    "parameters": training.parameters,
                    "initialization_hash": training.initialization_hash,
                    "measurement_block": measurement_block,
                    "batch_in_block": batch_index,
                    "measured_batch": (
                        measurement_block * BATCHES_PER_EPOCH
                        + batch_index
                    ),
                    "samples": batch.shape[0],
                    **values,
                    "train_step_ms": (
                        forward_ms
                        + backward_ms
                        + optimizer_ms
                        + zero_grad_ms
                    ),
                    "loss": float(loss.detach()),
                }
            )
    return totals, raw_rows


def run_configuration(
    configuration: Configuration,
    *,
    output: Path,
    raw_output: Path,
    device_arg: str,
    data_dir: Path,
    data_workers: int,
    sample_limit: int,
    verbose_errors: bool,
) -> None:
    device = get_device(device_arg)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    cleanup_device(device)
    set_seed(configuration.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    row = base_row(
        configuration,
        device=device,
        data_workers=data_workers,
        sample_limit=sample_limit,
    )
    scale = configuration.scale
    try:
        setup = setup_cirkit if configuration.backend == "cirkit" else setup_xe
        training = setup(
            width=WIDTH,
            height=HEIGHT,
            units=scale.units,
            categories=CATEGORIES,
            batch_size=scale.batch_size,
            region_graph=scale.region_graph,
            parameterization=scale.parameterization,
            factors=(scale.p, scale.q) if scale.p else None,
            seed=configuration.seed,
            device=device,
        )

        dataset = ImageNet64Dataset(
            data_dir,
            split="train",
            color_transform="grayscale",
            patch_size=None,
            patch_mode="random",
            sample_limit=sample_limit,
            sample_seed=configuration.seed,
            output_layout="flat",
        )
        run_block(
            training,
            dataset,
            configuration=configuration,
            device=device,
            data_workers=data_workers,
            epoch=0,
            batches=WARMUP_BATCHES,
            measurement_block=None,
        )
        measurements = []
        raw_rows: list[dict[str, object]] = []
        for block in range(MEASURED_EPOCHS):
            measurement, block_rows = run_block(
                training,
                dataset,
                configuration=configuration,
                device=device,
                data_workers=data_workers,
                epoch=block + 1,
                batches=BATCHES_PER_EPOCH,
                measurement_block=block,
            )
            measurements.append(measurement)
            raw_rows.extend(block_rows)

        def median_per_batch(field: str) -> float:
            return statistics.median(
                float(item[field]) / int(item["batches"])
                for item in measurements
            )

        measured_batches = sum(int(item["batches"]) for item in measurements)
        forward = median_per_batch("forward_loss_ms")
        backward = median_per_batch("backward_ms")
        optimizer_step = median_per_batch("optimizer_step_ms")
        memory = memory_snapshot(device)
        row.update(
            {
                "status": "ok",
                "monarch_layers": training.monarch_layers,
                "parameters": training.parameters,
                "initialization_hash": training.initialization_hash,
                "measured_batches": measured_batches,
                "forward_loss_ms_per_batch": forward,
                "backward_ms_per_batch": backward,
                "forward_backward_ms_per_batch": forward + backward,
                "optimizer_step_ms_per_batch": optimizer_step,
                "train_step_ms_per_batch": (
                    forward + backward + optimizer_step
                ),
                "peak_allocated_memory_bytes": memory["peak_memory_bytes"],
                "peak_reserved_memory_bytes": memory[
                    "peak_reserved_memory_bytes"
                ],
            }
        )
        append_raw_rows(raw_output, raw_rows)
        append_row(output, row)
        print(
            f"ok {configuration.backend:>6} {scale.parameterization:>7} "
            f"{scale.region_graph:>11} B={scale.batch_size:<3} "
            f"H={scale.units:<4} seed={configuration.seed}: "
            f"{forward + backward:.3f} ms/batch",
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
        cleanup_device(device)
        gc.collect()


def configurations(args: argparse.Namespace) -> list[Configuration]:
    blocks = [
        (seed, scale)
        for seed in args.seeds
        for scale in SCALES
        if scale.region_graph in args.graphs
        and scale.parameterization in args.parameterizations
    ]
    random.Random(20260728).shuffle(blocks)
    result: list[Configuration] = []
    for index, (seed, scale) in enumerate(blocks):
        backends = BACKENDS if index % 2 == 0 else tuple(reversed(BACKENDS))
        result.extend(Configuration(backend, seed, scale) for backend in backends)
    return result


def run_isolated(
    script: Path,
    configurations_: list[Configuration],
    *,
    output: Path,
    raw_output: Path,
    device: str,
    data_dir: Path,
    data_workers: int,
    sample_limit: int,
    verbose_errors: bool,
) -> None:
    completed = completed_keys(output, raw_output)
    remaining = [
        configuration
        for configuration in configurations_
        if configuration.key not in completed
    ]
    print(
        f"Monarch benchmark: {len(remaining)} remaining / "
        f"{len(configurations_)} total; output={output}",
        flush=True,
    )
    for index, configuration in enumerate(remaining, start=1):
        print(f"[{index}/{len(remaining)}] {configuration}", flush=True)
        command = [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--device",
            device,
            "--data-dir",
            str(data_dir),
            "--data-workers",
            str(data_workers),
            "--sample-limit",
            str(sample_limit),
            "--raw-output",
            str(raw_output),
            *configuration.child_arguments(),
        ]
        if verbose_errors:
            command.append("--verbose-errors")
        subprocess.run(command, check=False)


def main() -> None:
    args = parse_args()
    if args.single:
        run_configuration(
            args.single,
            output=args.output,
            raw_output=args.raw_output,
            device_arg=args.device,
            data_dir=args.data_dir,
            data_workers=args.data_workers,
            sample_limit=args.sample_limit,
            verbose_errors=args.verbose_errors,
        )
        return
    run_isolated(
        Path(__file__).resolve(),
        configurations(args),
        output=args.output,
        raw_output=args.raw_output,
        device=args.device,
        data_dir=args.data_dir,
        data_workers=args.data_workers,
        sample_limit=args.sample_limit,
        verbose_errors=args.verbose_errors,
    )


if __name__ == "__main__":
    main()
