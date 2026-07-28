from __future__ import annotations

import csv
import gc
import random
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from demo.cirkit import (
    cleanup_device,
    get_device,
    get_device_name,
    memory_snapshot,
    reset_peak_memory,
    run_epoch,
    run_warmup_epochs,
    set_seed,
    setup_cirkit_training,
    setup_xe_training,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

WARMUP_BATCHES = 30
MEASURED_EPOCHS = 3
BATCHES_PER_EPOCH = 30
MEASURED_BATCHES = MEASURED_EPOCHS * BATCHES_PER_EPOCH
SEEDS = (0, 1, 2, 3, 4)

CSV_FIELDS = (
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
    "forward_loss_ms_per_batch",
    "backward_ms_per_batch",
    "forward_backward_ms_per_batch",
    "optimizer_step_ms_per_batch",
    "train_step_ms_per_batch",
    "peak_allocated_memory_bytes",
    "peak_reserved_memory_bytes",
)


@dataclass(frozen=True)
class Variant:
    backend: str
    semiring: str = "scaled-max"
    shift_mode: str = "xe"
    optimize_group_order: bool = True


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
        return (
            self.variant,
            self.seed,
            self.region_graph,
            self.layer,
            self.batch_size,
            self.units,
        )

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


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("expected one or more non-negative integers")
    return parsed


def parse_layers(value: str) -> tuple[str, ...]:
    layers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not layers or set(layers) - {"cp", "tucker"}:
        raise ValueError("expected cp, tucker, or cp,tucker")
    return layers


def parse_single(values: list[str]) -> Configuration:
    if len(values) != 6:
        raise ValueError("internal configuration requires SEED GRAPH LAYER BATCH UNITS VARIANT")
    seed, graph, layer, batch, units, variant = values
    configuration = Configuration(
        seed=int(seed),
        region_graph=graph,
        layer=layer,
        batch_size=int(batch),
        units=int(units),
        variant=variant,
    )
    if (
        configuration.seed < 0
        or configuration.region_graph not in {"quad-tree-2", "quad-graph"}
        or configuration.layer not in {"cp", "tucker"}
        or configuration.batch_size <= 0
        or configuration.units <= 0
    ):
        raise ValueError("invalid internal configuration")
    return configuration


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def completed_keys(path: Path) -> set[tuple[str, int, str, str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, int, str, str, int, int]] = set()
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} does not use the publication schema. Move or remove it before running.")
        for row in reader:
            if row["status"] != "ok":
                continue
            completed.add(
                (
                    row["variant"],
                    int(row["seed"]),
                    row["region_graph"],
                    row["layer"],
                    int(row["batch_size"]),
                    int(row["units"]),
                )
            )
    return completed


def setup_training(
    configuration: Configuration,
    variant: Variant,
    *,
    device: torch.device,
):
    common = {
        "width": 28,
        "height": 28,
        "num_units": configuration.units,
        "sum_product_layer": configuration.layer,
        "region_graph": configuration.region_graph,
        "device": device,
        "dataset": "mnist",
        "data_dir": "datasets",
        "num_samples": 0,
        "pixel_values": 256,
        "lr": 0.01,
    }
    if variant.backend == "cirkit":
        return setup_cirkit_training(**common, semiring="lse-sum")
    if variant.backend != "xe":
        raise ValueError(f"unknown backend: {variant.backend}")
    return setup_xe_training(
        **common,
        batch_size=configuration.batch_size,
        semiring=variant.semiring,
        shift_mode=variant.shift_mode,
        optimize_group_order=variant.optimize_group_order,
        preorder_inputs=True,
        optimize_contraction_paths=True,
    )


def base_row(
    configuration: Configuration,
    *,
    device: torch.device,
) -> dict[str, object]:
    return {
        "variant": configuration.variant,
        "status": "",
        "error": "",
        "device_name": get_device_name(device),
        "seed": configuration.seed,
        "region_graph": configuration.region_graph,
        "layer": configuration.layer,
        "units": configuration.units,
        "batch_size": configuration.batch_size,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": 0,
    }


def run_configuration(
    configuration: Configuration,
    variants: dict[str, Variant],
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
    set_seed(configuration.seed)
    reset_peak_memory(device)
    row = base_row(configuration, device=device)
    try:
        step, optimizer, train_images, _program = setup_training(
            configuration,
            variants[configuration.variant],
            device=device,
        )
        # Model construction consumes a backend-dependent amount of randomness.
        # Reset before training to pair minibatch permutations across variants.
        set_seed(1_000_000 + configuration.seed)
        run_warmup_epochs(
            step,
            optimizer,
            train_images,
            batch_size=configuration.batch_size,
            max_batches=WARMUP_BATCHES,
            warmup_epochs=1,
            device=device,
        )
        measurements = [
            run_epoch(
                step,
                optimizer,
                train_images,
                batch_size=configuration.batch_size,
                max_batches=BATCHES_PER_EPOCH,
                device=device,
            )
            for _ in range(MEASURED_EPOCHS)
        ]
        measured_batches = sum(int(item["batches"]) for item in measurements)

        def median_per_batch(field: str) -> float:
            return statistics.median(float(item[field]) / int(item["batches"]) for item in measurements)

        forward = median_per_batch("forward_loss_ms")
        backward = median_per_batch("backward_ms")
        optimizer_step = median_per_batch("optimizer_step_ms")
        memory = memory_snapshot(device)
        row.update(
            {
                "status": "ok",
                "measured_batches": measured_batches,
                "forward_loss_ms_per_batch": forward,
                "backward_ms_per_batch": backward,
                "forward_backward_ms_per_batch": forward + backward,
                "optimizer_step_ms_per_batch": optimizer_step,
                "train_step_ms_per_batch": forward + backward + optimizer_step,
                "peak_allocated_memory_bytes": memory["peak_memory_bytes"],
                "peak_reserved_memory_bytes": memory["peak_reserved_memory_bytes"],
            }
        )
        append_row(output, row)
        print(
            f"ok {configuration.variant:>28} {configuration.layer:>6} "
            f"{configuration.region_graph:>11} B={configuration.batch_size:<3} "
            f"U={configuration.units:<4} seed={configuration.seed}: "
            f"{forward + backward:.3f} ms/batch",
            flush=True,
        )
    except Exception as error:
        row.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        append_row(output, row)
        print(f"failed {configuration}: {row['error']}", flush=True)
        if verbose_errors:
            traceback.print_exc()
        raise
    finally:
        cleanup_device(device)
        gc.collect()


def run_isolated(
    script: Path,
    configurations: list[Configuration],
    *,
    output: Path,
    device: str,
    verbose_errors: bool,
) -> None:
    completed = completed_keys(output)
    remaining = [configuration for configuration in configurations if configuration.key not in completed]
    print(
        f"publication benchmark: {len(remaining)} remaining / {len(configurations)} total; output={output}",
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
            *configuration.child_arguments(),
        ]
        if verbose_errors:
            command.append("--verbose-errors")
        subprocess.run(command, check=False)


def shuffled_blocks(
    blocks: list[tuple[int, str, str, int, int]],
    variants: tuple[str, ...],
    *,
    shuffle_seed: int,
) -> list[Configuration]:
    random.Random(shuffle_seed).shuffle(blocks)
    configurations: list[Configuration] = []
    for block_index, (seed, graph, layer, batch, units) in enumerate(blocks):
        ordered_variants = variants if block_index % 2 == 0 else tuple(reversed(variants))
        configurations.extend(Configuration(seed, graph, layer, batch, units, variant) for variant in ordered_variants)
    return configurations
