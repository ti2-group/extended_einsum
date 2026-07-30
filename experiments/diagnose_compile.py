from __future__ import annotations

# This file is also executed directly from experiments/.
# ruff: noqa: E402
import argparse
import csv
import gc
import os
import random
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch

import demo.cirkit as demo_cirkit
import experiments.monarch.model as monarch_model
from experiments.ablation import ABLATION_GRID, REGION_GRAPHS, VARIANTS
from experiments.common import parse_ints, parse_layers
from experiments.monarch.benchmark import CATEGORIES, HEIGHT, SCALES, WIDTH
from extended_einsum.preprocess import (
    FoldSameShapedOperations,
    OptimizeContractionPaths,
)

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = tuple(range(5))
SUITES = ("ablation", "monarch")
CSV_FIELDS = (
    "timestamp",
    "suite",
    "system",
    "variant",
    "run",
    "status",
    "error",
    "device_name",
    "torch_version",
    "cuda_version",
    "region_graph",
    "layer",
    "parameterization",
    "width",
    "height",
    "units",
    "batch_size",
    "monarch_p",
    "monarch_q",
    "setup_seconds",
    "xe_extraction_seconds",
    "folding_seconds",
    "contraction_path_seconds",
    "backend_lowering_seconds",
    "our_passes_seconds",
    "cirkit_lowering_seconds",
    "torch_compile_seconds",
    "compile_total_seconds",
    "runtime_instructions",
    "peak_rss_bytes",
)


@dataclass(frozen=True)
class Configuration:
    suite: str
    system: str
    variant: str
    run: int
    region_graph: str
    layer: str
    parameterization: str
    width: int
    height: int
    units: int
    batch_size: int
    p: int = 0
    q: int = 0

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.suite,
            self.system,
            self.variant,
            self.run,
            self.region_graph,
            self.layer,
            self.parameterization,
            self.width,
            self.height,
            self.units,
            self.batch_size,
            self.p,
            self.q,
        )

    def child_arguments(self) -> list[str]:
        return [
            "--_single",
            self.suite,
            self.system,
            self.variant,
            str(self.run),
            self.region_graph,
            self.layer,
            self.parameterization,
            str(self.width),
            str(self.height),
            str(self.units),
            str(self.batch_size),
            str(self.p),
            str(self.q),
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
    if len(values) != 13:
        raise ValueError(
            "internal configuration requires SUITE SYSTEM VARIANT RUN GRAPH "
            "LAYER PARAMETERIZATION WIDTH HEIGHT UNITS BATCH P Q"
        )
    (
        suite,
        system,
        variant,
        run,
        graph,
        layer,
        parameterization,
        width,
        height,
        units,
        batch,
        p,
        q,
    ) = values
    configuration = Configuration(
        suite=suite,
        system=system,
        variant=variant,
        run=int(run),
        region_graph=graph,
        layer=layer,
        parameterization=parameterization,
        width=int(width),
        height=int(height),
        units=int(units),
        batch_size=int(batch),
        p=int(p),
        q=int(q),
    )
    if (
        suite not in SUITES
        or system not in {"xe", "cirkit"}
        or configuration.run < 0
        or graph not in REGION_GRAPHS
        or layer not in {"cp", "tucker"}
        or parameterization not in {"dense", "monarch"}
        or min(
            configuration.width,
            configuration.height,
            configuration.units,
            configuration.batch_size,
        )
        <= 0
        or (
            parameterization == "monarch"
            and configuration.p * configuration.q != configuration.units
        )
        or (
            parameterization == "dense"
            and (configuration.p or configuration.q)
        )
        or (suite == "ablation" and parameterization != "dense")
        or (suite == "ablation" and variant not in VARIANTS)
        or (suite == "ablation" and VARIANTS[variant].backend != system)
        or (suite == "monarch" and variant != system)
        or (suite == "monarch" and layer != "cp")
    ):
        raise ValueError("invalid internal compile configuration")
    return configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process-isolated compiler breakdown for every publication "
            "ablation configuration and dense/Monarch matrix."
        )
    )
    parser.add_argument("--suites", default="ablation,monarch")
    parser.add_argument("--runs", default="0,1,2,3,4")
    parser.add_argument("--layers", default="cp,tucker")
    parser.add_argument("--graphs", default="quad-tree-2,quad-graph")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--parameterizations", default="dense,monarch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "compile_breakdown.csv",
    )
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--_single", nargs=13, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.suites = parse_names(
            args.suites,
            allowed=frozenset(SUITES),
            description="suites",
        )
        args.runs = parse_ints(args.runs)
        args.layers = parse_layers(args.layers)
        args.graphs = parse_names(
            args.graphs,
            allowed=frozenset(REGION_GRAPHS),
            description="region graphs",
        )
        args.variants = parse_names(
            args.variants,
            allowed=frozenset(VARIANTS),
            description="ablation variants",
        )
        args.parameterizations = parse_names(
            args.parameterizations,
            allowed=frozenset({"dense", "monarch"}),
            description="parameterizations",
        )
        args.single = parse_single(args._single) if args._single else None
        if args.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
    except ValueError as error:
        parser.error(str(error))
    return args


def configurations(args: argparse.Namespace) -> list[Configuration]:
    result: list[Configuration] = []
    if "ablation" in args.suites:
        for run in args.runs:
            for layer in args.layers:
                for graph in args.graphs:
                    for batch, units in ABLATION_GRID[layer]:
                        for variant in args.variants:
                            result.append(
                                Configuration(
                                    "ablation",
                                    VARIANTS[variant].backend,
                                    variant,
                                    run,
                                    graph,
                                    layer,
                                    "dense",
                                    28,
                                    28,
                                    units,
                                    batch,
                                )
                            )
    if "monarch" in args.suites:
        for run in args.runs:
            for scale in SCALES:
                if (
                    scale.region_graph not in args.graphs
                    or scale.parameterization not in args.parameterizations
                ):
                    continue
                for system in ("xe", "cirkit"):
                    result.append(
                        Configuration(
                            "monarch",
                            system,
                            system,
                            run,
                            scale.region_graph,
                            "cp",
                            scale.parameterization,
                            WIDTH,
                            HEIGHT,
                            scale.units,
                            scale.batch_size,
                            scale.p,
                            scale.q,
                        )
                    )
    random.Random(20260730).shuffle(result)
    return result


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as input_file:
            header = tuple(next(csv.reader(input_file), ()))
        if header != CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the compile-breakdown schema. "
                "Move or remove it before running."
            )
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def completed_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the compile-breakdown schema. "
                "Move or remove it before running."
            )
        return {
            row_key(row)
            for row in reader
            if row["status"] == "ok"
        }


def row_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        row["suite"],
        row["system"],
        row["variant"],
        int(row["run"]),
        row["region_graph"],
        row["layer"],
        row["parameterization"],
        int(row["width"]),
        int(row["height"]),
        int(row["units"]),
        int(row["batch_size"]),
        int(row["monarch_p"] or 0),
        int(row["monarch_q"] or 0),
    )


def get_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@contextmanager
def patched_attribute(target: object, name: str, value: object) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def timed_function(
    function: Callable[..., Any],
    timings: dict[str, float],
    name: str,
) -> Callable[..., Any]:
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + (
                time.perf_counter() - started
            )

    return wrapped


def timed_fold_class(timings: dict[str, float]):
    class TimedFold:
        @staticmethod
        def apply_with_input_depth_metadata(*args, **kwargs):
            function = timed_function(
                FoldSameShapedOperations.apply_with_input_depth_metadata,
                timings,
                "folding_seconds",
            )
            return function(*args, **kwargs)

    return TimedFold


def timed_path_class(timings: dict[str, float]):
    class TimedPaths:
        @staticmethod
        def apply(*args, **kwargs):
            function = timed_function(
                OptimizeContractionPaths.apply,
                timings,
                "contraction_path_seconds",
            )
            return function(*args, **kwargs)

    return TimedPaths


def timed_pipeline_context(timings: dict[str, float]):
    original_class = demo_cirkit.PipelineContext

    class TimedContext:
        def __init__(self, *args, **kwargs) -> None:
            self.context = original_class(*args, **kwargs)

        def compile(self, *args, **kwargs):
            function = timed_function(
                self.context.compile,
                timings,
                "cirkit_lowering_seconds",
            )
            return function(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.context, name)

    return TimedContext


def instrument_xe(
    module,
    timings: dict[str, float],
    *,
    translation_name: str,
) -> ExitStack:
    stack = ExitStack()
    translation = getattr(module, translation_name)
    stack.enter_context(
        patched_attribute(
            module,
            translation_name,
            timed_function(
                translation,
                timings,
                "xe_extraction_seconds",
            ),
        )
    )
    stack.enter_context(
        patched_attribute(
            module,
            "FoldSameShapedOperations",
            timed_fold_class(timings),
        )
    )
    stack.enter_context(
        patched_attribute(
            module,
            "OptimizeContractionPaths",
            timed_path_class(timings),
        )
    )
    stack.enter_context(
        patched_attribute(
            module,
            "torch_program_runner",
            timed_function(
                module.torch_program_runner,
                timings,
                "backend_lowering_seconds",
            ),
        )
    )
    return stack


def setup_ablation(
    configuration: Configuration,
    *,
    device: torch.device,
    timings: dict[str, float],
):
    variant = VARIANTS[configuration.variant]
    common = {
        "width": configuration.width,
        "height": configuration.height,
        "num_units": configuration.units,
        "sum_product_layer": configuration.layer,
        "region_graph": configuration.region_graph,
        "device": device,
        "dataset": "synthetic",
        "data_dir": "",
        "num_samples": configuration.batch_size,
        "pixel_values": 256,
        "lr": 0.01,
    }
    if configuration.system == "xe":
        with instrument_xe(
            demo_cirkit,
            timings,
            translation_name="translate_cirkit_to_xe",
        ):
            step, _optimizer, images, program = demo_cirkit.setup_xe_training(
                **common,
                batch_size=configuration.batch_size,
                semiring=variant.semiring,
                shift_mode=variant.shift_mode,
                optimize_group_order=variant.optimize_group_order,
                preorder_inputs=True,
                optimize_contraction_paths=True,
            )
    else:
        with patched_attribute(
            demo_cirkit,
            "PipelineContext",
            timed_pipeline_context(timings),
        ):
            step, _optimizer, images, program = (
                demo_cirkit.setup_cirkit_training(
                    **common,
                    semiring="lse-sum",
                )
            )
    return step, images[: configuration.batch_size], program


def setup_monarch(
    configuration: Configuration,
    *,
    device: torch.device,
    timings: dict[str, float],
):
    factors = (configuration.p, configuration.q) if configuration.p else None
    common = {
        "width": configuration.width,
        "height": configuration.height,
        "units": configuration.units,
        "categories": CATEGORIES,
        "batch_size": configuration.batch_size,
        "region_graph": configuration.region_graph,
        "parameterization": configuration.parameterization,
        "factors": factors,
        "seed": configuration.run,
        "device": device,
    }
    if configuration.system == "xe":
        with instrument_xe(
            monarch_model,
            timings,
            translation_name="translate_to_xe",
        ):
            training = monarch_model.setup_xe(**common)
    else:
        original_class = monarch_model.PipelineContext

        class TimedContext:
            def __init__(self, *args, **kwargs) -> None:
                self.context = original_class(*args, **kwargs)

            def compile(self, *args, **kwargs):
                function = timed_function(
                    self.context.compile,
                    timings,
                    "cirkit_lowering_seconds",
                )
                return function(*args, **kwargs)

            def __getattr__(self, name: str):
                return getattr(self.context, name)

        with patched_attribute(
            monarch_model,
            "PipelineContext",
            TimedContext,
        ):
            training = monarch_model.setup_cirkit(**common)
    batch = torch.randint(
        CATEGORIES,
        (configuration.batch_size, configuration.width * configuration.height),
        device=device,
    )
    return training.step, batch, training.runtime_program


def base_row(
    configuration: Configuration,
    *,
    device: torch.device | None,
) -> dict[str, object]:
    if device is None:
        device_name = ""
    elif device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = device.type
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "suite": configuration.suite,
        "system": configuration.system,
        "variant": configuration.variant,
        "run": configuration.run,
        "status": "",
        "error": "",
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "region_graph": configuration.region_graph,
        "layer": configuration.layer,
        "parameterization": configuration.parameterization,
        "width": configuration.width,
        "height": configuration.height,
        "units": configuration.units,
        "batch_size": configuration.batch_size,
        "monarch_p": configuration.p or "",
        "monarch_q": configuration.q or "",
    }


def run_configuration(
    configuration: Configuration,
    *,
    device_arg: str,
    output: Path,
    verbose_errors: bool,
) -> int:
    device = get_device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(configuration.run)
    timings: dict[str, float] = {}
    row = base_row(configuration, device=device)
    try:
        setup_started = time.perf_counter()
        if configuration.suite == "ablation":
            step, batch, program = setup_ablation(
                configuration,
                device=device,
                timings=timings,
            )
        else:
            step, batch, program = setup_monarch(
                configuration,
                device=device,
                timings=timings,
            )
        synchronize(device)
        timings["setup_seconds"] = time.perf_counter() - setup_started

        # torch.compile is lazy. The first forward and backward are both
        # required to materialize the same compiled training graph used by the
        # performance experiments.
        compile_started = time.perf_counter()
        loss = step(batch)
        loss.backward()
        synchronize(device)
        timings["torch_compile_seconds"] = (
            time.perf_counter() - compile_started
        )

        our_passes = sum(
            timings.get(field, 0.0)
            for field in (
                "xe_extraction_seconds",
                "folding_seconds",
                "contraction_path_seconds",
                "backend_lowering_seconds",
            )
        )
        frontend = (
            our_passes
            if configuration.system == "xe"
            else timings.get("cirkit_lowering_seconds", 0.0)
        )
        row.update(
            timings,
            status="ok",
            our_passes_seconds=our_passes if configuration.system == "xe" else "",
            compile_total_seconds=frontend + timings["torch_compile_seconds"],
            runtime_instructions=(
                len(program.instructions) if program is not None else ""
            ),
            peak_rss_bytes=(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        )
        append_row(output, row)
        print(
            f"ok {configuration.suite:>8} {configuration.variant:>28} "
            f"{configuration.layer:>6} {configuration.region_graph:>11} "
            f"B={configuration.batch_size:<3} H={configuration.units:<4} "
            f"run={configuration.run}: frontend={frontend:.3f}s "
            f"torch.compile={timings['torch_compile_seconds']:.3f}s",
            flush=True,
        )
        return 0
    except Exception as error:
        row.update(
            timings,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            peak_rss_bytes=(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        )
        append_row(output, row)
        print(f"failed {configuration}: {row['error']}", flush=True)
        if verbose_errors:
            traceback.print_exc()
        return 1
    finally:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_isolated(
    script: Path,
    selected: list[Configuration],
    *,
    output: Path,
    device: str,
    timeout_seconds: float,
    verbose_errors: bool,
) -> None:
    completed = completed_keys(output)
    remaining = [
        configuration
        for configuration in selected
        if configuration.key not in completed
    ]
    print(
        f"compile experiment: {len(remaining)} remaining / "
        f"{len(selected)} total; output={output}",
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
        try:
            # Inductor and Triton otherwise reuse persistent artifacts across
            # fresh Python processes. Give every measurement a private cache
            # so all repetitions perform real compilation.
            with tempfile.TemporaryDirectory(
                prefix="extended-einsum-compile-",
                dir="/tmp",
            ) as cache_directory:
                environment = os.environ.copy()
                environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                    Path(cache_directory) / "inductor"
                )
                environment["TRITON_CACHE_DIR"] = str(
                    Path(cache_directory) / "triton"
                )
                subprocess.run(
                    command,
                    check=False,
                    timeout=timeout_seconds,
                    env=environment,
                )
        except subprocess.TimeoutExpired:
            row = base_row(configuration, device=None)
            row.update(
                status="timeout",
                error=f"configuration exceeded {timeout_seconds:.1f} seconds",
            )
            append_row(output, row)
            print(
                f"timeout after {timeout_seconds:.1f}s: {configuration}",
                flush=True,
            )


def main() -> int:
    args = parse_args()
    if args.single:
        return run_configuration(
            args.single,
            device_arg=args.device,
            output=args.output,
            verbose_errors=args.verbose_errors,
        )
    run_isolated(
        Path(__file__).resolve(),
        configurations(args),
        output=args.output,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        verbose_errors=args.verbose_errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
