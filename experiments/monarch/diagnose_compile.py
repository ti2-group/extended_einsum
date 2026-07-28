from __future__ import annotations

# This file is also executed directly from experiments/monarch/.
# ruff: noqa: E402,I001

import argparse
import csv
import faulthandler
import os
import resource
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIRECTORY.parents[1]
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]
sys.path.insert(0, str(_ROOT))

import torch

import extended_einsum.interface as xe
import extended_einsum.preprocess as preprocess
from experiments.monarch.model import (
    build_symbolic_circuit,
    canonicalize_parameters,
    to_xe_expression,
)
from extended_einsum.preprocess import OptimizeContractionPaths

HERE = Path(__file__).resolve().parent
CSV_FIELDS = (
    "timestamp",
    "region_graph",
    "parameterization",
    "width",
    "height",
    "units",
    "batch_size",
    "monarch_p",
    "monarch_q",
    "seed",
    "timeout_seconds",
    "phase",
    "status",
    "seconds",
    "peak_rss_bytes",
    "details",
)

_active_phase = ""
_active_phase_token: object | None = None


class PhaseTimeout(TimeoutError):
    def __init__(self, phase: str, stack: str) -> None:
        super().__init__(f"{phase} exceeded its timeout")
        self.phase = phase
        self.stack = stack


def _alarm_handler(_signum: int, frame) -> None:
    stack = "".join(traceback.format_stack(frame))
    raise PhaseTimeout(_active_phase, stack)


def peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def append_row(path: Path, base: dict[str, object], **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        row = {**base, **values}
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def run_phase(
    name: str,
    function: Callable[[], Any],
    *,
    timeout_seconds: float,
    output: Path,
    base: dict[str, object],
    describe: Callable[[Any], str] = lambda _value: "",
) -> Any:
    global _active_phase, _active_phase_token
    _active_phase = name
    phase_token = object()
    _active_phase_token = phase_token
    print(f"START {name}", flush=True)
    started = time.perf_counter()

    def hard_timeout() -> None:
        if _active_phase_token is not phase_token:
            return
        elapsed = time.perf_counter() - started
        append_row(
            output,
            base,
            phase=name,
            status="hard-timeout",
            seconds=elapsed,
            peak_rss_bytes=peak_rss_bytes(),
            details=(
                "The Python timeout exception was intercepted by native "
                "code; the watchdog terminated the diagnostic."
            ),
        )
        print(
            f"HARD TIMEOUT {name} after {elapsed:.3f}s "
            f"(peak RSS {peak_rss_bytes() / 2**30:.2f} GiB)",
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(124)

    # A native-library callback can intercept a Python exception raised by the
    # signal handler. The watchdog makes the per-phase ceiling unconditional.
    watchdog = threading.Timer(timeout_seconds + 2.0, hard_timeout)
    watchdog.daemon = True
    watchdog.start()
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        value = function()
    except PhaseTimeout as error:
        elapsed = time.perf_counter() - started
        append_row(
            output,
            base,
            phase=name,
            status="timeout",
            seconds=elapsed,
            peak_rss_bytes=peak_rss_bytes(),
            details=f"timeout stack:\n{error.stack}",
        )
        print(
            f"TIMEOUT {name} after {elapsed:.3f}s "
            f"(peak RSS {peak_rss_bytes() / 2**30:.2f} GiB)",
            flush=True,
        )
        print(error.stack, flush=True)
        raise
    except Exception as error:
        elapsed = time.perf_counter() - started
        details = "".join(traceback.format_exception(error))
        append_row(
            output,
            base,
            phase=name,
            status="failed",
            seconds=elapsed,
            peak_rss_bytes=peak_rss_bytes(),
            details=details,
        )
        print(f"FAILED {name} after {elapsed:.3f}s: {error}", flush=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        _active_phase = ""
        _active_phase_token = None
        watchdog.cancel()

    elapsed = time.perf_counter() - started
    details = describe(value)
    append_row(
        output,
        base,
        phase=name,
        status="ok",
        seconds=elapsed,
        peak_rss_bytes=peak_rss_bytes(),
        details=details,
    )
    print(
        f"DONE {name} {elapsed:.3f}s "
        f"(peak RSS {peak_rss_bytes() / 2**30:.2f} GiB) {details}",
        flush=True,
    )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time the XE construction and preprocessing phases without "
            "calling torch.compile or executing on a GPU."
        )
    )
    parser.add_argument("--region-graph", default="quad-graph")
    parser.add_argument(
        "--parameterization",
        choices=("dense", "monarch"),
        default="monarch",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--monarch-factors", default="16,16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--optimize-group-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the future-consumer group-order heuristic during folding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "compile_diagnosis.csv",
    )
    args = parser.parse_args()
    try:
        factors = tuple(int(value) for value in args.monarch_factors.split(","))
        if len(factors) != 2:
            raise ValueError
        if args.parameterization == "dense":
            factors = None
        elif factors[0] * factors[1] != args.units:
            raise ValueError
        if (
            args.width <= 0
            or args.height <= 0
            or args.units <= 0
            or args.batch_size <= 0
            or args.timeout_seconds <= 0
        ):
            raise ValueError
    except ValueError:
        parser.error("invalid dimensions, factors, batch size, or timeout")
    args.factors = factors
    return args


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGALRM, _alarm_handler)
    factors = args.factors
    base = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "region_graph": args.region_graph,
        "parameterization": args.parameterization,
        "width": args.width,
        "height": args.height,
        "units": args.units,
        "batch_size": args.batch_size,
        "monarch_p": factors[0] if factors else "",
        "monarch_q": factors[1] if factors else "",
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
    }
    print(
        "This diagnostic stops before torch.compile and performs no GPU work.",
        flush=True,
    )
    try:
        symbolic = run_phase(
            "build_symbolic_circuit",
            lambda: build_symbolic_circuit(
                width=args.width,
                height=args.height,
                units=args.units,
                categories=256,
                region_graph=args.region_graph,
                parameterization=args.parameterization,
                factors=factors,
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda circuit: f"layers={len(circuit.layers)}",
        )
        canonical = run_phase(
            "canonicalize_parameters",
            lambda: canonicalize_parameters(symbolic, seed=args.seed),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda state: f"parameters={state.parameters}",
        )

        state_shape = factors or (args.units,)
        data = xe.array(
            torch.empty(
                (
                    args.width * args.height,
                    args.batch_size,
                    *state_shape,
                ),
                dtype=torch.float32,
            )
        )
        expression = run_phase(
            "construct_xe_expression",
            lambda: xe.log(
                to_xe_expression(
                    symbolic,
                    symbolic.layers[-1],
                    data,
                )
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: f"shape={value.shape}",
        )
        program, inputs = run_phase(
            "extract_xe_program",
            lambda: xe.extract_program(
                expression,
                stability_mode="logspace_max",
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: (
                f"instructions={len(value[0].instructions)} "
                f"inputs={value[0].n_inputs}"
            ),
        )
        same_index_product = (
            preprocess._has_same_index_product_contraction(program)
        )
        groups = run_phase(
            "discover_input_depth_groups",
            lambda: preprocess.group_identical_ops_by_input_depth(
                program,
                split_by_routing=same_index_product,
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: (
                f"groups={len(value)} "
                f"members={sum(len(group.members) for group in value)}"
            ),
        )
        ordered_groups = run_phase(
            "order_groups_by_input_access",
            lambda: preprocess._order_input_depth_groups_by_input_access(
                program,
                groups,
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: f"groups={len(value)}",
        )
        events = run_phase(
            "topologically_order_fold_events",
            lambda: preprocess._topologically_order_output_depth_events(
                program,
                ordered_groups,
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: f"events={len(value)}",
        )
        ordered_events = (
            run_phase(
                "optimize_group_member_order",
                lambda: (
                    preprocess._order_group_members_for_future_consumers(
                        events,
                        program,
                    )
                ),
                timeout_seconds=args.timeout_seconds,
                output=args.output,
                base=base,
                describe=lambda value: f"events={len(value)}",
            )
            if args.optimize_group_order
            else events
        )
        folded = run_phase(
            "rewrite_fold_events",
            lambda: preprocess._rewrite_output_depth_group_events(
                program,
                ordered_events,
                emit_fragmented_batch_gathers=True,
            ),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: (
                f"instructions={len(value.program.instructions)} "
                f"inputs={value.program.n_inputs}"
            ),
        )
        optimized = run_phase(
            "contraction_path_optimization",
            lambda: OptimizeContractionPaths.apply(folded.program),
            timeout_seconds=args.timeout_seconds,
            output=args.output,
            base=base,
            describe=lambda value: (
                f"instructions={len(value.instructions)} "
                f"inputs={value.n_inputs}"
            ),
        )
        print(
            "COMPLETE pipeline-only diagnosis: "
            f"parameters={canonical.parameters} extracted_inputs={len(inputs)} "
            f"runtime_instructions={len(optimized.instructions)}",
            flush=True,
        )
    except PhaseTimeout:
        return 124
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
