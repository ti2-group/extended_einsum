from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JAX_RESULTS = HERE / "results" / "ablation_jax.csv"
CORRECTNESS_RESULTS = HERE / "results" / "correctness.csv"

JAX_VARIANTS = (
    "xe",
    "logspace",
    "shift-gradients",
    "logspace-shift-gradients",
    "no-ordering",
)
REGION_GRAPHS = ("quad-tree-2", "quad-graph")
LAYERS = ("cp", "tucker")
SEEDS = tuple(range(5))
ABLATION_GRID = {
    "cp": ((256, 128), (512, 512)),
    "tucker": ((256, 32), (512, 64)),
}
AGREEMENT_VARIANTS = ("scaled-max", "logspace-max")
PRECISIONS = ("highest", "high")
STRESS_DEPTHS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
STRESS_VARIANTS = (
    "unstable-fp32",
    "unstable-fp64",
    "scaled-max-fp32",
    "logspace-max-fp32",
)
STRESS_GRIDS = (
    (64, "highest"),
    (64, "high"),
    (256, "high"),
    (512, "high"),
)
MNIST_UNITS = {"cp": 512, "tucker": 64}


@dataclass(frozen=True)
class UnitState:
    load: str
    active: str
    sub: str
    result: str
    status: str
    invocation_id: str


@dataclass(frozen=True)
class CsvState:
    successful: int
    expected: int
    failures: tuple[str, ...]
    duplicates: tuple[tuple[str, ...], ...]
    missing: tuple[tuple[str, ...], ...]
    unexpected: tuple[tuple[str, ...], ...]

    @property
    def complete(self) -> bool:
        return self.successful == self.expected and not self.failures and not self.duplicates and not self.missing and not self.unexpected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Wait for one exact JAX systemd invocation, verify all 200 JAX rows, then run and validate the numerical-correctness supplement."))
    parser.add_argument(
        "--unit",
        default="extended-einsum-jax-ablation.service",
    )
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--jax-results", type=Path, default=JAX_RESULTS)
    parser.add_argument(
        "--correctness-results",
        type=Path,
        default=CORRECTNESS_RESULTS,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="print the current JAX CSV state without waiting or launching",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1.0:
        parser.error("--poll-seconds must be at least one second")
    return args


def expected_jax_keys() -> set[tuple[str, ...]]:
    return {
        (
            variant,
            str(seed),
            graph,
            layer,
            str(batch),
            str(units),
        )
        for seed in SEEDS
        for graph in REGION_GRAPHS
        for layer in LAYERS
        for batch, units in ABLATION_GRID[layer]
        for variant in JAX_VARIANTS
    }


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.exists():
        return [], ()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), tuple(reader.fieldnames or ())


def jax_csv_state(path: Path) -> CsvState:
    rows, fields = read_csv_rows(path)
    required = {
        "variant",
        "status",
        "error",
        "seed",
        "region_graph",
        "layer",
        "units",
        "batch_size",
    }
    missing_fields = required - set(fields)
    if missing_fields:
        raise ValueError(f"{path} is missing JAX fields: {', '.join(sorted(missing_fields))}")
    expected = expected_jax_keys()
    successful_keys: list[tuple[str, ...]] = []
    failures: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        if row["status"] != "ok":
            failures.append(f"row {row_number}: {row['status']} {row['error']}".strip())
            continue
        successful_keys.append(
            (
                row["variant"],
                row["seed"],
                row["region_graph"],
                row["layer"],
                row["batch_size"],
                row["units"],
            )
        )
    counts: dict[tuple[str, ...], int] = {}
    for key in successful_keys:
        counts[key] = counts.get(key, 0) + 1
    actual = set(successful_keys)
    return CsvState(
        successful=len(actual & expected),
        expected=len(expected),
        failures=tuple(failures),
        duplicates=tuple(sorted(key for key, count in counts.items() if count > 1)),
        missing=tuple(sorted(expected - actual)),
        unexpected=tuple(sorted(actual - expected)),
    )


def unit_state(unit: str) -> UnitState:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,InvocationID",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return UnitState(
        load=values.get("LoadState", "not-found"),
        active=values.get("ActiveState", "inactive"),
        sub=values.get("SubState", "dead"),
        result=values.get("Result", ""),
        status=values.get("ExecMainStatus", ""),
        invocation_id=values.get("InvocationID", ""),
    )


def invocation_exit_failures(invocation_id: str) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "journalctl",
            "--user",
            f"USER_INVOCATION_ID={invocation_id}",
            "--no-pager",
            "-o",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    failures: list[str] = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        unit_result = record.get("UNIT_RESULT")
        exit_status = record.get("EXIT_STATUS")
        if unit_result not in {None, "", "success"}:
            failures.append(f"UNIT_RESULT={unit_result}")
        if exit_status not in {None, "", "0"}:
            failures.append(f"EXIT_STATUS={exit_status}")
    return tuple(dict.fromkeys(failures))


def wait_for_verified_jax(args: argparse.Namespace) -> None:
    observed_exact_invocation = False
    last_progress = -1
    while True:
        state = unit_state(args.unit)
        csv_state = jax_csv_state(args.jax_results)
        if csv_state.successful != last_progress:
            print(
                f"JAX gate: {csv_state.successful}/{csv_state.expected} successful, failures={len(csv_state.failures)}; unit={state.active}/{state.sub}",
                flush=True,
            )
            last_progress = csv_state.successful

        if state.active == "active":
            if state.invocation_id != args.invocation_id:
                raise RuntimeError(f"{args.unit} invocation changed from {args.invocation_id} to {state.invocation_id}")
            observed_exact_invocation = True
            time.sleep(args.poll_seconds)
            continue

        if not observed_exact_invocation:
            raise RuntimeError(f"never observed active invocation {args.invocation_id} for {args.unit}")
        journal_failures = invocation_exit_failures(args.invocation_id)
        if state.load != "not-found" and state.result not in {"", "success"}:
            raise RuntimeError(f"{args.unit} ended with result={state.result}, status={state.status}")
        if journal_failures:
            raise RuntimeError(f"{args.unit} recorded failure metadata: {', '.join(journal_failures)}")
        if not csv_state.complete:
            raise RuntimeError(format_incomplete_jax(csv_state))
        print(
            "JAX gate passed: the exact service invocation ended cleanly and the CSV has all 200 unique successful configurations.",
            flush=True,
        )
        return


def format_incomplete_jax(state: CsvState) -> str:
    details = [f"JAX CSV incomplete: {state.successful}/{state.expected} successful"]
    if state.failures:
        details.append(f"failures={state.failures[:3]}")
    if state.duplicates:
        details.append(f"duplicates={state.duplicates[:3]}")
    if state.missing:
        details.append(f"missing={state.missing[:3]}")
    if state.unexpected:
        details.append(f"unexpected={state.unexpected[:3]}")
    return "; ".join(details)


def expected_correctness_keys() -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for seed in SEEDS:
        for graph in REGION_GRAPHS:
            for layer in LAYERS:
                for precision in PRECISIONS:
                    for variant in AGREEMENT_VARIANTS:
                        keys.add(
                            (
                                "agreement",
                                str(seed),
                                graph,
                                layer,
                                "8",
                                "8",
                                "8",
                                "8",
                                "2",
                                "",
                                variant,
                                precision,
                                "cuda:0",
                                "True",
                            )
                        )
        for depth in STRESS_DEPTHS:
            for units, precision in STRESS_GRIDS:
                for variant in STRESS_VARIANTS:
                    keys.add(
                        (
                            "stress",
                            str(seed),
                            "",
                            "",
                            "",
                            "",
                            str(units),
                            "4",
                            "",
                            str(depth),
                            variant,
                            precision,
                            "cuda:0",
                            "False",
                        )
                    )
        for graph in REGION_GRAPHS:
            for layer in LAYERS:
                keys.add(
                    (
                        "mnist",
                        str(seed),
                        graph,
                        layer,
                        "28",
                        "28",
                        str(MNIST_UNITS[layer]),
                        "8",
                        "256",
                        "",
                        "scaled-max",
                        "high",
                        "cuda:0",
                        "True",
                    )
                )
    return keys


def correctness_row_key(row: dict[str, str]) -> tuple[str, ...]:
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
        "torch_compile",
    )
    return tuple(row[field] for field in fields)


def validate_correctness(path: Path) -> None:
    rows, fields = read_csv_rows(path)
    required = {
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
        "matmul_precision",
        "device",
        "torch_compile",
    }
    missing_fields = required - set(fields)
    if missing_fields:
        raise ValueError(f"{path} is missing correctness fields: {', '.join(sorted(missing_fields))}")
    expected = expected_correctness_keys()
    target_rows = [row for row in rows if correctness_row_key(row) in expected or row["suite"] in {"agreement", "stress", "mnist"} and row["device"] == "cuda:0"]
    failures = [f"{row['suite']} seed={row['seed']} {row['variant']}: {row['error']}" for row in target_rows if row["status"] != "ok"]
    successful_keys = [correctness_row_key(row) for row in target_rows if row["status"] == "ok"]
    counts: dict[tuple[str, ...], int] = {}
    for key in successful_keys:
        counts[key] = counts.get(key, 0) + 1
    actual = set(successful_keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    if failures or duplicates or actual != expected:
        raise RuntimeError(
            "correctness validation failed: "
            f"{len(actual & expected)}/{len(expected)} expected successes, "
            f"{len(failures)} failures, {len(duplicates)} duplicates, "
            f"{len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )
    print(
        f"Correctness validation passed: all {len(expected)} expected rows are unique and successful.",
        flush=True,
    )


def run_correctness(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(HERE / "correctness.py"),
        "--suites",
        "agreement,stress,mnist",
        "--device",
        "cuda",
        "--torch-compile",
        "--output",
        str(args.correctness_results),
        "--verbose-errors",
    ]
    print(f"Starting correctness run: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    for stress_units in (256, 512):
        stress_command = [
            sys.executable,
            str(HERE / "correctness.py"),
            "--suites",
            "stress",
            "--device",
            "cuda",
            "--stress-units",
            str(stress_units),
            "--precisions",
            "high",
            "--output",
            str(args.correctness_results),
            "--verbose-errors",
        ]
        print(f"Starting TF32 stress run: {' '.join(stress_command)}", flush=True)
        subprocess.run(stress_command, cwd=ROOT, check=True)
    validate_correctness(args.correctness_results)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "plot_correctness.py"),
            "--input",
            str(args.correctness_results),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.check_only:
        state = jax_csv_state(args.jax_results)
        print(f"{state.successful}/{state.expected} successful; failures={len(state.failures)}, duplicates={len(state.duplicates)}, missing={len(state.missing)}, unexpected={len(state.unexpected)}")
        return
    wait_for_verified_jax(args)
    run_correctness(args)


if __name__ == "__main__":
    main()
