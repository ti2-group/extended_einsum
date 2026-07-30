from __future__ import annotations

# This file is also executed directly from experiments/.
# ruff: noqa: E402
import argparse
import csv
import statistics
import sys
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.diagnose_compile import (
    CSV_FIELDS,
    DEFAULT_RUNS,
    Configuration,
    configurations,
    row_key,
)

HERE = Path(__file__).resolve().parent
GRAPH_LABELS = {
    "quad-tree-2": "Quad tree",
    "quad-graph": "Quad graph",
}
VARIANT_LABELS = {
    "cirkit": "Cirkit",
    "xe": "XE",
    "logspace": "Log space",
    "shift-gradients": "Shift gradients",
    "logspace-shift-gradients": "Log space + shift gradients",
    "no-ordering": "No ordering",
}


@dataclass(frozen=True)
class Summary:
    configuration: Configuration
    runs: int
    our_passes: tuple[float, float, float] | None
    cirkit_lowering: tuple[float, float, float] | None
    torch_compile: tuple[float, float, float]
    total: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the complete compiler-breakdown CSV into a "
            "supplementary LaTeX table."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "results" / "compile_breakdown.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "tables" / "compile_breakdown.tex",
    )
    parser.add_argument("--runs", default=",".join(map(str, DEFAULT_RUNS)))
    return parser.parse_args()


def interval(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def expected_configurations(runs: str) -> list[Configuration]:
    runner_args = Namespace(
        suites=("ablation", "monarch"),
        runs=tuple(int(value) for value in runs.split(",")),
        layers=("cp", "tucker"),
        graphs=("quad-tree-2", "quad-graph"),
        variants=tuple(VARIANT_LABELS),
        parameterizations=("dense", "monarch"),
    )
    return configurations(runner_args)


def load_results(path: Path, *, runs: str) -> tuple[Summary, ...]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"{path} does not use the compile-breakdown schema"
            )
        successful = [row for row in reader if row["status"] == "ok"]

    expected = expected_configurations(runs)
    expected_keys = {configuration.key for configuration in expected}
    rows_by_key: dict[tuple[object, ...], dict[str, str]] = {}
    for row in successful:
        key = row_key(row)
        if key not in expected_keys:
            continue
        if key in rows_by_key:
            raise ValueError(f"duplicate successful row: {key}")
        rows_by_key[key] = row
    missing = expected_keys - set(rows_by_key)
    if missing:
        preview = ", ".join(map(str, sorted(missing)[:3]))
        raise ValueError(
            f"incomplete compile experiment: {len(missing)} rows missing "
            f"(first: {preview})"
        )

    grouped: dict[
        tuple[object, ...],
        list[tuple[Configuration, dict[str, str]]],
    ] = defaultdict(list)
    for configuration in expected:
        group_key = (
            configuration.suite,
            configuration.system,
            configuration.variant,
            configuration.region_graph,
            configuration.layer,
            configuration.parameterization,
            configuration.width,
            configuration.height,
            configuration.units,
            configuration.batch_size,
            configuration.p,
            configuration.q,
        )
        grouped[group_key].append(
            (configuration, rows_by_key[configuration.key])
        )

    summaries = []
    for items in grouped.values():
        configuration = items[0][0]
        rows = [item[1] for item in items]
        summaries.append(
            Summary(
                configuration=configuration,
                runs=len(rows),
                our_passes=(
                    interval(
                        [float(row["our_passes_seconds"]) for row in rows]
                    )
                    if configuration.system == "xe"
                    else None
                ),
                cirkit_lowering=(
                    interval(
                        [
                            float(row["cirkit_lowering_seconds"])
                            for row in rows
                        ]
                    )
                    if configuration.system == "cirkit"
                    else None
                ),
                torch_compile=interval(
                    [float(row["torch_compile_seconds"]) for row in rows]
                ),
                total=interval(
                    [float(row["compile_total_seconds"]) for row in rows]
                ),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                0 if summary.configuration.suite == "ablation" else 1,
                summary.configuration.layer,
                summary.configuration.region_graph,
                summary.configuration.batch_size,
                summary.configuration.units,
                summary.configuration.parameterization,
                summary.configuration.variant,
            ),
        )
    )


def measurement(value: tuple[float, float, float] | None) -> str:
    if value is None:
        return r"\textemdash"
    median, minimum, maximum = value
    return rf"{median:.2f} [{minimum:.2f}, {maximum:.2f}]"


def ablation_rows(summaries: tuple[Summary, ...]) -> list[str]:
    rows = []
    for summary in summaries:
        configuration = summary.configuration
        if configuration.suite != "ablation":
            continue
        rows.append(
            f"{configuration.layer.upper()} & "
            f"{GRAPH_LABELS[configuration.region_graph]} & "
            f"{configuration.batch_size} & {configuration.units} & "
            f"{VARIANT_LABELS[configuration.variant]} & "
            f"{measurement(summary.our_passes)} & "
            f"{measurement(summary.cirkit_lowering)} & "
            f"{measurement(summary.torch_compile)} & "
            f"{measurement(summary.total)} \\\\"
        )
    return rows


def monarch_rows(summaries: tuple[Summary, ...]) -> list[str]:
    rows = []
    for summary in summaries:
        configuration = summary.configuration
        if configuration.suite != "monarch":
            continue
        factors = (
            rf"${configuration.p}\mathbin{{\times}}{configuration.q}$"
            if configuration.p
            else r"\textemdash"
        )
        rows.append(
            f"{GRAPH_LABELS[configuration.region_graph]} & "
            f"{configuration.parameterization.capitalize()} & "
            f"{configuration.units} & {factors} & "
            f"{configuration.system.upper()} & "
            f"{measurement(summary.our_passes)} & "
            f"{measurement(summary.cirkit_lowering)} & "
            f"{measurement(summary.torch_compile)} & "
            f"{measurement(summary.total)} \\\\"
        )
    return rows


def latex_table(summaries: tuple[Summary, ...]) -> str:
    run_counts = {summary.runs for summary in summaries}
    if len(run_counts) != 1:
        raise ValueError(f"inconsistent run counts: {sorted(run_counts)}")
    runs = run_counts.pop()
    lines = [
        "% Generated by experiments/compile_table.py; do not edit.",
        (
            f"% Entries are median [minimum, maximum] seconds over "
            f"{runs} process-isolated runs."
        ),
        r"\begin{tabular}{llrrlrrrr}",
        r"\toprule",
        r"Layer & Topology & $B$ & $H$ & Variant"
        r" & XE passes & Cirkit lowering & \texttt{torch.compile} & Total \\",
        r"\midrule",
        *ablation_rows(summaries),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tabular}{llrrlrrrr}",
        r"\toprule",
        r"Topology & Sum & $H$ & $P\times Q$ & System"
        r" & XE passes & Cirkit lowering & \texttt{torch.compile} & Total \\",
        r"\midrule",
        *monarch_rows(summaries),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summaries = load_results(args.input, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        latex_table(summaries),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
