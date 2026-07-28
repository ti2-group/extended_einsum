from __future__ import annotations

# This file is also executed directly from experiments/monarch/.
# ruff: noqa: E402,I001

import argparse
import csv
import statistics
import sys
from collections import defaultdict
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

HERE = Path(__file__).resolve().parent
EXPECTED_BACKENDS = frozenset({"cirkit", "xe"})
EXPECTED_SEEDS = frozenset(range(5))
GRAPH_ORDER = {"quad-tree-2": 0, "quad-graph": 1}
PARAMETERIZATION_ORDER = {"dense": 0, "monarch": 1}
GRAPH_LABELS = {
    "quad-tree-2": "Quad tree",
    "quad-graph": "Quad graph",
}


@dataclass(frozen=True)
class Configuration:
    region_graph: str
    parameterization: str
    units: int
    batch_size: int
    p: int | None
    q: int | None


@dataclass(frozen=True)
class Summary:
    configuration: Configuration
    parameters: int
    cirkit_ms: float
    xe_ms: float
    speedup: float
    speedup_min: float
    speedup_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically convert the completed Monarch CSV into "
            "main-paper and supplementary LaTeX tables."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "results" / "performance.csv",
    )
    parser.add_argument(
        "--main-output",
        type=Path,
        default=HERE / "tables" / "performance_main.tex",
    )
    parser.add_argument(
        "--supplement-output",
        type=Path,
        default=HERE / "tables" / "performance_supplement.tex",
    )
    return parser.parse_args()


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def load_results(path: Path) -> tuple[Summary, ...]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "backend",
            "status",
            "seed",
            "region_graph",
            "parameterization",
            "units",
            "batch_size",
            "monarch_p",
            "monarch_q",
            "parameters",
            "initialization_hash",
            "forward_backward_ms_per_batch",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = [
            row
            for row in reader
            if row["status"] == "ok"
        ]

    grouped: dict[
        Configuration,
        dict[tuple[str, int], dict[str, str]],
    ] = defaultdict(dict)
    for row in rows:
        configuration = Configuration(
            region_graph=row["region_graph"],
            parameterization=row["parameterization"],
            units=int(row["units"]),
            batch_size=int(row["batch_size"]),
            p=optional_int(row["monarch_p"]),
            q=optional_int(row["monarch_q"]),
        )
        key = (row["backend"], int(row["seed"]))
        if key in grouped[configuration]:
            raise ValueError(
                f"duplicate successful row for {configuration}, {key}"
            )
        grouped[configuration][key] = row

    summaries: list[Summary] = []
    expected_keys = {
        (backend, seed)
        for backend in EXPECTED_BACKENDS
        for seed in EXPECTED_SEEDS
    }
    for configuration, configuration_rows in grouped.items():
        if set(configuration_rows) != expected_keys:
            missing = sorted(expected_keys - set(configuration_rows))
            raise ValueError(
                f"incomplete configuration {configuration}; missing {missing}"
            )

        parameters = {
            int(row["parameters"])
            for row in configuration_rows.values()
        }
        if len(parameters) != 1:
            raise ValueError(
                f"parameter mismatch for {configuration}: {parameters}"
            )
        for seed in EXPECTED_SEEDS:
            hashes = {
                configuration_rows[(backend, seed)][
                    "initialization_hash"
                ]
                for backend in EXPECTED_BACKENDS
            }
            if len(hashes) != 1:
                raise ValueError(
                    f"initialization mismatch for {configuration}, seed {seed}"
                )

        cirkit = {
            seed: float(
                configuration_rows[("cirkit", seed)][
                    "forward_backward_ms_per_batch"
                ]
            )
            for seed in EXPECTED_SEEDS
        }
        xe = {
            seed: float(
                configuration_rows[("xe", seed)][
                    "forward_backward_ms_per_batch"
                ]
            )
            for seed in EXPECTED_SEEDS
        }
        speedups = [cirkit[seed] / xe[seed] for seed in EXPECTED_SEEDS]
        summaries.append(
            Summary(
                configuration=configuration,
                parameters=parameters.pop(),
                cirkit_ms=statistics.median(cirkit.values()),
                xe_ms=statistics.median(xe.values()),
                speedup=statistics.median(speedups),
                speedup_min=min(speedups),
                speedup_max=max(speedups),
            )
        )

    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                GRAPH_ORDER[summary.configuration.region_graph],
                PARAMETERIZATION_ORDER[
                    summary.configuration.parameterization
                ],
                summary.configuration.units,
            ),
        )
    )


def factorization(configuration: Configuration) -> str:
    if configuration.p is None or configuration.q is None:
        return ""
    return rf"{configuration.p}\mathbin{{\times}}{configuration.q}"


def matrix_size(configuration: Configuration) -> str:
    units = configuration.units
    factors = factorization(configuration)
    return (
        rf"${units}\mathbin{{\times}}{units}$ "
        rf"(${factors}$)"
    )


def main_table(summaries: tuple[Summary, ...]) -> str:
    monarch = [
        summary
        for summary in summaries
        if summary.configuration.parameterization == "monarch"
    ]
    lines = [
        "% Generated by experiments/monarch/table.py; do not edit.",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Topology & Monarch matrix $H\times H$ (factors $P\times Q$)"
        r" & XE speedup \\",
        r"\midrule",
    ]
    lines.extend(
        (
            f"{GRAPH_LABELS[summary.configuration.region_graph]} & "
            f"{matrix_size(summary.configuration)} & "
            rf"${summary.speedup:.3f}\mathbin{{\times}}$ \\"
        )
        for summary in monarch
    )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def supplement_table(summaries: tuple[Summary, ...]) -> str:
    lines = [
        "% Generated by experiments/monarch/table.py; do not edit.",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Topology & Sum & $H$ & $P\times Q$ & Parameters"
        r" & Cirkit (ms) & XE (ms) & Speedup & Seed range \\",
        r"\midrule",
    ]
    for summary in summaries:
        configuration = summary.configuration
        factor_cell = (
            f"${factorization(configuration)}$"
            if configuration.p is not None
            else r"\textemdash"
        )
        lines.append(
            f"{GRAPH_LABELS[configuration.region_graph]} & "
            f"{configuration.parameterization.capitalize()} & "
            f"{configuration.units} & "
            f"{factor_cell} & "
            f"{summary.parameters / 1e6:.1f}M & "
            f"{summary.cirkit_ms:.1f} & "
            f"{summary.xe_ms:.1f} & "
            rf"${summary.speedup:.3f}\mathbin{{\times}}$ & "
            rf"$[{summary.speedup_min:.3f},"
            rf"{summary.speedup_max:.3f}]\mathbin{{\times}}$ \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def write_table(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    summaries = load_results(args.input)
    write_table(args.main_output, main_table(summaries))
    write_table(args.supplement_output, supplement_table(summaries))


if __name__ == "__main__":
    main()
