from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import PLOTS_DIR, RESULTS_DIR
from experiments.plot_common import (
    GRAPH_LABELS,
    bootstrap_median,
    configure_style,
    read_results,
    save_pdf,
)
from experiments.speedup import SAFE_GRID

SERIES = (
    ("quad-tree-2", 256, "#0072B2", "o"),
    ("quad-tree-2", 512, "#56B4E9", "s"),
    ("quad-graph", 256, "#D55E00", "^"),
    ("quad-graph", 512, "#E69F00", "D"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication speedup and memory-reduction figures.")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "speedup.csv")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def paired_ratios(results: pd.DataFrame, numerator: str, denominator: str, metric: str) -> pd.DataFrame:
    key = ["seed", "region_graph", "layer", "batch_size", "units"]
    wide = results.pivot(index=key, columns="variant", values=metric)
    ratios = (wide[numerator] / wide[denominator]).rename("ratio").reset_index()
    return ratios.dropna(subset=["ratio"])


def select_publication_grid(results: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(False, index=results.index)
    for layer, graphs in SAFE_GRID.items():
        for graph, batches in graphs.items():
            for batch, units_values in batches.items():
                keep |= (
                    results["layer"].eq(layer)
                    & results["region_graph"].eq(graph)
                    & results["batch_size"].eq(batch)
                    & results["units"].isin(units_values)
                )
    selected = results.loc[keep].copy()
    groups = selected.groupby(
        ["variant", "region_graph", "layer", "batch_size", "units"]
    )
    incomplete = [
        (*group_key, sorted(set(range(5)) - set(group["seed"])))
        for group_key, group in groups
        if set(group["seed"]) != set(range(5))
    ]
    if incomplete:
        raise ValueError(
            f"publication grid is incomplete (missing seeds shown last): {incomplete[:8]}"
        )
    return selected


def plot_metric(
    ratios: pd.DataFrame,
    *,
    layer: str,
    ylabel: str,
    title: str,
    output: Path,
    lower_limit: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")
    layer_rows = ratios.loc[ratios["layer"].eq(layer)]
    for series_index, (graph, batch, color, marker) in enumerate(SERIES):
        rows = layer_rows.loc[layer_rows["region_graph"].eq(graph) & layer_rows["batch_size"].eq(batch)]
        if rows.empty:
            continue
        units_values = sorted(rows["units"].unique())
        centers, lows, highs = [], [], []
        for units_index, units in enumerate(units_values):
            values = rows.loc[rows["units"].eq(units), "ratio"].to_numpy()
            center, low, high = bootstrap_median(values, seed=10_000 * series_index + units_index)
            centers.append(center)
            lows.append(low)
            highs.append(high)
        x = np.asarray(units_values)
        centers_array = np.asarray(centers)
        ax.plot(
            x,
            centers_array,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=f"{GRAPH_LABELS[graph]}, batch {batch}",
        )
        ax.fill_between(x, lows, highs, color=color, alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="#666666", linewidth=1.0, zorder=0)
    ax.set_title(title)
    ax.set_xlabel("Units per input and sum layer")
    ax.set_ylabel(ylabel)
    if lower_limit is not None:
        ax.set_ylim(bottom=lower_limit)
    ax.legend(loc="lower right")
    save_pdf(fig, output)


def main() -> None:
    args = parse_args()
    configure_style()
    results = read_results(
        args.input,
        expected_variants={"cirkit", "xe"},
        require_complete_groups=False,
    )
    results = select_publication_grid(results)
    speedups = paired_ratios(results, "cirkit", "xe", "forward_backward_ms_per_batch")
    reductions = paired_ratios(results, "cirkit", "xe", "peak_allocated_memory_bytes")
    for layer in ("cp", "tucker"):
        plot_metric(
            speedups,
            layer=layer,
            ylabel="Speedup over Cirkit",
            title="Extended Einsum Speedup over Cirkit",
            output=args.output_dir / f"speedup_{layer}.pdf",
            lower_limit=1.2 if layer == "cp" else 1.0,
        )
        plot_metric(
            reductions,
            layer=layer,
            ylabel="Peak-memory reduction over Cirkit",
            title="Extended Einsum Peak-Memory Reduction",
            output=args.output_dir / f"memory_reduction_{layer}.pdf",
            lower_limit=1.0,
        )


if __name__ == "__main__":
    main()
