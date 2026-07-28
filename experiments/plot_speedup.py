from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
LAYER_LABELS = {"cp": "CP", "tucker": "Tucker"}
BACKEND_LABELS = {"xe": "Extended Einsum", "cirkit": "Cirkit"}
BACKEND_LINESTYLES = {"xe": "-", "cirkit": "--"}
GRAPH_STYLES = {
    "quad-tree-2": ("#0072B2", "o"),
    "quad-graph": ("#D55E00", "^"),
}


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
                keep |= results["layer"].eq(layer) & results["region_graph"].eq(graph) & results["batch_size"].eq(batch) & results["units"].isin(units_values)
    selected = results.loc[keep].copy()
    groups = selected.groupby(["variant", "region_graph", "layer", "batch_size", "units"])
    incomplete = [(*group_key, sorted(set(range(5)) - set(group["seed"]))) for group_key, group in groups if set(group["seed"]) != set(range(5))]
    if incomplete:
        raise ValueError(f"publication grid is incomplete (missing seeds shown last): {incomplete[:8]}")
    return selected


def plot_metric(
    ratios: pd.DataFrame,
    *,
    layer: str,
    ylabel: str,
    title: str,
    output: Path,
    lower_limit: float | None,
    legend_location: str = "lower right",
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
    ax.legend(loc=legend_location)
    save_pdf(fig, output)


def plot_raw_memory(
    results: pd.DataFrame,
    *,
    layer: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.2),
        sharey=True,
        layout="constrained",
    )
    layer_rows = results.loc[results["layer"].eq(layer)].copy()
    layer_rows["peak_allocated_memory_gib"] = layer_rows["peak_allocated_memory_bytes"] / float(1024**3)
    for batch_index, (batch, ax) in enumerate(zip((256, 512), axes, strict=True)):
        batch_rows = layer_rows.loc[layer_rows["batch_size"].eq(batch)]
        for graph_index, (graph, (color, marker)) in enumerate(GRAPH_STYLES.items()):
            for backend_index, backend in enumerate(("xe", "cirkit")):
                rows = batch_rows.loc[batch_rows["region_graph"].eq(graph) & batch_rows["variant"].eq(backend)]
                if rows.empty:
                    continue
                units_values = sorted(rows["units"].unique())
                centers, lows, highs = [], [], []
                for units_index, units in enumerate(units_values):
                    values = rows.loc[rows["units"].eq(units), "peak_allocated_memory_gib"].to_numpy()
                    center, low, high = bootstrap_median(
                        values,
                        seed=(100_000 * batch_index + 10_000 * graph_index + 1_000 * backend_index + units_index),
                    )
                    centers.append(center)
                    lows.append(low)
                    highs.append(high)
                x = np.asarray(units_values)
                ax.plot(
                    x,
                    centers,
                    color=color,
                    linestyle=BACKEND_LINESTYLES[backend],
                    marker=marker,
                    linewidth=2.0,
                    markersize=6,
                    label=f"{BACKEND_LABELS[backend]}, {GRAPH_LABELS[graph]}",
                )
                ax.fill_between(x, lows, highs, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"Batch size {batch}")
        ax.set_xlabel("Units per input and sum layer")
        if batch_index == 0:
            ax.set_ylabel("Peak allocated memory (GiB)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#303030",
            linestyle=BACKEND_LINESTYLES[backend],
            linewidth=2.0,
            label=BACKEND_LABELS[backend],
        )
        for backend in ("xe", "cirkit")
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=6,
            label=GRAPH_LABELS[graph],
        )
        for graph, (color, marker) in GRAPH_STYLES.items()
    )
    axes[1].legend(handles=legend_handles, loc="upper left", fontsize=8.5)
    fig.suptitle(f"{LAYER_LABELS[layer]} Peak Memory Usage")
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
        layer_label = LAYER_LABELS[layer]
        plot_metric(
            speedups,
            layer=layer,
            ylabel="Speedup over Cirkit",
            title=f"Extended Einsum {layer_label} Speedup over Cirkit",
            output=args.output_dir / f"speedup_{layer}.pdf",
            lower_limit=1.1 if layer == "cp" else 1.0,
        )
        plot_metric(
            reductions,
            layer=layer,
            ylabel="Peak-memory reduction over Cirkit",
            title=f"Extended Einsum {layer_label} Peak-Memory Reduction",
            output=args.output_dir / f"memory_reduction_{layer}.pdf",
            lower_limit=0.9,
            legend_location="lower left",
        )
        plot_raw_memory(
            results,
            layer=layer,
            output=args.output_dir / f"memory_usage_{layer}.pdf",
        )


if __name__ == "__main__":
    main()
