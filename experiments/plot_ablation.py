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

VARIANT_ORDER = (
    "xe",
    "logspace",
    "shift-gradients",
    "logspace-shift-gradients",
    "no-ordering",
    "cirkit",
)
VARIANT_LABELS = {
    "xe": "XE",
    "logspace": "Log-space",
    "shift-gradients": "Shift\ngradients",
    "logspace-shift-gradients": "Log-space +\nshift gradients",
    "no-ordering": "No consumer\nordering",
    "cirkit": "Cirkit",
}
FORWARD_COLOR = "#56B4E9"
BACKWARD_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CP and Tucker publication ablations.")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "ablation.csv")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def normalized_seed_rows(results: pd.DataFrame) -> pd.DataFrame:
    key = ["seed", "region_graph", "layer", "batch_size", "units"]
    reference = results.loc[results["variant"].eq("xe"), key + ["forward_backward_ms_per_batch"]].rename(columns={"forward_backward_ms_per_batch": "xe_total"})
    normalized = results.merge(reference, on=key, validate="many_to_one")
    normalized["forward"] = normalized["forward_loss_ms_per_batch"] / normalized["xe_total"]
    normalized["backward"] = normalized["backward_ms_per_batch"] / normalized["xe_total"]
    normalized["total"] = normalized["forward"] + normalized["backward"]
    return normalized


def plot_layer(rows: pd.DataFrame, *, layer: str, output: Path) -> None:
    selected = rows.loc[rows["layer"].eq(layer)]
    cells = sorted(
        selected[["region_graph", "batch_size", "units"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda item: (item[0], item[1], item[2]),
    )
    graphs = ("quad-tree-2", "quad-graph")
    size_pairs = sorted({(batch, units) for _, batch, units in cells})
    fig, axes = plt.subplots(
        len(graphs),
        len(size_pairs),
        figsize=(12.0, 7.0),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    x = np.arange(len(VARIANT_ORDER))
    for graph_index, graph in enumerate(graphs):
        for pair_index, (batch, units) in enumerate(size_pairs):
            ax = axes[graph_index, pair_index]
            cell = selected.loc[selected["region_graph"].eq(graph) & selected["batch_size"].eq(batch) & selected["units"].eq(units)]
            forward_values, backward_values, total_values = [], [], []
            lows, highs = [], []
            for variant_index, variant in enumerate(VARIANT_ORDER):
                variant_rows = cell.loc[cell["variant"].eq(variant)]
                forward_values.append(float(np.median(variant_rows["forward"])))
                backward_values.append(float(np.median(variant_rows["backward"])))
                center, low, high = bootstrap_median(
                    variant_rows["total"].to_numpy(),
                    seed=10_000 * graph_index + 100 * pair_index + variant_index,
                )
                total_values.append(center)
                lows.append(low)
                highs.append(high)
            forward_array = np.asarray(forward_values)
            backward_array = np.asarray(backward_values)
            total_array = np.asarray(total_values)
            ax.bar(x, forward_array, color=FORWARD_COLOR, label="Forward + loss")
            ax.bar(
                x,
                backward_array,
                bottom=forward_array,
                color=BACKWARD_COLOR,
                label="Backward",
            )
            ax.errorbar(
                x,
                total_array,
                yerr=np.vstack((total_array - lows, np.asarray(highs) - total_array)),
                fmt="none",
                ecolor="#202020",
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
            for position, total in zip(x, total_array, strict=True):
                if abs(total - 1.0) < 0.005:
                    continue
                ax.text(
                    position,
                    total + 0.035,
                    f"{(total - 1.0) * 100:+.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.axhline(1.0, color="#666666", linewidth=1.0)
            ax.set_title(f"{GRAPH_LABELS[graph]}, batch {batch}, units {units}")
            ax.set_xticks(x, [VARIANT_LABELS[item] for item in VARIANT_ORDER], rotation=20, ha="right")
            if pair_index == 0:
                ax.set_ylabel("Runtime relative to XE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.98), ncol=2)
    fig.suptitle(f"Extended Einsum {layer.upper()} Ablation", y=1.02)
    save_pdf(fig, output)


def main() -> None:
    args = parse_args()
    configure_style()
    results = read_results(args.input, expected_variants=set(VARIANT_ORDER))
    normalized = normalized_seed_rows(results)
    for layer in ("cp", "tucker"):
        plot_layer(normalized, layer=layer, output=args.output_dir / f"ablation_{layer}.pdf")


if __name__ == "__main__":
    main()
