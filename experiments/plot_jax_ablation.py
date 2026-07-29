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
    EXPECTED_SEEDS,
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
)
VARIANT_LABELS = {
    "xe": "XE",
    "logspace": "Log-space",
    "shift-gradients": "Shift\ngradients",
    "logspace-shift-gradients": "Log-space +\nshift gradients",
    "no-ordering": "No consumer\nordering",
}
BACKENDS = ("torch", "jax")
BACKEND_LABELS = {"torch": "PyTorch", "jax": "JAX", "cirkit": "Cirkit"}
BACKEND_COLORS = {"torch": "#0072B2", "jax": "#D55E00", "cirkit": "#7A7A7A"}
LAYER_LABELS = {"cp": "CP", "tucker": "Tucker"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the JAX/PyTorch Extended Einsum ablation comparison.")
    parser.add_argument("--torch-input", type=Path, default=RESULTS_DIR / "ablation.csv")
    parser.add_argument("--jax-input", type=Path, default=RESULTS_DIR / "ablation_jax.csv")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def read_jax_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    required = {
        "variant",
        "status",
        "seed",
        "region_graph",
        "layer",
        "units",
        "batch_size",
        "forward_backward_ms_per_batch",
    }
    missing = required - set(results)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    failed = results.loc[results["status"].ne("ok")]
    if not failed.empty:
        raise ValueError(f"{path} contains {len(failed)} failed rows")
    results = results.loc[results["variant"].isin(VARIANT_ORDER)].copy()
    key = ["variant", "seed", "region_graph", "layer", "batch_size", "units"]
    results = results.drop_duplicates(key, keep="last")
    groups = results.groupby(["variant", "region_graph", "layer", "batch_size", "units"])
    incomplete = [(*group_key, sorted(EXPECTED_SEEDS - set(group["seed"]))) for group_key, group in groups if set(group["seed"]) != EXPECTED_SEEDS]
    if incomplete:
        raise ValueError(f"{path} is incomplete (missing seeds shown last): {incomplete[:8]}")
    return results


def combine_results(torch_results: pd.DataFrame, jax_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "seed",
        "region_graph",
        "layer",
        "batch_size",
        "units",
        "forward_backward_ms_per_batch",
    ]
    torch_rows = torch_results.loc[torch_results["variant"].isin(VARIANT_ORDER), columns].assign(backend="torch")
    jax_rows = jax_results.loc[:, columns].assign(backend="jax")
    cirkit_rows = torch_results.loc[torch_results["variant"].eq("cirkit"), columns].assign(backend="cirkit")
    combined = pd.concat((torch_rows, jax_rows, cirkit_rows), ignore_index=True)
    expected = set(BACKENDS)
    groups = combined.loc[combined["variant"].isin(VARIANT_ORDER)].groupby(["variant", "seed", "region_graph", "layer", "batch_size", "units"])
    incomplete = [group_key for group_key, group in groups if set(group["backend"]) != expected]
    if incomplete:
        raise ValueError(f"backend pairing is incomplete: {incomplete[:8]}")
    return combined


def ratios_to_production_xe(cell: pd.DataFrame, *, backend: str, variant: str) -> np.ndarray:
    values = cell.loc[
        cell["backend"].eq(backend) & cell["variant"].eq(variant),
        ["seed", "forward_backward_ms_per_batch"],
    ].rename(columns={"forward_backward_ms_per_batch": "value"})
    reference = cell.loc[
        cell["backend"].eq("torch") & cell["variant"].eq("xe"),
        ["seed", "forward_backward_ms_per_batch"],
    ].rename(columns={"forward_backward_ms_per_batch": "xe"})
    paired = values.merge(reference, on="seed", validate="one_to_one")
    return (paired["value"] / paired["xe"]).to_numpy()


def percentage_label(ratio: float) -> str:
    percentage = round((ratio - 1.0) * 100)
    return "0%" if percentage == 0 else f"{percentage:+d}%"


def plot_layer(rows: pd.DataFrame, *, layer: str, output: Path) -> None:
    selected = rows.loc[rows["layer"].eq(layer)]
    graphs = ("quad-tree-2", "quad-graph")
    size_pairs = sorted(selected[["batch_size", "units"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        len(graphs),
        len(size_pairs),
        figsize=(11.5, 6.7),
        layout="constrained",
        squeeze=False,
    )
    x = np.arange(len(VARIANT_ORDER))
    cirkit_position = len(VARIANT_ORDER)
    width = 0.37
    offsets = {"torch": -width / 2, "jax": width / 2}

    for graph_index, graph in enumerate(graphs):
        for pair_index, (batch, units) in enumerate(size_pairs):
            ax = axes[graph_index, pair_index]
            cell = selected.loc[selected["region_graph"].eq(graph) & selected["batch_size"].eq(batch) & selected["units"].eq(units)]
            backend_summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for backend_index, backend in enumerate(BACKENDS):
                centers, lows, highs = [], [], []
                for variant_index, variant in enumerate(VARIANT_ORDER):
                    values = cell.loc[
                        cell["backend"].eq(backend) & cell["variant"].eq(variant),
                        "forward_backward_ms_per_batch",
                    ].to_numpy()
                    center, low, high = bootstrap_median(
                        values,
                        seed=(100_000 * graph_index + 10_000 * pair_index + 100 * backend_index + variant_index),
                    )
                    centers.append(center)
                    lows.append(low)
                    highs.append(high)
                center_array = np.asarray(centers)
                low_array = np.asarray(lows)
                high_array = np.asarray(highs)
                backend_summaries[backend] = (
                    center_array,
                    low_array,
                    high_array,
                )
                positions = x + offsets[backend]
                ax.bar(
                    positions,
                    center_array,
                    width,
                    color=BACKEND_COLORS[backend],
                    label=BACKEND_LABELS[backend],
                )
                ax.errorbar(
                    positions,
                    center_array,
                    yerr=np.vstack((center_array - low_array, high_array - center_array)),
                    fmt="none",
                    ecolor="#202020",
                    capsize=2.5,
                    linewidth=1.0,
                    zorder=4,
                )

            for backend_index, backend in enumerate(BACKENDS):
                _, _, backend_highs = backend_summaries[backend]
                for variant_index, variant in enumerate(VARIANT_ORDER):
                    if backend == "torch" and variant == "xe":
                        continue
                    ratio, _, _ = bootstrap_median(
                        ratios_to_production_xe(cell, backend=backend, variant=variant),
                        seed=(
                            900_000
                            + 100_000 * backend_index
                            + 10_000 * graph_index
                            + 100 * pair_index
                            + variant_index
                        ),
                    )
                    ax.annotate(
                        percentage_label(ratio),
                        xy=(
                            x[variant_index] + offsets[backend],
                            backend_highs[variant_index],
                        ),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )

            cirkit_values = cell.loc[
                cell["backend"].eq("cirkit") & cell["variant"].eq("cirkit"),
                "forward_backward_ms_per_batch",
            ].to_numpy()
            cirkit_center, cirkit_low, cirkit_high = bootstrap_median(
                cirkit_values,
                seed=800_000 + 10_000 * graph_index + 100 * pair_index,
            )
            ax.bar(
                cirkit_position,
                cirkit_center,
                width,
                color=BACKEND_COLORS["cirkit"],
                label=BACKEND_LABELS["cirkit"],
            )
            ax.errorbar(
                cirkit_position,
                cirkit_center,
                yerr=np.asarray([[cirkit_center - cirkit_low], [cirkit_high - cirkit_center]]),
                fmt="none",
                ecolor="#202020",
                capsize=2.5,
                linewidth=1.0,
                zorder=4,
            )
            torch_xe = cell.loc[
                cell["backend"].eq("torch") & cell["variant"].eq("xe"),
                ["seed", "forward_backward_ms_per_batch"],
            ].rename(columns={"forward_backward_ms_per_batch": "torch_xe"})
            cirkit_by_seed = cell.loc[
                cell["backend"].eq("cirkit") & cell["variant"].eq("cirkit"),
                ["seed", "forward_backward_ms_per_batch"],
            ].rename(columns={"forward_backward_ms_per_batch": "cirkit"})
            cirkit_ratios = cirkit_by_seed.merge(torch_xe, on="seed", validate="one_to_one")
            cirkit_ratio, _, _ = bootstrap_median(
                (cirkit_ratios["cirkit"] / cirkit_ratios["torch_xe"]).to_numpy(),
                seed=850_000 + 10_000 * graph_index + 100 * pair_index,
            )
            ax.annotate(
                percentage_label(cirkit_ratio),
                xy=(cirkit_position, cirkit_high),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

            ax.set_title(f"{GRAPH_LABELS[graph]}, batch {batch}, units {units}")
            ax.set_xticks(
                np.append(x, cirkit_position),
                [VARIANT_LABELS[variant] for variant in VARIANT_ORDER] + ["Cirkit"],
                rotation=18,
                ha="right",
            )
            ax.set_ylabel("Forward + backward (ms/batch)")
            ax.set_ylim(bottom=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    fig.suptitle(f"Extended Einsum {LAYER_LABELS[layer]} Backend Comparison", y=1.045)
    save_pdf(fig, output)


def main() -> None:
    args = parse_args()
    configure_style()
    torch_results = read_results(
        args.torch_input,
        expected_variants=set(VARIANT_ORDER) | {"cirkit"},
    )
    jax_results = read_jax_results(args.jax_input)
    combined = combine_results(torch_results, jax_results)
    for layer in ("cp", "tucker"):
        plot_layer(
            combined,
            layer=layer,
            output=args.output_dir / f"jax_torch_ablation_{layer}.pdf",
        )


if __name__ == "__main__":
    main()
