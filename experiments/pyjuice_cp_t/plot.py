from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plot_common import bootstrap_median, configure_style, save_pdf

HERE = Path(__file__).resolve().parent
BACKENDS = ("xe", "cirkit", "pyjuice")
LABELS = {"xe": "Extended Einsum", "cirkit": "Cirkit", "pyjuice": "PyJuice"}
COLORS = {"xe": "#0072B2", "cirkit": "#666666", "pyjuice": "#009E73"}
FORWARD_COLOR = "#56B4E9"
BACKWARD_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the parameter-matched CP-T comparison.")
    parser.add_argument("--input", type=Path, default=HERE / "results" / "comparison.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "plots")
    return parser.parse_args()


def load(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    if not results["status"].eq("ok").all():
        raise ValueError(f"{path} contains failed runs")
    if not results["parameter_match"].astype(str).str.lower().isin({"true", "1"}).all():
        raise ValueError(f"{path} contains parameter-mismatched runs")
    key = ["backend", "seed", "batch_size", "units"]
    results = results.drop_duplicates(key, keep="last")
    counts = results.groupby(["backend", "batch_size", "units"])["seed"].nunique()
    if not counts.eq(5).all():
        raise ValueError(f"{path} is incomplete:\n{counts}")
    return results


def plot_forward(results: pd.DataFrame, output: Path) -> None:
    pairs = ((256, 128), (512, 512))
    x = np.arange(len(pairs), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")
    for backend_index, backend in enumerate(BACKENDS):
        centers, lows, highs = [], [], []
        for pair_index, (batch, units) in enumerate(pairs):
            values = results.loc[
                results["backend"].eq(backend) & results["batch_size"].eq(batch) & results["units"].eq(units),
                "forward_microseconds_per_patch_batch",
            ].to_numpy()
            center, low, high = bootstrap_median(values, seed=100 * backend_index + pair_index)
            centers.append(center)
            lows.append(low)
            highs.append(high)
        positions = x + (backend_index - 1) * width
        ax.bar(positions, centers, width, color=COLORS[backend], label=LABELS[backend])
        ax.errorbar(
            positions,
            centers,
            yerr=np.vstack((np.asarray(centers) - lows, np.asarray(highs) - centers)),
            fmt="none",
            color="#202020",
            capsize=3,
        )
    ax.set_xticks(x, [f"Batch {batch}, units {units}" for batch, units in pairs])
    ax.set_ylabel("Forward time per CP-T patch (µs)")
    ax.set_title("Parameter-Matched CP-T Forward Pass")
    ax.legend(loc="upper left")
    save_pdf(fig, output)


def plot_forward_backward(results: pd.DataFrame, output: Path) -> None:
    pairs = ((256, 128), (512, 512))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained", sharey=False)
    x = np.arange(len(BACKENDS))
    for pair_index, ((batch, units), ax) in enumerate(zip(pairs, axes, strict=True)):
        forwards, backwards, totals, lows, highs = [], [], [], [], []
        for backend_index, backend in enumerate(BACKENDS):
            rows = results.loc[results["backend"].eq(backend) & results["batch_size"].eq(batch) & results["units"].eq(units)]
            forwards.append(float(np.median(rows["forward_ms_per_batch"])))
            backwards.append(float(np.median(rows["backward_ms_per_batch"])))
            center, low, high = bootstrap_median(
                rows["forward_backward_ms_per_batch"].to_numpy(),
                seed=100 * pair_index + backend_index,
            )
            totals.append(center)
            lows.append(low)
            highs.append(high)
        ax.bar(x, forwards, color=FORWARD_COLOR, label="Forward")
        ax.bar(x, backwards, bottom=forwards, color=BACKWARD_COLOR, label="Backward")
        ax.errorbar(
            x,
            totals,
            yerr=np.vstack((np.asarray(totals) - lows, np.asarray(highs) - totals)),
            fmt="none",
            color="#202020",
            capsize=3,
        )
        ax.set_xticks(x, [LABELS[item] for item in BACKENDS], rotation=15, ha="right")
        ax.set_title(f"Batch {batch}, units {units}")
        ax.set_ylabel("Time per batch (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Parameter-Matched CP-T Forward and Backward")
    save_pdf(fig, output)


def main() -> None:
    args = parse_args()
    configure_style()
    results = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_forward(results, args.output_dir / "forward_per_patch.pdf")
    plot_forward_backward(results, args.output_dir / "forward_backward.pdf")


if __name__ == "__main__":
    main()
