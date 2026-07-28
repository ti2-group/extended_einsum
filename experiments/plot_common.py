from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXPECTED_SEEDS = frozenset(range(5))
GRAPH_LABELS = {"quad-tree-2": "Quad tree", "quad-graph": "Quad graph"}


def configure_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.12)
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def read_results(
    path: Path,
    *,
    expected_variants: set[str],
    require_complete_groups: bool = True,
) -> pd.DataFrame:
    results = pd.read_csv(path)
    required = {
        "variant",
        "status",
        "seed",
        "region_graph",
        "layer",
        "units",
        "batch_size",
        "forward_loss_ms_per_batch",
        "backward_ms_per_batch",
        "forward_backward_ms_per_batch",
        "peak_allocated_memory_bytes",
    }
    missing = required - set(results)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    failed = results.loc[results["status"].ne("ok")]
    if not failed.empty:
        print(
            f"ignoring {len(failed)} failed configuration attempt(s) recorded in {path}"
        )
    results = results.loc[results["status"].eq("ok")].copy()
    results = results.loc[results["variant"].isin(expected_variants)].copy()
    for column in (
        "seed",
        "units",
        "batch_size",
        "forward_loss_ms_per_batch",
        "backward_ms_per_batch",
        "forward_backward_ms_per_batch",
        "peak_allocated_memory_bytes",
    ):
        results[column] = pd.to_numeric(results[column], errors="raise")
    key = ["variant", "seed", "region_graph", "layer", "batch_size", "units"]
    results = results.drop_duplicates(key, keep="last")
    groups = results.groupby(["variant", "region_graph", "layer", "batch_size", "units"])
    incomplete = [(*group_key, sorted(EXPECTED_SEEDS - set(group["seed"]))) for group_key, group in groups if set(group["seed"]) != EXPECTED_SEEDS]
    if require_complete_groups and incomplete:
        raise ValueError(f"{path} is incomplete (missing seeds shown last): {incomplete[:8]}")
    return results


def bootstrap_median(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 5000,
    confidence: float = 95.0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    bootstrapped = np.median(values[indices], axis=1)
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(bootstrapped, (tail, 100.0 - tail))
    return center, float(low), float(high)


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
