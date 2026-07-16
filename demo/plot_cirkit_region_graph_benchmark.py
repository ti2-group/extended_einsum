from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BACKEND_ORDER = ("XE logspace", "XE scaled", "Cirkit")
XE_BACKEND_ORDER = BACKEND_ORDER[:2]
BACKEND_PALETTE = {
    "XE logspace": "#0072B2",
    "XE scaled": "#D55E00",
    "Cirkit": "#4D4D4D",
}
BACKEND_MARKERS = {
    "XE logspace": "o",
    "XE scaled": "s",
    "Cirkit": "D",
}
REGION_GRAPH_ORDER = ("quad-tree-2", "quad-graph")
LAYER_ORDER = ("cp", "tucker")
EXPECTED_UNITS = {
    "cp": (64, 128, 256, 512),
    "tucker": (32, 64, 128),
}
EXPECTED_BATCH_SIZES = (256, 512)
EXPECTED_SEEDS = (0, 1, 2)

REQUIRED_COLUMNS = {
    "status",
    "backend_type",
    "region_graph",
    "sum_product_layer",
    "units",
    "batch_size",
    "seed",
    "semiring",
    "torch_compile",
    "epoch",
    "epochs",
    "forward_loss_ms",
    "backward_ms",
    "reserved_memory_bytes",
}

RUN_KEY = [
    "seed",
    "region_graph",
    "sum_product_layer",
    "units",
    "batch_size",
    "backend_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the compiled Cirkit region-graph benchmark as tightly cropped PDFs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cirkit_region_graph_benchmark_v3.csv"),
        help="Benchmark CSV produced by benchmark_cirkit_region_graphs.sh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cirkit_region_graph_plots"),
        help="Directory for generated PDFs.",
    )
    parser.add_argument("--confidence", type=float, default=95.0, help="Bootstrap confidence interval percentage across seeds.")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Bootstrap resamples used for each confidence interval.")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every requested run not covered by a recorded OOM configuration is complete.")
    parser.add_argument("--combined", action="store_true", help="Combine configurations into a 4-row by 2-column grid, producing three figures.")
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="pdf", help="Output format. 'both' writes PDF and PNG copies.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution in dots per inch.")
    args = parser.parse_args()
    if not 0.0 < args.confidence < 100.0:
        parser.error("--confidence must be between 0 and 100")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def backend_label(row: pd.Series) -> str:
    if row["backend_type"] == "cirkit":
        return "Cirkit"
    if row["backend_type"] != "xe":
        raise ValueError(f"Unknown backend type: {row['backend_type']!r}")
    if row["semiring"] == "lse-sum":
        return "XE logspace"
    if row["semiring"] == "scaled-max":
        return "XE scaled"
    raise ValueError(f"Unknown XE semiring: {row['semiring']!r}")


def load_seed_summaries(input_path: Path, *, require_complete: bool) -> pd.DataFrame:
    all_results = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS - set(all_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    compiled = all_results["torch_compile"].astype(str).str.lower().isin({"1", "true"})
    all_results = all_results.loc[compiled].copy()
    for column in ("units", "batch_size", "seed"):
        all_results[column] = pd.to_numeric(all_results[column], errors="raise")
    all_results["backend_label"] = all_results.apply(backend_label, axis=1)

    errors = all_results["error"] if "error" in all_results else pd.Series("", index=all_results.index)
    oom_rows = all_results.loc[all_results["status"].ne("ok") & errors.fillna("").str.contains("out of memory", case=False)]
    oom_signatures = {
        (
            row.region_graph,
            row.sum_product_layer,
            int(row.units),
            int(row.batch_size),
            row.backend_label,
        )
        for row in oom_rows.itertuples(index=False)
    }

    results = all_results.loc[all_results["status"].eq("ok")].copy()
    if results.empty:
        raise ValueError(f"{input_path} contains no successful torch.compile benchmark rows")

    numeric_columns = (
        "epoch",
        "epochs",
        "forward_loss_ms",
        "backward_ms",
        "reserved_memory_bytes",
    )
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="raise")

    results["runtime_seconds"] = (results["forward_loss_ms"] + results["backward_ms"]) / 1000.0
    # New benchmark files record the reservation high-water mark directly.
    # Older files retain the allocator's high-water reservation at epoch end,
    # so current reserved bytes are the best available measure of required VRAM.
    reserved_bytes = results["reserved_memory_bytes"]
    if "peak_reserved_memory_bytes" in results:
        peak_reserved_bytes = pd.to_numeric(results["peak_reserved_memory_bytes"], errors="coerce")
        reserved_bytes = peak_reserved_bytes.fillna(reserved_bytes)
    results["required_vram_gib"] = reserved_bytes / float(1024**3)

    # A resumed sweep can retain failed rows and, after an unusual interruption
    # during merging, repeated successful epochs. Keep only the latest copy so a
    # rerun cannot receive extra statistical weight.
    epoch_key = [*RUN_KEY, "epoch"]
    results = results.drop_duplicates(epoch_key, keep="last")

    summaries = (
        results.groupby(RUN_KEY, as_index=False, observed=True)
        .agg(
            runtime_seconds=("runtime_seconds", "median"),
            required_vram_gib=("required_vram_gib", "max"),
            measured_epochs=("epoch", "nunique"),
            expected_epochs=("epochs", "max"),
        )
        .sort_values(RUN_KEY)
    )
    complete_runs = summaries["measured_epochs"].eq(summaries["expected_epochs"])
    if not complete_runs.all():
        num_incomplete = int((~complete_runs).sum())
        print(f"warning: ignoring {num_incomplete} runs without all measured epochs", file=sys.stderr)
        summaries = summaries.loc[complete_runs].copy()

    observed = {
        (
            int(row.seed),
            row.region_graph,
            row.sum_product_layer,
            int(row.units),
            int(row.batch_size),
            row.backend_label,
        )
        for row in summaries.itertuples(index=False)
    }
    expected = {
        (seed, region_graph, layer, units, batch_size, backend)
        for seed in EXPECTED_SEEDS
        for region_graph in REGION_GRAPH_ORDER
        for layer in LAYER_ORDER
        for units in EXPECTED_UNITS[layer]
        for batch_size in EXPECTED_BATCH_SIZES
        for backend in BACKEND_ORDER
    }
    unavailable_runs = {
        run
        for run in expected
        if (run[1], run[2], run[3], run[4], run[5]) in oom_signatures
    }
    if unavailable_runs:
        print(
            f"note: excluding {len(unavailable_runs)} requested runs across {len(oom_signatures)} recorded OOM configurations",
            file=sys.stderr,
        )
    missing_runs = expected - unavailable_runs - observed
    if missing_runs:
        message = f"benchmark is incomplete: {len(missing_runs)} of {len(expected)} requested runs are missing"
        if require_complete:
            raise ValueError(message)
        print(f"warning: {message}", file=sys.stderr)

    if summaries.empty:
        raise ValueError(f"{input_path} contains no complete benchmark runs")
    return summaries


def speedup_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    comparison_key = ["seed", "region_graph", "sum_product_layer", "units", "batch_size"]
    runtimes = summaries.pivot(index=comparison_key, columns="backend_label", values="runtime_seconds")
    speedups: list[pd.DataFrame] = []
    for backend in XE_BACKEND_ORDER:
        if "Cirkit" not in runtimes or backend not in runtimes:
            continue
        speedup = (runtimes["Cirkit"] / runtimes[backend]).dropna().rename("speedup").reset_index()
        speedup["backend_label"] = backend
        speedups.append(speedup)
    if not speedups:
        return pd.DataFrame(columns=[*comparison_key, "speedup", "backend_label"])
    return pd.concat(speedups, ignore_index=True)


def configure_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.transparent": False,
        }
    )


def requested_extensions(output_format: str) -> tuple[str, ...]:
    return ("pdf", "png") if output_format == "both" else (output_format,)


def make_output_paths(output_dir: Path, stem: str, extensions: tuple[str, ...]) -> list[Path]:
    return [output_dir / f"{stem}.{extension}" for extension in extensions]


def save_tightly_cropped_figures(fig: plt.Figure, paths: list[Path], *, dpi: int) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Drawing first lets Matplotlib include the final text, legend, and tick
    # extents in the tight bounding box. Zero padding leaves no page margin while
    # retaining the complete artist bounding boxes.
    fig.canvas.draw()
    for path in paths:
        fig.savefig(path, format=path.suffix.removeprefix("."), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def draw_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    backend_order: tuple[str, ...],
    confidence: float,
    bootstrap_samples: int,
    reference_line: float | None = None,
    y_axis_bottom: float = 0.0,
    legend: bool = True,
) -> None:
    if reference_line is not None:
        ax.axhline(reference_line, color="#777777", linestyle="--", linewidth=1, zorder=0)
    if data.empty:
        ax.text(0.5, 0.5, "No complete data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Number of units")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=y_axis_bottom)
        return
    sns.lineplot(
        data=data,
        x="units",
        y=metric,
        hue="backend_label",
        hue_order=backend_order,
        palette=BACKEND_PALETTE,
        style="backend_label",
        style_order=backend_order,
        markers=BACKEND_MARKERS,
        dashes=False,
        estimator="median",
        errorbar=("ci", confidence),
        n_boot=bootstrap_samples,
        seed=0,
        markersize=5,
        linewidth=2,
        ax=ax,
    )
    units = sorted(data["units"].unique())
    ax.set_xticks(units)
    ax.set_xlabel("Number of units")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=y_axis_bottom)
    ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(title=None, loc="best")


def plot_line(
    data: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    backend_order: tuple[str, ...],
    confidence: float,
    bootstrap_samples: int,
    paths: list[Path],
    dpi: int,
    reference_line: float | None = None,
    y_axis_bottom: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0), layout="constrained")
    draw_line(
        ax,
        data,
        metric=metric,
        ylabel=ylabel,
        backend_order=backend_order,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        reference_line=reference_line,
        y_axis_bottom=y_axis_bottom,
    )
    ax.set_title(title)
    save_tightly_cropped_figures(fig, paths, dpi=dpi)


def plot_all(
    summaries: pd.DataFrame,
    *,
    output_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    extensions: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    speedups = speedup_summaries(summaries)
    generated_paths: list[Path] = []
    for region_graph in REGION_GRAPH_ORDER:
        for layer in LAYER_ORDER:
            for batch_size in EXPECTED_BATCH_SIZES:
                selection = (
                    summaries["region_graph"].eq(region_graph)
                    & summaries["sum_product_layer"].eq(layer)
                    & summaries["batch_size"].eq(batch_size)
                )
                subset = summaries.loc[selection]
                if subset.empty:
                    continue

                descriptor = f"{layer.upper()}, {region_graph}, batch {batch_size}"
                suffix = f"{layer}_{region_graph}_batch-{batch_size}"
                runtime_paths = make_output_paths(output_dir, f"runtime_{suffix}", extensions)
                plot_line(
                    subset,
                    metric="runtime_seconds",
                    ylabel="Forward + loss + backward per epoch (s)",
                    title=f"Runtime — {descriptor}",
                    backend_order=BACKEND_ORDER,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    paths=runtime_paths,
                    dpi=dpi,
                )
                generated_paths.extend(runtime_paths)

                memory_paths = make_output_paths(output_dir, f"peak-memory_{suffix}", extensions)
                plot_line(
                    subset,
                    metric="required_vram_gib",
                    ylabel="Required VRAM (GiB)",
                    title=f"Required VRAM — {descriptor}",
                    backend_order=BACKEND_ORDER,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    paths=memory_paths,
                    dpi=dpi,
                )
                generated_paths.extend(memory_paths)

                speedup_selection = (
                    speedups["region_graph"].eq(region_graph)
                    & speedups["sum_product_layer"].eq(layer)
                    & speedups["batch_size"].eq(batch_size)
                )
                speedup_subset = speedups.loc[speedup_selection]
                if speedup_subset.empty:
                    continue
                speedup_paths = make_output_paths(output_dir, f"speedup_{suffix}", extensions)
                plot_line(
                    speedup_subset,
                    metric="speedup",
                    ylabel="Speedup over Cirkit (×)",
                    title=f"Speedup — {descriptor}",
                    backend_order=XE_BACKEND_ORDER,
                    confidence=confidence,
                    bootstrap_samples=bootstrap_samples,
                    paths=speedup_paths,
                    dpi=dpi,
                    reference_line=1.0,
                    y_axis_bottom=0.9,
                )
                generated_paths.extend(speedup_paths)
    return generated_paths


def plot_combined_metric(
    data: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    figure_title: str,
    backend_order: tuple[str, ...],
    confidence: float,
    bootstrap_samples: int,
    paths: list[Path],
    dpi: int,
    reference_line: float | None = None,
    y_axis_bottom: float = 0.0,
) -> None:
    row_configurations = tuple((region_graph, layer) for region_graph in REGION_GRAPH_ORDER for layer in LAYER_ORDER)
    fig, axes = plt.subplots(len(row_configurations), len(EXPECTED_BATCH_SIZES), figsize=(11.4, 12.5), layout="constrained", squeeze=False)
    legend_artists: dict[str, object] = {}

    for row_index, (region_graph, layer) in enumerate(row_configurations):
        for column_index, batch_size in enumerate(EXPECTED_BATCH_SIZES):
            ax = axes[row_index, column_index]
            selection = (
                data["region_graph"].eq(region_graph)
                & data["sum_product_layer"].eq(layer)
                & data["batch_size"].eq(batch_size)
            )
            subset = data.loc[selection]
            draw_line(
                ax,
                subset,
                metric=metric,
                ylabel="",
                backend_order=backend_order,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                reference_line=reference_line,
                y_axis_bottom=y_axis_bottom,
                legend=True,
            )

            handles, labels = ax.get_legend_handles_labels()
            legend_artists.update(zip(labels, handles, strict=True))
            if ax.legend_ is not None:
                ax.legend_.remove()
            if row_index == 0:
                ax.set_title(f"Batch size {batch_size}")
            ax.set_xlabel("Number of units" if row_index == len(row_configurations) - 1 else "")
            if column_index == len(EXPECTED_BATCH_SIZES) - 1:
                ax.annotate(
                    f"{region_graph}\n{layer.upper()}",
                    xy=(1.03, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    annotation_clip=False,
                )

    ordered_labels = [label for label in backend_order if label in legend_artists]
    if ordered_labels:
        fig.legend(
            [legend_artists[label] for label in ordered_labels],
            ordered_labels,
            loc="outside lower center",
            ncol=len(ordered_labels),
            title=None,
        )
    fig.suptitle(figure_title)
    fig.supylabel(ylabel)
    save_tightly_cropped_figures(fig, paths, dpi=dpi)


def plot_combined(
    summaries: pd.DataFrame,
    *,
    output_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    extensions: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    speedups = speedup_summaries(summaries)
    specifications = (
        (
            summaries,
            "runtime_seconds",
            "Forward + loss + backward per epoch (s)",
            "Forward + loss + backward runtime",
            BACKEND_ORDER,
            "runtime_combined",
            None,
            0.0,
        ),
        (
            summaries,
            "required_vram_gib",
            "Required VRAM (GiB)",
            "Required VRAM",
            BACKEND_ORDER,
            "peak-memory_combined",
            None,
            0.0,
        ),
        (
            speedups,
            "speedup",
            "Speedup over Cirkit (×)",
            "Speedup over Cirkit",
            XE_BACKEND_ORDER,
            "speedup_combined",
            1.0,
            0.9,
        ),
    )
    generated_paths: list[Path] = []
    for data, metric, ylabel, title, backend_order, stem, reference_line, y_axis_bottom in specifications:
        paths = make_output_paths(output_dir, stem, extensions)
        plot_combined_metric(
            data,
            metric=metric,
            ylabel=ylabel,
            figure_title=title,
            backend_order=backend_order,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            paths=paths,
            dpi=dpi,
            reference_line=reference_line,
            y_axis_bottom=y_axis_bottom,
        )
        generated_paths.extend(paths)
    return generated_paths


def main() -> None:
    args = parse_args()
    configure_style()
    summaries = load_seed_summaries(args.input, require_complete=args.require_complete)
    extensions = requested_extensions(args.format)
    plot_function = plot_combined if args.combined else plot_all
    generated_paths = plot_function(
        summaries,
        output_dir=args.output_dir,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        extensions=extensions,
        dpi=args.dpi,
    )
    print(f"generated {len(generated_paths)} tightly cropped plot files in {args.output_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
