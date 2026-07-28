from __future__ import annotations

import argparse
import csv
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
SPEEDUP_CONFIGURATION_ORDER = (
    "Quad tree, batch 256",
    "Quad tree, batch 512",
    "Quad graph, batch 256",
    "Quad graph, batch 512",
)
SPEEDUP_CONFIGURATION_PALETTE = {
    "Quad tree, batch 256": "#0072B2",
    "Quad tree, batch 512": "#56B4E9",
    "Quad graph, batch 256": "#D55E00",
    "Quad graph, batch 512": "#E69F00",
}
SPEEDUP_CONFIGURATION_MARKERS = {
    "Quad tree, batch 256": "o",
    "Quad tree, batch 512": "s",
    "Quad graph, batch 256": "^",
    "Quad graph, batch 512": "D",
}
REGION_GRAPH_ORDER = ("quad-tree-2", "quad-graph")
LAYER_ORDER = ("cp", "tucker")
EXPECTED_UNITS = {
    "cp": (64, 128, 256, 512, 1024),
    "tucker": (32, 64, 128),
}
EXPECTED_BATCH_SIZES = (256, 512)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)

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

SCHEMA_EXTENSION_FIELDS = (
    "peak_reserved_memory_bytes",
    "peak_reserved_memory_mib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the compiled Cirkit region-graph benchmark as tightly cropped PDFs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cirkit_region_graph_benchmark_v4.csv"),
        help="Benchmark CSV produced by benchmark_cirkit_region_graphs.sh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cirkit_region_graph_plots"),
        help="Directory for generated PDFs.",
    )
    parser.add_argument(
        "--oom-reference-input",
        type=Path,
        default=None,
        help="Optional older benchmark CSV used only to identify configurations that are known to OOM.",
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
    if row["semiring"] in {"scaled-min", "scaled-max", "scaled-sum"}:
        return "XE scaled"
    raise ValueError(f"Unknown XE semiring: {row['semiring']!r}")


def read_benchmark_results(input_path: Path) -> pd.DataFrame:
    with input_path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{input_path} has no CSV header")

        fieldnames = list(header)
        missing_extension_fields = [field for field in SCHEMA_EXTENSION_FIELDS if field not in fieldnames]
        if missing_extension_fields and "reserved_memory_bytes" in fieldnames:
            insertion_index = fieldnames.index("reserved_memory_bytes")
            fieldnames[insertion_index:insertion_index] = missing_extension_fields

        records: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) == len(fieldnames):
                row_fieldnames = fieldnames
            elif fieldnames != header and len(row) == len(header):
                row_fieldnames = header
            else:
                raise ValueError(
                    f"{input_path}:{line_number} has {len(row)} fields; "
                    f"expected {len(header)} or {len(fieldnames)}"
                )
            records.append(dict(zip(row_fieldnames, row, strict=True)))

    return pd.DataFrame.from_records(records, columns=fieldnames)


def _recorded_oom_signatures(results: pd.DataFrame) -> set[tuple[str, str, int, int, str]]:
    if "error" not in results:
        return set()
    errors = results["error"].fillna("")
    oom_rows = results.loc[results["status"].ne("ok") & errors.str.contains("out of memory", case=False)]
    return {
        (
            row.region_graph,
            row.sum_product_layer,
            int(row.units),
            int(row.batch_size),
            row.backend_label,
        )
        for row in oom_rows.itertuples(index=False)
    }


def load_seed_summaries(
    input_path: Path,
    *,
    require_complete: bool,
    oom_reference_input: Path | None = None,
) -> pd.DataFrame:
    all_results = read_benchmark_results(input_path)
    missing_columns = REQUIRED_COLUMNS - set(all_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    compiled = all_results["torch_compile"].astype(str).str.lower().isin({"1", "true"})
    all_results = all_results.loc[compiled].copy()
    for column in ("units", "batch_size", "seed"):
        all_results[column] = pd.to_numeric(all_results[column], errors="raise")
    all_results["backend_label"] = all_results.apply(backend_label, axis=1)

    oom_signatures = _recorded_oom_signatures(all_results)
    if oom_reference_input is not None:
        oom_reference = read_benchmark_results(oom_reference_input)
        required_oom_columns = {
            "status",
            "error",
            "backend_type",
            "region_graph",
            "sum_product_layer",
            "units",
            "batch_size",
            "semiring",
        }
        missing_oom_columns = required_oom_columns - set(oom_reference.columns)
        if missing_oom_columns:
            missing = ", ".join(sorted(missing_oom_columns))
            raise ValueError(f"{oom_reference_input} is missing required OOM-reference columns: {missing}")
        if "torch_compile" in oom_reference:
            compiled_reference = oom_reference["torch_compile"].astype(str).str.lower().isin({"1", "true"})
            oom_reference = oom_reference.loc[compiled_reference].copy()
        for column in ("units", "batch_size"):
            oom_reference[column] = pd.to_numeric(oom_reference[column], errors="raise")
        oom_reference["backend_label"] = oom_reference.apply(backend_label, axis=1)
        oom_signatures.update(_recorded_oom_signatures(oom_reference))

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


def memory_reduction_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    comparison_key = ["seed", "region_graph", "sum_product_layer", "units", "batch_size"]
    memory = summaries.pivot(index=comparison_key, columns="backend_label", values="required_vram_gib")
    if "Cirkit" not in memory or "XE scaled" not in memory:
        return pd.DataFrame(columns=[*comparison_key, "memory_reduction_pct"])
    reduction = (
        100.0 * (1.0 - memory["XE scaled"] / memory["Cirkit"])
    ).dropna().rename("memory_reduction_pct")
    return reduction.reset_index()


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
        ax.set_xlabel("Units per input and sum layer")
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


def plot_relative_configuration_metric_by_layer(
    values: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
    output_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    extensions: tuple[str, ...],
    dpi: int,
    reference_line: float,
    y_axis_bottom: float | dict[str, float] | None = None,
    excluded_units: tuple[int, ...] = (),
) -> list[Path]:
    generated_paths: list[Path] = []
    graph_labels = {"quad-tree-2": "Quad tree", "quad-graph": "Quad graph"}
    values = values.loc[~values["units"].isin(excluded_units)].copy()
    values["configuration_label"] = [
        f"{graph_labels[region_graph]}, batch {int(batch_size)}"
        for region_graph, batch_size in zip(
            values["region_graph"],
            values["batch_size"],
            strict=True,
        )
    ]

    for layer in LAYER_ORDER:
        subset = values.loc[values["sum_product_layer"].eq(layer)]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.axhline(reference_line, color="#777777", linestyle="--", linewidth=1, zorder=0)
        sns.lineplot(
            data=subset,
            x="units",
            y=metric,
            hue="configuration_label",
            hue_order=SPEEDUP_CONFIGURATION_ORDER,
            palette=SPEEDUP_CONFIGURATION_PALETTE,
            style="configuration_label",
            style_order=SPEEDUP_CONFIGURATION_ORDER,
            markers=SPEEDUP_CONFIGURATION_MARKERS,
            dashes=False,
            estimator="median",
            errorbar=("ci", confidence),
            n_boot=bootstrap_samples,
            seed=0,
            markersize=5,
            linewidth=2,
            ax=ax,
        )
        ax.set_xticks(sorted(subset["units"].unique()))
        ax.set_xlabel("Units per input and sum layer")
        ax.set_ylabel(ylabel)
        layer_y_axis_bottom = (
            y_axis_bottom.get(layer)
            if isinstance(y_axis_bottom, dict)
            else y_axis_bottom
        )
        if layer_y_axis_bottom is not None:
            ax.set_ylim(bottom=layer_y_axis_bottom)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(title=None, loc="lower right")
        fig.tight_layout()

        paths = make_output_paths(output_dir, f"{output_stem}_{layer}", extensions)
        save_tightly_cropped_figures(fig, paths, dpi=dpi)
        generated_paths.extend(paths)
    return generated_paths


def plot_scaled_speedups_by_layer(
    speedups: pd.DataFrame,
    *,
    output_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    extensions: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    scaled = speedups.loc[speedups["backend_label"].eq("XE scaled")].copy()
    cp_64 = scaled["sum_product_layer"].eq("cp") & scaled["units"].eq(64)
    scaled = scaled.loc[~cp_64]
    return plot_relative_configuration_metric_by_layer(
        scaled,
        metric="speedup",
        ylabel="Speedup over Cirkit",
        title="Extended Einsum Speedup over Cirkit",
        output_stem="speedup",
        output_dir=output_dir,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        extensions=extensions,
        dpi=dpi,
        reference_line=1.0,
        y_axis_bottom={"cp": 1.2, "tucker": 1.05},
    )


def plot_scaled_memory_reduction_by_layer(
    reductions: pd.DataFrame,
    *,
    output_dir: Path,
    confidence: float,
    bootstrap_samples: int,
    extensions: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    return plot_relative_configuration_metric_by_layer(
        reductions,
        metric="memory_reduction_pct",
        ylabel="Memory reduction over Cirkit (%)",
        title="Extended Einsum Memory Reduction over Cirkit",
        output_stem="memory-reduction",
        output_dir=output_dir,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        extensions=extensions,
        dpi=dpi,
        reference_line=0.0,
    )


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
    memory_reductions = memory_reduction_summaries(summaries)
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

    generated_paths.extend(
        plot_scaled_speedups_by_layer(
            speedups,
            output_dir=output_dir,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            extensions=extensions,
            dpi=dpi,
        )
    )
    generated_paths.extend(
        plot_scaled_memory_reduction_by_layer(
            memory_reductions,
            output_dir=output_dir,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            extensions=extensions,
            dpi=dpi,
        )
    )
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
    extensions = requested_extensions(args.format)
    summaries = load_seed_summaries(
        args.input,
        require_complete=args.require_complete,
        oom_reference_input=args.oom_reference_input,
    )
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
