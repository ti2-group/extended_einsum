from __future__ import annotations

# This file is also executed directly from experiments/.
# ruff: noqa: E402,I001

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.correctness import (
    CSV_FIELDS,
    MNIST_TRAINING_CSV_FIELDS,
    MNIST_TRAINING_RESULTS,
    RESULTS,
)
from experiments.plot_common import GRAPH_LABELS, configure_style, save_pdf

NUMERIC_COLUMNS = (
    "seed",
    "width",
    "height",
    "units",
    "batch_size",
    "pixel_values",
    "depth",
    "forward_relative_l2",
    "forward_max_absolute_error",
    "data_gradient_relative_l2",
    "parameter_gradient_relative_l2",
    "worst_parameter_gradient_relative_l2",
    "gradient_max_absolute_error",
    "forward_finite_fraction",
    "gradient_finite_fraction",
    "reference_forward_finite_fraction",
    "reference_gradient_finite_fraction",
    "parameter_tensors",
    "parameters",
)
VARIANT_LABELS = {
    "unstable-fp32": "Raw FP32",
    "unstable-fp64": "Raw FP64",
    "scaled-max-fp32": "Scaled FP32",
    "logspace-max-fp32": "Log-space FP32",
}
VARIANT_STYLES = {
    "unstable-fp32": ("#CC79A7", "X"),
    "unstable-fp64": ("#666666", "s"),
    "scaled-max-fp32": ("#0072B2", "o"),
    "logspace-max-fp32": ("#D55E00", "^"),
}
LAYER_LABELS = {"cp": "CP", "tucker": "Tucker"}
TRAINING_VARIANT_LABELS = {
    "cirkit": "Cirkit",
    "logspace": "XE log-space",
    "scaled-max": "XE scaled-max",
}
TRAINING_VARIANT_STYLES = {
    "cirkit": ("#666666", "s"),
    "logspace": ("#D55E00", "^"),
    "scaled-max": ("#0072B2", "o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Render the correctness CSV as supplementary LaTeX tables and an underflow plot."))
    parser.add_argument("--input", type=Path, default=RESULTS)
    parser.add_argument(
        "--mnist-training-input",
        type=Path,
        default=MNIST_TRAINING_RESULTS,
    )
    parser.add_argument("--plot-dir", type=Path, default=_HERE / "plots")
    parser.add_argument("--table-dir", type=Path, default=_HERE / "tables")
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    missing = set(CSV_FIELDS) - set(results)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    failed = results.loc[results["status"].ne("ok")]
    if not failed.empty:
        print(f"ignoring {len(failed)} failed row(s) recorded in {path}")
    results = results.loc[results["status"].eq("ok")].copy()
    for column in NUMERIC_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")
    key = [
        "suite",
        "seed",
        "region_graph",
        "layer",
        "width",
        "height",
        "units",
        "batch_size",
        "pixel_values",
        "depth",
        "variant",
        "matmul_precision",
        "device",
        "device_name",
        "torch_compile",
    ]
    return results.drop_duplicates(key, keep="last")


def scientific(value: float) -> str:
    if math.isnan(value):
        return "--"
    if math.isinf(value):
        return r"$\infty$"
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    coefficient = value / 10**exponent
    return rf"${coefficient:.1f}\mathbin{{\times}}10^{{{exponent}}}$"


def percent(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{100.0 * value:.1f}\\%"


def read_mnist_training_results(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        print(f"no MNIST training results at {path}; skipping trajectory outputs")
        return pd.DataFrame()
    results = pd.read_csv(path)
    missing = set(MNIST_TRAINING_CSV_FIELDS) - set(results)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing))}"
        )
    failed = results.loc[results["status"].ne("ok")]
    if not failed.empty:
        print(
            f"ignoring {len(failed)} failed MNIST training attempt(s) "
            f"recorded in {path}"
        )
    results = results.loc[results["status"].eq("ok")].copy()
    for column in (
        "seed",
        "units",
        "batch_size",
        "epoch",
        "epochs",
        "batches",
        "samples",
        "avg_nll",
        "lr",
    ):
        results[column] = pd.to_numeric(results[column], errors="raise")
    if results.empty:
        return results
    configuration_columns = [
        "seed",
        "region_graph",
        "layer",
        "units",
        "batch_size",
        "epochs",
        "max_batches",
        "lr",
        "device_name",
    ]
    latest_configuration = tuple(
        results.iloc[-1][column]
        for column in configuration_columns
    )
    selected = results
    for column, value in zip(
        configuration_columns,
        latest_configuration,
        strict=True,
    ):
        if pd.isna(value):
            selected = selected.loc[selected[column].isna()]
        else:
            selected = selected.loc[selected[column].eq(value)]
    key = [*configuration_columns, "variant", "epoch"]
    selected = selected.drop_duplicates(key, keep="last")
    expected_epochs = set(
        range(1, int(selected["epochs"].iloc[0]) + 1)
    )
    incomplete = {
        variant: sorted(
            expected_epochs
            - set(group["epoch"].astype(int))
        )
        for variant, group in selected.groupby("variant")
        if set(group["epoch"].astype(int)) != expected_epochs
    }
    if incomplete:
        raise ValueError(
            f"{path} has incomplete MNIST trajectories: {incomplete}"
        )
    return selected


def plot_mnist_training(
    results: pd.DataFrame,
    output: Path,
) -> None:
    if results.empty:
        return
    fig, axis = plt.subplots(
        figsize=(6.8, 3.8),
        layout="constrained",
    )
    for variant in TRAINING_VARIANT_LABELS:
        selected = results.loc[
            results["variant"].eq(variant)
        ].sort_values("epoch")
        if selected.empty:
            continue
        color, marker = TRAINING_VARIANT_STYLES[variant]
        axis.plot(
            selected["epoch"],
            selected["avg_nll"],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=TRAINING_VARIANT_LABELS[variant],
        )
    epochs = int(results["epochs"].iloc[0])
    axis.set_xlim((1, epochs) if epochs > 1 else (0.5, 1.5))
    axis.set_xticks(
        np.unique(
            np.linspace(1, epochs, min(6, epochs), dtype=int)
        )
    )
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Average training NLL")
    axis.legend(ncol=3, loc="best")
    save_pdf(fig, output)
    print(f"wrote {output}")


def write_mnist_training_table(
    results: pd.DataFrame,
    output: Path,
) -> None:
    if results.empty:
        return
    final_nlls = {
        variant: float(
            group.sort_values("epoch")["avg_nll"].iloc[-1]
        )
        for variant, group in results.groupby("variant")
    }
    cirkit_final = final_nlls.get("cirkit", math.nan)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Implementation & Epoch 1 NLL & Final NLL & "
            r"Best NLL & Final $\Delta$ vs. Cirkit \\"
        ),
        r"\midrule",
    ]
    for variant in TRAINING_VARIANT_LABELS:
        selected = results.loc[
            results["variant"].eq(variant)
        ].sort_values("epoch")
        if selected.empty:
            continue
        first = float(selected["avg_nll"].iloc[0])
        final = float(selected["avg_nll"].iloc[-1])
        best = float(selected["avg_nll"].min())
        delta = final - cirkit_final
        lines.append(
            " & ".join(
                (
                    TRAINING_VARIANT_LABELS[variant],
                    f"{first:.4f}",
                    f"{final:.4f}",
                    f"{best:.4f}",
                    "--" if variant == "cirkit" else f"{delta:+.4f}",
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"wrote {output}")


def write_agreement_table(results: pd.DataFrame, output: Path) -> None:
    rows = results.loc[results["suite"].eq("agreement")]
    if rows.empty:
        print("no agreement rows; skipping agreement table")
        return
    grouped = rows.groupby(
        [
            "device_name",
            "torch_compile",
            "region_graph",
            "layer",
            "matmul_precision",
            "variant",
        ],
        sort=False,
    )
    lines = [
        r"\begin{tabular}{lllllrrrr}",
        r"\toprule",
        (
            r"Execution & Graph & Layer & FP32 math & Evaluation & "
            r"Forward rel. & Data grad rel. & Param. grad rel. & "
            r"Worst tensor rel. \\"
        ),
        r"\midrule",
    ]
    ordered_keys = sorted(
        grouped.groups,
        key=lambda key: (
            key[0],
            not bool(key[1]),
            REGION_ORDER.get(key[2], 99),
            LAYER_ORDER.get(key[3], 99),
            PRECISION_ORDER.get(key[4], 99),
            VARIANT_ORDER.get(key[5], 99),
        ),
    )
    for device, compiled, graph, layer, precision, variant in ordered_keys:
        group = grouped.get_group((device, compiled, graph, layer, precision, variant))
        if precision == "highest":
            precision_label = "IEEE"
        elif group["tf32_permitted"].astype(str).str.lower().eq("true").any():
            precision_label = "TF32-permitted"
        else:
            precision_label = "high"
        lines.append(
            " & ".join(
                (
                    f"{device} ({'compiled' if compiled else 'eager'})",
                    GRAPH_LABELS.get(graph, graph),
                    LAYER_LABELS.get(layer, layer),
                    precision_label,
                    "Scaled" if variant == "scaled-max" else "Log-space",
                    scientific(float(group["forward_relative_l2"].max())),
                    scientific(float(group["data_gradient_relative_l2"].max())),
                    scientific(float(group["parameter_gradient_relative_l2"].max())),
                    scientific(float(group["worst_parameter_gradient_relative_l2"].max())),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"wrote {output}")


REGION_ORDER = {"quad-tree-2": 0, "quad-graph": 1}
LAYER_ORDER = {"cp": 0, "tucker": 1}
PRECISION_ORDER = {"highest": 0, "high": 1}
VARIANT_ORDER = {"scaled-max": 0, "logspace-max": 1}


def write_mnist_table(results: pd.DataFrame, output: Path) -> None:
    rows = results.loc[results["suite"].eq("mnist")]
    if rows.empty:
        print("no MNIST rows; skipping MNIST table")
        return
    grouped = rows.groupby(
        ["device_name", "torch_compile", "region_graph", "layer", "units"],
        sort=False,
    )
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        (
            r"Execution & Graph & Layer & Units & Forward rel. & "
            r"Data grad rel. & Param. grad rel. & Finite gradients \\"
        ),
        r"\midrule",
    ]
    ordered_keys = sorted(
        grouped.groups,
        key=lambda key: (
            key[0],
            not bool(key[1]),
            REGION_ORDER.get(key[2], 99),
            LAYER_ORDER.get(key[3], 99),
            key[4],
        ),
    )
    for device, compiled, graph, layer, units in ordered_keys:
        group = grouped.get_group((device, compiled, graph, layer, units))
        lines.append(
            " & ".join(
                (
                    f"{device} ({'compiled' if compiled else 'eager'})",
                    GRAPH_LABELS.get(graph, graph),
                    LAYER_LABELS.get(layer, layer),
                    str(int(units)),
                    scientific(float(group["forward_relative_l2"].max())),
                    scientific(float(group["data_gradient_relative_l2"].max())),
                    scientific(float(group["parameter_gradient_relative_l2"].max())),
                    percent(float(group["gradient_finite_fraction"].min())),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"wrote {output}")


def finite_median(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if len(finite) else math.nan


def plot_underflow(results: pd.DataFrame, output: Path) -> None:
    rows = results.loc[results["suite"].eq("stress")]
    if rows.empty:
        print("no stress rows; skipping underflow plot")
        return
    variants = tuple(
        variant
        for variant in (
            "scaled-max-fp32",
            "logspace-max-fp32",
            "unstable-fp32",
            "unstable-fp64",
        )
        if variant in set(rows["variant"])
    )
    error_floor = np.finfo(np.float64).eps
    legend_entries: dict[str, object] = {}
    fig, (error_axis, finite_axis) = plt.subplots(
        2,
        1,
        figsize=(6.8, 6.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (1.25, 1.0)},
    )
    for variant in variants:
        available_precisions = set(rows.loc[rows["variant"].eq(variant), "matmul_precision"])
        precisions = (
            ("highest", "high")
            if variant in {"scaled-max-fp32", "logspace-max-fp32"}
            else ("highest",)
        )
        for precision in (item for item in precisions if item in available_precisions):
            selected = rows.loc[
                rows["variant"].eq(variant)
                & rows["matmul_precision"].eq(precision)
            ]
            units_label = ""
            if precision == "high" and variant in {"scaled-max-fp32", "logspace-max-fp32"}:
                stress_units = int(selected["units"].max())
                selected = selected.loc[selected["units"].eq(stress_units)]
                units_label = rf", $U={stress_units}$"
            grouped = selected.groupby("depth", sort=True)
            depths = np.asarray(sorted(grouped.groups), dtype=float)
            errors = np.asarray(
                [
                    max(
                        finite_median(
                            grouped.get_group(depth)["forward_relative_l2"]
                        ),
                        error_floor,
                    )
                    for depth in depths
                ]
            )
            finite_gradients = np.asarray([float(grouped.get_group(depth)["gradient_finite_fraction"].median()) for depth in depths])
            color, marker = VARIANT_STYLES[variant]
            linestyle = "--" if precision == "high" else "-"
            precision_label = (
                f" ({'TF32' if precision == 'high' else 'IEEE'}{units_label})"
                if len(precisions) > 1
                else ""
            )
            label = VARIANT_LABELS[variant] + precision_label
            (line,) = error_axis.plot(
                depths,
                errors,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                markeredgewidth=1.4 if variant.startswith("unstable") else 0.8,
                zorder=4 if variant.startswith("unstable") else 2,
                label=label,
            )
            legend_entries[label] = line
            finite_axis.plot(
                depths,
                finite_gradients,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                markeredgewidth=1.4 if variant.startswith("unstable") else 0.8,
                zorder=4 if variant.startswith("unstable") else 2,
            )
    error_axis.axhline(
        error_floor,
        color="#888888",
        linestyle=":",
        linewidth=0.9,
        zorder=1,
    )
    error_axis.text(
        float(rows["depth"].max()),
        1.35 * error_floor,
        r"$\epsilon_{\mathrm{FP64}}$",
        color="#666666",
        ha="right",
        va="bottom",
    )
    error_axis.set_yscale("log")
    error_axis.set_ylabel("Forward relative $L_2$ error")
    ordered_labels = [
        "Raw FP32",
        "Scaled FP32 (IEEE)",
        r"Scaled FP32 (TF32, $U=512$)",
        "Raw FP64",
        "Log-space FP32 (IEEE)",
        r"Log-space FP32 (TF32, $U=512$)",
    ]
    shown_labels = [label for label in ordered_labels if label in legend_entries]
    error_axis.legend(
        [legend_entries[label] for label in shown_labels],
        shown_labels,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.43),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#999999",
        fancybox=False,
        borderpad=0.6,
        columnspacing=1.2,
        handlelength=1.8,
        labelspacing=0.35,
    )
    finite_axis.set_xscale("log", base=2)
    finite_axis.set_ylim(-0.03, 1.03)
    finite_axis.set_yticks((0.0, 0.5, 1.0))
    finite_axis.set_ylabel("Finite gradient fraction")
    finite_axis.set_xlabel("Circuit depth")
    save_pdf(fig, output)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    configure_style()
    results = read_results(args.input)
    mnist_training_results = read_mnist_training_results(
        args.mnist_training_input
    )
    write_agreement_table(
        results,
        args.table_dir / "correctness_agreement.tex",
    )
    write_mnist_table(
        results,
        args.table_dir / "correctness_mnist.tex",
    )
    plot_underflow(
        results,
        args.plot_dir / "correctness_underflow.pdf",
    )
    plot_mnist_training(
        mnist_training_results,
        args.plot_dir / "correctness_mnist_training.pdf",
    )
    write_mnist_training_table(
        mnist_training_results,
        args.table_dir / "correctness_mnist_training.tex",
    )


if __name__ == "__main__":
    main()
