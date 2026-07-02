from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import random
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import matplotlib.pyplot as plt
import numpy as np
import torch
from cirkit.symbolic.layers import HadamardLayer, InputLayer, KroneckerLayer, SumLayer
from cirkit.templates import data_modalities, utils

import extended_einsum.interface as xe
from extended_einsum.interface.tensor_expression import Parameter
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.preprocess import FoldSameShapedOperations, OptimizeContractionPaths
from extended_einsum.visualization import plot_expression_dag

EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_symbolic_circuit(
    *,
    width: int,
    height: int,
    num_units: int,
    sum_product_layer: str,
):
    return data_modalities.image_data(
        (1, width, height),
        region_graph="quad-tree-2",
        input_layer="categorical",
        num_input_units=num_units,
        sum_product_layer=sum_product_layer,
        num_sum_units=num_units,
        sum_weight_param=utils.Parameterization(
            activation="softmax",
            initialization="normal",
        ),
    )


def get_scope_id(scope: utils.Scope) -> int:
    if len(scope) != 1:
        raise ValueError(f"Expected a singleton scope, got {scope!r}")
    return next(iter(scope))


def generate_symbols(count: int) -> str:
    if count > len(EINSUM_SYMBOLS):
        raise ValueError(f"Cannot generate {count} unique einsum symbols")
    return EINSUM_SYMBOLS[:count]


def to_xe_expression(symbolic_circuit, layer, data_by_scope):
    children = symbolic_circuit.layer_inputs(layer)
    child_nodes = [to_xe_expression(symbolic_circuit, child, data_by_scope) for child in children]

    if not children:
        scope_id = get_scope_id(layer.scope)
        return xe.select(data_by_scope, scope_id)

    if isinstance(layer, HadamardLayer):
        format_string = ",".join(["ab"] * len(child_nodes)) + "->ab"
        if not all(child.shape == child_nodes[0].shape for child in child_nodes):
            raise ValueError("Hadamard layer children must have the same shape")
        return xe.einsum(format_string, *child_nodes)

    if isinstance(layer, KroneckerLayer):
        child_indices = generate_symbols(len(child_nodes) + 1)
        batched_child_indices = [f"a{symbol}" for symbol in child_indices[1:]]
        format_string = ",".join(batched_child_indices) + "->" + "".join(child_indices)
        return xe.einsum(format_string, *child_nodes)

    if isinstance(layer, SumLayer):
        if len(children) != 1:
            raise ValueError("Sum layers are expected to have exactly one child")
        child = child_nodes[0]
        child_indices = generate_symbols(len(child.shape))
        weight_shape = child.shape[1:] + (layer.params["weight"].shape[0],)
        weight_indices = generate_symbols(len(child.shape) + 1)[1:]
        out_indices = child_indices[0] + weight_indices[-1]
        format_string = f"{child_indices},{weight_indices}->{out_indices}"
        weight_logits = Parameter(xe.array(torch.empty(weight_shape, dtype=torch.float32)))
        weights = xe.softmax(weight_logits, axis=0)
        return xe.einsum(format_string, child, weights)

    raise NotImplementedError(f"Unsupported Cirkit layer: {layer!r}")


def build_quad_tree_program(
    *,
    width: int,
    height: int,
    num_units: int,
    batch_size: int,
    sum_product_layer: str,
) -> RichProgram:
    symbolic_circuit = make_symbolic_circuit(
        width=width,
        height=height,
        num_units=num_units,
        sum_product_layer=sum_product_layer,
    )
    input_layer = next(layer for layer in symbolic_circuit.layers if isinstance(layer, InputLayer))
    data_by_scope = xe.array(
        torch.empty(
            (symbolic_circuit.num_variables, batch_size, input_layer.params["probs"].shape[0]),
            dtype=torch.float32,
        )
    )

    expression = to_xe_expression(symbolic_circuit, symbolic_circuit.layers[-1], data_by_scope)
    program, _inputs = xe.extract_program(expression, stability_mode="none")
    return program


def input_labels(program: RichProgram) -> dict[int, str]:
    data_index = 0
    parameter_index = 0
    labels: dict[int, str] = {}
    for input_id in range(program.n_inputs):
        if input_id in program.parameter_indices:
            labels[input_id] = f"Param {parameter_index}"
            parameter_index += 1
        else:
            labels[input_id] = f"Data {data_index}"
            data_index += 1
    return labels


def save_program_graph(
    program: RichProgram,
    *,
    output_path: Path,
    title: str,
    vertical_spacing: float,
    horizontal_spacing: float,
) -> None:
    ax = plot_expression_dag(
        program,
        input_labels=input_labels(program),
        show_tensor_ids=True,
        collapse_fused_einsums=False,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        max_operator_label_width=28,
    )
    ax.set_title(title, fontsize=11)
    ax.figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(ax.figure)


def summarize_program(program: RichProgram) -> str:
    operator_counts = Counter(instruction.operator.name for instruction in program.instructions)
    return f"inputs={program.n_inputs}, instructions={len(program.instructions)}, operators={dict(sorted(operator_counts.items()))}"


def generate_visualizations(
    *,
    layers: Iterable[str],
    output_dir: Path,
    width: int,
    height: int,
    num_units: int,
    batch_size: int,
    vertical_spacing: float,
    horizontal_spacing: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        original = build_quad_tree_program(
            width=width,
            height=height,
            num_units=num_units,
            batch_size=batch_size,
            sum_product_layer=layer,
        )
        folded = FoldSameShapedOperations.apply(original)
        optimized = OptimizeContractionPaths.apply(folded)

        stages = (
            ("01_raw", "before preprocessing", original),
            ("02_folded", "after folding preprocessing", folded),
            ("03_optimized_path", "after folding and contraction path optimization", optimized),
        )
        for file_prefix, stage_name, program in stages:
            output_path = output_dir / f"quad_tree_{layer}_{file_prefix}.pdf"
            save_program_graph(
                program,
                output_path=output_path,
                title=f"quad-tree-2 {layer}: {stage_name}",
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing,
            )
            print(f"{output_path}: {summarize_program(program)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save XE DAG visualizations for quad-tree CP and Tucker circuits.")
    parser.add_argument("--output-dir", type=Path, default=_THIS_DIR / "visualizations")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sum-product-layer", choices=("cp", "tucker", "both"), default="both")
    parser.add_argument("--vertical-spacing", type=float, default=1.0)
    parser.add_argument("--horizontal-spacing", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    layers = ("cp", "tucker") if args.sum_product_layer == "both" else (args.sum_product_layer,)
    generate_visualizations(
        layers=layers,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        num_units=args.units,
        batch_size=args.batch_size,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
    )


if __name__ == "__main__":
    main()
