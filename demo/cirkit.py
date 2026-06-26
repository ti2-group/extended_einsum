from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_THIS_DIR):
    sys.path.pop(0)

import torch
from cirkit.symbolic.layers import HadamardLayer, InputLayer, KroneckerLayer, SumLayer
from cirkit.templates import data_modalities, utils

import extended_einsum.interface as xe
from extended_einsum.language import Program, get_operator

WIDTH = 4
HEIGHT = 4
DEFAULT_UNITS = 64
DEFAULT_BATCH_SIZE = 256
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


def _to_xe_recursive(symbolic_circuit, layer, data_by_scope):
    children = symbolic_circuit.layer_inputs(layer)
    child_nodes = [
        _to_xe_recursive(
            symbolic_circuit,
            child,
            data_by_scope,
        )
        for child in children
    ]

    if not children:
        scope_id = get_scope_id(layer.scope)
        return data_by_scope[scope_id]

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
        weight_logits = torch.empty(weight_shape, dtype=torch.float32)
        weights = xe.softmax(weight_logits, axis=0)
        return xe.einsum(format_string, child, weights)

    raise NotImplementedError(f"Unsupported Cirkit layer: {layer!r}")


def translate_cirkit_to_xe(
    symbolic_circuit,
    *,
    batch_size: int,
    stability: str,
) -> tuple[RichProgram, list[object]]:
    input_layer = next(
        layer for layer in symbolic_circuit.layers if isinstance(layer, InputLayer)
    )
    data_by_scope = tuple(
        torch.empty(
            (batch_size, input_layer.params["probs"].shape[0]),
            dtype=torch.float32,
        )
        for _ in range(symbolic_circuit.num_variables)
    )

    expression = _to_xe_recursive(
        symbolic_circuit,
        symbolic_circuit.layers[-1],
        data_by_scope,
    )
    if stability == "scaled":
        expression = xe.log(expression)
    return xe.extract_program(expression, stability=stability)


def preprocess_xe_program(
    program: Program,
    inputs: Sequence[object],
    *,
    optimize_stacking: bool,
) -> Program:
    return program


def input_shape(value: object) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return tuple(int(dimension) for dimension in shape)


def format_input_shapes(inputs, limit: int) -> str:
    shapes = [input_shape(value) for value in inputs]
    preview = ", ".join(str(shape) for shape in shapes[:limit])
    if len(shapes) > limit:
        preview = f"{preview}, ... (+{len(shapes) - limit} more)"
    return f"[{preview}]"


def print_program_summary(
    name: str, program: Program, inputs: Sequence[object], *, shape_preview: int
) -> None:
    op_counts = Counter(
        get_operator(instruction) for instruction in program.instructions
    )
    print(f"{name}:")
    print(f"  inputs:       {program.n_inputs}")
    print(f"  instructions: {len(program.instructions)}")
    print(f"  output_ssa:   {program.output_ssa}")
    stability = getattr(program, "stability", None)
    if stability is not None:
        print(f"  stability:    {stability}")
    print(f"  input_shapes: {format_input_shapes(inputs, shape_preview)}")
    print(f"  operators:    {dict(sorted(op_counts.items()))}")


def print_instructions(program: Program, limit: int) -> None:
    if limit <= 0:
        return
    print(f"\nfirst {min(limit, len(program.instructions))} instruction(s):")
    for instruction in program.instructions[:limit]:
        print(f"  {instruction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a Cirkit symbolic image circuit to an XE program."
    )
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sum-product-layer", choices=("cp", "tucker"), default="cp")
    parser.add_argument(
        "--semiring",
        choices=("scaled-max", "lse-sum"),
        default="scaled-max",
        help="Numerical-stability mode to request from XE.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip the preprocessing compatibility hook.",
    )
    parser.add_argument(
        "--no-optimize-stacking",
        action="store_true",
        help="Retained for compatibility; has no effect.",
    )
    parser.add_argument(
        "--dump-instructions",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N final XE instructions.",
    )
    parser.add_argument(
        "--shape-preview",
        type=int,
        default=12,
        metavar="N",
        help="Print at most N input shapes per program summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbolic_circuit = make_symbolic_circuit(
        width=args.width,
        height=args.height,
        num_units=args.units,
        sum_product_layer=args.sum_product_layer,
    )
    program, inputs = translate_cirkit_to_xe(
        symbolic_circuit,
        batch_size=args.batch_size,
        stability="logspace" if args.semiring == "lse-sum" else "scaled",
    )

    print(
        "symbolic circuit: "
        f"layers={len(symbolic_circuit.layers)}, "
        f"variables={symbolic_circuit.num_variables}, "
        f"units={args.units}, "
        f"sum_product_layer={args.sum_product_layer}"
    )
    print_program_summary(
        "direct XE program", program, inputs, shape_preview=args.shape_preview
    )

    final_program = program
    if not args.no_preprocess:
        try:
            preprocessed = preprocess_xe_program(
                program,
                inputs,
                optimize_stacking=not args.no_optimize_stacking,
            )
        except NameError as error:
            print()
            print(f"preprocessing skipped: {error}")
            preprocessed_inputs = inputs
        else:
            final_program = (
                preprocessed.program
                if hasattr(preprocessed, "program")
                else preprocessed
            )
            preprocessed_inputs = (
                preprocessed.inputs if hasattr(preprocessed, "inputs") else inputs
            )
            print()
            print_program_summary(
                "preprocessed XE program",
                final_program,
                preprocessed_inputs,
                shape_preview=args.shape_preview,
            )
            batched_input_ids = getattr(preprocessed_inputs, "batched_input_ids", None)
            index_input_ids = getattr(preprocessed_inputs, "index_input_ids", None)
            dynamic_transforms = getattr(preprocessed, "dynamic_input_transforms", None)
            if batched_input_ids is not None:
                print(f"  batched_input_ids: {batched_input_ids}")
            if index_input_ids is not None:
                print(f"  index_input_ids:   {index_input_ids}")
            if dynamic_transforms is not None:
                print(f"  dynamic_transforms: {len(dynamic_transforms)}")

    print_instructions(final_program, args.dump_instructions)


if __name__ == "__main__":
    main()
