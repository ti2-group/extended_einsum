from __future__ import annotations

from collections import Counter

import pyjuice as juice
import pyjuice.nodes.distributions as distributions
from cirkit.symbolic.layers import HadamardLayer, InputLayer, SumLayer
from pyjuice.nodes import InputNodes, ProdNodes, SumNodes

from demo.cirkit import make_symbolic_circuit


def circuit_nodes(root: object) -> tuple[object, ...]:
    nodes: list[object] = []
    seen: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        pending.extend(node.chs)
    return tuple(nodes)


def build_cp_t_quad_tree(
    *,
    height: int,
    width: int,
    units: int,
    categories: int,
):
    """Construct the exact PyJuice counterpart of Cirkit's CP-T quad tree."""

    def pixel(row: int, column: int):
        return juice.inputs(
            var=row * width + column,
            num_node_blocks=1,
            block_size=units,
            dist=distributions.Categorical(num_cats=categories),
        )

    def cp_t(left, right, output_units: int):
        return juice.summate(
            juice.multiply(left, right),
            num_node_blocks=1,
            block_size=output_units,
        )

    grid = [[pixel(row, column) for column in range(width)] for row in range(height)]
    while len(grid) > 1 or len(grid[0]) > 1:
        next_height = (len(grid) + 1) // 2
        next_width = (len(grid[0]) + 1) // 2
        is_root = next_height == 1 and next_width == 1
        next_grid: list[list[object]] = []
        for row in range(0, len(grid), 2):
            next_row = []
            for column in range(0, len(grid[0]), 2):
                regions = [
                    grid[patch_row][patch_column]
                    for patch_row, patch_column in (
                        (row, column),
                        (row, column + 1),
                        (row + 1, column),
                        (row + 1, column + 1),
                    )
                    if patch_row < len(grid) and patch_column < len(grid[0])
                ]
                output_units = 1 if is_root else units
                if len(regions) == 1:
                    merged = regions[0]
                elif len(regions) == 2:
                    merged = cp_t(regions[0], regions[1], output_units)
                elif len(regions) == 4:
                    top = cp_t(regions[0], regions[1], units)
                    bottom = cp_t(regions[2], regions[3], units)
                    merged = cp_t(top, bottom, output_units)
                else:
                    raise RuntimeError(f"unexpected patch size: {len(regions)}")
                next_row.append(merged)
            next_grid.append(next_row)
        grid = next_grid
    return grid[0][0]


def expected_parameters(*, variables: int, patches: int, units: int, categories: int) -> int:
    return variables * units * categories + (patches - 1) * units**2 + units


def validate_structure(
    root: object,
    *,
    height: int,
    width: int,
    units: int,
    categories: int,
) -> dict[str, int]:
    symbolic = make_symbolic_circuit(
        width=width,
        height=height,
        num_units=units,
        sum_product_layer="cp-t",
        region_graph="quad-tree-2",
    )
    nodes = circuit_nodes(root)
    counts = Counter(type(node) for node in nodes)
    for pyjuice_type, cirkit_type in (
        (InputNodes, InputLayer),
        (ProdNodes, HadamardLayer),
        (SumNodes, SumLayer),
    ):
        cirkit_count = sum(isinstance(layer, cirkit_type) for layer in symbolic.layers)
        if counts[pyjuice_type] != cirkit_count:
            raise ValueError(f"{pyjuice_type.__name__}/{cirkit_type.__name__} count mismatch: {counts[pyjuice_type]} != {cirkit_count}")
    pyjuice_product_scopes = Counter(tuple(sorted(node.scope.to_list())) for node in nodes if isinstance(node, ProdNodes))
    cirkit_product_scopes = Counter(tuple(sorted(symbolic.layer_scope(layer))) for layer in symbolic.layers if isinstance(layer, HadamardLayer))
    if pyjuice_product_scopes != cirkit_product_scopes:
        raise ValueError("PyJuice and Cirkit product scopes differ")
    pyjuice_sum_signatures = Counter((tuple(sorted(node.scope.to_list())), node.num_nodes) for node in nodes if isinstance(node, SumNodes))
    cirkit_sum_signatures = Counter((tuple(sorted(symbolic.layer_scope(layer))), layer.num_output_units) for layer in symbolic.layers if isinstance(layer, SumLayer))
    if pyjuice_sum_signatures != cirkit_sum_signatures:
        raise ValueError("PyJuice and Cirkit sum scopes or output sizes differ")

    root.init_parameters()
    logical_parameters = sum(node.get_params().numel() for node in nodes if isinstance(node, (InputNodes, SumNodes)))
    variables = height * width
    patches = counts[ProdNodes]
    expected = expected_parameters(
        variables=variables,
        patches=patches,
        units=units,
        categories=categories,
    )
    if patches != variables - 1 or logical_parameters != expected:
        raise ValueError(f"invalid CP-T tree: patches={patches}, parameters={logical_parameters}, expected={expected}")
    return {
        "parameters": logical_parameters,
        "input_layers": counts[InputNodes],
        "product_layers": counts[ProdNodes],
        "sum_layers": counts[SumNodes],
    }
