from __future__ import annotations

import shlex
import shutil
import subprocess
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from extended_einsum.language.rich_operators import (
    OperatorConcat,
    OperatorEinsum,
    OperatorSelect,
    OperatorSlice,
    OperatorTake,
)
from extended_einsum.language.rich_program import RichProgram

_INPUT_KIND = "input"
_EXTRA_INPUT_KIND = "extra_input"
_OPERATION_KIND = "operation"
_EINSUM_KIND = "einsum"
_STACK_KIND = "stack"
_TAKE_KIND = "take"
_FUSED_EINSUM_KIND = "fused_einsum"
_OUTPUT_KIND = "output"
_TOL_WHITE = "#FFFFFF"
_TOL_YELLOW = "#DDAA33"
_TOL_RED = "#BB5566"
_TOL_BLUE = "#004488"
_TOL_BLACK = "#000000"
_GRAPHVIZ_RANKSEP = 0.58
_GRAPHVIZ_NODESEP = 0.35
_FALLBACK_RANKSEP = 1.0
_FALLBACK_NODESEP = 2.5
_ARROW_SHRINK = 12


def build_expression_dag(program: RichProgram) -> Any:
    """Build a NetworkX DAG for an extended einsum rich program.

    The graph uses SSA tensor ids as nodes. Edges point from an instruction
    result to the tensors it depends on, so original inputs are leaves.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("build_expression_dag requires networkx. Install the visualization extra: pip install 'extended-einsum[visualization]'.") from exc

    graph = nx.DiGraph()
    graph.add_nodes_from(range(program.output_ssa + 1))

    for input_id in range(program.n_inputs):
        graph.nodes[input_id]["kind"] = _INPUT_KIND
        graph.nodes[input_id]["operator"] = None

    for instruction_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + instruction_index
        graph.nodes[result_id]["kind"] = _OPERATION_KIND
        graph.nodes[result_id]["operator"] = instruction.operator.name
        if isinstance(instruction.operator, OperatorEinsum):
            graph.nodes[result_id]["format_string"] = instruction.operator.format_string
        for argument_position, argument_id in enumerate(instruction.argument_ssa_ids):
            graph.add_edge(result_id, argument_id, argument_position=argument_position)

    return graph


def plot_expression_dag(
    program: RichProgram,
    *,
    save_path: str | Path | None = None,
    ax: Any | None = None,
    input_labels: Sequence[str] | Mapping[int, str] | None = None,
    show: bool = False,
    show_edge_labels: bool = False,
    figsize: tuple[float, float] | None = None,
    collapse_fused_einsums: bool = True,
    show_tensor_ids: bool = True,
    vertical_spacing: float = 1.0,
    horizontal_spacing: float = 1.0,
    max_operator_label_width: int = 24,
) -> Any:
    """Plot an extended einsum expression DAG and optionally save the figure.

    Inputs are drawn as leaves. ``save_path`` may point to any Matplotlib-supported
    format; if omitted, the function only draws on ``ax`` or a new axes.
    """
    if vertical_spacing <= 0:
        raise ValueError("vertical_spacing must be positive.")
    if horizontal_spacing <= 0:
        raise ValueError("horizontal_spacing must be positive.")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError as exc:
        raise ImportError("plot_expression_dag requires matplotlib. Install the visualization extra: pip install 'extended-einsum[visualization]'.") from exc

    graph = _build_visual_expression_dag(
        program,
        input_labels=input_labels,
        collapse_fused_einsums=collapse_fused_einsums,
        show_tensor_ids=show_tensor_ids,
        max_operator_label_width=max_operator_label_width,
    )
    positions, graph_size = _layout_visual_graph(
        graph,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    if figsize is None:
        figsize = (max(graph_size[0], 2.0), max(graph_size[1], 2.0))

    positions = _scale_positions_to_figsize(positions, figsize)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#ffffff")
    ax.figure.patch.set_facecolor("#ffffff")

    for source, target in graph.edges:
        patch = FancyArrowPatch(
            positions[source],
            positions[target],
            arrowstyle="-|>",
            mutation_scale=11,
            color=_TOL_BLACK,
            linewidth=1.1,
            shrinkA=_ARROW_SHRINK,
            shrinkB=_ARROW_SHRINK,
            connectionstyle="arc3,rad=0.03",
            zorder=1,
        )
        ax.add_patch(patch)

    for node, data in graph.nodes(data=True):
        ax.text(
            *positions[node],
            data["label"],
            ha="center",
            va="center",
            fontsize=data["font_size"],
            color=data["text_color"],
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": data["fill_color"],
                "edgecolor": data["edge_color"],
                "linestyle": data["line_style"],
                "linewidth": data["line_width"],
            },
            zorder=2,
        )

    if show_edge_labels:
        for source, target, data in graph.edges(data=True):
            x = (positions[source][0] + positions[target][0]) / 2
            y = (positions[source][1] + positions[target][1]) / 2
            ax.text(x, y, str(data["argument_position"]), ha="center", va="center", fontsize=7, color=_TOL_BLACK)

    ax.set_axis_off()
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return ax


def _build_visual_expression_dag(
    program: RichProgram,
    *,
    input_labels: Sequence[str] | Mapping[int, str] | None,
    collapse_fused_einsums: bool,
    max_operator_label_width: int,
    show_tensor_ids: bool = True,
) -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("plot_expression_dag requires networkx. Install the visualization extra: pip install 'extended-einsum[visualization]'.") from exc

    source_graph = build_expression_dag(program)
    groups = _find_fused_einsum_groups(program) if collapse_fused_einsums else {}
    collapsed_node_by_source: dict[int, str] = {}
    skipped_nodes: set[int] = set()
    collapsed_labels: dict[str, str] = {}

    for einsum_node, group in groups.items():
        collapsed_node = f"fused_{einsum_node}"
        collapsed_labels[collapsed_node] = _operation_label(
            einsum_node,
            group["operator"],
            output_id=program.output_ssa,
            max_operator_label_width=max_operator_label_width,
            prefix="Fused",
            show_tensor_id=show_tensor_ids,
        )
        for node in group["stack_nodes"] | {einsum_node} | group["take_nodes"]:
            collapsed_node_by_source[node] = collapsed_node
            skipped_nodes.add(node)
        skipped_nodes.update(group["index_inputs"])

    graph = nx.DiGraph()
    for node in source_graph.nodes:
        if node in skipped_nodes:
            continue
        visual_node = collapsed_node_by_source.get(node, node)
        if visual_node in graph:
            continue
        if isinstance(visual_node, str):
            _add_visual_node(graph, visual_node, collapsed_labels[visual_node], _FUSED_EINSUM_KIND)
        else:
            kind = _node_kind(program, node)
            label = _node_label(
                program,
                node,
                input_labels=input_labels,
                show_tensor_id=show_tensor_ids,
                max_operator_label_width=max_operator_label_width,
            )
            _add_visual_node(graph, visual_node, label, kind)

    for collapsed_node, label in collapsed_labels.items():
        if collapsed_node not in graph:
            _add_visual_node(graph, collapsed_node, label, _FUSED_EINSUM_KIND)

    for source, target, data in source_graph.edges(data=True):
        visual_source = collapsed_node_by_source.get(source, source)
        visual_target = collapsed_node_by_source.get(target, target)
        if source in skipped_nodes and source not in collapsed_node_by_source:
            continue
        if target in skipped_nodes and target not in collapsed_node_by_source:
            continue
        if visual_source == visual_target:
            continue
        if visual_source not in graph or visual_target not in graph:
            continue
        graph.add_edge(
            visual_source,
            visual_target,
            argument_position=data["argument_position"],
        )
    return graph


def _add_visual_node(graph: Any, node: Any, label: str, kind: str) -> None:
    graph.add_node(node, label=label, kind=kind, **_node_style(kind))


def _node_style(kind: str) -> dict[str, Any]:
    if kind == _INPUT_KIND:
        return {
            "fill_color": _TOL_WHITE,
            "edge_color": _TOL_BLACK,
            "line_style": "dashed",
            "line_width": 1.35,
            "font_size": 8,
            "text_color": _TOL_BLACK,
        }
    if kind == _EXTRA_INPUT_KIND:
        return {
            "fill_color": _TOL_WHITE,
            "edge_color": _TOL_BLACK,
            "line_style": "dotted",
            "line_width": 0.9,
            "font_size": 7,
            "text_color": _TOL_BLACK,
        }
    if kind == _OUTPUT_KIND:
        return {
            "fill_color": _TOL_BLACK,
            "edge_color": _TOL_BLACK,
            "line_style": "solid",
            "line_width": 1.8,
            "font_size": 8,
            "text_color": _TOL_WHITE,
        }
    if kind == _FUSED_EINSUM_KIND:
        return {
            "fill_color": _TOL_YELLOW,
            "edge_color": _TOL_BLACK,
            "line_style": "dashdot",
            "line_width": 1.4,
            "font_size": 8,
            "text_color": _TOL_BLACK,
        }
    if kind in {_STACK_KIND, _TAKE_KIND}:
        return {
            "fill_color": _TOL_YELLOW,
            "edge_color": _TOL_BLACK,
            "line_style": "dashed",
            "line_width": 1.0,
            "font_size": 7,
            "text_color": _TOL_BLACK,
        }
    if kind == _EINSUM_KIND:
        return {
            "fill_color": _TOL_BLUE,
            "edge_color": _TOL_BLACK,
            "line_style": "solid",
            "line_width": 1.4,
            "font_size": 8,
            "text_color": _TOL_WHITE,
        }
    return {
        "fill_color": _TOL_RED,
        "edge_color": _TOL_BLACK,
        "line_style": "solid",
        "line_width": 1.2,
        "font_size": 8,
        "text_color": _TOL_BLACK,
    }


def _node_kind(program: RichProgram, node: int) -> str:
    if node >= program.n_inputs:
        if node == program.output_ssa:
            return _OUTPUT_KIND
        instruction = program.instructions[node - program.n_inputs]
        if instruction.operator.name == "stack":
            return _STACK_KIND
        if isinstance(instruction.operator, (OperatorTake, OperatorSelect, OperatorSlice, OperatorConcat)):
            return _TAKE_KIND
        if isinstance(instruction.operator, OperatorEinsum):
            return _EINSUM_KIND
        return _OPERATION_KIND

    take_index_inputs = {instruction.argument_ssa_ids[1] for instruction in program.instructions if isinstance(instruction.operator, OperatorTake) and len(instruction.argument_ssa_ids) == 2}
    if node in take_index_inputs:
        return _EXTRA_INPUT_KIND
    return _INPUT_KIND


def _node_label(
    program: RichProgram,
    node: int,
    *,
    input_labels: Sequence[str] | Mapping[int, str] | None,
    show_tensor_id: bool = True,
    max_operator_label_width: int,
) -> str:
    if node < program.n_inputs:
        if _node_kind(program, node) == _EXTRA_INPUT_KIND:
            return _label_with_optional_tensor_id(f"Idx {node}", node, show_tensor_id)
        return _label_with_optional_tensor_id(_input_label(input_labels, node), node, show_tensor_id)

    instruction = program.instructions[node - program.n_inputs]
    operator_label = instruction.operator.format_string if isinstance(instruction.operator, OperatorEinsum) else instruction.operator.name
    return _operation_label(
        node,
        operator_label,
        output_id=program.output_ssa,
        max_operator_label_width=max_operator_label_width,
        is_einsum=isinstance(instruction.operator, OperatorEinsum),
        show_tensor_id=show_tensor_id,
    )


def _operation_label(
    node: int,
    operator: str,
    *,
    output_id: int,
    max_operator_label_width: int,
    prefix: str | None = None,
    is_einsum: bool | None = None,
    show_tensor_id: bool = True,
) -> str:
    if prefix is None:
        if node == output_id:
            prefix = "Output"
        elif show_tensor_id:
            prefix = f"T{node}"
    operator_label = _wrap_operator(operator, max_operator_label_width) if (_is_einsum_operator(operator) if is_einsum is None else is_einsum) else operator
    if prefix is None:
        return operator_label
    return f"{prefix}\n{operator_label}"


def _label_with_optional_tensor_id(label: str, node: int, show_tensor_id: bool) -> str:
    if show_tensor_id:
        return f"{label}\nT{node}"
    return label


def _wrap_operator(operator: str, max_width: int) -> str:
    chunks = _operator_chunks(operator)
    lines: list[str] = []
    current = ""
    for chunk in chunks:
        candidate = f"{current}{chunk}"
        if current and len(candidate) > max_width:
            lines.append(current)
            current = chunk
        else:
            current = candidate
    if current:
        lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=max_width,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [line]
        )
    return "\n".join(wrapped)


def _operator_chunks(operator: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    index = 0
    while index < len(operator):
        if operator.startswith("->", index):
            if start < index:
                chunks.append(operator[start:index])
            chunks.append("->")
            index += 2
            start = index
        elif operator[index] == ",":
            chunks.append(operator[start : index + 1])
            index += 1
            start = index
        else:
            index += 1
    if start < len(operator):
        chunks.append(operator[start:])
    return chunks


def _is_einsum_operator(operator: str) -> bool:
    return "->" in operator


def _find_fused_einsum_groups(program: RichProgram) -> dict[int, dict[str, Any]]:
    children: dict[int, set[int]] = defaultdict(set)
    for child, child_parents in program.arguments_of_ssa_id.items():
        for parent in child_parents:
            children[parent].add(child)

    groups: dict[int, dict[str, Any]] = {}
    for node in range(program.n_inputs, program.output_ssa + 1):
        instruction = program.instructions[node - program.n_inputs]
        arguments = instruction.argument_ssa_ids
        if not isinstance(instruction.operator, OperatorEinsum):
            continue
        if not arguments:
            continue
        if not all(_operator_name_for_node(program, argument) == "stack" for argument in arguments):
            continue

        take_nodes = {child for child in children[node] if _operator_name_for_node(program, child) in {"take", "select"}}
        if len(take_nodes) < 2 or take_nodes != children[node]:
            continue

        stack_nodes = set(arguments)
        index_inputs: set[int] = set()
        for take_node in take_nodes:
            take_arguments = program.arguments_of_ssa_id[take_node]
            if len(take_arguments) == 2:
                index_inputs.add(take_arguments[1])
        groups[node] = {
            "operator": instruction.operator.format_string,
            "stack_nodes": stack_nodes,
            "take_nodes": take_nodes,
            "index_inputs": index_inputs,
        }
    return groups


def _operator_name_for_node(program: RichProgram, node: int) -> str | None:
    if node < program.n_inputs:
        return None
    return program.instructions[node - program.n_inputs].operator.name


def _layout_visual_graph(
    graph: Any,
    *,
    vertical_spacing: float = 1.0,
    horizontal_spacing: float = 1.0,
) -> tuple[dict[Any, tuple[float, float]], tuple[float, float]]:
    graphviz_layout = _graphviz_layout(
        graph,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    if graphviz_layout is not None:
        return graphviz_layout

    positions = _fallback_layered_positions(
        graph,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    width = max((x for x, _ in positions.values()), default=0.0) - min((x for x, _ in positions.values()), default=0.0)
    height = max((y for _, y in positions.values()), default=0.0) - min((y for _, y in positions.values()), default=0.0)
    return positions, (width + 2.0, height + 2.0)


def _graphviz_layout(
    graph: Any,
    *,
    vertical_spacing: float = 1.0,
    horizontal_spacing: float = 1.0,
) -> tuple[dict[Any, tuple[float, float]], tuple[float, float]] | None:
    if shutil.which("dot") is None:
        return None

    node_names = {node: f"n{index}" for index, node in enumerate(graph.nodes)}
    name_nodes = {name: node for node, name in node_names.items()}
    ranksep = _GRAPHVIZ_RANKSEP * vertical_spacing
    dot_lines = [
        "digraph G {",
        f'graph [rankdir=TB, ranksep="{ranksep:.3f}", nodesep="{_GRAPHVIZ_NODESEP:.3f}"];',
        'node [shape=box, margin="0.08,0.05", fontname="DejaVu Sans", fontsize=8];',
        'edge [color="#64748b"];',
    ]
    for node, name in node_names.items():
        label = _dot_escape(graph.nodes[node]["label"])
        dot_lines.append(f'{name} [label="{label}"];')
    for source, target in graph.edges:
        dot_lines.append(f"{node_names[source]} -> {node_names[target]};")
    dot_lines.append("}")

    try:
        result = subprocess.run(
            ["dot", "-Tplain"],
            input="\n".join(dot_lines),
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    positions: dict[Any, tuple[float, float]] = {}
    graph_size = (8.0, 5.0)
    for line in result.stdout.splitlines():
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] == "graph" and len(parts) >= 4:
            graph_size = (float(parts[2]), float(parts[3]))
        elif parts[0] == "node" and len(parts) >= 4:
            node = name_nodes.get(parts[1])
            if node is not None:
                positions[node] = (float(parts[2]), float(parts[3]))

    if len(positions) != len(graph.nodes):
        return None
    if horizontal_spacing != 1.0:
        positions = _scale_horizontal_positions(positions, horizontal_spacing)
        graph_size = (graph_size[0] * horizontal_spacing, graph_size[1])
    return positions, graph_size


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _scale_horizontal_positions(
    positions: dict[Any, tuple[float, float]],
    scale: float,
) -> dict[Any, tuple[float, float]]:
    if not positions:
        return {}
    min_x = min(x for x, _ in positions.values())
    max_x = max(x for x, _ in positions.values())
    center_x = (min_x + max_x) / 2
    return {node: (center_x + (x - center_x) * scale, y) for node, (x, y) in positions.items()}


def _fallback_layered_positions(
    graph: Any,
    *,
    vertical_spacing: float = 1.0,
    horizontal_spacing: float = 1.0,
) -> dict[Any, tuple[float, float]]:
    depths: dict[Any, int] = {}
    for node in reversed(list(graph.nodes)):
        parents = list(graph.successors(node))
        depths[node] = max((depths.get(parent, 0) for parent in parents), default=-1) + 1

    nodes_by_depth: dict[int, list[Any]] = defaultdict(list)
    for node, depth in depths.items():
        nodes_by_depth[depth].append(node)

    positions: dict[Any, tuple[float, float]] = {}
    for depth, nodes in nodes_by_depth.items():
        width = len(nodes) - 1
        for index, node in enumerate(nodes):
            positions[node] = (
                (index - width / 2) * _FALLBACK_NODESEP * horizontal_spacing,
                float(depth) * _FALLBACK_RANKSEP * vertical_spacing,
            )
    return positions


def _scale_positions_to_figsize(
    positions: dict[Any, tuple[float, float]],
    figsize: tuple[float, float],
) -> dict[Any, tuple[float, float]]:
    if not positions:
        return positions

    min_x = min(x for x, _ in positions.values())
    max_x = max(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    max_y = max(y for _, y in positions.values())
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    return {
        node: (
            ((x - min_x) / width) * figsize[0],
            ((y - min_y) / height) * figsize[1],
        )
        for node, (x, y) in positions.items()
    }


def _input_label(
    input_labels: Sequence[str] | Mapping[int, str] | None,
    input_id: int,
) -> str:
    if input_labels is None:
        return f"Input {input_id}"
    if isinstance(input_labels, Mapping):
        return input_labels.get(input_id, f"Input {input_id}")
    if input_id < len(input_labels):
        return input_labels[input_id]
    return f"Input {input_id}"
