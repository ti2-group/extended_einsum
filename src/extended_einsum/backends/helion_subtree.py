"""Subtree fusion for positive folded sum/product circuits.

Routing is specialized at compilation. Each task evaluates one subtree for a
sample tile in postorder; only log-valued separator nodes cross kernel launches.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion_categorical import CategoricalInput, _lookup_lines, _normalizer, _Preparation, _prepared
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import (
    OperatorConcat,
    OperatorEinsum,
    OperatorExp,
    OperatorLog,
    OperatorMultiply,
    OperatorSelect,
    OperatorSlice,
    OperatorSoftmax,
    OperatorStack,
    OperatorTake,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.utils import is_contraction_free_einsum, parse_format_string


@dataclass(frozen=True)
class _Input:
    source: int
    row: int | None
    boundary: bool = False


@dataclass
class _Node:
    operator: RichOperator | None
    arguments: tuple[int, ...]
    shape: tuple[int, ...]
    height: int
    parameter: bool
    input: _Input | None = None


class _Graph:
    def __init__(self, program: RichProgram, inputs: Sequence[torch.Tensor], axes: Sequence[int | None]):
        self.program, self.inputs, self.axes = program, inputs, axes
        self.nodes: list[_Node] = []
        self.cache: dict[tuple[int, int | None], int] = {}
        self.static: dict[int, tuple[torch.Tensor, int]] = {}
        self.routes: dict[int, list[int]] = {}

    def resolve(self, ssa: int, row: int | None = None) -> int:
        key = (ssa, row)
        if key in self.cache:
            return self.cache[key]
        p = self.program
        if ssa < p.n_inputs:
            shape = tuple(p.shapes[ssa][1:] if row is not None else p.shapes[ssa])
            if row is not None and not 0 <= row < p.shapes[ssa][0]:
                raise ValueError("Subtree routing index is out of bounds.")
            node = _Node(None, (), shape, 0, ssa in p.parameter_indices, _Input(ssa, row))
        else:
            instruction = p.instructions[ssa - p.n_inputs]
            op, args = instruction.operator, instruction.argument_ssa_ids
            if isinstance(op, OperatorSelect) and op.axis == 0:
                if row is not None:
                    raise NotImplementedError("Subtree scheduling requires a leading fold axis.")
                result = self.resolve(args[0], op.index)
                self.cache[key] = result
                return result
            if isinstance(op, OperatorTake) and op.axis == 0:
                if row is None or args[1] >= p.n_inputs:
                    raise NotImplementedError("Subtree routing requires input indices on the leading fold axis.")
                indices = self.inputs[args[1]]
                if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1:
                    raise NotImplementedError("Subtree routing requires one-dimensional integer inputs.")
                if args[1] not in self.routes:
                    try:
                        version = indices._version
                    except RuntimeError as exc:
                        raise ValueError("Static routing indices need version counters; create them outside torch.inference_mode().") from exc
                    self.static[args[1]] = (indices, version)
                    # Graph edges are copied once, never at execution.
                    self.routes[args[1]] = indices.detach().cpu().tolist()
                route = self.routes[args[1]]
                result = self.resolve(args[0], route[row])
                self.cache[key] = result
                return result
            if isinstance(op, OperatorSlice) and op.axis == 0 and row is not None:
                result = self.resolve(args[0], op.start + row)
                self.cache[key] = result
                return result
            if isinstance(op, OperatorStack) and op.axis == 0 and row is not None:
                result = self.resolve(args[row])
                self.cache[key] = result
                return result
            if isinstance(op, OperatorConcat) and op.axis == 0 and row is not None:
                offset = row
                for argument in args:
                    if offset < p.shapes[argument][0]:
                        result = self.resolve(argument, offset)
                        self.cache[key] = result
                        return result
                    offset -= p.shapes[argument][0]
                raise ValueError("Subtree concatenation index is out of bounds.")
            if isinstance(op, OperatorEinsum):
                labels, output = parse_format_string(op.format_string)
                if row is None:
                    local_args = tuple(self.resolve(a) for a in args)
                else:
                    fold = output[0]
                    local_args_list, local_labels = [], []
                    for a, label in zip(args, labels, strict=True):
                        if fold in label and label.index(fold) != 0:
                            raise NotImplementedError("Subtree einsums require a leading retained fold label.")
                        local_args_list.append(self.resolve(a, row if fold in label else None))
                        local_labels.append(label[1:] if fold in label else label)
                    local_args = tuple(local_args_list)
                    op = OperatorEinsum(",".join(local_labels) + "->" + output[1:])
            elif isinstance(op, (OperatorExp, OperatorLog, OperatorMultiply, OperatorSoftmax)):
                if isinstance(op, OperatorSoftmax) and row is not None:
                    dims = (op.axis,) if isinstance(op.axis, int) else op.axis
                    if 0 in dims:
                        raise NotImplementedError("Subtree scheduling cannot normalize across folded nodes.")
                    op = OperatorSoftmax(tuple(a - 1 for a in dims))
                local_args = tuple(self.resolve(a, row) for a in args)
            else:
                raise NotImplementedError(f"Subtree scheduling does not support {op.name!r} in this layout.")
            parameter = all(self.nodes[a].parameter for a in local_args)
            if isinstance(op, OperatorLog) and ssa != p.output_ssa and not parameter:
                raise NotImplementedError("Subtree scheduling requires positive internal circuit values and a final log.")
            height = max((self.nodes[a].height for a in local_args if not self.nodes[a].parameter), default=0)
            if isinstance(op, OperatorEinsum) and not parameter and not is_contraction_free_einsum(op.format_string):
                height += 1
            shape = op.propagate_shapes([self.nodes[a].shape for a in local_args])
            node = _Node(op, local_args, shape, height, parameter)
        result = len(self.nodes)
        self.nodes.append(node)
        self.cache[key] = result
        return result


@dataclass
class _Template:
    program: RichProgram
    inputs: list[_Input]
    axes: list[int | None]

    def key(self):
        return (
            tuple((a.source, a.boundary, axis) for a, axis in zip(self.inputs, self.axes, strict=True)),
            tuple((i.operator.name, i.operator.raw_extra_arguments, i.argument_ssa_ids) for i in self.program.instructions),
            tuple(self.program.shapes),
            self.program.parameter_indices,
        )


def _template(graph: _Graph, root: int, cuts: set[int], locations: dict[int, tuple[int, int]]) -> _Template:
    instructions, shapes, refs, axes, parameters = [], [], [], [], []
    input_shapes = []
    resolved = {}

    def visit(node_id):
        if node_id in resolved:
            return resolved[node_id]
        node = graph.nodes[node_id]
        boundary = node_id != root and node_id in cuts
        if boundary or node.input is not None:
            if boundary:
                source, row = locations[node_id]
                ref = _Input(source, row, True)
                axis = 0
            else:
                ref = node.input
                axis = graph.axes[ref.source]
                if axis is not None and ref.row is not None:
                    axis -= 1
            local_id = -1 - len(refs)
            refs.append(ref)
            axes.append(axis)
            input_shapes.append(node.shape)
            if node.parameter:
                parameters.append(local_id)
            if boundary:
                instructions.append(RichInstruction(OperatorExp(), (local_id,)))
                shapes.append(node.shape)
                local_id = len(instructions) - 1
        else:
            args = tuple(visit(a) for a in node.arguments)
            instructions.append(RichInstruction(node.operator, args))
            shapes.append(node.shape)
            local_id = len(instructions) - 1
        resolved[node_id] = local_id
        return local_id

    output = visit(root)
    if root in cuts:
        instructions.append(RichInstruction(OperatorLog(), (output,)))
        shapes.append(graph.nodes[root].shape)
    n = len(refs)

    def shift(i):
        return -1 - i if i < 0 else i + n

    local_program = RichProgram(
        [RichInstruction(i.operator, tuple(shift(a) for a in i.argument_ssa_ids)) for i in instructions],
        n,
        graph.program.stability_mode,
        ["dense"] * (n + len(instructions)),
        input_shapes + shapes,
        frozenset(shift(i) for i in parameters),
    )
    return _Template(local_program, refs, axes)


@dataclass
class _Stage:
    kernel: Any
    source: str
    inputs: list[_Input]
    routes: torch.Tensor
    roots: int

    def arguments(self, inputs, outputs):
        args = []
        for ref in self.inputs:
            value = outputs[ref.source] if ref.boundary else inputs[ref.source]
            args.extend(value.tensors if isinstance(value, CategoricalInput) else (value,))
        return args + [self.routes]


@dataclass
class HelionSubtreeProgram:
    stages: list[_Stage]
    input_specs: tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...]
    static_indices: dict[int, tuple[torch.Tensor, int]]
    subtree_depth: int
    last_uses: dict[int, int]
    categorical_specs: dict = field(default_factory=dict)
    preparations: list[_Preparation] = field(default_factory=list)

    @property
    def source(self) -> str:
        return "\n\n".join([p.code for p in self.preparations] + [stage.source for stage in self.stages])

    def _prepare_inputs(self, inputs, reference=False):
        result = list(inputs)
        for preparation in self.preparations:
            kernel = preparation.kernel
            if reference:
                import helion

                kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(kernel.fn)
            binding = result[preparation.source]
            result[preparation.source] = _prepared(binding, kernel(binding.tables))
        return result

    def _run(self, inputs: Sequence[torch.Tensor], reference: bool) -> torch.Tensor:
        from extended_einsum.backends.helion import HelionProgram

        HelionProgram(None, "", self.input_specs)._check(inputs)
        if {i for i, value in enumerate(inputs) if isinstance(value, CategoricalInput)} != set(self.categorical_specs):
            raise ValueError("Categorical input bindings must match the inputs used for compilation.")
        for i, (specs, logits, log_values) in self.categorical_specs.items():
            if inputs[i].logits != logits or inputs[i].log_values != log_values:
                raise ValueError("Categorical normalization and value-space modes must match compilation.")
            HelionProgram(None, "", specs)._check(inputs[i].tensors)
        if any(not a.is_contiguous() for a in inputs):
            raise ValueError("Subtree inputs must remain contiguous.")
        if not reference and inputs[0].device.type != "cuda":
            raise ValueError("Compiled Helion execution requires CUDA; use reference() on CPU.")
        for i, (tensor, version) in self.static_indices.items():
            if inputs[i] is not tensor or tensor._version != version:
                raise ValueError("Subtree routing inputs are static; recompile after replacing or modifying indices.")
        inputs = self._prepare_inputs(inputs, reference)
        outputs = {}
        for stage_id, stage in enumerate(self.stages):
            args = stage.arguments(inputs, outputs)
            if reference:
                import helion

                kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(stage.kernel.fn)
            else:
                kernel = stage.kernel
            outputs[stage_id] = kernel(*args)
            for source in {a.source for a in stage.inputs if a.boundary}:
                if self.last_uses[source] == stage_id:
                    del outputs[source]
        return outputs[len(self.stages) - 1][0]

    def __call__(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._run(inputs, False)

    def reference(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._run(inputs, True)


def compile_subtrees(
    program: RichProgram,
    inputs: Sequence[torch.Tensor | CategoricalInput],
    batch_axes: Sequence[int | None],
    *,
    block_size: int,
    subtree_depth: int,
    scale_interval: int,
    num_warps: int = 4,
    dot_precision: str = "ieee",
    reduction_block_size: int | None = None,
    grid_order: str = "roots",
    pipeline_stages: int = 1,
) -> HelionSubtreeProgram:
    from extended_einsum.backends.helion import _make_kernel, _MatrixLoad, _Recorder, _schedule, _Value

    if not isinstance(subtree_depth, int) or subtree_depth < 1:
        raise ValueError("subtree_depth must be positive.")
    if program.stability_mode == "unstable":
        raise NotImplementedError("Subtree frontiers require a stable evaluation mode.")
    if any(not a.is_contiguous() for a in inputs):
        raise NotImplementedError("Subtree scheduling currently requires contiguous inputs.")
    if any(a.numel() >= 2**31 for a in inputs):
        raise NotImplementedError("Subtree linear indexing currently requires fewer than 2**31 elements per input.")
    if not isinstance(program.instructions[-1].operator, OperatorLog) or len(program.shapes[-1]) != 2:
        raise NotImplementedError("Subtree scheduling currently requires a matrix-valued final log.")
    graph = _Graph(program, inputs, batch_axes)
    output = graph.resolve(program.output_ssa)
    cuts = {
        i
        for i, n in enumerate(graph.nodes)
        if isinstance(n.operator, OperatorEinsum) and not n.parameter and not is_contraction_free_einsum(n.operator.format_string) and n.height % subtree_depth == 0
    }
    cuts.difference_update(graph.nodes[output].arguments)
    levels = defaultdict(list)
    for node_id in cuts:
        levels[graph.nodes[node_id].height].append(node_id)
    levels[max(levels, default=0) + 1].append(output)
    locations, stages = {}, []
    for level in sorted(levels):
        groups = {}
        for root in sorted(levels[level]):
            template = _template(graph, root, cuts, locations)
            groups.setdefault(template.key(), []).append((root, template))
        for members in groups.values():
            template = members[0][1]
            p = template.program
            recorder = _Recorder(matmul=True, reduction_block_size=reduction_block_size, mask_padded_reductions=any(isinstance(a, CategoricalInput) for a in inputs), dot_precision=dot_precision)
            values, loads, views = [], [], []
            argument_names = []
            for i, (ref, shape, axis) in enumerate(zip(template.inputs, p.shapes[: p.n_inputs], template.axes, strict=True)):
                if axis not in (None, 0):
                    raise NotImplementedError("Subtree samples must follow the leading fold axis.")
                dtype = torch.float32 if ref.boundary else inputs[ref.source].dtype
                binding = inputs[ref.source] if not ref.boundary and isinstance(inputs[ref.source], CategoricalInput) else None
                argument_names.extend([f"arg{i}_{suffix}" for suffix in ("pixels", "tables", "variables", "banks")] if binding else [f"arg{i}"])
                row_offset = "0"
                if ref.row is not None:
                    if len(shape) != 2:
                        raise NotImplementedError("Subtree folded inputs must contain matrices.")
                    local_rows = "samples" if axis == 0 else f"hl.arange({shape[0]})"
                    column = [member.inputs[i].row for _, member in members]
                    step = column[1] - column[0] if len(column) > 1 else 0
                    # Regular region-graph routes need no dependent index
                    # load. Irregular routes retain the device lookup table.
                    affine = all(row == column[0] + j * step for j, row in enumerate(column))
                    row_code = f"({column[0]} + root * {step})" if affine else f"rows[root, {i}]"
                    if binding:
                        if axis != 0:
                            raise NotImplementedError("Categorical evidence requires the sample axis after the fold axis.")
                        # Vector row indices avoid scalar-dependent gather
                        # lowering and combine pixel selection with lookup.
                        categorical_views, categorical_loads = _lookup_lines(i, binding, row_code)
                        views.extend(categorical_views)
                        loads.extend(categorical_loads)
                        value = _Value(f"x{i}", torch.empty(shape, dtype=dtype, device="meta"), axis)
                        values.append(value)
                        recorder.remember_extents(value)
                        continue
                    views.append(f"    flat{i} = arg{i}.view([-1, arg{i}.size(-1)])\n")
                    row_offset = f"{row_code} * {shape[0]}"
                    index = [f"{row_code} * {shape[0]} + {local_rows}", ":"]
                    tensor_name = f"flat{i}"
                else:
                    if binding:
                        raise NotImplementedError("Categorical inputs require a leading fold axis selected by the circuit.")
                    index = ["samples" if j == axis else ":" for j in range(len(shape))]
                    tensor_name = f"arg{i}"
                mask = "" if axis is None else ", extra_mask=valid[:, None]"
                if axis is not None and len(shape) != 2:
                    raise NotImplementedError("Subtree data inputs must be matrices after removing the fold axis.")
                index_code = ", ".join("slice(None)" if a == ":" else a for a in index)
                load = f"hl.load({tensor_name}, [{index_code}]{mask})"
                loads.append(f"x{i} = {load}")
                matrix_load = _MatrixLoad(tensor_name, row_offset, tuple(shape)) if axis is None and len(shape) == 2 and dtype == torch.float32 else None
                value = _Value(f"x{i}", torch.empty(shape, dtype=dtype, device="meta"), axis, matrix_load)
                values.append(value)
                recorder.remember_extents(value)
            result = run_program(translate_to_backend_program(p, recorder, scale_interval=scale_interval), values)
            if result.batch_axis != 0 or len(result.shape) != 2:
                raise NotImplementedError("Subtree output must be a sample-by-unit matrix.")
            count, batch, units = len(members), *result.shape
            if count * batch * units >= 2**31:
                raise NotImplementedError("Subtree frontier indexing exceeds the 32-bit limit.")
            body = _schedule(loads, recorder.lines, result.name)
            body.append(f"hl.store(out, [root, samples, slice(None)], {result.name}, extra_mask=valid[:, None])")
            chunks = (batch + block_size - 1) // block_size
            grid = f"root, chunk in hl.grid([{count}, {chunks}])" if grid_order == "roots" else f"chunk, root in hl.grid([{chunks}, {count}])"
            source = (
                "def subtree_kernel(" + ", ".join(f"{name}: torch.Tensor" for name in argument_names) + ", rows: torch.Tensor):\n"
                f"    out = torch.empty([{count}, {batch}, {units}], dtype=torch.float32, device={argument_names[0]}.device)\n" + "".join(views) + f"    for {grid}:\n"
                f"        samples = chunk * {block_size} + hl.arange({block_size})\n"
                f"        valid = samples < {batch}\n" + "".join(f"        {line}\n" for statement in body for line in statement.splitlines()) + "    return out\n"
            )
            routes = torch.tensor([[r.row if r.row is not None else -1 for r in member.inputs] for _, member in members], dtype=torch.int32, device=inputs[0].device)
            kernel = _make_kernel(source, "subtree_kernel", num_warps=num_warps, dot_precision=dot_precision, pipeline_stages=pipeline_stages)
            stage_id = len(stages)
            stages.append(_Stage(kernel, source, template.inputs, routes, count))
            for row, (node_id, _) in enumerate(members):
                locations[node_id] = (stage_id, row)
    specs = tuple((tuple(a.shape), a.dtype, a.device) for a in inputs)
    last_uses = {a.source: i for i, stage in enumerate(stages) for a in stage.inputs if a.boundary}
    categorical_specs = {i: (tuple((tuple(t.shape), t.dtype, t.device) for t in a.tensors), a.logits, a.log_values) for i, a in enumerate(inputs) if isinstance(a, CategoricalInput)}
    preparations = [_normalizer(i, a) for i, a in enumerate(inputs) if isinstance(a, CategoricalInput) and a.logits]
    return HelionSubtreeProgram(stages, specs, graph.static, subtree_depth, last_uses, categorical_specs, preparations)
