"""Experimental, inference-only whole-program Helion compilation.

Each block evaluates the complete stable program for a batch tile. Intermediate
tensors are device-local values rather than global allocations between layers.
The schedule requires an independent batch axis; it deliberately rejects any
operation that mixes samples, including global stability shifts.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import torch

from extended_einsum.backend_translation.backend import BackendFunctions
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backend_translation.translate import translate_to_backend_program
from extended_einsum.backends.helion_categorical import CategoricalInput
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.utils import normalize_axis, parse_format_string

if TYPE_CHECKING:
    from extended_einsum.backends.helion_subtree import HelionSubtreeProgram


@dataclass(frozen=True)
class _MatrixLoad:
    """Reload a direct matrix operand a panel at a time, without a full tile."""

    tensor: str
    row_offset: str
    shape: tuple[int, int]

    def panel(self, axis: int, indices: str, *, tail: bool) -> str:
        rows, _ = self.shape
        row_indices = indices if axis == 0 else f"hl.arange({rows})"
        column_indices = indices if axis == 1 else "slice(None)"
        masks = []
        if tail:
            masks.append(f"({indices} < {self.shape[axis]})" + ("[:, None]" if axis == 0 else "[None, :]"))
        if axis == 1 and rows & (rows - 1):
            masks.append(f"(hl.arange({rows}) < {rows})[:, None]")
        mask = ", extra_mask=" + " & ".join(masks) if masks else ""
        return f"hl.load({self.tensor}, [{self.row_offset} + {row_indices}, {column_indices}]{mask})"


@dataclass(frozen=True)
class _Value:
    name: str
    meta: torch.Tensor
    batch_axis: int | None
    matrix_load: _MatrixLoad | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.meta.shape)


class _Recorder(BackendFunctions[_Value]):
    """Run the existing stability lowering on metadata and emit tile operations."""

    def __init__(self, *, matmul: bool = False, reduction_block_size: int | None = None, mask_padded_reductions: bool = False, dot_precision: str = "ieee") -> None:
        self.lines: list[str] = []
        self.extents: dict[int, str] = {}
        self.matmul = matmul
        self.reduction_block_size = reduction_block_size
        self.mask_padded_reductions = mask_padded_reductions
        self.dot_precision = dot_precision

    def _dot_operand(self, code: str) -> str:
        if self.dot_precision != "tf32":
            return code
        # Triton's native TF32 dot may truncate FP32 mantissas. Explicit
        # round-to-nearest avoids systematic bias in deep positive circuits.
        # Bitwise RNA rounding also works in Helion's CPU tile interpreter.
        # Preserve NaNs whose payload would otherwise round to infinity.
        rounded = f"torch.bitwise_and(torch.add(({code}).view(torch.int32), 4096), -8192).view(torch.float32)"
        return f"torch.where(torch.isnan({code}), {code}, {rounded})"

    def _matrix_product(self, x: _Value, weight: _Value, transpose: bool) -> _Value:
        meta_weight = weight.meta.T if transpose else weight.meta
        meta = torch.mm(x.meta, meta_weight)
        panel = self.reduction_block_size
        contracted = x.shape[1]
        if panel is None or contracted <= panel or weight.matrix_load is None:
            code = f"torch.transpose({weight.name}, 0, 1)" if transpose else weight.name
            return self.emit(f"torch.mm({self._dot_operand(x.name)}, {self._dot_operand(code)})", meta, 0)
        # A device loop bounds the live TF32x3 fragments. Unrolling these
        # products lets the compiler hoist all panels, recreating the large
        # register set that this schedule is intended to avoid.
        name = f"v{len(self.lines)}"
        indices, data, matrix, iteration = (f"{name}_{suffix}" for suffix in ("indices", "data", "weight", "k"))
        tail = contracted % panel != 0
        gather_indices = f"torch.clamp({indices}, max={contracted - 1})" if tail else indices
        gather = f"torch.gather({x.name}, 1, {gather_indices}.to(torch.int64)[None, :].expand([{x.name}.size(0), {panel}]))"
        if tail:
            gather = f"torch.where(({indices} < {contracted})[None, :], {gather}, 0.0)"
        load = weight.matrix_load.panel(1 if transpose else 0, indices, tail=tail)
        rhs = f"torch.transpose({matrix}, 0, 1)" if transpose else matrix
        self.lines.append(
            f"{name} = torch.zeros([{x.name}.size(0), {meta.shape[1]}], dtype=torch.float32, device={x.name}.device)\n"
            f"for {iteration} in range({(contracted + panel - 1) // panel}):\n"
            f"    {indices} = {iteration} * {panel} + hl.arange({panel})\n"
            f"    {data} = {gather}\n"
            f"    {matrix} = {load}\n"
            f"    {name} = torch.add({name}, torch.mm({self._dot_operand(data)}, {self._dot_operand(rhs)}))"
        )
        value = _Value(name, meta, 0)
        self.remember_extents(value)
        return value

    def remember_extents(self, value: _Value) -> None:
        for i, size in enumerate(value.shape):
            if i != value.batch_axis and size != 1:
                self.extents[size] = f"{value.name}.size({i})"

    def emit(self, code: str, meta: torch.Tensor, batch_axis: int | None) -> _Value:
        name = f"v{len(self.lines)}"
        self.lines.append(f"{name} = {code}")
        value = _Value(name, meta, batch_axis)
        self.remember_extents(value)
        return value

    def exp(self, array: _Value) -> _Value:
        return self.emit(f"torch.exp({array.name})", torch.exp(array.meta), array.batch_axis)

    def log(self, array: _Value) -> _Value:
        return self.emit(f"torch.log({array.name})", torch.log(array.meta), array.batch_axis)

    def stop_gradient(self, array: _Value) -> _Value:
        return array  # The entire kernel is inference-only.

    def _binary(self, left: _Value, right: _Value, op: str) -> _Value:
        meta = getattr(torch, op)(left.meta, right.meta)
        rank = meta.ndim
        axes = {a.batch_axis + rank - len(a.shape) for a in (left, right) if a.batch_axis is not None}
        if len(axes) > 1:
            raise NotImplementedError("Helion operands must align their independent batch axes.")
        axis = next(iter(axes), None)
        if axis is not None:
            for a in (left, right):
                padded_shape = (1,) * (rank - len(a.shape)) + a.shape
                if a.batch_axis is None and padded_shape[axis] != 1:
                    raise NotImplementedError("An untiled operand varies along the batch axis; specify its batch axis too.")
        return self.emit(f"torch.{op}({left.name}, {right.name})", meta, axis)

    def add(self, left: _Value, right: _Value) -> _Value:
        return self._binary(left, right, "add")

    def subtract(self, left: _Value, right: _Value) -> _Value:
        return self._binary(left, right, "sub")

    def multiply(self, left: _Value, right: _Value) -> _Value:
        return self._binary(left, right, "mul")

    def divide(self, left: _Value, right: _Value) -> _Value:
        return self._binary(left, right, "div")

    def maximum(self, left: _Value, right: _Value) -> _Value:
        return self._binary(left, right, "maximum")

    def _reduce(self, array: _Value, op: str, axis: int | tuple[int, ...] | None, keepdims: bool) -> _Value:
        axes = tuple(range(len(array.shape))) if axis is None else ((axis,) if isinstance(axis, int) else axis)
        if not axes:
            axes = tuple(range(len(array.shape)))  # Torch treats dim=() as all axes.
        axes = tuple(normalize_axis(a, len(array.shape)) for a in axes)
        if array.batch_axis in axes:
            raise NotImplementedError("Helion batch tiling cannot reduce across the batch axis (including global stability shifts).")
        batch_axis = array.batch_axis
        if batch_axis is not None and not keepdims:
            batch_axis -= sum(a < batch_axis for a in axes)
        if self.reduction_block_size is not None or self.mask_padded_reductions:
            # A panel accumulator introduces a new padded device extent.
            # Keep padding out of stability reductions even when Helion
            # unifies that extent with a full-axis input of the same size.
            neutral = {"sum": "0.0", "amax": "float('-inf')", "amin": "float('inf')"}[op]
            for dimension in axes:
                size = array.shape[dimension]
                if size & (size - 1):
                    index = ", ".join(":" if i == dimension else "None" for i in range(len(array.shape)))
                    array = self.emit(f"torch.where((hl.arange({size}) < {size})[{index}], {array.name}, {neutral})", array.meta, array.batch_axis)
        meta = getattr(torch, op)(array.meta, dim=axes, keepdim=keepdims)
        return self.emit(f"torch.{op}({array.name}, dim={axes!r}, keepdim={keepdims!r})", meta, batch_axis)

    def sum(self, array: _Value, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> _Value:
        return self._reduce(array, "sum", axis, keepdims)

    def max(self, array: _Value, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> _Value:
        return self._reduce(array, "amax", axis, keepdims)

    def min(self, array: _Value, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> _Value:
        return self._reduce(array, "amin", axis, keepdims)

    def softmax(self, array: _Value, axis: int | tuple[int, ...]) -> _Value:
        shifted = self.subtract(array, self.max(array, axis, keepdims=True))
        exp = self.exp(shifted)
        return self.divide(exp, self.sum(exp, axis, keepdims=True))

    def _shape_code(self, shape: tuple[int, ...], axis: int | None, batch_size_code: str) -> str:
        # Whole device axes can be padded to powers of two by Helion. Use
        # tensor extents, which are padded during compilation and logical in
        # reference mode, rather than baking logical lengths into a view.
        sizes = [batch_size_code if i == axis else self.extents.get(s, str(s)) for i, s in enumerate(shape)]
        return "[" + ", ".join(sizes) + "]"

    def reshape(self, array: _Value, shape: tuple[int, ...]) -> _Value:
        meta = array.meta.reshape(shape)
        shape = tuple(meta.shape)
        if tuple(s for s in array.shape if s != 1) != tuple(s for s in shape if s != 1):
            raise NotImplementedError("Helion stability reshapes currently only insert/remove singleton axes.")
        axis = None
        if array.batch_axis is not None:
            old = array.batch_axis
            candidates = [i for i, size in enumerate(shape) if size == array.shape[old] and prod(shape[:i]) == prod(array.shape[:old]) and prod(shape[i + 1 :]) == prod(array.shape[old + 1 :])]
            if not candidates:
                raise NotImplementedError("Helion reshape must preserve the batch axis without merging it with other axes.")
            axis = candidates[0]
        size = f"{array.name}.size({array.batch_axis})"
        return self.emit(f"torch.reshape({array.name}, {self._shape_code(shape, axis, size)})", meta, axis)

    def broadcast_to(self, array: _Value, shape: tuple[int, ...]) -> _Value:
        meta = torch.broadcast_to(array.meta, shape)
        axis = None if array.batch_axis is None else array.batch_axis + len(shape) - len(array.shape)
        size = f"{array.name}.size({array.batch_axis})"
        return self.emit(f"torch.broadcast_to({array.name}, {self._shape_code(shape, axis, size)})", meta, axis)

    def _join(self, arrays: Sequence[_Value], axis: int, op: str) -> _Value:
        rank = len(arrays[0].shape) + (op == "stack")
        axis = normalize_axis(axis, rank)
        if len({a.batch_axis for a in arrays}) != 1:
            raise NotImplementedError("Helion stack/concat operands must have the same batch axis.")
        batch_axis = arrays[0].batch_axis
        if op == "cat" and batch_axis == axis:
            raise NotImplementedError("Helion cannot concatenate along the batch axis.")
        if op == "stack" and batch_axis is not None:
            batch_axis += axis <= batch_axis
        meta = getattr(torch, op)([a.meta for a in arrays], dim=axis)
        return self.emit(f"torch.{op}([{', '.join(a.name for a in arrays)}], dim={axis})", meta, batch_axis)

    def stack(self, arrays: Sequence[_Value], axis: int) -> _Value:
        return self._join(arrays, axis, "stack")

    def concat(self, arrays: Sequence[_Value], axis: int) -> _Value:
        return self._join(arrays, axis, "cat")

    def take(self, array: _Value, indices: _Value, axis: int) -> _Value:
        axis = normalize_axis(axis, len(array.shape))
        if axis == array.batch_axis or indices.batch_axis is not None:
            raise NotImplementedError("Helion gathers must use shared indices along a non-batch axis.")
        meta = torch.index_select(array.meta, axis, indices.meta)
        view_shape = [f"{indices.name}.size(0)" if i == axis else "1" for i in range(len(array.shape))]
        expanded_shape = [f"{indices.name}.size(0)" if i == axis else f"{array.name}.size({i})" for i in range(len(array.shape))]
        index_code = f"{indices.name}.to(torch.int64).reshape([{', '.join(view_shape)}]).expand([{', '.join(expanded_shape)}])"
        return self.emit(f"torch.gather({array.name}, {axis}, {index_code})", meta, array.batch_axis)

    def select(self, array: _Value, axis: int, index: int) -> _Value:
        axis = normalize_axis(axis, len(array.shape))
        if axis == array.batch_axis:
            raise NotImplementedError("Helion cannot select along the batch axis.")
        if array.shape[axis] == 1 and index in (0, -1):
            return self.reshape(array, array.shape[:axis] + array.shape[axis + 1 :])
        # A singleton gather avoids Helion/Inductor's select ReinterpretView
        # lowering and also works for a non-leading folded axis.
        torch.select(array.meta, axis, index)  # Validate the logical bounds.
        index %= array.shape[axis]
        indices = self.emit(f"torch.full([1], {index}, dtype=torch.int64, device={array.name}.device)", torch.empty(1, dtype=torch.int64, device="meta"), None)
        selected = self.take(array, indices, axis)
        return self.reshape(selected, array.shape[:axis] + array.shape[axis + 1 :])

    def slice(self, array: _Value, start: int, stop: int, axis: int) -> _Value:
        axis = normalize_axis(axis, len(array.shape))
        if axis == array.batch_axis:
            raise NotImplementedError("Helion cannot slice along the batch axis.")
        indices = [":" for _ in array.shape]
        indices[axis] = f"{start}:{stop}"
        slices = [slice(None) for _ in array.shape]
        slices[axis] = slice(start, stop)
        return self.emit(f"{array.name}[{', '.join(indices)}]", array.meta[tuple(slices)], array.batch_axis)

    def einsum(self, format_string: str, *operands: _Value) -> _Value:
        inputs, output = parse_format_string(format_string)
        if any(len(set(labels)) != len(labels) for labels in inputs):
            raise NotImplementedError("Helion does not yet support repeated labels within an einsum operand.")
        batch_labels = {labels[a.batch_axis] for labels, a in zip(inputs, operands, strict=True) if a.batch_axis is not None}
        if len(batch_labels) > 1:
            raise NotImplementedError("Helion einsum operands must share one independent batch label.")
        batch_label = next(iter(batch_labels), None)
        if batch_label is not None:
            if batch_label not in output:
                raise NotImplementedError("Helion einsum cannot contract the batch label.")
            for labels, a in zip(inputs, operands, strict=True):
                if batch_label in labels and a.batch_axis is None and a.shape[labels.index(batch_label)] != 1:
                    raise NotImplementedError("An untiled einsum operand varies along the batch label; specify its batch axis too.")
        if self.matmul and len(output) == 2 and batch_label == output[0]:
            grouped = {}
            for labels, operand in zip(inputs, operands, strict=True):
                grouped[labels] = self.multiply(grouped[labels], operand) if labels in grouped else operand
            if len(grouped) == 2:
                data = [(labels, a) for labels, a in grouped.items() if a.batch_axis == 0 and len(labels) == 2 and labels[0] == batch_label]
                weights = [(labels, a) for labels, a in grouped.items() if a.batch_axis is None and len(labels) == 2]
                if len(data) == len(weights) == 1:
                    labels, x = data[0]
                    weight_labels, weight = weights[0]
                    if set(weight_labels) == {labels[1], output[1]} and labels[1] != output[1]:
                        transpose = weight_labels[0] == output[1]
                        return self._matrix_product(x, weight, transpose)
        # Explicit broadcast/reduce avoids the bmm decomposition's large dot
        # padding for tiny circuit units, and avoids specializing symbolic tile
        # lengths in opt_einsum's multi-operand path search. Keep contracted
        # axes trailing so reductions can consume their products immediately.
        contracted = "".join(dict.fromkeys(label for labels in inputs for label in labels if label not in output))
        union = output + contracted
        aligned = []
        for labels, operand in zip(inputs, operands, strict=True):
            order = [labels.index(label) for label in union if label in labels]
            if order != list(range(len(labels))):
                axis = None if operand.batch_axis is None else order.index(operand.batch_axis)
                operand = self.emit(f"torch.permute({operand.name}, {order!r})", operand.meta.permute(order), axis)
            for i, label in enumerate(union):
                if label not in labels:
                    axis = operand.batch_axis
                    if axis is not None:
                        axis += i <= axis
                    operand = self.emit(f"torch.unsqueeze({operand.name}, {i})", operand.meta.unsqueeze(i), axis)
            aligned.append(operand)
        result = aligned[0]
        for operand in aligned[1:]:
            result = self.multiply(result, operand)
        if contracted:
            result = self.sum(result, axis=tuple(range(len(output), len(union))))
        return result


@dataclass
class HelionProgram:
    """A shape-specialized kernel and its inspectable generated Python source.

    ``reference`` executes Helion's tile interpreter, including tail tiles, on
    CPU or CUDA. It validates lowering but does not validate compiled GPU code.
    """

    kernel: Any
    source: str
    input_specs: tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...]

    def _check(self, inputs: Sequence[torch.Tensor | CategoricalInput]) -> None:
        if len(inputs) != len(self.input_specs):
            raise ValueError(f"Expected {len(self.input_specs)} inputs, got {len(inputs)}.")
        for tensor, (shape, dtype, device) in zip(inputs, self.input_specs, strict=True):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device or tensor.layout != torch.strided:
                raise ValueError("Helion inputs must match the shapes, dtypes, and devices used for compilation.")
            if torch.is_grad_enabled() and tensor.requires_grad:
                raise ValueError("The experimental Helion compiler supports inference only; use torch.no_grad().")

    def __call__(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        self._check(inputs)
        if inputs[0].device.type != "cuda":
            raise ValueError("Compiled Helion execution requires CUDA; use reference() to check lowering on CPU.")
        return self.kernel(*inputs)

    def reference(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        import helion

        self._check(inputs)
        reference_kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(self.kernel.fn)
        return reference_kernel(*inputs)


def _schedule(loads: list[str], operations: list[str], output: str) -> list[str]:
    """Emit dependencies depth first, loading inputs just before their consumers.

    Reused DAG values are emitted once. This also drops unused stability
    bookkeeping instead of extending its live range to the end of the block.
    """
    assignments = {line.split(" = ", 1)[0]: line for line in loads + operations}
    dependencies = {}
    for name, line in assignments.items():
        statement = ast.parse(line)
        dependencies[name] = list(
            dict.fromkeys(node.id for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in assignments and node.id != name)
        )
    emitted = set()
    result = []
    pending = [(output, False)]
    while pending:
        name, ready = pending.pop()
        if name in emitted:
            continue
        if ready:
            result.append(assignments[name])
            emitted.add(name)
        else:
            pending.append((name, True))
            pending.extend((dependency, False) for dependency in reversed(dependencies[name]))
    return result


def _helion_module():
    try:
        import helion
    except ModuleNotFoundError as exc:
        if exc.name == "helion":
            raise ModuleNotFoundError("Install the experimental compiler with `pip install 'extended-einsum[helion]'`.") from exc
        raise
    return helion


def compile_helion(
    program: RichProgram,
    inputs: Sequence[torch.Tensor | CategoricalInput],
    *,
    batch_axes: Sequence[int | None],
    block_size: int | None = None,
    scale_interval: int = 3,
    schedule: str = "whole",
    subtree_depth: int = 2,
    num_warps: int = 4,
    dot_precision: str = "ieee",
    reduction_block_size: int | None = None,
    grid_order: str = "roots",
    pipeline_stages: int = 1,
) -> HelionProgram | HelionSubtreeProgram:
    """Compile stable inference using whole-program or bounded subtree fusion.

    ``batch_axes`` gives each input's independent sample axis, or ``None`` for
    shared parameters/indices. It is explicit because folded axis zero often
    denotes circuit nodes rather than samples. Operations are checked to ensure
    that blocks never need values from another sample tile. Unsupported graphs
    raise ``NotImplementedError``; there is no Torch fallback.

    ``schedule='whole'`` keeps all non-batch axes in each block (default batch
    tile 1). ``schedule='subtree'`` fuses up to ``subtree_depth`` contractions
    along a dependency path (default tile 32), with stable log-valued frontiers
    between kernels. Subtrees require contiguous positive folded circuits,
    leading fold axes, a final log matrix, and static routing indices. Recompile
    after changing routing; data and weights may change normally.

    Subtree matrix contractions use IEEE precision unless ``dot_precision`` is
    explicitly set to ``'tf32'`` or ``'tf32x3'``. Single-pass TF32 rounds operands
    to nearest (ties away from zero) before multiplication to avoid truncation
    bias. All data/frontiers and accumulation remain FP32. Backward is not generated.

    ``reduction_block_size`` streams direct matrix weights in panels instead
    of retaining their complete contraction axis. It is opt-in for subtree
    scheduling, can improve register occupancy for wide units, and changes
    floating-point summation order. Computed parameter matrices use the
    existing matrix product path. Sample/warp tiles still need tuning.

    ``grid_order='samples'`` schedules consecutive sample tiles of a subtree
    before moving to another root, improving weight locality for large graphs.
    The default ``'roots'`` traverses roots first. This is a subtree-only knob.

    ``pipeline_stages`` (1 to 4) overlaps panel loads with computation, using
    additional shared memory. It requires subtree scheduling with panels.

    A ``CategoricalInput`` can replace a folded evidence tensor when using
    ``schedule='subtree'`` and batch axis 1. It selects raw pixels and parameter
    banks inside each consuming task. Its logical shape must match the rich
    input. With ``logits=True``, table normalization is part of every call.
    """
    _helion_module()

    if len(inputs) != program.n_inputs or len(batch_axes) != program.n_inputs:
        raise ValueError("Input count and batch_axes must match program.n_inputs.")
    if block_size is None:
        block_size = 32 if schedule == "subtree" else 1
    if not isinstance(block_size, int) or block_size < 1 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two.")
    if not inputs:
        raise ValueError("Helion requires at least one input.")
    if schedule not in ("whole", "subtree"):
        raise ValueError("schedule must be 'whole' or 'subtree'.")
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16.")
    if dot_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("dot_precision must be 'ieee', 'tf32', or 'tf32x3'.")
    if grid_order not in ("roots", "samples"):
        raise ValueError("grid_order must be 'roots' or 'samples'.")
    if grid_order != "roots" and schedule != "subtree":
        raise ValueError("grid_order='samples' requires the subtree schedule.")
    if not isinstance(pipeline_stages, int) or isinstance(pipeline_stages, bool) or not 1 <= pipeline_stages <= 4:
        raise ValueError("pipeline_stages must be an integer from 1 to 4.")
    if pipeline_stages != 1 and (schedule != "subtree" or reduction_block_size is None):
        raise ValueError("pipeline_stages requires subtree scheduling with reduction_block_size.")
    if reduction_block_size is not None:
        if not isinstance(reduction_block_size, int) or reduction_block_size < 1 or reduction_block_size & (reduction_block_size - 1):
            raise ValueError("reduction_block_size must be a positive power of two.")
        if schedule != "subtree":
            raise ValueError("reduction_block_size requires the subtree schedule.")
    recorder = _Recorder()
    values = []
    batch_size = None
    batch_input = None
    loads = []
    normalized_axes = []
    for i, (tensor, axis) in enumerate(zip(inputs, batch_axes, strict=True)):
        if isinstance(tensor, CategoricalInput) and (schedule != "subtree" or axis != 1 or i in program.parameter_indices):
            raise ValueError("CategoricalInput requires a non-parameter subtree input with batch_axes=1.")
        if tuple(tensor.shape) != tuple(program.shapes[i]):
            raise ValueError(f"Input {i} shape does not match the rich program.")
        if tensor.device != inputs[0].device:
            raise ValueError("All Helion inputs must use the same device.")
        if tensor.layout != torch.strided or program.tensor_formats[i] != "dense":
            raise ValueError("Experimental Helion inputs must be dense strided tensors.")
        if tensor.dtype not in (torch.float32, torch.int32, torch.int64):
            raise ValueError("Experimental Helion inputs support float32 data and int32/int64 indices.")
        if any(s == 0 for s in tensor.shape):
            raise ValueError("Empty tensors are not supported by the experimental Helion compiler.")
        if axis is not None:
            axis = normalize_axis(axis, tensor.ndim)
            if batch_size is not None and tensor.shape[axis] != batch_size:
                raise ValueError("All tiled inputs must have the same batch size.")
            batch_size = tensor.shape[axis]
            batch_input = i
        normalized_axes.append(axis)
        index = ["tile" if a == axis else ":" for a in range(tensor.ndim)]
        loads.append(f"x{i} = arg{i}[{', '.join(index)}]" if index else f"x{i} = arg{i}[()]")
        value = _Value(f"x{i}", torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"), axis)
        values.append(value)
        recorder.remember_extents(value)
    if batch_size is None:
        raise ValueError("At least one input must have an independent batch axis.")
    if schedule == "subtree":
        from extended_einsum.backends.helion_subtree import compile_subtrees

        return compile_subtrees(
            program,
            inputs,
            normalized_axes,
            block_size=block_size,
            subtree_depth=subtree_depth,
            scale_interval=scale_interval,
            num_warps=num_warps,
            dot_precision=dot_precision,
            reduction_block_size=reduction_block_size,
            grid_order=grid_order,
            pipeline_stages=pipeline_stages,
        )
    backend_program = translate_to_backend_program(program, recorder, scale_interval=scale_interval)
    result = run_program(backend_program, values)
    if result.batch_axis is None:
        raise NotImplementedError("Helion output must retain the independent batch axis.")
    signature = ", ".join(f"arg{i}: torch.Tensor" for i in range(len(inputs)))
    output_index = ", ".join("tile" if i == result.batch_axis else ":" for i in range(len(result.shape)))
    body = _schedule(loads, recorder.lines, result.name) + [f"out[{output_index}] = {result.name}"]
    source = (
        f"def extended_einsum_kernel({signature}):\n"
        f"    out = torch.empty({list(result.shape)!r}, dtype={result.meta.dtype}, device=arg0.device)\n"
        f"    for tile in hl.tile(arg{batch_input}.size({normalized_axes[batch_input]}), block_size={block_size}):\n" + "".join(f"        {line}\n" for line in body) + "    return out\n"
    )
    kernel = _make_kernel(source, "extended_einsum_kernel", num_warps=num_warps)
    specs = tuple((tuple(a.shape), a.dtype, a.device) for a in inputs)
    return HelionProgram(kernel, source, specs)


def _make_kernel(source: str, name: str, *, num_warps: int = 4, dot_precision: str = "ieee", pipeline_stages: int = 1):
    import helion.language as hl

    helion = _helion_module()
    # Helion inspects Python source. Retain it in linecache so generated
    # functions have the same inspection behavior as file-backed functions.
    filename = f"<extended_einsum_helion_{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = {"torch": torch, "hl": hl}
    exec(compile(source, filename, "exec"), namespace)
    config = {"num_warps": num_warps}
    if pipeline_stages != 1:
        loops = sum(isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range" for n in ast.walk(ast.parse(source)))
        config.update(num_stages=pipeline_stages, range_num_stages=[pipeline_stages] * loops)
    return helion.kernel(config=helion.Config(**config), autotune_effort="none", dot_precision=dot_precision, fast_math=False, ref_mode=helion.RefMode.OFF)(namespace[name])
