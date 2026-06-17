from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from extended_einsum.backend import (
    BackendFunctions,
    MultiFormatBackendFunctions,
    SingleFormatBackendFunctions,
    TBackendArray,
)
from extended_einsum.format import (
    DenseFormat,
    DenseLogspaceFormat,
    DenseScaledFormat,
    TensorFormat,
)
from extended_einsum.language import (
    BINARY_OPERATORS,
    EINSUM_OPERATOR,
    SELECT_OPERATOR,
    SLICE_OPERATOR,
    SOFTMAX_OPERATOR,
    STACK_OPERATOR,
    TAKE_OPERATOR,
    UNARY_OPERATORS,
    Instruction,
    Program,
    get_arguments,
    get_einsum_format_string,
    get_operator,
    get_select_axis,
    get_select_index,
    get_slice_axis,
    get_slice_start,
    get_slice_stop,
    get_softmax_axis,
    get_stack_axis,
    get_take_axis,
    make_einsum_instruction,
    map_instruction_arguments,
)
from extended_einsum.shapes import (
    Shape,
    infer_binary_shape,
    infer_einsum_shape,
    infer_select_shape,
    infer_slice_shape,
    infer_softmax_shape,
    infer_stack_shape,
    infer_take_shape,
    infer_unary_shape,
)
from extended_einsum.utils import parse_format_string

_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MAX_REPLAN_PATH_ATTEMPTS = 3

AxisVar = tuple[str, int, int]


def choose_single_format_backend_functions(
    argument_tensor_format: TensorFormat,
    backend_functions: BackendFunctions[TBackendArray],
) -> SingleFormatBackendFunctions:
    match argument_tensor_format:
        case DenseFormat():
            return backend_functions.unary_dense_only
        case DenseLogspaceFormat():
            return backend_functions.unary_logspace_only
        case DenseScaledFormat():
            return backend_functions.unary_scaled_only
        case _:
            raise NotImplementedError()


def choose_multi_format_backend_functions(
    tensor_format_1: TensorFormat,
    tensor_format_2: TensorFormat,
    backend_functions: BackendFunctions[TBackendArray],
) -> MultiFormatBackendFunctions:
    match (tensor_format_1, tensor_format_2):
        case (DenseFormat(), DenseFormat()):
            return backend_functions.binary_dense_only
        case (DenseLogspaceFormat(), DenseLogspaceFormat()):
            return backend_functions.binary_logspace_only
        case (DenseScaledFormat(), DenseScaledFormat()):
            return backend_functions.binary_scaled_only
        case (DenseFormat(), DenseScaledFormat()):
            return backend_functions.binary_dense_scaled
        case (DenseScaledFormat(), DenseFormat()):
            return backend_functions.binary_scaled_dense
        case (DenseLogspaceFormat(), DenseFormat()):
            return backend_functions.binary_logspace_dense
        case (DenseFormat(), DenseLogspaceFormat()):
            return backend_functions.binary_dense_logspace
        case _:
            raise NotImplementedError()


@dataclass(frozen=True)
class RichProgram(Program):
    instructions: list[Instruction]
    n_inputs: int

    shapes: list[Shape]
    tensor_formats: list[TensorFormat]
    parameter_indices: list[int]

    def __post_init__(self) -> None:
        n_ssa_ids = self.n_inputs + len(self.instructions)

        if len(self.shapes) != n_ssa_ids:
            raise ValueError(
                f"Number of shapes ({len(self.shapes)}) must match the expected number of SSA IDs ({self.n_inputs} inputs + {len(self.instructions)} instructions)."
            )
        if len(self.tensor_formats) != n_ssa_ids:
            raise ValueError(
                f"Number of tensor formats ({len(self.tensor_formats)}) must match the expected number of SSA IDs ({self.n_inputs} inputs + {len(self.instructions)} instructions)."
            )


class PreparedInputs(tuple[Any, ...]):
    index_input_ids: tuple[int, ...]
    batched_input_ids: tuple[int, ...]

    def __new__(
        cls,
        values: Sequence[Any] = (),
        *,
        index_input_ids: Sequence[int] = (),
        batched_input_ids: Sequence[int] = (),
    ) -> PreparedInputs:
        prepared = super().__new__(cls, tuple(values))
        prepared.index_input_ids = tuple(index_input_ids)
        prepared.batched_input_ids = tuple(batched_input_ids)
        return prepared

    @property
    def index_inputs(self) -> tuple[Any, ...]:
        return tuple(self[input_id] for input_id in self.index_input_ids)

    @property
    def take_index_inputs(self) -> tuple[Any, ...]:
        return self.index_inputs

    @property
    def batched_inputs(self) -> tuple[Any, ...]:
        return tuple(self[input_id] for input_id in self.batched_input_ids)


PreprocessedProgram = tuple[Program, PreparedInputs]


@dataclass(frozen=True)
class DynamicInputTransform:
    input_id: int
    prepared_input_id: int
    program: Program
    inputs: PreparedInputs


@dataclass(frozen=True)
class PreprocessedProgramWithDynamicInputTransforms:
    program: Program
    inputs: PreparedInputs
    dynamic_input_transforms: tuple[DynamicInputTransform, ...]


@dataclass(frozen=True)
class _ScaleState:
    axis: int


ScaleState = _ScaleState | None
ScaleAxisKey = tuple[int, int]


@dataclass(frozen=True)
class ParsedEinsum:
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class CanonicalEinsum:
    signature: tuple[Any, ...]
    operator: str
    inputs: tuple[str, ...]
    output: str
    permutation: tuple[int, ...]
    operand_shapes: list[Shape]
    operand_scale_states: list[ScaleState]
    output_shape: Shape
    output_scale_axis: int | None


@dataclass(frozen=True)
class _Candidate:
    op_index: int
    result_id: int
    depth: int
    instruction: Instruction
    arguments: tuple[int, ...]
    dependencies: frozenset[int]
    canonical: CanonicalEinsum


@dataclass(frozen=True)
class _ReplannedBlock:
    op_indices: frozenset[int]
    boundary_arguments: tuple[int, ...]
    instructions: tuple[Instruction, ...]


ScheduledEvent = int | tuple[_Candidate, ...]


@dataclass(frozen=True)
class _InputArgument:
    input_id: int


@dataclass(frozen=True)
class _IndexInputArgument:
    input_index: int


@dataclass(frozen=True)
class _ResultArgument:
    instruction_index: int


@dataclass(frozen=True)
class _PendingDynamicInputTransform:
    input_id: int
    program: Program
    inputs: PreparedInputs


_RawArgument = _InputArgument | _IndexInputArgument | _ResultArgument
_RawInstruction = tuple[Any, ...]


@dataclass(frozen=True)
class _EinsumOp:
    instruction: Instruction
    parsed: ParsedEinsum
    operator: str
    output_scale_axis: int | None


@dataclass(frozen=True)
class _TensorRef:
    argument: _RawArgument


@dataclass(frozen=True)
class _SliceRef:
    source: _RawArgument
    index: int


@dataclass(frozen=True)
class _SliceRangeRef:
    source: _RawArgument
    axis: int
    start: int
    stop: int


@dataclass(frozen=True)
class _TakeRef:
    source: _RawArgument
    axis: int
    index: int


_TensorValueRef = _TensorRef | _SliceRef | _SliceRangeRef | _TakeRef


class _UnionFind[T]:
    def __init__(self) -> None:
        self._parents: dict[T, T] = {}

    def add(self, item: T) -> None:
        self._parents.setdefault(item, item)

    def find(self, item: T) -> T:
        self.add(item)
        parent = self._parents[item]
        if parent != item:
            parent = self.find(parent)
            self._parents[item] = parent
        return parent

    def union(self, first: T, second: T) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parents[second_root] = first_root


def preprocess_einsum_path(
    format_string: str,
    inputs: Sequence[Any],
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    *,
    min_group_size: int = 2,
    fold_softmax_operations: bool = True,
    fold_same_depth_only: bool = False,
    reorder_inputs: bool = True,
    optimize_stacking: bool = True,
    dynamic_input_ids: Sequence[int] = (),
    return_dynamic_input_transforms: bool = False,
) -> RichProgram:
    """Lower an einsum path and return a rewritten program plus prepared inputs."""
    input_tuple = tuple(inputs)
    input_shapes = _input_shapes(input_tuple)
    parsed = _parse_einsum(format_string)

    if path is None:
        from sesum import sr

        path, _, _ = sr.compute_path(  # pyright: ignore[reportAssignmentType]
            format_string,
            *input_shapes,
            is_linear=False,
        )
    assert path is not None

    program = _program_from_ssa_path(format_string, path)

    return fold_independent_einsums(
        program,
        input_tuple,
        min_group_size=min_group_size,
        fold_softmax_operations=fold_softmax_operations,
        fold_same_depth_only=fold_same_depth_only,
        reorder_inputs=reorder_inputs,
        optimize_stacking=optimize_stacking,
        dynamic_input_ids=dynamic_input_ids,
        return_dynamic_input_transforms=return_dynamic_input_transforms,
    )


def merge_replan_einsums(
    program: Program,
    inputs: Sequence[Any],
    *,
    minimize: str = "size",
) -> PreprocessedProgram:
    """Merge einsum-only regions, replan them, and return a rewritten program."""
    input_tuple = tuple(inputs)
    if program.n_inputs != len(input_tuple):
        raise ValueError("program.n_inputs must match the number of provided inputs")
    input_shapes = _input_shapes(input_tuple)
    input_scale_states = _input_scale_states(input_tuple)
    optimized_program = _merge_replan_einsums_by_shape(
        program,
        input_shapes,
        input_scale_states,
        minimize=minimize,
    )
    return optimized_program, PreparedInputs(input_tuple)


def fold_independent_einsums(
    program: Program,
    inputs: Sequence[Any],
    *,
    min_group_size: int = 2,
    fold_softmax_operations: bool = True,
    fold_same_depth_only: bool = False,
    reorder_inputs: bool = True,
    optimize_stacking: bool = True,
    dynamic_input_ids: Sequence[int] = (),
    return_dynamic_input_transforms: bool = False,
) -> RichProgram:
    """Fold independent einsums and return a rewritten program plus prepared inputs."""
    input_tuple = tuple(inputs)
    if program.n_inputs != len(input_tuple):
        raise ValueError("program.n_inputs must match the number of provided inputs")
    input_shapes = _input_shapes(input_tuple)
    input_scale_states = _input_scale_states(input_tuple)
    folded_program, index_inputs = _fold_independent_einsums_by_shape(
        program,
        input_shapes,
        input_scale_states,
        min_group_size=min_group_size,
        fold_softmax_operations=fold_softmax_operations,
        fold_same_depth_only=fold_same_depth_only,
        reorder_inputs=reorder_inputs,
        optimize_stacking=optimize_stacking,
        input_index_values=_input_index_values(input_tuple),
    )
    return _prepare_program_inputs(
        folded_program,
        input_tuple,
        index_inputs,
        hoist_static_stacks=True,
        dynamic_input_ids=dynamic_input_ids,
        extract_dynamic_input_transforms=return_dynamic_input_transforms,
    )


def _fold_independent_einsums_by_shape(
    program: Program,
    input_shapes: list[Shape],
    input_scale_states: list[ScaleState],
    *,
    min_group_size: int,
    fold_softmax_operations: bool,
    fold_same_depth_only: bool,
    reorder_inputs: bool,
    optimize_stacking: bool,
    input_index_values: dict[int, int] | None = None,
) -> tuple[Program, tuple[np.ndarray, ...]]:
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")

    input_count = len(input_shapes)
    known_input_indices = {} if input_index_values is None else input_index_values

    try:
        _shapes, _dependencies, _first_consumers, candidates = _analyze_program(
            program,
            input_shapes,
            input_scale_states,
            fold_softmax_operations=fold_softmax_operations,
            reorder_inputs=reorder_inputs,
        )
    except ValueError:
        return program, ()
    try:
        events = _schedule_ready_events(
            program,
            input_count,
            candidates,
            min_group_size,
            fold_same_depth_only,
        )
    except ValueError:
        return program, ()

    if optimize_stacking:
        events = _order_scheduled_groups(events, program, input_count)
    groups = [event for event in events if not isinstance(event, int)]
    if not groups:
        return program, ()

    index_inputs: list[np.ndarray] = []
    raw_instructions: list[_RawInstruction] = []
    tensor_map: dict[int, _TensorValueRef] = {
        tensor_id: _TensorRef(_InputArgument(tensor_id))
        for tensor_id in range(input_count)
    }
    raw_shapes: dict[_RawArgument, Shape] = {
        _InputArgument(input_id): input_shapes[input_id]
        for input_id in range(input_count)
    }
    materialized_slices: dict[_SliceRef, _RawArgument] = {}
    materialized_slice_ranges: dict[_SliceRangeRef, _RawArgument] = {}
    materialized_takes: dict[_TakeRef, _RawArgument] = {}

    def append_index_input(indices: np.ndarray) -> _IndexInputArgument:
        argument = _IndexInputArgument(len(index_inputs))
        index_inputs.append(indices)
        raw_shapes[argument] = tuple(int(dimension) for dimension in indices.shape)
        return argument

    def append_instruction(instruction: _RawInstruction) -> _ResultArgument:
        result = _ResultArgument(len(raw_instructions))
        raw_instructions.append(instruction)
        raw_shapes[result] = _infer_instruction_shape(
            instruction,
            tuple(raw_shapes[argument] for argument in get_arguments(instruction)),
        )
        return result

    def materialize_ref(
        ref: _TensorValueRef, *, force_new_slice: bool = False
    ) -> _RawArgument:
        if isinstance(ref, _TensorRef):
            return ref.argument
        if not force_new_slice:
            if isinstance(ref, _SliceRef):
                cached = materialized_slices.get(ref)
                if cached is not None:
                    return cached
            else:
                if isinstance(ref, _SliceRangeRef):
                    cached = materialized_slice_ranges.get(ref)
                    if cached is not None:
                        return cached
                else:
                    cached = materialized_takes.get(ref)
                    if cached is not None:
                        return cached
        if isinstance(ref, _SliceRangeRef):
            result = append_instruction(
                (SLICE_OPERATOR, (ref.source,), ref.axis, ref.start, ref.stop)
            )
            if not force_new_slice:
                materialized_slice_ranges[ref] = result
            return result

        index = ref.index
        axis = 0 if isinstance(ref, _SliceRef) else ref.axis
        result = append_instruction((SELECT_OPERATOR, (ref.source,), axis, index))
        if not force_new_slice:
            if isinstance(ref, _SliceRef):
                materialized_slices[ref] = result
            else:
                materialized_takes[ref] = result
        return result

    def materialize_batch(refs: tuple[_TensorValueRef, ...]) -> _RawArgument:
        if not refs:
            raise ValueError("cannot materialize an empty batch")
        if not optimize_stacking:
            take_result = _materialize_unoptimized_take_batch(
                refs,
                append_instruction,
                append_index_input,
            )
            if take_result is not None:
                return take_result
            materialized_arguments = tuple(materialize_ref(ref) for ref in refs)
            return append_instruction((STACK_OPERATOR, materialized_arguments))

        first_ref = refs[0]
        if isinstance(first_ref, _SliceRef) and all(
            isinstance(ref, _SliceRef) and ref.source == first_ref.source
            for ref in refs
        ):
            indices = tuple(ref.index for ref in refs)
            contiguous = _contiguous_range(indices)
            if contiguous is not None:
                start, stop = contiguous
                if start == 0 and raw_shapes[first_ref.source][0] == stop:
                    return first_ref.source
                return append_instruction(
                    (SLICE_OPERATOR, (first_ref.source,), 0, start, stop)
                )
            indices = np.asarray(indices, dtype=np.int64)
            index_argument = append_index_input(indices)
            return append_instruction(
                (TAKE_OPERATOR, (first_ref.source, index_argument))
            )

        if isinstance(first_ref, _TakeRef) and all(
            isinstance(ref, _TakeRef)
            and ref.source == first_ref.source
            and ref.axis == first_ref.axis
            for ref in refs
        ):
            batched_take = _materialize_take_batch(
                refs,
                first_ref,
                raw_shapes[first_ref.source],
                raw_instructions,
                append_instruction,
                append_index_input,
            )
            if batched_take is not None:
                return batched_take

        materialized_arguments = tuple(materialize_ref(ref) for ref in refs)
        return append_instruction((STACK_OPERATOR, materialized_arguments))

    for event in events:
        if not isinstance(event, int):
            representative = event[0].canonical
            grouped_arguments = _group_refs_by_canonical_operand(event, tensor_map)
            stacked_ids: list[_RawArgument] = []
            for operand_arguments in grouped_arguments:
                stacked_ids.append(materialize_batch(operand_arguments))

            batched_instruction = _make_batched_instruction(
                representative,
                tuple(stacked_ids),
            )
            batched_result_id = append_instruction(batched_instruction)

            for take_index, candidate in enumerate(event):
                tensor_map[candidate.result_id] = _SliceRef(
                    batched_result_id, take_index
                )
            continue

        op_index = event
        instruction = program.instructions[op_index]
        arguments = get_arguments(instruction)
        if get_operator(instruction) == TAKE_OPERATOR:
            source, index = arguments
            index_value = known_input_indices.get(index)
            if index_value is not None:
                source_argument = materialize_ref(tensor_map[source])
                source_shape = raw_shapes[source_argument]
                axis = get_take_axis(instruction)
                tensor_map[input_count + op_index] = _TakeRef(
                    source_argument,
                    axis,
                    index_value,
                )
                continue
        if get_operator(instruction) == SELECT_OPERATOR:
            (source,) = arguments
            source_argument = materialize_ref(tensor_map[source])
            axis = get_select_axis(instruction)
            index = get_select_index(instruction)
            tensor_map[input_count + op_index] = _TakeRef(
                source_argument,
                axis,
                index,
            )
            continue
        if get_operator(instruction) == SLICE_OPERATOR:
            (source,) = arguments
            source_argument = materialize_ref(tensor_map[source])
            axis = get_slice_axis(instruction)
            start = get_slice_start(instruction)
            stop = get_slice_stop(instruction)
            tensor_map[input_count + op_index] = _SliceRangeRef(
                source_argument,
                axis,
                start,
                stop,
            )
            continue
        if get_operator(instruction) == STACK_OPERATOR and not optimize_stacking:
            stacked_argument = materialize_batch(
                tuple(tensor_map[argument] for argument in arguments)
            )
            tensor_map[input_count + op_index] = _TensorRef(stacked_argument)
            continue
        mapped_arguments = tuple(
            materialize_ref(tensor_map[argument]) for argument in arguments
        )
        argument_map = dict(zip(arguments, mapped_arguments, strict=True))
        result = append_instruction(
            map_instruction_arguments(instruction, argument_map.__getitem__)
        )
        tensor_map[input_count + op_index] = _TensorRef(result)

    output_ref = tensor_map[program.output_ssa]
    output_argument = materialize_ref(output_ref, force_new_slice=True)
    if not _is_last_result(output_argument, raw_instructions):
        output_argument = _copy_to_last_result(
            output_argument,
            append_instruction,
            append_index_input,
        )

    optimized = _resolve_raw_instructions(
        raw_instructions, input_count, len(index_inputs)
    )
    return Program(
        instructions=optimized,
        n_inputs=input_count + len(index_inputs),
    ), tuple(index_inputs)


def _merge_replan_einsums_by_shape(
    program: Program,
    input_shapes: list[Shape],
    input_scale_states: list[ScaleState],
    *,
    minimize: str,
) -> Program:
    try:
        shapes, scale_states = _infer_tensor_info(
            program,
            input_shapes,
            input_scale_states,
        )
    except ValueError:
        return program

    einsum_ops = _valid_einsum_ops(program, shapes, scale_states)
    if not einsum_ops:
        return program

    consumers = _result_consumers(program, program.n_inputs)

    accepted_blocks = _replanned_maximal_blocks(
        program,
        shapes,
        einsum_ops,
        consumers,
        minimize=minimize,
    )

    if not accepted_blocks:
        return program

    return _rewrite_replanned_blocks(program, accepted_blocks)


def _infer_tensor_info(
    program: Program,
    input_shapes: list[Shape],
    input_scale_states: list[ScaleState],
) -> tuple[tuple[Shape, ...], tuple[ScaleState, ...]]:
    shapes = input_shapes.copy()
    scale_states = list(input_scale_states)
    for instruction in program.instructions:
        arguments = get_arguments(instruction)
        argument_shapes = [shapes[argument] for argument in arguments]
        argument_scale_states = [scale_states[argument] for argument in arguments]
        output_shape = _infer_instruction_shape(instruction, argument_shapes)
        shapes.append(output_shape)
        scale_states.append(
            _infer_instruction_tensor_format(
                instruction,
                argument_scale_states,
                output_shape,
                argument_shapes,
            )
        )
    return tuple(shapes), tuple(scale_states)


def _valid_einsum_ops(
    program: Program,
    shapes: tuple[Shape, ...],
    scale_states: tuple[ScaleState, ...],
) -> dict[int, _EinsumOp]:
    valid: dict[int, _EinsumOp] = {}
    for op_index, instruction in enumerate(program.instructions):
        operator = get_operator(instruction)
        if operator != "einsum":
            continue
        try:
            parsed = _parse_einsum(get_einsum_format_string(instruction))
        except Exception:
            continue
        arguments = get_arguments(instruction)
        if is_scaled_einsum_instruction(instruction) and not parsed.output:
            continue
        if (
            is_normal_einsum_instruction(instruction)
            or is_logspace_einsum_instruction(instruction)
        ) and any(scale_states[argument] is not None for argument in arguments):
            continue
        operand_shapes: tuple[tuple[int, ...], ...] = tuple(
            shapes[argument] for argument in arguments
        )
        if _infer_einsum_output_shape(parsed, operand_shapes) is None:
            continue
        output_scale_axis = (
            normalize_axis(
                scaled_einsum_output_axis(instruction),
                len(parsed.output),
            )
            if is_scaled_einsum_instruction(instruction)
            else None
        )
        valid[op_index] = _EinsumOp(
            instruction=instruction,
            parsed=parsed,
            operator=operator,
            output_scale_axis=output_scale_axis,
        )
    return valid


def _replanned_maximal_blocks(
    program: Program,
    shapes: tuple[Shape, ...],
    einsum_ops: dict[int, _EinsumOp],
    consumers: dict[int, list[tuple[int, int]]],
    *,
    minimize: str,
) -> dict[int, _ReplannedBlock]:
    links: dict[int, set[int]] = {op_index: set() for op_index in einsum_ops}

    for parent_op_index in sorted(einsum_ops):
        parent_result_id = program.n_inputs + parent_op_index
        result_consumers = consumers.get(parent_result_id, [])
        if len(result_consumers) != 1:
            continue

        consumer_op_index, _argument_position = result_consumers[0]
        if consumer_op_index not in einsum_ops:
            continue
        if (
            einsum_ops[parent_op_index].operator
            != einsum_ops[consumer_op_index].operator
        ):
            continue

        links[parent_op_index].add(consumer_op_index)
        links[consumer_op_index].add(parent_op_index)

    accepted_by_sink: dict[int, _ReplannedBlock] = {}
    visited: set[int] = set()
    for seed_op_index in sorted(einsum_ops):
        if seed_op_index in visited:
            continue

        component: set[int] = set()
        pending = [seed_op_index]
        while pending:
            op_index = pending.pop()
            if op_index in component:
                continue
            component.add(op_index)
            pending.extend(sorted(links[op_index] - component, reverse=True))

        visited.update(component)
        if len(component) < 2:
            continue

        sinks = _external_output_ops(program, component, consumers)
        if len(sinks) != 1:
            continue

        replanned = _build_replanned_block(
            program,
            shapes,
            einsum_ops,
            component,
            sinks[0],
            minimize=minimize,
        )
        if replanned is not None:
            accepted_by_sink[sinks[0]] = replanned

    return accepted_by_sink


def _external_output_ops(
    program: Program,
    block_ops: set[int],
    consumers: dict[int, list[tuple[int, int]]],
) -> list[int]:
    output_op_indices: list[int] = []
    for op_index in sorted(block_ops):
        result_id = program.n_inputs + op_index
        has_external_consumer = any(
            consumer_op_index not in block_ops
            for consumer_op_index, _argument_position in consumers.get(result_id, [])
        )
        if result_id == program.output_ssa or has_external_consumer:
            output_op_indices.append(op_index)
    return output_op_indices


def _build_replanned_block(
    program: Program,
    shapes: tuple[Shape, ...],
    einsum_ops: dict[int, _EinsumOp],
    block_ops: set[int],
    sink_op_index: int,
    *,
    minimize: str = "flops",
) -> _ReplannedBlock | None:
    merged = _build_merged_einsum(program, shapes, einsum_ops, block_ops, sink_op_index)
    if merged is None:
        return None
    merged_expression, boundary_arguments, boundary_shapes = merged
    if len(boundary_arguments) < 2:
        return None

    original_size_log2 = _block_peak_size_log2(program, shapes, block_ops)
    source_einsum_depths = _program_einsum_depths(program)
    original_einsum_depth = source_einsum_depths[program.n_inputs + sink_op_index]

    for _attempt in range(_MAX_REPLAN_PATH_ATTEMPTS):
        try:
            from sesum import sr

            path: list[tuple[int, int]]
            path, _flops_log10, planned_size_log2 = sr.compute_path(  # pyright: ignore[reportAssignmentType]
                merged_expression,
                *boundary_shapes,
                minimize=minimize,
                is_linear=False,
                max_repeats=128,
                skops_alpha=10,
                seed=_attempt,
            )
        except Exception:
            return None

        if planned_size_log2 > original_size_log2 + 1e-12:
            return None

        try:
            annotated_path = to_annotated_ssa_path(
                merged_expression,
                path,
                is_ascii=False,
            )
            sink_op = einsum_ops[sink_op_index]
            instructions = tuple(
                _planned_instruction_for_step(
                    sink_op,
                    step_format,
                    first,
                    second,
                    is_final=step_index == len(annotated_path) - 1,
                )
                for step_index, (first, second, step_format) in enumerate(
                    annotated_path
                )
            )
        except ValueError:
            return None

        if not instructions:
            return None
        planned_einsum_depth = _planned_einsum_output_depth(
            instructions,
            boundary_arguments,
            source_einsum_depths,
        )
        if planned_einsum_depth <= original_einsum_depth or (
            _attempt == _MAX_REPLAN_PATH_ATTEMPTS - 1
        ):
            return _ReplannedBlock(
                op_indices=frozenset(block_ops),
                boundary_arguments=boundary_arguments,
                instructions=instructions,
            )

    return None


def _program_einsum_depths(program: Program) -> tuple[int, ...]:
    depths = [0] * program.n_inputs
    for instruction in program.instructions:
        arguments = get_arguments(instruction)
        argument_depth = max((depths[argument] for argument in arguments), default=0)
        depths.append(argument_depth + int(get_operator(instruction) == "einsum"))
    return tuple(depths)


def _planned_einsum_output_depth(
    instructions: tuple[Instruction, ...],
    boundary_arguments: tuple[int, ...],
    source_depths: tuple[int, ...],
) -> int:
    local_depths = [source_depths[argument] for argument in boundary_arguments]
    for instruction in instructions:
        arguments = get_arguments(instruction)
        argument_depth = max(
            (local_depths[argument] for argument in arguments),
            default=0,
        )
        local_depths.append(argument_depth + int(get_operator(instruction) == "einsum"))
    return local_depths[-1]


def _planned_instruction_for_step(
    sink_op: _EinsumOp,
    step_format: str,
    first: int,
    second: int,
    *,
    is_final: bool,
) -> Instruction:
    if sink_op.operator == EINSUM_OPERATOR:
        return make_einsum_instruction(step_format, first, second)
    if sink_op.operator == LSE_SUM_EINSUM_OPERATOR:
        return make_logspace_einsum_instruction(
            step_format,
            first,
            second,
        )
    if sink_op.operator in SCALED_EINSUM_OPERATORS:
        parsed = _parse_einsum(step_format)
        if parsed is None or not parsed.output:
            raise ValueError("scaled replanning requires non-scalar pairwise outputs")
        output_axis = (
            sink_op.output_scale_axis
            if is_final and sink_op.output_scale_axis is not None
            else len(parsed.output) - 1
        )
        return make_scaled_einsum_instruction(
            sink_op.operator,
            step_format,
            first,
            second,
            output_axis,
        )
    raise ValueError(f"unsupported replanned einsum operator: {sink_op.operator!r}")


def _build_merged_einsum(
    program: Program,
    shapes: tuple[Shape, ...],
    einsum_ops: dict[int, _EinsumOp],
    block_ops: set[int],
    sink_op_index: int,
) -> tuple[str, tuple[int, ...], tuple[Shape, ...]] | None:
    uf = _UnionFind()
    output_vars_by_tensor: dict[int, tuple[AxisVar, ...]] = {}
    boundary_position_by_occurrence: dict[tuple[int, int], int] = {}
    boundary_arguments: list[int] = []

    def boundary_vars(
        op_index: int, argument_position: int, tensor_id: int
    ) -> tuple[AxisVar, ...]:
        occurrence = (op_index, argument_position)
        boundary_position = boundary_position_by_occurrence.get(occurrence)
        if boundary_position is None:
            boundary_position = len(boundary_arguments)
            boundary_position_by_occurrence[occurrence] = boundary_position
            boundary_arguments.append(tensor_id)
        return tuple(
            ("boundary", boundary_position, axis)
            for axis in range(len(shapes[tensor_id]))
        )

    for op_index in sorted(block_ops):
        parsed = einsum_ops[op_index].parsed
        arguments = get_arguments(program.instructions[op_index])
        if len(parsed.inputs) != len(arguments):
            return None

        input_vars: list[tuple[AxisVar, ...]] = []
        for argument_position, argument in enumerate(arguments):
            if (
                argument >= program.n_inputs
                and argument - program.n_inputs in block_ops
            ):
                argument_vars = output_vars_by_tensor.get(argument)
                if argument_vars is None:
                    return None
            else:
                argument_vars = boundary_vars(op_index, argument_position, argument)
            input_vars.append(argument_vars)

        for subscript, argument_vars in zip(parsed.inputs, input_vars, strict=True):
            if len(subscript) != len(argument_vars):
                return None

        result_id = program.n_inputs + op_index
        output_vars = tuple(
            ("output", op_index, axis) for axis in range(len(parsed.output))
        )
        output_vars_by_tensor[result_id] = output_vars

        label_representatives: dict[str, AxisVar] = {}
        for subscript, argument_vars in zip(parsed.inputs, input_vars, strict=True):
            for label, axis_var in zip(subscript, argument_vars, strict=True):
                _union_label_occurrence(uf, label_representatives, label, axis_var)
        for label, axis_var in zip(parsed.output, output_vars, strict=True):
            _union_label_occurrence(uf, label_representatives, label, axis_var)

    sink_output_vars = output_vars_by_tensor.get(program.n_inputs + sink_op_index)
    if sink_output_vars is None:
        return None

    boundary_subscripts: list[str] = []
    label_by_root: dict[AxisVar, str] = {}
    for boundary_position, tensor_id in enumerate(boundary_arguments):
        subscript = _labels_for_axis_vars(
            tuple(
                ("boundary", boundary_position, axis)
                for axis in range(len(shapes[tensor_id]))
            ),
            uf,
            label_by_root,
        )
        if subscript is None:
            return None
        boundary_subscripts.append(subscript)

    output_subscript = _labels_for_axis_vars(sink_output_vars, uf, label_by_root)
    if output_subscript is None or len(set(output_subscript)) != len(output_subscript):
        return None

    expression = f"{','.join(boundary_subscripts)}->{output_subscript}"
    boundary_shapes = tuple(shapes[argument] for argument in boundary_arguments)
    parsed = _parse_einsum(expression)
    if parsed is None:
        return None
    if (
        _infer_einsum_output_shape(parsed, boundary_shapes)
        != shapes[program.n_inputs + sink_op_index]
    ):
        return None

    return expression, tuple(boundary_arguments), boundary_shapes


def _union_label_occurrence(
    uf: _UnionFind,
    label_representatives: dict[str, AxisVar],
    label: str,
    axis_var: AxisVar,
) -> None:
    uf.add(axis_var)
    representative = label_representatives.setdefault(label, axis_var)
    uf.union(representative, axis_var)


def _labels_for_axis_vars(
    axis_vars: tuple[AxisVar, ...],
    uf: _UnionFind,
    label_by_root: dict[AxisVar, str],
) -> str | None:
    labels: list[str] = []
    for axis_var in axis_vars:
        root = uf.find(axis_var)
        label = label_by_root.get(root)
        if label is None:
            if len(label_by_root) == len(_LABELS):
                return None
            label = _LABELS[len(label_by_root)]
            label_by_root[root] = label
        labels.append(label)
    return "".join(labels)


def _block_peak_size_log2(
    program: Program,
    shapes: tuple[Shape, ...],
    block_ops: set[int],
) -> float:
    peak_size = 0
    for op_index in block_ops:
        shape = shapes[program.n_inputs + op_index]
        peak_size = max(peak_size, int(np.prod(shape, dtype=np.int64)))
    if peak_size <= 0:
        return float("-inf")
    return float(np.log2(peak_size))


def _rewrite_replanned_blocks(
    program: Program,
    accepted_by_sink: dict[int, _ReplannedBlock],
) -> Program:
    block_ops = {
        op_index for block in accepted_by_sink.values() for op_index in block.op_indices
    }
    tensor_map: dict[int, int] = {
        input_id: input_id for input_id in range(program.n_inputs)
    }
    instructions: list[Instruction] = []

    def append_instruction(instruction: Instruction) -> int:
        result_id = program.n_inputs + len(instructions)
        instructions.append(instruction)
        return result_id

    for op_index, instruction in enumerate(program.instructions):
        arguments = get_arguments(instruction)
        block = accepted_by_sink.get(op_index)
        result_id = program.n_inputs + op_index

        if block is not None:
            local_tensor_map: dict[int, int] = {
                local_id: tensor_map[argument]
                for local_id, argument in enumerate(block.boundary_arguments)
            }
            for step_index, planned_instruction in enumerate(block.instructions):
                planned_arguments = get_arguments(planned_instruction)
                mapped_arguments = tuple(
                    local_tensor_map[argument] for argument in planned_arguments
                )
                argument_map = dict(
                    zip(planned_arguments, mapped_arguments, strict=True)
                )
                planned_result_id = append_instruction(
                    map_instruction_arguments(
                        planned_instruction, argument_map.__getitem__
                    )
                )
                local_tensor_map[len(block.boundary_arguments) + step_index] = (
                    planned_result_id
                )
            tensor_map[result_id] = planned_result_id
            continue

        if op_index in block_ops:
            continue

        mapped_arguments = tuple(tensor_map[argument] for argument in arguments)
        argument_map = dict(zip(arguments, mapped_arguments, strict=True))
        tensor_map[result_id] = append_instruction(
            map_instruction_arguments(instruction, argument_map.__getitem__)
        )

    return Program(instructions=instructions, n_inputs=program.n_inputs)


def _input_shapes(inputs: tuple[Any, ...]) -> list[Shape]:
    shapes: list[Shape] = []
    for input_index, value in enumerate(inputs):
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ValueError(f"input {input_index} does not expose a shape")
        shapes.append(tuple(int(dimension) for dimension in shape))
    return shapes


def _input_scale_states(inputs: tuple[Any, ...]) -> list[ScaleState]:
    return [
        _ScaleState(value.scale_axis) if isinstance(value, ScaledTensor) else None
        for value in inputs
    ]


def _input_index_values(inputs: tuple[Any, ...]) -> dict[int, int]:
    index_values: dict[int, int] = {}
    for input_id, value in enumerate(inputs):
        if isinstance(value, ScaledTensor):
            continue
        shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
        if shape:
            continue
        try:
            index_values[input_id] = int(value.item())
        except AttributeError:
            try:
                index_values[input_id] = int(value)
            except (TypeError, ValueError):
                continue
        except (TypeError, ValueError):
            continue
    return index_values


def _prepare_program_inputs(
    program: Program,
    original_inputs: tuple[Any, ...],
    index_inputs: tuple[np.ndarray, ...],
    *,
    hoist_static_stacks: bool,
    dynamic_input_ids: Sequence[int],
    extract_dynamic_input_transforms: bool,
) -> RichProgram:
    if program.n_inputs != len(original_inputs) + len(index_inputs):
        raise ValueError("program input count does not match prepared inputs")

    dynamic_input_id_set = _validated_dynamic_input_ids(
        dynamic_input_ids,
        len(original_inputs),
    )
    backend_reference = original_inputs[0] if original_inputs else None
    generated_index_input_ids = tuple(
        range(len(original_inputs), len(original_inputs) + len(index_inputs))
    )
    base_inputs = [
        *original_inputs,
        *(
            _index_array_like(index_input, backend_reference)
            for index_input in index_inputs
        ),
    ]
    hoisted_stack_values: list[Any] = []
    hoisted_stack_input_by_result: dict[int, int] = {}

    for instruction_index, instruction in enumerate(program.instructions):
        operator = get_operator(instruction)
        arguments = get_arguments(instruction)
        result_id = program.n_inputs + instruction_index

        if (
            hoist_static_stacks
            and operator == STACK_OPERATOR
            and all(
                argument < len(original_inputs) and argument not in dynamic_input_id_set
                for argument in arguments
            )
        ):
            hoisted_stack_input_by_result[result_id] = len(base_inputs) + len(
                hoisted_stack_values
            )
            hoisted_stack_values.append(
                _stack_input_values(
                    tuple(base_inputs[argument] for argument in arguments)
                )
            )

    candidate_inputs = tuple([*base_inputs, *hoisted_stack_values])
    candidate_input_count = len(candidate_inputs)
    tensor_map: dict[int, int] = {
        input_id: input_id for input_id in range(program.n_inputs)
    }
    tensor_map.update(hoisted_stack_input_by_result)

    instructions: list[Instruction] = []
    for instruction_index, instruction in enumerate(program.instructions):
        arguments = get_arguments(instruction)
        result_id = program.n_inputs + instruction_index
        if result_id in hoisted_stack_input_by_result:
            continue

        mapped_arguments = tuple(tensor_map[argument] for argument in arguments)
        tensor_map[result_id] = candidate_input_count + len(instructions)
        argument_map = dict(zip(arguments, mapped_arguments, strict=True))
        instructions.append(
            map_instruction_arguments(instruction, argument_map.__getitem__)
        )

    output_id = tensor_map[program.output_ssa]
    prepared_program = Program(
        instructions=instructions,
        n_inputs=candidate_input_count,
    )

    pending_transforms: tuple[_PendingDynamicInputTransform, ...] = ()
    if extract_dynamic_input_transforms:
        prepared_program, candidate_inputs, output_id, pending_transforms = (
            _extract_dynamic_input_transforms(
                prepared_program,
                candidate_inputs,
                output_id,
                dynamic_input_id_set,
            )
        )

    compacted_program, compacted_values, input_id_map = (
        _compact_program_inputs_with_input_map(
            prepared_program,
            candidate_inputs,
            output_id,
        )
    )
    compacted_index_input_ids = tuple(
        input_id_map[input_id]
        for input_id in generated_index_input_ids
        if input_id in input_id_map
    )
    compacted_batched_input_ids = tuple(
        input_id_map[input_id]
        for input_id in hoisted_stack_input_by_result.values()
        if input_id in input_id_map
    )
    compacted_program, compacted_inputs, final_input_id_map = (
        _append_index_inputs_to_end(
            compacted_program,
            compacted_values,
            compacted_index_input_ids,
            compacted_batched_input_ids,
        )
    )
    input_id_map = {
        input_id: final_input_id_map[compacted_input_id]
        for input_id, compacted_input_id in input_id_map.items()
    }

    if not extract_dynamic_input_transforms:
        return compacted_program, compacted_inputs

    transforms = tuple(
        DynamicInputTransform(
            input_id=transform.input_id,
            prepared_input_id=input_id_map[transform.input_id],
            program=transform.program,
            inputs=transform.inputs,
        )
        for transform in pending_transforms
        if transform.input_id in input_id_map
    )
    return RichProgram(
        instructions=compacted_program.instructions,
        n_inputs=compacted_program.n_inputs,
    )


def _validated_dynamic_input_ids(
    dynamic_input_ids: Sequence[int],
    input_count: int,
) -> frozenset[int]:
    dynamic_input_id_set = frozenset(dynamic_input_ids)
    if len(dynamic_input_id_set) != len(dynamic_input_ids):
        raise ValueError("dynamic_input_ids must be unique")
    for input_id in dynamic_input_id_set:
        if input_id < 0 or input_id >= input_count:
            raise ValueError(
                f"dynamic_input_id {input_id} is out of range for {input_count} inputs"
            )
    return dynamic_input_id_set


def _extract_dynamic_input_transforms(
    program: Program,
    inputs: tuple[Any, ...],
    output_id: int,
    dynamic_input_ids: frozenset[int],
) -> tuple[Program, tuple[Any, ...], int, tuple[_PendingDynamicInputTransform, ...]]:
    if not dynamic_input_ids or not program.instructions:
        return program, inputs, output_id, ()

    live_instruction_indices = _live_instruction_indices(program, output_id)
    transformable_by_dynamic = _transformable_nodes_by_dynamic(
        program,
        dynamic_input_ids,
        live_instruction_indices,
    )
    consumers = _result_consumers(program, program.n_inputs)
    extracted_nodes: set[int] = set()
    replacement_by_result: dict[int, int] = {}
    pending_transforms: list[_PendingDynamicInputTransform] = []
    rewritten_inputs = list(inputs)
    extracted_input_ids: set[int] = set()

    for input_id in sorted(dynamic_input_ids):
        transformable_nodes = transformable_by_dynamic.get(input_id, frozenset())
        if not transformable_nodes:
            continue

        boundary_nodes = tuple(
            sorted(
                result_id
                for result_id in transformable_nodes
                if result_id == output_id
                or any(
                    consumer_op_index in live_instruction_indices
                    and program.n_inputs + consumer_op_index not in transformable_nodes
                    for consumer_op_index, _argument_position in consumers.get(
                        result_id, []
                    )
                )
            )
        )
        if len(boundary_nodes) != 1:
            continue

        transform_output = boundary_nodes[0]
        transform_node_ids = _transform_subgraph_nodes(
            program,
            transform_output,
            input_id,
            transformable_nodes,
        )
        if not transform_node_ids:
            continue
        if _dynamic_input_has_external_raw_consumers(
            program,
            input_id,
            transform_node_ids,
            live_instruction_indices,
        ):
            continue
        if not _transform_subgraph_is_closed(
            program,
            transform_output,
            transform_node_ids,
            consumers,
            live_instruction_indices,
        ):
            continue

        transform = _build_dynamic_input_transform(
            program,
            inputs,
            input_id,
            transform_output,
            transform_node_ids,
        )
        if transform is None:
            continue

        transformed_value = _execute_take_stack_program(
            transform.program,
            transform.inputs,
        )
        rewritten_inputs[input_id] = transformed_value
        extracted_nodes.update(transform_node_ids)
        replacement_by_result[transform_output] = input_id
        pending_transforms.append(transform)
        extracted_input_ids.add(input_id)

    if pending_transforms:
        program, output_id = _remove_extracted_transform_nodes(
            program,
            output_id,
            extracted_nodes,
            replacement_by_result,
        )
        inputs = tuple(rewritten_inputs)

    program, inputs, reorder_transforms = _rewrite_dynamic_input_takes_as_slices(
        program,
        inputs,
        output_id,
        dynamic_input_ids - extracted_input_ids,
        dynamic_input_ids,
    )
    pending_transforms.extend(reorder_transforms)

    if not pending_transforms:
        return program, inputs, output_id, ()

    return program, inputs, output_id, tuple(pending_transforms)


@dataclass(frozen=True)
class _DynamicTakeRewrite:
    axis: int
    start: int
    stop: int


def _rewrite_dynamic_input_takes_as_slices(
    program: Program,
    inputs: tuple[Any, ...],
    output_id: int,
    dynamic_input_ids: frozenset[int],
    all_dynamic_input_ids: frozenset[int],
) -> tuple[Program, tuple[Any, ...], tuple[_PendingDynamicInputTransform, ...]]:
    if not dynamic_input_ids:
        return program, inputs, ()

    live_instruction_indices = _live_instruction_indices(program, output_id)
    rewritten_inputs = list(inputs)
    pending_transforms: list[_PendingDynamicInputTransform] = []
    rewrites: dict[int, _DynamicTakeRewrite] = {}

    for input_id in sorted(dynamic_input_ids):
        try:
            planned = _plan_dynamic_input_take_rewrites(
                program,
                inputs,
                input_id,
                output_id,
                live_instruction_indices,
                all_dynamic_input_ids,
            )
        except Exception:
            continue
        axis, permutation, input_rewrites = planned
        index_input = _index_array_like(
            np.asarray(permutation, dtype=np.int64),
            inputs[input_id],
        )
        transform_program = Program(
            instructions=[(TAKE_OPERATOR, (0, 1), axis)],
            n_inputs=2,
        )
        transform = _PendingDynamicInputTransform(
            input_id=input_id,
            program=transform_program,
            inputs=(inputs[input_id], index_input),
        )
        rewritten_inputs[input_id] = _contiguous_input_value(
            _execute_take_stack_program(
                transform_program,
                transform.inputs,
            )
        )
        rewrites.update(input_rewrites)
        pending_transforms.append(transform)

    if not pending_transforms:
        return program, inputs, ()

    instructions = list(program.instructions)
    for instruction_index, rewrite in rewrites.items():
        instruction = instructions[instruction_index]
        source, _index = get_arguments(instruction)
        instructions[instruction_index] = (
            SLICE_OPERATOR,
            (source,),
            rewrite.axis,
            rewrite.start,
            rewrite.stop,
        )

    return (
        Program(instructions=instructions, n_inputs=program.n_inputs),
        tuple(rewritten_inputs),
        tuple(pending_transforms),
    )


def _plan_dynamic_input_take_rewrites(
    program: Program,
    inputs: tuple[Any, ...],
    input_id: int,
    output_id: int,
    live_instruction_indices: set[int],
    dynamic_input_ids: frozenset[int],
) -> tuple[int, tuple[int, ...], dict[int, _DynamicTakeRewrite]]:
    if output_id == input_id:
        raise RuntimeError("TODO: idk why this results in a runtime error")

    source_shape = tuple(
        int(dimension) for dimension in getattr(inputs[input_id], "shape", ())
    )
    if not source_shape:
        raise Value

    axis: int | None = None
    permutation: list[int] = []
    rewrites: dict[int, _DynamicTakeRewrite] = {}

    for instruction_index in sorted(live_instruction_indices):
        instruction = program.instructions[instruction_index]
        arguments = get_arguments(instruction)
        if input_id not in arguments:
            continue
        if get_operator(instruction) != TAKE_OPERATOR or len(arguments) != 2:
            return None
        source, index = arguments
        if (
            source != input_id
            or index >= program.n_inputs
            or index in dynamic_input_ids
        ):
            return None

        take_index_values = _input_index_vector(inputs[index])
        if take_index_values is None:
            return None

        normalized_axis = normalize_axis(take_axis(instruction), len(source_shape))
        if axis is None:
            axis = normalized_axis
        elif axis != normalized_axis:
            return None

        start = len(permutation)
        permutation.extend(take_index_values)
        rewrites[instruction_index] = _DynamicTakeRewrite(
            axis=normalized_axis,
            start=start,
            stop=len(permutation),
        )

    if axis is None or not rewrites:
        return None

    return axis, tuple(permutation), rewrites


def _input_index_vector(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, ScaledTensor):
        return None
    if _is_torch_value(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 1:
        return None
    return tuple(int(index) for index in array.tolist())


def _transformable_nodes_by_dynamic(
    program: Program,
    dynamic_input_ids: frozenset[int],
    live_instruction_indices: set[int],
) -> dict[int, frozenset[int]]:
    dynamic_dependencies: dict[int, frozenset[int]] = {
        input_id: (
            frozenset((input_id,)) if input_id in dynamic_input_ids else frozenset()
        )
        for input_id in range(program.n_inputs)
    }
    transformable: dict[int, set[int]] = {
        input_id: set() for input_id in dynamic_input_ids
    }

    for instruction_index, instruction in enumerate(program.instructions):
        arguments = get_arguments(instruction)
        dependencies = frozenset().union(
            *(dynamic_dependencies[argument] for argument in arguments)
        )
        result_id = program.n_inputs + instruction_index
        dynamic_dependencies[result_id] = dependencies

        if len(dependencies) != 1:
            continue
        input_id = next(iter(dependencies))
        if (
            _is_transformable_dynamic_instruction(
                program,
                instruction,
                arguments,
                input_id,
                dynamic_dependencies,
                transformable[input_id],
            )
            and instruction_index in live_instruction_indices
        ):
            transformable[input_id].add(result_id)

    return {
        input_id: frozenset(result_ids)
        for input_id, result_ids in transformable.items()
    }


def _is_transformable_dynamic_instruction(
    program: Program,
    instruction: Instruction,
    arguments: tuple[int, ...],
    input_id: int,
    dynamic_dependencies: dict[int, frozenset[int]],
    transformable_nodes: set[int],
) -> bool:
    operator = get_operator(instruction)
    if operator == SLICE_OPERATOR:
        if len(arguments) != 1:
            return False
        source = arguments[0]
        return source == input_id or source in transformable_nodes
    if operator == TAKE_OPERATOR:
        if len(arguments) != 2:
            return False
        source, index = arguments
        if dynamic_dependencies[index]:
            return False
        return source == input_id or source in transformable_nodes
    if operator == SELECT_OPERATOR:
        if len(arguments) != 1:
            return False
        source = arguments[0]
        return source == input_id or source in transformable_nodes
    if operator != STACK_OPERATOR:
        return False
    return all(
        argument == input_id
        or (argument >= program.n_inputs and argument in transformable_nodes)
        for argument in arguments
    )


def _transform_subgraph_nodes(
    program: Program,
    output_id: int,
    input_id: int,
    transformable_nodes: frozenset[int],
) -> frozenset[int]:
    pending = [output_id]
    result_ids: set[int] = set()
    while pending:
        tensor_id = pending.pop()
        if tensor_id == input_id or tensor_id < program.n_inputs:
            continue
        if tensor_id not in transformable_nodes:
            return frozenset()
        if tensor_id in result_ids:
            continue
        result_ids.add(tensor_id)
        instruction = program.instructions[tensor_id - program.n_inputs]
        for argument in get_arguments(instruction):
            if argument >= program.n_inputs or argument == input_id:
                pending.append(argument)
    return frozenset(result_ids)


def _dynamic_input_has_external_raw_consumers(
    program: Program,
    input_id: int,
    transform_node_ids: frozenset[int],
    live_instruction_indices: set[int],
) -> bool:
    for instruction_index, instruction in enumerate(program.instructions):
        if instruction_index not in live_instruction_indices:
            continue
        if input_id not in get_arguments(instruction):
            continue
        if program.n_inputs + instruction_index not in transform_node_ids:
            return True
    return False


def _transform_subgraph_is_closed(
    program: Program,
    output_id: int,
    transform_node_ids: frozenset[int],
    consumers: dict[int, list[tuple[int, int]]],
    live_instruction_indices: set[int],
) -> bool:
    for result_id in transform_node_ids:
        if result_id == output_id:
            continue
        for consumer_op_index, _argument_position in consumers.get(result_id, []):
            if consumer_op_index not in live_instruction_indices:
                continue
            if program.n_inputs + consumer_op_index not in transform_node_ids:
                return False
    return True


def _build_dynamic_input_transform(
    program: Program,
    inputs: tuple[Any, ...],
    input_id: int,
    output_id: int,
    transform_node_ids: frozenset[int],
) -> _PendingDynamicInputTransform | None:
    static_input_ids = tuple(
        sorted(
            {
                argument
                for result_id in transform_node_ids
                for argument in get_arguments(
                    program.instructions[result_id - program.n_inputs]
                )
                if argument < program.n_inputs and argument != input_id
            }
        )
    )
    input_id_map = {
        input_id: 0,
        **{
            static_input_id: static_position + 1
            for static_position, static_input_id in enumerate(static_input_ids)
        },
    }
    result_id_map: dict[int, int] = {}
    instructions: list[Instruction] = []

    for result_id in sorted(transform_node_ids):
        instruction = program.instructions[result_id - program.n_inputs]
        arguments = get_arguments(instruction)
        argument_map = {
            argument: (
                result_id_map[argument]
                if argument >= program.n_inputs
                else input_id_map[argument]
            )
            for argument in arguments
        }
        result_id_map[result_id] = len(input_id_map) + len(instructions)
        instructions.append(
            map_instruction_arguments(instruction, argument_map.__getitem__)
        )

    if output_id not in result_id_map:
        return None

    transform_program = Program(
        instructions=instructions,
        n_inputs=len(input_id_map),
    )
    if transform_program.output_ssa != result_id_map[output_id]:
        return None

    return _PendingDynamicInputTransform(
        input_id=input_id,
        program=transform_program,
        inputs=tuple(inputs[argument] for argument in (input_id, *static_input_ids)),
    )


def _remove_extracted_transform_nodes(
    program: Program,
    output_id: int,
    extracted_nodes: set[int],
    replacement_by_result: dict[int, int],
) -> tuple[Program, int]:
    tensor_map: dict[int, int] = {
        input_id: input_id for input_id in range(program.n_inputs)
    }
    tensor_map.update(replacement_by_result)
    instructions: list[Instruction] = []

    for instruction_index, instruction in enumerate(program.instructions):
        old_result_id = program.n_inputs + instruction_index
        if old_result_id in extracted_nodes:
            continue

        arguments = get_arguments(instruction)
        mapped_arguments = tuple(tensor_map[argument] for argument in arguments)
        argument_map = dict(zip(arguments, mapped_arguments, strict=True))
        tensor_map[old_result_id] = program.n_inputs + len(instructions)
        instructions.append(
            map_instruction_arguments(instruction, argument_map.__getitem__)
        )

    return Program(instructions=instructions, n_inputs=program.n_inputs), tensor_map[
        output_id
    ]


def _compact_program_inputs(
    program: Program,
    inputs: tuple[Any, ...],
    output_id: int,
) -> PreprocessedProgram:
    compacted_program, compacted_inputs, _input_id_map = (
        _compact_program_inputs_with_input_map(program, inputs, output_id)
    )
    return compacted_program, compacted_inputs


def _compact_program_inputs_with_input_map(
    program: Program,
    inputs: tuple[Any, ...],
    output_id: int,
) -> tuple[Program, PreparedInputs, dict[int, int]]:
    used_input_ids: set[int] = set()
    live_instruction_indices = _live_instruction_indices(program, output_id)

    for instruction_index in live_instruction_indices:
        for argument in get_arguments(program.instructions[instruction_index]):
            if argument < program.n_inputs:
                used_input_ids.add(argument)

    if output_id < program.n_inputs:
        used_input_ids.add(output_id)

    used_input_ids = sorted(used_input_ids)

    input_id_map = {
        old_input_id: new_input_id
        for new_input_id, old_input_id in enumerate(used_input_ids)
    }
    new_input_count = len(used_input_ids)
    result_id_map: dict[int, int] = {}

    def map_tensor_id(tensor_id: int) -> int:
        if tensor_id < program.n_inputs:
            return input_id_map[tensor_id]
        return result_id_map[tensor_id]

    instructions: list[Instruction] = []
    for instruction_index, instruction in enumerate(program.instructions):
        if instruction_index not in live_instruction_indices:
            continue
        old_result_id = program.n_inputs + instruction_index
        result_id_map[old_result_id] = new_input_count + len(instructions)
        instructions.append(map_instruction_arguments(instruction, map_tensor_id))

    compacted_program = Program(instructions=instructions, n_inputs=new_input_count)
    return (
        compacted_program,
        tuple(inputs[input_id] for input_id in used_input_ids),
        input_id_map,
    )


def _append_index_inputs_to_end(
    program: Program,
    inputs: tuple[Any, ...],
    index_input_ids: tuple[int, ...],
    batched_input_ids: tuple[int, ...],
) -> tuple[Program, PreparedInputs, dict[int, int]]:
    index_input_id_set = set(index_input_ids)
    if not index_input_id_set:
        return (
            program,
            PreparedInputs(inputs, batched_input_ids=batched_input_ids),
            {input_id: input_id for input_id in range(program.n_inputs)},
        )

    non_index_input_ids = tuple(
        input_id
        for input_id in range(program.n_inputs)
        if input_id not in index_input_id_set
    )
    reordered_input_ids = (*non_index_input_ids, *index_input_ids)
    input_id_map = {
        old_input_id: new_input_id
        for new_input_id, old_input_id in enumerate(reordered_input_ids)
    }

    def map_tensor_id(tensor_id: int) -> int:
        if tensor_id < program.n_inputs:
            return input_id_map[tensor_id]
        return tensor_id

    reordered_program = Program(
        instructions=[
            map_instruction_arguments(instruction, map_tensor_id)
            for instruction in program.instructions
        ],
        n_inputs=program.n_inputs,
    )
    reordered_inputs = PreparedInputs(
        tuple(inputs[input_id] for input_id in reordered_input_ids),
        index_input_ids=tuple(input_id_map[input_id] for input_id in index_input_ids),
        batched_input_ids=tuple(
            input_id_map[input_id]
            for input_id in batched_input_ids
            if input_id in input_id_map
        ),
    )
    return reordered_program, reordered_inputs, input_id_map


def _live_instruction_indices(program: Program, output_id: int) -> set[int]:
    live_instruction_indices: set[int] = set()
    pending = [output_id]

    while pending:
        tensor_id = pending.pop()
        if tensor_id < program.n_inputs:
            continue

        instruction_index = tensor_id - program.n_inputs
        if instruction_index in live_instruction_indices:
            continue
        live_instruction_indices.add(instruction_index)
        pending.extend(get_arguments(program.instructions[instruction_index]))

    return live_instruction_indices


def _program_from_ssa_path(
    format_string: str,
    ssa_path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> Program:
    instructions: list[Instruction] = []
    for first, second, operator in to_annotated_ssa_path(
        format_string,
        ssa_path,
        is_ascii=True,
    ):
        instructions.append(make_einsum_instruction(operator, first, second))
    index_strings, _ = parse_format_string(format_string)
    return Program(instructions=instructions, n_inputs=len(index_strings))


def to_annotated_ssa_path(
    format_string: str,
    ssa_path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    is_ascii: bool = False,
) -> list[tuple[int, int, str]]:
    """Annotate an SSA path with the pairwise einsum string for each step."""
    inputs, output = format_string.replace(" ", "").split("->")
    inputs = inputs.split(",")
    if len(inputs) < 2:
        raise ValueError("einsum expressions involving one tensor are not supported")

    histogram = Counter(format_string.replace(" ", ""))
    annotated_ssa_path: list[tuple[int, int, str]] = []

    for step_index, (first, second) in enumerate(ssa_path, start=1):
        t1 = inputs[first]
        t2 = inputs[second]
        visited: set[str] = set()
        unique_indices: list[str] = []

        for char in t1 + t2:
            if char not in visited:
                unique_indices.append(char)
                visited.add(char)
            histogram[char] -= 1

        if step_index == len(ssa_path):
            t3 = output
        else:
            t3 = "".join(char for char in unique_indices if histogram[char] > 0)
            for char in t3:
                histogram[char] += 1

        pairwise_expression = f"{t1},{t2}->{t3}"
        if is_ascii:
            pairwise_expression = _to_ascii_einsum(pairwise_expression)

        annotated_ssa_path.append((first, second, pairwise_expression))
        inputs.append(t3)

    return annotated_ssa_path


def _to_ascii_einsum(expression: str) -> str:
    ascii_index = 0
    char_mapping: dict[str, str] = {}
    parts: list[str] = []
    for char in expression:
        if char in ",->":
            parts.append(char)
            continue
        if char not in char_mapping:
            if ascii_index == len(_LABELS):
                raise RuntimeError(
                    f"ERROR: {expression} cannot be converted to ASCII, "
                    "it is too large."
                )
            char_mapping[char] = _LABELS[ascii_index]
            ascii_index += 1
        parts.append(char_mapping[char])
    return "".join(parts)


def _analyze_program(
    program: Program,
    input_shapes: list[Shape],
    input_scale_states: list[ScaleState],
    *,
    fold_softmax_operations: bool,
    reorder_inputs: bool,
) -> tuple[list[Shape], dict[int, frozenset[int]], dict[int, int], list[_Candidate]]:
    shapes = input_shapes.copy()
    scale_states = list(input_scale_states)
    depths = [-1] * len(input_shapes)
    dependencies: dict[int, frozenset[int]] = {
        tensor_id: frozenset() for tensor_id in range(len(input_shapes))
    }
    first_consumers: dict[int, int] = {}
    candidates: list[_Candidate] = []

    for op_index, instruction in enumerate(program.instructions):
        arguments = get_arguments(instruction)
        for argument in arguments:
            if argument >= len(input_shapes):
                first_consumers.setdefault(argument, op_index)

        result_id = len(input_shapes) + op_index
        depth = max((depths[argument] for argument in arguments), default=-1) + 1
        dependency_set: set[int] = set()
        for argument in arguments:
            dependency_set.update(dependencies[argument])
            if argument >= len(input_shapes):
                dependency_set.add(argument)
        dependencies[result_id] = frozenset(dependency_set)

        argument_shapes = [shapes[argument] for argument in arguments]
        argument_scale_states = [scale_states[argument] for argument in arguments]
        output_shape = _infer_instruction_shape(instruction, argument_shapes)
        shapes.append(output_shape)
        output_scale_state = _infer_instruction_tensor_format(
            instruction,
            argument_scale_states,
            output_shape,
            argument_shapes,
        )
        scale_states.append(output_scale_state)
        depths.append(depth)

        canonical = _canonicalize_instruction(
            instruction,
            argument_shapes,
            argument_scale_states,
            output_shape,
            fold_softmax_operations=fold_softmax_operations,
            reorder_inputs=reorder_inputs,
        )
        if canonical is not None:
            candidates.append(
                _Candidate(
                    op_index=op_index,
                    result_id=result_id,
                    depth=depth,
                    instruction=instruction,
                    arguments=tuple(arguments),
                    dependencies=dependencies[result_id],
                    canonical=canonical,
                )
            )

    return shapes, dependencies, first_consumers, candidates


def _schedule_ready_events(
    program: Program,
    input_count: int,
    candidates: list[_Candidate],
    min_group_size: int,
    fold_same_depth_only: bool,
) -> list[ScheduledEvent]:
    candidate_by_index = {candidate.op_index: candidate for candidate in candidates}
    softmax_future_keys = _softmax_future_keys(
        program,
        input_count,
        candidate_by_index,
    )
    remaining = set(range(len(program.instructions)))
    available = set(range(input_count))
    events: list[ScheduledEvent] = []

    while remaining:
        ready = [
            op_index
            for op_index in sorted(remaining)
            if all(
                argument in available
                for argument in get_arguments(program.instructions[op_index])
            )
        ]
        if not ready:
            raise ValueError(
                "program contains instructions whose inputs are unavailable"
            )

        non_candidate = next(
            (op_index for op_index in ready if op_index not in candidate_by_index),
            None,
        )
        if non_candidate is not None:
            events.append(non_candidate)
            remaining.remove(non_candidate)
            available.add(input_count + non_candidate)
            continue

        ready_group = _select_ready_group(
            ready,
            candidate_by_index,
            min_group_size,
            softmax_future_keys,
            fold_same_depth_only,
        )
        if ready_group is not None:
            events.append(ready_group)
            for candidate in ready_group:
                remaining.remove(candidate.op_index)
                available.add(candidate.result_id)
            continue

        op_index = ready[0]
        events.append(op_index)
        remaining.remove(op_index)
        available.add(input_count + op_index)

    return events


def _select_ready_group(
    ready: list[int],
    candidate_by_index: dict[int, _Candidate],
    min_group_size: int,
    softmax_future_keys: dict[int, int],
    fold_same_depth_only: bool,
) -> tuple[_Candidate, ...] | None:
    grouped_candidates: dict[tuple[Any, ...], list[_Candidate]] = {}
    for op_index in ready:
        candidate = candidate_by_index.get(op_index)
        if candidate is None:
            continue
        group_key = _candidate_group_key(
            candidate,
            softmax_future_keys,
            fold_same_depth_only=fold_same_depth_only,
        )
        grouped_candidates.setdefault(group_key, []).append(candidate)

    for group in grouped_candidates.values():
        if len(group) >= min_group_size:
            return tuple(group)
    return None


def _candidate_group_key(
    candidate: _Candidate,
    softmax_future_keys: dict[int, int],
    *,
    fold_same_depth_only: bool,
) -> tuple[Any, ...]:
    key: tuple[Any, ...]
    if candidate.canonical.operator != SOFTMAX_OPERATOR:
        key = candidate.canonical.signature
    else:
        key = (
            *candidate.canonical.signature,
            softmax_future_keys.get(candidate.result_id),
        )
    if fold_same_depth_only:
        return (*key, ("depth", candidate.depth))
    return key


def _order_scheduled_groups(
    events: list[ScheduledEvent],
    program: Program,
    input_count: int,
) -> list[ScheduledEvent]:
    consumers = _result_consumers(program, input_count)
    event_index_by_op: dict[int, int] = {}

    for event_index, event in enumerate(events):
        if isinstance(event, int):
            event_index_by_op[event] = event_index
            continue
        for candidate in event:
            event_index_by_op[candidate.op_index] = event_index

    ordered_groups: dict[int, tuple[_Candidate, ...]] = {}
    for event_index in reversed(range(len(events))):
        event = events[event_index]
        if isinstance(event, int):
            continue
        ordered_groups[event_index] = tuple(
            sorted(
                event,
                key=lambda candidate: _candidate_future_order_key(
                    candidate,
                    events,
                    consumers,
                    event_index_by_op,
                    ordered_groups,
                ),
            )
        )

    return [
        ordered_groups[event_index] if event_index in ordered_groups else event
        for event_index, event in enumerate(events)
    ]


def _result_consumers(
    program: Program, input_count: int
) -> dict[int, list[tuple[int, int]]]:
    consumers: dict[int, list[tuple[int, int]]] = {}
    for op_index, instruction in enumerate(program.instructions):
        arguments = get_arguments(instruction)
        for argument_position, argument in enumerate(arguments):
            if argument >= input_count:
                consumers.setdefault(argument, []).append((op_index, argument_position))
    return consumers


def _softmax_future_keys(
    program: Program,
    input_count: int,
    candidate_by_index: dict[int, _Candidate],
) -> dict[int, int]:
    consumers = _result_consumers(program, input_count)
    interned_signatures: dict[tuple[Any, ...], int] = {}

    def intern(signature: tuple[Any, ...]) -> int:
        signature_id = interned_signatures.get(signature)
        if signature_id is not None:
            return signature_id
        signature_id = len(interned_signatures)
        interned_signatures[signature] = signature_id
        return signature_id

    def consumer_key(result_id: int) -> int:
        uses = consumers.get(result_id, [])
        if not uses:
            return intern(("output",))
        return intern(
            tuple(
                sorted(
                    _consumer_key(
                        program,
                        candidate_by_index,
                        consumer_op_index,
                        argument_position,
                    )
                    for consumer_op_index, argument_position in uses
                )
            )
        )

    return {
        candidate.result_id: consumer_key(candidate.result_id)
        for candidate in candidate_by_index.values()
        if candidate.canonical.operator == SOFTMAX_OPERATOR
    }


def _consumer_key(
    program: Program,
    candidate_by_index: dict[int, _Candidate],
    consumer_op_index: int,
    argument_position: int,
) -> tuple[Any, ...]:
    consumer_candidate = candidate_by_index.get(consumer_op_index)
    if consumer_candidate is not None:
        return (
            "candidate",
            consumer_candidate.canonical.signature,
            _canonical_operand_position(consumer_candidate, argument_position),
        )

    return (
        "instruction",
        _instruction_future_signature(program.instructions[consumer_op_index]),
    )


def _instruction_future_signature(instruction: Instruction) -> tuple[Any, ...]:
    operator = get_operator(instruction)
    if is_einsum_instruction(instruction):
        signature: tuple[Any, ...] = (
            operator,
            einsum_format(instruction),
            len(get_arguments(instruction)),
        )
        if is_scaled_einsum_instruction(instruction):
            return (*signature, scaled_einsum_output_axis(instruction))
        return signature
    if operator == SOFTMAX_OPERATOR:
        return (operator, softmax_axis(instruction))
    if operator == TAKE_OPERATOR:
        return (operator, take_axis(instruction))
    if operator == SELECT_OPERATOR:
        return (operator, select_axis(instruction), select_index(instruction))
    if operator == SLICE_OPERATOR:
        return (
            operator,
            slice_axis(instruction),
            slice_start(instruction),
            slice_stop(instruction),
        )
    if operator == STACK_OPERATOR:
        return (operator, len(get_arguments(instruction)))
    return (operator, len(get_arguments(instruction)))


def _candidate_future_order_key(
    candidate: _Candidate,
    events: list[ScheduledEvent],
    consumers: dict[int, list[tuple[int, int]]],
    event_index_by_op: dict[int, int],
    ordered_groups: dict[int, tuple[_Candidate, ...]],
) -> tuple[int, ...]:
    order_keys: list[tuple[int, ...]] = []
    for consumer_op_index, argument_position in consumers.get(candidate.result_id, []):
        consumer_event_index = event_index_by_op[consumer_op_index]
        consumer_event = events[consumer_event_index]
        if isinstance(consumer_event, int):
            order_keys.append(
                (
                    consumer_event_index,
                    0,
                    consumer_op_index,
                    argument_position,
                )
            )
            continue

        ordered_group = ordered_groups.get(consumer_event_index, consumer_event)
        consumer_position_by_op = {
            consumer_candidate.op_index: position
            for position, consumer_candidate in enumerate(ordered_group)
        }
        consumer_candidate = ordered_group[consumer_position_by_op[consumer_op_index]]
        canonical_position = _canonical_operand_position(
            consumer_candidate, argument_position
        )
        order_keys.append(
            (
                consumer_event_index,
                0,
                canonical_position,
                consumer_position_by_op[consumer_op_index],
                argument_position,
            )
        )

    if not order_keys:
        return (len(events), candidate.op_index)
    return min(order_keys)


def _canonical_operand_position(candidate: _Candidate, original_position: int) -> int:
    for canonical_position, candidate_original_position in enumerate(
        candidate.canonical.permutation
    ):
        if candidate_original_position == original_position:
            return canonical_position
    raise ValueError("candidate argument position is not part of its permutation")


def _group_refs_by_canonical_operand(
    group: tuple[_Candidate, ...],
    tensor_map: dict[int, _TensorValueRef],
) -> list[tuple[_TensorValueRef, ...]]:
    operand_count = len(group[0].canonical.permutation)
    grouped: list[list[_TensorValueRef]] = [[] for _ in range(operand_count)]
    for candidate in group:
        for canonical_position, original_position in enumerate(
            candidate.canonical.permutation
        ):
            grouped[canonical_position].append(
                tensor_map[candidate.arguments[original_position]]
            )
    return [tuple(arguments) for arguments in grouped]


def _contiguous_range(indices: tuple[int, ...]) -> tuple[int, int] | None:
    if not indices:
        return None
    start = indices[0]
    for offset, index in enumerate(indices):
        if index != start + offset:
            return None
    return start, start + len(indices)


def _materialize_unoptimized_take_batch(
    refs: tuple[_TensorValueRef, ...],
    append_instruction: Callable[[_RawInstruction], _ResultArgument],
    append_index_input: Callable[[np.ndarray], _IndexInputArgument],
) -> _RawArgument | None:
    first_ref = refs[0]
    if isinstance(first_ref, _SliceRef) and all(
        isinstance(ref, _SliceRef) and ref.source == first_ref.source for ref in refs
    ):
        index_argument = append_index_input(
            np.asarray([ref.index for ref in refs], dtype=np.int64)
        )
        return append_instruction((TAKE_OPERATOR, (first_ref.source, index_argument)))

    if (
        isinstance(first_ref, _TakeRef)
        and first_ref.axis == 0
        and all(
            isinstance(ref, _TakeRef)
            and ref.source == first_ref.source
            and ref.axis == first_ref.axis
            for ref in refs
        )
    ):
        index_argument = append_index_input(
            np.asarray([ref.index for ref in refs], dtype=np.int64)
        )
        return append_instruction(
            (TAKE_OPERATOR, (first_ref.source, index_argument), first_ref.axis)
        )

    if (
        isinstance(first_ref, _SliceRangeRef)
        and first_ref.axis == 0
        and all(
            isinstance(ref, _SliceRangeRef)
            and ref.source == first_ref.source
            and ref.axis == first_ref.axis
            and ref.stop - ref.start == first_ref.stop - first_ref.start
            for ref in refs
        )
    ):
        length = first_ref.stop - first_ref.start
        if length < 0:
            return None
        indices = np.asarray(
            [np.arange(ref.start, ref.stop, dtype=np.int64) for ref in refs],
            dtype=np.int64,
        )
        index_argument = append_index_input(indices)
        return append_instruction(
            (TAKE_OPERATOR, (first_ref.source, index_argument), first_ref.axis)
        )

    return None


def _materialize_take_batch(
    refs: tuple[_TensorValueRef, ...],
    first_ref: _TakeRef,
    source_shape: Shape,
    raw_instructions: list[_RawInstruction],
    append_instruction: Callable[[_RawInstruction], _ResultArgument],
    append_index_input: Callable[[np.ndarray], _IndexInputArgument],
) -> _RawArgument | None:
    if first_ref.axis != 0:
        return None

    indices = tuple(ref.index for ref in refs if isinstance(ref, _TakeRef))
    if len(indices) != len(refs):
        return None

    contiguous = _contiguous_range(indices)
    if contiguous is not None:
        start, stop = contiguous
        if start == 0 and stop == source_shape[0]:
            return first_ref.source
        return append_instruction(
            (SLICE_OPERATOR, (first_ref.source,), first_ref.axis, start, stop)
        )

    index_argument = append_index_input(np.asarray(indices, dtype=np.int64))
    commuted = _commute_take_batch_through_softmax(
        first_ref,
        index_argument,
        raw_instructions,
        append_instruction,
    )
    if commuted is not None:
        return commuted
    return append_instruction(
        (TAKE_OPERATOR, (first_ref.source, index_argument), first_ref.axis)
    )


def _commute_take_batch_through_softmax(
    take_ref: _TakeRef,
    index_argument: _IndexInputArgument,
    raw_instructions: list[_RawInstruction],
    append_instruction: Callable[[_RawInstruction], _ResultArgument],
) -> _RawArgument | None:
    if not isinstance(take_ref.source, _ResultArgument):
        return None
    source_instruction = raw_instructions[take_ref.source.instruction_index]
    if get_operator(source_instruction) != SOFTMAX_OPERATOR:
        return None
    (softmax_source,) = get_arguments(source_instruction)
    source_axis = softmax_axis(source_instruction)
    if take_ref.axis == source_axis:
        return None

    index_rank = 1
    target_axis = source_axis
    if take_ref.axis < source_axis:
        target_axis = source_axis - 1 + index_rank

    taken = append_instruction(
        (TAKE_OPERATOR, (softmax_source, index_argument), take_ref.axis)
    )
    return append_instruction((SOFTMAX_OPERATOR, (taken,), target_axis))


def _is_last_result(
    argument: _RawArgument, raw_instructions: list[_RawInstruction]
) -> bool:
    return (
        isinstance(argument, _ResultArgument)
        and argument.instruction_index == len(raw_instructions) - 1
    )


def _copy_to_last_result(
    argument: _RawArgument,
    append_instruction: Callable[[_RawInstruction], _ResultArgument],
    append_index_input: Callable[[np.ndarray], _IndexInputArgument],
) -> _RawArgument:
    stack_result = append_instruction((STACK_OPERATOR, (argument,)))
    return append_instruction((SELECT_OPERATOR, (stack_result,), 0, 0))


def _resolve_raw_instructions(
    raw_instructions: list[_RawInstruction],
    input_count: int,
    index_input_count: int,
) -> list[Instruction]:
    def resolve_argument(argument: _RawArgument) -> int:
        if isinstance(argument, _InputArgument):
            return argument.input_id
        if isinstance(argument, _IndexInputArgument):
            return input_count + argument.input_index
        return input_count + index_input_count + argument.instruction_index

    return [
        map_instruction_arguments(instruction, resolve_argument)
        for instruction in raw_instructions
    ]


def _batched_operator(inputs: tuple[str, ...], output: str) -> str:
    used_labels = set(output + "".join(inputs))
    batch_label = next((label for label in _LABELS if label not in used_labels), None)
    if batch_label is None:
        raise ValueError("could not allocate batch label for fused einsum")
    batched_inputs = ",".join(f"{batch_label}{subscript}" for subscript in inputs)
    return f"{batched_inputs}->{batch_label}{output}"


def _make_batched_instruction(
    canonical: CanonicalEinsum,
    arguments: tuple[_RawArgument, ...],
) -> _RawInstruction:
    if canonical.operator == EINSUM_OPERATOR:
        batched_format = _batched_operator(canonical.inputs, canonical.output)
        return make_einsum_instruction(batched_format, *arguments)
    if canonical.operator == LSE_SUM_EINSUM_OPERATOR:
        batched_format = _batched_operator(canonical.inputs, canonical.output)
        return make_logspace_einsum_instruction(
            batched_format,
            *arguments,
        )
    if canonical.operator in SCALED_EINSUM_OPERATORS:
        if len(arguments) != 2:
            raise ValueError("scaled batched einsum folding requires binary einsums")
        if canonical.output_scale_axis is None:
            raise ValueError("scaled batched einsum requires an output scale axis")
        batched_format = _batched_operator(canonical.inputs, canonical.output)
        return make_scaled_einsum_instruction(
            canonical.operator,
            batched_format,
            arguments[0],
            arguments[1],
            canonical.output_scale_axis + 1,
        )
    if canonical.operator == SOFTMAX_OPERATOR:
        if len(arguments) != 1:
            raise ValueError("batched softmax folding requires one operand")
        if canonical.output_scale_axis is None:
            raise ValueError("batched softmax requires an axis")
        return (SOFTMAX_OPERATOR, (arguments[0],), canonical.output_scale_axis + 1)
    raise ValueError(f"unsupported batched operator: {canonical.operator!r}")


def _canonicalize_instruction(
    instruction: Instruction,
    operand_shapes: list[Shape],
    operand_scale_states: list[ScaleState],
    output_shape: Shape,
    *,
    fold_softmax_operations: bool,
    reorder_inputs: bool,
) -> CanonicalEinsum | None:
    canonical = _canonicalize_instruction_einsum(
        instruction,
        operand_shapes,
        operand_scale_states,
        output_shape,
        reorder_inputs=reorder_inputs,
    )
    if canonical is not None:
        return canonical
    if not fold_softmax_operations:
        return None
    return _canonicalize_instruction_softmax(
        instruction,
        operand_shapes,
        operand_scale_states,
        output_shape,
    )


def _canonicalize_instruction_einsum(
    instruction: Instruction,
    operand_shapes: list[Shape],
    operand_scale_states: list[ScaleState],
    output_shape: Shape,
    *,
    reorder_inputs: bool,
) -> CanonicalEinsum | None:
    if not is_einsum_instruction(instruction):
        return None
    operator = get_operator(instruction)
    parsed = _parse_einsum(einsum_format(instruction))
    if parsed is None or len(parsed.inputs) != len(operand_shapes):
        return None
    if is_scaled_einsum_instruction(instruction) and not parsed.output:
        return None

    if (
        is_normal_einsum_instruction(instruction)
        or is_logspace_einsum_instruction(instruction)
    ) and any(scale_state is not None for scale_state in operand_scale_states):
        return None

    output_scale_axis = (
        normalize_axis(scaled_einsum_output_axis(instruction), len(parsed.output))
        if is_scaled_einsum_instruction(instruction)
        else None
    )
    best: CanonicalEinsum | None = None
    if not reorder_inputs:
        candidate_permutations = (tuple(range(len(parsed.inputs))),)
    elif is_logspace_einsum_instruction(instruction):
        candidate_permutations = (
            (0, *permutation)
            for permutation in permutations(range(1, len(parsed.inputs)))
        )
    else:
        candidate_permutations = permutations(range(len(parsed.inputs)))

    for permutation in candidate_permutations:
        permuted_inputs = tuple(parsed.inputs[index] for index in permutation)
        permuted_shapes = tuple(operand_shapes[index] for index in permutation)
        permuted_scale_states = tuple(
            operand_scale_states[index] for index in permutation
        )
        normalized = _normalize_labels(permuted_inputs, parsed.output)
        if normalized is None:
            return None
        normalized_inputs, normalized_output = normalized
        signature = (
            operator,
            normalized_inputs,
            normalized_output,
            permuted_shapes,
            tuple(
                _scale_axis_key(scale_state) for scale_state in permuted_scale_states
            ),
            output_shape,
            _scale_axis_key(output_scale_axis),
        )
        canonical = CanonicalEinsum(
            signature=signature,
            operator=operator,
            inputs=normalized_inputs,
            output=normalized_output,
            permutation=tuple(permutation),
            operand_shapes=permuted_shapes,
            operand_scale_states=permuted_scale_states,
            output_shape=output_shape,
            output_scale_axis=output_scale_axis,
        )
        if best is None or canonical.signature < best.signature:
            best = canonical

    return best


def _canonicalize_instruction_softmax(
    instruction: Instruction,
    operand_shapes: list[Shape],
    operand_scale_states: list[ScaleState],
    output_shape: Shape,
) -> CanonicalEinsum | None:
    if get_operator(instruction) != SOFTMAX_OPERATOR:
        return None
    if len(operand_shapes) != 1 or len(operand_scale_states) != 1:
        return None
    if operand_scale_states[0] is not None:
        return None
    if not operand_shapes[0] or output_shape != operand_shapes[0]:
        return None

    axis = get_softmax_axis(instruction)
    return CanonicalEinsum(
        signature=(
            SOFTMAX_OPERATOR,
            operand_shapes[0],
            output_shape,
            axis,
        ),
        operator=SOFTMAX_OPERATOR,
        inputs=(),
        output="",
        permutation=(0,),
        operand_shapes=operand_shapes,
        operand_scale_states=operand_scale_states,
        output_shape=output_shape,
        output_scale_axis=axis,
    )


def _scale_axis_key(scale_state: ScaleState | int) -> ScaleAxisKey:
    if scale_state is None:
        return (0, -1)
    if isinstance(scale_state, _ScaleState):
        return (1, scale_state.axis)
    return (1, scale_state)


def _normalize_labels(
    inputs: tuple[str, ...], output: str
) -> tuple[tuple[str, ...], str] | None:
    label_map: dict[str, str] = {}

    def normalize_subscript(subscript: str) -> str | None:
        normalized_labels: list[str] = []
        for label in subscript:
            if label not in label_map:
                if len(label_map) >= len(_LABELS):
                    return None
                label_map[label] = _LABELS[len(label_map)]
            normalized_labels.append(label_map[label])
        return "".join(normalized_labels)

    normalized_inputs = []
    for subscript in inputs:
        normalized = normalize_subscript(subscript)
        if normalized is None:
            return None
        normalized_inputs.append(normalized)
    normalized_output = normalize_subscript(output)
    if normalized_output is None:
        return None
    return tuple(normalized_inputs), normalized_output


def _infer_instruction_shape(
    instruction: Instruction,
    operand_shapes: list[Shape],
) -> Shape:
    operator = get_operator(instruction)
    match operator:
        case "einsum":
            format_string = get_einsum_format_string(instruction)
            return infer_einsum_shape(format_string, operand_shapes)
        case "stack":
            axis = get_stack_axis(instruction)
            return infer_stack_shape(operand_shapes, axis)
        case "take":
            axis = get_take_axis(instruction)
            return infer_take_shape(operand_shapes[0], operand_shapes[1], axis)
        case "softmax":
            return infer_softmax_shape(operand_shapes[0])
        case "select":
            axis = get_select_axis(instruction)
            return infer_select_shape(operand_shapes[0], axis)
        case "slice":
            start = get_slice_start(instruction)
            stop = get_slice_stop(instruction)
            axis = get_slice_axis(instruction)
            return infer_slice_shape(operand_shapes[0], start, stop, axis)
        case operator if operator in UNARY_OPERATORS:
            return infer_unary_shape(operand_shapes[0])
        case operator if operator in BINARY_OPERATORS:
            return infer_binary_shape(*operand_shapes)
        case _:
            raise ValueError(f"unsupported operator for shape inference: {operator!r}")


def _infer_instruction_tensor_format(
    instruction: Instruction,
    operand_tensor_formats: list[ScaleState],
    output_shape: Shape,
    operand_shapes: list[Shape],
) -> ScaleState:
    operator = get_operator(instruction)
    if is_normal_einsum_instruction(instruction):
        if any(scale_state is not None for scale_state in operand_tensor_formats):
            raise ValueError("normal einsum does not accept scaled inputs")
        return None
    if is_logspace_einsum_instruction(instruction):
        if any(scale_state is not None for scale_state in operand_tensor_formats):
            raise ValueError("logspace einsum does not accept scaled inputs")
        return None
    if is_scaled_einsum_instruction(instruction):
        return _ScaleState(
            normalize_axis(scaled_einsum_output_axis(instruction), len(output_shape))
        )
    if operator == "log":
        return None
    if operator in UNARY_OPERATORS:
        if any(scale_state is not None for scale_state in operand_tensor_formats):
            raise ValueError(f"unary operator {operator!r} does not support scaling")
        return None
    if operator == SOFTMAX_OPERATOR:
        if any(scale_state is not None for scale_state in operand_tensor_formats):
            raise ValueError("softmax does not support scaled tensors")
        return None
    if operator in BINARY_OPERATORS:
        scaled = [
            scale_state
            for scale_state in operand_tensor_formats
            if scale_state is not None
        ]
        if not scaled:
            return None
        if operator != "*":
            raise ValueError(f"binary operator {operator!r} does not support scaling")
        first_axis = scaled[0].axis
        if any(scale_state.axis != first_axis for scale_state in scaled):
            raise ValueError("scaled multiplication requires matching scale axes")
        return _ScaleState(first_axis)
    if operator == STACK_OPERATOR:
        scaled = [
            scale_state
            for scale_state in operand_tensor_formats
            if scale_state is not None
        ]
        if not scaled:
            return None
        if len(scaled) != len(operand_tensor_formats):
            raise ValueError("cannot stack mixed scaled and unscaled tensors")
        first_axis = scaled[0].axis
        if any(scale_state.axis != first_axis for scale_state in scaled):
            raise ValueError("cannot stack scaled tensors with different scale axes")
        return _ScaleState(first_axis + 1)
    if operator == TAKE_OPERATOR:
        source_scale_state = operand_tensor_formats[0]
        if source_scale_state is None:
            return None
        axis = normalize_axis(take_axis(instruction), len(operand_shapes[0]))
        if source_scale_state.axis == axis:
            raise ValueError("cannot take away the scale axis")
        index_rank = len(operand_shapes[1])
        if source_scale_state.axis < axis:
            return source_scale_state
        return _ScaleState(source_scale_state.axis - 1 + index_rank)
    if operator == SELECT_OPERATOR:
        source_scale_state = operand_tensor_formats[0]
        if source_scale_state is None:
            return None
        axis = normalize_axis(select_axis(instruction), len(operand_shapes[0]))
        if source_scale_state.axis == axis:
            raise ValueError("cannot select away the scale axis")
        if source_scale_state.axis < axis:
            return source_scale_state
        return _ScaleState(source_scale_state.axis - 1)
    if operator == SLICE_OPERATOR:
        return operand_tensor_formats[0]
    return None


def _parse_einsum(expression: str) -> ParsedEinsum:
    index_strings, output_string = parse_format_string(expression)
    return ParsedEinsum(tuple(index_strings), output_string)


def _format_einsum(parsed: ParsedEinsum) -> str:
    return f"{','.join(parsed.inputs)}->{parsed.output}"
