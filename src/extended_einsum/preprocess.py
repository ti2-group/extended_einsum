from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override

import numpy as np

from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import OperatorEinsum
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.shapes import (
    Shape,
    infer_einsum_shape,
)
from extended_einsum.utils import parse_format_string

_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_COMMUTATIVE_OPERATORS = frozenset({"+", "*"})


class _EinsumLabelAllocator:
    def __init__(
        self,
        *,
        source_to_label: dict[str, str] | None = None,
        next_label_index: int = 0,
    ) -> None:
        self._source_to_label = (
            {} if source_to_label is None else source_to_label.copy()
        )
        self._next_label_index = next_label_index

    def copy(self) -> _EinsumLabelAllocator:
        return _EinsumLabelAllocator(
            source_to_label=self._source_to_label,
            next_label_index=self._next_label_index,
        )

    def new_label(self) -> str:
        label_index = self._next_label_index
        self._next_label_index += 1
        if label_index < len(_LABELS):
            return _LABELS[label_index]
        return chr(0x100 + label_index - len(_LABELS))

    def get_label(self, source_label: str) -> str:
        if source_label not in self._source_to_label:
            self._source_to_label[source_label] = self.new_label()
        return self._source_to_label[source_label]

    def normalize_subscript(self, subscript: str) -> str:
        return "".join(self.get_label(label) for label in subscript)

    def normalize_expression(self, expression: str) -> str:
        parts: list[str] = []
        for char in expression:
            if char in ",->":
                parts.append(char)
                continue
            parts.append(self.get_label(char))
        return "".join(parts)

    @classmethod
    def remap_expression(cls, expression: str) -> str:
        return cls().normalize_expression(expression)


def choose_single_format_backend_functions(
    argument_tensor_format: TensorFormat,
    backend_functions: BackendFunctions[TBackendArrayCovariant],
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
    backend_functions: BackendFunctions[TBackendArrayCovariant],
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

    stability: Literal["none", "scaled", "logspace"]
    shapes: list[Shape]
    tensor_formats: list[TensorFormat]
    parameter_indices: list[int]
    consumers_of_ssa_id: list[list[int]]

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

    def strip(self) -> Program:
        return Program(self.instructions, self.n_inputs)


class PreprocessingRoutine(Protocol):
    @staticmethod
    def apply(program: RichProgram) -> RichProgram: ...


@dataclass(frozen=True)
class OutputDepthOpGroupMember:
    op_index: int
    result_id: int
    arguments: tuple[int, ...]
    canonical_argument_order: tuple[int, ...]


@dataclass(frozen=True)
class _OutputDepthOpSignature:
    depth: int
    operator: Operator
    canonical_instruction_specific_arguments: tuple[Any, ...]
    operand_shapes: tuple[Shape, ...]
    operand_tensor_formats: tuple[TensorFormat, ...]


@dataclass(frozen=True)
class OutputDepthOpGroup(_OutputDepthOpSignature):
    members: tuple[OutputDepthOpGroupMember, ...]


def group_identical_ops_by_output_depth(
    program: RichProgram,
    *,
    min_group_size: int = 2,
) -> tuple[OutputDepthOpGroup, ...]:
    """Group live, canonically identical ops by nearest output depth."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")

    output_depths = _nearest_output_depths(program)
    result_dependencies = _instruction_result_dependencies(program)
    buckets: dict[_OutputDepthOpSignature, list[OutputDepthOpGroupMember]] = {}

    for op_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + op_index
        depth = output_depths.get(result_id)
        if depth is None:
            continue

        signature, member = _canonicalize_output_depth_op(program, op_index, depth)
        buckets.setdefault(signature, []).append(member)

    groups: list[OutputDepthOpGroup] = []
    for signature, members in buckets.items():
        for safe_group in _split_dependency_safe_op_groups(
            members,
            result_dependencies,
        ):
            if len(safe_group) < min_group_size:
                continue
            canonical_args = signature.canonical_instruction_specific_arguments
            groups.append(
                OutputDepthOpGroup(
                    depth=signature.depth,
                    operator=signature.operator,
                    canonical_instruction_specific_arguments=canonical_args,
                    operand_shapes=signature.operand_shapes,
                    operand_tensor_formats=signature.operand_tensor_formats,
                    members=safe_group,
                )
            )

    return tuple(sorted(groups, key=_output_depth_group_sort_key))


def _nearest_output_depths(program: Program) -> dict[int, int]:
    depths: dict[int, int] = {}
    pending: deque[tuple[int, int]] = deque([(program.output_ssa, 0)])

    while pending:
        ssa_id, depth = pending.popleft()
        known_depth = depths.get(ssa_id)
        if known_depth is not None and known_depth <= depth:
            continue
        depths[ssa_id] = depth

        if ssa_id < program.n_inputs:
            continue

        instruction = program.instructions[ssa_id - program.n_inputs]
        for argument in get_arguments(instruction):
            pending.append((argument, depth + 1))

    return depths


def _instruction_result_dependencies(program: Program) -> dict[int, frozenset[int]]:
    dependencies: dict[int, frozenset[int]] = {
        input_id: frozenset() for input_id in range(program.n_inputs)
    }

    for op_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + op_index
        result_dependencies: set[int] = set()
        for argument in get_arguments(instruction):
            result_dependencies.update(dependencies[argument])
            if argument >= program.n_inputs:
                result_dependencies.add(argument)
        dependencies[result_id] = frozenset(result_dependencies)

    return dependencies


def _canonicalize_output_depth_op(
    program: RichProgram,
    op_index: int,
    depth: int,
) -> tuple[_OutputDepthOpSignature, OutputDepthOpGroupMember]:
    instruction = program.instructions[op_index]
    operator = get_operator(instruction)
    if operator == EINSUM_OPERATOR:
        return _canonicalize_output_depth_einsum(program, op_index, depth)

    arguments = get_arguments(instruction)
    canonical_argument_order = _canonical_argument_order_for_operator(
        operator,
        arguments,
        program,
    )
    return _make_output_depth_op_group_entry(
        program,
        op_index,
        depth,
        operator,
        get_instruction_specific_arguments(instruction),
        canonical_argument_order,
    )


def _make_output_depth_op_group_entry(
    program: RichProgram,
    op_index: int,
    depth: int,
    operator: Operator,
    canonical_args: tuple[Any, ...],
    canonical_argument_order: tuple[int, ...],
) -> tuple[_OutputDepthOpSignature, OutputDepthOpGroupMember]:
    arguments = get_arguments(program.instructions[op_index])
    ordered_arguments = tuple(arguments[index] for index in canonical_argument_order)
    operand_shapes = tuple(program.shapes[argument] for argument in ordered_arguments)
    operand_tensor_formats = tuple(
        program.tensor_formats[argument] for argument in ordered_arguments
    )

    return (
        _OutputDepthOpSignature(
            depth=depth,
            operator=operator,
            canonical_instruction_specific_arguments=canonical_args,
            operand_shapes=operand_shapes,
            operand_tensor_formats=operand_tensor_formats,
        ),
        OutputDepthOpGroupMember(
            op_index=op_index,
            result_id=program.n_inputs + op_index,
            arguments=arguments,
            canonical_argument_order=canonical_argument_order,
        ),
    )


def _canonical_argument_order_for_operator(
    operator: Operator,
    arguments: tuple[int, ...],
    program: RichProgram,
) -> tuple[int, ...]:
    if operator not in _COMMUTATIVE_OPERATORS:
        return tuple(range(len(arguments)))

    return tuple(
        sorted(
            range(len(arguments)),
            key=lambda index: (
                program.shapes[arguments[index]],
                program.tensor_formats[arguments[index]].sort_key,
                index,
            ),
        )
    )


def _canonicalize_output_depth_einsum(
    program: RichProgram,
    op_index: int,
    depth: int,
) -> tuple[_OutputDepthOpSignature, OutputDepthOpGroupMember]:
    instruction = program.instructions[op_index]
    arguments = get_arguments(instruction)

    input_strings, output_string = parse_format_string(
        get_einsum_format_string(instruction)
    )

    if len(input_strings) != len(arguments):
        raise ValueError("einsum input count must match argument count")

    # Labels alone do not identify indistinguishable operands; include tensor
    # format metadata so canonical permutations keep tensor slots aligned.
    argument_sort_keys = tuple(
        (program.tensor_formats[argument].sort_key,) for argument in arguments
    )
    canonical_format_string, canonical_argument_order = (
        _canonicalize_einsum_string_from_output(
            input_strings,
            output_string,
            argument_sort_keys,
        )
    )
    return _make_output_depth_op_group_entry(
        program,
        op_index,
        depth,
        EINSUM_OPERATOR,
        (canonical_format_string,),
        canonical_argument_order,
    )


def _canonicalize_einsum_string_from_output(
    input_strings: list[str],
    output_string: str,
    argument_sort_keys: tuple[tuple[Any, ...], ...],
) -> tuple[str, tuple[int, ...]]:
    if len(input_strings) != len(argument_sort_keys):
        raise ValueError("argument sort keys must match einsum inputs")

    label_allocator = _EinsumLabelAllocator()
    normalized_output = label_allocator.normalize_subscript(output_string)

    def argument_sort_key(argument_position: int) -> tuple[Any, ...]:
        return (
            label_allocator.copy().normalize_subscript(
                input_strings[argument_position]
            ),
            argument_sort_keys[argument_position],
            argument_position,
        )

    canonical_argument_order = tuple(
        sorted(range(len(input_strings)), key=argument_sort_key)
    )
    normalized_inputs = tuple(
        label_allocator.normalize_subscript(input_strings[index])
        for index in canonical_argument_order
    )
    return (
        f"{','.join(normalized_inputs)}->{normalized_output}",
        canonical_argument_order,
    )


def _split_dependency_safe_op_groups(
    members: list[OutputDepthOpGroupMember],
    result_dependencies: dict[int, frozenset[int]],
) -> list[tuple[OutputDepthOpGroupMember, ...]]:
    groups: list[list[OutputDepthOpGroupMember]] = []
    for member in sorted(members, key=lambda item: item.op_index):
        for group in groups:
            if all(
                not _op_members_depend_on_each_other(
                    member,
                    existing_member,
                    result_dependencies,
                )
                for existing_member in group
            ):
                group.append(member)
                break
        else:
            groups.append([member])
    return [tuple(group) for group in groups]


def _op_members_depend_on_each_other(
    first: OutputDepthOpGroupMember,
    second: OutputDepthOpGroupMember,
    result_dependencies: dict[int, frozenset[int]],
) -> bool:
    first_result_id = first.result_id
    second_result_id = second.result_id
    return (
        second_result_id in result_dependencies[first_result_id]
        or first_result_id in result_dependencies[second_result_id]
    )


def _output_depth_group_sort_key(group: OutputDepthOpGroup) -> tuple[Any, ...]:
    return (
        group.depth,
        group.operator,
        group.canonical_instruction_specific_arguments,
        group.operand_shapes,
        tuple(tensor_format.sort_key for tensor_format in group.operand_tensor_formats),
        tuple(member.op_index for member in group.members),
    )


@dataclass(frozen=True)
class _ConnectedEinsumComponent:
    op_indices: frozenset[int]
    sink_op_index: int
    boundary_arguments: tuple[int, ...]
    format_string: str


def extract_connected_einsum_components(
    program: RichProgram,
) -> tuple[_ConnectedEinsumComponent, ...]:
    """Find einsum components by walking from the program output to its inputs."""
    components: list[_ConnectedEinsumComponent] = []
    visited: set[int] = set()
    pending_ssa_ids = [program.output_ssa]

    # Walk the complete program from its output towards the inputs. Non-einsum
    # operations separate components, but their arguments may lead to more components.
    while pending_ssa_ids:
        ssa_id = pending_ssa_ids.pop()
        if ssa_id < program.n_inputs:
            continue

        instruction_index = ssa_id - program.n_inputs
        if instruction_index in visited:
            continue

        instruction = program.instructions[instruction_index]
        if not isinstance(instruction.operator, OperatorEinsum):
            visited.add(instruction_index)
            pending_ssa_ids.extend(instruction.argument_ssa_ids)
            continue

        sink_op_index = instruction_index
        _sink_inputs, sink_output = parse_format_string(
            instruction.operator.format_string
        )
        label_allocator = _EinsumLabelAllocator()

        # Every queued einsum carries the component-wide labels that its output
        # must use. This propagates consistent labels from consumers to producers.
        relabeled_output = "".join(label_allocator.new_label() for _ in sink_output)
        component: set[int] = set()
        boundary_arguments: list[int] = []
        boundary_strings: list[str] = []
        pending_einsum_instructions = [(instruction_index, relabeled_output)]
        while pending_einsum_instructions:
            einsum_op_index, expected_output = pending_einsum_instructions.pop()
            if einsum_op_index in visited:
                continue

            visited.add(einsum_op_index)
            component.add(einsum_op_index)
            einsum_instruction = program.instructions[einsum_op_index]
            assert isinstance(einsum_instruction.operator, OperatorEinsum)
            input_strings, output_string = parse_format_string(
                einsum_instruction.operator.format_string
            )
            label_map = dict(zip(output_string, expected_output, strict=True))

            for argument_ssa_id, input_string in zip(
                einsum_instruction.argument_ssa_ids, input_strings, strict=True
            ):
                # Preserve labels shared with the output and allocate labels for
                # contraction indices that are first encountered at this operation.
                relabeled_input_labels: list[str] = []
                for label in input_string:
                    if label not in label_map:
                        label_map[label] = label_allocator.new_label()
                    relabeled_input_labels.append(label_map[label])
                relabeled_input = "".join(relabeled_input_labels)

                if argument_ssa_id >= program.n_inputs:
                    argument_instruction_index = argument_ssa_id - program.n_inputs
                    # A uniquely consumed einsum producer belongs to this component.
                    # Everything else is a boundary input and a new traversal root.
                    if (
                        isinstance(
                            program.instructions[argument_instruction_index].operator,
                            OperatorEinsum,
                        )
                        and len(program.consumers_of_ssa_id[argument_ssa_id]) == 1
                    ):
                        pending_einsum_instructions.append(
                            (argument_instruction_index, relabeled_input)
                        )
                        continue

                boundary_arguments.append(argument_ssa_id)
                boundary_strings.append(relabeled_input)
                pending_ssa_ids.append(argument_ssa_id)

        if len(component) >= 2:
            components.append(
                _ConnectedEinsumComponent(
                    op_indices=frozenset(component),
                    sink_op_index=sink_op_index,
                    boundary_arguments=tuple(boundary_arguments),
                    format_string=(f"{','.join(boundary_strings)}->{relabeled_output}"),
                )
            )

    return tuple(components)


class OptimizeContractionPaths(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram:
        # Compute a replacement contraction path for each extracted component.
        blocks: dict[
            int,
            tuple[frozenset[int], tuple[int, ...], tuple[RichInstruction, ...]],
        ] = {}
        for component in extract_connected_einsum_components(program):
            if len(component.boundary_arguments) < 2:
                continue

            boundary_shapes = tuple(
                program.shapes[argument] for argument in component.boundary_arguments
            )

            from sesum import sr

            try:
                path, _flops, _size = sr.compute_path(
                    component.format_string,
                    *boundary_shapes,
                    minimize="size",
                    is_linear=False,
                    max_repeats=128,
                    skops_alpha=10,
                )
            except (RuntimeError, ValueError):
                continue

            planned_instructions = tuple(
                RichInstruction(
                    operator=OperatorEinsum(step_format),
                    argument_ssa_ids=(first, second),
                )
                for first, second, step_format in to_annotated_ssa_path(
                    component.format_string,
                    path,
                    prefer_ascii=True,
                )
            )
            blocks[component.sink_op_index] = (
                component.op_indices,
                component.boundary_arguments,
                planned_instructions,
            )

        if not blocks:
            return program

        # Operations in a planned component are omitted from the old program and
        # replaced by the new pairwise contractions when its sink is reached.
        block_ops = {
            op_index
            for component, _arguments, _instructions in blocks.values()
            for op_index in component
        }
        tensor_map = {input_id: input_id for input_id in range(program.n_inputs)}
        instructions: list[RichInstruction] = []
        shapes = program.shapes[: program.n_inputs]
        tensor_formats = program.tensor_formats[: program.n_inputs]

        # Rebuild in topological order while mapping every old SSA ID to the SSA ID
        # of its value in the rewritten program.
        for op_index, instruction in enumerate(program.instructions):
            result_id = program.n_inputs + op_index
            block = blocks.get(op_index)

            if block is not None:
                # Planned paths use local IDs: boundary inputs first, followed by
                # intermediate results. Translate those IDs as instructions are added.
                _component, boundary_arguments, planned_instructions = block
                local_tensor_map = {
                    local_id: tensor_map[argument]
                    for local_id, argument in enumerate(boundary_arguments)
                }
                block_format = program.tensor_formats[result_id]
                for step_index, planned_instruction in enumerate(planned_instructions):
                    assert isinstance(planned_instruction.operator, OperatorEinsum)
                    mapped_arguments = tuple(
                        local_tensor_map[argument]
                        for argument in planned_instruction.argument_ssa_ids
                    )
                    argument_map = dict(
                        zip(
                            planned_instruction.argument_ssa_ids,
                            mapped_arguments,
                            strict=True,
                        )
                    )
                    mapped_instruction = _map_instruction_arguments(
                        planned_instruction, argument_map.__getitem__
                    )
                    planned_result_id = program.n_inputs + len(instructions)
                    instructions.append(mapped_instruction)
                    shapes.append(
                        infer_einsum_shape(
                            planned_instruction.operator.format_string,
                            [shapes[argument] for argument in mapped_arguments],
                        )
                    )
                    tensor_formats.append(block_format)
                    local_tensor_map[len(boundary_arguments) + step_index] = (
                        planned_result_id
                    )
                tensor_map[result_id] = planned_result_id
                continue

            if op_index in block_ops:
                continue

            argument_map = {
                argument: tensor_map[argument]
                for argument in instruction.argument_ssa_ids
            }
            tensor_map[result_id] = program.n_inputs + len(instructions)
            instructions.append(
                _map_instruction_arguments(instruction, argument_map.__getitem__)
            )
            shapes.append(program.shapes[result_id])
            tensor_formats.append(program.tensor_formats[result_id])

        return RichProgram(
            instructions=instructions,
            n_inputs=program.n_inputs,
            stability_mode=program.stability_mode,
            shapes=shapes,
            tensor_formats=tensor_formats,
            parameter_indices=program.parameter_indices,
        )


class FoldSameShapedOperations(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram: ...


class OptimizeMemoryLayout(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram: ...


def fold_independent_einsums(
    program: RichProgram,
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
    prefer_ascii: bool = False,
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
        if prefer_ascii:
            pairwise_expression = _EinsumLabelAllocator.remap_expression(
                pairwise_expression
            )

        annotated_ssa_path.append((first, second, pairwise_expression))
        inputs.append(t3)

    return annotated_ssa_path
