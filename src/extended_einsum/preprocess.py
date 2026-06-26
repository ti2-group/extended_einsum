from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, override

from extended_einsum.language.rich_instruction import RichInstruction, map_instruction_arguments
from extended_einsum.language.rich_operators import (
    OperatorEinsum,
    OperatorSelect,
    OperatorSlice,
    OperatorSoftmax,
    OperatorStack,
    OperatorTake,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import TensorFormat
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
        self._source_to_label = {} if source_to_label is None else source_to_label.copy()
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
    operator: RichOperator
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

    for op_index, _ in enumerate(program.instructions):
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
            program.n_inputs,
        ):
            if len(safe_group) < min_group_size:
                continue
            groups.append(
                OutputDepthOpGroup(
                    depth=signature.depth,
                    operator=signature.operator,
                    operand_shapes=signature.operand_shapes,
                    operand_tensor_formats=signature.operand_tensor_formats,
                    members=safe_group,
                )
            )

    return tuple(sorted(groups, key=_output_depth_group_sort_key))


def _nearest_output_depths(program: RichProgram) -> dict[int, int]:
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
        for argument in instruction.argument_ssa_ids:
            pending.append((argument, depth + 1))

    return depths


def _instruction_result_dependencies(program: RichProgram) -> dict[int, frozenset[int]]:
    dependencies: dict[int, frozenset[int]] = {input_id: frozenset() for input_id in range(program.n_inputs)}

    for op_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + op_index
        result_dependencies: set[int] = set()
        for argument in instruction.argument_ssa_ids:
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
    if isinstance(instruction.operator, OperatorEinsum):
        return _canonicalize_output_depth_einsum(program, op_index, depth)

    canonical_argument_order = _canonical_argument_order_for_operator(
        instruction.operator,
        instruction.argument_ssa_ids,
        program,
    )
    return _make_output_depth_op_group_entry(
        program,
        op_index,
        depth,
        instruction.operator,
        canonical_argument_order,
    )


def _make_output_depth_op_group_entry(
    program: RichProgram,
    op_index: int,
    depth: int,
    operator: RichOperator,
    canonical_argument_order: tuple[int, ...],
) -> tuple[_OutputDepthOpSignature, OutputDepthOpGroupMember]:
    argument_ssa_ids = program.instructions[op_index].argument_ssa_ids
    ordered_arguments = tuple(argument_ssa_ids[index] for index in canonical_argument_order)
    operand_shapes = tuple(program.shapes[argument] for argument in ordered_arguments)
    operand_tensor_formats: tuple[TensorFormat, ...] = tuple(program.tensor_formats[argument] for argument in ordered_arguments)

    return (
        _OutputDepthOpSignature(
            depth=depth,
            operator=operator,
            operand_shapes=operand_shapes,
            operand_tensor_formats=operand_tensor_formats,
        ),
        OutputDepthOpGroupMember(
            op_index=op_index,
            result_id=program.n_inputs + op_index,
            arguments=argument_ssa_ids,
            canonical_argument_order=canonical_argument_order,
        ),
    )


def _canonical_argument_order_for_operator(
    operator: RichOperator,
    arguments: tuple[int, ...],
    program: RichProgram,
) -> tuple[int, ...]:
    if operator.name not in _COMMUTATIVE_OPERATORS:
        return tuple(range(len(arguments)))

    return tuple(
        sorted(
            range(len(arguments)),
            key=lambda index: (
                program.shapes[arguments[index]],
                program.tensor_formats[arguments[index]],
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
    assert isinstance(instruction.operator, OperatorEinsum)
    input_strings, output_string = parse_format_string(instruction.operator.format_string)
    if len(input_strings) != len(instruction.argument_ssa_ids):
        raise ValueError(f"einsum input count {len(input_strings)} must match argument count {len(instruction.argument_ssa_ids)}")

    # Labels alone do not identify indistinguishable operands; include tensor
    # format metadata so canonical permutations keep tensor slots aligned.
    argument_sort_keys = tuple((program.tensor_formats[argument],) for argument in instruction.argument_ssa_ids)
    canonical_format_string, canonical_argument_order = _canonicalize_einsum_string_from_output(
        input_strings,
        output_string,
        argument_sort_keys,
    )
    return _make_output_depth_op_group_entry(
        program,
        op_index,
        depth,
        OperatorEinsum(canonical_format_string),
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
            label_allocator.copy().normalize_subscript(input_strings[argument_position]),
            argument_sort_keys[argument_position],
            argument_position,
        )

    canonical_argument_order = tuple(sorted(range(len(input_strings)), key=argument_sort_key))
    normalized_inputs = tuple(label_allocator.normalize_subscript(input_strings[index]) for index in canonical_argument_order)
    return (
        f"{','.join(normalized_inputs)}->{normalized_output}",
        canonical_argument_order,
    )


def _split_dependency_safe_op_groups(
    members: list[OutputDepthOpGroupMember],
    result_dependencies: dict[int, frozenset[int]],
    n_inputs: int,
) -> list[tuple[OutputDepthOpGroupMember, ...]]:
    groups: list[list[OutputDepthOpGroupMember]] = []
    for member in sorted(members, key=lambda item: item.op_index):
        for group in groups:
            if _op_member_can_join_group(member, group, result_dependencies, n_inputs):
                group.append(member)
                break
        else:
            groups.append([member])
    return [tuple(group) for group in groups]


def _op_member_can_join_group(
    member: OutputDepthOpGroupMember,
    group: list[OutputDepthOpGroupMember],
    result_dependencies: dict[int, frozenset[int]],
    n_inputs: int,
) -> bool:
    member_result_arguments = _member_result_arguments(member, n_inputs)
    if len(member_result_arguments) != len(set(member_result_arguments)):
        return False

    if any(len(existing_result_arguments := _member_result_arguments(existing_member, n_inputs)) != len(set(existing_result_arguments)) for existing_member in group):
        return False

    used_result_arguments = {argument for existing_member in group for argument in _member_result_arguments(existing_member, n_inputs)}
    if used_result_arguments.intersection(member_result_arguments):
        return False

    return all(
        not _op_members_depend_on_each_other(
            member,
            existing_member,
            result_dependencies,
        )
        for existing_member in group
    )


def _member_result_arguments(
    member: OutputDepthOpGroupMember,
    n_inputs: int,
) -> tuple[int, ...]:
    return tuple(argument for argument in member.arguments if argument >= n_inputs)


def _op_members_depend_on_each_other(
    first: OutputDepthOpGroupMember,
    second: OutputDepthOpGroupMember,
    result_dependencies: dict[int, frozenset[int]],
) -> bool:
    first_result_id = first.result_id
    second_result_id = second.result_id
    return second_result_id in result_dependencies[first_result_id] or first_result_id in result_dependencies[second_result_id]


def _output_depth_group_sort_key(group: OutputDepthOpGroup) -> tuple[Any, ...]:
    return (
        group.depth,
        group.operator.name,
        group.operand_shapes,
        tuple(group.operand_tensor_formats),
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
        _sink_inputs, sink_output = parse_format_string(instruction.operator.format_string)
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
            input_strings, output_string = parse_format_string(einsum_instruction.operator.format_string)
            label_map = dict(zip(output_string, expected_output, strict=True))

            for argument_ssa_id, input_string in zip(einsum_instruction.argument_ssa_ids, input_strings, strict=True):
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
                        pending_einsum_instructions.append((argument_instruction_index, relabeled_input))
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

            boundary_shapes = tuple(program.shapes[argument] for argument in component.boundary_arguments)

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
        block_ops = {op_index for component, _arguments, _instructions in blocks.values() for op_index in component}
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
                local_tensor_map = {local_id: tensor_map[argument] for local_id, argument in enumerate(boundary_arguments)}
                block_format = program.tensor_formats[result_id]
                for step_index, planned_instruction in enumerate(planned_instructions):
                    assert isinstance(planned_instruction.operator, OperatorEinsum)
                    mapped_arguments = tuple(local_tensor_map[argument] for argument in planned_instruction.argument_ssa_ids)
                    argument_map = dict(
                        zip(
                            planned_instruction.argument_ssa_ids,
                            mapped_arguments,
                            strict=True,
                        )
                    )
                    mapped_instruction = map_instruction_arguments(planned_instruction, argument_map.__getitem__)
                    planned_result_id = program.n_inputs + len(instructions)
                    instructions.append(mapped_instruction)
                    shapes.append(
                        infer_einsum_shape(
                            planned_instruction.operator.format_string,
                            [shapes[argument] for argument in mapped_arguments],
                        )
                    )
                    tensor_formats.append(block_format)
                    local_tensor_map[len(boundary_arguments) + step_index] = planned_result_id
                tensor_map[result_id] = planned_result_id
                continue

            if op_index in block_ops:
                continue

            argument_map = {argument: tensor_map[argument] for argument in instruction.argument_ssa_ids}
            tensor_map[result_id] = program.n_inputs + len(instructions)
            instructions.append(map_instruction_arguments(instruction, argument_map.__getitem__))
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


_BATCHABLE_OPERATOR_NAMES = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "1/",
        "+",
        "-",
        "*",
        "/",
        "einsum",
        "stack",
        "slice",
        "select",
        "softmax",
    }
)


@dataclass(frozen=True)
class _TensorRef:
    ssa_id: int


@dataclass(frozen=True)
class _BatchElementRef:
    source_ssa_id: int
    index: int


@dataclass(frozen=True)
class FoldSameShapedOperationsResult:
    program: RichProgram
    batched_result_orders: tuple[tuple[int, ...], ...]
    non_parameter_stack_orders: tuple[tuple[int, ...], ...]


_TensorValueRef = _TensorRef | _BatchElementRef
_ScheduledOutputDepthEvent = int | OutputDepthOpGroup


class FoldSameShapedOperations(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram:
        return FoldSameShapedOperations.apply_with_metadata(program).program

    @staticmethod
    def apply_with_metadata(program: RichProgram) -> FoldSameShapedOperationsResult:
        groups = tuple(group for group in group_identical_ops_by_output_depth(program) if _operator_can_be_batched(group.operator))
        if not groups:
            return FoldSameShapedOperationsResult(
                program=program,
                batched_result_orders=(),
                non_parameter_stack_orders=(),
            )

        events = _schedule_output_depth_group_events(program, groups)
        if not any(isinstance(event, OutputDepthOpGroup) for event in events):
            return FoldSameShapedOperationsResult(
                program=program,
                batched_result_orders=(),
                non_parameter_stack_orders=(),
            )

        ordered_events = _order_output_depth_group_events(events, program)
        return _rewrite_output_depth_group_events(program, ordered_events)


def _operator_can_be_batched(operator: RichOperator) -> bool:
    return operator.name in _BATCHABLE_OPERATOR_NAMES and not isinstance(operator, OperatorTake)


def _schedule_output_depth_group_events(
    program: RichProgram,
    groups: Sequence[OutputDepthOpGroup],
) -> tuple[_ScheduledOutputDepthEvent, ...]:
    group_index_by_op_index: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for member in group.members:
            group_index_by_op_index[member.op_index] = group_index

    remaining = set(range(len(program.instructions)))
    available = set(range(program.n_inputs))
    disabled_groups: set[int] = set()
    events: list[_ScheduledOutputDepthEvent] = []

    while remaining:
        ready = [op_index for op_index in sorted(remaining) if all(argument in available for argument in program.instructions[op_index].argument_ssa_ids)]
        if not ready:
            raise ValueError("program contains instructions whose inputs are unavailable")

        progress = False
        for op_index in ready:
            group_index = group_index_by_op_index.get(op_index)
            if group_index is None or group_index in disabled_groups:
                events.append(op_index)
                remaining.remove(op_index)
                available.add(program.n_inputs + op_index)
                progress = True
                break

            group = groups[group_index]
            if all(member.op_index in ready for member in group.members):
                events.append(group)
                for member in group.members:
                    remaining.remove(member.op_index)
                    available.add(member.result_id)
                progress = True
                break

        if progress:
            continue

        # A partial group is ready but cannot be emitted as a whole. Keep the
        # program moving by leaving that group unbatched.
        disabled_groups.add(group_index_by_op_index[ready[0]])

    return tuple(events)


def _order_output_depth_group_events(
    events: tuple[_ScheduledOutputDepthEvent, ...],
    program: RichProgram,
) -> tuple[_ScheduledOutputDepthEvent, ...]:
    consumers = _result_consumers_with_positions(program)
    event_index_by_op_index: dict[int, int] = {}
    for event_index, event in enumerate(events):
        if isinstance(event, int):
            event_index_by_op_index[event] = event_index
            continue
        for member in event.members:
            event_index_by_op_index[member.op_index] = event_index

    ordered_groups: dict[int, OutputDepthOpGroup] = {}
    for event_index in reversed(range(len(events))):
        event = events[event_index]
        if isinstance(event, int):
            continue

        ordered_members = tuple(
            sorted(
                event.members,
                key=lambda member: _group_member_future_order_key(
                    member,
                    events,
                    consumers,
                    event_index_by_op_index,
                    ordered_groups,
                ),
            )
        )
        ordered_groups[event_index] = replace(event, members=ordered_members)

    return tuple(ordered_groups[event_index] if event_index in ordered_groups else event for event_index, event in enumerate(events))


def _result_consumers_with_positions(
    program: RichProgram,
) -> dict[int, list[tuple[int, int]]]:
    consumers: dict[int, list[tuple[int, int]]] = {}
    for op_index, instruction in enumerate(program.instructions):
        for argument_position, argument in enumerate(instruction.argument_ssa_ids):
            consumers.setdefault(argument, []).append((op_index, argument_position))
    return consumers


def _group_member_future_order_key(
    member: OutputDepthOpGroupMember,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    consumers: dict[int, list[tuple[int, int]]],
    event_index_by_op_index: dict[int, int],
    ordered_groups: dict[int, OutputDepthOpGroup],
) -> tuple[int, ...]:
    order_keys: list[tuple[int, ...]] = []
    for consumer_op_index, argument_position in consumers.get(member.result_id, []):
        consumer_event_index = event_index_by_op_index[consumer_op_index]
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
        consumer_position_by_op_index = {consumer_member.op_index: position for position, consumer_member in enumerate(ordered_group.members)}
        consumer_member = ordered_group.members[consumer_position_by_op_index[consumer_op_index]]
        canonical_position = _canonical_operand_position(
            consumer_member,
            argument_position,
        )
        order_keys.append(
            (
                consumer_event_index,
                0,
                canonical_position,
                consumer_position_by_op_index[consumer_op_index],
                argument_position,
            )
        )

    if not order_keys:
        return (len(events), member.op_index)
    return min(order_keys)


def _canonical_operand_position(
    member: OutputDepthOpGroupMember,
    original_position: int,
) -> int:
    for canonical_position, member_original_position in enumerate(member.canonical_argument_order):
        if member_original_position == original_position:
            return canonical_position
    raise ValueError("argument position is not part of the member's canonical argument order")


def _rewrite_output_depth_group_events(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> FoldSameShapedOperationsResult:
    instructions: list[RichInstruction] = []
    shapes = program.shapes[: program.n_inputs]
    tensor_formats = program.tensor_formats[: program.n_inputs]
    tensor_map: dict[int, _TensorValueRef] = {input_id: _TensorRef(input_id) for input_id in range(program.n_inputs)}
    materialized_elements: dict[_BatchElementRef, int] = {}
    batched_result_orders: list[tuple[int, ...]] = []
    non_parameter_stack_orders: list[tuple[int, ...]] = []

    def append_instruction(
        instruction: RichInstruction,
        *,
        output_shape: Shape | None = None,
        output_format: TensorFormat | None = None,
    ) -> int:
        result_id = program.n_inputs + len(instructions)
        instructions.append(instruction)
        argument_shapes = [shapes[argument] for argument in instruction.argument_ssa_ids]
        shapes.append(instruction.operator.propagate_shapes(argument_shapes) if output_shape is None else output_shape)
        if output_format is None:
            output_format = _infer_generated_tensor_format(
                instruction.operator,
                [tensor_formats[argument] for argument in instruction.argument_ssa_ids],
            )
        tensor_formats.append(output_format)
        return result_id

    def ref_shape(ref: _TensorValueRef) -> Shape:
        if isinstance(ref, _TensorRef):
            return shapes[ref.ssa_id]
        source_shape = shapes[ref.source_ssa_id]
        if not source_shape:
            raise ValueError("batched tensor has no batch axis")
        return source_shape[1:]

    def ref_format(ref: _TensorValueRef) -> TensorFormat:
        if isinstance(ref, _TensorRef):
            return tensor_formats[ref.ssa_id]
        return tensor_formats[ref.source_ssa_id]

    def materialize_ref(ref: _TensorValueRef) -> int:
        if isinstance(ref, _TensorRef):
            return ref.ssa_id

        cached = materialized_elements.get(ref)
        if cached is not None:
            return cached

        result_id = append_instruction(
            RichInstruction(
                operator=OperatorSelect(axis=0, index=ref.index),
                argument_ssa_ids=(ref.source_ssa_id,),
            ),
            output_shape=ref_shape(ref),
            output_format=ref_format(ref),
        )
        materialized_elements[ref] = result_id
        return result_id

    def materialize_batch(refs: tuple[_TensorValueRef, ...]) -> int:
        if not refs:
            raise ValueError("cannot materialize an empty batch")

        first_ref = refs[0]
        if isinstance(first_ref, _BatchElementRef) and all(isinstance(ref, _BatchElementRef) and ref.source_ssa_id == first_ref.source_ssa_id for ref in refs):
            indices = tuple(ref.index for ref in refs)
            contiguous = _contiguous_range(indices)
            if contiguous is not None:
                start, stop = contiguous
                if start == 0 and shapes[first_ref.source_ssa_id][0] == stop:
                    return first_ref.source_ssa_id
                return append_instruction(
                    RichInstruction(
                        operator=OperatorSlice(start=start, stop=stop, axis=0),
                        argument_ssa_ids=(first_ref.source_ssa_id,),
                    ),
                    output_format=ref_format(first_ref),
                )

        input_order = _input_stack_order(refs, program.n_inputs)
        if input_order is not None and any(input_id not in program.parameter_indices for input_id in input_order):
            non_parameter_stack_orders.append(input_order)

        materialized_arguments = tuple(materialize_ref(ref) for ref in refs)
        return append_instruction(
            RichInstruction(
                operator=OperatorStack(axis=0),
                argument_ssa_ids=materialized_arguments,
            ),
            output_format=ref_format(first_ref),
        )

    for event in events:
        if not isinstance(event, int):
            grouped_refs = _group_refs_by_canonical_operand(event, tensor_map)
            batched_arguments = tuple(materialize_batch(refs) for refs in grouped_refs)
            batched_operator = _make_batched_operator(event.operator)
            batched_result_id = append_instruction(
                RichInstruction(
                    operator=batched_operator,
                    argument_ssa_ids=batched_arguments,
                ),
                output_format=program.tensor_formats[event.members[0].result_id],
            )
            batched_result_orders.append(tuple(member.result_id for member in event.members))
            for index, member in enumerate(event.members):
                tensor_map[member.result_id] = _BatchElementRef(batched_result_id, index)
            continue

        instruction = program.instructions[event]
        result_id = program.n_inputs + event
        mapped_arguments = tuple(materialize_ref(tensor_map[argument]) for argument in instruction.argument_ssa_ids)
        mapped_result_id = append_instruction(
            RichInstruction(
                operator=instruction.operator,
                argument_ssa_ids=mapped_arguments,
            ),
            output_shape=program.shapes[result_id],
            output_format=program.tensor_formats[result_id],
        )
        tensor_map[result_id] = _TensorRef(mapped_result_id)

    output_id = materialize_ref(tensor_map[program.output_ssa])
    if not _is_last_result(output_id, program.n_inputs, instructions):
        output_id = _copy_to_last_result(
            output_id,
            append_instruction,
            shapes,
            tensor_formats,
        )
    if not _is_last_result(output_id, program.n_inputs, instructions):
        raise RuntimeError("rewritten program output was not materialized as the last result")

    rewritten_program = RichProgram(
        instructions=instructions,
        n_inputs=program.n_inputs,
        stability_mode=program.stability_mode,
        shapes=shapes,
        tensor_formats=tensor_formats,
        parameter_indices=program.parameter_indices,
    )
    return FoldSameShapedOperationsResult(
        program=rewritten_program,
        batched_result_orders=tuple(batched_result_orders),
        non_parameter_stack_orders=tuple(non_parameter_stack_orders),
    )


def _group_refs_by_canonical_operand(
    group: OutputDepthOpGroup,
    tensor_map: dict[int, _TensorValueRef],
) -> list[tuple[_TensorValueRef, ...]]:
    operand_count = len(group.members[0].canonical_argument_order)
    grouped: list[list[_TensorValueRef]] = [[] for _ in range(operand_count)]
    for member in group.members:
        for canonical_position, original_position in enumerate(member.canonical_argument_order):
            grouped[canonical_position].append(tensor_map[member.arguments[original_position]])
    return [tuple(refs) for refs in grouped]


def _input_stack_order(
    refs: tuple[_TensorValueRef, ...],
    n_inputs: int,
) -> tuple[int, ...] | None:
    input_ids: list[int] = []
    for ref in refs:
        if not isinstance(ref, _TensorRef) or ref.ssa_id >= n_inputs:
            return None
        input_ids.append(ref.ssa_id)
    return tuple(input_ids)


def _contiguous_range(indices: tuple[int, ...]) -> tuple[int, int] | None:
    if not indices:
        return None
    start = indices[0]
    for offset, index in enumerate(indices):
        if index != start + offset:
            return None
    return start, start + len(indices)


def _make_batched_operator(operator: RichOperator) -> RichOperator:
    match operator:
        case OperatorEinsum(format_string):
            input_strings, output_string = parse_format_string(format_string)
            return OperatorEinsum(_batched_einsum_format(input_strings, output_string))
        case OperatorStack(axis):
            return OperatorStack(axis + 1)
        case OperatorSlice(start, stop, axis):
            return OperatorSlice(start, stop, axis + 1)
        case OperatorSelect(axis, index):
            return OperatorSelect(axis + 1, index)
        case OperatorSoftmax(axis):
            return OperatorSoftmax(axis + 1)
        case OperatorTake(_):
            raise ValueError("take cannot be batched with the current primitive")
        case _:
            return operator


def _batched_einsum_format(input_strings: Sequence[str], output_string: str) -> str:
    batch_label = _new_unused_einsum_label(f"{''.join(input_strings)}{output_string}")
    batched_inputs = ",".join(f"{batch_label}{input_string}" for input_string in input_strings)
    return f"{batched_inputs}->{batch_label}{output_string}"


def _new_unused_einsum_label(used_labels: str) -> str:
    used = set(used_labels)
    for label in _LABELS:
        if label not in used:
            return label

    extended_index = 0
    while True:
        label = chr(0x100 + extended_index)
        if label not in used:
            return label
        extended_index += 1


def _infer_generated_tensor_format(
    operator: RichOperator,
    argument_formats: Sequence[TensorFormat],
) -> TensorFormat:
    if not argument_formats:
        raise ValueError(f"operator {operator.name} has no arguments")
    first_format = argument_formats[0]
    if isinstance(operator, (OperatorStack, OperatorEinsum)) and any(format != first_format for format in argument_formats[1:]):
        raise ValueError(f"operator {operator.name} requires matching tensor formats")
    return first_format


def _is_last_result(
    ssa_id: int,
    n_inputs: int,
    instructions: Sequence[RichInstruction],
) -> bool:
    return ssa_id == n_inputs + len(instructions) - 1


def _copy_to_last_result(
    ssa_id: int,
    append_instruction: Any,
    shapes: Sequence[Shape],
    tensor_formats: Sequence[TensorFormat],
) -> int:
    stack_result = append_instruction(
        RichInstruction(
            operator=OperatorStack(axis=0),
            argument_ssa_ids=(ssa_id,),
        ),
        output_format=tensor_formats[ssa_id],
    )
    return append_instruction(
        RichInstruction(
            operator=OperatorSelect(axis=0, index=0),
            argument_ssa_ids=(stack_result,),
        ),
        output_shape=shapes[ssa_id],
        output_format=tensor_formats[ssa_id],
    )


class OptimizeMemoryLayout(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram:
        return program  # TODO


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
            pairwise_expression = _EinsumLabelAllocator.remap_expression(pairwise_expression)

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
                raise RuntimeError(f"ERROR: {expression} cannot be converted to ASCII, it is too large.")
            char_mapping[char] = _LABELS[ascii_index]
            ascii_index += 1
        parts.append(char_mapping[char])
    return "".join(parts)
