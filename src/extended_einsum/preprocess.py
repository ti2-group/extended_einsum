from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, override

from extended_einsum.language.rich_instruction import RichInstruction, map_instruction_arguments
from extended_einsum.language.rich_operators import (
    OperatorConcat,
    OperatorEinsum,
    OperatorSelect,
    OperatorSlice,
    OperatorSoftmax,
    OperatorStack,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import TensorFormat
from extended_einsum.shapes import (
    Shape,
    infer_einsum_shape,
)
from extended_einsum.utils import is_contraction_free_einsum, parse_format_string

_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_COMMUTATIVE_OPERATORS = frozenset({"+", "*"})
_POINTWISE_OPERATOR_NAMES = frozenset(
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
    }
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
_CONTRACTION_PATH_ATTEMPTS = 8
_GROUP_ORDER_BEAM_WIDTH = 16


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
    """Group live, batchable, canonically identical ops by nearest output depth."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")

    output_depths = _nearest_output_depths(program)
    result_dependencies = _instruction_result_dependencies(program)
    groups: list[OutputDepthOpGroup] = []

    grouped_input_pointwise_ops = _append_input_pointwise_groups(
        program,
        output_depths,
        result_dependencies,
        groups,
        min_group_size,
    )

    buckets: dict[_OutputDepthOpSignature, list[OutputDepthOpGroupMember]] = {}

    for op_index, _ in enumerate(program.instructions):
        if op_index in grouped_input_pointwise_ops:
            continue
        if program.instructions[op_index].operator.name not in _BATCHABLE_OPERATOR_NAMES:
            continue
        result_id = program.n_inputs + op_index
        depth = output_depths.get(result_id)
        if depth is None:
            continue

        signature, member = _canonicalize_output_depth_op(program, op_index, depth)
        buckets.setdefault(signature, []).append(member)

    for signature, members in buckets.items():
        for safe_group in _split_dependency_safe_op_groups(
            members,
            result_dependencies,
        ):
            for source_safe_group in _split_operand_source_safe_op_groups(program, safe_group):
                if len(source_safe_group) < min_group_size:
                    continue
                groups.append(
                    OutputDepthOpGroup(
                        depth=signature.depth,
                        operator=signature.operator,
                        operand_shapes=signature.operand_shapes,
                        operand_tensor_formats=signature.operand_tensor_formats,
                        members=source_safe_group,
                    )
                )

    return _refine_groups_by_materialization_sources(
        program,
        tuple(groups),
        min_group_size=min_group_size,
    )


def _append_input_pointwise_groups(
    program: RichProgram,
    output_depths: dict[int, int],
    result_dependencies: dict[int, frozenset[int]],
    groups: list[OutputDepthOpGroup],
    min_group_size: int,
) -> frozenset[int]:
    effective_min_group_size = max(2, min_group_size)
    buckets: dict[_OutputDepthOpSignature, list[OutputDepthOpGroupMember]] = {}

    for op_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + op_index
        if result_id not in output_depths:
            continue
        if not _is_input_pointwise_instruction(program, instruction):
            continue

        signature, member = _canonicalize_output_depth_op(program, op_index, depth=-1)
        buckets.setdefault(signature, []).append(member)

    grouped_op_indices: set[int] = set()
    for signature, members in buckets.items():
        for safe_group in _split_dependency_safe_op_groups(
            members,
            result_dependencies,
        ):
            if len(safe_group) < effective_min_group_size:
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
            grouped_op_indices.update(member.op_index for member in safe_group)

    return frozenset(grouped_op_indices)


def _is_input_pointwise_instruction(
    program: RichProgram,
    instruction: RichInstruction,
) -> bool:
    return instruction.operator.name in _POINTWISE_OPERATOR_NAMES and all(argument < program.n_inputs for argument in instruction.argument_ssa_ids)


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
) -> list[tuple[OutputDepthOpGroupMember, ...]]:
    groups: list[list[OutputDepthOpGroupMember]] = []
    for member in sorted(members, key=lambda item: item.op_index):
        for group in groups:
            # A batched operator cannot consume the same value twice in one operand batch.
            if _op_member_can_join_group(member, group, result_dependencies):
                group.append(member)
                break
        else:
            groups.append([member])
    return [tuple(group) for group in groups]


def _split_operand_source_safe_op_groups(
    program: RichProgram,
    members: tuple[OutputDepthOpGroupMember, ...],
) -> tuple[tuple[OutputDepthOpGroupMember, ...], ...]:
    buckets: dict[tuple[str, ...], list[OutputDepthOpGroupMember]] = {}
    for member in members:
        source_signature = tuple(
            _operand_source_kind(
                program,
                member.arguments[original_position],
            )
            for original_position in member.canonical_argument_order
        )
        buckets.setdefault(source_signature, []).append(member)
    return tuple(tuple(source_members) for source_members in buckets.values())


def _operand_source_kind(
    program: RichProgram,
    argument: int,
) -> str:
    if argument < program.n_inputs:
        return "input-access"

    instruction = program.instructions[argument - program.n_inputs]
    if isinstance(instruction.operator, (OperatorSelect, OperatorSlice)) and instruction.argument_ssa_ids[0] < program.n_inputs:
        return "input-access"

    return "result"


def _refine_groups_by_materialization_sources(
    program: RichProgram,
    groups: tuple[OutputDepthOpGroup, ...],
    *,
    min_group_size: int,
) -> tuple[OutputDepthOpGroup, ...]:
    refined_groups = tuple(sorted(groups, key=_output_depth_group_sort_key))
    stack_arguments_by_result_id = _axis0_stack_arguments_by_result_id(program)
    while True:
        group_index_by_result_id = {
            member.result_id: group_index
            for group_index, group in enumerate(refined_groups)
            for member in group.members
        }
        next_groups: list[OutputDepthOpGroup] = []
        changed = False

        for group in refined_groups:
            buckets: dict[tuple[tuple[Any, ...], ...], list[OutputDepthOpGroupMember]] = {}
            group_result_ids = frozenset(member.result_id for member in group.members)
            for member in group.members:
                source_signature = tuple(
                    (
                        _argument_materialization_group_source_key(
                            program,
                            member.arguments[original_position],
                            group_index_by_result_id,
                        ),
                        _stack_boundary_source_key(
                            member.arguments[original_position],
                            group_result_ids,
                            stack_arguments_by_result_id,
                        ),
                    )
                    for original_position in member.canonical_argument_order
                )
                result_stack_boundary = _stack_boundary_source_key(
                    member.result_id,
                    group_result_ids,
                    stack_arguments_by_result_id,
                )
                source_signature = (*source_signature, (("result-stack-boundary",), result_stack_boundary))
                buckets.setdefault(source_signature, []).append(member)

            kept_member_groups = tuple(
                tuple(members) for members in buckets.values() if len(members) >= min_group_size
            )
            if kept_member_groups != (group.members,):
                changed = True
            next_groups.extend(replace(group, members=members) for members in kept_member_groups)

        next_refined_groups = tuple(sorted(next_groups, key=_output_depth_group_sort_key))
        if not changed:
            return next_refined_groups
        refined_groups = next_refined_groups


def _argument_materialization_group_source_key(
    program: RichProgram,
    argument: int,
    group_index_by_result_id: dict[int, int],
) -> tuple[Any, ...]:
    direct_input_source = _direct_input_access_source_key(program, argument)
    if direct_input_source is not None:
        return direct_input_source

    group_index = group_index_by_result_id.get(argument)
    if group_index is not None:
        return ("producer-group", group_index)

    return ("ssa", argument)


def _direct_input_access_source_key(
    program: RichProgram,
    argument: int,
) -> tuple[Any, ...] | None:
    if argument < program.n_inputs:
        return ("input",)

    instruction = program.instructions[argument - program.n_inputs]
    if isinstance(instruction.operator, (OperatorSelect, OperatorSlice)) and instruction.argument_ssa_ids[0] < program.n_inputs:
        return ("input-view", instruction.argument_ssa_ids[0])

    return None


def _axis0_stack_arguments_by_result_id(
    program: RichProgram,
) -> dict[int, tuple[tuple[int, tuple[int, ...]], ...]]:
    stack_arguments_by_result_id: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for op_index, instruction in enumerate(program.instructions):
        if not isinstance(instruction.operator, OperatorStack) or instruction.operator.axis != 0:
            continue
        stack_arguments = tuple(instruction.argument_ssa_ids)
        for argument in stack_arguments:
            stack_arguments_by_result_id.setdefault(argument, []).append((op_index, stack_arguments))
    return {result_id: tuple(stack_arguments) for result_id, stack_arguments in stack_arguments_by_result_id.items()}


def _stack_boundary_source_key(
    result_id: int,
    group_result_ids: frozenset[int],
    stack_arguments_by_result_id: dict[int, tuple[tuple[int, tuple[int, ...]], ...]],
) -> tuple[Any, ...]:
    stack_arguments = stack_arguments_by_result_id.get(result_id, ())
    if not stack_arguments:
        return ("none",)

    keys: list[tuple[Any, ...]] = []
    for stack_op_index, arguments in stack_arguments:
        argument_ids = frozenset(arguments)
        if argument_ids == group_result_ids and len(arguments) == len(group_result_ids):
            keys.append(("complete", stack_op_index))
            continue
        keys.append(("partial", stack_op_index, result_id))
    return tuple(keys)


def _op_member_can_join_group(
    member: OutputDepthOpGroupMember,
    group: list[OutputDepthOpGroupMember],
    result_dependencies: dict[int, frozenset[int]],
) -> bool:

    used_result_arguments = {argument for existing_member in group for argument in existing_member.arguments}
    if used_result_arguments.intersection(member.arguments):
        return False

    return all(
        not _op_members_depend_on_each_other(
            member,
            existing_member,
            result_dependencies,
        )
        for existing_member in group
    )


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


def _is_outer_product_einsum(format_string: str) -> bool:
    """Whether a contraction-free einsum expands at least one operand with new output axes."""

    input_strings, output_string = parse_format_string(format_string)
    output_labels = set(output_string)
    return is_contraction_free_einsum(format_string) and any(set(input_string) != output_labels for input_string in input_strings)


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
                    argument_operator = program.instructions[argument_instruction_index].operator
                    fusable_stable_outer_product = (
                        program.stability_mode != "unstable"
                        and isinstance(argument_operator, OperatorEinsum)
                        and _is_outer_product_einsum(argument_operator.format_string)
                    )
                    cheap_stable_product = (
                        program.stability_mode != "unstable"
                        and isinstance(argument_operator, OperatorEinsum)
                        and is_contraction_free_einsum(argument_operator.format_string)
                        and not fusable_stable_outer_product
                    )
                    # A uniquely consumed einsum producer belongs to this component.
                    # Stable contraction-free products are boundaries unless
                    # they are outer products that can be fused with their
                    # later reduction.
                    if (
                        isinstance(argument_operator, OperatorEinsum)
                        and len(program.consumers_of_ssa_id[argument_ssa_id]) == 1
                        and not cheap_stable_product
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
        result_depths = _ssa_depths_from_inputs(program)
        blocks: dict[
            int,
            tuple[frozenset[int], tuple[int, ...], tuple[RichInstruction, ...]],
        ] = {}
        for component in extract_connected_einsum_components(program):
            if len(component.boundary_arguments) < 2:
                continue

            fuse_stable_outer_product = program.stability_mode != "unstable" and any(
                isinstance(program.instructions[op_index].operator, OperatorEinsum)
                and _is_outer_product_einsum(program.instructions[op_index].operator.format_string)
                for op_index in component.op_indices
            )
            if fuse_stable_outer_product and len(component.op_indices) == 2:
                planned_instructions = (
                    RichInstruction(
                        operator=OperatorEinsum(component.format_string),
                        argument_ssa_ids=tuple(range(len(component.boundary_arguments))),
                    ),
                )
            elif fuse_stable_outer_product:
                # Keep larger connected subgraphs unchanged. Fusing an entire
                # hierarchy would create a high-arity einsum instead of one
                # Tucker kernel per outer-product/reduction pair.
                continue
            else:
                boundary_shapes = tuple(program.shapes[argument] for argument in component.boundary_arguments)
                planned_instructions = _compute_depth_preserving_contraction_plan(
                    component,
                    boundary_shapes,
                    result_depths,
                    program.n_inputs,
                    prioritize_output_labels=program.stability_mode in {"scaled_min", "scaled_sum"},
                )
                if planned_instructions is None:
                    continue

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


def _ssa_depths_from_inputs(program: RichProgram) -> dict[int, int]:
    depths = {input_id: 0 for input_id in range(program.n_inputs)}
    for op_index, instruction in enumerate(program.instructions):
        result_id = program.n_inputs + op_index
        depths[result_id] = max((depths[argument] for argument in instruction.argument_ssa_ids), default=-1) + 1
    return depths


def _compute_depth_preserving_contraction_plan(
    component: _ConnectedEinsumComponent,
    boundary_shapes: tuple[Shape, ...],
    result_depths: dict[int, int],
    n_inputs: int,
    *,
    prioritize_output_labels: bool = False,
) -> tuple[RichInstruction, ...] | None:
    from sesum import sr

    original_sink_depth = result_depths[n_inputs + component.sink_op_index]
    for seed in range(_CONTRACTION_PATH_ATTEMPTS):
        try:
            path, _flops, _size = sr.compute_path(
                component.format_string,
                *boundary_shapes,
                seed=seed,
                minimize="size",
                is_linear=False,
                max_repeats=128,
                skops_alpha=10,
            )
            planned_instructions = tuple(
                RichInstruction(
                    operator=OperatorEinsum(step_format),
                    argument_ssa_ids=(first, second),
                )
                for first, second, step_format in to_annotated_ssa_path(
                    component.format_string,
                    path,
                    prefer_ascii=True,
                    prioritize_output_labels=prioritize_output_labels,
                )
            )
        except (RuntimeError, ValueError):
            continue

        planned_depth = _planned_component_output_depth(
            component,
            planned_instructions,
            result_depths,
        )
        if planned_depth <= original_sink_depth:
            return planned_instructions

    return None


def _planned_component_output_depth(
    component: _ConnectedEinsumComponent,
    planned_instructions: tuple[RichInstruction, ...],
    result_depths: dict[int, int],
) -> int:
    local_depths = {local_id: result_depths[argument] for local_id, argument in enumerate(component.boundary_arguments)}
    for step_index, instruction in enumerate(planned_instructions):
        local_result_id = len(component.boundary_arguments) + step_index
        local_depths[local_result_id] = max(local_depths[argument] for argument in instruction.argument_ssa_ids) + 1
    return local_depths[len(component.boundary_arguments) + len(planned_instructions) - 1]

@dataclass(frozen=True)
class _TensorRef:
    ssa_id: int


@dataclass(frozen=True)
class _InputRef:
    ssa_id: int
    original_input_id: int


@dataclass(frozen=True)
class _BatchElementRef:
    source_ssa_id: int
    index: int
    original_input_id: int | None = None
    original_ssa_id: int | None = None


@dataclass(frozen=True)
class _RewriteInputPlan:
    n_inputs: int
    shapes: list[Shape]
    tensor_formats: list[TensorFormat]
    parameter_indices: frozenset[int]
    tensor_map: dict[int, _TensorValueRef]
    parameter_stack_input_by_order: dict[tuple[int, ...], int]
    input_axis0_orders: dict[int, tuple[int, ...]]
    input_axis0_position_by_index: dict[int, dict[int, int]]


@dataclass(frozen=True)
class FoldSameShapedOperationsResult:
    program: RichProgram
    batched_result_orders: tuple[tuple[int, ...], ...]
    non_parameter_stack_orders: tuple[tuple[int, ...], ...]
    parameter_stack_orders: tuple[tuple[int, ...], ...]
    input_axis0_orders: dict[int, tuple[int, ...]]
    concatenated_batch_orders: tuple[tuple[int, ...], ...]


_TensorValueRef = _TensorRef | _InputRef | _BatchElementRef
_ScheduledOutputDepthEvent = int | OutputDepthOpGroup


class FoldSameShapedOperations(PreprocessingRoutine):
    @override
    @staticmethod
    def apply(program: RichProgram) -> RichProgram:
        result = FoldSameShapedOperations.apply_with_metadata(program)
        return result.program

    @staticmethod
    def apply_with_metadata(program: RichProgram) -> FoldSameShapedOperationsResult:
        groups = group_identical_ops_by_output_depth(program)
        if not groups:
            return FoldSameShapedOperationsResult(
                program=program,
                batched_result_orders=(),
                non_parameter_stack_orders=(),
                parameter_stack_orders=(),
                input_axis0_orders={},
                concatenated_batch_orders=(),
            )

        events = _topologically_order_output_depth_events(program, groups)
        ordered_events = _order_group_members_for_future_consumers(events, program)
        return _rewrite_output_depth_group_events(program, ordered_events)


def _topologically_order_output_depth_events(
    program: RichProgram,
    groups: Sequence[OutputDepthOpGroup],
) -> tuple[_ScheduledOutputDepthEvent, ...]:
    """Topologically order ungrouped ops and whole op groups as rewrite events.

    This decides where a whole group block is emitted relative to other ops. It
    intentionally does not choose the order of members inside a group.
    """
    group_index_by_op_index: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for member in group.members:
            group_index_by_op_index[member.op_index] = group_index

    block_by_group_index: dict[int, int] = {}
    block_by_op_index: dict[int, int] = {}
    blocks: list[_ScheduledOutputDepthEvent] = []
    block_order_keys: list[tuple[int, int]] = []

    for op_index in range(len(program.instructions)):
        group_index = group_index_by_op_index.get(op_index)
        if group_index is None:
            block_index = len(blocks)
            blocks.append(op_index)
            block_order_keys.append((op_index, op_index))
            block_by_op_index[op_index] = block_index
            continue

        block_index = block_by_group_index.get(group_index)
        if block_index is None:
            group = groups[group_index]
            member_op_indices = tuple(member.op_index for member in group.members)
            block_index = len(blocks)
            block_by_group_index[group_index] = block_index
            blocks.append(group)
            if group.depth < 0:
                block_order_keys.append((-1, min(member_op_indices)))
            else:
                block_order_keys.append((min(member_op_indices), max(member_op_indices)))

        block_by_op_index[op_index] = block_index

    dependencies: list[set[int]] = [set() for _ in blocks]
    for op_index, instruction in enumerate(program.instructions):
        consumer_block_index = block_by_op_index[op_index]
        for argument in instruction.argument_ssa_ids:
            if argument < program.n_inputs:
                continue
            producer_block_index = block_by_op_index[argument - program.n_inputs]
            if producer_block_index == consumer_block_index:
                continue
            dependencies[consumer_block_index].add(producer_block_index)

    events: list[_ScheduledOutputDepthEvent] = []
    remaining_block_indices = set(range(len(blocks)))
    while remaining_block_indices:
        ready_block_index = min(
            (block_index for block_index in remaining_block_indices if not dependencies[block_index]),
            key=block_order_keys.__getitem__,
            default=None,
        )
        if ready_block_index is None:
            raise ValueError("output-depth groups contain cyclic block dependencies")

        events.append(blocks[ready_block_index])
        remaining_block_indices.remove(ready_block_index)
        for block_dependencies in dependencies:
            block_dependencies.discard(ready_block_index)

    return tuple(events)


def _order_group_members_for_future_consumers(
    events: tuple[_ScheduledOutputDepthEvent, ...],
    program: RichProgram,
) -> tuple[_ScheduledOutputDepthEvent, ...]:
    """Sort members inside each group to align batch axes with future consumers."""
    consumers = _result_consumers_with_positions(program)
    event_index_by_op_index: dict[int, int] = {}
    group_position_by_result_id: dict[int, tuple[int, int]] = {}
    for event_index, event in enumerate(events):
        if isinstance(event, int):
            event_index_by_op_index[event] = event_index
            continue
        for position, member in enumerate(event.members):
            event_index_by_op_index[member.op_index] = event_index
            group_position_by_result_id[member.result_id] = (event_index, position)

    ordered_groups: dict[int, OutputDepthOpGroup] = {}
    # Producer group member order determines batch-axis order. Walk backward so
    # consumer groups are ordered before the producers they consume.
    for event_index in reversed(range(len(events))):
        event = events[event_index]
        if isinstance(event, int):
            continue

        candidates = _group_member_order_candidates(
            event,
            program,
            events,
            consumers,
            event_index_by_op_index,
            group_position_by_result_id,
            ordered_groups,
        )
        ordered_members = min(
            candidates,
            key=lambda members: _group_member_order_score(
                event,
                members,
                program,
                events,
                consumers,
                event_index_by_op_index,
                group_position_by_result_id,
                ordered_groups,
            ),
        )
        ordered_groups[event_index] = replace(event, members=ordered_members)

    ordered_events = tuple(ordered_groups[event_index] if event_index in ordered_groups else event for event_index, event in enumerate(events))
    return _optimize_group_member_orders_by_materialization_cost(ordered_events, program)


def _group_member_order_candidates(
    group: OutputDepthOpGroup,
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    consumers: dict[int, list[tuple[int, int]]],
    event_index_by_op_index: dict[int, int],
    group_position_by_result_id: dict[int, tuple[int, int]],
    ordered_groups: dict[int, OutputDepthOpGroup],
) -> tuple[tuple[OutputDepthOpGroupMember, ...], ...]:
    candidates: list[tuple[OutputDepthOpGroupMember, ...]] = []

    def add_candidate(members: Sequence[OutputDepthOpGroupMember]) -> None:
        ordered_members = tuple(members)
        if ordered_members not in candidates:
            candidates.append(ordered_members)

    add_candidate(group.members)
    add_candidate(
        sorted(
            group.members,
            key=lambda member: _group_member_future_order_key(
                member,
                events,
                consumers,
                event_index_by_op_index,
                ordered_groups,
            ),
        )
    )

    operand_count = len(group.members[0].canonical_argument_order)
    for canonical_position in range(operand_count):
        add_candidate(
            sorted(
                group.members,
                key=lambda member, position=canonical_position: (
                    _member_materialization_order_key(
                        program,
                        member,
                        position,
                        group_position_by_result_id,
                    ),
                    _group_member_future_order_key(
                        member,
                        events,
                        consumers,
                        event_index_by_op_index,
                        ordered_groups,
                    ),
                ),
            )
        )
        add_candidate(
            sorted(
                group.members,
                key=lambda member, position=canonical_position: (
                    _member_materialization_source_key(
                        program,
                        member,
                        position,
                        group_position_by_result_id,
                    ),
                    _group_member_future_order_key(
                        member,
                        events,
                        consumers,
                        event_index_by_op_index,
                        ordered_groups,
                    ),
                    _member_materialization_index(
                        program,
                        member,
                        position,
                        group_position_by_result_id,
                    ),
                ),
            )
        )

    add_candidate(
        sorted(
            group.members,
            key=lambda member: (
                tuple(
                    _member_materialization_order_key(
                        program,
                        member,
                        canonical_position,
                        group_position_by_result_id,
                    )
                    for canonical_position in range(operand_count)
                ),
                member.op_index,
            ),
        )
    )
    add_candidate(
        sorted(
            group.members,
            key=lambda member: (
                tuple(
                    _member_materialization_order_key(
                        program,
                        member,
                        canonical_position,
                        group_position_by_result_id,
                    )
                    for canonical_position in reversed(range(operand_count))
                ),
                member.op_index,
            ),
        )
    )

    return tuple(candidates)


def _group_member_order_score(
    group: OutputDepthOpGroup,
    members: tuple[OutputDepthOpGroupMember, ...],
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    consumers: dict[int, list[tuple[int, int]]],
    event_index_by_op_index: dict[int, int],
    group_position_by_result_id: dict[int, tuple[int, int]],
    ordered_groups: dict[int, OutputDepthOpGroup],
) -> tuple[Any, ...]:
    future_segment_count = _future_consumer_segment_count(
        group,
        members,
        events,
        consumers,
        event_index_by_op_index,
        ordered_groups,
    )
    source_run_count = _materialization_source_run_count(
        group,
        members,
        program,
        group_position_by_result_id,
    )
    future_order_keys = tuple(
        _group_member_future_order_key(
            member,
            events,
            consumers,
            event_index_by_op_index,
            ordered_groups,
        )
        for member in members
    )
    return (
        future_segment_count,
        source_run_count,
        future_order_keys,
        tuple(member.op_index for member in members),
    )


def _optimize_group_member_orders_by_materialization_cost(
    events: tuple[_ScheduledOutputDepthEvent, ...],
    program: RichProgram,
) -> tuple[_ScheduledOutputDepthEvent, ...]:
    group_event_indices = tuple(event_index for event_index, event in enumerate(events) if not isinstance(event, int))
    states = (events,)
    # Some useful reorderings require several dependent groups to move together;
    # a narrow beam keeps those temporary regressions available until they pay off.
    for event_index in group_event_indices:
        candidate_states: list[tuple[_ScheduledOutputDepthEvent, ...]] = []
        for state in states:
            event = state[event_index]
            if isinstance(event, int):
                continue

            for candidate_members in _group_order_candidates_for_state(
                program,
                state,
                event_index,
            ):
                state_events = list(state)
                state_events[event_index] = replace(event, members=candidate_members)
                candidate_states.append(tuple(state_events))

        states = _best_unique_order_states(program, candidate_states)

    return min(states, key=lambda state: _order_state_score(program, state))


def _group_order_candidates_for_state(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    event_index: int,
) -> tuple[tuple[OutputDepthOpGroupMember, ...], ...]:
    event = events[event_index]
    if isinstance(event, int):
        return ()

    consumers = _result_consumers_with_positions(program)
    event_index_by_op_index: dict[int, int] = {}
    group_position_by_result_id: dict[int, tuple[int, int]] = {}
    ordered_groups: dict[int, OutputDepthOpGroup] = {}
    for candidate_event_index, candidate_event in enumerate(events):
        if isinstance(candidate_event, int):
            event_index_by_op_index[candidate_event] = candidate_event_index
            continue
        ordered_groups[candidate_event_index] = candidate_event
        for position, member in enumerate(candidate_event.members):
            event_index_by_op_index[member.op_index] = candidate_event_index
            group_position_by_result_id[member.result_id] = (
                candidate_event_index,
                position,
            )

    return _group_member_order_candidates(
        event,
        program,
        events,
        consumers,
        event_index_by_op_index,
        group_position_by_result_id,
        ordered_groups,
    )


def _best_unique_order_states(
    program: RichProgram,
    states: Sequence[tuple[_ScheduledOutputDepthEvent, ...]],
) -> tuple[tuple[_ScheduledOutputDepthEvent, ...], ...]:
    unique_states = {_order_state_key(state): state for state in states}
    return tuple(
        sorted(
            unique_states.values(),
            key=lambda state: _order_state_score(program, state),
        )[:_GROUP_ORDER_BEAM_WIDTH]
    )


def _order_state_score(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> tuple[Any, ...]:
    return (
        _estimated_total_materialization_instruction_count(program, events),
        _order_state_key(events),
    )


def _order_state_key(
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (event,) if isinstance(event, int) else tuple(member.op_index for member in event.members)
        for event in events
    )


def _estimated_total_materialization_instruction_count(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> int:
    position_by_result_id = _group_member_position_by_result_id(events)
    input_axis0_position_by_index = {
        input_id: {original_index: position for position, original_index in enumerate(input_order)}
        for input_id, input_order in _input_axis0_orders_for_events(program, events).items()
    }
    return sum(
        _estimated_event_materialization_instruction_count(
            program,
            events,
            event_index,
            position_by_result_id,
            input_axis0_position_by_index,
        )
        for event_index, event in enumerate(events)
        if not isinstance(event, int)
    )


def _estimated_event_materialization_instruction_count(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    event_index: int,
    position_by_result_id: dict[int, tuple[int, int]],
    input_axis0_position_by_index: dict[int, dict[int, int]],
) -> int:
    event = events[event_index]
    if isinstance(event, int):
        return 0

    instruction_count = 0
    operand_count = len(event.members[0].canonical_argument_order)
    for canonical_position in range(operand_count):
        refs = tuple(
            _estimated_materialization_ref(
                program,
                member.arguments[member.canonical_argument_order[canonical_position]],
                position_by_result_id,
                input_axis0_position_by_index,
            )
            for member in event.members
        )
        instruction_count += _estimated_ref_materialization_instruction_count(
            program,
            events,
            refs,
        )
    return instruction_count


def _group_member_position_by_result_id(
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> dict[int, tuple[int, int]]:
    position_by_result_id: dict[int, tuple[int, int]] = {}
    for event_index, event in enumerate(events):
        if isinstance(event, int):
            continue
        for position, member in enumerate(event.members):
            position_by_result_id[member.result_id] = (event_index, position)
    return position_by_result_id


def _estimated_materialization_ref(
    program: RichProgram,
    argument: int,
    position_by_result_id: dict[int, tuple[int, int]],
    input_axis0_position_by_index: dict[int, dict[int, int]],
) -> tuple[tuple[Any, ...], int]:
    select_key = _direct_axis0_select_key(program, argument)
    if select_key is not None and select_key[0] < program.n_inputs:
        input_id, index = select_key
        mapped_index = input_axis0_position_by_index.get(input_id, {}).get(index, index)
        return ("input-select", input_id), mapped_index

    group_position = position_by_result_id.get(argument)
    if group_position is not None:
        event_index, position = group_position
        return ("producer-group", event_index), position

    if argument < program.n_inputs:
        return ("input", argument), 0

    return ("ssa", argument), 0


def _estimated_ref_materialization_instruction_count(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    refs: tuple[tuple[tuple[Any, ...], int], ...],
) -> int:
    segments = _estimated_ref_segments(refs)
    if len(segments) <= 1:
        return sum(
            0 if _estimated_segment_is_full_source(program, events, source, start, stop) else 1
            for source, start, stop in segments
        )
    return 1 + sum(
        0 if _estimated_segment_is_full_source(program, events, source, start, stop) else 1
        for source, start, stop in segments
    )


def _estimated_ref_segments(
    refs: tuple[tuple[tuple[Any, ...], int], ...],
) -> tuple[tuple[tuple[Any, ...], int, int], ...]:
    if not refs:
        return ()

    segments: list[tuple[tuple[Any, ...], int, int]] = []
    segment_source, start = refs[0]
    stop = start + 1
    for source, index in refs[1:]:
        if source == segment_source and index == stop:
            stop += 1
            continue
        segments.append((segment_source, start, stop))
        segment_source = source
        start = index
        stop = index + 1

    segments.append((segment_source, start, stop))
    return tuple(segments)


def _estimated_segment_is_full_source(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
    source: tuple[Any, ...],
    start: int,
    stop: int,
) -> bool:
    source_kind = source[0]
    if source_kind == "input-select":
        axis_size = program.shapes[source[1]][0]
    elif source_kind == "producer-group":
        source_event = events[source[1]]
        if isinstance(source_event, int):
            return False
        axis_size = len(source_event.members)
    else:
        return False
    return start == 0 and stop == axis_size


def _future_consumer_segment_count(
    group: OutputDepthOpGroup,
    members: tuple[OutputDepthOpGroupMember, ...],
    events: tuple[_ScheduledOutputDepthEvent, ...],
    consumers: dict[int, list[tuple[int, int]]],
    event_index_by_op_index: dict[int, int],
    ordered_groups: dict[int, OutputDepthOpGroup],
) -> int:
    position_by_result_id = {member.result_id: position for position, member in enumerate(members)}
    segment_count = 0
    scored_operands: set[tuple[int, int]] = set()

    for member in group.members:
        for consumer_op_index, _argument_position in consumers.get(member.result_id, []):
            consumer_event_index = event_index_by_op_index[consumer_op_index]
            consumer_event = events[consumer_event_index]
            if isinstance(consumer_event, int):
                continue

            ordered_consumer = ordered_groups.get(consumer_event_index, consumer_event)
            operand_count = len(ordered_consumer.members[0].canonical_argument_order)
            for canonical_position in range(operand_count):
                score_key = (consumer_event_index, canonical_position)
                if score_key in scored_operands:
                    continue

                positions: list[int | None] = []
                has_group_argument = False
                for consumer_member in ordered_consumer.members:
                    original_position = consumer_member.canonical_argument_order[canonical_position]
                    argument = consumer_member.arguments[original_position]
                    position = position_by_result_id.get(argument)
                    positions.append(position)
                    has_group_argument = has_group_argument or position is not None

                if not has_group_argument:
                    continue

                scored_operands.add(score_key)
                segment_count += _position_segment_count(positions)

    return segment_count


def _position_segment_count(positions: Sequence[int | None]) -> int:
    segment_count = 0
    previous_position: int | None = None
    for position in positions:
        if position is None:
            previous_position = None
            continue
        if previous_position is None or position != previous_position + 1:
            segment_count += 1
        previous_position = position
    return segment_count


def _materialization_source_run_count(
    group: OutputDepthOpGroup,
    members: tuple[OutputDepthOpGroupMember, ...],
    program: RichProgram,
    group_position_by_result_id: dict[int, tuple[int, int]],
) -> int:
    run_count = 0
    operand_count = len(group.members[0].canonical_argument_order)
    for canonical_position in range(operand_count):
        previous_source: tuple[Any, ...] | None = None
        for member in members:
            source_key = _member_materialization_source_key(
                program,
                member,
                canonical_position,
                group_position_by_result_id,
            )
            if source_key == previous_source:
                continue
            run_count += 1
            previous_source = source_key
    return run_count


def _member_materialization_order_key(
    program: RichProgram,
    member: OutputDepthOpGroupMember,
    canonical_position: int,
    group_position_by_result_id: dict[int, tuple[int, int]],
) -> tuple[Any, ...]:
    return (
        _member_materialization_source_key(
            program,
            member,
            canonical_position,
            group_position_by_result_id,
        ),
        _member_materialization_index(
            program,
            member,
            canonical_position,
            group_position_by_result_id,
        ),
        member.op_index,
    )


def _member_materialization_source_key(
    program: RichProgram,
    member: OutputDepthOpGroupMember,
    canonical_position: int,
    group_position_by_result_id: dict[int, tuple[int, int]],
) -> tuple[Any, ...]:
    source_kind, source_id, _index = _member_materialization_source(
        program,
        member,
        canonical_position,
        group_position_by_result_id,
    )
    source_kind_order = {
        "producer-group": 0,
        "input-select": 1,
        "input": 2,
        "ssa": 3,
    }[source_kind]
    return (source_kind_order, source_id)


def _member_materialization_index(
    program: RichProgram,
    member: OutputDepthOpGroupMember,
    canonical_position: int,
    group_position_by_result_id: dict[int, tuple[int, int]],
) -> int:
    _source_kind, _source_id, index = _member_materialization_source(
        program,
        member,
        canonical_position,
        group_position_by_result_id,
    )
    return index


def _member_materialization_source(
    program: RichProgram,
    member: OutputDepthOpGroupMember,
    canonical_position: int,
    group_position_by_result_id: dict[int, tuple[int, int]],
) -> tuple[str, int, int]:
    original_position = member.canonical_argument_order[canonical_position]
    argument = member.arguments[original_position]

    select_key = _direct_axis0_select_key(program, argument)
    if select_key is not None and select_key[0] < program.n_inputs:
        input_id, index = select_key
        return "input-select", input_id, index

    group_position = group_position_by_result_id.get(argument)
    if group_position is not None:
        event_index, position = group_position
        return "producer-group", event_index, position

    if argument < program.n_inputs:
        return "input", argument, 0

    return "ssa", argument, 0


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


def _direct_axis0_select_key(
    program: RichProgram,
    ssa_id: int,
) -> tuple[int, int] | None:
    if ssa_id < program.n_inputs:
        return None

    instruction = program.instructions[ssa_id - program.n_inputs]
    if not isinstance(instruction.operator, OperatorSelect):
        return None
    if instruction.operator.axis != 0:
        return None
    return instruction.argument_ssa_ids[0], instruction.operator.index


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
    parameter_stack_orders = _parameter_input_stack_orders_for_events(program, events)
    input_axis0_orders = _input_axis0_orders_for_events(program, events)
    input_plan = _make_rewrite_input_plan(program, parameter_stack_orders, input_axis0_orders)

    instructions: list[RichInstruction] = []
    shapes = input_plan.shapes.copy()
    tensor_formats = input_plan.tensor_formats.copy()
    tensor_map = input_plan.tensor_map.copy()
    materialized_elements: dict[tuple[int, int], int] = {}
    materialized_segments: dict[tuple[int, int, int], int] = {}
    concatenated_batches: dict[tuple[tuple[int, int], ...], int] = {}
    batched_result_orders: list[tuple[int, ...]] = []
    non_parameter_stack_orders: list[tuple[int, ...]] = []
    concatenated_batch_orders: list[tuple[int, ...]] = []

    def append_instruction(
        instruction: RichInstruction,
        *,
        output_shape: Shape | None = None,
        output_format: TensorFormat | None = None,
    ) -> int:
        result_id = input_plan.n_inputs + len(instructions)
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
        if isinstance(ref, (_TensorRef, _InputRef)):
            return shapes[ref.ssa_id]
        source_shape = shapes[ref.source_ssa_id]
        if not source_shape:
            raise ValueError("batched tensor has no batch axis")
        return source_shape[1:]

    def ref_format(ref: _TensorValueRef) -> TensorFormat:
        if isinstance(ref, (_TensorRef, _InputRef)):
            return tensor_formats[ref.ssa_id]
        return tensor_formats[ref.source_ssa_id]

    def materialize_ref(ref: _TensorValueRef) -> int:
        if isinstance(ref, (_TensorRef, _InputRef)):
            return ref.ssa_id

        element_key = _batch_element_key(ref)
        cached = materialized_elements.get(element_key)
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
        materialized_elements[element_key] = result_id
        return result_id

    def materialize_batch_segment(ref: _BatchElementRef, start: int, stop: int) -> int:
        if start == 0 and shapes[ref.source_ssa_id][0] == stop:
            return ref.source_ssa_id
        segment_key = (ref.source_ssa_id, start, stop)
        cached = materialized_segments.get(segment_key)
        if cached is not None:
            return cached

        result_id = append_instruction(
            RichInstruction(
                operator=OperatorSlice(start=start, stop=stop, axis=0),
                argument_ssa_ids=(ref.source_ssa_id,),
            ),
            output_format=ref_format(ref),
        )
        materialized_segments[segment_key] = result_id
        return result_id

    def materialize_batch_elements(refs: tuple[_TensorValueRef, ...]) -> int | None:
        if not all(isinstance(ref, _BatchElementRef) for ref in refs):
            return None

        batch_refs = tuple(ref for ref in refs if isinstance(ref, _BatchElementRef))
        batch_key = tuple(_batch_element_key(ref) for ref in batch_refs)
        cached = concatenated_batches.get(batch_key)
        if cached is not None:
            return cached

        segments = _batch_ref_segments(batch_refs)
        if len(segments) == 1:
            ref, start, stop = segments[0]
            return materialize_batch_segment(ref, start, stop)

        segment_ids = tuple(materialize_batch_segment(ref, start, stop) for ref, start, stop in segments)
        result_id = append_instruction(
            RichInstruction(
                operator=OperatorConcat(axis=0),
                argument_ssa_ids=segment_ids,
            ),
            output_format=ref_format(batch_refs[0]),
        )
        concatenated_batches[batch_key] = result_id
        original_order = _original_ssa_order(batch_refs)
        if original_order is not None:
            concatenated_batch_orders.append(original_order)
        return result_id

    def materialize_batch(refs: tuple[_TensorValueRef, ...]) -> int:
        if not refs:
            raise ValueError("cannot materialize an empty batch")

        input_order = _original_input_order(refs)
        if input_order is not None:
            packed_parameter_input = input_plan.parameter_stack_input_by_order.get(input_order)
            if packed_parameter_input is not None:
                return packed_parameter_input
            if any(input_id not in program.parameter_indices for input_id in input_order):
                non_parameter_stack_orders.append(input_order)

        materialized_batch_elements = materialize_batch_elements(refs)
        if materialized_batch_elements is not None:
            return materialized_batch_elements

        first_ref = refs[0]
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
                tensor_map[member.result_id] = _BatchElementRef(
                    source_ssa_id=batched_result_id,
                    index=index,
                    original_ssa_id=member.result_id,
                )
            continue

        instruction = program.instructions[event]
        result_id = program.n_inputs + event
        if isinstance(instruction.operator, OperatorStack) and instruction.operator.axis == 0:
            mapped_refs = tuple(tensor_map[argument] for argument in instruction.argument_ssa_ids)
            mapped_result_id = materialize_batch(mapped_refs)
            tensor_map[result_id] = _TensorRef(mapped_result_id)
            continue

        if isinstance(instruction.operator, OperatorSelect) and instruction.operator.axis == 0:
            mapped_source = materialize_ref(tensor_map[instruction.argument_ssa_ids[0]])
            mapped_index = input_plan.input_axis0_position_by_index.get(instruction.argument_ssa_ids[0], {}).get(
                instruction.operator.index,
                instruction.operator.index,
            )
            tensor_map[result_id] = _BatchElementRef(
                source_ssa_id=mapped_source,
                index=mapped_index,
                original_ssa_id=result_id,
            )
            continue

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
    if not _is_last_result(output_id, input_plan.n_inputs, instructions):
        output_id = _copy_to_last_result(
            output_id,
            append_instruction,
            shapes,
            tensor_formats,
        )
    if not _is_last_result(output_id, input_plan.n_inputs, instructions):
        raise RuntimeError("rewritten program output was not materialized as the last result")

    rewritten_program = RichProgram(
        instructions=instructions,
        n_inputs=input_plan.n_inputs,
        stability_mode=program.stability_mode,
        shapes=shapes,
        tensor_formats=tensor_formats,
        parameter_indices=input_plan.parameter_indices,
    )
    return FoldSameShapedOperationsResult(
        program=rewritten_program,
        batched_result_orders=tuple(batched_result_orders),
        non_parameter_stack_orders=tuple(non_parameter_stack_orders),
        parameter_stack_orders=parameter_stack_orders,
        input_axis0_orders=input_plan.input_axis0_orders,
        concatenated_batch_orders=tuple(concatenated_batch_orders),
    )


def _make_rewrite_input_plan(
    program: RichProgram,
    parameter_stack_orders: tuple[tuple[int, ...], ...],
    input_axis0_orders: dict[int, tuple[int, ...]],
) -> _RewriteInputPlan:
    used_input_ids = {
        argument
        for instruction in program.instructions
        for argument in instruction.argument_ssa_ids
        if argument < program.n_inputs
    }
    packed_parameter_input_ids = {input_id for input_order in parameter_stack_orders for input_id in input_order}
    retained_input_ids = tuple(input_id for input_id in range(program.n_inputs) if input_id in used_input_ids and input_id not in packed_parameter_input_ids)
    retained_input_id_map = {input_id: new_input_id for new_input_id, input_id in enumerate(retained_input_ids)}
    parameter_stack_input_by_order = {
        input_order: len(retained_input_ids) + stack_index for stack_index, input_order in enumerate(parameter_stack_orders)
    }
    retained_input_axis0_orders = {
        input_id: input_order
        for input_id, input_order in input_axis0_orders.items()
        if input_id in retained_input_id_map
    }
    input_axis0_position_by_index = {
        input_id: {original_index: position for position, original_index in enumerate(input_order)}
        for input_id, input_order in retained_input_axis0_orders.items()
    }

    tensor_map: dict[int, _TensorValueRef] = {
        input_id: _InputRef(new_input_id, input_id) for input_id, new_input_id in retained_input_id_map.items()
    }
    for input_order, packed_input_id in parameter_stack_input_by_order.items():
        for index, input_id in enumerate(input_order):
            tensor_map.setdefault(
                input_id,
                _BatchElementRef(
                    source_ssa_id=packed_input_id,
                    index=index,
                    original_input_id=input_id,
                    original_ssa_id=input_id,
                ),
            )

    return _RewriteInputPlan(
        n_inputs=len(retained_input_ids) + len(parameter_stack_orders),
        shapes=[
            *(program.shapes[input_id] for input_id in retained_input_ids),
            *(_stacked_input_shape(program, input_order) for input_order in parameter_stack_orders),
        ],
        tensor_formats=[
            *(program.tensor_formats[input_id] for input_id in retained_input_ids),
            *(_stacked_input_format(program, input_order) for input_order in parameter_stack_orders),
        ],
        parameter_indices=frozenset(
            retained_input_id_map[input_id] for input_id in retained_input_ids if input_id in program.parameter_indices
        ).union(parameter_stack_input_by_order.values()),
        tensor_map=tensor_map,
        parameter_stack_input_by_order=parameter_stack_input_by_order,
        input_axis0_orders=retained_input_axis0_orders,
        input_axis0_position_by_index=input_axis0_position_by_index,
    )


def _parameter_input_stack_orders_for_events(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> tuple[tuple[int, ...], ...]:
    parameter_stack_orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for event in events:
        if isinstance(event, int):
            continue
        for input_order in _input_orders_by_canonical_operand(event, program.n_inputs):
            if not all(input_id in program.parameter_indices for input_id in input_order):
                continue
            if input_order in seen:
                continue
            seen.add(input_order)
            parameter_stack_orders.append(input_order)
    return tuple(parameter_stack_orders)


def _input_axis0_orders_for_events(
    program: RichProgram,
    events: tuple[_ScheduledOutputDepthEvent, ...],
) -> dict[int, tuple[int, ...]]:
    candidate_runs_by_input: dict[int, list[tuple[tuple[int, int], tuple[int, ...]]]] = {}
    for event_index, event in enumerate(events):
        if isinstance(event, int):
            continue

        indices_by_input: dict[int, list[int]] = {}
        operand_count = len(event.members[0].canonical_argument_order)
        for canonical_position in range(operand_count):
            select_batch = _direct_input_axis0_select_batch(program, event, canonical_position)
            if select_batch is None:
                continue
            input_id, indices = select_batch
            indices_by_input.setdefault(input_id, []).extend(indices)

        for input_id, indices in indices_by_input.items():
            if len(indices) < 2 or len(indices) != len(set(indices)):
                continue
            axis_size = program.shapes[input_id][0]
            if any(index < 0 or index >= axis_size for index in indices):
                continue

            score = (len(indices), -event_index)
            candidate_runs_by_input.setdefault(input_id, []).append((score, tuple(indices)))

    input_orders: dict[int, tuple[int, ...]] = {}
    for input_id, candidate_runs in candidate_runs_by_input.items():
        axis_size = program.shapes[input_id][0]
        selected: set[int] = set()
        ordered_indices: list[int] = []
        for _score, indices in sorted(candidate_runs, key=lambda candidate: (-candidate[0][0], -candidate[0][1])):
            if any(index in selected for index in indices):
                continue
            ordered_indices.extend(indices)
            selected.update(indices)

        if len(ordered_indices) < 2:
            continue

        completed_order = (*ordered_indices, *(index for index in range(axis_size) if index not in selected))
        if completed_order == tuple(range(axis_size)):
            continue
        input_orders[input_id] = completed_order

    return input_orders


def _direct_input_axis0_select_batch(
    program: RichProgram,
    group: OutputDepthOpGroup,
    canonical_position: int,
) -> tuple[int, tuple[int, ...]] | None:
    input_id: int | None = None
    indices: list[int] = []
    for member in group.members:
        original_position = member.canonical_argument_order[canonical_position]
        select_key = _direct_axis0_select_key(program, member.arguments[original_position])
        if select_key is None:
            return None
        source_input_id, index = select_key
        if source_input_id >= program.n_inputs:
            return None
        if input_id is None:
            input_id = source_input_id
        elif input_id != source_input_id:
            return None
        indices.append(index)

    if input_id is None:
        return None
    return input_id, tuple(indices)


def _input_orders_by_canonical_operand(
    group: OutputDepthOpGroup,
    n_inputs: int,
) -> list[tuple[int, ...]]:
    operand_count = len(group.members[0].canonical_argument_order)
    grouped: list[list[int] | None] = [[] for _ in range(operand_count)]
    for member in group.members:
        for canonical_position, original_position in enumerate(member.canonical_argument_order):
            argument = member.arguments[original_position]
            if argument >= n_inputs:
                grouped[canonical_position] = None
                continue
            if grouped[canonical_position] is None:
                continue
            grouped[canonical_position].append(argument)
    return [tuple(input_order) for input_order in grouped if input_order is not None and input_order]


def _stacked_input_shape(
    program: RichProgram,
    input_order: tuple[int, ...],
) -> Shape:
    input_shapes = [program.shapes[input_id] for input_id in input_order]
    return OperatorStack(axis=0).propagate_shapes(input_shapes)


def _stacked_input_format(
    program: RichProgram,
    input_order: tuple[int, ...],
) -> TensorFormat:
    input_formats = [program.tensor_formats[input_id] for input_id in input_order]
    return _infer_generated_tensor_format(OperatorStack(axis=0), input_formats)


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


def _batch_element_key(ref: _BatchElementRef) -> tuple[int, int]:
    return ref.source_ssa_id, ref.index


def _batch_ref_segments(
    refs: tuple[_BatchElementRef, ...],
) -> tuple[tuple[_BatchElementRef, int, int], ...]:
    if not refs:
        raise ValueError("cannot segment an empty batch")

    segments: list[tuple[_BatchElementRef, int, int]] = []
    segment_ref = refs[0]
    start = refs[0].index
    stop = start + 1
    for ref in refs[1:]:
        if ref.source_ssa_id == segment_ref.source_ssa_id and ref.index == stop:
            stop += 1
            continue
        segments.append((segment_ref, start, stop))
        segment_ref = ref
        start = ref.index
        stop = start + 1

    segments.append((segment_ref, start, stop))
    return tuple(segments)


def _original_ssa_order(
    refs: tuple[_BatchElementRef, ...],
) -> tuple[int, ...] | None:
    ssa_ids: list[int] = []
    for ref in refs:
        if ref.original_ssa_id is None:
            return None
        ssa_ids.append(ref.original_ssa_id)
    return tuple(ssa_ids)


def _original_input_order(
    refs: tuple[_TensorValueRef, ...],
) -> tuple[int, ...] | None:
    input_ids: list[int] = []
    for ref in refs:
        if isinstance(ref, _InputRef):
            input_ids.append(ref.original_input_id)
            continue
        if isinstance(ref, _BatchElementRef) and ref.original_input_id is not None:
            input_ids.append(ref.original_input_id)
            continue
        return None
    return tuple(input_ids)


def _make_batched_operator(operator: RichOperator) -> RichOperator:
    match operator:
        case OperatorEinsum(format_string):
            input_strings, output_string = parse_format_string(format_string)
            return OperatorEinsum(_batched_einsum_format(input_strings, output_string))
        case OperatorStack(axis):
            return OperatorStack(axis + 1)
        case OperatorConcat(axis):
            return OperatorConcat(axis + 1)
        case OperatorSlice(start, stop, axis):
            return OperatorSlice(start, stop, axis + 1)
        case OperatorSelect(axis, index):
            return OperatorSelect(axis + 1, index)
        case OperatorSoftmax(axis):
            if isinstance(axis, int):
                return OperatorSoftmax(axis + 1)
            return OperatorSoftmax(tuple(item + 1 for item in axis))
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
    if isinstance(operator, (OperatorStack, OperatorConcat, OperatorEinsum)) and any(format != first_format for format in argument_formats[1:]):
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


def to_annotated_ssa_path(
    format_string: str,
    ssa_path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    prefer_ascii: bool = False,
    *,
    prioritize_output_labels: bool = False,
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
            if prioritize_output_labels:
                output_prefix = "".join(char for char in output if char in t3)
                t3 = output_prefix + "".join(char for char in t3 if char not in output_prefix)
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
