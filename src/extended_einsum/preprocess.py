from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, override

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
