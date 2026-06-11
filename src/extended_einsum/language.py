from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

UnaryOperator = Literal["sin", "cos", "tan", "exp", "log", "sqrt", "1/"]
BinaryOperator = Literal["+", "-", "*", "/", "**"]
EinsumOperator = Literal["einsum"]
ScaledEinsumOperator = Literal["scaled_einsum_sum", "scaled_einsum_max"]
StackOperator = Literal["stack"]
TakeOperator = Literal["take"]
SliceOperator = Literal["slice"]
SoftmaxOperator = Literal["softmax"]

EINSUM_OPERATOR: EinsumOperator = "einsum"
SCALED_EINSUM_SUM_OPERATOR: ScaledEinsumOperator = "scaled_einsum_sum"
SCALED_EINSUM_MAX_OPERATOR: ScaledEinsumOperator = "scaled_einsum_max"
SCALED_EINSUM_OPERATORS: frozenset[ScaledEinsumOperator] = frozenset(
    (SCALED_EINSUM_SUM_OPERATOR, SCALED_EINSUM_MAX_OPERATOR)
)
STACK_OPERATOR: StackOperator = "stack"
TAKE_OPERATOR: TakeOperator = "take"
SLICE_OPERATOR: SliceOperator = "slice"
SOFTMAX_OPERATOR: SoftmaxOperator = "softmax"

Operator = (
    UnaryOperator
    | BinaryOperator
    | StackOperator
    | TakeOperator
    | SliceOperator
    | SoftmaxOperator
    | EinsumOperator
    | ScaledEinsumOperator
)

# operator, operand_ids, instruction specific arguments
Instruction = tuple[Operator, tuple[int, ...], tuple[Any, ...]]


def make_stack_instruction(operand_ids: tuple[int, ...], axis: int) -> Instruction:
    return (STACK_OPERATOR, operand_ids, (axis,))


def make_take_instruction(operand_id: int, indices_id: int, axis: int) -> Instruction:
    return (TAKE_OPERATOR, (operand_id, indices_id), (axis,))


def make_slice_instruction(
    operand_id: int, start: int, stop: int, axis: int
) -> Instruction:
    return (SLICE_OPERATOR, (operand_id,), (start, stop, axis))


def make_softmax_instruction(operand_id: int, axis: int) -> Instruction:
    return (SOFTMAX_OPERATOR, (operand_id,), (axis,))


def make_einsum_instruction(
    format_string: str,
    *argument_ids: int,
) -> Instruction:
    return (EINSUM_OPERATOR, tuple(argument_ids), (format_string,))


def make_scaled_einsum_instruction(
    operator: ScaledEinsumOperator,
    format_string: str,
    lhs_id: int,
    rhs_id: int,
    output_scale_axis: int,
) -> Instruction:
    return (operator, (lhs_id, rhs_id), (format_string, output_scale_axis))


def get_operator(instruction: Instruction) -> Operator:
    return instruction[0]


def get_arguments(instruction: Instruction) -> tuple[int, ...]:
    return instruction[1]


def get_instruction_specific_arguments(instruction: Instruction) -> tuple[Any, ...]:
    return instruction[2]


def map_instruction_arguments(
    instruction: Instruction,
    mapper: Callable[[int], int],
) -> Instruction:
    operator, arguments, instruction_specific_arguments = instruction
    return (
        operator,
        tuple(mapper(argument) for argument in arguments),
        instruction_specific_arguments,
    )


def is_normal_einsum_instruction(instruction: tuple[Any, ...]) -> bool:
    return get_operator(instruction) == EINSUM_OPERATOR


def is_scaled_einsum_instruction(instruction: tuple[Any, ...]) -> bool:
    return get_operator(instruction) in SCALED_EINSUM_OPERATORS


def is_einsum_instruction(instruction: tuple[Any, ...]) -> bool:
    return is_normal_einsum_instruction(instruction) or is_scaled_einsum_instruction(
        instruction
    )


def get_format_string_einsum(instruction: Instruction) -> str:
    return get_instruction_specific_arguments(instruction)[0]


def get_output_axis_scaled_einsum(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[1]


def get_axis_take(instruction: Instruction) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def slice_start(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def slice_stop(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[1]


def slice_axis(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[2]


def softmax_axis(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[0]


@dataclass
class Program:
    instructions: list[Instruction]
    n_inputs: int

    @property
    def output_ssa(self) -> int:
        return self.n_inputs + len(self.instructions) - 1
