from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

UnaryOperator = Literal["exp", "log"]
BinaryOperator = Literal["+", "-", "*", "/"]
EinsumOperator = Literal["einsum"]
StackOperator = Literal["stack"]
TakeOperator = Literal["take"]
SliceOperator = Literal["slice"]
SoftmaxOperator = Literal["softmax"]
SelectOperator = Literal["select"]

UNARY_OPERATORS: frozenset[UnaryOperator] = frozenset(get_args(UnaryOperator))
BINARY_OPERATORS: frozenset[BinaryOperator] = frozenset(get_args(BinaryOperator))
EINSUM_OPERATOR: EinsumOperator = "einsum"
STACK_OPERATOR: StackOperator = "stack"
TAKE_OPERATOR: TakeOperator = "take"
SLICE_OPERATOR: SliceOperator = "slice"
SOFTMAX_OPERATOR: SoftmaxOperator = "softmax"
SELECT_OPERATOR: SelectOperator = "select"

Operator = (
    UnaryOperator
    | BinaryOperator
    | StackOperator
    | TakeOperator
    | SelectOperator
    | SliceOperator
    | SoftmaxOperator
    | EinsumOperator
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


def get_einsum_format_string(instruction: Instruction) -> str:
    return get_instruction_specific_arguments(instruction)[0]


def get_scaled_einsum_output_axis(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[1]


def get_stack_axis(instruction: Instruction) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def get_take_axis(instruction: Instruction) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def get_select_axis(instruction: Instruction) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def get_select_index(instruction: Instruction) -> int:
    return get_instruction_specific_arguments(instruction)[1]


def get_slice_start(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[0]


def get_slice_stop(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[1]


def get_slice_axis(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[2]


def get_softmax_axis(instruction: tuple[Any, ...]) -> int:
    return get_instruction_specific_arguments(instruction)[0]


@dataclass(frozen=True)
class Program:
    instructions: list[Instruction]
    n_inputs: int
    # ssa_id_to_tensor_format: list[TensorFormat]

    # def __post_init__(self):
    #     if len(self.ssa_id_to_tensor_format) != len(self.instructions) + self.n_inputs:
    #         raise ValueError(
    #             f"Number of tensor formats ({len(self.ssa_id_to_tensor_format)}) must match the expected number of SSA IDs ({self.n_inputs} inputs + {len(self.instructions)} instructions)."
    #         )

    @property
    def output_ssa(self) -> int:
        return self.n_inputs + len(self.instructions) - 1
