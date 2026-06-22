from dataclasses import dataclass
from typing import Any, Literal, Protocol

UnaryOperator = Literal["sin", "cos", "tan", "exp", "log", "sqrt", "1/"]
BinaryOperator = Literal["+", "-", "*", "/", "**"]
EinsumOperator = Literal["einsum"]
StackOperator = Literal["stack"]
TakeOperator = Literal["take"]
SliceOperator = Literal["slice"]
SoftmaxOperator = Literal["softmax"]

OperatorName = (
    UnaryOperator
    | BinaryOperator
    | EinsumOperator
    | StackOperator
    | TakeOperator
    | SliceOperator
    | SoftmaxOperator
)


RawInstruction = tuple[OperatorName, tuple[int, ...], tuple[Any, ...]]


class Operator(Protocol):
    @property
    def name(self) -> OperatorName: ...

    @property
    def raw_extra_arguments(self) -> tuple[Any, ...]: ...

    def check_inputs(self, operands: list[Any]) -> bool: ...

    def to_instruction(self, operand_ids: tuple[int, ...]) -> RawInstruction:
        return (self.name, operand_ids, self.raw_extra_arguments)


@dataclass(frozen=True)
class Program:
    instructions: list[RawInstruction]
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
