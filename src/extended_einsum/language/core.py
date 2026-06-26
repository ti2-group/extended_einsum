from dataclasses import dataclass
from typing import Any, Literal

UnaryOperator = Literal["sin", "cos", "tan", "exp", "log", "sqrt", "1/"]
BinaryOperator = Literal["+", "-", "*", "/", "**"]
EinsumOperator = Literal["einsum"]
StackOperator = Literal["stack"]
TakeOperator = Literal["take"]
SliceOperator = Literal["slice"]
SelectOperator = Literal["select"]
SoftmaxOperator = Literal["softmax"]

OperatorName = UnaryOperator | BinaryOperator | EinsumOperator | StackOperator | TakeOperator | SliceOperator | SelectOperator | SoftmaxOperator
ArgumentSSAIds = tuple[int, ...]
ExtraArguments = tuple[Any, ...]

RawInstruction = tuple[OperatorName, ArgumentSSAIds, ExtraArguments]


@dataclass(frozen=True)
class RawProgram:
    instructions: list[RawInstruction]
    n_inputs: int

    @property
    def output_ssa(self) -> int:
        return self.n_inputs + len(self.instructions) - 1
