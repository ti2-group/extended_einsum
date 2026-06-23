from dataclasses import dataclass
from typing import Any, Protocol, override

import numpy as np

from extended_einsum.language.core import OperatorName, RawInstruction
from extended_einsum.language.types import HasShape, Shape
from extended_einsum.utils import parse_format_string

################################
# main protocol, everything in here inherits from this
################################


class RichOperator(Protocol):
    @property
    def name(self) -> OperatorName: ...

    @property
    def raw_extra_arguments(self) -> tuple[Any, ...]: ...

    def check_inputs(self, operands: list[Any]) -> None: ...

    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape: ...

    def to_instruction(self, operand_ids: Shape) -> RawInstruction:
        return (self.name, operand_ids, self.raw_extra_arguments)


################################
# some functions are the same for many operators
################################


class UnaryArithmeticOperator(RichOperator, Protocol):
    @override
    def check_inputs(self, operands: list[Any]) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"The {self.name} operator takes exactly one argument, but {len(operands)} were given."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        return input_shapes[0]


class BinaryArithmeticOperator(RichOperator, Protocol):
    @override
    def check_inputs(self, operands: list[Any]) -> None:
        if len(operands) != 2:
            raise ValueError(
                f"The {self.name} operator takes exactly two arguments, but {len(operands)} were given."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        return np.broadcast_shapes(input_shapes[0], input_shapes[1])


class NoExtraArgumentOperator(RichOperator, Protocol):
    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return ()


################################
# complete operators
################################


@dataclass(frozen=True)
class OperatorSin(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "sin"


@dataclass(frozen=True)
class OperatorCos(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "cos"


@dataclass(frozen=True)
class OperatorTan(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "tan"


@dataclass(frozen=True)
class OperatorExp(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "exp"


@dataclass(frozen=True)
class OperatorLog(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "log"


@dataclass(frozen=True)
class OperatorSqrt(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "sqrt"


@dataclass(frozen=True)
class OperatorInverse(UnaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "1/"


@dataclass(frozen=True)
class OperatorAdd(BinaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "+"


@dataclass(frozen=True)
class OperatorSubtract(BinaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "-"


@dataclass(frozen=True)
class OperatorMultiply(BinaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "*"


@dataclass(frozen=True)
class OperatorDivide(BinaryArithmeticOperator, NoExtraArgumentOperator):
    @property
    @override
    def name(self) -> OperatorName:
        return "/"


@dataclass(frozen=True)
class OperatorStack(RichOperator):
    axis: int

    @property
    @override
    def name(self) -> OperatorName:
        return "stack"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.axis,)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        if len(operands) == 0:
            raise ValueError("stack requires at least one argument")
        non_stacked_shape = operands[0].shape
        if any(operand.shape != non_stacked_shape for operand in operands[1:]):
            raise ValueError(
                "The stack operator requires all arguments to have the same shape along the stack axis."
            )
        if not 0 <= self.axis <= len(non_stacked_shape):
            raise ValueError(
                f"The stack operator wants to stack axis {self.axis} but the operands only have {len(non_stacked_shape)} axes. Bounds are 0 <= axis <= {len(non_stacked_shape)}."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        non_stacked_shape = input_shapes[0]
        return (
            *non_stacked_shape[: self.axis],
            len(input_shapes),
            *non_stacked_shape[self.axis :],
        )


@dataclass(frozen=True)
class OperatorTake(RichOperator):
    axis: int

    @property
    @override
    def name(self) -> OperatorName:
        return "take"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.axis,)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        if len(operands) != 2:
            raise ValueError(
                f"The take operator takes exactly two arguments, but {len(operands)} were given."
            )
        if len(operands[0].shape) == 0:
            raise ValueError(
                f"The take operator takes a non-scalar as first arguments, but the first argument has shape {operands[0].shape}."
            )
        if len(operands[1].shape) != 1:
            raise ValueError(
                f"The take operator takes a vector as second arguments, but the second argument has shape {operands[1].shape}."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        return (
            *input_shapes[0][: self.axis],
            *input_shapes[1],
            *input_shapes[0][self.axis + 1 :],
        )


@dataclass(frozen=True)
class OperatorSlice(RichOperator):
    start: int
    stop: int
    axis: int

    @property
    @override
    def name(self) -> OperatorName:
        return "slice"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.start, self.stop, self.axis)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"The slice operator takes exactly one argument, but {len(operands)} were given."
            )
        if len(operands[0].shape) == 0:
            raise ValueError(
                f"The slice operator takes a non-scalar as first arguments, but the first argument has shape {operands[0].shape}."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        return (
            *input_shapes[0][: self.axis],
            self.stop - self.start,
            *input_shapes[0][self.axis + 1 :],
        )


@dataclass(frozen=True)
class OperatorSelect(RichOperator):
    axis: int
    index: int

    @property
    @override
    def name(self) -> OperatorName:
        return "select"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.axis, self.index)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"The select operator takes exactly one argument, but {len(operands)} were given."
            )
        if len(operands[0].shape) == 0:
            raise ValueError(
                f"The select operator takes a non-scalar as first arguments, but the first argument has shape {operands[0].shape}."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        # TODO: is this the correct shape? or do we just delete the middle axis?
        return (
            *input_shapes[0][: self.axis],
            1,
            *input_shapes[0][self.axis + 1 :],
        )


@dataclass(frozen=True)
class OperatorSoftmax(RichOperator):
    axis: int

    @property
    @override
    def name(self) -> OperatorName:
        return "softmax"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.axis,)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"The softmax operator takes exactly one argument, but {len(operands)} were given."
            )
        if len(operands[0].shape) == 0:
            raise ValueError(
                f"The softmax operator takes a non-scalar as first arguments, but the first argument has shape {operands[0].shape}."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        return input_shapes[0]


@dataclass(frozen=True)
class OperatorEinsum(RichOperator):
    format_string: str

    def __post_init__(self):
        index_strings, output_string = parse_format_string(self.format_string)
        symbols_in_output_string = set(output_string)
        symbols_in_index_strings = set().union(*index_strings)
        if any(
            symbol not in symbols_in_index_strings
            for symbol in symbols_in_output_string
        ):
            raise ValueError(
                f"Einsum format string error: an output symbol in {output_string} is not present in the input index strings."
            )

    @property
    @override
    def name(self) -> OperatorName:
        return "einsum"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return (self.format_string,)

    @override
    def check_inputs(self, operands: list[HasShape]) -> None:
        index_strings, output_string = parse_format_string(self.format_string)
        if len(index_strings) != len(operands):
            raise ValueError(
                f"The number of indices in the einsum format string ({len(index_strings)}) does not match the number of arguments ({len(operands)})."
            )

    @override
    def propagate_shapes(self, input_shapes: list[Shape]) -> Shape:
        raise NotImplementedError
