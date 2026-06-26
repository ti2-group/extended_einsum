from dataclasses import dataclass
from typing import Generic, override

from extended_einsum.backend import TBackendArray, get_backend_of_array
from extended_einsum.format import DenseArray, SparseArray
from extended_einsum.interface.tensor_expression import (
    Parameter,
    TensorExpression,
)
from extended_einsum.language.rich_operators import (
    OperatorCos,
    OperatorEinsum,
    OperatorExp,
    OperatorInverse,
    OperatorLog,
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorSqrt,
    OperatorStack,
    OperatorTake,
    OperatorTan,
)
from extended_einsum.language.types import Array, Backend, Shape, TArray, TensorFormat
from extended_einsum.utils import normalize_axis, parse_format_string


@dataclass(frozen=True)
class BackendArrayWrapper(Array, Generic[TBackendArray]):
    backend_array: TBackendArray
    _format: TensorFormat

    @property
    @override
    def shape(self) -> Shape:
        return tuple(self.backend_array.shape)

    @property
    @override
    def backend(self) -> Backend:
        return get_backend_of_array(self.backend_array)

    @property
    @override
    def format(self) -> TensorFormat:
        return self._format


def array(backend_array: TBackendArray, format: TensorFormat = "dense") -> DenseArray[TBackendArray] | SparseArray[TBackendArray]:
    match format:
        case "dense":
            return DenseArray(backend_array)
        case "sparse":
            return SparseArray(backend_array)
        case _:
            raise ValueError(f"Unsupported format: {format}")


def exp(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorExp(), [a])


def log(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorLog(), [a])


def sin(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorSin(), [a])


def cos(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorCos(), [a])


def tan(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorTan(), [a])


def sqrt(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorSqrt(), [a])


def inverse(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorInverse(), [a])


def einsum(
    format_string: str,
    *operands: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    index_strings, output_string = parse_format_string(format_string)
    if len(index_strings) != len(operands):
        raise ValueError(f"format string {format_string} has {len(index_strings)} indices, but {len(operands)} operands.")
    all_input_symbols = frozenset("".join(index_strings))
    if any(output_symbol not in all_input_symbols for output_symbol in output_string):
        raise ValueError(f"format string {format_string} contains output symbols that are not present in the operands.")
    return TensorExpression(OperatorEinsum(format_string), list(operands))


def stack(
    operands: list[TensorExpression[TArray] | Parameter[TArray] | TArray],
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(operands[0].shape))
    if len(operands) == 0:
        raise ValueError("stack requires at least one argument")
    if any(operand.shape != operands[0].shape for operand in operands[1:]):
        raise ValueError("The stack operator requires all arguments to have the same shape along the stack axis.")
    return TensorExpression(OperatorStack(axis), operands)


def take(
    source: TensorExpression[TArray] | TArray,
    index: TensorExpression[TArray] | TArray,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    if not source.shape:
        raise ValueError("The take operator requires an operand with a leading axis.")
    if not index.shape:
        raise ValueError("The take operator requires an index with a leading axis.")
    return TensorExpression(OperatorTake(axis), [source, index])


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorSlice(start, stop, axis), [source])


def softmax(
    a: TensorExpression[TArray] | TArray,
    axis: int = 0,
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor."""

    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axis = normalize_axis(axis, len(a.shape))
    return TensorExpression(OperatorSoftmax(axis), [a])
