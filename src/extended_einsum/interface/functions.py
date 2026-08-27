from __future__ import annotations

from typing import cast

from extended_einsum.interface.tensor_expression import (
    OperatorExpression,
    TensorExpression,
    as_expression,
)
from extended_einsum.language.rich_operators import (
    OperatorCos,
    OperatorEinsum,
    OperatorExp,
    OperatorInverse,
    OperatorLog,
    OperatorSelect,
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorSqrt,
    OperatorStack,
    OperatorTake,
    OperatorTan,
)
from extended_einsum.language.types import TArray
from extended_einsum.utils import normalize_axis, parse_format_string


def exp(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorExp(), [a])


def log(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorLog(), [a])


def sin(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorSin(), [a])


def cos(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorCos(), [a])


def tan(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorTan(), [a])


def sqrt(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorSqrt(), [a])


def inverse(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return OperatorExpression(OperatorInverse(), [a])


def einsum(
    format_string: str,
    *operands: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    index_strings, output_string = parse_format_string(format_string)
    if len(index_strings) != len(operands):
        operand_word = "operand was" if len(operands) == 1 else "operands were"
        raise ValueError(f"The einsum format string {format_string!r} has {len(index_strings)} input terms, but {len(operands)} {operand_word} given.")
    all_input_symbols = frozenset("".join(index_strings))
    missing_symbols = sorted(set(output_string) - all_input_symbols)
    if missing_symbols:
        raise ValueError(f"The einsum format string {format_string!r} has output symbols that do not appear in any input term: {', '.join(missing_symbols)}.")
    return OperatorExpression(OperatorEinsum(format_string), list(operands))


def stack(
    operands: list[TensorExpression[TArray] | TArray],
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    if len(operands) == 0:
        raise ValueError("stack requires at least one operand")
    first_operand = cast("TensorExpression[TArray]", as_expression(operands[0]))
    # the stack axis refers to the output, which has one more axis than the operands
    axis = normalize_axis(axis, len(first_operand.shape) + 1)
    return OperatorExpression(OperatorStack(axis), operands)


def take(
    source: TensorExpression[TArray] | TArray,
    index: TensorExpression[TArray] | TArray,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = as_expression(source)
    index = as_expression(index)
    if not source.shape:
        raise ValueError("The take operator requires a non-scalar source, but the source has shape ().")
    if not index.shape:
        raise ValueError("The take operator requires an index vector with one axis, but the index has shape ().")
    axis = normalize_axis(axis, len(source.shape))
    return OperatorExpression(OperatorTake(axis), [source, index])


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = as_expression(source)
    axis = normalize_axis(axis, len(source.shape))
    return OperatorExpression(OperatorSlice(start, stop, axis), [source])


def select(
    source: TensorExpression[TArray] | TArray,
    index: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = as_expression(source)
    axis = normalize_axis(axis, len(source.shape))
    return OperatorExpression(OperatorSelect(axis, index), [source])


def softmax(
    a: TensorExpression[TArray] | TArray,
    *,
    axis: int | tuple[int, ...],
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor along the given axis or axes."""

    a = as_expression(a)
    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axes = (axis,) if isinstance(axis, int) else axis
    if not axes:
        raise ValueError("softmax requires at least one axis")
    normalized_axes = tuple(normalize_axis(item, len(a.shape)) for item in axes)
    normalized_axis: int | tuple[int, ...] = normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    return OperatorExpression(OperatorSoftmax(normalized_axis), [a])
