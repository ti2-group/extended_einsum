from extended_einsum.backend import DenseArray, TArray, TBackendArray
from extended_einsum.interface.operator import (
    InterfaceEinsumOperator,
    InterfaceSliceOperator,
    InterfaceSoftmaxOperator,
    InterfaceStackOperator,
    InterfaceTakeOperator,
    InterfaceUnaryOperator,
)
from extended_einsum.interface.tensor_expression import TensorExpression
from extended_einsum.utils import normalize_axis, parse_format_string


def array(a: TBackendArray, is_parameter: bool = False) -> DenseArray[TBackendArray]:
    return DenseArray(a, is_parameter=is_parameter)


def exp(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(InterfaceUnaryOperator("exp"), [a])


def log(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(InterfaceUnaryOperator("log"), [a])


def einsum(
    format_string: str,
    *operands: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    index_strings, output_string = parse_format_string(format_string)
    if len(index_strings) != len(operands):
        raise ValueError(
            f"format string {format_string} has {len(index_strings)} indices, but {len(operands)} operands."
        )
    all_input_symbols = frozenset("".join(index_strings))
    if any(output_symbol not in all_input_symbols for output_symbol in output_string):
        raise ValueError(
            f"format string {format_string} contains output symbols that are not present in the operands."
        )
    return TensorExpression(InterfaceEinsumOperator(format_string), list(operands))


def stack(
    operands: list[TensorExpression[TArray] | TArray],
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(operands[0].shape))
    if len(operands) == 0:
        raise ValueError("stack requires at least one argument")
    if any(operand.shape != operands[0].shape for operand in operands[1:]):
        raise ValueError(
            "The stack operator requires all arguments to have the same shape along the stack axis."
        )
    return TensorExpression(InterfaceStackOperator(axis), operands)


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
    return TensorExpression(InterfaceTakeOperator(axis), [source, index])


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(InterfaceSliceOperator(start, stop, axis), [source])


def softmax(
    a: TensorExpression[TArray] | TArray,
    axis: int = 0,
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor."""

    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axis = normalize_axis(axis, len(a.shape))
    return TensorExpression(InterfaceSoftmaxOperator(axis), [a])
