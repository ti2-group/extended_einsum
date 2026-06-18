from extended_einsum.backend import Array, TBackendArray, TBackendArrayCovariant
from extended_einsum.interface.operator import (
    InterfaceEinsumOperator,
    InterfaceSliceOperator,
    InterfaceSoftmaxOperator,
    InterfaceStackOperator,
    InterfaceTakeOperator,
    InterfaceUnaryOperator,
)
from extended_einsum.interface.tensor_expression import Parameter, TensorExpression
from extended_einsum.utils import normalize_axis, parse_format_string


def parameter(a: TBackendArrayCovariant) -> Parameter[TBackendArrayCovariant]:
    return Parameter(a)


def exp(
    a: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
) -> TensorExpression[TBackendArray]:
    return TensorExpression(InterfaceUnaryOperator("exp"), [a])


def log(
    a: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
) -> TensorExpression[TBackendArray]:
    return TensorExpression(InterfaceUnaryOperator("log"), [a])


def einsum(
    format_string: str,
    *operands: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
) -> TensorExpression[TBackendArray]:
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
    operands: list[
        TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray
    ],
    *,
    axis: int = 0,
) -> TensorExpression[TBackendArray]:
    axis = normalize_axis(axis, len(operands[0].shape))
    if len(operands) == 0:
        raise ValueError("stack requires at least one argument")
    if any(operand.shape != operands[0].shape for operand in operands[1:]):
        raise ValueError(
            "The stack operator requires all arguments to have the same shape along the stack axis."
        )
    return TensorExpression(InterfaceStackOperator(axis), operands)  # pyright: ignore[reportArgumentType]


def take(
    source: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
    index: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
    *,
    axis: int = 0,
) -> TensorExpression[TBackendArray]:
    axis = normalize_axis(axis, len(source.shape))
    if not source.shape:
        raise ValueError("The take operator requires an operand with a leading axis.")
    if not index.shape:
        raise ValueError("The take operator requires an index with a leading axis.")
    return TensorExpression(InterfaceTakeOperator(axis), [source, index])


def slice(
    source: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TBackendArray]:
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(InterfaceSliceOperator(start, stop, axis), [source])


def softmax(
    a: TensorExpression[TBackendArray] | Array[TBackendArray] | TBackendArray,
    axis: int = 0,
) -> TensorExpression[TBackendArray]:
    """Applies the softmax function to the input tensor."""

    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axis = normalize_axis(axis, len(a.shape))
    return TensorExpression(InterfaceSoftmaxOperator(axis), [a])
