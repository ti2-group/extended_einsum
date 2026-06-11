from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from exteinsum.language import (
    EINSUM_OPERATOR,
    SCALED_EINSUM_OPERATORS,
    SLICE_OPERATOR,
    SOFTMAX_OPERATOR,
    STACK_OPERATOR,
    TAKE_OPERATOR,
    BinaryOperator,
    Program,
    UnaryOperator,
    einsum_format,
    instruction_arguments,
    instruction_operator,
    normalize_axis,
    scaled_einsum_output_axis,
    slice_axis,
    slice_start,
    slice_stop,
    softmax_axis,
    take_axis,
)
from exteinsum.translations._scaled import (
    BackendOps,
    binary_value,
    normal_einsum,
    scaled_einsum,
    slice_value,
    softmax_value,
    stack_values,
    take_value,
    unary_value,
)

UNARY_OPERATOR_TO_NUMPY: dict[UnaryOperator, Callable[[npt.NDArray], npt.NDArray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
}

BINARY_OPERATOR_TO_NUMPY: dict[
    BinaryOperator, Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "**": np.pow,
}


def numpy_einsum_helper(
    format_string: str, operands: Sequence[npt.NDArray]
) -> npt.NDArray:
    return np.einsum(format_string, *operands)


def einsum_to_numpy(
    format_string: str,
) -> Callable[[Sequence[npt.NDArray]], npt.NDArray]:
    return partial(numpy_einsum_helper, format_string)


def _numpy_softmax(value: npt.NDArray, axis: int) -> npt.NDArray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp_value = np.exp(shifted)
    return exp_value / np.sum(exp_value, axis=axis, keepdims=True)


def _numpy_slice(source: npt.NDArray, axis: int, start: int, stop: int) -> npt.NDArray:
    normalized_axis = normalize_axis(axis, len(source.shape))
    slices = [slice(None)] * source.ndim
    slices[normalized_axis] = slice(start, stop)
    return source[tuple(slices)]


NUMPY_OPS = BackendOps(
    exp=np.exp,
    log=np.log,
    sum=np.sum,
    max=np.max,
    stack=lambda values: np.stack(list(values), axis=0),
    take=lambda source, index, axis: np.take(source, index, axis=axis),
    slice=_numpy_slice,
    softmax=_numpy_softmax,
    reshape=np.reshape,
    einsum=lambda format_string, operands: np.einsum(format_string, *operands),
)


def execute_program_numpy(program: Program, inputs: Sequence[Any]) -> Any:
    tensors: list[Any] = list(inputs)
    for instruction in program.instructions:
        operator = instruction_operator(instruction)
        arguments = instruction_arguments(instruction)
        if operator == STACK_OPERATOR:
            result = stack_values(
                [tensors[argument] for argument in arguments], NUMPY_OPS
            )
        elif operator == TAKE_OPERATOR:
            result = take_value(
                tensors[arguments[0]],
                tensors[arguments[1]],
                take_axis(instruction),
                NUMPY_OPS,
            )
        elif operator == SLICE_OPERATOR:
            result = slice_value(
                tensors[arguments[0]],
                slice_axis(instruction),
                slice_start(instruction),
                slice_stop(instruction),
                NUMPY_OPS,
            )
        elif operator == SOFTMAX_OPERATOR:
            result = softmax_value(
                tensors[arguments[0]],
                softmax_axis(instruction),
                NUMPY_OPS,
            )
        elif operator in UNARY_OPERATOR_TO_NUMPY:
            result = unary_value(
                operator,
                tensors[arguments[0]],
                UNARY_OPERATOR_TO_NUMPY[cast(UnaryOperator, operator)],
                NUMPY_OPS,
            )
        elif operator in BINARY_OPERATOR_TO_NUMPY:
            result = binary_value(
                operator,
                tensors[arguments[0]],
                tensors[arguments[1]],
                BINARY_OPERATOR_TO_NUMPY[cast(BinaryOperator, operator)],
            )
        elif operator == EINSUM_OPERATOR:
            result = normal_einsum(
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                NUMPY_OPS,
            )
        elif operator in SCALED_EINSUM_OPERATORS:
            result = scaled_einsum(
                cast(Any, operator),
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                scaled_einsum_output_axis(instruction),
                NUMPY_OPS,
            )
        else:
            raise ValueError(f"unsupported instruction operator: {operator!r}")
        tensors.append(result)
    return tensors[-1]
