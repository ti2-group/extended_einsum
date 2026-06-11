from __future__ import annotations

from typing import Any, Generic

from extended_einsum.backend import TArray
from extended_einsum.language import (
    EINSUM_OPERATOR,
    SCALED_EINSUM_MAX_OPERATOR,
    SCALED_EINSUM_OPERATORS,
    SCALED_EINSUM_SUM_OPERATOR,
    SLICE_OPERATOR,
    SOFTMAX_OPERATOR,
    TAKE_OPERATOR,
    Instruction,
    Operator,
    Program,
    make_einsum_instruction,
    make_scaled_einsum_instruction,
    make_slice_instruction,
    make_softmax_instruction,
    make_take_instruction,
    map_instruction_arguments,
)
from extended_einsum.scale import ScaledTensor
from extended_einsum.utils import normalize_axis, parse_format_string, propagate_shapes


class TensorExpression(Generic[TArray]):
    def __init__(
        self,
        operator: Operator,
        arguments: list[TensorExpression[TArray] | TArray | ScaledTensor[TArray]],
        *,
        format_string: str | None = None,
        scale_axis: int | None = None,
        axis: int | None = None,
        slice_start: int | None = None,
        slice_stop: int | None = None,
    ) -> None:
        self.operator = operator
        self.arguments = arguments
        self.format_string = format_string
        self.scale_axis = scale_axis
        self.axis = axis
        self.slice_start = slice_start
        self.slice_stop = slice_stop
        self._shape = propagate_shapes(
            operator,
            [tuple(argument.shape) for argument in arguments],
            format_string=format_string,
            axis=axis,
            slice_start=slice_start,
            slice_stop=slice_stop,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __add__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression("+", [self, other])

    def __sub__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression("-", [self, other])

    def __mul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression("*", [self, other])

    def __truediv__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression("/", [self, other])

    def __pow__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression("**", [self, other])

    def __matmul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return einsum("ik, kj -> ij", self, other)


def sin(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("sin", [a])


def cos(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("cos", [a])


def tan(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("tan", [a])


def exp(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("exp", [a])


def log(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("log", [a])


def sqrt(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression("sqrt", [a])


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
    return TensorExpression(
        EINSUM_OPERATOR, list(operands), format_string=format_string
    )


def scaled_einsum_sum(
    format_string: str,
    lhs: TensorExpression[TArray] | TArray,
    rhs: TensorExpression[TArray] | TArray,
    *,
    scale_axis: int = -1,
) -> TensorExpression[TArray]:
    return TensorExpression(
        SCALED_EINSUM_SUM_OPERATOR,
        [lhs, rhs],
        format_string=format_string,
        scale_axis=scale_axis,
    )


def scaled_einsum_max(
    format_string: str,
    lhs: TensorExpression[TArray] | TArray,
    rhs: TensorExpression[TArray] | TArray,
    *,
    scale_axis: int = -1,
) -> TensorExpression[TArray]:
    return TensorExpression(
        SCALED_EINSUM_MAX_OPERATOR,
        [lhs, rhs],
        format_string=format_string,
        scale_axis=scale_axis,
    )


def take(
    source: TensorExpression[TArray] | TArray,
    index: TensorExpression[TArray] | TArray,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    return TensorExpression(TAKE_OPERATOR, [source, index], axis=axis)


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    return TensorExpression(
        SLICE_OPERATOR,
        [source],
        axis=axis,
        slice_start=start,
        slice_stop=stop,
    )


def softmax(
    a: TensorExpression[TArray] | TArray,
    axis: int | None = None,
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor."""

    if axis is None:
        axis = len(a.shape) - 1

    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axis = normalize_axis(axis, len(a.shape))
    if axis < 0:
        axis += len(a.shape)
    if axis < 0 or axis >= len(a.shape):
        raise ValueError(
            "Axis positions must be between 0 and the number of dimensions of the input tensor."
        )

    return TensorExpression(SOFTMAX_OPERATOR, [a], axis=axis)


def actual_value(value: Any) -> Any:
    if not isinstance(value, ScaledTensor):
        return value
    backend_module = type(value.value).__module__.split(".", maxsplit=1)[0]
    if backend_module == "torch":
        import torch

        return value.value * torch.exp(value.log_scale)
    if backend_module in {"jax", "jaxlib"}:
        import jax.numpy as jnp

        return value.value * jnp.exp(value.log_scale)
    import numpy as np

    return value.value * np.exp(value.log_scale)


def _compile_recursive(
    tensor_expression: TensorExpression[TArray] | TArray,
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[Instruction],
    input_tensors: list[Any],
) -> int:
    # get a unique key for this expression
    expression_key = id(tensor_expression)

    # if the expression is a tensor, just add it to the inputs
    if not isinstance(tensor_expression, TensorExpression):
        if expression_key in input_ssa_ids:
            return input_ssa_ids[expression_key]
        input_ssa_ids[expression_key] = -1 - len(input_ssa_ids)
        input_tensors.append(tensor_expression)
        return input_ssa_ids[expression_key]

    # if we have already seen this expression, return its SSA-ID
    if expression_key in ssa_ids:
        return ssa_ids[expression_key]

    # recursively compile the children first
    argument_ssa_ids = [
        _compile_recursive(
            argument,
            ssa_ids,
            input_ssa_ids,
            instructions,
            input_tensors,
        )
        for argument in tensor_expression.arguments
    ]

    # add the instruction to the program
    ssa_ids[expression_key] = len(ssa_ids)
    if tensor_expression.operator == EINSUM_OPERATOR:
        if tensor_expression.format_string is None or not argument_ssa_ids:
            raise ValueError(
                "explicit einsum requires a format string and at least one operand"
            )
        instructions.append(
            make_einsum_instruction(
                tensor_expression.format_string,
                *argument_ssa_ids,
            )
        )
    elif tensor_expression.operator in SCALED_EINSUM_OPERATORS:
        if (
            tensor_expression.format_string is None
            or tensor_expression.scale_axis is None
            or len(argument_ssa_ids) != 2
        ):
            raise ValueError(
                "scaled einsum requires a format string, scale axis, and two operands"
            )
        instructions.append(
            make_scaled_einsum_instruction(
                tensor_expression.operator,
                tensor_expression.format_string,
                argument_ssa_ids[0],
                argument_ssa_ids[1],
                tensor_expression.scale_axis,
            )
        )
    elif tensor_expression.operator == TAKE_OPERATOR:
        instructions.append(
            make_take_instruction(
                argument_ssa_ids[0],
                argument_ssa_ids[1],
                0 if tensor_expression.axis is None else tensor_expression.axis,
            )
        )
    elif tensor_expression.operator == SLICE_OPERATOR:
        if (
            tensor_expression.axis is None
            or tensor_expression.slice_start is None
            or tensor_expression.slice_stop is None
            or len(argument_ssa_ids) != 1
        ):
            raise ValueError("slice requires an axis, start, stop, and one operand")
        instructions.append(
            make_slice_instruction(
                argument_ssa_ids[0],
                tensor_expression.slice_start,
                tensor_expression.slice_stop,
                tensor_expression.axis,
            )
        )
    elif tensor_expression.operator == SOFTMAX_OPERATOR:
        if tensor_expression.axis is None or len(argument_ssa_ids) != 1:
            raise ValueError("softmax requires an axis and one operand")
        instructions.append(
            make_softmax_instruction(argument_ssa_ids[0], tensor_expression.axis)
        )
    else:
        instructions.append((tensor_expression.operator, tuple(argument_ssa_ids), ()))  # pyright: ignore[reportArgumentType]
    return ssa_ids[expression_key]


def compile(
    tensor_expression: TensorExpression[TArray],
) -> tuple[Program, list[Any]]:
    """Compiles a tensor expression into a program and a list of arguments."""

    instructions: list[Instruction] = []
    ssa_ids: dict[int, int] = {}
    input_ssa_ids: dict[int, int] = {}
    input_tensors: list[Any] = []
    _compile_recursive(
        tensor_expression,
        ssa_ids,
        input_ssa_ids,
        instructions,
        input_tensors,
    )

    n_inputs = len(input_ssa_ids)

    def shift_argument(old_argument: int) -> int:
        return old_argument + n_inputs if old_argument >= 0 else -1 - old_argument

    for i, instruction in enumerate(instructions):
        instructions[i] = map_instruction_arguments(instruction, shift_argument)
    return Program(instructions=instructions, n_inputs=n_inputs), input_tensors
