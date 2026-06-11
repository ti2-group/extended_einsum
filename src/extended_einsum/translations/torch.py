from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import torch
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

UNARY_OPERATOR_TO_TORCH: dict[UnaryOperator, Callable[[torch.Tensor], torch.Tensor]] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
}

BINARY_OPERATOR_TO_TORCH: dict[
    BinaryOperator, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "+": torch.add,
    "-": torch.sub,
    "*": torch.mul,
    "/": torch.div,
    "**": torch.pow,
}


def torch_einsum_helper(
    format_string: str, operands: Sequence[torch.Tensor]
) -> torch.Tensor:
    return torch.einsum(format_string, *operands)


def einsum_to_torch(
    format_string: str,
) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
    return partial(torch_einsum_helper, format_string)


def _torch_take(source: torch.Tensor, index: torch.Tensor, axis: int) -> torch.Tensor:
    take_axis = normalize_axis(axis, len(source.shape))
    result = torch.index_select(source, dim=take_axis, index=index.reshape(-1))
    return result.reshape(
        (*source.shape[:take_axis], *index.shape, *source.shape[take_axis + 1 :])
    )


def _torch_slice(
    source: torch.Tensor, axis: int, start: int, stop: int
) -> torch.Tensor:
    normalized_axis = normalize_axis(axis, len(source.shape))
    slices = [slice(None)] * source.ndim
    slices[normalized_axis] = slice(start, stop)
    return source[tuple(slices)]


TORCH_OPS = BackendOps(
    exp=torch.exp,
    log=torch.log,
    sum=torch.sum,
    max=torch.amax,
    stack=lambda values: torch.stack(list(values), dim=0),
    take=_torch_take,
    slice=_torch_slice,
    softmax=lambda value, axis: torch.softmax(value, dim=axis),
    reshape=torch.reshape,
    einsum=lambda format_string, operands: torch.einsum(format_string, *operands),
)


def execute_program_torch(program: Program, inputs: Sequence[Any]) -> Any:
    tensors: list[Any] = list(inputs)
    for instruction in program.instructions:
        operator = instruction_operator(instruction)
        arguments = instruction_arguments(instruction)
        if operator == STACK_OPERATOR:
            result = stack_values(
                [tensors[argument] for argument in arguments], TORCH_OPS
            )
        elif operator == TAKE_OPERATOR:
            result = take_value(
                tensors[arguments[0]],
                tensors[arguments[1]],
                take_axis(instruction),
                TORCH_OPS,
            )
        elif operator == SLICE_OPERATOR:
            result = slice_value(
                tensors[arguments[0]],
                slice_axis(instruction),
                slice_start(instruction),
                slice_stop(instruction),
                TORCH_OPS,
            )
        elif operator == SOFTMAX_OPERATOR:
            result = softmax_value(
                tensors[arguments[0]],
                softmax_axis(instruction),
                TORCH_OPS,
            )
        elif operator in UNARY_OPERATOR_TO_TORCH:
            result = unary_value(
                operator,
                tensors[arguments[0]],
                UNARY_OPERATOR_TO_TORCH[cast(UnaryOperator, operator)],
                TORCH_OPS,
            )
        elif operator in BINARY_OPERATOR_TO_TORCH:
            result = binary_value(
                operator,
                tensors[arguments[0]],
                tensors[arguments[1]],
                BINARY_OPERATOR_TO_TORCH[cast(BinaryOperator, operator)],
            )
        elif operator == EINSUM_OPERATOR:
            result = normal_einsum(
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                TORCH_OPS,
            )
        elif operator in SCALED_EINSUM_OPERATORS:
            result = scaled_einsum(
                cast(Any, operator),
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                scaled_einsum_output_axis(instruction),
                TORCH_OPS,
            )
        else:
            raise ValueError(f"unsupported instruction operator: {operator!r}")
        tensors.append(result)
    return tensors[-1]


def compile_program_torch(
    program: Program,
) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
    return torch.compile(partial(execute_program_torch, program))
