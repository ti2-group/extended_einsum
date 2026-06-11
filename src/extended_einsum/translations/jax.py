from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast, override

import jax
import jax.numpy as jnp

from extended_einsum.backend import BackendTranslation
from extended_einsum.language import (
    EINSUM_OPERATOR,
    SCALED_EINSUM_OPERATORS,
    SLICE_OPERATOR,
    SOFTMAX_OPERATOR,
    STACK_OPERATOR,
    TAKE_OPERATOR,
    BinaryOperator,
    Program,
    UnaryOperator,
    slice_axis,
    slice_start,
    slice_stop,
    softmax_axis,
)
from extended_einsum.scale import ScaledTensor

# from extended_einsum.translations._scaled import (
#     binary_value,
#     normal_einsum,
#     scaled_einsum,
#     slice_value,
#     softmax_value,
#     stack_values,
#     take_value,
#     unary_value,
# )
from extended_einsum.utils import normalize_axis

jax.tree_util.register_dataclass(
    ScaledTensor,
    data_fields=["value", "log_scale"],
    meta_fields=["scale_axis"],
)

UNARY_OPERATOR_TO_JAX: dict[UnaryOperator, Callable[[jax.Array], jax.Array]] = {
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
}

BINARY_OPERATOR_TO_JAX: dict[
    BinaryOperator, Callable[[jax.Array, jax.Array], jax.Array]
] = {
    "+": jnp.add,
    "-": jnp.subtract,
    "*": jnp.multiply,
    "/": jnp.divide,
    "**": jnp.pow,
}


class JaxTranslation(BackendTranslation[jax.Array]):
    @override
    @staticmethod
    def exp(array: jax.Array) -> jax.Array:
        return jnp.exp(array)

    @override
    @staticmethod
    def log(array: jax.Array) -> jax.Array:
        return jnp.log(array)

    @override
    @staticmethod
    def sum(array: jax.Array, axis: int | None = None) -> jax.Array:
        return jnp.sum(array, axis=axis)

    @override
    @staticmethod
    def max(array: jax.Array, axis: int | None = None) -> jax.Array:
        return jnp.max(array, axis=axis)

    @override
    @staticmethod
    def stack(arrays: list[jax.Array], axis: int = 0) -> jax.Array:
        return jnp.stack(list(arrays), axis=axis)

    @override
    @staticmethod
    def take(array: jax.Array, indices: jax.Array, axis: int = 0) -> jax.Array:
        return jnp.take(array, indices, axis=axis)

    @override
    @staticmethod
    def slice(array: jax.Array, start: int, stop: int, axis: int = 0) -> jax.Array:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]

    @override
    @staticmethod
    def softmax(array: jax.Array, axis: int = 0) -> jax.Array:
        return jax.nn.softmax(array, axis=axis)

    @override
    @staticmethod
    def einsum(format_string: str, *operands: jax.Array) -> jax.Array:
        return jnp.einsum(format_string, *operands)


class JaxScaledTranslation(BackendTranslation[ScaledTensor[jax.Array]]):
    @override
    @staticmethod
    def exp(array: ScaledTensor[jax.Array]) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def log(array: ScaledTensor[jax.Array]) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def sum(
        array: ScaledTensor[jax.Array], axis: int | None = None
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def max(
        array: ScaledTensor[jax.Array], axis: int | None = None
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def stack(
        arrays: list[ScaledTensor[jax.Array]], axis: int = 0
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def take(
        array: ScaledTensor[jax.Array], indices: ScaledTensor[jax.Array], axis: int = 0
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def slice(
        array: ScaledTensor[jax.Array], start: int, stop: int, axis: int = 0
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def softmax(
        array: ScaledTensor[jax.Array], axis: int = 0
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def einsum(
        format_string: str, *operands: ScaledTensor[jax.Array]
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()


def execute_program_jax(program: Program, inputs: Sequence[Any]) -> Any:
    tensors: list[Any] = list(inputs)
    for instruction in program.instructions:
        operator = instruction_operator(instruction)
        arguments = instruction_arguments(instruction)
        if operator == STACK_OPERATOR:
            result = stack_values(
                [tensors[argument] for argument in arguments], JAX_OPS
            )
        elif operator == TAKE_OPERATOR:
            result = take_value(
                tensors[arguments[0]],
                tensors[arguments[1]],
                take_axis(instruction),
                JAX_OPS,
            )
        elif operator == SLICE_OPERATOR:
            result = slice_value(
                tensors[arguments[0]],
                slice_axis(instruction),
                slice_start(instruction),
                slice_stop(instruction),
                JAX_OPS,
            )
        elif operator == SOFTMAX_OPERATOR:
            result = softmax_value(
                tensors[arguments[0]],
                softmax_axis(instruction),
                JAX_OPS,
            )
        elif operator in UNARY_OPERATOR_TO_JAX:
            result = unary_value(
                operator,
                tensors[arguments[0]],
                UNARY_OPERATOR_TO_JAX[cast(UnaryOperator, operator)],
                JAX_OPS,
            )
        elif operator in BINARY_OPERATOR_TO_JAX:
            result = binary_value(
                operator,
                tensors[arguments[0]],
                tensors[arguments[1]],
                BINARY_OPERATOR_TO_JAX[cast(BinaryOperator, operator)],
            )
        elif operator == EINSUM_OPERATOR:
            result = normal_einsum(
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                JAX_OPS,
            )
        elif operator in SCALED_EINSUM_OPERATORS:
            result = scaled_einsum(
                cast(Any, operator),
                einsum_format(instruction),
                [tensors[argument] for argument in arguments],
                scaled_einsum_output_axis(instruction),
                JAX_OPS,
            )
        else:
            raise ValueError(f"unsupported instruction operator: {operator!r}")
        tensors.append(result)
    return tensors[-1]


def compile_program_jax(
    program: Program,
    argument_signatures: Sequence[jax.ShapeDtypeStruct | ScaledTensor[Any]]
    | None = None,
) -> Callable[[Sequence[Any]], Any]:
    jit_prepared = jax.jit(partial(execute_program_jax, program))
    if argument_signatures is None:
        return jit_prepared
    return jit_prepared.trace(argument_signatures).lower().compile()


def extract_signature(array: jax.Array | ScaledTensor[jax.Array]) -> Any:
    if isinstance(array, ScaledTensor):
        return ScaledTensor(
            extract_signature(array.value),
            extract_signature(array.log_scale),
            array.scale_axis,
        )
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def execute_program_sliced_jax(
    program: Program, inputs: Sequence[jax.Array], sliced_indices: list[str]
) -> jax.Array: ...
