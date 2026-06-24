from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import jax
import jax.numpy as jnp

from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.language.core import RawProgram
from extended_einsum.runtime import run_program
from extended_einsum.utils import normalize_axis

# jax.tree_util.register_dataclass(
#     ScaledTensor,
#     data_fields=["value", "log_scale"],
#     meta_fields=["scale_axis"],
# )


class JaxTranslation(BackendFunctions[jax.Array]):
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
    def sum(array: jax.Array, axis: int) -> jax.Array:
        return jnp.sum(array, axis=axis)

    @override
    @staticmethod
    def max(array: jax.Array, axis: int) -> jax.Array:
        return jnp.max(array, axis=axis)

    @override
    @staticmethod
    def stack(arrays: Sequence[jax.Array], axis: int) -> jax.Array:
        return jnp.stack(arrays, axis=axis)

    @override
    @staticmethod
    def take(array: jax.Array, indices: jax.Array, axis: int) -> jax.Array:
        return jnp.take(array, indices, axis=axis)

    @override
    @staticmethod
    def select(array: jax.Array, axis: int, index: int) -> jax.Array:
        return jnp.take(array, index, axis=axis)

    @override
    @staticmethod
    def slice(array: jax.Array, start: int, stop: int, axis: int) -> jax.Array:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]

    @override
    @staticmethod
    def softmax(array: jax.Array, axis: int) -> jax.Array:
        return jax.nn.softmax(array, axis=axis)

    @override
    @staticmethod
    def einsum(format_string: str, *operands: jax.Array) -> jax.Array:
        return jnp.einsum(format_string, *operands)

    @override
    @staticmethod
    def add(summand_array_1: jax.Array, summand_array_2: jax.Array) -> jax.Array:
        return summand_array_1 + summand_array_2

    @override
    @staticmethod
    def subtract(minuend_array: jax.Array, subtrahend_array: jax.Array) -> jax.Array:
        return minuend_array - subtrahend_array

    @override
    @staticmethod
    def multiply(factor_array_1: jax.Array, factor_array_2: jax.Array) -> jax.Array:
        return factor_array_1 * factor_array_2

    @override
    @staticmethod
    def divide(dividend_array: jax.Array, divisor_array: jax.Array) -> jax.Array:
        return dividend_array / divisor_array


class JaxCompiler(BackendCompiler[jax.Array]):
    @override
    @staticmethod
    def compile(
        program: RawProgram,
        arguments: Sequence[jax.Array],
        backend_functions_per_instruction: list[BackendFunctions[jax.Array]],
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        jit_prepared = jax.jit(
            partial(
                run_program,
                program,
                backend_functions_per_instruction=backend_functions_per_instruction,  # pyright: ignore[reportArgumentType]
            )
        )
        return jit_prepared.trace(arguments).lower().compile()
