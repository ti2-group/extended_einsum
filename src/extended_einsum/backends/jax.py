from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import jax
import jax.numpy as jnp

from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions, BackendProgram
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.utils import normalize_axis


class JaxBackendFunctions(BackendFunctions[jax.Array]):
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
    def max(array: jax.Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> jax.Array:
        return jnp.max(array, axis=axis, keepdims=keepdims)

    @override
    @staticmethod
    def min(array: jax.Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> jax.Array:
        return jnp.min(array, axis=axis, keepdims=keepdims)

    @override
    @staticmethod
    def reshape(array: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.reshape(array, shape)

    @override
    @staticmethod
    def stack(arrays: Sequence[jax.Array], axis: int) -> jax.Array:
        return jnp.stack(list(arrays), axis=axis)

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
        normalized_axis = normalize_axis(axis, array.ndim)
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
        program: BackendProgram[jax.Array],
        inputs: Sequence[jax.Array],
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        if len(inputs) != program.n_inputs:
            raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")
        jit_prepared = jax.jit(partial(run_program, program))
        return jit_prepared.trace(list(inputs)).lower().compile()
