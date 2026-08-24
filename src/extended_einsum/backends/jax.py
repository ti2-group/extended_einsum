from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import jax
import jax.numpy as jnp

from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions, BackendProgram
from extended_einsum.backend_translation.runtime import run_program


class JaxBackendFunctions(BackendFunctions[jax.Array]):
    @override
    def stop_gradient(self, array: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(array)

    @override
    def exp(self, array: jax.Array) -> jax.Array:
        return jnp.exp(array)

    @override
    def log(self, array: jax.Array) -> jax.Array:
        return jnp.log(array)

    @override
    def sum(self, array: jax.Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> jax.Array:
        return jnp.sum(array, axis=axis, keepdims=keepdims)

    @override
    def max(self, array: jax.Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> jax.Array:
        return jnp.max(array, axis=axis, keepdims=keepdims)

    @override
    def min(self, array: jax.Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> jax.Array:
        return jnp.min(array, axis=axis, keepdims=keepdims)

    @override
    def maximum(self, array_1: jax.Array, array_2: jax.Array) -> jax.Array:
        return jnp.maximum(array_1, array_2)

    @override
    def reshape(self, array: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.reshape(array, shape)

    @override
    def broadcast_to(self, array: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.broadcast_to(array, shape)

    @override
    def stack(self, arrays: Sequence[jax.Array], axis: int) -> jax.Array:
        return jnp.stack(list(arrays), axis=axis)

    @override
    def concat(self, arrays: Sequence[jax.Array], axis: int) -> jax.Array:
        return jnp.concatenate(list(arrays), axis=axis)

    @override
    def take(self, array: jax.Array, indices: jax.Array, axis: int) -> jax.Array:
        return jnp.take(array, indices, axis=axis)

    @override
    def softmax(self, array: jax.Array, axis: int | tuple[int, ...]) -> jax.Array:
        return jax.nn.softmax(array, axis=axis)

    @override
    def einsum(self, format_string: str, *operands: jax.Array) -> jax.Array:
        return jnp.einsum(format_string, *operands)


class JaxCompiler(BackendCompiler[jax.Array]):
    @override
    def compile(
        self,
        program: BackendProgram[jax.Array],
        inputs: Sequence[jax.Array],
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        if len(inputs) != program.n_inputs:
            raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")
        jit_prepared = jax.jit(partial(run_program, program))
        return jit_prepared.trace(list(inputs)).lower().compile()
