from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, override

import jax
import jax.numpy as jnp

from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.language import Program
from extended_einsum.runtime import run_program
from extended_einsum.scale import ScaledTensor
from extended_einsum.utils import normalize_axis

jax.tree_util.register_dataclass(
    ScaledTensor,
    data_fields=["value", "log_scale"],
    meta_fields=["scale_axis"],
)


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
    def sum(array: jax.Array, axis: int | None = None) -> jax.Array:
        return jnp.sum(array, axis=axis)

    @override
    @staticmethod
    def max(array: jax.Array, axis: int | None = None) -> jax.Array:
        return jnp.max(array, axis=axis)

    @override
    @staticmethod
    def stack(arrays: Sequence[jax.Array], axis: int = 0) -> jax.Array:
        return jnp.stack(arrays, axis=axis)

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


class JaxScaledTranslation(BackendFunctions[ScaledTensor[jax.Array]]):
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
        arrays: Sequence[ScaledTensor[jax.Array]], axis: int = 0
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

    @override
    @staticmethod
    def add(
        summand_array_1: ScaledTensor[jax.Array],
        summand_array_2: ScaledTensor[jax.Array],
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def subtract(
        minuend_array: ScaledTensor[jax.Array],
        subtrahend_array: ScaledTensor[jax.Array],
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def multiply(
        factor_array_1: ScaledTensor[jax.Array], factor_array_2: ScaledTensor[jax.Array]
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def divide(
        dividend_array: ScaledTensor[jax.Array], divisor_array: ScaledTensor[jax.Array]
    ) -> ScaledTensor[jax.Array]:
        raise NotImplementedError()


def extract_signature(array: jax.Array | ScaledTensor[jax.Array]) -> Any:
    if isinstance(array, ScaledTensor):
        return ScaledTensor(
            extract_signature(array.value),
            extract_signature(array.log_scale),
            array.scale_axis,
        )
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


class JaxCompiler(BackendCompiler[jax.Array]):
    @override
    @staticmethod
    def compile(
        program: Program, arguments: Sequence[jax.Array]
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        argument_signatures = [extract_signature(argument) for argument in arguments]
        jit_prepared = jax.jit(
            partial(run_program, program, backend_functions=JaxTranslation())
        )
        return jit_prepared.trace(argument_signatures).lower().compile()
