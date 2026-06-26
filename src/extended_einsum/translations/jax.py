from collections.abc import Sequence
from functools import partial
from typing import Callable, override

import jax
import jax.numpy as jnp

from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.format import DenseArray, SparseArray
from extended_einsum.language.core import RawProgram
from extended_einsum.runtime import run_program
from extended_einsum.utils import normalize_axis

# jax.tree_util.register_dataclass(
#     ScaledTensor,
#     data_fields=["value", "log_scale"],
#     meta_fields=["scale_axis"],
# )


class JaxDenseUnstableImplementation(
    BackendFunctions[DenseArray[jax.Array]],
):
    @override
    @staticmethod
    def exp(array: DenseArray[jax.Array]) -> DenseArray[jax.Array]:
        return DenseArray(jnp.exp(array.backend_array))

    @override
    @staticmethod
    def log(array: DenseArray[jax.Array]) -> DenseArray[jax.Array]:
        return DenseArray(jnp.log(array.backend_array))

    @override
    @staticmethod
    def sum(array: DenseArray[jax.Array], axis: int) -> DenseArray[jax.Array]:
        return DenseArray(jnp.sum(array.backend_array, axis=axis))

    @override
    @staticmethod
    def max(array: DenseArray[jax.Array], axis: int) -> DenseArray[jax.Array]:
        return DenseArray(jnp.max(array.backend_array, axis=axis))

    @override
    @staticmethod
    def stack(arrays: Sequence[DenseArray[jax.Array]], axis: int) -> DenseArray[jax.Array]:
        return DenseArray(jnp.stack([array.backend_array for array in arrays], axis=axis))

    @override
    @staticmethod
    def take(array: DenseArray[jax.Array], indices: DenseArray[jax.Array], axis: int) -> DenseArray[jax.Array]:
        return DenseArray(jnp.take(array.backend_array, indices.backend_array, axis=axis))

    @override
    @staticmethod
    def select(array: DenseArray[jax.Array], axis: int, index: int) -> DenseArray[jax.Array]:
        return DenseArray(jnp.take(array.backend_array, index, axis=axis))

    @override
    @staticmethod
    def slice(array: DenseArray[jax.Array], start: int, stop: int, axis: int) -> DenseArray[jax.Array]:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.backend_array.ndim
        slices[normalized_axis] = slice(start, stop)
        return DenseArray(array.backend_array[tuple(slices)])

    @override
    @staticmethod
    def softmax(array: DenseArray[jax.Array], axis: int) -> DenseArray[jax.Array]:
        return DenseArray(jax.nn.softmax(array.backend_array, axis=axis))

    @override
    @staticmethod
    def einsum(format_string: str, *operands: DenseArray[jax.Array]) -> DenseArray[jax.Array]:
        return DenseArray(jnp.einsum(format_string, *[operand.backend_array for operand in operands]))

    @override
    @staticmethod
    def add(
        summand_array_1: DenseArray[jax.Array],
        summand_array_2: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(summand_array_1.backend_array + summand_array_2.backend_array)

    @override
    @staticmethod
    def subtract(
        minuend_array: DenseArray[jax.Array],
        subtrahend_array: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(minuend_array.backend_array - subtrahend_array.backend_array)

    @override
    @staticmethod
    def multiply(
        factor_array_1: DenseArray[jax.Array],
        factor_array_2: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(factor_array_1.backend_array * factor_array_2.backend_array)

    @override
    @staticmethod
    def divide(
        dividend_array: DenseArray[jax.Array],
        divisor_array: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(dividend_array.backend_array / divisor_array.backend_array)


class JaxCompiler(BackendCompiler[DenseArray[jax.Array] | SparseArray[jax.Array]]):
    @override
    @staticmethod
    def compile(
        program: RawProgram,
        arguments: Sequence[DenseArray[jax.Array] | SparseArray[jax.Array]],
        backend_functions_per_instruction: list[BackendFunctions[DenseArray[jax.Array] | SparseArray[jax.Array]]],
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        jit_prepared = jax.jit(
            partial(
                run_program,
                program,
                backend_functions_per_instruction=backend_functions_per_instruction,  # pyright: ignore[reportArgumentType]
            )
        )
        return jit_prepared.trace(arguments).lower().compile()
