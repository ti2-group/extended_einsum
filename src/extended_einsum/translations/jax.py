from collections.abc import Sequence
from functools import partial
from typing import override

import jax
import jax.numpy as jnp

from extended_einsum.backend import BackendCompiler
from extended_einsum.language.core import RawProgram
from extended_einsum.runtime import run_program
from extended_einsum.utils import normalize_axis

# jax.tree_util.register_dataclass(
#     ScaledTensor,
#     data_fields=["value", "log_scale"],
#     meta_fields=["scale_axis"],
# )


class JaxDenseImplementation(
    SingleFormatBackendFunctions[DenseArray[jax.Array]],
    MultiFormatBackendFunctions[
        DenseArray[jax.Array], DenseArray[jax.Array], DenseArray[jax.Array]
    ],
):
    @override
    @staticmethod
    def exp(array: DenseArray[jax.Array]) -> DenseArray[jax.Array]:
        return DenseArray(jnp.exp(array._backend_array))

    @override
    @staticmethod
    def log(array: DenseArray[jax.Array]) -> DenseArray[jax.Array]:
        return DenseArray(jnp.log(array._backend_array))

    @override
    @staticmethod
    def softmax(array: DenseArray[jax.Array], axis: int = 0) -> DenseArray[jax.Array]:
        return DenseArray(jax.nn.softmax(array._backend_array, axis=axis))

    @override
    @staticmethod
    def stack(
        arrays: Sequence[DenseArray[jax.Array]], axis: int = 0
    ) -> DenseArray[jax.Array]:
        return DenseArray(
            jnp.stack([array._backend_array for array in arrays], axis=axis)
        )

    @override
    @staticmethod
    def slice(
        array: DenseArray[jax.Array], start: int, stop: int, axis: int = 0
    ) -> DenseArray[jax.Array]:
        slices = [slice(None)] * array._backend_array.ndim
        slices[axis] = slice(start, stop)
        return DenseArray(array._backend_array[tuple(slices)])

    @override
    @staticmethod
    def take(
        array: DenseArray[jax.Array], indices: DenseArray[jax.Array], axis: int = 0
    ) -> DenseArray[jax.Array]:
        return DenseArray(jnp.take(array._backend_array, indices._backend_array, axis))

    @override
    @staticmethod
    def select(array: jax.Array, axis: int, index: int) -> jax.Array:
        return jnp.take(array, index, axis=axis)

    @override
    @staticmethod
    def slice(array: jax.Array, start: int, stop: int, axis: int = 0) -> jax.Array:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]

    @override
    @staticmethod
    def add(
        summand_array_1: DenseArray[jax.Array],
        summand_array_2: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(
            summand_array_1._backend_array + summand_array_2._backend_array
        )

    @override
    @staticmethod
    def subtract(
        minuend_array: DenseArray[jax.Array],
        subtrahend_array: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(
            minuend_array._backend_array - subtrahend_array._backend_array
        )

    @override
    @staticmethod
    def multiply(
        factor_array_1: DenseArray[jax.Array],
        factor_array_2: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(factor_array_1._backend_array * factor_array_2._backend_array)

    @override
    @staticmethod
    def divide(
        dividend_array: DenseArray[jax.Array],
        divisor_array: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(dividend_array._backend_array / divisor_array._backend_array)


class JaxScaledImplementation(
    SingleFormatBackendFunctions[ScaledArray[jax.Array]],
    MultiFormatBackendFunctions[
        ScaledArray[jax.Array], ScaledArray[jax.Array], ScaledArray[jax.Array]
    ],
):
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
        program: RawProgram, arguments: Sequence[jax.Array]
    ) -> Callable[[Sequence[jax.Array]], jax.Array]:
        jax_translation = JaxTranslation()
        jit_prepared = jax.jit(
            partial(
                run_program,
                program,
                backend_functions_per_instruction=[  # pyright: ignore[reportArgumentType]
                    jax_translation for _ in program.instructions
                ],
            )
        )
        return jit_prepared.trace(arguments).lower().compile()
