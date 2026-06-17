from collections.abc import Sequence
from typing import override

import jax
import jax.numpy as jnp

from extended_einsum.backend import (
    BackendFunctions,
    DenseArray,
    LogSpaceArray,
    MultiFormatBackendFunctions,
    ScaledArray,
    SingleFormatBackendFunctions,
)

jax.tree_util.register_dataclass(
    ScaledArray[jax.Array],
    data_fields=["backend_array", "log_scale"],
    meta_fields=["scale_axis", "is_parameter"],
)


class JaxDenseImplementation(
    SingleFormatBackendFunctions[DenseArray[jax.Array]],
    MultiFormatBackendFunctions[
        DenseArray[jax.Array], DenseArray[jax.Array], DenseArray[jax.Array]
    ],
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
    def softmax(array: DenseArray[jax.Array], axis: int = 0) -> DenseArray[jax.Array]:
        return DenseArray(jax.nn.softmax(array.backend_array, axis=axis))

    @override
    @staticmethod
    def stack(
        arrays: Sequence[DenseArray[jax.Array]], axis: int = 0
    ) -> DenseArray[jax.Array]:
        return DenseArray(
            jnp.stack([array.backend_array for array in arrays], axis=axis)
        )

    @override
    @staticmethod
    def slice(
        array: DenseArray[jax.Array], start: int, stop: int, axis: int = 0
    ) -> DenseArray[jax.Array]:
        slices = [slice(None)] * array.backend_array.ndim
        slices[axis] = slice(start, stop)
        return DenseArray(array.backend_array[tuple(slices)])

    @override
    @staticmethod
    def take(
        array: DenseArray[jax.Array], indices: DenseArray[jax.Array], axis: int = 0
    ) -> DenseArray[jax.Array]:
        return DenseArray(jnp.take(array.backend_array, indices.backend_array, axis))

    @override
    @staticmethod
    def einsum(
        format_string: str,
        operand_1: DenseArray[jax.Array],
        operand_2: DenseArray[jax.Array],
    ) -> DenseArray[jax.Array]:
        return DenseArray(
            jnp.einsum(format_string, operand_1.backend_array, operand_2.backend_array)
        )

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


class JaxScaledImplementation(
    SingleFormatBackendFunctions[ScaledArray[jax.Array]],
    MultiFormatBackendFunctions[
        ScaledArray[jax.Array], ScaledArray[jax.Array], ScaledArray[jax.Array]
    ],
):
    @override
    @staticmethod
    def exp(array: ScaledArray[jax.Array]) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def log(array: ScaledArray[jax.Array]) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def softmax(array: ScaledArray[jax.Array], axis: int = 0) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def stack(
        arrays: Sequence[ScaledArray[jax.Array]], axis: int = 0
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def take(
        array: ScaledArray[jax.Array], indices: ScaledArray[jax.Array], axis: int = 0
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def slice(
        array: ScaledArray[jax.Array], start: int, stop: int, axis: int = 0
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def einsum(
        format_string: str,
        operand_1: ScaledArray[jax.Array],
        operand_2: ScaledArray[jax.Array],
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def add(
        summand_array_1: ScaledArray[jax.Array],
        summand_array_2: ScaledArray[jax.Array],
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def subtract(
        minuend_array: ScaledArray[jax.Array],
        subtrahend_array: ScaledArray[jax.Array],
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def multiply(
        factor_array_1: ScaledArray[jax.Array],
        factor_array_2: ScaledArray[jax.Array],
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def divide(
        dividend_array: ScaledArray[jax.Array],
        divisor_array: ScaledArray[jax.Array],
    ) -> ScaledArray[jax.Array]:
        raise NotImplementedError()


class JaxLogspaceImplementation(
    SingleFormatBackendFunctions[LogSpaceArray[jax.Array]],
    MultiFormatBackendFunctions[
        LogSpaceArray[jax.Array], LogSpaceArray[jax.Array], LogSpaceArray[jax.Array]
    ],
):
    @override
    @staticmethod
    def exp(array: LogSpaceArray[jax.Array]) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def log(array: LogSpaceArray[jax.Array]) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def softmax(
        array: LogSpaceArray[jax.Array], axis: int = 0
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def stack(
        arrays: Sequence[LogSpaceArray[jax.Array]], axis: int = 0
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def take(
        array: LogSpaceArray[jax.Array],
        indices: LogSpaceArray[jax.Array],
        axis: int = 0,
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def slice(
        array: LogSpaceArray[jax.Array], start: int, stop: int, axis: int = 0
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def einsum(
        format_string: str,
        operand_1: LogSpaceArray[jax.Array],
        operand_2: LogSpaceArray[jax.Array],
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def add(
        summand_array_1: LogSpaceArray[jax.Array],
        summand_array_2: LogSpaceArray[jax.Array],
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def subtract(
        minuend_array: LogSpaceArray[jax.Array],
        subtrahend_array: LogSpaceArray[jax.Array],
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def multiply(
        factor_array_1: LogSpaceArray[jax.Array],
        factor_array_2: LogSpaceArray[jax.Array],
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()

    @override
    @staticmethod
    def divide(
        dividend_array: LogSpaceArray[jax.Array],
        divisor_array: LogSpaceArray[jax.Array],
    ) -> LogSpaceArray[jax.Array]:
        raise NotImplementedError()


# def extract_signature(array: jax.Array | ScaledTensor[jax.Array]) -> Any:
#     if isinstance(array, ScaledTensor):
#         return ScaledTensor(
#             extract_signature(array.value),
#             extract_signature(array.log_scale),
#             array.scale_axis,
#         )
#     return jax.ShapeDtypeStruct(array.shape, array.dtype)


JAX_BACKEND_FUNCTIONS = BackendFunctions[jax.Array](
    unary_dense_only=JaxDenseImplementation(),
    binary_dense_only=JaxDenseImplementation(),
    unary_logspace_only=JaxLogspaceImplementation(),
    binary_logspace_only=JaxLogspaceImplementation(),
    unary_scaled_only=JaxScaledImplementation(),
    binary_scaled_only=JaxScaledImplementation(),
    binary_dense_scaled=JaxScaledImplementation(),
    binary_scaled_dense=JaxScaledImplementation(),
    binary_logspace_dense=JaxLogspaceImplementation(),
    binary_dense_logspace=JaxLogspaceImplementation(),
)


# class JaxCompiler(BackendCompiler[jax.Array]):
#     @override
#     @staticmethod
#     def compile(
#         program: Program,
#         arguments: Sequence[jax.Array],
#         backend_implementations: list[
#             SingleFormatBackendFunctions | MultiFormatBackendFunctions
#         ],
#     ) -> Callable[[Sequence[jax.Array]], jax.Array]:
#         argument_signatures = [extract_signature(argument) for argument in arguments]
#         jit_prepared = jax.jit(
#             partial(
#                 run_program, program, backend_implementations=backend_implementations
#             )
#         )
#         return jit_prepared.trace(argument_signatures).lower().compile()
