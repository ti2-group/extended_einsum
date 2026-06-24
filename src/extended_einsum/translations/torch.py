from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import torch

from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.format import DenseArray
from extended_einsum.language.core import RawProgram
from extended_einsum.runtime import run_program
from extended_einsum.utils import normalize_axis

# jax.tree_util.register_dataclass(
#     ScaledTensor,
#     data_fields=["value", "log_scale"],
#     meta_fields=["scale_axis"],
# )


class TorchTranslation(BackendFunctions[DenseArray[torch.Tensor]]):
    @override
    @staticmethod
    def exp(array: DenseArray[torch.Tensor]) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.exp(array.backend_array))

    @override
    @staticmethod
    def log(array: DenseArray[torch.Tensor]) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.log(array.backend_array))

    @override
    @staticmethod
    def sum(array: DenseArray[torch.Tensor], axis: int) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.sum(array.backend_array, dim=axis))

    @override
    @staticmethod
    def max(array: DenseArray[torch.Tensor], axis: int) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.max(array.backend_array, dim=axis))  # pyright: ignore[reportArgumentType]

    @override
    @staticmethod
    def stack(
        arrays: Sequence[DenseArray[torch.Tensor]], axis: int
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(
            torch.stack([array.backend_array for array in arrays], dim=axis)
        )

    @override
    @staticmethod
    def take(
        array: DenseArray[torch.Tensor], indices: DenseArray[torch.Tensor], axis: int
    ) -> DenseArray[torch.Tensor]:
        # return torch.take(array, indices, dim=axis)
        raise NotImplementedError("Torch doesn't support take with an axis.")

    @override
    @staticmethod
    def select(
        array: DenseArray[torch.Tensor], axis: int, index: int
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.select(array.backend_array, dim=axis, index=index))

    @override
    @staticmethod
    def slice(
        array: DenseArray[torch.Tensor], start: int, stop: int, axis: int
    ) -> DenseArray[torch.Tensor]:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.backend_array.ndim
        slices[normalized_axis] = slice(start, stop)
        return DenseArray(array.backend_array[tuple(slices)])

    @override
    @staticmethod
    def softmax(array: DenseArray[torch.Tensor], axis: int) -> DenseArray[torch.Tensor]:
        return DenseArray(torch.softmax(array.backend_array, dim=axis))

    @override
    @staticmethod
    def einsum(
        format_string: str, *operands: DenseArray[torch.Tensor]
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(
            torch.einsum(
                format_string, *[operand.backend_array for operand in operands]
            )
        )

    @override
    @staticmethod
    def add(
        summand_array_1: DenseArray[torch.Tensor],
        summand_array_2: DenseArray[torch.Tensor],
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(summand_array_1.backend_array + summand_array_2.backend_array)

    @override
    @staticmethod
    def subtract(
        minuend_array: DenseArray[torch.Tensor],
        subtrahend_array: DenseArray[torch.Tensor],
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(minuend_array.backend_array - subtrahend_array.backend_array)

    @override
    @staticmethod
    def multiply(
        factor_array_1: DenseArray[torch.Tensor],
        factor_array_2: DenseArray[torch.Tensor],
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(factor_array_1.backend_array * factor_array_2.backend_array)

    @override
    @staticmethod
    def divide(
        dividend_array: DenseArray[torch.Tensor],
        divisor_array: DenseArray[torch.Tensor],
    ) -> DenseArray[torch.Tensor]:
        return DenseArray(dividend_array.backend_array / divisor_array.backend_array)


class TorchCompiler(BackendCompiler[torch.Tensor]):
    @override
    @staticmethod
    def compile(
        program: RawProgram,
        arguments: Sequence[torch.Tensor],
        backend_functions_per_instruction: list[BackendFunctions[torch.Tensor]],
    ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        return torch.compile(  # pyright: ignore[reportReturnType] - this is just because torch.Shape isn't exactly a tuple
            partial(
                run_program,
                program,
                backend_functions_per_instruction=backend_functions_per_instruction,  # pyright: ignore[reportArgumentType]
            )
        )
