from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import torch

from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.language.core import RawProgram
from extended_einsum.runtime import run_program
from extended_einsum.utils import normalize_axis

# jax.tree_util.register_dataclass(
#     ScaledTensor,
#     data_fields=["value", "log_scale"],
#     meta_fields=["scale_axis"],
# )


class TorchTranslation(BackendFunctions[torch.Tensor]):
    @override
    @staticmethod
    def exp(array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @override
    @staticmethod
    def log(array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @override
    @staticmethod
    def sum(array: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.sum(array, dim=axis)

    @override
    @staticmethod
    def max(array: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.max(array, dim=axis)  # pyright: ignore[reportReturnType]

    @override
    @staticmethod
    def stack(arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)  # pyright: ignore[reportArgumentType]

    @override
    @staticmethod
    def take(array: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        # return torch.take(array, indices, dim=axis)
        raise NotImplementedError("Torch doesn't support take with an axis.")

    @override
    @staticmethod
    def select(array: torch.Tensor, axis: int, index: int) -> torch.Tensor:
        return torch.select(array, dim=axis, index=index)

    @override
    @staticmethod
    def slice(array: torch.Tensor, start: int, stop: int, axis: int) -> torch.Tensor:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]

    @override
    @staticmethod
    def softmax(array: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.softmax(array, dim=axis)

    @override
    @staticmethod
    def einsum(format_string: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(format_string, *operands)

    @override
    @staticmethod
    def add(
        summand_array_1: torch.Tensor, summand_array_2: torch.Tensor
    ) -> torch.Tensor:
        return summand_array_1 + summand_array_2

    @override
    @staticmethod
    def subtract(
        minuend_array: torch.Tensor, subtrahend_array: torch.Tensor
    ) -> torch.Tensor:
        return minuend_array - subtrahend_array

    @override
    @staticmethod
    def multiply(
        factor_array_1: torch.Tensor, factor_array_2: torch.Tensor
    ) -> torch.Tensor:
        return factor_array_1 * factor_array_2

    @override
    @staticmethod
    def divide(
        dividend_array: torch.Tensor, divisor_array: torch.Tensor
    ) -> torch.Tensor:
        return dividend_array / divisor_array


class TorchCompiler(BackendCompiler[torch.Tensor]):
    @override
    @staticmethod
    def compile(
        program: RawProgram, arguments: Sequence[torch.Tensor]
    ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        torch_translation = TorchTranslation()
        return torch.compile(  # pyright: ignore[reportReturnType] - this is just because torch.Shape isn't exactly a tuple
            partial(
                run_program,
                program,
                backend_functions_per_instruction=[  # pyright: ignore[reportArgumentType]
                    torch_translation for _ in program.instructions
                ],
            )
        )
