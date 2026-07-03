from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import torch

from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions, BackendProgram
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.utils import normalize_axis


class TorchBackendFunctions(BackendFunctions[torch.Tensor]):
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
    def sum(array: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        if axis is None:
            return torch.sum(array)
        return torch.sum(array, dim=axis)

    @override
    @staticmethod
    def max(array: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        if axis is None:
            return torch.max(array)
        return torch.amax(array, dim=axis)

    @override
    @staticmethod
    def min(array: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        if axis is None:
            return torch.min(array)
        return torch.amin(array, dim=axis)

    @override
    @staticmethod
    def stack(arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(list(arrays), dim=axis)

    @override
    @staticmethod
    def take(array: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.index_select(array, dim=axis, index=indices)

    @override
    @staticmethod
    def select(array: torch.Tensor, axis: int, index: int) -> torch.Tensor:
        return torch.select(array, dim=axis, index=index)

    @override
    @staticmethod
    def slice(array: torch.Tensor, start: int, stop: int, axis: int) -> torch.Tensor:
        normalized_axis = normalize_axis(axis, array.ndim)
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
    def add(summand_array_1: torch.Tensor, summand_array_2: torch.Tensor) -> torch.Tensor:
        return summand_array_1 + summand_array_2

    @override
    @staticmethod
    def subtract(minuend_array: torch.Tensor, subtrahend_array: torch.Tensor) -> torch.Tensor:
        return minuend_array - subtrahend_array

    @override
    @staticmethod
    def multiply(factor_array_1: torch.Tensor, factor_array_2: torch.Tensor) -> torch.Tensor:
        return factor_array_1 * factor_array_2

    @override
    @staticmethod
    def divide(dividend_array: torch.Tensor, divisor_array: torch.Tensor) -> torch.Tensor:
        return dividend_array / divisor_array


class TorchCompiler(BackendCompiler[torch.Tensor]):
    @override
    @staticmethod
    def compile(
        program: BackendProgram[torch.Tensor],
        inputs: Sequence[torch.Tensor],
    ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        if len(inputs) != program.n_inputs:
            raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")
        return torch.compile(partial(run_program, program))
