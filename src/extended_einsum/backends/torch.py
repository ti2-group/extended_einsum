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
    def stop_gradient(array: torch.Tensor) -> torch.Tensor:
        return array.detach()

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
    def sum(array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            if keepdims:
                return torch.sum(array, dim=tuple(range(array.ndim)), keepdim=True)
            return torch.sum(array)
        return torch.sum(array, dim=axis, keepdim=keepdims)

    @override
    @staticmethod
    def max(array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        # Stable translations use maxima as numerical reference shifts.  The
        # value-selecting max has the same forward result as amax but avoids
        # its costly tie-distributing backward graph.
        if axis is None:
            if not array.ndim:
                return array
            result = torch.max(array.reshape(-1), dim=0).values
            return result.reshape((1,) * array.ndim) if keepdims else result
        axes = (axis,) if isinstance(axis, int) else axis
        normalized_axes = tuple(item if item >= 0 else item + array.ndim for item in axes)
        result = array
        for reduction_axis in normalized_axes:
            result = torch.max(result, dim=reduction_axis, keepdim=True).values
        if not keepdims:
            for reduction_axis in sorted(normalized_axes, reverse=True):
                result = result.squeeze(reduction_axis)
        return result

    @override
    @staticmethod
    def min(array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            if keepdims:
                return torch.amin(array, dim=tuple(range(array.ndim)), keepdim=True)
            return torch.min(array)
        return torch.amin(array, dim=axis, keepdim=keepdims)

    @override
    @staticmethod
    def maximum(array_1: torch.Tensor, array_2: torch.Tensor) -> torch.Tensor:
        return torch.maximum(array_1, array_2)

    @override
    @staticmethod
    def reshape(array: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(array, shape)

    @override
    @staticmethod
    def broadcast_to(array: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(array, shape)

    @override
    @staticmethod
    def stack(arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(list(arrays), dim=axis)

    @override
    @staticmethod
    def concat(arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.cat(list(arrays), dim=axis)

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
    def softmax(array: torch.Tensor, axis: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(axis, int):
            return torch.softmax(array, dim=axis)
        normalized_axes = tuple(item if item >= 0 else item + array.ndim for item in axis)
        if normalized_axes == tuple(range(normalized_axes[0], array.ndim)):
            flattened = torch.flatten(array, start_dim=normalized_axes[0])
            return torch.softmax(flattened, dim=normalized_axes[0]).reshape(array.shape)
        shifted = array - torch.amax(array, dim=axis, keepdim=True)
        exp_array = torch.exp(shifted)
        return exp_array / torch.sum(exp_array, dim=axis, keepdim=True)

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
