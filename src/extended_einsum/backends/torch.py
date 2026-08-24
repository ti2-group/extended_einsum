from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import torch

from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions, BackendProgram
from extended_einsum.backend_translation.runtime import run_program


class TorchBackendFunctions(BackendFunctions[torch.Tensor]):
    @override
    def stop_gradient(self, array: torch.Tensor) -> torch.Tensor:
        return array.detach()

    @override
    def exp(self, array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @override
    def log(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @override
    def sum(self, array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            if keepdims:
                return torch.sum(array, dim=tuple(range(array.ndim)), keepdim=True)
            return torch.sum(array)
        return torch.sum(array, dim=axis, keepdim=keepdims)

    @override
    def max(self, array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            if keepdims:
                return torch.amax(
                    array,
                    dim=tuple(range(array.ndim)),
                    keepdim=True,
                )
            return torch.amax(array)
        return torch.amax(array, dim=axis, keepdim=keepdims)

    @override
    def min(self, array: torch.Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            if keepdims:
                return torch.amin(array, dim=tuple(range(array.ndim)), keepdim=True)
            return torch.min(array)
        return torch.amin(array, dim=axis, keepdim=keepdims)

    @override
    def maximum(self, array_1: torch.Tensor, array_2: torch.Tensor) -> torch.Tensor:
        return torch.maximum(array_1, array_2)

    @override
    def reshape(self, array: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(array, shape)

    @override
    def broadcast_to(self, array: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(array, shape)

    @override
    def stack(self, arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(list(arrays), dim=axis)

    @override
    def concat(self, arrays: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.cat(list(arrays), dim=axis)

    @override
    def take(self, array: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.index_select(array, dim=axis, index=indices)

    @override
    def softmax(self, array: torch.Tensor, axis: int | tuple[int, ...]) -> torch.Tensor:
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
    def einsum(self, format_string: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(format_string, *operands)


class TorchCompiler(BackendCompiler[torch.Tensor]):
    def compile(
        self,
        program: BackendProgram[torch.Tensor],
        inputs: Sequence[torch.Tensor],
    ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        if len(inputs) != program.n_inputs:
            raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")
        return torch.compile(partial(run_program, program))
