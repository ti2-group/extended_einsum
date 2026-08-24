from collections.abc import Sequence
from typing import override

import numpy as np
import numpy.typing as npt

from extended_einsum.backend_translation.backend import BackendFunctions
from extended_einsum.backend_translation.runtime import DefaultCompiler


class NumpyBackendFunctions(BackendFunctions[npt.NDArray]):
    """Reference backend implementing only the required primitives; everything else uses the BackendFunctions defaults."""

    @override
    def exp(self, array: npt.NDArray) -> npt.NDArray:
        return np.exp(array)

    @override
    def log(self, array: npt.NDArray) -> npt.NDArray:
        return np.log(array)

    @override
    def sum(self, array: npt.NDArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> npt.NDArray:
        return np.sum(array, axis=axis, keepdims=keepdims)

    @override
    def max(self, array: npt.NDArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> npt.NDArray:
        return np.max(array, axis=axis, keepdims=keepdims)

    @override
    def min(self, array: npt.NDArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> npt.NDArray:
        return np.min(array, axis=axis, keepdims=keepdims)

    @override
    def maximum(self, array_1: npt.NDArray, array_2: npt.NDArray) -> npt.NDArray:
        return np.maximum(array_1, array_2)

    @override
    def reshape(self, array: npt.NDArray, shape: tuple[int, ...]) -> npt.NDArray:
        return np.reshape(array, shape)

    @override
    def broadcast_to(self, array: npt.NDArray, shape: tuple[int, ...]) -> npt.NDArray:
        return np.broadcast_to(array, shape)

    @override
    def stack(self, arrays: Sequence[npt.NDArray], axis: int) -> npt.NDArray:
        return np.stack(list(arrays), axis=axis)

    @override
    def concat(self, arrays: Sequence[npt.NDArray], axis: int) -> npt.NDArray:
        return np.concatenate(list(arrays), axis=axis)

    @override
    def take(self, array: npt.NDArray, indices: npt.NDArray, axis: int) -> npt.NDArray:
        return np.take(array, indices, axis=axis)

    @override
    def einsum(self, format_string: str, *operands: npt.NDArray) -> npt.NDArray:
        return np.einsum(format_string, *operands)


class NumpyCompiler(DefaultCompiler[npt.NDArray]):
    """numpy has no JIT compilation, so the program is simply interpreted."""
