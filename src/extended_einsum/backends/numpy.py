from collections.abc import Callable, Sequence
from functools import partial
from typing import override

import numpy as np
import numpy.typing as npt

from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions, BackendProgram
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.utils import normalize_axis


class NumpyBackendFunctions(BackendFunctions[npt.NDArray]):
    @override
    @staticmethod
    def exp(array: npt.NDArray) -> npt.NDArray:
        return np.exp(array)

    @override
    @staticmethod
    def log(array: npt.NDArray) -> npt.NDArray:
        return np.log(array)

    @override
    @staticmethod
    def sum(array: npt.NDArray, axis: int | None = None) -> npt.NDArray:
        return np.sum(array, axis=axis)

    @override
    @staticmethod
    def max(array: npt.NDArray, axis: int | None = None) -> npt.NDArray:
        return np.max(array, axis=axis)

    @override
    @staticmethod
    def stack(arrays: Sequence[npt.NDArray], axis: int) -> npt.NDArray:
        return np.stack(list(arrays), axis=axis)

    @override
    @staticmethod
    def take(array: npt.NDArray, indices: npt.NDArray, axis: int) -> npt.NDArray:
        return np.take(array, indices, axis=axis)

    @override
    @staticmethod
    def select(array: npt.NDArray, axis: int, index: int) -> npt.NDArray:
        return np.take(array, index, axis=axis)

    @override
    @staticmethod
    def slice(array: npt.NDArray, start: int, stop: int, axis: int) -> npt.NDArray:
        normalized_axis = normalize_axis(axis, array.ndim)
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]

    @override
    @staticmethod
    def softmax(array: npt.NDArray, axis: int) -> npt.NDArray:
        shifted = array - np.max(array, axis=axis, keepdims=True)
        exp_array = np.exp(shifted)
        return exp_array / np.sum(exp_array, axis=axis, keepdims=True)

    @override
    @staticmethod
    def einsum(format_string: str, *operands: npt.NDArray) -> npt.NDArray:
        return np.einsum(format_string, *operands)

    @override
    @staticmethod
    def add(summand_array_1: npt.NDArray, summand_array_2: npt.NDArray) -> npt.NDArray:
        return summand_array_1 + summand_array_2

    @override
    @staticmethod
    def subtract(minuend_array: npt.NDArray, subtrahend_array: npt.NDArray) -> npt.NDArray:
        return minuend_array - subtrahend_array

    @override
    @staticmethod
    def multiply(factor_array_1: npt.NDArray, factor_array_2: npt.NDArray) -> npt.NDArray:
        return factor_array_1 * factor_array_2

    @override
    @staticmethod
    def divide(dividend_array: npt.NDArray, divisor_array: npt.NDArray) -> npt.NDArray:
        return dividend_array / divisor_array


class NumpyCompiler(BackendCompiler[npt.NDArray]):
    @override
    @staticmethod
    def compile(
        program: BackendProgram[npt.NDArray],
        inputs: Sequence[npt.NDArray],
    ) -> Callable[[Sequence[npt.NDArray]], npt.NDArray]:
        # numpy has no JIT compilation, so the program is simply interpreted
        return partial(run_program, program)
