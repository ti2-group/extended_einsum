from collections.abc import Sequence
from typing import Callable, Literal, Protocol, TypeVar

import jax
import numpy as np
import torch

Backend = Literal["torch", "numpy", "jax"]


class Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...] | torch.Size: ...


TArray = TypeVar("TArray", bound=Array)


class BackendFunctions(Protocol[TArray]):
    @staticmethod
    def exp(array: TArray) -> TArray: ...

    @staticmethod
    def log(array: TArray) -> TArray: ...

    @staticmethod
    def sum(array: TArray, axis: int | None = None) -> TArray: ...

    @staticmethod
    def max(array: TArray, axis: int | None = None) -> TArray: ...

    @staticmethod
    def stack(arrays: Sequence[TArray], axis: int = 0) -> TArray: ...

    @staticmethod
    def take(array: TArray, indices: TArray, axis: int = 0) -> TArray: ...

    @staticmethod
    def slice(array: TArray, start: int, stop: int, axis: int = 0) -> TArray: ...

    @staticmethod
    def softmax(array: TArray, axis: int = 0) -> TArray: ...

    @staticmethod
    def einsum(format_string: str, *operands: TArray) -> TArray: ...

    @staticmethod
    def add(summand_array_1: TArray, summand_array_2: TArray) -> TArray: ...

    @staticmethod
    def subtract(minuend_array: TArray, subtrahend_array: TArray) -> TArray: ...

    @staticmethod
    def multiply(factor_array_1: TArray, factor_array_2: TArray) -> TArray: ...

    @staticmethod
    def divide(dividend_array: TArray, divisor_array: TArray) -> TArray: ...


class BackendCompiler(Protocol[TArray]):
    @staticmethod
    def compile(
        program: Program, arguments: Sequence[TArray]
    ) -> Callable[[Sequence[TArray]], TArray]: ...


def get_backend_of_array(array: Array) -> Backend:
    if isinstance(array, torch.Tensor):
        return "torch"
    elif isinstance(array, np.ndarray):
        return "numpy"
    elif isinstance(array, jax.Array):
        return "jax"
    else:
        raise ValueError(f"Unsupported array type: {type(array)}")
