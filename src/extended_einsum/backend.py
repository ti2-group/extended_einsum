from collections.abc import Sequence
from typing import Callable, Literal, Protocol, TypeVar

import jax
import numpy as np
import torch

from extended_einsum.language import Program

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
    def stack(arrays: list[TArray], axis: int = 0) -> TArray: ...

    @staticmethod
    def take(array: TArray, indices: TArray, axis: int = 0) -> TArray: ...

    @staticmethod
    def slice(array: TArray, start: int, stop: int, axis: int = 0) -> TArray: ...

    @staticmethod
    def softmax(array: TArray, axis: int = 0) -> TArray: ...

    @staticmethod
    def einsum(format_string: str, *operands: TArray) -> TArray: ...


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
