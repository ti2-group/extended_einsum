from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
import torch

from extended_einsum.language.types import Backend, HasShape


class BackendArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...] | torch.Size: ...


TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


class BackendFunctions(Protocol[TBackendArray]):
    @staticmethod
    def stop_gradient(array: TBackendArray) -> TBackendArray: ...
    # Paper "Detached reference shifts" (sec:numerical-stability): translations
    # call this on every shift/normalizer whose derivative cancels exactly.

    @staticmethod
    def exp(array: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def log(array: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def sum(array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @staticmethod
    def max(array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @staticmethod
    def min(array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @staticmethod
    def maximum(array_1: TBackendArray, array_2: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def reshape(array: TBackendArray, shape: tuple[int, ...]) -> TBackendArray: ...

    @staticmethod
    def broadcast_to(array: TBackendArray, shape: tuple[int, ...]) -> TBackendArray: ...

    @staticmethod
    def stack(arrays: Sequence[TBackendArray], axis: int) -> TBackendArray: ...

    @staticmethod
    def concat(arrays: Sequence[TBackendArray], axis: int) -> TBackendArray: ...

    @staticmethod
    def take(array: TBackendArray, indices: TBackendArray, axis: int) -> TBackendArray: ...

    @staticmethod
    def select(array: TBackendArray, axis: int, index: int) -> TBackendArray: ...

    @staticmethod
    def slice(array: TBackendArray, start: int, stop: int, axis: int) -> TBackendArray: ...

    @staticmethod
    def softmax(array: TBackendArray, axis: int | tuple[int, ...]) -> TBackendArray: ...

    @staticmethod
    def einsum(format_string: str, *operands: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def add(summand_array_1: TBackendArray, summand_array_2: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def subtract(minuend_array: TBackendArray, subtrahend_array: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def multiply(factor_array_1: TBackendArray, factor_array_2: TBackendArray) -> TBackendArray: ...

    @staticmethod
    def divide(dividend_array: TBackendArray, divisor_array: TBackendArray) -> TBackendArray: ...


@dataclass(frozen=True)
class BackendProgram(Generic[TBackendArray]):
    backend_calls: list[Callable[[Sequence[TBackendArray]], TBackendArray]]
    call_arguments: list[tuple[int, ...]]
    n_inputs: int


class BackendCompiler(Protocol[TBackendArray]):
    @staticmethod
    def compile(
        program: BackendProgram[TBackendArray],
        inputs: Sequence[TBackendArray],
    ) -> Callable[[Sequence[TBackendArray]], TBackendArray]: ...


def get_backend_of_array(array: HasShape) -> Backend:
    if isinstance(array, torch.Tensor):
        return "torch"
    elif isinstance(array, np.ndarray):
        return "numpy"

    try:
        import jax
    except ModuleNotFoundError:
        jax = None  # type: ignore[assignment]

    if jax is not None and isinstance(array, jax.Array):
        return "jax"

    raise ValueError(f"Unsupported array type: {type(array)}")
