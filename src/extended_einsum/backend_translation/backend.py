from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar, cast

from extended_einsum.utils import normalize_axis


class BackendArray(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


class BackendFunctions(ABC, Generic[TBackendArray]):
    """The array operations a backend must provide to execute programs.

    The abstract methods are the primitives every backend has to implement.
    The remaining methods have default implementations composed from the
    primitives or from the standard Python operator and indexing protocols of
    the array type; override them when the backend offers a faster or more
    precise native version (e.g. a fused softmax), or when its arrays do not
    support the standard operators.

    ``exp``, ``log``, ``einsum``, ``stack``, ``concat``, ``take``, ``select``,
    ``slice``, ``softmax``, ``add``, ``subtract``, ``multiply``, and ``divide``
    execute the corresponding user-facing operators. ``sum``, ``max``, ``min``,
    ``maximum``, ``reshape``, ``broadcast_to``, and ``stop_gradient`` are
    additionally required by the stable evaluation strategies (the scaled and
    logspace stability modes).
    """

    @abstractmethod
    def exp(self, array: TBackendArray) -> TBackendArray: ...

    @abstractmethod
    def log(self, array: TBackendArray) -> TBackendArray: ...

    @abstractmethod
    def sum(self, array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @abstractmethod
    def max(self, array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @abstractmethod
    def min(self, array: TBackendArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> TBackendArray: ...

    @abstractmethod
    def maximum(self, array_1: TBackendArray, array_2: TBackendArray) -> TBackendArray: ...

    @abstractmethod
    def reshape(self, array: TBackendArray, shape: tuple[int, ...]) -> TBackendArray: ...

    @abstractmethod
    def broadcast_to(self, array: TBackendArray, shape: tuple[int, ...]) -> TBackendArray: ...

    @abstractmethod
    def stack(self, arrays: Sequence[TBackendArray], axis: int) -> TBackendArray: ...

    @abstractmethod
    def concat(self, arrays: Sequence[TBackendArray], axis: int) -> TBackendArray: ...

    @abstractmethod
    def take(self, array: TBackendArray, indices: TBackendArray, axis: int) -> TBackendArray: ...

    @abstractmethod
    def einsum(self, format_string: str, *operands: TBackendArray) -> TBackendArray: ...

    def stop_gradient(self, array: TBackendArray) -> TBackendArray:
        """Detaches the array from gradient tracking; the identity for backends without automatic differentiation."""

        return array

    def select(self, array: TBackendArray, axis: int, index: int) -> TBackendArray:
        normalized_axis = normalize_axis(axis, len(array.shape))
        item: list[slice | int] = [slice(None)] * len(array.shape)
        item[normalized_axis] = index
        return cast(TBackendArray, cast(Any, array)[tuple(item)])

    def slice(self, array: TBackendArray, start: int, stop: int, axis: int) -> TBackendArray:
        normalized_axis = normalize_axis(axis, len(array.shape))
        slices = [slice(None)] * len(array.shape)
        slices[normalized_axis] = slice(start, stop)
        return array[tuple(slices)]  # pyright: ignore[reportIndexIssue]

    def softmax(self, array: TBackendArray, axis: int | tuple[int, ...]) -> TBackendArray:
        shifted = self.subtract(array, self.max(array, axis=axis, keepdims=True))
        exp_array = self.exp(shifted)
        return self.divide(exp_array, self.sum(exp_array, axis=axis, keepdims=True))

    def add(self, summand_array_1: TBackendArray, summand_array_2: TBackendArray) -> TBackendArray:
        return summand_array_1 + summand_array_2  # pyright: ignore[reportOperatorIssue]

    def subtract(self, minuend_array: TBackendArray, subtrahend_array: TBackendArray) -> TBackendArray:
        return minuend_array - subtrahend_array  # pyright: ignore[reportOperatorIssue]

    def multiply(self, factor_array_1: TBackendArray, factor_array_2: TBackendArray) -> TBackendArray:
        return factor_array_1 * factor_array_2  # pyright: ignore[reportOperatorIssue]

    def divide(self, dividend_array: TBackendArray, divisor_array: TBackendArray) -> TBackendArray:
        return dividend_array / divisor_array  # pyright: ignore[reportOperatorIssue]


@dataclass(frozen=True)
class BackendProgram(Generic[TBackendArray]):
    backend_calls: list[Callable[[Sequence[TBackendArray]], TBackendArray]]
    call_arguments: list[tuple[int, ...]]
    n_inputs: int


class BackendCompiler(Protocol[TBackendArray]):
    def compile(
        self,
        program: BackendProgram[TBackendArray],
        inputs: Sequence[TBackendArray],
    ) -> Callable[[Sequence[TBackendArray]], TBackendArray]: ...
