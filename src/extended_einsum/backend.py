from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Protocol, TypeVar, override

import jax
import numpy as np
import torch

Backend = Literal["torch", "numpy", "jax"]


class BackendArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...] | torch.Size: ...


def get_backend_of_array(array: BackendArray) -> Backend:
    match array:
        case np.ndarray:
            return "numpy"
        case torch.Tensor:
            return "torch"
        case jax.Array:
            return "jax"
        case _:
            raise NotImplementedError("Unsupported backend array type")


TBackendArrayCovariant = TypeVar(
    "TBackendArrayCovariant", bound=BackendArray, covariant=True
)
TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


class Array(Protocol[TBackendArrayCovariant]):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def backend_array(self) -> TBackendArrayCovariant: ...

    @property
    def format(self) -> TensorFormat: ...


@dataclass(frozen=True)
class DenseArray(Array[TBackendArrayCovariant]):
    _backend_array: TBackendArrayCovariant

    @property
    @override
    def backend_array(self) -> TBackendArrayCovariant:
        return self._backend_array

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend_array.shape)

    @property
    @override
    def format(self) -> TensorFormat:
        return DenseFormat()


@dataclass(frozen=True)
class LogSpaceArray(Array[TBackendArrayCovariant]):
    _backend_array: TBackendArrayCovariant

    @property
    @override
    def backend_array(self) -> TBackendArrayCovariant:
        return self._backend_array

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend_array.shape)

    @property
    @override
    def format(self) -> TensorFormat:
        return DenseLogspaceFormat()


@dataclass(frozen=True)
class ScaledArray(Array[TBackendArrayCovariant]):
    _backend_array: TBackendArrayCovariant
    log_scale: TBackendArrayCovariant
    scale_axis: int

    @property
    @override
    def backend_array(self) -> TBackendArrayCovariant:
        return self._backend_array

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend_array.shape)

    @property
    @override
    def format(self) -> TensorFormat:
        return DenseScaledFormat(axis=self.scale_axis)


TArray = TypeVar("TArray", bound=Array)
TArrayInput1 = TypeVar("TArrayInput1", bound=Array, contravariant=True)
TArrayInput2 = TypeVar("TArrayInput2", bound=Array, contravariant=True)
TArrayOutput = TypeVar("TArrayOutput", bound=Array, covariant=True)


class SingleFormatBackendFunctions(Protocol[TArray]):
    @staticmethod
    def exp(array: TArray) -> TArray: ...

    @staticmethod
    def log(array: TArray) -> TArray: ...

    @staticmethod
    def softmax(array: TArray, axis: int = 0) -> TArray: ...

    @staticmethod
    def stack(arrays: Sequence[TArray], axis: int = 0) -> TArray: ...

    @staticmethod
    def slice(array: TArray, start: int, stop: int, axis: int = 0) -> TArray: ...

    @staticmethod
    def select(array: TArray, axis: int, index: int) -> TArray: ...


class MultiFormatBackendFunctions(Protocol[TArrayInput1, TArrayInput2, TArrayOutput]):
    @staticmethod
    def take(
        array: TArrayInput1, indices: TArrayInput2, axis: int = 0
    ) -> TArrayOutput: ...

    @staticmethod
    def einsum(
        format_string: str, operand_1: TArrayInput1, operand_2: TArrayInput2
    ) -> TArrayOutput: ...

    @staticmethod
    def add(
        summand_array_1: TArrayInput1, summand_array_2: TArrayInput2
    ) -> TArrayOutput: ...

    @staticmethod
    def subtract(
        minuend_array: TArrayInput1, subtrahend_array: TArrayInput2
    ) -> TArrayOutput: ...

    @staticmethod
    def multiply(
        factor_array_1: TArrayInput1, factor_array_2: TArrayInput2
    ) -> TArrayOutput: ...

    @staticmethod
    def divide(
        dividend_array: TArrayInput1, divisor_array: TArrayInput2
    ) -> TArrayOutput: ...


@dataclass(frozen=True)
class BackendFunctions(Generic[TBackendArrayCovariant]):
    unary_dense_only: SingleFormatBackendFunctions[DenseArray[TBackendArrayCovariant]]
    unary_logspace_only: SingleFormatBackendFunctions[
        LogSpaceArray[TBackendArrayCovariant]
    ]
    unary_scaled_only: SingleFormatBackendFunctions[ScaledArray[TBackendArrayCovariant]]

    binary_dense_only: MultiFormatBackendFunctions[
        DenseArray[TBackendArrayCovariant],
        DenseArray[TBackendArrayCovariant],
        DenseArray[TBackendArrayCovariant],
    ]
    binary_logspace_only: MultiFormatBackendFunctions[
        LogSpaceArray[TBackendArrayCovariant],
        LogSpaceArray[TBackendArrayCovariant],
        LogSpaceArray[TBackendArrayCovariant],
    ]
    binary_scaled_only: MultiFormatBackendFunctions[
        ScaledArray[TBackendArrayCovariant],
        ScaledArray[TBackendArrayCovariant],
        ScaledArray[TBackendArrayCovariant],
    ]
    binary_dense_scaled: MultiFormatBackendFunctions[
        DenseArray[TBackendArrayCovariant],
        ScaledArray[TBackendArrayCovariant],
        ScaledArray[TBackendArrayCovariant],
    ]
    binary_scaled_dense: MultiFormatBackendFunctions[
        ScaledArray[TBackendArrayCovariant],
        DenseArray[TBackendArrayCovariant],
        ScaledArray[TBackendArrayCovariant],
    ]
    binary_logspace_dense: MultiFormatBackendFunctions[
        LogSpaceArray[TBackendArrayCovariant],
        DenseArray[TBackendArrayCovariant],
        LogSpaceArray[TBackendArrayCovariant],
    ]
    binary_dense_logspace: MultiFormatBackendFunctions[
        DenseArray[TBackendArrayCovariant],
        LogSpaceArray[TBackendArrayCovariant],
        LogSpaceArray[TBackendArrayCovariant],
    ]


class BackendCompiler(Protocol[TBackendArray]):
    @staticmethod
    def compile(
        program: Program,
        arguments: Sequence[TBackendArray],
        backend_implementations: list[
            SingleFormatBackendFunctions | MultiFormatBackendFunctions
        ],
    ) -> Callable[[Sequence[TBackendArray]], TBackendArray]: ...
