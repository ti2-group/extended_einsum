from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Protocol, TypeVar

import torch

from extended_einsum.format import (
    DenseFormat,
    DenseLogspaceFormat,
    DenseScaledFormat,
    TensorFormat,
)
from extended_einsum.language import Program

Backend = Literal["torch", "numpy", "jax"]


class BackendArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...] | torch.Size: ...


TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


class Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def format(self) -> TensorFormat: ...

    @property
    def is_parameter(self) -> bool: ...


@dataclass(frozen=True)
class DenseArray(Generic[TBackendArray]):
    backend_array: TBackendArray
    is_parameter: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.backend_array.shape)

    @property
    def format(self) -> TensorFormat:
        return DenseFormat()


@dataclass(frozen=True)
class LogSpaceArray(Generic[TBackendArray]):
    backend_array: TBackendArray
    is_parameter: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.backend_array.shape)

    @property
    def format(self) -> TensorFormat:
        return DenseLogspaceFormat()


@dataclass(frozen=True)
class ScaledArray(Generic[TBackendArray]):
    backend_array: TBackendArray
    log_scale: TBackendArray
    scale_axis: int
    is_parameter: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.backend_array.shape)

    @property
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
class BackendFunctions(Generic[TBackendArray]):
    unary_dense_only: SingleFormatBackendFunctions[DenseArray[TBackendArray]]
    unary_logspace_only: SingleFormatBackendFunctions[LogSpaceArray[TBackendArray]]
    unary_scaled_only: SingleFormatBackendFunctions[ScaledArray[TBackendArray]]

    binary_dense_only: MultiFormatBackendFunctions[
        DenseArray[TBackendArray], DenseArray[TBackendArray], DenseArray[TBackendArray]
    ]
    binary_logspace_only: MultiFormatBackendFunctions[
        LogSpaceArray[TBackendArray],
        LogSpaceArray[TBackendArray],
        LogSpaceArray[TBackendArray],
    ]
    binary_scaled_only: MultiFormatBackendFunctions[
        ScaledArray[TBackendArray],
        ScaledArray[TBackendArray],
        ScaledArray[TBackendArray],
    ]
    binary_dense_scaled: MultiFormatBackendFunctions[
        DenseArray[TBackendArray],
        ScaledArray[TBackendArray],
        ScaledArray[TBackendArray],
    ]
    binary_scaled_dense: MultiFormatBackendFunctions[
        ScaledArray[TBackendArray],
        DenseArray[TBackendArray],
        ScaledArray[TBackendArray],
    ]
    binary_logspace_dense: MultiFormatBackendFunctions[
        LogSpaceArray[TBackendArray],
        DenseArray[TBackendArray],
        LogSpaceArray[TBackendArray],
    ]
    binary_dense_logspace: MultiFormatBackendFunctions[
        DenseArray[TBackendArray],
        LogSpaceArray[TBackendArray],
        LogSpaceArray[TBackendArray],
    ]


class BackendCompiler(Protocol[TBackendArray]):
    @staticmethod
    def compile(
        program: Program,
        arguments: Sequence[TBackendArray],
    ) -> Callable[[Sequence[TBackendArray]], TBackendArray]: ...
