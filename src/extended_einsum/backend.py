from typing import Protocol, TypeVar

import torch


class Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...] | torch.Size: ...


TArray = TypeVar("TArray", bound=Array)


class BackendTranslation(Protocol[TArray]):
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
