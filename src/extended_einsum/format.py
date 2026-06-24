from dataclasses import dataclass
from typing import Generic, override

from extended_einsum.backend import TBackendArray, get_backend_of_array
from extended_einsum.language.types import Array, Backend, TensorFormat


@dataclass(frozen=True)
class DenseArray(Array, Generic[TBackendArray]):
    _backend_array: TBackendArray

    @property
    def backend_array(self) -> TBackendArray:
        return self._backend_array

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend_array.shape)

    @property
    @override
    def format(self) -> TensorFormat:
        return "dense"

    @property
    @override
    def backend(self) -> Backend:
        return get_backend_of_array(self._backend_array)


@dataclass(frozen=True)
class SparseArray(Array, Generic[TBackendArray]):
    _backend_array: TBackendArray

    @property
    def backend_array(self) -> TBackendArray:
        return self._backend_array

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend_array.shape)

    @property
    @override
    def format(self) -> TensorFormat:
        return "sparse"

    @property
    @override
    def backend(self) -> Backend:
        return get_backend_of_array(self._backend_array)
