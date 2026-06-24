from typing import Literal, Protocol

Shape = tuple[int, ...]

TensorFormat = Literal["dense", "sparse"]

StabilityMode = Literal["none", "scaled", "logspace"]

Backend = Literal["torch", "numpy", "jax"]


class HasShape(Protocol):
    @property
    def shape(self) -> Shape: ...


class HasBackend(Protocol):
    @property
    def backend(self) -> Backend: ...


class HasFormat(Protocol):
    @property
    def format(self) -> TensorFormat: ...
