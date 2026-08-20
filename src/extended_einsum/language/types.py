from typing import Literal, Protocol, TypeVar

Shape = tuple[int, ...]

TensorFormat = Literal["dense", "sparse"]

StabilityMode = Literal[
    "unstable",
    "scaled_min",
    "scaled_max",
    "scaled_sum",
    "logspace_min",
    "logspace_max",
]

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


class Array(HasShape, HasBackend, HasFormat, Protocol): ...


TArray = TypeVar("TArray", bound=Array)
