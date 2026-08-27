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

Backend = str


class HasShape(Protocol):
    @property
    def shape(self) -> Shape: ...


TArray = TypeVar("TArray", bound=HasShape)
