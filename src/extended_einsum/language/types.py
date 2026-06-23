from typing import Literal, Protocol

Shape = tuple[int, ...]

TensorFormat = Literal["dense", "sparse"]

StabilityMode = Literal["none", "scaled", "logspace"]


class HasShape(Protocol):
    @property
    def shape(self) -> Shape: ...
