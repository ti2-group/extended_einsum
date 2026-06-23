from typing import Literal, Protocol

Shape = tuple[int, ...]

TensorFormat = Literal["dense", "sparse"]


class HasShape(Protocol):
    @property
    def shape(self) -> Shape: ...
