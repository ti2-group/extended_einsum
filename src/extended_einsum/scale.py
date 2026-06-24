from dataclasses import dataclass
from typing import Generic

from extended_einsum.backend import BackendFunctions, TBackendArray
from extended_einsum.utils import normalize_axis


@dataclass(frozen=True)
class ScaledTensor(Generic[TBackendArray]):
    value: TBackendArray
    log_scale: TBackendArray
    scale_axis: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scale_axis",
            normalize_axis(self.scale_axis, len(self.shape)),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def actual_value(self, translation: BackendFunctions) -> TBackendArray:
        return self.value * translation.exp(self.log_scale)
