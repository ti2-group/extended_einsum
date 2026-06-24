from dataclasses import dataclass


@dataclass(frozen=True)
class DenseFormat:
    @property
    def sort_key(self) -> str:
        return "dense"


@dataclass(frozen=True)
class DenseLogspaceFormat:
    @property
    def sort_key(self) -> str:
        return "logspace"


@dataclass(frozen=True)
class DenseScaledFormat:
    axis: int

    @property
    def sort_key(self) -> str:
        return f"scaled:{self.axis}"


TensorFormat = DenseFormat | DenseLogspaceFormat | DenseScaledFormat
