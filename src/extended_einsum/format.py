from dataclasses import dataclass


@dataclass(frozen=True)
class DenseFormat:
    pass


@dataclass(frozen=True)
class DenseLogspaceFormat:
    pass


@dataclass(frozen=True)
class DenseScaledFormat:
    axis: int


TensorFormat = DenseFormat | DenseLogspaceFormat | DenseScaledFormat
