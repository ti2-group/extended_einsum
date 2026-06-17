from dataclasses import dataclass

from extended_einsum.language import BinaryOperator, UnaryOperator


@dataclass(frozen=True)
class InterfaceUnaryOperator:
    operator: UnaryOperator


@dataclass(frozen=True)
class InterfaceBinaryOperator:
    operator: BinaryOperator


@dataclass(frozen=True)
class InterfaceEinsumOperator:
    format_string: str


@dataclass(frozen=True)
class InterfaceStackOperator:
    axis: int


@dataclass(frozen=True)
class InterfaceTakeOperator:
    axis: int


@dataclass(frozen=True)
class InterfaceSoftmaxOperator:
    axis: int


# @dataclass(frozen=True)
# class InterfaceSumOperator:
#     axis: int


@dataclass(frozen=True)
class InterfaceSliceOperator:
    start: int
    stop: int
    axis: int


@dataclass(frozen=True)
class InterfaceSelectOperator:
    axis: int
    index: int


InterfaceOperator = (
    InterfaceUnaryOperator
    | InterfaceBinaryOperator
    | InterfaceEinsumOperator
    | InterfaceStackOperator
    | InterfaceTakeOperator
    | InterfaceSelectOperator
    | InterfaceSoftmaxOperator
    | InterfaceSliceOperator
)
