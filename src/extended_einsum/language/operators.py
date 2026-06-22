from dataclasses import dataclass
from typing import Any, override

from extended_einsum.language.core import Operator, OperatorName


@dataclass(frozen=True)
class OperatorAdd(Operator):
    @property
    @override
    def name(self) -> OperatorName:
        return "+"

    @property
    @override
    def raw_extra_arguments(self) -> tuple[Any, ...]:
        return ()

    @override
    def check_inputs(self, operands: list[Any]) -> bool:
        return len(operands) == 2
