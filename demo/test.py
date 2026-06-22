from dataclasses import dataclass
from typing import Any, Protocol, override

from extended_einsum.language.core import OperatorName, RawInstruction


class Operator(Protocol):
    @property
    def name(self) -> OperatorName: ...

    @property
    def raw_extra_arguments(self) -> tuple[Any, ...]: ...

    def check_inputs(self, operands: list[Any]) -> bool: ...

    def to_instruction(self, operand_ids: tuple[int, ...]) -> RawInstruction:
        return (self.name, operand_ids, self.raw_extra_arguments)


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


def main():
    test = TestImplementation()
    test.tell(test.ask())
    print(test.value)


if __name__ == "__main__":
    main()
