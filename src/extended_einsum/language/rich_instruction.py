from dataclasses import dataclass
from typing import Callable

from extended_einsum.language.core import ArgumentSSAIds
from extended_einsum.language.rich_operators import RichOperator


@dataclass(frozen=True)
class RichInstruction:
    operator: RichOperator
    argument_ssa_ids: ArgumentSSAIds


def map_instruction_arguments(instruction: RichInstruction, shift_argument: Callable[[int], int]) -> RichInstruction:
    return RichInstruction(
        operator=instruction.operator,
        argument_ssa_ids=tuple(map(shift_argument, instruction.argument_ssa_ids)),
    )
