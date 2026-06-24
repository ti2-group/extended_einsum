from collections import defaultdict
from dataclasses import dataclass, field

from extended_einsum.language.core import ArgumentSSAIds, RawProgram
from extended_einsum.language.rich_operators import RichOperator
from extended_einsum.language.types import Shape, StabilityMode, TensorFormat

RichInstruction = tuple[RichOperator, ArgumentSSAIds]


@dataclass(frozen=True)
class RichProgram:
    instructions: list[RichInstruction]
    n_inputs: int
    stability_mode: StabilityMode

    tensor_formats: list[TensorFormat]
    shapes: list[Shape]

    parameter_indices: frozenset[int]

    arguments_of_ssa_id: dict[int, ArgumentSSAIds] = field(init=False)
    consumers_of_ssa_id: dict[int, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        n_ssa_ids = self.n_inputs + len(self.instructions)

        if len(self.shapes) != n_ssa_ids:
            raise ValueError(
                f"Number of shapes ({len(self.shapes)}) must match the expected number of SSA IDs ({self.n_inputs} inputs + {len(self.instructions)} instructions)."
            )
        if len(self.tensor_formats) != n_ssa_ids:
            raise ValueError(
                f"Number of tensor formats ({len(self.tensor_formats)}) must match the expected number of SSA IDs ({self.n_inputs} inputs + {len(self.instructions)} instructions)."
            )

        # map from ssa id to its arguments
        arguments_of_ssa_id: dict[int, ArgumentSSAIds] = defaultdict(tuple)
        for i, (_, arguments) in enumerate(self.instructions):
            arguments_of_ssa_id[self.n_inputs + i] = arguments

        # for each ssa id remember the ssa ids where it is used as an argument
        consumers_of_ssa_id: dict[int, list[int]] = defaultdict(list)
        for i, (_, arguments) in enumerate(self.instructions):
            for argument in arguments:
                # self.n_inputs + i is the ssa id that the instruction writes to
                consumers_of_ssa_id[argument].append(self.n_inputs + i)

        # overwrite the generated fields
        object.__setattr__(self, "arguments_of_ssa_id", arguments_of_ssa_id)
        object.__setattr__(self, "consumers_of_ssa_id", consumers_of_ssa_id)

    @property
    def output_ssa(self) -> int:
        return self.n_inputs + len(self.instructions) - 1

    def to_raw_program(self) -> RawProgram:
        return RawProgram(
            instructions=[
                (
                    rich_operator.name,
                    argument_ssa_ids,
                    rich_operator.raw_extra_arguments,
                )
                for rich_operator, argument_ssa_ids in self.instructions
            ],
            n_inputs=self.n_inputs,
        )
