from collections.abc import Sequence
from typing import TypeVar

from extended_einsum.backend_translation.backend import BackendArray, BackendProgram

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


def run_program(
    program: BackendProgram[TBackendArray],
    inputs: Sequence[TBackendArray],
) -> TBackendArray:
    tensors: list[TBackendArray] = list(inputs)
    for backend_call, argument_ids in zip(program.backend_calls, program.call_arguments):
        argument_tensors = [tensors[argument] for argument in argument_ids]
        result = backend_call(argument_tensors)
        tensors.append(result)
    return tensors[-1]
