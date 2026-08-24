from collections.abc import Callable, Sequence
from functools import partial
from typing import Generic, TypeVar

from extended_einsum.backend_translation.backend import BackendArray, BackendCompiler, BackendProgram

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


def run_program(
    program: BackendProgram[TBackendArray],
    inputs: Sequence[TBackendArray],
) -> TBackendArray:
    if len(inputs) != program.n_inputs:
        raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")

    tensors: list[TBackendArray] = list(inputs)
    for backend_call, argument_ids in zip(program.backend_calls, program.call_arguments):
        argument_tensors = [tensors[argument] for argument in argument_ids]
        result = backend_call(argument_tensors)
        tensors.append(result)
    return tensors[-1]


class DefaultCompiler(BackendCompiler[TBackendArray], Generic[TBackendArray]):
    """Fallback compiler that interprets the program call by call.

    Used for backends without JIT compilation and as the default when a custom
    backend registers no compiler of its own.
    """

    def compile(
        self,
        program: BackendProgram[TBackendArray],
        inputs: Sequence[TBackendArray],
    ) -> Callable[[Sequence[TBackendArray]], TBackendArray]:
        if len(inputs) != program.n_inputs:
            raise ValueError(f"The number of inputs ({len(inputs)}) does not match the number of inputs ({program.n_inputs}) in the program.")
        return partial(run_program, program)
