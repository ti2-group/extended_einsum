from collections.abc import Sequence
from typing import Any

from extended_einsum.backend import BackendFunctions
from extended_einsum.language.core import OperatorName, RawProgram
from extended_einsum.language.types import HasShape


def execute_operator(
    operator: OperatorName,
    tensor_arguments: Sequence[HasShape],
    instruction_specific_arguments: tuple[Any, ...],
    backend_functions: BackendFunctions[HasShape],
) -> HasShape:
    match operator:
        case "stack":
            axis = instruction_specific_arguments[0]
            return backend.stack(tensor_arguments, axis=axis)
        case "take":
            axis = instruction_specific_arguments[0]
            return backend_functions.take(
                tensor_arguments[0], tensor_arguments[1], axis
            )
        case "select":
            axis = instruction_specific_arguments[0]
            index = instruction_specific_arguments[1]
            return backend_functions.select(tensor_arguments[0], axis, index)
        case "slice":
            start = instruction_specific_arguments[0]
            stop = instruction_specific_arguments[1]
            axis = instruction_specific_arguments[2]
            return backend.slice(tensor_arguments[0], start, stop, axis)
        case "softmax":
            axis = instruction_specific_arguments[0]
            return backend.softmax(tensor_arguments[0], axis)
        case "einsum":
            format_string = instruction_specific_arguments[0]
            return backend.einsum(format_string, *tensor_arguments)
        case "exp":
            return backend.exp(tensor_arguments[0])
        case "log":
            return backend.log(tensor_arguments[0])
        case "+":
            return backend.add(tensor_arguments[0], tensor_arguments[1])
        case "-":
            return backend.subtract(tensor_arguments[0], tensor_arguments[1])
        case "*":
            return backend.multiply(tensor_arguments[0], tensor_arguments[1])
        case "/":
            return backend.divide(tensor_arguments[0], tensor_arguments[1])
        case _:
            raise NotImplementedError()


def run_program(
    program: RawProgram,
    inputs: Sequence[HasShape],
    backend_functions_per_instruction: list[BackendFunctions[HasShape]],
) -> HasShape:
    tensors: list[HasShape] = list(inputs)
    for i, (operator, arguments, instruction_specific_arguments) in enumerate(
        program.instructions
    ):
        argument_tensors = [tensors[argument] for argument in arguments]
        result = execute_operator(
            operator,
            argument_tensors,
            instruction_specific_arguments,
            backend_functions_per_instruction[i],
        )
        tensors.append(result)
    return tensors[-1]
