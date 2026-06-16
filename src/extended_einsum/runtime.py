from collections.abc import Sequence
from typing import Any

from extended_einsum.backend import BackendFunctions, TArray
from extended_einsum.language import (
    Operator,
    Program,
    get_arguments,
    get_instruction_specific_arguments,
    get_operator,
)


def execute_operator(
    operator: Operator,
    tensor_arguments: Sequence[TArray],
    instruction_specific_arguments: tuple[Any, ...],
    backend_functions: BackendFunctions[TArray],
) -> TArray:
    match operator:
        case "stack":
            axis = instruction_specific_arguments[0]
            return backend_functions.stack(tensor_arguments, axis=axis)
        case "take":
            axis = instruction_specific_arguments[0]
            return backend_functions.take(
                tensor_arguments[0], tensor_arguments[1], axis
            )
        case "slice":
            start = instruction_specific_arguments[0]
            stop = instruction_specific_arguments[1]
            axis = instruction_specific_arguments[2]
            return backend_functions.slice(tensor_arguments[0], start, stop, axis)
        case "softmax":
            axis = instruction_specific_arguments[0]
            return backend_functions.softmax(tensor_arguments[0], axis)
        case "einsum":
            format_string = instruction_specific_arguments[0]
            return backend_functions.einsum(format_string, *tensor_arguments)
        case "exp":
            return backend_functions.exp(tensor_arguments[0])
        case "log":
            return backend_functions.log(tensor_arguments[0])
        case "+":
            return backend_functions.add(tensor_arguments[0], tensor_arguments[1])
        case "-":
            return backend_functions.subtract(tensor_arguments[0], tensor_arguments[1])
        case "*":
            return backend_functions.multiply(tensor_arguments[0], tensor_arguments[1])
        case "/":
            return backend_functions.divide(tensor_arguments[0], tensor_arguments[1])
        case _:
            raise NotImplementedError()


def run_program(
    program: Program,
    inputs: Sequence[TArray],
    backend_functions: BackendFunctions[TArray],
) -> TArray:
    tensors: list[TArray] = list(inputs)
    for instruction in program.instructions:
        operator = get_operator(instruction)
        arguments = get_arguments(instruction)
        instruction_specific_arguments = get_instruction_specific_arguments(instruction)
        result = execute_operator(
            operator,
            [tensors[argument] for argument in arguments],
            instruction_specific_arguments,
            backend_functions,
        )
        tensors.append(result)
    return tensors[-1]
