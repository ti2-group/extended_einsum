from collections.abc import Sequence
from typing import Any

from extended_einsum.backend import (
    MultiFormatBackendFunctions,
    SingleFormatBackendFunctions,
    TArray,
)
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
    backend: SingleFormatBackendFunctions | MultiFormatBackendFunctions,
) -> TArray:
    match operator:
        case "stack":
            axis = instruction_specific_arguments[0]
            return backend.stack(tensor_arguments, axis=axis)
        case "take":
            axis = instruction_specific_arguments[0]
            return backend.take(tensor_arguments[0], tensor_arguments[1], axis)
        case "select":
            axis = instruction_specific_arguments[0]
            index = instruction_specific_arguments[1]
            return backend.select(tensor_arguments[0], axis, index)
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
    program: Program,
    inputs: Sequence[TArray],
    backend_implementations: list[
        SingleFormatBackendFunctions | MultiFormatBackendFunctions
    ],
) -> TArray:
    tensors: list[TArray] = list(inputs)
    for instruction, backend_implementation in zip(
        program.instructions, backend_implementations
    ):
        operator = get_operator(instruction)
        arguments = get_arguments(instruction)
        instruction_specific_arguments = get_instruction_specific_arguments(instruction)
        result = execute_operator(
            operator,
            [tensors[argument] for argument in arguments],
            instruction_specific_arguments,
            backend_implementation,
        )
        tensors.append(result)
    return tensors[-1]
