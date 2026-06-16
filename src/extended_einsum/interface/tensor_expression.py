from __future__ import annotations

from typing import Any, Generic

from extended_einsum.backend import BackendCompiler, TArray, get_backend_of_array
from extended_einsum.interface.operator import (
    InterfaceBinaryOperator,
    InterfaceEinsumOperator,
    InterfaceOperator,
    InterfaceSliceOperator,
    InterfaceSoftmaxOperator,
    InterfaceStackOperator,
    InterfaceTakeOperator,
    InterfaceUnaryOperator,
)
from extended_einsum.language import (
    Instruction,
    Program,
    make_einsum_instruction,
    make_slice_instruction,
    make_softmax_instruction,
    make_stack_instruction,
    make_take_instruction,
    map_instruction_arguments,
)
from extended_einsum.scale import ScaledTensor
from extended_einsum.translations.translations import BACKEND_TO_COMPILER
from extended_einsum.utils import get_axis_sizes, parse_format_string


class TensorExpression(Generic[TArray]):
    def __init__(
        self,
        interface_operator: InterfaceOperator,
        arguments: list[TensorExpression[TArray] | TArray],
        keyword_arguments: InterfaceOperator | None = None,
    ) -> None:
        self.interface_operator = interface_operator
        self.arguments = arguments
        self.keyword_arguments = keyword_arguments
        self._shape = _propagate_shapes(
            interface_operator,
            [tuple(argument.shape) for argument in arguments],
        )
        # check that the backends are consistent
        if len(arguments) == 0:
            raise ValueError("Tensor expression must have at least one argument.")
        self.backend = get_backend_of_argument(arguments[0])
        for argument in arguments[1:]:
            if get_backend_of_argument(argument) != self.backend:
                raise ValueError(
                    f"Tensor expression has arguments with different backends: {self.backend} and {get_backend_of_argument(argument)}."
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def materialize(self) -> TArray:
        program, arguments = compile(self)
        compiler: BackendCompiler[TArray] = BACKEND_TO_COMPILER[self.backend]
        backend_code = compiler.compile(program, arguments)
        return backend_code(arguments)

    def __add__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(InterfaceBinaryOperator("+"), [self, other])

    def __sub__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(InterfaceBinaryOperator("-"), [self, other])

    def __mul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(InterfaceBinaryOperator("*"), [self, other])

    def __truediv__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(InterfaceBinaryOperator("/"), [self, other])

    def __matmul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(InterfaceEinsumOperator("ik, kj -> ij"), [self, other])


def get_backend_of_argument(
    argument: TensorExpression[TArray] | ScaledTensor[TArray] | TArray,
) -> str:
    if isinstance(argument, TensorExpression):
        return argument.backend
    if isinstance(argument, ScaledTensor):
        return get_backend_of_array(argument.value)
    return get_backend_of_array(argument)


def _compile_recursive(
    tensor_expression: TensorExpression[TArray] | TArray,
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[Instruction],
    input_tensors: list[Any],
) -> int:
    # get a unique key for this expression
    expression_key = id(tensor_expression)

    # if the expression is a tensor, just add it to the inputs
    if not isinstance(tensor_expression, TensorExpression):
        if expression_key in input_ssa_ids:
            return input_ssa_ids[expression_key]
        input_ssa_ids[expression_key] = -1 - len(input_ssa_ids)
        input_tensors.append(tensor_expression)
        return input_ssa_ids[expression_key]

    # if we have already seen this expression, return its SSA-ID
    if expression_key in ssa_ids:
        return ssa_ids[expression_key]

    # recursively compile the children first
    argument_ssa_ids = [
        _compile_recursive(
            argument,
            ssa_ids,
            input_ssa_ids,
            instructions,
            input_tensors,
        )
        for argument in tensor_expression.arguments
    ]

    # add the instruction to the program
    ssa_ids[expression_key] = len(ssa_ids)
    match tensor_expression.interface_operator:
        case InterfaceEinsumOperator(format_string):
            instructions.append(
                make_einsum_instruction(format_string, *argument_ssa_ids)
            )

        case InterfaceStackOperator(axis):
            instructions.append(make_stack_instruction(tuple(argument_ssa_ids), axis))

        case InterfaceTakeOperator(axis):
            instructions.append(
                make_take_instruction(argument_ssa_ids[0], argument_ssa_ids[1], axis)
            )

        case InterfaceSoftmaxOperator(axis):
            instructions.append(make_softmax_instruction(argument_ssa_ids[0], axis))

        case InterfaceSliceOperator(start, stop, axis):
            instructions.append(
                make_slice_instruction(
                    argument_ssa_ids[0],
                    start,
                    stop,
                    axis,
                )
            )

        case InterfaceUnaryOperator(operator):
            instructions.append((operator, tuple(argument_ssa_ids), ()))

        case InterfaceBinaryOperator(operator):
            instructions.append((operator, tuple(argument_ssa_ids), ()))

    return ssa_ids[expression_key]


def compile(
    tensor_expression: TensorExpression[TArray],
) -> tuple[Program, list[Any]]:
    """Compiles a tensor expression into a program and a list of arguments."""

    instructions: list[Instruction] = []
    ssa_ids: dict[int, int] = {}
    input_ssa_ids: dict[int, int] = {}
    input_tensors: list[Any] = []
    _compile_recursive(
        tensor_expression,
        ssa_ids,
        input_ssa_ids,
        instructions,
        input_tensors,
    )

    n_inputs = len(input_ssa_ids)

    def shift_argument(old_argument: int) -> int:
        return old_argument + n_inputs if old_argument >= 0 else -1 - old_argument

    for i, instruction in enumerate(instructions):
        instructions[i] = map_instruction_arguments(instruction, shift_argument)
    return Program(instructions=instructions, n_inputs=n_inputs), input_tensors


def _propagate_shapes(
    interface_operator: InterfaceOperator,
    argument_shapes: list[tuple[int, ...]],
) -> tuple[int, ...]:
    """Propagates the shapes of tensors to the shape of the resulting tensor.

    Parameters
    ----------
    operator : InterfaceOperator
        The operator to be applied to the tensors. This is a dataclass with extra argument information.
    argument_shapes : list[tuple[int, ...]]
        The shapes of the tensors to be applied to the operator.

    Returns
    -------
    tuple[int, ...]
        The shape of the resulting tensor.

    Raises
    ------
    ValueError
        If the number of arguments is not compatible with the operator.
    RuntimeError
        If the shapes of the tensors are incompatible with the operator.
    """

    match interface_operator:
        case InterfaceUnaryOperator(operator):
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The {operator} operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return argument_shapes[0]

        case InterfaceBinaryOperator(operator):
            if len(argument_shapes) != 2:
                raise ValueError(
                    f"The {operator} operator takes exactly two arguments, but {len(argument_shapes)} were given."
                )
            if argument_shapes[0] != argument_shapes[1]:
                raise RuntimeError(
                    f"The shapes of the tensors are incompatible with the {operator} operator: {argument_shapes[0]} and {argument_shapes[1]}."
                )
            return argument_shapes[0]

        case InterfaceStackOperator(axis):
            if len(argument_shapes) == 0:
                raise ValueError("stack requires at least one argument")
            non_stacked_shape = argument_shapes[0]
            if any(shape != non_stacked_shape for shape in argument_shapes[1:]):
                raise ValueError(
                    "The stack operator requires all arguments to have the same shape along the stack axis."
                )
            return (
                *non_stacked_shape[:axis],
                len(argument_shapes),
                *non_stacked_shape[axis + 1 :],
            )

        case InterfaceTakeOperator(axis):
            if len(argument_shapes) != 2:
                raise ValueError(
                    f"The take operator takes exactly two arguments, but {len(argument_shapes)} were given."
                )
            return (
                *argument_shapes[0][:axis],
                *argument_shapes[1],
                *argument_shapes[0][axis + 1 :],
            )

        case InterfaceSoftmaxOperator():
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The softmax operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return argument_shapes[0]

        case InterfaceSliceOperator(start, stop, axis):
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The slice operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return (
                *argument_shapes[0][:axis],
                stop - start,
                *argument_shapes[0][axis + 1 :],
            )

        case InterfaceEinsumOperator(format_string):
            # parse the format string and check that the number of arguments matches the number of index strings
            index_strings, output_string = parse_format_string(format_string)
            if len(index_strings) != len(argument_shapes):
                raise ValueError(
                    f"The number of indices in the einsum format string ({len(index_strings)}) does not match the number of arguments ({len(argument_shapes)})."
                )
            # find the shape of the output tensor
            axis_sizes = get_axis_sizes(index_strings, argument_shapes)
            for index in output_string:
                if index not in axis_sizes:
                    raise ValueError(
                        f"Einsum format string error: the output index {index} is not present in the input index strings."
                    )
            return tuple(axis_sizes[index] for index in output_string)
