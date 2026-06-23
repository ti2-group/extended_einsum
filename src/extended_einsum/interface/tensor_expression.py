from __future__ import annotations

from typing import Callable, Generic, Protocol, TypeVar

from extended_einsum.backend import BackendCompiler, HasBackend
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorDivide,
    OperatorEinsum,
    OperatorMultiply,
    OperatorSubtract,
    RichOperator,
)
from extended_einsum.language.rich_program import RichInstruction, RichProgram
from extended_einsum.language.types import HasShape, Shape, StabilityMode, TensorFormat
from extended_einsum.translations.translations import BACKEND_TO_COMPILER


class Array(HasShape, HasBackend, Protocol): ...


TArray = TypeVar("TArray", bound=Array)


class TensorExpression(Generic[TArray]):
    def __init__(
        self,
        operator: RichOperator,
        arguments: list[TensorExpression[TArray] | TArray],
    ) -> None:
        self.operator = operator
        self.arguments = arguments
        self._shape = operator.propagate_shapes(
            [tuple(argument.shape) for argument in arguments]
        )
        # check that the backends are consistent
        if len(arguments) == 0:
            raise ValueError("Tensor expression must have at least one argument.")
        self.backend = arguments[0].backend
        for argument in arguments[1:]:
            if argument.backend != self.backend:
                raise ValueError(
                    f"Tensor expression has arguments with different backends: {self.backend} and {argument.backend}."
                )

    @property
    def shape(self) -> Shape:
        return self._shape

    def materialize(self, stability_mode: StabilityMode) -> TArray:
        rich_program, arguments = compile(self, stability_mode)
        raw_program = rich_program.to_raw_program()
        compiler: BackendCompiler[TArray] = BACKEND_TO_COMPILER[self.backend]
        backend_code = compiler.compile(raw_program, arguments)
        return backend_code(arguments)

    def __add__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(OperatorAdd(), [self, other])

    def __sub__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(OperatorSubtract(), [self, other])

    def __mul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(OperatorMultiply(), [self, other])

    def __truediv__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(OperatorDivide(), [self, other])

    def __matmul__(
        self, other: TensorExpression[TArray] | TArray
    ) -> TensorExpression[TArray]:
        return TensorExpression(OperatorEinsum("ik, kj -> ij"), [self, other])

    def __getitem__(self, index: int | slice) -> TensorExpression[TArray]:
        raise NotImplementedError(
            "Indexing is not yet implemented. This should produce a select, take, or slice operator."
        )


def _compile_recursive(
    tensor_expression: TensorExpression[TArray] | TArray,
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[RichInstruction],
    input_tensors: list[TArray],
    parameter_positions: list[int],
    shapes: dict[int, Shape],
    tensor_formats: dict[int, TensorFormat],
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

    # recursively compile the children
    argument_ssa_ids = tuple(
        _compile_recursive(
            argument,
            ssa_ids,
            input_ssa_ids,
            instructions,
            input_tensors,
            parameter_positions,
            shapes,
            tensor_formats,
        )
        for argument in tensor_expression.arguments
    )

    # add the instruction to the program
    ssa_ids[expression_key] = len(ssa_ids)
    instructions.append((tensor_expression.operator, argument_ssa_ids))

    return ssa_ids[expression_key]


def _map_instruction_arguments(
    instruction: RichInstruction, shift_argument: Callable[[int], int]
) -> RichInstruction:
    operator, argument_ssa_ids = instruction
    return operator, tuple(shift_argument(argument) for argument in argument_ssa_ids)


def compile(
    tensor_expression: TensorExpression[TArray],
    stability_mode: StabilityMode,
) -> tuple[RichProgram, list[TArray]]:
    """Compiles a tensor expression into a rich program and a list of arguments.

    Parameters
    ----------
    tensor_expression : TensorExpression[TArray]
        Root of the tensor expression to be compiled.
    stability_mode : StabilityMode
        The stability mode to be used for the compiled program.

    Returns
    -------
    tuple[RichProgram, list[TArray]]
        A tuple containing the compiled rich program and a list of arguments.
    """

    instructions: list[RichInstruction] = []
    ssa_ids: dict[int, int] = {}
    input_ssa_ids: dict[int, int] = {}
    input_tensors: list[TArray] = []
    parameter_positions: list[int] = []
    shapes: dict[int, Shape] = {}
    tensor_formats: dict[int, TensorFormat] = {}
    _compile_recursive(
        tensor_expression,
        ssa_ids,
        input_ssa_ids,
        instructions,
        input_tensors,
        parameter_positions,
        shapes,
        tensor_formats,
    )

    # prepare shifting ssa ids
    n_inputs = len(input_ssa_ids)

    def shift_ssa_id(old_ssa_id: int) -> int:
        return old_ssa_id + n_inputs if old_ssa_id >= 0 else -1 - old_ssa_id

    # shift ssa ids
    for i, instruction in enumerate(instructions):
        instructions[i] = _map_instruction_arguments(instruction, shift_ssa_id)
    parameter_positions = [shift_ssa_id(position) for position in parameter_positions]
    shapes = {shift_ssa_id(key): value for key, value in shapes.items()}
    tensor_formats = {shift_ssa_id(key): value for key, value in tensor_formats.items()}

    # turn dicts into lists
    shapes_list: list[Shape] = [()] * n_inputs
    for i, shape in shapes.items():
        shapes_list[i] = shape
    tensor_formats_list: list[TensorFormat] = ["dense"] * n_inputs
    for i, tensor_format in tensor_formats.items():
        tensor_formats_list[i] = tensor_format

    return (
        RichProgram(
            instructions=instructions,
            n_inputs=n_inputs,
            stability_mode=stability_mode,
            shapes=shapes_list,
            tensor_formats=tensor_formats_list,
            parameter_indices=frozenset(parameter_positions),
        ),
        input_tensors,
    )
