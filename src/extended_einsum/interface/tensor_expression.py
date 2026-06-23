from __future__ import annotations

from typing import Any, Callable, Generic

from extended_einsum.backend import BackendCompiler, TArray
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorDivide,
    OperatorEinsum,
    OperatorMultiply,
    OperatorSubtract,
    RichOperator,
)
from extended_einsum.language.rich_program import RichInstruction, RichProgram
from extended_einsum.language.types import Shape
from extended_einsum.translations.translations import BACKEND_TO_COMPILER


@dataclass(frozen=True)
class Parameter[TBackendArray]:
    backend_array: TBackendArray

    @property
    def shape(self) -> Shape:
        return tuple(self.backend_array.shape)  # pyright: ignore[reportAttributeAccessIssue]


class TensorExpression(Generic[TBackendArray]):
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
        argument_formats = [
            argument.format if hasattr(argument, "format") else DenseFormat()  # pyright: ignore[reportAttributeAccessIssue]
            for argument in arguments
        ]
        self.format = _propagate_tensor_format(
            interface_operator,
            argument_formats,
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

    def materialize(
        self, stability: Literal["none", "scaled", "logspace"] = "none"
    ) -> TBackendArray:
        program, arguments = extract_program(self, stability)
        compiler: BackendCompiler[TBackendArray] = BACKEND_TO_COMPILER[self.backend]
        # TODO: the specific backend implementations should be clear after preprocessing. also we need to preprocess here
        backend_implementations = ...
        backend_code = compiler.compile(program, arguments, backend_implementations)
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


def _extract_program_recursive(
    tensor_expression: TensorExpression[TBackendArray] | TBackendArray,
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[RichInstruction],
    input_tensors: list[Any],
) -> int:
    # get a unique key for this expression
    expression_key = id(tensor_expression)

    # if the expression is a tensor, just add it to the inputs
    if not isinstance(tensor_expression, TensorExpression):
        # if we already found this input, return its SSA-ID
        if expression_key in input_ssa_ids:
            return input_ssa_ids[expression_key]
        # if not, continue counting left of 0
        input_ssa_ids[expression_key] = -1 - len(input_ssa_ids)
        # add it to the list of inputs and maybe also add it to the list of parameters
        if isinstance(tensor_expression, Parameter):
            input_tensors.append(tensor_expression.backend_array)
            parameter_positions.append(input_ssa_ids[expression_key])
        elif hasattr(tensor_expression, "backend_array"):
            input_tensors.append(tensor_expression.backend_array)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            input_tensors.append(tensor_expression)
        # add shape and format information
        shapes[input_ssa_ids[expression_key]] = tensor_expression.shape
        tensor_formats[input_ssa_ids[expression_key]] = (
            tensor_expression.format  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(tensor_expression, "format")
            else DenseFormat()
        )
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
            input_tensors,  # pyright: ignore[reportArgumentType]
            parameter_positions,
            shapes,
            tensor_formats,
            consumers_of_ssa_id,
        )
        for argument in tensor_expression.arguments
    )

    # add the instruction to the program
    ssa_ids[expression_key] = len(ssa_ids)
    instructions.append((tensor_expression.operator, argument_ssa_ids))

    # add shape and format information
    shapes[ssa_ids[expression_key]] = tensor_expression.shape
    tensor_formats[ssa_ids[expression_key]] = tensor_expression.format
    # add this expression as a consumer of all its arguments
    for argument_ssa_id in argument_ssa_ids:
        consumers_of_ssa_id[argument_ssa_id].append(ssa_ids[expression_key])
    return ssa_ids[expression_key]


def _map_instruction_arguments(
    instruction: RichInstruction, shift_argument: Callable[[int], int]
) -> RichInstruction:
    operator, argument_ssa_ids = instruction
    return operator, tuple(shift_argument(argument) for argument in argument_ssa_ids)


def compile(
    tensor_expression: TensorExpression[TArray],
) -> tuple[RichProgram, list[Any]]:
    """Compiles a tensor expression into a program and a list of arguments."""

    instructions: list[RichInstruction] = []
    ssa_ids: dict[int, int] = {}
    input_ssa_ids: dict[int, int] = {}
    input_tensors: list[Any] = []
    parameter_positions: list[int] = []
    shapes: dict[int, Shape] = {}
    tensor_formats: dict[int, TensorFormat] = {}
    consumers_of_ssa_id: dict[int, list[int]] = defaultdict(list)
    _extract_program_recursive(
        tensor_expression,
        ssa_ids,
        input_ssa_ids,
        instructions,
        input_tensors,
        parameter_positions,
        shapes,
        tensor_formats,
        consumers_of_ssa_id,
    )

    # prepare shifting ssa ids
    n_inputs = len(input_ssa_ids)

    def shift_ssa_id(old_ssa_id: int) -> int:
        return old_ssa_id + n_inputs if old_ssa_id >= 0 else -1 - old_ssa_id

    # shift ssa ids
    for i, instruction in enumerate(instructions):
        instructions[i] = _map_instruction_arguments(instruction, shift_argument)
    return RichProgram(instructions=instructions, n_inputs=n_inputs), input_tensors


def _propagate_shapes(
    interface_operator: InterfaceOperator,
    argument_shapes: list[Shape],
) -> Shape:
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
            return infer_unary_shape(argument_shapes[0])

        case InterfaceBinaryOperator(operator):
            if len(argument_shapes) != 2:
                raise ValueError(
                    f"The {operator} operator takes exactly two arguments, but {len(argument_shapes)} were given."
                )
            return infer_binary_shape(argument_shapes[0], argument_shapes[1])

        case InterfaceStackOperator(axis):
            return infer_stack_shape(argument_shapes, axis)

        case InterfaceTakeOperator(axis):
            if len(argument_shapes) != 2:
                raise ValueError(
                    f"The take operator takes exactly two arguments, but {len(argument_shapes)} were given."
                )
            return infer_take_shape(argument_shapes[0], argument_shapes[1], axis)

        case InterfaceSoftmaxOperator(_):
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The softmax operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return infer_softmax_shape(argument_shapes[0])

        case InterfaceSliceOperator(start, stop, axis):
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The slice operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return infer_slice_shape(argument_shapes[0], start, stop, axis)

        case InterfaceEinsumOperator(format_string):
            return infer_einsum_shape(format_string, argument_shapes)

        case InterfaceSelectOperator(axis, _):
            if len(argument_shapes) != 1:
                raise ValueError(
                    f"The select operator takes exactly one argument, but {len(argument_shapes)} were given."
                )
            return infer_select_shape(argument_shapes[0], axis)
