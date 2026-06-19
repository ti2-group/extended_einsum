from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, Literal

from extended_einsum.backend import (
    Array,
    BackendCompiler,
    TBackendArray,
    get_backend_of_array,
)
from extended_einsum.format import (
    DenseFormat,
    DenseLogspaceFormat,
    DenseScaledFormat,
    TensorFormat,
)
from extended_einsum.interface.operator import (
    InterfaceBinaryOperator,
    InterfaceEinsumOperator,
    InterfaceOperator,
    InterfaceSelectOperator,
    InterfaceSliceOperator,
    InterfaceSoftmaxOperator,
    InterfaceStackOperator,
    InterfaceTakeOperator,
    InterfaceUnaryOperator,
)
from extended_einsum.language import (
    Instruction,
    make_einsum_instruction,
    make_select_instruction,
    make_slice_instruction,
    make_softmax_instruction,
    make_stack_instruction,
    make_take_instruction,
    map_instruction_arguments,
)
from extended_einsum.preprocess import RichProgram
from extended_einsum.shapes import (
    Shape,
    infer_binary_shape,
    infer_einsum_shape,
    infer_select_shape,
    infer_slice_shape,
    infer_softmax_shape,
    infer_stack_shape,
    infer_take_shape,
    infer_unary_shape,
)
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
        interface_operator: InterfaceOperator,
        arguments: list[
            TensorExpression[TBackendArray]
            | Parameter[TBackendArray]
            | Array[TBackendArray]
            | TBackendArray
        ],
        keyword_arguments: InterfaceOperator | None = None,
    ) -> None:
        self.interface_operator = interface_operator
        self.arguments = arguments
        self.keyword_arguments = keyword_arguments
        # propagate shapes and formats
        self._shape = _propagate_shapes(
            interface_operator,
            [tuple(argument.shape) for argument in arguments],
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
        self.backend = get_backend_of_argument(arguments[0])
        for argument in arguments[1:]:
            if get_backend_of_argument(argument) != self.backend:
                raise ValueError(
                    f"Tensor expression has arguments with different backends: {self.backend} and {get_backend_of_argument(argument)}."
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
        self,
        other: TensorExpression[TBackendArray]
        | Parameter[TBackendArray]
        | TBackendArray,
    ) -> TensorExpression[TBackendArray]:
        return TensorExpression(InterfaceBinaryOperator("+"), [self, other])

    def __sub__(
        self,
        other: TensorExpression[TBackendArray]
        | Parameter[TBackendArray]
        | TBackendArray,
    ) -> TensorExpression[TBackendArray]:
        return TensorExpression(InterfaceBinaryOperator("-"), [self, other])

    def __mul__(
        self,
        other: TensorExpression[TBackendArray]
        | Parameter[TBackendArray]
        | TBackendArray,
    ) -> TensorExpression[TBackendArray]:
        return TensorExpression(InterfaceBinaryOperator("*"), [self, other])

    def __truediv__(
        self,
        other: TensorExpression[TBackendArray]
        | Parameter[TBackendArray]
        | TBackendArray,
    ) -> TensorExpression[TBackendArray]:
        return TensorExpression(InterfaceBinaryOperator("/"), [self, other])

    def __matmul__(
        self,
        other: TensorExpression[TBackendArray]
        | Parameter[TBackendArray]
        | TBackendArray,
    ) -> TensorExpression[TBackendArray]:
        return TensorExpression(InterfaceEinsumOperator("ik, kj -> ij"), [self, other])


def get_backend_of_argument(
    argument: TensorExpression[TBackendArray]
    | Parameter[TBackendArray]
    | TBackendArray,
) -> str:
    if isinstance(argument, TensorExpression):
        return argument.backend
    if isinstance(argument, Parameter):
        return get_backend_of_array(argument.backend_array)
    return get_backend_of_array(argument)


def _extract_program_recursive(
    tensor_expression: TensorExpression[TBackendArray] | TBackendArray,
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[Instruction],
    input_tensors: list[TBackendArray | Parameter[TBackendArray]],
    parameter_positions: list[int],
    shapes: dict[int, Shape],
    tensor_formats: dict[int, TensorFormat],
    consumers_of_ssa_id: dict[int, list[int]],
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

    # recursively compile the children first
    argument_ssa_ids = [
        _extract_program_recursive(
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

        case InterfaceSelectOperator(axis, index):
            instructions.append(
                make_select_instruction(argument_ssa_ids[0], axis, index)
            )

        case InterfaceUnaryOperator(operator):
            instructions.append((operator, tuple(argument_ssa_ids), ()))

        case InterfaceBinaryOperator(operator):
            instructions.append((operator, tuple(argument_ssa_ids), ()))

    # add shape and format information
    shapes[input_ssa_ids[expression_key]] = tensor_expression.shape
    tensor_formats[input_ssa_ids[expression_key]] = tensor_expression.format
    # add this expression as a consumer of all its arguments
    for argument_ssa_id in argument_ssa_ids:
        consumers_of_ssa_id[argument_ssa_id].append(input_ssa_ids[expression_key])
    return ssa_ids[expression_key]


def extract_program(
    tensor_expression: TensorExpression[TBackendArray],
    stability: Literal["none", "scaled", "logspace"] = "none",
) -> tuple[RichProgram, list[Any]]:
    """Compiles a tensor expression into a program and a list of arguments."""

    # extract information
    instructions: list[Instruction] = []
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
        instructions[i] = map_instruction_arguments(instruction, shift_ssa_id)
    parameter_positions = [shift_ssa_id(position) for position in parameter_positions]
    shapes = {shift_ssa_id(key): value for key, value in shapes.items()}
    tensor_formats = {shift_ssa_id(key): value for key, value in tensor_formats.items()}
    consumers_of_ssa_id = {
        shift_ssa_id(argument): [shift_ssa_id(consumer) for consumer in value]
        for argument, value in consumers_of_ssa_id.items()
    }

    # turn dicts into lists
    shapes_list: list[Shape] = [()] * n_inputs
    for i, shape in shapes.items():
        shapes_list[i] = shape
    tensor_formats_list: list[TensorFormat] = [DenseFormat()] * n_inputs
    for i, tensor_format in tensor_formats.items():
        tensor_formats_list[i] = tensor_format
    consumers_of_ssa_id_list = [[] for _ in range(n_inputs + len(instructions))]
    for i, consumers in consumers_of_ssa_id.items():
        consumers_of_ssa_id_list[i] = consumers

    return (
        RichProgram(
            instructions=instructions,
            n_inputs=n_inputs,
            stability=stability,
            shapes=shapes_list,
            tensor_formats=tensor_formats_list,
            parameter_indices=parameter_positions,
            consumers_of_ssa_id=consumers_of_ssa_id_list,
        ),
        input_tensors,
    )


def _propagate_tensor_format(
    interface_operator: InterfaceOperator, argument_formats: list[TensorFormat]
) -> TensorFormat:
    # find the argument signature
    match interface_operator:
        case InterfaceUnaryOperator(operator):
            if len(argument_formats) != 1:
                raise ValueError(
                    f"The {operator} operator takes exactly one argument, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0]]

        case InterfaceBinaryOperator(operator):
            if len(argument_formats) != 2:
                raise ValueError(
                    f"The {operator} operator takes exactly two arguments, but {len(argument_formats)} were given."
                )
            if argument_formats[0] != argument_formats[1]:
                raise ValueError(
                    f"The {operator} operator requires arguments with the same format, but {argument_formats[0]} and {argument_formats[1]} were given."
                )
            format_signature = [argument_formats[0], argument_formats[1]]

        case InterfaceStackOperator(_):
            if len(argument_formats) < 1:
                raise ValueError(
                    f"The stack operator requires at least one argument, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0]]
            if any(format != argument_formats[0] for format in argument_formats[1:]):
                raise ValueError(
                    f"The stack operator requires all arguments to have the same format, but {argument_formats[0]} and {argument_formats[1:]} were given."
                )

        case InterfaceTakeOperator(_):
            if len(argument_formats) != 2:
                raise ValueError(
                    f"The take operator takes exactly two arguments, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0], argument_formats[1]]

        case InterfaceSoftmaxOperator(_):
            if len(argument_formats) != 1:
                raise ValueError(
                    f"The softmax operator takes exactly one argument, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0]]

        case InterfaceSliceOperator(_, _, _):
            if len(argument_formats) != 1:
                raise ValueError(
                    f"The slice operator takes exactly one argument, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0]]

        case InterfaceEinsumOperator(_):
            # TODO: this is a hack
            format_signature = [argument_formats[0]]
            if any(format != argument_formats[0] for format in argument_formats[1:]):
                raise ValueError(
                    f"The einsum operator requires all arguments to have the same format, but {argument_formats[0]} and {argument_formats[1:]} were given."
                )

        case InterfaceSelectOperator(_, _):
            if len(argument_formats) != 1:
                raise ValueError(
                    f"The select operator takes exactly one argument, but {len(argument_formats)} were given."
                )
            format_signature = [argument_formats[0]]

    # decise the output format
    match format_signature:
        case [format]:
            return format
        case [DenseFormat(), DenseFormat()]:
            return DenseFormat()
        case [DenseScaledFormat(axis1), DenseScaledFormat(_)]:
            return DenseScaledFormat(axis1)
        case [DenseLogspaceFormat(), DenseLogspaceFormat()]:
            return DenseLogspaceFormat()
        case [DenseFormat(), DenseScaledFormat(axis)]:
            return DenseScaledFormat(axis)
        case [DenseScaledFormat(axis), DenseFormat()]:
            return DenseScaledFormat(axis)
        case [DenseLogspaceFormat(), DenseFormat()]:
            return DenseLogspaceFormat()
        case [DenseFormat(), DenseLogspaceFormat()]:
            return DenseLogspaceFormat()
        case _:
            raise NotImplementedError()


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
