from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, cast, override

from extended_einsum.backend_translation import BackendCompiler, BackendProgram, translate_to_backend_program
from extended_einsum.backends.registry import get_backend_compiler, get_backend_functions
from extended_einsum.language.rich_instruction import RichInstruction, map_instruction_arguments
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorConcat,
    OperatorCos,
    OperatorDivide,
    OperatorEinsum,
    OperatorExp,
    OperatorInverse,
    OperatorLog,
    OperatorMultiply,
    OperatorSelect,
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorSqrt,
    OperatorStack,
    OperatorSubtract,
    OperatorTake,
    OperatorTan,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import (
    Backend,
    HasBackend,
    HasFormat,
    HasShape,
    Shape,
    StabilityMode,
    TArray,
    TensorFormat,
)


@dataclass(frozen=True)
class Parameter(Generic[TArray]):
    array: TArray

    @property
    def shape(self) -> Shape:
        return tuple(self.array.shape)

    @property
    def backend(self) -> Backend:
        return self.array.backend

    @property
    def format(self) -> TensorFormat:
        return self.array.format


class TensorExpression(HasShape, HasBackend, HasFormat, Generic[TArray]):
    def __init__(
        self,
        operator: RichOperator,
        arguments: list[TensorExpression[TArray] | Parameter[TArray] | TArray],
    ) -> None:
        self.operator = operator
        self.arguments = arguments
        operator.check_inputs(arguments)
        self._shape = operator.propagate_shapes([tuple(argument.shape) for argument in arguments])
        self._format: TensorFormat = _propagate_tensor_format(
            operator,
            [argument.format for argument in arguments],
        )
        # check that the backends are consistent
        if len(arguments) == 0:
            raise ValueError("Tensor expression must have at least one argument.")
        self._backend: Backend = arguments[0].backend
        for argument in arguments[1:]:
            if argument.backend != self._backend:
                raise ValueError(f"Tensor expression has arguments with different backends: {self.backend} and {argument.backend}.")

        # these only exist once the tensor has materialized
        self._rich_program: RichProgram | None = None
        self._input_arguments: list[TArray] | None = None
        self._backend_program: BackendProgram[TArray] | None = None
        self._backend_code: Callable[[Sequence[TArray]], TArray] | None = None

    @property
    @override
    def shape(self) -> Shape:
        return self._shape

    @property
    @override
    def backend(self) -> Backend:
        return self._backend

    @property
    @override
    def format(self) -> TensorFormat:
        return self._format

    def materialize(self, stability_mode: StabilityMode = "unstable") -> TArray:
        self._rich_program, self._input_arguments = extract_program(self, stability_mode)
        backend_functions = get_backend_functions(self.backend)
        self._backend_program = translate_to_backend_program(self._rich_program, backend_functions)
        compiler: BackendCompiler[TArray] = get_backend_compiler(self.backend)
        backend_inputs = [argument.backend_array for argument in self._input_arguments]
        self._backend_code = compiler.compile(self._backend_program, backend_inputs)
        backend_array = self._backend_code(backend_inputs)

        # Import locally to avoid the module cycle between the expression and
        # public interface-function modules.
        from extended_einsum.interface.functions import BackendArrayWrapper

        return cast(TArray, BackendArrayWrapper(backend_array, self.backend, self.format))

    def __add__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return TensorExpression(OperatorAdd(), [self, other])

    def __sub__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return TensorExpression(OperatorSubtract(), [self, other])

    def __mul__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return TensorExpression(OperatorMultiply(), [self, other])

    def __truediv__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return TensorExpression(OperatorDivide(), [self, other])

    def __matmul__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return TensorExpression(OperatorEinsum("ik, kj -> ij"), [self, other])

    def __getitem__(self, index: int | slice) -> TensorExpression[TArray]:
        raise NotImplementedError("Indexing is not yet implemented. This should produce a select, take, or slice operator.")


def _compile_recursive(
    tensor_expression: TensorExpression[TArray] | Parameter[TArray] | TArray,
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
        # if we already found this input, return its SSA-ID
        if expression_key in input_ssa_ids:
            return input_ssa_ids[expression_key]
        # if not, continue counting left of 0
        input_ssa_ids[expression_key] = -1 - len(input_ssa_ids)
        # add it to the list of inputs and maybe also add it to the list of parameters
        if isinstance(tensor_expression, Parameter):
            input_tensors.append(tensor_expression.array)
            parameter_positions.append(input_ssa_ids[expression_key])
        else:
            input_tensors.append(tensor_expression)
        # add shape and format information
        shapes[input_ssa_ids[expression_key]] = tensor_expression.shape
        tensor_formats[input_ssa_ids[expression_key]] = tensor_expression.format
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
        )
        for argument in tensor_expression.arguments
    )

    # add the instruction to the program
    ssa_ids[expression_key] = len(ssa_ids)
    instructions.append(RichInstruction(tensor_expression.operator, argument_ssa_ids))
    shapes[ssa_ids[expression_key]] = tensor_expression.shape
    tensor_formats[ssa_ids[expression_key]] = tensor_expression.format
    return ssa_ids[expression_key]


def extract_program(
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
        instructions[i] = map_instruction_arguments(instruction, shift_ssa_id)
    parameter_positions = [shift_ssa_id(position) for position in parameter_positions]
    shapes = {shift_ssa_id(key): value for key, value in shapes.items()}
    tensor_formats = {shift_ssa_id(key): value for key, value in tensor_formats.items()}

    # turn dicts into lists
    n_ssa_ids = n_inputs + len(ssa_ids)
    shapes_list: list[Shape] = [()] * n_ssa_ids
    for i, shape in shapes.items():
        shapes_list[i] = shape
    tensor_formats_list: list[TensorFormat] = ["dense"] * n_ssa_ids
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


def _propagate_tensor_format(operator: RichOperator, argument_formats: list[TensorFormat]) -> TensorFormat:
    format_signature: list[TensorFormat]
    # find the argument signature
    match operator:
        case OperatorSin() | OperatorCos() | OperatorTan() | OperatorExp() | OperatorLog() | OperatorSqrt() | OperatorInverse():
            if len(argument_formats) != 1:
                raise ValueError(f"The {operator} operator takes exactly one argument, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0]]

        case OperatorAdd() | OperatorSubtract() | OperatorMultiply() | OperatorDivide():
            if len(argument_formats) != 2:
                raise ValueError(f"The {operator} operator takes exactly two arguments, but {len(argument_formats)} were given.")
            if argument_formats[0] != argument_formats[1]:
                raise ValueError(f"The {operator} operator requires arguments with the same format, but {argument_formats[0]} and {argument_formats[1]} were given.")
            format_signature = [argument_formats[0], argument_formats[1]]

        case OperatorStack(_) | OperatorConcat(_):
            if len(argument_formats) < 1:
                raise ValueError(f"The {operator.name} operator requires at least one argument, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0]]
            if any(format != argument_formats[0] for format in argument_formats[1:]):
                raise ValueError(f"The {operator.name} operator requires all arguments to have the same format, but {argument_formats[0]} and {argument_formats[1:]} were given.")

        case OperatorTake(_):
            if len(argument_formats) != 2:
                raise ValueError(f"The take operator takes exactly two arguments, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0], argument_formats[1]]

        case OperatorSelect(_, _):
            if len(argument_formats) != 1:
                raise ValueError(f"The select operator takes exactly one argument, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0]]

        case OperatorSlice(_, _, _):
            if len(argument_formats) != 1:
                raise ValueError(f"The slice operator takes exactly one argument, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0]]

        case OperatorSoftmax(_):
            if len(argument_formats) != 1:
                raise ValueError(f"The softmax operator takes exactly one argument, but {len(argument_formats)} were given.")
            format_signature = [argument_formats[0]]

        case OperatorEinsum(_):
            # TODO: this is a hack
            format_signature = [argument_formats[0]]
            if any(format != argument_formats[0] for format in argument_formats[1:]):
                raise ValueError(f"The einsum operator requires all arguments to have the same format, but {argument_formats[0]} and {argument_formats[1:]} were given.")

        case _:
            raise NotImplementedError(f"Operator {operator} is not yet supported.")

    # decise the output format
    match format_signature:
        case [format]:
            return format
        case ["dense", "dense"]:
            return "dense"
        case ["dense", "sparse"]:
            return "sparse"
        case ["sparse", "dense"]:
            return "sparse"
        case ["sparse", "sparse"]:
            return "sparse"
        case _:
            raise NotImplementedError()
