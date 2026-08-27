from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, cast, get_args, override

from extended_einsum.backend_translation import BackendCompiler, translate_to_backend_program
from extended_einsum.backends.registry import get_backend_compiler, get_backend_functions, get_backend_of_array
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
    Shape,
    StabilityMode,
    TArray,
    TensorFormat,
)


class TensorExpression(ABC, Generic[TArray]):
    """A lazy tensor computation over raw backend arrays.

    Expressions form a graph whose leaves are ``TensorLeaf`` nodes wrapping
    raw backend arrays (e.g. ``numpy.ndarray`` or ``torch.Tensor``) and whose
    inner nodes are ``OperatorExpression`` nodes. ``materialize`` compiles and
    executes the graph on the arrays' backend and returns a raw backend array.
    """

    @property
    @abstractmethod
    def shape(self) -> Shape: ...

    @property
    @abstractmethod
    def backend(self) -> Backend: ...

    @property
    @abstractmethod
    def format(self) -> TensorFormat: ...

    def materialize(self, stability_mode: StabilityMode = "unstable") -> TArray:
        _check_stability_mode(stability_mode)
        rich_program, backend_inputs = extract_program(self, stability_mode)
        backend_functions = get_backend_functions(self.backend)
        backend_program = translate_to_backend_program(rich_program, backend_functions)
        compiler: BackendCompiler[TArray] = get_backend_compiler(self.backend)
        backend_code = compiler.compile(backend_program, backend_inputs)
        return backend_code(backend_inputs)

    def __add__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorAdd(), [self, other])

    def __radd__(self, other: TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorAdd(), [other, self])

    def __sub__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorSubtract(), [self, other])

    def __rsub__(self, other: TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorSubtract(), [other, self])

    def __mul__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorMultiply(), [self, other])

    def __rmul__(self, other: TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorMultiply(), [other, self])

    def __truediv__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorDivide(), [self, other])

    def __rtruediv__(self, other: TArray) -> TensorExpression[TArray]:
        return OperatorExpression(OperatorDivide(), [other, self])

    def __matmul__(self, other: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
        return matmul_expression(self, other)

    def __rmatmul__(self, other: TArray) -> TensorExpression[TArray]:
        return matmul_expression(other, self)

    def __getitem__(self, item: int | slice | tuple[int | slice, ...]) -> TensorExpression[TArray]:
        return getitem_expression(self, item)


class TensorLeaf(TensorExpression[TArray]):
    """A tensor expression holding a raw backend array.

    Raw arrays passed to expression functions become leaves automatically when
    their backend is detectable; construct one explicitly for backends
    registered without an ``is_array`` predicate (``backend=...``), for
    non-default tensor formats (``format=...``), or to mark a learnable
    parameter (``is_parameter=True``, recorded in the extracted program's
    ``parameter_indices``).
    """

    def __init__(
        self,
        array: TArray,
        format: TensorFormat = "dense",
        *,
        backend: Backend | None = None,
        is_parameter: bool = False,
    ) -> None:
        if backend is None:
            backend = get_backend_of_array(cast(Any, array))
        else:
            # validate the name eagerly so a typo fails here, not at materialize
            get_backend_functions(backend)
        self.array = array
        self.is_parameter = is_parameter
        self._backend = backend
        self._format: TensorFormat = format

    @property
    @override
    def shape(self) -> Shape:
        return tuple(self.array.shape)

    @property
    @override
    def backend(self) -> Backend:
        return self._backend

    @property
    @override
    def format(self) -> TensorFormat:
        return self._format

    @override
    def materialize(self, stability_mode: StabilityMode = "unstable") -> TArray:
        """A leaf is already materialized; returns its raw backend array."""

        _check_stability_mode(stability_mode)
        return self.array

    @override
    def __repr__(self) -> str:
        return f"<TensorLeaf shape={self.shape} backend={self._backend!r} format={self._format!r}>"


class OperatorExpression(TensorExpression[TArray]):
    """A tensor expression applying an operator to argument expressions."""

    def __init__(
        self,
        operator: RichOperator,
        arguments: list[TensorExpression[TArray] | TArray],
    ) -> None:
        coerced_arguments = [cast("TensorExpression[TArray]", as_expression(argument)) for argument in arguments]
        if len(coerced_arguments) == 0:
            raise ValueError("Tensor expression must have at least one argument.")
        self.operator = operator
        self.arguments = coerced_arguments
        operator.check_inputs(coerced_arguments)
        self._shape = operator.propagate_shapes([argument.shape for argument in coerced_arguments])
        self._format: TensorFormat = _propagate_tensor_format(
            operator,
            [argument.format for argument in coerced_arguments],
        )
        # check that the backends are consistent
        self._backend: Backend = coerced_arguments[0].backend
        for argument in coerced_arguments[1:]:
            if argument.backend != self._backend:
                raise ValueError(f"Tensor expression has arguments with different backends: {self.backend} and {argument.backend}.")

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

    @override
    def __repr__(self) -> str:
        return f"<TensorExpression {self.operator.name} shape={self._shape} backend={self._backend!r}>"


def as_expression(argument: TArray | TensorExpression[TArray]) -> TensorExpression[TArray]:
    """Coerces an operand into a tensor expression.

    Expressions pass through unchanged. Raw backend arrays (e.g. a
    ``numpy.ndarray`` or ``torch.Tensor``) become leaves automatically when
    their backend can be detected. Python scalars and other unsupported
    objects raise a ``TypeError`` explaining what to do instead.
    """

    if isinstance(argument, TensorExpression):
        return argument
    if isinstance(argument, (bool, int, float, complex)):
        raise TypeError(f"Tensor expressions do not support Python scalars (got {argument!r}). Use a 0-d backend array instead, e.g. numpy.array({argument!r}).")
    try:
        return TensorLeaf(argument)
    except ValueError as error:
        raise TypeError(f"Tensor expressions do not support {type(argument).__name__} operands. Convert it to a backend array first, e.g. with numpy.asarray or torch.as_tensor.") from error


def _check_stability_mode(stability_mode: StabilityMode) -> None:
    if stability_mode not in get_args(StabilityMode):
        raise ValueError(f"Unknown stability mode {stability_mode!r}. Valid modes are: {', '.join(get_args(StabilityMode))}.")


def matmul_expression(left: TensorExpression[TArray] | TArray, right: TensorExpression[TArray] | TArray) -> TensorExpression[TArray]:
    """Builds the einsum expression for ``left @ right`` with numpy's 1-D/2-D matmul semantics."""

    left = as_expression(left)
    right = as_expression(right)
    match (len(left.shape), len(right.shape)):
        case (1, 1):
            format_string = "k, k -> "
        case (1, 2):
            format_string = "k, kj -> j"
        case (2, 1):
            format_string = "ik, k -> i"
        case (2, 2):
            format_string = "ik, kj -> ij"
        case _:
            raise ValueError(f"The @ operator supports operands with one or two axes, but got shapes {left.shape} and {right.shape}. Use einsum for higher-dimensional products.")
    return OperatorExpression(OperatorEinsum(format_string), [left, right])


def getitem_expression(source: TensorExpression[TArray] | TArray, item: int | slice | tuple[int | slice, ...]) -> TensorExpression[TArray]:
    """Builds select and slice expressions for ``source[item]`` with numpy's basic indexing semantics (no step)."""

    entries = item if isinstance(item, tuple) else (item,)
    if len(entries) == 0:
        raise TypeError("indexing with an empty tuple is not supported.")
    expression = as_expression(source)
    axis = 0
    for entry in entries:
        if axis >= len(expression.shape):
            raise IndexError(f"too many indices: the expression has shape {expression.shape}, but {len(entries)} indices were given.")
        axis_size = expression.shape[axis]
        if isinstance(entry, int):
            if not -axis_size <= entry < axis_size:
                raise IndexError(f"index {entry} is out of bounds for axis {axis} with size {axis_size}.")
            index = entry + axis_size if entry < 0 else entry
            expression = OperatorExpression(OperatorSelect(axis, index), [expression])
            # selecting removes the axis, so the next entry applies to the same position
        elif isinstance(entry, slice):
            if entry.step not in (None, 1):
                raise ValueError(f"slicing with a step is not supported (got step {entry.step!r}).")
            start, stop, _ = entry.indices(axis_size)
            expression = OperatorExpression(OperatorSlice(start, stop, axis), [expression])
            axis += 1
        else:
            raise TypeError(f"indices must be integers, slices, or tuples of those, but got {type(entry).__name__}.")
    return expression


def _leaf_input_key(leaf: TensorLeaf[TArray]) -> int:
    return id(leaf.array)


def _expression_key(expression: TensorExpression[TArray]) -> int:
    return id(expression)


def _compile_expression_graph(
    root: TensorExpression[TArray],
    ssa_ids: dict[int, int],
    input_ssa_ids: dict[int, int],
    instructions: list[RichInstruction],
    input_tensors: list[TArray],
    parameter_positions: list[int],
    shapes: dict[int, Shape],
    tensor_formats: dict[int, TensorFormat],
) -> int:
    def register_input(leaf: TensorLeaf[TArray]) -> int:
        # inputs are deduplicated by the identity of the underlying backend
        # array, so reusing the same raw array yields a single program input
        input_key = _leaf_input_key(leaf)
        # if we already found this input, return its SSA-ID
        if input_key in input_ssa_ids:
            return input_ssa_ids[input_key]
        # if not, continue counting left of 0
        input_ssa_ids[input_key] = -1 - len(input_ssa_ids)
        # add it to the list of inputs and maybe also add it to the list of parameters
        input_tensors.append(leaf.array)
        if leaf.is_parameter:
            parameter_positions.append(input_ssa_ids[input_key])
        # add shape and format information
        shapes[input_ssa_ids[input_key]] = leaf.shape
        tensor_formats[input_ssa_ids[input_key]] = leaf.format
        return input_ssa_ids[input_key]

    # Iterative post-order traversal with stack: [(expression, arguments_compiled)]
    stack: list[tuple[TensorExpression[TArray], bool]] = [(root, False)]
    while stack:
        tensor_expression, arguments_compiled = stack.pop()

        if isinstance(tensor_expression, TensorLeaf):
            register_input(tensor_expression)
            continue
        assert isinstance(tensor_expression, OperatorExpression), f"Unsupported expression node type: {type(tensor_expression).__name__}."

        expression_key = _expression_key(tensor_expression)
        # if we have already seen this expression, its instruction exists
        if expression_key in ssa_ids:
            continue

        if not arguments_compiled:
            # revisit this expression once all of its arguments are compiled
            stack.append((tensor_expression, True))
            # push the arguments in reverse so they are compiled left to right,
            # preserving the SSA numbering of the recursive formulation
            for argument in reversed(tensor_expression.arguments):
                stack.append((argument, False))
            continue

        argument_ssa_ids = tuple(input_ssa_ids[_leaf_input_key(argument)] if isinstance(argument, TensorLeaf) else ssa_ids[id(argument)] for argument in tensor_expression.arguments)

        # add the instruction to the program
        ssa_ids[expression_key] = len(ssa_ids)
        instructions.append(RichInstruction(tensor_expression.operator, argument_ssa_ids))
        shapes[ssa_ids[expression_key]] = tensor_expression.shape
        tensor_formats[ssa_ids[expression_key]] = tensor_expression.format

    if isinstance(root, TensorLeaf):
        return input_ssa_ids[_leaf_input_key(root)]
    return ssa_ids[_expression_key(root)]


def extract_program(
    tensor_expression: TensorExpression[TArray],
    stability_mode: StabilityMode,
) -> tuple[RichProgram, list[TArray]]:
    """Compiles a tensor expression into a rich program and its raw input arrays.

    Parameters
    ----------
    tensor_expression : TensorExpression[TArray]
        Root of the tensor expression to be compiled.
    stability_mode : StabilityMode
        The stability mode to be used for the compiled program.

    Returns
    -------
    tuple[RichProgram, list[TArray]]
        A tuple containing the compiled rich program and the raw backend
        arrays that are its inputs, in program input order.
    """

    instructions: list[RichInstruction] = []
    ssa_ids: dict[int, int] = {}
    input_ssa_ids: dict[int, int] = {}
    input_tensors: list[TArray] = []
    parameter_positions: list[int] = []
    shapes: dict[int, Shape] = {}
    tensor_formats: dict[int, TensorFormat] = {}
    _compile_expression_graph(
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
