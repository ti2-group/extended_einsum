import os
from typing import get_args

from extended_einsum.language import (
    EINSUM_OPERATOR,
    SCALED_EINSUM_OPERATORS,
    SLICE_OPERATOR,
    SOFTMAX_OPERATOR,
    TAKE_OPERATOR,
    BinaryOperator,
    Operator,
    Program,
    UnaryOperator,
    get_arguments,
)


def ensure_directories(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def unify_indices(program: Program) -> Program:
    """Builds a new program that has consistent index strings for tensors.

    Parameters
    ----------
    program : Program
        Programm that possibly refers to the same axis of the same tensor with different characters in different einsum calls.

    Returns
    -------
    Program
        Program that refers to the axes of each tensor with the same characters respectively always.
    """

    # trivial identity implementation for now (placeholder)
    return program


def get_ssa_parents(program: Program) -> dict[int, tuple[int, ...]]:
    """Builds a dictionary that maps each SSA-IDs to the SSA-IDs of its parents."""

    # the SSA-IDs 0 to program.n_inputs - 1 are occupied by the inputs, so the first intermediate tensor has SSA-ID program.n_inputs
    parents: dict[int, tuple[int, ...]] = {i: () for i in range(program.n_inputs)}
    for i, instruction in enumerate(program.instructions):
        # each instruction writes to a new SSA-ID, so we just fill the parents with the SSA-IDs of the arguments
        parents[program.n_inputs + i] = get_arguments(instruction)
    return parents


def parse_format_string(format_string: str) -> tuple[list[str], str]:
    """Parses a format string into a list of index strings and an output index string.

    Parameters
    ----------
    format_string : str
        The format string to be parsed.

    Returns
    -------
    tuple[list[str], str]
        A tuple containing a list of index strings and an output index string.
    """

    arrow_split = format_string.split("->")
    if len(arrow_split) != 2:
        raise ValueError(f'"{format_string}" is not a valid einsum format string.')
    index_string_part, output_string_part = arrow_split
    index_strings = [
        index_string.strip() for index_string in index_string_part.split(",")
    ]
    output_string = output_string_part.strip()

    return index_strings, output_string


def _get_axis_sizes(
    index_strings: list[str], tensor_shapes: list[tuple[int, ...]]
) -> dict[str, int]:
    axis_sizes: dict[str, int] = {}
    for index_string, tensor_shape in zip(index_strings, tensor_shapes):
        for index in index_string:
            if index not in axis_sizes:
                axis_sizes[index] = tensor_shape[index_string.index(index)]
            elif axis_sizes[index] != tensor_shape[index_string.index(index)]:
                raise RuntimeError(
                    f"Incompatible shapes for index {index_string}: {tensor_shape} and {axis_sizes[index]}."
                )
    return axis_sizes


def propagate_shapes(
    operator: Operator,
    argument_shapes: list[tuple[int, ...]],
    format_string: str | None = None,
    axis: int | None = None,
    slice_start: int | None = None,
    slice_stop: int | None = None,
) -> tuple[int, ...]:
    """Propagates the shapes of tensors to the shape of the resulting tensor.

    Parameters
    ----------
    operator : UnaryOperator | BinaryOperator | Literal["einsum", "id"]
        The operator to be applied to the tensors.
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

    # unary operators
    if operator in get_args(UnaryOperator):
        if len(argument_shapes) != 1:
            raise ValueError(
                f"The {operator} operator takes exactly one argument, but {len(argument_shapes)} were given."
            )
        return argument_shapes[0]

    # softmax operator
    if operator == SOFTMAX_OPERATOR:
        if len(argument_shapes) != 1:
            raise ValueError(
                f"The {operator} operator takes exactly one argument, but {len(argument_shapes)} were given."
            )
        if not argument_shapes[0]:
            raise ValueError("softmax requires an input tensor with at least one axis")
        normalize_axis(0 if axis is None else axis, len(argument_shapes[0]))
        return argument_shapes[0]

    # binary operators
    if operator in get_args(BinaryOperator):
        if len(argument_shapes) != 2:
            raise ValueError(
                f"The {operator} operator takes exactly two arguments, but {len(argument_shapes)} were given."
            )
        if argument_shapes[0] != argument_shapes[1]:
            raise RuntimeError(
                f"The shapes of the tensors are incompatible with the {operator} operator: {argument_shapes[0]} and {argument_shapes[1]}."
            )
        return argument_shapes[0]

    # take operator
    if operator == TAKE_OPERATOR:
        if len(argument_shapes) != 2:
            raise ValueError(
                f"The take operator takes exactly two arguments, but {len(argument_shapes)} were given."
            )
        if not argument_shapes[0]:
            raise ValueError(
                "The take operator requires an operand with a leading axis."
            )
        take_axis = normalize_axis(0 if axis is None else axis, len(argument_shapes[0]))
        return (
            *argument_shapes[0][:take_axis],
            *argument_shapes[1],
            *argument_shapes[0][take_axis + 1 :],
        )

    # slice operator
    if operator == SLICE_OPERATOR:
        if len(argument_shapes) != 1:
            raise ValueError(
                "The slice operator takes exactly one argument, but "
                f"{len(argument_shapes)} were given."
            )
        if slice_start is None or slice_stop is None:
            raise ValueError("slice shape propagation requires start and stop")
        if not argument_shapes[0]:
            raise ValueError(
                "The slice operator requires an operand with at least one axis."
            )
        slice_axis = normalize_axis(
            0 if axis is None else axis,
            len(argument_shapes[0]),
        )
        start, stop, step = slice(slice_start, slice_stop).indices(
            argument_shapes[0][slice_axis]
        )
        if step != 1:
            raise ValueError("slice operator only supports step 1")
        return (
            *argument_shapes[0][:slice_axis],
            max(0, stop - start),
            *argument_shapes[0][slice_axis + 1 :],
        )

    # einsum operators
    if operator == EINSUM_OPERATOR or operator in SCALED_EINSUM_OPERATORS:
        if format_string is None:
            raise ValueError(
                "explicit einsum shape propagation requires a format string"
            )
        index_strings, output_string = parse_format_string(format_string)
    else:
        index_strings, output_string = parse_format_string(operator)

    if len(index_strings) != len(argument_shapes):
        raise ValueError(
            f"The number of indices in the einsum format string ({len(index_strings)}) does not match the number of arguments ({len(argument_shapes)})."
        )

    # find the shape of the output tensor
    axis_sizes = _get_axis_sizes(index_strings, argument_shapes)
    for index in output_string:
        if index not in axis_sizes:
            raise ValueError(
                f"The output index {index} is not present in the input indices."
            )
    return tuple(axis_sizes[index] for index in output_string)


def normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("axis normalization requires a positive rank")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return normalized
