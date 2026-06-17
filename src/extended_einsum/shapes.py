import numpy as np

from extended_einsum.utils import parse_format_string

Shape = tuple[int, ...]


def get_axis_sizes(
    index_strings: list[str], tensor_shapes: list[Shape]
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


def infer_unary_shape(operand_shape: Shape) -> Shape:
    return operand_shape


def infer_binary_shape(operand_shape_1: Shape, operand_shape_2: Shape) -> Shape:
    return tuple(np.broadcast_shapes(operand_shape_1, operand_shape_2))


def infer_take_shape(source_shape: Shape, index_shape: Shape, axis: int) -> Shape:
    if len(source_shape) == 0:
        raise ValueError("take requires an operand with at least one axis")
    if len(index_shape) != 1:
        raise ValueError("take requires an index with rank 1")
    return (
        *source_shape[:axis],
        *index_shape,
        *source_shape[axis + 1 :],
    )


def infer_stack_shape(source_shapes: list[Shape], axis: int) -> Shape:
    if len(source_shapes) == 0:
        raise ValueError("stack requires at least one operand")
    non_stacked_shape = source_shapes[0]
    if any(shape != non_stacked_shape for shape in source_shapes[1:]):
        raise ValueError(
            "The stack operator requires all arguments to have the same shape along the stack axis."
        )
    return (
        *non_stacked_shape[:axis],
        len(source_shapes),
        *non_stacked_shape[axis + 1 :],
    )


def infer_slice_shape(source_shape: Shape, start: int, stop: int, axis: int) -> Shape:
    if len(source_shape) == 0:
        raise ValueError("slice requires an operand with at least one axis")
    return (
        *source_shape[:axis],
        stop - start,
        *source_shape[axis + 1 :],
    )


def infer_einsum_shape(format_string: str, argument_shapes: list[Shape]) -> Shape:
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


def infer_softmax_shape(operand_shape: Shape) -> Shape:
    if len(operand_shape) == 0:
        raise ValueError("softmax requires an operand with at least one axis")
    return operand_shape


def infer_select_shape(source_shape: Shape, axis: int) -> Shape:
    if len(source_shape) == 0:
        raise ValueError("select requires an operand with at least one axis")
    if len(source_shape) != 1:
        raise ValueError("select requires an operand with rank 1")
    return (
        *source_shape[:axis],
        *source_shape[axis + 1 :],
    )
