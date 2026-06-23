import os


def ensure_directories(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


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


def get_axis_sizes(
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


def normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("axis normalization requires a positive rank")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return normalized
