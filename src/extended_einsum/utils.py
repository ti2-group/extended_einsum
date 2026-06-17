import os

from extended_einsum.language import (
    Program,
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


def normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("axis normalization requires a positive rank")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return normalized
