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
    if len(arrow_split) == 1:
        raise ValueError(f'The einsum format string "{format_string}" has no output: an explicit "->" output is required (implicit-output format strings are not supported), e.g. "ij,jk->ik".')
    if len(arrow_split) != 2:
        raise ValueError(f'The einsum format string "{format_string}" contains more than one "->".')
    index_string_part, output_string_part = arrow_split
    index_strings = [index_string.strip() for index_string in index_string_part.split(",")]
    output_string = output_string_part.strip()

    return index_strings, output_string


def is_contraction_free_einsum(format_string: str) -> bool:
    """Whether an einsum only broadcasts and multiplies operands without reducing labels."""

    input_strings, output_string = parse_format_string(format_string)
    output_labels = set(output_string)
    return (
        len(output_string) == len(output_labels)
        and all(len(input_string) == len(set(input_string)) and set(input_string) <= output_labels for input_string in input_strings)
        and set().union(*map(set, input_strings)) == output_labels
    )


def normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("axis normalization requires a positive rank")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return normalized


class UnionFind[T]:
    def __init__(self) -> None:
        self._parents: dict[T, T] = {}

    def add(self, item: T) -> None:
        self._parents.setdefault(item, item)

    def find(self, item: T) -> T:
        self.add(item)
        parent = self._parents[item]
        if parent != item:
            parent = self.find(parent)
            self._parents[item] = parent
        return parent

    def union(self, first: T, second: T) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self._parents[second_root] = first_root
