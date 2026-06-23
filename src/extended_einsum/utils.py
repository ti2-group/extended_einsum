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
