"""Validate that release archives contain package and metadata files only."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

BANNED_TOP_LEVEL = {"demo", "examples", "results", "scripts", "tests"}


def archive_members(path: Path) -> Iterable[PurePosixPath]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            yield from (PurePosixPath(name) for name in archive.namelist())
        return

    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            yield from (PurePosixPath(member.name) for member in archive.getmembers())
        return

    raise ValueError(f"Unsupported distribution archive: {path}")


def normalized_members(path: Path) -> list[PurePosixPath]:
    members = list(archive_members(path))
    if path.name.endswith(".tar.gz"):
        roots = {member.parts[0] for member in members if member.parts}
        if len(roots) != 1:
            raise AssertionError(f"{path} should contain exactly one archive root, found {sorted(roots)}")
        return [PurePosixPath(*member.parts[1:]) for member in members if len(member.parts) > 1]
    return members


def validate(path: Path) -> None:
    members = normalized_members(path)
    top_level = {member.parts[0] for member in members if member.parts}
    unexpected = top_level & BANNED_TOP_LEVEL
    if unexpected:
        raise AssertionError(f"{path} contains excluded top-level paths: {sorted(unexpected)}")

    names = {member.name for member in members}
    if "LICENSE" not in names:
        raise AssertionError(f"{path} does not contain LICENSE")

    if path.suffix == ".whl":
        if "METADATA" not in names or "WHEEL" not in names or "py.typed" not in names:
            raise AssertionError(f"{path} is missing wheel metadata or the typing marker")
        allowed = {"extended_einsum"}
        unexpected_wheel_paths = {item for item in top_level if item not in allowed and not item.startswith("extended_einsum-")}
        if unexpected_wheel_paths:
            raise AssertionError(f"{path} contains unexpected wheel paths: {sorted(unexpected_wheel_paths)}")
    elif "README.md" not in names:
        raise AssertionError(f"{path} does not contain README.md")


def main(arguments: list[str]) -> None:
    if not arguments:
        raise SystemExit("usage: check_distribution.py DIST [DIST ...]")
    for argument in arguments:
        path = Path(argument)
        validate(path)
        print(f"validated {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
