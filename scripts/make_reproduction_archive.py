#!/usr/bin/env python3
"""Build and validate an anonymized paper-reproduction source archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "extended-einsum-paper-reproduction"
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / f"{ARCHIVE_ROOT}.zip"

# The exporter itself is deliberately outside the payload. These are the
# source and artifact trees needed to inspect, run, test, and plot the work.
INCLUDED_ROOT_FILES = (
    ".python-version",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
)
INCLUDED_ROOT_DIRECTORIES = (
    "datasets",
    "demo",
    "examples",
    "experiments",
    "src",
    "tests",
)
EXCLUDED_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}

REQUIRED_REPRODUCTION_FILES = (
    "experiments/results/ablation.csv",
    "experiments/results/ablation_jax.csv",
    "experiments/results/correctness.csv",
    "experiments/results/speedup.csv",
    "experiments/monarch/results/performance.csv",
    "experiments/monarch/results/performance_batches.csv",
    "experiments/pyjuice_cp_t/results/comparison.csv",
    "datasets/MNIST/raw/train-images-idx3-ubyte",
    "datasets/MNIST/raw/train-labels-idx1-ubyte",
)

# This file never enters the archive. Keeping the denylist here makes it easy
# to audit and extend without embedding the denied text in the exported work.
DENIED_SPELLINGS = (
    "Christoph",
    "Chrsitoph",
    "Staudt",
    "Maurice",
    "Wenig",
    "Jena",
    "Ti2",
    "FSU",
    "Friedrich Schiller",
    "Theoretische Informatik",
    "Theoretical Computer Science",
    "University",
    "Universities",
    "Universtiy",
    "Universität",
    "Universitaet",
)


def flexible_pattern(spelling: str) -> re.Pattern[str]:
    """Match a spelling case-insensitively, including separator obfuscation."""

    characters = [re.escape(character) for character in spelling if character.isalnum()]
    body = r"[\W_]*".join(characters)
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])", re.IGNORECASE | re.UNICODE)


DENIED_PATTERNS = tuple(flexible_pattern(spelling) for spelling in DENIED_SPELLINGS)
DENIED_BINARY_PATTERNS = tuple(
    re.compile(
        rf"(?<![^\W_]){re.escape(spelling)}(?![^\W_])",
        re.IGNORECASE | re.UNICODE,
    )
    for spelling in DENIED_SPELLINGS
)


def denied_matches(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return sorted({match.group(0) for pattern in DENIED_PATTERNS for match in pattern.finditer(normalized)})


def denied_binary_matches(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return sorted({match.group(0) for pattern in DENIED_BINARY_PATTERNS for match in pattern.finditer(normalized)})


def should_exclude(relative_path: Path) -> bool:
    parts = relative_path.parts
    return (
        any(part.startswith(".git") for part in parts)
        or any(part in EXCLUDED_DIRECTORY_NAMES for part in parts)
        or relative_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES
        or (relative_path.parent == Path("experiments/monarch/results") and relative_path.name.startswith("compile_diagnosis"))
    )


def selected_files() -> list[Path]:
    selected: list[Path] = []
    for name in INCLUDED_ROOT_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required project file is missing: {name}")
        selected.append(path)

    for name in INCLUDED_ROOT_DIRECTORIES:
        directory = PROJECT_ROOT / name
        if not directory.exists():
            if name == "datasets":
                continue
            raise FileNotFoundError(f"required project directory is missing: {name}")
        for path in directory.rglob("*"):
            relative_path = path.relative_to(PROJECT_ROOT)
            if should_exclude(relative_path):
                continue
            if path.is_symlink():
                raise RuntimeError(f"refusing to archive symlink: {relative_path}")
            if path.is_file():
                selected.append(path)

    return sorted(set(selected), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def validate_required_inputs(selected: Iterable[Path]) -> None:
    selected_names = {path.relative_to(PROJECT_ROOT).as_posix() for path in selected}
    missing = sorted(set(REQUIRED_REPRODUCTION_FILES) - selected_names)
    if missing:
        formatted = "\n  ".join(missing)
        raise RuntimeError(f"paper-reproduction inputs are missing:\n  {formatted}")


def apply_file_specific_cleanup(relative_path: PurePosixPath, text: str) -> str:
    if relative_path == PurePosixPath("README.md"):
        text = re.sub(
            r"^\[!\[(?:CI|PyPI|Python)\].*\n",
            "",
            text,
            flags=re.MULTILINE,
        )
        text = text.replace(
            "https://github.com/ti2-group/extended_einsum/tree/main/examples",
            "examples/",
        )
        text = text.replace(
            "https://github.com/ti2-group/extended_einsum/blob/main/CONTRIBUTING.md",
            "CONTRIBUTING.md",
        )
        text = text.replace(
            "https://github.com/ti2-group/extended_einsum/blob/main/PUBLISHING.md",
            "PUBLISHING.md",
        )
        text = text.replace(
            "https://github.com/ti2-group/extended_einsum/blob/main/LICENSE",
            "LICENSE",
        )
        text = text.replace(
            "Copyright © 2026 FSU Theoretical Computer Science II.",
            "Copyright © 2026 contributors.",
        )
        text = re.sub(
            r"See \[CONTRIBUTING\.md\]\(CONTRIBUTING\.md\) for development "
            r"details and \[PUBLISHING\.md\]\(PUBLISHING\.md\) for the release "
            r"process\.\n?",
            "",
            text,
        )

    elif relative_path == PurePosixPath("pyproject.toml"):
        text = re.sub(
            r"^authors = \[\n.*?^\]\n",
            'authors = [{ name = "Anonymous contributors" }]\n',
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        text = re.sub(
            r"^\[project\.urls\]\n.*?(?=^\[|\Z)",
            "",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )

    elif relative_path == PurePosixPath("LICENSE"):
        text = re.sub(
            r"^Copyright \(c\) 2026 .*$",
            "Copyright (c) 2026 contributors",
            text,
            flags=re.MULTILINE,
        )

    elif relative_path == PurePosixPath("experiments/run_all.sh"):
        text = re.sub(
            r'^UV_EXECUTABLE="\$\{EXTENDED_EINSUM_UV_BIN:-.*\}"$',
            'UV_EXECUTABLE="${EXTENDED_EINSUM_UV_BIN:-uv}"',
            text,
            flags=re.MULTILINE,
        )

    return text


def sanitize_file(relative_path: PurePosixPath, data: bytes) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data

    text = apply_file_specific_cleanup(relative_path, text)
    for pattern in DENIED_PATTERNS:
        text = pattern.sub("anonymous", text)
    return text.encode("utf-8")


def printable_strings(data: bytes, minimum_length: int = 6) -> str:
    ascii_runs = re.findall(rb"[\x20-\x7e]{%d,}" % minimum_length, data)
    utf16_runs = re.findall(rb"(?:[\x20-\x7e]\x00){%d,}" % minimum_length, data)
    decoded = [run.decode("ascii", errors="ignore") for run in ascii_runs]
    decoded.extend(run[::2].decode("ascii", errors="ignore") for run in utf16_runs)
    return "\n".join(decoded)


def inspect_pdf(path: Path) -> str:
    tools = ("pdfinfo", "pdftotext")
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("PDF validation requires Poppler utilities; missing: " + ", ".join(missing))

    outputs: list[str] = []
    commands = (("pdfinfo", str(path)), ("pdftotext", str(path), "-"))
    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise RuntimeError(f"could not inspect PDF {path}: {detail}")
        outputs.append(result.stdout)
    return "\n".join(outputs)


def validate_staged_file(path: Path, relative_path: PurePosixPath) -> None:
    name_matches = denied_matches(relative_path.as_posix())
    if name_matches:
        raise RuntimeError(f"denied marker in archive path {relative_path}: {', '.join(name_matches)}")

    data = path.read_bytes()
    try:
        visible_text = data.decode("utf-8")
    except UnicodeDecodeError:
        if path.suffix.lower() == ".pdf":
            visible_text = inspect_pdf(path)
            content_matches = denied_matches(visible_text)
        else:
            visible_text = printable_strings(data)
            content_matches = denied_binary_matches(visible_text)
    else:
        content_matches = denied_matches(visible_text)
    if content_matches:
        raise RuntimeError(f"denied marker in archive content {relative_path}: " + ", ".join(content_matches))


def reproduction_notes(dataset_files: list[PurePosixPath]) -> bytes:
    image_shards = [path for path in dataset_files if path.suffix.lower() == ".npz" and "imagenet" in path.as_posix().lower()]
    image_data_note = (
        "The locally available ImageNet64 NPZ shards are included."
        if image_shards
        else (
            "The official ImageNet64 NPZ shards were not present in the source "
            "checkout and therefore cannot be redistributed here. Download "
            "ImageNet64 directly from the [official ImageNet download page]"
            "(https://www.image-net.org/download-images.php), prepare NPZ shards "
            "with `data` and `labels` arrays, and place them under `datasets/` "
            "before rerunning the Monarch measurement. The expected layouts are "
            "documented in `experiments/monarch/README.md`."
        )
    )
    text = f"""
## Reproducing the results

1. Install `uv`.
2. Run `uv sync --group demo --group dev`.
3. Follow `experiments/README.md` for individual measurements, smoke tests,
   validation, and plotting.
4. Run `CUDA_VISIBLE_DEVICES=0 experiments/run_all.sh` for the sequential
   publication workflow.

MNIST is included when it is available in the source checkout. {image_data_note}

`SHA256SUMS` records every payload file other than itself. The archive builder
checks paths, readable text, PDF metadata, and extracted PDF text before it
writes the final archive.
"""
    return text.encode("utf-8")


def write_staging_tree(stage_root: Path, selected: list[Path]) -> list[PurePosixPath]:
    payload_paths: list[PurePosixPath] = []
    dataset_paths: list[PurePosixPath] = []
    for source in selected:
        relative_path = PurePosixPath(source.relative_to(PROJECT_ROOT).as_posix())
        destination = stage_root.joinpath(*relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sanitize_file(relative_path, source.read_bytes()))
        mode = 0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644
        destination.chmod(mode)
        payload_paths.append(relative_path)
        if relative_path.parts[0] == "datasets":
            dataset_paths.append(relative_path)

    notes_path = PurePosixPath("REPRODUCTION.md")
    stage_root.joinpath(*notes_path.parts).write_bytes(reproduction_notes(dataset_paths))
    payload_paths.append(notes_path)

    checksums = []
    for relative_path in sorted(payload_paths, key=PurePosixPath.as_posix):
        data = stage_root.joinpath(*relative_path.parts).read_bytes()
        checksums.append(f"{hashlib.sha256(data).hexdigest()}  {relative_path.as_posix()}")
    checksum_path = PurePosixPath("SHA256SUMS")
    stage_root.joinpath(*checksum_path.parts).write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    payload_paths.append(checksum_path)
    return sorted(payload_paths, key=PurePosixPath.as_posix)


def validate_staging_tree(stage_root: Path, payload_paths: Iterable[PurePosixPath]) -> None:
    for relative_path in payload_paths:
        if any(part.startswith(".git") for part in relative_path.parts):
            raise RuntimeError(f"version-control metadata selected: {relative_path}")
        validate_staged_file(stage_root.joinpath(*relative_path.parts), relative_path)


def source_date_epoch() -> int:
    value = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        epoch = int(value)
    except ValueError as error:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
    return epoch


def build_archive(
    output: Path,
    stage_root: Path,
    payload_paths: Iterable[PurePosixPath],
    epoch: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        zip_epoch = max(epoch, 315532800)
        zip_timestamp = time.gmtime(zip_epoch)[:6]
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative_path in payload_paths:
                source = stage_root.joinpath(*relative_path.parts)
                archive_name = PurePosixPath(ARCHIVE_ROOT) / relative_path
                info = zipfile.ZipInfo(archive_name.as_posix(), date_time=zip_timestamp)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = stat.S_IMODE(source.stat().st_mode) << 16
                archive.writestr(
                    info,
                    source.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_finished_archive(
    output: Path,
    stage_root: Path,
    payload_paths: list[PurePosixPath],
) -> None:
    expected_names = {(PurePosixPath(ARCHIVE_ROOT) / relative_path).as_posix() for relative_path in payload_paths}
    with zipfile.ZipFile(output, "r") as archive:
        members = archive.infolist()
        actual_names = {member.filename for member in members}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise RuntimeError(f"archive member mismatch; missing={missing}, extra={extra}")
        if any(member.is_dir() for member in members):
            raise RuntimeError("archive contains a non-regular-file member")
        if any(part.startswith(".git") for name in actual_names for part in PurePosixPath(name).parts):
            raise RuntimeError("archive contains version-control metadata")

        for member in members:
            relative_path = PurePosixPath(*PurePosixPath(member.filename).parts[1:])
            archived_digest = hashlib.sha256(archive.read(member)).digest()
            staged_digest = hashlib.sha256(stage_root.joinpath(*relative_path.parts).read_bytes()).digest()
            if archived_digest != staged_digest:
                raise RuntimeError(f"archive content mismatch: {member.filename}")


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    output = args.output.expanduser().resolve()
    selected = selected_files()
    validate_required_inputs(selected)

    with tempfile.TemporaryDirectory(prefix="paper-reproduction-") as temporary:
        stage_root = Path(temporary) / ARCHIVE_ROOT
        stage_root.mkdir()
        payload_paths = write_staging_tree(stage_root, selected)
        validate_staging_tree(stage_root, payload_paths)
        build_archive(output, stage_root, payload_paths, source_date_epoch())
        validate_finished_archive(output, stage_root, payload_paths)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"created {output}")
    print(f"files: {len(payload_paths)}")
    print(f"bytes: {output.stat().st_size}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
