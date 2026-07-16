from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"


def run_example(name: str, tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *arguments],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "name",
    ["torch_quickstart.py", "numpy_backend.py", "stable_materialization.py"],
)
def test_base_examples_run_outside_repository(name: str, tmp_path: Path) -> None:
    completed = run_example(name, tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_jax_example_runs_outside_repository(tmp_path: Path) -> None:
    pytest.importorskip("jax")

    completed = run_example("jax_backend.py", tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_visualization_example_runs_outside_repository(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    output = tmp_path / "expression.png"

    completed = run_example("visualize_expression.py", tmp_path, str(output))

    assert completed.returncode == 0, completed.stderr
    assert output.stat().st_size > 0
