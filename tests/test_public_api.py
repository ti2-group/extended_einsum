from importlib.metadata import version

import numpy as np
import pytest
import torch

import extended_einsum as xe
import extended_einsum.interface as interface
from extended_einsum.backends import registry

PUBLIC_OPERATIONS = {
    "TensorExpression",
    "array",
    "cos",
    "einsum",
    "exp",
    "extract_program",
    "inverse",
    "log",
    "select",
    "sin",
    "slice",
    "softmax",
    "sqrt",
    "stack",
    "take",
    "tan",
}


def test_top_level_api_matches_interface_namespace() -> None:
    assert PUBLIC_OPERATIONS <= set(xe.__all__)
    assert PUBLIC_OPERATIONS == set(interface.__all__)
    for name in PUBLIC_OPERATIONS:
        assert getattr(xe, name) is getattr(interface, name)


def test_version_comes_from_distribution_metadata() -> None:
    assert xe.__version__ == version("extended-einsum")


def test_torch_backend_is_available_in_base_install() -> None:
    source_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = xe.array(source_torch)

    result = xe.exp(source).materialize()

    torch.testing.assert_close(result.backend_array, torch.exp(source_torch))


def test_numpy_backend_is_available_in_base_install() -> None:
    source_numpy = np.array([[1.0, 2.0], [3.0, 4.0]])
    source = xe.array(source_numpy)

    result = xe.exp(source).materialize()

    np.testing.assert_allclose(result.backend_array, np.exp(source_numpy))


def test_missing_jax_backend_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(registry.BACKEND_TO_FUNCTIONS, "jax", raising=False)
    monkeypatch.delitem(registry.BACKEND_TO_COMPILER, "jax", raising=False)

    with pytest.raises(ModuleNotFoundError, match=r"extended-einsum\[jax\]"):
        registry.get_backend_functions("jax")
    with pytest.raises(ModuleNotFoundError, match=r"extended-einsum\[jax\]"):
        registry.get_backend_compiler("jax")


def test_extract_program_handles_expressions_deeper_than_the_recursion_limit() -> None:
    # Learned tree-structured circuits (e.g. HCLTs) produce expression graphs
    # far deeper than Python's default recursion limit.
    depth = 5_000
    source = xe.array(np.array([1.0, 2.0]))
    expression = xe.exp(source)
    for _ in range(depth - 1):
        expression = xe.exp(expression)

    program, inputs = xe.extract_program(expression, stability_mode="unstable")

    assert len(program.instructions) == depth
    assert program.n_inputs == 1
    assert len(inputs) == 1


def test_materialize_handles_expressions_deeper_than_the_recursion_limit() -> None:
    depth = 2_048
    source_numpy = np.ones((2, 2))
    source = xe.array(source_numpy)
    expression = xe.log(xe.exp(source))
    for _ in range(depth):
        expression = expression * source

    result = expression.materialize()

    np.testing.assert_allclose(result.backend_array, source_numpy)
