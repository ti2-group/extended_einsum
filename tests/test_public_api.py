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
