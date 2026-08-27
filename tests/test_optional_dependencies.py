import builtins
from typing import Any

import pytest

from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import OperatorExp
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.visualization import build_expression_dag, plot_expression_dag


def simple_program() -> RichProgram:
    return RichProgram(
        instructions=[RichInstruction(OperatorExp(), (0,))],
        n_inputs=1,
        stability_mode="unstable",
        shapes=[(2,), (2,)],
        tensor_formats=["dense", "dense"],
        parameter_indices=frozenset(),
    )


def reject_import(monkeypatch: pytest.MonkeyPatch, blocked_name: str) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked_name or name.startswith(f"{blocked_name}."):
            raise ImportError(f"blocked optional dependency: {blocked_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_networkx_error_names_missing_dependency_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    reject_import(monkeypatch, "networkx")

    with pytest.raises(ImportError, match=r"requires networkx.*extended-einsum\[visualization\]"):
        build_expression_dag(simple_program())


def test_matplotlib_error_names_missing_dependency_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    reject_import(monkeypatch, "matplotlib")

    with pytest.raises(ImportError, match=r"requires matplotlib.*extended-einsum\[visualization\]"):
        plot_expression_dag(simple_program())
