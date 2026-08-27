"""Conformance checks for extended-einsum execution backends.

Custom backend authors register their backend and then call ``check_backend``
to run every supported operator through every stability mode, comparing the
results against the numpy reference backend.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from extended_einsum import interface
from extended_einsum.interface.tensor_expression import OperatorExpression, TensorExpression, TensorLeaf
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorConcat,
    OperatorDivide,
    OperatorMultiply,
    OperatorSubtract,
)
from extended_einsum.language.types import Backend, StabilityMode

_STABILITY_MODES: tuple[StabilityMode, ...] = (
    "unstable",
    "scaled_min",
    "scaled_max",
    "scaled_sum",
    "logspace_min",
    "logspace_max",
)


@dataclass(frozen=True)
class _ConformanceCase:
    name: str
    inputs: tuple[np.ndarray, ...]
    build: Callable[..., TensorExpression]


def _conformance_cases() -> list[_ConformanceCase]:
    # strictly positive inputs so that the scaled and logspace stability modes
    # are applicable to every case; the subtraction operands are chosen so
    # that the difference stays strictly positive as well
    a = np.array([[1.5, 2.0, 0.5], [3.0, 1.0, 2.5]])
    b = np.array([[0.5, 1.5, 0.25], [1.0, 0.5, 2.0]])
    v = np.array([2.0, 0.5, 1.5])
    m = np.array([[0.5, 1.0, 2.0], [1.5, 0.25, 0.75], [2.5, 1.25, 0.5]])
    indices = np.array([2, 0])

    return [
        _ConformanceCase("add", (a, b), lambda x, y: OperatorExpression(OperatorAdd(), [x, y])),
        _ConformanceCase("subtract", (a, b), lambda x, y: OperatorExpression(OperatorSubtract(), [x, y])),
        _ConformanceCase("multiply", (a, b), lambda x, y: OperatorExpression(OperatorMultiply(), [x, y])),
        _ConformanceCase("divide", (a, b), lambda x, y: OperatorExpression(OperatorDivide(), [x, y])),
        _ConformanceCase("exp", (a,), interface.exp),
        _ConformanceCase("log", (a,), interface.log),
        _ConformanceCase("softmax", (a,), lambda x: interface.softmax(x, axis=1)),
        _ConformanceCase("einsum_contraction", (a, m), lambda x, y: interface.einsum("ij, jk -> ik", x, y)),
        _ConformanceCase("einsum_contraction_free", (v, a), lambda x, y: interface.einsum("j, ij -> ij", x, y)),
        _ConformanceCase("einsum_full_reduction", (a,), lambda x: interface.einsum("ij -> ", x)),
        _ConformanceCase("stack_leading_axis", (a, b), lambda x, y: interface.stack([x, y], axis=0)),
        _ConformanceCase("stack_trailing_axis", (a, b), lambda x, y: interface.stack([x, y], axis=1)),
        _ConformanceCase("concat_leading_axis", (a, b), lambda x, y: OperatorExpression(OperatorConcat(0), [x, y])),
        _ConformanceCase("concat_trailing_axis", (a, b), lambda x, y: OperatorExpression(OperatorConcat(1), [x, y])),
        _ConformanceCase("take", (a, indices), lambda x, i: interface.take(x, i, axis=1)),
        _ConformanceCase("select", (a,), lambda x: interface.select(x, 1, axis=0)),
        _ConformanceCase("slice", (a,), lambda x: interface.slice(x, 0, 2, axis=1)),
        _ConformanceCase("composite", (a, b), lambda x, y: interface.log(interface.exp(x) + interface.exp(y)) * interface.softmax(x, axis=1)),
    ]


def check_backend(
    backend: Backend,
    *,
    from_numpy: Callable[[np.ndarray], Any],
    to_numpy: Callable[[Any], np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Checks a registered backend against the numpy reference backend.

    Runs every supported operator through every stability mode using the
    backend as registered (including its compiler) and compares the results
    against the numpy backend in unstable mode. Operator/mode combinations
    that raise ``NotImplementedError`` are skipped, matching the library's
    contract for unsupported combinations.

    Parameters
    ----------
    backend : Backend
        Name of the backend to check; must already be registered.
    from_numpy : Callable[[np.ndarray], Any]
        Converts a numpy input array into one of the backend's arrays.
    to_numpy : Callable[[Any], np.ndarray]
        Converts one of the backend's arrays back into a numpy array.
    rtol, atol : float
        Tolerances for the comparison, e.g. loosened for backends computing
        in single precision.

    Raises
    ------
    AssertionError
        If any operator/mode combination produces a wrong result or raises an
        unexpected error. The message lists every failing combination.
    """

    failures: list[str] = []
    for case in _conformance_cases():
        reference_expression = case.build(*[TensorLeaf(input_array, backend="numpy") for input_array in case.inputs])
        reference = np.asarray(reference_expression.materialize("unstable"))
        for stability_mode in _STABILITY_MODES:
            wrapped_inputs = [TensorLeaf(from_numpy(input_array), backend=backend) for input_array in case.inputs]
            try:
                result = case.build(*wrapped_inputs).materialize(stability_mode)
            except NotImplementedError:
                continue
            except Exception as error:  # noqa: BLE001 - collected into the conformance report
                failures.append(f"{case.name} [{stability_mode}]: raised {type(error).__name__}: {error}")
                continue
            result_numpy = np.asarray(to_numpy(result))
            if result_numpy.shape != reference.shape:
                failures.append(f"{case.name} [{stability_mode}]: result shape {result_numpy.shape} does not match reference shape {reference.shape}")
            elif not np.allclose(result_numpy, reference, rtol=rtol, atol=atol):
                failures.append(f"{case.name} [{stability_mode}]: result does not match the numpy reference (max abs diff {np.max(np.abs(result_numpy - reference))})")
    if failures:
        raise AssertionError(f"The backend {backend!r} failed {len(failures)} conformance check(s):\n" + "\n".join(failures))
