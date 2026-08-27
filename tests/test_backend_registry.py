from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

import extended_einsum as xe
from extended_einsum.backend_translation.backend import BackendFunctions
from extended_einsum.backends.numpy import NumpyBackendFunctions
from extended_einsum.backends.registry import (
    get_backend_compiler,
    get_backend_functions,
    get_backend_of_array,
    register_backend,
)
from extended_einsum.testing import check_backend


@dataclass(frozen=True)
class BoxedArray:
    """Minimal custom array type wrapping numpy, supporting the standard operator and indexing protocols."""

    data: npt.NDArray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: "BoxedArray") -> "BoxedArray":
        return BoxedArray(self.data + other.data)

    def __sub__(self, other: "BoxedArray") -> "BoxedArray":
        return BoxedArray(self.data - other.data)

    def __mul__(self, other: "BoxedArray") -> "BoxedArray":
        return BoxedArray(self.data * other.data)

    def __truediv__(self, other: "BoxedArray") -> "BoxedArray":
        return BoxedArray(self.data / other.data)

    def __getitem__(self, item: object) -> "BoxedArray":
        return BoxedArray(np.asarray(self.data[item]))


class BoxedBackendFunctions(BackendFunctions[BoxedArray]):
    """Implements only the abstract primitives; every derived operation uses the BackendFunctions defaults."""

    def exp(self, array: BoxedArray) -> BoxedArray:
        return BoxedArray(np.exp(array.data))

    def log(self, array: BoxedArray) -> BoxedArray:
        return BoxedArray(np.log(array.data))

    def sum(self, array: BoxedArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BoxedArray:
        return BoxedArray(np.asarray(np.sum(array.data, axis=axis, keepdims=keepdims)))

    def max(self, array: BoxedArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BoxedArray:
        return BoxedArray(np.asarray(np.max(array.data, axis=axis, keepdims=keepdims)))

    def min(self, array: BoxedArray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> BoxedArray:
        return BoxedArray(np.asarray(np.min(array.data, axis=axis, keepdims=keepdims)))

    def maximum(self, array_1: BoxedArray, array_2: BoxedArray) -> BoxedArray:
        return BoxedArray(np.maximum(array_1.data, array_2.data))

    def reshape(self, array: BoxedArray, shape: tuple[int, ...]) -> BoxedArray:
        return BoxedArray(np.reshape(array.data, shape))

    def broadcast_to(self, array: BoxedArray, shape: tuple[int, ...]) -> BoxedArray:
        return BoxedArray(np.broadcast_to(array.data, shape))

    def stack(self, arrays: Sequence[BoxedArray], axis: int) -> BoxedArray:
        return BoxedArray(np.stack([array.data for array in arrays], axis=axis))

    def concat(self, arrays: Sequence[BoxedArray], axis: int) -> BoxedArray:
        return BoxedArray(np.concatenate([array.data for array in arrays], axis=axis))

    def take(self, array: BoxedArray, indices: BoxedArray, axis: int) -> BoxedArray:
        return BoxedArray(np.take(array.data, indices.data, axis=axis))

    def einsum(self, format_string: str, *operands: BoxedArray) -> BoxedArray:
        return BoxedArray(np.einsum(format_string, *[operand.data for operand in operands]))


def _is_boxed_array(array: object) -> bool:
    return isinstance(array, BoxedArray)


register_backend("boxed", BoxedBackendFunctions(), is_array=_is_boxed_array)


def test_custom_backend_is_detected_and_materializes_with_default_compiler() -> None:
    source_numpy = np.array([[1.0, 2.0], [3.0, 4.0]])
    expression = xe.exp(BoxedArray(source_numpy))
    assert expression.backend == "boxed"

    result = expression.materialize()

    np.testing.assert_allclose(result.data, np.exp(source_numpy))


def test_custom_backend_passes_conformance_including_default_implementations() -> None:
    check_backend("boxed", from_numpy=BoxedArray, to_numpy=lambda array: array.data)


def test_later_registered_detection_predicate_takes_precedence() -> None:
    class SubclassArray(np.ndarray):
        pass

    register_backend("boxed_subclass", NumpyBackendFunctions(), is_array=lambda array: type(array) is SubclassArray)

    subclass_array = np.array([1.0, 2.0]).view(SubclassArray)
    assert get_backend_of_array(subclass_array) == "boxed_subclass"
    assert get_backend_of_array(np.array([1.0, 2.0])) == "numpy"


def test_explicit_backend_name_overrides_detection() -> None:
    leaf = xe.TensorLeaf(np.array([1.0, 2.0]), backend="boxed")
    assert leaf.backend == "boxed"


def test_incomplete_backend_functions_subclass_fails_at_instantiation() -> None:
    class IncompleteBackendFunctions(BackendFunctions[BoxedArray]):
        def exp(self, array: BoxedArray) -> BoxedArray:
            return BoxedArray(np.exp(array.data))

    with pytest.raises(TypeError):
        IncompleteBackendFunctions()  # type: ignore[abstract]


def test_register_backend_rejects_duck_typed_object_missing_methods() -> None:
    class DuckFunctions:
        def exp(self, array: BoxedArray) -> BoxedArray:
            return BoxedArray(np.exp(array.data))

    with pytest.raises(TypeError, match="missing the methods"):
        register_backend("duck", DuckFunctions())  # type: ignore[arg-type]


def test_register_backend_rejects_invalid_names_and_compilers() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        register_backend("", BoxedBackendFunctions())
    with pytest.raises(TypeError, match="callable compile method"):
        register_backend("boxed_bad_compiler", BoxedBackendFunctions(), compiler=object())  # type: ignore[arg-type]


def test_unknown_backend_name_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="No backend is registered under the name 'nope'"):
        get_backend_functions("nope")
    with pytest.raises(ValueError, match="register_backend"):
        get_backend_compiler("nope")


def test_undetectable_array_type_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unsupported array type"):
        xe.TensorLeaf([1.0, 2.0])  # type: ignore[arg-type]
