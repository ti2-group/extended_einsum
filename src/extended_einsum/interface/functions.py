from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, override

from extended_einsum.backend_translation import BackendArray
from extended_einsum.backends.registry import get_backend_functions, get_backend_of_array
from extended_einsum.interface.tensor_expression import (
    Parameter,
    TensorExpression,
    as_expression_argument,
    getitem_expression,
    matmul_expression,
)
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorCos,
    OperatorDivide,
    OperatorEinsum,
    OperatorExp,
    OperatorInverse,
    OperatorLog,
    OperatorMultiply,
    OperatorSelect,
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorSqrt,
    OperatorStack,
    OperatorSubtract,
    OperatorTake,
    OperatorTan,
)
from extended_einsum.language.types import Array, Backend, Shape, StabilityMode, TArray, TensorFormat
from extended_einsum.utils import normalize_axis, parse_format_string

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


@dataclass(frozen=True)
class BackendArrayWrapper(Array, Generic[TBackendArray]):
    backend_array: TBackendArray
    _backend: Backend
    _format: TensorFormat

    @property
    @override
    def shape(self) -> Shape:
        return tuple(self.backend_array.shape)

    @property
    @override
    def backend(self) -> Backend:
        return self._backend

    @property
    @override
    def format(self) -> TensorFormat:
        return self._format

    def materialize(self, stability_mode: StabilityMode = "unstable") -> BackendArrayWrapper[TBackendArray]:
        """A wrapped array is already materialized; returns itself."""

        return self

    def __repr__(self) -> str:
        return f"array({self.backend_array!r}, format={self._format!r}, backend={self._backend!r})"

    def __add__(self, other: TensorExpression[Any] | Any) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return TensorExpression(OperatorAdd(), [self, other])

    def __sub__(self, other: TensorExpression[Any] | Any) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return TensorExpression(OperatorSubtract(), [self, other])

    def __mul__(self, other: TensorExpression[Any] | Any) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return TensorExpression(OperatorMultiply(), [self, other])

    def __truediv__(self, other: TensorExpression[Any] | Any) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return TensorExpression(OperatorDivide(), [self, other])

    def __matmul__(self, other: TensorExpression[Any] | Any) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return matmul_expression(self, other)

    def __getitem__(self, item: int | slice | tuple[int | slice, ...]) -> TensorExpression[BackendArrayWrapper[TBackendArray]]:
        return getitem_expression(self, item)


def array(backend_array: TBackendArray, format: TensorFormat = "dense", *, backend: Backend | None = None) -> BackendArrayWrapper[TBackendArray]:
    """Wraps a backend array for use in tensor expressions.

    The backend is detected from the array type; for backends registered
    without an ``is_array`` predicate, pass the backend name explicitly.
    """

    if backend is None:
        backend = get_backend_of_array(backend_array)
    else:
        # validate the name eagerly so a typo fails here, not at materialize
        get_backend_functions(backend)
    return BackendArrayWrapper(backend_array, backend, format)


def exp(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorExp(), [a])


def log(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorLog(), [a])


def sin(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorSin(), [a])


def cos(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorCos(), [a])


def tan(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorTan(), [a])


def sqrt(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorSqrt(), [a])


def inverse(
    a: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    return TensorExpression(OperatorInverse(), [a])


def einsum(
    format_string: str,
    *operands: TensorExpression[TArray] | TArray,
) -> TensorExpression[TArray]:
    index_strings, output_string = parse_format_string(format_string)
    if len(index_strings) != len(operands):
        operand_word = "operand was" if len(operands) == 1 else "operands were"
        raise ValueError(f"The einsum format string {format_string!r} has {len(index_strings)} input terms, but {len(operands)} {operand_word} given.")
    all_input_symbols = frozenset("".join(index_strings))
    missing_symbols = sorted(set(output_string) - all_input_symbols)
    if missing_symbols:
        raise ValueError(f"The einsum format string {format_string!r} has output symbols that do not appear in any input term: {', '.join(missing_symbols)}.")
    return TensorExpression(OperatorEinsum(format_string), list(operands))


def stack(
    operands: list[TensorExpression[TArray] | Parameter[TArray] | TArray],
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    if len(operands) == 0:
        raise ValueError("stack requires at least one operand")
    first_operand = cast("TensorExpression[TArray] | Parameter[TArray] | TArray", as_expression_argument(operands[0]))
    # the stack axis refers to the output, which has one more axis than the operands
    axis = normalize_axis(axis, len(first_operand.shape) + 1)
    return TensorExpression(OperatorStack(axis), operands)


def take(
    source: TensorExpression[TArray] | TArray,
    index: TensorExpression[TArray] | TArray,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = cast("TensorExpression[TArray] | TArray", as_expression_argument(source))
    index = cast("TensorExpression[TArray] | TArray", as_expression_argument(index))
    if not source.shape:
        raise ValueError("The take operator requires a non-scalar source, but the source has shape ().")
    if not index.shape:
        raise ValueError("The take operator requires an index vector with one axis, but the index has shape ().")
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorTake(axis), [source, index])


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = cast("TensorExpression[TArray] | TArray", as_expression_argument(source))
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorSlice(start, stop, axis), [source])


def select(
    source: TensorExpression[TArray] | TArray,
    index: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    source = cast("TensorExpression[TArray] | TArray", as_expression_argument(source))
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorSelect(axis, index), [source])


def softmax(
    a: TensorExpression[TArray] | TArray,
    *,
    axis: int | tuple[int, ...],
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor along the given axis or axes."""

    a = cast("TensorExpression[TArray] | TArray", as_expression_argument(a))
    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axes = (axis,) if isinstance(axis, int) else axis
    if not axes:
        raise ValueError("softmax requires at least one axis")
    normalized_axes = tuple(normalize_axis(item, len(a.shape)) for item in axes)
    normalized_axis: int | tuple[int, ...] = normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    return TensorExpression(OperatorSoftmax(normalized_axis), [a])
