from dataclasses import dataclass
from typing import Generic, TypeVar, override

from extended_einsum.backend_translation import BackendArray
from extended_einsum.backends.registry import get_backend_of_array
from extended_einsum.interface.tensor_expression import (
    Parameter,
    TensorExpression,
)
from extended_einsum.language.rich_operators import (
    OperatorCos,
    OperatorEinsum,
    OperatorExp,
    OperatorInverse,
    OperatorLog,
    OperatorSelect,
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorSqrt,
    OperatorStack,
    OperatorTake,
    OperatorTan,
)
from extended_einsum.language.types import Array, Backend, Shape, TArray, TensorFormat
from extended_einsum.utils import normalize_axis, parse_format_string

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)

# Paper "Beyond Standard Einsum": these constructors deliberately expose
# einsum together with first-class intermediates, stack/take/select/slice,
# nonlinearities, softmax, and elementwise arithmetic. Keeping these as a
# restricted tensor language preserves the contraction and dataflow structure
# used by the compiler passes in sec:compiler-optimizations.


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


def array(backend_array: TBackendArray, format: TensorFormat = "dense", *, backend: Backend | None = None) -> BackendArrayWrapper[TBackendArray]:
    """Wraps a backend array for use in tensor expressions.

    The backend is detected from the array type; for backends registered
    without an ``is_array`` predicate, pass the backend name explicitly.
    """

    if backend is None:
        backend = get_backend_of_array(backend_array)
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
        raise ValueError(f"format string {format_string} has {len(index_strings)} indices, but {len(operands)} operands.")
    all_input_symbols = frozenset("".join(index_strings))
    if any(output_symbol not in all_input_symbols for output_symbol in output_string):
        raise ValueError(f"format string {format_string} contains output symbols that are not present in the operands.")
    return TensorExpression(OperatorEinsum(format_string), list(operands))


def stack(
    operands: list[TensorExpression[TArray] | Parameter[TArray] | TArray],
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(operands[0].shape))
    if len(operands) == 0:
        raise ValueError("stack requires at least one argument")
    if any(operand.shape != operands[0].shape for operand in operands[1:]):
        raise ValueError("The stack operator requires all arguments to have the same shape along the stack axis.")
    return TensorExpression(OperatorStack(axis), operands)


def take(
    source: TensorExpression[TArray] | TArray,
    index: TensorExpression[TArray] | TArray,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    if not source.shape:
        raise ValueError("The take operator requires an operand with a leading axis.")
    if not index.shape:
        raise ValueError("The take operator requires an index with a leading axis.")
    return TensorExpression(OperatorTake(axis), [source, index])


def slice(
    source: TensorExpression[TArray] | TArray,
    start: int,
    stop: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorSlice(start, stop, axis), [source])


def select(
    source: TensorExpression[TArray] | TArray,
    index: int,
    *,
    axis: int = 0,
) -> TensorExpression[TArray]:
    axis = normalize_axis(axis, len(source.shape))
    return TensorExpression(OperatorSelect(axis, index), [source])


def softmax(
    a: TensorExpression[TArray] | TArray,
    axis: int | tuple[int, ...] = 0,
) -> TensorExpression[TArray]:
    """Applies the softmax function to the input tensor."""

    if not a.shape:
        raise ValueError("softmax requires an input tensor with at least one axis")

    axes = (axis,) if isinstance(axis, int) else axis
    if not axes:
        raise ValueError("softmax requires at least one axis")
    normalized_axes = tuple(normalize_axis(item, len(a.shape)) for item in axes)
    normalized_axis: int | tuple[int, ...] = normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    return TensorExpression(OperatorSoftmax(normalized_axis), [a])
