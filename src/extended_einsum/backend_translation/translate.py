from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Generic, Literal, TypeVar

from extended_einsum.backend_translation.backend import BackendArray, BackendFunctions, BackendProgram
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorDivide,
    OperatorEinsum,
    OperatorExp,
    OperatorLog,
    OperatorMultiply,
    OperatorSelect,
    OperatorSlice,
    OperatorSoftmax,
    OperatorStack,
    OperatorSubtract,
    OperatorTake,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.utils import parse_format_string

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)


def translate_to_backend_program(rich_program: RichProgram, backend_functions: BackendFunctions[TBackendArray]) -> BackendProgram[TBackendArray]:
    match rich_program.stability_mode:
        case "unstable":
            return _translate_unstable(rich_program, backend_functions)
        case "scaled_min" | "scaled_sum":
            return _translate_scaled(rich_program, backend_functions, rich_program.stability_mode)
        case "logspace_min" | "logspace_max":
            return _translate_logspace(rich_program, backend_functions, rich_program.stability_mode)
        case _:
            raise NotImplementedError(f"The translation of rich programs to backend programs is not implemented for stability mode {rich_program.stability_mode}.")


def _unwrap_and_execute(backend_function: Callable[..., TBackendArray], tensor_arguments: Sequence[TBackendArray]) -> TBackendArray:
    return backend_function(*tensor_arguments)


def _to_backend_call(backend_function: Callable[..., TBackendArray]) -> Callable[[Sequence[TBackendArray]], TBackendArray]:
    return partial(_unwrap_and_execute, backend_function)


def _operator_to_backend_call(operator: RichOperator, backend_functions: BackendFunctions[TBackendArray]) -> Callable[[Sequence[TBackendArray]], TBackendArray]:
    """Binds an operator to its backend function, resulting in a call that takes only the tensor arguments."""

    match operator:
        case OperatorExp():
            backend_function = backend_functions.exp
        case OperatorLog():
            backend_function = backend_functions.log
        case OperatorAdd():
            backend_function = backend_functions.add
        case OperatorSubtract():
            backend_function = backend_functions.subtract
        case OperatorMultiply():
            backend_function = backend_functions.multiply
        case OperatorDivide():
            backend_function = backend_functions.divide
        case OperatorStack(axis):
            # stack takes the tensors as a sequence instead of separate arguments
            return partial(backend_functions.stack, axis=axis)
        case OperatorTake(axis):
            backend_function = partial(backend_functions.take, axis=axis)
        case OperatorSelect(axis, index):
            backend_function = partial(backend_functions.select, axis=axis, index=index)
        case OperatorSlice(start, stop, axis):
            backend_function = partial(backend_functions.slice, axis=axis, start=start, stop=stop)
        case OperatorSoftmax(axis):
            backend_function = partial(backend_functions.softmax, axis=axis)
        case OperatorEinsum(format_string):
            backend_function = partial(backend_functions.einsum, format_string)
        case _:
            raise NotImplementedError(f"The operator {operator.name} has no backend function.")
    return _to_backend_call(backend_function)


@dataclass
class _ProgramBuilder(Generic[TBackendArray]):
    """Collects backend calls. The input tensors occupy the positions 0 to n_inputs - 1, each call result gets the next free position."""

    n_inputs: int
    backend_calls: list[Callable[[Sequence[TBackendArray]], TBackendArray]] = field(default_factory=list)
    call_arguments: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def last_position(self) -> int:
        return self.n_inputs + len(self.backend_calls) - 1

    def append(self, backend_call: Callable[[Sequence[TBackendArray]], TBackendArray], argument_positions: tuple[int, ...]) -> int:
        self.backend_calls.append(backend_call)
        self.call_arguments.append(argument_positions)
        return self.last_position

    def wrap_and_append(self, backend_function: Callable[..., TBackendArray], argument_positions: tuple[int, ...]) -> int:
        return self.append(_to_backend_call(backend_function), argument_positions)

    def build(self) -> BackendProgram[TBackendArray]:
        return BackendProgram(self.backend_calls, self.call_arguments, self.n_inputs)


def _translate_unstable(rich_program: RichProgram, backend_functions: BackendFunctions[TBackendArray]) -> BackendProgram[TBackendArray]:
    builder = _ProgramBuilder(rich_program.n_inputs)
    for instruction in rich_program.instructions:
        backend_call = _operator_to_backend_call(instruction.operator, backend_functions)
        builder.append(backend_call, instruction.argument_ssa_ids)
    return builder.build()


# in the scaled translation, an SSA value is available in some of three parts:
# "raw" is the value itself, "normalized" and "log_scale" form a scaled pair with raw = normalized * exp(log_scale), where the log scale is a scalar
_ScaledPart = Literal["raw", "normalized", "log_scale"]
_ScaledPositions = dict[tuple[int, _ScaledPart], int]


def _translate_scaled(rich_program: RichProgram, backend_functions: BackendFunctions[TBackendArray], stability_mode: Literal["scaled_min", "scaled_sum"]) -> BackendProgram[TBackendArray]:
    """Translates a rich program into a backend program whose intermediate values are scaled pairs of a normalized tensor and a scalar log scale.

    Scaling is lazy: a value stays a raw tensor until an operator consumes it as a scaled pair, then it is normalized by its total sum
    (scaled_sum) or its minimum (scaled_min), which assumes strictly positive values. The log and softmax operators eliminate the scale again,
    so their results are stored as raw tensors.
    """

    builder = _ProgramBuilder(rich_program.n_inputs)
    positions: _ScaledPositions = {(ssa_id, "raw"): ssa_id for ssa_id in range(rich_program.n_inputs)}
    for instruction_index, instruction in enumerate(rich_program.instructions):
        _append_scaled_instruction(
            builder,
            backend_functions,
            positions,
            instruction,
            rich_program.n_inputs + instruction_index,
            stability_mode,
        )
    output_position = _raw_position(builder, backend_functions, positions, rich_program.output_ssa)
    if output_position != builder.last_position:
        raise ValueError("The raw value of the program's output must be the last computed tensor, because the runtime returns the last tensor.")
    return builder.build()


def _raw_position(builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], positions: _ScaledPositions, ssa_id: int) -> int:
    """Position of the value as a raw tensor, converting it from its scaled pair at most once."""

    if (ssa_id, "raw") not in positions:
        scale_factor_position = builder.wrap_and_append(backend_functions.exp, (positions[ssa_id, "log_scale"],))
        positions[ssa_id, "raw"] = builder.wrap_and_append(backend_functions.multiply, (positions[ssa_id, "normalized"], scale_factor_position))
    return positions[ssa_id, "raw"]


def _scaled_positions(
    builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], positions: _ScaledPositions, ssa_id: int, stability_mode: Literal["scaled_min", "scaled_sum"]
) -> tuple[int, int]:
    """Positions of the value as a scaled pair, converting it from its raw tensor at most once by normalizing with its total sum (scaled_sum) or its minimum (scaled_min)."""

    if (ssa_id, "normalized") not in positions:
        raw_position = positions[ssa_id, "raw"]
        match stability_mode:
            case "scaled_min":
                total_position = builder.wrap_and_append(backend_functions.min, (raw_position,))
            case "scaled_sum":
                total_position = builder.wrap_and_append(backend_functions.sum, (raw_position,))
        positions[ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (raw_position, total_position))
        positions[ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.log, (total_position,))
    return positions[ssa_id, "normalized"], positions[ssa_id, "log_scale"]


def _append_rescaled_values(builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], scaled_pairs: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Brings scaled pairs to a common log scale, the maximum of their log scales, and returns the positions of the rescaled values and of the common log scale."""

    stacked_scales_position = builder.append(partial(backend_functions.stack, axis=0), tuple(log_scale_position for _, log_scale_position in scaled_pairs))
    common_scale_position = builder.wrap_and_append(backend_functions.max, (stacked_scales_position,))
    rescaled_positions: list[int] = []
    for normalized_position, log_scale_position in scaled_pairs:
        shift_position = builder.wrap_and_append(backend_functions.subtract, (log_scale_position, common_scale_position))
        factor_position = builder.wrap_and_append(backend_functions.exp, (shift_position,))
        rescaled_positions.append(builder.wrap_and_append(backend_functions.multiply, (normalized_position, factor_position)))
    return rescaled_positions, common_scale_position


def _append_scaled_einsum(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _ScaledPositions,
    format_string: str,
    argument_ssa_ids: tuple[int, ...],
    result_ssa_id: int,
    stability_mode: Literal["scaled_min", "scaled_sum"],
) -> None:
    """Einsum on the normalized operands, renormalized by the total sum of its result; the operand log scales add up in the result's log scale."""

    scaled_pairs = [_scaled_positions(builder, backend_functions, positions, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
    raw_einsum_position = builder.wrap_and_append(partial(backend_functions.einsum, format_string), tuple(normalized_position for normalized_position, _ in scaled_pairs))
    match stability_mode:
        case "scaled_min":
            raw_norm_position = builder.wrap_and_append(backend_functions.min, (raw_einsum_position,))
        case "scaled_sum":
            raw_norm_position = builder.wrap_and_append(backend_functions.sum, (raw_einsum_position,))
    positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (raw_einsum_position, raw_norm_position))
    log_scale_position = builder.wrap_and_append(backend_functions.log, (raw_norm_position,))
    for _, operand_scale_position in scaled_pairs:
        log_scale_position = builder.wrap_and_append(backend_functions.add, (log_scale_position, operand_scale_position))
    positions[result_ssa_id, "log_scale"] = log_scale_position


def _append_scaled_instruction(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _ScaledPositions,
    instruction: RichInstruction,
    result_ssa_id: int,
    stability_mode: Literal["scaled_min", "scaled_sum"],
) -> None:
    argument_ssa_ids = instruction.argument_ssa_ids
    match instruction.operator:
        case OperatorExp():
            # exp(a) = exp(a - m) * e^m with the scalar m = max(a): the raw argument moves into the log scale, keeping the normalized values in (0, 1]
            # (entries whose exponent is far below the maximum underflow to zero, the same tradeoff as in softmax)
            raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
            log_scale_position = builder.wrap_and_append(backend_functions.max, (raw_position,))
            shifted_position = builder.wrap_and_append(backend_functions.subtract, (raw_position, log_scale_position))
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.exp, (shifted_position,))
            positions[result_ssa_id, "log_scale"] = log_scale_position
        case OperatorLog():
            # log(n * e^s) = log(n) + s, which is safe to store as a raw tensor
            normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, argument_ssa_ids[0], stability_mode)
            log_position = builder.wrap_and_append(backend_functions.log, (normalized_position,))
            positions[result_ssa_id, "raw"] = builder.wrap_and_append(backend_functions.add, (log_position, log_scale_position))
        case OperatorSoftmax(_):
            # softmax is shifted by its maximum internally and its result sums to one along the axis, so raw in and raw out is safe
            raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
            positions[result_ssa_id, "raw"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (raw_position,))
        case OperatorAdd() | OperatorSubtract():
            # n1 * e^s1 ± n2 * e^s2 = (n1 * e^(s1 - m) ± n2 * e^(s2 - m)) * e^m with m = max(s1, s2)
            combine = backend_functions.add if isinstance(instruction.operator, OperatorAdd) else backend_functions.subtract
            scaled_pairs = [_scaled_positions(builder, backend_functions, positions, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
            rescaled_positions, common_scale_position = _append_rescaled_values(builder, backend_functions, scaled_pairs)
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(combine, tuple(rescaled_positions))
            positions[result_ssa_id, "log_scale"] = common_scale_position
        case OperatorMultiply():
            # the normalized values multiply and the log scales add up
            (normalized_1, log_scale_1), (normalized_2, log_scale_2) = (
                _scaled_positions(builder, backend_functions, positions, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids
            )
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.multiply, (normalized_1, normalized_2))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.add, (log_scale_1, log_scale_2))
        case OperatorDivide():
            # the normalized values divide and the divisor's log scale is subtracted from the dividend's
            (normalized_1, log_scale_1), (normalized_2, log_scale_2) = (
                _scaled_positions(builder, backend_functions, positions, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids
            )
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (normalized_1, normalized_2))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.subtract, (log_scale_1, log_scale_2))
        case OperatorStack(axis):
            # bring all operands to a common log scale, then stack the rescaled values
            scaled_pairs = [_scaled_positions(builder, backend_functions, positions, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
            rescaled_positions, common_scale_position = _append_rescaled_values(builder, backend_functions, scaled_pairs)
            positions[result_ssa_id, "normalized"] = builder.append(partial(backend_functions.stack, axis=axis), tuple(rescaled_positions))
            positions[result_ssa_id, "log_scale"] = common_scale_position
        case OperatorTake(_):
            # indexing the normalized values does not change the scalar log scale; the indices are integer positions and always used raw
            normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, argument_ssa_ids[0], stability_mode)
            indices_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[1])
            positions[result_ssa_id, "normalized"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (normalized_position, indices_position))
            positions[result_ssa_id, "log_scale"] = log_scale_position
        case OperatorSelect(_, _) | OperatorSlice(_, _, _):
            # indexing the normalized values does not change the scalar log scale
            normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, argument_ssa_ids[0], stability_mode)
            positions[result_ssa_id, "normalized"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (normalized_position,))
            positions[result_ssa_id, "log_scale"] = log_scale_position
        case OperatorEinsum(format_string):
            _append_scaled_einsum(builder, backend_functions, positions, format_string, argument_ssa_ids, result_ssa_id, stability_mode)
        case _:
            raise NotImplementedError(f"The operator {instruction.operator.name} has no scaled backend translation.")


# in the logspace translation, an SSA value is available in some of two parts:
# "raw" is the value itself and "logspace" is its natural logarithm, with raw = exp(logspace)
_LogspacePart = Literal["raw", "logspace"]
_LogspacePositions = dict[tuple[int, _LogspacePart], int]


def _parameter_derived_ssa_ids(rich_program: RichProgram) -> frozenset[int]:
    """SSA values whose complete dependency chain consists of parameters."""

    parameter_derived = set(rich_program.parameter_indices)
    for instruction_index, instruction in enumerate(rich_program.instructions):
        if instruction.argument_ssa_ids and all(argument_ssa_id in parameter_derived for argument_ssa_id in instruction.argument_ssa_ids):
            parameter_derived.add(rich_program.n_inputs + instruction_index)
    return frozenset(parameter_derived)


def _translate_logspace(rich_program: RichProgram, backend_functions: BackendFunctions[TBackendArray], stability_mode: Literal["logspace_min", "logspace_max"]) -> BackendProgram[TBackendArray]:
    """Translates a rich program into a backend program whose intermediate values are the natural logarithms of the actual values.

    Conversion is lazy: a value stays a raw tensor until an operator consumes it in logspace, then it is converted with log,
    which assumes strictly positive values. The exp and log operators only move values between the raw and logspace parts, so a
    chain like log(einsum(exp(a), exp(b))) never materializes the raw exponentials.
    """

    builder = _ProgramBuilder(rich_program.n_inputs)
    positions: _LogspacePositions = {(ssa_id, "raw"): ssa_id for ssa_id in range(rich_program.n_inputs)}
    parameter_derived_ssa_ids = _parameter_derived_ssa_ids(rich_program)
    for instruction_index, instruction in enumerate(rich_program.instructions):
        _append_logspace_instruction(
            builder,
            backend_functions,
            positions,
            instruction,
            rich_program.n_inputs + instruction_index,
            stability_mode,
            parameter_derived_ssa_ids,
        )
    output_position = _as_raw_position(builder, backend_functions, positions, rich_program.output_ssa)
    if output_position != builder.last_position:
        raise ValueError("The raw value of the program's output must be the last computed tensor, because the runtime returns the last tensor.")
    return builder.build()


def _as_raw_position(builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], positions: _LogspacePositions, ssa_id: int) -> int:
    """Position of the value as a raw tensor, converting it from its logspace part at most once."""

    if (ssa_id, "raw") not in positions:
        positions[ssa_id, "raw"] = builder.wrap_and_append(backend_functions.exp, (positions[ssa_id, "logspace"],))
    return positions[ssa_id, "raw"]


def _as_logspace_position(builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], positions: _LogspacePositions, ssa_id: int) -> int:
    """Position of the value's natural logarithm, converting it from its raw tensor at most once, which assumes strictly positive values."""

    if (ssa_id, "logspace") not in positions:
        positions[ssa_id, "logspace"] = builder.wrap_and_append(backend_functions.log, (positions[ssa_id, "raw"],))
    return positions[ssa_id, "logspace"]


def _append_shifted(
    builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], log_positions: list[int], stability_mode: Literal["logspace_min", "logspace_max"]
) -> tuple[list[int], int]:
    """Shifts logspace values by the scalar minimum/maximum over all of them, returning the positions of the shifted logspace values and of the shift."""

    norm_backend_function = backend_functions.max if stability_mode == "logspace_max" else backend_functions.min
    norm_positions = tuple(builder.wrap_and_append(norm_backend_function, (log_position,)) for log_position in log_positions)
    # TODO: maybe don't stack and aggregate but just aggregate here? we may need to add another backend function for this
    stacked_maxima_position = builder.append(partial(backend_functions.stack, axis=0), norm_positions)
    shift_position = builder.wrap_and_append(norm_backend_function, (stacked_maxima_position,))
    log_shifted_positions: list[int] = []
    for log_position in log_positions:
        log_shifted_positions.append(builder.wrap_and_append(backend_functions.subtract, (log_position, shift_position)))
    return log_shifted_positions, shift_position


def _reshape_einsum_shift(backend_functions: BackendFunctions[TBackendArray], retained_labels: str, output_string: str, shift: TBackendArray) -> TBackendArray:
    """Reshapes a shift over retained einsum labels so that it broadcasts over the complete output."""

    retained_dimensions = iter(shift.shape)
    output_shape = tuple(next(retained_dimensions) if output_label in retained_labels else 1 for output_label in output_string)
    return backend_functions.reshape(shift, output_shape)


def _append_logspace_einsum(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _LogspacePositions,
    format_string: str,
    argument_ssa_ids: tuple[int, ...],
    result_ssa_id: int,
    stability_mode: Literal["logspace_min", "logspace_max"],
    parameter_derived_ssa_ids: frozenset[int],
) -> None:
    """Stable einsum that keeps parameter-derived weights linear and shifts other operands in log space."""

    norm_backend_function = backend_functions.max if stability_mode == "logspace_max" else backend_functions.min
    input_strings, output_string = parse_format_string(format_string)
    if len(input_strings) != len(argument_ssa_ids):
        raise ValueError(f"The einsum format string has {len(input_strings)} inputs, but the instruction has {len(argument_ssa_ids)} arguments.")

    raw_argument_positions: list[int] = []
    shift_positions: list[tuple[int, str, str]] = []
    for input_string, argument_ssa_id in zip(input_strings, argument_ssa_ids, strict=True):
        if argument_ssa_id in parameter_derived_ssa_ids:
            raw_argument_positions.append(_as_raw_position(builder, backend_functions, positions, argument_ssa_id))
            continue

        log_argument_position = _as_logspace_position(builder, backend_functions, positions, argument_ssa_id)
        retained_labels = "".join(output_label for output_label in output_string if output_label in input_string)
        reduction_axes = tuple(axis for axis, input_label in enumerate(input_string) if input_label not in output_string)
        if not reduction_axes:
            shift_position = log_argument_position
        elif retained_labels:
            shift_position = builder.wrap_and_append(partial(norm_backend_function, axis=reduction_axes, keepdims=True), (log_argument_position,))
        else:
            shift_position = builder.wrap_and_append(norm_backend_function, (log_argument_position,))
        log_shifted_position = builder.wrap_and_append(backend_functions.subtract, (log_argument_position, shift_position))
        raw_argument_positions.append(builder.wrap_and_append(backend_functions.exp, (log_shifted_position,)))
        shift_positions.append((shift_position, input_string, retained_labels))

    raw_einsum_position = builder.wrap_and_append(partial(backend_functions.einsum, format_string), tuple(raw_argument_positions))
    log_einsum_position = builder.wrap_and_append(backend_functions.log, (raw_einsum_position,))
    for shift_position, input_string, retained_labels in shift_positions:
        if retained_labels:
            shift_position = builder.wrap_and_append(partial(backend_functions.einsum, f"{input_string}->{retained_labels}"), (shift_position,))
            shift_position = builder.wrap_and_append(partial(_reshape_einsum_shift, backend_functions, retained_labels, output_string), (shift_position,))
        log_einsum_position = builder.wrap_and_append(backend_functions.add, (log_einsum_position, shift_position))
    positions[result_ssa_id, "logspace"] = log_einsum_position


def _append_logspace_instruction(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _LogspacePositions,
    instruction: RichInstruction,
    result_ssa_id: int,
    stability_mode: Literal["logspace_min", "logspace_max"],
    parameter_derived_ssa_ids: frozenset[int],
) -> None:
    argument_ssa_ids = instruction.argument_ssa_ids
    match instruction.operator:
        case OperatorExp():
            # log(exp(a)) = a: the raw argument already is the logarithm of the result, so no computation is needed
            positions[result_ssa_id, "logspace"] = _as_raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
        case OperatorLog():
            # log(a) is the logspace part of a, so the result is available as a raw tensor without computation
            positions[result_ssa_id, "raw"] = _as_logspace_position(builder, backend_functions, positions, argument_ssa_ids[0])
        case OperatorSoftmax(_):
            # softmax is shifted by its maximum internally and takes arguments of arbitrary sign, so raw in and raw out is safe
            raw_argument_position = _as_raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
            positions[result_ssa_id, "raw"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (raw_argument_position,))
        case OperatorAdd() | OperatorSubtract():
            # TODO: maybe use minimum instead of maximum here to completely prevent underflow?
            # log(e^a ± e^b) = log(e^(a - m) ± e^(b - m)) + m with the scalar m = max(a, b), so only values far below the maximum underflow
            # (a subtraction whose result is negative cannot be represented in logspace and produces NaN)
            aggregate = backend_functions.add if isinstance(instruction.operator, OperatorAdd) else backend_functions.subtract
            log_argument_positions = [_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids]
            # add -m
            log_shifted_positions, shift_position = _append_shifted(builder, backend_functions, log_argument_positions, stability_mode)
            # exponentiate
            raw_shifted_positions = [builder.wrap_and_append(backend_functions.exp, (log_shifted_position,)) for log_shifted_position in log_shifted_positions]
            # ±
            raw_aggregated_position = builder.wrap_and_append(aggregate, tuple(raw_shifted_positions))
            # log
            log_aggregated_position = builder.wrap_and_append(backend_functions.log, (raw_aggregated_position,))
            # add m
            positions[result_ssa_id, "logspace"] = builder.wrap_and_append(backend_functions.add, (log_aggregated_position, shift_position))
        case OperatorMultiply():
            # log(a * b) = log(a) + log(b)
            log_argument_1, log_argument_2 = (_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids)
            positions[result_ssa_id, "logspace"] = builder.wrap_and_append(backend_functions.add, (log_argument_1, log_argument_2))
        case OperatorDivide():
            # log(a / b) = log(a) - log(b)
            log_argument_1, log_argument_2 = (_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids)
            positions[result_ssa_id, "logspace"] = builder.wrap_and_append(backend_functions.subtract, (log_argument_1, log_argument_2))
        case OperatorStack(_) | OperatorSelect(_, _) | OperatorSlice(_, _, _):
            # stacking and indexing commute with the elementwise logarithm
            log_argument_positions = [_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids]
            positions[result_ssa_id, "logspace"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), tuple(log_argument_positions))
        case OperatorTake(_):
            # indexing commutes with the elementwise logarithm; the indices are integer positions and always used raw
            log_argument_position = _as_logspace_position(builder, backend_functions, positions, argument_ssa_ids[0])
            indices_position = _as_raw_position(builder, backend_functions, positions, argument_ssa_ids[1])
            positions[result_ssa_id, "logspace"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (log_argument_position, indices_position))
        case OperatorEinsum(format_string):
            _append_logspace_einsum(builder, backend_functions, positions, format_string, argument_ssa_ids, result_ssa_id, stability_mode, parameter_derived_ssa_ids)
        case _:
            raise NotImplementedError(f"The operator {instruction.operator.name} has no logspace backend translation.")
