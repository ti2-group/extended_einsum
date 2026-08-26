from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Generic, Literal, TypeVar

from extended_einsum.backend_translation.backend import BackendArray, BackendFunctions, BackendProgram
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorConcat,
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
from extended_einsum.language.types import Shape
from extended_einsum.shapes import infer_binary_shape
from extended_einsum.utils import is_contraction_free_einsum, normalize_axis, parse_format_string

TBackendArray = TypeVar("TBackendArray", bound=BackendArray)

# Paper implementation map ("Automatic Numerical Stability",
# sec:numerical-stability):
# - _append_logspace_einsum implements "Log-space einsum".
# - every stop_gradient applied to a shift/normalizer implements "Detached
#   reference shifts".
# - _scaled_positions and _append_scaled_einsum implement "Scaled evaluation".
# The remaining instruction translators propagate those representations through
# the extended language described in "Beyond Standard Einsum".


def translate_to_backend_program(
    rich_program: RichProgram,
    backend_functions: BackendFunctions[TBackendArray],
    *,
    scale_interval: int = 3,
) -> BackendProgram[TBackendArray]:
    """Insert the paper's stability transformation while lowering the IR.

    This is the final backend-independent compiler stage in Figure
    ``fig:pipeline`` and sec:compiler-optimizations. The selected stability mode
    chooses direct evaluation, conventional log-space evaluation, or the
    paper's scaled representation before backend compilation. Scaled modes
    always normalize raw inputs before their first contraction, then normalize
    contraction outputs every ``scale_interval`` dependency depths.
    """

    if scale_interval <= 0:
        raise ValueError("The scale interval must be positive.")
    match rich_program.stability_mode:
        case "unstable":
            return _translate_unstable(rich_program, backend_functions)
        case "scaled_min" | "scaled_max" | "scaled_sum":
            return _translate_scaled(
                rich_program,
                backend_functions,
                rich_program.stability_mode,
                scale_interval=scale_interval,
            )
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
        case OperatorConcat(axis):
            # concat takes the tensors as a sequence instead of separate arguments
            return partial(backend_functions.concat, axis=axis)
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


# Paper "Scaled evaluation": represent every positive value as
# X = X_hat * exp(s). "raw" is X, while "normalized" and "log_scale" are
# X_hat and s. The scale broadcasts to X; folded tensors keep one scale per
# last-axis fiber.
_ScaledPart = Literal["raw", "normalized", "log_scale"]
_ScaledPositions = dict[tuple[int, _ScaledPart], int]
_ScaledShapes = dict[int, Shape]


def _translate_scaled(
    rich_program: RichProgram,
    backend_functions: BackendFunctions[TBackendArray],
    stability_mode: Literal["scaled_min", "scaled_max", "scaled_sum"],
    *,
    scale_interval: int,
) -> BackendProgram[TBackendArray]:
    """Translates a rich program into a backend program whose intermediate values are scaled pairs of a normalized tensor and a broadcastable log scale.

    Paper reference: "Scaled evaluation" in sec:numerical-stability.
    Scaling is lazy: a value stays a raw tensor until an operator consumes it as a scaled pair, then each last-axis fiber is normalized by its sum
    (scaled_sum), maximum (scaled_max), or minimum (scaled_min), which assumes strictly positive values. The log and softmax operators eliminate the scale again,
    so their results are stored as raw tensors. Parameter-derived values stay linear in einsums.
    """

    builder = _ProgramBuilder(rich_program.n_inputs)
    positions: _ScaledPositions = {(ssa_id, "raw"): ssa_id for ssa_id in range(rich_program.n_inputs)}
    scale_shapes: _ScaledShapes = {}
    parameter_derived_ssa_ids = _parameter_derived_ssa_ids(rich_program)
    contraction_ages = {ssa_id: 0 for ssa_id in range(rich_program.n_inputs)}
    for instruction_index, instruction in enumerate(rich_program.instructions):
        result_ssa_id = rich_program.n_inputs + instruction_index
        source_contraction_age = max(
            (
                contraction_ages[argument_ssa_id]
                for argument_ssa_id in instruction.argument_ssa_ids
                if argument_ssa_id not in parameter_derived_ssa_ids
            ),
            default=0,
        )
        is_data_contraction = (
            isinstance(instruction.operator, OperatorEinsum)
            and result_ssa_id not in parameter_derived_ssa_ids
            and not is_contraction_free_einsum(instruction.operator.format_string)
        )
        result_contraction_age = source_contraction_age + int(is_data_contraction)
        normalize_einsum_output = (
            is_data_contraction
            and result_contraction_age >= scale_interval
        )
        _append_scaled_instruction(
            builder,
            backend_functions,
            positions,
            scale_shapes,
            rich_program.shapes,
            instruction,
            result_ssa_id,
            stability_mode,
            parameter_derived_ssa_ids,
            normalize_einsum_output=normalize_einsum_output,
        )
        if normalize_einsum_output or isinstance(
            instruction.operator,
            (OperatorExp, OperatorSoftmax),
        ):
            result_contraction_age = 0
        contraction_ages[result_ssa_id] = result_contraction_age
    output_position = _raw_position(builder, backend_functions, positions, rich_program.output_ssa)
    if output_position != builder.last_position:
        raise ValueError("The raw value of the program's output must be the last computed tensor, because the runtime returns the last tensor.")
    return builder.build()


def _raw_position(builder: _ProgramBuilder[TBackendArray], backend_functions: BackendFunctions[TBackendArray], positions: _ScaledPositions, ssa_id: int) -> int:
    """Recover ``X = X_hat * exp(s)`` from paper "Scaled evaluation" once."""

    if (ssa_id, "raw") not in positions:
        scale_factor_position = builder.wrap_and_append(backend_functions.exp, (positions[ssa_id, "log_scale"],))
        positions[ssa_id, "raw"] = builder.wrap_and_append(backend_functions.multiply, (positions[ssa_id, "normalized"], scale_factor_position))
    return positions[ssa_id, "raw"]


def _scaled_positions(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _ScaledPositions,
    scale_shapes: _ScaledShapes,
    value_shapes: list[Shape],
    ssa_id: int,
    stability_mode: Literal["scaled_min", "scaled_max", "scaled_sum"],
) -> tuple[int, int]:
    """Construct the scaled pair from paper "Scaled evaluation".

    For maximum normalization this emits exactly
    ``n = sg(max(X))``, ``X_hat = X / n``, and ``s = log(n)`` for each
    last-axis fiber. The min/sum modes are implementation variants using the
    same representation identity.
    """

    if (ssa_id, "normalized") not in positions:
        raw_position = positions[ssa_id, "raw"]
        match stability_mode:
            case "scaled_min":
                total_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.min), (raw_position,))
            case "scaled_max":
                total_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.max), (raw_position,))
            case "scaled_sum":
                total_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.sum), (raw_position,))
        # Paper "Detached reference shifts": n only chooses a representation,
        # because (X / n) * exp(log(n)) = X. Detaching n is therefore exact and
        # removes the reduction's backward graph.
        total_position = builder.wrap_and_append(backend_functions.stop_gradient, (total_position,))
        positions[ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (raw_position, total_position))
        positions[ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.log, (total_position,))
        scale_shapes[ssa_id] = _fiber_scale_shape(value_shapes[ssa_id])
    return positions[ssa_id, "normalized"], positions[ssa_id, "log_scale"]


def _fiber_scale_shape(value_shape: Shape) -> Shape:
    if not value_shape:
        return ()
    return (*value_shape[:-1], 1)


def _fiber_norm(backend_functions: BackendFunctions[TBackendArray], norm_backend_function: Callable[..., TBackendArray], array: TBackendArray) -> TBackendArray:
    """Reduces the final value axis while retaining every leading batch or folded-operation axis."""

    if not array.shape:
        return norm_backend_function(array)
    return norm_backend_function(array, axis=-1, keepdims=True)


def _slice_broadcast_scale(backend_call: Callable[[Sequence[TBackendArray]], TBackendArray], axis: int, scale: TBackendArray) -> TBackendArray:
    """Slices a broadcastable scale only when it varies along the sliced axis."""

    if not scale.shape:
        return scale
    normalized_axis = axis if axis >= 0 else axis + len(scale.shape)
    if scale.shape[normalized_axis] == 1:
        return scale
    return backend_call((scale,))


def _select_broadcast_scale(backend_functions: BackendFunctions[TBackendArray], axis: int, index: int, scale: TBackendArray) -> TBackendArray:
    """Selects from a scale, using index zero when the selected axis is broadcast."""

    if not scale.shape:
        return scale
    normalized_axis = axis if axis >= 0 else axis + len(scale.shape)
    scale_index = 0 if scale.shape[normalized_axis] == 1 else index
    return backend_functions.select(scale, axis, scale_index)


def _take_broadcast_scale(backend_functions: BackendFunctions[TBackendArray], axis: int, scale: TBackendArray, indices: TBackendArray) -> TBackendArray:
    """Takes from a broadcastable scale only when it varies along the indexed axis."""

    if not scale.shape:
        return scale
    normalized_axis = axis if axis >= 0 else axis + len(scale.shape)
    if scale.shape[normalized_axis] == 1:
        return scale
    return backend_functions.take(scale, indices, axis)


def _append_rescaled_values(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    scaled_pairs: list[tuple[int, int]],
    scaled_shapes: list[Shape],
    *,
    reduce_scales_to_scalar: bool = False,
) -> tuple[list[int], int, Shape]:
    """Align scaled values using the reference-shift rule from the paper.

    Paper reference: the contracted-scale alignment at the end of "Scaled
    evaluation". Each value absorbs ``exp(s - s_bar)`` while ``s_bar`` is
    retained as the common output scale.
    """

    if not scaled_pairs:
        raise ValueError("At least one scaled pair is required")
    if len(scaled_pairs) != len(scaled_shapes):
        raise ValueError("Every scaled pair must have a scale shape")
    scale_positions = [log_scale_position for _, log_scale_position in scaled_pairs]
    if reduce_scales_to_scalar:
        scale_positions = [builder.wrap_and_append(backend_functions.max, (scale_position,)) for scale_position in scale_positions]
    common_scale_position = scale_positions[0]
    common_scale_shape = () if reduce_scales_to_scalar else scaled_shapes[0]
    for log_scale_position, scale_shape in zip(scale_positions[1:], scaled_shapes[1:], strict=True):
        common_scale_position = builder.wrap_and_append(backend_functions.maximum, (common_scale_position, log_scale_position))
        if not reduce_scales_to_scalar:
            common_scale_shape = infer_binary_shape(common_scale_shape, scale_shape)
    # Paper "Detached reference shifts": s_bar is subtracted before
    # exponentiation and restored afterward, so its derivative cancels exactly.
    common_scale_position = builder.wrap_and_append(backend_functions.stop_gradient, (common_scale_position,))
    rescaled_positions: list[int] = []
    for normalized_position, log_scale_position in scaled_pairs:
        shift_position = builder.wrap_and_append(backend_functions.subtract, (log_scale_position, common_scale_position))
        factor_position = builder.wrap_and_append(backend_functions.exp, (shift_position,))
        rescaled_positions.append(builder.wrap_and_append(backend_functions.multiply, (normalized_position, factor_position)))
    return rescaled_positions, common_scale_position, common_scale_shape


def _reshape_stack_scale(backend_functions: BackendFunctions[TBackendArray], axis: int, scale: TBackendArray) -> TBackendArray:
    """Inserts the stack axis into a non-scalar broadcast scale."""

    if not scale.shape:
        return scale
    normalized_axis = axis if axis >= 0 else axis + len(scale.shape) + 1
    return backend_functions.reshape(scale, (*scale.shape[:normalized_axis], 1, *scale.shape[normalized_axis:]))


def _append_scaled_einsum(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _ScaledPositions,
    scale_shapes: _ScaledShapes,
    value_shapes: list[Shape],
    format_string: str,
    argument_ssa_ids: tuple[int, ...],
    result_ssa_id: int,
    stability_mode: Literal["scaled_min", "scaled_max", "scaled_sum"],
    parameter_derived_ssa_ids: frozenset[int],
    short_contraction_labels: tuple[str, ...],
    *,
    normalize_output: bool,
) -> None:
    """Implement an einsum on the paper's ``(X_hat, s)`` representation.

    Paper reference: "Scaled evaluation". Linear parameter-derived operands
    remain unchanged; data operands keep normalized values and broadcastable
    log scales. Contracted scale axes are aligned before the dense contraction,
    and the result is normalized for direct consumption by the next layer.
    """

    input_strings, output_string = parse_format_string(format_string)
    if len(input_strings) != len(argument_ssa_ids):
        raise ValueError(f"The einsum format string has {len(input_strings)} inputs, but the instruction has {len(argument_ssa_ids)} arguments.")
    if result_ssa_id in parameter_derived_ssa_ids:
        raw_argument_positions = tuple(_raw_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids)
        positions[result_ssa_id, "raw"] = builder.wrap_and_append(partial(backend_functions.einsum, format_string), raw_argument_positions)
        return

    einsum_argument_positions: list[int] = []
    shift_positions: list[tuple[int, Shape, bool, str, str]] = []
    for input_string, argument_ssa_id in zip(input_strings, argument_ssa_ids, strict=True):
        if argument_ssa_id in parameter_derived_ssa_ids:
            einsum_argument_positions.append(_raw_position(builder, backend_functions, positions, argument_ssa_id))
            continue

        normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode)
        scale_shape = scale_shapes[argument_ssa_id]
        value_shape_is_known = len(value_shapes[argument_ssa_id]) == len(input_string)
        scale_shape_is_scalar = value_shape_is_known and not scale_shape
        scale_shape_is_known = value_shape_is_known and (not scale_shape or len(scale_shape) == len(input_string))
        retained_labels = "".join(output_label for output_label in output_string if output_label in input_string)
        reduction_axes = tuple(axis for axis, input_label in enumerate(input_string) if input_label not in output_string)
        if reduction_axes and not scale_shape_is_scalar and (not scale_shape_is_known or any(scale_shape[axis] != 1 for axis in reduction_axes)):
            # Paper's Tucker example: when s varies on a contracted label,
            # compute s_bar = sg(max(s)), absorb exp(s - s_bar) into X_hat,
            # and propagate only s_bar to the output.
            common_scale_position = builder.wrap_and_append(partial(backend_functions.max, axis=reduction_axes, keepdims=True), (log_scale_position,))
            # This reduction only selects a reference scale for a contraction;
            # its derivative cancels between rescaling and scale restoration.
            common_scale_position = builder.wrap_and_append(backend_functions.stop_gradient, (common_scale_position,))
            scale_delta_position = builder.wrap_and_append(backend_functions.subtract, (log_scale_position, common_scale_position))
            scale_factor_position = builder.wrap_and_append(backend_functions.exp, (scale_delta_position,))
            normalized_position = builder.wrap_and_append(backend_functions.multiply, (normalized_position, scale_factor_position))
            common_scale_shape = tuple(1 if axis in reduction_axes else dimension for axis, dimension in enumerate(scale_shape))
        else:
            common_scale_position = log_scale_position
            common_scale_shape = scale_shape
        einsum_argument_positions.append(normalized_position)
        shift_positions.append((common_scale_position, common_scale_shape, scale_shape_is_known, input_string, retained_labels))

    # Paper "Log-space einsum" notes that contraction-free Hadamard/Kronecker
    # products need no shift. In scaled form, multiply/broadcast X_hat and add
    # the log scales without renormalizing a materialized outer product.
    if is_contraction_free_einsum(format_string) and all(scale_shape_is_known for _, _, scale_shape_is_known, _, _ in shift_positions):
        positions[result_ssa_id, "normalized"] = builder.wrap_and_append(partial(backend_functions.einsum, format_string), tuple(einsum_argument_positions))
        result_scale_position: int | None = None
        result_scale_shape: Shape = ()
        for operand_scale_position, operand_scale_shape, _, input_string, _ in shift_positions:
            if operand_scale_shape and input_string != output_string:
                operand_scale_position = builder.wrap_and_append(partial(_broadcast_einsum_operand, backend_functions, input_string, output_string), (operand_scale_position,))
                operand_scale_shape = tuple(operand_scale_shape[input_string.index(output_label)] if output_label in input_string else 1 for output_label in output_string)
            if result_scale_position is None:
                result_scale_position = operand_scale_position
                result_scale_shape = operand_scale_shape
            else:
                result_scale_position = builder.wrap_and_append(backend_functions.add, (result_scale_position, operand_scale_position))
                result_scale_shape = infer_binary_shape(result_scale_shape, operand_scale_shape)
        if result_scale_position is None:
            raise ValueError("A contraction-free scaled einsum must have at least one non-parameter operand.")
        positions[result_ssa_id, "log_scale"] = result_scale_position
        scale_shapes[result_ssa_id] = result_scale_shape
        return

    # Paper "Scaled evaluation": contract normalized data with unchanged
    # linear weights. The compiler marks very short same-index contractions;
    # expose those as multiply/reduce in backend IR instead of tiny GEMVs.
    if not short_contraction_labels:
        raw_einsum_position = builder.wrap_and_append(partial(backend_functions.einsum, format_string), tuple(einsum_argument_positions))
    else:
        expanded_string = output_string + "".join(short_contraction_labels)
        aligned_positions = tuple(
            builder.wrap_and_append(
                partial(
                    _broadcast_einsum_operand,
                    backend_functions,
                    input_string,
                    expanded_string,
                ),
                (argument_position,),
            )
            for input_string, argument_position in zip(
                input_strings,
                einsum_argument_positions,
                strict=True,
            )
        )
        product_position = builder.wrap_and_append(
            backend_functions.multiply,
            aligned_positions,
        )
        raw_einsum_position = builder.wrap_and_append(
            partial(
                backend_functions.sum,
                axis=tuple(range(len(output_string), len(expanded_string))),
            ),
            (product_position,),
        )
    log_scale_position: int | None
    if normalize_output:
        match stability_mode:
            case "scaled_min":
                raw_norm_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.min), (raw_einsum_position,))
            case "scaled_max":
                raw_norm_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.max), (raw_einsum_position,))
            case "scaled_sum":
                raw_norm_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.sum), (raw_einsum_position,))
        # Paper "Detached reference shifts": output normalization changes only
        # the (X_hat, s) representation, so stop-gradient preserves the
        # derivative.
        raw_norm_position = builder.wrap_and_append(backend_functions.stop_gradient, (raw_norm_position,))
        positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (raw_einsum_position, raw_norm_position))
        log_scale_position = builder.wrap_and_append(backend_functions.log, (raw_norm_position,))
        result_scale_shape = _fiber_scale_shape(value_shapes[result_ssa_id])
    else:
        positions[result_ssa_id, "normalized"] = raw_einsum_position
        log_scale_position = None
        result_scale_shape = ()
    # Restore every operand's retained scale on the corresponding output labels,
    # which is the "+ s" step of the paper's scaled sum-layer construction.
    for operand_scale_position, operand_scale_shape, scale_shape_is_known, input_string, retained_labels in shift_positions:
        scale_already_broadcasts_to_output = (
            scale_shape_is_known
            and len(operand_scale_shape) == len(output_string)
            and all(dimension == 1 or input_label == output_label for dimension, input_label, output_label in zip(operand_scale_shape, input_string, output_string, strict=True))
        )
        if (not scale_shape_is_known or operand_scale_shape) and not scale_already_broadcasts_to_output:
            operand_scale_position = builder.wrap_and_append(partial(backend_functions.einsum, f"{input_string}->{retained_labels}"), (operand_scale_position,))
        if retained_labels and (not scale_shape_is_known or operand_scale_shape) and not scale_already_broadcasts_to_output:
            operand_scale_position = builder.wrap_and_append(partial(_reshape_einsum_shift, backend_functions, retained_labels, output_string), (operand_scale_position,))
        if operand_scale_shape and not scale_already_broadcasts_to_output:
            operand_scale_shape = tuple(operand_scale_shape[input_string.index(output_label)] if output_label in input_string else 1 for output_label in output_string) if retained_labels else ()
        if log_scale_position is None:
            log_scale_position = operand_scale_position
            result_scale_shape = operand_scale_shape
        else:
            log_scale_position = builder.wrap_and_append(backend_functions.add, (log_scale_position, operand_scale_position))
            result_scale_shape = infer_binary_shape(result_scale_shape, operand_scale_shape)
    if log_scale_position is None:
        raise ValueError("A scaled data einsum must have at least one non-parameter operand.")
    positions[result_ssa_id, "log_scale"] = log_scale_position
    scale_shapes[result_ssa_id] = result_scale_shape


def _append_scaled_instruction(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _ScaledPositions,
    scale_shapes: _ScaledShapes,
    value_shapes: list[Shape],
    instruction: RichInstruction,
    result_ssa_id: int,
    stability_mode: Literal["scaled_min", "scaled_max", "scaled_sum"],
    parameter_derived_ssa_ids: frozenset[int],
    *,
    normalize_einsum_output: bool,
) -> None:
    """Propagate the pair from paper "Scaled evaluation" through the IR.

    Multiplication/division add/subtract scales; layout operations transform
    values and their broadcastable scales together; log reconstructs
    ``log(X_hat) + s``; and softmax stays raw because the backend primitive
    already performs its stable shift.
    """

    argument_ssa_ids = instruction.argument_ssa_ids
    match instruction.operator:
        case OperatorExp():
            # Paper "Scaled evaluation" applied to explicit exp:
            # exp(a) = exp(a - m) * exp(m), with m = sg(max(a)) per fiber.
            # The raw argument moves into the log scale and normalized values
            # remain in (0, 1].
            # (entries whose exponent is far below the maximum underflow to zero, the same tradeoff as in softmax)
            raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
            log_scale_position = builder.wrap_and_append(partial(_fiber_norm, backend_functions, backend_functions.max), (raw_position,))
            # exp(x - m) * exp(m) is independent of how m is selected.
            log_scale_position = builder.wrap_and_append(backend_functions.stop_gradient, (log_scale_position,))
            shifted_position = builder.wrap_and_append(backend_functions.subtract, (raw_position, log_scale_position))
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.exp, (shifted_position,))
            positions[result_ssa_id, "log_scale"] = log_scale_position
            scale_shapes[result_ssa_id] = _fiber_scale_shape(value_shapes[result_ssa_id])
        case OperatorLog():
            # Paper "Scaled evaluation" root reconstruction:
            # log(X_hat * exp(s)) = log(X_hat) + s.
            normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_ids[0], stability_mode)
            log_position = builder.wrap_and_append(backend_functions.log, (normalized_position,))
            positions[result_ssa_id, "raw"] = builder.wrap_and_append(backend_functions.add, (log_position, log_scale_position))
        case OperatorSoftmax(_):
            # softmax is shifted by its maximum internally and its result sums to one along the axis, so raw in and raw out is safe
            raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
            positions[result_ssa_id, "raw"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (raw_position,))
        case OperatorAdd() | OperatorSubtract():
            # Paper's scale-alignment rule:
            # X1_hat*e^s1 ± X2_hat*e^s2 =
            # (X1_hat*e^(s1-m) ± X2_hat*e^(s2-m))*e^m.
            combine = backend_functions.add if isinstance(instruction.operator, OperatorAdd) else backend_functions.subtract
            scaled_pairs = [_scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
            rescaled_positions, common_scale_position, common_scale_shape = _append_rescaled_values(
                builder, backend_functions, scaled_pairs, [scale_shapes[argument_ssa_id] for argument_ssa_id in argument_ssa_ids]
            )
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(combine, tuple(rescaled_positions))
            positions[result_ssa_id, "log_scale"] = common_scale_position
            scale_shapes[result_ssa_id] = common_scale_shape
        case OperatorMultiply():
            # the normalized values multiply and the log scales add up
            (normalized_1, log_scale_1), (normalized_2, log_scale_2) = (
                _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids
            )
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.multiply, (normalized_1, normalized_2))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.add, (log_scale_1, log_scale_2))
            scale_shapes[result_ssa_id] = infer_binary_shape(scale_shapes[argument_ssa_ids[0]], scale_shapes[argument_ssa_ids[1]])
        case OperatorDivide():
            # the normalized values divide and the divisor's log scale is subtracted from the dividend's
            (normalized_1, log_scale_1), (normalized_2, log_scale_2) = (
                _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids
            )
            positions[result_ssa_id, "normalized"] = builder.wrap_and_append(backend_functions.divide, (normalized_1, normalized_2))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(backend_functions.subtract, (log_scale_1, log_scale_2))
            scale_shapes[result_ssa_id] = infer_binary_shape(scale_shapes[argument_ssa_ids[0]], scale_shapes[argument_ssa_ids[1]])
        case OperatorStack(axis):
            # bring all operands to a common log scale, then stack the rescaled values
            scaled_pairs = [_scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
            rescaled_positions, common_scale_position, common_scale_shape = _append_rescaled_values(
                builder, backend_functions, scaled_pairs, [scale_shapes[argument_ssa_id] for argument_ssa_id in argument_ssa_ids]
            )
            positions[result_ssa_id, "normalized"] = builder.append(partial(backend_functions.stack, axis=axis), tuple(rescaled_positions))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(partial(_reshape_stack_scale, backend_functions, axis), (common_scale_position,))
            normalized_axis = normalize_axis(axis, len(value_shapes[result_ssa_id])) if value_shapes[result_ssa_id] else 0
            scale_shapes[result_ssa_id] = common_scale_shape if not common_scale_shape else (*common_scale_shape[:normalized_axis], 1, *common_scale_shape[normalized_axis:])
        case OperatorConcat(axis):
            scaled_pairs = [_scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_id, stability_mode) for argument_ssa_id in argument_ssa_ids]
            normalized_axis = normalize_axis(axis, len(value_shapes[result_ssa_id]))
            if normalized_axis == len(value_shapes[result_ssa_id]) - 1:
                # Concatenating within a fiber requires one common scale for
                # all of its segments.
                rescaled_positions, common_scale_position, common_scale_shape = _append_rescaled_values(
                    builder,
                    backend_functions,
                    scaled_pairs,
                    [scale_shapes[argument_ssa_id] for argument_ssa_id in argument_ssa_ids],
                    reduce_scales_to_scalar=True,
                )
                positions[result_ssa_id, "normalized"] = builder.append(partial(backend_functions.concat, axis=axis), tuple(rescaled_positions))
                positions[result_ssa_id, "log_scale"] = common_scale_position
                scale_shapes[result_ssa_id] = common_scale_shape
            else:
                # Concatenation across fibers can preserve every operand's
                # scale exactly. Reducing these to one scalar can underflow
                # complete fibers before their next normalization.
                normalized_positions = tuple(normalized_position for normalized_position, _ in scaled_pairs)
                normalized_positions = list(normalized_positions)
                expanded_scale_positions: list[int] = []
                for pair_index, (
                    argument_ssa_id,
                    (_, log_scale_position),
                ) in enumerate(
                    zip(argument_ssa_ids, scaled_pairs, strict=True)
                ):
                    target_scale_shape = _fiber_scale_shape(value_shapes[argument_ssa_id])
                    scale_shape = scale_shapes[argument_ssa_id]
                    if scale_shape and scale_shape[-1] != 1:
                        common_scale_position = builder.wrap_and_append(
                            partial(
                                backend_functions.max,
                                axis=-1,
                                keepdims=True,
                            ),
                            (log_scale_position,),
                        )
                        common_scale_position = builder.wrap_and_append(
                            backend_functions.stop_gradient,
                            (common_scale_position,),
                        )
                        scale_delta_position = builder.wrap_and_append(
                            backend_functions.subtract,
                            (
                                log_scale_position,
                                common_scale_position,
                            ),
                        )
                        scale_factor_position = builder.wrap_and_append(
                            backend_functions.exp,
                            (scale_delta_position,),
                        )
                        normalized_positions[pair_index] = (
                            builder.wrap_and_append(
                                backend_functions.multiply,
                                (
                                    normalized_positions[pair_index],
                                    scale_factor_position,
                                ),
                            )
                        )
                        log_scale_position = common_scale_position
                        scale_shape = (*scale_shape[:-1], 1)
                    if scale_shape != target_scale_shape:
                        log_scale_position = builder.wrap_and_append(partial(backend_functions.broadcast_to, shape=target_scale_shape), (log_scale_position,))
                    expanded_scale_positions.append(log_scale_position)
                positions[result_ssa_id, "normalized"] = builder.append(partial(backend_functions.concat, axis=axis), tuple(normalized_positions))
                positions[result_ssa_id, "log_scale"] = builder.append(partial(backend_functions.concat, axis=axis), tuple(expanded_scale_positions))
                scale_shapes[result_ssa_id] = _fiber_scale_shape(value_shapes[result_ssa_id])
        case OperatorTake(_):
            # The indices are integer positions and always used raw. Index a
            # broadcastable scale only when it varies along the selected axis.
            normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_ids[0], stability_mode)
            indices_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[1])
            positions[result_ssa_id, "normalized"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (normalized_position, indices_position))
            positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(partial(_take_broadcast_scale, backend_functions, instruction.operator.axis), (log_scale_position, indices_position))
            source_scale_shape = scale_shapes[argument_ssa_ids[0]]
            scale_axis = normalize_axis(instruction.operator.axis, len(source_scale_shape)) if source_scale_shape else 0
            if not source_scale_shape or source_scale_shape[scale_axis] == 1:
                scale_shapes[result_ssa_id] = source_scale_shape
            else:
                value_axis = normalize_axis(instruction.operator.axis, len(value_shapes[result_ssa_id]))
                scale_shapes[result_ssa_id] = (*source_scale_shape[:scale_axis], value_shapes[result_ssa_id][value_axis], *source_scale_shape[scale_axis + 1 :])
        case OperatorSelect(axis, index):
            # Keep raw indexed values raw so selecting a variable from a
            # variables-by-batch input happens before fiber normalization.
            if (argument_ssa_ids[0], "raw") in positions:
                raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
                positions[result_ssa_id, "raw"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (raw_position,))
            else:
                normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_ids[0], stability_mode)
                backend_call = _operator_to_backend_call(instruction.operator, backend_functions)
                positions[result_ssa_id, "normalized"] = builder.append(backend_call, (normalized_position,))
                positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(partial(_select_broadcast_scale, backend_functions, axis, index), (log_scale_position,))
                source_scale_shape = scale_shapes[argument_ssa_ids[0]]
                scale_axis = normalize_axis(axis, len(source_scale_shape)) if source_scale_shape else 0
                scale_shapes[result_ssa_id] = source_scale_shape if not source_scale_shape else (*source_scale_shape[:scale_axis], *source_scale_shape[scale_axis + 1 :])
        case OperatorSlice(_, _, axis):
            if (argument_ssa_ids[0], "raw") in positions:
                raw_position = _raw_position(builder, backend_functions, positions, argument_ssa_ids[0])
                positions[result_ssa_id, "raw"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (raw_position,))
            else:
                normalized_position, log_scale_position = _scaled_positions(builder, backend_functions, positions, scale_shapes, value_shapes, argument_ssa_ids[0], stability_mode)
                backend_call = _operator_to_backend_call(instruction.operator, backend_functions)
                positions[result_ssa_id, "normalized"] = builder.append(backend_call, (normalized_position,))
                positions[result_ssa_id, "log_scale"] = builder.wrap_and_append(partial(_slice_broadcast_scale, backend_call, axis), (log_scale_position,))
                source_scale_shape = scale_shapes[argument_ssa_ids[0]]
                scale_axis = normalize_axis(axis, len(source_scale_shape)) if source_scale_shape else 0
                if not source_scale_shape or source_scale_shape[scale_axis] == 1:
                    scale_shapes[result_ssa_id] = source_scale_shape
                else:
                    value_axis = normalize_axis(axis, len(value_shapes[result_ssa_id]))
                    scale_shapes[result_ssa_id] = (*source_scale_shape[:scale_axis], value_shapes[result_ssa_id][value_axis], *source_scale_shape[scale_axis + 1 :])
        case OperatorEinsum(format_string, short_contraction_labels):
            _append_scaled_einsum(
                builder,
                backend_functions,
                positions,
                scale_shapes,
                value_shapes,
                format_string,
                argument_ssa_ids,
                result_ssa_id,
                stability_mode,
                parameter_derived_ssa_ids,
                short_contraction_labels,
                normalize_output=normalize_einsum_output,
            )
        case _:
            raise NotImplementedError(f"The operator {instruction.operator.name} has no scaled backend translation.")


# Paper "Log-space einsum": an SSA value is available as the positive value
# itself ("raw") or as its natural logarithm ("logspace"), with
# raw = exp(logspace).
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

    Paper reference: "Log-space einsum" in sec:numerical-stability.
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
            rich_program.shapes,
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
    """Apply the paper's detached reference shift to log-space values.

    This is ``m = sg(max(ell))`` from "Detached reference shifts"; the min mode
    is an implementation variant with the same cancellation identity.
    """

    norm_backend_function = backend_functions.max if stability_mode == "logspace_max" else backend_functions.min
    norm_positions = tuple(builder.wrap_and_append(norm_backend_function, (log_position,)) for log_position in log_positions)
    # TODO: maybe don't stack and aggregate but just aggregate here? we may need to add another backend function for this
    stacked_maxima_position = builder.append(partial(backend_functions.stack, axis=0), norm_positions)
    shift_position = builder.wrap_and_append(norm_backend_function, (stacked_maxima_position,))
    # The common shift is only a numerical reference. Its derivative cancels
    # between exp(log_value - shift) and the final + shift, so detaching it
    # preserves the represented derivative and avoids reduction backward work.
    shift_position = builder.wrap_and_append(backend_functions.stop_gradient, (shift_position,))
    log_shifted_positions: list[int] = []
    for log_position in log_positions:
        log_shifted_positions.append(builder.wrap_and_append(backend_functions.subtract, (log_position, shift_position)))
    return log_shifted_positions, shift_position


def _reshape_einsum_shift(backend_functions: BackendFunctions[TBackendArray], retained_labels: str, output_string: str, shift: TBackendArray) -> TBackendArray:
    """Reshapes a shift over retained einsum labels so that it broadcasts over the complete output."""

    retained_dimensions = iter(shift.shape)
    output_shape = tuple(next(retained_dimensions) if output_label in retained_labels else 1 for output_label in output_string)
    return backend_functions.reshape(shift, output_shape)


def _broadcast_einsum_operand(backend_functions: BackendFunctions[TBackendArray], input_string: str, output_string: str, operand: TBackendArray) -> TBackendArray:
    """Reorders and reshapes a contraction-free einsum operand for broadcasting over the output."""

    retained_labels = "".join(output_label for output_label in output_string if output_label in input_string)
    if input_string != retained_labels:
        operand = backend_functions.einsum(f"{input_string}->{retained_labels}", operand)
    retained_dimensions = iter(operand.shape)
    output_shape = tuple(next(retained_dimensions) if output_label in retained_labels else 1 for output_label in output_string)
    return backend_functions.reshape(operand, output_shape)


def _append_logspace_einsum(
    builder: _ProgramBuilder[TBackendArray],
    backend_functions: BackendFunctions[TBackendArray],
    positions: _LogspacePositions,
    format_string: str,
    argument_ssa_ids: tuple[int, ...],
    result_ssa_id: int,
    stability_mode: Literal["logspace_min", "logspace_max"],
    parameter_derived_ssa_ids: frozenset[int],
    short_contraction_labels: tuple[str, ...],
) -> None:
    """Implement the paper's shifted log-space einsum formula.

    For each data operand this computes its retained-index shift ``m``, forms
    ``exp(ell - m)``, executes an ordinary linear einsum with parameter weights,
    and returns ``log(result) + m``. See "Log-space einsum" and "Detached
    reference shifts" in sec:numerical-stability.
    """

    norm_backend_function = backend_functions.max if stability_mode == "logspace_max" else backend_functions.min
    input_strings, output_string = parse_format_string(format_string)
    if len(input_strings) != len(argument_ssa_ids):
        raise ValueError(f"The einsum format string has {len(input_strings)} inputs, but the instruction has {len(argument_ssa_ids)} arguments.")

    # Paper "Log-space einsum": a contraction-free Hadamard or Kronecker product
    # is broadcasted addition in log space, so omit the exp/einsum/log sequence.
    if is_contraction_free_einsum(format_string) and all(argument_ssa_id not in parameter_derived_ssa_ids for argument_ssa_id in argument_ssa_ids):
        log_argument_positions = [_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids]
        broadcast_positions = [
            log_argument_position
            if input_string == output_string
            else builder.wrap_and_append(partial(_broadcast_einsum_operand, backend_functions, input_string, output_string), (log_argument_position,))
            for input_string, log_argument_position in zip(input_strings, log_argument_positions, strict=True)
        ]
        result_position = broadcast_positions[0]
        product_function = getattr(
            backend_functions,
            "logspace_product",
            backend_functions.add,
        )
        for log_argument_position in broadcast_positions[1:]:
            result_position = builder.wrap_and_append(
                product_function,
                (result_position, log_argument_position),
            )
        positions[result_ssa_id, "logspace"] = result_position
        return

    raw_argument_positions: list[int] = []
    shift_positions: list[tuple[int, str, str]] = []
    for input_string, argument_ssa_id in zip(input_strings, argument_ssa_ids, strict=True):
        if argument_ssa_id in parameter_derived_ssa_ids:
            raw_argument_positions.append(_as_raw_position(builder, backend_functions, positions, argument_ssa_id))
            continue

        log_argument_position = _as_logspace_position(builder, backend_functions, positions, argument_ssa_id)
        retained_labels = "".join(output_label for output_label in output_string if output_label in input_string)
        reduction_axes = tuple(axis for axis, input_label in enumerate(input_string) if input_label not in output_string)
        # Paper formula m = reduce_max(ell) over exactly the contracted labels;
        # retained labels remain so m broadcasts over the corresponding output.
        if not reduction_axes:
            shift_position = log_argument_position
        elif retained_labels:
            shift_position = builder.wrap_and_append(partial(norm_backend_function, axis=reduction_axes, keepdims=True), (log_argument_position,))
        else:
            shift_position = builder.wrap_and_append(norm_backend_function, (log_argument_position,))
        # Paper "Detached reference shifts": m is subtracted before contraction
        # and restored afterward, hence its derivative path contributes zero.
        shift_position = builder.wrap_and_append(backend_functions.stop_gradient, (shift_position,))
        log_shifted_position = builder.wrap_and_append(backend_functions.subtract, (log_argument_position, shift_position))
        raw_argument_positions.append(builder.wrap_and_append(backend_functions.exp, (log_shifted_position,)))
        shift_positions.append((shift_position, input_string, retained_labels))

    # Central operation from paper "Log-space einsum": contract the shifted
    # positive operands with unchanged linear weights.
    if not short_contraction_labels:
        raw_einsum_position = builder.wrap_and_append(partial(backend_functions.einsum, format_string), tuple(raw_argument_positions))
    else:
        # This is an elementwise weighted reduction, not a dense contraction:
        # one operand's labels are a subset of the other's and the contracted
        # fiber is very short. Make that structure explicit in backend IR so
        # tensor compilers can fuse the multiply and reduction instead of
        # dispatching thousands of tiny strided GEMVs.
        expanded_string = output_string + "".join(short_contraction_labels)
        aligned_positions = tuple(
            builder.wrap_and_append(
                partial(
                    _broadcast_einsum_operand,
                    backend_functions,
                    input_string,
                    expanded_string,
                ),
                (raw_argument_position,),
            )
            for input_string, raw_argument_position in zip(
                input_strings,
                raw_argument_positions,
                strict=True,
            )
        )
        product_position = builder.wrap_and_append(
            backend_functions.multiply,
            aligned_positions,
        )
        raw_einsum_position = builder.wrap_and_append(
            partial(
                backend_functions.sum,
                axis=tuple(
                    range(len(output_string), len(expanded_string))
                ),
            ),
            (product_position,),
        )
    log_einsum_position = builder.wrap_and_append(backend_functions.log, (raw_einsum_position,))
    # Restore every operand shift on the retained output labels: the final
    # ``+ m`` in the paper's log Y formula.
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
    value_shapes: list[Shape],
) -> None:
    """Propagate the representation from paper "Log-space einsum" through IR.

    Products/divisions become addition/subtraction, layout operations commute
    with elementwise log, and explicit exp/log only move between stored parts.
    """

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
        case OperatorStack(_) | OperatorConcat(_) | OperatorSelect(_, _) | OperatorSlice(_, _, _):
            # stacking, concatenation, and indexing commute with the elementwise logarithm
            log_argument_positions = [_as_logspace_position(builder, backend_functions, positions, argument_ssa_id) for argument_ssa_id in argument_ssa_ids]
            positions[result_ssa_id, "logspace"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), tuple(log_argument_positions))
        case OperatorTake(_):
            # indexing commutes with the elementwise logarithm; the indices are integer positions and always used raw
            log_argument_position = _as_logspace_position(builder, backend_functions, positions, argument_ssa_ids[0])
            indices_position = _as_raw_position(builder, backend_functions, positions, argument_ssa_ids[1])
            positions[result_ssa_id, "logspace"] = builder.append(_operator_to_backend_call(instruction.operator, backend_functions), (log_argument_position, indices_position))
        case OperatorEinsum(format_string, short_contraction_labels):
            _append_logspace_einsum(builder, backend_functions, positions, format_string, argument_ssa_ids, result_ssa_id, stability_mode, parameter_derived_ssa_ids, short_contraction_labels)
        case _:
            raise NotImplementedError(f"The operator {instruction.operator.name} has no logspace backend translation.")
