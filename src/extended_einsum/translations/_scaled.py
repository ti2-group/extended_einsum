from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from extended_einsum.backend import BackendTranslation
from extended_einsum.language import (
    SCALED_EINSUM_MAX_OPERATOR,
    SCALED_EINSUM_SUM_OPERATOR,
    ScaledEinsumOperator,
)
from extended_einsum.scale import ScaledTensor, normalize_axis
from extended_einsum.utils import parse_format_string


def stack_values(values: Sequence[Any], ops: BackendTranslation) -> Any:
    if not values:
        raise ValueError("cannot stack an empty input list")
    scaled_flags = [isinstance(value, ScaledTensor) for value in values]
    if any(scaled_flags) and not all(scaled_flags):
        raise ValueError("cannot stack a mix of scaled and unscaled tensors")
    if not scaled_flags[0]:
        return ops.stack(values)

    scaled_values = [value for value in values if isinstance(value, ScaledTensor)]
    axis = scaled_values[0].scale_axis
    if any(value.scale_axis != axis for value in scaled_values):
        raise ValueError("cannot stack scaled tensors with different scale axes")
    return ScaledTensor(
        ops.stack([value.value for value in scaled_values]),
        ops.stack([value.log_scale for value in scaled_values]),
        axis + 1,
    )


def take_value(source: Any, index: Any, axis: int, ops: BackendOps) -> Any:
    if not isinstance(source, ScaledTensor):
        return ops.take(source, index, axis)
    take_axis = normalize_axis(axis, len(source.shape))
    if source.scale_axis == take_axis:
        raise ValueError("cannot take away the scale axis of a scaled tensor")
    index_shape = tuple(int(dimension) for dimension in getattr(index, "shape", ()))
    scale_shape = tuple(
        int(dimension) for dimension in getattr(source.log_scale, "shape", ())
    )
    if not scale_shape:
        log_scale = source.log_scale
    elif scale_shape[take_axis] != 1:
        log_scale = ops.take(source.log_scale, index, take_axis)
    else:
        log_scale = ops.reshape(
            source.log_scale,
            (
                *scale_shape[:take_axis],
                *((1,) * len(index_shape)),
                *scale_shape[take_axis + 1 :],
            ),
        )
    scale_axis = (
        source.scale_axis
        if source.scale_axis < take_axis
        else source.scale_axis - 1 + len(index_shape)
    )
    return ScaledTensor(
        ops.take(source.value, index, take_axis),
        log_scale,
        scale_axis,
    )


def slice_value(source: Any, axis: int, start: int, stop: int, ops: BackendOps) -> Any:
    if not isinstance(source, ScaledTensor):
        return ops.slice(source, axis, start, stop)
    slice_axis = normalize_axis(axis, len(source.shape))
    scale_shape = tuple(
        int(dimension) for dimension in getattr(source.log_scale, "shape", ())
    )
    if not scale_shape or scale_shape[slice_axis] == 1:
        log_scale = source.log_scale
    else:
        log_scale = ops.slice(source.log_scale, slice_axis, start, stop)
    return ScaledTensor(
        ops.slice(source.value, slice_axis, start, stop),
        log_scale,
        source.scale_axis,
    )


def softmax_value(operand: Any, axis: int, ops: BackendOps) -> Any:
    if isinstance(operand, ScaledTensor):
        raise ValueError("softmax does not support scaled tensors")
    return ops.softmax(operand, axis)


def unary_value(
    operator: str,
    operand: Any,
    raw_unary: Callable[[Any], Any],
    ops: BackendOps,
) -> Any:
    if not isinstance(operand, ScaledTensor):
        return raw_unary(operand)
    if operator != "log":
        raise ValueError(f"unary operator {operator!r} does not support scaled tensors")
    return raw_unary(operand.value) + operand.log_scale


def binary_value(
    operator: str,
    lhs: Any,
    rhs: Any,
    raw_binary: Callable[[Any, Any], Any],
) -> Any:
    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor):
        return raw_binary(lhs, rhs)
    if operator != "*":
        raise ValueError(
            f"binary operator {operator!r} does not support scaled tensors"
        )
    if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
        if lhs.scale_axis != rhs.scale_axis:
            raise ValueError("scaled multiplication requires matching scale axes")
        return ScaledTensor(
            raw_binary(lhs.value, rhs.value),
            lhs.log_scale + rhs.log_scale,
            lhs.scale_axis,
        )
    if isinstance(lhs, ScaledTensor):
        return ScaledTensor(raw_binary(lhs.value, rhs), lhs.log_scale, lhs.scale_axis)
    if isinstance(rhs, ScaledTensor):
        return ScaledTensor(raw_binary(lhs, rhs.value), rhs.log_scale, rhs.scale_axis)
    raise AssertionError("unreachable")


def normal_einsum(format_string: str, operands: Sequence[Any], ops: BackendOps) -> Any:
    if any(isinstance(operand, ScaledTensor) for operand in operands):
        raise ValueError("normal einsum does not accept scaled tensors")
    return ops.einsum(format_string, operands)


def scaled_einsum(
    operator: ScaledEinsumOperator,
    format_string: str,
    operands: Sequence[Any],
    output_scale_axis: int,
    ops: BackendOps,
) -> ScaledTensor[Any]:
    if len(operands) != 2:
        raise ValueError("scaled einsum requires exactly two operands")
    input_labels, output_labels = _parse_scaled_einsum_format(format_string)
    output_axis = normalize_axis(output_scale_axis, len(output_labels))
    output_safe_labels = set(output_labels)
    output_safe_labels.discard(output_labels[output_axis])

    # print(f"scaled_einsum: operator={operator}, format_string={format_string}")
    # print(f"  input_labels={input_labels}, output_labels={output_labels}")
    # print(f"  output_axis={output_axis}, output_safe_labels={output_safe_labels}")

    adjusted_operands: list[Any] = []
    accumulated_scale: Any | None = None
    for operand, labels in zip(operands, input_labels, strict=True):
        if not isinstance(operand, ScaledTensor):
            adjusted_operands.append(operand)
            continue

        _validate_scaled_operand(operand, labels)
        value, propagated_scale = _fold_unsafe_scale(
            operand,
            labels,
            output_safe_labels,
            ops,
        )
        adjusted_operands.append(value)

        # aligned_scale = _align_scale_to_output(
        #     propagated_scale,
        #     labels,
        #     output_labels,
        #     ops,
        # )
        accumulated_scale = (
            propagated_scale
            if accumulated_scale is None
            else accumulated_scale + propagated_scale
        )

    stable = ops.einsum(format_string, adjusted_operands)
    if operator == SCALED_EINSUM_SUM_OPERATOR:
        factor = ops.sum(stable, axis=output_axis, keepdims=True)
    elif operator == SCALED_EINSUM_MAX_OPERATOR:
        factor = ops.max(stable, axis=output_axis, keepdims=True)
    else:
        raise ValueError(f"unsupported scaled einsum operator: {operator!r}")

    # print(
    #     f" factor shape={getattr(factor, 'shape', 'scalar')}, stable shape={getattr(stable, 'shape', 'scalar')}"
    # )
    if accumulated_scale is None:
        accumulated_scale = factor * 0
    return ScaledTensor(
        stable / factor, accumulated_scale + ops.log(factor), output_axis
    )


def _parse_scaled_einsum_format(format_string: str) -> tuple[tuple[str, str], str]:
    input_labels, output_labels = parse_format_string(format_string.replace(" ", ""))
    if len(input_labels) != 2:
        raise ValueError("scaled einsum requires a binary format string")
    if not output_labels:
        raise ValueError("scaled einsum requires a non-scalar output")
    if any(len(set(labels)) != len(labels) for labels in input_labels):
        raise ValueError("scaled einsum does not support repeated labels in an operand")
    if len(set(output_labels)) != len(output_labels):
        raise ValueError("scaled einsum does not support repeated output labels")
    return (input_labels[0], input_labels[1]), output_labels


def _validate_scaled_operand(operand: ScaledTensor[Any], labels: str) -> None:
    if len(labels) != len(operand.shape):
        raise ValueError("scaled operand rank does not match einsum labels")
    scale_shape = tuple(int(dimension) for dimension in operand.log_scale.shape)
    if scale_shape == ():
        return
    if len(scale_shape) != len(operand.shape):
        raise ValueError("scaled tensor log_scale must be scalar or match value rank")
    for axis, (scale_size, value_size) in enumerate(zip(scale_shape, operand.shape)):
        if scale_size not in (1, value_size):
            raise ValueError("scaled tensor log_scale is not broadcastable to value")
        if axis == operand.scale_axis and scale_size != 1:
            raise ValueError("scaled tensor log_scale must be singleton on scale_axis")


def _fold_unsafe_scale(
    operand: ScaledTensor[Any],
    labels: str,
    output_safe_labels: set[str],
    ops: BackendOps,
) -> tuple[Any, Any]:
    scale_shape = tuple(int(dimension) for dimension in operand.log_scale.shape)
    if scale_shape == ():
        return operand.value, operand.log_scale

    dependency_axes = [axis for axis, size in enumerate(scale_shape) if size != 1]
    unsafe_axes = tuple(
        axis for axis in dependency_axes if labels[axis] not in output_safe_labels
    )
    if not unsafe_axes:
        return operand.value, operand.log_scale

    print(f"  folding unsafe scale for operand with labels {labels}")
    print(
        f"    scale_shape={scale_shape}, dependency_axes={dependency_axes}, unsafe_axes={unsafe_axes}"
    )

    shift = ops.max(operand.log_scale, axis=unsafe_axes, keepdims=True)
    adjusted_value = operand.value * ops.exp(operand.log_scale - shift)
    return adjusted_value, shift


def _align_scale_to_output(
    scale: Any,
    input_labels: str,
    output_labels: str,
    ops: BackendOps,
) -> Any:
    scale_shape = tuple(int(dimension) for dimension in getattr(scale, "shape", ()))
    if scale_shape == ():
        print("  aligning scale: scale is scalar, no reshape needed")
        return ops.reshape(scale, (1,) * len(output_labels))

    kept_labels = "".join(label for label in output_labels if label in input_labels)
    if kept_labels:
        print(
            f"  aligning scale: keeping labels {kept_labels} from input_labels {input_labels}"
        )
        reordered = ops.einsum(f"{input_labels}->{kept_labels}", [scale])
    else:
        print("  aligning scale: no labels kept, reducing to scalar")
        reordered = ops.einsum(f"{input_labels}->", [scale])

    target_shape: list[int] = []
    source_axis = 0
    for label in output_labels:
        if label in kept_labels:
            target_shape.append(int(reordered.shape[source_axis]))
            source_axis += 1
        else:
            target_shape.append(1)
    return ops.reshape(reordered, tuple(target_shape))
