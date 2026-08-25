import numpy as np
import numpy.typing as npt
import pytest
import torch

from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backend_translation.translate import translate_to_backend_program
from extended_einsum.backends.numpy import NumpyBackendFunctions
from extended_einsum.backends.torch import TorchBackendFunctions
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
    OperatorSin,
    OperatorSlice,
    OperatorSoftmax,
    OperatorStack,
    OperatorSubtract,
    OperatorTake,
    RichOperator,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import Shape, StabilityMode

BACKEND_FUNCTIONS = NumpyBackendFunctions()


def _dense_program(
    instructions: list[RichInstruction],
    n_inputs: int,
    stability_mode: StabilityMode = "unstable",
    parameter_indices: frozenset[int] = frozenset(),
    shapes: list[Shape] | None = None,
) -> RichProgram:
    n_ssa_ids = n_inputs + len(instructions)
    return RichProgram(
        instructions=instructions,
        n_inputs=n_inputs,
        stability_mode=stability_mode,
        tensor_formats=["dense"] * n_ssa_ids,
        shapes=[()] * n_ssa_ids if shapes is None else shapes,
        parameter_indices=parameter_indices,
    )


def _single_instruction_program(operator: RichOperator, n_arguments: int, stability_mode: StabilityMode) -> RichProgram:
    return _dense_program([RichInstruction(operator, tuple(range(n_arguments)))], n_inputs=n_arguments, stability_mode=stability_mode)


# Scaled translation supports three fiber normalizers; log space supports two shifts.
SCALED_MODES: list[StabilityMode] = ["scaled_min", "scaled_max", "scaled_sum"]
LOGSPACE_MODES: list[StabilityMode] = ["logspace_min", "logspace_max"]
STABLE_MODES: list[StabilityMode] = [*SCALED_MODES, *LOGSPACE_MODES]
ALL_MODES: list[StabilityMode] = ["unstable", *STABLE_MODES]


_RNG = np.random.default_rng(seed=0)
MATRIX = _RNG.standard_normal((3, 4))
OTHER_MATRIX = _RNG.standard_normal((3, 4))
POSITIVE_MATRIX = np.abs(_RNG.standard_normal((3, 4))) + 0.5
OTHER_POSITIVE_MATRIX = np.abs(_RNG.standard_normal((3, 4))) + 0.5
POSITIVE_RIGHT_MATRIX = np.abs(_RNG.standard_normal((4, 5))) + 0.5
RIGHT_MATRIX = _RNG.standard_normal((4, 5))
INDICES = np.array([2, 0, 3])
TENSOR = _RNG.standard_normal((2, 3, 4))
MULTI_AXIS_SOFTMAX = np.exp(TENSOR - np.max(TENSOR, axis=(1, 2), keepdims=True))
MULTI_AXIS_SOFTMAX /= np.sum(MULTI_AXIS_SOFTMAX, axis=(1, 2), keepdims=True)


################################
# translation of single instructions (unstable)
################################


UNSTABLE_TRANSLATION_CASES: list[tuple[RichOperator, list[npt.NDArray], npt.NDArray]] = [
    (OperatorExp(), [MATRIX], np.exp(MATRIX)),
    (OperatorLog(), [POSITIVE_MATRIX], np.log(POSITIVE_MATRIX)),
    (OperatorAdd(), [MATRIX, OTHER_MATRIX], MATRIX + OTHER_MATRIX),
    (OperatorSubtract(), [MATRIX, OTHER_MATRIX], MATRIX - OTHER_MATRIX),
    (OperatorMultiply(), [MATRIX, OTHER_MATRIX], MATRIX * OTHER_MATRIX),
    (OperatorDivide(), [MATRIX, POSITIVE_MATRIX], MATRIX / POSITIVE_MATRIX),
    (OperatorStack(axis=1), [MATRIX, OTHER_MATRIX], np.stack([MATRIX, OTHER_MATRIX], axis=1)),
    (OperatorTake(axis=1), [MATRIX, INDICES], MATRIX[:, INDICES]),
    (OperatorSelect(axis=0, index=1), [MATRIX], MATRIX[1]),
    (OperatorSlice(start=1, stop=3, axis=1), [MATRIX], MATRIX[:, 1:3]),
    (OperatorSoftmax(axis=1), [MATRIX], np.exp(MATRIX) / np.sum(np.exp(MATRIX), axis=1, keepdims=True)),
    (OperatorSoftmax(axis=(1, 2)), [TENSOR], MULTI_AXIS_SOFTMAX),
    (OperatorEinsum("ik, kj -> ij"), [MATRIX, RIGHT_MATRIX], MATRIX @ RIGHT_MATRIX),
]
UNSTABLE_TRANSLATION_CASE_IDS = [operator.name for operator, _, _ in UNSTABLE_TRANSLATION_CASES]


@pytest.mark.parametrize(("operator", "tensor_arguments", "expected_result"), UNSTABLE_TRANSLATION_CASES, ids=UNSTABLE_TRANSLATION_CASE_IDS)
def test_unstable_translation_computes_actual_result(operator: RichOperator, tensor_arguments: list[npt.NDArray], expected_result: npt.NDArray) -> None:
    rich_program = _single_instruction_program(operator, len(tensor_arguments), "unstable")

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, tensor_arguments)

    np.testing.assert_allclose(result, expected_result)


def test_unstable_translation_produces_one_call_per_instruction() -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorAdd(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorMultiply(), (2, 0)),  # ssa id 3
        ],
        n_inputs=2,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)

    assert backend_program.n_inputs == 2
    assert len(backend_program.backend_calls) == 2
    assert backend_program.call_arguments == [(0, 1), (2, 0)]


@pytest.mark.parametrize(
    "format_string",
    ["dacb,dbc->dab", "xyzw,xwz->xyw"],
)
def test_logspace_short_same_index_contraction_matches_general_einsum(
    format_string: str,
) -> None:
    log_values = torch.randn(3, 5, 2, 7, requires_grad=True)
    weights = (torch.rand(3, 7, 2) + 0.5).requires_grad_()
    expected_log_values = log_values.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),
            RichInstruction(OperatorEinsum(format_string), (2, 1)),
            RichInstruction(OperatorLog(), (3,)),
        ],
        n_inputs=2,
        stability_mode="logspace_max",
        parameter_indices=frozenset({1}),
        shapes=[
            (3, 5, 2, 7),
            (3, 7, 2),
            (3, 5, 2, 7),
            (3, 5, 7),
            (3, 5, 7),
        ],
    )
    result = run_program(
        translate_to_backend_program(
            rich_program,
            TorchBackendFunctions(),
        ),
        (log_values, weights),
    )
    expected = torch.log(
        torch.einsum(
            format_string,
            torch.exp(expected_log_values),
            expected_weights,
        )
    )
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(result.sum(), (log_values, weights))
    expected_gradients = torch.autograd.grad(
        expected.sum(),
        (expected_log_values, expected_weights),
    )
    for gradient, expected_gradient in zip(
        gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("stability_mode", ALL_MODES)
def test_translation_rejects_operator_without_backend_function(stability_mode: StabilityMode) -> None:
    rich_program = _single_instruction_program(OperatorSin(), 1, stability_mode)

    with pytest.raises(NotImplementedError, match="sin"):
        translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)


def test_run_program_rejects_wrong_number_of_inputs() -> None:
    rich_program = _single_instruction_program(OperatorExp(), 1, "unstable")
    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)

    with pytest.raises(ValueError, match="number of inputs"):
        run_program(backend_program, [MATRIX, MATRIX])


################################
# translation of single instructions (stable modes)
################################

# single-instruction programs compared against the actual result. values that the stable translations convert
# are strictly positive, because the scaled translation normalizes them by their total sum or minimum and the
# logspace translation takes their logarithm
STABLE_TRANSLATION_CASES: list[tuple[str, RichOperator, list[npt.NDArray], npt.NDArray]] = [
    ("exp", OperatorExp(), [MATRIX], np.exp(MATRIX)),
    ("log", OperatorLog(), [POSITIVE_MATRIX], np.log(POSITIVE_MATRIX)),
    ("add", OperatorAdd(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX + OTHER_POSITIVE_MATRIX),
    # the minuend is constructed so that the difference is exactly POSITIVE_MATRIX, keeping the result positive
    ("subtract", OperatorSubtract(), [POSITIVE_MATRIX + OTHER_POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX),
    ("multiply", OperatorMultiply(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX),
    ("divide", OperatorDivide(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX / OTHER_POSITIVE_MATRIX),
    ("stack", OperatorStack(axis=1), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], np.stack([POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], axis=1)),
    ("take", OperatorTake(axis=1), [POSITIVE_MATRIX, INDICES], POSITIVE_MATRIX[:, INDICES]),
    ("select", OperatorSelect(axis=0, index=1), [POSITIVE_MATRIX], POSITIVE_MATRIX[1]),
    ("slice", OperatorSlice(start=1, stop=3, axis=1), [POSITIVE_MATRIX], POSITIVE_MATRIX[:, 1:3]),
    ("softmax", OperatorSoftmax(axis=1), [MATRIX], np.exp(MATRIX) / np.sum(np.exp(MATRIX), axis=1, keepdims=True)),
    ("multi-axis-softmax", OperatorSoftmax(axis=(1, 2)), [TENSOR], MULTI_AXIS_SOFTMAX),
    ("einsum-matmul", OperatorEinsum("ik, kj -> ij"), [POSITIVE_MATRIX, POSITIVE_RIGHT_MATRIX], POSITIVE_MATRIX @ POSITIVE_RIGHT_MATRIX),
    ("einsum-total-sum", OperatorEinsum("ik ->"), [POSITIVE_MATRIX], np.sum(POSITIVE_MATRIX)),
]
STABLE_TRANSLATION_CASE_IDS = [case_name for case_name, _, _, _ in STABLE_TRANSLATION_CASES]


@pytest.mark.parametrize("stability_mode", STABLE_MODES)
@pytest.mark.parametrize(("case_name", "operator", "tensor_arguments", "expected_result"), STABLE_TRANSLATION_CASES, ids=STABLE_TRANSLATION_CASE_IDS)
def test_stable_translation_computes_actual_result(case_name: str, operator: RichOperator, tensor_arguments: list[npt.NDArray], expected_result: npt.NDArray, stability_mode: StabilityMode) -> None:
    rich_program = _single_instruction_program(operator, len(tensor_arguments), stability_mode)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, tensor_arguments)

    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


################################
# translation of single instructions (scaled)
################################


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_translation_leaves_raw_only_values_unscaled(stability_mode: StabilityMode) -> None:
    # softmax consumes its argument raw and produces a raw result, so the input must not be scaled at all
    rich_program = _single_instruction_program(OperatorSoftmax(axis=1), 1, stability_mode)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)

    assert len(backend_program.backend_calls) == 1
    assert backend_program.call_arguments == [(0,)]


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_translation_scales_each_value_at_most_once(stability_mode: StabilityMode) -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorMultiply(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorMultiply(), (2, 0)),  # ssa id 3, consumes input 0 as a scaled pair a second time
        ],
        n_inputs=2,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX])

    # scaling the two inputs (4 calls each), two multiplications (2 calls each), and the raw conversion of the output (2 calls)
    assert len(backend_program.backend_calls) == 14
    np.testing.assert_allclose(result, POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX * POSITIVE_MATRIX, rtol=1e-9)


def test_scaled_translation_rejects_non_positive_scale_interval() -> None:
    rich_program = _single_instruction_program(OperatorLog(), 1, "scaled_sum")

    with pytest.raises(ValueError, match="scale interval"):
        translate_to_backend_program(
            rich_program,
            BACKEND_FUNCTIONS,
            scale_interval=0,
        )


def test_scaled_translation_defaults_to_three_contractions_between_output_normalizations() -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("bi,oi->bo"), (0, 1)),
            RichInstruction(OperatorEinsum("bi,oi->bo"), (2, 1)),
            RichInstruction(OperatorEinsum("bi,oi->bo"), (3, 1)),
            RichInstruction(OperatorLog(), (4,)),
        ],
        n_inputs=2,
        stability_mode="scaled_sum",
        parameter_indices=frozenset({1}),
        shapes=[(3, 4), (4, 4), (3, 4), (3, 4), (3, 4), (3, 4)],
    )
    inputs = [POSITIVE_MATRIX, np.full((4, 4), 0.125)]

    default_interval = translate_to_backend_program(
        rich_program,
        BACKEND_FUNCTIONS,
    )
    explicit_interval = translate_to_backend_program(
        rich_program,
        BACKEND_FUNCTIONS,
        scale_interval=3,
    )
    every_layer = translate_to_backend_program(
        rich_program,
        BACKEND_FUNCTIONS,
        scale_interval=1,
    )

    np.testing.assert_allclose(
        run_program(default_interval, inputs),
        run_program(every_layer, inputs),
        rtol=1e-9,
        atol=1e-12,
    )
    assert len(default_interval.backend_calls) == len(
        explicit_interval.backend_calls
    )
    assert default_interval.call_arguments == explicit_interval.call_arguments
    assert len(default_interval.backend_calls) < len(every_layer.backend_calls)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_translation_of_exp_and_log_survives_overflowing_exponentials(stability_mode: StabilityMode) -> None:
    # exp moves the maximum of its raw argument into the log scale and log adds it back, so the roundtrip survives exponentials that overflow raw tensors
    # (exp stores its result directly as a scaled pair and log reads it back, so no min/sum normalization runs and both scaled variants agree here)
    huge_matrix = 800.0 + MATRIX
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 1
            RichInstruction(OperatorLog(), (1,)),  # ssa id 2
        ],
        n_inputs=1,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [huge_matrix])

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, huge_matrix, rtol=1e-9)


def test_scaled_sum_einsum_keeps_parameter_weights_linear_and_scales_each_data_row() -> None:
    log_values = np.array([[0.0, -1.0], [-1000.0, -1001.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 2
            RichInstruction(OperatorEinsum("bi,io->bo"), (2, 1)),  # ssa id 3
            RichInstruction(OperatorLog(), (3,)),  # ssa id 4
        ],
        n_inputs=2,
        stability_mode="scaled_sum",
        parameter_indices=frozenset({1}),
        shapes=[(2, 2)] * 5,
    )

    backend_program = translate_to_backend_program(
        rich_program,
        BACKEND_FUNCTIONS,
        scale_interval=1,
    )
    result = run_program(backend_program, [log_values, weights])

    row_maxima = np.max(log_values, axis=1, keepdims=True)
    expected_result = row_maxima + np.log(np.einsum("bi,io->bo", np.exp(log_values - row_maxima), weights))
    assert np.all(np.isfinite(result))
    assert len(backend_program.backend_calls) == 12
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def test_scaled_sum_preserves_folded_and_batch_scales_through_slicing() -> None:
    log_values = np.array(
        [
            [[0.0, -1.0], [-1000.0, -1001.0]],
            [[-10.0, -11.0], [-1010.0, -1011.0]],
            [[-20.0, -21.0], [-1020.0, -1021.0]],
            [[-30.0, -31.0], [-1030.0, -1031.0]],
        ]
    )
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 2
            RichInstruction(OperatorSlice(start=1, stop=3, axis=0), (2,)),  # ssa id 3
            RichInstruction(OperatorEinsum("fbi,io->fbo"), (3, 1)),  # ssa id 4
            RichInstruction(OperatorLog(), (4,)),  # ssa id 5
        ],
        n_inputs=2,
        stability_mode="scaled_sum",
        parameter_indices=frozenset({1}),
        shapes=[(4, 2, 2), (2, 2), (4, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [log_values, weights])

    sliced_values = log_values[1:3]
    fiber_maxima = np.max(sliced_values, axis=-1, keepdims=True)
    expected_result = fiber_maxima + np.log(np.einsum("fbi,io->fbo", np.exp(sliced_values - fiber_maxima), weights))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def test_scaled_sum_elementwise_einsum_propagates_scales_directly() -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("ij,ij->ij"), (0, 1)),  # ssa id 2
            RichInstruction(OperatorLog(), (2,)),  # ssa id 3
        ],
        n_inputs=2,
        stability_mode="scaled_sum",
        shapes=[(3, 4)] * 4,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX])

    # Normalize each input once, then use one einsum and one scale addition;
    # no output renormalization or scale reshaping is necessary.
    assert len(backend_program.backend_calls) == 12
    np.testing.assert_allclose(result, np.log(POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX), rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_kronecker_einsum_broadcasts_scales_without_renormalizing(stability_mode: StabilityMode) -> None:
    left = POSITIVE_MATRIX[:, :2]
    right = OTHER_POSITIVE_MATRIX[:, :3]
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("bi,bj->bij"), (0, 1)),
            RichInstruction(OperatorLog(), (2,)),
        ],
        n_inputs=2,
        stability_mode=stability_mode,
        shapes=[left.shape, right.shape, (3, 2, 3), (3, 2, 3)],
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [left, right])

    # Normalize both operands, multiply them, broadcast their two scales, and
    # consume the scaled result directly in log form.
    assert len(backend_program.backend_calls) == 14
    np.testing.assert_allclose(result, np.log(left[:, :, None] * right[:, None, :]), rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_concat_preserves_independent_fiber_scales(stability_mode: StabilityMode) -> None:
    large = np.full((2, 3), 1.0e300)
    small = np.full((1, 3), 1.0e-300)
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorConcat(axis=0), (0, 1)),
            RichInstruction(OperatorLog(), (2,)),
        ],
        n_inputs=2,
        stability_mode=stability_mode,
        shapes=[large.shape, small.shape, (3, 3), (3, 3)],
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [large, small])

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, np.log(np.concatenate((large, small), axis=0)), rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_concat_canonicalizes_scales_that_vary_along_fibers(
    stability_mode: StabilityMode,
) -> None:
    left = POSITIVE_MATRIX[:2, :3]
    right = 1.0e3 * OTHER_POSITIVE_MATRIX[:2, :3]
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("ab->ba"), (0,)),
            RichInstruction(OperatorEinsum("ab->ba"), (1,)),
            RichInstruction(OperatorConcat(axis=0), (2, 3)),
            RichInstruction(OperatorLog(), (4,)),
        ],
        n_inputs=2,
        stability_mode=stability_mode,
        shapes=[
            left.shape,
            right.shape,
            (3, 2),
            (3, 2),
            (6, 2),
            (6, 2),
        ],
    )

    backend_program = translate_to_backend_program(
        rich_program,
        BACKEND_FUNCTIONS,
    )
    result = run_program(backend_program, [left, right])
    expected = np.log(
        np.concatenate((left.T, right.T), axis=0)
    )

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_concat_preserves_gradients(stability_mode: StabilityMode) -> None:
    input_values = [POSITIVE_MATRIX[:2, :3], 1.0e3 * OTHER_POSITIVE_MATRIX[:1, :3]]
    instructions = [
        RichInstruction(OperatorConcat(axis=0), (0, 1)),
        RichInstruction(OperatorEinsum("ij->"), (2,)),
    ]
    shapes = [(2, 3), (1, 3), (3, 3), ()]

    def gradients(mode: StabilityMode) -> tuple[torch.Tensor, ...]:
        inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in input_values]
        rich_program = _dense_program(instructions, n_inputs=2, stability_mode=mode, shapes=shapes)
        result = run_program(translate_to_backend_program(rich_program, TorchBackendFunctions()), inputs)
        return torch.autograd.grad(result, inputs)

    baseline_gradients = gradients("unstable")
    scaled_gradients = gradients(stability_mode)
    for gradient, baseline_gradient in zip(scaled_gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(gradient, baseline_gradient)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_reassociated_tucker_contractions_track_non_fiber_scale_shape(stability_mode: StabilityMode) -> None:
    data_left = np.abs(_RNG.standard_normal((2, 5, 3))) + 0.5
    data_right = np.abs(_RNG.standard_normal((2, 5, 4))) + 0.5
    weights = np.abs(_RNG.standard_normal((2, 6, 3, 4))) + 0.5
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("abcd,aed->abce"), (2, 1)),
            RichInstruction(OperatorEinsum("abc,adcb->abd"), (0, 3)),
            RichInstruction(OperatorLog(), (4,)),
        ],
        n_inputs=3,
        stability_mode=stability_mode,
        parameter_indices=frozenset({2}),
        shapes=[
            data_left.shape,
            data_right.shape,
            weights.shape,
            (2, 6, 3, 5),
            (2, 5, 6),
            (2, 5, 6),
        ],
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [data_left, data_right, weights])

    intermediate = np.einsum("abcd,aed->abce", weights, data_right)
    expected = np.log(np.einsum("abc,adcb->abd", data_left, intermediate))
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_mixing_einsum_accepts_scalar_scale_from_stack(stability_mode: StabilityMode) -> None:
    first = np.abs(_RNG.standard_normal((5, 3))) + 0.5
    second = np.abs(_RNG.standard_normal((5, 3))) + 0.5
    weights = np.array([[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorStack(axis=1), (0, 1)),
            RichInstruction(OperatorEinsum("bhu,uh->bu"), (3, 2)),
            RichInstruction(OperatorLog(), (4,)),
        ],
        n_inputs=3,
        stability_mode=stability_mode,
        parameter_indices=frozenset({2}),
        shapes=[first.shape, second.shape, weights.shape, (5, 2, 3), (5, 3), (5, 3)],
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [first, second, weights])

    expected = np.log(np.einsum("bhu,uh->bu", np.stack([first, second], axis=1), weights))
    np.testing.assert_allclose(result, expected, rtol=1e-9)


################################
# translation of single instructions (logspace)
################################


@pytest.mark.parametrize("stability_mode", LOGSPACE_MODES)
def test_logspace_translation_of_exp_and_log_is_free(stability_mode: StabilityMode) -> None:
    # exp and log only move values between the raw and logspace parts, so the roundtrip compiles to no backend calls at all
    huge_matrix = 800.0 + MATRIX
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 1
            RichInstruction(OperatorLog(), (1,)),  # ssa id 2
        ],
        n_inputs=1,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [huge_matrix])

    assert len(backend_program.backend_calls) == 0
    np.testing.assert_array_equal(result, huge_matrix)


@pytest.mark.parametrize("stability_mode", LOGSPACE_MODES)
def test_logspace_translation_converts_each_value_at_most_once(stability_mode: StabilityMode) -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorMultiply(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorMultiply(), (2, 0)),  # ssa id 3, consumes input 0 in logspace a second time
        ],
        n_inputs=2,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX])

    # converting the two inputs to logspace, two logspace multiplications, and the raw conversion of the output
    assert len(backend_program.backend_calls) == 5
    np.testing.assert_allclose(result, POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX * POSITIVE_MATRIX, rtol=1e-9)


def test_logspace_max_einsum_keeps_parameter_weights_linear_and_shifts_each_data_row() -> None:
    log_values = np.array([[0.0, -1.0], [-1000.0, -1001.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 2
            RichInstruction(OperatorEinsum("bi, io -> bo"), (2, 1)),  # ssa id 3
            RichInstruction(OperatorLog(), (3,)),  # ssa id 4
        ],
        n_inputs=2,
        stability_mode="logspace_max",
        parameter_indices=frozenset({1}),
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [log_values, weights])

    row_maxima = np.max(log_values, axis=1, keepdims=True)
    expected_result = row_maxima + np.log(np.einsum("bi,io->bo", np.exp(log_values - row_maxima), weights))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


@pytest.mark.parametrize("stability_mode", LOGSPACE_MODES)
def test_logspace_elementwise_einsum_is_addition(stability_mode: StabilityMode) -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("ij,ij->ij"), (0, 1)),
            RichInstruction(OperatorLog(), (2,)),
        ],
        n_inputs=2,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX])

    # Convert both inputs to log space and add them; the final log instruction
    # only changes which representation is returned and requires no call.
    assert len(backend_program.backend_calls) == 3
    np.testing.assert_allclose(result, np.log(POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX), rtol=1e-9)


@pytest.mark.parametrize("stability_mode", LOGSPACE_MODES)
def test_logspace_kronecker_einsum_is_broadcast_addition(stability_mode: StabilityMode) -> None:
    left = POSITIVE_MATRIX[:, :2]
    right = OTHER_POSITIVE_MATRIX[:, :3]
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("bi,bj->bij"), (0, 1)),
            RichInstruction(OperatorLog(), (2,)),
        ],
        n_inputs=2,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [left, right])

    # Two logarithms, two broadcast reshapes, and one addition. In particular,
    # this must not lower to exp/einsum/log as a general contraction would.
    assert len(backend_program.backend_calls) == 5
    np.testing.assert_allclose(result, np.log(left[:, :, None] * right[:, None, :]), rtol=1e-9)


def test_logspace_max_einsum_shifts_rows_by_output_label_after_axis_reordering() -> None:
    log_values = np.array([[0.0, -1000.0], [-1.0, -1001.0]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 2
            RichInstruction(OperatorEinsum("ib, io -> bo"), (2, 1)),  # ssa id 3; batch/row label b is axis 1
            RichInstruction(OperatorLog(), (3,)),  # ssa id 4
        ],
        n_inputs=2,
        stability_mode="logspace_max",
        parameter_indices=frozenset({1}),
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [log_values, weights])

    column_maxima = np.max(log_values, axis=0, keepdims=True)
    expected_result = column_maxima.T + np.log(np.einsum("ib,io->bo", np.exp(log_values - column_maxima), weights))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def test_logspace_max_einsum_keeps_softmax_derived_parameters_linear() -> None:
    log_values = np.array([[0.0, -1.0], [-1000.0, -1001.0]])
    weight_logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorSoftmax(axis=0), (1,)),  # ssa id 2, still parameter-derived
            RichInstruction(OperatorExp(), (0,)),  # ssa id 3
            RichInstruction(OperatorEinsum("bi, io -> bo"), (3, 2)),  # ssa id 4
            RichInstruction(OperatorLog(), (4,)),  # ssa id 5
        ],
        n_inputs=2,
        stability_mode="logspace_max",
        parameter_indices=frozenset({1}),
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [log_values, weight_logits])

    weights = np.exp(weight_logits - np.max(weight_logits, axis=0, keepdims=True))
    weights /= np.sum(weights, axis=0, keepdims=True)
    row_maxima = np.max(log_values, axis=1, keepdims=True)
    expected_result = row_maxima + np.log(np.einsum("bi,io->bo", np.exp(log_values - row_maxima), weights))
    # softmax, the shifted data operand, the detached numerical shift, the
    # einsum, and restoring the row shift require ten calls;
    # converting and shifting the softmax result as another log-space operand would require five more.
    assert len(backend_program.backend_calls) == 10
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


################################
# full translation pipeline
################################


@pytest.mark.parametrize("stability_mode", ALL_MODES)
def test_translate_to_backend_program_end_to_end(stability_mode: StabilityMode) -> None:
    # a test-local generator keeps the inputs independent of which tests ran before
    rng = np.random.default_rng(seed=1)
    left_matrix = np.abs(rng.standard_normal((3, 4))) + 0.5
    right_matrix = np.abs(rng.standard_normal((4, 5))) + 0.5
    bias_matrix = np.abs(rng.standard_normal((3, 5))) + 0.5
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("ik, kj -> ij"), (0, 1)),  # ssa id 3
            RichInstruction(OperatorAdd(), (3, 2)),  # ssa id 4
            RichInstruction(OperatorSoftmax(axis=1), (4,)),  # ssa id 5
            RichInstruction(OperatorStack(axis=0), (5, 5)),  # ssa id 6
        ],
        n_inputs=3,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [left_matrix, right_matrix, bias_matrix])

    logits = left_matrix @ right_matrix + bias_matrix
    softmaxed = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    expected_result = np.stack([softmaxed, softmaxed], axis=0)
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


@pytest.mark.parametrize("stability_mode", ALL_MODES)
def test_translate_to_backend_program_with_mixed_representations(stability_mode: StabilityMode) -> None:
    # in scaled mode this program mixes representations: the einsum introduces a scale, the multiplication
    # and addition combine it with raw inputs, and the logarithm eliminates it again
    rng = np.random.default_rng(seed=2)
    left_matrix = np.abs(rng.standard_normal((3, 4))) + 0.5
    right_matrix = np.abs(rng.standard_normal((4, 5))) + 0.5
    factor_matrix = np.abs(rng.standard_normal((3, 5))) + 0.5
    summand_matrix = np.abs(rng.standard_normal((3, 5))) + 0.5
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorEinsum("ik, kj -> ij"), (0, 1)),  # ssa id 4
            RichInstruction(OperatorMultiply(), (4, 2)),  # ssa id 5
            RichInstruction(OperatorAdd(), (5, 3)),  # ssa id 6
            RichInstruction(OperatorLog(), (6,)),  # ssa id 7
        ],
        n_inputs=4,
        stability_mode=stability_mode,
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [left_matrix, right_matrix, factor_matrix, summand_matrix])

    expected_result = np.log((left_matrix @ right_matrix) * factor_matrix + summand_matrix)
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def _huge_chain_program(stability_mode: StabilityMode, n_factors: int) -> RichProgram:
    """A chain of matrix products, normalized by the total sum of the final product."""
    instructions: list[RichInstruction] = []
    running_ssa_id = 0
    for factor_index in range(1, n_factors):
        instructions.append(RichInstruction(OperatorEinsum("ij, jk -> ik"), (running_ssa_id, factor_index)))
        running_ssa_id = n_factors + factor_index - 1
    instructions.append(RichInstruction(OperatorEinsum("ij ->"), (running_ssa_id,)))
    total_ssa_id = running_ssa_id + 1
    instructions.append(RichInstruction(OperatorDivide(), (running_ssa_id, total_ssa_id)))
    return _dense_program(instructions, n_inputs=n_factors, stability_mode=stability_mode)


N_HUGE_FACTORS = 9
RAW_CHAIN_FACTORS = [0.5 + _RNG.random((3, 3)) for _ in range(N_HUGE_FACTORS)]
HUGE_CHAIN_FACTORS = [1.0e70 * raw_factor for raw_factor in RAW_CHAIN_FACTORS]


@pytest.mark.parametrize("stability_mode", STABLE_MODES)
def test_stable_modes_survive_huge_intermediate_values(stability_mode: StabilityMode) -> None:
    rich_program = _huge_chain_program(stability_mode, N_HUGE_FACTORS)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, HUGE_CHAIN_FACTORS)

    # the huge scalar factors cancel out in the normalization, so the expected result is computable from the raw factors
    raw_chain_product = RAW_CHAIN_FACTORS[0]
    for raw_factor in RAW_CHAIN_FACTORS[1:]:
        raw_chain_product = raw_chain_product @ raw_factor
    expected_result = raw_chain_product / np.sum(raw_chain_product)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


################################
# agreement between stability modes
################################

# every stability mode must compute the same result as the unstable baseline, up to numeric error
POSITIVE_BIAS_MATRIX = np.abs(_RNG.standard_normal((3, 5))) + 0.5
POSITIVE_VALUE_MATRIX = np.abs(_RNG.standard_normal((5, 2))) + 0.5
POSITIVE_SQUARE_MATRICES = [np.abs(_RNG.standard_normal((3, 3))) + 0.5 for _ in range(3)]

# single-operator programs covering every translated operator, plus composite programs that mix representations.
# inputs that the scaled translation consumes as scaled pairs are positive, because it normalizes them by their total sum
AGREEMENT_CASES: list[tuple[str, list[RichInstruction], list[npt.NDArray]]] = [
    (
        "exp",
        [RichInstruction(OperatorExp(), (0,))],
        [MATRIX],
    ),
    (
        "log",
        [RichInstruction(OperatorLog(), (0,))],
        [POSITIVE_MATRIX],
    ),
    (
        "add",
        [RichInstruction(OperatorAdd(), (0, 1))],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
    (
        "subtract-with-mixed-sign-result",
        [RichInstruction(OperatorSubtract(), (0, 1))],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
    (
        "multiply",
        [RichInstruction(OperatorMultiply(), (0, 1))],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
    (
        "divide",
        [RichInstruction(OperatorDivide(), (0, 1))],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
    (
        "stack-with-mismatched-scales",
        [RichInstruction(OperatorStack(axis=0), (0, 1, 2))],
        [POSITIVE_MATRIX, 1.0e6 * OTHER_POSITIVE_MATRIX, 1.0e-6 * POSITIVE_MATRIX],
    ),
    (
        "take",
        [RichInstruction(OperatorTake(axis=1), (0, 1))],
        [POSITIVE_MATRIX, INDICES],
    ),
    (
        "select",
        [RichInstruction(OperatorSelect(axis=0, index=1), (0,))],
        [POSITIVE_MATRIX],
    ),
    (
        "slice",
        [RichInstruction(OperatorSlice(start=1, stop=3, axis=1), (0,))],
        [POSITIVE_MATRIX],
    ),
    (
        "softmax",
        [RichInstruction(OperatorSoftmax(axis=1), (0,))],
        [MATRIX],
    ),
    (
        "einsum-matmul",
        [RichInstruction(OperatorEinsum("ik, kj -> ij"), (0, 1))],
        [POSITIVE_MATRIX, POSITIVE_RIGHT_MATRIX],
    ),
    (
        "einsum-total-sum",
        [RichInstruction(OperatorEinsum("ik ->"), (0,))],
        [POSITIVE_MATRIX],
    ),
    (
        "attention-block",
        [
            RichInstruction(OperatorEinsum("ik, kj -> ij"), (0, 1)),  # ssa id 4
            RichInstruction(OperatorAdd(), (4, 2)),  # ssa id 5
            RichInstruction(OperatorSoftmax(axis=1), (5,)),  # ssa id 6
            RichInstruction(OperatorEinsum("ij, jk -> ik"), (6, 3)),  # ssa id 7
        ],
        [POSITIVE_MATRIX, POSITIVE_RIGHT_MATRIX, POSITIVE_BIAS_MATRIX, POSITIVE_VALUE_MATRIX],
    ),
    (
        "logsumexp",
        [
            RichInstruction(OperatorExp(), (0,)),  # ssa id 1
            RichInstruction(OperatorEinsum("ij ->"), (1,)),  # ssa id 2
            RichInstruction(OperatorLog(), (2,)),  # ssa id 3
        ],
        [MATRIX],
    ),
    (
        "reused-input",
        [
            RichInstruction(OperatorMultiply(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorAdd(), (2, 0)),  # ssa id 3, consumes input 0 a second time
            RichInstruction(OperatorLog(), (3,)),  # ssa id 4
        ],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
    (
        "normalized-chain-product",
        [
            RichInstruction(OperatorEinsum("ij, jk -> ik"), (0, 1)),  # ssa id 3
            RichInstruction(OperatorEinsum("ij, jk -> ik"), (3, 2)),  # ssa id 4
            RichInstruction(OperatorEinsum("ij ->"), (4,)),  # ssa id 5
            RichInstruction(OperatorDivide(), (4, 5)),  # ssa id 6
        ],
        [1.0e3 * square_matrix for square_matrix in POSITIVE_SQUARE_MATRICES],
    ),
    (
        "einsum-with-repeated-argument",
        [RichInstruction(OperatorEinsum("ij, ij ->"), (0, 0))],
        [POSITIVE_MATRIX],
    ),
    (
        "value-consumed-both-raw-and-converted",
        [
            RichInstruction(OperatorMultiply(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorSoftmax(axis=1), (2,)),  # ssa id 3, consumes ssa id 2 raw
            RichInstruction(OperatorMultiply(), (3, 2)),  # ssa id 4, consumes ssa id 2 as a scaled pair / in logspace
        ],
        [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX],
    ),
]
AGREEMENT_CASE_IDS = [case_name for case_name, _, _ in AGREEMENT_CASES]


@pytest.mark.parametrize("stability_mode", STABLE_MODES)
@pytest.mark.parametrize(("case_name", "instructions", "tensor_arguments"), AGREEMENT_CASES, ids=AGREEMENT_CASE_IDS)
def test_stability_modes_compute_same_result(case_name: str, instructions: list[RichInstruction], tensor_arguments: list[npt.NDArray], stability_mode: StabilityMode) -> None:
    if stability_mode in LOGSPACE_MODES and case_name == "subtract-with-mixed-sign-result":
        pytest.skip("logspace cannot represent the negative values in the subtraction result")

    def run_in_mode(mode: StabilityMode) -> npt.NDArray:
        rich_program = _dense_program(instructions, n_inputs=len(tensor_arguments), stability_mode=mode)
        backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
        return run_program(backend_program, tensor_arguments)

    baseline_result = run_in_mode("unstable")
    result = run_in_mode(stability_mode)

    assert result.shape == baseline_result.shape
    np.testing.assert_allclose(result, baseline_result, rtol=1e-9)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_detached_normalizers_preserve_gradients(stability_mode: StabilityMode) -> None:
    instructions = [
        RichInstruction(OperatorStack(axis=1), (0, 1)),  # ssa id 3
        RichInstruction(OperatorEinsum("bhu,uh->bu"), (3, 2)),  # ssa id 4
        RichInstruction(OperatorEinsum("bu->"), (4,)),  # ssa id 5
    ]
    shapes = [(2, 3), (2, 3), (3, 2), (2, 2, 3), (2, 3), ()]
    input_values = [
        POSITIVE_MATRIX[:2, :3],
        1.0e3 * OTHER_POSITIVE_MATRIX[:2, :3],
        np.abs(_RNG.standard_normal((3, 2))) + 0.5,
    ]

    def value_and_grad(mode: StabilityMode) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in input_values]
        rich_program = _dense_program(instructions, n_inputs=3, stability_mode=mode, parameter_indices=frozenset({2}), shapes=shapes)
        backend_program = translate_to_backend_program(rich_program, TorchBackendFunctions())
        result = run_program(backend_program, inputs)
        gradients = torch.autograd.grad(result, inputs)
        return result, list(gradients)

    baseline_result, baseline_gradients = value_and_grad("unstable")
    result, gradients = value_and_grad(stability_mode)

    torch.testing.assert_close(result, baseline_result)
    for gradient, baseline_gradient in zip(gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(gradient, baseline_gradient)


@pytest.mark.parametrize("stability_mode", SCALED_MODES)
def test_scaled_detached_reference_scales_preserve_gradients(stability_mode: StabilityMode) -> None:
    first = np.asarray([[0.2, -0.3, 0.7], [1.1, -0.4, 0.5]])
    second = np.asarray([[-0.2, 0.8, 0.1]])
    instructions = [
        RichInstruction(OperatorExp(), (0,)),  # ssa id 2
        RichInstruction(OperatorExp(), (1,)),  # ssa id 3
        RichInstruction(OperatorConcat(axis=0), (2, 3)),  # ssa id 4
        RichInstruction(OperatorEinsum("ij->j"), (4,)),  # ssa id 5
        RichInstruction(OperatorLog(), (5,)),  # ssa id 6
        RichInstruction(OperatorEinsum("j->"), (6,)),  # ssa id 7
    ]
    shapes = [first.shape, second.shape, first.shape, second.shape, (3, 3), (3,), (3,), ()]

    def value_and_grad(mode: StabilityMode) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (first, second)]
        rich_program = _dense_program(instructions, n_inputs=2, stability_mode=mode, shapes=shapes)
        result = run_program(translate_to_backend_program(rich_program, TorchBackendFunctions()), inputs)
        return result, torch.autograd.grad(result, inputs)

    baseline_result, baseline_gradients = value_and_grad("unstable")
    result, gradients = value_and_grad(stability_mode)

    torch.testing.assert_close(result, baseline_result)
    for gradient, baseline_gradient in zip(gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(gradient, baseline_gradient)


@pytest.mark.parametrize("stability_mode", LOGSPACE_MODES)
@pytest.mark.parametrize("fused_tucker", [False, True])
def test_logspace_detached_shifts_preserve_tucker_gradients(stability_mode: StabilityMode, fused_tucker: bool) -> None:
    left = np.abs(_RNG.standard_normal((2, 3))) + 0.5
    right = np.abs(_RNG.standard_normal((2, 4))) + 0.5
    weights = np.abs(_RNG.standard_normal((5, 3, 4))) + 0.5
    if fused_tucker:
        instructions = [
            RichInstruction(OperatorEinsum("bi,bj,kij->bk"), (0, 1, 2)),
            RichInstruction(OperatorEinsum("bk->"), (3,)),
            RichInstruction(OperatorLog(), (4,)),
        ]
        shapes = [left.shape, right.shape, weights.shape, (2, 5), (), ()]
    else:
        instructions = [
            RichInstruction(OperatorEinsum("bi,bj->bij"), (0, 1)),
            RichInstruction(OperatorEinsum("bij,kij->bk"), (3, 2)),
            RichInstruction(OperatorEinsum("bk->"), (4,)),
            RichInstruction(OperatorLog(), (5,)),
        ]
        shapes = [left.shape, right.shape, weights.shape, (2, 3, 4), (2, 5), (), ()]

    def value_and_grad(mode: StabilityMode) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (left, right, weights)]
        rich_program = _dense_program(
            instructions,
            n_inputs=3,
            stability_mode=mode,
            parameter_indices=frozenset({2}),
            shapes=shapes,
        )
        result = run_program(translate_to_backend_program(rich_program, TorchBackendFunctions()), inputs)
        return result, torch.autograd.grad(result, inputs)

    baseline_result, baseline_gradients = value_and_grad("unstable")
    result, gradients = value_and_grad(stability_mode)

    torch.testing.assert_close(result, baseline_result)
    for gradient, baseline_gradient in zip(gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(gradient, baseline_gradient)


def test_unstable_mode_overflows_on_huge_intermediate_values() -> None:
    rich_program = _huge_chain_program("unstable", N_HUGE_FACTORS)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    with np.errstate(over="ignore", invalid="ignore"):
        result = run_program(backend_program, HUGE_CHAIN_FACTORS)

    assert not np.any(np.isfinite(result))
