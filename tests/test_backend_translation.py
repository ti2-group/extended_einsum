import numpy as np
import numpy.typing as npt
import pytest

from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backend_translation.translate import translate_to_backend_program
from extended_einsum.backends.numpy import NumpyBackendFunctions
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import (
    OperatorAdd,
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
from extended_einsum.language.types import StabilityMode

BACKEND_FUNCTIONS = NumpyBackendFunctions()


def _dense_program(
    instructions: list[RichInstruction],
    n_inputs: int,
    stability_mode: StabilityMode = "unstable",
    parameter_indices: frozenset[int] = frozenset(),
) -> RichProgram:
    n_ssa_ids = n_inputs + len(instructions)
    return RichProgram(
        instructions=instructions,
        n_inputs=n_inputs,
        stability_mode=stability_mode,
        tensor_formats=["dense"] * n_ssa_ids,
        shapes=[()] * n_ssa_ids,
        parameter_indices=parameter_indices,
    )


def _single_instruction_program(operator: RichOperator, n_arguments: int, stability_mode: StabilityMode) -> RichProgram:
    return _dense_program([RichInstruction(operator, tuple(range(n_arguments)))], n_inputs=n_arguments, stability_mode=stability_mode)


# the scaled and logspace translations each come in two normalizing variants
SCALED_MODES: list[StabilityMode] = ["scaled_min", "scaled_sum"]
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

    # scaling the two inputs (3 calls each), two multiplications (2 calls each), and the raw conversion of the output (2 calls)
    assert len(backend_program.backend_calls) == 12
    np.testing.assert_allclose(result, POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX * POSITIVE_MATRIX, rtol=1e-9)


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
    # softmax, the shifted data operand, the einsum, and restoring the row shift require nine calls;
    # converting and shifting the softmax result as another log-space operand would require five more.
    assert len(backend_program.backend_calls) == 9
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


def test_unstable_mode_overflows_on_huge_intermediate_values() -> None:
    rich_program = _huge_chain_program("unstable", N_HUGE_FACTORS)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    with np.errstate(over="ignore", invalid="ignore"):
        result = run_program(backend_program, HUGE_CHAIN_FACTORS)

    assert not np.any(np.isfinite(result))
