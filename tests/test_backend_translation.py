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


def _dense_program(instructions: list[RichInstruction], n_inputs: int, stability_mode: StabilityMode = "unstable") -> RichProgram:
    n_ssa_ids = n_inputs + len(instructions)
    return RichProgram(
        instructions=instructions,
        n_inputs=n_inputs,
        stability_mode=stability_mode,
        tensor_formats=["dense"] * n_ssa_ids,
        shapes=[()] * n_ssa_ids,
        parameter_indices=frozenset(),
    )


def _single_instruction_program(operator: RichOperator, n_arguments: int, stability_mode: StabilityMode) -> RichProgram:
    return _dense_program([RichInstruction(operator, tuple(range(n_arguments)))], n_inputs=n_arguments, stability_mode=stability_mode)


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


@pytest.mark.parametrize("stability_mode", ["unstable", "scaled"])
def test_translation_rejects_operator_without_backend_function(stability_mode: StabilityMode) -> None:
    rich_program = _single_instruction_program(OperatorSin(), 1, stability_mode)

    with pytest.raises(NotImplementedError, match="sin"):
        translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)


################################
# translation of single instructions (scaled)
################################

# single-instruction programs on positive inputs, compared against the actual result.
# the values are positive because the scaled translation normalizes raw tensors by their total sum
SCALED_TRANSLATION_CASES: list[tuple[RichOperator, list[npt.NDArray], npt.NDArray]] = [
    (OperatorExp(), [MATRIX], np.exp(MATRIX)),
    (OperatorLog(), [POSITIVE_MATRIX], np.log(POSITIVE_MATRIX)),
    (OperatorAdd(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX + OTHER_POSITIVE_MATRIX),
    (OperatorSubtract(), [POSITIVE_MATRIX + OTHER_POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX),
    (OperatorMultiply(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX),
    (OperatorDivide(), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], POSITIVE_MATRIX / OTHER_POSITIVE_MATRIX),
    (OperatorStack(axis=1), [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], np.stack([POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX], axis=1)),
    (OperatorTake(axis=1), [POSITIVE_MATRIX, INDICES], POSITIVE_MATRIX[:, INDICES]),
    (OperatorSelect(axis=0, index=1), [POSITIVE_MATRIX], POSITIVE_MATRIX[1]),
    (OperatorSlice(start=1, stop=3, axis=1), [POSITIVE_MATRIX], POSITIVE_MATRIX[:, 1:3]),
    (OperatorSoftmax(axis=1), [MATRIX], np.exp(MATRIX) / np.sum(np.exp(MATRIX), axis=1, keepdims=True)),
    (OperatorEinsum("ik, kj -> ij"), [POSITIVE_MATRIX, POSITIVE_RIGHT_MATRIX], POSITIVE_MATRIX @ POSITIVE_RIGHT_MATRIX),
    (OperatorEinsum("ik ->"), [POSITIVE_MATRIX], np.sum(POSITIVE_MATRIX)),
]
SCALED_TRANSLATION_CASE_IDS = [f"{operator.name}-{case_index}" for case_index, (operator, _, _) in enumerate(SCALED_TRANSLATION_CASES)]


@pytest.mark.parametrize(("operator", "tensor_arguments", "expected_result"), SCALED_TRANSLATION_CASES, ids=SCALED_TRANSLATION_CASE_IDS)
def test_scaled_translation_computes_actual_result(operator: RichOperator, tensor_arguments: list[npt.NDArray], expected_result: npt.NDArray) -> None:
    rich_program = _single_instruction_program(operator, len(tensor_arguments), "scaled")

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, tensor_arguments)

    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def test_scaled_translation_leaves_raw_only_values_unscaled() -> None:
    # softmax consumes its argument raw and produces a raw result, so the input must not be scaled at all
    rich_program = _single_instruction_program(OperatorSoftmax(axis=1), 1, "scaled")

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)

    assert len(backend_program.backend_calls) == 1
    assert backend_program.call_arguments == [(0,)]


def test_scaled_translation_scales_each_value_at_most_once() -> None:
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorMultiply(), (0, 1)),  # ssa id 2
            RichInstruction(OperatorMultiply(), (2, 0)),  # ssa id 3, consumes input 0 as a scaled pair a second time
        ],
        n_inputs=2,
        stability_mode="scaled",
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [POSITIVE_MATRIX, OTHER_POSITIVE_MATRIX])

    # scaling the two inputs (3 calls each), two multiplications (2 calls each), and the raw conversion of the output (2 calls)
    assert len(backend_program.backend_calls) == 12
    np.testing.assert_allclose(result, POSITIVE_MATRIX * OTHER_POSITIVE_MATRIX * POSITIVE_MATRIX, rtol=1e-9)


def test_scaled_translation_of_exp_and_log_survives_overflowing_exponentials() -> None:
    # exp moves the maximum of its raw argument into the log scale and log adds it back, so the roundtrip survives exponentials that overflow raw tensors
    huge_matrix = 800.0 + MATRIX
    rich_program = _dense_program(
        instructions=[
            RichInstruction(OperatorExp(), (0,)),  # ssa id 1
            RichInstruction(OperatorLog(), (1,)),  # ssa id 2
        ],
        n_inputs=1,
        stability_mode="scaled",
    )

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, [huge_matrix])

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, huge_matrix, rtol=1e-9)


def test_logspace_translation_is_not_implemented_yet() -> None:
    rich_program = _single_instruction_program(OperatorExp(), 1, "logspace")

    with pytest.raises(NotImplementedError, match="logspace"):
        translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)


################################
# full translation pipeline
################################


@pytest.mark.parametrize("stability_mode", ["unstable", "scaled"])
def test_translate_to_backend_program_end_to_end(stability_mode: StabilityMode) -> None:
    left_matrix = np.abs(_RNG.standard_normal((3, 4))) + 0.5
    right_matrix = np.abs(_RNG.standard_normal((4, 5))) + 0.5
    bias_matrix = np.abs(_RNG.standard_normal((3, 5))) + 0.5
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


@pytest.mark.parametrize("stability_mode", ["unstable", "scaled"])
def test_translate_to_backend_program_with_mixed_representations(stability_mode: StabilityMode) -> None:
    # in scaled mode this program mixes representations: the einsum introduces a scale, the multiplication
    # and addition combine it with raw inputs, and the logarithm eliminates it again
    left_matrix = np.abs(_RNG.standard_normal((3, 4))) + 0.5
    right_matrix = np.abs(_RNG.standard_normal((4, 5))) + 0.5
    factor_matrix = np.abs(_RNG.standard_normal((3, 5))) + 0.5
    summand_matrix = np.abs(_RNG.standard_normal((3, 5))) + 0.5
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


def test_scaled_mode_survives_huge_intermediate_values() -> None:
    rich_program = _huge_chain_program("scaled", N_HUGE_FACTORS)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    result = run_program(backend_program, HUGE_CHAIN_FACTORS)

    # the huge scalar factors cancel out in the normalization, so the expected result is computable from the raw factors
    raw_chain_product = RAW_CHAIN_FACTORS[0]
    for raw_factor in RAW_CHAIN_FACTORS[1:]:
        raw_chain_product = raw_chain_product @ raw_factor
    expected_result = raw_chain_product / np.sum(raw_chain_product)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected_result, rtol=1e-9)


def test_unstable_mode_overflows_on_huge_intermediate_values() -> None:
    rich_program = _huge_chain_program("unstable", N_HUGE_FACTORS)

    backend_program = translate_to_backend_program(rich_program, BACKEND_FUNCTIONS)
    with np.errstate(over="ignore", invalid="ignore"):
        result = run_program(backend_program, HUGE_CHAIN_FACTORS)

    assert not np.any(np.isfinite(result))
