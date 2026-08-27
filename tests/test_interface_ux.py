"""Regression tests for the interface ergonomics: operators on wrapped arrays,
numpy-compatible semantics, eager validation, and actionable error messages."""

import numpy as np
import pytest

import extended_einsum as xe


@pytest.fixture
def matrix() -> np.ndarray:
    return np.arange(6.0).reshape(2, 3)


class TestWrappedArrayOperators:
    def test_adding_two_wrapped_arrays_builds_an_expression(self, matrix: np.ndarray) -> None:
        result = (xe.array(matrix) + xe.array(matrix)).materialize()

        np.testing.assert_allclose(result.backend_array, matrix + matrix)

    def test_all_arithmetic_operators_work_on_wrapped_arrays(self, matrix: np.ndarray) -> None:
        left = xe.array(matrix + 1.0)
        right = xe.array(np.ones_like(matrix))

        np.testing.assert_allclose((left - right).materialize().backend_array, matrix)
        np.testing.assert_allclose((left * right).materialize().backend_array, matrix + 1.0)
        np.testing.assert_allclose((left / right).materialize().backend_array, matrix + 1.0)

    def test_matmul_on_wrapped_arrays(self, matrix: np.ndarray) -> None:
        other = np.ones((3, 2))

        result = (xe.array(matrix) @ xe.array(other)).materialize()

        np.testing.assert_allclose(result.backend_array, matrix @ other)

    def test_wrapped_array_materialize_returns_itself(self, matrix: np.ndarray) -> None:
        wrapped = xe.array(matrix)

        assert wrapped.materialize() is wrapped

    def test_raw_array_as_right_operand_is_wrapped_automatically(self, matrix: np.ndarray) -> None:
        result = (xe.array(matrix) + np.ones_like(matrix)).materialize()

        np.testing.assert_allclose(result.backend_array, matrix + 1.0)


class TestMatmulSemantics:
    def test_vector_matrix_combinations_match_numpy(self) -> None:
        vector = np.arange(3.0)
        matrix = np.arange(6.0).reshape(3, 2)

        np.testing.assert_allclose((xe.array(vector) @ xe.array(matrix)).materialize().backend_array, vector @ matrix)
        np.testing.assert_allclose((xe.array(matrix.T) @ xe.array(vector)).materialize().backend_array, matrix.T @ vector)
        np.testing.assert_allclose((xe.array(vector) @ xe.array(vector)).materialize().backend_array, vector @ vector)

    def test_higher_dimensional_matmul_points_to_einsum(self) -> None:
        cube = xe.array(np.ones((2, 2, 2)))

        with pytest.raises(ValueError, match="einsum"):
            cube @ cube

    def test_expression_matmul_with_raw_right_operand(self) -> None:
        matrix = np.ones((2, 2))

        result = (xe.exp(xe.array(matrix)) @ np.eye(2)).materialize()

        np.testing.assert_allclose(result.backend_array, np.exp(matrix))


class TestIndexing:
    def test_integer_slice_and_tuple_indexing_match_numpy(self, matrix: np.ndarray) -> None:
        wrapped = xe.array(matrix)

        np.testing.assert_allclose(wrapped[0].materialize().backend_array, matrix[0])
        np.testing.assert_allclose(wrapped[-1].materialize().backend_array, matrix[-1])
        np.testing.assert_allclose(wrapped[:, 1].materialize().backend_array, matrix[:, 1])
        np.testing.assert_allclose(wrapped[0:2, 1:].materialize().backend_array, matrix[0:2, 1:])

    def test_indexing_works_on_expressions(self, matrix: np.ndarray) -> None:
        expression = xe.exp(xe.array(matrix))

        np.testing.assert_allclose(expression[1].materialize().backend_array, np.exp(matrix)[1])

    def test_out_of_bounds_index_raises_index_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(IndexError, match="out of bounds"):
            xe.array(matrix)[5]

    def test_step_slicing_is_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="step"):
            xe.array(matrix)[::2]

    def test_chained_integer_and_slice_indexing_matches_numpy(self) -> None:
        cube = np.arange(24.0).reshape(2, 3, 4)
        wrapped = xe.array(cube)

        np.testing.assert_allclose(wrapped[1, 0:2].materialize().backend_array, cube[1, 0:2])
        np.testing.assert_allclose(wrapped[:, 1, 2].materialize().backend_array, cube[:, 1, 2])
        np.testing.assert_allclose(wrapped[0, :, -1].materialize().backend_array, cube[0, :, -1])

    def test_empty_tuple_indexing_is_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="empty tuple"):
            xe.array(matrix)[()]

    def test_too_many_indices_raise_index_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(IndexError, match="too many indices"):
            xe.array(matrix)[0, 0, 0]

    def test_unsupported_index_types_raise_type_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="integers, slices"):
            xe.array(matrix)["row"]  # pyright: ignore[reportArgumentType]


class TestStackSemantics:
    def test_negative_axis_matches_numpy(self, matrix: np.ndarray) -> None:
        result = xe.stack([xe.array(matrix), xe.array(matrix)], axis=-1).materialize()

        np.testing.assert_allclose(result.backend_array, np.stack([matrix, matrix], axis=-1))

    def test_axis_equal_to_rank_matches_numpy(self, matrix: np.ndarray) -> None:
        result = xe.stack([xe.array(matrix), xe.array(matrix)], axis=2).materialize()

        np.testing.assert_allclose(result.backend_array, np.stack([matrix, matrix], axis=2))

    def test_empty_operand_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            xe.stack([])

    def test_shape_mismatch_reports_the_shapes(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match=r"same shape.*\(2, 3\).*\(3, 2\)"):
            xe.stack([xe.array(matrix), xe.array(matrix.T)])


class TestArgumentCoercion:
    def test_raw_backend_arrays_are_wrapped_automatically(self, matrix: np.ndarray) -> None:
        result = xe.einsum("ij,jk->ik", matrix, np.ones((3, 2))).materialize()

        np.testing.assert_allclose(result.backend_array, matrix @ np.ones((3, 2)))

    def test_python_scalars_raise_an_actionable_type_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="extended_einsum.array"):
            xe.exp(xe.array(matrix)) + 2.0

    def test_python_lists_raise_an_actionable_type_error(self) -> None:
        with pytest.raises(TypeError, match="extended_einsum.array"):
            xe.einsum("ij->i", [[1.0, 2.0]])

    def test_raw_torch_tensors_are_wrapped_automatically(self) -> None:
        torch = pytest.importorskip("torch")

        result = xe.einsum("ij,jk->ik", torch.ones(2, 3), torch.ones(3, 2)).materialize()

        assert result.backend == "torch"
        torch.testing.assert_close(result.backend_array, torch.ones(2, 3) @ torch.ones(3, 2))

    def test_numpy_scalars_are_rejected_like_python_scalars(self, matrix: np.ndarray) -> None:
        # numpy scalars subclass the Python number types, so they hit the same guard
        with pytest.raises(TypeError, match="extended_einsum.array"):
            xe.exp(xe.array(matrix)) + np.float64(2.0)

    def test_bool_scalars_are_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="scalars"):
            xe.exp(xe.array(matrix)) + True


class TestEagerValidation:
    def test_unknown_stability_mode_lists_the_valid_modes(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="scaled_max"):
            xe.exp(xe.array(matrix)).materialize("stable")

    def test_unknown_backend_name_fails_at_wrap_time(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="No backend is registered"):
            xe.array(matrix, backend="nunpy")

    def test_softmax_requires_an_explicit_axis(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="axis"):
            xe.softmax(xe.array(matrix))  # pyright: ignore[reportCallIssue]

    def test_valid_explicit_backend_name_is_accepted(self, matrix: np.ndarray) -> None:
        result = xe.exp(xe.array(matrix, backend="numpy")).materialize()

        np.testing.assert_allclose(result.backend_array, np.exp(matrix))

    def test_materialize_stores_no_compilation_state_on_the_expression(self, matrix: np.ndarray) -> None:
        expression = xe.exp(xe.array(matrix))
        attributes_before = set(vars(expression))

        expression.materialize()

        assert set(vars(expression)) == attributes_before

    def test_materialize_is_repeatable_with_different_stability_modes(self, matrix: np.ndarray) -> None:
        expression = xe.exp(xe.array(matrix))

        first = expression.materialize("unstable")
        second = expression.materialize("logspace_max")

        np.testing.assert_allclose(second.backend_array, first.backend_array)


class TestErrorMessages:
    def test_missing_arrow_explains_the_required_form(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="->"):
            xe.einsum("ij,jk", xe.array(matrix), xe.array(matrix.T))

    def test_operand_count_mismatch_names_terms_and_operands(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="2 input terms, but 1 operand was given"):
            xe.einsum("ij,jk->ik", xe.array(matrix))

    def test_unknown_output_symbols_are_named(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="z"):
            xe.einsum("ij,jk->iz", xe.array(matrix), xe.array(matrix.T))

    def test_axis_size_conflicts_raise_value_error_naming_both_operands(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="operand 1 .* operand 0"):
            xe.einsum("ij,jk->ik", xe.array(matrix), xe.array(matrix))

    def test_multiple_arrows_are_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match='more than one "->"'):
            xe.einsum("ij->j->", xe.array(matrix))

    def test_whitespace_in_format_strings_is_still_tolerated(self, matrix: np.ndarray) -> None:
        result = xe.einsum("ik, kj -> ij", xe.array(matrix), xe.array(matrix.T)).materialize()

        np.testing.assert_allclose(result.backend_array, matrix @ matrix.T)

    def test_mixed_backends_name_both_backends(self, matrix: np.ndarray) -> None:
        torch = pytest.importorskip("torch")

        with pytest.raises(ValueError, match="different backends: numpy and torch"):
            xe.exp(xe.array(matrix)) + xe.array(torch.ones(2, 3))

    def test_take_scalar_source_and_index_errors_say_non_scalar(self, matrix: np.ndarray) -> None:
        scalar = xe.array(np.array(1.0))
        index = xe.array(np.array([0, 1]))

        with pytest.raises(ValueError, match="non-scalar source"):
            xe.take(scalar, index)
        with pytest.raises(ValueError, match="index vector"):
            xe.take(xe.array(matrix), scalar)


class TestSoftmaxSemantics:
    def test_negative_axis_is_normalized(self, matrix: np.ndarray) -> None:
        result = xe.softmax(xe.array(matrix), axis=-1).materialize()

        expected = np.exp(matrix) / np.exp(matrix).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.backend_array, expected)

    def test_tuple_axes_normalize_over_all_named_axes(self, matrix: np.ndarray) -> None:
        result = xe.softmax(xe.array(matrix), axis=(0, 1)).materialize()

        np.testing.assert_allclose(result.backend_array.sum(), 1.0)


class TestReprs:
    def test_expression_repr_shows_operator_shape_and_backend(self, matrix: np.ndarray) -> None:
        representation = repr(xe.exp(xe.array(matrix)))

        assert "exp" in representation
        assert "(2, 3)" in representation
        assert "numpy" in representation

    def test_wrapped_array_repr_has_no_private_field_names(self, matrix: np.ndarray) -> None:
        representation = repr(xe.array(matrix))

        assert "_backend" not in representation
        assert "numpy" in representation


class TestPreprocessingPipeline:
    def test_fold_and_optimize_are_public_and_executable(self, matrix: np.ndarray) -> None:
        expression = xe.einsum("ik,kj->ij", xe.exp(xe.array(matrix)), xe.array(np.ones((3, 2))))
        program, inputs = xe.extract_program(expression, stability_mode="unstable")

        program = xe.FoldSameShapedOperations.apply(program)
        program = xe.OptimizeContractionPaths.apply(program)
        backend_program = xe.translate_to_backend_program(program, xe.get_backend_functions("numpy"))
        result = xe.run_program(backend_program, [wrapped.backend_array for wrapped in inputs])

        np.testing.assert_allclose(result, np.exp(matrix) @ np.ones((3, 2)))
