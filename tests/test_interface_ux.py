"""Regression tests for the interface ergonomics: raw backend arrays as
operands, numpy-compatible semantics, eager validation, and actionable error
messages."""

import numpy as np
import pytest

import extended_einsum as xe


@pytest.fixture
def matrix() -> np.ndarray:
    return np.arange(6.0).reshape(2, 3)


class TestLeafOperators:
    def test_adding_two_leaves_builds_an_expression(self, matrix: np.ndarray) -> None:
        result = (xe.TensorLeaf(matrix) + xe.TensorLeaf(matrix)).materialize()

        np.testing.assert_allclose(result, matrix + matrix)

    def test_all_arithmetic_operators_work_on_leaves(self, matrix: np.ndarray) -> None:
        left = xe.TensorLeaf(matrix + 1.0)
        right = xe.TensorLeaf(np.ones_like(matrix))

        np.testing.assert_allclose((left - right).materialize(), matrix)
        np.testing.assert_allclose((left * right).materialize(), matrix + 1.0)
        np.testing.assert_allclose((left / right).materialize(), matrix + 1.0)

    def test_matmul_on_leaves(self, matrix: np.ndarray) -> None:
        other = np.ones((3, 2))

        result = (xe.TensorLeaf(matrix) @ xe.TensorLeaf(other)).materialize()

        np.testing.assert_allclose(result, matrix @ other)

    def test_leaf_materialize_returns_the_raw_array(self, matrix: np.ndarray) -> None:
        leaf = xe.TensorLeaf(matrix)

        assert leaf.materialize() is matrix

    def test_raw_array_as_right_operand_becomes_a_leaf_automatically(self, matrix: np.ndarray) -> None:
        result = (xe.TensorLeaf(matrix) + np.ones_like(matrix)).materialize()

        np.testing.assert_allclose(result, matrix + 1.0)

    def test_raw_array_as_left_operand_becomes_a_leaf_automatically(self, matrix: np.ndarray) -> None:
        result = (matrix + xe.exp(xe.TensorLeaf(matrix))).materialize()

        np.testing.assert_allclose(result, matrix + np.exp(matrix))

    def test_raw_torch_tensor_as_left_operand_becomes_a_leaf_automatically(self) -> None:
        torch = pytest.importorskip("torch")

        result = (torch.ones(2, 3) * xe.exp(torch.ones(2, 3))).materialize()

        torch.testing.assert_close(result, torch.exp(torch.ones(2, 3)))


class TestMatmulSemantics:
    def test_vector_matrix_combinations_match_numpy(self) -> None:
        vector = np.arange(3.0)
        matrix = np.arange(6.0).reshape(3, 2)

        np.testing.assert_allclose((xe.TensorLeaf(vector) @ matrix).materialize(), vector @ matrix)
        np.testing.assert_allclose((xe.TensorLeaf(matrix.T) @ vector).materialize(), matrix.T @ vector)
        np.testing.assert_allclose((xe.TensorLeaf(vector) @ vector).materialize(), vector @ vector)

    def test_higher_dimensional_matmul_points_to_einsum(self) -> None:
        cube = xe.TensorLeaf(np.ones((2, 2, 2)))

        with pytest.raises(ValueError, match="einsum"):
            cube @ cube

    def test_expression_matmul_with_raw_operands_on_both_sides(self) -> None:
        matrix = np.ones((2, 2))

        np.testing.assert_allclose((xe.exp(matrix) @ np.eye(2)).materialize(), np.exp(matrix))
        np.testing.assert_allclose((np.eye(2) @ xe.exp(matrix)).materialize(), np.exp(matrix))


class TestIndexing:
    def test_integer_slice_and_tuple_indexing_match_numpy(self, matrix: np.ndarray) -> None:
        leaf = xe.TensorLeaf(matrix)

        np.testing.assert_allclose(leaf[0].materialize(), matrix[0])
        np.testing.assert_allclose(leaf[-1].materialize(), matrix[-1])
        np.testing.assert_allclose(leaf[:, 1].materialize(), matrix[:, 1])
        np.testing.assert_allclose(leaf[0:2, 1:].materialize(), matrix[0:2, 1:])

    def test_indexing_works_on_expressions(self, matrix: np.ndarray) -> None:
        expression = xe.exp(matrix)

        np.testing.assert_allclose(expression[1].materialize(), np.exp(matrix)[1])

    def test_out_of_bounds_index_raises_index_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(IndexError, match="out of bounds"):
            xe.TensorLeaf(matrix)[5]

    def test_step_slicing_is_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="step"):
            xe.TensorLeaf(matrix)[::2]

    def test_chained_integer_and_slice_indexing_matches_numpy(self) -> None:
        cube = np.arange(24.0).reshape(2, 3, 4)
        leaf = xe.TensorLeaf(cube)

        np.testing.assert_allclose(leaf[1, 0:2].materialize(), cube[1, 0:2])
        np.testing.assert_allclose(leaf[:, 1, 2].materialize(), cube[:, 1, 2])
        np.testing.assert_allclose(leaf[0, :, -1].materialize(), cube[0, :, -1])

    def test_empty_tuple_indexing_is_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="empty tuple"):
            xe.TensorLeaf(matrix)[()]

    def test_too_many_indices_raise_index_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(IndexError, match="too many indices"):
            xe.TensorLeaf(matrix)[0, 0, 0]

    def test_unsupported_index_types_raise_type_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="integers, slices"):
            xe.TensorLeaf(matrix)["row"]  # pyright: ignore[reportArgumentType]


class TestStackSemantics:
    def test_negative_axis_matches_numpy(self, matrix: np.ndarray) -> None:
        result = xe.stack([matrix, matrix], axis=-1).materialize()

        np.testing.assert_allclose(result, np.stack([matrix, matrix], axis=-1))

    def test_axis_equal_to_rank_matches_numpy(self, matrix: np.ndarray) -> None:
        result = xe.stack([matrix, matrix], axis=2).materialize()

        np.testing.assert_allclose(result, np.stack([matrix, matrix], axis=2))

    def test_empty_operand_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            xe.stack([])

    def test_shape_mismatch_reports_the_shapes(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match=r"same shape.*\(2, 3\).*\(3, 2\)"):
            xe.stack([matrix, matrix.T])


class TestArgumentCoercion:
    def test_raw_backend_arrays_become_leaves_automatically(self, matrix: np.ndarray) -> None:
        result = xe.einsum("ij,jk->ik", matrix, np.ones((3, 2))).materialize()

        np.testing.assert_allclose(result, matrix @ np.ones((3, 2)))

    def test_reused_raw_arrays_become_a_single_program_input(self, matrix: np.ndarray) -> None:
        expression = xe.einsum("ij,ij->", matrix, matrix)

        program, inputs = xe.extract_program(expression, stability_mode="unstable")

        assert program.n_inputs == 1
        assert len(inputs) == 1
        assert inputs[0] is matrix

    def test_python_scalars_raise_an_actionable_type_error(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="0-d backend array"):
            xe.exp(matrix) + 2.0

    def test_python_lists_raise_an_actionable_type_error(self) -> None:
        with pytest.raises(TypeError, match="backend array"):
            xe.einsum("ij->i", [[1.0, 2.0]])

    def test_raw_torch_tensors_become_leaves_automatically(self) -> None:
        torch = pytest.importorskip("torch")

        result = xe.einsum("ij,jk->ik", torch.ones(2, 3), torch.ones(3, 2))

        assert result.backend == "torch"
        torch.testing.assert_close(result.materialize(), torch.ones(2, 3) @ torch.ones(3, 2))

    def test_numpy_scalars_are_rejected_like_python_scalars(self, matrix: np.ndarray) -> None:
        # numpy scalars subclass the Python number types, so they hit the same guard
        with pytest.raises(TypeError, match="0-d backend array"):
            xe.exp(matrix) + np.float64(2.0)

    def test_bool_scalars_are_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="scalars"):
            xe.exp(matrix) + True


class TestEagerValidation:
    def test_unknown_stability_mode_lists_the_valid_modes(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="scaled_max"):
            xe.exp(matrix).materialize("stable")

    def test_unknown_backend_name_fails_at_leaf_construction(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="No backend is registered"):
            xe.TensorLeaf(matrix, backend="nunpy")

    def test_softmax_requires_an_explicit_axis(self, matrix: np.ndarray) -> None:
        with pytest.raises(TypeError, match="axis"):
            xe.softmax(matrix)  # pyright: ignore[reportCallIssue]

    def test_valid_explicit_backend_name_is_accepted(self, matrix: np.ndarray) -> None:
        result = xe.exp(xe.TensorLeaf(matrix, backend="numpy")).materialize()

        np.testing.assert_allclose(result, np.exp(matrix))

    def test_materialize_stores_no_compilation_state_on_the_expression(self, matrix: np.ndarray) -> None:
        expression = xe.exp(matrix)
        attributes_before = set(vars(expression))

        expression.materialize()

        assert set(vars(expression)) == attributes_before

    def test_materialize_is_repeatable_with_different_stability_modes(self, matrix: np.ndarray) -> None:
        expression = xe.exp(matrix)

        first = expression.materialize("unstable")
        second = expression.materialize("logspace_max")

        np.testing.assert_allclose(second, first)


class TestErrorMessages:
    def test_missing_arrow_explains_the_required_form(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="->"):
            xe.einsum("ij,jk", matrix, matrix.T)

    def test_operand_count_mismatch_names_terms_and_operands(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="2 input terms, but 1 operand was given"):
            xe.einsum("ij,jk->ik", matrix)

    def test_unknown_output_symbols_are_named(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="z"):
            xe.einsum("ij,jk->iz", matrix, matrix.T)

    def test_axis_size_conflicts_raise_value_error_naming_both_operands(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="operand 1 .* operand 0"):
            xe.einsum("ij,jk->ik", matrix, matrix)

    def test_multiple_arrows_are_rejected(self, matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match='more than one "->"'):
            xe.einsum("ij->j->", matrix)

    def test_whitespace_in_format_strings_is_still_tolerated(self, matrix: np.ndarray) -> None:
        result = xe.einsum("ik, kj -> ij", matrix, matrix.T).materialize()

        np.testing.assert_allclose(result, matrix @ matrix.T)

    def test_mixed_backends_name_both_backends(self, matrix: np.ndarray) -> None:
        torch = pytest.importorskip("torch")

        with pytest.raises(ValueError, match="different backends: numpy and torch"):
            xe.exp(matrix) + torch.ones(2, 3)

    def test_take_scalar_source_and_index_errors_say_non_scalar(self, matrix: np.ndarray) -> None:
        scalar = np.array(1.0)
        index = np.array([0, 1])

        with pytest.raises(ValueError, match="non-scalar source"):
            xe.take(scalar, index)
        with pytest.raises(ValueError, match="index vector"):
            xe.take(matrix, scalar)


class TestSoftmaxSemantics:
    def test_negative_axis_is_normalized(self, matrix: np.ndarray) -> None:
        result = xe.softmax(matrix, axis=-1).materialize()

        expected = np.exp(matrix) / np.exp(matrix).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected)

    def test_tuple_axes_normalize_over_all_named_axes(self, matrix: np.ndarray) -> None:
        result = xe.softmax(matrix, axis=(0, 1)).materialize()

        np.testing.assert_allclose(result.sum(), 1.0)


class TestReprs:
    def test_expression_repr_shows_operator_shape_and_backend(self, matrix: np.ndarray) -> None:
        representation = repr(xe.exp(matrix))

        assert "exp" in representation
        assert "(2, 3)" in representation
        assert "numpy" in representation

    def test_leaf_repr_has_no_private_field_names(self, matrix: np.ndarray) -> None:
        representation = repr(xe.TensorLeaf(matrix))

        assert "_backend" not in representation
        assert "numpy" in representation


class TestParameters:
    def test_parameter_leaves_are_recorded_in_the_program(self, matrix: np.ndarray) -> None:
        weights = xe.TensorLeaf(np.ones((3, 2)), is_parameter=True)
        expression = xe.einsum("ij,jk->ik", matrix, weights)

        program, inputs = xe.extract_program(expression, stability_mode="unstable")

        assert program.n_inputs == 2
        assert program.parameter_indices == frozenset({1})
        assert inputs[1] is weights.array


class TestPreprocessingPipeline:
    def test_fold_and_optimize_are_public_and_executable(self, matrix: np.ndarray) -> None:
        expression = xe.einsum("ik,kj->ij", xe.exp(matrix), np.ones((3, 2)))
        program, inputs = xe.extract_program(expression, stability_mode="unstable")

        program = xe.FoldSameShapedOperations.apply(program)
        program = xe.OptimizeContractionPaths.apply(program)
        backend_program = xe.translate_to_backend_program(program, xe.get_backend_functions("numpy"))
        result = xe.run_program(backend_program, inputs)

        np.testing.assert_allclose(result, np.exp(matrix) @ np.ones((3, 2)))
