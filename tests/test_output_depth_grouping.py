import unittest
from unittest.mock import patch

import torch

from extended_einsum.backend_translation import run_program, translate_to_backend_program
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import (
    OperatorAdd,
    OperatorConcat,
    OperatorEinsum,
    OperatorExp,
    OperatorLog,
    OperatorSelect,
    OperatorSoftmax,
    OperatorStack,
    OperatorSubtract,
    OperatorTake,
)
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.preprocess import (
    AnnotateShortSameIndexContractions,
    FoldSameShapedOperations,
    OptimizeContractionPaths,
    extract_connected_einsum_components,
    group_identical_ops_by_output_depth,
    to_annotated_ssa_path,
)
from extended_einsum.shapes import Shape


def _add_instruction(first: int, second: int) -> RichInstruction:
    return RichInstruction(OperatorAdd(), (first, second))


def _subtract_instruction(first: int, second: int) -> RichInstruction:
    return RichInstruction(OperatorSubtract(), (first, second))


def _log_instruction(argument: int) -> RichInstruction:
    return RichInstruction(OperatorLog(), (argument,))


def _exp_instruction(argument: int) -> RichInstruction:
    return RichInstruction(OperatorExp(), (argument,))


def _softmax_instruction(argument: int, axis: int | tuple[int, ...]) -> RichInstruction:
    return RichInstruction(OperatorSoftmax(axis), (argument,))


def _select_instruction(argument: int, index: int, axis: int = 0) -> RichInstruction:
    return RichInstruction(OperatorSelect(axis, index), (argument,))


def _take_instruction(argument: int, indices: int, axis: int = 0) -> RichInstruction:
    return RichInstruction(OperatorTake(axis), (argument, indices))


def _stack_instruction(*arguments: int, axis: int = 0) -> RichInstruction:
    return RichInstruction(OperatorStack(axis), arguments)


def _einsum_instruction(format_string: str, *arguments: int) -> RichInstruction:
    return RichInstruction(OperatorEinsum(format_string), arguments)


def _program(
    *,
    instructions: list[RichInstruction],
    n_inputs: int,
    shapes: list[Shape],
    parameter_indices: frozenset[int] = frozenset(),
) -> RichProgram:
    return RichProgram(
        instructions=instructions,
        n_inputs=n_inputs,
        stability_mode="unstable",
        shapes=shapes,
        tensor_formats=["dense" for _ in shapes],
        parameter_indices=parameter_indices,
    )


class ConcatOperatorTests(unittest.TestCase):
    def test_concat_propagates_axis_shape(self) -> None:
        operator = OperatorConcat(axis=0)

        self.assertEqual(operator.propagate_shapes([(2, 3), (4, 3)]), (6, 3))

    def test_concat_runs_on_torch_backend(self) -> None:
        program = RichProgram(
            instructions=[RichInstruction(OperatorConcat(axis=0), (0, 1))],
            n_inputs=2,
            stability_mode="unstable",
            shapes=[(2, 3), (1, 3), (3, 3)],
            tensor_formats=["dense", "dense", "dense"],
            parameter_indices=frozenset(),
        )
        first = torch.ones((2, 3))
        second = torch.zeros((1, 3))
        backend_program = translate_to_backend_program(program, TorchBackendFunctions())
        result = run_program(backend_program, [first, second])

        torch.testing.assert_close(
            result,
            torch.cat([first, second], dim=0),
        )


class OutputDepthOpGroupingTests(unittest.TestCase):
    def test_groups_identical_non_einsum_ops_at_same_output_depth(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _add_instruction(2, 3),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].depth, -1)
        self.assertEqual(groups[0].operator.name, "log")
        self.assertEqual(groups[0].operator.raw_extra_arguments, ())
        self.assertEqual(
            tuple(member.op_index for member in groups[0].members),
            (0, 1),
        )

    def test_keeps_operator_specific_arguments_in_the_group_signature(self) -> None:
        program = _program(
            instructions=[
                _softmax_instruction(0, axis=0),
                _softmax_instruction(1, axis=1),
                _add_instruction(2, 3),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(groups, ())

    def test_ignores_non_batchable_ops(self) -> None:
        program = _program(
            instructions=[
                _take_instruction(0, 1),
                _take_instruction(2, 3),
                _add_instruction(4, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (1,),
                (2, 3),
                (1,),
                (1, 3),
                (1, 3),
                (1, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program, min_group_size=1)

        self.assertEqual(
            tuple((group.operator.name, tuple(member.op_index for member in group.members)) for group in groups),
            (("+", (2,)),),
        )

    def test_canonicalizes_commutative_binary_operand_order(self) -> None:
        program = _program(
            instructions=[
                _add_instruction(0, 1),
                _add_instruction(3, 2),
                _add_instruction(4, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (1, 3),
                (2, 3),
                (1, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].operator.name, "+")
        self.assertEqual(groups[0].operand_shapes, ((1, 3), (2, 3)))
        members_by_op = {member.op_index: member for member in groups[0].members}
        self.assertEqual(members_by_op[0].canonical_argument_order, (1, 0))
        self.assertEqual(members_by_op[1].canonical_argument_order, (0, 1))

    def test_keeps_noncommutative_binary_operand_order(self) -> None:
        program = _program(
            instructions=[
                _subtract_instruction(0, 1),
                _subtract_instruction(3, 2),
                _add_instruction(4, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (1, 3),
                (2, 3),
                (1, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(groups, ())

    def test_groups_identical_einsums_at_same_output_depth(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ab,bc->ac", 0, 1),
                _einsum_instruction("ab,bc->ac", 2, 3),
                _add_instruction(4, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (3, 4),
                (2, 3),
                (3, 4),
                (2, 4),
                (2, 4),
                (2, 4),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].depth, 1)
        self.assertEqual(groups[0].operator.name, "einsum")
        self.assertEqual(
            groups[0].operator.raw_extra_arguments,
            ("ac,cb->ab",),
        )
        self.assertEqual(
            tuple(member.op_index for member in groups[0].members),
            (0, 1),
        )

    def test_canonicalizes_renamed_and_reordered_operands(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ab,bc->ac", 0, 1),
                _einsum_instruction("yz,xy->xz", 2, 3),
                _add_instruction(4, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (3, 4),
                (3, 4),
                (2, 3),
                (2, 4),
                (2, 4),
                (2, 4),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].operator.name, "einsum")
        self.assertEqual(
            groups[0].operator.raw_extra_arguments,
            ("ac,cb->ab",),
        )
        self.assertEqual(groups[0].operand_shapes, ((2, 3), (3, 4)))
        members_by_op = {member.op_index: member for member in groups[0].members}
        self.assertEqual(members_by_op[0].canonical_argument_order, (0, 1))
        self.assertEqual(members_by_op[1].canonical_argument_order, (1, 0))

    def test_canonicalizes_three_reordered_operands(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ij,jk,kl->il", 0, 1, 2),
                _einsum_instruction("zw,xy,yz->xw", 3, 4, 5),
                _add_instruction(6, 7),
            ],
            n_inputs=6,
            shapes=[
                (2, 3),
                (3, 4),
                (4, 5),
                (4, 5),
                (2, 3),
                (3, 4),
                (2, 5),
                (2, 5),
                (2, 5),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].operator.name, "einsum")
        self.assertEqual(
            groups[0].operator.raw_extra_arguments,
            ("ac,db,cd->ab",),
        )
        self.assertEqual(groups[0].operand_shapes, ((2, 3), (4, 5), (3, 4)))
        members_by_op = {member.op_index: member for member in groups[0].members}
        self.assertEqual(members_by_op[0].canonical_argument_order, (0, 2, 1))
        self.assertEqual(members_by_op[1].canonical_argument_order, (1, 0, 2))

    def test_does_not_group_identical_einsums_at_different_output_depths(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ab,bc->ac", 0, 1),
                _einsum_instruction("ab,bc->ac", 2, 3),
                _log_instruction(4),
                _add_instruction(6, 5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (3, 4),
                (2, 3),
                (3, 4),
                (2, 4),
                (2, 4),
                (2, 4),
                (2, 4),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(groups, ())

    def test_splits_ops_with_input_and_einsum_operand_sources(self) -> None:
        program = _program(
            instructions=[
                _select_instruction(0, 1),
                _einsum_instruction("ab,bc->ac", 1, 2),
                _add_instruction(5, 3),
                _add_instruction(6, 4),
                _stack_instruction(7, 8),
            ],
            n_inputs=5,
            shapes=[
                (2, 2, 3),
                (2, 3),
                (3, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(groups, ())

    def test_groups_input_pointwise_operations_across_output_depths(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _log_instruction(2),
                _add_instruction(3, 4),
                _add_instruction(6, 5),
            ],
            n_inputs=3,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        groups = group_identical_ops_by_output_depth(program)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].depth, -1)
        self.assertEqual(tuple(member.op_index for member in groups[0].members), (0, 1, 2))

    def test_ignores_unreachable_ops(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ab,bc->ac", 0, 1),
                _einsum_instruction("ab,bc->ac", 2, 3),
                _log_instruction(5),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (3, 4),
                (2, 3),
                (3, 4),
                (2, 4),
                (2, 4),
                (2, 4),
            ],
        )

        groups = group_identical_ops_by_output_depth(program, min_group_size=1)

        self.assertEqual(
            tuple(tuple(member.op_index for member in group.members) for group in groups),
            ((2,), (1,)),
        )

    def test_splits_same_depth_dependent_candidates(self) -> None:
        program = _program(
            instructions=[
                _einsum_instruction("ab,bc->ac", 0, 1),
                _einsum_instruction("ab,bc->ac", 3, 2),
                _add_instruction(3, 4),
            ],
            n_inputs=3,
            shapes=[
                (2, 2),
                (2, 2),
                (2, 2),
                (2, 2),
                (2, 2),
                (2, 2),
            ],
        )

        grouped_pairs = group_identical_ops_by_output_depth(program)
        singleton_groups = group_identical_ops_by_output_depth(
            program,
            min_group_size=1,
        )

        self.assertEqual(grouped_pairs, ())
        self.assertEqual(len(singleton_groups), 3)
        self.assertEqual(
            tuple(tuple(member.op_index for member in group.members) for group in singleton_groups),
            ((2,), (0,), (1,)),
        )

    def test_canonicalizes_when_unique_labels_exceed_ascii_capacity(self) -> None:
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0"
        program = _program(
            instructions=[
                _einsum_instruction(f"{labels}->a", 0),
            ],
            n_inputs=1,
            shapes=[
                (1,) * len(labels),
                (1,),
            ],
        )

        groups = group_identical_ops_by_output_depth(program, min_group_size=1)

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].operator.raw_extra_arguments,
            ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\u0100->a",),
        )

    def test_splits_candidates_that_reuse_the_same_previous_result(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _add_instruction(2, 1),
                _add_instruction(2, 1),
                _stack_instruction(3, 4),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 2, 3),
            ],
        )

        grouped_pairs = group_identical_ops_by_output_depth(program)
        singleton_groups = group_identical_ops_by_output_depth(program, min_group_size=1)

        self.assertEqual(grouped_pairs, ())
        self.assertEqual(
            tuple(tuple(member.op_index for member in group.members) for group in singleton_groups),
            ((3,), (1,), (2,), (0,)),
        )

    def test_folds_identical_unary_operations_into_one_batched_operation(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _add_instruction(2, 3),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = FoldSameShapedOperations.apply(program)

        self.assertEqual(
            [instruction.operator.name for instruction in folded.instructions],
            ["stack", "log", "select", "select", "+"],
        )
        self.assertEqual(result.program, folded)
        self.assertEqual(result.batched_result_orders, ((2, 3),))
        self.assertEqual(result.non_parameter_stack_orders, ((0, 1),))
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0, 1))
        self.assertEqual(folded.instructions[1].argument_ssa_ids, (2,))
        self.assertEqual(folded.shapes, [(2, 3), (2, 3), (2, 2, 3), (2, 2, 3), (2, 3), (2, 3), (2, 3)])

    def test_folding_shifts_every_softmax_axis_past_batch_axis(self) -> None:
        program = _program(
            instructions=[
                _softmax_instruction(0, axis=(1, 2)),
                _softmax_instruction(1, axis=(1, 2)),
                _add_instruction(2, 3),
            ],
            n_inputs=2,
            shapes=[
                (5, 3, 4),
                (5, 3, 4),
                (5, 3, 4),
                (5, 3, 4),
                (5, 3, 4),
            ],
            parameter_indices=frozenset({0, 1}),
        )

        folded = FoldSameShapedOperations.apply(program)

        softmax = next(instruction.operator for instruction in folded.instructions if instruction.operator.name == "softmax")
        self.assertEqual(softmax.axis, (2, 3))

    def test_uses_packed_parameter_input_instead_of_stack_instruction(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _add_instruction(2, 3),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
            parameter_indices=frozenset({0, 1}),
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = result.program

        self.assertEqual(
            [instruction.operator.name for instruction in folded.instructions],
            ["log", "select", "select", "+"],
        )
        self.assertEqual(folded.n_inputs, 1)
        self.assertEqual(folded.parameter_indices, frozenset({0}))
        self.assertEqual(result.parameter_stack_orders, ((0, 1),))
        self.assertEqual(result.non_parameter_stack_orders, ())
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0,))
        self.assertEqual(folded.shapes, [(2, 2, 3), (2, 2, 3), (2, 3), (2, 3), (2, 3)])

    def test_reuses_packed_parameter_input_for_later_scalar_parameter_use(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _add_instruction(2, 0),
                _add_instruction(4, 3),
            ],
            n_inputs=2,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
            parameter_indices=frozenset({0, 1}),
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = result.program

        self.assertEqual(folded.n_inputs, 1)
        self.assertEqual(folded.parameter_indices, frozenset({0}))
        self.assertEqual(result.parameter_stack_orders, ((0, 1),))
        self.assertNotIn("stack", [instruction.operator.name for instruction in folded.instructions])
        self.assertTrue(any(instruction.operator.name == "select" and instruction.argument_ssa_ids == (0,) for instruction in folded.instructions))

    def test_schedules_group_after_ungrouped_producers(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _log_instruction(4),
                _add_instruction(3, 5),
            ],
            n_inputs=3,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        folded = FoldSameShapedOperations.apply(program)

        self.assertEqual(
            [instruction.operator.name for instruction in folded.instructions],
            ["stack", "log", "select", "log", "select", "+"],
        )
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0, 1))
        self.assertEqual(folded.instructions[1].argument_ssa_ids, (2,))

    def test_splits_producer_batches_for_partial_stack_consumer(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _exp_instruction(2),
                _exp_instruction(3),
                _add_instruction(8, 4),
                _add_instruction(9, 5),
                _add_instruction(10, 6),
                _add_instruction(11, 7),
                _stack_instruction(12, 13, 14, 15),
            ],
            n_inputs=8,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (4, 2, 3),
            ],
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = result.program

        self.assertEqual(
            [instruction.operator.name for instruction in folded.instructions],
            [
                "stack",
                "log",
                "stack",
                "exp",
                "select",
                "+",
                "select",
                "+",
                "select",
                "+",
                "select",
                "+",
                "stack",
            ],
        )
        self.assertEqual(result.concatenated_batch_orders, ())
        self.assertNotIn("concat", [instruction.operator.name for instruction in folded.instructions])
        self.assertEqual(folded.shapes[-1], (4, 2, 3))

    def test_reuses_identical_slice_segments_for_multiple_operands(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _log_instruction(2),
                _log_instruction(3),
                _add_instruction(4, 4),
                _add_instruction(5, 5),
                _stack_instruction(8, 9, 6, 7),
            ],
            n_inputs=4,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 2, 3),
            ],
        )

        folded = FoldSameShapedOperations.apply(program)
        add_instruction = next(instruction for instruction in folded.instructions if instruction.operator.name == "+")

        self.assertNotIn("concat", [instruction.operator.name for instruction in folded.instructions])
        self.assertNotIn("slice", [instruction.operator.name for instruction in folded.instructions])
        self.assertEqual(add_instruction.argument_ssa_ids[0], add_instruction.argument_ssa_ids[1])

    def test_orders_producer_batches_so_consumers_can_use_slices(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _log_instruction(1),
                _log_instruction(2),
                _add_instruction(7, 3),
                _add_instruction(8, 4),
                _subtract_instruction(6, 5),
                _stack_instruction(9, 10, 11),
            ],
            n_inputs=6,
            shapes=[
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (3, 2, 3),
            ],
        )

        folded = FoldSameShapedOperations.apply(program)

        self.assertNotIn("concat", [instruction.operator.name for instruction in folded.instructions])
        self.assertNotIn("slice", [instruction.operator.name for instruction in folded.instructions])
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0, 1, 2))

    def test_replaces_stacked_ordered_selects_with_slice(self) -> None:
        program = _program(
            instructions=[
                _select_instruction(0, 1),
                _select_instruction(0, 2),
                _log_instruction(1),
                _log_instruction(2),
                _add_instruction(3, 4),
            ],
            n_inputs=1,
            shapes=[
                (3, 2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
            ],
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = result.program

        self.assertEqual(
            [instruction.operator.name for instruction in folded.instructions],
            ["slice", "log", "select", "select", "+"],
        )
        self.assertEqual(result.input_axis0_orders, {0: (1, 2, 0)})
        self.assertEqual(folded.instructions[0].operator.raw_extra_arguments, (0, 2, 0))
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0,))

    def test_orders_direct_select_inputs_before_future_consumers(self) -> None:
        program = _program(
            instructions=[
                _select_instruction(0, 0),
                _select_instruction(0, 1),
                _select_instruction(0, 2),
                _select_instruction(0, 3),
                _log_instruction(1),
                _log_instruction(2),
                _log_instruction(3),
                _log_instruction(4),
                _add_instruction(5, 7),
                _add_instruction(6, 8),
                _stack_instruction(9, 10),
            ],
            n_inputs=1,
            shapes=[
                (4, 2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 2, 3),
            ],
        )

        result = FoldSameShapedOperations.apply_with_metadata(program)
        folded = result.program

        self.assertEqual(result.input_axis0_orders, {})
        self.assertEqual(folded.instructions[0].operator.name, "log")
        self.assertEqual(folded.instructions[0].argument_ssa_ids, (0,))
        self.assertNotEqual(folded.instructions[0].operator.name, "stack")

    def test_input_depth_can_use_spatial_input_order_without_future_consumers(
        self,
    ) -> None:
        program = _program(
            instructions=[
                _select_instruction(0, 0),
                _select_instruction(0, 2),
                _select_instruction(0, 1),
                _select_instruction(0, 3),
                _log_instruction(1),
                _log_instruction(2),
                _log_instruction(3),
                _log_instruction(4),
                _stack_instruction(5, 6, 7, 8),
            ],
            n_inputs=1,
            shapes=[
                (4, 2),
                *((2,) for _ in range(8)),
                (4, 2),
            ],
        )

        depth_first = FoldSameShapedOperations.apply_with_input_depth_metadata(
            program,
            optimize_group_order=False,
            order_by_input_access=False,
        )
        input_ordered = (
            FoldSameShapedOperations.apply_with_input_depth_metadata(
                program,
                optimize_group_order=False,
                order_by_input_access=True,
            )
        )

        self.assertEqual(depth_first.batched_result_orders, ((5, 6, 7, 8),))
        self.assertEqual(input_ordered.batched_result_orders, ((5, 7, 6, 8),))
        self.assertEqual(depth_first.input_axis0_orders, {0: (0, 2, 1, 3)})
        self.assertEqual(input_ordered.input_axis0_orders, {})


class EinsumLabelAllocationTests(unittest.TestCase):
    def test_extract_connected_einsum_components_prefers_ascii_before_extended(
        self,
    ) -> None:
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0"
        program = _program(
            instructions=[
                _einsum_instruction(f"{labels}->a", 0),
                _einsum_instruction("a,b->ab", 2, 1),
            ],
            n_inputs=2,
            shapes=[
                (1,) * len(labels),
                (2,),
                (1,),
                (1, 2),
            ],
        )

        components = extract_connected_einsum_components(program)

        self.assertEqual(len(components), 1)
        self.assertTrue(components[0].format_string.endswith("->ab"))
        self.assertIn("Ā", components[0].format_string)

    def test_to_annotated_ssa_path_prefers_clean_ascii_per_expression(self) -> None:
        annotated_path = to_annotated_ssa_path(
            "pq,qr,rs->ps",
            [(0, 1), (2, 3)],
            prefer_ascii=True,
        )

        self.assertEqual(
            annotated_path,
            [
                (0, 1, "ab,bc->ac"),
                (2, 3, "ab,ca->cb"),
            ],
        )

    def test_to_annotated_ssa_path_falls_back_to_extended_labels(self) -> None:
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01"

        annotated_path = to_annotated_ssa_path(
            f"{labels},{labels}->{labels}",
            [(0, 1)],
            prefer_ascii=True,
        )

        self.assertEqual(len(annotated_path), 1)
        self.assertIn("Ā", annotated_path[0][2])

    def test_to_annotated_ssa_path_can_keep_final_output_labels_first(self) -> None:
        annotated_path = to_annotated_ssa_path(
            "foij,fbi,fbj->fbo",
            [(0, 2), (1, 3)],
            prioritize_output_labels=True,
        )

        self.assertEqual(
            annotated_path,
            [
                (0, 2, "foij,fbj->fboi"),
                (1, 3, "fbi,fboi->fbo"),
            ],
        )


class OptimizeContractionPathsTests(unittest.TestCase):
    def test_fuses_stable_outer_product_with_reduction(self) -> None:
        for stability_mode in (
            "scaled_min",
            "scaled_max",
            "scaled_sum",
            "logspace_min",
            "logspace_max",
        ):
            with self.subTest(stability_mode=stability_mode):
                program = RichProgram(
                    instructions=[
                        _einsum_instruction("bi,bj->bij", 0, 1),
                        _einsum_instruction("bij,oij->bo", 3, 2),
                    ],
                    n_inputs=3,
                    stability_mode=stability_mode,
                    shapes=[(5, 3), (5, 4), (2, 3, 4), (5, 3, 4), (5, 2)],
                    tensor_formats=["dense"] * 5,
                    parameter_indices=frozenset({2}),
                )

                optimized = OptimizeContractionPaths.apply(program)
                left = torch.rand((5, 3)) + 0.5
                right = torch.rand((5, 4)) + 0.5
                weights = torch.rand((2, 3, 4)) + 0.5
                backend_program = translate_to_backend_program(optimized, TorchBackendFunctions())
                result = run_program(backend_program, [left, right, weights])

                self.assertEqual(len(optimized.instructions), 1)
                self.assertEqual(len(optimized.instructions[0].argument_ssa_ids), 3)
                torch.testing.assert_close(result, torch.einsum("bi,bj,oij->bo", left, right, weights))

    def test_retries_until_path_does_not_increase_dag_depth(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _einsum_instruction("bc,cd->bd", 1, 2),
                _einsum_instruction("ab,bd->ad", 3, 4),
            ],
            n_inputs=3,
            shapes=[
                (2, 3),
                (3, 4),
                (4, 5),
                (2, 3),
                (3, 5),
                (2, 5),
            ],
        )

        with patch(
            "sesum.sr.compute_path",
            side_effect=[
                ([(0, 1), (3, 2)], 0.0, 0.0),
                ([(1, 2), (0, 3)], 0.0, 0.0),
            ],
        ) as compute_path:
            optimized = OptimizeContractionPaths.apply(program)

        self.assertEqual(compute_path.call_count, 2)
        self.assertEqual(
            [instruction.argument_ssa_ids for instruction in optimized.instructions],
            [(0,), (1, 2), (3, 4)],
        )

    def test_keeps_program_when_all_candidate_paths_increase_dag_depth(self) -> None:
        program = _program(
            instructions=[
                _log_instruction(0),
                _einsum_instruction("bc,cd->bd", 1, 2),
                _einsum_instruction("ab,bd->ad", 3, 4),
            ],
            n_inputs=3,
            shapes=[
                (2, 3),
                (3, 4),
                (4, 5),
                (2, 3),
                (3, 5),
                (2, 5),
            ],
        )

        with patch(
            "sesum.sr.compute_path",
            return_value=([(0, 1), (3, 2)], 0.0, 0.0),
        ) as compute_path:
            optimized = OptimizeContractionPaths.apply(program)

        self.assertEqual(compute_path.call_count, 8)
        self.assertEqual(optimized, program)


class AnnotateShortSameIndexContractionsTests(unittest.TestCase):
    def test_marks_short_weighted_reduction_without_splitting_instruction(self) -> None:
        program = RichProgram(
            instructions=[_einsum_instruction("abc,ac->ab", 0, 1)],
            n_inputs=2,
            stability_mode="scaled_max",
            shapes=[(2, 3, 4), (2, 4), (2, 3)],
            tensor_formats=["dense"] * 3,
            parameter_indices=frozenset({1}),
        )

        annotated = AnnotateShortSameIndexContractions.apply(program)

        self.assertEqual(len(annotated.instructions), 1)
        operator = annotated.instructions[0].operator
        self.assertIsInstance(operator, OperatorEinsum)
        assert isinstance(operator, OperatorEinsum)
        self.assertEqual(operator.short_contraction_labels, ("c",))
        self.assertEqual(operator.raw_extra_arguments, ("abc,ac->ab",))
        self.assertEqual(annotated.shapes, program.shapes)

    def test_does_not_mark_long_weighted_reduction(self) -> None:
        program = RichProgram(
            instructions=[_einsum_instruction("abc,ac->ab", 0, 1)],
            n_inputs=2,
            stability_mode="scaled_max",
            shapes=[(2, 3, 5), (2, 5), (2, 3)],
            tensor_formats=["dense"] * 3,
            parameter_indices=frozenset({1}),
        )

        self.assertIs(AnnotateShortSameIndexContractions.apply(program), program)

    def test_annotated_contraction_preserves_stable_results(self) -> None:
        data = torch.rand((2, 3, 4)) + 0.5
        weights = torch.rand((2, 4)) + 0.5
        expected = torch.einsum("abc,ac->ab", data, weights)

        for stability_mode in ("scaled_max", "logspace_max"):
            with self.subTest(stability_mode=stability_mode):
                program = RichProgram(
                    instructions=[_einsum_instruction("abc,ac->ab", 0, 1)],
                    n_inputs=2,
                    stability_mode=stability_mode,
                    shapes=[(2, 3, 4), (2, 4), (2, 3)],
                    tensor_formats=["dense"] * 3,
                    parameter_indices=frozenset({1}),
                )
                annotated = AnnotateShortSameIndexContractions.apply(program)
                backend_program = translate_to_backend_program(
                    annotated,
                    TorchBackendFunctions(),
                )

                actual = run_program(backend_program, [data, weights])

                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
