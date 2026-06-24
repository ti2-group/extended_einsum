import unittest

from extended_einsum.format import DenseFormat
from extended_einsum.language import Instruction, get_arguments, make_einsum_instruction
from extended_einsum.preprocess import (
    RichProgram,
    group_identical_ops_by_output_depth,
)
from extended_einsum.shapes import Shape


def _add_instruction(first: int, second: int) -> Instruction:
    return ("+", (first, second), ())


def _subtract_instruction(first: int, second: int) -> Instruction:
    return ("-", (first, second), ())


def _log_instruction(argument: int) -> Instruction:
    return ("log", (argument,), ())


def _softmax_instruction(argument: int, axis: int) -> Instruction:
    return ("softmax", (argument,), (axis,))


def _program(
    *,
    instructions: list[Instruction],
    n_inputs: int,
    shapes: list[Shape],
) -> RichProgram:
    consumers = [[] for _ in shapes]
    for op_index, instruction in enumerate(instructions):
        consumer_id = n_inputs + op_index
        for argument in get_arguments(instruction):
            consumers[argument].append(consumer_id)

    return RichProgram(
        instructions=instructions,
        n_inputs=n_inputs,
        stability="none",
        shapes=shapes,
        tensor_formats=[DenseFormat() for _ in shapes],
        parameter_indices=[],
        consumers_of_ssa_id=consumers,
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
        self.assertEqual(groups[0].depth, 1)
        self.assertEqual(groups[0].operator, "log")
        self.assertEqual(groups[0].canonical_instruction_specific_arguments, ())
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
        self.assertEqual(groups[0].operator, "+")
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
                make_einsum_instruction("ab,bc->ac", 0, 1),
                make_einsum_instruction("ab,bc->ac", 2, 3),
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
        self.assertEqual(groups[0].operator, "einsum")
        self.assertEqual(
            groups[0].canonical_instruction_specific_arguments,
            ("ac,cb->ab",),
        )
        self.assertEqual(
            tuple(member.op_index for member in groups[0].members),
            (0, 1),
        )

    def test_canonicalizes_renamed_and_reordered_operands(self) -> None:
        program = _program(
            instructions=[
                make_einsum_instruction("ab,bc->ac", 0, 1),
                make_einsum_instruction("yz,xy->xz", 2, 3),
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
        self.assertEqual(groups[0].operator, "einsum")
        self.assertEqual(
            groups[0].canonical_instruction_specific_arguments,
            ("ac,cb->ab",),
        )
        self.assertEqual(groups[0].operand_shapes, ((2, 3), (3, 4)))
        members_by_op = {member.op_index: member for member in groups[0].members}
        self.assertEqual(members_by_op[0].canonical_argument_order, (0, 1))
        self.assertEqual(members_by_op[1].canonical_argument_order, (1, 0))

    def test_canonicalizes_three_reordered_operands(self) -> None:
        program = _program(
            instructions=[
                make_einsum_instruction("ij,jk,kl->il", 0, 1, 2),
                make_einsum_instruction("zw,xy,yz->xw", 3, 4, 5),
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
        self.assertEqual(groups[0].operator, "einsum")
        self.assertEqual(
            groups[0].canonical_instruction_specific_arguments,
            ("ac,db,cd->ab",),
        )
        self.assertEqual(groups[0].operand_shapes, ((2, 3), (4, 5), (3, 4)))
        members_by_op = {member.op_index: member for member in groups[0].members}
        self.assertEqual(members_by_op[0].canonical_argument_order, (0, 2, 1))
        self.assertEqual(members_by_op[1].canonical_argument_order, (1, 0, 2))

    def test_does_not_group_identical_einsums_at_different_output_depths(self) -> None:
        program = _program(
            instructions=[
                make_einsum_instruction("ab,bc->ac", 0, 1),
                make_einsum_instruction("ab,bc->ac", 2, 3),
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

    def test_ignores_unreachable_ops(self) -> None:
        program = _program(
            instructions=[
                make_einsum_instruction("ab,bc->ac", 0, 1),
                make_einsum_instruction("ab,bc->ac", 2, 3),
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
            tuple(
                tuple(member.op_index for member in group.members)
                for group in groups
            ),
            ((2,), (1,)),
        )

    def test_splits_same_depth_dependent_candidates(self) -> None:
        program = _program(
            instructions=[
                make_einsum_instruction("ab,bc->ac", 0, 1),
                make_einsum_instruction("ab,bc->ac", 3, 2),
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
            tuple(
                tuple(member.op_index for member in group.members)
                for group in singleton_groups
            ),
            ((2,), (0,), (1,)),
        )

    def test_raises_when_unique_labels_exceed_canonical_capacity(self) -> None:
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0"
        program = _program(
            instructions=[
                make_einsum_instruction(f"{labels}->a", 0),
            ],
            n_inputs=1,
            shapes=[
                (1,) * len(labels),
                (1,),
            ],
        )

        with self.assertRaisesRegex(ValueError, "more unique labels"):
            group_identical_ops_by_output_depth(program, min_group_size=1)


if __name__ == "__main__":
    unittest.main()
