import csv
from collections import Counter

import pytest
import torch
from cirkit.symbolic.layers import SumLayer

from demo.cirkit import append_row, make_symbolic_circuit, preprocess_xe_program, set_seed, setup_xe_training, translate_cirkit_to_xe
from experiments.ablation import VARIANTS
from extended_einsum.preprocess import FoldSameShapedOperations


@pytest.fixture(autouse=True)
def run_compiled_steps_eagerly_in_unit_tests(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function, **_kwargs: function,
    )


def test_scaled_ablation_defaults_to_maximum_normalization() -> None:
    assert VARIANTS["xe"].semiring == "scaled-max"
    assert VARIANTS["xe"].optimize_group_order
    assert VARIANTS["shift-gradients"].semiring == "scaled-max"
    assert VARIANTS["shift-gradients"].shift_mode == "differentiable"
    assert VARIANTS["shift-gradients"].optimize_group_order
    assert VARIANTS["logspace"].semiring == "lse-sum"
    assert VARIANTS["logspace"].shift_mode == "xe"
    assert VARIANTS["logspace"].optimize_group_order
    assert not VARIANTS["no-ordering"].optimize_group_order
    assert VARIANTS["logspace-shift-gradients"].semiring == "lse-sum"
    assert VARIANTS["logspace-shift-gradients"].shift_mode == "differentiable"
    assert VARIANTS["logspace-shift-gradients"].optimize_group_order


def test_cp_counterfactuals_remove_one_production_optimization() -> None:
    production = VARIANTS["xe"]
    unordered = VARIANTS["no-ordering"]
    differentiable = VARIANTS["shift-gradients"]
    logspace = VARIANTS["logspace"]
    unoptimized_stability = VARIANTS["logspace-shift-gradients"]

    assert (
        production.shift_mode,
        production.semiring,
        production.optimize_group_order,
    ) == ("xe", "scaled-max", True)
    assert (
        unordered.shift_mode,
        unordered.semiring,
        unordered.optimize_group_order,
    ) == ("xe", "scaled-max", False)
    assert (
        differentiable.shift_mode,
        differentiable.semiring,
        differentiable.optimize_group_order,
    ) == ("differentiable", "scaled-max", True)
    assert (
        logspace.shift_mode,
        logspace.semiring,
        logspace.optimize_group_order,
    ) == ("xe", "lse-sum", True)
    assert (
        unoptimized_stability.shift_mode,
        unoptimized_stability.semiring,
        unoptimized_stability.optimize_group_order,
    ) == ("differentiable", "lse-sum", True)


def test_tucker_sum_weights_use_output_first_layout_and_normalize_all_inputs() -> None:
    circuit = make_symbolic_circuit(width=4, height=4, num_units=3, sum_product_layer="tucker")
    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability="logspace_max")

    einsum_formats = Counter(instruction.operator.format_string for instruction in program.instructions if instruction.operator.name == "einsum")
    softmax_axes = Counter(instruction.operator.axis for instruction in program.instructions if instruction.operator.name == "softmax")

    # d is the output unit, while b and c are the two contiguous Tucker input
    # axes. This matches Cirkit's (output, flattened-input) weight layout.
    assert einsum_formats["abc,dbc->ad"] == 15
    assert softmax_axes == {(1, 2): 15}

    folded = preprocess_xe_program(program, optimize_stacking=True)
    folded_softmax_axes = Counter(instruction.operator.axis for instruction in folded.instructions if instruction.operator.name == "softmax")
    assert set(folded_softmax_axes) == {(1, 2), (2, 3)}


def test_quad_graph_preserves_shared_layers_and_compact_mixing_weights() -> None:
    circuit = make_symbolic_circuit(width=4, height=4, num_units=3, sum_product_layer="cp", region_graph="quad-graph")
    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability="logspace_max")

    num_sum_layers = sum(isinstance(layer, SumLayer) for layer in circuit.layers)
    parameter_shapes = Counter(program.shapes[input_id] for input_id in program.parameter_indices)

    # Every shared symbolic sum layer is translated exactly once. The five
    # multi-partition regions use compact (unit, arity) mixing logits.
    assert program.n_inputs == num_sum_layers + 1
    assert parameter_shapes[(3, 2)] == 4
    assert parameter_shapes[(1, 2)] == 1
    assert sum(instruction.operator.name == "stack" for instruction in program.instructions) == 5


@pytest.mark.parametrize("region_graph", ["quad-tree-2", "quad-graph"])
@pytest.mark.parametrize(
    "stability_mode",
    ["logspace_max", "scaled_min", "scaled_max", "scaled_sum"],
)
def test_stable_tucker_preprocessing_uses_nary_fusion(
    region_graph: str,
    stability_mode: str,
) -> None:
    circuit = make_symbolic_circuit(width=4, height=4, num_units=3, sum_product_layer="tucker", region_graph=region_graph)
    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability=stability_mode)

    fused = preprocess_xe_program(program, optimize_stacking=True)

    assert any(instruction.operator.format_string.count(",") == 2 for instruction in fused.instructions if instruction.operator.name == "einsum")


@pytest.mark.parametrize(
    "options",
    [
        {
            "shift_mode": "differentiable",
            "optimize_group_order": False,
            "optimize_contraction_paths": False,
        },
        {
            "shift_mode": "xe",
            "optimize_group_order": False,
            "optimize_contraction_paths": False,
        },
        {
            "shift_mode": "xe",
            "optimize_group_order": False,
            "optimize_contraction_paths": True,
        },
        {
            "shift_mode": "xe",
            "optimize_group_order": True,
            "optimize_contraction_paths": True,
        },
    ],
)
def test_tucker_ablation_variants_preserve_loss(options) -> None:
    common = {
        "width": 4,
        "height": 4,
        "num_units": 3,
        "batch_size": 2,
        "sum_product_layer": "tucker",
        "region_graph": "quad-graph",
        "device": torch.device("cpu"),
        "dataset": "synthetic",
        "data_dir": "datasets",
        "num_samples": 4,
        "pixel_values": 8,
        "semiring": "lse-sum",
        "lr": 0.01,
    }
    set_seed(7)
    reference_step, _optimizer, reference_images, _program = setup_xe_training(
        **common,
        shift_mode="xe",
        optimize_group_order=False,
        optimize_contraction_paths=False,
    )
    set_seed(7)
    candidate_step, _optimizer, candidate_images, _program = setup_xe_training(
        **common,
        **options,
    )

    torch.testing.assert_close(
        candidate_step(candidate_images[:2]),
        reference_step(reference_images[:2]),
    )


@pytest.mark.parametrize("region_graph", ["quad-tree-2", "quad-graph"])
@pytest.mark.parametrize(
    "semiring",
    ["lse-sum", "scaled-min", "scaled-max", "scaled-sum"],
)
def test_cp_t_training_step_is_finite(region_graph: str, semiring: str) -> None:
    set_seed(7)
    step, _optimizer, images, program = setup_xe_training(
        width=4,
        height=4,
        num_units=3,
        batch_size=2,
        sum_product_layer="cp-t",
        region_graph=region_graph,
        device=torch.device("cpu"),
        dataset="synthetic",
        data_dir="datasets",
        num_samples=4,
        pixel_values=8,
        semiring=semiring,
        lr=0.01,
    )

    loss = step(images[:2])
    loss.backward()

    assert torch.isfinite(loss)
    assert program.instructions


def test_new_benchmark_csv_records_reproducibility_fields(tmp_path) -> None:
    output = tmp_path / "benchmark.csv"

    append_row(
        output,
        {
            "backend": "xe-quad-graph-lse-sum-torch-compile",
            "backend_type": "xe",
            "region_graph": "quad-graph",
            "seed": 2,
            "warmup_epochs": 5,
        },
    )

    with output.open(newline="") as output_file:
        row = next(csv.DictReader(output_file))
    assert row["backend_type"] == "xe"
    assert row["region_graph"] == "quad-graph"
    assert row["seed"] == "2"
    assert row["warmup_epochs"] == "5"


def test_append_row_keeps_an_existing_legacy_csv_schema(tmp_path) -> None:
    output = tmp_path / "legacy.csv"
    output.write_text("backend,status\n")

    append_row(output, {"backend": "xe-lse-sum", "status": "ok", "seed": 2})

    with output.open(newline="") as output_file:
        rows = list(csv.reader(output_file))
    assert rows == [["backend", "status"], ["xe-lse-sum", "ok"]]


def test_xe_order_counterfactual_preserves_training_loss() -> None:
    common = {
        "width": 4,
        "height": 4,
        "num_units": 3,
        "batch_size": 2,
        "sum_product_layer": "cp",
        "region_graph": "quad-tree-2",
        "device": torch.device("cpu"),
        "dataset": "synthetic",
        "data_dir": "datasets",
        "num_samples": 4,
        "pixel_values": 8,
        "semiring": "lse-sum",
        "lr": 0.01,
    }
    set_seed(7)
    first_step, _optimizer, first_images, _program = setup_xe_training(
        **common,
        shift_mode="xe",
        optimize_group_order=False,
    )
    set_seed(7)
    second_step, _optimizer, second_images, _program = setup_xe_training(
        **common,
        shift_mode="xe",
        optimize_group_order=True,
    )

    torch.testing.assert_close(first_step(first_images[:2]), second_step(second_images[:2]))


@pytest.mark.parametrize(
    ("sum_product_layer", "expected_counts"),
    [
        (
            "cp",
            {
                "einsum": 32,
                "concat": 3,
                "take": 45,
                "select": 2,
            },
        ),
        (
            "tucker",
            {
                "einsum": 25,
                "concat": 3,
                "take": 28,
                "select": 2,
            },
        ),
    ],
)
def test_input_depth_fold_reproduces_metadata_free_quad_graph_structure(
    sum_product_layer,
    expected_counts,
) -> None:
    circuit = make_symbolic_circuit(
        width=28,
        height=28,
        num_units=2,
        sum_product_layer=sum_product_layer,
        region_graph="quad-graph",
    )
    program, _inputs = translate_cirkit_to_xe(
        circuit,
        batch_size=2,
        stability="logspace_max",
    )

    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=False,
    ).program

    operator_counts = Counter(
        instruction.operator.name for instruction in folded.instructions
    )
    for operator, expected_count in expected_counts.items():
        assert operator_counts[operator] == expected_count


def test_input_depth_fold_emits_fragmented_batch_gathers_directly() -> None:
    circuit = make_symbolic_circuit(width=4, height=4, num_units=3, sum_product_layer="cp")
    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability="logspace_max")
    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        optimize_group_order=False,
    )

    assert folded.gather_index_orders
    index_input_ids = set(
        range(
            folded.program.n_inputs - len(folded.gather_index_orders),
            folded.program.n_inputs,
        )
    )
    take_instructions = [
        instruction
        for instruction in folded.program.instructions
        if instruction.operator.name == "take"
    ]
    assert len(take_instructions) == len(folded.gather_index_orders)
    assert {
        instruction.argument_ssa_ids[1]
        for instruction in take_instructions
    } == index_input_ids
    for instruction in folded.program.instructions:
        if instruction.operator.name != "concat":
            continue
        assert all(
            argument < folded.program.n_inputs
            or folded.program.instructions[
                argument - folded.program.n_inputs
            ].operator.name
            != "slice"
            for argument in instruction.argument_ssa_ids
        )
