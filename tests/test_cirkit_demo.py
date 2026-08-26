import csv
from collections import Counter

import pytest
import torch
from cirkit.symbolic.layers import SumLayer

from demo.cirkit import (
    append_row,
    make_symbolic_circuit,
    preprocess_xe_program,
    resolve_categorical_lookup,
    set_seed,
    setup_xe_training,
    translate_cirkit_to_xe,
)
from extended_einsum.preprocess import FoldSameShapedOperations


@pytest.mark.parametrize(
    ("layer", "units", "expected"),
    [
        ("cp", 64, "advanced"),
        ("cp", 128, "flattened"),
        ("cp", 512, "flattened"),
        ("cp-t", 64, "gather"),
        ("cp-t", 512, "gather"),
        ("tucker", 32, "gather"),
        ("tucker", 64, "gather"),
    ],
)
def test_categorical_lookup_auto_selection(layer, units, expected) -> None:
    assert (
        resolve_categorical_lookup(
            "auto",
            sum_product_layer=layer,
            num_units=units,
        )
        == expected
    )


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
@pytest.mark.parametrize("stability_mode", ["logspace_max", "scaled_sum"])
def test_stable_tucker_preprocessing_uses_nary_fusion(
    region_graph: str,
    stability_mode: str,
) -> None:
    circuit = make_symbolic_circuit(width=4, height=4, num_units=3, sum_product_layer="tucker", region_graph=region_graph)
    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability=stability_mode)

    fused = preprocess_xe_program(program, optimize_stacking=True)

    assert any(instruction.operator.format_string.count(",") == 2 for instruction in fused.instructions if instruction.operator.name == "einsum")


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


def test_xe_gather_lookup_preserves_training_loss_and_gradients() -> None:
    common = {
        "width": 4,
        "height": 4,
        "num_units": 3,
        "batch_size": 2,
        "sum_product_layer": "cp-t",
        "region_graph": "quad-tree-2",
        "device": torch.device("cpu"),
        "dataset": "synthetic",
        "data_dir": "datasets",
        "num_samples": 4,
        "pixel_values": 8,
        "semiring": "lse-sum",
        "lr": 0.01,
    }
    set_seed(11)
    advanced_step, advanced_optimizer, advanced_images, _program = (
        setup_xe_training(
            **common,
            categorical_lookup="advanced",
        )
    )
    set_seed(11)
    gather_step, gather_optimizer, gather_images, _program = setup_xe_training(
        **common,
        categorical_lookup="gather",
    )

    torch.testing.assert_close(advanced_images, gather_images)
    advanced_loss = advanced_step(advanced_images[:2])
    gather_loss = gather_step(gather_images[:2])
    torch.testing.assert_close(advanced_loss, gather_loss)
    advanced_loss.backward()
    gather_loss.backward()
    advanced_parameters = advanced_optimizer.param_groups[0]["params"]
    gather_parameters = gather_optimizer.param_groups[0]["params"]
    assert len(advanced_parameters) == len(gather_parameters)
    for advanced_parameter, gather_parameter in zip(
        advanced_parameters,
        gather_parameters,
        strict=True,
    ):
        torch.testing.assert_close(
            advanced_parameter.grad,
            gather_parameter.grad,
        )


def test_input_depth_fold_routes_same_index_products_without_gathers_by_default() -> None:
    circuit = make_symbolic_circuit(
        width=28,
        height=28,
        num_units=3,
        sum_product_layer="cp-t",
    )
    program, _inputs = translate_cirkit_to_xe(
        circuit,
        batch_size=2,
        stability="logspace_max",
    )

    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(program)
    gathered = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        gather_fragmented_batches=True,
    )

    assert folded.gather_index_orders == ()
    assert all(
        instruction.operator.name != "take"
        for instruction in folded.program.instructions
    )
    assert gathered.gather_index_orders


def test_input_depth_fold_auto_gathers_fragmented_shared_graph_batches() -> None:
    circuit = make_symbolic_circuit(
        width=4,
        height=4,
        num_units=3,
        sum_product_layer="cp",
        region_graph="quad-graph",
    )
    program, _inputs = translate_cirkit_to_xe(
        circuit,
        batch_size=2,
        stability="logspace_max",
    )

    folded = FoldSameShapedOperations.apply_with_input_depth_metadata(program)
    gathered = FoldSameShapedOperations.apply_with_input_depth_metadata(
        program,
        gather_fragmented_batches=True,
    )

    assert folded.gather_index_orders
    assert len(folded.gather_index_orders) < len(gathered.gather_index_orders)
    assert any(
        instruction.operator.name == "take"
        for instruction in folded.program.instructions
    )
