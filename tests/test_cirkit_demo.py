import csv
from collections import Counter

import pytest

from cirkit.symbolic.layers import SumLayer

from demo.cirkit import append_row, make_symbolic_circuit, preprocess_xe_program, translate_cirkit_to_xe


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
