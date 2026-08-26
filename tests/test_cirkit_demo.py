import csv
from collections import Counter

import numpy as np
import pytest
import torch
from cirkit.symbolic.layers import SumLayer

from demo.cirkit import append_row, learn_clt_tree, load_or_learn_clt_tree, make_symbolic_circuit, preprocess_xe_program, translate_cirkit_to_xe


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


def _correlated_synthetic_data(num_variables: int, num_categories: int, num_samples: int = 400) -> "torch.Tensor":
    torch.manual_seed(0)
    base = torch.randint(num_categories, size=(num_samples, 1))
    noise = torch.randint(2, size=(num_samples, num_variables))
    return (base + noise.cumsum(dim=1)) % num_categories


def test_learn_clt_tree_bins_data_and_returns_a_rooted_tree() -> None:
    # 256 pixel values with 8 bins exercises the pre-binning workaround for
    # Cirkit's num_bins path, which allocates counts for the unbinned categories.
    data = _correlated_synthetic_data(num_variables=12, num_categories=256)

    tree = learn_clt_tree(data, pixel_values=256, num_bins=8)

    assert tree.shape == (12,)
    assert (tree == -1).sum() == 1
    assert all(-1 <= parent < 12 for parent in tree)


def test_learn_clt_tree_rejects_insufficient_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        learn_clt_tree(torch.zeros((0, 12), dtype=torch.long), pixel_values=4, num_bins=4)


def test_load_or_learn_clt_tree_reuses_the_cached_tree(tmp_path) -> None:
    data = _correlated_synthetic_data(num_variables=8, num_categories=4)
    cache_path = tmp_path / "hclt_trees" / "tree.npy"

    tree = load_or_learn_clt_tree(data, pixel_values=4, num_bins=4, cache_path=cache_path)
    cached = load_or_learn_clt_tree(torch.zeros((0, 8), dtype=torch.long), pixel_values=4, num_bins=4, cache_path=cache_path)

    assert cache_path.exists()
    assert (tree == cached).all()


@pytest.mark.parametrize("sum_product_layer", ["cp", "tucker"])
@pytest.mark.parametrize("stability_mode", ["logspace_max", "scaled_sum"])
def test_chow_liu_tree_circuit_translates_without_mixing_layers(sum_product_layer: str, stability_mode: str) -> None:
    data = _correlated_synthetic_data(num_variables=12, num_categories=4)
    tree = learn_clt_tree(data, pixel_values=4, num_bins=4)
    circuit = make_symbolic_circuit(
        width=3,
        height=4,
        num_units=3,
        sum_product_layer=sum_product_layer,
        region_graph="chow-liu-tree",
        pixel_values=4,
        clt_tree=tree,
    )

    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability=stability_mode)
    folded = preprocess_xe_program(program, optimize_stacking=True)

    # Tree region graphs have one partition per region, so no mixing layers
    # appear and every symbolic sum layer is translated exactly once.
    num_sum_layers = sum(isinstance(layer, SumLayer) for layer in circuit.layers)
    assert program.n_inputs == num_sum_layers + 1
    assert not any(instruction.operator.name == "stack" for instruction in program.instructions)
    assert any(instruction.operator.name == "einsum" for instruction in folded.instructions)


def test_chow_liu_tree_translation_handles_trees_deeper_than_the_recursion_limit() -> None:
    # A path-shaped tree maximizes depth: with 500 variables the layer graph is
    # far deeper than Python's default recursion limit.
    num_variables = 500
    tree = np.arange(-1, num_variables - 1)
    circuit = make_symbolic_circuit(
        width=num_variables,
        height=1,
        num_units=2,
        sum_product_layer="cp",
        region_graph="chow-liu-tree",
        pixel_values=4,
        clt_tree=tree,
    )

    program, _inputs = translate_cirkit_to_xe(circuit, batch_size=2, stability="logspace_max")

    assert sum(instruction.operator.name == "select" for instruction in program.instructions) == num_variables


def test_make_symbolic_circuit_requires_a_learned_tree_for_chow_liu() -> None:
    with pytest.raises(ValueError, match="clt_tree"):
        make_symbolic_circuit(width=2, height=2, num_units=2, sum_product_layer="cp", region_graph="chow-liu-tree")
    with pytest.raises(ValueError, match="4"):
        make_symbolic_circuit(
            width=2,
            height=2,
            num_units=2,
            sum_product_layer="cp",
            region_graph="chow-liu-tree",
            clt_tree=np.array([-1, 0]),
        )
