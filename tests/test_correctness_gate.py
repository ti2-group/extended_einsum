import csv

from experiments.run_correctness_after_jax import (
    correctness_row_key,
    expected_correctness_keys,
    expected_jax_keys,
    jax_csv_state,
    validate_correctness,
)


def write_rows(path, fields, rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_expected_gate_sizes_match_publication_grids() -> None:
    assert len(expected_jax_keys()) == 200
    assert len(expected_correctness_keys()) == 820


def test_jax_gate_requires_every_unique_success(tmp_path) -> None:
    output = tmp_path / "jax.csv"
    fields = (
        "variant",
        "status",
        "error",
        "seed",
        "region_graph",
        "layer",
        "batch_size",
        "units",
    )
    rows = [
        {
            "variant": variant,
            "status": "ok",
            "error": "",
            "seed": seed,
            "region_graph": graph,
            "layer": layer,
            "batch_size": batch,
            "units": units,
        }
        for variant, seed, graph, layer, batch, units in expected_jax_keys()
    ]
    write_rows(output, fields, rows)

    complete = jax_csv_state(output)
    assert complete.complete
    assert complete.successful == 200

    rows.append(rows[0])
    write_rows(output, fields, rows)
    duplicated = jax_csv_state(output)
    assert not duplicated.complete
    assert duplicated.duplicates


def test_correctness_gate_accepts_exact_target_grid(tmp_path) -> None:
    output = tmp_path / "correctness.csv"
    key_fields = (
        "suite",
        "seed",
        "region_graph",
        "layer",
        "width",
        "height",
        "units",
        "batch_size",
        "pixel_values",
        "depth",
        "variant",
        "matmul_precision",
        "device",
        "torch_compile",
    )
    fields = (*key_fields, "status", "error")
    rows = [
        {
            **dict(zip(key_fields, key, strict=True)),
            "status": "ok",
            "error": "",
        }
        for key in expected_correctness_keys()
    ]
    write_rows(output, fields, rows)

    validate_correctness(output)
    assert {correctness_row_key(row) for row in rows} == expected_correctness_keys()
