import csv

import pandas as pd
import pytest

from demo.plot_cirkit_region_graph_benchmark import load_seed_summaries


@pytest.mark.parametrize(
    ("peak_reserved_bytes", "expected_gib"),
    [(None, 2.0), (3 * 1024**3, 3.0)],
)
def test_memory_summary_uses_required_reserved_vram(
    tmp_path,
    peak_reserved_bytes: int | None,
    expected_gib: float,
) -> None:
    row = {
        "status": "ok",
        "backend_type": "xe",
        "region_graph": "quad-tree-2",
        "sum_product_layer": "cp",
        "units": 64,
        "batch_size": 256,
        "seed": 0,
        "semiring": "lse-sum",
        "torch_compile": True,
        "epoch": 0,
        "epochs": 1,
        "forward_loss_ms": 100.0,
        "backward_ms": 200.0,
        "reserved_memory_bytes": 2 * 1024**3,
    }
    if peak_reserved_bytes is not None:
        row["peak_reserved_memory_bytes"] = peak_reserved_bytes
    input_path = tmp_path / "benchmark.csv"
    pd.DataFrame([row]).to_csv(input_path, index=False)

    summary = load_seed_summaries(input_path, require_complete=False).iloc[0]

    assert summary["required_vram_gib"] == pytest.approx(expected_gib)


def test_memory_summary_reads_mixed_legacy_and_current_csv_rows(tmp_path) -> None:
    legacy_row = {
        "status": "ok",
        "backend_type": "xe",
        "region_graph": "quad-tree-2",
        "sum_product_layer": "cp",
        "units": 1024,
        "batch_size": 256,
        "seed": 0,
        "semiring": "lse-sum",
        "torch_compile": True,
        "epoch": 0,
        "epochs": 2,
        "forward_loss_ms": 100.0,
        "backward_ms": 200.0,
        "reserved_memory_bytes": 2 * 1024**3,
    }
    current_row = {
        **legacy_row,
        "epoch": 1,
        "peak_reserved_memory_bytes": 3 * 1024**3,
        "peak_reserved_memory_mib": 3 * 1024,
    }
    legacy_header = list(legacy_row)
    current_header = list(legacy_header)
    insertion_index = current_header.index("reserved_memory_bytes")
    current_header[insertion_index:insertion_index] = [
        "peak_reserved_memory_bytes",
        "peak_reserved_memory_mib",
    ]
    input_path = tmp_path / "mixed-schema-benchmark.csv"
    with input_path.open("w", newline="") as input_file:
        writer = csv.writer(input_file)
        writer.writerow(legacy_header)
        writer.writerow([legacy_row[field] for field in legacy_header])
        writer.writerow([current_row[field] for field in current_header])

    summary = load_seed_summaries(input_path, require_complete=False).iloc[0]

    assert summary["measured_epochs"] == 2
    assert summary["required_vram_gib"] == pytest.approx(3.0)
