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
