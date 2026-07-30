import argparse
import csv

from experiments.ablation import VARIANTS
from experiments.compile_table import latex_table, load_results
from experiments.diagnose_compile import (
    CSV_FIELDS,
    Configuration,
    append_row,
    completed_keys,
    configurations,
    parse_single,
)


def _args(*, runs=(0,)):
    return argparse.Namespace(
        suites=("ablation", "monarch"),
        runs=runs,
        layers=("cp", "tucker"),
        graphs=("quad-tree-2", "quad-graph"),
        variants=tuple(VARIANTS),
        parameterizations=("dense", "monarch"),
    )


def test_publication_grid_has_five_process_isolated_runs() -> None:
    selected = configurations(_args(runs=tuple(range(5))))

    assert len(selected) == 280
    assert len({configuration.key for configuration in selected}) == 280
    assert sum(configuration.suite == "ablation" for configuration in selected) == 240
    assert sum(configuration.suite == "monarch" for configuration in selected) == 40


def test_internal_configuration_round_trip() -> None:
    original = Configuration(
        "monarch",
        "xe",
        "xe",
        3,
        "quad-graph",
        "cp",
        "monarch",
        64,
        64,
        256,
        256,
        16,
        16,
    )

    assert parse_single(original.child_arguments()[1:]) == original


def test_compile_table_requires_and_summarizes_every_run(tmp_path) -> None:
    results = tmp_path / "compile.csv"
    selected = configurations(_args())
    for index, configuration in enumerate(selected):
        value = 1.0 + index / 1000.0
        row = {
            "suite": configuration.suite,
            "system": configuration.system,
            "variant": configuration.variant,
            "run": configuration.run,
            "status": "ok",
            "region_graph": configuration.region_graph,
            "layer": configuration.layer,
            "parameterization": configuration.parameterization,
            "width": configuration.width,
            "height": configuration.height,
            "units": configuration.units,
            "batch_size": configuration.batch_size,
            "monarch_p": configuration.p or "",
            "monarch_q": configuration.q or "",
            "our_passes_seconds": (
                value if configuration.system == "xe" else ""
            ),
            "cirkit_lowering_seconds": (
                value if configuration.system == "cirkit" else ""
            ),
            "torch_compile_seconds": value + 1.0,
            "compile_total_seconds": 2.0 * value + 1.0,
        }
        append_row(results, row)

    with results.open(newline="") as input_file:
        assert tuple(next(csv.reader(input_file))) == CSV_FIELDS
    assert completed_keys(results) == {
        configuration.key for configuration in selected
    }

    summaries = load_results(results, runs="0")
    table = latex_table(summaries)
    assert len(summaries) == 56
    assert "median [minimum, maximum] seconds over 1 process-isolated runs" in table
    assert "Log space + shift gradients" in table
    assert r"$16\mathbin{\times}16$" in table
    assert r"\texttt{torch.compile}" in table
