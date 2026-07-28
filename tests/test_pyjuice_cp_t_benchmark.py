from __future__ import annotations

import pytest

pytest.importorskip("pyjuice")

from demo.benchmark_pyjuice_cp_t import (
    build_pyjuice_cp_t_quad_tree,
    expected_parameter_count,
    validate_parameter_matched_structure,
)


@pytest.mark.parametrize(("height", "width"), [(2, 2), (4, 4), (7, 7)])
def test_pyjuice_cp_t_exactly_matches_cirkit_structure_and_parameters(
    height: int,
    width: int,
) -> None:
    units = 4
    categories = 8
    root = build_pyjuice_cp_t_quad_tree(
        height=height,
        width=width,
        num_units=units,
        num_categories=categories,
    )

    metadata = validate_parameter_matched_structure(
        root,
        height=height,
        width=width,
        num_units=units,
        num_categories=categories,
    )

    assert metadata["product_layers"] == height * width - 1
    assert metadata["sum_layers"] == height * width - 1
    assert metadata["logical_model_parameters"] == expected_parameter_count(
        variables=height * width,
        patches=height * width - 1,
        units=units,
        num_categories=categories,
    )
