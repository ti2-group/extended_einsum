from __future__ import annotations

import pytest

pytest.importorskip("pyjuice")

from experiments.pyjuice_cp_t.model import (
    build_cp_t_quad_tree,
    expected_parameters,
    validate_structure,
)


@pytest.mark.parametrize(("height", "width"), [(2, 2), (4, 4), (7, 7)])
def test_pyjuice_cp_t_exactly_matches_cirkit_structure_and_parameters(
    height: int,
    width: int,
) -> None:
    units = 4
    categories = 8
    root = build_cp_t_quad_tree(
        height=height,
        width=width,
        units=units,
        categories=categories,
    )

    metadata = validate_structure(
        root,
        height=height,
        width=width,
        units=units,
        categories=categories,
    )

    assert metadata["product_layers"] == height * width - 1
    assert metadata["sum_layers"] == height * width - 1
    assert metadata["parameters"] == expected_parameters(
        variables=height * width,
        patches=height * width - 1,
        units=units,
        categories=categories,
    )
