import math

import torch

from experiments.correctness import (
    compare,
    evaluate,
    make_circuit_inputs,
    make_circuit_program,
    make_depth_inputs,
    make_depth_program,
)
from experiments.plot_correctness import scientific


def test_optimized_stability_modes_match_unstable_float64_reference() -> None:
    program = make_circuit_program(
        width=4,
        height=4,
        units=3,
        batch_size=2,
        layer="cp",
        region_graph="quad-tree-2",
    )
    inputs = make_circuit_inputs(
        program,
        seed=7,
        pixel_values=2,
    )
    reference = evaluate(
        program,
        inputs,
        mode="unstable",
        dtype=torch.float64,
        device=torch.device("cpu"),
        matmul_precision="highest",
        optimize=False,
    )

    for mode in ("scaled_max", "logspace_max"):
        candidate = evaluate(
            program,
            inputs,
            mode=mode,
            dtype=torch.float32,
            device=torch.device("cpu"),
            matmul_precision="highest",
            optimize=True,
        )
        metrics = compare(
            candidate,
            reference,
            parameter_ids=program.parameter_indices,
        )

        assert metrics.forward_relative_l2 < 1.0e-5
        assert metrics.data_gradient_relative_l2 < 1.0e-4
        assert metrics.parameter_gradient_relative_l2 < 1.0e-4
        assert metrics.forward_finite_fraction == 1.0
        assert metrics.gradient_finite_fraction == 1.0


def test_depth_stress_exposes_raw_fp32_underflow() -> None:
    program = make_depth_program(
        depth=32,
        units=3,
        batch_size=2,
    )
    inputs = make_depth_inputs(
        program,
        seed=3,
        factor_scale=0.01,
    )
    common = {
        "program": program,
        "canonical_inputs": inputs,
        "device": torch.device("cpu"),
        "matmul_precision": "highest",
    }
    raw = evaluate(
        **common,
        mode="unstable",
        dtype=torch.float32,
        optimize=True,
        optimize_contractions=False,
    )
    scaled = evaluate(
        **common,
        mode="scaled_max",
        dtype=torch.float32,
        optimize=True,
        optimize_contractions=False,
    )
    logspace = evaluate(
        **common,
        mode="logspace_max",
        dtype=torch.float32,
        optimize=True,
        optimize_contractions=False,
    )

    assert not torch.all(torch.isfinite(raw.output))
    assert all(torch.all(torch.isfinite(gradient)) for gradient in scaled.gradients.values())
    assert all(torch.all(torch.isfinite(gradient)) for gradient in logspace.gradients.values())


def test_circuit_inputs_are_valid_selected_categorical_probabilities() -> None:
    program = make_circuit_program(
        width=4,
        height=4,
        units=3,
        batch_size=2,
        layer="tucker",
        region_graph="quad-graph",
    )
    observations = torch.arange(32).reshape(2, 16) % 4
    values = make_circuit_inputs(
        program,
        seed=11,
        pixel_values=4,
        observations=observations,
    )

    assert values[0].shape == (16, 2, 3)
    assert torch.all(values[0] > 0.0)
    assert torch.all(values[0] < 1.0)
    assert len(values) == program.n_inputs


def test_torch_compile_path_preserves_autograd(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function, **_kwargs: function,
    )
    program = make_depth_program(
        depth=2,
        units=3,
        batch_size=2,
    )
    inputs = make_depth_inputs(
        program,
        seed=5,
        factor_scale=0.1,
    )

    result = evaluate(
        program,
        inputs,
        mode="scaled_max",
        dtype=torch.float32,
        device=torch.device("cpu"),
        matmul_precision="highest",
        optimize=True,
        optimize_contractions=False,
        torch_compile_run=True,
    )

    assert torch.all(torch.isfinite(result.output))
    assert all(torch.all(torch.isfinite(gradient)) for gradient in result.gradients.values())


def test_scientific_formats_finite_and_nonfinite_values() -> None:
    assert scientific(0.0) == "0"
    assert "10^{-7}" in scientific(2.5e-7)
    assert scientific(math.inf) == r"$\infty$"
