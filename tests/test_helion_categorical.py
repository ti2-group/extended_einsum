"""Raw evidence selection is part of the compiled circuit, not a cached input."""

from dataclasses import replace

import pytest
import torch

from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion import CategoricalInput, compile_helion
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import OperatorEinsum
from tests.test_helion import MODES, benchmark


def case(device, *, logits=False, log_values=True, units=9, categories=5, batch=7, depth=3):
    generator = torch.Generator(device=device).manual_seed(941)
    variables = 2**depth + 3
    pixels = torch.randint(categories, (batch, variables), generator=generator, device=device)
    tables = 1.5 * torch.randn(4, categories, units, generator=generator, device=device)
    if not logits:
        tables = tables.log_softmax(dim=1)
    ids = torch.randperm(variables, generator=generator, device=device)[: 2**depth]
    banks = (ids * 3 + 1) % 4
    binding = CategoricalInput(pixels, tables, ids, banks, logits=logits, log_values=log_values)
    p, inputs, axes = benchmark.folded_circuit(batch=batch, units=units, depth=depth, device=device, stability="logspace_max", categorical=binding)
    if not log_values:
        # An identity replaces the benchmark's exp(log evidence).
        p = replace(p, instructions=[RichInstruction(OperatorEinsum("fbi->fbi"), (0,)), *p.instructions[1:]])
    return p, inputs, axes


def expected(program, inputs):
    values = [v.materialize() if isinstance(v, CategoricalInput) else v for v in inputs]
    return run_program(translate_to_backend_program(program, TorchBackendFunctions()), values)


@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("log_values", [False, True])
@pytest.mark.parametrize("mode", MODES[1:])
def test_categorical_reference_modes_and_routing(logits, log_values, mode):
    p, inputs, axes = case("cpu", logits=logits, log_values=log_values)
    p = replace(p, stability_mode=mode)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, reduction_block_size=4)
    torch.testing.assert_close(runner.reference(inputs), expected(p, inputs), atol=1e-4, rtol=1e-6)
    assert len(runner.preparations) == int(logits)
    assert "categories0" in runner.source
    for method in ("advanced", "gather", "flattened"):
        torch.testing.assert_close(inputs[0].materialize(method), inputs[0].materialize())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_dynamic_pixels_tables_and_routes(device):
    p, inputs, axes = case(device, logits=True)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, dot_precision="tf32x3")
    run = runner.reference if device == "cpu" else runner
    first = run(inputs)
    torch.testing.assert_close(first, expected(p, inputs))
    binding = inputs[0]
    # Replacements and in-place updates must both affect the very next call.
    changed = replace(binding, pixels=(binding.pixels + 1) % 5, tables=binding.tables.flip(1).contiguous())
    inputs = [changed, *inputs[1:]]
    torch.testing.assert_close(run(inputs), expected(p, inputs))
    changed.variable_ids.copy_(changed.variable_ids.roll(1))
    changed.table_ids.copy_(changed.table_ids.roll(2))
    torch.testing.assert_close(run(inputs), expected(p, inputs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("log_values", [False, True])
@pytest.mark.parametrize("units,categories", [(9, 5), (65, 7), (128, 256)])
@pytest.mark.parametrize("pipelined", [False, True])
def test_categorical_cuda_partial_tiles_and_large_tables(logits, log_values, units, categories, pipelined):
    p, inputs, axes = case("cuda", logits=logits, log_values=log_values, units=units, categories=categories, batch=17, depth=2)
    runner = compile_helion(
        p,
        inputs,
        batch_axes=axes,
        schedule="subtree",
        block_size=16,
        num_warps=8,
        reduction_block_size=32,
        dot_precision="tf32x3",
        pipeline_stages=2 if pipelined else 1,
    )
    torch.testing.assert_close(runner(inputs), expected(p, inputs), atol=1e-4, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_uint8_pixels_and_guarded_invalid_addresses():
    p, inputs, axes = case("cuda", units=7)
    inputs[0] = replace(inputs[0], pixels=inputs[0].pixels.to(torch.uint8))
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4)
    torch.testing.assert_close(runner(inputs), expected(p, inputs))
    # Bad categories and routes must never read neighboring tables or images.
    inputs[0].pixels.fill_(255)
    assert torch.isnan(runner(inputs)).all()
    inputs[0].variable_ids.fill_(2**40)
    inputs[0].table_ids.fill_(-1)
    assert torch.isnan(runner(inputs)).all()
    torch.cuda.synchronize()


def test_categorical_payload_and_mode_guards():
    p, inputs, axes = case("cpu", logits=True)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree")
    with pytest.raises(ValueError, match="Categorical input bindings"):
        runner.reference([inputs[0].materialize(), *inputs[1:]])
    with pytest.raises(ValueError, match="modes must match"):
        runner.reference([replace(inputs[0], logits=False), *inputs[1:]])
    with pytest.raises(ValueError, match="shapes, dtypes"):
        runner.reference([replace(inputs[0], tables=inputs[0].tables[:, :-1].contiguous()), *inputs[1:]])
    with pytest.raises(ValueError, match="CategoricalInput requires"):
        compile_helion(p, inputs, batch_axes=axes)
    with pytest.raises(ValueError, match="Categorical tensors must be contiguous"):
        replace(inputs[0], pixels=inputs[0].pixels.T)


def test_categorical_autograd_guard():
    p, inputs, axes = case("cpu", logits=True)
    inputs[0].tables.requires_grad_(True)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree")
    with pytest.raises(ValueError, match="inference only"):
        runner.reference(inputs)
    with torch.no_grad():
        torch.testing.assert_close(runner.reference(inputs), expected(p, inputs))


@pytest.mark.parametrize("grid_order", ["roots", "samples"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_isolated_lookup_partial_sample_unit_and_fold_tiles(grid_order, device):
    import helion

    from extended_einsum.backends.helion_categorical import _lookup_kernel

    _, inputs, _ = case(device)
    binding = replace(inputs[0], variable_ids=inputs[0].variable_ids[:7], table_ids=inputs[0].table_ids[:7])
    kernel, _ = _lookup_kernel(binding, sample_block=4, unit_block=4, grid_order=grid_order)
    if device == "cpu":
        kernel = helion.kernel(ref_mode=helion.RefMode.EAGER)(kernel.fn)
    torch.testing.assert_close(kernel(*binding.tensors), binding.materialize(), atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_sample_first_categorical_subtrees(device):
    p, inputs, axes = case(device)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, grid_order="samples", reduction_block_size=4, dot_precision="tf32x3")
    actual = runner.reference(inputs) if device == "cpu" else runner(inputs)
    torch.testing.assert_close(actual, expected(p, inputs))
    assert "for chunk, root in hl.grid" in runner.source


def test_invalid_grid_order():
    p, inputs, axes = case("cpu")
    with pytest.raises(ValueError, match="grid_order must"):
        compile_helion(p, inputs, batch_axes=axes, schedule="subtree", grid_order="invalid")
    with pytest.raises(ValueError, match="requires the subtree"):
        compile_helion(p, inputs, batch_axes=axes, grid_order="samples")


@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("log_values", [False, True])
@pytest.mark.parametrize("mode", MODES[1:])
def test_pipelined_categorical_reference_modes(logits, log_values, mode):
    p, inputs, axes = case("cpu", logits=logits, log_values=log_values)
    p = replace(p, stability_mode=mode)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, reduction_block_size=4, pipeline_stages=2)
    torch.testing.assert_close(runner.reference(inputs), expected(p, inputs), atol=1e-4, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("pipeline_stages", [1, 3])
def test_pipelined_categorical_256_units_updates_and_invalid_addresses(pipeline_stages):
    p, inputs, axes = case("cuda", units=256, categories=256, batch=33, depth=2)
    inputs[0] = replace(inputs[0], pixels=inputs[0].pixels.to(torch.uint8))
    inputs[0].pixels[0].fill_(255)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=32, num_warps=16, reduction_block_size=32, pipeline_stages=pipeline_stages, dot_precision="tf32x3")
    with torch.no_grad():
        torch.testing.assert_close(runner(inputs), expected(p, inputs), atol=1e-4, rtol=1e-6)
        inputs[0].pixels.copy_(inputs[0].pixels.roll(1, 0))
        inputs[0].tables.add_(0.1)
        inputs[0].variable_ids.copy_(inputs[0].variable_ids.roll(1))
        torch.testing.assert_close(runner(inputs), expected(p, inputs), atol=1e-4, rtol=1e-6)
        inputs[0].variable_ids.fill_(2**40)
        inputs[0].table_ids.fill_(-1)
        assert torch.isnan(runner(inputs)).all()
        torch.cuda.synchronize()


@pytest.mark.parametrize("stages", [0, 5, 1.5, True])
def test_invalid_pipeline_stages(stages):
    p, inputs, axes = case("cpu")
    with pytest.raises(ValueError, match="pipeline_stages must"):
        compile_helion(p, inputs, batch_axes=axes, schedule="subtree", pipeline_stages=stages)


def test_panel_optimization_requires_panels():
    p, inputs, axes = case("cpu")
    with pytest.raises(ValueError, match="requires subtree scheduling with reduction_block_size"):
        compile_helion(p, inputs, batch_axes=axes, schedule="subtree", pipeline_stages=2)


def test_tf32_rounding_ties_signs_and_nonfinite_values():
    from extended_einsum.backends.helion import _Recorder

    x = torch.tensor([1 + 2**-12, 1 + 2**-11, -1 - 2**-12, -1 - 2**-11, 0.0, float("inf"), float("-inf"), float("nan")])
    rounded = eval(_Recorder(dot_precision="tf32")._dot_operand("x"), {"torch": torch, "x": x})
    expected = torch.tensor([1.0, 1 + 2**-10, -1.0, -1 - 2**-10, 0.0, float("inf"), float("-inf"), float("nan")])
    torch.testing.assert_close(rounded, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
@pytest.mark.parametrize("units", [64, 65])
def test_tf32_avoids_systematic_truncation_bias(device, units):
    p, inputs, axes = case(device, units=units, batch=17, depth=4)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=16, num_warps=8, reduction_block_size=32, dot_precision="tf32")
    with torch.no_grad():
        actual = runner.reference(inputs) if device == "cpu" else runner(inputs)
        reference = expected(p, inputs)
    torch.testing.assert_close(actual, reference, atol=0.01, rtol=0)
    assert abs((actual - reference).mean().item()) < 0.005
