"""Subtree boundaries preserve stability and specialize circuit routing safely."""

import pytest
import torch

import extended_einsum as xe
from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion import compile_helion
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.interface.tensor_expression import Parameter
from tests.test_helion import MODES, benchmark


@pytest.mark.parametrize("stability", MODES[1:])
@pytest.mark.parametrize("subtree_depth", [1, 2, 4])
def test_subtree_reference_matches_stable_program(stability, subtree_depth):
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=3, device="cpu", stability=stability)
    runner = compile_helion(p, inputs, batch_axes=axes, block_size=4, schedule="subtree", subtree_depth=subtree_depth)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    actual = runner.reference(inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert len(runner.stages) == {1: 4, 2: 2, 4: 1}[subtree_depth]


def test_irregular_routes_and_shared_descendants():
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=2, device="cpu", stability="logspace_max")
    inputs[3].copy_(torch.tensor([3, 1]))
    inputs[4].copy_(torch.tensor([0, 3]))  # leaf 3 is reused by both parents
    inputs[6].fill_(1)
    inputs[7].fill_(0)
    runner = compile_helion(p, inputs, batch_axes=axes, block_size=4, schedule="subtree", subtree_depth=2)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner.reference(inputs), expected)
    # Data may change without specializing values or rerouting the graph.
    changed = [inputs[0] + 0.25, *inputs[1:]]
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), changed)
    torch.testing.assert_close(runner.reference(changed), expected)


def test_static_routing_guards():
    p, inputs, axes = benchmark.folded_circuit(batch=3, units=3, depth=2, device="cpu", stability="scaled_max")
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree")
    replacement = list(inputs)
    replacement[3] = inputs[3].clone()
    with pytest.raises(ValueError, match="routing inputs are static"):
        runner.reference(replacement)
    inputs[3].add_(0)
    with pytest.raises(ValueError, match="routing inputs are static"):
        runner.reference(inputs)


@pytest.mark.parametrize("stability", MODES[1:])
@pytest.mark.parametrize("offset", [-100.0, 100.0])
def test_panelled_reference_with_partial_tiles_and_irregular_routes(stability, offset):
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=9, depth=3, device="cpu", stability=stability, log_offset=offset)
    inputs[3].copy_(torch.tensor([6, 0, 4, 1]))
    inputs[4].copy_(torch.tensor([3, 5, 6, 2]))
    runner = compile_helion(p, inputs, batch_axes=axes, block_size=4, schedule="subtree", reduction_block_size=4)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner.reference(inputs), expected)
    assert "for v" in runner.source and "torch.gather" in runner.source


@pytest.mark.parametrize("layout", ["io", "oi"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_panelled_nonsquare_shared_weights(layout, device):
    # Unequal widths and partial panels exercise both weight axes and masks.
    x = xe.array(torch.randn(2, 7, 33, device=device) - 100)
    weight_shape = (33, 17) if layout == "io" else (17, 33)
    weight = Parameter(xe.array(torch.rand(weight_shape, device=device) + 0.1))
    leaves = xe.einsum(f"fbi,{layout}->fbo", xe.exp(x), weight)
    root_shape = (17, 5) if layout == "io" else (5, 17)
    root_weight = Parameter(xe.array(torch.rand(root_shape, device=device) + 0.1))
    expression = xe.log(xe.einsum(f"bi,bi,{layout}->bo", xe.select(leaves, 0), xe.select(leaves, 1), root_weight))
    p, wrapped = xe.extract_program(expression, "logspace_max")
    inputs = [a.backend_array for a in wrapped]
    axes = [1 if a.ndim == 3 else None for a in inputs]
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, reduction_block_size=16, dot_precision="tf32x3")
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    actual = runner.reference(inputs) if device == "cpu" else runner(inputs)
    torch.testing.assert_close(actual, expected)


def test_panelled_computed_weights_keep_full_matrix_evaluation():
    x = xe.array(torch.randn(2, 7, 9) - 100)
    weight = Parameter(xe.array(torch.rand(9, 7) + 0.1))
    leaves = xe.einsum("fbi,io->fbo", xe.exp(x), xe.softmax(weight, axis=0))
    root_weight = Parameter(xe.array(torch.rand(7, 5) + 0.1))
    expression = xe.log(xe.einsum("bi,bi,io->bo", xe.select(leaves, 0), xe.select(leaves, 1), root_weight))
    p, wrapped = xe.extract_program(expression, "logspace_max")
    inputs = [a.backend_array for a in wrapped]
    axes = [1 if a.ndim == 3 else None for a in inputs]
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, reduction_block_size=4)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner.reference(inputs), expected)
    assert "torch.mm" in runner.source and "for v" in runner.source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("stability", ["scaled_max", "logspace_max"])
@pytest.mark.parametrize("units", [65, 128, 256])
@pytest.mark.parametrize("dot_precision", ["ieee", "tf32x3"])
def test_panelled_cuda_larger_widths(stability, units, dot_precision):
    p, inputs, axes = benchmark.folded_circuit(batch=17, units=units, depth=2, device="cuda", stability=stability)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=16, num_warps=8, reduction_block_size=32, dot_precision=dot_precision)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner(inputs), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("stability", MODES[1:])
@pytest.mark.parametrize("offset", [-100.0, 100.0])
def test_panelled_cuda_extreme_partial_units(stability, offset):
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=17, depth=3, device="cuda", stability=stability, log_offset=offset)
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=4, reduction_block_size=8, dot_precision="tf32x3")
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner(inputs), expected)


@pytest.mark.parametrize("panel", [0, -1, 3])
def test_invalid_reduction_panel(panel):
    p, inputs, axes = benchmark.folded_circuit(batch=3, units=5, depth=1, device="cpu", stability="logspace_max")
    with pytest.raises(ValueError, match="positive power of two"):
        compile_helion(p, inputs, batch_axes=axes, schedule="subtree", reduction_block_size=panel)


def test_reduction_panel_requires_subtree_schedule():
    p, inputs, axes = benchmark.folded_circuit(batch=3, units=5, depth=1, device="cpu", stability="logspace_max")
    with pytest.raises(ValueError, match="requires the subtree schedule"):
        compile_helion(p, inputs, batch_axes=axes, reduction_block_size=4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_shared_unfolded_weights_and_changing_unit_widths(device):
    x = xe.array(torch.randn(2, 7, 3, device=device) - 100)
    weight = Parameter(xe.array(torch.rand(3, 5, device=device) + 0.1))
    leaves = xe.einsum("fbi,io->fbo", xe.exp(x), xe.softmax(weight, axis=1))
    root_weight = Parameter(xe.array(torch.rand(2, 5, device=device) + 0.1))
    expression = xe.log(xe.einsum("bi,bi,oi->bo", xe.select(leaves, 0), xe.select(leaves, 1), root_weight))
    p, wrapped = xe.extract_program(expression, "logspace_max")
    inputs = [a.backend_array for a in wrapped]
    axes = [1 if tuple(a.shape) == (2, 7, 3) else None for a in inputs]
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", subtree_depth=1)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    actual = runner.reference(inputs) if device == "cpu" else runner(inputs)
    torch.testing.assert_close(actual, expected)


def test_subtree_rejects_unstable_or_noncontiguous_inputs():
    p, inputs, axes = benchmark.folded_circuit(batch=3, units=3, depth=2, device="cpu", stability="unstable")
    with pytest.raises(NotImplementedError, match="stable evaluation"):
        compile_helion(p, inputs, batch_axes=axes, schedule="subtree")
    p, inputs, axes = benchmark.folded_circuit(batch=3, units=3, depth=2, device="cpu", stability="scaled_max")
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree")
    changed = [inputs[0].transpose(0, 1).contiguous().transpose(0, 1), *inputs[1:]]
    with pytest.raises(ValueError, match="remain contiguous"):
        runner.reference(changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("stability", MODES[1:])
@pytest.mark.parametrize("offset", [-100.0, 100.0])
def test_subtree_cuda_extreme_values(stability, offset):
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=3, device="cuda", stability=stability, log_offset=offset)
    runner = compile_helion(p, inputs, batch_axes=axes, block_size=4, schedule="subtree")
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    actual = runner(inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("stability", ["scaled_max", "logspace_max"])
@pytest.mark.parametrize("units", [31, 64])
@pytest.mark.parametrize("dot_precision", ["ieee", "tf32x3"])
def test_subtree_cuda_larger_units(stability, units, dot_precision):
    p, inputs, axes = benchmark.folded_circuit(batch=17, units=units, depth=3, device="cuda", stability=stability)
    runner = compile_helion(p, inputs, batch_axes=axes, block_size=16, schedule="subtree", dot_precision=dot_precision)
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner(inputs), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_subtree_cuda_nonaffine_routing():
    p, inputs, axes = benchmark.folded_circuit(batch=7, units=7, depth=3, device="cuda", stability="logspace_max")
    inputs[3].copy_(torch.tensor([6, 0, 4, 1], device="cuda"))
    inputs[4].copy_(torch.tensor([3, 5, 6, 2], device="cuda"))
    runner = compile_helion(p, inputs, batch_axes=axes, schedule="subtree", block_size=16, dot_precision="tf32x3")
    expected = run_program(translate_to_backend_program(p, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner(inputs), expected)
