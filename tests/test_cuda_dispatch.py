"""Dispatch keeps stability, gradients, dynamic inputs and optional imports safe."""

from dataclasses import replace

import pytest
import torch

from extended_einsum.cuda import MODES, CategoricalInput, compile_circuit


def inputs(device="cpu", units=9, batch=17, leaves=8):
    g = torch.Generator(device=device).manual_seed(42)
    pixels = torch.randint(7, (batch, leaves + 3), device=device, generator=g)
    tables = torch.randn(4, 7, units, device=device, generator=g).log_softmax(1)
    variables = torch.randperm(leaves + 3, device=device, generator=g)[:leaves]
    banks = torch.arange(leaves, device=device) % 4
    binding = CategoricalInput(pixels, tables, variables, banks)
    weights = [torch.rand(leaves >> i, units, units, device=device, generator=g).softmax(-1) for i in range(leaves.bit_length())]
    return binding, weights


@pytest.mark.parametrize("mode", MODES)
def test_cpu_fallback_has_gradients_and_dynamic_data(mode):
    binding, weights = inputs()
    binding = replace(binding, tables=binding.tables.requires_grad_())
    weights = [w.requires_grad_() for w in weights]
    runner = compile_circuit(binding, weights, stability=mode)
    assert runner.selection.backend == "torch"
    y = runner(binding, weights)
    y.sum().backward()
    assert binding.tables.grad is not None
    assert all(w.grad is not None for w in weights)
    binding = replace(binding, pixels=(binding.pixels + 1) % 7)
    assert not torch.equal(y, runner(binding, weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("units", [9, 65, 256])
def test_triton_modes_routes_zeros_and_grad_fallback(mode, units):
    binding, weights = inputs("cuda", units=units)
    runner = compile_circuit(binding, weights, stability=mode, backend="triton", precision="tf32")
    reference = compile_circuit(binding, weights, stability=mode, backend="torch")
    for offset in [-100.0, 100.0]:
        changed = replace(binding, tables=binding.tables + offset)
        with torch.no_grad():
            torch.testing.assert_close(runner(changed, weights), reference(changed, weights), atol=0.02, rtol=0)
    # Gradient requests made after inference compilation use differentiable Torch.
    changed = replace(binding, tables=binding.tables.requires_grad_())
    runner(changed, weights).sum().backward()
    assert changed.tables.grad is not None
    with torch.no_grad():
        binding.variable_ids.fill_(2**40)
        assert torch.isnan(runner(binding, weights)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_graph_and_current_stream():
    binding, weights = inputs("cuda", units=65)
    runner = compile_circuit(binding, weights, backend="triton", precision="tf32")
    ref = compile_circuit(binding, weights, backend="torch")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        runner(binding, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = runner(binding, weights)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(y, ref(binding, weights), atol=0.01, rtol=0)
            binding.pixels.add_(1).remainder_(7)
            weights[0].mul_(0.9)
    torch.cuda.current_stream().wait_stream(stream)
