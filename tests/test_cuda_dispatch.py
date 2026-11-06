"""Compiled Triton dispatch: whole-program matching and explicit rejection."""

from dataclasses import replace

import pytest
import torch

import extended_einsum as xe
from extended_einsum.backend_translation import run_program, translate_to_backend_program
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.backends.triton import MODES
from extended_einsum.compiled import UnsupportedProgramError
from extended_einsum.inputs import CategoricalInput
from extended_einsum.interface.tensor_expression import Parameter

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(device="cpu", units=9, batch=17, leaves=8):
    g = torch.Generator(device=device).manual_seed(42)
    pixels = torch.randint(7, (batch, leaves + 3), device=device, generator=g)
    tables = torch.randn(4, 7, units, device=device, generator=g).log_softmax(1)
    variables = torch.randperm(leaves + 3, device=device, generator=g)[:leaves]
    banks = torch.arange(leaves, device=device) % 4
    binding = CategoricalInput(pixels, tables, variables, banks)
    weights = [torch.rand(leaves >> i, units, units, device=device, generator=g).softmax(-1) for i in range(leaves.bit_length())]
    return binding, weights


def program_for(binding, weights, mode="logspace_max", *, log=True):
    source = xe.array(binding, backend="triton")
    current = xe.exp(source) if log else source
    routing = []
    for level, weight in enumerate(weights):
        w = Parameter(xe.array(weight, backend="triton"))
        if level:
            f = weight.shape[0]
            ids = [torch.arange(side, 2 * f, 2, device=binding.device) for side in (0, 1)]
            routing.extend(ids)
            a, b = [xe.take(current, xe.array(i, backend="triton")) for i in ids]
            current = xe.einsum("fbi,fbi,foi->fbo", a, b, w)
        else:
            current = xe.einsum("fbi,foi->fbo", current, w)
    expression = xe.log(xe.select(current, 0))
    program, wrapped = xe.extract_program(expression, stability_mode=mode)
    values = [v.backend_array for v in wrapped]
    constants = [i for i, v in enumerate(values) if any(v is r for r in routing)]
    return program, values, constants, expression


def reference(program, values):
    actual = [v.materialize() if isinstance(v, CategoricalInput) else v for v in values]
    return run_program(translate_to_backend_program(program, TorchBackendFunctions()), actual)


def compile_tree(binding, weights, mode="logspace_max", **options):
    program, values, constants, expression = program_for(binding, weights, mode)
    runner = xe.compile_program(program, values, backend="triton", constant_inputs=constants, **options)
    return runner, program, values, constants, expression


@pytest.mark.parametrize("mode", MODES)
def test_cpu_is_rejected(mode):
    binding, weights = inputs()
    with pytest.raises(UnsupportedProgramError, match="CUDA"):
        compile_tree(binding, weights, mode, precision="tf32")


def test_unsupported_program_and_precision_are_rejected():
    expression = xe.sin(xe.array(torch.ones(3), backend="triton"))
    with pytest.raises(UnsupportedProgramError, match="stability"):
        expression.materialize()
    with pytest.raises(UnsupportedProgramError, match="requires precision"):
        expression.materialize(stability_mode="logspace_max")
    with pytest.raises(UnsupportedProgramError, match="unsupported 'sin'"):
        expression.materialize(stability_mode="logspace_max", precision="tf32")


@CUDA
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("units", [2, 9, 65, 256])
def test_triton_modes_routes_and_no_gradient_fallback(mode, units):
    binding, weights = inputs("cuda", units=units)
    runner, program, values, _, _ = compile_tree(binding, weights, mode, precision="tf32")
    data_id = next(i for i, v in enumerate(values) if v is binding)
    for offset in [-100.0, 100.0]:
        changed = list(values)
        changed[data_id] = replace(binding, tables=binding.tables + offset)
        torch.testing.assert_close(runner(changed), reference(program, changed), atol=0.02, rtol=0)
    changed = list(values)
    changed[data_id] = replace(binding, tables=binding.tables.requires_grad_())
    with pytest.raises(UnsupportedProgramError, match="inference only"):
        runner(changed)
    with torch.no_grad():
        binding.variable_ids.fill_(2**40)
        assert torch.isnan(runner(values)).all()


@CUDA
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_dense_inputs_and_materialize(log, mode):
    binding, weights = inputs("cuda", units=9, leaves=4)
    dense = binding.materialize()
    dense = dense if log else dense.exp()
    program, values, constants, expression = program_for(dense, weights, mode, log=log)
    runner = xe.compile_program(program, values, backend="triton", constant_inputs=constants, precision="tf32")
    expected = reference(program, values)
    torch.testing.assert_close(runner(values), expected, atol=0.01, rtol=0)
    torch.testing.assert_close(expression.materialize(stability_mode=mode, constant_inputs=constants, precision="tf32").backend_array, expected, atol=0.01, rtol=0)


@CUDA
def test_triton_graph_and_current_stream():
    binding, weights = inputs("cuda", units=65)
    runner, program, values, _, _ = compile_tree(binding, weights, precision="tf32")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        runner(values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = runner(values)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(y, reference(program, values), atol=0.01, rtol=0)
            binding.pixels.add_(1).remainder_(7)
            weights[0].mul_(0.9)
    torch.cuda.current_stream().wait_stream(stream)


@CUDA
def test_rejects_mutated_routing_metadata_and_implicit_constants():
    binding, weights = inputs("cuda", units=9)
    runner, program, values, constants, _ = compile_tree(binding, weights, precision="tf32")
    with pytest.raises(UnsupportedProgramError, match="constant_inputs"):
        xe.compile_program(program, values, backend="triton", precision="tf32")
    changed = list(values)
    changed[constants[0]] = changed[constants[0]].clone()
    with pytest.raises(UnsupportedProgramError, match="routing changed"):
        runner(changed)
    values[constants[0]].add_(1)
    with pytest.raises(UnsupportedProgramError, match="routing changed"):
        runner(values)
    with pytest.raises(UnsupportedProgramError, match="even/odd"):
        xe.compile_program(program, values, backend="triton", constant_inputs=constants, precision="tf32")


@CUDA
def test_compilation_and_execution_reject_gradients():
    binding, weights = inputs("cuda", units=128, batch=128, leaves=4)
    binding = replace(binding, tables=binding.tables.requires_grad_())
    with pytest.raises(UnsupportedProgramError, match="inference only"):
        compile_tree(binding, weights, precision="tf32")
    with torch.no_grad():
        runner, program, values, _, _ = compile_tree(binding, weights, "scaled_max", precision="tf32")
        assert runner.selection.backend == "triton"
        assert runner.selection.options["pipeline_stages"] == 3
        assert torch.isfinite(runner(values)).all()
    with pytest.raises(UnsupportedProgramError, match="inference only"):
        runner(values)


@CUDA
def test_inference_consumes_normalized_probability_floor():
    floor = torch.tensor(torch.finfo(torch.float32).tiny, device="cuda").log()
    table = torch.stack([torch.stack([floor * 0, floor]), torch.stack([floor, floor * 0])]).expand(2, -1, -1).contiguous()
    binding = CategoricalInput(torch.zeros((17, 2), dtype=torch.uint8, device="cuda"), table, torch.arange(2, device="cuda"))
    leaf = table.exp().clone()
    root = torch.stack([floor, floor * 0]).exp().reshape(1, 1, 2).expand(1, 2, 2).contiguous()
    with torch.no_grad():
        runner, _, values, _, _ = compile_tree(binding, [leaf, root], precision="tf32")
        torch.testing.assert_close(runner(values), floor.expand(17, 2), atol=3e-5, rtol=0)


@CUDA
def test_input_metadata_precision_and_scaled_training_rejection():
    binding, weights = inputs("cuda", units=9)
    runner, program, values, constants, _ = compile_tree(binding, weights, precision="tf32")
    w_id = next(i for i, v in enumerate(values) if v is weights[0])
    for value in (weights[0].half(), weights[0].transpose(1, 2), weights[0].cpu()):
        changed = list(values)
        changed[w_id] = value
        with pytest.raises(UnsupportedProgramError):
            runner(changed)
        with pytest.raises(UnsupportedProgramError):
            xe.compile_program(program, changed, backend="triton", constant_inputs=constants, precision="tf32")
    from extended_einsum.tree import balanced_tree, compile_training

    model = balanced_tree(pixels=4, units=9, categories=7, device="cuda")
    with pytest.raises(UnsupportedProgramError, match="scaled training is not supported"):
        compile_training(model, 17, stability="scaled_max")
