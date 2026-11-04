"""Whole-program lowering checks; real CUDA execution is tested when available."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import extended_einsum as xe
from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion import compile_helion
from extended_einsum.backends.torch import TorchBackendFunctions

pytest.importorskip("helion")

spec = importlib.util.spec_from_file_location("benchmark_helion", Path(__file__).parents[1] / "demo" / "benchmark_helion.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

MODES = ["unstable", "scaled_min", "scaled_max", "scaled_sum", "logspace_min", "logspace_max"]


@pytest.mark.parametrize("stability", MODES)
@pytest.mark.parametrize("block_size", [1, 2, 8])
def test_folded_circuit_reference_matches_torch(stability, block_size):
    program, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=2, device="cpu", stability=stability, log_offset=-0.5)
    runner = compile_helion(program, inputs, batch_axes=axes, block_size=block_size, scale_interval=1)
    expected = run_program(translate_to_backend_program(program, TorchBackendFunctions(), scale_interval=1), inputs)
    torch.testing.assert_close(runner.reference(inputs), expected)


@pytest.mark.parametrize("stability", MODES[1:])
@pytest.mark.parametrize("log_offset", [-100.0, 100.0])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_stable_extreme_inputs(stability, log_offset, device):
    program, inputs, axes = benchmark.folded_circuit(batch=3, units=3, depth=3, device=device, stability=stability, log_offset=log_offset)
    runner = compile_helion(program, inputs, batch_axes=axes, block_size=2, scale_interval=1)
    expected = run_program(translate_to_backend_program(program, TorchBackendFunctions(), scale_interval=1), inputs)
    result = runner.reference(inputs) if device == "cpu" else runner(inputs)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("stability", ["scaled_max", "logspace_max"])
@pytest.mark.parametrize("block_size", [1, 2])
def test_codegen_is_one_kernel_with_no_intermediate_global_stores(stability, block_size, tmp_path):
    program, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=1, device="cpu", stability=stability)
    runner = compile_helion(program, inputs, batch_axes=axes, block_size=block_size)
    code = runner.kernel.bind(tuple(inputs)).to_triton_code()
    assert code.count("@triton.jit") == 1
    assert code.count("_launcher(") == 1
    stores = [line.strip() for line in code.splitlines() if "tl.store(" in line]
    assert len(stores) == 1
    assert "tl.store(out " in stores[0]
    assert "input_precision='tf32'" not in code

    # Compile the actual generated source all the way to an SM80 CUDA binary.
    # This requires no GPU/driver and catches invalid Triton shapes that source
    # generation and the reference interpreter alone would both miss.
    pytest.importorskip("triton.backends.nvidia.compiler")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source_path = tmp_path / "generated.py"
    source_path.write_text(code)
    generated_spec = importlib.util.spec_from_file_location("generated_helion", source_path)
    generated = importlib.util.module_from_spec(generated_spec)
    generated_spec.loader.exec_module(generated)
    binaries = []

    def compile_cuda(fn, grid, *args, **kwargs):
        signature, constants = {}, {}
        for i, (name, arg) in enumerate(zip(fn.arg_names, args, strict=True)):
            if i in fn.constexprs:
                constants[name] = arg
            else:
                signature[name] = "*" + {torch.float32: "fp32", torch.int64: "i64", torch.int32: "i32"}[arg.dtype]
        options = {key: kwargs[key] for key in ("num_warps", "num_stages")}
        compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants), target=GPUTarget("cuda", 80, 32), options=options)
        binaries.append(compiled.asm["cubin"])

    generated.extended_einsum_kernel(*inputs, _launcher=compile_cuda)
    assert len(binaries) == 1 and binaries[0]


def test_batch_axis_moves_through_layout_and_broadcasts():
    x = torch.rand(3, 7, 5) + 0.1
    a = xe.array(x)
    selected = xe.select(a, 1, axis=0)
    stacked = xe.stack([selected, selected], axis=0)
    expression = xe.log(xe.einsum("bi->ib", xe.select(stacked, 0, axis=0)))
    program, arrays = xe.extract_program(expression, "logspace_max")
    inputs = [a.backend_array for a in arrays]
    runner = compile_helion(program, inputs, batch_axes=[1], block_size=2)
    torch.testing.assert_close(runner.reference(inputs), x[1].T.log())


def test_int32_routing_indices():
    program, inputs, axes = benchmark.folded_circuit(batch=3, units=3, depth=1, device="cpu", stability="scaled_max")
    inputs = [a.int() if a.dtype == torch.int64 else a for a in inputs]
    runner = compile_helion(program, inputs, batch_axes=axes, block_size=2)
    expected = run_program(translate_to_backend_program(program, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner.reference(inputs), expected)


@pytest.mark.parametrize(
    "expression,axes",
    [
        (lambda a: xe.einsum("bi->i", a), [0]),
        (lambda a: xe.softmax(a, axis=0), [0]),
        (lambda a: xe.select(a, 0, axis=0), [0]),
    ],
)
def test_cross_batch_operations_are_rejected(expression, axes):
    program, arrays = xe.extract_program(expression(xe.array(torch.rand(7, 5))), "unstable")
    with pytest.raises(NotImplementedError, match="batch"):
        compile_helion(program, [a.backend_array for a in arrays], batch_axes=axes)


def test_inference_and_specialization_guards():
    x = torch.rand(7, 5, requires_grad=True)
    program, _ = xe.extract_program(xe.exp(xe.array(x)), "scaled_max")
    runner = compile_helion(program, [x], batch_axes=[0])
    with pytest.raises(ValueError, match="inference only"):
        runner.reference([x])
    with torch.no_grad():
        torch.testing.assert_close(runner.reference([x]), x.exp())
    with pytest.raises(ValueError, match="shapes, dtypes"):
        runner.reference([x.detach().double()])
    with pytest.raises(ValueError, match="power of two"):
        compile_helion(program, [x], batch_axes=[0], block_size=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for compiled Helion execution")
@pytest.mark.parametrize("stability", MODES)
@pytest.mark.parametrize("block_size", [1, 4])
def test_compiled_cuda_matches_stable_torch(stability, block_size):
    program, inputs, axes = benchmark.folded_circuit(batch=7, units=5, depth=2, device="cuda", stability=stability, log_offset=-0.5)
    runner = compile_helion(program, inputs, batch_axes=axes, block_size=block_size)
    expected = run_program(translate_to_backend_program(program, TorchBackendFunctions()), inputs)
    torch.testing.assert_close(runner(inputs), expected, atol=5e-5, rtol=5e-5)
