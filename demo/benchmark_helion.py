"""Check and benchmark stable folded circuits compiled with Helion.

CPU lowering check:
    python demo/benchmark_helion.py --reference-only --batch 17 --units 5
CUDA correctness and timings (compilation excluded):
    python demo/benchmark_helion.py --batch 256 --units 16 --depth 3
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from extended_einsum.backend_translation import translate_to_backend_program
from extended_einsum.backend_translation.runtime import run_program
from extended_einsum.backends.helion import CategoricalInput, compile_helion
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.language.rich_instruction import RichInstruction
from extended_einsum.language.rich_operators import OperatorEinsum, OperatorExp, OperatorLog, OperatorSelect, OperatorTake
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import StabilityMode


def folded_circuit(
    *,
    batch: int,
    units: int,
    depth: int,
    device: str,
    stability: StabilityMode,
    log_offset: float = -100.0,
    categorical: CategoricalInput | None = None,
) -> tuple[RichProgram, list[torch.Tensor | CategoricalInput], list[int | None]]:
    """Binary sum/product tree with folded routing and shared linear weights."""
    if min(batch, units, depth) < 1:
        raise ValueError("batch, units, and depth must be positive.")
    generator = torch.Generator(device=device).manual_seed(42)
    leaves = 2**depth
    if categorical is not None and categorical.shape != (leaves, batch, units):
        raise ValueError("Categorical input must match the folded circuit's leaves, batch, and units.")
    inputs = [categorical if categorical is not None else torch.randn(leaves, batch, units, generator=generator, device=device) + log_offset]
    axes: list[int | None] = [1]
    weights = []
    routing = []
    for level in range(depth + 1):
        folds = leaves // (2**level)
        weights.append(len(inputs))
        weight = torch.rand(folds, units, units, generator=generator, device=device) + 0.1
        inputs.append(weight / weight.sum(-1, keepdim=True))
        axes.append(None)
        if level:
            routing.append((len(inputs), len(inputs) + 1))
            inputs.extend([torch.arange(0, 2 * folds, 2, device=device), torch.arange(1, 2 * folds, 2, device=device)])
            axes.extend([None, None])

    shapes = [tuple(x.shape) for x in inputs]
    instructions = []

    def append(operator, args, shape):
        result = len(inputs) + len(instructions)
        instructions.append(RichInstruction(operator, tuple(args)))
        shapes.append(shape)
        return result

    current = append(OperatorExp(), (0,), shapes[0])
    current = append(OperatorEinsum("fbi,foi->fbo"), (current, weights[0]), shapes[0])
    for level, (left_ids, right_ids) in enumerate(routing, start=1):
        shape = (leaves // (2**level), batch, units)
        left = append(OperatorTake(0), (current, left_ids), shape)
        right = append(OperatorTake(0), (current, right_ids), shape)
        current = append(OperatorEinsum("fbi,fbi,foi->fbo"), (left, right, weights[level]), shape)
    current = append(OperatorSelect(0, 0), (current,), (batch, units))
    append(OperatorLog(), (current,), (batch, units))
    program = RichProgram(
        instructions,
        len(inputs),
        stability,
        ["dense"] * len(shapes),
        shapes,
        frozenset(weights),
    )
    return program, inputs, axes


def cuda_time_ms(fn: Callable[[], torch.Tensor], repeats: int, *, timing: str = "cuda-graph") -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if timing == "cuda-graph":
        from triton.testing import do_bench_cudagraph

        # Capture warmed calls to separate device work from Python dispatch.
        return do_bench_cudagraph(fn, rep=20 * repeats, return_mode="median")
    measurements = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / 20)
    return statistics.median(measurements)


def kernel_stage_reference(stage, inputs):
    import helion

    return helion.kernel(ref_mode=helion.RefMode.EAGER)(stage.kernel.fn)(*inputs)


def kernel_resources(kernel, inputs):
    from helion.runtime import default_launcher

    launches = []
    device = torch.cuda.get_device_properties(inputs[0].device)
    occupancy = ctypes.CDLL("libcuda.so.1").cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    occupancy.restype = ctypes.c_int

    def record_launch(fn, grid, *args, **kwargs):
        compiled = default_launcher(fn, grid, *args, **kwargs)
        blocks = ctypes.c_int()
        threads = compiled.metadata.num_warps * 32
        status = occupancy(ctypes.byref(blocks), compiled.function, threads, compiled.metadata.shared)
        if status:
            raise RuntimeError(f"CUDA occupancy query failed with driver error {status}.")
        launches.append(
            {
                "grid": list(grid),
                "registers_per_thread": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "num_warps": compiled.metadata.num_warps,
                "max_resident_blocks_per_sm": blocks.value,
                # Resource ceiling, not a measurement of achieved occupancy.
                "theoretical_occupancy": blocks.value * threads / device.max_threads_per_multi_processor,
                "grid_occupancy_ceiling": min(math.prod(grid), blocks.value * device.multi_processor_count) * threads / (device.multi_processor_count * device.max_threads_per_multi_processor),
                "grid_waves_at_max_residency": math.prod(grid) / (blocks.value * device.multi_processor_count),
            }
        )
        return compiled

    compiled = kernel.bind(tuple(inputs)).compile_config()
    result = compiled(*inputs, _launcher=record_launch)
    torch.cuda.synchronize()
    return result, launches


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=257)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--stability", choices=["scaled_max", "scaled_sum", "scaled_min", "logspace_max", "logspace_min"], default="logspace_max")
    parser.add_argument("--scale-interval", type=int, default=3)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--schedule", choices=["whole", "subtree"], default="whole")
    parser.add_argument("--subtree-depth", type=int, default=2)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--dot-precision", choices=["ieee", "tf32x3"], default="ieee")
    parser.add_argument("--reduction-block-size", type=int, help="Stream direct matrix weights in panels along the contraction axis.")
    parser.add_argument("--pipeline-stages", type=int, default=1, help="Software pipeline depth for subtree reduction panels.")
    parser.add_argument("--timing", choices=["cuda-graph", "cuda-events"], default="cuda-graph", help="CUDA graphs isolate device time; CUDA events include gaps from Python dispatch.")
    parser.add_argument("--reference-only", action="store_true", help="Check Helion's CPU tile interpreter; do not report GPU timings.")
    parser.add_argument("--resources", action="store_true", help="Record per-stage time, registers, spills, shared memory, and the CUDA occupancy ceiling.")
    parser.add_argument("--output", type=Path, help="Save measurements as JSON.")
    parser.add_argument("--source-dir", type=Path, help="Save generated Helion and Triton source for inspection.")
    args = parser.parse_args()
    if not args.reference_only and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --reference-only to check lowering on CPU.")
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if args.resources and args.reference_only:
        parser.error("--resources requires compiled CUDA execution.")
    device = "cpu" if args.reference_only else "cuda"
    torch.set_float32_matmul_precision("highest")
    program, inputs, axes = folded_circuit(batch=args.batch, units=args.units, depth=args.depth, device=device, stability=args.stability)
    backend_program = translate_to_backend_program(program, TorchBackendFunctions(), scale_interval=args.scale_interval)

    def eager():
        return run_program(backend_program, inputs)

    expected = eager()
    # FP64 uses the same stable program and represents the original FP32 inputs.
    precise_inputs = [x.double() if x.is_floating_point() else x for x in inputs]
    precise = run_program(backend_program, precise_inputs)
    torch.testing.assert_close(expected.double(), precise, atol=1e-4, rtol=1e-6)
    if not torch.isfinite(expected).all():
        raise AssertionError("The stable Torch reference contains non-finite outputs.")
    report = {
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": torch.__version__,
        "batch": args.batch,
        "units": args.units,
        "depth": args.depth,
        "stability": args.stability,
        "scale_interval": args.scale_interval,
        "schedule": args.schedule,
        "subtree_depth": args.subtree_depth if args.schedule == "subtree" else None,
        "num_warps": args.num_warps,
        "dot_precision": args.dot_precision,
        "reduction_block_size": args.reduction_block_size,
        "pipeline_stages": args.pipeline_stages,
        # Count multiply and add separately. This excludes stability operations
        # and the extra instructions used to implement TF32x3 products.
        "dense_contraction_flops": 2 * (2 ** (args.depth + 1) - 1) * args.batch * args.units**2,
        "execution": "helion-reference" if args.reference_only else "compiled-cuda",
        "timing": None if args.reference_only else args.timing,
        "torch_fp64_max_abs_error": (expected.double() - precise).abs().max().item(),
        "variants": [],
    }
    if not args.reference_only:
        compiled = torch.compile(lambda tensors: run_program(backend_program, tensors), fullgraph=True)
        torch.testing.assert_close(compiled(inputs), expected, atol=1e-4, rtol=1e-6)
        report["torch_eager_ms"] = cuda_time_ms(eager, args.repeats, timing=args.timing)
        report["torch_compile_ms"] = cuda_time_ms(lambda: compiled(inputs), args.repeats, timing=args.timing)
    for block_size in args.block_sizes:
        kernel = compile_helion(
            program,
            inputs,
            batch_axes=axes,
            block_size=block_size,
            scale_interval=args.scale_interval,
            schedule=args.schedule,
            subtree_depth=args.subtree_depth,
            num_warps=args.num_warps,
            dot_precision=args.dot_precision,
            reduction_block_size=args.reduction_block_size,
            pipeline_stages=args.pipeline_stages,
        )
        actual = kernel.reference(inputs) if args.reference_only else kernel(inputs)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-6)
        variant = {"block_size": block_size, "max_abs_error": (actual - expected).abs().max().item()}
        variant["max_abs_error_fp64"] = (actual.double() - precise).abs().max().item()
        variant["kernel_count"] = len(kernel.stages) if args.schedule == "subtree" else 1
        if not args.reference_only:
            variant["ms"] = cuda_time_ms(lambda: kernel(inputs), args.repeats, timing=args.timing)
            variant["speedup_vs_torch_compile"] = report["torch_compile_ms"] / variant["ms"]
            variant["useful_dense_tflops"] = report["dense_contraction_flops"] / (variant["ms"] * 1e9)
        report["variants"].append(variant)
        if args.source_dir or args.resources:
            if args.source_dir:
                args.source_dir.mkdir(parents=True, exist_ok=True)
                (args.source_dir / f"helion_b{block_size}.py").write_text(kernel.source)
            if args.schedule == "subtree":
                outputs = []
                stage_resources = []
                for stage_id, stage in enumerate(kernel.stages):
                    stage_inputs = stage.arguments(inputs, outputs)
                    if args.resources:
                        output, launches = kernel_resources(stage.kernel, stage_inputs)
                        stage_ms = cuda_time_ms(lambda: stage.kernel(*stage_inputs), args.repeats, timing=args.timing)
                        stage_resources.append({"stage": stage_id, "roots": stage.roots, "frontier_bytes": output.numel() * output.element_size(), "isolated_ms": stage_ms, "launches": launches})
                    else:
                        output = stage.kernel(*stage_inputs) if not args.reference_only else kernel_stage_reference(stage, stage_inputs)
                    outputs.append(output)
                    if args.source_dir:
                        code = stage.kernel.bind(tuple(stage_inputs)).to_triton_code()
                        (args.source_dir / f"triton_b{block_size}_s{stage_id}.py").write_text(code)
                if args.resources:
                    variant["resources"] = stage_resources
            else:
                if args.resources:
                    _, variant["resources"] = kernel_resources(kernel.kernel, inputs)
                if args.source_dir:
                    code = kernel.kernel.bind(tuple(inputs)).to_triton_code()
                    (args.source_dir / f"triton_b{block_size}.py").write_text(code)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
