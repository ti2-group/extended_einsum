"""Matched full steps from raw pixels, including softmax/Adam or EM/Anemone.

python -m demo.benchmark_cuda_training --pixels 784 --batch 128 256 --units 32 256 --output training.json
No MNIST download is needed: pixel selection uses synthetic uint8 28x28 images.
Timing uses warmed CUDA graphs; compilation and CPU transfers are excluded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from extended_einsum.tree import balanced_tree, compile_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=784)
    parser.add_argument("--batch", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--units", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--trainer", nargs="+", default=["adam", "em", "anemone"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-compile-reference", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        pixels=args.pixels,
        scope="raw uint8 pixels, routing, normalization, forward, backward, optimizer update; warm CUDA graph device time",
        results=[],
    )
    report["command"] = sys.argv
    report["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path("src/extended_einsum/tree.py"), Path("src/extended_einsum/backends/triton_training.py"))
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        report["results"] = previous["results"]
    for batch in args.batch:
        for units in args.units:
            for trainer in args.trainer:
                if any((row["batch"], row["units"], row["trainer"]) == (batch, units, trainer) for row in report["results"]):
                    continue
                models = [balanced_tree(pixels=args.pixels, units=units, device="cuda", parameterization="logits" if trainer == "adam" else "normalized_log") for _ in range(2)]
                images = torch.randint(256, (batch, args.pixels), device="cuda", dtype=torch.uint8)
                runners = [compile_training(model, batch, trainer=trainer, backend=backend, precision="tf32") for model, backend in zip(models, ("torch", "triton"), strict=True)]
                if not args.no_compile_reference:
                    ref = runners[0]
                    ref.loss = torch.compile(ref.loss, fullgraph=True, dynamic=False)
                    if trainer != "adam":
                        ref.model.partition = torch.compile(ref.model.partition, fullgraph=True, dynamic=False)
                        ref._apply_update = torch.compile(ref._apply_update, fullgraph=True, dynamic=False)
                initial = [copy.deepcopy(m.state_dict()) for m in models]
                losses = [r.step(images).item() for r in runners]
                row = dict(batch=batch, units=units, trainer=trainer, initial_loss=losses, abs_loss_difference=abs(losses[0] - losses[1]), backends=[])
                for model, runner, state in zip(models, runners, initial, strict=True):
                    model.load_state_dict(state)
                    for _ in range(3):
                        runner.step(images)
                    ms = [do_bench_cudagraph(lambda: runner.step(images), rep=args.rep) for _ in range(3)]
                    row["backends"].append(dict(backend=runner.selection.backend, ms=ms, options=runner.selection.options, compiled_reference=not args.no_compile_reference))
                print(row, flush=True)
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                del runners, models, initial
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
