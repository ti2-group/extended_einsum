from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (
    RESULTS_DIR,
    Configuration,
    Variant,
    parse_ints,
    parse_layers,
    parse_single,
    run_configuration,
    run_isolated,
    shuffled_blocks,
)

VARIANTS = {
    "cirkit": Variant("cirkit", semiring="lse-sum"),
    "xe": Variant("xe", semiring="scaled-max", shift_mode="xe", optimize_group_order=True),
}

# Explicitly list only configurations established to fit in GPU memory.
SAFE_GRID = {
    "cp": {
        "quad-tree-2": {256: (128, 256, 512, 1024), 512: (128, 256, 512, 1024)},
        "quad-graph": {256: (128, 256, 512), 512: (128, 256, 512)},
    },
    "tucker": {
        "quad-tree-2": {256: (32, 64), 512: (32, 64)},
        "quad-graph": {256: (32, 64), 512: (32, 64)},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication speedup benchmark: production XE (scaled-max) versus untouched Cirkit.")
    parser.add_argument("--layers", default="cp,tucker")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "speedup.csv")
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--_single", nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.layers = parse_layers(args.layers)
        args.seeds = parse_ints(args.seeds)
        args.single = parse_single(args._single) if args._single else None
    except ValueError as error:
        parser.error(str(error))
    return args


def configurations(args: argparse.Namespace) -> list[Configuration]:
    blocks = [
        (seed, graph, layer, batch, units)
        for seed in args.seeds
        for layer in args.layers
        for graph, batches in SAFE_GRID[layer].items()
        for batch, units_values in batches.items()
        for units in units_values
    ]
    return shuffled_blocks(blocks, tuple(VARIANTS), shuffle_seed=20260728)


def main() -> None:
    args = parse_args()
    if args.single:
        run_configuration(
            args.single,
            VARIANTS,
            output=args.output,
            device_arg=args.device,
            verbose_errors=args.verbose_errors,
        )
        return
    run_isolated(
        Path(__file__).resolve(),
        configurations(args),
        output=args.output,
        device=args.device,
        verbose_errors=args.verbose_errors,
    )


if __name__ == "__main__":
    main()
