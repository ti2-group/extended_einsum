# Publication experiments

This directory is the self-contained entry point for the paper measurements.
It intentionally contains only the current publication protocol: there are no
versioned result files, legacy backends, or compatibility columns.

## Layout

| Experiment | Runner | Plotter | Results | Figures |
|---|---|---|---|---|
| CP and Tucker speedup/memory | `speedup.py` | `plot_speedup.py` | `results/speedup.csv` | `plots/speedup_{cp,tucker}.pdf`, `plots/memory_{reduction,usage}_{cp,tucker}.pdf` |
| CP and Tucker ablations | `ablation.py` | `plot_ablation.py` | `results/ablation.csv` | `plots/ablation_{cp,tucker}.pdf` |
| Parameter-matched CP-T comparison | `pyjuice_cp_t/benchmark.py` | `pyjuice_cp_t/plot.py` | `pyjuice_cp_t/results/comparison.csv` | `pyjuice_cp_t/plots/*.pdf` |
| Dense/Monarch full-image comparison | `monarch/benchmark.py` | `monarch/table.py` | `monarch/results/performance.csv` | `monarch/tables/performance_{main,supplement}.tex` |

Runners resume automatically from successful rows. Every configuration is run
in a fresh Python process, so compiled graphs and CUDA allocator state cannot
leak from one configuration into the next. To make a genuinely fresh
measurement, move or remove the corresponding CSV before starting the runner.

## Common protocol

- Dataset: MNIST training split, 28 x 28 pixels and 256 categorical values.
- Optimizer: fused Adam with learning rate 0.01.
- Compilation: the complete forward-and-loss function is compiled with
  `torch.compile(mode="reduce-overhead")`; forward, backward, and optimizer
  components are timed separately with CUDA events.
- Warmup: 30 complete training batches.
- Measurement: three consecutive blocks of 30 batches (90 measured batches).
  The CSV stores the median per-batch time across the three block summaries.
- Replication: five seeds, 0 through 4.
- Pairing: the minibatch RNG is reset after model construction so variants use
  the same seed-specific batch permutations. Seed/configuration blocks are
  shuffled, and variant order is reversed in alternating blocks.
- Isolation: one fresh process per backend/seed/graph/layer/batch/unit
  configuration.
- Runtime metric: forward plus loss plus backward. The optimizer is reported in
  the CSV but excluded from speedup and ablation figures.
- Memory metric: the maximum allocated and reserved CUDA memory across setup,
  compilation, warmup, and measurement.

Each successful CSV row is one seed/configuration summary. The compact schema
contains identifiers, the five per-batch timing fields, and the two CUDA
high-water marks; setup times, duplicated compile flags, epoch rows, NLLs, and
current allocator state are deliberately omitted.

## Speedup experiment

The only systems are untouched Cirkit and the current Extended Einsum
implementation: scaled-max stability, detached shifts, input-depth folding,
contraction-path optimization, and consumer ordering. Cirkit remains its native
log-space baseline. Both CP and Tucker are evaluated on quad trees and quad
graphs at batches 256 and 512. The grid in `speedup.py` is explicit and excludes
the configurations already established to run out of memory. CP widths start
at 64 units for every graph and batch-size series.

Run and plot:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/speedup.py
uv run --group demo python experiments/plot_speedup.py
```

For a quick smoke test, restrict the layers and seeds:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/speedup.py \
  --layers cp --seeds 0
```

## Ablation experiment

The ablation contains six variants for both CP and Tucker:

1. `xe`: scaled-max, detached shifts, consumer ordering (the reference).
2. `logspace`: log-space arithmetic with detached shifts.
3. `shift-gradients`: scaled-max with differentiable shifts.
4. `logspace-shift-gradients`: log-space arithmetic with differentiable shifts;
   this jointly removes both numerical-stability optimizations.
5. `no-ordering`: scaled-max with detached shifts but no consumer ordering.
6. `cirkit`: untouched Cirkit.

Thus `logspace` isolates arithmetic, `shift-gradients` isolates stop-gradient,
and `logspace-shift-gradients` measures their merged effect. “No ordering” is a
separate production counterfactual; it does not alter folding, input
preordering, stability, or contraction paths.

CP uses batch/unit pairs 256/128 and 512/512. Tucker uses 256/32 and 512/64.
Every pair is run on both region graphs.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/ablation.py
uv run --group demo python experiments/plot_ablation.py
```

## Monarch full-image experiment

The Monarch experiment uses full `64 x 64` grayscale ImageNet64 images and CP
quad-tree/quad-graph circuits. Within each scale point, Cirkit and XE use
identical canonical logits, trainable tensors, and deterministic image orders.
Cirkit performs its native folding; XE uses only the current input-depth
folding implementation. Dense and Monarch parameterizations are both included,
with larger hidden widths for the structured Monarch sums. In addition to the
compact summary CSV, the runner records one raw timing row per measured batch.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo \
  python experiments/monarch/benchmark.py
uv run --group demo python experiments/monarch/table.py
```

See [`monarch/README.md`](monarch/README.md) for the layouts, factor shapes,
safe grid, and matching guarantees.

## Complete sequential run

`run_all.sh` executes the speedup, ablation, CP-T, and Monarch experiments
sequentially on one visible GPU, then renders every PDF and LaTeX table.
Sequential execution avoids cross-experiment GPU and compilation interference.

```bash
CUDA_VISIBLE_DEVICES=0 experiments/run_all.sh
```

The same script is suitable for a user service. The current run was launched
as:

```bash
systemd-run --user \
  --unit=extended-einsum-publication \
  --description="Extended Einsum publication experiments" \
  --collect \
  --property=Restart=no \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=EXTENDED_EINSUM_UV_CACHE=/tmp/extended-einsum-uv-cache \
  --setenv=EXTENDED_EINSUM_MPL_CACHE=/tmp/extended-einsum-publication-matplotlib \
  --working-directory="$PWD" \
  "$PWD/experiments/run_all.sh"
```

Inspect it with:

```bash
systemctl --user status extended-einsum-publication.service
journalctl --user -fu extended-einsum-publication.service
```
