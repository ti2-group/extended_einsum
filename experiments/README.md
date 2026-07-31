# Publication experiments

This directory is the self-contained entry point for the paper measurements.

## Layout

| Experiment | Runner | Plotter | Results | Figures |
|---|---|---|---|---|
| CP and Tucker speedup/memory | `speedup.py` | `plot_speedup.py` | `results/speedup.csv` | `plots/speedup_{cp,tucker}.pdf`, `plots/memory_{reduction,usage}_{cp,tucker}.pdf` |
| CP and Tucker ablations | `ablation.py` | `plot_ablation.py` | `results/ablation.csv` | `plots/ablation_{cp,tucker}.pdf` |
| Compiler breakdown | `diagnose_compile.py` | `compile_table.py` | `results/compile_breakdown.csv` | `tables/compile_breakdown.tex` |
| JAX CP and Tucker ablations | `ablation_jax.py` | `plot_jax_ablation.py` | `results/ablation_jax.csv` | `plots/jax_torch_ablation_{cp,tucker}.pdf` |
| Numerical correctness, underflow, and MNIST convergence | `correctness.py` | `plot_correctness.py` | `results/correctness.csv`, `results/correctness_mnist_training.csv` | `tables/correctness_{agreement,mnist,mnist_training}.tex`, `plots/correctness_{underflow,mnist_training}.pdf` |
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
high-water marks.

## Numerical correctness and underflow

The supplementary correctness runner is independent of the timing protocol and
writes only `experiments/results/correctness.csv`. It computes normwise relative
errors

```text
||candidate - reference||_2 / max(||reference||_2, tiny)
```

for forward log likelihoods, input-likelihood gradients, all sum-parameter
gradients together, and the worst individual parameter tensor. It also records
maximum absolute errors and the fractions of finite forward and gradient
entries.

The default CPU run contains two suites:

1. `agreement`: optimized scaled-max and log-space FP32 programs versus an
   unoptimized, unstabilized FP64 reference, for CP and Tucker on both the quad
   tree and quad graph. It evaluates both IEEE FP32 matmuls (`highest`) and the
   publication setting (`high`).
2. `stress`: an alternating product/sum circuit at depths 4 through 1024. Raw
   FP32, raw FP64, scaled FP32, and log-space FP32 are compared with an
   unoptimized log-space FP64 reference. The width-64 circuit is evaluated
   under both IEEE FP32 (`highest`) and TF32-permitted (`high`) settings;
   complete width-256 and width-512 `high` sweeps ensure the A6000 dispatches
   TF32-capable matrix multiplications. The 0.01-scale factors make raw FP32
   underflow near depth 32 and raw FP64 underflow at larger depths.

For a CPU smoke test that does not enter the publication CSV:

```bash
uv run --group demo python experiments/correctness.py \
  --device cpu --seeds 0 --output /tmp/correctness-smoke.csv
```

The agreement suite uses the same FX/`torch.compile` path as the performance experiment. The
synthetic stress suite deliberately leaves contraction-path fusion and
`torch.compile` disabled so it isolates stability lowering across hundreds of
dependent layers rather than attempting to fuse the artificial chain. The
first command covers agreement, width-64 stress, and MNIST; the next two add
the width-256 and width-512 `high`-precision stress sweeps:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/correctness.py \
  --suites agreement,stress,mnist --device cuda --torch-compile --download
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/correctness.py \
  --suites stress --device cuda --stress-units 256 --precisions high
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/correctness.py \
  --suites stress --device cuda --stress-units 512 --precisions high
uv run --group demo python experiments/plot_correctness.py
```

The MNIST suite evaluates identical batches and parameters with the complete
optimized scaled and log-space pipelines. It uses 512 CP units and 64 Tucker
units on both region graphs. MNIST is never downloaded unless `--download` is
passed. All commands resume from successful rows in the shared
`results/correctness.csv`.

The separate `mnist-training` suite is an end-to-end convergence check. It
trains a 512-unit CP circuit on the MNIST training split for 20 epochs with
batch size 512, comparing untouched Cirkit log-space execution with Extended
Einsum log-space and scaled-max execution. Each epoch uses all 117 complete
fixed-shape batches (59,904 samples); the 96-sample remainder changes with the
seeded permutation and is dropped. The default region graph is the quad tree;
`--mnist-training-region-graph quad-graph` selects the shared quad graph
instead. Each implementation runs in a fresh process. There is no training
warmup, so epoch 1 begins from the seeded initialization. Setup RNG consumption
is isolated by resetting the minibatch seed before training, which gives all
three implementations the same epoch permutations. The two Extended Einsum
variants also share identical initial parameters. Cirkit uses its native
seeded parameter initialization and is therefore a convergence baseline, not
a bitwise parameter-matched trajectory.

One row per epoch is written to
`results/correctness_mnist_training.csv`. A complete variant trajectory is
committed together, and later invocations skip variants for which all 20
epochs are already present. The plotter generates a combined NLL trajectory
and a final-NLL comparison table:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/correctness.py \
  --suites mnist-training --device cuda --download
uv run --group demo python experiments/plot_correctness.py
```

For a short pipeline check, cap the number of epochs and batches without
writing into the publication CSV:

```bash
uv run --group demo python experiments/correctness.py \
  --suites mnist-training --device cpu \
  --mnist-training-epochs 2 --mnist-training-units 8 \
  --mnist-training-batch-size 8 --mnist-training-max-batches 2 \
  --mnist-training-output /tmp/correctness-mnist-training-smoke.csv
```

All non-index model inputs and parameters use `torch.float32`; gather indices
use `torch.int64`; autocast is disabled. The `high` setting retains FP32
storage and outputs but permits TF32 or another reduced-mantissa internal
algorithm for eligible CUDA matrix multiplications. The unstabilized reference
uses FP64 and `highest`; the MNIST comparison uses the production `high`
setting for both stable implementations. Every row records the storage and
reference dtypes, matmul settings, PyTorch/CUDA versions, device, finite-value
fractions, compilation state, and parameter counts.

## Speedup experiment

The only systems are Cirkit and the current Extended Einsum
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

## Compiler breakdown

The compiler experiment reuses the complete ablation grid and the four
dense/Monarch full-image configurations. It measures five independent runs in
fresh processes with private Inductor and Triton cache directories, preventing
compiled artifacts from carrying across repetitions. For XE, the CSV records
expression extraction, input-depth
folding and consumer ordering, contraction-path optimization, and backend/FX
lowering separately, as well as their sum. For Cirkit, it records native
lowering. In both systems, the `torch.compile` measurement is the synchronized
first forward and backward call: PyTorch compilation is lazy, and both calls
are needed to compile the training graph used by the performance experiments.
The reported compile time therefore includes one execution of the compiled
graph.

Run the experiment on the same GPU used for the runtime measurements, then
generate the supplementary table:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo \
  python experiments/diagnose_compile.py
uv run --group demo python experiments/compile_table.py
```

The runner resumes successful configuration/run pairs from
`results/compile_breakdown.csv`. Each configuration has a 15-minute
parent-enforced timeout by default. The generated
`tables/compile_breakdown.tex` reports the median and
`[minimum, maximum]` in seconds across the five process-isolated runs. Filters
such as `--suites`, `--runs`, `--layers`, `--graphs`, `--variants`, and
`--parameterizations` are intended for smoke runs; the table generator
deliberately requires the complete grid for the selected run IDs.

### JAX ablation

The five Extended Einsum variants can additionally be run through JAX/XLA:

```bash
uv sync --extra jax-cuda --group demo --group dev
CUDA_VISIBLE_DEVICES=0 uv run python experiments/ablation_jax.py
uv run --group demo python experiments/plot_jax_ablation.py
```

This writes `experiments/results/ablation_jax.csv` and uses the same CP and
Tucker grids, run seeds, 30 warm-up batches, and 90 measured batches as the
PyTorch ablation. Every variant/configuration/run is launched in a fresh
process. The forward loss, `value_and_grad`, and Adam update are each lowered
and compiled with their concrete parameter and batch shapes before warm-up, so
compilation is excluded from all measurements.

XLA stages forward and reverse-mode differentiation as one executable and does
not expose a synchronization boundary equivalent to PyTorch's
`loss.backward()`. The primary JAX metric is therefore the directly timed
compiled `value_and_grad` execution (`forward_backward_ms_per_batch`). The CSV
also contains a separately timed compiled forward pass and a diagnostic
`backward_estimate_ms_per_batch`, computed as `value_and_grad - forward`.
Comparisons should use the directly measured combined time rather than treating
that estimate as an independently measured backward phase.

## Monarch full-image experiment

The Monarch experiment uses full `64 x 64` grayscale ImageNet64 images and CP
quad-tree/quad-graph circuits. Within each scale point, Cirkit and XE use
identical canonical logits, trainable tensors, and deterministic image orders.
Cirkit performs its native folding and XE uses only the current input-depth
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
