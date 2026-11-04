# Optional custom CUDA kernels

Install `extended-einsum[cuda]` for Triton, or `[helion]` to experiment with the
alternative whole-program/subtree compiler. Importing the base package or the
public CUDA APIs does not import either compiler. CPU installations still work.
These APIs are experimental and explicitly opt in to custom execution.

## Tree inference

```python
import torch
from extended_einsum.cuda import CategoricalInput, compile_circuit

# pixels[B,V] integers; log_tables[banks,categories,U] FP32.
# Routes map each leaf to a pixel and a categorical parameter bank.
evidence = CategoricalInput(pixels, log_tables, variable_ids, table_ids)
# One contiguous [F >> level, output_unit, input_unit] tensor per level,
# including the leaf projection. F is a power of two, root output width is U.
runner = compile_circuit(evidence, weights, stability="scaled_max",
                         precision="tf32", backend="auto")
print(runner.selection)  # backend, reason and launch options
with torch.no_grad():
    log_scores = runner(evidence, weights)
```

`logspace_max`, `logspace_min`, `scaled_max`, `scaled_sum`, and `scaled_min`
remain separate algorithms. Scaled modes retain values and row log scales;
logspace modes retain log values. Every call includes raw categorical selection.
Pixel, variable-route, category-bank, table and weight contents may change between calls.
There is no implicit cache of normalized or packed weights. Tables are log
probabilities, weights are nonnegative probabilities (positive for scaled_min).
Parameter normalization is the caller's responsibility.

Auto selects a measured Triton profile on SM86 for the tested unit/batch range,
with conservative Torch fallback elsewhere. It does not run timing experiments
or change precision/global Torch flags. Profiles are starting points, not a
claim of universal optimality. `backend="triton"`, `"torch"`, or `"helion"`
provides a reproducible override. Helion is an optional explicit comparison;
it is not required for the Triton tree path. An existing inference runner uses
Torch when a later call requests gradients, so gradients are never dropped.

Regular Triton inference currently requires explicit `precision="tf32"`.
`tf32x3` (default) and `ieee` use Torch/Helion. Precision options govern custom
kernels; Torch fallback respects the caller's Torch precision settings. TF32 rounds matrix operands and
accumulates in FP32; approximate exp/log intrinsics are used. Likelihood-relative
errors accumulate with depth. Validate precision on your own model.

For arbitrary supported independent-batch `RichProgram`s, `compile_cuda` accepts
`program`, `inputs`, and `batch_axes`. Explicit `backend="helion"` exposes the
whole-program/subtree compiler. Unsupported operations and autograd use Torch
in auto mode. This compiler does not recognize general circuits as tree models.

## Training without a dependency on a research experiment

```python
from extended_einsum.tree import balanced_tree, compile_training

model = balanced_tree(pixels=784, units=256, categories=256, device="cuda",
                      parameterization="normalized_log")
trainer = compile_training(model, batch_size=128, trainer="anemone",
                           precision="tf32")
loss = trainer.step(images)  # raw integer pixels [128,784]
```

This is an unshared CP tree; `project_leaves=False` constructs CP-T. Pixel counts
need not be powers of two. `TreeCircuit` and `Projection` accept explicit
irregular binary-tree topology and folded parameter storage. Each node and each
parameter element has one owner; shared DAGs require accumulating flows and
are rejected. Arbitrary pixel routing (including repeated pixels) is supported
through `model.variable_ids`; categorical parameter banks are separate per leaf.

| Setup | Parameters | Automatic path on profiled CUDA shapes |
|---|---|---|
| Logspace Anemone | normalized log probabilities | custom forward/backward and fused Eq. 19 update |
| Logspace EM | normalized log probabilities | same kernels, alpha=1 and momentum=0 |
| Logspace softmax Adam | logits | custom circuit forward/backward, Torch log_softmax/Jacobian and fused Adam |
| Scaled training | either parameterization | differentiable Torch scaled algorithm |
| CPU, unsupported hardware/shape or stricter auto precision | either | Torch |

For Adam use `parameterization="logits"` and `trainer="adam"`; the original
logits remain the optimizer parameters. The custom backward returns gradients
with respect to log probabilities and autograd applies the log_softmax Jacobian.
Adam can amplify roundoff in nearly zero gradients into larger individual
logit changes; check likelihoods and aggregate gradients rather than requiring
identical logits across implementations. Normalized Anemone parameters do
**not** pass through a softmax. The fused update
includes data flows, model flows, momentum bias correction, a float32 floor and
log normalization. Its GPU step counter advances on CUDA graph replay.

Both trainers expose `loss(images)` for first-order autograd and `step(images)`
for an optimizer update. Custom double backward is unsupported. Save the model's
`state_dict()` and the trainer's `state_dict()` separately. Restore both before
resuming with the same optimizer hyperparameters. Compile a new trainer after
moving/replacing model parameters or changing topology/batch size. Warm forward,
backward and optimizer kernels on a side stream before CUDA graph capture.

The default synthetic tree builder groups adjacent pixels and is independent
of any Cirkit region graph. Prior MNIST epoch results below used the original
quad-tree model; do not compare their absolute loss with this builder's loss.

## Evidence and reproduction

`benchmarks/inference-prior.csv`, `indexing-prior.csv`, and
`anemone-mnist-prior.csv` preserve the original prototype measurements on an RTX
A6000, Torch 2.12, Triton 3.7 and Helion 1.5.2. They are historical evidence,
not fresh measurements of the packaged source. Inferencing measurements include
raw pixel lookup plus the complete circuit. Training measurements include lookup,
exp/log, forward/backward, model flows and parameter updates; the original 784
pixel Anemone 2x2 experiment gave about 1.59–2.30x speedups and held-out BPD
agreement within 9.2e-6. Individual low-flow log parameters can differ much more.

Fresh packaged-source measurements and tuning records are stored separately.
The comparison scripts exclude compilation and CPU transfers and use warmed CUDA
graph replay. Training compares the same initialization and tree on each backend.

```bash
python -m demo.benchmark_cuda_training --pixels 784 --batch 128 256 \
    --units 32 256 --trainer adam em anemone --output training.json
python -m pytest tests/test_cuda_dispatch.py tests/test_cuda_training.py
python -m pytest tests/test_helion.py tests/test_helion_categorical.py tests/test_helion_subtree.py
```

Triton retains the best register subtree schedule. cuTile, cuBLASDx, Gluon,
interleaved/packed weight storage and ping-pong alternatives did not consistently
beat it in prior experiments and are not part of this optional runtime.

## Consolidating the measured tree paths on Triton

The additional scaled sweep compared block sizes 16/32/64, 4/8/16 warps,
32/64 reduction panels and one/three pipeline stages. Helion-style integer
rounding, two-way loop unrolling, early operand rounding, smaller reduction
panels and guarded reciprocal normalization were also checked. Wider panels
and larger blocks increased spills or exceeded SM86 shared-memory limits.
Early rounding removed a small spill but did not improve time consistently;
the reciprocal and extra unrolling variants were slower and were discarded.

The retained improvement uses three pipeline stages for scaled 64/128-unit
inference (logspace small-unit inference keeps one). At B128/F16/U128 this
reduced the measured Triton time from approximately 0.028 to 0.025 ms. Helion
still measured about 0.024 ms. At B128/F1024/U256 the corresponding comparison
was 1.84 vs 1.81 ms. Auto uses Triton for these profiled tree paths to provide
one custom runtime, accepting the small scaled-mode gap. It does not claim
Triton is always the fastest backend.

`triton-scaled-tuning.json` records the initial tile sweep; its Helion comparison
used root-major grid ordering and should not be treated as the winning Helion
baseline. `triton-reciprocal-tuning.json` alternates measurements against the
correct sample-major Helion schedule. Measurements can vary with GPU load;
use the supplied scripts to confirm choices on the target device.

## Packaged training measurements

All twelve cases completed on the A6000 with 784 synthetic categorical pixels,
TF32 circuit operations, and raw lookup plus complete training steps in the
measurement. Each entry is custom / compiled Torch time in milliseconds.

| Batch | Units | Softmax Adam | EM | Anemone |
|---:|---:|---:|---:|---:|
| 128 | 32 | 1.59 / 1.75 | 0.66 / 1.97 | 0.76 / 1.91 |
| 128 | 256 | 19.55 / 20.98 | 8.42 / 26.35 | 10.56 / 26.19 |
| 256 | 32 | 2.06 / 2.53 | 1.12 / 2.74 | 1.20 / 2.73 |
| 256 | 256 | 25.20 / 27.23 | 14.03 / 32.61 | 16.60 / 32.50 |

The full repeats and initial likelihood differences are in
`benchmarks/training-packaged.json` and `.csv`. Initial NLL differences were at
most 0.002 nats per image. These are synthetic timing and initial-parity checks,
not new MNIST training-quality results. The matched Torch reference differentiates
the partition for model flows; the historical MNIST reference used a manually
optimized top-down flow pass and a different tree. Keep those comparisons separate.
The benchmark resumed after four cases using `dynamic=False` to avoid a Torch
Inductor `SymFloat` compiler failure on a subsequent shape. No eager reference
was substituted into the timing table.
