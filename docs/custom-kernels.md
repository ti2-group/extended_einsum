# Optional custom CUDA kernels

Install `extended-einsum[cuda]` for Triton. Importing the base package or the
public CUDA APIs does not import Triton. CPU installations still work.
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
claim of universal optimality. `backend="triton"` or `"torch"` provides an explicit override. An existing inference runner uses
Torch when a later call requests gradients, so gradients are never dropped.

Regular Triton inference currently requires explicit `precision="tf32"`.
`tf32x3` (default) and `ieee` use Torch. Precision options govern custom
kernels; Torch fallback respects the caller's Torch precision settings. TF32 rounds matrix operands and
accumulates in FP32; approximate exp/log intrinsics are used. Likelihood-relative
errors accumulate with depth. Validate precision on your own model.

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

## Cirkit input kernels

The input implementation is separate from the circuit implementation:
`extended_einsum/inputs.py` holds evidence bindings and the Cirkit adapter;
`backends/triton_inputs.py` holds the input evaluation and gradient kernels.
There is no Helion backend or dependency in this runtime.

```python
from extended_einsum.inputs import compile_input

# A Cirkit Torch input layer, with its existing parameters and semiring.
input_fn = compile_input(cirkit_input_layer)
scores = input_fn(raw_data)  # raw_data[batch, variables] -> [fold, batch, units]
# Gradients propagate to the original parameters and to continuous data.
scores.backward(upstream_flows)
print(input_fn.selection)
```

Cirkit remains an optional dependency. The adapter supports categorical,
embedding, binomial (including Bernoulli), Gaussian (with optional log partition),
polynomial, constant-value and evidence layers. It preserves Cirkit's linear or
log output space. Categorical logits are **unnormalized log values**, as in Cirkit;
they are not passed through a softmax. This differs from the explicit
`CategoricalInput(logits=True).materialize()` normalization option.

FP32 CUDA input evaluation fuses raw-variable selection with the distribution or
function calculation. Custom first-order backward reduces input-parameter flows;
continuous input gradients accumulate correctly for repeated variable routes.
The adapter reads parameter values on every call. It bypasses only provable
identity gathers on direct tensor parameters; changed fold routing falls back
to Cirkit's parameter graph. Recapture CUDA graphs after changing parameter-fold
routing or replacing parameters. Raw data, parameter values and variable-scope
indices can change in place between replays. General parameter transforms stay owned by Cirkit.

Constants transform their small parameter vector before broadcasting. Evidence
is evaluated once before broadcasting across the batch. These paths avoid full
batch-sized output writes. Outputs may be strided or broadcast views and should be treated as read-only.
Use `clone()` for a writable copy, or request contiguity when a consuming
operation requires it.

The selected tiles depend on the input family and unit count. `tile`,
`output_layout`, `table_layout`, `backward_units` and `backward_categories` offer
explicit overrides. No parameter pack is cached; a requested packed table is
rebuilt on every call. CPU, complex semirings and unsupported precision/shape
setups preserve Cirkit's Torch execution. Custom kernels assume valid
parameter domains, for example positive Gaussian standard deviations.
Polynomial singleton batches retain the installed Cirkit implementation's
behavior via Torch. Custom backward supports batches up to 512; larger training
batches fall back to Torch.

This adapter can supply tensors to existing extended-einsum expressions. The
specialized `TreeCircuit` trainer above continues to model categorical leaves;
the adapter does not introduce Gaussian EM update rules into that trainer.
