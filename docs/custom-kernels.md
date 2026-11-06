# Optional custom CUDA kernels

Install `extended-einsum[cuda]` for Triton. Importing the base package or the
public CUDA APIs does not import Triton. CPU installations still work.
These APIs are experimental and explicitly opt in to custom execution.

## Compiled-only backend

`register_compiled_backend(name, compiler)` registers a compiler that receives
`RichProgram` and example inputs directly, before lowering into eager array
operations. Its `compile(program, inputs, **options)` returns a callable taking
an input sequence. A compiled backend does not implement `BackendFunctions`.
Example inputs specialize shapes, dtypes, devices and storage layouts; their
values remain dynamic unless explicitly declared constant.

Use `compile_program(program, inputs, backend="triton", precision="tf32", ...)`
with an extracted or preprocessed program. This is the same `RichProgram` used
by the Cirkit demo. Supported patterns compile into fused kernels; other
programs raise `UnsupportedProgramError`. There is no graph partitioning,
Torch fallback, or eager Triton execution. Existing Torch/JAX/NumPy backends
retain their primitive lowering and compiler interfaces.

Expressions whose arrays use `backend="triton"` also compile this way through
`expression.materialize(stability_mode=..., precision="tf32", ...)`. Torch
tensors still detect as the Torch backend by default; Triton must be selected
explicitly. Registration and imports do not load the optional Triton runtime.

## Tree inference

The initial compiler recognizes complete binary trees with a leaf projection,
equal unit counts, even/odd sibling routing, and a final root selection and log.
It accepts either positive dense inputs, `exp(dense_log_inputs)`, or
`exp(CategoricalInput(...))`. Other folded graphs are rejected. The compiler
matches existing IR; it does not construct a replacement tree program.

```python
import torch
import extended_einsum as xe
from extended_einsum.interface.tensor_expression import Parameter

# Two folded leaves, batch B, units U. W_leaf[2,U,U], W_root[1,U,U].
x = xe.array(log_inputs, backend="triton")  # log_inputs[2,B,U]
w0 = Parameter(xe.array(W_leaf, backend="triton"))
w1 = Parameter(xe.array(W_root, backend="triton"))
left_ids = torch.tensor([0], device=log_inputs.device)
right_ids = torch.tensor([1], device=log_inputs.device)
leaf = xe.einsum("fbi,foi->fbo", xe.exp(x), w0)
left = xe.take(leaf, xe.array(left_ids, backend="triton"))
right = xe.take(leaf, xe.array(right_ids, backend="triton"))
root = xe.einsum("fbi,fbi,foi->fbo", left, right, w1)
expression = xe.log(xe.select(root, 0))
program, wrapped_inputs = xe.extract_program(expression, stability_mode="scaled_max")
inputs = [value.backend_array for value in wrapped_inputs]
constant_ids = [i for i, value in enumerate(inputs)
                if value is left_ids or value is right_ids]
with torch.no_grad():
    runner = xe.compile_program(program, inputs, backend="triton",
                                precision="tf32", constant_inputs=constant_ids)
    log_scores = runner(inputs)
print(runner.selection)
```

Sibling routing inputs must be explicitly listed in `constant_inputs`. Their
values are checked at compilation, then their identity and version are guarded
on each call. Recompile after changing them. Construct these tensors outside
`torch.inference_mode()` so they have version counters. Capture CUDA graphs
only after compilation/warmup; changing compiled constants requires recapture.

`CategoricalInput` lives in `extended_einsum.inputs`. Its pixel, variable-route,
category-bank, table and weight contents remain dynamic. Every inference call
includes raw categorical selection. Dense inputs must be contiguous; adapter
outputs may require `.contiguous()` before compilation. No normalized or packed
weight cache is retained. Tables are log probabilities, weights are nonnegative
linear probabilities (positive for scaled_min). Normalization is the caller's
responsibility.

`logspace_max`, `logspace_min`, `scaled_max`, `scaled_sum`, and `scaled_min`
remain separate algorithms. Scaled modes retain values and row log scales;
logspace modes retain log values. Launch settings use the measured unit-dependent
profiles, with no timing experiment or global precision change at compilation.

Inference currently requires real FP32 inputs on SM80+ CUDA and explicit
`precision="tf32"`. Unsupported precision, CPU execution and autograd requests
are rejected at compilation and on subsequent calls. Use the separate Torch
backend when needed. TF32 rounds matrix operands and accumulates in FP32;
approximate exp/log intrinsics are used. Likelihood errors accumulate with depth.

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

| Setup | Parameters | Execution |
|---|---|---|
| Logspace Anemone | normalized log probabilities | custom forward/backward and fused Eq. 19 update |
| Logspace EM | normalized log probabilities | same kernels, alpha=1 and momentum=0 |
| Logspace softmax Adam | logits | custom circuit forward/backward, Torch log_softmax/Jacobian and fused Adam |
| Scaled training, CPU, unsupported hardware/configuration | either | rejected |

Training requires SM80+ CUDA and `logspace_max`. Its matrix kernels support
`tf32`, `tf32x3` (default) and `ieee`. There is no automatic backend switch.

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
identity gathers on direct tensor parameters; changed fold routing uses
the original Cirkit parameter graph. Recapture CUDA graphs after changing parameter-fold
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
rebuilt on every call. CPU, complex semirings, polynomial singleton batches,
unsupported compositions and unsupported precision/shape setups are rejected.
Custom kernels assume valid parameter domains, for example positive Gaussian
standard deviations. Custom backward supports batches up to 512; larger
training batches are rejected. `input_fn.reference(data)` is an explicit native
Cirkit reference for validation; execution never switches to it automatically.

This adapter can supply tensors to existing extended-einsum expressions. The
specialized `TreeCircuit` trainer above continues to model categorical leaves;
the adapter does not introduce Gaussian EM update rules into that trainer.
