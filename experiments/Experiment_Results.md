# Experiments and results

This document describes the experiments represented by the PDFs in
`experiments/plots/` and `experiments/pyjuice_cp_t/plots/`, and summarizes
their results. Times are milliseconds per batch unless stated otherwise.

## Experimental setup

### CP and Tucker experiments

We evaluate probabilistic circuits over 28 x 28 MNIST images with 256
categorical values per input. Both systems use fused Adam with learning rate
0.01. The complete forward-and-loss function is compiled with
`torch.compile(mode="reduce-overhead")`; forward, backward, and optimizer
components are timed separately with CUDA events. Reported runtime is forward
plus loss plus backward and excludes the optimizer update.

Every backend, seed, region graph, layer, batch size, and unit count is executed
in a fresh Python process. Each run performs 30 warmup batches followed by three
blocks of 30 measured batches. The CSV records the median per-batch time across
the three blocks. We use five seeds (0--4), pair minibatch permutations across
systems, shuffle configuration blocks, and reverse system order in alternating
blocks. Uncertainty is reported with 95% bootstrap intervals over the five
paired seed measurements.

Peak memory is the maximum CUDA allocation observed across model construction,
compilation, warmup, and measurement. The speedup experiment compares untouched
Cirkit with production Extended Einsum: scaled-max stability, detached shifts,
input-depth folding, contraction-path optimization, and consumer ordering.
Configurations known not to fit on the 48 GiB GPU are excluded.

### Parameter-matched CP-T experiment

CP-T denotes the *transposed canonical polyadic* sum-product layer. Given two
vectors of child values \(x,y\in\mathbb{R}^{U}\), it first forms their
element-wise product and then applies a weighted sum:

```text
z_k = x_k y_k,             h_j = sum_k W_jk z_k.
```

Thus, each CP-T layer is a Hadamard product followed by one dense sum layer. It
reverses the order used by the CP layer, which first applies separate dense
transformations to the two children and then multiplies their outputs
element-wise. In log space, the CP-T computation is equivalently
`h_j = logsumexp_k(log W_jk + x_k + y_k)`. We call each product--sum merge of
two child regions a CP-T *patch*. A binary tree over 784 image variables has
783 such internal patches.

This construction is particularly suitable for comparison with PyJuice:
PyJuice represents the same operation explicitly as a product node followed by
a sum node, without introducing the additional structural product layer that
made the earlier CP comparison parameter-mismatched. The CP-T experiment uses
the same 28 x 28 quad tree in PyJuice 2.6.1, Extended Einsum, and Cirkit. Before
timing, the implementation verifies equal input, product, and sum counts; equal
product and sum scopes; equal output widths; and the exact logical parameter
count

```text
784 * U * 256 + 782 * U^2 + U.
```

We evaluate batch/unit pairs 256/64, 512/64, 256/128, and 512/512 with five
seeds. Each backend/configuration/seed runs in a fresh process for 30 warmup and
90 measured batches. All systems receive the same seed-specific synthetic
categorical batch repeatedly, removing data-loading variation. Speedup over
PyJuice is computed as PyJuice time divided by the corresponding system's time;
values above one therefore favor that system.

## Extended Einsum compared with Cirkit

### Runtime

We measure forward-plus-backward speedup relative to untouched Cirkit for CP
and Tucker sum-product layers. The experiments cover quad trees and quad
graphs, batch sizes 256 and 512, and the feasible range of unit counts. Quad
graphs stop at 512 units because larger configurations exceed available GPU
memory. The plots report 95% bootstrap intervals over five paired seeds.

Extended Einsum is faster for every evaluated CP configuration
(`speedup_cp.pdf`), with the largest gains on higher-width quad-graph circuits.
The Tucker gains (`speedup_tucker.pdf`) are smaller but remain consistent
across the feasible grid. Their growth on wider quad graphs is consistent with
improved handling of reused intermediates, although this experiment does not
isolate that mechanism.

### Peak GPU memory

We compare both relative and absolute peak allocated GPU memory. The relative
measure is Cirkit peak memory divided by Extended Einsum peak memory, so values
above one mean that Extended Einsum uses less memory. Separate results are
reported for CP (`memory_reduction_cp.pdf`) and Tucker
(`memory_reduction_tucker.pdf`), with 95% bootstrap intervals over five seeds.
The corresponding absolute-memory plots (`memory_usage_cp.pdf` and
`memory_usage_tucker.pdf`) show the GiB footprint at batch sizes 256 and 512.
Together, these measurements show the memory reduction in context and identify
configurations approaching the 48 GiB device limit.

## Optimization ablations

### CP layers

The CP ablation (`ablation_cp.pdf`) measures the contribution of stable
arithmetic and consumer-aware memory ordering. `Log-space` changes only the
stable arithmetic, `Shift gradients` makes the numerical shifts
differentiable, `Log-space + shift gradients` combines both counterfactuals,
and `No consumer ordering` disables only consumer-aware memory ordering.
Runtime is normalized per seed to production Extended Einsum (XE), and
untouched Cirkit provides an external reference. The results show that
differentiating through the stability shifts and disabling consumer ordering
both increase CP runtime.

### Tucker layers

The Tucker ablation (`ablation_tucker.pdf`) evaluates the same counterfactuals
and normalization. Unlike CP, consumer ordering is slightly detrimental for
Tucker layers.

This difference does not arise from the Tucker contractions themselves: with
and without ordering, Extended Einsum generates the same
einsum equations, operand shapes, contraction paths, and batched matrix
multiplications. Instead, ordering changes how folded intermediate tensors are
routed to their consumers. It replaces many indexed gathers with contiguous
slices, without reducing the total number of program instructions. Although
the slices are inexpensive in the forward pass, their gradients introduce
slice-backward materializations and separate reduction kernels. By contrast,
PyTorch can fuse the indexed-gather backward operations with the surrounding
Tucker multiplication and reduction work. Consequently, the forward times and
matrix-multiplication costs remain nearly unchanged, while the ordered
variant's backward pass is a few percent slower.

## JAX and PyTorch backend comparison

### Experimental setup

We execute the five Extended Einsum ablation variants through both the
PyTorch/Inductor and JAX/XLA backends on the same RTX A6000 GPU. The circuit
structure, folded tensor program, contraction-path optimization, consumer
ordering, stability mode, tensor shapes, batch sizes, and unit counts are
identical across backends. Each backend/configuration/run is isolated in a
fresh process and uses 30 warmup followed by 90 measured batches. Results are
the median over five runs with 95% bootstrap intervals.

For JAX, the forward loss, `value_and_grad`, and Adam update are separately
traced, lowered, and compiled for the concrete parameter and batch shapes
before warmup. Compilation time is excluded. XLA stages the forward and
reverse-mode computations as one executable, so the directly comparable
quantity is combined forward-plus-backward time. Optimizer time is excluded
for both backends.

The seed labels identify independent runs, but JAX and PyTorch use their native
random-number generators and therefore do not receive bitwise-identical
initial values or minibatch permutations. This does not alter the static tensor
program or dense kernel shapes and the observed run-to-run ranges are small,
but the comparison should be interpreted as a backend execution comparison
rather than a numerical trajectory comparison.

### CP results

The CP backend experiment (`jax_torch_ablation_cp.pdf`) measures absolute
forward-plus-backward time for the production implementation and all four
counterfactuals. Untouched Cirkit is included once as an external reference.
Every percentage uses PyTorch production XE in the same configuration as its
common reference, and uncertainty is reported with 95% bootstrap intervals
over five independent runs.

The optimized CP program transfers well to XLA. JAX is only 0.3% slower than
PyTorch on the largest quad graph and 7--22% slower on the remaining
configurations.

Untouched Cirkit requires 34--47% more time than PyTorch XE, providing the
external baseline for these within-XE backend and ablation comparisons.
The relative ablations are consistent across compilers: consumer ordering
reduces JAX runtime by 15--26%, while differentiating through the reference
shifts increases it by 17--31%. Thus, the two principal CP optimizations are
not artifacts of Inductor's code generation.

### Tucker results

The Tucker backend experiment (`jax_torch_ablation_tucker.pdf`) uses the same
normalization, external Cirkit reference, and uncertainty calculation as the
CP experiment.

Tucker contractions exhibit a stronger backend dependence. Production JAX
requires 1.39--1.87x the PyTorch time, and the gap does not disappear at the
larger problem sizes. PyTorch XE nevertheless remains 8--22% faster than
untouched Cirkit on the same configurations. Because both backends receive the
same folded program structure and contraction paths, these measurements point
to different lowering and fusion quality for the Tucker contraction and
routing pattern rather than Python overhead or shape polymorphism. In
particular, disabling consumer ordering slows JAX by 6--30%, whereas it makes
PyTorch approximately 2--5% faster. The layout selected by consumer ordering
is therefore favorable to XLA even though Inductor can exploit the alternative
indexed-gather route.

The numerical-stability variants also reveal an XLA-specific interaction.
Log-space arithmetic alone is 5--14% slower than scaled-max under JAX, but
combining log space with differentiable shifts makes the Tucker program
3--29% faster than production JAX. The reversal is stable across five runs and
is largest for the 512/64 quad graph. We interpret it as a compiler interaction
rather than an algorithmic advantage of differentiable shifts: PyTorch does
not show the reversal, and the mathematical transformation adds rather than
removes gradient dependencies. Accordingly, the main optimization claims
should continue to use the PyTorch ablation, while the JAX results demonstrate
both the portability of the CP gains and the backend sensitivity of Tucker
lowering.

## Parameter-matched comparison with PyJuice

### Forward execution

We compare Extended Einsum, Cirkit, and PyJuice 2.6.1 on exactly
parameter-matched CP-T quad trees. The configurations are 64 units at batch
sizes 256 and 512, 128 units at batch size 256, and 512 units at batch size
512. Forward time is divided by the 783 internal patches in the binary tree,
and uncertainty is reported with 95% bootstrap intervals over five seeds
(`pyjuice_cp_t/plots/forward_per_patch.pdf`).

Extended Einsum ranges from 0.93x to 1.38x PyJuice forward speed, while Cirkit
ranges from 0.97x to 1.53x. The advantage is clearest for the largest
512/512 configuration: Extended Einsum is 1.38x and Cirkit is 1.53x as fast as
PyJuice. At the smaller configurations, no single system wins consistently.

This forward comparison is the cleanest cross-system result because circuit
structure, scopes, widths, parameters, inputs, warmup, and measurement counts
are matched. The execution stacks still differ: Extended Einsum and Cirkit use
PyTorch compilation, while PyJuice uses its native Triton and CUDA graph path.
On the RTX A6000 (compute capability 8.6), PyJuice 2.6.1 cannot use its
compute-capability-9.0 TMA kernels and falls back to Triton. Results on
Hopper-class hardware may therefore differ.

PyJuice's custom kernels are designed to support block-sparse circuit
computations, whereas the circuits evaluated here are fully dense. Extended
Einsum and Cirkit can therefore lower their dense contractions to highly
optimized dense matrix-multiplication kernels. PyJuice's more general
block-sparse execution path does not attain the same dense-kernel efficiency
in this setting. This is an important specialization trade-off rather than a
claim that dense execution is preferable for circuits that contain exploitable
structural sparsity.

### Forward and backward execution

The forward-plus-backward experiment
(`pyjuice_cp_t/plots/forward_backward.pdf`) uses the same parameter-matched
circuits. Extended Einsum is 1.20--1.72x as fast as PyJuice across the four
configurations. Cirkit is slower than PyJuice in three configurations and
1.26x as fast at 512/512.

Backward work is less directly comparable than forward execution. PyJuice's
positive EM flows and the logit gradients computed by Extended Einsum and
Cirkit traverse the same circuit but are different mathematical quantities.
Gradient or flow clearing is outside the timed regions, and no optimizer
update is included. These measurements therefore compare implementation-level
execution and do not show that one system performs an equivalent training
update faster.

## CP-T versus CP

CP and CP-T differ in operation order. CP transforms each child with a dense
sum and then multiplies the transformed vectors, whereas CP-T first multiplies
corresponding child units and then applies one dense sum. The table compares
Extended Einsum with Cirkit on both structures using the same quad-tree
batch/unit configurations. The comparison holds batch size and width fixed; it
does not claim equal parameter counts between CP and CP-T. Values are medians
over five paired seeds; the range is the minimum and maximum seed-wise speedup.

| Layer | Batch/units | Cirkit | Extended Einsum | XE speedup | Seed range |
|---|---:|---:|---:|---:|---:|
| CP | 256/64 | 4.514 ms | 3.264 ms | 1.382x | 1.380--1.383x |
| CP-T | 256/64 | 4.324 ms | 3.172 ms | 1.363x | 1.278--1.389x |
| CP | 512/64 | 7.340 ms | 5.665 ms | 1.296x | 1.286--1.297x |
| CP-T | 512/64 | 6.235 ms | 4.414 ms | 1.415x | 1.302--1.462x |
| CP | 256/128 | 9.952 ms | 6.776 ms | 1.470x | 1.468--1.472x |
| CP-T | 256/128 | 8.154 ms | 5.324 ms | 1.531x | 1.388--1.538x |
| CP | 512/512 | 88.956 ms | 65.277 ms | 1.362x | 1.361--1.365x |
| CP-T | 512/512 | 60.577 ms | 44.369 ms | 1.364x | 1.361--1.369x |

CP-T reduces absolute runtime for both systems, but it does not remove the
Extended Einsum advantage. At 64 units, the CP-T speedup is 1.36x for batch
256 and 1.41x for batch 512, compared with 1.38x and 1.30x for CP. At 256/128,
the median speedup increases from 1.47x for CP to 1.53x for CP-T. At 512/512,
the results are effectively identical: 1.362x for CP and 1.364x for CP-T. The
agreement across widths indicates that the production optimizations generalize
beyond the CP lowering used in the main experiment.

The speedups relative to PyJuice are summarized below. Values are PyJuice time
divided by the corresponding Extended Einsum or Cirkit time.

| Batch/units | XE forward | Cirkit forward | XE forward + backward | Cirkit forward + backward |
|---|---:|---:|---:|---:|
| 256/64 | 1.04x | 1.15x | 1.20x | 0.89x |
| 512/64 | 0.93x | 1.17x | 1.27x | 0.90x |
| 256/128 | 0.95x | 0.97x | 1.21x | 0.79x |
| 512/512 | 1.38x | 1.53x | 1.72x | 1.26x |

The backward qualification above applies to the forward-plus-backward values.

## Matched Monarch full-image comparison

The Monarch experiment measures matched Cirkit and Extended Einsum models on
full 64 x 64 grayscale ImageNet64 images at batch size 256. The two systems
train the same parameter tensors within each configuration. We report the
median paired forward-plus-backward ratio over five seeds at high-width,
jointly feasible points: H=512 for the quad tree and H=256 for the more
intermediate-intensive quad graph. These are conservative shared feasibility
points, not claimed hardware maxima.

Extended Einsum reduces Monarch forward-plus-backward time by 1.41x for the
quad graph and 1.82x for the quad tree relative to Cirkit. The supplementary
table reports absolute times, parameter counts, seed ranges, and Dense results.
