# Experimental setup, results text, and figure captions

This document accompanies the PDFs in `experiments/plots/` and
`experiments/pyjuice_cp_t/plots/`. Times are milliseconds per batch unless
stated otherwise.

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
blocks. Confidence bands and error bars are 95% bootstrap intervals over the
five paired seed measurements.

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
categorical batch repeatedly, removing data-loading variation. The bar labels
report speedup over PyJuice, computed as PyJuice time divided by the
corresponding bar time; values above one therefore favor the indicated system.

## Main comparison captions

### `speedup_cp.pdf`

**Extended Einsum speedup for CP sum-product layers.** Forward-plus-backward
time is normalized to untouched Cirkit; values above one favor Extended
Einsum. Lines distinguish quad trees and quad graphs at batch sizes 256 and
512, while shaded regions show 95% bootstrap intervals over five paired seeds.
Quad graphs stop at 512 units because larger configurations exceed available
GPU memory. Extended Einsum is faster throughout the evaluated grid, with the
largest gains on the higher-width quad-graph circuits.

### `speedup_tucker.pdf`

**Extended Einsum speedup for Tucker sum-product layers.**
Forward-plus-backward time is normalized to untouched Cirkit. Lines distinguish
region graph and batch size; shaded regions show 95% bootstrap intervals over
five paired seeds. The gains are smaller than for CP but remain consistent
across the feasible grid. Their growth with wider quad graphs is consistent
with improved handling of reused intermediates, although the measurements do
not isolate that mechanism.

### `memory_reduction_cp.pdf` and `memory_reduction_tucker.pdf`

**Peak-memory reduction of Extended Einsum relative to Cirkit.** Each point is
Cirkit peak allocated memory divided by Extended Einsum peak allocated memory;
values above one indicate lower memory use for Extended Einsum, while values
below one indicate higher memory use. The horizontal line at one marks equal
peak memory. Lines identify region graph and batch size, and shaded regions
show 95% bootstrap intervals over five seeds. The CP and Tucker layer types are
shown in separate figures.

### `memory_usage_cp.pdf` and `memory_usage_tucker.pdf`

**Absolute peak allocated GPU memory for Extended Einsum and Cirkit.** The two
panels separate batch sizes 256 and 512 and use a shared GiB scale, allowing the
two systems to be compared directly without crowding eight series into one
panel. Color and marker identify the region graph; solid lines denote Extended
Einsum and dashed lines denote Cirkit. Shaded regions show 95% bootstrap
intervals over five seeds. These figures expose the absolute memory footprint
behind the relative reductions and make configurations approaching the 48 GiB
device limit explicit.

## Ablation captions

### `ablation_cp.pdf`

**CP stability and consumer-ordering ablation.** Runtime is normalized per seed
to production Extended Einsum (XE). Bars stack forward-plus-loss and backward
time; error bars are 95% bootstrap intervals for the total, and annotations
report the percentage slowdown relative to XE. `Log-space` changes only the
stable arithmetic, `Shift gradients` makes the numerical shifts
differentiable, `Log-space + shift gradients` combines both counterfactuals,
and `No consumer ordering` disables only consumer-aware memory ordering.
Untouched Cirkit is included as an external reference.

### `ablation_tucker.pdf`

**Tucker stability and consumer-ordering ablation.** Runtime is normalized per
seed to production Extended Einsum. Forward-plus-loss and backward components
are stacked within each bar; error bars show 95% bootstrap intervals for the
total, and annotations report the percentage change relative to XE. The
counterfactuals are identical to the CP ablation, allowing the numerical
stability and ordering effects to be compared across layer types.

Unlike for CP, consumer ordering is slightly detrimental for the Tucker
layers. This difference does not arise from the Tucker contractions
themselves: with and without ordering, Extended Einsum generates the same
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

## PyJuice captions and comparability

### `pyjuice_cp_t/plots/forward_per_patch.pdf`

**Forward time for exactly parameter-matched CP-T quad trees.** A CP-T patch
multiplies corresponding units from two child regions and then maps the
products through a dense sum layer. Bars compare Extended Einsum, Cirkit, and
PyJuice 2.6.1 at 64 units for batch sizes 256 and 512, together with the
original batch/unit pairs 256/128 and 512/512. Error bars show 95% bootstrap
intervals over five seeds. Values above the bars are speedups over PyJuice;
values above one favor the indicated system. Time is divided by the 783
internal CP-T patches in the 28 x 28 binary quad tree.

### `pyjuice_cp_t/plots/forward_backward.pdf`

**Forward and backward time for exactly parameter-matched CP-T quad trees.**
Each CP-T layer consists of an element-wise product of corresponding child
units followed by a dense sum. Bars stack forward and backward time, error bars
show 95% bootstrap intervals for their sum, and labels give speedup over
PyJuice. The comparison is a systems timing result, not an optimizer-equivalent
training comparison: Extended Einsum and Cirkit compute log-likelihood
gradients with respect to logits, whereas native PyJuice backward computes
positive EM parameter flows.

The forward comparison is the cleanest cross-system result because circuit
structure, scopes, widths, parameters, inputs, warmup, and measurement counts
are matched. Nevertheless, the execution stacks differ: Extended Einsum and
Cirkit use PyTorch compilation, while PyJuice uses its native Triton and CUDA
graph path. On the RTX A6000 (compute capability 8.6), PyJuice 2.6.1 cannot use
its compute-capability-9.0 TMA kernels and falls back to Triton. Results on
Hopper-class hardware may therefore differ.

PyJuice's custom kernels are designed to support block-sparse circuit
computations, whereas the circuits evaluated here are fully dense. Extended
Einsum and Cirkit can therefore lower their dense contractions to highly
optimized dense matrix-multiplication kernels. PyJuice's more general
block-sparse execution path does not attain the same dense-kernel efficiency
in this setting. This is an important specialization trade-off rather than a
claim that dense execution is preferable for circuits that contain exploitable
structural sparsity.

Backward work is less directly comparable. PyJuice's positive EM flows and the
logit gradients computed by Extended Einsum and Cirkit traverse the same
parameter-matched circuit but are different mathematical quantities. Gradient
or flow clearing is outside the timed regions, and no optimizer update is
included. Accordingly, the forward-plus-backward figure should be presented as
an implementation-level comparison with an explicit caveat, not as evidence
that one system performs the same training update faster.

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

**Matched Monarch runtime on full ImageNet64 images.** Cirkit and XE train the
same parameter tensors within each row. Speedup is the median paired
forward-plus-backward ratio over five seeds; all runs use full 64 x 64
grayscale images and batch size 256. The main table reports only the high-width
Monarch points: H=512 for the quad tree and H=256 for the more
intermediate-intensive quad graph. These are conservative shared feasibility
points, not claimed hardware maxima. The supplementary table reports absolute
times, parameter counts, seed ranges, and the Dense rows.

Suggested main text: “On full ImageNet64 images, XE reduces Monarch
forward--backward time by 1.41x for the quad graph and 1.82x for the quad
tree relative to Cirkit.”
