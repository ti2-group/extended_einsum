# Parameter-matched PyJuice CP-T comparison

This experiment compares PyJuice 2.6.1, production Extended Einsum, and
untouched Cirkit on the same 28 x 28 CP-T quad tree. The builders are validated
before timing: input/product/sum layer counts, product scopes, sum scopes,
output widths, and the logical parameter count must all match. For `U` units,
the count is

```text
784 * U * 256 + 782 * U^2 + U.
```

The configurations are batch 256/units 64, batch 512/units 64, batch 256/units
128, and batch 512/units 512. Each uses 30 warmup batches, 90 measured batches,
and seeds 0--4. Every
backend/configuration/seed runs in a fresh process, with shuffled paired blocks
and alternating backend order. Each backend receives the same seed-specific
synthetic categorical batch repeatedly, avoiding data-loading effects in this
kernel-focused comparison. The CSV contains one median summary per seed.

PyJuice is pinned because exact publication reproduction is more important than
silently changing dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo --with pyjuice==2.6.1 \
  python experiments/pyjuice_cp_t/benchmark.py
uv run --group demo python experiments/pyjuice_cp_t/plot.py
```

The forward comparison is like-for-like. The backward bars require an important
qualification: Extended Einsum and Cirkit compute ordinary log-likelihood
gradients with respect to logits, whereas PyJuice's public native backward path
computes positive EM parameter flows. Those quantities have the same
parameter-matched circuit and traverse the same data-dependent circuit, but
they are not the same optimization update; the CSV records
`backward_quantity` explicitly. Consequently, `forward_per_patch.pdf` is the
primary cross-system result and `forward_backward.pdf` is a transparent systems
comparison, not a claim of optimizer-equivalent training work.

PyJuice uses custom kernels designed to support block-sparse circuit
computations. This experiment deliberately uses an exactly matched dense
circuit, for which Extended Einsum and Cirkit can rely on highly optimized
dense matrix-multiplication kernels. PyJuice's more general sparse-capable path
does not reach the same dense-kernel efficiency here; this observation should
not be extrapolated to circuits with exploitable structural sparsity.
