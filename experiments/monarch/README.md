# Matched Cirkit/XE Monarch experiment

This experiment compares native Cirkit folding with Extended Einsum's current
input-depth folding for full `64 x 64` grayscale ImageNet64 CP circuits. It is
a runtime and peak-memory experiment, not a likelihood-quality experiment.
Images are read lazily from the official downsampled ImageNet NPZ shards and
converted to deterministic 8-bit integer luminance.

## Matched Monarch construction

For hidden width `H = P Q`, both backends train the same two independent
raw-logit tensors:

```text
A[j,k,i] : (Q,P,P)
B[k,l,j] : (P,Q,Q)
```

Only the final contracted axes are normalized:

```text
u[b,k,j] = sum_i softmax(A[j,k,:])[i] x[b,i,j]
y[b,k,l] = sum_j softmax(B[k,l,:])[j] u[b,k,j]
```

Each replaced hidden `H -> H` sum therefore has exactly `H(P+Q)` trainable
logits. Categorical leaves and the `H -> 1` root remain dense. Quad-graph
partition mixing uses the same compact `H x arity` parameterization in both
backends.

`cirkit.py` defines the symbolic and compiled Cirkit layer. `xe.py` defines the
two XE contractions directly. `model.py` builds the shared symbolic CP circuit,
installs canonical backend-independent logits, lowers it to each backend, and
asserts equal unique trainable parameter counts.

## Publication protocol

The runner follows the conventions in `experiments/README.md`:

- one fresh process per backend, seed, graph, and parameterization;
- five seeds;
- 30 complete warmup batches;
- three blocks of 30 measured optimizer batches;
- the median per-batch timing across the three blocks;
- `torch.compile(mode="reduce-overhead")` over the complete loss function;
- LSE sum-product semantics in both backends;
- forward/loss, backward, optimizer, allocated memory, and reserved memory
  recorded separately.
- the same deterministic shard-local ImageNet64 sample order for paired arms.

The conservative 48 GiB grid is:

| Region graph | Parameterization | Hidden width | Factors | Batch |
|---|---|---:|---:|---:|
| Quad tree | Dense | 128 | — | 256 |
| Quad tree | Monarch | 512 | 16 x 32 | 256 |
| Quad graph | Dense | 64 | — | 256 |
| Quad graph | Monarch | 256 | 16 x 16 | 256 |

The widths differ deliberately: the dense and Monarch rows are representative
scalability points, while the Cirkit/XE comparison within each row is exactly
model- and parameter-matched.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo \
  python experiments/monarch/benchmark.py --data-dir datasets

uv run --group demo python experiments/monarch/table.py
```

Results are written to `experiments/monarch/results/performance.csv` and the
deterministically generated main-paper and supplementary tables to
`experiments/monarch/tables/performance_{main,supplement}.tex`. The main-paper
table contains only the two Monarch rows and reports speed only; memory remains
available in the CSV for diagnostic analysis. A second CSV,
`experiments/monarch/results/performance_batches.csv`, stores every one of the
90 measured batches per configuration, including data loading, forward/loss,
backward, optimizer, zero-gradient, model-step, end-to-end, and loss values.
Each row also carries the dataset, device, model shape, parameter count, and
initialization hash, so the raw file can be analyzed independently. The compact
CSV retains the three-block median summaries used in the table.

Successful configurations are resumed only when both the summary row and all
90 raw batch rows are present. Restrict a smoke run with `--seeds`, `--graphs`,
`--parameterizations`, or a deterministic `--sample-limit`.

## Compiler-phase diagnosis

`diagnose_compile.py` times symbolic construction, canonical initialization,
XE expression construction, program extraction, input-depth group discovery,
input-access ordering, topological scheduling, exact group-member ordering,
fold rewriting, and contraction-path optimization without invoking
`torch.compile` or using a GPU. Every phase has an independent timeout and a
hard watchdog, and records a stack trace if interrupted.

```bash
uv run --group demo python experiments/monarch/diagnose_compile.py \
  --region-graph quad-graph \
  --parameterization monarch \
  --units 256 \
  --monarch-factors 16,16 \
  --timeout-seconds 180
```

The benchmark path always keeps group-member ordering enabled.
`--no-optimize-group-order` exists only to isolate base input-depth folding in
this diagnostic; it is not an optimization mode used by the experiments.
