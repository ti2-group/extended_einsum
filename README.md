# extended-einsum

[![CI](https://github.com/ti2-group/extended_einsum/actions/workflows/ci.yml/badge.svg)](https://github.com/ti2-group/extended_einsum/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/extended-einsum.svg)](https://pypi.org/project/extended-einsum/)
[![Python](https://img.shields.io/pypi/pyversions/extended-einsum.svg)](https://pypi.org/project/extended-einsum/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ti2-group/extended_einsum/blob/main/LICENSE)

`extended-einsum` builds tensor-expression graphs, rewrites contraction programs, and executes them with numerically stable evaluation strategies. PyTorch is the primary backend; NumPy is included in the base installation and JAX is optional.

The package is an expression engine, not a drop-in replacement for `torch.einsum` or `numpy.einsum`. Wrap backend arrays with `xe.array`, compose operations lazily, then call `materialize`.

## Installation

Python 3.12, 3.13, and 3.14 are supported.

```bash
pip install extended-einsum
```

Optional features are installed explicitly:

```bash
pip install "extended-einsum[jax]"
pip install "extended-einsum[jax-cuda]"
pip install "extended-einsum[visualization]"
```

The CUDA extra follows JAX's CUDA 13 installation. PyTorch device support is determined by the PyTorch build available from your configured package index.

## Quickstart

```python
import torch
import extended_einsum as xe

left_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
right_torch = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

left = xe.array(left_torch)
right = xe.array(right_torch)
expression = xe.einsum("ik,kj->ij", xe.exp(left), right)
result = expression.materialize(stability_mode="unstable")

assert torch.allclose(
    result.backend_array,
    torch.einsum("ik,kj->ij", torch.exp(left_torch), right_torch),
)
```

The same interface accepts NumPy arrays. With the `jax` extra installed it also accepts JAX arrays:

```python
import numpy as np
import extended_einsum as xe

source = xe.array(np.arange(6.0).reshape(2, 3))
result = xe.softmax(source, axis=1).materialize()
print(result.backend_array)
```

## Expressions and stability

The public interface currently includes:

- `array`, `einsum`, `stack`, `take`, `slice`, and `select`
- `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `inverse`, and `softmax`
- `TensorExpression.materialize` and `extract_program`

Supported execution backends are PyTorch, NumPy, and optional JAX. Rich programs can be evaluated in `unstable`, `scaled_min`, `scaled_max`, `scaled_sum`, `logspace_min`, or `logspace_max` mode. Support is operator-dependent; unsupported combinations raise `NotImplementedError` instead of silently changing semantics.

The preprocessing API also provides expression folding and contraction-path optimization. DAG plotting is available from `extended_einsum.visualization` when the visualization extra is installed.

## Examples

All introductory scripts in [`examples/`](https://github.com/ti2-group/extended_einsum/tree/main/examples) are standalone: copy one into another project with `extended-einsum` installed and run it from any working directory.

```bash
python examples/torch_quickstart.py
python examples/numpy_backend.py
python examples/stable_materialization.py
python examples/jax_backend.py                    # requires [jax]
python examples/visualize_expression.py graph.png # requires [visualization]
```

The advanced [`demo/cirkit.py`](demo/cirkit.py) integration requires the repository demo dependencies and currently Python 3.12 or 3.13 because of Cirkit's SciPy constraint. The publication experiments use it as the shared Cirkit/XE model adapter.

## Reproducing the paper experiments

The complete, current reproduction workflow is documented in
[`experiments/README.md`](experiments/README.md). The publication suite has
four isolated experiments:

- CP and Tucker runtime and peak-memory comparisons between production
  Extended Einsum and untouched Cirkit.
- CP and Tucker ablations of log-space arithmetic, shift gradients, their
  combined effect, and consumer ordering.
- An exactly parameter-matched CP-T comparison with PyJuice 2.6.1, production
  Extended Einsum, and Cirkit.
- A matched dense/Monarch CP comparison on full `64 x 64` grayscale ImageNet64
  inputs, using native Cirkit folding and Extended Einsum input-depth folding.

Every configuration runs in a fresh Python process. The common training
protocol uses 30 warmup batches, 90 measured batches, and five seeds. The
compact CSVs contain one summary row per seed and configuration.

Run the complete suite sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 experiments/run_all.sh
```

Or run and plot the experiments individually:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/speedup.py
CUDA_VISIBLE_DEVICES=0 uv run --group demo python experiments/ablation.py
CUDA_VISIBLE_DEVICES=0 uv run --group demo --with pyjuice==2.6.1 \
  python experiments/pyjuice_cp_t/benchmark.py
CUDA_VISIBLE_DEVICES=0 uv run --group demo \
  python experiments/monarch/benchmark.py

uv run --group demo python experiments/plot_speedup.py
uv run --group demo python experiments/plot_ablation.py
uv run --group demo python experiments/pyjuice_cp_t/plot.py
uv run --group demo python experiments/monarch/table.py
```

Results are written to `experiments/results/` and
`experiments/pyjuice_cp_t/results/`. Paper-ready PDFs are written to
`experiments/plots/` and `experiments/pyjuice_cp_t/plots/`. Generated CSVs are
excluded from distributions. The Monarch experiment keeps its results under
`experiments/monarch/results/` and its LaTeX tables under
`experiments/monarch/tables/`.

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev --group demo --extra jax --extra visualization
uv run ruff check .
uv run pytest
uv build
```

See [CONTRIBUTING.md](https://github.com/ti2-group/extended_einsum/blob/main/CONTRIBUTING.md) for development details and [PUBLISHING.md](https://github.com/ti2-group/extended_einsum/blob/main/PUBLISHING.md) for the release process.

## License

Copyright © 2026 FSU Theoretical Computer Science II. Distributed under the [MIT License](https://github.com/ti2-group/extended_einsum/blob/main/LICENSE).
