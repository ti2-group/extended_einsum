# extended-einsum

[![CI](https://github.com/ti2-group/extended_einsum/actions/workflows/ci.yml/badge.svg)](https://github.com/ti2-group/extended_einsum/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/extended-einsum.svg)](https://pypi.org/project/extended-einsum/)
[![Python](https://img.shields.io/pypi/pyversions/extended-einsum.svg)](https://pypi.org/project/extended-einsum/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ti2-group/extended_einsum/blob/main/LICENSE)

`extended-einsum` builds tensor-expression graphs, rewrites contraction programs, and executes them with numerically stable evaluation strategies. PyTorch is the primary backend; NumPy is included in the base installation and JAX is optional.

The package is an expression engine, not a drop-in replacement for `torch.einsum` or `numpy.einsum`. Pass backend arrays directly to the expression functions, compose operations lazily, then call `materialize`.

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

left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
right = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

expression = xe.einsum("ik,kj->ij", xe.exp(left), right)
result = expression.materialize(stability_mode="unstable")

assert torch.allclose(
    result,
    torch.einsum("ik,kj->ij", torch.exp(left), right),
)
```

The same interface accepts NumPy arrays. With the `jax` extra installed it also accepts JAX arrays:

```python
import numpy as np
import extended_einsum as xe

result = xe.softmax(np.arange(6.0).reshape(2, 3), axis=1).materialize()
print(result)
```

## Expressions and stability

The public interface currently includes:

- `einsum`, `stack`, `take`, `slice`, and `select`
- `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `inverse`, and `softmax`
- the arithmetic operators `+`, `-`, `*`, `/`, and `@` (numpy matmul semantics for 1-D/2-D operands), and basic indexing (`expression[0]`, `expression[:, 1:3]`)
- `TensorExpression.materialize` (returns a raw backend array) and `extract_program`

Raw backend arrays passed to these functions (and used as operands of the arithmetic operators, on either side) become expression leaves automatically when their backend is detectable. Construct a `xe.TensorLeaf` explicitly for backends registered without an `is_array` predicate (`backend=...`), for non-default tensor formats (`format=...`), or to mark a learnable parameter (`is_parameter=True`).

Supported execution backends are PyTorch, NumPy, and optional JAX. Rich programs can be evaluated in `unstable`, `scaled_min`, `scaled_max`, `scaled_sum`, `logspace_min`, or `logspace_max` mode. Support is operator-dependent; unsupported combinations raise `NotImplementedError` instead of silently changing semantics.

Custom execution backends plug in through `extended_einsum.register_backend`: subclass `extended_einsum.BackendFunctions` (only a small set of primitives is abstract; derived operations such as `softmax`, `select`, and the arithmetic operators have defaults composed from them), then register it under a name, optionally with a compiler and an `is_array` predicate for automatic backend detection of raw arrays. Without a compiler, programs are interpreted call by call; without a predicate, wrap arrays with `xe.TensorLeaf(data, backend="yourname")`. Validate an implementation with `extended_einsum.testing.check_backend`, which runs every operator through every stability mode against the NumPy reference backend.

The preprocessing API provides expression folding (`FoldSameShapedOperations`) and contraction-path optimization (`OptimizeContractionPaths`). Extract a program from an expression, rewrite it, then translate and run it:

```python
program, inputs = xe.extract_program(expression, stability_mode="unstable")
program = xe.FoldSameShapedOperations.apply(program)
program = xe.OptimizeContractionPaths.apply(program)
backend_program = xe.translate_to_backend_program(program, xe.get_backend_functions("numpy"))
result = xe.run_program(backend_program, inputs)
```

DAG plotting is available from `extended_einsum.visualization` when the visualization extra is installed.

## Examples

All introductory scripts in [`examples/`](https://github.com/ti2-group/extended_einsum/tree/main/examples) are standalone: copy one into another project with `extended-einsum` installed and run it from any working directory.

```bash
python examples/torch_quickstart.py
python examples/numpy_backend.py
python examples/stable_materialization.py
python examples/jax_backend.py                    # requires [jax]
python examples/visualize_expression.py graph.png # requires [visualization]
```

The advanced [`demo/cirkit.py`](https://github.com/ti2-group/extended_einsum/blob/main/demo/cirkit.py) integration is also standalone, but requires the repository demo dependencies and currently Python 3.12 or 3.13 because of Cirkit's SciPy constraint. Its defaults use a small synthetic circuit; training and benchmark sweeps are opt-in. Besides the fixed `quad-tree-2` and `quad-graph` region graphs, `--region-graph chow-liu-tree` builds a Hidden Chow-Liu Tree circuit whose structure is learned from the training data (binned with `--clt-bins` and cached on disk). Generated benchmark data and plots are repository artifacts and are never included in distributions.

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
