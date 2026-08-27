# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Hidden Chow-Liu Tree (HCLT) circuits in the Cirkit demo: `--region-graph chow-liu-tree` learns the tree structure from the training data (binned mutual information, cached on disk under the data directory) and records the structure-learning time and bin count in the benchmark CSV.
- Raw backend arrays (e.g. `numpy.ndarray`, `torch.Tensor`) passed to expression functions or used as operands of the arithmetic operators (on either side) become expression leaves automatically when their backend is detectable; Python scalars and other unsupported operands raise a `TypeError` explaining what to pass instead.
- `xe.TensorLeaf`: a tensor expression holding a raw backend array, for backends registered without an `is_array` predicate (`backend=...`), non-default tensor formats (`format=...`), and marking learnable parameters (`is_parameter=True`, replacing the internal `Parameter` wrapper).
- Basic indexing on expressions: `expression[0]`, `expression[:, 1:3]`, and tuples of integers and slices build `select` and `slice` operators (steps are not supported).
- `@` follows numpy's matmul semantics for all 1-D/2-D operand combinations; higher-dimensional operands raise a `ValueError` pointing to `einsum`.
- The preprocessing pipeline is part of the top-level API: `FoldSameShapedOperations`, `OptimizeContractionPaths`, `PreprocessingRoutine`, `RichProgram`, `translate_to_backend_program`, `run_program`, and `get_backend_functions`.
- `TensorExpression` has an informative `repr` showing the operator, shape, and backend.

### Changed

- **Breaking:** `xe.array` and the wrapped-array type are removed. Expressions are built directly from raw backend arrays; `materialize` and the input list of `extract_program` return raw backend arrays instead of wrapped ones.
- **Breaking:** reusing the same raw array in an expression produces a single program input (inputs are deduplicated by array identity).
- **Breaking:** `stack` interprets `axis` relative to the output rank, matching `numpy.stack`: `axis=-1` now stacks along a new trailing axis, and `axis=rank` is accepted.
- **Breaking:** `softmax` requires an explicit keyword-only `axis` argument; the previous default of `axis=0` silently normalized the wrong axis for code ported from other libraries.
- `materialize` validates the stability mode eagerly and raises a `ValueError` listing the valid modes, instead of a late `NotImplementedError` indistinguishable from an unsupported operator/mode combination.
- `xe.TensorLeaf(..., backend=...)` validates the backend name at construction instead of at materialize.
- `materialize` no longer stores intermediate compilation state on the expression object.
- Error messages across `einsum`, `stack`, `take`, and format-string parsing name the offending operand, shapes, and terms consistently; einsum axis-size conflicts raise `ValueError` instead of `RuntimeError`; format strings without `->` explain that an explicit output is required; visualization import errors name the `[visualization]` extra.

### Fixed

- `extract_program` compiles expression graphs iteratively, so expressions deeper than Python's recursion limit (such as learned tree-structured circuits) no longer raise `RecursionError`.
- `stack([])` raises a clear `ValueError` instead of an `IndexError`.

## 0.1.0 - 2026-07-16

### Added

- Lazy tensor-expression construction and rich-program extraction.
- PyTorch and NumPy execution backends, with optional JAX support.
- Unstable, scaled, and log-space evaluation strategies.
- Expression folding, contraction-path optimization, and DAG visualization.
- Standalone examples and Cirkit integration demos.

### Infrastructure

- Python 3.12–3.14 test coverage.
- Reproducible uv-based builds and PyPI Trusted Publishing workflows.
