# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Hidden Chow-Liu Tree (HCLT) circuits in the Cirkit demo: `--region-graph chow-liu-tree` learns the tree structure from the training data (binned mutual information, cached on disk under the data directory) and records the structure-learning time and bin count in the benchmark CSV.

### Fixed

- `extract_program` compiles expression graphs iteratively, so expressions deeper than Python's recursion limit (such as learned tree-structured circuits) no longer raise `RecursionError`.

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
