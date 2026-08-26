# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/).

## 1.1.1 - 2026-08-26

### Added

- Added preprocessing annotations for short same-index weighted reductions so stable backends can lower them to fused multiply-and-reduce operations without changing the rich-program structure.
- Expanded the Cirkit demo with configurable folding, input lookup, contraction-path, stability-scaling, compilation, optimizer, and memory-measurement strategies.

### Changed

- Reworked input-depth folding to choose between contiguous slice/concatenate routing and indexed gathers based on program structure and estimated routing cost.
- Removed the ineffective input-access fold-ordering path and narrowed the Cirkit demo's routing override to automatic routing or explicitly forced gathers.
- Updated automatic categorical lookup selection for scaled 64-unit CP quad graphs.
- Changed scaled evaluation to normalize according to contraction depth, with a configurable interval and a default of three contractions, while still normalizing at representation boundaries.
- Improved consumer-aware fold ordering, input preordering, parameter packing, and gather materialization for shared and deeply folded expression graphs.
- Extended Cirkit benchmark output with reproducibility, compilation, routing, optimizer, and incremental memory metrics.

### Performance

- Reduced compilation and execution overhead for stable folded circuits by avoiding unnecessary intermediate normalizations and routing operations.
- Optimized short stable contractions to avoid inefficient dense matrix-vector lowering while preserving forward values and gradients.

## 1.1.0 - 2026-08-24

### Added

- Added `register_backend` for custom execution backends, including optional array-type detection and custom compilers.
- Exported `BackendFunctions` and `BackendCompiler` from the top-level package for backend implementers.
- Added `extended_einsum.testing.check_backend` to validate registered backends across the supported operators and stability modes against NumPy.
- Added a default interpreter compiler for backends that do not provide their own compiler.

### Changed

- Replaced the monolithic backend protocol with an abstract primitive interface and reusable default implementations for indexing, softmax, and arithmetic operations.
- Moved built-in NumPy, PyTorch, and optional JAX backend selection to the same registry used by custom backends.
- Improved errors for unknown backends, missing optional JAX support, unsupported array types, and incomplete backend implementations.

## 1.0.0 - 2026-08-21

### Added

- Added `scaled_max` evaluation alongside the existing unstable, scaled-minimum, scaled-sum, and log-space strategies.
- Added input-depth folding as an alternative to output-depth grouping, including input-access ordering and routing-aware group splitting.
- Added depth-preserving optimization of connected einsum components and pairwise contraction paths.

### Changed

- Reworked same-shaped operation folding to use consumer-aware group ordering and expose input permutations and parameter stack layouts as preprocessing metadata.
- Improved stable handling of folded and concatenated values whose scale factors vary across fibers.
- Removed the alpha-status warning after validating the public expression API and core stability modes.

### Performance

- Reduced materialization operations through broader folding, canonicalized einsum grouping, optimized fold-axis layouts, and input preordering.
- Improved execution of large repeated tensor-expression graphs while keeping contraction rewrites from increasing DAG depth.

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
