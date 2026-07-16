# Contributing

Contributions and bug reports are welcome through the GitHub repository.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the full CPU development environment:

```bash
uv sync --group dev --group demo --extra jax --extra visualization
```

Use Python 3.12 or 3.13 for the full Cirkit demo environment; Cirkit's current SciPy constraint is not compatible with Python 3.14. The core package, JAX extra, and visualization extra are tested on Python 3.12–3.14. The `jax-cuda` extra is intentionally excluded from the standard development environment and CI.

Before opening a pull request, run:

```bash
uv run ruff check .
uv run pytest
uv build
uvx twine check dist/*
```

Tests must be deterministic and should not require a GPU, network access, downloaded datasets, or a repository-relative working directory. Keep generated benchmark outputs out of package source and tests.

## Project conventions

- Public operations are exported from both `extended_einsum` and the backwards-compatible `extended_einsum.interface` namespace.
- New optional integrations must import lazily and provide an actionable installation error.
- Add user-visible changes to `CHANGELOG.md`.
- Do not change numerical semantics or silently fall back between stability modes.

Releases are produced only by the trusted publishing workflows described in [PUBLISHING.md](PUBLISHING.md).
