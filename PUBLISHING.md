# Publishing

Releases are built by GitHub Actions and published with PyPI Trusted Publishing. Do not create or store a long-lived PyPI API token.

## One-time setup

1. Confirm that the `extended-einsum` project name is available on PyPI and TestPyPI.
2. In the GitHub repository, create protected environments named `pypi` and `testpypi`. Add required reviewers to `pypi`.
3. Configure a pending publisher for a new project, or a trusted publisher for an existing project, with these exact identities:

   | Index | Owner | Repository | Workflow | Environment |
   | --- | --- | --- | --- | --- |
   | PyPI | `ti2-group` | `extended_einsum` | `publish.yml` | `pypi` |
   | TestPyPI | `ti2-group` | `extended_einsum` | `test-publish.yml` | `testpypi` |

The publisher configuration must match the workflow filename and environment exactly.

## Preflight on TestPyPI

1. Set the intended version in `pyproject.toml` and update `CHANGELOG.md`.
2. Ensure CI passes on the release commit.
3. Run the **Publish to TestPyPI** workflow manually.
4. Install from TestPyPI in a clean environment. PyTorch and the other dependencies should still come from PyPI:

   ```bash
   python -m venv /tmp/extended-einsum-test
   /tmp/extended-einsum-test/bin/pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     extended-einsum==0.1.0
   /tmp/extended-einsum-test/bin/python -c "import extended_einsum as xe; print(xe.__version__)"
   ```

TestPyPI versions cannot be overwritten. Increment the development version before repeating a failed upload.

## Production release

1. Merge the tested release commit into `main`.
2. Create a GitHub Release whose tag is exactly `vX.Y.Z`, matching the version in `pyproject.toml`.
3. Publish the GitHub Release. The `publish.yml` workflow validates the tag, rebuilds and checks the distributions, and pauses at the protected `pypi` environment before publishing the exact checked artifacts.

PyPI files are immutable. Fixes after publication require a new version.
