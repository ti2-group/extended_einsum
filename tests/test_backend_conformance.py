import numpy as np
import pytest
import torch

from extended_einsum.backends.registry import register_backend
from extended_einsum.backends.torch import TorchBackendFunctions
from extended_einsum.testing import check_backend


def test_numpy_backend_conformance() -> None:
    check_backend("numpy", from_numpy=lambda array: array, to_numpy=np.asarray)


def test_torch_backend_functions_conformance() -> None:
    # An interpreted registration of the torch functions: running the full
    # conformance matrix through torch.compile would dominate the test suite's
    # runtime, and the compiled path is covered by the other torch tests.
    register_backend("torch_interpreted", TorchBackendFunctions())

    check_backend(
        "torch_interpreted",
        from_numpy=torch.from_numpy,
        to_numpy=lambda tensor: tensor.detach().cpu().numpy(),
    )


def test_jax_backend_functions_conformance() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from extended_einsum.backends.jax import JaxBackendFunctions

    register_backend("jax_interpreted", JaxBackendFunctions())

    # jax computes in single precision by default, so the tolerances are loosened
    check_backend(
        "jax_interpreted",
        from_numpy=jnp.asarray,
        to_numpy=np.asarray,
        rtol=1e-4,
        atol=1e-6,
    )
