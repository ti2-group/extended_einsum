from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions
from extended_einsum.backends.jax import JaxBackendFunctions, JaxCompiler
from extended_einsum.backends.numpy import NumpyBackendFunctions, NumpyCompiler
from extended_einsum.backends.torch import TorchBackendFunctions, TorchCompiler
from extended_einsum.language.types import Backend

BACKEND_TO_FUNCTIONS: dict[Backend, BackendFunctions] = {
    "numpy": NumpyBackendFunctions(),
    "torch": TorchBackendFunctions(),
    "jax": JaxBackendFunctions(),
}

BACKEND_TO_COMPILER: dict[Backend, BackendCompiler] = {
    "numpy": NumpyCompiler(),
    "torch": TorchCompiler(),
    "jax": JaxCompiler(),
}
