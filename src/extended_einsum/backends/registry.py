from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions
from extended_einsum.backends.numpy import NumpyBackendFunctions, NumpyCompiler
from extended_einsum.backends.torch import TorchBackendFunctions, TorchCompiler
from extended_einsum.language.types import Backend

BACKEND_TO_FUNCTIONS: dict[Backend, BackendFunctions] = {
    "numpy": NumpyBackendFunctions(),
    "torch": TorchBackendFunctions(),
}

BACKEND_TO_COMPILER: dict[Backend, BackendCompiler] = {
    "numpy": NumpyCompiler(),
    "torch": TorchCompiler(),
}

try:
    from extended_einsum.backends.jax import JaxBackendFunctions, JaxCompiler
except ModuleNotFoundError as exc:
    if exc.name not in {"jax", "jaxlib"}:
        raise
else:
    BACKEND_TO_FUNCTIONS["jax"] = JaxBackendFunctions()
    BACKEND_TO_COMPILER["jax"] = JaxCompiler()


def get_backend_functions(backend: Backend) -> BackendFunctions:
    try:
        return BACKEND_TO_FUNCTIONS[backend]
    except KeyError:
        if backend == "jax":
            raise ModuleNotFoundError("The JAX backend is optional. Install it with `pip install 'extended-einsum[jax]'`.") from None
        raise ValueError(f"Unsupported backend: {backend}") from None


def get_backend_compiler(backend: Backend) -> BackendCompiler:
    try:
        return BACKEND_TO_COMPILER[backend]
    except KeyError:
        if backend == "jax":
            raise ModuleNotFoundError("The JAX backend is optional. Install it with `pip install 'extended-einsum[jax]'`.") from None
        raise ValueError(f"Unsupported backend: {backend}") from None
