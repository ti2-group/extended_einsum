from collections.abc import Callable

import numpy as np
import torch

from extended_einsum.backend_translation import BackendArray
from extended_einsum.backend_translation.backend import BackendCompiler, BackendFunctions
from extended_einsum.backend_translation.runtime import DefaultCompiler
from extended_einsum.backends.numpy import NumpyBackendFunctions, NumpyCompiler
from extended_einsum.backends.torch import TorchBackendFunctions, TorchCompiler
from extended_einsum.language.types import Backend

BACKEND_TO_FUNCTIONS: dict[Backend, BackendFunctions] = {}
BACKEND_TO_COMPILER: dict[Backend, BackendCompiler] = {}
BACKEND_TO_IS_ARRAY: dict[Backend, Callable[[object], bool]] = {}

# Every method a backend has to provide, whether abstract or defaulted. Objects
# that subclass BackendFunctions get this via the ABC; duck-typed objects are
# checked against it explicitly at registration time.
_REQUIRED_FUNCTION_NAMES = tuple(name for name, value in vars(BackendFunctions).items() if callable(value) and not name.startswith("_"))


def register_backend(
    name: Backend,
    functions: BackendFunctions,
    compiler: BackendCompiler | None = None,
    *,
    is_array: Callable[[object], bool] | None = None,
) -> None:
    """Registers an execution backend under the given name.

    Parameters
    ----------
    name : Backend
        The name under which the backend is registered. Registering an already
        registered name replaces the previous backend.
    functions : BackendFunctions
        The backend's array operations, preferably a ``BackendFunctions``
        subclass so that missing primitives fail at instantiation time.
    compiler : BackendCompiler | None
        Optional compiler for translated programs. Without one, programs are
        interpreted call by call.
    is_array : Callable[[object], bool] | None
        Optional predicate deciding whether an object is one of this backend's
        arrays, used by ``extended_einsum.array`` to detect the backend of an
        unwrapped array. Later-registered predicates take precedence, so a
        custom backend whose arrays subclass a built-in array type still
        detects correctly. Without a predicate, arrays must be wrapped with an
        explicit ``backend=name`` argument.
    """

    if not name:
        raise ValueError("The backend name must be a non-empty string.")
    missing_function_names = [function_name for function_name in _REQUIRED_FUNCTION_NAMES if not callable(getattr(functions, function_name, None))]
    if missing_function_names:
        raise TypeError(f"The backend functions for {name!r} are missing the methods: {', '.join(sorted(missing_function_names))}.")
    if compiler is None:
        compiler = DefaultCompiler()
    elif not callable(getattr(compiler, "compile", None)):
        raise TypeError(f"The compiler for backend {name!r} must provide a callable compile method.")

    BACKEND_TO_FUNCTIONS[name] = functions
    BACKEND_TO_COMPILER[name] = compiler
    if is_array is not None:
        BACKEND_TO_IS_ARRAY[name] = is_array


def get_backend_functions(backend: Backend) -> BackendFunctions:
    try:
        return BACKEND_TO_FUNCTIONS[backend]
    except KeyError:
        raise _unknown_backend_error(backend) from None


def get_backend_compiler(backend: Backend) -> BackendCompiler:
    try:
        return BACKEND_TO_COMPILER[backend]
    except KeyError:
        raise _unknown_backend_error(backend) from None


def get_backend_of_array(array: BackendArray) -> Backend:
    # reversed so that later-registered custom predicates win over built-ins
    for name, is_array in reversed(BACKEND_TO_IS_ARRAY.items()):
        if is_array(array):
            return name
    raise ValueError(f"Unsupported array type: {type(array)}. Wrap it with an explicit backend name via array(..., backend=...), or register its backend with an is_array predicate.")


def _unknown_backend_error(backend: Backend) -> Exception:
    if backend == "jax":
        return ModuleNotFoundError("The JAX backend is optional. Install it with `pip install 'extended-einsum[jax]'`.")
    return ValueError(f"No backend is registered under the name {backend!r}. Registered backends: {sorted(BACKEND_TO_FUNCTIONS)}. Register custom backends with register_backend.")


def _is_numpy_array(array: object) -> bool:
    return isinstance(array, np.ndarray)


def _is_torch_array(array: object) -> bool:

    return isinstance(array, torch.Tensor)


register_backend("numpy", NumpyBackendFunctions(), NumpyCompiler(), is_array=_is_numpy_array)
register_backend("torch", TorchBackendFunctions(), TorchCompiler(), is_array=_is_torch_array)

try:
    from extended_einsum.backends.jax import JaxBackendFunctions, JaxCompiler
except ModuleNotFoundError as exc:
    if exc.name not in {"jax", "jaxlib"}:
        raise
else:
    import jax

    def _is_jax_array(array: object) -> bool:

        return isinstance(array, jax.Array)

    register_backend("jax", JaxBackendFunctions(), JaxCompiler(), is_array=_is_jax_array)
