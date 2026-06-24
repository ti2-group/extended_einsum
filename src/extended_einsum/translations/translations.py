from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.language.types import Backend
from extended_einsum.translations.jax import JaxCompiler, JaxTranslation
from extended_einsum.translations.torch import TorchCompiler, TorchTranslation

BACKEND_TO_TRANSLATION: dict[Backend, BackendFunctions] = {
    "jax": JaxTranslation(),
    "torch": TorchTranslation(),
}

BACKEND_TO_COMPILER: dict[Backend, BackendCompiler] = {
    "jax": JaxCompiler(),
    "torch": TorchCompiler(),
}
