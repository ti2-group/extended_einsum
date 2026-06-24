from extended_einsum.backend import BackendCompiler, BackendFunctions
from extended_einsum.language.types import Backend, StabilityMode
from extended_einsum.translations.jax import JaxCompiler, JaxTranslation
from extended_einsum.translations.torch import TorchCompiler, TorchTranslation

STABILITY_AND_BACKEND_TO_TRANSLATION: dict[
    StabilityMode, dict[Backend, BackendFunctions]
] = {
    "none": {
        "jax": JaxTranslation(),
        "torch": TorchTranslation(),
    },
    "scaled": {
        "jax": JaxTranslation(),
        "torch": TorchTranslation(),
    },
    "logspace": {
        "jax": JaxTranslation(),
        "torch": TorchTranslation(),
    },
}

BACKEND_TO_COMPILER: dict[Backend, BackendCompiler] = {
    "jax": JaxCompiler(),
    "torch": TorchCompiler(),
}
