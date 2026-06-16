from extended_einsum.backend import BackendCompiler, BackendFunctions
from src.extended_einsum.translations.jax import JaxCompiler, JaxTranslation

BACKEND_TO_TRANSLATION: dict[str, BackendFunctions] = {
    "jax": JaxTranslation(),
}

BACKEND_TO_COMPILER: dict[str, BackendCompiler] = {
    "jax": JaxCompiler(),
}
