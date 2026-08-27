from importlib.metadata import PackageNotFoundError, version

from extended_einsum.backend_translation import BackendCompiler as BackendCompiler
from extended_einsum.backend_translation import BackendFunctions as BackendFunctions
from extended_einsum.backend_translation import run_program as run_program
from extended_einsum.backend_translation import translate_to_backend_program as translate_to_backend_program
from extended_einsum.backends.registry import get_backend_functions as get_backend_functions
from extended_einsum.backends.registry import register_backend as register_backend
from extended_einsum.interface import TensorExpression as TensorExpression
from extended_einsum.interface import TensorLeaf as TensorLeaf
from extended_einsum.interface import cos as cos
from extended_einsum.interface import einsum as einsum
from extended_einsum.interface import exp as exp
from extended_einsum.interface import extract_program as extract_program
from extended_einsum.interface import inverse as inverse
from extended_einsum.interface import log as log
from extended_einsum.interface import select as select
from extended_einsum.interface import sin as sin
from extended_einsum.interface import slice as slice
from extended_einsum.interface import softmax as softmax
from extended_einsum.interface import sqrt as sqrt
from extended_einsum.interface import stack as stack
from extended_einsum.interface import take as take
from extended_einsum.interface import tan as tan
from extended_einsum.language.rich_program import RichProgram as RichProgram
from extended_einsum.preprocess import FoldSameShapedOperations as FoldSameShapedOperations
from extended_einsum.preprocess import OptimizeContractionPaths as OptimizeContractionPaths
from extended_einsum.preprocess import PreprocessingRoutine as PreprocessingRoutine

try:
    __version__ = version("extended-einsum")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "BackendCompiler",
    "BackendFunctions",
    "FoldSameShapedOperations",
    "OptimizeContractionPaths",
    "PreprocessingRoutine",
    "RichProgram",
    "TensorExpression",
    "TensorLeaf",
    "__version__",
    "cos",
    "einsum",
    "exp",
    "extract_program",
    "get_backend_functions",
    "inverse",
    "log",
    "register_backend",
    "run_program",
    "select",
    "sin",
    "slice",
    "softmax",
    "sqrt",
    "stack",
    "take",
    "tan",
    "translate_to_backend_program",
]
