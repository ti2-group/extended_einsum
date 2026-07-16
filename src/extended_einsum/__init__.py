from importlib.metadata import PackageNotFoundError, version

from extended_einsum.interface import TensorExpression as TensorExpression
from extended_einsum.interface import array as array
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

try:
    __version__ = version("extended-einsum")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "TensorExpression",
    "__version__",
    "array",
    "cos",
    "einsum",
    "exp",
    "extract_program",
    "inverse",
    "log",
    "select",
    "sin",
    "slice",
    "softmax",
    "sqrt",
    "stack",
    "take",
    "tan",
]
