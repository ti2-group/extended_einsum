"""XE construction for the untied generalized Monarch transformation.

The canonical layouts match Section 3.1 of the Monarch Circuits paper:

    A[j, k, i] has shape (Q, P, P)
    B[k, l, j] has shape (P, Q, Q)

Only the last (contracted) axis of each raw-logit tensor is normalized.
"""

from dataclasses import dataclass

import torch

import extended_einsum.interface as xe
from extended_einsum.interface.tensor_expression import Parameter


@dataclass(frozen=True)
class XEMonarchParameters:
    """The two independent XE raw-logit parameters."""

    factor_a: Parameter
    factor_b: Parameter
    p: int
    q: int

    @property
    def num_parameters(self) -> int:
        return self.q * self.p * self.p + self.p * self.q * self.q


def factor_shapes(p: int, q: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return canonical ``A`` and ``B`` shapes for hidden width ``H=P*Q``."""
    if p <= 0 or q <= 0:
        raise ValueError(f"Monarch factors must be positive, found p={p}, q={q}")
    return (q, p, p), (p, q, q)


def make_parameters(
    p: int,
    q: int,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> XEMonarchParameters:
    """Create independent canonical XE raw-logit tensors on CPU."""
    factor_a_shape, factor_b_shape = factor_shapes(p, q)
    factor_a = Parameter(xe.array(torch.randn(factor_a_shape, generator=generator, dtype=dtype)))
    factor_b = Parameter(xe.array(torch.randn(factor_b_shape, generator=generator, dtype=dtype)))
    return XEMonarchParameters(factor_a=factor_a, factor_b=factor_b, p=p, q=q)


def transform(
    inputs,
    factor_a_logits,
    factor_b_logits,
):
    """Apply the two Monarch contractions to an XE tensor of shape ``(B,P,Q)``.

    This implements

    ``u[b,k,j] = sum_i softmax(A[j,k,:])[i] * x[b,i,j]``

    followed by

    ``y[b,k,l] = sum_j softmax(B[k,l,:])[j] * u[b,k,j]``.
    """
    if len(inputs.shape) != 3:
        raise ValueError(f"Expected inputs with shape (batch,P,Q), found {inputs.shape}")
    p = int(inputs.shape[1])
    q = int(inputs.shape[2])
    expected_a, expected_b = factor_shapes(p, q)
    if tuple(factor_a_logits.shape) != expected_a:
        raise ValueError(f"Expected factor A shape {expected_a}, found {factor_a_logits.shape}")
    if tuple(factor_b_logits.shape) != expected_b:
        raise ValueError(f"Expected factor B shape {expected_b}, found {factor_b_logits.shape}")

    factor_a = xe.softmax(factor_a_logits, axis=2)
    factor_b = xe.softmax(factor_b_logits, axis=2)
    intermediate = xe.einsum("bij,jki->bkj", inputs, factor_a)
    return xe.einsum("bkj,klj->bkl", intermediate, factor_b)


def transform_with_parameters(inputs, parameters: XEMonarchParameters):
    """Apply :func:`transform` using a parameter bundle from :func:`make_parameters`."""
    return transform(inputs, parameters.factor_a, parameters.factor_b)
