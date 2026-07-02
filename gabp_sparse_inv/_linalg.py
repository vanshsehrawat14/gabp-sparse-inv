"""Shared batched SPD linear-algebra helpers.

These back every selected-inverse kernel (chain, star, ...): a batched Cholesky
with an informative SPD-failure message and an explicit SPD-block inverse from a
Cholesky factor. Keeping them in one place means there is a single, tested
batched-Cholesky path regardless of the matrix structure being inverted.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cholesky_spd", "inv_via_chol"]


def cholesky_spd(block: Tensor, *, name: str) -> Tensor:
    """Batched Cholesky via ``cholesky_ex`` with an informative SPD failure message.

    ``cholesky_ex`` avoids the CUDA-side error-check synchronization of
    ``torch.linalg.cholesky`` and returns an ``info`` code per batch element.

    ``name`` identifies the failing pivot in the error message (e.g. ``"D_3"`` or
    ``"S (center Schur complement)"``).
    """
    chol, info = torch.linalg.cholesky_ex(block)
    if torch.any(info != 0):
        bad = torch.nonzero(info != 0, as_tuple=False)
        raise torch.linalg.LinAlgError(
            f"pivot block {name} lost positive-definiteness during factorization "
            f"(cholesky_ex info != 0 at batch indices {bad.tolist()}); matrix is "
            f"not SPD or is too ill-conditioned"
        )
    return chol


def inv_via_chol(chol: Tensor) -> Tensor:
    """Explicit inverse of an SPD block from its Cholesky factor."""
    eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device)
    eye = eye.expand_as(chol)
    return torch.cholesky_solve(eye, chol)
