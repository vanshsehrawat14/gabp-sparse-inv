"""Shared batched linear-algebra helpers.

These back every selected-inverse kernel (chain, star, ...): a batched Cholesky
with an informative SPD-failure message, an explicit SPD-block inverse from a
Cholesky factor, and the element-growth guard of the static no-pivot
non-symmetric eliminations. Keeping them in one place means there is a single,
tested path regardless of the matrix structure being inverted.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cholesky_spd", "inv_via_chol", "block_amax", "check_pivot_growth"]


def cholesky_spd(block: Tensor, *, name: str) -> Tensor:
    """Batched Cholesky via ``cholesky_ex`` with an informative SPD failure message.

    ``cholesky_ex`` reports failure through an ``info`` code per batch element instead
    of raising inside the kernel. Inspecting ``info`` here still costs one host
    synchronization on accelerators.

    ``name`` identifies the failing pivot in the error message (e.g. ``"D_3"`` or
    ``"S (center Schur complement)"``).
    """
    chol, info = torch.linalg.cholesky_ex(block)
    if torch.any(info != 0):
        where = (
            f"batch indices {torch.nonzero(info != 0, as_tuple=False).tolist()}"
            if info.ndim > 0
            else "an unbatched input"
        )
        raise torch.linalg.LinAlgError(
            f"pivot block {name} lost positive-definiteness during factorization "
            f"(cholesky_ex info != 0 for {where}); matrix is not SPD or is too "
            f"ill-conditioned"
        )
    return chol


def inv_via_chol(chol: Tensor) -> Tensor:
    """Explicit inverse of an SPD block from its Cholesky factor."""
    eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device)
    eye = eye.expand_as(chol)
    return torch.cholesky_solve(eye, chol)


def block_amax(x: Tensor, dims: int = 2) -> Tensor:
    """Largest absolute entry over the trailing ``dims`` dimensions, detached."""
    return x.detach().abs().amax(dim=tuple(range(-dims, 0)))


def check_pivot_growth(growth: Tensor, scale: Tensor, dtype: torch.dtype) -> None:
    """Refuse a static no-pivot elimination whose element growth is unsafe.

    ``growth`` is the largest entry magnitude of any reduced pivot, reduced coupling
    block or Schur update formed during elimination, and ``scale`` the largest input
    entry, both per batch element. Without pivoting the backward error of block LU is
    proportional to ``growth / scale`` times the unit roundoff, so beyond
    ``1 / sqrt(eps)`` the computed inverse blocks can be wrong in every digit even
    when ``A`` is well conditioned. Non-finite growth (overflow, or ``inf``/``nan``
    input) is refused as well.
    """
    ratio = growth / scale.clamp_min(torch.finfo(dtype).tiny)
    limit = torch.finfo(dtype).eps ** -0.5
    if not bool(torch.all(ratio <= limit)):
        raise torch.linalg.LinAlgError(
            f"element growth {float(ratio.max()):.3g} exceeds {limit:.3g} = 1/sqrt(eps) for {dtype} "
            "during static no-pivot elimination; the result would be numerically "
            "unreliable. Use an elimination order that keeps the leading pivots "
            "well conditioned, or a pivoted LU."
        )
