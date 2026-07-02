"""Block-chain (block-tridiagonal) selected matrix inversion.

Given a symmetric positive-definite block-tridiagonal matrix ``A``, this module computes the
blocks of ``A^{-1}`` that lie on ``A``'s sparsity pattern (the diagonal and first
sub-diagonal blocks) without ever forming the dense inverse.

Algorithm: block ``LDL^T`` factorization down the chain followed by the Takahashi
back-recursion. Per step the work is ``O(b**3)`` on ``b x b`` blocks, for a total of
``O(L * b**3)`` time and ``O(L * b**2)`` storage. With ``N = L * b`` this is
``O(N * b**2)`` -- linear in ``N`` only at fixed block size ``b``.

All ``b x b`` block operations are batched over any leading dimensions of the inputs
(``[..., L, b, b]``). The only Python-level loop is over the sequential chain index
``L``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._linalg import cholesky_spd, inv_via_chol
from .layout import BlockTridiag

__all__ = ["selected_inverse_chain", "ChainFactors"]


class ChainFactors:
    """Factors from the forward block-``LDL^T`` pass (returned when requested).

    Attributes
    ----------
    chol_D:
        Cholesky factors of the block pivots ``D_i``: ``[..., L, b, b]`` lower-triangular.
    ell:
        Unit-lower sub-diagonal factor blocks ``ell_i`` for ``i = 2..L``:
        ``[..., L-1, b, b]`` (``ell[..., k]`` corresponds to ``ell_{k+2}``).
    """

    __slots__ = ("chol_D", "ell")

    def __init__(self, chol_D: Tensor, ell: Tensor) -> None:
        self.chol_D = chol_D
        self.ell = ell


def _cholesky(block: Tensor, *, block_index: int) -> Tensor:
    """Batched SPD Cholesky of pivot ``D_{block_index}`` (see :func:`cholesky_spd`)."""
    return cholesky_spd(block, name=f"D_{block_index}")


def selected_inverse_chain(
    diag: Tensor,
    lower: Tensor,
    *,
    check: bool = False,
    return_factors: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Selected inverse of a symmetric block-tridiagonal SPD matrix.

    Parameters
    ----------
    diag:
        Diagonal blocks ``A_ii``: ``[..., L, b, b]`` (each block symmetric).
    lower:
        Sub-diagonal blocks ``A_{i+1,i}``: ``[..., L-1, b, b]``.
    check:
        If ``True``, validate inputs (shape / dtype / symmetry) before factorizing.
    return_factors:
        If ``True``, also return the :class:`ChainFactors` from the forward pass.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs are upcast and the
        result cast back to the input dtype). Used by the precision sweep to model
        low-precision *storage* while computing factorizations in a dtype the LAPACK
        backend supports (``cholesky``/``cholesky_solve`` have no half kernel on CPU).

    Returns
    -------
    (G_diag, G_lower) or (G_diag, G_lower, factors):
        ``G_diag`` ``[..., L, b, b]`` holds the on-pattern diagonal blocks of
        ``A^{-1}``; ``G_lower`` ``[..., L-1, b, b]`` holds the sub-diagonal blocks
        (``G_{i,i+1} = G_lower[..., i].mT``).
    """
    if check:
        BlockTridiag(diag=diag, lower=lower).validate()

    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        lower = lower.to(compute_dtype)

    L = diag.shape[-3]
    b = diag.shape[-1]
    batch = diag.shape[:-3]

    # ---- forward block-LDL^T factorization (sequential over L) --------------
    chol_D = diag.new_empty((*batch, L, b, b))
    ell = diag.new_empty((*batch, max(L - 1, 0), b, b))

    chol_prev = _cholesky(diag[..., 0, :, :], block_index=1)
    chol_D[..., 0, :, :] = chol_prev
    for i in range(1, L):
        C = lower[..., i - 1, :, :]              # A_{i+1,i} == A_{i,i-1} in 1-based terms
        # Right-apply D_{i-1}^{-1}: ell_i = C @ D_{i-1}^{-1}.
        # cholesky_solve(B, chol) solves D X = B (left-apply), so solve for D^{-1} C^T
        # and transpose.  Writing cholesky_solve(C, chol) would give D^{-1} C -- wrong.
        X = torch.cholesky_solve(C.mT, chol_prev)   # X = D_{i-1}^{-1} C^T
        ell_i = X.mT                                # = C @ D_{i-1}^{-1}
        D_i = diag[..., i, :, :] - C @ X            # = A_ii - C D_{i-1}^{-1} C^T
        D_i = 0.5 * (D_i + D_i.mT)                  # symmetrize the pivot (policy: see derivations.md §6)
        chol_prev = _cholesky(D_i, block_index=i + 1)
        chol_D[..., i, :, :] = chol_prev
        ell[..., i - 1, :, :] = ell_i

    # ---- Takahashi back-recursion (selected inverse) ------------------------
    G_diag = diag.new_empty((*batch, L, b, b))
    G_lower = diag.new_empty((*batch, max(L - 1, 0), b, b))

    G_next = inv_via_chol(chol_D[..., L - 1, :, :])   # G_LL = D_L^{-1}
    G_diag[..., L - 1, :, :] = G_next
    for i in range(L - 2, -1, -1):
        ell_ip1 = ell[..., i, :, :]                    # ell_{i+1}
        G_lo = -(G_next @ ell_ip1)                     # G_{i+1,i} = -G_{i+1,i+1} ell_{i+1}
        Dinv = inv_via_chol(chol_D[..., i, :, :])
        G_ii = Dinv - ell_ip1.mT @ G_lo                # = D_i^{-1} + ell^T G_{i+1,i+1} ell
        G_ii = 0.5 * (G_ii + G_ii.mT)                  # enforce symmetry
        G_lower[..., i, :, :] = G_lo
        G_diag[..., i, :, :] = G_ii
        G_next = G_ii

    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_lower = G_lower.to(in_dtype)
        chol_D = chol_D.to(in_dtype)
        ell = ell.to(in_dtype)

    if return_factors:
        return G_diag, G_lower, ChainFactors(chol_D=chol_D, ell=ell)
    return G_diag, G_lower
