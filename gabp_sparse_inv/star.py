"""Block-star (block-arrowhead) selected matrix inversion.

A star is the smallest *branching* tree: one center block ``A_00`` coupled to ``K``
leaf blocks ``A_jj`` via ``A_0j``, with no leaf--leaf coupling. Given such a
symmetric positive-definite block-arrowhead matrix, this module computes the blocks of
``A^{-1}`` that lie on the pattern -- the center diagonal ``G_00``, the leaf
diagonals ``G_jj``, and the center<->leaf cross blocks ``G_{j,0}`` -- without ever
forming the dense inverse.

Algorithm (eliminate leaves, i.e. Gaussian belief propagation collect/distribute
on a depth-1 tree). With ``B_j = A_jj^{-1} A_{j,0}`` and the center Schur
complement ``S = A_00 - sum_j A_0j B_j``:

    G_00   = S^{-1}
    G_{j,0} = -B_j G_00                 (cross block; G_{0,j} = G_{j,0}^T)
    G_jj   = A_jj^{-1} + B_j G_00 B_j^T

The leaves are mutually independent, so every per-leaf operation is *batched over
the leaf axis* ``K`` -- the only reduction is the sum that forms ``S``. There is no
Python-level loop over leaves. Cost ``O(K * b**3)`` time, ``O(K * b**2)`` storage.

The off-pattern leaf--leaf inverse blocks ``G_{j,k} = B_j G_00 B_k^T`` (``j != k``)
are generally nonzero but are *not* part of the selected target and are never
formed. See ``docs/derivations.md``.

All ``b x b`` block operations are batched over any leading dimensions of the
inputs (``[..., K, b, b]``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._linalg import cholesky_spd, inv_via_chol
from .layout import BlockStar

__all__ = ["selected_inverse_star", "StarFactors"]


class StarFactors:
    """Factors from the leaf-elimination (collect) pass (returned when requested).

    Attributes
    ----------
    chol_leaf:
        Cholesky factors of the leaf pivots ``A_jj``: ``[..., K, b, b]``.
    chol_S:
        Cholesky factor of the center Schur complement ``S``: ``[..., b, b]``.
    B:
        Eliminated coupling ``B_j = A_jj^{-1} A_{j,0}``: ``[..., K, b, b]``.
    """

    __slots__ = ("chol_leaf", "chol_S", "B")

    def __init__(self, chol_leaf: Tensor, chol_S: Tensor, B: Tensor) -> None:
        self.chol_leaf = chol_leaf
        self.chol_S = chol_S
        self.B = B


def selected_inverse_star(
    center: Tensor,
    leaf_diag: Tensor,
    coupling: Tensor,
    *,
    check: bool = False,
    return_factors: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Selected inverse of a symmetric block-arrowhead (star) SPD matrix.

    Parameters
    ----------
    center:
        Center diagonal block ``A_00``: ``[..., b, b]`` (symmetric).
    leaf_diag:
        Leaf diagonal blocks ``A_jj``: ``[..., K, b, b]`` (each symmetric).
    coupling:
        Center->leaf coupling blocks ``A_0j``: ``[..., K, b, b]``
        (``A_{j,0} = coupling[..., j].mT``).
    check:
        If ``True``, validate inputs (shape / dtype / symmetry) before factorizing.
    return_factors:
        If ``True``, also return the :class:`StarFactors` from the collect pass.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs are upcast and the
        result cast back to the input dtype). Mirrors the chain kernel: models
        low-precision *storage* while computing in a dtype the LAPACK backend
        supports (``cholesky``/``cholesky_solve`` have no half kernel on CPU).

    Returns
    -------
    (G_center, G_leaf_diag, G_cross) or (..., factors):
        ``G_center`` ``[..., b, b]`` is the center diagonal block ``G_00``;
        ``G_leaf_diag`` ``[..., K, b, b]`` holds the leaf diagonal blocks ``G_jj``;
        ``G_cross`` ``[..., K, b, b]`` holds the cross blocks ``G_{j,0}`` (so
        ``G_{0,j} = G_cross[..., j].mT``).
    """
    if check:
        BlockStar(center=center, leaf_diag=leaf_diag, coupling=coupling).validate()

    in_dtype = center.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        center = center.to(compute_dtype)
        leaf_diag = leaf_diag.to(compute_dtype)
        coupling = coupling.to(compute_dtype)

    # ---- collect: eliminate every leaf (batched over the leaf axis K) -------
    chol_leaf = cholesky_spd(leaf_diag, name="A_jj (leaf)")
    # B_j = A_jj^{-1} A_{j,0}; A_{j,0} = coupling_j^T.  cholesky_solve(B, L) solves
    # A_jj X = B, so feeding coupling^T gives B_j directly.
    B = torch.cholesky_solve(coupling.mT, chol_leaf)          # [..., K, b, b]
    # Schur term per leaf: A_0j B_j = coupling_j @ B_j; reduce over the leaf axis.
    schur = (coupling @ B).sum(dim=-3)                        # [..., b, b]
    S = center - schur
    S = 0.5 * (S + S.mT)                                      # symmetrize the pivot
    chol_S = cholesky_spd(S, name="S (center Schur complement)")

    # ---- distribute: center marginal, then each leaf (batched over K) -------
    G_center = inv_via_chol(chol_S)                          # G_00 = S^{-1}
    G_center = 0.5 * (G_center + G_center.mT)

    G_center_K = G_center.unsqueeze(-3)                      # broadcast over leaves
    G_cross = -(B @ G_center_K)                              # G_{j,0} = -B_j G_00
    leaf_inv = inv_via_chol(chol_leaf)                       # A_jj^{-1}
    G_leaf = leaf_inv + B @ G_center_K @ B.mT                # G_jj = A_jj^{-1} + B G_00 B^T
    G_leaf = 0.5 * (G_leaf + G_leaf.mT)

    if compute_dtype is not None and compute_dtype != in_dtype:
        G_center = G_center.to(in_dtype)
        G_leaf = G_leaf.to(in_dtype)
        G_cross = G_cross.to(in_dtype)
        chol_leaf = chol_leaf.to(in_dtype)
        chol_S = chol_S.to(in_dtype)
        B = B.to(in_dtype)

    if return_factors:
        return G_center, G_leaf, G_cross, StarFactors(chol_leaf=chol_leaf, chol_S=chol_S, B=B)
    return G_center, G_leaf, G_cross
