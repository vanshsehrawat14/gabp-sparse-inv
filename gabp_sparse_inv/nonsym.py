"""Selected inverse of non-symmetric block lower-bidiagonal matrices.

The non-symmetric analogue of the chain (Phase 1). Given a general (non-symmetric)
block lower-bidiagonal matrix ``M`` -- diagonal blocks ``M_ii`` and one sub-diagonal
``M_{i+1,i}``, zero super-diagonal -- compute the blocks of ``G = M^{-1}`` on ``M``'s
own pattern (diagonal + first sub-diagonal) without forming the dense inverse:

    G_ii     = M_ii^{-1}                                   (diagonal)
    G_{i+1,i} = -M_{i+1,i+1}^{-1} M_{i+1,i} M_ii^{-1}      (first sub-diagonal)

Both blocks are non-trivial (contrast the *unit*-diagonal case ``(I - A)^{-1}``, whose
diagonal selected blocks are trivially ``I``). Off-pattern blocks of ``G`` are nonzero
but excluded. See ``docs/derivations.md`` §9 for the derivation.

Unlike the SPD chain, the selected blocks here are **fully local** -- ``G_ii`` depends
only on ``M_ii`` and ``G_{i+1,i}`` only on its two endpoint diagonals plus the coupling
-- so there is no sequential collect/distribute sweep: the forward is one batched block
inverse and one batched product, the "easy case" the roadmap targets for validating the
non-symmetric machinery.

**Differentiable backward (``selinv_bidiag``).** The matrix-level adjoint of the inverse
is ``dG = -G (dM) G`` (folklore); restricted to the bidiagonal pattern it is a local,
``O(n)`` rule (``docs/derivations.md`` §9.2), implemented here as a hand-written analytic
backward (the diagonal inverse is not re-differentiated by autograd -- the saved
``G_diag = M_ii^{-1}`` carries the inverse VJP). First-order gradients only.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .layout import BlockBidiag, _as_parent_tensor, tree_orders

__all__ = [
    "selected_inverse_bidiag",
    "selinv_bidiag",
    "SelInvBidiag",
    "selected_inverse_tril",
    "selinv_tril",
    "SelInvTril",
    "selected_inverse_nonsym_tree",
]


def selected_inverse_bidiag(
    diag: Tensor,
    lower: Tensor,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Selected inverse of a general block lower-bidiagonal matrix ``M``.

    Parameters
    ----------
    diag:
        Diagonal blocks ``M_ii``: ``[..., n, b, b]`` (general square blocks).
    lower:
        Sub-diagonal blocks ``lower[i] = M_{i+1,i}``: ``[..., n-1, b, b]``.
    check:
        If ``True``, validate the block-bidiagonal inputs before computing.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast
        back); mirrors the SPD kernels' low-precision-storage path.

    Returns
    -------
    (G_diag, G_lower):
        ``G_diag[i] = (M^{-1})_{ii}`` ``[..., n, b, b]`` and
        ``G_lower[i] = (M^{-1})_{i+1,i}`` ``[..., n-1, b, b]``.
    """
    if check:
        BlockBidiag(diag=diag, lower=lower).validate()

    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        lower = lower.to(compute_dtype)

    Dinv = torch.linalg.inv(diag)                       # G_ii = M_ii^{-1}
    G_diag = Dinv
    # G_{i+1,i} = -D_{i+1}^{-1} M_{i+1,i} D_i^{-1}; empty (no-op) when n == 1.
    G_lower = -(Dinv[..., 1:, :, :] @ lower @ Dinv[..., :-1, :, :])

    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_lower = G_lower.to(in_dtype)
    return G_diag, G_lower


class SelInvBidiag(torch.autograd.Function):
    """Autograd Function: selected inverse of a block lower-bidiagonal ``M``.

    Forward returns ``(G_diag, G_lower)``; backward is the analytic local adjoint of
    ``docs/derivations.md`` §9.2. Use :func:`selinv_bidiag` rather than ``apply``.
    """

    @staticmethod
    def forward(ctx, diag, lower):
        Dinv = torch.linalg.inv(diag)
        G_diag = Dinv
        G_lower = -(Dinv[..., 1:, :, :] @ lower @ Dinv[..., :-1, :, :])
        ctx.save_for_backward(Dinv, lower)
        return G_diag, G_lower

    @staticmethod
    def backward(ctx, bGd, bGl):
        # Cotangents: bGd = df/dG_diag [...,n,b,b]; bGl = df/dG_lower [...,n-1,b,b].
        # Returns (bd, bc): df/d diag and df/d lower (the autograd-sense gradients).
        Dinv, lower = ctx.saved_tensors
        Dl = Dinv[..., :-1, :, :]                       # D_i^{-1},   i = 0..n-2
        Dr = Dinv[..., 1:, :, :]                        # D_{i+1}^{-1}, i = 0..n-2

        # Accumulate the cotangent on E := Dinv (= G_diag), then map through the
        # inverse VJP  bar_D = -E^T bar_E E^T.
        bE = torch.zeros_like(Dinv) if bGd is None else bGd.clone()
        if bGl is not None:
            # G_lower[i] = -Dr[i] @ lower[i] @ Dl[i].  VJP onto the two diagonal factors:
            #   right factor D_i^{-1}   -> slots 0..n-2
            bE[..., :-1, :, :] = bE[..., :-1, :, :] - lower.mT @ Dr.mT @ bGl
            #   left factor  D_{i+1}^{-1} -> slots 1..n-1
            bE[..., 1:, :, :] = bE[..., 1:, :, :] - bGl @ Dl.mT @ lower.mT
            # ... and onto the coupling (middle factor):
            bc = -(Dr.mT @ bGl @ Dl.mT)
        else:
            bc = torch.zeros_like(lower)

        bd = -(Dinv.mT @ bE @ Dinv.mT)
        return bd, bc


def selinv_bidiag(diag: Tensor, lower: Tensor, *, check: bool = False):
    """Autograd-connected selected inverse of a block lower-bidiagonal ``M``.

    Same forward result as :func:`selected_inverse_bidiag`, but ``(G_diag, G_lower)``
    are connected to ``diag`` and ``lower`` through the analytic backward of
    :class:`SelInvBidiag`. First-order gradients only.
    """
    if check:
        BlockBidiag(diag=diag, lower=lower).validate()
    return SelInvBidiag.apply(diag, lower)


# --------------------------------------------------------------------------- #
# Dense triangular instance: T = (I - A)^{-1}, the DeltaNet chunk inverse.
# --------------------------------------------------------------------------- #
def _eye_like(A: Tensor) -> Tensor:
    return torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)


def selected_inverse_tril(
    A: Tensor,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Dense lower-triangular inverse ``T = (I - A)^{-1}`` -- the DeltaNet chunk inverse.

    The dense instance of §9 (``docs/derivations.md`` §9.4): ``A`` is strictly
    lower-triangular (only ``A.tril(-1)`` is used), so ``M = I - A`` is *unit* lower
    triangular and ``T`` is unit lower triangular and dense. This is the chunk inverse
    of DeltaNet / gated linear attention. The "selected" set is the full lower triangle
    (no sparsity win); a plain blocked triangular solve is the baseline, so the
    contribution is the analytic self-adjoint backward of :func:`selinv_tril`, not the
    forward inverse.

    Parameters
    ----------
    A:
        Strictly-lower-triangular matrix ``[..., N, N]`` (the part above and on the
        diagonal is ignored).
    check:
        If ``True``, require a square trailing shape.
    compute_dtype:
        If given, solve in this dtype and cast the result back (low-precision storage).

    Returns
    -------
    T:
        ``(I - A)^{-1}`` ``[..., N, N]`` (unit lower triangular).
    """
    if check and (A.ndim < 2 or A.shape[-1] != A.shape[-2]):
        raise ValueError(f"A must be square [..., N, N]; got {tuple(A.shape)}")

    in_dtype = A.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        A = A.to(compute_dtype)

    eye = _eye_like(A)
    M = eye - A.tril(-1)
    T = torch.linalg.solve_triangular(M, eye, upper=False, unitriangular=True)

    if compute_dtype is not None and compute_dtype != in_dtype:
        T = T.to(in_dtype)
    return T


class SelInvTril(torch.autograd.Function):
    """Autograd Function for ``T = (I - A)^{-1}`` with the analytic self-adjoint backward.

    Backward (``docs/derivations.md`` §9.4): ``dT = T (dA) T``, so for a loss ``f(T)``
    with cotangent ``bar_T`` the gradient restricted to the strictly-lower pattern is
    ``bar_A = tril(T^T bar_T T^T, -1)``. Use :func:`selinv_tril`.
    """

    @staticmethod
    def forward(ctx, A):
        eye = _eye_like(A)
        T = torch.linalg.solve_triangular(eye - A.tril(-1), eye, upper=False, unitriangular=True)
        ctx.save_for_backward(T)
        return T

    @staticmethod
    def backward(ctx, bT):
        (T,) = ctx.saved_tensors
        return (T.mT @ bT @ T.mT).tril(-1)


def selinv_tril(A: Tensor, *, check: bool = False) -> Tensor:
    """Autograd-connected ``T = (I - A)^{-1}`` (DeltaNet chunk inverse).

    Same forward as :func:`selected_inverse_tril`, with the analytic backward of
    :class:`SelInvTril` (``bar_A = tril(T^T bar_T T^T, -1)``). First-order gradients only.
    """
    if check and (A.ndim < 2 or A.shape[-1] != A.shape[-2]):
        raise ValueError(f"A must be square [..., N, N]; got {tuple(A.shape)}")
    return SelInvTril.apply(A)


# --------------------------------------------------------------------------- #
# Non-symmetric *tree*-structured selected inverse (zero fill).
# --------------------------------------------------------------------------- #
def selected_inverse_nonsym_tree(
    diag: Tensor,
    edge_pc: Tensor,
    edge_cp: Tensor,
    parent: Tensor | Sequence[int],
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Selected inverse of a general (non-symmetric) block matrix whose graph is a tree.

    The zero-fill non-symmetric rung between the block lower-bidiagonal case
    (:func:`selected_inverse_bidiag`) and the general LU selected inverse: ``M`` is a
    block matrix whose off-diagonal pattern is a tree, but ``M_{uv}`` and ``M_{vu}`` are
    **independent** (``M_{uv} != M_{vu}^T``). Both directed blocks per tree edge are kept.

    Because a tree is a perfect elimination order with no fill, a block ``LU`` (here ``LDU``)
    factorization in collect order has the tree pattern, and the two-sided Takahashi
    recurrence gives every selected block exactly (``docs/derivations.md`` §2 generalized
    to the non-symmetric case; verified against the dense inverse and by ``gradcheck``):

        collect (leaves -> root):  D_v = M_vv - sum_{c in children(v)} M_{v,c} D_c^{-1} M_{c,v}
                                   ellL_v = M_{p,v} D_v^{-1},  ellU_v = D_v^{-1} M_{v,p}
        distribute (root -> leaves):  G_{p,v} = -G_pp ellL_v,   G_{v,p} = -ellU_v G_pp,
                                      G_vv = D_v^{-1} + ellU_v G_pp ellL_v

    When ``M`` is symmetric (``edge_cp[v] = edge_pc[v].mT``) these reduce block-for-block to
    the SPD tree kernel. Each ``D_v`` must be invertible (general blocks; no SPD assumption).

    Parameters
    ----------
    diag:
        Node diagonal blocks ``M_vv``: ``[..., n, b, b]`` (general square blocks).
    edge_pc:
        Parent-row, child-col blocks ``edge_pc[v] = M_{p(v), v}``: ``[..., n, b, b]``
        (the root slot is unused / treated as zero).
    edge_cp:
        Child-row, parent-col blocks ``edge_cp[v] = M_{v, p(v)}``: ``[..., n, b, b]``
        (root slot unused). For symmetric ``M`` this equals ``edge_pc[v].mT``.
    parent:
        1-D ``long`` parent array of length ``n`` (``parent[root] = -1``); shared across
        leading batch dims (not batched).
    check:
        If ``True``, validate shapes / parent topology first.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast back).

    Returns
    -------
    (G_diag, G_pc, G_cp):
        ``G_diag[v] = (M^{-1})_{vv}``; ``G_pc[v] = (M^{-1})_{p(v), v}`` and
        ``G_cp[v] = (M^{-1})_{v, p(v)}`` (both ``[..., n, b, b]``, root slot zero).

    Notes
    -----
    Functional / autograd-traceable (the inverses are ``torch.linalg.inv``): gradients of
    any loss over the selected blocks flow to ``diag``, ``edge_pc`` and ``edge_cp`` by
    reverse-mode, with no custom backward. First- and higher-order both available via autograd.
    """
    parent_t = _as_parent_tensor(parent)
    if check:
        n_chk = diag.shape[-3]
        if parent_t.numel() != n_chk:
            raise ValueError(f"parent length {parent_t.numel()} != n={n_chk}")
        for t, nm in ((edge_pc, "edge_pc"), (edge_cp, "edge_cp")):
            if t.shape != diag.shape:
                raise ValueError(f"{nm} must match diag shape {tuple(diag.shape)}; got {tuple(t.shape)}")

    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_pc = edge_pc.to(compute_dtype)
        edge_cp = edge_cp.to(compute_dtype)

    plist = parent_t.tolist()
    root, children, collect_order = tree_orders(parent_t)
    n = diag.shape[-3]

    # ---- collect: eliminate children into parents (leaves -> root) -----------
    Dinv: list[Tensor | None] = [None] * n          # D_v^{-1} (Schur pivot inverse)
    ellL: list[Tensor | None] = [None] * n          # M_{p,v} D_v^{-1}
    ellU: list[Tensor | None] = [None] * n          # D_v^{-1} M_{v,p}
    for v in collect_order:
        Dv = diag[..., v, :, :]
        for c in children[v]:
            # subtract child Schur term  M_{v,c} D_c^{-1} M_{c,v}
            Dv = Dv - edge_pc[..., c, :, :] @ Dinv[c] @ edge_cp[..., c, :, :]
        Dv_inv = torch.linalg.inv(Dv)
        Dinv[v] = Dv_inv
        if v != root:
            ellL[v] = edge_pc[..., v, :, :] @ Dv_inv
            ellU[v] = Dv_inv @ edge_cp[..., v, :, :]

    # ---- distribute: root marginal, then each child (root -> leaves) ---------
    G_d: list[Tensor | None] = [None] * n
    G_pc_blocks: list[Tensor] = [None] * n          # type: ignore[list-item]
    G_cp_blocks: list[Tensor] = [None] * n          # type: ignore[list-item]
    zero = diag.new_zeros(diag.shape[:-3] + (diag.shape[-1], diag.shape[-1]))
    G_d[root] = Dinv[root]
    for v in reversed(collect_order):
        if v == root:
            continue
        p = plist[v]
        Gpp = G_d[p]
        G_pc_blocks[v] = -(Gpp @ ellL[v])           # G_{p(v), v}
        G_cp_blocks[v] = -(ellU[v] @ Gpp)           # G_{v, p(v)}
        G_d[v] = Dinv[v] + ellU[v] @ Gpp @ ellL[v]
    G_pc_blocks[root] = zero
    G_cp_blocks[root] = zero

    G_diag = torch.stack(G_d, dim=-3)
    G_pc = torch.stack(G_pc_blocks, dim=-3)
    G_cp = torch.stack(G_cp_blocks, dim=-3)

    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_pc = G_pc.to(in_dtype)
        G_cp = G_cp.to(in_dtype)
    return G_diag, G_pc, G_cp
