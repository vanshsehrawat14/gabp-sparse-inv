"""General block-tree selected matrix inversion.

Given a symmetric positive-definite block matrix whose block graph is an **arbitrary
rooted tree** (given by a ``parent`` array), compute the blocks of ``A^{-1}`` on the
tree pattern -- the node diagonals ``G_vv`` and the parent-child cross blocks
``G_{p(v),v}`` -- without forming the dense inverse. This is the general kernel of
which :func:`gabp_sparse_inv.selected_inverse_chain` (path) and
:func:`gabp_sparse_inv.selected_inverse_star` (depth-1 tree) are special cases.

Algorithm (the two-pass collect/distribute of ``docs/derivations.md`` eqs. 3-6 =
Gaussian belief propagation on a tree). With ``edge[v] = A_{p(v),v} = U_v`` and pivot
``D_v``:

    collect  (children -> parents):  D_v = A_vv - sum_{c in ch(v)} U_c D_c^{-1} U_c^T
    distribute (parents -> children): ell_v = U_v D_v^{-1},
                                      G_{p(v),v} = - G_{p(v),p(v)} ell_v,
                                      G_vv       = D_v^{-1} + ell_v^T G_{p(v),p(v)} ell_v

The schedule follows the validated topological orders from
:func:`gabp_sparse_inv.layout.tree_orders`. The only Python-level loop is the
``O(n)`` sweep over nodes; every ``b x b`` block op is batched over leading dims.
``O(n b^3)`` time, ``O(n b^2)`` storage.

This is a **correctness reference** kernel: the sequential per-node loop is
launch-latency-bound on GPU. Sibling/level-set batching (and a differentiable
backward) belong to the differentiable-selinv work -- see ``docs/derivations.md`` §8.

Pivots are symmetrized before factorization (policy; see ``derivations.md`` §6).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._linalg import cholesky_spd, inv_via_chol
from .layout import BlockTree, _as_parent_tensor, tree_levels, tree_orders

__all__ = ["selected_inverse_tree", "TreeFactors"]


class TreeFactors:
    """Factors from the collect (leaves-to-root) pass (returned when requested).

    Attributes
    ----------
    chol_D:
        Cholesky factors of the node pivots ``D_v``: ``[..., n, b, b]`` lower-triangular.
    ell:
        Factor blocks ``ell_v = A_{p(v),v} D_v^{-1}``: ``[..., n, b, b]``. The slot for
        the root is zero (the root has no parent edge).
    parent:
        The (validated) 1-D ``long`` parent array, length ``n``.
    """

    __slots__ = ("chol_D", "ell", "parent")

    def __init__(self, chol_D: Tensor, ell: Tensor, parent: Tensor) -> None:
        self.chol_D = chol_D
        self.ell = ell
        self.parent = parent


def _level_index_tensors(levels, root, plist, device):
    """Per-level ``(node_idx, nonroot_idx, parent_idx)`` long tensors for batching."""
    out = []
    for level in levels:
        node_idx = torch.tensor(level, dtype=torch.long, device=device)
        nonroot = [v for v in level if v != root]
        nonroot_idx = torch.tensor(nonroot, dtype=torch.long, device=device)
        parent_idx = torch.tensor([plist[v] for v in nonroot], dtype=torch.long, device=device)
        out.append((node_idx, nonroot_idx, parent_idx))
    return out


def _selected_inverse_tree_batched(diag, edge, parent_t, root, plist, return_factors):
    """Level-set batched forward: one batched Cholesky + one ``index_add_`` per level.

    Mathematically identical to the per-node loop (validated block-for-block in
    ``tests/test_tree.py``); the antichain levels come from :func:`tree_levels`.
    """
    levels = tree_levels(parent_t)
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    dev = diag.device
    lvl_idx = _level_index_tensors(levels, root, plist, dev)

    chol_D = diag.new_empty((*batch, n, b, b))
    ell = diag.new_zeros((*batch, n, b, b))

    # ---- collect: levels in increasing height (children before parents) ----
    P = diag.clone()
    for h, (node_idx, nonroot_idx, parent_idx) in enumerate(lvl_idx):
        Pl = P.index_select(-3, node_idx)
        Pl = 0.5 * (Pl + Pl.mT)                       # symmetrize pivots (policy)
        chol_l = cholesky_spd(Pl, name=f"D(level h={h})")
        chol_D.index_copy_(-3, node_idx, chol_l)
        if nonroot_idx.numel() > 0:
            U = edge.index_select(-3, nonroot_idx)
            chol_nr = chol_D.index_select(-3, nonroot_idx)
            ell_nr = torch.cholesky_solve(U.mT, chol_nr).mT
            ell.index_copy_(-3, nonroot_idx, ell_nr)
            P = P.index_add(-3, parent_idx, -(ell_nr @ U.mT))

    # ---- distribute: root marginal, then levels in decreasing height ----
    G_diag = diag.new_empty((*batch, n, b, b))
    G_edge = diag.new_zeros((*batch, n, b, b))
    ridx = torch.tensor([root], dtype=torch.long, device=dev)
    G_root = inv_via_chol(chol_D.index_select(-3, ridx))
    G_diag.index_copy_(-3, ridx, 0.5 * (G_root + G_root.mT))
    for node_idx, nonroot_idx, parent_idx in reversed(lvl_idx):
        if nonroot_idx.numel() == 0:
            continue
        ell_nr = ell.index_select(-3, nonroot_idx)
        Gpp = G_diag.index_select(-3, parent_idx)     # parent already computed
        G_edge.index_copy_(-3, nonroot_idx, -(Gpp @ ell_nr))
        Dinv_nr = inv_via_chol(chol_D.index_select(-3, nonroot_idx))
        G_vv = Dinv_nr + ell_nr.mT @ Gpp @ ell_nr
        G_diag.index_copy_(-3, nonroot_idx, 0.5 * (G_vv + G_vv.mT))

    if return_factors:
        return G_diag, G_edge, TreeFactors(chol_D=chol_D, ell=ell, parent=parent_t)
    return G_diag, G_edge


def selected_inverse_tree(
    diag: Tensor,
    edge: Tensor,
    parent: Tensor | Sequence[int],
    *,
    check: bool = False,
    return_factors: bool = False,
    compute_dtype: torch.dtype | None = None,
    batched: bool = False,
):
    """Selected inverse of an SPD block matrix whose block graph is a tree.

    Parameters
    ----------
    diag:
        Node diagonal blocks ``A_vv``: ``[..., n, b, b]`` (each symmetric).
    edge:
        Edge blocks ``edge[v] = A_{p(v),v}`` (parent-row, child-col): ``[..., n, b, b]``.
        The slot for the root is unused (treated as zero); ``A_{v,p(v)} = edge[v].mT``.
    parent:
        1-D ``long`` parent array of length ``n`` (``parent[root] = -1``); the topology
        is shared across leading batch dims and is not batched.
    check:
        If ``True``, validate inputs (shape / dtype / symmetry / topology) first.
    return_factors:
        If ``True``, also return the :class:`TreeFactors` from the collect pass.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast
        back). Mirrors the chain/star kernels: models low-precision *storage* while
        computing in a dtype the LAPACK backend supports.
    batched:
        If ``True``, use the level-set batched path: one batched Cholesky and one
        ``index_add_`` scatter per height antichain (:func:`tree_levels`) instead of
        the per-node Python loop. Mathematically identical (validated block-for-block);
        amortizes kernel-launch latency on GPU. The per-node loop remains the default
        correctness reference.

    Returns
    -------
    (G_diag, G_edge) or (G_diag, G_edge, factors):
        ``G_diag`` ``[..., n, b, b]`` holds the node diagonal blocks ``G_vv``;
        ``G_edge`` ``[..., n, b, b]`` holds the cross blocks ``G_{p(v),v}`` (so
        ``G_{v,p(v)} = G_edge[v].mT``). The root slot of ``G_edge`` is zero.
    """
    if check:
        BlockTree(diag=diag, edge=edge, parent=parent).validate()

    parent = _as_parent_tensor(parent)
    plist = parent.tolist()
    root, _children, collect_order = tree_orders(parent)

    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge = edge.to(compute_dtype)

    if batched:
        result = _selected_inverse_tree_batched(
            diag, edge, parent, root, plist, return_factors
        )
        if compute_dtype is not None and compute_dtype != in_dtype:
            result = tuple(
                (x.to(in_dtype) if isinstance(x, Tensor) else x) for x in result
            )
            if not return_factors:
                return result[0], result[1]
            G_diag, G_edge, factors = result
            factors.chol_D = factors.chol_D.to(in_dtype)
            factors.ell = factors.ell.to(in_dtype)
            return G_diag, G_edge, factors
        return result

    n = diag.shape[-3]
    b = diag.shape[-1]
    batch = diag.shape[:-3]

    chol_D = diag.new_empty((*batch, n, b, b))
    ell = diag.new_zeros((*batch, n, b, b))         # root slot stays zero

    # ---- collect: eliminate children into parents (leaves -> root) ----------
    P = diag.clone()                                 # working pivots D_v
    for v in collect_order:
        Pv = 0.5 * (P[..., v, :, :] + P[..., v, :, :].mT)   # symmetrize pivot (policy)
        chol_v = cholesky_spd(Pv, name=f"D_{v}")
        chol_D[..., v, :, :] = chol_v
        if v != root:
            U_v = edge[..., v, :, :]                  # A_{p(v),v}
            # ell_v = U_v D_v^{-1} = (D_v^{-1} U_v^T)^T = cholesky_solve(U_v^T, chol_v)^T.
            ell_v = torch.cholesky_solve(U_v.mT, chol_v).mT
            ell[..., v, :, :] = ell_v
            # Subtract this child's Schur term U_v D_v^{-1} U_v^T from the parent pivot.
            pv = plist[v]
            P[..., pv, :, :] = P[..., pv, :, :] - ell_v @ U_v.mT

    # ---- distribute: root marginal, then each child (root -> leaves) --------
    G_diag = diag.new_empty((*batch, n, b, b))
    G_edge = diag.new_zeros((*batch, n, b, b))       # root slot stays zero

    G_root = inv_via_chol(chol_D[..., root, :, :])
    G_diag[..., root, :, :] = 0.5 * (G_root + G_root.mT)
    for v in reversed(collect_order):
        if v == root:
            continue
        pv = plist[v]
        ell_v = ell[..., v, :, :]
        Gpp = G_diag[..., pv, :, :]                   # parent already computed
        G_pv = -(Gpp @ ell_v)                         # G_{p(v),v}
        G_edge[..., v, :, :] = G_pv
        Dv_inv = inv_via_chol(chol_D[..., v, :, :])
        G_vv = Dv_inv + ell_v.mT @ Gpp @ ell_v        # = D_v^{-1} + ell^T G_pp ell
        G_diag[..., v, :, :] = 0.5 * (G_vv + G_vv.mT)

    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_edge = G_edge.to(in_dtype)
        chol_D = chol_D.to(in_dtype)
        ell = ell.to(in_dtype)

    if return_factors:
        return G_diag, G_edge, TreeFactors(chol_D=chol_D, ell=ell, parent=parent)
    return G_diag, G_edge
