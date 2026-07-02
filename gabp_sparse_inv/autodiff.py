"""Differentiable selected inverse on SPD block trees.

:class:`SelInvTree` wraps the forward kernel
(:func:`gabp_sparse_inv.tree.selected_inverse_tree`) with a **hand-written analytic
backward**: the reverse two-pass collect/distribute of ``docs/derivations.md`` §8.3,
reusing the forward factors ``(chol_D, ell)`` and the selected blocks ``G_diag``.

The backward is the *self-adjoint* schedule proved in §8.2 -- a reverse-distribute
sweep (leaves->root) followed by a reverse-collect sweep (root->leaves) on the same
elimination tree. It costs ``O((|V|+|E|) b^3)`` time and ``O((|V|+|E|) b^2)`` memory,
identical to the forward pass, and never tapes the per-node loop through autograd
(the shipped forward loop is in fact *not* autograd-traceable -- it writes into
preallocated buffers -- which is part of why this analytic backward exists).

Both passes have a per-node ``loop`` form (the correctness reference) and a level-set
``batched`` form (one batched solve + one ``index_add_`` per height antichain, the
same batching the forward uses); the two agree block-for-block.

Convention (``docs/derivations.md`` §8.1): the differentiable inputs are the stored
blocks ``diag`` (node diagonals ``A_vv``, symmetric) and ``edge`` (``edge[v] =
A_{p(v),v}``). The returned cotangents ``barAd[v] = d f / d diag[v]`` and
``barAe[v] = d f / d edge[v]`` are gradients in the autograd sense w.r.t. those input
tensors -- exactly what :func:`torch.autograd.gradcheck` verifies.

**First-order only for this module.** ``backward`` runs under no-grad (the standard
custom-Function contract). For Hessian-vector products use the functional junction kernels
or ``selected_inverse_tree(batched=True)``; both pass ``gradgradcheck``
(``tests/test_double_backward.py``).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._linalg import inv_via_chol
from .layout import _as_parent_tensor, tree_levels, tree_orders
from .tree import _level_index_tensors, selected_inverse_tree

__all__ = ["selinv_tree", "SelInvTree"]


def _sym(x: Tensor) -> Tensor:
    """Symmetrize the trailing ``b x b`` blocks: ``(X + X^T)/2``."""
    return 0.5 * (x + x.mT)


def _backward_loop(ctx, barGd, barGe):
    """Per-node reverse two-pass (eqs. 13-14); the correctness reference."""
    edge, G_diag, ell, chol_D = ctx.saved_tensors
    root, children, collect_order, plist = ctx.root, ctx.children, ctx.collect_order, ctx.plist
    Dinv = inv_via_chol(chol_D)

    zeros = torch.zeros_like(G_diag)
    bG = barGd.clone() if barGd is not None else zeros.clone()
    if barGe is None:
        barGe = zeros
    bDinv = zeros.clone()
    bell = zeros.clone()
    bU = zeros.clone()

    # Pass 1: reverse-distribute (leaves -> root), eq. (13).
    for v in collect_order:
        M = _sym(bG[..., v, :, :])
        bDinv[..., v, :, :] = bDinv[..., v, :, :] + M
        if v != root:
            p = plist[v]
            Gpp = G_diag[..., p, :, :]
            ell_v = ell[..., v, :, :]
            bge = barGe[..., v, :, :]
            bell[..., v, :, :] = bell[..., v, :, :] + Gpp @ ell_v @ (M + M.mT) - Gpp @ bge
            bG[..., p, :, :] = bG[..., p, :, :] + ell_v @ M @ ell_v.mT - bge @ ell_v.mT

    # Pass 2: reverse-collect (root -> leaves), eq. (14).
    barAd = zeros.clone()
    for v in reversed(collect_order):
        Dinv_v = Dinv[..., v, :, :]
        if v != root:
            ell_v = ell[..., v, :, :]
            U_v = edge[..., v, :, :]
            bell_v = bell[..., v, :, :]
            bU[..., v, :, :] = bU[..., v, :, :] + bell_v @ Dinv_v
            bDinv[..., v, :, :] = bDinv[..., v, :, :] + U_v.mT @ bell_v
        bP = _sym(-(Dinv_v @ bDinv[..., v, :, :] @ Dinv_v))
        barAd[..., v, :, :] = bP
        for c in children[v]:
            bP_v = bP
            bell[..., c, :, :] = bell[..., c, :, :] - bP_v @ edge[..., c, :, :]
            bU[..., c, :, :] = bU[..., c, :, :] - bP_v @ ell[..., c, :, :]

    return barAd, bU


def _backward_batched(ctx, barGd, barGe):
    """Level-set reverse two-pass: vectorizes (13)-(14) over height antichains.

    Same arithmetic as :func:`_backward_loop`, scheduled by :func:`tree_levels` so each
    height level is one batched block op + one ``index_add_`` -- the backward mirror of
    the batched forward (the schedule is self-adjoint).
    """
    edge, G_diag, ell, chol_D = ctx.saved_tensors
    root, plist, levels = ctx.root, ctx.plist, ctx.levels
    Dinv = inv_via_chol(chol_D)
    dev = G_diag.device
    lvl = _level_index_tensors(levels, root, plist, dev)

    zeros = torch.zeros_like(G_diag)
    bG = barGd.clone() if barGd is not None else zeros.clone()
    barGe = barGe if barGe is not None else zeros
    bDinv = zeros.clone()
    bell = zeros.clone()
    bU = zeros.clone()
    bP = zeros.clone()
    barAd = zeros.clone()

    # Pass 1: reverse-distribute, increasing height.
    for node_idx, nonroot_idx, parent_idx in lvl:
        bGl = bG.index_select(-3, node_idx)
        Ml = _sym(bGl)
        bDinv.index_copy_(-3, node_idx, bDinv.index_select(-3, node_idx) + Ml)
        if nonroot_idx.numel() > 0:
            M = _sym(bG.index_select(-3, nonroot_idx))
            ell_v = ell.index_select(-3, nonroot_idx)
            Gpp = G_diag.index_select(-3, parent_idx)
            bge = barGe.index_select(-3, nonroot_idx)
            add_bell = Gpp @ ell_v @ (M + M.mT) - Gpp @ bge
            bell.index_copy_(-3, nonroot_idx, bell.index_select(-3, nonroot_idx) + add_bell)
            bG = bG.index_add(-3, parent_idx, ell_v @ M @ ell_v.mT - bge @ ell_v.mT)

    # Pass 2: reverse-collect, decreasing height. Root (top level, alone) first.
    ridx = torch.tensor([root], dtype=torch.long, device=dev)
    Dinv_r = Dinv.index_select(-3, ridx)
    bP_r = _sym(-(Dinv_r @ bDinv.index_select(-3, ridx) @ Dinv_r))
    bP.index_copy_(-3, ridx, bP_r)
    barAd.index_copy_(-3, ridx, bP_r)
    for node_idx, nonroot_idx, parent_idx in reversed(lvl):
        if nonroot_idx.numel() == 0:
            continue
        ell_v = ell.index_select(-3, nonroot_idx)
        U_v = edge.index_select(-3, nonroot_idx)
        Dinv_v = Dinv.index_select(-3, nonroot_idx)
        bP_par = bP.index_select(-3, parent_idx)
        bell_v = bell.index_select(-3, nonroot_idx) - bP_par @ U_v        # (14d)
        bU_v = bU.index_select(-3, nonroot_idx) - bP_par @ ell_v          # (14d)
        bU_v = bU_v + bell_v @ Dinv_v                                     # (14a)
        bDinv_v = bDinv.index_select(-3, nonroot_idx) + U_v.mT @ bell_v   # (14a)
        bP_v = _sym(-(Dinv_v @ bDinv_v @ Dinv_v))                         # (14b)
        bell.index_copy_(-3, nonroot_idx, bell_v)
        bU.index_copy_(-3, nonroot_idx, bU_v)
        bP.index_copy_(-3, nonroot_idx, bP_v)
        barAd.index_copy_(-3, nonroot_idx, bP_v)

    return barAd, bU


class SelInvTree(torch.autograd.Function):
    """Autograd Function: selected inverse of an SPD block tree with analytic backward.

    Use :func:`selinv_tree` rather than calling ``apply`` directly; it resolves and
    validates the tree topology once and routes through here.
    """

    @staticmethod
    def forward(ctx, diag, edge, parent_t, root, children, collect_order, batched):
        levels = tree_levels(parent_t) if batched else None
        G_diag, G_edge, factors = selected_inverse_tree(
            diag, edge, parent_t, return_factors=True, batched=batched
        )
        ctx.save_for_backward(edge, G_diag, factors.ell, factors.chol_D)
        ctx.root = root
        ctx.children = children
        ctx.collect_order = collect_order
        ctx.plist = parent_t.tolist()
        ctx.levels = levels
        ctx.batched = batched
        return G_diag, G_edge

    @staticmethod
    def backward(ctx, barGd, barGe):
        if ctx.batched:
            barAd, barAe = _backward_batched(ctx, barGd, barGe)
        else:
            barAd, barAe = _backward_loop(ctx, barGd, barGe)
        return barAd, barAe, None, None, None, None, None


def selinv_tree(
    diag: Tensor,
    edge: Tensor,
    parent: Tensor | Sequence[int],
    *,
    check: bool = False,
    batched: bool = False,
):
    """Autograd-connected selected inverse of an SPD block tree.

    Same forward result as :func:`gabp_sparse_inv.selected_inverse_tree`, but the
    returned ``(G_diag, G_edge)`` are connected to ``diag`` and ``edge`` through the
    analytic backward of :class:`SelInvTree`. Gradients flow to ``diag`` and ``edge``;
    ``parent`` is a static (non-differentiable) topology argument.

    Parameters
    ----------
    diag, edge, parent:
        As in :func:`gabp_sparse_inv.selected_inverse_tree`.
    check:
        If ``True``, validate the block-tree inputs before computing.
    batched:
        If ``True``, use the level-set batched forward and backward (same result;
        amortizes launch latency on GPU). Default is the per-node reference loop.

    Returns
    -------
    (G_diag, G_edge):
        ``G_diag[v] = G_vv`` and ``G_edge[v] = G_{p(v),v}`` (root slot zero), with
        autograd support w.r.t. ``diag`` and ``edge``.
    """
    if check:
        from .layout import BlockTree

        BlockTree(diag=diag, edge=edge, parent=parent).validate()
    parent_t = _as_parent_tensor(parent)
    root, children, collect_order = tree_orders(parent_t)
    return SelInvTree.apply(diag, edge, parent_t, root, children, collect_order, batched)
