"""Exact Gaussian sampling from a structured SPD precision (tree / junction).

Draw ``x ~ N(0, A^{-1})`` where ``A`` is an SPD **precision** given in a tree
(:class:`~gabp_sparse_inv.layout.BlockTree`) or general-sparse
(:class:`~gabp_sparse_inv.layout.BlockSparseSym`, the junction kernel) layout, reusing the
same ``LDL^T`` factorization as the selected inverse / solve. This is the other M-JOSS
statistical op alongside the log-determinant (``tree_logdet`` / ``junction_logdet``).

With ``A = L D L^T`` and each pivot factored ``D_v = C_v C_v^T`` (Cholesky), set ``M = L C``
(block-diagonal ``C``, so ``M`` is lower-triangular in elimination order). Then ``A = M M^T``
and, for ``z ~ N(0, I)``,

    x = M^{-T} z   =>   Cov(x) = M^{-T} I M^{-1} = (M M^T)^{-1} = A^{-1}.

The transform ``x = M^{-T} z`` is two substitutions: ``u_v = C_v^{-T} z_v`` (a per-node
upper-triangular solve with the pivot factor) followed by the back-solve ``L^T x = u`` on the
elimination structure (the distribute half of the linear solve). It is deterministic; the
public samplers just draw ``z`` and apply it. Drawing the ``num_samples`` realizations as the
trailing axis means a single vectorized pass produces the whole batch.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .gmrf import _tree_collect_batched
from .junction import _factor_junction, _normalize_edge_index
from .layout import _as_parent_tensor, tree_orders

__all__ = ["sample_gaussian_tree", "sample_gaussian_junction"]


def _sample_transform_tree(diag, edge, parent_t, z):
    """Deterministic ``x = M^{-T} z`` for the tree precision (``z``/``x``: ``[..., n, b, k]``)."""
    chol_D, ell, _logdet, root, _lvl = _tree_collect_batched(diag, edge, parent_t)
    _root, _children, collect = tree_orders(parent_t)
    plist = parent_t.tolist()
    n = diag.shape[-3]
    # u_v = C_v^{-T} z_v : an upper-triangular solve with the pivot factor, batched over v.
    u = torch.linalg.solve_triangular(chol_D.mT, z, upper=True)        # [..., n, b, k]
    # back-solve L^T x = u : parents before children, x_v = u_v - ell[v]^T x_{p(v)}.
    x = [None] * n
    x[root] = u[..., root, :, :]
    for v in reversed(collect):
        if v == root:
            continue
        x[v] = u[..., v, :, :] - ell[..., v, :, :].mT @ x[plist[v]]
    return torch.stack(x, dim=-3)                                     # [..., n, b, k]


def _sample_transform_junction(diag, edge_index, edge_val, z, order):
    """Deterministic ``x = M^{-T} z`` for the junction precision (``z``/``x``: ``[..., n, b, k]``)."""
    order, _pos, U, _S_pairs, chol, ell = _factor_junction(diag, edge_index, edge_val, order)
    n = diag.shape[-3]
    u = {v: torch.linalg.solve_triangular(chol[v].mT, z[..., v, :, :], upper=True) for v in range(n)}
    # back-solve L^T x = u : decreasing elimination order, x_v = u_v - sum_{w in U_v} ell_{wv}^T x_w.
    x = {}
    for v in reversed(order):
        acc = u[v]
        for w in U[v]:
            acc = acc - ell[(w, v)].mT @ x[w]
        x[v] = acc
    return torch.stack([x[v] for v in range(n)], dim=-3)              # [..., n, b, k]


def sample_gaussian_tree(
    diag: Tensor,
    edge: Tensor,
    parent: Tensor | Sequence[int],
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
):
    """Exact samples ``x ~ N(0, A^{-1})`` for a tree-structured SPD precision ``A``.

    Parameters
    ----------
    diag, edge, parent:
        The tree precision in :class:`~gabp_sparse_inv.layout.BlockTree` storage
        (``diag``/``edge`` ``[..., n, b, b]``, ``edge[v] = A_{p(v), v}``).
    num_samples:
        Number of independent samples to draw.
    generator:
        Optional ``torch.Generator`` for reproducibility (must match ``diag``'s device).

    Returns
    -------
    x:
        Samples shaped ``[num_samples, ..., n, b]`` with exact covariance ``A^{-1}``.
    """
    p = _as_parent_tensor(parent)
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    z = torch.randn((*batch, n, b, num_samples), generator=generator, dtype=diag.dtype, device=diag.device)
    x = _sample_transform_tree(diag, edge, p, z)                      # [..., n, b, num_samples]
    return x.movedim(-1, 0)                                           # [num_samples, ..., n, b]


def sample_gaussian_junction(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    num_samples: int,
    *,
    order: Tensor | Sequence[int] | None = None,
    generator: torch.Generator | None = None,
):
    """Exact samples ``x ~ N(0, A^{-1})`` for a general sparse SPD precision ``A``.

    Parameters
    ----------
    diag, edge_index, edge_val, order:
        The precision in :class:`~gabp_sparse_inv.layout.BlockSparseSym` storage, as in
        :func:`~gabp_sparse_inv.selected_inverse_junction` (``order`` defaults to greedy
        min-degree).
    num_samples:
        Number of independent samples to draw.
    generator:
        Optional ``torch.Generator`` for reproducibility (must match ``diag``'s device).

    Returns
    -------
    x:
        Samples shaped ``[num_samples, ..., n, b]`` with exact covariance ``A^{-1}``.
    """
    edge_index = _normalize_edge_index(edge_index)
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    z = torch.randn((*batch, n, b, num_samples), generator=generator, dtype=diag.dtype, device=diag.device)
    x = _sample_transform_junction(diag, edge_index, edge_val, z, order)
    return x.movedim(-1, 0)                                           # [num_samples, ..., n, b]
