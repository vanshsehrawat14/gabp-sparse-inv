"""Differentiable hierarchical (tree-structured) Gaussian Markov random fields.

A latent multiresolution Gaussian field on a rooted tree with Gaussian observations:

    x_root ~ N(0, root_prec^{-1} I),
    x_v | x_{p(v)} ~ N(a x_{p(v)}, kappa_v^{-1} I)   (branching AR / innovation process),
    y_v = x_v + eps_v,   eps_v ~ N(0, sigma2 I).

The latent precision ``Q(theta)`` is **tree-structured** (sparse on the tree pattern),
so every quantity needed for learning is computed in ``O(n)`` by the tree kernels of
this package rather than the ``O(N^3)`` of a dense Gaussian:

* the **log marginal likelihood** ``log p(y | theta)`` -- via tree log-determinants
  (:func:`tree_logdet`) and a tree linear solve (:func:`tree_solve`);
* the **posterior marginal variances** ``diag(Q_post^{-1})`` -- via the differentiable
  selected inverse :func:`gabp_sparse_inv.selinv_tree`. This is the objective that
  *needs* the general (beyond-log-det) selinv adjoint of ``docs/derivations.md`` §8 --
  the log-det gradient alone does not provide ``d/dtheta diag(Q_post^{-1})``.

Gradients w.r.t. the hyperparameters ``theta = (a, kappa, root_prec, sigma2)`` flow
through autograd. Hyperparameter learning therefore scales linearly in ``n`` where a
dense ``torch.linalg.inv`` + autograd baseline is ``O(N^3)`` time / ``O(N^2)`` memory
and becomes infeasible.

All nodes here are scalar (block size ``b = 1``); the kernels themselves support
general ``b``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ._linalg import cholesky_spd
from .autodiff import selinv_tree
from .layout import _as_parent_tensor, tree_levels, tree_orders
from .tree import _level_index_tensors

__all__ = [
    "node_depths",
    "tree_gmrf_precision",
    "tree_logdet",
    "tree_solve",
    "marginal_log_likelihood",
    "posterior_marginal_variances",
    "sample_tree_gmrf",
    "fit_marginal_likelihood",
]


def _resolve_dtype(*vals) -> torch.dtype:
    """Floating dtype to build the model in: the first float tensor's, else float64."""
    for v in vals:
        if isinstance(v, Tensor) and v.is_floating_point():
            return v.dtype
    return torch.float64


def node_depths(parent: Tensor | Sequence[int]) -> Tensor:
    """Depth of each node (root = 0), as a 1-D long tensor."""
    p = _as_parent_tensor(parent)
    root, _children, collect = tree_orders(p)
    plist = p.tolist()
    depth = [0] * int(p.numel())
    for v in reversed(collect):          # parents before children
        if v != root:
            depth[v] = depth[plist[v]] + 1
    return torch.tensor(depth, dtype=torch.long)


def _kappa_vector(parent_t: Tensor, kappa, n: int, dtype, device) -> Tensor:
    """Broadcast ``kappa`` (scalar or per-node) to a ``[n]`` tensor."""
    if isinstance(kappa, Tensor) and kappa.ndim >= 1 and kappa.shape[-1] == n:
        return kappa.to(dtype=dtype, device=device)
    k = torch.as_tensor(kappa, dtype=dtype, device=device)
    return k * torch.ones(n, dtype=dtype, device=device)


def tree_gmrf_precision(parent, a, kappa, root_prec):
    """Build the latent precision ``Q(theta)`` blocks for the branching AR model.

    Returns ``(diag, edge)`` with shapes ``[n, 1, 1]``: ``Q_vv`` on the diagonal and
    ``edge[v] = Q_{p(v), v}`` (root slot zero), differentiable in ``a``, ``kappa``,
    ``root_prec``. With innovation precisions ``kappa_v`` and coupling ``a``,

        Q_vv = (kappa_v if v != root else root_prec) + sum_{c in ch(v)} a^2 kappa_c,
        Q_{p(v), v} = -a kappa_v.
    """
    p = _as_parent_tensor(parent)
    n = int(p.numel())
    root, _children, _collect = tree_orders(p)
    dtype = _resolve_dtype(
        a if isinstance(a, Tensor) else None,
        kappa if isinstance(kappa, Tensor) else None,
        root_prec if isinstance(root_prec, Tensor) else None,
    )
    device = a.device if isinstance(a, Tensor) else torch.device("cpu")
    a = torch.as_tensor(a, dtype=dtype, device=device)
    root_prec = torch.as_tensor(root_prec, dtype=dtype, device=device)
    kappa_v = _kappa_vector(p, kappa, n, dtype, device)

    is_root = torch.zeros(n, dtype=torch.bool, device=device)
    is_root[root] = True
    base = torch.where(is_root, root_prec.expand(n), kappa_v)

    nonroot_mask = ~is_root
    parent_idx = p.to(device).clamp_min(0)[nonroot_mask]          # parents of non-root nodes
    child_kappa = (a * a) * kappa_v[nonroot_mask]
    child_contrib = torch.zeros(n, dtype=dtype, device=device).index_add(
        0, parent_idx, child_kappa
    )
    diag_scalar = base + child_contrib

    edge_scalar = -a * kappa_v
    edge_scalar = torch.where(is_root, torch.zeros_like(edge_scalar), edge_scalar)

    diag = diag_scalar.reshape(n, 1, 1)
    edge = edge_scalar.reshape(n, 1, 1)
    return diag, edge


def _tree_collect(diag, edge, parent_t):
    """Out-of-place block ``LDL^T`` collect: returns ``Dinv, ell, logdet`` + topology.

    Autograd-traceable (no in-place buffer writes). ``logdet = log det Q`` summed over
    the SPD pivots; ``Dinv[v] = D_v^{-1}``; ``ell[v] = Q_{p(v),v} D_v^{-1}``.
    """
    root, children, collect = tree_orders(parent_t)
    plist = parent_t.tolist()
    n, b = diag.shape[-3], diag.shape[-1]
    eye = torch.eye(b, dtype=diag.dtype, device=diag.device)

    P = [diag[..., v, :, :] for v in range(n)]
    Dinv: dict[int, Tensor] = {}
    ell: dict[int, Tensor] = {}
    logdet = diag.new_zeros(diag.shape[:-3])
    for v in collect:
        Dv = 0.5 * (P[v] + P[v].mT)
        Lv = torch.linalg.cholesky(Dv)
        logdet = logdet + 2.0 * torch.log(torch.diagonal(Lv, dim1=-2, dim2=-1)).sum(-1)
        Dinv[v] = torch.cholesky_solve(eye.expand_as(Dv), Lv)
        if v != root:
            U = edge[..., v, :, :]
            ell[v] = U @ Dinv[v]
            P[plist[v]] = P[plist[v]] - ell[v] @ U.mT
    return Dinv, ell, logdet, root, collect, plist, children


def _tree_collect_batched(diag, edge, parent_t):
    """Level-set batched, autograd-traceable collect: returns ``chol_D, ell, logdet``.

    One batched Cholesky per height antichain (:func:`tree_levels`) instead of the
    per-node loop of :func:`_tree_collect`, amortizing kernel-launch latency. Uses the
    **out-of-place** ``index_copy``/``index_add`` (not the ``_`` in-place forms of
    ``tree.py``'s selinv forward) so the whole pass stays differentiable through
    autograd. Mathematically identical to the loop (validated in ``tests/test_gmrf.py``).

    ``chol_D[v]`` is the Cholesky of pivot ``D_v``; ``ell[v] = Q_{p(v),v} D_v^{-1}``
    (root slot zero); ``logdet = log det Q``. Also returns ``root`` and the per-level
    index tensors so :func:`tree_solve` can batch its substitutions on the same schedule.
    """
    root, _children, _collect = tree_orders(parent_t)
    plist = parent_t.tolist()
    levels = tree_levels(parent_t)
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    lvl_idx = _level_index_tensors(levels, root, plist, diag.device)

    chol_D = diag.new_zeros((*batch, n, b, b))
    ell = diag.new_zeros((*batch, n, b, b))
    P = diag.clone()
    for h, (node_idx, nonroot_idx, parent_idx) in enumerate(lvl_idx):
        Pl = P.index_select(-3, node_idx)
        Pl = 0.5 * (Pl + Pl.mT)                       # symmetrize pivots (policy)
        chol_l = cholesky_spd(Pl, name=f"D(level h={h})")
        chol_D = chol_D.index_copy(-3, node_idx, chol_l)
        if nonroot_idx.numel() > 0:
            U = edge.index_select(-3, nonroot_idx)
            chol_nr = chol_D.index_select(-3, nonroot_idx)
            ell_nr = torch.cholesky_solve(U.mT, chol_nr).mT
            ell = ell.index_copy(-3, nonroot_idx, ell_nr)
            P = P.index_add(-3, parent_idx, -(ell_nr @ U.mT))

    logdet = 2.0 * torch.log(torch.diagonal(chol_D, dim1=-2, dim2=-1)).sum((-2, -1))
    return chol_D, ell, logdet, root, lvl_idx


def tree_logdet(diag, edge, parent, *, batched=False):
    """``log det Q`` for a tree-structured SPD block matrix (differentiable, ``O(n)``).

    With ``batched=True`` use the level-set schedule (one batched Cholesky per height
    antichain) instead of the per-node loop; mathematically identical.
    """
    p = _as_parent_tensor(parent)
    if batched:
        return _tree_collect_batched(diag, edge, p)[2]
    return _tree_collect(diag, edge, p)[2]


def tree_solve(diag, edge, parent, rhs, *, batched=False):
    """Solve ``Q x = rhs`` for a tree-structured SPD ``Q`` (differentiable, ``O(n)``).

    ``rhs`` has shape ``[..., n, b]``; returns ``x`` of the same shape. Two-pass:
    forward RHS elimination (leaves->root) then back-substitution (root->leaves). With
    ``batched=True`` both passes run on the level-set schedule (same result).
    """
    p = _as_parent_tensor(parent)
    if batched:
        return _tree_solve_batched(diag, edge, p, rhs)
    Dinv, ell, _logdet, root, collect, plist, _children = _tree_collect(diag, edge, p)
    n = diag.shape[-3]
    r = [rhs[..., v, :].unsqueeze(-1) for v in range(n)]      # [..., b, 1]
    for v in collect:
        if v != root:
            r[plist[v]] = r[plist[v]] - ell[v] @ r[v]
    x = [None] * n
    x[root] = Dinv[root] @ r[root]
    for v in reversed(collect):
        if v == root:
            continue
        x[v] = Dinv[v] @ r[v] - ell[v].mT @ x[plist[v]]
    return torch.stack([xi.squeeze(-1) for xi in x], dim=-2)


def _tree_solve_batched(diag, edge, parent_t, rhs):
    """Level-set batched version of :func:`tree_solve` (out-of-place, autograd-safe)."""
    chol_D, ell, _logdet, root, lvl_idx = _tree_collect_batched(diag, edge, parent_t)
    dev = diag.device
    r = rhs.unsqueeze(-1)                                     # [..., n, b, 1]
    # forward RHS elimination: leaves -> root (increasing height).
    for _node_idx, nonroot_idx, parent_idx in lvl_idx:
        if nonroot_idx.numel() > 0:
            ell_nr = ell.index_select(-3, nonroot_idx)
            r_nr = r.index_select(-3, nonroot_idx)
            r = r.index_add(-3, parent_idx, -(ell_nr @ r_nr))
    # back-substitution: root marginal, then root -> leaves (decreasing height).
    x = torch.zeros_like(r)
    ridx = torch.tensor([root], dtype=torch.long, device=dev)
    x = x.index_copy(-3, ridx, torch.cholesky_solve(
        r.index_select(-3, ridx), chol_D.index_select(-3, ridx)))
    for _node_idx, nonroot_idx, parent_idx in reversed(lvl_idx):
        if nonroot_idx.numel() == 0:
            continue
        chol_nr = chol_D.index_select(-3, nonroot_idx)
        r_nr = r.index_select(-3, nonroot_idx)
        ell_nr = ell.index_select(-3, nonroot_idx)
        x_par = x.index_select(-3, parent_idx)
        x_nr = torch.cholesky_solve(r_nr, chol_nr) - ell_nr.mT @ x_par
        x = x.index_copy(-3, nonroot_idx, x_nr)
    return x.squeeze(-1)


def marginal_log_likelihood(parent, y, a, kappa, root_prec, sigma2, *, batched=False):
    """Exact log marginal likelihood ``log p(y | theta)`` of the tree GMRF.

    ``y`` has shape ``[..., n, 1]``. Uses the evidence identity for ``y ~ N(0, Q^{-1} +
    sigma2 I)`` with ``Q_post = Q + sigma2^{-1} I`` (tree-structured): every term is a
    tree log-det or solve, so the whole quantity and its gradient are ``O(n)``. With
    ``batched=True`` the log-dets and solve run on the level-set schedule (same result).
    """
    p = _as_parent_tensor(parent)
    diag, edge = tree_gmrf_precision(p, a, kappa, root_prec)
    n, b = diag.shape[-3], diag.shape[-1]
    eye = torch.eye(b, dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)
    R = 1.0 / sigma2

    logdet_Q = tree_logdet(diag, edge, p, batched=batched)
    diag_post = diag + R * eye
    logdet_Qpost = tree_logdet(diag_post, edge, p, batched=batched)
    logdet_R = n * b * torch.log(R)

    Ry = R * y
    mu = tree_solve(diag_post, edge, p, Ry, batched=batched)
    yRy = (R * (y * y)).sum(dim=(-1, -2))
    quad2 = (Ry * mu).sum(dim=(-1, -2))

    N = n * b
    return (
        0.5 * logdet_Q
        + 0.5 * logdet_R
        - 0.5 * logdet_Qpost
        - 0.5 * yRy
        + 0.5 * quad2
        - 0.5 * N * math.log(2.0 * math.pi)
    )


def posterior_marginal_variances(parent, a, kappa, root_prec, sigma2, *, batched=False):
    """Posterior marginal variances ``diag(Q_post^{-1})`` via the selected inverse.

    Returns a ``[..., n]`` tensor of scalar variances. The gradient w.r.t. ``theta``
    flows through the **general** selinv adjoint (``selinv_tree``), not just the log-det
    case -- this is the objective that uniquely needs ``docs/derivations.md`` §8.
    """
    p = _as_parent_tensor(parent)
    diag, edge = tree_gmrf_precision(p, a, kappa, root_prec)
    eye = torch.eye(diag.shape[-1], dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)
    diag_post = diag + (1.0 / sigma2) * eye
    G_diag, _G_edge = selinv_tree(diag_post, edge, p, batched=batched)
    return torch.diagonal(G_diag, dim1=-2, dim2=-1).sum(-1)


def sample_tree_gmrf(parent, a, kappa, root_prec, *, seed=0, dtype=torch.float64):
    """Sample a latent field ``x ~ N(0, Q^{-1})`` via the generative branching process."""
    p = _as_parent_tensor(parent)
    n = int(p.numel())
    root, _children, collect = tree_orders(p)
    plist = p.tolist()
    g = torch.Generator().manual_seed(seed)
    kappa_v = _kappa_vector(p, kappa, n, dtype, torch.device("cpu"))
    a = torch.as_tensor(a, dtype=dtype)
    root_prec = torch.as_tensor(root_prec, dtype=dtype)
    z = torch.randn(n, generator=g, dtype=dtype)
    x = [torch.zeros((), dtype=dtype) for _ in range(n)]
    x[root] = z[root] / root_prec.sqrt()
    for v in reversed(collect):          # parents before children
        if v != root:
            x[v] = a * x[plist[v]] + z[v] / kappa_v[v].sqrt()
    return torch.stack(x).reshape(n, 1)


def fit_marginal_likelihood(
    parent, y, *, init=None, steps=300, lr=0.05, seed=0, batched=False
):
    """Learn ``(a, kappa, root_prec, sigma2)`` by maximizing the marginal likelihood.

    Positive parameters are optimized in softplus-unconstrained space. Returns a dict of
    recovered hyperparameters and the per-step negative-log-likelihood trace. With
    ``batched=True`` each step uses the level-set schedule for the GMRF objective.
    """
    p = _as_parent_tensor(parent)
    dtype = y.dtype
    init = init or {}

    def _inv_softplus(x):
        x = torch.as_tensor(x, dtype=dtype)
        return torch.log(torch.expm1(x.clamp_min(1e-6)))

    raw_a = torch.tensor(float(init.get("a", 0.5)), dtype=dtype, requires_grad=True)
    raw_kappa = _inv_softplus(init.get("kappa", 1.0)).clone().requires_grad_(True)
    raw_root = _inv_softplus(init.get("root_prec", 1.0)).clone().requires_grad_(True)
    raw_sigma2 = _inv_softplus(init.get("sigma2", 1.0)).clone().requires_grad_(True)
    params = [raw_a, raw_kappa, raw_root, raw_sigma2]
    opt = torch.optim.Adam(params, lr=lr)

    sp = torch.nn.functional.softplus
    trace = []
    for _ in range(steps):
        opt.zero_grad()
        # sum the per-sample log-likelihood over any leading iid batch dimension.
        nll = -marginal_log_likelihood(
            p, y, raw_a, sp(raw_kappa), sp(raw_root), sp(raw_sigma2), batched=batched
        ).sum()
        nll.backward()
        opt.step()
        trace.append(float(nll.detach()))

    return {
        "a": float(raw_a.detach()),
        "kappa": float(sp(raw_kappa).detach()),
        "root_prec": float(sp(raw_root).detach()),
        "sigma2": float(sp(raw_sigma2).detach()),
        "nll_trace": trace,
    }
