"""Differentiable loopy (grid / general-graph) Gaussian Markov random field.

The junction-kernel counterpart of the tree GMRF (:mod:`gabp_sparse_inv.gmrf`): a latent
Gaussian field on an **arbitrary graph** (the 4-neighbour grid is the motivating loopy
case) with Gaussian observations, learned by exact marginal likelihood and posterior
marginal variances. Where the tree version uses ``tree_logdet`` / ``tree_solve`` /
``selinv_tree``, this uses their general-sparse siblings ``junction_logdet`` /
``junction_solve`` / ``selinv_junction`` on the filled (chordal) pattern -- so the graph
may be loopy and the kernel handles it *exactly* (not iteratively).

Latent precision (a conditional-autoregression / CAR model on the graph ``G = (V, E)``):

    Q(kappa, a) = kappa * (I + a * L),      L = D - W  (the graph Laplacian),

with ``kappa > 0`` and ``a >= 0``. ``L`` is PSD, so ``Q`` is SPD; in fact
``Q_vv = kappa (1 + a deg_v)`` and the off-diagonal row sum is ``kappa a deg_v``, so
``Q`` is strictly block-diagonally dominant (``Q_vv - sum_j |Q_vj| = kappa > 0``). The
edge blocks are ``Q_{ij} = -kappa a`` for ``(i, j) in E``. All nodes are scalar
(``b = 1``); the kernels themselves support general ``b``.

The observation model and learning objectives are identical to the tree case:

    x ~ N(0, Q^{-1}),   y = x + eps,   eps ~ N(0, sigma2 I),

so ``y ~ N(0, Q^{-1} + sigma2 I)``; the evidence and the posterior variances
``diag(Q_post^{-1})`` (with ``Q_post = Q + sigma2^{-1} I``) are exact and differentiable
in ``theta = (kappa, a, sigma2)`` by autograd through the junction kernels.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .generators import grid_edges
from .junction import junction_logdet, junction_solve, selinv_junction

__all__ = [
    "grid_gmrf_precision",
    "grid_node_degrees",
    "junction_marginal_log_likelihood",
    "junction_posterior_marginal_variances",
    "fit_grid_marginal_likelihood",
]


def _normalize_ei(edge_index: Tensor) -> Tensor:
    """Validate and coerce an edge index to a CPU long ``[2, m]`` tensor."""
    ei = edge_index.detach().to(device="cpu", dtype=torch.long)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, m]; got {tuple(ei.shape)}")
    return ei


def grid_node_degrees(n: int, edge_index: Tensor) -> Tensor:
    """Degree of each node (number of incident edges) as a ``[n]`` long tensor."""
    ei = _normalize_ei(edge_index)
    deg = torch.zeros(n, dtype=torch.long)
    deg.index_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
    deg.index_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))
    return deg


def grid_gmrf_precision(rows: int, cols: int, kappa, a):
    """Build the CAR latent precision ``Q = kappa (I + a L)`` for a ``rows x cols`` grid.

    Returns ``(diag, edge_index, edge_val)`` in the junction layout: ``diag`` ``[n, 1, 1]``
    holds ``Q_vv = kappa (1 + a deg_v)``; ``edge_index`` ``[2, m]`` is :func:`grid_edges`
    (``i > j``); ``edge_val`` ``[m, 1, 1]`` holds ``Q_{ij} = -kappa a``. Differentiable in
    ``kappa`` and ``a`` (tensors or floats); ``n = rows * cols``.
    """
    n = rows * cols
    edge_index = grid_edges(rows, cols)
    m = edge_index.shape[1]
    dtype = torch.float64
    for val in (kappa, a):
        if isinstance(val, Tensor) and val.is_floating_point():
            dtype = val.dtype
            break
    kappa = torch.as_tensor(kappa, dtype=dtype)
    a = torch.as_tensor(a, dtype=dtype)
    deg = grid_node_degrees(n, edge_index).to(dtype)

    diag_scalar = kappa * (1.0 + a * deg)                      # [n]
    diag = diag_scalar.reshape(n, 1, 1)
    edge_scalar = (-kappa * a).expand(m)                       # [m]
    edge_val = edge_scalar.reshape(m, 1, 1)
    return diag, edge_index, edge_val


def junction_marginal_log_likelihood(diag, edge_index, edge_val, y, sigma2, *, order=None):
    """Exact log marginal likelihood ``log p(y | theta)`` of a junction-layout GMRF.

    ``y`` has shape ``[..., n, 1]``; the precision ``Q`` is given by ``(diag, edge_index,
    edge_val)``. Uses the evidence identity for ``y ~ N(0, Q^{-1} + sigma2 I)`` with
    ``Q_post = Q + sigma2^{-1} I`` (same chordal pattern), so every term is a junction
    log-det or solve and the whole quantity / its gradient flow by autograd. ``order`` is
    an optional shared elimination order (e.g. :func:`elimination_order_nested_dissection`).
    """
    n, b = diag.shape[-3], diag.shape[-1]
    eye = torch.eye(b, dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)
    R = 1.0 / sigma2

    logdet_Q = junction_logdet(diag, edge_index, edge_val, order)
    diag_post = diag + R * eye
    logdet_Qpost = junction_logdet(diag_post, edge_index, edge_val, order)
    logdet_R = n * b * torch.log(R)

    Ry = R * y
    mu = junction_solve(diag_post, edge_index, edge_val, Ry, order)
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


def junction_posterior_marginal_variances(diag, edge_index, edge_val, sigma2, *, order=None):
    """Posterior marginal variances ``diag(Q_post^{-1})`` via the junction selected inverse.

    Returns a ``[..., n]`` tensor of scalar variances (``Q_post = Q + sigma2^{-1} I``). The
    gradient w.r.t. ``theta`` flows through the junction selinv adjoint (``selinv_junction``).
    """
    eye = torch.eye(diag.shape[-1], dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)
    diag_post = diag + (1.0 / sigma2) * eye
    G_diag, _S_index, _G_lower = selinv_junction(diag_post, edge_index, edge_val, order)
    return torch.diagonal(G_diag, dim1=-2, dim2=-1).sum(-1)


def fit_grid_marginal_likelihood(
    rows, cols, y, *, init=None, steps=300, lr=0.05, order=None
):
    """Learn ``(kappa, a, sigma2)`` of the grid CAR-GMRF by maximizing marginal likelihood.

    Positive parameters are optimized in softplus-unconstrained space (``a >= 0`` keeps
    ``Q`` SPD). Returns a dict of recovered hyperparameters and the per-step NLL trace.
    ``y`` has shape ``[..., n, 1]``; ``order`` is an optional shared elimination order.
    """
    dtype = y.dtype
    init = init or {}

    def _inv_softplus(x):
        x = torch.as_tensor(x, dtype=dtype)
        return torch.log(torch.expm1(x.clamp_min(1e-6)))

    raw_kappa = _inv_softplus(init.get("kappa", 1.0)).clone().requires_grad_(True)
    raw_a = _inv_softplus(init.get("a", 0.5)).clone().requires_grad_(True)
    raw_sigma2 = _inv_softplus(init.get("sigma2", 1.0)).clone().requires_grad_(True)
    params = [raw_kappa, raw_a, raw_sigma2]
    opt = torch.optim.Adam(params, lr=lr)
    sp = torch.nn.functional.softplus

    trace = []
    for _ in range(steps):
        opt.zero_grad()
        diag, ei, edge_val = grid_gmrf_precision(rows, cols, sp(raw_kappa), sp(raw_a))
        nll = -junction_marginal_log_likelihood(
            diag, ei, edge_val, y, sp(raw_sigma2), order=order
        ).sum()
        nll.backward()
        opt.step()
        trace.append(float(nll.detach()))

    return {
        "kappa": float(sp(raw_kappa).detach()),
        "a": float(sp(raw_a).detach()),
        "sigma2": float(sp(raw_sigma2).detach()),
        "nll_trace": trace,
    }
