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
in ``theta = (kappa, a, sigma2)`` by autograd through the junction kernels. The evidence
factors ``B = sigma2 Q_post = I + sigma2 Q`` so that it stays accurate as ``sigma2 -> 0``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .generators import grid_edges
from .junction import JunctionAnalysis, _normalize_edge_index as _normalize_ei, analyze_junction
from .layout import _check_edge_index

__all__ = [
    "grid_gmrf_precision",
    "grid_node_degrees",
    "junction_marginal_log_likelihood",
    "junction_posterior_marginal_variances",
    "fit_grid_marginal_likelihood",
    "fit_grid_marginal_likelihood_analytic",
]




def _resolve_analysis(n, edge_index, order, analysis):
    """Return a compatible persistent symbolic plan, checking any supplied plan."""
    ei = _normalize_ei(edge_index)
    if analysis is None:
        return analyze_junction(n, ei, order)
    if not isinstance(analysis, JunctionAnalysis):
        raise TypeError(f"analysis must be a JunctionAnalysis; got {type(analysis).__name__}")
    if analysis.n != n or not torch.equal(analysis.edge_index, ei):
        raise ValueError("analysis pattern does not match the supplied precision pattern")
    if order is not None:
        requested = order.tolist() if isinstance(order, Tensor) else list(order)
        if requested != analysis.order:
            raise ValueError("order disagrees with the supplied analysis")
    return analysis


def grid_node_degrees(n: int, edge_index: Tensor) -> Tensor:
    """Degree of each node (number of incident edges) as a ``[n]`` long tensor."""
    ei = _normalize_ei(edge_index)
    _check_edge_index(ei, n)
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


def _junction_matvec(diag, edge_index, edge_val, x):
    """``Q x`` for junction storage; ``x`` is ``[..., n, b]``."""
    xc = x.unsqueeze(-1)
    out = diag @ xc
    if edge_index.shape[1] > 0:
        i, j = edge_index[0].to(x.device), edge_index[1].to(x.device)
        out = out.index_add(-3, i, edge_val @ xc.index_select(-3, j))
        out = out.index_add(-3, j, edge_val.mT @ xc.index_select(-3, i))
    return out.squeeze(-1)


def junction_marginal_log_likelihood(
    diag, edge_index, edge_val, y, sigma2, *, order=None, analysis=None
):
    """Exact log marginal likelihood ``log p(y | theta)`` of a junction-layout GMRF.

    ``y`` has shape ``[..., n, 1]``; the precision ``Q`` is given by ``(diag, edge_index,
    edge_val)`` and ``y ~ N(0, Q^{-1} + sigma2 I)``. With ``B = I + sigma2 Q`` (same
    chordal pattern) and the posterior mean ``mu = B^{-1} y``,

        log p(y) = 1/2 log det Q - 1/2 log det B
                   - 1/2 (mu' Q mu + sigma2 ||Q mu||^2) - N/2 log(2 pi).

    Every term is a junction log-det, solve or sparse product, and none grows like
    ``1 / sigma2``, so the value and its gradient stay accurate as ``sigma2 -> 0``.
    ``order`` is an optional shared elimination order (e.g.
    :func:`elimination_order_nested_dissection`). Pass a compatible
    :class:`~gabp_sparse_inv.JunctionAnalysis` to reuse symbolic work; ``B`` is
    numerically factored once and serves both its logdet and solve.
    """
    n, b = diag.shape[-3], diag.shape[-1]
    eye = torch.eye(b, dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)

    analysis = _resolve_analysis(n, edge_index, order, analysis)
    logdet_Q = analysis.factor(diag, edge_val).logdet()
    factor_B = analysis.factor(eye + sigma2 * diag, sigma2 * edge_val)
    logdet_B = factor_B.logdet()
    mu = factor_B.solve(y)
    Qmu = _junction_matvec(diag, analysis.edge_index, edge_val, mu)
    quad = (mu * Qmu).sum(dim=(-1, -2)) + sigma2 * (Qmu * Qmu).sum(dim=(-1, -2))

    return 0.5 * logdet_Q - 0.5 * logdet_B - 0.5 * quad - 0.5 * n * b * math.log(2.0 * math.pi)


def junction_posterior_marginal_variances(
    diag, edge_index, edge_val, sigma2, *, order=None, analysis=None
):
    """Posterior marginal variances ``diag(Q_post^{-1})`` via the junction selected inverse.

    Returns the per-coordinate variances (``Q_post = Q + sigma2^{-1} I``): a ``[..., n]``
    tensor for scalar nodes (``b = 1``) and ``[..., n, b]`` for block nodes. The gradient
    w.r.t. ``theta`` flows through the functional selected-inverse schedule. A supplied
    :class:`~gabp_sparse_inv.JunctionAnalysis` reuses the symbolic plan.
    """
    eye = torch.eye(diag.shape[-1], dtype=diag.dtype, device=diag.device)
    sigma2 = torch.as_tensor(sigma2, dtype=diag.dtype, device=diag.device)
    diag_post = diag + (1.0 / sigma2) * eye
    analysis = _resolve_analysis(diag.shape[-3], edge_index, order, analysis)
    variances = analysis.factor(diag_post, edge_val).marginal_diag()
    return variances.squeeze(-1) if variances.shape[-1] == 1 else variances


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
    analysis = analyze_junction(rows * cols, grid_edges(rows, cols), order)

    trace = []
    for _ in range(steps):
        opt.zero_grad()
        diag, ei, edge_val = grid_gmrf_precision(rows, cols, sp(raw_kappa), sp(raw_a))
        nll = -junction_marginal_log_likelihood(
            diag, ei, edge_val, y, sp(raw_sigma2), analysis=analysis
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


def _marginal_nll_and_grads(
    rows, cols, y, kappa, a, sigma2, order=None, analysis=None
):
    """Tape-free ``(nll, d_kappa, d_a, d_sigma2)`` of ``nll = -junction_marginal_log_likelihood(...).sum()``.

    Same quantity and gradient as autograd through :func:`junction_marginal_log_likelihood`,
    assembled from classical identities instead of the tape. With ``B = I + sigma2 Q`` and
    ``mu = B^{-1} y``: ``d log det Q = <Q^{-1}, dQ>`` and
    ``d log det B = sigma2 <B^{-1}, dQ> + tr(B^{-1} Q) d sigma2`` on the pattern (selected
    inverses); the quadratic ``mu' Q mu + sigma2 ||Q mu||^2`` is the minimum over ``x`` of
    ``x' Q x + ||y - x||^2 / sigma2``, so its derivatives are ``<mu mu', dQ>`` and
    ``-||Q mu||^2 d sigma2``. Runs entirely under ``no_grad`` with numeric storage
    proportional to the factor pattern: two selected inverses and one solve per call, from
    two numeric factors. Scalar nodes only (``b = 1``, the grid CAR layout); validated
    against the taped path in ``tests/test_gmrf_grid.py``.
    """
    with torch.no_grad():
        dtype = y.dtype
        n = rows * cols
        kappa = torch.as_tensor(kappa, dtype=dtype)
        a = torch.as_tensor(a, dtype=dtype)
        s2 = torch.as_tensor(sigma2, dtype=dtype)

        diag, ei, edge_val = grid_gmrf_precision(rows, cols, kappa, a)
        analysis = _resolve_analysis(n, ei, order, analysis)
        deg = grid_node_degrees(n, ei).to(dtype)
        yb = y.reshape(-1, n, 1)
        nb = yb.shape[0]

        def _selinv_with_logdet(d, e):
            factor = analysis.factor(d, e)
            G_diag, S_index, G_lower = factor.selected_inverse()
            return G_diag.reshape(n), S_index, G_lower.reshape(-1), factor.logdet(), factor

        Gq_d, S_index, Gq_l, logdet_Q, _factor_Q = _selinv_with_logdet(diag, edge_val)
        Gb_d, _, Gb_l, logdet_B, factor_B = _selinv_with_logdet(1.0 + s2 * diag, s2 * edge_val)

        # Positions of the original edges inside the filled pattern S (both i > j).
        skey = S_index[0] * n + S_index[1]
        sorted_skey, sort_idx = torch.sort(skey)
        ekey = ei[0] * n + ei[1]
        edge_pos = sort_idx[torch.searchsorted(sorted_skey, ekey)]
        if not torch.equal(skey[edge_pos], ekey):
            raise RuntimeError("original edges not found in the filled pattern S")
        Gq_e, Gb_e = Gq_l[edge_pos], Gb_l[edge_pos]

        mu = factor_B.solve(yb)
        Qmu = _junction_matvec(diag, ei, edge_val, mu).reshape(nb, n)
        mu = mu.reshape(nb, n)
        quad = (mu * Qmu).sum() + s2 * (Qmu * Qmu).sum()

        # ll summed over batch (mirrors junction_marginal_log_likelihood term by term).
        ll = (
            0.5 * nb * logdet_Q - 0.5 * nb * logdet_B - 0.5 * quad
            - 0.5 * nb * n * math.log(2.0 * math.pi)
        )

        # d(ll)/dQ on the pattern: logdet terms scale with nb; the quadratic sums the
        # per-batch -0.5 mu mu' outer products (diag and original-edge entries only).
        mu_i, mu_j = mu[:, ei[0]], mu[:, ei[1]]
        d_grad = 0.5 * nb * (Gq_d - s2 * Gb_d) - 0.5 * (mu * mu).sum(0)
        e_grad = 0.5 * nb * (Gq_e - s2 * Gb_e) - 0.5 * (mu_i * mu_j).sum(0)

        # Chain to theta: dQ_ii = (1 + a deg_i) dkappa + kappa deg_i da; dQ_ij(edge) =
        # -a dkappa - kappa da; off-diagonal entries count twice (symmetric contraction).
        d_kappa = (d_grad * (1.0 + a * deg)).sum() + 2.0 * (e_grad * (-a)).sum()
        d_a = (d_grad * (kappa * deg)).sum() + 2.0 * (e_grad * (-kappa)).sum()
        # d(ll)/d sigma2 = -0.5 nb tr(B^{-1} Q) + 0.5 ||Q mu||^2, summed over the batch.
        tr_BinvQ = (Gb_d * diag.reshape(n)).sum() + 2.0 * (Gb_e * edge_val.reshape(-1)).sum()
        d_sigma2 = -0.5 * nb * tr_BinvQ + 0.5 * (Qmu * Qmu).sum()

        return -ll, -d_kappa, -d_a, -d_sigma2


def fit_grid_marginal_likelihood_analytic(
    rows, cols, y, *, init=None, steps=300, lr=0.05, order=None
):
    """Drop-in :func:`fit_grid_marginal_likelihood` with tape-free analytic gradients.

    Identical objective, parameterization (softplus-unconstrained), optimizer, and return
    dict as the taped fit; only the gradient computation differs: it is assembled analytically
    per step by :func:`_marginal_nll_and_grads` instead of reverse-mode through the
    factorization tape, so numeric tensor storage stays proportional to the factor pattern
    instead of growing with the tape (the current level-set symbolic schedule may still scale
    with squared front sizes)
    (measured 2026-07-08: the taped fit at ``n = 16384`` holds ~382 GiB). Deterministic:
    identical gradients give an identical Adam trajectory up to float reassociation.
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
    analysis = analyze_junction(rows * cols, grid_edges(rows, cols), order)

    trace = []
    for _ in range(steps):
        opt.zero_grad()
        with torch.no_grad():
            nll, d_kappa, d_a, d_sigma2 = _marginal_nll_and_grads(
                rows,
                cols,
                y,
                sp(raw_kappa),
                sp(raw_a),
                sp(raw_sigma2),
                analysis=analysis,
            )
            # Chain through softplus: d(raw) = d(param) * sigmoid(raw).
            raw_kappa.grad = d_kappa * torch.sigmoid(raw_kappa)
            raw_a.grad = d_a * torch.sigmoid(raw_a)
            raw_sigma2.grad = d_sigma2 * torch.sigmoid(raw_sigma2)
        opt.step()
        trace.append(float(nll))

    return {
        "kappa": float(sp(raw_kappa).detach()),
        "a": float(sp(raw_a).detach()),
        "sigma2": float(sp(raw_sigma2).detach()),
        "nll_trace": trace,
    }
