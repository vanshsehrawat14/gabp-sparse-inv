"""Correctness gates for the loopy (grid) GMRF on the junction kernel (D1).

Mirrors the tree-GMRF suite (`tests/test_gmrf.py`) on a 4-neighbour grid, where the graph
is loopy and the junction kernel handles it exactly via the filled pattern. Every term is
checked against a dense fp64 oracle: the CAR precision `Q = kappa (I + a L)` is SPD with the
right structure; the marginal likelihood equals a dense multivariate-normal log-prob; the
posterior marginal variances equal `diag(inv(Q_post))`; gradients of both objectives match
dense autograd; the result is invariant to the elimination order; and hyperparameters are
recovered from sampled data.
"""

from __future__ import annotations

import math

import pytest
import torch

from gabp_sparse_inv import (
    elimination_order_nested_dissection,
    fit_grid_marginal_likelihood,
    grid_edges,
    grid_gmrf_precision,
    grid_node_degrees,
    junction_marginal_log_likelihood,
    junction_posterior_marginal_variances,
    sample_gaussian_junction,
)
from gabp_sparse_inv.junction import _normalize_edge_index

torch.manual_seed(0)


def _laplacian(rows, cols):
    """Constant graph Laplacian L = D - W (fp64) for the differentiable dense oracle."""
    n = rows * cols
    ei = _normalize_edge_index(grid_edges(rows, cols))
    W = torch.zeros(n, n, dtype=torch.float64)
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        W[i, j] = W[j, i] = 1.0
    return torch.diag(W.sum(1)) - W


def _dense_Q(rows, cols, kappa, a):
    n = rows * cols
    eye = torch.eye(n, dtype=torch.float64)
    return kappa * eye + (kappa * a) * _laplacian(rows, cols)


def _dense_mll(rows, cols, y, kappa, a, sigma2):
    n = rows * cols
    eye = torch.eye(n, dtype=torch.float64)
    Sigma = torch.linalg.inv(_dense_Q(rows, cols, kappa, a)) + sigma2 * eye
    yv = y.reshape(-1)
    _sign, logabsdet = torch.linalg.slogdet(Sigma)
    quad = yv @ torch.linalg.solve(Sigma, yv)
    return -0.5 * (n * math.log(2.0 * math.pi) + logabsdet + quad)


def _dense_pv(rows, cols, kappa, a, sigma2):
    n = rows * cols
    eye = torch.eye(n, dtype=torch.float64)
    Qpost = _dense_Q(rows, cols, kappa, a) + (1.0 / sigma2) * eye
    return torch.diagonal(torch.linalg.inv(Qpost))


# --------------------------------------------------------------------------- #
# Precision structure.
# --------------------------------------------------------------------------- #
def test_degrees_and_precision_structure():
    rows, cols = 3, 4
    n = rows * cols
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, 1.3, 0.4)
    deg = grid_node_degrees(n, ei)
    # Corner deg 2, edge deg 3, interior deg 4 on a 3x4 grid.
    assert int(deg.max()) == 4 and int(deg.min()) == 2
    # Q_vv = kappa (1 + a deg_v); Q_ij = -kappa a.
    expect_diag = 1.3 * (1.0 + 0.4 * deg.to(torch.float64))
    assert torch.allclose(diag.reshape(-1), expect_diag, atol=1e-12)
    assert torch.allclose(edge_val.reshape(-1), torch.full((ei.shape[1],), -1.3 * 0.4, dtype=torch.float64))


def test_precision_is_spd():
    diag, ei, edge_val = grid_gmrf_precision(4, 5, 2.0, 0.9)
    Q = _dense_Q(4, 5, torch.tensor(2.0, dtype=torch.float64), torch.tensor(0.9, dtype=torch.float64))
    assert bool((torch.linalg.eigvalsh(Q) > 0).all())


# --------------------------------------------------------------------------- #
# Marginal likelihood + posterior variances vs dense oracle.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols", [(4, 5), (3, 6), (5, 5)])
def test_marginal_likelihood_matches_dense(rows, cols):
    n = rows * cols
    kappa, a, sigma2 = 1.3, 0.4, 0.7
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, kappa, a)
    y = torch.randn(n, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(rows))
    ll = junction_marginal_log_likelihood(diag, ei, edge_val, y, sigma2)
    ll_ref = _dense_mll(rows, cols, y,
                        torch.tensor(kappa, dtype=torch.float64),
                        torch.tensor(a, dtype=torch.float64),
                        torch.tensor(sigma2, dtype=torch.float64))
    assert torch.allclose(ll, ll_ref, atol=1e-9)


@pytest.mark.parametrize("rows,cols", [(4, 5), (3, 6), (5, 5)])
def test_posterior_variances_match_dense(rows, cols):
    kappa, a, sigma2 = 1.3, 0.4, 0.7
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, kappa, a)
    pv = junction_posterior_marginal_variances(diag, ei, edge_val, sigma2)
    pv_ref = _dense_pv(rows, cols,
                       torch.tensor(kappa, dtype=torch.float64),
                       torch.tensor(a, dtype=torch.float64),
                       torch.tensor(sigma2, dtype=torch.float64))
    assert torch.allclose(pv, pv_ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# Gradients vs dense autograd (the differentiable thread).
# --------------------------------------------------------------------------- #
def _theta():
    return (torch.tensor(1.3, dtype=torch.float64, requires_grad=True),
            torch.tensor(0.4, dtype=torch.float64, requires_grad=True),
            torch.tensor(0.7, dtype=torch.float64, requires_grad=True))


def test_marginal_likelihood_gradients_match_dense():
    rows, cols = 4, 5
    n = rows * cols
    y = torch.randn(n, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(2))

    k, a, s = _theta()
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, k, a)
    junction_marginal_log_likelihood(diag, ei, edge_val, y, s).backward()
    gk, ga, gs = k.grad.clone(), a.grad.clone(), s.grad.clone()

    k2, a2, s2 = _theta()
    _dense_mll(rows, cols, y, k2, a2, s2).backward()
    assert torch.allclose(gk, k2.grad, atol=1e-7)
    assert torch.allclose(ga, a2.grad, atol=1e-7)
    assert torch.allclose(gs, s2.grad, atol=1e-7)


def test_posterior_variance_gradients_match_dense():
    rows, cols = 4, 5
    w = torch.randn(rows * cols, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    k, a, s = _theta()
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, k, a)
    (junction_posterior_marginal_variances(diag, ei, edge_val, s) * w).sum().backward()
    gk, ga, gs = k.grad.clone(), a.grad.clone(), s.grad.clone()

    k2, a2, s2 = _theta()
    (_dense_pv(rows, cols, k2, a2, s2) * w).sum().backward()
    assert torch.allclose(gk, k2.grad, atol=1e-7)
    assert torch.allclose(ga, a2.grad, atol=1e-7)
    assert torch.allclose(gs, s2.grad, atol=1e-7)


# --------------------------------------------------------------------------- #
# Order-invariance: nested dissection gives identical objectives (different fill).
# --------------------------------------------------------------------------- #
def test_objectives_order_invariant():
    rows, cols = 5, 5
    n = rows * cols
    diag, ei, edge_val = grid_gmrf_precision(rows, cols, 1.3, 0.4)
    y = torch.randn(n, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    nd = elimination_order_nested_dissection(rows, cols)
    ll = junction_marginal_log_likelihood(diag, ei, edge_val, y, 0.7)
    ll_nd = junction_marginal_log_likelihood(diag, ei, edge_val, y, 0.7, order=nd)
    pv = junction_posterior_marginal_variances(diag, ei, edge_val, 0.7)
    pv_nd = junction_posterior_marginal_variances(diag, ei, edge_val, 0.7, order=nd)
    assert torch.allclose(ll, ll_nd, atol=1e-10)
    assert torch.allclose(pv, pv_nd, atol=1e-10)


# --------------------------------------------------------------------------- #
# Hyperparameter recovery from sampled data.
# --------------------------------------------------------------------------- #
def test_recovers_hyperparameters():
    rows, cols = 4, 4
    true = dict(kappa=1.5, a=0.6, sigma2=0.4)
    diag, ei, edge_val = grid_gmrf_precision(
        rows, cols, torch.tensor(true["kappa"], dtype=torch.float64),
        torch.tensor(true["a"], dtype=torch.float64))
    x = sample_gaussian_junction(diag, ei, edge_val, 150)            # [S, n, 1]
    g = torch.Generator().manual_seed(1)
    y = x + math.sqrt(true["sigma2"]) * torch.randn(x.shape, generator=g, dtype=torch.float64)

    out = fit_grid_marginal_likelihood(
        rows, cols, y, steps=120, lr=0.15, init=dict(kappa=1.0, a=0.3, sigma2=1.0))
    assert out["nll_trace"][-1] < out["nll_trace"][0]               # learning made progress
    assert abs(out["kappa"] - true["kappa"]) < 0.6
    assert abs(out["a"] - true["a"]) < 0.3
    assert abs(out["sigma2"] - true["sigma2"]) < 0.2
