"""Validation gates for the differentiable tree-GMRF application (``gmrf.py``).

Every tree-tractable quantity is checked against a dense oracle, and the two learning
objectives are checked for gradient correctness against dense autograd:

* ``tree_logdet`` / ``tree_solve`` / ``marginal_log_likelihood`` /
  ``posterior_marginal_variances`` vs dense ``logdet`` / ``solve`` / MVN / ``inv``;
* the marginal-likelihood gradient (log-det case) and the posterior-variance gradient
  (the *general* selinv adjoint -- with a nonzero coupling gradient, distinguishing it
  from the log-det-only case) vs dense autograd;
* hyperparameter recovery on synthetic data.
"""

from __future__ import annotations

import math

import pytest
import torch

from gabp_sparse_inv import BlockTree
from gabp_sparse_inv.gmrf import (
    fit_marginal_likelihood,
    marginal_log_likelihood,
    posterior_marginal_variances,
    sample_tree_gmrf,
    tree_gmrf_precision,
    tree_logdet,
    tree_solve,
)
from gabp_sparse_inv.layout import _as_parent_tensor

torch.manual_seed(0)
PARENTS = {
    "balanced": [-1, 0, 0, 1, 1, 2, 2],
    "deep": [-1, 0, 1, 2, 3, 4, 5, 6],
    "wide": [-1, 0, 0, 0, 0, 0],
}
THETA = dict(a=0.7, kappa=1.5, root_prec=2.0, sigma2=0.3)


def _dense_Q(parent, a, kappa, root_prec):
    diag, edge = tree_gmrf_precision(parent, a, kappa, root_prec)
    return BlockTree(diag=diag, edge=edge, parent=_as_parent_tensor(parent)).to_dense()


@pytest.mark.parametrize("name", list(PARENTS))
def test_precision_spd_and_logdet(name):
    parent = PARENTS[name]
    Q = _dense_Q(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    eig = torch.linalg.eigvalsh(Q.double())
    assert float(eig.min()) > 0
    diag, edge = tree_gmrf_precision(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    ld = tree_logdet(diag, edge, parent)
    assert torch.allclose(ld, torch.logdet(Q), atol=1e-10)


@pytest.mark.parametrize("name", list(PARENTS))
def test_solve_matches_dense(name):
    parent = PARENTS[name]
    n = len(parent)
    diag, edge = tree_gmrf_precision(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    Q = _dense_Q(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    b = torch.randn(n, 1, dtype=torch.float64)
    x = tree_solve(diag, edge, parent, b)
    assert torch.allclose(x, torch.linalg.solve(Q, b), atol=1e-10)


@pytest.mark.parametrize("name", list(PARENTS))
def test_marginal_likelihood_matches_dense(name):
    parent = PARENTS[name]
    n = len(parent)
    y = torch.randn(n, 1, dtype=torch.float64)
    Q = _dense_Q(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    Sigma = torch.linalg.inv(Q) + THETA["sigma2"] * torch.eye(n, dtype=torch.float64)
    dense = torch.distributions.MultivariateNormal(
        torch.zeros(n, dtype=torch.float64), covariance_matrix=Sigma
    ).log_prob(y.squeeze(-1))
    ll = marginal_log_likelihood(parent, y, **THETA)
    assert torch.allclose(ll, dense, atol=1e-9)


@pytest.mark.parametrize("name", list(PARENTS))
def test_posterior_variances_match_dense(name):
    parent = PARENTS[name]
    n = len(parent)
    Q = _dense_Q(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    Qpost = Q + (1.0 / THETA["sigma2"]) * torch.eye(n, dtype=torch.float64)
    var_dense = torch.diagonal(torch.linalg.inv(Qpost))
    var = posterior_marginal_variances(parent, **THETA)
    assert torch.allclose(var, var_dense, atol=1e-10)
    # batched path agrees with the loop path.
    var_b = posterior_marginal_variances(parent, **THETA, batched=True)
    assert torch.allclose(var, var_b, atol=1e-12)


def test_marginal_likelihood_grad_vs_dense():
    parent = PARENTS["balanced"]
    n = len(parent)
    y = torch.randn(n, 1, dtype=torch.float64)

    def grads(dense: bool):
        a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
        k = torch.tensor(THETA["kappa"], dtype=torch.float64, requires_grad=True)
        r = torch.tensor(THETA["root_prec"], dtype=torch.float64, requires_grad=True)
        s = torch.tensor(THETA["sigma2"], dtype=torch.float64, requires_grad=True)
        if dense:
            diag, edge = tree_gmrf_precision(parent, a, k, r)
            Q = BlockTree(diag=diag, edge=edge, parent=_as_parent_tensor(parent)).to_dense()
            Sigma = torch.linalg.inv(Q) + s * torch.eye(n, dtype=torch.float64)
            ll = torch.distributions.MultivariateNormal(
                torch.zeros(n, dtype=torch.float64), covariance_matrix=Sigma
            ).log_prob(y.squeeze(-1))
        else:
            ll = marginal_log_likelihood(parent, y, a, k, r, s)
        ll.backward()
        return torch.tensor([a.grad, k.grad, r.grad, s.grad])

    assert torch.allclose(grads(False), grads(True), atol=1e-8)


def test_posterior_variance_grad_vs_dense_and_uses_general_adjoint():
    parent = PARENTS["balanced"]
    n = len(parent)
    target = torch.full((n,), 0.2, dtype=torch.float64)

    def grads(dense: bool):
        a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
        k = torch.tensor(THETA["kappa"], dtype=torch.float64, requires_grad=True)
        r = torch.tensor(THETA["root_prec"], dtype=torch.float64, requires_grad=True)
        s = torch.tensor(THETA["sigma2"], dtype=torch.float64, requires_grad=True)
        if dense:
            diag, edge = tree_gmrf_precision(parent, a, k, r)
            Q = BlockTree(diag=diag, edge=edge, parent=_as_parent_tensor(parent)).to_dense()
            Qpost = Q + (1.0 / s) * torch.eye(n, dtype=torch.float64)
            var = torch.diagonal(torch.linalg.inv(Qpost))
        else:
            var = posterior_marginal_variances(parent, a, k, r, s)
        ((var - target) ** 2).sum().backward()
        return a, torch.tensor([a.grad, k.grad, r.grad, s.grad])

    a_tree, g_tree = grads(False)
    _, g_dense = grads(True)
    assert torch.allclose(g_tree, g_dense, atol=1e-9)
    # The coupling gradient is nonzero: the objective depends on off-diagonal precision
    # blocks through G_edge, i.e. it exercises the general selinv adjoint (not log-det).
    assert abs(float(g_tree[0])) > 1e-6


def test_hyperparameter_recovery():
    parent = PARENTS["balanced"]
    n = len(parent)
    a0, k0, r0, s0 = THETA["a"], THETA["kappa"], THETA["root_prec"], THETA["sigma2"]
    fields = torch.cat([sample_tree_gmrf(parent, a0, k0, r0, seed=s) for s in range(3000)], dim=1)
    g = torch.Generator().manual_seed(123)
    y = (fields + math.sqrt(s0) * torch.randn(n, 3000, generator=g, dtype=torch.float64))
    y = y.T.reshape(3000, n, 1)
    res = fit_marginal_likelihood(parent, y, steps=400, lr=0.05)
    assert abs(res["a"] - a0) < 0.1
    assert abs(res["kappa"] - k0) < 0.25
    assert abs(res["root_prec"] - r0) < 0.4
    assert abs(res["sigma2"] - s0) < 0.1
    assert res["nll_trace"][-1] < res["nll_trace"][0]


def test_batched_marginal_likelihood_shape():
    parent = PARENTS["balanced"]
    n = len(parent)
    y = torch.randn(5, n, 1, dtype=torch.float64)
    ll = marginal_log_likelihood(parent, y, **THETA)
    assert ll.shape == (5,)


@pytest.mark.parametrize("name", list(PARENTS))
def test_batched_logdet_solve_match_loop(name):
    parent = PARENTS[name]
    n = len(parent)
    diag, edge = tree_gmrf_precision(parent, THETA["a"], THETA["kappa"], THETA["root_prec"])
    ld = tree_logdet(diag, edge, parent)
    ld_b = tree_logdet(diag, edge, parent, batched=True)
    assert torch.allclose(ld, ld_b, atol=1e-12)

    rhs = torch.randn(3, n, 1, dtype=torch.float64)          # leading batch dim
    x = tree_solve(diag, edge, parent, rhs)
    x_b = tree_solve(diag, edge, parent, rhs, batched=True)
    assert torch.allclose(x, x_b, atol=1e-12)


@pytest.mark.parametrize("name", list(PARENTS))
def test_batched_marginal_likelihood_matches_loop(name):
    parent = PARENTS[name]
    n = len(parent)
    y = torch.randn(4, n, 1, dtype=torch.float64)
    ll = marginal_log_likelihood(parent, y, **THETA)
    ll_b = marginal_log_likelihood(parent, y, **THETA, batched=True)
    assert torch.allclose(ll, ll_b, atol=1e-11)


def test_batched_marginal_likelihood_grad_matches_loop():
    parent = PARENTS["balanced"]
    n = len(parent)
    y = torch.randn(n, 1, dtype=torch.float64)

    def grads(batched: bool):
        a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
        k = torch.tensor(THETA["kappa"], dtype=torch.float64, requires_grad=True)
        r = torch.tensor(THETA["root_prec"], dtype=torch.float64, requires_grad=True)
        s = torch.tensor(THETA["sigma2"], dtype=torch.float64, requires_grad=True)
        marginal_log_likelihood(parent, y, a, k, r, s, batched=batched).backward()
        return torch.tensor([a.grad, k.grad, r.grad, s.grad])

    assert torch.allclose(grads(False), grads(True), atol=1e-11)


def test_scaling_module_smoke(tmp_path):
    from gabp_sparse_inv.bench.gmrf_scaling import balanced_binary_parent, main

    assert balanced_binary_parent(7) == [-1, 0, 0, 1, 1, 2, 2]
    rc = main(["--values", "7", "15", "--samples", "4", "--dense-max-n", "15",
               "--out", str(tmp_path / "s")])
    assert rc == 0
    assert (tmp_path / "s.json").exists()
