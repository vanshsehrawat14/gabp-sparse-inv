"""The tape-free analytic fit gradient equals autograd through the taped path.

`_marginal_nll_and_grads` / `fit_grid_marginal_likelihood_analytic` re-derive the marginal
NLL gradient from the selected-inverse identities; these tests pin them to the existing
autograd path (`junction_marginal_log_likelihood` + backward) at machine precision, so the
analytic route is a drop-in for the exact-arm fit.
"""

import pytest
import torch

from gabp_sparse_inv.gmrf_grid import (
    _marginal_nll_and_grads,
    fit_grid_marginal_likelihood,
    fit_grid_marginal_likelihood_analytic,
    grid_gmrf_precision,
    junction_marginal_log_likelihood,
)


def _taped_nll_and_grads(rows, cols, y, kappa, a, sigma2, order=None):
    kappa = torch.tensor(kappa, dtype=y.dtype, requires_grad=True)
    a = torch.tensor(a, dtype=y.dtype, requires_grad=True)
    sigma2 = torch.tensor(sigma2, dtype=y.dtype, requires_grad=True)
    diag, ei, ev = grid_gmrf_precision(rows, cols, kappa, a)
    nll = -junction_marginal_log_likelihood(diag, ei, ev, y, sigma2, order=order).sum()
    gk, ga, gs = torch.autograd.grad(nll, (kappa, a, sigma2))
    return nll.detach(), gk, ga, gs


@pytest.mark.parametrize("theta", [(1.0, 0.5, 1.0), (2.0, 4.0, 0.1), (0.7, 64.0, 1e-3)])
@pytest.mark.parametrize("shape", [(3, 4), (5, 5)])
def test_analytic_grads_match_autograd(shape, theta):
    rows, cols = shape
    torch.manual_seed(0)
    y = torch.randn(rows * cols, 1, dtype=torch.float64)
    ref = _taped_nll_and_grads(rows, cols, y, *theta)
    got = _marginal_nll_and_grads(rows, cols, y, *theta)
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, rtol=1e-9, atol=1e-9)


def test_analytic_grads_match_autograd_batched():
    torch.manual_seed(1)
    y = torch.randn(3, 20, 1, dtype=torch.float64)
    ref = _taped_nll_and_grads(4, 5, y, 1.3, 2.0, 0.25)
    got = _marginal_nll_and_grads(4, 5, y, 1.3, 2.0, 0.25)
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, rtol=1e-9, atol=1e-9)


def test_analytic_fit_matches_taped_fit():
    torch.manual_seed(2)
    rows, cols = 4, 5
    y = torch.randn(rows * cols, 1, dtype=torch.float64)
    init = {"kappa": 1.0, "a": 0.5, "sigma2": 1.0}
    ref = fit_grid_marginal_likelihood(rows, cols, y, init=init, steps=12, lr=0.05)
    got = fit_grid_marginal_likelihood_analytic(rows, cols, y, init=init, steps=12, lr=0.05)
    assert got["nll_trace"] == pytest.approx(ref["nll_trace"], rel=1e-9, abs=1e-9)
    for key in ("kappa", "a", "sigma2"):
        assert got[key] == pytest.approx(ref[key], rel=1e-8)


@pytest.mark.parametrize("sigma2", [1e-8, 1e-14])
def test_analytic_grads_match_dense_oracle_at_small_noise(sigma2):
    # Pins the tape-free gradient to a dense covariance oracle, not only to the taped
    # path, where both once shared the cancelling Woodbury form.
    rows, cols, kappa, a = 3, 4, 1.5, 3.0
    torch.manual_seed(4)
    y = torch.randn(2, rows * cols, 1, dtype=torch.float64)

    def dense_nll(k, aa, s):
        n = rows * cols
        diag, ei, ev = grid_gmrf_precision(rows, cols, k, aa)
        Q = torch.diag(diag.reshape(n))
        Q = Q.index_put((ei[0], ei[1]), ev.reshape(-1)).index_put((ei[1], ei[0]), ev.reshape(-1))
        Sigma = torch.linalg.inv(Q) + s * torch.eye(n, dtype=torch.float64)
        mvn = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), Sigma)
        return -mvn.log_prob(y.squeeze(-1)).sum()

    params = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (kappa, a, sigma2)]
    ref = dense_nll(*params)
    ref_grads = torch.autograd.grad(ref, params)
    got = _marginal_nll_and_grads(rows, cols, y, kappa, a, sigma2)
    torch.testing.assert_close(got[0], ref.detach(), rtol=1e-10, atol=0.0)
    for g, r in zip(got[1:], ref_grads):
        torch.testing.assert_close(g, r, rtol=1e-7, atol=1e-9)
