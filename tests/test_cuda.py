"""CUDA smoke test: the main entry points run on a CUDA device and match the CPU result.

Skipped when no CUDA device is present. The package is device-agnostic PyTorch code, but
CI has no GPU, so this file is the check to run on GPU hardware.
"""

from __future__ import annotations

import pytest
import torch

import gabp_sparse_inv as gs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
F64 = torch.float64


def _match(cpu_out, gpu_out):
    for c, g in zip(cpu_out, gpu_out):
        if isinstance(c, torch.Tensor) and c.is_floating_point():
            assert g.device.type == "cuda"
            torch.testing.assert_close(g.cpu(), c, rtol=1e-9, atol=1e-12)


def _grads(fn, inputs, device):
    xs = [x.to(device).requires_grad_(True) for x in inputs]
    out = fn(*xs)
    loss = sum(o.square().sum() for o in out if isinstance(o, torch.Tensor) and o.is_floating_point())
    return torch.autograd.grad(loss, xs)


def test_junction_routes_on_cuda():
    A = gs.random_spd_graph(16, gs.grid_edges(4, 4), 2, seed=0, diag_load=1.0)
    ei, order = A.edge_index, gs.elimination_order_nested_dissection(4, 4)
    d, ev = A.diag, A.edge_val
    dc, evc = d.cuda(), ev.cuda()
    rhs = torch.randn(16, 2, dtype=F64)
    for batched in (False, True):
        _match(gs.selected_inverse_junction(d, ei, ev, order, batched=batched),
               gs.selected_inverse_junction(dc, ei, evc, order, batched=batched))
    _match([gs.junction_solve(d, ei, ev, rhs)], [gs.junction_solve(dc, ei, evc, rhs.cuda())])
    _match([gs.junction_logdet(d, ei, ev)], [gs.junction_logdet(dc, ei, evc)])
    analysis = gs.analyze_junction(16, ei, order)
    f_cpu, f_gpu = analysis.factor(d, ev, batched=True), analysis.factor(dc, evc, batched=True)
    _match([f_cpu.logdet(), f_cpu.solve(rhs), f_cpu.marginal_diag()],
           [f_gpu.logdet(), f_gpu.solve(rhs.cuda()), f_gpu.marginal_diag()])
    assert f_gpu.sample(3).device.type == "cuda"
    assert gs.sample_gaussian_junction(dc, ei, evc, 3).device.type == "cuda"
    for fn in (lambda d, e: gs.selinv_junction(d, ei, e, order, batched=True),
               lambda d, e: gs.selinv_junction_analytic(d, ei, e, order)):
        _match(_grads(fn, (d, ev), "cpu"), _grads(fn, (d, ev), "cuda"))


def test_tree_chain_star_on_cuda():
    T = gs.random_spd_tree(9, 2, seed=1, diag_load=1.0)
    rhs = torch.randn(9, 2, dtype=F64)
    for batched in (False, True):
        fn = lambda d, e, batched=batched: gs.selected_inverse_tree(d, e, T.parent, batched=batched)  # noqa: E731
        _match(fn(T.diag, T.edge), fn(T.diag.cuda(), T.edge.cuda()))
        _match(_grads(fn, (T.diag, T.edge), "cpu"), _grads(fn, (T.diag, T.edge), "cuda"))
        sel = lambda d, e, batched=batched: gs.selinv_tree(d, e, T.parent, batched=batched)  # noqa: E731
        _match(_grads(sel, (T.diag, T.edge), "cpu"), _grads(sel, (T.diag, T.edge), "cuda"))
    _match([gs.tree_logdet(T.diag, T.edge, T.parent), gs.tree_solve(T.diag, T.edge, T.parent, rhs)],
           [gs.tree_logdet(T.diag.cuda(), T.edge.cuda(), T.parent),
            gs.tree_solve(T.diag.cuda(), T.edge.cuda(), T.parent, rhs.cuda())])
    assert gs.sample_gaussian_tree(T.diag.cuda(), T.edge.cuda(), T.parent, 2).device.type == "cuda"
    C = gs.random_spd_chain(5, 2, seed=0, diag_load=1.0)
    _match(gs.selected_inverse_chain(C.diag, C.lower), gs.selected_inverse_chain(C.diag.cuda(), C.lower.cuda()))
    S = gs.random_spd_star(4, 2, seed=0)
    _match(gs.selected_inverse_star(S.center, S.leaf_diag, S.coupling),
           gs.selected_inverse_star(S.center.cuda(), S.leaf_diag.cuda(), S.coupling.cuda()))


def test_nonsymmetric_kernels_on_cuda():
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[1, 2, 3, 4, 5, 5], [0, 1, 0, 3, 4, 2]])
    lo = 0.3 * torch.randn(6, 2, 2, generator=gen, dtype=F64)
    up = 0.3 * torch.randn(6, 2, 2, generator=gen, dtype=F64)
    dg = 0.2 * torch.randn(6, 2, 2, generator=gen, dtype=F64) + 3 * torch.eye(2, dtype=F64)
    rhs = torch.randn(6, 2, dtype=F64)
    _match(gs.selected_inverse_nonsym_junction(dg, ei, lo, up),
           gs.selected_inverse_nonsym_junction(dg.cuda(), ei, lo.cuda(), up.cuda()))
    for transpose in (False, True):
        _match([gs.nonsym_junction_solve(dg, ei, lo, up, rhs, transpose=transpose)],
               [gs.nonsym_junction_solve(dg.cuda(), ei, lo.cuda(), up.cuda(), rhs.cuda(), transpose=transpose)])
    fn = lambda d, a, b: gs.selinv_nonsym_junction_analytic(d, ei, a, b)  # noqa: E731
    _match(_grads(fn, (dg, lo, up), "cpu"), _grads(fn, (dg, lo, up), "cuda"))
    parent = [-1, 0, 0, 1, 1, 2]
    _match(gs.selected_inverse_nonsym_tree(dg, lo, up, parent),
           gs.selected_inverse_nonsym_tree(dg.cuda(), lo.cuda(), up.cuda(), parent))
    M = gs.random_nonsym_bidiag(5, 2, seed=0)
    _match(gs.selected_inverse_bidiag(M.diag, M.lower), gs.selected_inverse_bidiag(M.diag.cuda(), M.lower.cuda()))
    a = (0.3 * torch.randn(6, 6, generator=gen, dtype=F64)).tril(-1)
    _match([gs.selected_inverse_tril(a)], [gs.selected_inverse_tril(a.cuda())])


def test_gmrf_helpers_on_cuda():
    y = torch.randn(9, 1, dtype=F64)
    cpu = gs.grid_gmrf_precision(3, 3, torch.tensor(1.0, dtype=F64), 0.5)
    gpu = gs.grid_gmrf_precision(3, 3, torch.tensor(1.0, dtype=F64, device="cuda"), 0.5)
    _match([gs.junction_marginal_log_likelihood(*cpu, y, 0.3),
            gs.junction_posterior_marginal_variances(*cpu, 0.3)],
           [gs.junction_marginal_log_likelihood(*gpu, y.cuda(), 0.3),
            gs.junction_posterior_marginal_variances(*gpu, 0.3)])
    for fit in (gs.fit_grid_marginal_likelihood, ):
        a, b = fit(3, 3, y, steps=3), fit(3, 3, y.cuda(), steps=3)
        assert abs(a["kappa"] - b["kappa"]) < 1e-8 and abs(a["sigma2"] - b["sigma2"]) < 1e-8
    parent = [-1, 0, 0, 1, 1, 2, 2, 3, 3]
    a, b = gs.fit_marginal_likelihood(parent, y, steps=3), gs.fit_marginal_likelihood(parent, y.cuda(), steps=3)
    assert abs(a["a"] - b["a"]) < 1e-8 and abs(a["sigma2"] - b["sigma2"]) < 1e-8
