"""Gates for the hand-written analytic backward of the junction selected inverses.

``selinv_junction_analytic`` / ``selinv_nonsym_junction_analytic`` (``junction_autodiff.py``)
run the explicit reverse two-sweep clique recurrence of ``docs/derivations.md`` §8.4 / §10.3 --
the tape-free filled-pattern analogue of the tree analytic backward. The gates: identical
forward and machine-precision gradient agreement with the autograd-through-the-functional-forward
path (the validated oracle), ``gradcheck``, the reduction of the non-symmetric backward to the
symmetric one on symmetric input, leading batch dims, the zero-fill tree case, and the tape-free
property (a constant-size autograd graph independent of ``n``).
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    grid_edges,
    random_spd_graph,
    selinv_junction,
    selinv_junction_analytic,
    selinv_nonsym_junction,
    selinv_nonsym_junction_analytic,
)
from gabp_sparse_inv.junction import _normalize_edge_index

torch.manual_seed(0)


def _nonsym_graph(n, edges, b, *, seed, diag_load=4.0, off=0.3):
    g = torch.Generator().manual_seed(seed)
    pairs = sorted({(max(int(edges[0, k]), int(edges[1, k])), min(int(edges[0, k]), int(edges[1, k])))
                    for k in range(edges.shape[1])})
    ei = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    m = ei.shape[1]
    eye = torch.eye(b, dtype=torch.float64)
    diag = diag_load * eye + 0.1 * torch.randn(n, b, b, generator=g, dtype=torch.float64)
    el = off * torch.randn(m, b, b, generator=g, dtype=torch.float64)
    eu = off * torch.randn(m, b, b, generator=g, dtype=torch.float64)
    return diag, ei, el, eu


def _grads_sym(fn, diag, ei, ev, bGd, bGl):
    d = diag.clone().requires_grad_(True)
    e = ev.clone().requires_grad_(True)
    Gd, _Si, Gl = fn(d, ei, e)
    ((Gd * bGd).sum() + (Gl * bGl).sum()).backward()
    return Gd.detach(), Gl.detach(), d.grad, e.grad


def _grads_nonsym(fn, diag, ei, el, eu, bGd, bGl, bGu):
    d = diag.clone().requires_grad_(True)
    a = el.clone().requires_grad_(True)
    c = eu.clone().requires_grad_(True)
    Gd, _Si, Gl, Gu = fn(d, ei, a, c)
    ((Gd * bGd).sum() + (Gl * bGl).sum() + (Gu * bGu).sum()).backward()
    return Gd.detach(), d.grad, a.grad, c.grad


def _count_graph_nodes(*outs):
    seen, stack = set(), [o.grad_fn for o in outs if o.grad_fn is not None]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for nf, _ in fn.next_functions:
            if nf is not None:
                stack.append(nf)
    return len(seen)


# --------------------------------------------------------------------------- #
# Analytic backward == autograd-through-the-functional-forward (machine precision).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,seed", [(3, 3, 1), (4, 3, 2), (2, 5, 3)])
def test_sym_analytic_matches_functional_grid(rows, cols, seed):
    sp = random_spd_graph(rows * cols, grid_edges(rows, cols), 2, seed=seed, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    Gd0, _Si, Gl0 = selinv_junction(sp.diag, ei, sp.edge_val)
    bGd, bGl = torch.randn_like(Gd0), torch.randn_like(Gl0)
    Gda, Gla, ad, ae = _grads_sym(selinv_junction_analytic, sp.diag, ei, sp.edge_val, bGd, bGl)
    Gdf, Glf, fd, fe = _grads_sym(selinv_junction, sp.diag, ei, sp.edge_val, bGd, bGl)
    assert torch.equal(Gda, Gdf) and torch.equal(Gla, Glf)        # identical forward
    assert torch.allclose(ad, fd, atol=1e-10) and torch.allclose(ae, fe, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sym_analytic_matches_functional_random_graph(seed):
    n, b = 10, 2
    gen = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    edges = pairs or [(1, 0)]
    sp = random_spd_graph(n, edges, b, seed=seed, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    Gd0, _Si, Gl0 = selinv_junction(sp.diag, ei, sp.edge_val)
    bGd, bGl = torch.randn_like(Gd0), torch.randn_like(Gl0)
    _, _, ad, ae = _grads_sym(selinv_junction_analytic, sp.diag, ei, sp.edge_val, bGd, bGl)
    _, _, fd, fe = _grads_sym(selinv_junction, sp.diag, ei, sp.edge_val, bGd, bGl)
    assert torch.allclose(ad, fd, atol=1e-10) and torch.allclose(ae, fe, atol=1e-10)


@pytest.mark.parametrize("rows,cols,seed", [(3, 3, 1), (2, 5, 2), (4, 3, 4)])
def test_nonsym_analytic_matches_functional_grid(rows, cols, seed):
    diag, ei, el, eu = _nonsym_graph(rows * cols, grid_edges(rows, cols), 2, seed=seed)
    Gd0, _Si, Gl0, Gu0 = selinv_nonsym_junction(diag, ei, el, eu)
    bGd, bGl, bGu = torch.randn_like(Gd0), torch.randn_like(Gl0), torch.randn_like(Gu0)
    Ga, ad, al, au = _grads_nonsym(selinv_nonsym_junction_analytic, diag, ei, el, eu, bGd, bGl, bGu)
    Gf, fd, fl, fu = _grads_nonsym(selinv_nonsym_junction, diag, ei, el, eu, bGd, bGl, bGu)
    assert torch.equal(Ga, Gf)
    assert torch.allclose(ad, fd, atol=1e-10)
    assert torch.allclose(al, fl, atol=1e-10) and torch.allclose(au, fu, atol=1e-10)


# --------------------------------------------------------------------------- #
# gradcheck (fp64) against finite differences.
# --------------------------------------------------------------------------- #
def test_sym_gradcheck():
    sp = random_spd_graph(4, grid_edges(2, 2), 2, seed=7, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    d = sp.diag.clone().requires_grad_(True)
    e = sp.edge_val.clone().requires_grad_(True)

    def f(d, e):
        Gd, _Si, Gl = selinv_junction_analytic(d, ei, e)
        return torch.cat([Gd.reshape(-1), Gl.reshape(-1)])

    assert torch.autograd.gradcheck(f, (d, e), atol=1e-6, rtol=1e-4)


def test_nonsym_gradcheck():
    diag, ei, el, eu = _nonsym_graph(4, grid_edges(2, 2), 2, seed=9)
    d = diag.clone().requires_grad_(True)
    a = el.clone().requires_grad_(True)
    c = eu.clone().requires_grad_(True)

    def f(d, a, c):
        Gd, _Si, Gl, Gu = selinv_nonsym_junction_analytic(d, ei, a, c)
        return torch.cat([Gd.reshape(-1), Gl.reshape(-1), Gu.reshape(-1)])

    assert torch.autograd.gradcheck(f, (d, a, c), atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Reductions and batching.
# --------------------------------------------------------------------------- #
def _sym(x):
    return 0.5 * (x + x.mT)


def test_nonsym_analytic_reduces_to_symmetric():
    # On symmetric input (A_{j,i} = A_{i,j}^T) the non-symmetric kernel reproduces the symmetric
    # one. Forward: G_diag/G_lower equal, G_upper = G_lower^T. Backward (§8.1 convention): the
    # symmetric kernel symmetrizes the diagonal input and folds the two edge slots A_{i,j},
    # A_{j,i}=A_{i,j}^T into one parameter, so its grads are sym(d.grad) and a.grad + c.grad^T.
    sp = random_spd_graph(9, grid_edges(3, 3), 2, seed=6, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    Gd0, _Si, Gl0 = selinv_junction(sp.diag, ei, sp.edge_val)
    bGd, bGl = torch.randn_like(Gd0), torch.randn_like(Gl0)
    Gds, Gls, ad_s, ae_s = _grads_sym(selinv_junction_analytic, sp.diag, ei, sp.edge_val, bGd, bGl)

    d = sp.diag.clone().requires_grad_(True)
    a = sp.edge_val.clone().requires_grad_(True)
    c = sp.edge_val.mT.clone().requires_grad_(True)
    Gd, _Si, Gl, Gu = selinv_nonsym_junction_analytic(d, ei, a, c)
    assert torch.allclose(Gd, Gds, atol=1e-10) and torch.allclose(Gl, Gls, atol=1e-10)
    assert torch.allclose(Gu, Gls.mT, atol=1e-10)                # G_{j,i} = G_{i,j}^T
    ((Gd * bGd).sum() + (Gl * bGl).sum()).backward()             # bGu = 0: same scalar as the sym loss
    assert torch.allclose(_sym(d.grad), ad_s, atol=1e-10)
    assert torch.allclose(a.grad + c.grad.mT, ae_s, atol=1e-10)


def test_sym_batch_matches_per_item():
    sp = random_spd_graph(6, grid_edges(2, 3), 2, seed=8, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    batch = 4
    dB = sp.diag.expand(batch, *sp.diag.shape).clone() + 0.01 * torch.randn(batch, *sp.diag.shape)
    eB = sp.edge_val.expand(batch, *sp.edge_val.shape).clone() + 0.01 * torch.randn(batch, *sp.edge_val.shape)
    dBr = dB.clone().requires_grad_(True)
    eBr = eB.clone().requires_grad_(True)
    Gd, _Si, Gl = selinv_junction_analytic(dBr, ei, eBr)
    bGd, bGl = torch.randn_like(Gd), torch.randn_like(Gl)
    ((Gd * bGd).sum() + (Gl * bGl).sum()).backward()
    for k in range(batch):
        dk = dB[k].clone().requires_grad_(True)
        ek = eB[k].clone().requires_grad_(True)
        Gdk, _Si, Glk = selinv_junction_analytic(dk, ei, ek)
        ((Gdk * bGd[k]).sum() + (Glk * bGl[k]).sum()).backward()
        assert torch.allclose(dBr.grad[k], dk.grad, atol=1e-12)
        assert torch.allclose(eBr.grad[k], ek.grad, atol=1e-12)


def test_zero_fill_tree_matches_functional():
    # A path is a tree (zero fill): the analytic backward must still match the functional path.
    n, b = 8, 2
    edges = torch.tensor([[i + 1 for i in range(n - 1)], [i for i in range(n - 1)]], dtype=torch.long)
    sp = random_spd_graph(n, edges, b, seed=3, diag_load=1.0)
    ei = _normalize_edge_index(sp.edge_index)
    Gd0, Si, Gl0 = selinv_junction(sp.diag, ei, sp.edge_val)
    assert Si.shape[1] == ei.shape[1]                            # zero fill
    bGd, bGl = torch.randn_like(Gd0), torch.randn_like(Gl0)
    _, _, ad, ae = _grads_sym(selinv_junction_analytic, sp.diag, ei, sp.edge_val, bGd, bGl)
    _, _, fd, fe = _grads_sym(selinv_junction, sp.diag, ei, sp.edge_val, bGd, bGl)
    assert torch.allclose(ad, fd, atol=1e-10) and torch.allclose(ae, fe, atol=1e-10)


def test_single_node():
    diag = (2.0 * torch.eye(2, dtype=torch.float64)).reshape(1, 2, 2)
    ei = torch.zeros((2, 0), dtype=torch.long)
    ev = torch.zeros(0, 2, 2, dtype=torch.float64)
    d = diag.clone().requires_grad_(True)
    Gd, _Si, _Gl = selinv_junction_analytic(d, ei, ev)
    Gd.pow(2).sum().backward()
    # d/dA (A^-1)^2 summed: cotangent on A_vv = -G (2G) G with G = A^-1 = 0.5 I.
    assert d.grad is not None and float(d.grad.abs().max()) > 0


# --------------------------------------------------------------------------- #
# Tape-free: the analytic backward keeps a constant-size autograd graph.
# --------------------------------------------------------------------------- #
def test_tape_free_constant_graph():
    sizes = [(3, 3), (5, 5), (7, 7)]
    functional_counts = []
    for rows, cols in sizes:
        sp = random_spd_graph(rows * cols, grid_edges(rows, cols), 1, seed=1, diag_load=1.0)
        ei = _normalize_edge_index(sp.edge_index)
        d = sp.diag.clone().requires_grad_(True)
        e = sp.edge_val.clone().requires_grad_(True)
        Gd, _Si, Gl = selinv_junction(d, ei, e)
        functional_counts.append(_count_graph_nodes(Gd, Gl))
        d2 = sp.diag.clone().requires_grad_(True)
        e2 = sp.edge_val.clone().requires_grad_(True)
        Gd2, _Si, Gl2 = selinv_junction_analytic(d2, ei, e2)
        # The analytic Function is a single autograd node regardless of n (+ the two input leaves).
        assert _count_graph_nodes(Gd2, Gl2) == 3
    # The functional tape grows with n; the analytic graph does not.
    assert functional_counts[0] < functional_counts[-1]
