"""Double-backward / Hessian-vector-product gates (B1).

Second-order autograd (needed for Laplace approximation / natural-gradient uses) comes
*for free* through the functional kernels, with no hand-written second-order rule:

* the **junction** selected inverse / solve / log-det are written functionally, so
  ``gradgradcheck`` passes directly;
* the **tree** selected inverse double-backs through its *level-set batched* functional
  forward (``selected_inverse_tree(batched=True)``); the per-node reference loop and the
  memory-lean analytic ``selinv_tree`` backward are **first-order only** (in-place block
  writes / a once-differentiable analytic adjoint) -- use the batched functional forward
  when HVPs are required.

* the hand-written analytic backwards (``selinv_tree``, ``selinv_junction_analytic``,
  ``selinv_nonsym_junction_analytic``) read saved factors that are not connected to the
  inputs. 0.4.0 let autograd differentiate them anyway and returned Hessian-vector products
  off by 60%; they now refuse every ``create_graph=True`` request.

These tests pin those facts so a regression (or a future manual second-order rule) is visible.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    BlockStar,
    BlockTree,
    BlockTridiag,
    grid_edges,
    junction_logdet,
    junction_solve,
    nonsym_junction_solve,
    random_nonsym_bidiag,
    random_spd_chain,
    random_spd_graph,
    random_spd_star,
    random_spd_tree,
    selected_inverse_chain,
    selected_inverse_nonsym_tree,
    selected_inverse_star,
    selected_inverse_tree,
    selinv_bidiag,
    selinv_junction,
    selinv_junction_analytic,
    selinv_nonsym_junction,
    selinv_nonsym_junction_analytic,
    selinv_tree,
    selinv_tril,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Junction: functional => double-backward works for selinv / solve / logdet.
# --------------------------------------------------------------------------- #
def _grid_mat(s, b, seed):
    n = s * s
    ei = grid_edges(s, s)
    return n, ei, random_spd_graph(n, ei, b, seed=seed, diag_load=1.0)


def test_junction_selinv_gradgradcheck():
    n, ei, mat = _grid_mat(3, 2, seed=2)
    d = mat.diag.clone().requires_grad_(True)
    v = mat.edge_val.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda d, v: selinv_junction(d, ei, v)[0], (d, v), atol=1e-4, rtol=1e-3
    )


def test_junction_solve_gradgradcheck():
    n, ei, mat = _grid_mat(3, 2, seed=3)
    b = torch.randn(n, 2, dtype=torch.float64)
    d = mat.diag.clone().requires_grad_(True)
    v = mat.edge_val.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda d, v: junction_solve(d, ei, v, b), (d, v), atol=1e-4, rtol=1e-3
    )


def test_junction_logdet_gradgradcheck():
    n, ei, mat = _grid_mat(3, 2, seed=4)
    d = mat.diag.clone().requires_grad_(True)
    v = mat.edge_val.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda d, v: junction_logdet(d, ei, v), (d, v), atol=1e-4, rtol=1e-3
    )


def test_junction_hvp_matches_finite_difference():
    """An explicit HVP through the junction solve matches a finite-difference of the gradient.

    HVP(v) = d/dε ∇f(θ + ε v)|_{0}; the create-graph autograd HVP must match the central
    difference of the first gradient along ``v`` (validates the second-order path end to end,
    in the kernel's own parametrization so there is no symmetrization mismatch).
    """
    n, ei, mat = _grid_mat(3, 2, seed=5)
    b = torch.randn(n, 2, dtype=torch.float64)
    # Symmetric perturbation directions (the kernel reads diag symmetrically).
    vec_v = torch.randn_like(mat.edge_val)
    raw = torch.randn_like(mat.diag)
    vec_d = 0.5 * (raw + raw.mT)

    def loss(d, v):
        return (junction_solve(d, ei, v, b) ** 2).sum()

    def grad_at(d0, v0):
        d = d0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        return torch.autograd.grad(loss(d, v), (d, v))

    # Autograd HVP.
    d = mat.diag.clone().requires_grad_(True)
    v = mat.edge_val.clone().requires_grad_(True)
    g = torch.autograd.grad(loss(d, v), (d, v), create_graph=True)
    dot = (g[0] * vec_d).sum() + (g[1] * vec_v).sum()
    hvp_d, hvp_v = torch.autograd.grad(dot, (d, v))

    # Central finite difference of the gradient.
    h = 1e-6
    gp = grad_at(mat.diag + h * vec_d, mat.edge_val + h * vec_v)
    gm = grad_at(mat.diag - h * vec_d, mat.edge_val - h * vec_v)
    fd_d = (gp[0] - gm[0]) / (2 * h)
    fd_v = (gp[1] - gm[1]) / (2 * h)
    assert torch.allclose(hvp_d, fd_d, atol=1e-5), (hvp_d - fd_d).abs().max()
    assert torch.allclose(hvp_v, fd_v, atol=1e-5), (hvp_v - fd_v).abs().max()


# --------------------------------------------------------------------------- #
# Tree: the level-set batched functional forward double-backs.
# --------------------------------------------------------------------------- #
def test_tree_batched_functional_gradgradcheck():
    bt = random_spd_tree(6, 2, seed=1, kind="random", diag_load=1.0)
    d = bt.diag.clone().requires_grad_(True)
    e = bt.edge.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda d, e: selected_inverse_tree(d, e, bt.parent, batched=True),
        (d, e), atol=1e-4, rtol=1e-3,
    )


def test_tree_per_node_loop_is_first_order_only():
    """The per-node reference loop uses in-place block writes => no second order (documented)."""
    bt = random_spd_tree(6, 2, seed=1, kind="random", diag_load=1.0)
    d = bt.diag.clone().requires_grad_(True)
    e = bt.edge.clone().requires_grad_(True)
    with pytest.raises(RuntimeError):
        torch.autograd.gradgradcheck(
            lambda d, e: selected_inverse_tree(d, e, bt.parent, batched=False),
            (d, e), atol=1e-4, rtol=1e-3,
        )


# --------------------------------------------------------------------------- #
# Analytic first-order backwards refuse every second-derivative request.
# --------------------------------------------------------------------------- #
def _nonsym_problem():
    gen = torch.Generator().manual_seed(0)
    pairs = [(1, 0), (2, 1), (3, 0), (4, 3), (5, 4), (5, 2)]
    ei = torch.tensor(list(zip(*pairs)))
    lower = 0.3 * torch.randn(6, 2, 2, generator=gen, dtype=torch.float64)
    upper = 0.3 * torch.randn(6, 2, 2, generator=gen, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    diag = 0.2 * torch.randn(6, 2, 2, generator=gen, dtype=torch.float64) + 3 * eye
    return diag, ei, lower, upper


def _analytic_cases():
    bt = random_spd_tree(6, 2, seed=1, kind="random", diag_load=1.0)
    _n, ei, mat = _grid_mat(2, 2, seed=3)
    diag, nei, lower, upper = _nonsym_problem()
    return [
        (lambda d, e: selinv_tree(d, e, bt.parent)[0], (bt.diag, bt.edge)),
        (lambda d, e: selinv_tree(d, e, bt.parent, batched=True)[0], (bt.diag, bt.edge)),
        (lambda d, v: selinv_junction_analytic(d, ei, v)[0], (mat.diag, mat.edge_val)),
        (lambda d, lo, up: selinv_nonsym_junction_analytic(d, nei, lo, up)[0], (diag, lower, upper)),
    ]


@pytest.mark.parametrize("case", range(4))
def test_analytic_backward_refuses_second_derivatives(case):
    fn, inputs = _analytic_cases()[case]
    xs = [x.clone().requires_grad_(True) for x in inputs]
    # first order still works
    grads = torch.autograd.grad((fn(*xs) ** 2).sum(), xs)
    assert all(torch.isfinite(g).all() for g in grads)
    # every way of asking for a differentiable gradient raises instead of returning a value
    # (with once_differentiable alone, functional.hessian silently returned zeros)
    with pytest.raises(RuntimeError, match="first-order analytic backward"):
        torch.autograd.grad((fn(*xs) ** 2).sum(), xs, create_graph=True)
    with pytest.raises(RuntimeError, match="first-order analytic backward"):
        torch.autograd.functional.hessian(lambda *z: (fn(*z) ** 2).sum(), tuple(inputs))
    with pytest.raises(RuntimeError, match="first-order analytic backward"):
        torch.autograd.gradgradcheck(fn, xs)


# --------------------------------------------------------------------------- #
# Functional paths that do support second derivatives: dense HVP / gradgradcheck.
# --------------------------------------------------------------------------- #
def _sym(x):
    return 0.5 * (x + x.mT)


def _hvp(loss, inputs):
    xs = [x.clone().requires_grad_(True) for x in inputs]
    grads = torch.autograd.grad(loss(*xs), xs, create_graph=True)
    return torch.autograd.grad(sum(g.sum() for g in grads), xs)


def _assert_hvp_close(kernel_loss, dense_loss, inputs):
    for got, want in zip(_hvp(kernel_loss, inputs), _hvp(dense_loss, inputs)):
        assert float((got - want).abs().max()) <= 1e-10 * float(want.abs().max())


def _pattern_sum_sq(G, blocks, b):
    return sum(G[i * b:(i + 1) * b, j * b:(j + 1) * b].square().sum() for i, j in blocks)


def test_chain_and_star_hvp_match_dense():
    # Diagonal blocks are symmetric by contract, so both sides see sym(d).
    c = random_spd_chain(4, 2, seed=0, diag_load=1.0)

    def chain_kernel(d, lo):
        Gd, Gl = selected_inverse_chain(_sym(d), lo)
        return Gd.square().sum() + Gl.square().sum()

    def chain_dense(d, lo):
        G = torch.linalg.inv(BlockTridiag(diag=_sym(d), lower=lo).to_dense())
        return _pattern_sum_sq(G, [(i, i) for i in range(4)] + [(i + 1, i) for i in range(3)], 2)

    _assert_hvp_close(chain_kernel, chain_dense, (c.diag, c.lower))
    st = random_spd_star(3, 2, seed=0)

    def star_kernel(cen, leaves, cpl):
        Gc, Gl, Gw = selected_inverse_star(_sym(cen), _sym(leaves), cpl)
        return Gc.square().sum() + Gl.square().sum() + Gw.square().sum()

    def star_dense(cen, leaves, cpl):
        A = BlockStar(center=_sym(cen), leaf_diag=_sym(leaves), coupling=cpl).to_dense()
        blocks = [(0, 0)] + [(j, j) for j in range(1, 4)] + [(0, j) for j in range(1, 4)]
        return _pattern_sum_sq(torch.linalg.inv(A), blocks, 2)

    _assert_hvp_close(star_kernel, star_dense, (st.center, st.leaf_diag, st.coupling))


def test_tree_batched_hvp_matches_dense():
    bt = random_spd_tree(6, 2, seed=1, kind="random", diag_load=1.0)
    plist = bt.parent.tolist()
    pairs = [(v, v) for v in range(6)] + [(plist[v], v) for v in range(6) if plist[v] >= 0]

    def kernel(d, e):
        Gd, Ge = selected_inverse_tree(_sym(d), e, bt.parent, batched=True)
        return Gd.square().sum() + Ge.square().sum()

    def dense(d, e):
        G = torch.linalg.inv(BlockTree(diag=_sym(d), edge=e, parent=bt.parent).to_dense())
        return _pattern_sum_sq(G, pairs, 2)

    _assert_hvp_close(kernel, dense, (bt.diag, bt.edge))


def test_nonsym_functional_paths_gradgradcheck():
    diag, ei, lower, upper = _nonsym_problem()
    xs = [t.clone().requires_grad_(True) for t in (diag, lower, upper)]
    rhs = torch.randn(6, 2, dtype=torch.float64)
    assert torch.autograd.gradgradcheck(lambda d, lo, up: selinv_nonsym_junction(d, ei, lo, up)[0], xs)
    for transpose in (False, True):
        assert torch.autograd.gradgradcheck(
            lambda d, lo, up: nonsym_junction_solve(d, ei, lo, up, rhs, transpose=transpose), xs)
    parent = [-1, 0, 0, 1]
    gen = torch.Generator().manual_seed(1)
    eye = torch.eye(2, dtype=torch.float64)
    td = 0.2 * torch.randn(4, 2, 2, generator=gen, dtype=torch.float64) + 3 * eye
    pc = 0.3 * torch.randn(4, 2, 2, generator=gen, dtype=torch.float64)
    cp = 0.3 * torch.randn(4, 2, 2, generator=gen, dtype=torch.float64)
    ts = [t.clone().requires_grad_(True) for t in (td, pc, cp)]
    assert torch.autograd.gradgradcheck(lambda d, a, b: selected_inverse_nonsym_tree(d, a, b, parent), ts)


def test_bidiag_and_tril_custom_backwards_double_back():
    m = random_nonsym_bidiag(5, 2, seed=0)
    xs = [m.diag.clone().requires_grad_(True), m.lower.clone().requires_grad_(True)]
    assert torch.autograd.gradgradcheck(lambda d, lo: selinv_bidiag(d, lo), xs)
    a = (0.3 * torch.randn(6, 6, dtype=torch.float64)).tril(-1).requires_grad_(True)
    assert torch.autograd.gradgradcheck(lambda x: selinv_tril(x), (a,))
