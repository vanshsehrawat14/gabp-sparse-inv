"""Validation gates for the differentiable selected inverse (``selinv_tree``).

The analytic two-pass backward of ``docs/derivations.md`` §8.3 is validated against
two independent autograd oracles, each link to ~1e-9 in fp64:

* ``torch.autograd.gradcheck`` on controlled-conditioning topologies;
* autograd through an out-of-place *functional twin* of the (3)-(6) recurrence
  (``_selinv_functional``) -- the structured O(n) oracle; the shipped loop
  ``selected_inverse_tree`` is itself *not* autograd-traceable (in-place buffers),
  which is part of why the hand-written backward exists;
* autograd through a *parametrized dense* matrix (``to_dense -> inv -> extract``),
  which is the convention-faithful oracle for the on-pattern cotangent
  ``bar_A|_S = -[A^{-1} bar_G A^{-1}]_S`` (§8.1: the dense matrix is symmetrized to
  match the kernel's pivot-symmetrization convention).

A minimal end-to-end learnability test closes the milestone: gradient descent through
``selinv_tree`` recovers a scalar parameter from a target marginal-variance trace.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import random_spd_tree, selinv_tree
from gabp_sparse_inv.layout import _as_parent_tensor, tree_orders

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _selinv_functional(diag, edge, parent):
    """Out-of-place, autograd-traceable reference of the (3)-(6) recurrence.

    The shipped loop ``selected_inverse_tree`` writes into preallocated buffers and is
    *not* autograd-traceable (in-place version conflicts) -- which is part of why the
    analytic backward exists. This functional twin matches it block-for-block and
    serves as the structured autograd oracle (the dense oracle is the other link).
    """
    p = _as_parent_tensor(parent)
    plist = p.tolist()
    root, children, collect = tree_orders(p)
    n = diag.shape[-3]
    sym = lambda x: 0.5 * (x + x.mT)  # noqa: E731

    P = [diag[..., v, :, :] for v in range(n)]
    Dinv, ell = {}, {}
    for v in collect:
        Dinv[v] = torch.linalg.inv(sym(P[v]))
        if v != root:
            U_v = edge[..., v, :, :]
            ell[v] = U_v @ Dinv[v]
            P[plist[v]] = P[plist[v]] - ell[v] @ U_v.mT

    Gd = [None] * n
    Ge = [torch.zeros_like(diag[..., 0, :, :]) for _ in range(n)]
    Gd[root] = sym(Dinv[root])
    for v in reversed(collect):
        if v == root:
            continue
        Gpp = Gd[plist[v]]
        Ge[v] = -(Gpp @ ell[v])
        Gd[v] = sym(Dinv[v] + ell[v].mT @ Gpp @ ell[v])
    return torch.stack(Gd, dim=-3), torch.stack(Ge, dim=-3)



def _dense_selinv_from_blocks(diag, edge, plist, *, symmetrize=True):
    """Differentiable dense oracle: assemble A from (diag, edge), invert, extract.

    The assembled matrix is symmetrized (the off-diagonal edge blocks are already
    placed symmetrically; this symmetrizes the diagonal blocks) so the oracle matches
    the kernel's pivot-symmetrization convention (derivations.md §6/§8.1).
    """
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    mat = diag.new_zeros((*diag.shape[:-3], N, N))
    for v in range(n):
        rv = slice(v * b, (v + 1) * b)
        mat[..., rv, rv] = diag[..., v, :, :]
        p = plist[v]
        if p != -1:
            rp = slice(p * b, (p + 1) * b)
            mat[..., rp, rv] = edge[..., v, :, :]
            mat[..., rv, rp] = edge[..., v, :, :].mT
    if symmetrize:
        mat = 0.5 * (mat + mat.mT)
    G = torch.linalg.inv(mat)
    Gd = torch.stack(
        [G[..., v * b:(v + 1) * b, v * b:(v + 1) * b] for v in range(n)], dim=-3
    )
    Ge_blocks = []
    for v in range(n):
        p = plist[v]
        if p == -1:
            Ge_blocks.append(torch.zeros_like(Gd[..., 0, :, :]))
        else:
            Ge_blocks.append(G[..., p * b:(p + 1) * b, v * b:(v + 1) * b])
    Ge = torch.stack(Ge_blocks, dim=-3)
    return Gd, Ge


def _grads(fn, diag0, edge0, parent, loss_fn):
    """Run ``fn(diag, edge, parent) -> (G_diag, G_edge)``, backprop ``loss_fn``."""
    diag = diag0.clone().requires_grad_(True)
    edge = edge0.clone().requires_grad_(True)
    Gd, Ge = fn(diag, edge, parent)
    loss_fn(Gd, Ge).backward()
    gd = diag.grad if diag.grad is not None else torch.zeros_like(diag)
    ge = edge.grad if edge.grad is not None else torch.zeros_like(edge)
    return Gd.detach(), Ge.detach(), gd.clone(), ge.clone()


# A small bank of on-pattern losses (each induces a different symmetric bar_G).
def _loss_logdet(Gd, Ge):
    # d(log det A) cotangent is bar_G = I on diagonals, 0 on edges.
    eye = torch.eye(Gd.shape[-1], dtype=Gd.dtype)
    return (Gd * eye).sum()


def _loss_marg_var(Gd, Ge):
    return torch.diagonal(Gd, dim1=-2, dim2=-1).sum()


def _loss_selected_frob(Gd, Ge):
    return (Gd * Gd).sum() + 2.0 * (Ge * Ge).sum()


def _make_random_loss(n, b, seed):
    g = torch.Generator().manual_seed(seed)
    wd = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    we = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    return lambda Gd, Ge: (Gd * wd).sum() + (Ge * we).sum()


LOSSES = [_loss_logdet, _loss_marg_var, _loss_selected_frob]


# --------------------------------------------------------------------------- #
# 1. gradcheck on controlled-conditioning topologies (B1: not random).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "star", "balanced"])
@pytest.mark.parametrize("n,b", [(6, 2), (7, 3), (9, 2)])
def test_gradcheck_controlled(kind, n, b):
    bt = random_spd_tree(n, b, seed=n * 10 + b, diag_load=5.0, kind=kind)
    diag = bt.diag.clone().double().requires_grad_(True)
    edge = bt.edge.clone().double().requires_grad_(True)
    parent = bt.parent

    def f(d, e):
        return selinv_tree(d, e, parent)

    assert torch.autograd.gradcheck(f, (diag, edge), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("kind", ["path", "balanced"])
def test_gradcheck_batched(kind):
    n, b = 9, 2
    bt = random_spd_tree(n, b, seed=n + b, diag_load=5.0, kind=kind)
    diag = bt.diag.clone().double().requires_grad_(True)
    edge = bt.edge.clone().double().requires_grad_(True)
    parent = bt.parent
    assert torch.autograd.gradcheck(
        lambda d, e: selinv_tree(d, e, parent, batched=True),
        (diag, edge), atol=1e-6, rtol=1e-4,
    )


# --------------------------------------------------------------------------- #
# 1b. Level-set batched == per-node loop, forward AND backward, block-for-block.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "star", "balanced", "random"])
@pytest.mark.parametrize("n,b", [(1, 3), (8, 2), (20, 3)])
def test_batched_equals_loop(kind, n, b):
    bt = random_spd_tree(n, b, seed=n * 5 + b, diag_load=3.0, kind=kind)
    loss_fn = _make_random_loss(n, b, seed=n + b)
    Gd_l, Ge_l, gd_l, ge_l = _grads(
        lambda d, e, p: selinv_tree(d, e, p, batched=False),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    Gd_b, Ge_b, gd_b, ge_b = _grads(
        lambda d, e, p: selinv_tree(d, e, p, batched=True),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(Gd_l, Gd_b, atol=1e-12)
    assert torch.allclose(Ge_l, Ge_b, atol=1e-12)
    assert torch.allclose(gd_l, gd_b, atol=1e-12), (gd_l - gd_b).abs().max()
    assert torch.allclose(ge_l, ge_b, atol=1e-12), (ge_l - ge_b).abs().max()


# --------------------------------------------------------------------------- #
# 2. Analytic backward vs autograd through the reference loop (~1e-10).
#    The reference kernel is plain cholesky/solve/matmul, hence autograd-traceable.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "star", "balanced", "random"])
@pytest.mark.parametrize("loss_fn", LOSSES)
def test_analytic_vs_loop_autograd(kind, loss_fn):
    n, b = 10, 3
    bt = random_spd_tree(n, b, seed=hash((kind,)) % 9973, diag_load=4.0, kind=kind)
    Gd_a, Ge_a, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    Gd_l, Ge_l, gd_l, ge_l = _grads(
        lambda d, e, p: _selinv_functional(d, e, p),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(Gd_a, Gd_l, atol=1e-10)
    assert torch.allclose(Ge_a, Ge_l, atol=1e-10)
    assert torch.allclose(gd_a, gd_l, atol=1e-9), (gd_a - gd_l).abs().max()
    assert torch.allclose(ge_a, ge_l, atol=1e-9), (ge_a - ge_l).abs().max()


# --------------------------------------------------------------------------- #
# 3. Adjoint-restriction identity: analytic == autograd-through-parametrized-dense.
#    Closes the §8 theorem numerically for general bar_G (beyond log-det).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "star", "balanced", "random"])
def test_adjoint_restriction_vs_dense(kind):
    n, b = 8, 3
    bt = random_spd_tree(n, b, seed=n + b + len(kind), diag_load=4.0, kind=kind)
    plist = bt.parent.tolist()
    loss_fn = _make_random_loss(n, b, seed=7)

    _, _, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    _, _, gd_o, ge_o = _grads(
        lambda d, e, p: _dense_selinv_from_blocks(d, e, plist),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(gd_a, gd_o, atol=1e-9), (gd_a - gd_o).abs().max()
    assert torch.allclose(ge_a, ge_o, atol=1e-9), (ge_a - ge_o).abs().max()


# --------------------------------------------------------------------------- #
# 4. Leading batch dimensions.
# --------------------------------------------------------------------------- #
def test_batched_grads():
    n, b = 7, 2
    bt = random_spd_tree(n, b, seed=3, diag_load=4.0, kind="balanced", batch_shape=(4,))
    loss_fn = _make_random_loss(n, b, seed=11)
    # The random loss weights broadcast over the batch dim.
    _, _, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    _, _, gd_l, ge_l = _grads(
        lambda d, e, p: _selinv_functional(d, e, p),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert gd_a.shape == bt.diag.shape
    assert torch.allclose(gd_a, gd_l, atol=1e-9)
    assert torch.allclose(ge_a, ge_l, atol=1e-9)


# --------------------------------------------------------------------------- #
# 5. Edge cases: n=1, scalar b=1, explicit parent array.
# --------------------------------------------------------------------------- #
def test_single_node():
    bt = random_spd_tree(1, 3, seed=1, diag_load=3.0)
    loss_fn = _make_random_loss(1, 3, seed=2)
    _, _, gd_a, _ = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    _, _, gd_o, _ = _grads(
        lambda d, e, p: _dense_selinv_from_blocks(d, e, bt.parent.tolist()),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(gd_a, gd_o, atol=1e-9)


def test_scalar_blocks():
    bt = random_spd_tree(6, 1, seed=4, diag_load=3.0, kind="balanced")
    loss_fn = _make_random_loss(6, 1, seed=5)
    _, _, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    _, _, gd_o, ge_o = _grads(
        lambda d, e, p: _dense_selinv_from_blocks(d, e, bt.parent.tolist()),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(gd_a, gd_o, atol=1e-9)
    assert torch.allclose(ge_a, ge_o, atol=1e-9)


def test_explicit_parent():
    parent = [-1, 0, 0, 1, 1, 2]
    bt = random_spd_tree(6, 2, seed=8, diag_load=4.0, parent=parent)
    loss_fn = _make_random_loss(6, 2, seed=9)
    _, _, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), bt.diag, bt.edge, bt.parent, loss_fn
    )
    _, _, gd_o, ge_o = _grads(
        lambda d, e, p: _dense_selinv_from_blocks(d, e, parent),
        bt.diag, bt.edge, bt.parent, loss_fn,
    )
    assert torch.allclose(gd_a, gd_o, atol=1e-9)
    assert torch.allclose(ge_a, ge_o, atol=1e-9)


# --------------------------------------------------------------------------- #
# 6. fp32 path (looser tolerance).
# --------------------------------------------------------------------------- #
def test_fp32_grads_loose():
    n, b = 8, 3
    bt = random_spd_tree(n, b, seed=6, diag_load=5.0, kind="balanced")
    loss_fn = _loss_marg_var
    diag32, edge32 = bt.diag.float(), bt.edge.float()
    _, _, gd_a, ge_a = _grads(
        lambda d, e, p: selinv_tree(d, e, p), diag32, edge32, bt.parent, loss_fn
    )
    _, _, gd_l, ge_l = _grads(
        lambda d, e, p: _selinv_functional(d, e, p),
        diag32, edge32, bt.parent, loss_fn,
    )
    assert torch.allclose(gd_a, gd_l, atol=1e-4)
    assert torch.allclose(ge_a, ge_l, atol=1e-4)


# --------------------------------------------------------------------------- #
# 7. End-to-end learnability: recover a scalar diagonal-load parameter through
#    selinv_tree from a target sum-of-marginal-variances.
# --------------------------------------------------------------------------- #
def test_learnability_recovers_scalar():
    n, b = 12, 2
    base = random_spd_tree(n, b, seed=21, diag_load=0.0, kind="balanced")
    parent = base.parent
    eye = torch.eye(b, dtype=torch.float64).expand(n, b, b)

    def marg_var_trace(theta):
        diag = base.diag + theta * eye
        Gd, _ = selinv_tree(diag, base.edge, parent)
        return torch.diagonal(Gd, dim1=-2, dim2=-1).sum()

    theta_star = torch.tensor(3.0, dtype=torch.float64)
    target = marg_var_trace(theta_star).detach()

    theta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.2)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = (marg_var_trace(theta) - target) ** 2
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 1e-8, f"did not converge: final loss {losses[-1]:.2e}"
    assert abs(theta.item() - theta_star.item()) < 1e-2
