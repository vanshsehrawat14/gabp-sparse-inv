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

These tests pin those facts so a regression (or a future manual second-order rule) is visible.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    grid_edges,
    junction_logdet,
    junction_solve,
    random_spd_graph,
    random_spd_tree,
    selected_inverse_tree,
    selinv_junction,
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
