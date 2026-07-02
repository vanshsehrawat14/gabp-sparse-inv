"""Gates for the DEQ / fixed-point impact rung (``demos/deq_fixedpoint.py``).

The deep-equilibrium backward is a non-symmetric solve with the equilibrium Jacobian
``(I - J)``; this module shows it is (a) **exact** via the structured selected-inverse
factorization (matches a dense implicit-diff oracle and autograd through an unrolled solver to
machine precision) and (b) **robust as ``rho(J) -> 1``**, where the standard iterative
(Neumann) backward degrades like ``rho^K``. Both the affine equilibrium (autograd through
``nonsym_junction_solve``) and the nonlinear DEQ (custom IFT backward) are gated here.
"""

from __future__ import annotations

import torch

from gabp_sparse_inv import grid_edges
from gabp_sparse_inv.demos.deq_fixedpoint import (
    affine_fixed_point,
    assemble_blocks,
    backward_accuracy_sweep,
    deq_fixed_point,
    fit_affine_fixed_point,
    im_minus_jac_blocks,
    random_coupling,
    spmv,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Operator helpers: the sparse block mat-vec and the I - diag(s) W assembly.
# --------------------------------------------------------------------------- #
def test_spmv_matches_dense():
    ei = grid_edges(3, 3)
    n, b = 9, 2
    Wd, Wl, Wu = random_coupling(n, ei, b, seed=0, rho=0.7)
    A = assemble_blocks(Wd, ei, Wl, Wu)
    v = torch.randn(n, b, dtype=torch.float64)
    assert torch.allclose(spmv(Wd, ei, Wl, Wu, v).reshape(-1), A @ v.reshape(-1), atol=1e-12)
    assert torch.allclose(
        spmv(Wd, ei, Wl, Wu, v, transpose=True).reshape(-1), A.T @ v.reshape(-1), atol=1e-12
    )


def test_im_minus_jac_blocks_matches_dense():
    ei = grid_edges(2, 3)
    n, b = 6, 2
    Wd, Wl, Wu = random_coupling(n, ei, b, seed=2, rho=0.7)
    W = assemble_blocks(Wd, ei, Wl, Wu)
    # Affine case J = W.
    Ad, Al, Au = im_minus_jac_blocks(Wd, ei, Wl, Wu)
    assert torch.allclose(assemble_blocks(Ad, ei, Al, Au), torch.eye(n * b, dtype=torch.float64) - W, atol=1e-12)
    # Nonlinear case J = diag(s) W.
    s = torch.rand(n, b, dtype=torch.float64) + 0.5
    Ad, Al, Au = im_minus_jac_blocks(Wd, ei, Wl, Wu, s)
    ref = torch.eye(n * b, dtype=torch.float64) - torch.diag(s.reshape(-1)) @ W
    assert torch.allclose(assemble_blocks(Ad, ei, Al, Au), ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# Affine equilibrium: forward solves z = W z + b; backward is exact vs a dense oracle.
# --------------------------------------------------------------------------- #
def test_affine_fixed_point_is_equilibrium():
    ei = grid_edges(2, 3)
    n, b = 6, 2
    Wd, Wl, Wu = random_coupling(n, ei, b, seed=1, rho=0.8)
    binj = torch.randn(n, b, dtype=torch.float64)
    z = affine_fixed_point(Wd, ei, Wl, Wu, binj)
    assert (z - (spmv(Wd, ei, Wl, Wu, z) + binj)).abs().max() < 1e-10      # z* = W z* + b
    A = assemble_blocks(torch.eye(b, dtype=torch.float64) - Wd, ei, -Wl, -Wu)
    zref = torch.linalg.solve(A, binj.reshape(-1)).reshape(n, b)
    assert torch.allclose(z, zref, atol=1e-10)


def test_affine_backward_matches_dense_implicit_diff():
    ei = grid_edges(3, 3)
    n, b = 9, 2
    Wd0, Wl0, Wu0 = random_coupling(n, ei, b, seed=4, rho=0.9)
    binj = torch.randn(n, b, dtype=torch.float64)
    target = torch.randn(n, b, dtype=torch.float64)

    def grads(use_dense):
        leaves = [t.clone().requires_grad_(True) for t in (Wd0, Wl0, Wu0)]
        if use_dense:
            A = assemble_blocks(torch.eye(b, dtype=torch.float64) - leaves[0], ei, -leaves[1], -leaves[2])
            z = torch.linalg.solve(A, binj.reshape(-1)).reshape(n, b)
        else:
            z = affine_fixed_point(leaves[0], ei, leaves[1], leaves[2], binj)
        torch.mean((z - target) ** 2).backward()
        return [t.grad.clone() for t in leaves]

    for gs, gd in zip(grads(False), grads(True)):
        assert torch.allclose(gs, gd, atol=1e-9)


# --------------------------------------------------------------------------- #
# The result: exact structured backward stays machine-accurate as rho(J) -> 1,
# while the iterative (Neumann-K) backward degrades like rho^K.
# --------------------------------------------------------------------------- #
def test_exact_backward_robust_as_rho_to_one():
    res = backward_accuracy_sweep(grid_edges(3, 3), 9, 2, rhos=(0.5, 0.9, 0.99, 0.999), Ks=(8, 16, 32), seed=0)
    # (a) the exact selected-inverse backward is machine-accurate at every rho.
    for rho, row in res.items():
        assert row["exact"] < (1e-7 if rho >= 0.999 else 1e-9), (rho, row["exact"])
    # (b) in the stiff regime the iterative backward is orders of magnitude worse.
    stiff = res[0.99]
    assert stiff["neumann32"] > 0.1
    assert stiff["neumann32"] / stiff["exact"] > 1e6
    # (c) at fixed K the iterative error grows monotonically toward rho -> 1.
    assert res[0.999]["neumann32"] > res[0.9]["neumann32"] > res[0.5]["neumann32"]


# --------------------------------------------------------------------------- #
# Nonlinear DEQ: the custom IFT backward equals autograd through an unrolled solver.
# --------------------------------------------------------------------------- #
def test_nonlinear_deq_ift_matches_unrolled():
    ei = grid_edges(3, 3)
    n, b = 9, 2
    base = random_coupling(n, ei, b, seed=3, rho=0.5) + (torch.randn(n, b, dtype=torch.float64),)
    w = torch.randn(n, b, dtype=torch.float64)

    def ift_grads():
        leaves = [t.clone().requires_grad_(True) for t in base]
        z = deq_fixed_point(leaves[0], ei, leaves[1], leaves[2], leaves[3], max_iter=500, tol=1e-13)
        (w * z).sum().backward()
        return z.detach(), [t.grad.clone() for t in leaves]

    def unrolled_grads():
        leaves = [t.clone().requires_grad_(True) for t in base]
        z = torch.zeros(n, b, dtype=torch.float64)
        for _ in range(500):
            z = torch.tanh(spmv(leaves[0], ei, leaves[1], leaves[2], z) + leaves[3])
        (w * z).sum().backward()
        return z.detach(), [t.grad for t in leaves]

    z_ift, g_ift = ift_grads()
    z_ref, g_ref = unrolled_grads()
    assert (z_ift - z_ref).abs().max() < 1e-10
    for a, c in zip(g_ift, g_ref):
        assert (a - c).norm() / c.norm() < 1e-8


def test_nonlinear_deq_gradcheck():
    ei = grid_edges(2, 2)
    n, b = 4, 2
    leaves = [t.clone().requires_grad_(True) for t in
              random_coupling(n, ei, b, seed=5, rho=0.5) + (torch.randn(n, b, dtype=torch.float64),)]

    def f(Wd, Wl, Wu, binj):
        return deq_fixed_point(Wd, ei, Wl, Wu, binj, max_iter=500, tol=1e-13)

    assert torch.autograd.gradcheck(f, tuple(leaves), atol=1e-5, rtol=1e-3)


# --------------------------------------------------------------------------- #
# End-to-end: a layer trains through the exact backward.
# --------------------------------------------------------------------------- #
def test_learnability_through_exact_backward():
    mse0, mse1 = fit_affine_fixed_point(grid_edges(3, 3), 9, 2, rho=0.9, steps=300, lr=0.05, seed=0)
    assert mse1 < mse0 * 1e-3
