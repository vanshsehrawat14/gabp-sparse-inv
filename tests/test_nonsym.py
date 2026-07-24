"""Correctness gates for the non-symmetric block lower-bidiagonal selected inverse.

Mirrors the SPD chain/autodiff gates (dense oracle, condition-aware tolerance,
independent on-pattern residuals, gradcheck, analytic-vs-dense adjoint, batch dims,
fp32/bf16) for the non-symmetric kernel ``selected_inverse_bidiag`` / ``selinv_bidiag``.
The dense oracle is ``torch.linalg.inv`` of the assembled (triangular, non-symmetric)
matrix ``M``; there is no symmetrization convention to reconcile, so analytic and dense
gradients agree to ~1e-9.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    BlockBidiag,
    random_nonsym_bidiag,
    selected_inverse_bidiag,
    selected_inverse_tril,
    selinv_bidiag,
    selinv_tril,
)

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _oracle_selected(bt: BlockBidiag):
    """Selected blocks of M^{-1} via a dense inverse (test oracle)."""
    n, b = bt.num_blocks, bt.block_size
    G = torch.linalg.inv(bt.to_dense().to(torch.float64))
    od = torch.stack([G[..., i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(n)], dim=-3)
    if n > 1:
        ol = torch.stack(
            [G[..., (i + 1) * b:(i + 2) * b, i * b:(i + 1) * b] for i in range(n - 1)], dim=-3
        )
    else:
        ol = G.new_zeros((*G.shape[:-2], 0, b, b))
    return od, ol


def _forward_error(Gd, Gl, bt: BlockBidiag) -> float:
    """Relative Frobenius error of the selected blocks vs the dense oracle."""
    od, ol = _oracle_selected(bt)
    num = (Gd.to(torch.float64) - od).pow(2).sum() + (Gl.to(torch.float64) - ol).pow(2).sum()
    den = od.pow(2).sum() + ol.pow(2).sum()
    return float((num / den).sqrt())


def _assemble_M(diag, lower):
    """Autograd-traceable assembly of the dense lower-bidiagonal M from its blocks."""
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    M = diag.new_zeros((*diag.shape[:-3], N, N))
    for i in range(n):
        r = slice(i * b, (i + 1) * b)
        M[..., r, r] = diag[..., i, :, :]
        if i < n - 1:
            c = slice((i + 1) * b, (i + 2) * b)
            M[..., c, r] = lower[..., i, :, :]
    return M


def _dense_selinv_from_blocks(diag, lower):
    """Differentiable dense oracle: assemble M, invert, extract selected blocks."""
    n, b = diag.shape[-3], diag.shape[-1]
    G = torch.linalg.inv(_assemble_M(diag, lower))
    Gd = torch.stack([G[..., i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(n)], dim=-3)
    if n > 1:
        Gl = torch.stack(
            [G[..., (i + 1) * b:(i + 2) * b, i * b:(i + 1) * b] for i in range(n - 1)], dim=-3
        )
    else:
        Gl = diag.new_zeros((*diag.shape[:-3], 0, b, b))
    return Gd, Gl


def _grads(fn, diag0, lower0, loss_fn):
    diag = diag0.clone().requires_grad_(True)
    lower = lower0.clone().requires_grad_(True)
    Gd, Gl = fn(diag, lower)
    loss_fn(Gd, Gl).backward()
    gd = diag.grad if diag.grad is not None else torch.zeros_like(diag)
    gl = lower.grad if lower.grad is not None else torch.zeros_like(lower)
    return Gd.detach(), Gl.detach(), gd.clone(), gl.clone()


def _make_random_loss(n, b, seed):
    g = torch.Generator().manual_seed(seed)
    wd = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    wl = torch.randn(max(n - 1, 0), b, b, generator=g, dtype=torch.float64)
    return lambda Gd, Gl: (Gd * wd).sum() + (Gl * wl).sum()


# --------------------------------------------------------------------------- #
# Forward: well-conditioned fp64 matches dense oracle to ~1e-10.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,b", [(8, 4), (16, 3), (32, 2), (5, 6)])
def test_well_conditioned_fp64_gate(n, b):
    bt = random_nonsym_bidiag(n, b, seed=n * 100 + b, diag_load=4.0)
    kappa = float(torch.linalg.cond(bt.to_dense()))
    assert kappa < 1e3, f"fixture not well conditioned (kappa={kappa:.2e})"
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower, check=True)
    assert _forward_error(Gd, Gl, bt) < 1e-10


# --------------------------------------------------------------------------- #
# Ill-conditioned fp64: condition-aware tolerance.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diag_load", [1.0, 1e-1, 1e-2])
def test_ill_conditioned_fp64_condition_aware(diag_load):
    bt = random_nonsym_bidiag(24, 4, seed=7, diag_load=diag_load)
    kappa = float(torch.linalg.cond(bt.to_dense()))
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower)
    fe = _forward_error(Gd, Gl, bt)
    tol = 1e3 * kappa * EPS64
    assert fe < tol, f"fe={fe:.3e} exceeds condition-aware tol={tol:.3e} (kappa={kappa:.3e})"


# --------------------------------------------------------------------------- #
# Independent on-pattern residuals: the defining equations of the selected inverse.
#   D_i G_ii = I                         (diagonal block of M G = I)
#   D_{i+1} G_{i+1,i} + C_i G_ii = 0     ((i+1,i) block of M G = 0)
# --------------------------------------------------------------------------- #
def test_independent_residuals():
    n, b = 20, 4
    bt = random_nonsym_bidiag(n, b, seed=3, diag_load=2.0)
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower)
    eye = torch.eye(b, dtype=torch.float64)
    res_diag = (bt.diag @ Gd - eye).abs().amax()
    res_off = (bt.diag[1:] @ Gl + bt.lower @ Gd[:-1]).abs().amax()
    assert float(res_diag) < 1e-10
    assert float(res_off) < 1e-10


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_n1():
    bt = random_nonsym_bidiag(1, 5, seed=1, diag_load=1.0)
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower)
    assert torch.allclose(Gd[0], torch.linalg.inv(bt.diag[0]), atol=1e-10)
    assert Gl.shape[-3] == 0


def test_scalar_blocks_b1():
    bt = random_nonsym_bidiag(30, 1, seed=2, diag_load=1.0)
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower)
    assert _forward_error(Gd, Gl, bt) < 1e-10


def test_leading_batch_matches_per_item():
    n, b, batch = 12, 3, 4
    bt = random_nonsym_bidiag(n, b, seed=5, diag_load=1.5, batch_shape=(batch,))
    Gd, Gl = selected_inverse_bidiag(bt.diag, bt.lower)
    for k in range(batch):
        gd_k, gl_k = selected_inverse_bidiag(bt.diag[k], bt.lower[k])
        assert torch.allclose(Gd[k], gd_k, atol=1e-12)
        assert torch.allclose(Gl[k], gl_k, atol=1e-12)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_lower_shape():
    bt = random_nonsym_bidiag(5, 3, seed=0, diag_load=1.0)
    with pytest.raises(ValueError):
        BlockBidiag(diag=bt.diag, lower=bt.lower[:-1]).validate()


def test_validate_rejects_nonsquare_blocks():
    diag = torch.randn(4, 3, 2)
    with pytest.raises(ValueError):
        BlockBidiag(diag=diag, lower=torch.randn(3, 3, 2)).validate()


def test_check_flag_validates():
    bt = random_nonsym_bidiag(5, 3, seed=0, diag_load=1.0)
    with pytest.raises(ValueError):
        selected_inverse_bidiag(bt.diag, bt.lower[:-1], check=True)


# --------------------------------------------------------------------------- #
# Precision.
# --------------------------------------------------------------------------- #
def test_fp32_relative_accuracy():
    bt = random_nonsym_bidiag(16, 4, seed=4, diag_load=2.0)
    Gd, Gl = selected_inverse_bidiag(bt.diag.float(), bt.lower.float())
    assert _forward_error(Gd, Gl, bt) < 1e-4


def test_bf16_store_low_compute_fp32():
    bt = random_nonsym_bidiag(8, 4, seed=6, diag_load=4.0)
    diag16, lower16 = bt.diag.to(torch.bfloat16), bt.lower.to(torch.bfloat16)
    Gd, Gl = selected_inverse_bidiag(diag16, lower16, compute_dtype=torch.float32)
    assert Gd.dtype == torch.bfloat16  # cast back to storage dtype
    assert _forward_error(Gd, Gl, bt) < 5e-1  # bf16 storage is coarse; sanity only


# --------------------------------------------------------------------------- #
# Backward: gradcheck on controlled-conditioning fixtures.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,b", [(6, 2), (7, 3), (9, 2), (1, 3)])
def test_gradcheck(n, b):
    bt = random_nonsym_bidiag(n, b, seed=n * 10 + b, diag_load=4.0)
    diag = bt.diag.clone().requires_grad_(True)
    lower = bt.lower.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda d, l: selinv_bidiag(d, l), (diag, lower), atol=1e-6, rtol=1e-4
    )


# --------------------------------------------------------------------------- #
# Backward: analytic == autograd through the parametrized dense inverse.
#   The adjoint-restriction identity for general bar_G (beyond what gradcheck samples).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,b", [(10, 3), (15, 2), (4, 4)])
def test_analytic_vs_dense_adjoint(n, b):
    bt = random_nonsym_bidiag(n, b, seed=n + b, diag_load=4.0)
    loss_fn = _make_random_loss(n, b, seed=n * 3 + b)
    Gd_a, Gl_a, gd_a, gl_a = _grads(lambda d, l: selinv_bidiag(d, l), bt.diag, bt.lower, loss_fn)
    Gd_o, Gl_o, gd_o, gl_o = _grads(_dense_selinv_from_blocks, bt.diag, bt.lower, loss_fn)
    assert torch.allclose(Gd_a, Gd_o, atol=1e-10)
    assert torch.allclose(Gl_a, Gl_o, atol=1e-10)
    assert torch.allclose(gd_a, gd_o, atol=1e-9), (gd_a - gd_o).abs().max()
    assert torch.allclose(gl_a, gl_o, atol=1e-9), (gl_a - gl_o).abs().max()


def test_batched_grads():
    n, b = 7, 2
    bt = random_nonsym_bidiag(n, b, seed=3, diag_load=4.0, batch_shape=(4,))
    loss_fn = _make_random_loss(n, b, seed=11)  # weights broadcast over the batch dim
    _, _, gd_a, gl_a = _grads(lambda d, l: selinv_bidiag(d, l), bt.diag, bt.lower, loss_fn)
    _, _, gd_o, gl_o = _grads(_dense_selinv_from_blocks, bt.diag, bt.lower, loss_fn)
    assert gd_a.shape == bt.diag.shape
    assert torch.allclose(gd_a, gd_o, atol=1e-9)
    assert torch.allclose(gl_a, gl_o, atol=1e-9)


# --------------------------------------------------------------------------- #
# Dense triangular instance: T = (I - A)^{-1}, the DeltaNet chunk inverse (§9.4).
# --------------------------------------------------------------------------- #
def _strict_lower(N, *, seed, scale=0.3, batch=()):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(*batch, N, N, generator=g, dtype=torch.float64).tril(-1)


@pytest.mark.parametrize("N,batch", [(12, ()), (8, (3,)), (16, (2, 2))])
def test_tril_forward_vs_dense(N, batch):
    A = _strict_lower(N, seed=N, batch=batch)
    T = selected_inverse_tril(A, check=True)
    ref = torch.linalg.inv(torch.eye(N, dtype=torch.float64) - A)
    assert (T - ref).abs().max() < 1e-10
    # Unit lower triangular.
    assert torch.allclose(T.triu(1), torch.zeros_like(T), atol=1e-12)
    assert torch.allclose(T.diagonal(dim1=-2, dim2=-1), torch.ones(*batch, N, dtype=torch.float64))


def test_tril_ignores_upper_and_diagonal():
    A = _strict_lower(10, seed=1)
    perturbed = A + torch.triu(torch.randn(10, 10, dtype=torch.float64))  # change diag+upper
    assert torch.allclose(selected_inverse_tril(A), selected_inverse_tril(perturbed), atol=1e-12)


@pytest.mark.parametrize("N", [5, 8, 11])
def test_tril_gradcheck(N):
    A = _strict_lower(N, seed=N + 1).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a: selinv_tril(a), (A,), atol=1e-6, rtol=1e-4)


def test_tril_analytic_vs_autograd_solve():
    """Analytic transpose-form VJP equals autograd through solve_triangular."""
    N = 9
    w = torch.randn(N, N, dtype=torch.float64)

    def oracle(a):
        eye = torch.eye(N, dtype=a.dtype)
        return torch.linalg.solve_triangular(eye - a.tril(-1), eye, upper=False, unitriangular=True)

    a1 = _strict_lower(N, seed=2).requires_grad_(True)
    (selinv_tril(a1) * w).sum().backward()
    a2 = a1.detach().clone().requires_grad_(True)
    (oracle(a2) * w).sum().backward()
    assert torch.allclose(a1.grad, a2.grad, atol=1e-10)
    # Gradient lives on the strictly-lower pattern only.
    assert torch.allclose(a1.grad.triu(0), torch.zeros_like(a1.grad), atol=1e-12)


def test_tril_fp32():
    A = _strict_lower(12, seed=3).float()
    T = selected_inverse_tril(A, compute_dtype=torch.float32)
    ref = torch.linalg.inv(torch.eye(12) - A)
    assert (T - ref).abs().max() < 1e-4
