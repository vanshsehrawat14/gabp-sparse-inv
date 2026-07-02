"""Correctness gates for the block-chain selected inverse."""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import BlockTridiag, random_spd_chain, selected_inverse_chain
from gabp_sparse_inv.bench import metrics

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


def _oracle_selected(bt: BlockTridiag):
    return metrics.dense_oracle(bt)


# --------------------------------------------------------------------------- #
# Phase 1 gate: well-conditioned fp64 matches dense oracle to ~1e-10.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("L,b", [(8, 4), (16, 3), (32, 2), (5, 6)])
def test_well_conditioned_fp64_gate(L, b):
    bt = random_spd_chain(L, b, seed=L * 100 + b, diag_load=5.0)  # well conditioned
    eig = torch.linalg.eigvalsh(bt.to_dense())
    kappa = float(eig.max() / eig.min())
    assert kappa < 1e3, f"fixture not well conditioned (kappa={kappa:.2e})"
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower, check=True)
    fe = metrics.forward_error(G_diag, G_lower, bt)
    assert fe.normwise < 1e-10, f"normwise={fe.normwise:.2e}, kappa={kappa:.2e}"
    assert fe.worst_block < 1e-9


# --------------------------------------------------------------------------- #
# Ill-conditioned fp64: condition-aware tolerance, with diagnostics on failure.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diag_load", [1e-2, 1e-4, 1e-6])
def test_ill_conditioned_fp64_condition_aware(diag_load):
    L, b = 24, 4
    bt = random_spd_chain(L, b, seed=7, diag_load=diag_load)
    eig = torch.linalg.eigvalsh(bt.to_dense())
    kappa = float(eig[-1] / eig[0])
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower)
    fe = metrics.forward_error(G_diag, G_lower, bt)
    # Forward error scales ~ C * kappa * eps; use a generous constant.
    tol = 1e3 * kappa * EPS64
    assert fe.normwise < tol, (
        f"normwise={fe.normwise:.3e} exceeds condition-aware tol={tol:.3e} "
        f"(kappa={kappa:.3e}, worst_block={fe.worst_block:.3e})"
    )


# --------------------------------------------------------------------------- #
# Independent residuals (fp64) + SPD/symmetry properties.
# --------------------------------------------------------------------------- #
def test_independent_residuals_and_properties():
    L, b = 20, 4
    bt = random_spd_chain(L, b, seed=3, diag_load=2.0)
    G_diag, G_lower, factors = selected_inverse_chain(bt.diag, bt.lower, return_factors=True)

    assert metrics.factorization_residual(bt, factors) < 1e-9
    assert metrics.selected_inverse_residual(bt, G_diag, G_lower) < 1e-9

    # G_ii symmetric and SPD.
    assert torch.allclose(G_diag, G_diag.mT, atol=1e-12)
    eig = torch.linalg.eigvalsh(G_diag)
    assert (eig > 0).all(), "G_ii not SPD"


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_L1():
    bt = random_spd_chain(1, 5, seed=1, diag_load=1.0)
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower)
    ref = torch.linalg.inv(bt.diag[0])
    assert torch.allclose(G_diag[0], ref, atol=1e-10)
    assert G_lower.shape[-3] == 0


def test_scalar_blocks_b1():
    L = 30
    bt = random_spd_chain(L, 1, seed=2, diag_load=1.0)
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower)
    fe = metrics.forward_error(G_diag, G_lower, bt)
    assert fe.normwise < 1e-10


def test_long_chain_compounding():
    bt = random_spd_chain(512, 2, seed=11, diag_load=1.0)
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower)
    assert metrics.selected_inverse_residual(bt, G_diag, G_lower) < 1e-8


def test_leading_batch_matches_per_item():
    L, b, batch = 12, 3, 4
    bt = random_spd_chain(L, b, seed=5, diag_load=1.5, batch_shape=(batch,))
    Gd, Gl = selected_inverse_chain(bt.diag, bt.lower)
    for k in range(batch):
        gd_k, gl_k = selected_inverse_chain(bt.diag[k], bt.lower[k])
        assert torch.allclose(Gd[k], gd_k, atol=1e-12)
        assert torch.allclose(Gl[k], gl_k, atol=1e-12)


def test_high_kappa_runs():
    bt = random_spd_chain(16, 4, seed=9, diag_load=1e-5)
    G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower)
    assert torch.isfinite(G_diag).all() and torch.isfinite(G_lower).all()


# --------------------------------------------------------------------------- #
# API / validation.
# --------------------------------------------------------------------------- #
def test_validate_rejects_nonsymmetric():
    bt = random_spd_chain(4, 3, seed=0, diag_load=1.0)
    bt.diag[0, 0, 1] += 1.0  # break symmetry
    with pytest.raises(ValueError):
        selected_inverse_chain(bt.diag, bt.lower, check=True)


def test_validate_rejects_bad_lower_shape():
    bt = random_spd_chain(5, 3, seed=0, diag_load=1.0)
    with pytest.raises(ValueError):
        BlockTridiag(diag=bt.diag, lower=bt.lower[:-1]).validate()


def test_validate_rejects_nonsquare_blocks():
    diag = torch.randn(4, 3, 2)
    with pytest.raises(ValueError):
        BlockTridiag(diag=diag, lower=torch.randn(3, 3, 2)).validate()


# --------------------------------------------------------------------------- #
# Precision: fp32 loose bound; bf16 records the compute path (representation expt).
# --------------------------------------------------------------------------- #
def test_fp32_relative_accuracy():
    bt = random_spd_chain(16, 4, seed=4, diag_load=2.0)
    diag32, lower32 = bt.diag.to(torch.float32), bt.lower.to(torch.float32)
    G_diag, G_lower = selected_inverse_chain(diag32, lower32)
    fe = metrics.forward_error(G_diag, G_lower, bt)
    assert fe.normwise < 1e-4  # loose; not an fp64-style gate


def test_compute_dtype_fp32_storage_fp64_compute_matches_oracle():
    # fp32 storage + fp64 compute: the result (fp32) must match the dense inverse of the
    # fp32-rounded matrix to fp32-storage precision -- isolates the upcast/cast-back path
    # (the native-fp32 test above also carries fp32 *compute* error, hence its looser tol).
    bt = random_spd_chain(16, 4, seed=6, diag_load=2.0)
    diag32, lower32 = bt.diag.to(torch.float32), bt.lower.to(torch.float32)
    G_diag, G_lower = selected_inverse_chain(diag32, lower32, compute_dtype=torch.float64)
    assert G_diag.dtype == torch.float32 and G_lower.dtype == torch.float32
    bt32 = BlockTridiag(diag=diag32, lower=lower32)  # oracle = inv of the fp32-rounded A
    fe = metrics.forward_error(G_diag, G_lower, bt32)
    assert fe.normwise < 1e-5


def test_bf16_store_low_compute_fp32():
    from gabp_sparse_inv.bench.harness import resolve_precision

    spec = resolve_precision("bf16", torch.device("cpu"))
    # No native half Cholesky on CPU -> compute must be promoted to fp32.
    assert spec.compute == torch.float32
    assert spec.input_storage == torch.bfloat16
    assert spec.output == torch.bfloat16
    assert spec.runnable

    bt = random_spd_chain(8, 4, seed=6, diag_load=4.0)
    diag16, lower16 = bt.diag.to(torch.bfloat16), bt.lower.to(torch.bfloat16)
    G_diag, G_lower = selected_inverse_chain(diag16, lower16, compute_dtype=torch.float32)
    assert G_diag.dtype == torch.bfloat16  # cast back to storage dtype
    fe = metrics.forward_error(G_diag, G_lower, bt)
    assert fe.normwise < 5e-1  # bf16 storage is coarse; just sanity
