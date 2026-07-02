"""Correctness gates for the block-star (block-arrowhead) selected inverse."""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    BlockStar,
    random_spd_star,
    selected_inverse_chain,
    selected_inverse_star,
)
from gabp_sparse_inv.bench import metrics

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


# --------------------------------------------------------------------------- #
# Phase 2 gate: well-conditioned fp64 matches the dense oracle to ~1e-10.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K,b", [(5, 3), (8, 4), (16, 2), (1, 6)])
def test_well_conditioned_fp64_gate(K, b):
    st = random_spd_star(K, b, seed=K * 100 + b, diag_load=5.0)  # well conditioned
    eig = torch.linalg.eigvalsh(st.to_dense())
    kappa = float(eig.max() / eig.min())
    assert kappa < 1e3, f"fixture not well conditioned (kappa={kappa:.2e})"
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling, check=True)
    fe = metrics.forward_error_star(G_c, G_l, G_x, st)
    assert fe.normwise < 1e-10, f"normwise={fe.normwise:.2e}, kappa={kappa:.2e}"
    assert fe.worst_block < 1e-9


# --------------------------------------------------------------------------- #
# Ill-conditioned fp64: condition-aware tolerance.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("diag_load", [1e-2, 1e-4, 1e-6])
def test_ill_conditioned_fp64_condition_aware(diag_load):
    K, b = 12, 4
    st = random_spd_star(K, b, seed=7, diag_load=diag_load)
    eig = torch.linalg.eigvalsh(st.to_dense())
    kappa = float(eig[-1] / eig[0])
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)
    fe = metrics.forward_error_star(G_c, G_l, G_x, st)
    tol = 1e3 * kappa * EPS64
    assert fe.normwise < tol, (
        f"normwise={fe.normwise:.3e} exceeds condition-aware tol={tol:.3e} "
        f"(kappa={kappa:.3e}, worst_block={fe.worst_block:.3e})"
    )


# --------------------------------------------------------------------------- #
# Independent residuals (fp64) + SPD/symmetry properties.
# --------------------------------------------------------------------------- #
def test_independent_residuals_and_properties():
    K, b = 10, 4
    st = random_spd_star(K, b, seed=3, diag_load=2.0)
    G_c, G_l, G_x, factors = selected_inverse_star(
        st.center, st.leaf_diag, st.coupling, return_factors=True
    )

    assert metrics.factorization_residual_star(st, factors) < 1e-9
    assert metrics.selected_inverse_residual_star(st, G_c, G_l, G_x) < 1e-9

    # G_00 and G_jj symmetric and SPD.
    assert torch.allclose(G_c, G_c.mT, atol=1e-12)
    assert torch.allclose(G_l, G_l.mT, atol=1e-12)
    assert (torch.linalg.eigvalsh(G_c) > 0).all(), "G_00 not SPD"
    assert (torch.linalg.eigvalsh(G_l) > 0).all(), "G_jj not SPD"


# --------------------------------------------------------------------------- #
# K=1 star is exactly a 2-block chain: cross-check the two kernels.
# --------------------------------------------------------------------------- #
def test_K1_matches_chain():
    st = random_spd_star(1, 4, seed=4, diag_load=1.5)
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)

    # Equivalent 2-block chain: block 0 = center, block 1 = leaf 0,
    # sub-diagonal A_{1,0} = coupling[0]^T.
    diag = torch.stack([st.center, st.leaf_diag[0]])
    lower = st.coupling[0].mT.unsqueeze(0)
    Gd, Gl = selected_inverse_chain(diag, lower)
    assert torch.allclose(G_c, Gd[0], atol=1e-12)
    assert torch.allclose(G_l[0], Gd[1], atol=1e-12)
    assert torch.allclose(G_x[0], Gl[0], atol=1e-12)   # G_{1,0}


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_large_K():
    st = random_spd_star(256, 2, seed=11, diag_load=1.0)
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)
    assert metrics.selected_inverse_residual_star(st, G_c, G_l, G_x) < 1e-8


def test_scalar_blocks_b1():
    st = random_spd_star(40, 1, seed=2, diag_load=1.0)
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)
    fe = metrics.forward_error_star(G_c, G_l, G_x, st)
    assert fe.normwise < 1e-10


def test_leading_batch_matches_per_item():
    K, b, batch = 6, 3, 4
    st = random_spd_star(K, b, seed=5, diag_load=1.5, batch_shape=(batch,))
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)
    for k in range(batch):
        c_k, l_k, x_k = selected_inverse_star(
            st.center[k], st.leaf_diag[k], st.coupling[k]
        )
        assert torch.allclose(G_c[k], c_k, atol=1e-12)
        assert torch.allclose(G_l[k], l_k, atol=1e-12)
        assert torch.allclose(G_x[k], x_k, atol=1e-12)


def test_high_kappa_runs():
    st = random_spd_star(16, 4, seed=9, diag_load=1e-5)
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)
    assert torch.isfinite(G_c).all() and torch.isfinite(G_l).all() and torch.isfinite(G_x).all()


# --------------------------------------------------------------------------- #
# API / validation.
# --------------------------------------------------------------------------- #
def test_validate_rejects_nonsymmetric_center():
    st = random_spd_star(4, 3, seed=0, diag_load=1.0)
    st.center[0, 1] += 1.0  # break symmetry
    with pytest.raises(ValueError):
        selected_inverse_star(st.center, st.leaf_diag, st.coupling, check=True)


def test_validate_rejects_bad_coupling_shape():
    st = random_spd_star(5, 3, seed=0, diag_load=1.0)
    with pytest.raises(ValueError):
        BlockStar(center=st.center, leaf_diag=st.leaf_diag, coupling=st.coupling[:-1]).validate()


# --------------------------------------------------------------------------- #
# Precision: fp32 loose bound; bf16 records the compute path.
# --------------------------------------------------------------------------- #
def test_fp32_relative_accuracy():
    st = random_spd_star(16, 4, seed=4, diag_load=2.0)
    c32, l32, x32 = st.center.to(torch.float32), st.leaf_diag.to(torch.float32), st.coupling.to(torch.float32)
    G_c, G_l, G_x = selected_inverse_star(c32, l32, x32)
    fe = metrics.forward_error_star(G_c, G_l, G_x, st)
    assert fe.normwise < 1e-4  # loose; not an fp64-style gate


def test_compute_dtype_fp32_storage_fp64_compute_matches_oracle():
    # fp32 storage + fp64 compute: the result (fp32) must match the dense inverse of the
    # fp32-rounded matrix to fp32-storage precision -- isolates the upcast/cast-back path
    # (the native-fp32 test above also carries fp32 *compute* error, hence its looser tol).
    st = random_spd_star(16, 4, seed=6, diag_load=2.0)
    c32, l32, x32 = st.center.to(torch.float32), st.leaf_diag.to(torch.float32), st.coupling.to(torch.float32)
    G_c, G_l, G_x = selected_inverse_star(c32, l32, x32, compute_dtype=torch.float64)
    assert G_c.dtype == torch.float32 and G_l.dtype == torch.float32 and G_x.dtype == torch.float32
    st32 = BlockStar(center=c32, leaf_diag=l32, coupling=x32)  # oracle = inv of the fp32-rounded A
    fe = metrics.forward_error_star(G_c, G_l, G_x, st32)
    assert fe.normwise < 1e-5


def test_bf16_store_low_compute_fp32():
    st = random_spd_star(8, 4, seed=6, diag_load=4.0)
    c16, l16, x16 = st.center.to(torch.bfloat16), st.leaf_diag.to(torch.bfloat16), st.coupling.to(torch.bfloat16)
    G_c, G_l, G_x = selected_inverse_star(c16, l16, x16, compute_dtype=torch.float32)
    assert G_c.dtype == torch.bfloat16  # cast back to storage dtype
    fe = metrics.forward_error_star(G_c, G_l, G_x, st)
    assert fe.normwise < 5e-1  # bf16 storage is coarse; just sanity
