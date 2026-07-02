"""Correctness gates for the general block-tree selected inverse."""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    BlockTree,
    random_spd_tree,
    selected_inverse_chain,
    selected_inverse_star,
    selected_inverse_tree,
)
from gabp_sparse_inv.bench import metrics

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


def _kappa(bt: BlockTree) -> float:
    eig = torch.linalg.eigvalsh(bt.to_dense().to(torch.float64))
    return float(eig[..., -1].max() / eig[..., 0].min())


# --------------------------------------------------------------------------- #
# Well-conditioned fp64 gate on controlled-conditioning topologies.
# (path/balanced trees have bounded depth-driven conditioning, unlike random.)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "balanced", "star"])
@pytest.mark.parametrize("n,b", [(7, 3), (15, 2), (8, 4)])
def test_well_conditioned_fp64_gate(kind, n, b):
    bt = random_spd_tree(n, b, seed=n * 100 + b, diag_load=5.0, kind=kind)
    kappa = _kappa(bt)
    assert kappa < 1e3, f"fixture not well conditioned (kappa={kappa:.2e})"
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent, check=True)
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    assert fe.normwise < 1e-10, f"normwise={fe.normwise:.2e}, kappa={kappa:.2e}"
    assert fe.worst_block < 1e-9


# --------------------------------------------------------------------------- #
# Random trees: condition-aware tolerance (no absolute floor on uncontrolled kappa).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("diag_load", [2.0, 1e-2, 1e-4])
def test_random_trees_condition_aware(seed, diag_load):
    n, b = 24, 3
    bt = random_spd_tree(n, b, seed=seed, diag_load=diag_load, kind="random")
    kappa = _kappa(bt)
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    tol = 1e3 * kappa * EPS64
    assert fe.normwise < tol, (
        f"normwise={fe.normwise:.3e} exceeds condition-aware tol={tol:.3e} "
        f"(kappa={kappa:.3e}, worst_block={fe.worst_block:.3e})"
    )


# --------------------------------------------------------------------------- #
# Root invariance: a path tree (rooted at node 0) must agree block-for-block with
# the chain kernel (rooted at the *last* node). Two eliminations of one matrix.
# --------------------------------------------------------------------------- #
def test_path_tree_matches_chain_root_invariance():
    from gabp_sparse_inv import random_spd_chain

    L, b = 9, 4
    bt = random_spd_chain(L, b, seed=5, diag_load=1.5)
    diag, lower = bt.diag, bt.lower               # lower[i] = A_{i+1,i}

    # Path tree rooted at 0: parent[v] = v-1, edge[v] = A_{v-1,v} = lower[v-1]^T.
    parent = [-1] + list(range(0, L - 1))
    edge = torch.zeros_like(diag)
    edge[1:] = lower.mT
    G_diag, G_edge = selected_inverse_tree(diag, edge, parent)

    Gd, Gl = selected_inverse_chain(diag, lower)
    # Diagonal blocks agree directly.
    assert torch.allclose(G_diag, Gd, atol=1e-11)
    # Edge: tree stores G_{v-1,v}; chain stores G_lower[v-1] = G_{v,v-1} = G_{v-1,v}^T.
    for v in range(1, L):
        assert torch.allclose(G_edge[v], Gl[v - 1].mT, atol=1e-11)


# --------------------------------------------------------------------------- #
# Depth-1 tree must agree block-for-block with the star kernel.
# --------------------------------------------------------------------------- #
def test_star_tree_matches_star_kernel():
    from gabp_sparse_inv import random_spd_star

    K, b = 6, 3
    st = random_spd_star(K, b, seed=8, diag_load=2.0)
    # Tree: node 0 = center, nodes 1..K = leaves, parent[j]=0.
    # edge[j] = A_{0,j} = coupling[j-1].
    n = K + 1
    diag = torch.empty((n, b, b), dtype=st.center.dtype)
    diag[0] = st.center
    diag[1:] = st.leaf_diag
    edge = torch.zeros((n, b, b), dtype=st.center.dtype)
    edge[1:] = st.coupling
    parent = [-1] + [0] * K

    G_diag, G_edge = selected_inverse_tree(diag, edge, parent)
    G_c, G_l, G_x = selected_inverse_star(st.center, st.leaf_diag, st.coupling)

    assert torch.allclose(G_diag[0], G_c, atol=1e-11)
    assert torch.allclose(G_diag[1:], G_l, atol=1e-11)
    # tree G_edge[j] = G_{0,j}; star G_cross[j-1] = G_{j,0} = G_{0,j}^T.
    for j in range(1, n):
        assert torch.allclose(G_edge[j], G_x[j - 1].mT, atol=1e-11)


# --------------------------------------------------------------------------- #
# Independent residuals (fp64) + SPD/symmetry of the diagonal blocks.
# --------------------------------------------------------------------------- #
def test_independent_residuals_and_properties():
    n, b = 20, 4
    bt = random_spd_tree(n, b, seed=3, diag_load=2.0, kind="random")
    G_diag, G_edge, factors = selected_inverse_tree(
        bt.diag, bt.edge, bt.parent, return_factors=True
    )
    assert metrics.factorization_residual_tree(bt, factors) < 1e-9
    assert metrics.selected_inverse_residual_tree(bt, G_diag, G_edge) < 1e-9

    assert torch.allclose(G_diag, G_diag.mT, atol=1e-12)
    assert (torch.linalg.eigvalsh(G_diag) > 0).all(), "G_vv not SPD"


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_single_node():
    bt = random_spd_tree(1, 5, seed=1, diag_load=1.0)
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    ref = torch.linalg.inv(bt.diag[0])
    assert torch.allclose(G_diag[0], ref, atol=1e-11)
    assert G_edge.shape == bt.edge.shape  # root edge slot present and zero
    assert float(G_edge.abs().max()) == 0.0


def test_scalar_blocks_b1():
    bt = random_spd_tree(30, 1, seed=2, diag_load=1.0, kind="random")
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    tol = 1e3 * _kappa(bt) * EPS64
    assert fe.normwise < tol


def test_large_tree_residual():
    bt = random_spd_tree(300, 2, seed=11, diag_load=1.0, kind="random")
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    assert metrics.selected_inverse_residual_tree(bt, G_diag, G_edge) < 1e-8


def test_leading_batch_matches_per_item():
    n, b, batch = 10, 3, 4
    bt = random_spd_tree(n, b, seed=5, diag_load=1.5, kind="random", batch_shape=(batch,))
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    for k in range(batch):
        d_k, e_k = selected_inverse_tree(bt.diag[k], bt.edge[k], bt.parent)
        assert torch.allclose(G_diag[k], d_k, atol=1e-12)
        assert torch.allclose(G_edge[k], e_k, atol=1e-12)


def test_explicit_parent_array():
    # A hand-built tree:   0 - 1 - 3
    #                       \  2
    parent = [-1, 0, 0, 1]
    bt = random_spd_tree(4, 3, seed=7, diag_load=2.0, parent=parent)
    assert bt.parent.tolist() == parent
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent, check=True)
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    assert fe.normwise < 1e-10


# --------------------------------------------------------------------------- #
# API / validation.
# --------------------------------------------------------------------------- #
def test_validate_rejects_nonsymmetric_diag():
    bt = random_spd_tree(5, 3, seed=0, diag_load=1.0)
    bt.diag[0, 0, 1] += 1.0  # break symmetry
    with pytest.raises(ValueError):
        selected_inverse_tree(bt.diag, bt.edge, bt.parent, check=True)


def test_validate_rejects_cyclic_parent():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge = torch.zeros(3, 2, 2)
    with pytest.raises(ValueError):
        BlockTree(diag=diag, edge=edge, parent=[1, 2, 0]).validate()  # 3-cycle, no root


def test_validate_rejects_out_of_range_parent():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge = torch.zeros(3, 2, 2)
    with pytest.raises(ValueError):
        BlockTree(diag=diag, edge=edge, parent=[-1, 0, 9]).validate()


def test_validate_rejects_multiple_roots():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge = torch.zeros(3, 2, 2)
    with pytest.raises(ValueError):
        BlockTree(diag=diag, edge=edge, parent=[-1, -1, 0]).validate()


def test_validate_rejects_nonzero_root_edge():
    bt = random_spd_tree(5, 3, seed=0, diag_load=1.0)
    bt.edge[bt.root] += 1.0
    with pytest.raises(ValueError):
        bt.validate()


# --------------------------------------------------------------------------- #
# Precision: fp32 loose bound; bf16 store-low / compute-fp32 path.
# --------------------------------------------------------------------------- #
def test_fp32_relative_accuracy():
    bt = random_spd_tree(16, 4, seed=4, diag_load=3.0, kind="balanced")
    d32, e32 = bt.diag.to(torch.float32), bt.edge.to(torch.float32)
    G_diag, G_edge = selected_inverse_tree(d32, e32, bt.parent)
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    assert fe.normwise < 1e-4  # loose; not an fp64-style gate


def test_compute_dtype_fp32_storage_fp64_compute_matches_oracle():
    # fp32 storage + fp64 compute: the result (fp32) must match the dense inverse of the
    # fp32-rounded matrix to fp32-storage precision -- isolates the upcast/cast-back path
    # (the native-fp32 test above also carries fp32 *compute* error, hence its looser tol).
    bt = random_spd_tree(16, 4, seed=6, diag_load=3.0, kind="balanced")
    d32, e32 = bt.diag.to(torch.float32), bt.edge.to(torch.float32)
    G_diag, G_edge = selected_inverse_tree(d32, e32, bt.parent, compute_dtype=torch.float64)
    assert G_diag.dtype == torch.float32 and G_edge.dtype == torch.float32
    bt32 = BlockTree(diag=d32, edge=e32, parent=bt.parent)  # oracle = inv of the fp32-rounded A
    fe = metrics.forward_error_tree(G_diag, G_edge, bt32)
    assert fe.normwise < 1e-5


def test_bf16_store_low_compute_fp32():
    bt = random_spd_tree(12, 4, seed=6, diag_load=4.0, kind="balanced")
    d16, e16 = bt.diag.to(torch.bfloat16), bt.edge.to(torch.bfloat16)
    G_diag, G_edge = selected_inverse_tree(d16, e16, bt.parent, compute_dtype=torch.float32)
    assert G_diag.dtype == torch.bfloat16  # cast back to storage dtype
    fe = metrics.forward_error_tree(G_diag, G_edge, bt)
    assert fe.normwise < 5e-1  # bf16 storage is coarse; just sanity


# --------------------------------------------------------------------------- #
# Trace identities read off the selected inverse (the C3 one-liner; APPLICATIONS.md;
# junction gates the same). trace(A^-1) = sum_v tr(G_vv);
# trace(A^-1 B) = <G_diag, Bd> + 2 <G_edge, Be> for symmetric B on the tree pattern.
# --------------------------------------------------------------------------- #
def test_trace_identities_from_selected_inverse():
    bt = random_spd_tree(13, 3, seed=5, diag_load=2.0, kind="balanced")
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    inv = torch.linalg.inv(bt.to_dense().to(torch.float64))

    tr_Ainv = torch.diagonal(G_diag, dim1=-2, dim2=-1).sum()
    assert torch.allclose(tr_Ainv, torch.trace(inv), atol=1e-10)

    n, b = bt.num_nodes, bt.block_size
    plist = bt.parent.tolist()
    g = torch.Generator().manual_seed(3)
    Bd = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    Bd = Bd + Bd.mT                                          # symmetric diagonal blocks
    Be = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    N = n * b
    Bm = torch.zeros(N, N, dtype=torch.float64)
    for v in range(n):
        Bm[v * b:(v + 1) * b, v * b:(v + 1) * b] = Bd[v]
        pv = plist[v]
        if pv == -1:
            Be[v] = 0.0                                      # root edge slot is off-pattern
            continue
        Bm[pv * b:(pv + 1) * b, v * b:(v + 1) * b] = Be[v]   # B_{p,v}
        Bm[v * b:(v + 1) * b, pv * b:(pv + 1) * b] = Be[v].mT

    tr_AinvB = (G_diag * Bd).sum() + 2.0 * (G_edge * Be).sum()
    assert torch.allclose(tr_AinvB, torch.trace(inv @ Bm), atol=1e-10)


# --------------------------------------------------------------------------- #
# Adjoint-of-selinv smoke check (de-risks the differentiable thread; see
# docs/derivations.md §8). Validates the on-pattern cotangent of log-det against
# autograd-through-dense on the reference kernel.
# --------------------------------------------------------------------------- #
def test_adjoint_logdet_matches_autograd():
    # f = log det A. Then bar_A = A^{-T} = A^{-1}; restricted to the tree pattern the
    # gradient is exactly the selected inverse (diag + edge) blocks. Check that the
    # selected inverse equals autograd's d(logdet)/dA projected onto the pattern.
    n, b = 8, 3
    bt = random_spd_tree(n, b, seed=2, diag_load=2.0, kind="random")
    G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)

    A = bt.to_dense().clone().requires_grad_(True)
    logdet = torch.logdet(A)
    (grad,) = torch.autograd.grad(logdet, A)        # d logdet / dA = A^{-1} (sym)
    plist = bt.parent.tolist()
    # Diagonal blocks of the grad must equal G_diag.
    for v in range(n):
        gv = grad[v * b:(v + 1) * b, v * b:(v + 1) * b]
        assert torch.allclose(G_diag[v], gv, atol=1e-9)
        p = plist[v]
        if p != -1:
            # grad block (p,v) = (A^{-1})_{p,v} = G_edge[v].
            ge = grad[p * b:(p + 1) * b, v * b:(v + 1) * b]
            assert torch.allclose(G_edge[v], ge, atol=1e-9)
