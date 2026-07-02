"""Tests for the seeded SPD block-tridiagonal generator."""

from __future__ import annotations

import torch

from gabp_sparse_inv import (
    condition_number,
    grid_edges,
    random_spd_chain,
    random_spd_laplacian,
    random_spd_star,
    selected_inverse_junction,
)

EPS64 = torch.finfo(torch.float64).eps


def test_generated_matrix_is_spd():
    bt = random_spd_chain(10, 4, seed=0, diag_load=1.0)
    eig = torch.linalg.eigvalsh(bt.to_dense())
    assert (eig > 0).all()


def test_on_pattern():
    # Reconstructing the dense matrix and projecting back is a no-op for an
    # in-pattern matrix: off-tridiagonal blocks must be exactly zero.
    L, b = 8, 3
    bt = random_spd_chain(L, b, seed=1, diag_load=1.0)
    dense = bt.to_dense()
    for i in range(L):
        for j in range(L):
            if abs(i - j) > 1:
                block = dense[i * b:(i + 1) * b, j * b:(j + 1) * b]
                assert torch.count_nonzero(block) == 0


def test_kappa_monotone_decreasing_in_diag_load():
    # Deterministic for a fixed base: kappa(A0 + cI) is strictly decreasing in c.
    loads = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
    kappas = [float(condition_number(random_spd_chain(12, 3, seed=2, diag_load=c)))
              for c in loads]
    for a, b in zip(kappas, kappas[1:]):
        assert b < a, f"kappa not decreasing: {kappas}"


def test_seed_determinism():
    a = random_spd_chain(6, 4, seed=42, diag_load=1.0)
    b = random_spd_chain(6, 4, seed=42, diag_load=1.0)
    c = random_spd_chain(6, 4, seed=43, diag_load=1.0)
    assert torch.equal(a.diag, b.diag) and torch.equal(a.lower, b.lower)
    assert not torch.equal(a.diag, c.diag)


def test_validate_passes():
    bt = random_spd_chain(5, 4, seed=3, diag_load=1.0)
    bt.validate()  # should not raise


def test_batched_generation():
    bt = random_spd_chain(7, 3, seed=4, diag_load=1.0, batch_shape=(2, 3))
    assert bt.diag.shape == (2, 3, 7, 3, 3)
    assert bt.lower.shape == (2, 3, 6, 3, 3)


# --------------------------------------------------------------------------- #
# Star generator.
# --------------------------------------------------------------------------- #
def test_star_is_spd():
    st = random_spd_star(10, 4, seed=0, diag_load=1.0)
    eig = torch.linalg.eigvalsh(st.to_dense())
    assert (eig > 0).all()


def test_star_on_pattern():
    # Leaf--leaf blocks must be exactly zero in the generated matrix.
    K, b = 8, 3
    st = random_spd_star(K, b, seed=1, diag_load=1.0)
    dense = st.to_dense()
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if i != j:
                block = dense[i * b:(i + 1) * b, j * b:(j + 1) * b]
                assert torch.count_nonzero(block) == 0


def test_star_kappa_monotone_decreasing_in_diag_load():
    loads = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
    kappas = [float(condition_number(random_spd_star(12, 3, seed=2, diag_load=c)))
              for c in loads]
    for a, b in zip(kappas, kappas[1:]):
        assert b < a, f"kappa not decreasing: {kappas}"


def test_star_seed_determinism():
    a = random_spd_star(6, 4, seed=42, diag_load=1.0)
    b = random_spd_star(6, 4, seed=42, diag_load=1.0)
    c = random_spd_star(6, 4, seed=43, diag_load=1.0)
    assert torch.equal(a.center, b.center) and torch.equal(a.coupling, b.coupling)
    assert not torch.equal(a.center, c.center)


def test_star_batched_generation():
    st = random_spd_star(7, 3, seed=4, diag_load=1.0, batch_shape=(2, 3))
    assert st.center.shape == (2, 3, 3, 3)
    assert st.leaf_diag.shape == (2, 3, 7, 3, 3)
    assert st.coupling.shape == (2, 3, 7, 3, 3)


# --------------------------------------------------------------------------- #
# Laplacian SPD graph generator (well-scaled, high-conditioning).
# --------------------------------------------------------------------------- #
def test_laplacian_is_spd():
    edges = grid_edges(3, 3)
    for eps in (1.0, 1e-1, 1e-2, 1e-3):
        sp = random_spd_laplacian(9, edges, 2, eps=eps, seed=0)
        eig = torch.linalg.eigvalsh(sp.to_dense())
        assert (eig > 0).all(), f"not SPD at eps={eps}"


def test_laplacian_kappa_monotone_in_eps():
    # Smaller eps -> larger condition number (the knob in the opposite direction
    # from diag_load): kappa(eps=1e-3) > kappa(eps=1e-1) strictly.
    edges = grid_edges(4, 4)
    epses = [1.0, 1e-1, 1e-2, 1e-3]
    kappas = [float(condition_number(random_spd_laplacian(16, edges, 1, eps=e, seed=1)))
              for e in epses]
    for a, b in zip(kappas, kappas[1:]):
        assert b > a, f"kappa not increasing as eps shrinks: {kappas}"


def test_laplacian_pattern_no_fill_baked_in():
    # The returned edge_index must equal the canonicalized input pattern (i > j),
    # not a filled pattern.
    edges = grid_edges(3, 3)
    sp = random_spd_laplacian(9, edges, 2, eps=1e-2, seed=2)
    want = {(max(int(a), int(c)), min(int(a), int(c)))
            for a, c in zip(edges[0].tolist(), edges[1].tolist())}
    got = {(int(sp.edge_index[0, k]), int(sp.edge_index[1, k]))
           for k in range(sp.num_edges)}
    assert got == want


def test_laplacian_oracle_on_grid():
    edges = grid_edges(3, 3)
    sp = random_spd_laplacian(9, edges, 2, eps=1e-2, seed=3)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val, check=True)
    A = sp.to_dense().to(torch.float64)
    inv = torch.linalg.inv(A)
    b = sp.block_size
    kappa = float(condition_number(sp))
    atol = max(1e3 * kappa * EPS64, 1e-10)
    for v in range(sp.num_nodes):
        blk = inv[v * b:(v + 1) * b, v * b:(v + 1) * b]
        assert torch.allclose(G_diag[v].to(torch.float64), blk, atol=atol), f"diag {v}"
    ij = S_index.tolist()
    for k in range(S_index.shape[1]):
        i, j = ij[0][k], ij[1][k]
        blk = inv[i * b:(i + 1) * b, j * b:(j + 1) * b]
        assert torch.allclose(G_lower[k].to(torch.float64), blk, atol=atol), f"edge {(i, j)}"


def test_laplacian_seed_determinism():
    edges = grid_edges(2, 3)
    a = random_spd_laplacian(6, edges, 2, eps=1e-2, seed=42)
    b = random_spd_laplacian(6, edges, 2, eps=1e-2, seed=42)
    c = random_spd_laplacian(6, edges, 2, eps=1e-2, seed=43)
    assert torch.equal(a.diag, b.diag) and torch.equal(a.edge_val, b.edge_val)
    assert not torch.equal(a.edge_val, c.edge_val)


def test_laplacian_batched_and_scalar_block():
    edges = grid_edges(2, 3)
    sp = random_spd_laplacian(6, edges, 1, eps=1e-2, seed=4, batch_shape=(2,))
    assert sp.diag.shape == (2, 6, 1, 1)
    assert sp.edge_val.shape == (2, sp.num_edges, 1, 1)
    eig = torch.linalg.eigvalsh(sp.to_dense())
    assert (eig > 0).all()


def test_laplacian_no_edges_is_block_diagonal():
    sp = random_spd_laplacian(4, [], 2, eps=0.5, seed=5)
    assert sp.num_edges == 0
    eye = torch.eye(2, dtype=torch.float64)
    for v in range(4):
        assert torch.allclose(sp.diag[v], 0.5 * eye, atol=1e-12)


def test_laplacian_rejects_nonpositive_eps():
    import pytest
    with pytest.raises(ValueError):
        random_spd_laplacian(4, grid_edges(2, 2), 2, eps=0.0)
