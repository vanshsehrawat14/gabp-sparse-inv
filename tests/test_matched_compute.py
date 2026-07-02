"""Tests for the matched-compute B2 solvers and the structure-exact FLOP counter."""

import torch

from gabp_sparse_inv.generators import grid_edges
from gabp_sparse_inv.bench.matched_compute import (
    spd_matvec,
    jacobi_solve,
    cg_solve,
    neumann_series,
    matvec_flops,
    factor_flops,
    exact_solve_flops,
    exact_solve_adjoint_flops,
    iter_flops,
    iter_solve_adjoint_flops,
    n_match,
    neumann_term_flops,
    k_match_neumann,
)


def _scalar_laplacian(rows, cols, *, eps=0.3, seed=0):
    """Random SPD scalar grid Laplacian A = L(w) + eps I in the scalar-node rep."""
    ei = grid_edges(rows, cols)
    i_idx, j_idx = ei[0], ei[1]
    n = rows * cols
    g = torch.Generator().manual_seed(seed)
    w = 0.5 + torch.rand(ei.shape[1], generator=g, dtype=torch.float64)
    diag = torch.full((n,), eps, dtype=torch.float64)
    diag = diag.index_add(0, i_idx, w).index_add(0, j_idx, w)
    return ei, i_idx, j_idx, w, diag  # edge_val = -w


def _dense(diag, i_idx, j_idx, edge_val):
    n = diag.shape[0]
    A = torch.diag(diag.clone())
    for k in range(i_idx.shape[0]):
        A[i_idx[k], j_idx[k]] = edge_val[k]
        A[j_idx[k], i_idx[k]] = edge_val[k]
    return A


def test_solvers_converge_to_dense_solve():
    ei, i_idx, j_idx, w, diag = _scalar_laplacian(4, 4, eps=0.5)
    edge_val = -w
    A = _dense(diag, i_idx, j_idx, edge_val)
    b = torch.randn(diag.shape[0], generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    x_true = torch.linalg.solve(A, b)
    x_j = jacobi_solve(diag, i_idx, j_idx, edge_val, b, 400)
    x_c = cg_solve(diag, i_idx, j_idx, edge_val, b, 60)
    assert (x_j - x_true).abs().max() < 1e-6
    assert (x_c - x_true).abs().max() < 1e-6


def test_neumann_series_matches_explicit_sum():
    M = torch.tensor([[0.2, 0.1], [0.0, 0.3]], dtype=torch.float64)
    b = torch.tensor([1.0, -2.0], dtype=torch.float64)
    got = neumann_series(lambda x: M @ x, b, 4)
    want = b + M @ b + M @ (M @ b) + M @ (M @ (M @ b))
    assert (got - want).abs().max() < 1e-12


def test_matvec_matches_dense():
    ei, i_idx, j_idx, w, diag = _scalar_laplacian(3, 3)
    edge_val = -w
    A = _dense(diag, i_idx, j_idx, edge_val)
    x = torch.randn(diag.shape[0], generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    assert (spd_matvec(diag, i_idx, j_idx, edge_val, x) - A @ x).abs().max() < 1e-12


def test_matvec_reads_all_edges_global_not_local():
    """Guard: B2's operator is GLOBAL -- the matvec must depend on every edge."""
    ei, i_idx, j_idx, w, diag = _scalar_laplacian(3, 3)
    edge_val = -w
    x = torch.ones(diag.shape[0], dtype=torch.float64)
    base = spd_matvec(diag, i_idx, j_idx, edge_val, x)
    for k in range(edge_val.shape[0]):
        bumped = edge_val.clone()
        bumped[k] += 1.0
        assert not torch.allclose(spd_matvec(diag, i_idx, j_idx, bumped, x), base), f"edge {k} ignored"


def test_fill_scaling_grid_vs_path():
    """factor_flops must grow with fill: a loopy grid fills, a path (tree) does not."""
    # path graph on n nodes (a tree -> zero fill -> factor ~ n)
    n = 16
    path = torch.tensor([[i for i in range(1, n)], [i - 1 for i in range(1, n)]], dtype=torch.long)
    f_path = factor_flops(n, path, b=1)
    # 4x4 grid (loopy, same n) -> fill -> strictly more factor work
    ei = grid_edges(4, 4)
    f_grid = factor_flops(16, ei, b=1)
    assert f_grid > f_path


def test_n_match_positive_and_grows_with_size():
    """N_match ~ Theta(sqrt(n)) on grids: a bigger grid affords B2 more matched iterations."""
    k_small = n_match(16, grid_edges(4, 4), b=1, k=1, method="jacobi")
    k_big = n_match(64, grid_edges(8, 8), b=1, k=1, method="jacobi")
    assert k_small >= 1 and k_big >= 1
    assert k_big > k_small


def test_exact_solve_exceeds_one_matvec():
    ei = grid_edges(4, 4)
    assert exact_solve_flops(16, ei, b=1, k=1) > matvec_flops(16, ei.shape[1], b=1, k=1)
    assert exact_solve_adjoint_flops(16, ei, b=1, k=1) > exact_solve_flops(16, ei, b=1, k=1)


def test_iter_flops_linear_in_K():
    ei = grid_edges(4, 4)
    f1 = iter_flops(16, ei.shape[1], 1, method="jacobi")
    f10 = iter_flops(16, ei.shape[1], 10, method="jacobi")
    assert abs(f10 - 10 * f1) < 1e-9
    assert iter_solve_adjoint_flops(16, ei.shape[1], 3, method="jacobi") == 2 * iter_flops(
        16, ei.shape[1], 3, method="jacobi"
    )


def test_n_match_defaults_to_solve_plus_adjoint_anchor():
    ei = grid_edges(4, 4)
    k_train = n_match(16, ei, b=1, k=1, method="jacobi")
    k_fwd = n_match(16, ei, b=1, k=1, method="jacobi", target="solve")
    assert k_train >= k_fwd


def test_neumann_k_match_positive():
    n = 32
    assert k_match_neumann(n, n - 1, b=1, k=1) >= 1
    assert neumann_term_flops(n, n - 1, b=1, k=1) > 0
