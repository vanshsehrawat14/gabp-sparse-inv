"""Correctness gates for the general non-symmetric (LU / Erisman-Tinney) selected inverse.

``selected_inverse_nonsym_junction`` is the headline open rung: the selected inverse of a
general non-symmetric block matrix on an arbitrary structurally-symmetric pattern, on the
*filled* set ``S`` (both orientations). Gates: a dense ``torch.linalg.inv`` oracle on loopy
grids / random graphs / a zero-fill tree, the symmetric special case reducing block-for-block
to the SPD junction kernel, an oracle-free ``A G = I`` residual, batching, ``compute_dtype``,
and ``gradcheck`` of the differentiable path.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    grid_edges,
    junction_solve,
    random_spd_graph,
    selected_inverse_junction,
)
from gabp_sparse_inv.nonsym_junction import (
    nonsym_junction_solve,
    selected_inverse_nonsym_junction,
    selinv_nonsym_junction,
)

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


# --------------------------------------------------------------------------- #
# Fixtures: random non-symmetric, diagonally-dominant (no-pivot-safe) problems.
# --------------------------------------------------------------------------- #
def _make_nonsym_graph(n, edges, b, *, seed, diag_load=4.0, off=0.3):
    g = torch.Generator().manual_seed(seed)
    if not isinstance(edges, torch.Tensor):
        edges = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.zeros((2, 0), dtype=torch.long)
    pairs = sorted({(max(int(edges[0, k]), int(edges[1, k])), min(int(edges[0, k]), int(edges[1, k])))
                    for k in range(edges.shape[1]) if edges[0, k] != edges[1, k]})
    ei = torch.tensor(list(zip(*pairs)), dtype=torch.long) if pairs else torch.zeros((2, 0), dtype=torch.long)
    m = ei.shape[1]
    eye = torch.eye(b, dtype=torch.float64)
    diag = diag_load * eye + 0.1 * torch.randn(*(()), n, b, b, generator=g, dtype=torch.float64)
    el = off * torch.randn(m, b, b, generator=g, dtype=torch.float64)
    eu = off * torch.randn(m, b, b, generator=g, dtype=torch.float64)
    return diag, ei, el, eu


def _assemble(diag, ei, el, eu):
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    A = diag.new_zeros((*diag.shape[:-3], N, N))
    for v in range(n):
        A[..., v * b:(v + 1) * b, v * b:(v + 1) * b] = diag[..., v, :, :]
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        A[..., i * b:(i + 1) * b, j * b:(j + 1) * b] = el[..., k, :, :]
        A[..., j * b:(j + 1) * b, i * b:(i + 1) * b] = eu[..., k, :, :]
    return A


def _oracle_err(diag, ei, el, eu, Gd, Si, Gl, Gu):
    n, b = diag.shape[-3], diag.shape[-1]
    inv = torch.linalg.inv(_assemble(diag, ei, el, eu).to(torch.float64))
    err = max(float((Gd[..., v, :, :].to(torch.float64) - inv[..., v * b:(v + 1) * b, v * b:(v + 1) * b]).abs().max())
              for v in range(n))
    for k in range(Si.shape[1]):
        i, j = int(Si[0, k]), int(Si[1, k])
        err = max(err, float((Gl[..., k, :, :].to(torch.float64) - inv[..., i * b:(i + 1) * b, j * b:(j + 1) * b]).abs().max()))
        err = max(err, float((Gu[..., k, :, :].to(torch.float64) - inv[..., j * b:(j + 1) * b, i * b:(i + 1) * b]).abs().max()))
    return err


# --------------------------------------------------------------------------- #
# Dense oracle: loopy grids (genuine fill), random graphs, a zero-fill tree.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(2, 2, 2), (3, 3, 2), (4, 3, 1), (2, 5, 2)])
def test_dense_oracle_grid(rows, cols, b):
    diag, ei, el, eu = _make_nonsym_graph(rows * cols, grid_edges(rows, cols), b, seed=rows * 10 + cols)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, el, eu, check=True)
    assert Si.shape[1] > ei.shape[1]                       # loopy grid -> fill
    assert _oracle_err(diag, ei, el, eu, Gd, Si, Gl, Gu) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dense_oracle_random_graph(seed):
    n, b = 10, 2
    gen = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    edges = torch.tensor(list(zip(*pairs)), dtype=torch.long) if pairs else torch.tensor([[1], [0]])
    diag, ei, el, eu = _make_nonsym_graph(n, edges, b, seed=seed)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, el, eu)
    assert _oracle_err(diag, ei, el, eu, Gd, Si, Gl, Gu) < 1e-9


def test_dense_oracle_tree_zero_fill():
    # A path is a tree: a perfect elimination order exists, so no fill.
    n, b = 8, 2
    edges = torch.tensor([[i + 1 for i in range(n - 1)], [i for i in range(n - 1)]], dtype=torch.long)
    diag, ei, el, eu = _make_nonsym_graph(n, edges, b, seed=5)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, el, eu)
    assert Si.shape[1] == ei.shape[1]                      # zero fill
    assert _oracle_err(diag, ei, el, eu, Gd, Si, Gl, Gu) < 1e-9


# --------------------------------------------------------------------------- #
# Symmetric special case reduces block-for-block to the SPD junction kernel.
# --------------------------------------------------------------------------- #
def test_reduces_to_symmetric_junction():
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=6, diag_load=1.0)
    Gd_s, Si_s, Gl_s = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    Gd_n, Si_n, Gl_n, Gu_n = selected_inverse_nonsym_junction(
        sp.diag, sp.edge_index, sp.edge_val, sp.edge_val.mT
    )
    assert torch.equal(Si_n, Si_s)
    assert torch.allclose(Gd_n, Gd_s, atol=1e-10)
    assert torch.allclose(Gl_n, Gl_s, atol=1e-10)
    assert torch.allclose(Gu_n, Gl_s.mT, atol=1e-10)       # G_{j,i} = G_{i,j}^T when symmetric


# --------------------------------------------------------------------------- #
# Oracle-free on-pattern residual: (A G)_ii = A_ii G_ii + sum_j A_ij G_ji = I,
# from the selected blocks alone (A is sparse on the input pattern).
# --------------------------------------------------------------------------- #
def test_onpattern_residual_oracle_free():
    n, b = 9, 2
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(3, 3), b, seed=7)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, el, eu)
    lutL = {(int(Si[0, k]), int(Si[1, k])): Gl[k] for k in range(Si.shape[1])}   # G_{i,j}
    lutU = {(int(Si[0, k]), int(Si[1, k])): Gu[k] for k in range(Si.shape[1])}   # G_{j,i}
    R = [diag[i] @ Gd[i] for i in range(n)]
    ij = ei.tolist()
    for k in range(ei.shape[1]):
        i, j = ij[0][k], ij[1][k]                          # i > j
        R[i] = R[i] + el[k] @ lutU[(i, j)]                 # A_{i,j} G_{j,i}
        R[j] = R[j] + eu[k] @ lutL[(i, j)]                 # A_{j,i} G_{i,j}
    eye = torch.eye(b, dtype=torch.float64)
    for i in range(n):
        assert torch.allclose(R[i], eye, atol=1e-9), f"residual node {i}"


# --------------------------------------------------------------------------- #
# Batching / precision / autograd / edge cases.
# --------------------------------------------------------------------------- #
def test_leading_batch_matches_per_item():
    n, b, batch = 6, 2, 4
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 3), b, seed=8)
    dB = diag.expand(batch, n, b, b) + 0.01 * torch.randn(batch, n, b, b)
    lB = el.expand(batch, *el.shape) + 0.01 * torch.randn(batch, *el.shape)
    uB = eu.expand(batch, *eu.shape) + 0.01 * torch.randn(batch, *eu.shape)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(dB, ei, lB, uB)
    for k in range(batch):
        gd, si, gl, gu = selected_inverse_nonsym_junction(dB[k], ei, lB[k], uB[k])
        assert torch.allclose(Gd[k], gd, atol=1e-12) and torch.allclose(Gl[k], gl, atol=1e-12)
        assert torch.allclose(Gu[k], gu, atol=1e-12) and torch.equal(Si, si)


def test_compute_dtype_fp32_storage_fp64_compute():
    diag, ei, el, eu = _make_nonsym_graph(9, grid_edges(3, 3), 2, seed=9)
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(
        diag.float(), ei, el.float(), eu.float(), compute_dtype=torch.float64
    )
    assert Gd.dtype == torch.float32
    assert _oracle_err(diag, ei, el, eu, Gd, Si, Gl, Gu) < 1e-5


def test_gradcheck():
    n, b = 4, 2
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 2), b, seed=1)
    d = diag.clone().requires_grad_(True)
    a = el.clone().requires_grad_(True)
    c = eu.clone().requires_grad_(True)

    def f(d, a, c):
        Gd, _Si, Gl, Gu = selinv_nonsym_junction(d, ei, a, c)
        return torch.cat([Gd.reshape(-1), Gl.reshape(-1), Gu.reshape(-1)])

    assert torch.autograd.gradcheck(f, (d, a, c), atol=1e-6, rtol=1e-4)


def test_grad_flows_to_all_inputs():
    n, b = 6, 2
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 3), b, seed=2)
    d = diag.clone().requires_grad_(True)
    a = el.clone().requires_grad_(True)
    c = eu.clone().requires_grad_(True)
    Gd, _Si, Gl, Gu = selinv_nonsym_junction(d, ei, a, c)
    (Gd.pow(2).sum() + Gl.pow(2).sum() + Gu.pow(2).sum()).backward()
    for t in (d, a, c):
        assert t.grad is not None and float(t.grad.abs().sum()) > 0


def test_single_node():
    diag = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 2.0
    Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(
        diag, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, 3, 3, dtype=torch.float64),
        torch.zeros(0, 3, 3, dtype=torch.float64),
    )
    assert torch.allclose(Gd[0], torch.linalg.inv(diag[0]), atol=1e-12)
    assert Si.shape == (2, 0) and Gl.shape[-3] == 0 and Gu.shape[-3] == 0


def test_check_rejects_bad_index():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    with pytest.raises(ValueError):
        selected_inverse_nonsym_junction(
            diag, torch.tensor([[0], [1]]), torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), check=True
        )


# --------------------------------------------------------------------------- #
# Non-symmetric solve  x = A^{-1} b  (and the transpose A^{-T} b): the solve
# sibling of the selected inverse and the DEQ implicit-diff adjoint primitive.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b,k", [(3, 3, 2, 1), (3, 3, 2, 3), (2, 4, 1, 2)])
def test_solve_dense_oracle_grid(rows, cols, b, k):
    n = rows * cols
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(rows, cols), b, seed=rows * 7 + cols)
    A = _assemble(diag, ei, el, eu)
    rhs = torch.randn(n, b, k, dtype=torch.float64)
    rhs_flat = rhs.permute(2, 0, 1).reshape(k, n * b).T               # [N, k]
    x = nonsym_junction_solve(diag, ei, el, eu, rhs, check=True)
    xt = nonsym_junction_solve(diag, ei, el, eu, rhs, transpose=True)
    x_flat = x.permute(2, 0, 1).reshape(k, n * b).T
    xt_flat = xt.permute(2, 0, 1).reshape(k, n * b).T
    assert torch.allclose(x_flat, torch.linalg.solve(A, rhs_flat), atol=1e-9)
    assert torch.allclose(xt_flat, torch.linalg.solve(A.T, rhs_flat), atol=1e-9)


def test_solve_multi_rhs_equals_per_column():
    n, b, k = 9, 2, 4
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(3, 3), b, seed=11)
    rhs = torch.randn(n, b, k, dtype=torch.float64)
    x = nonsym_junction_solve(diag, ei, el, eu, rhs)
    for c in range(k):
        xc = nonsym_junction_solve(diag, ei, el, eu, rhs[..., c])
        assert torch.allclose(x[..., c], xc, atol=1e-12)


def test_solve_vector_rhs_shape_matches():
    n, b = 6, 2
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 3), b, seed=12)
    rhs = torch.randn(n, b, dtype=torch.float64)
    x = nonsym_junction_solve(diag, ei, el, eu, rhs)
    assert x.shape == (n, b)


def test_solve_reduces_to_symmetric_junction():
    # On symmetric input (edge_upper = edge_lower^T) the non-symmetric solve must match
    # junction_solve, and the transpose solve must equal the forward (A = A^T).
    sp = random_spd_graph(9, grid_edges(3, 3), 2, seed=6, diag_load=1.0)
    rhs = torch.randn(9, 2, 2, dtype=torch.float64)
    xs = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs)
    xn = nonsym_junction_solve(sp.diag, sp.edge_index, sp.edge_val, sp.edge_val.mT, rhs)
    xnt = nonsym_junction_solve(sp.diag, sp.edge_index, sp.edge_val, sp.edge_val.mT, rhs, transpose=True)
    assert torch.allclose(xn, xs, atol=1e-10)
    assert torch.allclose(xnt, xs, atol=1e-10)


def test_solve_leading_batch_matches_per_item():
    n, b, batch = 6, 2, 4
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 3), b, seed=13)
    dB = diag.expand(batch, n, b, b) + 0.01 * torch.randn(batch, n, b, b)
    lB = el.expand(batch, *el.shape) + 0.01 * torch.randn(batch, *el.shape)
    uB = eu.expand(batch, *eu.shape) + 0.01 * torch.randn(batch, *eu.shape)
    rB = torch.randn(batch, n, b, dtype=torch.float64)
    xB = nonsym_junction_solve(dB, ei, lB, uB, rB)
    for kk in range(batch):
        assert torch.allclose(xB[kk], nonsym_junction_solve(dB[kk], ei, lB[kk], uB[kk], rB[kk]), atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_solve_gradcheck(transpose):
    n, b = 4, 2
    diag, ei, el, eu = _make_nonsym_graph(n, grid_edges(2, 2), b, seed=14)
    d = diag.clone().requires_grad_(True)
    a = el.clone().requires_grad_(True)
    c = eu.clone().requires_grad_(True)
    rr = torch.randn(n, b, dtype=torch.float64, requires_grad=True)

    def f(d, a, c, rr):
        return nonsym_junction_solve(d, ei, a, c, rr, transpose=transpose)

    assert torch.autograd.gradcheck(f, (d, a, c, rr), atol=1e-6, rtol=1e-4)


def test_solve_compute_dtype_fp32_storage_fp64_compute():
    diag, ei, el, eu = _make_nonsym_graph(9, grid_edges(3, 3), 2, seed=15)
    A = _assemble(diag, ei, el, eu)
    rhs = torch.randn(9, 2, dtype=torch.float64)
    x = nonsym_junction_solve(
        diag.float(), ei, el.float(), eu.float(), rhs.float(), compute_dtype=torch.float64
    )
    assert x.dtype == torch.float32
    ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(9, 2)
    assert (x.double() - ref).abs().max() < 1e-5


def test_solve_rejects_bad_rhs():
    diag, ei, el, eu = _make_nonsym_graph(6, grid_edges(2, 3), 2, seed=16)
    with pytest.raises(ValueError):
        nonsym_junction_solve(diag, ei, el, eu, torch.randn(5, 2))      # wrong n


def test_solve_single_node():
    diag = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 2.0
    rhs = torch.randn(1, 3, dtype=torch.float64)
    x = nonsym_junction_solve(
        diag, torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0, 3, 3, dtype=torch.float64), torch.zeros(0, 3, 3, dtype=torch.float64), rhs,
    )
    assert torch.allclose(x[0], torch.linalg.solve(diag[0], rhs[0]), atol=1e-12)
