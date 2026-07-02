"""Correctness gates for the general sparse (junction-tree) selected inverse.

The kernel computes the selected inverse on the *filled* pattern S = pattern(L+L^T);
the gold check is equality to the dense inverse on S, plus a block-for-block
cross-check against the tree kernel on the zero-fill (tree) special case and a
gradcheck of the differentiable path. See docs/derivations.md §2.1, §8.4.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    BlockSparseSym,
    condition_number,
    grid_edges,
    junction_logdet,
    junction_solve,
    random_spd_graph,
    random_spd_tree,
    selected_inverse_junction,
    selected_inverse_tree,
    selinv_junction,
    tree_logdet,
    tree_solve,
)

torch.manual_seed(0)
EPS64 = torch.finfo(torch.float64).eps


def _oracle_check(sp: BlockSparseSym, G_diag, S_index, G_lower, *, atol):
    """Every returned block on S must equal the dense inverse's corresponding block."""
    A = sp.to_dense().to(torch.float64)
    inv = torch.linalg.inv(A)
    b = sp.block_size
    for v in range(sp.num_nodes):
        blk = inv[..., v * b:(v + 1) * b, v * b:(v + 1) * b]
        assert torch.allclose(G_diag[..., v, :, :].to(torch.float64), blk, atol=atol), f"diag {v}"
    ij = S_index.tolist()
    for k in range(S_index.shape[1]):
        i, j = ij[0][k], ij[1][k]
        assert i > j, "S_index must be lower-triangular (i > j)"
        blk = inv[..., i * b:(i + 1) * b, j * b:(j + 1) * b]
        assert torch.allclose(G_lower[..., k, :, :].to(torch.float64), blk, atol=atol), f"edge {(i, j)}"


# --------------------------------------------------------------------------- #
# Dense oracle on a grid (loopy: fill genuinely occurs -> the junction path).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(3, 3, 2), (4, 3, 1), (2, 5, 2)])
def test_dense_oracle_grid(rows, cols, b):
    n = rows * cols
    edges = grid_edges(rows, cols)
    sp = random_spd_graph(n, edges, b, seed=rows * 10 + cols, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val, check=True)

    # The grid is loopy, so elimination must create fill: S strictly larger than input.
    assert S_index.shape[1] > sp.num_edges, "expected fill on a loopy grid"

    kappa = float(condition_number(sp))
    _oracle_check(sp, G_diag, S_index, G_lower, atol=max(1e3 * kappa * EPS64, 1e-10))


# --------------------------------------------------------------------------- #
# Dense oracle on a random sparse graph.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dense_oracle_random_graph(seed):
    n, b = 10, 2
    gen = torch.Generator().manual_seed(seed)
    # Random Erdos-Renyi-ish graph: keep each candidate edge with prob ~0.35.
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    if not pairs:
        pairs = [(1, 0)]
    edges = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    sp = random_spd_graph(n, edges, b, seed=seed, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    kappa = float(condition_number(sp))
    _oracle_check(sp, G_diag, S_index, G_lower, atol=max(1e3 * kappa * EPS64, 1e-10))


# --------------------------------------------------------------------------- #
# Tree reduction: on a tree (zero fill) the junction kernel must agree
# block-for-block with the dedicated tree kernel -- two eliminations of one matrix.
# --------------------------------------------------------------------------- #
def test_tree_reduction_matches_tree_kernel():
    n, b = 12, 3
    bt = random_spd_tree(n, b, seed=4, diag_load=2.0, kind="random")
    plist = bt.parent.tolist()

    pairs, vals = [], []
    for v in range(n):
        p = plist[v]
        if p == -1:
            continue
        if v > p:                      # edge[v] = A_{p,v}; want A_{v,p} = A_{p,v}^T
            pairs.append((v, p))
            vals.append(bt.edge[v].mT)
        else:
            pairs.append((p, v))
            vals.append(bt.edge[v])
    edge_index = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    edge_val = torch.stack(vals, dim=0)

    G_diag, S_index, G_lower = selected_inverse_junction(bt.diag, edge_index, edge_val)
    T_diag, T_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent)

    # A tree has a zero-fill perfect elimination order; min-degree finds one.
    assert S_index.shape[1] == edge_index.shape[1], "tree must have zero fill"
    assert torch.allclose(G_diag, T_diag, atol=1e-10)

    lut = {(int(S_index[0, k]), int(S_index[1, k])): G_lower[k] for k in range(S_index.shape[1])}
    for v in range(n):
        p = plist[v]
        if p == -1:
            continue
        i, j = (v, p) if v > p else (p, v)
        Gij = lut[(i, j)]                                   # G_{i,j}
        expected = Gij if p > v else Gij.mT                 # tree stores G_{p,v}
        assert torch.allclose(T_edge[v], expected, atol=1e-10), f"edge at node {v}"


# --------------------------------------------------------------------------- #
# Differentiable path: gradcheck through the fill (a 2x2 grid is a 4-cycle).
# --------------------------------------------------------------------------- #
def test_gradcheck_with_fill():
    n, b = 4, 2
    edges = grid_edges(2, 2)
    sp = random_spd_graph(n, edges, b, seed=1, diag_load=2.0)
    diag = sp.diag.clone().requires_grad_(True)
    edge_val = sp.edge_val.clone().requires_grad_(True)

    def f(d, e):
        G_diag, _S, G_lower = selinv_junction(d, sp.edge_index, e)
        return torch.cat([G_diag.reshape(-1), G_lower.reshape(-1)])

    assert torch.autograd.gradcheck(f, (diag, edge_val), atol=1e-6, rtol=1e-4)


def test_grad_flows_to_inputs():
    n, b = 6, 2
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=2, diag_load=2.0)
    diag = sp.diag.clone().requires_grad_(True)
    edge_val = sp.edge_val.clone().requires_grad_(True)
    G_diag, _S, G_lower = selinv_junction(diag, sp.edge_index, edge_val)
    loss = torch.diagonal(G_diag, dim1=-2, dim2=-1).sum() + G_lower.pow(2).sum()
    loss.backward()
    assert diag.grad is not None and float(diag.grad.abs().sum()) > 0
    assert edge_val.grad is not None and float(edge_val.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# junction_solve: differentiable sparse SPD solve A x = b on the filled pattern.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(3, 3, 2), (4, 3, 1), (2, 5, 2)])
def test_solve_oracle_grid(rows, cols, b):
    n = rows * cols
    sp = random_spd_graph(n, grid_edges(rows, cols), b, seed=rows * 7 + cols, diag_load=1.0)
    rhs = torch.randn(n, b, dtype=torch.float64)
    x = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs, check=True)
    A = sp.to_dense().to(torch.float64)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(n, b)
    kappa = float(condition_number(sp))
    assert torch.allclose(x, x_ref, atol=max(1e2 * kappa * EPS64, 1e-10))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solve_oracle_random_graph(seed):
    n, b = 9, 2
    gen = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    if not pairs:
        pairs = [(1, 0)]
    edges = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    sp = random_spd_graph(n, edges, b, seed=seed, diag_load=1.0)
    rhs = torch.randn(n, b, dtype=torch.float64)
    x = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs)
    A = sp.to_dense().to(torch.float64)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(n, b)
    kappa = float(condition_number(sp))
    assert torch.allclose(x, x_ref, atol=max(1e2 * kappa * EPS64, 1e-10))


def test_solve_multi_rhs_matches_columns():
    n, b, k = 6, 2, 3
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=11, diag_load=1.5)
    B = torch.randn(n, b, k, dtype=torch.float64)
    X = junction_solve(sp.diag, sp.edge_index, sp.edge_val, B)
    assert X.shape == (n, b, k)
    for c in range(k):
        xc = junction_solve(sp.diag, sp.edge_index, sp.edge_val, B[..., c])
        assert torch.allclose(X[..., c], xc, atol=1e-12)


def test_solve_leading_batch_matches_per_item():
    n, b, batch = 6, 2, 4
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=12, diag_load=1.5, batch_shape=(batch,))
    rhs = torch.randn(batch, n, b, dtype=torch.float64)
    X = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs)
    for kk in range(batch):
        xk = junction_solve(sp.diag[kk], sp.edge_index, sp.edge_val[kk], rhs[kk])
        assert torch.allclose(X[kk], xk, atol=1e-12)


def test_solve_gradcheck():
    n, b = 4, 2
    sp = random_spd_graph(n, grid_edges(2, 2), b, seed=13, diag_load=2.0)
    diag = sp.diag.clone().requires_grad_(True)
    edge_val = sp.edge_val.clone().requires_grad_(True)
    rhs = torch.randn(n, b, dtype=torch.float64, requires_grad=True)

    def f(d, e, r):
        return junction_solve(d, sp.edge_index, e, r)

    assert torch.autograd.gradcheck(f, (diag, edge_val, rhs), atol=1e-6, rtol=1e-4)


def test_solve_tree_reduction_matches_tree_solve():
    n, b = 12, 3
    bt = random_spd_tree(n, b, seed=14, diag_load=2.0, kind="random")
    plist = bt.parent.tolist()
    pairs, vals = [], []
    for v in range(n):
        p = plist[v]
        if p == -1:
            continue
        if v > p:
            pairs.append((v, p)); vals.append(bt.edge[v].mT)
        else:
            pairs.append((p, v)); vals.append(bt.edge[v])
    edge_index = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    edge_val = torch.stack(vals, dim=0)
    rhs = torch.randn(n, b, dtype=torch.float64)
    x_j = junction_solve(bt.diag, edge_index, edge_val, rhs)
    x_t = tree_solve(bt.diag, bt.edge, bt.parent, rhs)
    assert torch.allclose(x_j, x_t, atol=1e-10)


def test_solve_single_node():
    sp = random_spd_graph(1, [], 3, seed=15, diag_load=1.0)
    rhs = torch.randn(1, 3, dtype=torch.float64)
    x = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs)
    x_ref = torch.linalg.solve(sp.diag[0], rhs[0])
    assert torch.allclose(x[0], x_ref, atol=1e-11)


def test_solve_rejects_bad_rhs_shape():
    sp = random_spd_graph(4, grid_edges(2, 2), 2, seed=16, diag_load=1.0)
    with pytest.raises(ValueError):
        junction_solve(sp.diag, sp.edge_index, sp.edge_val, torch.randn(4, 3))  # wrong block size


# --------------------------------------------------------------------------- #
# junction_logdet: log det A from the LDL^T factor (differentiable).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(3, 3, 2), (4, 3, 1), (2, 5, 2)])
def test_logdet_oracle_grid(rows, cols, b):
    n = rows * cols
    sp = random_spd_graph(n, grid_edges(rows, cols), b, seed=rows * 5 + cols, diag_load=1.0)
    ld = junction_logdet(sp.diag, sp.edge_index, sp.edge_val, check=True)
    ref = torch.logdet(sp.to_dense().to(torch.float64))
    assert torch.allclose(ld, ref, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_logdet_oracle_random_graph(seed):
    n, b = 9, 2
    gen = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    if not pairs:
        pairs = [(1, 0)]
    edges = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    sp = random_spd_graph(n, edges, b, seed=seed, diag_load=1.0)
    ld = junction_logdet(sp.diag, sp.edge_index, sp.edge_val)
    ref = torch.logdet(sp.to_dense().to(torch.float64))
    assert torch.allclose(ld, ref, atol=1e-9, rtol=1e-9)


def test_logdet_order_invariant():
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=6, diag_load=1.0)
    a = junction_logdet(sp.diag, sp.edge_index, sp.edge_val)
    rev = list(range(n - 1, -1, -1))
    c = junction_logdet(sp.diag, sp.edge_index, sp.edge_val, order=rev)
    assert torch.allclose(a, c, atol=1e-10)


def test_logdet_gradcheck():
    n, b = 4, 2
    sp = random_spd_graph(n, grid_edges(2, 2), b, seed=7, diag_load=2.0)
    diag = sp.diag.clone().requires_grad_(True)
    edge_val = sp.edge_val.clone().requires_grad_(True)

    def f(d, e):
        return junction_logdet(d, sp.edge_index, e)

    assert torch.autograd.gradcheck(f, (diag, edge_val), atol=1e-6, rtol=1e-4)


def test_logdet_leading_batch():
    n, b, batch = 6, 2, 3
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=8, diag_load=1.5, batch_shape=(batch,))
    ld = junction_logdet(sp.diag, sp.edge_index, sp.edge_val)
    assert ld.shape == (batch,)
    ref = torch.logdet(sp.to_dense().to(torch.float64))
    assert torch.allclose(ld, ref, atol=1e-9, rtol=1e-9)


def test_logdet_tree_reduction_matches_tree_logdet():
    n, b = 12, 3
    bt = random_spd_tree(n, b, seed=9, diag_load=2.0, kind="random")
    plist = bt.parent.tolist()
    pairs, vals = [], []
    for v in range(n):
        p = plist[v]
        if p == -1:
            continue
        if v > p:
            pairs.append((v, p)); vals.append(bt.edge[v].mT)
        else:
            pairs.append((p, v)); vals.append(bt.edge[v])
    edge_index = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    edge_val = torch.stack(vals, dim=0)
    ld_j = junction_logdet(bt.diag, edge_index, edge_val)
    ld_t = tree_logdet(bt.diag, bt.edge, bt.parent)
    assert torch.allclose(ld_j, ld_t, atol=1e-10)


def test_logdet_single_node_and_no_edges():
    sp = random_spd_graph(1, [], 3, seed=10, diag_load=1.0)
    assert torch.allclose(junction_logdet(sp.diag, sp.edge_index, sp.edge_val),
                          torch.logdet(sp.diag[0]), atol=1e-11)
    sp2 = random_spd_graph(5, [], 2, seed=11, diag_load=1.0)   # block diagonal
    ref = sum(torch.logdet(sp2.diag[v]) for v in range(5))
    assert torch.allclose(junction_logdet(sp2.diag, sp2.edge_index, sp2.edge_val), ref, atol=1e-10)


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_single_node():
    sp = random_spd_graph(1, [], 4, seed=1, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    ref = torch.linalg.inv(sp.diag[0])
    assert torch.allclose(G_diag[0], ref, atol=1e-11)
    assert S_index.shape == (2, 0) and G_lower.shape[-3] == 0


def test_no_edges_is_block_diagonal():
    n, b = 5, 3
    sp = random_spd_graph(n, [], b, seed=3, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    assert S_index.shape[1] == 0
    for v in range(n):
        assert torch.allclose(G_diag[v], torch.linalg.inv(sp.diag[v]), atol=1e-11)


def test_scalar_blocks_b1():
    n = 9
    sp = random_spd_graph(n, grid_edges(3, 3), 1, seed=5, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    kappa = float(condition_number(sp))
    _oracle_check(sp, G_diag, S_index, G_lower, atol=max(1e3 * kappa * EPS64, 1e-10))


def test_leading_batch_matches_per_item():
    n, b, batch = 6, 2, 4
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=7, diag_load=1.5, batch_shape=(batch,))
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    for k in range(batch):
        d_k, s_k, l_k = selected_inverse_junction(sp.diag[k], sp.edge_index, sp.edge_val[k])
        assert torch.allclose(G_diag[k], d_k, atol=1e-12)
        assert torch.allclose(G_lower[k], l_k, atol=1e-12)
        assert torch.equal(S_index, s_k)


def test_explicit_order_matches_min_degree_result():
    # The selected inverse is elimination-order invariant: a different (valid) order
    # gives the same blocks on the pattern it returns.
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=8, diag_load=1.0)
    Gd_a, Si_a, _ = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    rev = list(range(n - 1, -1, -1))
    Gd_b, Si_b, Gl_b = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val, order=rev)
    assert torch.allclose(Gd_a, Gd_b, atol=1e-10)
    # Both are correct selected inverses (compare the second to the oracle on its own S).
    kappa = float(condition_number(sp))
    _oracle_check(sp, Gd_b, Si_b, Gl_b, atol=max(1e3 * kappa * EPS64, 1e-10))


# --------------------------------------------------------------------------- #
# Precision: fp32 loose bound.
# --------------------------------------------------------------------------- #
def test_fp32_relative_accuracy():
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=4, diag_load=3.0)
    d32 = sp.diag.to(torch.float32)
    e32 = sp.edge_val.to(torch.float32)
    G_diag, S_index, G_lower = selected_inverse_junction(d32, sp.edge_index, e32)
    A = sp.to_dense().to(torch.float64)
    inv = torch.linalg.inv(A)
    # Worst on-pattern block error, relative to the inverse scale.
    scale = inv.abs().amax()
    err = 0.0
    for v in range(n):
        blk = inv[v * b:(v + 1) * b, v * b:(v + 1) * b]
        err = max(err, float((G_diag[v].to(torch.float64) - blk).abs().amax()))
    assert err < 1e-4 * float(scale)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def test_validate_rejects_nonsymmetric_diag():
    sp = random_spd_graph(5, grid_edges(1, 5), 3, seed=0, diag_load=1.0)
    sp.diag[0, 0, 1] += 1.0
    with pytest.raises(ValueError):
        selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val, check=True)


def test_validate_rejects_upper_triangular_index():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge_index = torch.tensor([[0], [1]])         # i=0 < j=1: not lower-triangular
    edge_val = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError):
        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()


def test_validate_rejects_out_of_range_index():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge_index = torch.tensor([[9], [0]])
    edge_val = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError):
        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()


def test_validate_rejects_duplicate_edges():
    diag = torch.eye(2).expand(3, 2, 2).clone()
    edge_index = torch.tensor([[1, 1], [0, 0]])
    edge_val = torch.zeros(2, 2, 2)
    with pytest.raises(ValueError):
        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()


# --------------------------------------------------------------------------- #
# A2: compute_dtype (store-low / compute path) + informative SPD-failure message.
# --------------------------------------------------------------------------- #
def _grid_problem(rows=3, cols=3, b=2, seed=7, diag_load=1.0):
    n = rows * cols
    sp = random_spd_graph(n, grid_edges(rows, cols), b, seed=seed, diag_load=diag_load)
    return sp


def test_compute_dtype_none_equals_native():
    sp = _grid_problem()
    a = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    c = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val, compute_dtype=None)
    assert torch.equal(a[0], c[0]) and torch.equal(a[2], c[2])


def test_compute_dtype_fp32_storage_fp64_compute_matches_oracle():
    # fp32 storage + fp64 compute: the result (fp32) must match the dense inverse of the
    # fp32-rounded matrix to fp32 storage precision -- confirms the upcast/cast-back path.
    sp = _grid_problem(diag_load=1.0)
    diag32 = sp.diag.to(torch.float32)
    edge32 = sp.edge_val.to(torch.float32)
    G_diag, S_index, G_lower = selected_inverse_junction(
        diag32, sp.edge_index, edge32, compute_dtype=torch.float64)
    assert G_diag.dtype == torch.float32 and G_lower.dtype == torch.float32
    A32 = BlockSparseSym(diag=diag32, edge_index=sp.edge_index, edge_val=edge32).to_dense().to(torch.float64)
    inv = torch.linalg.inv(A32)
    b = sp.block_size
    for v in range(sp.num_nodes):
        blk = inv[v * b:(v + 1) * b, v * b:(v + 1) * b]
        assert torch.allclose(G_diag[v].to(torch.float64), blk, atol=1e-5), f"diag {v}"


def test_compute_dtype_bf16_storage_runs():
    # bf16 has no native CPU Cholesky; the store-low/compute-fp32 path must still run and
    # return bf16, landing in a loose neighbourhood of the oracle (bf16 storage is coarse).
    sp = _grid_problem(diag_load=2.0)
    diagbf = sp.diag.to(torch.bfloat16)
    edgebf = sp.edge_val.to(torch.bfloat16)
    G_diag, S_index, G_lower = selected_inverse_junction(
        diagbf, sp.edge_index, edgebf, compute_dtype=torch.float32)
    assert G_diag.dtype == torch.bfloat16
    inv = torch.linalg.inv(sp.to_dense().to(torch.float64))
    b = sp.block_size
    err = max(float((G_diag[v].to(torch.float64) - inv[v * b:(v + 1) * b, v * b:(v + 1) * b]).abs().max())
              for v in range(sp.num_nodes))
    assert err < 5e-1                                   # coarse sanity only


def test_solve_and_logdet_compute_dtype_match_oracle():
    sp = _grid_problem(diag_load=1.0)
    diag32, edge32 = sp.diag.to(torch.float32), sp.edge_val.to(torch.float32)
    A32 = BlockSparseSym(diag=diag32, edge_index=sp.edge_index, edge_val=edge32).to_dense().to(torch.float64)
    n, b = sp.num_nodes, sp.block_size
    rhs = torch.randn(n, b, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    x = junction_solve(diag32, sp.edge_index, edge32, rhs, compute_dtype=torch.float64)
    assert x.dtype == torch.float32
    x_ref = torch.linalg.solve(A32, rhs.to(torch.float64).reshape(-1)).reshape(n, b)
    assert torch.allclose(x.to(torch.float64), x_ref, atol=1e-4)
    ld = junction_logdet(diag32, sp.edge_index, edge32, compute_dtype=torch.float64)
    assert ld.dtype == torch.float32
    assert abs(float(ld) - float(torch.logdet(A32))) < 1e-3


def test_non_spd_input_names_failing_pivot():
    # A clearly non-SPD diagonal must raise a LinAlgError that *names* the pivot block,
    # not a bare backend error -- the informative-error half of A2.
    n, b = 4, 2
    diag = torch.eye(b).expand(n, b, b).clone()
    diag[2] = -5.0 * torch.eye(b)                       # node 2 indefinite
    edges = torch.tensor([[1, 2, 3], [0, 1, 2]])
    edge_val = 0.1 * torch.randn(3, b, b, generator=torch.Generator().manual_seed(0))
    with pytest.raises(torch.linalg.LinAlgError) as ei:
        selected_inverse_junction(diag, edges, edge_val)
    assert "pivot block" in str(ei.value) and "D_" in str(ei.value)


# --------------------------------------------------------------------------- #
# Trace identities read off the selected inverse (the C3 one-liner; APPLICATIONS.md).
#   trace(A^-1)   = sum_v tr(G_vv)
#   trace(A^-1 B) = <G_diag, Bd> + 2 <G_lower, Bl>   for symmetric B on the filled pattern S
# --------------------------------------------------------------------------- #
def test_trace_inverse_and_trace_inverse_times_b():
    sp = _grid_problem(diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    inv = torch.linalg.inv(sp.to_dense().to(torch.float64))

    tr_Ainv = torch.diagonal(G_diag, dim1=-2, dim2=-1).sum((-1, -2))
    assert torch.allclose(tr_Ainv, torch.trace(inv), atol=1e-10)

    n, b = sp.num_nodes, sp.block_size
    g = torch.Generator().manual_seed(3)
    Bd = torch.randn(n, b, b, generator=g, dtype=torch.float64)
    Bd = Bd + Bd.mT                                          # symmetric diagonal blocks
    mS = S_index.shape[1]
    Bl = torch.randn(mS, b, b, generator=g, dtype=torch.float64)
    # Dense B on the same filled pattern S (symmetric).
    N = n * b
    Bm = torch.zeros(N, N, dtype=torch.float64)
    for v in range(n):
        Bm[v * b:(v + 1) * b, v * b:(v + 1) * b] = Bd[v]
    sij = S_index.tolist()
    for k in range(mS):
        i, j = sij[0][k], sij[1][k]
        Bm[i * b:(i + 1) * b, j * b:(j + 1) * b] = Bl[k]
        Bm[j * b:(j + 1) * b, i * b:(i + 1) * b] = Bl[k].mT

    tr_AinvB = (G_diag * Bd).sum() + 2.0 * (G_lower * Bl).sum()
    assert torch.allclose(tr_AinvB, torch.trace(inv @ Bm), atol=1e-10)


# --------------------------------------------------------------------------- #
# Oracle-free on-pattern residual: A G = I checked block-wise from the selected
# blocks alone (no dense inverse). A is sparse on the *input* pattern, so the
# diagonal block of A A^{-1} closes over only the selected blocks:
#   (A G)_ii = A_ii G_ii + sum_{input nbr j} A_ij G_ji = I.
# This is the junction analogue of the SPD chain/star/tree residual checks.
# --------------------------------------------------------------------------- #
def test_onpattern_residual_oracle_free():
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=23, diag_load=1.0)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    lut = {(int(S_index[0, k]), int(S_index[1, k])): G_lower[k] for k in range(S_index.shape[1])}

    def Gblk(a, c):                                   # G_{a,c}; A^{-1} is symmetric
        if a == c:
            return G_diag[a]
        return lut[(a, c)] if (a, c) in lut else lut[(c, a)].mT

    R = [sp.diag[i] @ G_diag[i] for i in range(n)]
    ij = sp.edge_index.tolist()
    for k in range(sp.edge_index.shape[1]):
        i, j = ij[0][k], ij[1][k]                     # i > j
        Aij = sp.edge_val[k]
        R[i] = R[i] + Aij @ Gblk(j, i)                # A_ij G_ji
        R[j] = R[j] + Aij.mT @ Gblk(i, j)             # A_ji G_ij
    eye = torch.eye(b, dtype=torch.float64)
    atol = max(1e3 * float(condition_number(sp)) * EPS64, 1e-10)
    for i in range(n):
        assert torch.allclose(R[i], eye, atol=atol), f"residual node {i}"


def test_solve_order_invariant():
    # The solve x = A^{-1} b is independent of the elimination order (only fill changes).
    n, b = 9, 2
    sp = random_spd_graph(n, grid_edges(3, 3), b, seed=24, diag_load=1.0)
    rhs = torch.randn(n, b, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    x_a = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs)
    x_b = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs, order=list(range(n - 1, -1, -1)))
    assert torch.allclose(x_a, x_b, atol=1e-10)


def test_compute_dtype_backward_matches_fp64_reference():
    # fp32 storage + fp64 compute: the *backward* (the flagged gap -- gradcheck itself needs
    # fp64 inputs) must match the pure-fp64 gradient at the same fp32-rounded inputs, to
    # ~fp32 precision. A broken mixed-precision backward would be off by O(1), not 1e-3.
    sp = _grid_problem(diag_load=1.0)
    d32, e32 = sp.diag.to(torch.float32), sp.edge_val.to(torch.float32)

    d64 = d32.double().clone().requires_grad_(True)
    e64 = e32.double().clone().requires_grad_(True)
    Gd, _S, Gl = selinv_junction(d64, sp.edge_index, e64)
    (Gd.pow(2).sum() + Gl.pow(2).sum()).backward()

    d = d32.clone().requires_grad_(True)
    e = e32.clone().requires_grad_(True)
    Gd2, _S2, Gl2 = selected_inverse_junction(d, sp.edge_index, e, compute_dtype=torch.float64)
    (Gd2.pow(2).sum() + Gl2.pow(2).sum()).backward()

    assert d.grad.dtype == torch.float32 and e.grad.dtype == torch.float32
    assert float((d.grad.double() - d64.grad).norm()) < 1e-2 * float(d64.grad.norm())
    assert float((e.grad.double() - e64.grad).norm()) < 1e-2 * float(e64.grad.norm())
