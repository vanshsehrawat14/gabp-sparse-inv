"""Correctness gates for the level-set *batched* junction selected inverse (T6).

The batched path (`selected_inverse_junction(..., batched=True)` /
`selinv_junction(..., batched=True)`) reschedules the per-node reference loop into one
sweep over elimination *levels* (antichains of mutually non-adjacent filled-graph nodes),
batching each level's clique work. The reference per-node kernel stays the correctness
oracle: every test here asserts the batched result equals the reference **block-for-block**
(`~1e-10`), forward and (via autograd) backward, plus a `gradcheck` of the batched path and
a scaling note on the loop-count reduction. See docs/ROADMAP.md (T6) / docs/derivations.md §8.4.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    grid_edges,
    random_spd_graph,
    random_spd_tree,
    selected_inverse_junction,
    selinv_junction,
)
from gabp_sparse_inv.junction import _batched_symbolic

torch.manual_seed(0)


def _assert_matches_reference(diag, edge_index, edge_val, *, order=None, atol=1e-10):
    """Batched forward must equal the per-node reference block-for-block (same S order)."""
    Gd_a, Si_a, Gl_a = selected_inverse_junction(diag, edge_index, edge_val, order=order)
    Gd_b, Si_b, Gl_b = selected_inverse_junction(
        diag, edge_index, edge_val, order=order, batched=True
    )
    assert torch.equal(Si_a, Si_b), "filled pattern S_index must match the reference exactly"
    assert torch.allclose(Gd_a, Gd_b, atol=atol), "G_diag mismatch vs reference"
    assert torch.allclose(Gl_a, Gl_b, atol=atol), "G_lower mismatch vs reference"
    return Gd_b, Si_b, Gl_b


# --------------------------------------------------------------------------- #
# Forward: block-for-block equality with the reference on loopy grids (fill).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(3, 3, 2), (4, 3, 1), (2, 5, 2), (5, 5, 1), (6, 4, 2)])
def test_forward_matches_reference_grid(rows, cols, b):
    n = rows * cols
    sp = random_spd_graph(n, grid_edges(rows, cols), b, seed=rows * 10 + cols, diag_load=1.0)
    _, S_index, _ = _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)
    assert S_index.shape[1] > sp.num_edges, "expected genuine fill on a loopy grid"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_forward_matches_reference_random_graph(seed):
    n, b = 10, 2
    gen = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i) if torch.rand((), generator=gen) < 0.35]
    if not pairs:
        pairs = [(1, 0)]
    edges = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    sp = random_spd_graph(n, edges, b, seed=seed, diag_load=1.0)
    _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)


def test_forward_matches_reference_scalar_b1():
    sp = random_spd_graph(9, grid_edges(3, 3), 1, seed=5, diag_load=1.0)
    _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)


def test_forward_matches_reference_tree_zero_fill():
    # A tree has a zero-fill perfect elimination order; the batched path must still
    # reduce block-for-block to the reference (which itself reduces to the tree kernel).
    bt = random_spd_tree(12, 3, seed=4, diag_load=2.0, kind="random")
    plist = bt.parent.tolist()
    pairs, vals = [], []
    for v in range(12):
        p = plist[v]
        if p == -1:
            continue
        if v > p:
            pairs.append((v, p)); vals.append(bt.edge[v].mT)
        else:
            pairs.append((p, v)); vals.append(bt.edge[v])
    edge_index = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    edge_val = torch.stack(vals, dim=0)
    _, S_index, _ = _assert_matches_reference(bt.diag, edge_index, edge_val)
    assert S_index.shape[1] == edge_index.shape[1], "tree must stay zero-fill"


# --------------------------------------------------------------------------- #
# Edge cases (mirror the reference's): single node, no edges, batch, order.
# --------------------------------------------------------------------------- #
def test_forward_single_node():
    sp = random_spd_graph(1, [], 4, seed=1, diag_load=1.0)
    _, S_index, G_lower = _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)
    assert S_index.shape == (2, 0) and G_lower.shape[-3] == 0


def test_forward_no_edges_block_diagonal():
    # All nodes are independent -> a single elimination level of n nodes (the wide case).
    sp = random_spd_graph(5, [], 3, seed=3, diag_load=1.0)
    _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)
    _, _, _, _, _, levels = _batched_symbolic(5, sp.edge_index, None, torch.device("cpu"))
    assert len(levels) == 1


def test_forward_leading_batch():
    sp = random_spd_graph(6, grid_edges(2, 3), 2, seed=7, diag_load=1.5, batch_shape=(4,))
    Gd_b, _, Gl_b = _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val)
    # Batched path's leading-batch result must also match a per-item batched run.
    for k in range(4):
        d_k, _s, l_k = selected_inverse_junction(
            sp.diag[k], sp.edge_index, sp.edge_val[k], batched=True
        )
        assert torch.allclose(Gd_b[k], d_k, atol=1e-12)
        assert torch.allclose(Gl_b[k], l_k, atol=1e-12)


def test_forward_explicit_order():
    n = 9
    sp = random_spd_graph(n, grid_edges(3, 3), 2, seed=8, diag_load=1.0)
    rev = list(range(n - 1, -1, -1))
    _assert_matches_reference(sp.diag, sp.edge_index, sp.edge_val, order=rev)


# --------------------------------------------------------------------------- #
# Backward: autograd through the batched path must equal the reference gradients.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_matches_reference(seed):
    sp = random_spd_graph(9, grid_edges(3, 3), 2, seed=seed, diag_load=2.0)

    def grads(batched):
        d = sp.diag.clone().requires_grad_(True)
        e = sp.edge_val.clone().requires_grad_(True)
        Gd, _S, Gl = selinv_junction(d, sp.edge_index, e, batched=batched)
        loss = Gd.pow(2).sum() + Gl.pow(2).sum() + torch.diagonal(Gd, dim1=-2, dim2=-1).sum()
        loss.backward()
        return d.grad, e.grad

    gd_loop, ge_loop = grads(False)
    gd_bat, ge_bat = grads(True)
    assert torch.allclose(gd_loop, gd_bat, atol=1e-9)
    assert torch.allclose(ge_loop, ge_bat, atol=1e-9)


def test_gradcheck_batched():
    sp = random_spd_graph(4, grid_edges(2, 2), 2, seed=1, diag_load=2.0)
    diag = sp.diag.clone().requires_grad_(True)
    edge_val = sp.edge_val.clone().requires_grad_(True)

    def f(d, e):
        Gd, _S, Gl = selinv_junction(d, sp.edge_index, e, batched=True)
        return torch.cat([Gd.reshape(-1), Gl.reshape(-1)])

    assert torch.autograd.gradcheck(f, (diag, edge_val), atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# compute_dtype routes through the batched path too (fp32 store / fp64 compute).
# --------------------------------------------------------------------------- #
def test_compute_dtype_batched_matches_reference():
    sp = random_spd_graph(9, grid_edges(3, 3), 2, seed=7, diag_load=1.0)
    d32, e32 = sp.diag.to(torch.float32), sp.edge_val.to(torch.float32)
    Gd_a, _Sa, Gl_a = selected_inverse_junction(
        d32, sp.edge_index, e32, compute_dtype=torch.float64
    )
    Gd_b, _Sb, Gl_b = selected_inverse_junction(
        d32, sp.edge_index, e32, compute_dtype=torch.float64, batched=True
    )
    assert Gd_b.dtype == torch.float32 and Gl_b.dtype == torch.float32
    assert torch.allclose(Gd_a, Gd_b, atol=1e-6) and torch.allclose(Gl_a, Gl_b, atol=1e-6)


# --------------------------------------------------------------------------- #
# Scaling note: the Python loop-count drops from n to the elimination-tree height.
# Sub-linear on a 2-D grid (~sqrt(n)); a diagnostic, not a wall-clock claim (the GPU
# launch-latency win is un-measured on this CPU env).
# --------------------------------------------------------------------------- #
def test_level_count_sublinear_on_grid():
    counts = {}
    for side in (3, 5, 7, 9):
        n = side * side
        sp = random_spd_graph(n, grid_edges(side, side), 1, seed=1, diag_load=1.0)
        _, _, _, _, _, levels = _batched_symbolic(n, sp.edge_index, None, torch.device("cpu"))
        counts[n] = len(levels)
        assert len(levels) < n, "batched loop must run over fewer levels than nodes"
    # Loop-count reduction n/levels grows as the grid grows (sub-linear level count).
    assert counts[81] / 81 < counts[9] / 9
