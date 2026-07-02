"""Gates for exact Gaussian sampling from a structured SPD precision (tree / junction).

The covariance is checked **exactly** (no Monte-Carlo noise) by applying the deterministic
sampling transform x = M^{-T} z to the standard basis: the resulting matrix T satisfies
T T^T = A^{-1} iff A = M M^T, which is the whole correctness claim. A loose statistical
check on actual random draws is a sanity gate. The junction sampler on a tree pattern (with
the tree's elimination order) must reproduce the tree sampler block-for-block.
"""

from __future__ import annotations

import torch

from gabp_sparse_inv import (
    grid_edges,
    random_spd_graph,
    random_spd_tree,
    sample_gaussian_junction,
    sample_gaussian_tree,
)
from gabp_sparse_inv.layout import _as_parent_tensor, tree_orders
from gabp_sparse_inv.sampling import _sample_transform_junction, _sample_transform_tree

torch.manual_seed(0)


def _tree_to_sparse(bt):
    """Build a junction (edge_index, edge_val) from a BlockTree (i > j orientation)."""
    plist = bt.parent.tolist()
    pairs, vals = [], []
    for v in range(bt.num_nodes):
        p = plist[v]
        if p == -1:
            continue
        if v > p:
            pairs.append((v, p)); vals.append(bt.edge[v].mT)
        else:
            pairs.append((p, v)); vals.append(bt.edge[v])
    return torch.tensor(list(zip(*pairs)), dtype=torch.long), torch.stack(vals, dim=0)


def test_tree_sampler_exact_covariance():
    n, b = 8, 2
    bt = random_spd_tree(n, b, seed=0, diag_load=1.0, kind="random")
    N = n * b
    z = torch.eye(N, dtype=torch.float64).reshape(n, b, N)            # basis as the columns
    T = _sample_transform_tree(bt.diag, bt.edge, _as_parent_tensor(bt.parent), z).reshape(N, N)
    cov = T @ T.mT
    A = bt.to_dense().to(torch.float64)
    assert torch.allclose(cov, torch.linalg.inv(A), atol=1e-10)


def test_junction_sampler_exact_covariance_grid():
    rows, cols, b = 3, 3, 2
    n = rows * cols
    sp = random_spd_graph(n, grid_edges(rows, cols), b, seed=1, diag_load=1.0)
    N = n * b
    z = torch.eye(N, dtype=torch.float64).reshape(n, b, N)
    T = _sample_transform_junction(sp.diag, sp.edge_index, sp.edge_val, z, None).reshape(N, N)
    cov = T @ T.mT
    A = sp.to_dense().to(torch.float64)
    assert torch.allclose(cov, torch.linalg.inv(A), atol=1e-10)


def test_junction_matches_tree_sampler_on_tree_pattern():
    # Same deterministic z and the tree's own elimination order => block-for-block equal M.
    n, b = 10, 2
    bt = random_spd_tree(n, b, seed=2, diag_load=2.0, kind="random")
    edge_index, edge_val = _tree_to_sparse(bt)
    _root, _children, collect = tree_orders(_as_parent_tensor(bt.parent))
    z = torch.randn(n, b, 5, dtype=torch.float64)
    xt = _sample_transform_tree(bt.diag, bt.edge, _as_parent_tensor(bt.parent), z)
    xj = _sample_transform_junction(bt.diag, edge_index, edge_val, z, collect)
    assert torch.allclose(xt, xj, atol=1e-10)


def test_sampler_output_shape_and_batch():
    n, b, batch = 6, 2, 3
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=3, diag_load=1.5, batch_shape=(batch,))
    g = torch.Generator().manual_seed(0)
    x = sample_gaussian_junction(sp.diag, sp.edge_index, sp.edge_val, 7, generator=g)
    assert x.shape == (7, batch, n, b)

    bt = random_spd_tree(n, b, seed=4, diag_load=1.0, kind="balanced")
    xt = sample_gaussian_tree(bt.diag, bt.edge, bt.parent, 5)
    assert xt.shape == (5, n, b)


def test_junction_sampler_empirical_covariance():
    # MC-noisy sanity gate: empirical covariance of many draws ~ A^{-1} on the pattern.
    n, b = 6, 1
    sp = random_spd_graph(n, grid_edges(2, 3), b, seed=5, diag_load=1.0)
    g = torch.Generator().manual_seed(0)
    S = 200_000
    X = sample_gaussian_junction(sp.diag, sp.edge_index, sp.edge_val, S, generator=g)  # [S, n, b]
    Xf = X.reshape(S, n * b)
    emp = (Xf.mT @ Xf) / S
    inv = torch.linalg.inv(sp.to_dense().to(torch.float64))
    assert (emp - inv).abs().max() < 0.03   # loose: pure Monte-Carlo noise at this S


def test_tree_sampler_empirical_covariance():
    n, b = 7, 1
    bt = random_spd_tree(n, b, seed=6, diag_load=1.0, kind="random")
    g = torch.Generator().manual_seed(0)
    S = 200_000
    X = sample_gaussian_tree(bt.diag, bt.edge, bt.parent, S, generator=g)
    Xf = X.reshape(S, n * b)
    emp = (Xf.mT @ Xf) / S
    inv = torch.linalg.inv(bt.to_dense().to(torch.float64))
    assert (emp - inv).abs().max() < 0.03
