"""Reach bound of the routed-field Jacobi arm (Proposition 1 of the attainability paper).

The reach argument constrains source-Krylov rules: before the first product they may
form only scalar multiples of the source, and thereafter only linear combinations or
products of the sparse iteration matrix with an available vector. Global transforms,
arbitrary probes, and uncounted nonlocal preconditioners are outside the class. The
paper therefore checks the shipped arm rather than relying on the idealization.

``K`` damped-Jacobi sweeps from a zero start spend ``K - 1`` products with ``A``, so
the support bound at ``K`` sweeps is ``K - 1`` hops.
"""

from collections import deque

import torch

from gabp_sparse_inv.demos.maze_grid import GaBPMazeGrid, gen_dataset, reach_floor_sweep


def _hop_distance(edge_index, n_nodes, source):
    """BFS distance from ``source`` on the undirected grid pattern."""
    adj = [[] for _ in range(n_nodes)]
    for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[i].append(j)
        adj[j].append(i)
    dist = [-1] * n_nodes
    dist[source] = 0
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def _point_source_feats(n_nodes, source):
    """Fixed geometry features with a unit point source ``b = e_source``."""
    gen = torch.Generator().manual_seed(0)
    feats = torch.zeros(n_nodes, 3, dtype=torch.float64)
    feats[:, :2] = torch.rand(n_nodes, 2, generator=gen, dtype=torch.float64)
    feats[source, 2] = 1.0
    return feats


def test_k_sweeps_stay_within_k_minus_one_hops():
    rows = cols = 6
    n_nodes, source = rows * cols, 0
    for K in (1, 2, 4, 7):
        model = GaBPMazeGrid(rows, cols, solve_steps=K).double()
        dist = _hop_distance(model.edge_index, n_nodes, source)
        out = model(_point_source_feats(n_nodes, source))
        reached = [v for v in range(n_nodes) if out[v].abs() > 0]
        assert reached, f"K={K} produced an identically zero field"
        assert max(dist[v] for v in reached) <= K - 1, (
            f"K={K} reached {max(dist[v] for v in reached)} hops, bound is {K - 1}"
        )


def test_the_reach_bound_is_attained():
    """Sharpness: the frontier at exactly ``K - 1`` hops is populated."""
    rows = cols = 6
    n_nodes, source = rows * cols, 0
    for K in (2, 4, 7):
        model = GaBPMazeGrid(rows, cols, solve_steps=K).double()
        dist = _hop_distance(model.edge_index, n_nodes, source)
        out = model(_point_source_feats(n_nodes, source))
        frontier = [v for v in range(n_nodes) if dist[v] == K - 1]
        assert any(out[v].abs() > 0 for v in frontier), (
            f"K={K} did not reach its own frontier at {K - 1} hops"
        )


def test_matched_budget_cannot_cross_the_test_grids():
    """The preregistered ``K_match = 7`` reaches 6 hops against diameters 22, 46 and 94."""
    source = 0
    for size, diameter in ((12, 22), (24, 46), (48, 94)):
        model = GaBPMazeGrid(size, size, solve_steps=7).double()
        n_nodes = size * size
        dist = _hop_distance(model.edge_index, n_nodes, source)
        out = model(_point_source_feats(n_nodes, source))
        assert max(dist) == diameter
        assert all(out[v] == 0 for v in range(n_nodes) if dist[v] > 6)


def test_the_floor_is_exactly_the_arms_error_beyond_reach():
    """The support-to-loss bridge, checked on the arm rather than on the idealization.

    Beyond its reach the arm predicts zero, so its squared error there is the field's
    own energy. That identity is what turns the support bound into a loss bound.
    """
    rows = cols = 6
    n_nodes, num = rows * cols, 8
    edge_index, feats, target = gen_dataset(rows, cols, num, seed=1000)
    src = feats[..., 2].argmax(dim=-1)
    dist = torch.tensor([_hop_distance(edge_index, n_nodes, int(s)) for s in src])
    for K in (1, 2, 4, 7):
        model = GaBPMazeGrid(rows, cols, solve_steps=K).double()
        with torch.no_grad():
            beyond = (dist > K - 1).to(target.dtype)
            err = torch.sum(((model(feats) - target) * beyond) ** 2)
            energy = torch.sum((target * beyond) ** 2)
        assert torch.allclose(err, energy, rtol=0, atol=0), (
            f"K={K}: beyond-reach error {err} is not the field energy {energy}"
        )


def test_the_floor_vanishes_once_the_reach_covers_the_grid():
    """The bound is honest about where it stops binding: a budget past the diameter
    constrains nothing, so the dose-response there rests on measurement alone."""
    res = reach_floor_sweep(6, 6, budgets=(4, 7, 8, 16, 32), n_test=8)
    assert res["diameter"] == 10
    assert res[4] > res[7] > res[8] > 0.0        # strictly binding below the diameter
    assert res[16] == 0.0 and res[32] == 0.0     # reach 15 and 31 cover diameter 10


def test_arm_carries_no_global_term():
    """A global bias, normalisation across nodes, or pooled term would make the output
    nonzero for a zero source without spending any product with ``A``."""
    rows = cols = 6
    n_nodes = rows * cols
    for K in (1, 4, 7):
        model = GaBPMazeGrid(rows, cols, solve_steps=K).double()
        feats = _point_source_feats(n_nodes, 0)
        feats[:, 2] = 0.0
        assert torch.all(model(feats) == 0), (
            f"K={K} produced a nonzero field from a zero source"
        )


def test_old_transpose_power_span_claim_has_a_counterexample():
    """For nonsymmetric ``J``, one ``A.T`` product need not lie in span{b, Jb}."""
    J = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64) - J
    b = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert torch.linalg.matrix_rank(torch.stack([b, J @ b])) == 1
    assert torch.linalg.matrix_rank(torch.stack([b, A.T @ b])) == 2


def test_old_zero_start_claim_did_not_exclude_a_global_transform():
    """A zero initial iterate alone does not stop a rule from globally pooling ``b``."""
    b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    globally_pooled = torch.ones_like(b) * b.sum()
    assert torch.count_nonzero(globally_pooled) == 3
