"""Input contracts: malformed values, indices and topology fail loudly instead of computing."""

from __future__ import annotations

import math

import pytest
import torch

import gabp_sparse_inv as gs
from gabp_sparse_inv import (
    selected_inverse_chain,
    selected_inverse_junction,
    selected_inverse_star,
    selected_inverse_tree,
    selected_inverse_bidiag,
)
from gabp_sparse_inv._linalg import cholesky_spd
from gabp_sparse_inv.nonsym_junction import selected_inverse_nonsym_junction

F64 = torch.float64


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_check_rejects_non_finite_blocks_in_every_layout(bad):
    two = torch.full((2, 1, 1), 2.0, dtype=F64)
    one = torch.full((1, 1, 1), 0.5, dtype=F64)
    poisoned = two.clone()
    poisoned[0, 0, 0] = bad
    ei = torch.tensor([[1], [0]])
    calls = [
        lambda: selected_inverse_chain(poisoned, one, check=True),
        lambda: selected_inverse_star(poisoned[0], two, one.expand(2, 1, 1), check=True),
        lambda: selected_inverse_tree(poisoned, torch.cat([torch.zeros_like(one), one]), [-1, 0], check=True),
        lambda: selected_inverse_junction(poisoned, ei, one, check=True),
        lambda: selected_inverse_bidiag(poisoned, one, check=True),
        lambda: selected_inverse_nonsym_junction(poisoned, ei, one, one, check=True),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="finite"):
            call()


@pytest.mark.parametrize("index", [torch.tensor([[1.9], [0.9]]), [[1.9], [0.9]], torch.tensor([[True], [False]])])
def test_non_integer_edge_indices_are_rejected_not_truncated(index):
    diag = torch.full((2, 1, 1), 2.0, dtype=F64)
    with pytest.raises(TypeError, match="integers"):
        selected_inverse_junction(diag, index, torch.full((1, 1, 1), 0.5, dtype=F64))


def test_empty_python_edge_list_is_still_accepted():
    diag = torch.full((2, 1, 1), 2.0, dtype=F64)
    Gd, S, _Gl = selected_inverse_junction(diag, [[], []], torch.zeros((0, 1, 1), dtype=F64))
    torch.testing.assert_close(Gd.reshape(2), torch.full((2,), 0.5, dtype=F64))
    assert S.shape == (2, 0)


def test_cholesky_failure_message_names_an_unbatched_input():
    with pytest.raises(torch.linalg.LinAlgError, match="unbatched input"):
        cholesky_spd(torch.tensor([[-1.0]], dtype=F64), name="D_0")


# --------------------------------------------------------------------------- #
# Structural checks run on every call, ``check`` or not (0.4.0 summed both-direction
# and duplicate edge lists, truncated float parents and ignored count mismatches).
# --------------------------------------------------------------------------- #
def _junction_calls(diag, ei, ev):
    """Every public entry point that reads a junction edge list, all with check=False."""
    n = diag.shape[-3]
    up = ev.mT
    rhs = torch.ones(n, diag.shape[-1], dtype=diag.dtype)
    return {
        "selected_inverse_junction": lambda: gs.selected_inverse_junction(diag, ei, ev),
        "selected_inverse_junction batched": lambda: gs.selected_inverse_junction(diag, ei, ev, batched=True),
        "selinv_junction": lambda: gs.selinv_junction(diag, ei, ev),
        "selinv_junction batched": lambda: gs.selinv_junction(diag, ei, ev, batched=True),
        "selinv_junction_analytic": lambda: gs.selinv_junction_analytic(diag, ei, ev),
        "junction_solve": lambda: gs.junction_solve(diag, ei, ev, rhs),
        "junction_logdet": lambda: gs.junction_logdet(diag, ei, ev),
        "sample_gaussian_junction": lambda: gs.sample_gaussian_junction(diag, ei, ev, 2),
        "analyze_junction": lambda: gs.analyze_junction(n, ei),
        "BlockSparseSym.validate": lambda: gs.BlockSparseSym(diag=diag, edge_index=ei, edge_val=ev).validate(),
        "grid_node_degrees": lambda: gs.grid_node_degrees(n, ei),
        "selected_inverse_nonsym_junction": lambda: gs.selected_inverse_nonsym_junction(diag, ei, ev, up),
        "selinv_nonsym_junction": lambda: gs.selinv_nonsym_junction(diag, ei, ev, up),
        "selinv_nonsym_junction_analytic": lambda: gs.selinv_nonsym_junction_analytic(diag, ei, ev, up),
        "nonsym_junction_solve": lambda: gs.nonsym_junction_solve(diag, ei, ev, up, rhs),
    }


def _spd_graph():
    return gs.random_spd_graph(6, gs.grid_edges(2, 3), 2, seed=0, diag_load=2.0)


_JUNCTION_ENTRY_POINTS = [
    "selected_inverse_junction", "selected_inverse_junction batched", "selinv_junction",
    "selinv_junction batched", "selinv_junction_analytic", "junction_solve", "junction_logdet",
    "sample_gaussian_junction", "analyze_junction", "BlockSparseSym.validate", "grid_node_degrees",
    "selected_inverse_nonsym_junction", "selinv_nonsym_junction", "selinv_nonsym_junction_analytic",
    "nonsym_junction_solve",
]


@pytest.mark.parametrize("name", _JUNCTION_ENTRY_POINTS)
def test_both_directions_and_duplicates_are_refused_everywhere(name):
    A = _spd_graph()
    both = torch.cat([A.edge_index, A.edge_index.flip(0)], dim=1)
    both_val = torch.cat([A.edge_val, A.edge_val.mT], dim=0)
    with pytest.raises(ValueError, match="lower-triangular"):
        _junction_calls(A.diag, both, both_val)[name]()
    dup = torch.cat([A.edge_index, A.edge_index[:, :1]], dim=1)
    dup_val = torch.cat([A.edge_val, A.edge_val[:1]], dim=0)
    with pytest.raises(ValueError, match="duplicate"):
        _junction_calls(A.diag, dup, dup_val)[name]()
    upper = A.edge_index.flip(0)
    with pytest.raises(ValueError, match="keep = edge_index"):
        _junction_calls(A.diag, upper, A.edge_val.mT)[name]()
    for bad in (torch.tensor([[6], [0]]), torch.tensor([[-1], [0]])):
        with pytest.raises(ValueError, match="must lie in"):
            _junction_calls(A.diag, bad, A.edge_val[:1])[name]()


def test_documented_conversion_of_a_both_direction_list_is_exact():
    A = _spd_graph()
    both = torch.cat([A.edge_index, A.edge_index.flip(0)], dim=1)
    both_val = torch.cat([A.edge_val, A.edge_val.mT], dim=0)       # A_ji = A_ij^T
    keep = both[0] > both[1]
    Gd, S, Gl = gs.selected_inverse_junction(A.diag, both[:, keep], both_val[keep])
    G = torch.linalg.inv(A.to_dense())
    torch.testing.assert_close(Gd, torch.stack([G[2 * v:2 * v + 2, 2 * v:2 * v + 2] for v in range(6)]))
    # non-symmetric kernels take the dropped (j, i) blocks as edge_upper
    up = torch.zeros_like(both_val[keep])
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(both.T.tolist())}
    for k, (i, j) in enumerate(both[:, keep].T.tolist()):
        up[k] = both_val[lookup[(j, i)]]
    Gd2 = gs.selected_inverse_nonsym_junction(A.diag, both[:, keep], both_val[keep], up)[0]
    torch.testing.assert_close(Gd2, Gd)


@pytest.mark.parametrize("parent", [torch.tensor([1.7, -1.0, 1.2]), [1.7, -1.0, 1.2], [1.0, -1.0, 1.0],
                                    torch.tensor([True, False, True])])
def test_non_integer_parents_are_rejected_not_truncated(parent):
    T = gs.random_spd_tree(3, 2, seed=1, diag_load=1.0)
    rhs = torch.ones(3, 2, dtype=F64)
    calls = [
        lambda: gs.selected_inverse_tree(T.diag, T.edge, parent),
        lambda: gs.selected_inverse_tree(T.diag, T.edge, parent, check=True),
        lambda: gs.selected_inverse_tree(T.diag, T.edge, parent, batched=True),
        lambda: gs.selinv_tree(T.diag, T.edge, parent),
        lambda: gs.selected_inverse_nonsym_tree(T.diag, T.edge, T.edge, parent),
        lambda: gs.tree_logdet(T.diag, T.edge, parent),
        lambda: gs.tree_solve(T.diag, T.edge, parent, rhs),
        lambda: gs.sample_gaussian_tree(T.diag, T.edge, parent, 2),
        lambda: gs.BlockTree(diag=T.diag, edge=T.edge, parent=parent),
        lambda: gs.marginal_log_likelihood(parent, torch.ones(3, 1, dtype=F64), 0.5, 1.0, 1.0, 0.1),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="integers"):
            call()


def test_non_integer_edge_index_in_the_container_is_rejected():
    with pytest.raises(TypeError, match="integers"):
        gs.BlockSparseSym(diag=torch.full((2, 1, 1), 2.0, dtype=F64),
                          edge_index=torch.tensor([[1.9], [0.2]]), edge_val=torch.full((1, 1, 1), 0.1, dtype=F64))


def test_block_count_mismatches_are_refused_without_check():
    A = _spd_graph()
    extra = torch.cat([A.edge_val, torch.ones(3, 2, 2, dtype=F64)], dim=0)
    with pytest.raises(ValueError, match="edge_val must have shape"):
        gs.selected_inverse_junction(A.diag, A.edge_index, extra)
    T = gs.random_spd_tree(5, 2, seed=1, diag_load=1.0)
    with pytest.raises(ValueError, match="parent must have length"):
        gs.selected_inverse_tree(T.diag, T.edge, [-1, 0, 1])           # 3 parents for 5 nodes
    C = gs.random_spd_chain(4, 2, seed=0, diag_load=1.0)
    with pytest.raises(ValueError, match="lower must have shape"):
        gs.selected_inverse_chain(C.diag, C.lower[:1])
    M = gs.random_nonsym_bidiag(4, 2, seed=0)
    with pytest.raises(ValueError, match="lower must have shape"):
        gs.selected_inverse_bidiag(M.diag, M.lower[:1])                # would broadcast in 0.4.0
    with pytest.raises(ValueError, match="lower must have shape"):
        gs.selinv_bidiag(M.diag, M.lower[:1])
    S = gs.random_spd_star(3, 2, seed=0)
    with pytest.raises(ValueError, match="coupling must have shape"):
        gs.selected_inverse_star(S.center, S.leaf_diag, S.coupling[:1])


def test_empty_inputs_are_refused_clearly():
    with pytest.raises(ValueError, match="at least one node"):
        gs.selected_inverse_junction(torch.zeros(0, 2, 2), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2, 2))
    with pytest.raises(ValueError, match="at least one node"):
        gs.selected_inverse_chain(torch.zeros(0, 2, 2), torch.zeros(0, 2, 2))


def _spd_checked_calls(diag):
    """check=True through every SPD layout, with ``diag`` as the (n=2) diagonal blocks."""
    b = diag.shape[-1]
    zero = torch.zeros(1, b, b, dtype=diag.dtype)
    return [
        lambda: gs.selected_inverse_chain(diag, zero, check=True),
        lambda: gs.selected_inverse_star(diag[0], diag[1:], zero, check=True),
        lambda: gs.selected_inverse_tree(diag, torch.cat([zero, zero]), [-1, 0], check=True),
        lambda: gs.selected_inverse_junction(diag, torch.zeros(2, 0, dtype=torch.long),
                                             torch.zeros(0, b, b, dtype=diag.dtype), check=True),
    ]


def test_symmetry_is_judged_per_block():
    # 0.4.0 compared against max(1, largest entry anywhere): a fully asymmetric block of tiny
    # entries passed, and so did an asymmetric unit block sitting next to a 1e12 block.
    tiny = torch.tensor([[1e-12, 5e-11], [0.0, 1e-12]], dtype=F64)
    masked = torch.tensor([[1.0, 1e-3], [0.0, 1.0]], dtype=F64)
    for bad in (torch.stack([tiny, torch.eye(2, dtype=F64)]), torch.stack([masked, 1e12 * torch.eye(2, dtype=F64)])):
        for call in _spd_checked_calls(bad):
            with pytest.raises(ValueError, match="not symmetric"):
                call()


def test_roundoff_level_asymmetry_still_passes():
    for scale in (1e-12, 1.0, 1e12):
        block = scale * torch.tensor([[2.0, 0.5], [0.5 * (1 + 1e-14), 2.0]], dtype=F64)
        for call in _spd_checked_calls(torch.stack([block, block])):
            call()


def test_pivot_failures_name_the_node_in_every_path():
    # Node 2 is indefinite. 0.4.0 called it D_3 in the chain (1-based), named only the level
    # in the batched paths, and tree_logdet surfaced a bare backend error.
    b = 2
    eye = torch.eye(b, dtype=F64)
    diag = torch.stack([3 * eye, 3 * eye, -5 * eye, 3 * eye, 3 * eye])
    lower = torch.full((4, b, b), 0.1, dtype=F64)
    edge = torch.cat([torch.zeros(1, b, b, dtype=F64), lower])     # path tree rooted at node 0
    parent = [-1, 0, 1, 2, 3]
    ei, order = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]]), [0, 1, 2, 3, 4]
    named = {
        "D_2": [
            lambda: gs.selected_inverse_chain(diag, lower),
            lambda: gs.selected_inverse_tree(diag, edge, parent),
            lambda: gs.selected_inverse_junction(diag, ei, lower, order),
            lambda: gs.tree_logdet(diag, edge, parent),
        ],
        "node(s) [2]": [
            lambda: gs.selected_inverse_tree(diag, edge, parent, batched=True),
            lambda: gs.selected_inverse_junction(diag, ei, lower, order, batched=True),
            lambda: gs.tree_logdet(diag, edge, parent, batched=True),
        ],
    }
    for text, calls in named.items():
        for call in calls:
            with pytest.raises(torch.linalg.LinAlgError, match=r"positive-definiteness") as err:
                call()
            assert text in str(err.value), str(err.value)
