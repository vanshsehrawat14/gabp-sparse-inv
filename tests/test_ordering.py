"""Nested-dissection grid ordering + order-invariance / fill gates for the junction kernel.

``elimination_order_nested_dissection`` produces a deterministic, dimension-aware
elimination order for a ``rows x cols`` 4-neighbour grid (the :func:`grid_edges`
convention). Two properties are gated:

* **Order-invariance** -- the selected inverse on the *input* pattern (node diagonals and
  the original grid edges) is identical regardless of elimination order (min-degree,
  nested dissection, lexicographic): only the *fill* differs, never the result.
* **Asymptotic fill** -- on a sufficiently large square grid nested dissection produces
  strictly less fill than a lexicographic (row-major) order.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    elimination_order_min_degree,
    elimination_order_nested_dissection,
    grid_edges,
    random_spd_graph,
    selected_inverse_junction,
)
from gabp_sparse_inv.junction import _normalize_edge_index, _symbolic

torch.manual_seed(0)


def _nbr(n, ei):
    nbr = [set() for _ in range(n)]
    for a, c in zip(ei[0].tolist(), ei[1].tolist()):
        nbr[a].add(c)
        nbr[c].add(a)
    return nbr


def _fill(n, ei, order):
    return len(_symbolic(n, ei, order)[3])


# --------------------------------------------------------------------------- #
# The helper returns a valid permutation for every grid shape.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 6), (6, 1), (5, 5), (6, 4), (3, 7), (8, 8)])
def test_nested_dissection_is_a_permutation(rows, cols):
    order = elimination_order_nested_dissection(rows, cols)
    assert sorted(order) == list(range(rows * cols))


def test_nested_dissection_rejects_bad_dims():
    with pytest.raises(ValueError):
        elimination_order_nested_dissection(0, 5)


# --------------------------------------------------------------------------- #
# Order-invariance of the selected inverse on the input pattern.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,b", [(5, 5, 2), (6, 4, 3), (4, 7, 2)])
def test_selected_inverse_is_order_invariant(rows, cols, b):
    n = rows * cols
    ei = grid_edges(rows, cols)
    mat = random_spd_graph(n, ei, b, seed=rows * 10 + cols, diag_load=1.0)
    eix = _normalize_edge_index(ei)

    nd = elimination_order_nested_dissection(rows, cols)
    md = elimination_order_min_degree(n, _nbr(n, eix))
    lex = list(range(n))

    base_d, base_S, base_L = selected_inverse_junction(mat.diag, ei, mat.edge_val, lex)
    # Map (i, j) -> block for the baseline filled result, to compare on the input pattern.
    base_map = {(int(i), int(j)): base_L[..., k, :, :]
                for k, (i, j) in enumerate(zip(base_S[0].tolist(), base_S[1].tolist()))}

    for order in (nd, md):
        d, S, L = selected_inverse_junction(mat.diag, ei, mat.edge_val, order)
        assert torch.allclose(d, base_d, atol=1e-10)
        # Every *input* edge block must match the baseline regardless of fill/order.
        for k in range(eix.shape[1]):
            i, j = int(eix[0, k]), int(eix[1, k])
            blk = L[..., _row_of(S, i, j), :, :]
            assert torch.allclose(blk, base_map[(i, j)], atol=1e-10)


def _row_of(S, i, j):
    cols = list(zip(S[0].tolist(), S[1].tolist()))
    return cols.index((i, j))


# --------------------------------------------------------------------------- #
# Asymptotic fill: nested dissection beats lexicographic on a large square grid.
# --------------------------------------------------------------------------- #
def test_nested_dissection_reduces_fill_vs_lexicographic():
    s = 16
    n = s * s
    eix = _normalize_edge_index(grid_edges(s, s))
    f_nd = _fill(n, eix, elimination_order_nested_dissection(s, s))
    f_lex = _fill(n, eix, list(range(n)))
    assert f_nd < f_lex, f"nd fill {f_nd} not below lexicographic {f_lex}"


def test_nested_dissection_fill_advantage_grows_with_size():
    ratios = []
    for s in (12, 24):
        n = s * s
        eix = _normalize_edge_index(grid_edges(s, s))
        f_nd = _fill(n, eix, elimination_order_nested_dissection(s, s))
        f_lex = _fill(n, eix, list(range(n)))
        ratios.append(f_nd / f_lex)
    assert ratios[1] < ratios[0], f"nd/lex fill ratio should shrink with size; got {ratios}"
