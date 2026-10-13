"""Input contracts: malformed values and indices fail loudly instead of computing."""

from __future__ import annotations

import math

import pytest
import torch

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
