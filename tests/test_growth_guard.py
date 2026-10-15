"""The no-pivot growth guard is local and matched to each operation's error law.

0.4.0 divided the growth by the largest input entry of the whole matrix, so one large
block anywhere (even a disconnected one) disabled the guard and the kernels returned
wrong blocks. It also measured only the Schur entries and used one limit, 1/sqrt(eps),
for solves and selected inverses alike. These tests pin the criterion of
``_linalg.check_pivot_growth``: factor products against local row/column scales, with
``eps**-1/2`` for solves and ``eps**-1/4`` for selected inverses.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import nonsym_junction as nonsym_junction_module
from gabp_sparse_inv import (
    nonsym_junction_solve,
    selected_inverse_junction,
    selected_inverse_nonsym_junction,
    selected_inverse_nonsym_tree,
    selinv_nonsym_junction,
    selinv_nonsym_junction_analytic,
)

F64, F32 = torch.float64, torch.float32


def _pair_plus_node(eps, big, dtype=F64, coupling=None, scale=1.0):
    """``[[eps, 1], [2, 1]]`` (cond -> 2.62) plus node 2 with diagonal ``big``.

    ``coupling=(a21, a12)`` connects node 2 to node 1; otherwise node 2 is isolated.
    Returns the junction inputs and the dense matrix.
    """
    diag = torch.tensor([[[eps]], [[1.0]], [[big]]], dtype=dtype) * scale
    M = torch.zeros(3, 3, dtype=F64)
    M[0, 0], M[1, 1], M[2, 2], M[1, 0], M[0, 1] = eps, 1.0, big, 2.0, 1.0
    if coupling is None:
        ei = torch.tensor([[1], [0]])
        lower = torch.tensor([[[2.0]]], dtype=dtype) * scale
        upper = torch.tensor([[[1.0]]], dtype=dtype) * scale
    else:
        a21, a12 = coupling
        ei = torch.tensor([[1, 2], [0, 1]])
        lower = torch.tensor([[[2.0]], [[a21]]], dtype=dtype) * scale
        upper = torch.tensor([[[1.0]], [[a12]]], dtype=dtype) * scale
        M[2, 1], M[1, 2] = a21, a12
    return diag, ei, lower, upper, M * scale


def _assert_every_route_refuses(diag, ei, lower, upper, order):
    rhs = torch.ones(3, 1, dtype=diag.dtype)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        selected_inverse_nonsym_junction(diag, ei, lower, upper, order, check=True)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        selinv_nonsym_junction(diag, ei, lower, upper, order)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        selinv_nonsym_junction_analytic(diag, ei, lower, upper, order)
    for transpose in (False, True):
        with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
            nonsym_junction_solve(diag, ei, lower, upper, rhs, order, transpose=transpose)


@pytest.mark.parametrize("dtype,eps,big", [
    (F64, 1e-18, 1e-12), (F64, 1e-18, 1.0), (F64, 1e-18, 1e6), (F64, 1e-18, 1e12),
    (F32, 2e-8, 1.0), (F32, 2e-8, 1e3), (F32, 2e-8, 1e5), (F32, 1e-9, 1e6),
])
def test_disconnected_block_cannot_hide_growth(dtype, eps, big):
    diag, ei, lower, upper, _ = _pair_plus_node(eps, big, dtype)
    _assert_every_route_refuses(diag, ei, lower, upper, [0, 1, 2])


@pytest.mark.parametrize("coupling", [(1.0, 1.0), (1e-3, 1e-3), (1e6, 1e-6)])
def test_coupled_large_block_cannot_hide_growth(coupling):
    diag, ei, lower, upper, _ = _pair_plus_node(1e-18, 1e12, coupling=coupling)
    _assert_every_route_refuses(diag, ei, lower, upper, [0, 1, 2])


@pytest.mark.parametrize("dtype,eps,big,c", [(F64, 1e-18, 1e12, 1.0), (F32, 1e-9, 1e6, 1e-3)])
def test_tree_kernel_guard_is_local(dtype, eps, big, c):
    # tree 0 - 1 - 2 rooted at 1: M = [[eps, 1, 0], [2, 1, c], [0, c, big]]
    diag = torch.tensor([[[eps]], [[1.0]], [[big]]], dtype=dtype)
    edge_pc = torch.tensor([[[2.0]], [[0.0]], [[c]]], dtype=dtype)
    edge_cp = torch.tensor([[[1.0]], [[0.0]], [[c]]], dtype=dtype)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        selected_inverse_nonsym_tree(diag, edge_pc, edge_cp, [1, -1, 1])


@pytest.mark.parametrize("scale", [2.0 ** -40, 1e-10, 1e10, 2.0 ** 40])
def test_global_rescaling_keeps_the_decision(scale):
    diag, ei, lower, upper, M = _pair_plus_node(1e-18, 1.0, scale=scale)
    _assert_every_route_refuses(diag, ei, lower, upper, [0, 1, 2])
    # the safe order stays accepted and normwise accurate at every scale (G_11 ~ -5e-19 is
    # tiny next to the other blocks, so no stable method resolves it relatively)
    Gd, _S, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, lower, upper, [1, 0, 2])
    inv = torch.linalg.inv(M)
    got = torch.stack([Gd[0, 0, 0], Gd[1, 0, 0], Gl[0, 0, 0], Gu[0, 0, 0]])
    want = torch.stack([inv[0, 0], inv[1, 1], inv[1, 0], inv[0, 1]])
    torch.testing.assert_close(got, want, rtol=0, atol=1e-14 * float(want.abs().max()))
    torch.testing.assert_close(Gd[2, 0, 0], inv[2, 2], rtol=1e-15, atol=0)


def _dominant_tree_problem(n, b, seed, dtype=F64):
    """Strictly block-diagonally dominant non-symmetric matrix on a random tree."""
    gen = torch.Generator().manual_seed(seed)
    parent = [-1] + [int(torch.randint(0, v, (1,), generator=gen)) for v in range(1, n)]
    ei = torch.tensor([[v for v in range(1, n)], [parent[v] for v in range(1, n)]])
    m = n - 1
    lower = torch.randn(m, b, b, generator=gen, dtype=F64)
    upper = torch.randn(m, b, b, generator=gen, dtype=F64)
    diag = 0.3 * torch.randn(n, b, b, generator=gen, dtype=F64)
    rowsum = torch.zeros(n, dtype=F64)
    for k in range(m):
        i, j = int(ei[0, k]), int(ei[1, k])
        rowsum[i] += lower[k].abs().sum(-1).max()
        rowsum[j] += upper[k].abs().sum(-1).max()
    diag = diag + (2.0 * rowsum + 1.0)[:, None, None] * torch.eye(b, dtype=F64)
    return diag.to(dtype), ei, lower.to(dtype), upper.to(dtype)


def _dense(diag, ei, lower, upper):
    n, b = diag.shape[-3], diag.shape[-1]
    A = torch.zeros(n * b, n * b, dtype=F64)
    for v in range(n):
        A[v * b:(v + 1) * b, v * b:(v + 1) * b] = diag[v].double()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        A[i * b:(i + 1) * b, j * b:(j + 1) * b] = lower[k].double()
        A[j * b:(j + 1) * b, i * b:(i + 1) * b] = upper[k].double()
    return A


@pytest.mark.parametrize("seed", range(6))
def test_row_and_column_scaling_of_a_dominant_matrix_is_accepted(seed):
    # Block row/column scaling by factors spread over 12 decades leaves no-pivot elimination
    # exactly as safe as before, so the guard must not refuse; each selected block keeps
    # its relative accuracy.
    n, b = 8, 2
    diag, ei, lower, upper = _dominant_tree_problem(n, b, seed)
    gen = torch.Generator().manual_seed(100 + seed)
    r = 10.0 ** (12 * torch.rand(n, generator=gen, dtype=F64) - 6)
    c = 10.0 ** (12 * torch.rand(n, generator=gen, dtype=F64) - 6)
    diag = r[:, None, None] * diag * c[:, None, None]
    lower = r[ei[0]][:, None, None] * lower * c[ei[1]][:, None, None]
    upper = r[ei[1]][:, None, None] * upper * c[ei[0]][:, None, None]
    Gd, S, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, lower, upper)
    inv = torch.linalg.inv(_dense(diag, ei, lower, upper))
    for v in range(n):
        want = inv[v * b:(v + 1) * b, v * b:(v + 1) * b]
        assert float((Gd[v] - want).abs().max() / want.abs().max()) < 1e-10
    for k in range(S.shape[1]):
        i, j = int(S[0, k]), int(S[1, k])
        want = inv[i * b:(i + 1) * b, j * b:(j + 1) * b]
        assert float((Gl[k] - want).abs().max() / want.abs().max()) < 1e-10


def test_batch_elements_are_judged_separately():
    safe_big = _pair_plus_node(3.0, 1.0, scale=1e12)
    safe_small = _pair_plus_node(3.0, 1.0, scale=1e-12)
    bad = _pair_plus_node(1e-18, 1.0)
    ei = safe_big[1]
    stack = lambda *ps: [torch.stack([p[i] for p in ps]) for i in (0, 2, 3)]   # noqa: E731
    diag, lower, upper = stack(safe_big, safe_small)
    Gd, _S, _Gl, _Gu = selected_inverse_nonsym_junction(diag, ei, lower, upper, [0, 1, 2])
    torch.testing.assert_close(Gd[0, 0, 0, 0], torch.linalg.inv(safe_big[4])[0, 0], rtol=1e-12, atol=0)
    torch.testing.assert_close(Gd[1, 0, 0, 0], torch.linalg.inv(safe_small[4])[0, 0], rtol=1e-12, atol=0)
    diag, lower, upper = stack(safe_big, bad)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        selected_inverse_nonsym_junction(diag, ei, lower, upper, [0, 1, 2])


def test_ill_conditioned_but_safely_eliminable_matrix_is_accepted():
    # [[1, 1], [1, 1 + d]] has cond ~ 4/d but no growth from either pivot order.
    d = 1e-10
    diag = torch.tensor([[[1.0]], [[1.0 + d]]], dtype=F64)
    one = torch.tensor([[[1.0]]], dtype=F64)
    Gd, _S, Gl, Gu = selected_inverse_nonsym_junction(diag, torch.tensor([[1], [0]]), one, one, [0, 1])
    inv = torch.linalg.inv(torch.tensor([[1.0, 1.0], [1.0, 1.0 + d]], dtype=F64))
    cond_u = 4.0 / d * torch.finfo(F64).eps
    got = torch.stack([Gd[0, 0, 0], Gd[1, 0, 0], Gl[0, 0, 0], Gu[0, 0, 0]])
    want = torch.stack([inv[0, 0], inv[1, 1], inv[1, 0], inv[0, 1]])
    assert float(((got - want) / want).abs().max()) < 10 * cond_u


def test_spd_kernels_are_unaffected_by_a_disconnected_scale():
    # Cholesky needs no growth guard; the small component keeps full accuracy.
    diag = torch.tensor([[[2.0]], [[2.0]], [[1e12]]], dtype=F64)
    Gd, _S, Gl = selected_inverse_junction(diag, torch.tensor([[1], [0]]), torch.tensor([[[1.0]]], dtype=F64))
    inv = torch.linalg.inv(torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=F64))
    torch.testing.assert_close(torch.stack([Gd[0, 0, 0], Gd[1, 0, 0], Gl[0, 0, 0]]),
                               torch.stack([inv[0, 0], inv[1, 1], inv[1, 0]]), rtol=1e-15, atol=0)


def _three_node(d):
    """Scalar nodes; node 0 has pivot ``d`` and couples to nodes 1 and 2, which couple too."""
    M = torch.tensor([[d, 1.0, 1.0], [1.0, 1.0, 0.5], [2.0, 0.3, 1.0]], dtype=F64)
    ei = torch.tensor([[1, 2, 2], [0, 0, 1]])
    lower = torch.stack([M[1, 0], M[2, 0], M[2, 1]]).reshape(3, 1, 1)
    upper = torch.stack([M[0, 1], M[0, 2], M[1, 2]]).reshape(3, 1, 1)
    return torch.diagonal(M).reshape(3, 1, 1).clone(), ei, lower, upper, M


def test_selected_inverse_uses_the_stricter_limit(monkeypatch):
    # cond(M) = 5.6 and eliminating node 0 first grows entries by 1e7. A solve keeps about
    # u * growth accuracy; the selected inverse loses about u * growth**2, so only the solve
    # is accepted.
    diag, ei, lower, upper, M = _three_node(1e-7)
    rhs = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    x = nonsym_junction_solve(diag, ei, lower, upper, rhs, [0, 1, 2])
    torch.testing.assert_close(x.reshape(3), torch.linalg.solve(M, rhs.reshape(3)), rtol=1e-8, atol=0)
    for fn in (selected_inverse_nonsym_junction, selinv_nonsym_junction, selinv_nonsym_junction_analytic):
        with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
            fn(diag, ei, lower, upper, [0, 1, 2])
    # The refusal is not cosmetic: unguarded, the selected inverse is wrong in the third digit.
    monkeypatch.setattr(nonsym_junction_module, "check_pivot_growth", lambda *args: None)
    Gd = selected_inverse_nonsym_junction(diag, ei, lower, upper, [0, 1, 2])[0]
    assert float((Gd.reshape(3) - torch.diagonal(torch.linalg.inv(M))).abs().max()) > 1e-4


def test_selected_inverse_with_mild_growth_is_accepted():
    diag, ei, lower, upper, M = _three_node(1e-3)   # growth 1e3 < eps**-1/4; error ~ u * 1e6
    Gd, S, Gl, Gu = selected_inverse_nonsym_junction(diag, ei, lower, upper, [0, 1, 2])
    G = torch.linalg.inv(M)
    torch.testing.assert_close(Gd.reshape(3), torch.diagonal(G), rtol=0, atol=1e-9)
    for k in range(S.shape[1]):
        i, j = int(S[0, k]), int(S[1, k])
        torch.testing.assert_close(Gl[k, 0, 0], G[i, j], rtol=0, atol=1e-9)
        torch.testing.assert_close(Gu[k, 0, 0], G[j, i], rtol=0, atol=1e-9)


# A float32 tree (b = 2) found by a randomized search: its third pivot block is nearly singular
# (cond ~1e3), so the Schur entries grow only ~70x while the multipliers reach ~250. Measured on
# the Schur entries alone the solve passed 1/sqrt(eps) and came back wrong by ~1.6e-2; the
# factor product ||L|| ||D|| ||U|| exposes it.
_BLOCK_DIAG = [
    [[1.32459962, -0.847527921], [0.184109628, -0.261150002]],
    [[-0.742084026, -1.36627769], [0.538344741, 0.773586094]],
    [[2.40413106e-10, 2.11103617e-11], [-4.77599627e-10, -2.17550075e-10]],
    [[-0.689692795, -0.0445649475], [1.26866293, -0.0873004571]],
    [[-0.423422784, -0.306350023], [0.058731284, 0.765275538]],
    [[-0.314292639, -0.191784054], [0.295993149, 0.310011446]],
    [[0.338734925, -0.0855050907], [0.300398111, 0.377307773]],
]
_BLOCK_LOWER = [
    [[-0.444194019, 0.873988569], [-0.560843706, 0.655399024]],
    [[-0.739023745, -1.16598678], [0.447975785, 0.790261626]],
    [[-0.524009585, 0.841767013], [0.0771628916, 1.26780593]],
    [[0.116109967, 0.60290575], [0.940656424, -0.237142071]],
    [[0.0413374789, -1.16457438], [1.41832328, -0.00787506253]],
    [[-0.635557711, -0.824221313], [-2.89519286, 1.24651444]],
]
_BLOCK_UPPER = [
    [[-0.735293269, 1.71887898], [-0.118113548, -0.619832516]],
    [[1.06719136, 0.180206537], [0.560040474, -0.55420804]],
    [[1.71542799, -0.691467524], [-0.838608682, 1.52911711]],
    [[0.101322539, 2.20277643], [-0.325533152, -1.081002]],
    [[0.412261456, 0.418751299], [0.710394979, -1.12701321]],
    [[-1.07092106, -0.122146912], [0.684043825, 0.236305028]],
]


def test_block_pivot_conditioning_is_measured(monkeypatch):
    diag = torch.tensor(_BLOCK_DIAG, dtype=F32)
    lower = torch.tensor(_BLOCK_LOWER, dtype=F32)
    upper = torch.tensor(_BLOCK_UPPER, dtype=F32)
    ei = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 1, 2, 3, 4]])
    rhs = torch.linspace(-1, 1, 14, dtype=F32).reshape(7, 2)
    with pytest.raises(torch.linalg.LinAlgError, match="element growth"):
        nonsym_junction_solve(diag, ei, lower, upper, rhs)
    monkeypatch.setattr(nonsym_junction_module, "check_pivot_growth", lambda *args: None)
    x = nonsym_junction_solve(diag, ei, lower, upper, rhs).reshape(14).double()
    want = torch.linalg.solve(_dense(diag, ei, lower, upper), rhs.reshape(14).double())
    assert float((x - want).abs().max() / want.abs().max()) > 1e-3
