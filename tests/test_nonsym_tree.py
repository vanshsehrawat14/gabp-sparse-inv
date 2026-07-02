"""Correctness gates for the non-symmetric *tree* selected inverse.

``selected_inverse_nonsym_tree`` is the zero-fill non-symmetric rung between the block
lower-bidiagonal kernel and the general LU selected inverse: the off-diagonal pattern is
a tree, but ``M_{uv}`` and ``M_{vu}`` are independent. Gates mirror the SPD-tree and
non-symmetric-bidiagonal suites: a dense ``torch.linalg.inv`` oracle on the assembled
matrix, the symmetric special case reducing to the SPD-tree kernel, batch consistency,
``compute_dtype``, and autograd (``gradcheck`` + analytic-vs-dense adjoint + higher order).
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    selected_inverse_nonsym_tree,
    selected_inverse_tree,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Topologies + fixture assembly.
# --------------------------------------------------------------------------- #
def _parent(kind: str, n: int, *, seed: int = 0) -> list[int]:
    if kind == "path":
        return [-1] + list(range(n - 1))
    if kind == "star":
        return [-1] + [0] * (n - 1)
    if kind == "balanced":
        return [-1] + [(v - 1) // 2 for v in range(1, n)]
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        out = [-1]
        for v in range(1, n):
            out.append(int(torch.randint(0, v, (1,), generator=g)))
        return out
    raise ValueError(kind)


def _make_nonsym_tree(n, b, parent, *, seed, diag_load=4.0, off_scale=0.3, batch=()):
    """Random non-symmetric block-tree blocks (each pivot well-conditioned)."""
    g = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64)

    eye = torch.eye(b, dtype=torch.float64)
    diag = rn(*batch, n, b, b) + diag_load * eye
    edge_pc = off_scale * rn(*batch, n, b, b)   # edge_pc[v] = M_{p(v), v}
    edge_cp = off_scale * rn(*batch, n, b, b)   # edge_cp[v] = M_{v, p(v)}
    return diag, edge_pc, edge_cp


def _assemble(diag, edge_pc, edge_cp, parent):
    """Autograd-traceable dense assembly of the non-symmetric tree matrix M."""
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    M = diag.new_zeros((*diag.shape[:-3], N, N))
    for v in range(n):
        r = slice(v * b, (v + 1) * b)
        M[..., r, r] = diag[..., v, :, :]
    for v in range(n):
        p = parent[v]
        if p < 0:
            continue
        rp, rv = slice(p * b, (p + 1) * b), slice(v * b, (v + 1) * b)
        M[..., rp, rv] = edge_pc[..., v, :, :]   # M_{p,v}
        M[..., rv, rp] = edge_cp[..., v, :, :]   # M_{v,p}
    return M


def _oracle_selected(diag, edge_pc, edge_cp, parent):
    n, b = diag.shape[-3], diag.shape[-1]
    G = torch.linalg.inv(_assemble(diag, edge_pc, edge_cp, parent).to(torch.float64))

    def blk(i, j):
        return G[..., i * b:(i + 1) * b, j * b:(j + 1) * b]

    od = torch.stack([blk(v, v) for v in range(n)], dim=-3)
    opc, ocp = [], []
    for v in range(n):
        p = parent[v]
        if p < 0:
            z = G.new_zeros((*G.shape[:-2], b, b))
            opc.append(z)
            ocp.append(z)
        else:
            opc.append(blk(p, v))
            ocp.append(blk(v, p))
    return od, torch.stack(opc, dim=-3), torch.stack(ocp, dim=-3)


def _selected_from_dense(diag, edge_pc, edge_cp, parent):
    """Differentiable dense oracle: assemble, invert, gather selected blocks."""
    return _oracle_selected(diag, edge_pc, edge_cp, parent)


def _rel_err(got, ref):
    num = sum((g.to(torch.float64) - r).pow(2).sum() for g, r in zip(got, ref))
    den = sum(r.pow(2).sum() for r in ref)
    return float((num / den).sqrt())


# --------------------------------------------------------------------------- #
# Forward: fp64 matches the dense oracle across topologies.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["path", "star", "balanced", "random"])
@pytest.mark.parametrize("n,b", [(8, 3), (15, 2), (6, 4)])
def test_forward_matches_oracle(kind, n, b):
    parent = _parent(kind, n, seed=n + b)
    diag, epc, ecp = _make_nonsym_tree(n, b, parent, seed=n * 7 + b)
    G = selected_inverse_nonsym_tree(diag, epc, ecp, parent, check=True)
    ref = _oracle_selected(diag, epc, ecp, parent)
    assert _rel_err(G, ref) < 1e-9


# --------------------------------------------------------------------------- #
# Symmetric special case reduces block-for-block to the SPD-tree kernel.
# --------------------------------------------------------------------------- #
def test_symmetric_reduces_to_spd_tree():
    from gabp_sparse_inv import random_spd_tree

    bt = random_spd_tree(12, 3, seed=2, kind="random", diag_load=1.0)
    parent = bt.parent.tolist()
    edge_pc = bt.edge                       # M_{p(v),v}
    edge_cp = bt.edge.mT                    # symmetric: M_{v,p(v)} = M_{p(v),v}^T
    Gd, Gpc, Gcp = selected_inverse_nonsym_tree(bt.diag, edge_pc, edge_cp, parent)
    Gd_s, Ge_s = selected_inverse_tree(bt.diag, bt.edge, bt.parent)
    assert torch.allclose(Gd, Gd_s, atol=1e-10)
    assert torch.allclose(Gpc, Ge_s, atol=1e-10)
    assert torch.allclose(Gcp, Gpc.mT, atol=1e-10)   # G_{v,p} = G_{p,v}^T when symmetric


# --------------------------------------------------------------------------- #
# Edge cases / batching / precision.
# --------------------------------------------------------------------------- #
def test_n1():
    diag, epc, ecp = _make_nonsym_tree(1, 5, [-1], seed=1)
    Gd, Gpc, Gcp = selected_inverse_nonsym_tree(diag, epc, ecp, [-1])
    assert torch.allclose(Gd[0], torch.linalg.inv(diag[0]), atol=1e-10)
    assert torch.allclose(Gpc[0], torch.zeros(5, 5, dtype=torch.float64))
    assert torch.allclose(Gcp[0], torch.zeros(5, 5, dtype=torch.float64))


def test_scalar_blocks_b1():
    parent = _parent("random", 30, seed=4)
    diag, epc, ecp = _make_nonsym_tree(30, 1, parent, seed=9)
    G = selected_inverse_nonsym_tree(diag, epc, ecp, parent)
    assert _rel_err(G, _oracle_selected(diag, epc, ecp, parent)) < 1e-9


def test_leading_batch_matches_per_item():
    n, b, batch = 10, 3, 4
    parent = _parent("random", n, seed=5)
    diag, epc, ecp = _make_nonsym_tree(n, b, parent, seed=5, batch=(batch,))
    Gd, Gpc, Gcp = selected_inverse_nonsym_tree(diag, epc, ecp, parent)
    for k in range(batch):
        gd, gpc, gcp = selected_inverse_nonsym_tree(diag[k], epc[k], ecp[k], parent)
        assert torch.allclose(Gd[k], gd, atol=1e-12)
        assert torch.allclose(Gpc[k], gpc, atol=1e-12)
        assert torch.allclose(Gcp[k], gcp, atol=1e-12)


def test_compute_dtype_fp32_storage_fp64_compute():
    parent = _parent("balanced", 15, seed=3)
    diag, epc, ecp = _make_nonsym_tree(15, 3, parent, seed=8)
    G32 = selected_inverse_nonsym_tree(
        diag.float(), epc.float(), ecp.float(), parent, compute_dtype=torch.float64
    )
    assert G32[0].dtype == torch.float32          # cast back to storage dtype
    assert _rel_err(G32, _oracle_selected(diag, epc, ecp, parent)) < 1e-5


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def test_check_rejects_bad_edge_shape():
    parent = _parent("path", 5, seed=0)
    diag, epc, ecp = _make_nonsym_tree(5, 3, parent, seed=0)
    with pytest.raises(ValueError):
        selected_inverse_nonsym_tree(diag, epc[:-1], ecp, parent, check=True)


def test_check_rejects_parent_length_mismatch():
    parent = _parent("path", 5, seed=0)
    diag, epc, ecp = _make_nonsym_tree(5, 3, parent, seed=0)
    with pytest.raises(ValueError):
        selected_inverse_nonsym_tree(diag, epc, ecp, parent[:-1], check=True)


# --------------------------------------------------------------------------- #
# Backward: gradcheck, analytic-vs-dense adjoint, higher order.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,n,b", [("path", 5, 2), ("star", 6, 2), ("balanced", 7, 2)])
def test_gradcheck(kind, n, b):
    parent = _parent(kind, n, seed=n)
    diag, epc, ecp = _make_nonsym_tree(n, b, parent, seed=n + b, diag_load=4.0)
    d = diag.clone().requires_grad_(True)
    a = epc.clone().requires_grad_(True)
    c = ecp.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda d, a, c: selected_inverse_nonsym_tree(d, a, c, parent),
        (d, a, c), atol=1e-6, rtol=1e-4,
    )


def _grads(fn, diag0, epc0, ecp0, w):
    d = diag0.clone().requires_grad_(True)
    a = epc0.clone().requires_grad_(True)
    c = ecp0.clone().requires_grad_(True)
    Gd, Gpc, Gcp = fn(d, a, c)
    ((Gd * w[0]).sum() + (Gpc * w[1]).sum() + (Gcp * w[2]).sum()).backward()
    return d.grad.clone(), a.grad.clone(), c.grad.clone()


@pytest.mark.parametrize("kind,n,b", [("random", 10, 3), ("path", 8, 2), ("balanced", 7, 3)])
def test_analytic_vs_dense_adjoint(kind, n, b):
    parent = _parent(kind, n, seed=n * 2 + b)
    diag, epc, ecp = _make_nonsym_tree(n, b, parent, seed=n + b, diag_load=4.0)
    g = torch.Generator().manual_seed(n * 13 + b)
    w = [torch.randn(n, b, b, generator=g, dtype=torch.float64) for _ in range(3)]
    gk = _grads(lambda d, a, c: selected_inverse_nonsym_tree(d, a, c, parent), diag, epc, ecp, w)
    go = _grads(lambda d, a, c: _selected_from_dense(d, a, c, parent), diag, epc, ecp, w)
    for x, y in zip(gk, go):
        assert torch.allclose(x, y, atol=1e-9), (x - y).abs().max()


def test_gradgradcheck_small():
    parent = _parent("path", 4, seed=1)
    diag, epc, ecp = _make_nonsym_tree(4, 2, parent, seed=2, diag_load=4.0)
    d = diag.clone().requires_grad_(True)
    a = epc.clone().requires_grad_(True)
    c = ecp.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda d, a, c: selected_inverse_nonsym_tree(d, a, c, parent),
        (d, a, c), atol=1e-5, rtol=1e-4,
    )


# --------------------------------------------------------------------------- #
# Oracle-free on-pattern residual: M G = I block-wise from the selected blocks
# alone (no dense inverse). On a tree node v's neighbours are its parent and
# children, so the diagonal block of M M^{-1} closes over the selected set:
#   (M G)_vv = M_vv G_vv + M_{v,p} G_{p,v} + sum_{c in children} M_{v,c} G_{c,v} = I.
# --------------------------------------------------------------------------- #
def test_onpattern_residual_oracle_free():
    parent = _parent("random", 12, seed=7)
    diag, epc, ecp = _make_nonsym_tree(12, 3, parent, seed=15, diag_load=4.0)
    Gd, Gpc, Gcp = selected_inverse_nonsym_tree(diag, epc, ecp, parent)
    n, b = diag.shape[-3], diag.shape[-1]
    children: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    eye = torch.eye(b, dtype=torch.float64)
    for v in range(n):
        R = diag[v] @ Gd[v]
        if parent[v] >= 0:
            R = R + ecp[v] @ Gpc[v]                   # M_{v,p} G_{p,v}
        for c in children[v]:
            R = R + epc[c] @ Gcp[c]                   # M_{v,c} G_{c,v}
        assert torch.allclose(R, eye, atol=1e-9), f"residual node {v}"


def _ill_conditioned_path(n=8, b=2, delta=1e-3, off=0.05, seed=0):
    """A path tree with one tiny diagonal block -> kappa ~ 1/delta, pivots safe.

    Identity diagonals keep every Schur pivot ~ I (invertible) while the single
    ``delta * I`` block drives the condition number up in a controlled, non-flaky way.
    """
    g = torch.Generator().manual_seed(seed)
    eye = torch.eye(b, dtype=torch.float64)
    diag = eye.expand(n, b, b).clone()
    diag[n // 2] = delta * eye
    epc = off * torch.randn(n, b, b, generator=g, dtype=torch.float64)
    ecp = off * torch.randn(n, b, b, generator=g, dtype=torch.float64)
    parent = [-1] + list(range(n - 1))
    return diag, epc, ecp, parent


def test_ill_conditioned_fp32_accuracy_tracks_kappa():
    # Genuine fp32 storage AND compute on an ill-conditioned instance: the on-pattern error
    # must stay within ~kappa * u_fp32 (the condition-aware forward-error story), closing the
    # SPD-tree-vs-nonsym-tree gap (the nonsym suite previously had no ill-conditioned gate).
    diag, epc, ecp, parent = _ill_conditioned_path(n=8, b=2, delta=1e-3)
    kappa = float(torch.linalg.cond(_assemble(diag, epc, ecp, parent)))
    assert kappa > 1e2                                # genuinely ill-conditioned
    G32 = selected_inverse_nonsym_tree(diag.float(), epc.float(), ecp.float(), parent)
    assert G32[0].dtype == torch.float32
    rel = _rel_err(G32, _oracle_selected(diag, epc, ecp, parent))
    u32 = torch.finfo(torch.float32).eps
    assert rel < max(1e4 * kappa * u32, 1e-4)


def test_bf16_storage_runs():
    # bf16 storage with fp32 compute must run and return bf16 in a loose neighbourhood of the
    # oracle (bf16 storage is coarse) -- the store-low path the SPD kernels also expose.
    parent = _parent("balanced", 15, seed=3)
    diag, epc, ecp = _make_nonsym_tree(15, 3, parent, seed=8, diag_load=4.0)
    G = selected_inverse_nonsym_tree(
        diag.bfloat16(), epc.bfloat16(), ecp.bfloat16(), parent, compute_dtype=torch.float32
    )
    assert G[0].dtype == torch.bfloat16
    assert _rel_err(G, _oracle_selected(diag, epc, ecp, parent)) < 5e-1
