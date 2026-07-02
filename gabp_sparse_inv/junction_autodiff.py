"""Hand-written analytic backward for the filled-pattern (junction) selected inverses.

The autograd-facing :func:`~gabp_sparse_inv.selinv_junction` and
:func:`~gabp_sparse_inv.selinv_nonsym_junction` already give exact gradients by reverse-mode
through the functional forward -- which *is* the self-adjoint schedule of
``docs/derivations.md`` §8.4 / §10.3. This module is the **tape-free** version: a custom
``torch.autograd.Function`` whose backward runs the explicit reverse two-sweep clique
recurrence directly on the saved factors ``(chol/Dinv, ell/Lf/Uf, G|_S)``, with **no autograd
tape** over the per-node loop. It is the filled-pattern generalization of the tree analytic
backward in :mod:`gabp_sparse_inv.autodiff` (§8.3): the tree's single ``parent`` becomes the
clique ``U_v`` of later-eliminated neighbours, and the reverse-distribute / reverse-collect
sweeps carry cotangents over those cliques.

The reverse two passes (symmetric case; the non-symmetric case carries independent lower/upper
factors but the same structure):

- **reverse-distribute** (increasing elimination order): reverse of the Takahashi
  back-substitution (7): turns ``barG`` into factor cotangents ``bell`` and pivot-inverse
  cotangents ``bDinv``, propagating ``barG`` along ``S`` to lower-ordered nodes;
- **reverse-collect** (decreasing elimination order): reverse of the sparse block ``LDL^T``
  (resp. ``LDU``): the reverse clique-Schur update feeds factor cotangents, then the factor and
  pivot reverses produce the input cotangents ``barA`` on the diagonal and on the original edges
  (fill blocks are not inputs, so they carry no gradient).

Each block touched lies in ``S`` (chordality), so the schedule is ``O(fill)`` time and memory
like the forward -- the §8.4 self-adjoint cost, now realized without the autograd tape's
intermediates. Validated block-for-block against the autograd path (machine precision) and by
``gradcheck`` in ``tests/test_junction_autodiff.py``.

**First-order only** (the standard custom-``Function`` contract: ``backward`` runs under
no-grad). For Hessian-vector products use the functional :func:`~gabp_sparse_inv.selinv_junction`
/ :func:`~gabp_sparse_inv.selinv_nonsym_junction`, which double-back through autograd.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._linalg import inv_via_chol
from .junction import _normalize_edge_index, _selinv_junction_core
from .nonsym_junction import _selinv_nonsym_junction_core

__all__ = [
    "selinv_junction_analytic",
    "selinv_nonsym_junction_analytic",
    "SelInvJunction",
    "SelInvNonsymJunction",
]


def _sym(x: Tensor) -> Tensor:
    return 0.5 * (x + x.mT)


# --------------------------------------------------------------------------- #
# Symmetric SPD junction: analytic backward.
# --------------------------------------------------------------------------- #
def _backward_junction(ctx, barGd, barGl):
    """Reverse two-sweep (§8.4) for the SPD junction selected inverse."""
    chol_stack, ell_stack, G_diag, G_lower = ctx.saved_tensors
    order, pos, U, S_pairs = ctx.symbolic
    S_index = ctx.S_index
    n = ctx.n

    chol = {v: chol_stack[..., v, :, :] for v in range(n)}
    ell = {key: ell_stack[..., k, :, :] for k, key in enumerate(S_pairs)}
    Dv = {v: chol[v] @ chol[v].mT for v in range(n)}
    Dinv = {v: inv_via_chol(chol[v]) for v in range(n)}

    # G blocks in elimination-position orientation (G_off[(hi,lo)] = G_{hi,lo}).
    G_d = {v: G_diag[..., v, :, :] for v in range(n)}
    G_off = {}
    si = S_index.tolist()
    for k in range(S_index.shape[1]):
        i, j = si[0][k], si[1][k]                       # node i > j
        hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
        G_off[(hi, lo)] = G_lower[..., k, :, :] if hi == i else G_lower[..., k, :, :].mT

    def Gget(a, c):
        if a == c:
            return G_d[a]
        return G_off[(a, c)] if pos[a] > pos[c] else G_off[(c, a)].mT

    zero = torch.zeros_like(G_diag[..., 0, :, :])
    bG_d = {v: (barGd[..., v, :, :].clone() if barGd is not None else zero.clone()) for v in range(n)}
    bG_off = {key: zero.clone() for key in S_pairs}
    if barGl is not None:
        for k in range(S_index.shape[1]):
            i, j = si[0][k], si[1][k]
            hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
            bG_off[(hi, lo)] = bG_off[(hi, lo)] + (barGl[..., k, :, :] if hi == i else barGl[..., k, :, :].mT)

    def addG(a, c, X):
        if a == c:
            bG_d[a] = bG_d[a] + X
        elif pos[a] > pos[c]:
            bG_off[(a, c)] = bG_off[(a, c)] + X
        else:
            bG_off[(c, a)] = bG_off[(c, a)] + X.mT

    bell = {key: zero.clone() for key in S_pairs}
    bDinv = {v: zero.clone() for v in range(n)}
    bD = {v: zero.clone() for v in range(n)}
    bWd = {v: zero.clone() for v in range(n)}
    bW = {key: zero.clone() for key in S_pairs}

    # Pass 1 -- reverse-distribute (increasing elimination order).
    for v in order:
        Uv = U[v]
        M = _sym(bG_d[v])                                          # reverse of G_vv = sym(...)
        bDinv[v] = bDinv[v] + M
        for u in Uv:
            bell[(u, v)] = bell[(u, v)] - G_off[(u, v)] @ M.mT
            bG_off[(u, v)] = bG_off[(u, v)] - ell[(u, v)] @ M
        for w in Uv:                                              # reverse of G_{wv} = -sum_u G_{wu} ell_{uv}
            c = bG_off[(w, v)]
            for u in Uv:
                bell[(u, v)] = bell[(u, v)] - Gget(w, u).mT @ c
                addG(w, u, -c @ ell[(u, v)].mT)

    # Pass 2 -- reverse-collect (decreasing elimination order).
    barAd = {}
    for v in reversed(order):
        Uv = U[v]
        Dvv, Dinv_v = Dv[v], Dinv[v]
        for w in Uv:                                              # reverse clique-Schur update
            for wp in Uv:
                if pos[w] < pos[wp]:
                    continue
                bt = -(bWd[w] if w == wp else bW[(w, wp)])
                ell_w, ell_wp = ell[(w, v)], ell[(wp, v)]
                bell[(w, v)] = bell[(w, v)] + bt @ ell_wp @ Dvv.mT
                bD[v] = bD[v] + ell_w.mT @ bt @ ell_wp
                bell[(wp, v)] = bell[(wp, v)] + bt.mT @ ell_w @ Dvv
        for w in Uv:                                              # reverse factor ell_{wv} = W_{wv} Dinv_v
            be = bell[(w, v)]
            W_wv = ell[(w, v)] @ Dvv
            bW[(w, v)] = bW[(w, v)] + be @ Dinv_v
            bDinv[v] = bDinv[v] + W_wv.mT @ be
        bD[v] = bD[v] - Dinv_v @ bDinv[v] @ Dinv_v                # reverse Dinv = D^{-1}
        bWd[v] = bWd[v] + _sym(bD[v])                             # reverse D = sym(Wd)
        barAd[v] = bWd[v]

    barAd_t = torch.stack([barAd[v] for v in range(n)], dim=-3)
    ei = ctx.edge_index.tolist()
    m = ctx.edge_index.shape[1]
    if m:
        cols = []
        for k in range(m):
            i, j = ei[0][k], ei[1][k]
            hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
            cols.append(bW[(hi, lo)] if hi == i else bW[(hi, lo)].mT)
        barAe_t = torch.stack(cols, dim=-3)
    else:
        barAe_t = G_diag.new_zeros((*G_diag.shape[:-3], 0, ctx.b, ctx.b))
    return barAd_t, barAe_t


class SelInvJunction(torch.autograd.Function):
    """SPD junction selected inverse with the analytic §8.4 backward (use :func:`selinv_junction_analytic`)."""

    @staticmethod
    def forward(ctx, diag, edge_val, edge_index, order):
        n, b = diag.shape[-3], diag.shape[-1]
        G_diag, S_index, G_lower, factors = _selinv_junction_core(
            diag, edge_index, edge_val, order, return_factors=True
        )
        order_r, pos, U, S_pairs, chol, ell = factors
        chol_stack = torch.stack([chol[v] for v in range(n)], dim=-3)
        ell_stack = (
            torch.stack([ell[key] for key in S_pairs], dim=-3)
            if S_pairs else diag.new_zeros((*diag.shape[:-3], 0, b, b))
        )
        ctx.save_for_backward(chol_stack, ell_stack, G_diag, G_lower)
        ctx.symbolic = (order_r, pos, U, S_pairs)
        ctx.edge_index, ctx.S_index, ctx.n, ctx.b = edge_index, S_index, n, b
        return G_diag, S_index, G_lower

    @staticmethod
    def backward(ctx, barGd, barSi, barGl):
        barAd, barAe = _backward_junction(ctx, barGd, barGl)
        return barAd, barAe, None, None


def selinv_junction_analytic(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    order: Tensor | Sequence[int] | None = None,
):
    """SPD junction selected inverse with a hand-written, tape-free analytic backward.

    Same forward result as :func:`~gabp_sparse_inv.selinv_junction`, but gradients flow through
    the explicit reverse two-sweep clique recurrence of ``docs/derivations.md`` §8.4 (no autograd
    tape over the per-node loop), reusing the forward factors. ``O(fill)`` time and memory.
    Gradients to ``diag`` and ``edge_val``; **first-order only** (use the functional
    :func:`~gabp_sparse_inv.selinv_junction` for Hessian-vector products).
    """
    edge_index = _normalize_edge_index(edge_index)
    return SelInvJunction.apply(diag, edge_val, edge_index, order)


# --------------------------------------------------------------------------- #
# General non-symmetric (LU) junction: analytic backward.
# --------------------------------------------------------------------------- #
def _backward_nonsym_junction(ctx, barGd, barGl, barGu):
    """Reverse two-sweep (§10.3) for the non-symmetric junction selected inverse."""
    Dinv_stack, Lf_stack, Uf_stack, G_diag, G_lower, G_upper = ctx.saved_tensors
    order, pos, U, S_pairs = ctx.symbolic
    S_index = ctx.S_index
    n, b = ctx.n, ctx.b

    Dinv = {v: Dinv_stack[..., v, :, :] for v in range(n)}
    Dv = {v: torch.linalg.inv(Dinv[v]) for v in range(n)}
    Lf = {key: Lf_stack[..., k, :, :] for k, key in enumerate(S_pairs)}
    Uf = {key: Uf_stack[..., k, :, :] for k, key in enumerate(S_pairs)}

    # G blocks: G_L[(hi,lo)] = G_{hi,lo}, G_U[(hi,lo)] = G_{lo,hi}.
    G_d = {v: G_diag[..., v, :, :] for v in range(n)}
    G_L, G_U = {}, {}
    si = S_index.tolist()
    for k in range(S_index.shape[1]):
        i, j = si[0][k], si[1][k]
        hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
        gl, gu = G_lower[..., k, :, :], G_upper[..., k, :, :]
        G_L[(hi, lo)], G_U[(hi, lo)] = (gl, gu) if hi == i else (gu, gl)

    def Gget(a, c):
        if a == c:
            return G_d[a]
        return G_L[(a, c)] if pos[a] > pos[c] else G_U[(c, a)]

    zero = torch.zeros_like(G_diag[..., 0, :, :])
    bG_d = {v: (barGd[..., v, :, :].clone() if barGd is not None else zero.clone()) for v in range(n)}
    bG_L = {key: zero.clone() for key in S_pairs}
    bG_U = {key: zero.clone() for key in S_pairs}

    def addG(a, c, X):
        if a == c:
            bG_d[a] = bG_d[a] + X
        elif pos[a] > pos[c]:
            bG_L[(a, c)] = bG_L[(a, c)] + X
        else:
            bG_U[(c, a)] = bG_U[(c, a)] + X

    for k in range(S_index.shape[1]):
        i, j = si[0][k], si[1][k]
        if barGl is not None:
            addG(i, j, barGl[..., k, :, :])
        if barGu is not None:
            addG(j, i, barGu[..., k, :, :])

    bLf = {key: zero.clone() for key in S_pairs}
    bUf = {key: zero.clone() for key in S_pairs}
    bDinv = {v: zero.clone() for v in range(n)}
    bD = {v: zero.clone() for v in range(n)}
    bWd = {v: zero.clone() for v in range(n)}
    bWL = {key: zero.clone() for key in S_pairs}
    bWU = {key: zero.clone() for key in S_pairs}

    # Pass 1 -- reverse-distribute (increasing elimination order).
    for v in order:
        Uv = U[v]
        bgd = bG_d[v]                                             # reverse G_vv = Dinv_v - sum_u U_{vu} G_{uv}
        bDinv[v] = bDinv[v] + bgd
        for u in Uv:
            bUf[(u, v)] = bUf[(u, v)] - bgd @ G_L[(u, v)].mT
            bG_L[(u, v)] = bG_L[(u, v)] - Uf[(u, v)].mT @ bgd
        for w in Uv:                                             # reverse G_{vw} = -sum_u U_{vu} G_{uw}
            c = bG_U[(w, v)]
            for u in Uv:
                bUf[(u, v)] = bUf[(u, v)] - c @ Gget(u, w).mT
                addG(u, w, -Uf[(u, v)].mT @ c)
        for w in Uv:                                             # reverse G_{wv} = -sum_u G_{wu} L_{uv}
            c = bG_L[(w, v)]
            for u in Uv:
                addG(w, u, -c @ Lf[(u, v)].mT)
                bLf[(u, v)] = bLf[(u, v)] - Gget(w, u).mT @ c

    # Pass 2 -- reverse-collect (decreasing elimination order).
    barAd = {}
    for v in reversed(order):
        Uv = U[v]
        Dvv, Dinv_v = Dv[v], Dinv[v]
        for w in Uv:                                             # reverse clique-Schur: term = L_{wv} D_v U_{vwp}
            for wp in Uv:
                if w == wp:
                    bout = bWd[w]
                elif pos[w] > pos[wp]:
                    bout = bWL[(w, wp)]
                else:
                    bout = bWU[(wp, w)]
                bt = -bout
                Lf_w, Uf_wp = Lf[(w, v)], Uf[(wp, v)]
                bLf[(w, v)] = bLf[(w, v)] + bt @ Uf_wp.mT @ Dvv.mT
                bD[v] = bD[v] + Lf_w.mT @ bt @ Uf_wp.mT
                bUf[(wp, v)] = bUf[(wp, v)] + Dvv.mT @ Lf_w.mT @ bt
        for w in Uv:                                             # reverse factors L=WL Dinv, U=Dinv WU
            WL_wv, WU_wv = Lf[(w, v)] @ Dvv, Dvv @ Uf[(w, v)]
            bWL[(w, v)] = bWL[(w, v)] + bLf[(w, v)] @ Dinv_v.mT
            bDinv[v] = bDinv[v] + WL_wv.mT @ bLf[(w, v)]
            bWU[(w, v)] = bWU[(w, v)] + Dinv_v.mT @ bUf[(w, v)]
            bDinv[v] = bDinv[v] + bUf[(w, v)] @ WU_wv.mT
        bD[v] = bD[v] - Dinv_v.mT @ bDinv[v] @ Dinv_v.mT         # reverse Dinv = D^{-1} (general)
        bWd[v] = bWd[v] + bD[v]                                  # reverse D = Wd (no symmetrization)
        barAd[v] = bWd[v]

    barAd_t = torch.stack([barAd[v] for v in range(n)], dim=-3)
    ei = ctx.edge_index.tolist()
    m = ctx.edge_index.shape[1]
    if m:
        lo_cols, up_cols = [], []
        for k in range(m):
            i, j = ei[0][k], ei[1][k]
            hi, lo = (i, j) if pos[i] > pos[j] else (j, i)
            if hi == i:
                lo_cols.append(bWL[(i, j)]); up_cols.append(bWU[(i, j)])
            else:
                lo_cols.append(bWU[(j, i)]); up_cols.append(bWL[(j, i)])
        barAel = torch.stack(lo_cols, dim=-3)
        barAeu = torch.stack(up_cols, dim=-3)
    else:
        barAel = G_diag.new_zeros((*G_diag.shape[:-3], 0, b, b))
        barAeu = barAel.clone()
    return barAd_t, barAel, barAeu


class SelInvNonsymJunction(torch.autograd.Function):
    """Non-symmetric junction selected inverse with the analytic §10.3 backward."""

    @staticmethod
    def forward(ctx, diag, edge_lower, edge_upper, edge_index, order):
        n, b = diag.shape[-3], diag.shape[-1]
        G_diag, S_index, G_lower, G_upper, factors = _selinv_nonsym_junction_core(
            diag, edge_index, edge_lower, edge_upper, order, return_factors=True
        )
        order_r, pos, U, S_pairs, Dinv, Lf, Uf = factors
        Dinv_stack = torch.stack([Dinv[v] for v in range(n)], dim=-3)
        empty = diag.new_zeros((*diag.shape[:-3], 0, b, b))
        Lf_stack = torch.stack([Lf[k] for k in S_pairs], dim=-3) if S_pairs else empty
        Uf_stack = torch.stack([Uf[k] for k in S_pairs], dim=-3) if S_pairs else empty.clone()
        ctx.save_for_backward(Dinv_stack, Lf_stack, Uf_stack, G_diag, G_lower, G_upper)
        ctx.symbolic = (order_r, pos, U, S_pairs)
        ctx.edge_index, ctx.S_index, ctx.n, ctx.b = edge_index, S_index, n, b
        return G_diag, S_index, G_lower, G_upper

    @staticmethod
    def backward(ctx, barGd, barSi, barGl, barGu):
        barAd, barAel, barAeu = _backward_nonsym_junction(ctx, barGd, barGl, barGu)
        return barAd, barAel, barAeu, None, None


def selinv_nonsym_junction_analytic(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_lower: Tensor,
    edge_upper: Tensor,
    order: Tensor | Sequence[int] | None = None,
):
    """Non-symmetric junction selected inverse with a hand-written, tape-free analytic backward.

    Same forward result as :func:`~gabp_sparse_inv.selinv_nonsym_junction`, but gradients flow
    through the explicit reverse two-sweep clique recurrence of ``docs/derivations.md`` §10.3 --
    the non-symmetric analogue of the §8.4 schedule, carrying independent lower/upper factor
    cotangents -- with no autograd tape. ``O(fill)`` time and memory. Gradients to ``diag``,
    ``edge_lower`` and ``edge_upper``; **first-order only** (use the functional
    :func:`~gabp_sparse_inv.selinv_nonsym_junction` for Hessian-vector products).
    """
    edge_index = _normalize_edge_index(edge_index)
    return SelInvNonsymJunction.apply(diag, edge_lower, edge_upper, edge_index, order)
