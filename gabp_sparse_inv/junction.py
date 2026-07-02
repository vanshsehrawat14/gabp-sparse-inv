"""General sparse SPD selected inverse via the elimination tree (junction tree).

This is the Phase-4 generalization of the tree kernel (:mod:`gabp_sparse_inv.tree`):
the input SPD block matrix may have an **arbitrary** sparsity pattern, not just a
tree. The implementation picks an elimination order, symbolically completes the pattern to its
**chordal (filled)** closure ``S = pattern(L + L^T)`` (``docs/derivations.md`` §2.1),
and compute every block of ``A^{-1}`` on ``S`` -- the node diagonals plus the filled
off-diagonal blocks -- without ever forming the dense inverse.

Trees are the zero-fill special case (``S`` = the input pattern, ``U_v`` =
``{parent}``); there the recurrences below reduce *block-for-block* to ``tree.py``
eqs. (5)-(6), which the tests assert.

Algorithm (the multi-neighbour Takahashi / Erisman-Tinney recurrence; derived and
proved self-adjoint in ``docs/derivations.md`` §8.4). Number the nodes by their
elimination order; for a node ``v`` let ``U_v`` be its neighbours eliminated *later*.
By chordality ``U_v`` is a **clique** in the filled graph -- which is exactly why
every block the distribute step reads is itself on ``S`` and already computed. With
pivots ``D_v`` and factor blocks ``ell_{wv} = A_{wv} D_v^{-1}``:

    collect (eliminate v, low -> high order):
        D_v        = A_vv          (after earlier Schur updates have landed in it)
        ell_{wv}   = A_{wv} D_v^{-1}                          for w in U_v
        A_{w,w'}  -= ell_{wv} D_v ell_{w'v}^T                 for w, w' in U_v   (fill)

    distribute (back-substitute, high -> low order):
        G_{wv}     = - sum_{u in U_v} G_{wu} ell_{uv}         for w in U_v
        G_vv       =   D_v^{-1} - sum_{u in U_v} ell_{uv}^T G_{uv}

**Differentiability.** The core is written functionally with differentiable block
ops (``cholesky`` / ``cholesky_solve``), so reverse-mode autograd *through it* is the
``S``-local self-adjoint schedule proved in §8.4 -- no custom backward needed for
correct gradients. :func:`selinv_junction` is that autograd-facing entry point. A
hand-written analytic clique backward (the §8.3 analogue, which avoids the autograd
tape's ``O(fill)`` memory) is a tracked follow-up; this per-node loop is the
correctness reference, exactly as ``tree.py`` preceded the batched ``autodiff.py``.

**Scope.** A minimal, dependency-free reference: a greedy **min-degree** elimination
order and a per-node Python loop with dict-of-blocks sparse storage. No
supernodal/multifrontal engineering and no external ordering library (AMD/METIS) --
those are deliberate non-goals (``docs/ROADMAP.md``); a better ordering only changes
the *amount* of fill, not the result.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._linalg import cholesky_spd, inv_via_chol

__all__ = [
    "selected_inverse_junction",
    "selinv_junction",
    "junction_solve",
    "junction_logdet",
    "elimination_order_min_degree",
    "elimination_order_nested_dissection",
]


def _normalize_edge_index(edge_index: Tensor | Sequence[Sequence[int]]) -> Tensor:
    """Normalize an edge index to a 1-D CPU ``long`` ``[2, m]`` tensor (not batched)."""
    if isinstance(edge_index, Tensor):
        ei = edge_index.detach().to(device="cpu", dtype=torch.long)
    else:
        ei = torch.tensor(edge_index, dtype=torch.long, device="cpu")
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, m]; got {tuple(ei.shape)}")
    return ei


def elimination_order_min_degree(n: int, nbr: list[set[int]]) -> list[int]:
    """Greedy minimum-degree elimination order on the fill graph.

    Repeatedly eliminate the remaining node of least current degree, connecting its
    surviving neighbours into a clique (the fill it creates). Ties break to the lowest
    index. This is the classic heuristic; it is *not* optimal (nested dissection does
    better on grids), but ordering quality only affects how much fill appears, never
    the computed selected inverse.
    """
    work = [set(s) for s in nbr]
    eliminated = [False] * n
    order: list[int] = []
    for _ in range(n):
        v = min((u for u in range(n) if not eliminated[u]), key=lambda u: len(work[u]))
        eliminated[v] = True
        order.append(v)
        survivors = [w for w in work[v] if not eliminated[w]]
        for a in survivors:
            work[a].discard(v)
            for c in survivors:
                if a != c:
                    work[a].add(c)
    return order


def elimination_order_nested_dissection(rows: int, cols: int) -> list[int]:
    """Nested-dissection elimination order for a ``rows x cols`` 4-neighbour grid.

    Node ``(r, c)`` has id ``r * cols + c`` (the :func:`grid_edges` convention).
    Recursively bisect the longer side of each sub-rectangle by its middle line: order
    the two halves first (eliminated earlier) and the separator line last. On a 2-D grid
    this gives ``O(n log n)`` fill and ``O(n^{3/2})`` work -- asymptotically below the
    ``O(n^{3/2})`` fill of a lexicographic (row-major) order (empirically the fill ratio
    vs lexicographic falls from ~0.7 at 12x12 to ~0.47 at 32x32). It is a deterministic,
    dimension-aware alternative to the pattern-only greedy
    :func:`elimination_order_min_degree` (which is competitive, and often lower fill on
    *small* grids, but has no asymptotic grid guarantee). The selected inverse is
    **invariant** to the order -- only the *amount* of fill changes.

    Returns a permutation of ``range(rows * cols)`` (separators last) suitable as the
    ``order`` argument of :func:`selected_inverse_junction` / :func:`junction_solve` /
    :func:`junction_logdet`.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1; got {rows}x{cols}")
    order: list[int] = []

    def rec(r0: int, r1: int, c0: int, c1: int) -> None:
        if r0 > r1 or c0 > c1:
            return
        if r0 == r1 and c0 == c1:
            order.append(r0 * cols + c0)
            return
        if (r1 - r0) >= (c1 - c0):                       # bisect the taller side
            rm = (r0 + r1) // 2
            rec(r0, rm - 1, c0, c1)
            rec(rm + 1, r1, c0, c1)
            order.extend(rm * cols + c for c in range(c0, c1 + 1))
        else:                                            # bisect the wider side
            cm = (c0 + c1) // 2
            rec(r0, r1, c0, cm - 1)
            rec(r0, r1, cm + 1, c1)
            order.extend(r * cols + cm for r in range(r0, r1 + 1))

    rec(0, rows - 1, 0, cols - 1)
    return order


def _symbolic(n, edge_index, order):
    """Symbolic factorization: returns ``(order, pos, U, S_pairs)``.

    ``U[v]`` lists the neighbours of ``v`` (in the *filled* graph) eliminated after
    ``v``, sorted by elimination position; ``pos[v]`` is ``v``'s rank in ``order``.
    ``S_pairs`` is the list of off-diagonal selected blocks as ``(hi, lo)`` tuples
    oriented by *elimination position* (``pos[hi] > pos[lo]``). The diagonal is
    implicitly all of ``0..n-1``.
    """
    nbr: list[set[int]] = [set() for _ in range(n)]
    ij = edge_index.tolist()
    for a, c in zip(ij[0], ij[1]):
        if a == c:
            raise ValueError(f"self-loop in edge_index at node {a}")
        nbr[a].add(c)
        nbr[c].add(a)

    if order is None:
        order = elimination_order_min_degree(n, nbr)
    else:
        order = (order.tolist() if isinstance(order, Tensor) else list(order))
        if sorted(order) != list(range(n)):
            raise ValueError("order must be a permutation of range(n)")

    pos = [0] * n
    for rank, v in enumerate(order):
        pos[v] = rank

    # Elimination game on a working copy: eliminating v cliques its higher neighbours.
    work = [set(s) for s in nbr]
    U: list[list[int]] = [[] for _ in range(n)]
    for v in order:
        higher = sorted((w for w in work[v] if pos[w] > pos[v]), key=lambda w: pos[w])
        U[v] = higher
        for a in higher:
            for c in higher:
                if a != c:
                    work[a].add(c)

    S_pairs = [(w, v) for v in range(n) for w in U[v]]  # (hi, lo) by elimination pos
    return order, pos, U, S_pairs


def _factor_junction(diag, edge_index, edge_val, order):
    """Shared sparse block ``LDL^T`` factorization (the collect pass).

    Returns ``(order, pos, U, S_pairs, chol, ell)`` where, in elimination order, the
    pivot blocks are ``D_v`` with Cholesky factor ``chol[v]`` and the strictly-lower
    factor blocks are ``ell[(w, v)] = A_{wv} D_v^{-1}`` for ``w in U_v`` (keyed by
    elimination-position orientation, ``pos[w] > pos[v]``). Both the selected inverse
    (:func:`_selinv_junction_core`) and the solve (:func:`_solve_junction_core`) build
    on this single factorization path. Functional / autograd-traceable.
    """
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    m = edge_index.shape[1]

    order, pos, U, S_pairs = _symbolic(n, edge_index, order)

    def sym(x):
        return 0.5 * (x + x.mT)

    # ---- working matrix on S (pos-oriented). Out-of-place only (autograd-safe). ----
    Wd = {v: diag[..., v, :, :] for v in range(n)}                 # diagonal blocks
    W = {key: diag.new_zeros((*batch, b, b)) for key in S_pairs}    # off-diagonal blocks
    ij = edge_index.tolist()
    for k in range(m):
        i, j = ij[0][k], ij[1][k]
        blk = edge_val[..., k, :, :]                               # A_{i,j} (i > j node idx)
        if pos[i] > pos[j]:
            W[(i, j)] = W[(i, j)] + blk
        else:
            W[(j, i)] = W[(j, i)] + blk.mT                         # store A_{j,i} = A_{i,j}^T

    # ---- collect: sparse block Cholesky with the clique (Schur) fill update --------
    chol = {}
    ell = {}
    for v in order:
        Dv = sym(Wd[v])
        Rv = cholesky_spd(Dv, name=f"D_{v}")          # informative SPD-failure message
        chol[v] = Rv
        Uv = U[v]
        for w in Uv:
            A_wv = W[(w, v)]                                       # (hi=w, lo=v)
            ell[(w, v)] = torch.cholesky_solve(A_wv.mT, Rv).mT     # A_wv D_v^{-1}
        # A_{w,w'} -= ell_{wv} D_v ell_{w'v}^T over the clique U_v (each off block once).
        for w in Uv:
            ell_wv = ell[(w, v)]
            for wp in Uv:
                if pos[w] < pos[wp]:
                    continue                                      # the (wp, w) pass owns it
                term = ell_wv @ (Dv @ ell[(wp, v)].mT)
                if w == wp:
                    Wd[w] = Wd[w] - term
                else:
                    W[(w, wp)] = W[(w, wp)] - term

    return order, pos, U, S_pairs, chol, ell


def _selinv_junction_core(diag, edge_index, edge_val, order, return_factors=False):
    """Functional (autograd-traceable) forward. Returns ``(G_diag, S_index, G_lower)``.

    Storage is a dict of ``b x b`` blocks keyed by elimination-position orientation
    ``(hi, lo)`` (so ``hi`` is eliminated after ``lo``); ``G_diag``/``G_lower`` are
    re-emitted in *node-index* orientation (``i > j``) to match the input layout.

    With ``return_factors=True`` also returns the collect-pass factors
    ``(order, pos, U, S_pairs, chol, ell)`` (the analytic backward reuses them; see
    :mod:`gabp_sparse_inv.junction_autodiff`).
    """
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]

    order, pos, U, S_pairs, chol, ell = _factor_junction(diag, edge_index, edge_val, order)

    def sym(x):
        return 0.5 * (x + x.mT)

    # ---- distribute: Takahashi back-substitution restricted to S -------------------
    G_d = {}
    G_off = {}

    def Gget(a, c):                                                # G_{a,c} (row a, col c)
        if a == c:
            return G_d[a]
        if pos[a] > pos[c]:
            return G_off[(a, c)]
        return G_off[(c, a)].mT                                   # G symmetric

    for v in reversed(order):
        Uv = U[v]
        for w in Uv:                                              # cross blocks G_{w,v}
            acc = None
            for u in Uv:
                t = Gget(w, u) @ ell[(u, v)]
                acc = t if acc is None else acc + t
            G_off[(w, v)] = -acc if acc is not None else diag.new_zeros((*batch, b, b))
        Dv_inv = inv_via_chol(chol[v])                            # diagonal block G_vv
        acc = None
        for u in Uv:
            t = ell[(u, v)].mT @ G_off[(u, v)]
            acc = t if acc is None else acc + t
        G_vv = Dv_inv if acc is None else Dv_inv - acc
        G_d[v] = sym(G_vv)

    # ---- emit in node-index orientation (i > j) ------------------------------------
    G_diag = torch.stack([G_d[v] for v in range(n)], dim=-3)
    if S_pairs:
        idx_list, blk_list = [], []
        for (w, v) in S_pairs:
            i, j = (w, v) if w > v else (v, w)                    # node-index orientation
            idx_list.append((i, j))
            blk_list.append(Gget(i, j))
        S_index = torch.tensor(list(zip(*idx_list)), dtype=torch.long)
        G_lower = torch.stack(blk_list, dim=-3)
    else:
        S_index = torch.zeros((2, 0), dtype=torch.long)
        G_lower = diag.new_zeros((*batch, 0, b, b))
    if return_factors:
        return G_diag, S_index, G_lower, (order, pos, U, S_pairs, chol, ell)
    return G_diag, S_index, G_lower


class _JunctionLevel:
    """Flattened per-level index tensors for the batched (level-set) schedule.

    One :class:`_JunctionLevel` holds, for a single elimination level (an antichain of
    mutually non-adjacent nodes in the filled graph), the long index tensors that drive
    the batched collect and distribute block ops. ``nodes`` are the level's nodes; the
    remaining fields enumerate the clique work as flat pair / triple lists, so each level
    is a constant number of batched ``index_select`` / ``index_add`` calls rather than a
    Python loop over nodes and their cliques. Block indices ``< n`` address node diagonals
    and ``>= n`` address off-diagonal slots (slot ``k`` -> index ``n + k``), so one unified
    block array carries both with a single indexing scheme.
    """

    __slots__ = (
        "nodes",                                  # level nodes (Cholesky / G_vv)
        "ell_dest", "ell_lo",                     # ell_{wv}: dest slot, pivot node v
        "sc_ellw", "sc_ellwp", "sc_v", "sc_dest", # collect Schur fill over the clique
        "cr_dest", "cr_ell", "cr_gidx", "cr_gT",  # distribute cross blocks G_{w,v}
        "dg_v", "dg_ell",                         # distribute diagonal G_vv accumulation
    )

    def __init__(self, **fields):
        for key in self.__slots__:
            setattr(self, key, fields[key])


def _batched_symbolic(n, edge_index, order, device):
    """Symbolic plan for the level-set batched schedule (values-independent).

    Extends :func:`_symbolic` with the **elimination levels** ``level[v] = 1 +
    max(level[u])`` over ``v``'s *lower* filled-neighbours ``u`` (those eliminated before
    ``v``). Adjacent nodes are always ancestor/descendant in the elimination tree, so
    same-level nodes are pairwise non-adjacent in the filled graph -- which is exactly why
    a whole level's collect updates (and, in reverse, its distribute reads) are
    independent and batch. Returns ``(nS, init_dest, init_mask, S_index, emit_mask,
    levels)``: ``nS`` off-diagonal slots in the same order as :func:`_symbolic`'s
    ``S_pairs`` (so results align block-for-block with the reference), the input-scatter
    and emit index/transpose tensors, and a list of :class:`_JunctionLevel`.
    """
    order, pos, U, S_pairs = _symbolic(n, edge_index, order)
    slot = {pair: k for k, pair in enumerate(S_pairs)}
    nS = len(S_pairs)

    # ---- elimination levels (a node's lower filled-neighbours fix its level) ----------
    lower: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        for w in U[u]:
            lower[w].append(u)                       # u is a lower neighbour of w
    level = [0] * n
    for v in order:                                  # increasing pos: lower nbrs leveled
        level[v] = 0 if not lower[v] else 1 + max(level[u] for u in lower[v])
    num_levels = (max(level) + 1) if n else 0
    by_level: list[list[int]] = [[] for _ in range(num_levels)]
    for v in range(n):
        by_level[level[v]].append(v)

    def lt(xs):
        return torch.tensor(xs, dtype=torch.long, device=device)

    def bt(xs):
        return torch.tensor(xs, dtype=torch.bool, device=device)

    # ---- input scatter: edge A_{i,j} (node i>j) -> off-slot (elimination orientation) -
    ij = edge_index.tolist()
    init_dest, init_mask = [], []
    for k in range(edge_index.shape[1]):
        i, j = ij[0][k], ij[1][k]
        if pos[i] > pos[j]:
            init_dest.append(n + slot[(i, j)]); init_mask.append(False)
        else:
            init_dest.append(n + slot[(j, i)]); init_mask.append(True)   # store A_{i,j}^T

    # ---- emit: off-slot (elimination orientation) -> node orientation (i>j) + transpose
    s_cols, emit_mask = [], []
    for (w, v) in S_pairs:
        if w > v:
            s_cols.append((w, v)); emit_mask.append(False)
        else:
            s_cols.append((v, w)); emit_mask.append(True)
    S_index = (torch.tensor(list(zip(*s_cols)), dtype=torch.long)
               if s_cols else torch.zeros((2, 0), dtype=torch.long))

    # ---- per-level flattened clique work (mirrors the reference recurrences) ----------
    levels = []
    for L in range(num_levels):
        nodes = by_level[L]
        ell_dest, ell_lo = [], []
        sc_ellw, sc_ellwp, sc_v, sc_dest = [], [], [], []
        cr_dest, cr_ell, cr_gidx, cr_gT = [], [], [], []
        dg_v, dg_ell = [], []
        for v in nodes:
            Uv = U[v]
            for w in Uv:                              # ell_{wv} = A_{wv} D_v^{-1}
                ell_dest.append(slot[(w, v)]); ell_lo.append(v)
            for w in Uv:                              # A_{w,w'} -= ell_{wv} D_v ell_{w'v}^T
                for wp in Uv:
                    if pos[w] < pos[wp]:
                        continue                      # the (wp, w) pass owns this block
                    sc_ellw.append(slot[(w, v)])
                    sc_ellwp.append(slot[(wp, v)])
                    sc_v.append(v)
                    sc_dest.append(w if w == wp else (n + slot[(w, wp)]))
            for w in Uv:                              # G_{w,v} = -sum_u G_{w,u} ell_{u,v}
                for u in Uv:
                    cr_dest.append(n + slot[(w, v)])
                    cr_ell.append(slot[(u, v)])
                    if w == u:
                        cr_gidx.append(w); cr_gT.append(False)            # G_{w,u} = G_uu
                    elif pos[w] > pos[u]:
                        cr_gidx.append(n + slot[(w, u)]); cr_gT.append(False)
                    else:
                        cr_gidx.append(n + slot[(u, w)]); cr_gT.append(True)
            for u in Uv:                              # G_vv = D_v^{-1} - sum_u ell_{u,v}^T G_{u,v}
                dg_v.append(v); dg_ell.append(slot[(u, v)])
        levels.append(_JunctionLevel(
            nodes=lt(nodes),
            ell_dest=lt(ell_dest), ell_lo=lt(ell_lo),
            sc_ellw=lt(sc_ellw), sc_ellwp=lt(sc_ellwp), sc_v=lt(sc_v), sc_dest=lt(sc_dest),
            cr_dest=lt(cr_dest), cr_ell=lt(cr_ell), cr_gidx=lt(cr_gidx), cr_gT=bt(cr_gT),
            dg_v=lt(dg_v), dg_ell=lt(dg_ell),
        ))

    return nS, lt(init_dest), bt(init_mask), S_index, bt(emit_mask), levels


def _selinv_junction_batched(diag, edge_index, edge_val, order):
    """Level-set batched forward (functional / autograd-traceable).

    Mathematically identical to the per-node reference :func:`_selinv_junction_core`
    (validated block-for-block in ``tests/test_junction_batched.py``), but the Python loop
    runs over *elimination levels* -- ``O(elimination-tree height)``: ``~log n`` balanced,
    ``~sqrt(n)`` on a 2-D grid -- instead of over nodes, with each level a handful of
    batched block ops. The unified working array ``Mbuf`` (and result ``Gbuf``) stack the
    ``n`` node diagonals over the ``nS`` off-diagonal slots, so the variable-size clique
    updates become flat ``index_select`` / ``index_add`` scatters. Reverse-mode autograd
    through it is the §8.4 self-adjoint schedule, so ``selinv_junction(..., batched=True)``
    gets exact gradients with no custom backward -- exactly as the per-node path does.
    """
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    dev = diag.device

    nS, init_dest, init_mask, S_index, emit_mask, levels = _batched_symbolic(
        n, edge_index, order, dev
    )

    def sym(x):
        return 0.5 * (x + x.mT)

    # ---- unified working block array: [ D (n node diagonals) | W (nS off-blocks) ] ----
    Mbuf = torch.cat([diag, diag.new_zeros((*batch, nS, b, b))], dim=-3)
    if init_dest.numel() > 0:
        contrib = torch.where(init_mask[:, None, None], edge_val.mT, edge_val)
        Mbuf = Mbuf.index_add(-3, init_dest, contrib)

    chol_buf = diag.new_zeros((*batch, n, b, b))     # write-once per node (index_add from 0)
    Dsym_buf = diag.new_zeros((*batch, n, b, b))      # symmetrized pivots, for the Schur term
    ell_buf = diag.new_zeros((*batch, nS, b, b))      # write-once per slot

    # ---- collect: sparse block Cholesky + clique Schur fill, increasing level ----------
    for h, lv in enumerate(levels):
        Dl = sym(Mbuf.index_select(-3, lv.nodes))
        chol_l = cholesky_spd(Dl, name=f"D(level {h})")
        chol_buf = chol_buf.index_add(-3, lv.nodes, chol_l)
        Dsym_buf = Dsym_buf.index_add(-3, lv.nodes, Dl)
        if lv.ell_dest.numel() > 0:
            A_wv = Mbuf.index_select(-3, lv.ell_dest + n)
            chol_v = chol_buf.index_select(-3, lv.ell_lo)
            ell_l = torch.cholesky_solve(A_wv.mT, chol_v).mT
            ell_buf = ell_buf.index_add(-3, lv.ell_dest, ell_l)
        if lv.sc_dest.numel() > 0:
            ellw = ell_buf.index_select(-3, lv.sc_ellw)
            ellwp = ell_buf.index_select(-3, lv.sc_ellwp)
            Dv = Dsym_buf.index_select(-3, lv.sc_v)
            Mbuf = Mbuf.index_add(-3, lv.sc_dest, -(ellw @ (Dv @ ellwp.mT)))

    # ---- distribute: Takahashi back-substitution restricted to S, decreasing level -----
    Gbuf = diag.new_zeros((*batch, n + nS, b, b))
    for lv in reversed(levels):
        if lv.cr_dest.numel() > 0:                    # cross blocks G_{w,v}
            Gsrc = Gbuf.index_select(-3, lv.cr_gidx)
            Gsrc = torch.where(lv.cr_gT[:, None, None], Gsrc.mT, Gsrc)
            Gbuf = Gbuf.index_add(-3, lv.cr_dest, -(Gsrc @ ell_buf.index_select(-3, lv.cr_ell)))
        Dinv_l = inv_via_chol(chol_buf.index_select(-3, lv.nodes))   # diagonal blocks G_vv
        if lv.dg_v.numel() > 0:
            elluv = ell_buf.index_select(-3, lv.dg_ell)
            Guv = Gbuf.index_select(-3, lv.dg_ell + n)
            acc = diag.new_zeros((*batch, n, b, b)).index_add(-3, lv.dg_v, elluv.mT @ Guv)
            Gvv = Dinv_l - acc.index_select(-3, lv.nodes)
        else:
            Gvv = Dinv_l
        Gbuf = Gbuf.index_add(-3, lv.nodes, sym(Gvv))

    G_diag = Gbuf[..., :n, :, :]
    if nS > 0:
        G_off = Gbuf[..., n:, :, :]
        G_lower = torch.where(emit_mask[:, None, None], G_off.mT, G_off)
    else:
        G_lower = diag.new_zeros((*batch, 0, b, b))
    return G_diag, S_index, G_lower


def selected_inverse_junction(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
    batched: bool = False,
):
    """Selected inverse of a general sparse SPD block matrix on its filled pattern.

    Parameters
    ----------
    diag:
        Node diagonal blocks ``A_vv``: ``[..., n, b, b]`` (each symmetric).
    edge_index:
        ``[2, m]`` long tensor of off-diagonal block positions ``(i, j)`` with
        ``i > j`` (node index); the upper triangle is implied by symmetry. Shared
        across leading batch dims (not batched).
    edge_val:
        Off-diagonal blocks ``edge_val[..., k, :, :] = A_{i_k, j_k}``: ``[..., m, b, b]``.
    order:
        Optional explicit elimination order (a permutation of ``range(n)``). Defaults
        to a greedy minimum-degree order computed from the pattern.
    check:
        If ``True``, validate inputs (shapes / dtype / symmetry / index range) first.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast
        back). Mirrors the chain/star/tree kernels: models low-precision *storage* while
        computing in a dtype the LAPACK backend supports (e.g. bf16 storage, fp32 compute).
    batched:
        If ``True``, use the level-set batched path: the Python loop runs over elimination
        levels (``O(tree height)``) instead of nodes, with each level a handful of batched
        block ops scattered with ``index_add`` (the junction analogue of
        ``selected_inverse_tree(batched=True)``). Mathematically identical (validated
        block-for-block); amortizes kernel-launch latency on GPU. The per-node loop remains
        the default correctness reference.

    Returns
    -------
    (G_diag, S_index, G_lower):
        ``G_diag`` ``[..., n, b, b]`` holds the node diagonals ``G_vv``. ``S_index``
        ``[2, m_S]`` is the **filled** off-diagonal pattern (``i > j``; a superset of
        ``edge_index`` when elimination creates fill), and ``G_lower``
        ``[..., m_S, b, b]`` holds ``G_{i_k, j_k}`` (so ``G_{j,i} = G_lower[k].mT``).
    """
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=_normalize_edge_index(edge_index), edge_val=edge_val).validate()
    edge_index = _normalize_edge_index(edge_index)
    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_val = edge_val.to(compute_dtype)
    forward = _selinv_junction_batched if batched else _selinv_junction_core
    G_diag, S_index, G_lower = forward(diag, edge_index, edge_val, order)
    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_lower = G_lower.to(in_dtype)
    return G_diag, S_index, G_lower


def _solve_junction_core(diag, edge_index, edge_val, rhs, order):
    """Functional (autograd-traceable) sparse SPD solve ``x = A^{-1} b`` on the fill.

    Uses the shared ``LDL^T`` factorization (:func:`_factor_junction`) and a standard
    block triangular solve in elimination order: ``L y = b`` (increasing order),
    ``D z = y`` (per-node, via the Cholesky factor), ``L^T x = z`` (decreasing order).
    ``rhs`` is normalized internally to ``[..., n, b, k]``; the result matches.
    """
    n, b = diag.shape[-3], diag.shape[-1]
    order, pos, U, S_pairs, chol, ell = _factor_junction(diag, edge_index, edge_val, order)

    # incoming[w] = nodes u eliminated before w with w in U_u (i.e. ell[(w, u)] exists).
    incoming: list[list[int]] = [[] for _ in range(n)]
    for (w, v) in ell:
        incoming[w].append(v)

    # ---- forward solve  L y = b   (increasing elimination order) -------------------
    y = {}
    for v in order:
        acc = rhs[..., v, :, :]
        for u in incoming[v]:
            acc = acc - ell[(v, u)] @ y[u]
        y[v] = acc

    # ---- diagonal solve  D z = y   (per node, via the Cholesky factor) -------------
    z = {v: torch.cholesky_solve(y[v], chol[v]) for v in range(n)}

    # ---- back solve  L^T x = z   (decreasing elimination order) --------------------
    x = {}
    for v in reversed(order):
        acc = z[v]
        for w in U[v]:
            acc = acc - ell[(w, v)].mT @ x[w]
        x[v] = acc

    return torch.stack([x[v] for v in range(n)], dim=-3)           # [..., n, b, k]


def junction_solve(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    b: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Differentiable sparse SPD linear solve ``x = A^{-1} b`` on a general pattern.

    Solves ``A x = b`` for a symmetric positive-definite block matrix ``A`` given in the
    :class:`~gabp_sparse_inv.layout.BlockSparseSym` layout (node diagonals ``diag`` and
    lower-triangular off-diagonal blocks ``edge_val`` on ``edge_index``), via the same
    symbolic min-degree elimination + sparse block ``LDL^T`` factorization as
    :func:`selected_inverse_junction`. The factorization path is shared (one
    :func:`_factor_junction`), so this is the solve sibling of the selected inverse.

    Parameters
    ----------
    diag, edge_index, edge_val, order:
        As in :func:`selected_inverse_junction` (``diag`` ``[..., n, b, b]``;
        ``edge_index`` ``[2, m]`` with ``i > j``; ``edge_val`` ``[..., m, b, b]``;
        ``order`` an optional elimination order, default greedy min-degree).
    b:
        Right-hand side ``[..., n, b]`` (a single vector per node) or ``[..., n, b, k]``
        (``k`` right-hand sides). The result has the matching shape.
    check:
        If ``True``, validate the matrix inputs first.
    compute_dtype:
        If given, all linear algebra runs in this dtype (``diag``, ``edge_val`` and ``b``
        upcast, result cast back). Mirrors :func:`selected_inverse_junction`.

    Returns
    -------
    x:
        The solution ``A^{-1} b`` shaped like ``b`` (``[..., n, b]`` or ``[..., n, b, k]``).

    Notes
    -----
    Functional / autograd-traceable: gradients of any loss over ``x`` flow to ``diag``,
    ``edge_val`` and ``b`` by reverse-mode (no custom backward).
    """
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=_normalize_edge_index(edge_index), edge_val=edge_val).validate()
    edge_index = _normalize_edge_index(edge_index)
    n, bs = diag.shape[-3], diag.shape[-1]
    vector_rhs = b.shape[-2] == n and b.shape[-1] == bs            # [..., n, b]
    matrix_rhs = b.ndim >= 3 and b.shape[-3] == n and b.shape[-2] == bs  # [..., n, b, k]
    if not (vector_rhs or matrix_rhs):
        raise ValueError(
            f"b must have shape [..., n, b] or [..., n, b, k] with n={n}, b={bs}; got {tuple(b.shape)}"
        )
    rhs = b.unsqueeze(-1) if vector_rhs else b                     # -> [..., n, b, k]
    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_val = edge_val.to(compute_dtype)
        rhs = rhs.to(compute_dtype)
    x = _solve_junction_core(diag, edge_index, edge_val, rhs, order)
    if compute_dtype is not None and compute_dtype != in_dtype:
        x = x.to(in_dtype)
    return x.squeeze(-1) if vector_rhs else x


def junction_logdet(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Log-determinant ``log det A`` from the junction ``LDL^T`` factorization.

    With ``A = L D L^T`` and unit block-lower-triangular ``L`` (so ``det L = 1``),
    ``log det A = sum_v log det D_v = sum_v 2 * sum(log diag(R_v))`` where ``R_v`` is the
    Cholesky factor of the pivot ``D_v`` (``chol[v]`` from the shared
    :func:`_factor_junction` collect pass). Differentiable (the sibling of
    :func:`~gabp_sparse_inv.tree_logdet`).

    Parameters
    ----------
    diag, edge_index, edge_val, order, check:
        As in :func:`selected_inverse_junction`.
    compute_dtype:
        If given, the factorization runs in this dtype (inputs upcast, result cast back).

    Returns
    -------
    logdet:
        ``log det A`` with shape equal to the leading batch dims (a scalar per problem).
    """
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=_normalize_edge_index(edge_index), edge_val=edge_val).validate()
    edge_index = _normalize_edge_index(edge_index)
    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_val = edge_val.to(compute_dtype)
    n = diag.shape[-3]
    _order, _pos, _U, _S_pairs, chol, _ell = _factor_junction(diag, edge_index, edge_val, order)
    total = None
    for v in range(n):
        ld = 2.0 * torch.log(torch.diagonal(chol[v], dim1=-2, dim2=-1)).sum(-1)
        total = ld if total is None else total + ld
    if compute_dtype is not None and compute_dtype != in_dtype:
        total = total.to(in_dtype)
    return total


def selinv_junction(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_val: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    batched: bool = False,
):
    """Autograd-facing selected inverse on the filled pattern (gradients to the inputs).

    Identical result to :func:`selected_inverse_junction`; the name mirrors the other
    ``selinv_*`` kernels. The functional forward is itself differentiable, so gradients
    of any loss over ``G_diag`` / ``G_lower`` flow to ``diag`` and ``edge_val`` by
    reverse-mode through the ``S``-local schedule -- which is the self-adjoint schedule
    proved in ``docs/derivations.md`` §8.4. (A hand-written analytic clique backward,
    which avoids the autograd tape's memory, is a tracked follow-up.)

    ``batched=True`` selects the level-set batched forward (see
    :func:`selected_inverse_junction`); the result and its gradients are identical (the
    batched path is likewise functional, so reverse-mode gives the same self-adjoint
    backward), and only the scheduling -- per elimination level vs per node -- differs.
    """
    edge_index = _normalize_edge_index(edge_index)
    forward = _selinv_junction_batched if batched else _selinv_junction_core
    return forward(diag, edge_index, edge_val, order)
