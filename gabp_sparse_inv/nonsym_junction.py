"""General non-symmetric (LU / Erisman-Tinney) selected inverse on the filled pattern.

The non-symmetric generalization of :mod:`gabp_sparse_inv.junction`, and the headline
open rung of the program (``docs/ROADMAP.md`` step 5): the input block matrix ``A`` has an
**arbitrary structurally-symmetric** sparsity pattern (``(i,j)`` present iff ``(j,i)`` is),
but ``A_{ij}`` and ``A_{ji}`` are **independent** (``A != A^T``). The implementation picks an elimination
order, complete the pattern to its chordal closure ``S`` (the *same* fill as the symmetric
case, since the pattern is structurally symmetric), and compute every block of ``A^{-1}`` on
``S`` -- the node diagonals plus **both** off-diagonal orientations -- without forming the
dense inverse.

Algorithm (block ``LDU`` + the two-sided Takahashi / Erisman-Tinney 1975 recurrence). Number
nodes by elimination order; for node ``v`` let ``U_v`` be its later-eliminated neighbours (a
clique in the filled graph). The symmetric kernel keeps one factor ``ell``; here the lower and
upper factors differ and are tracked separately:

    collect (eliminate v, low -> high order):
        D_v      = A_vv                      (after earlier Schur updates)
        L_{wv}   = A_{wv} D_v^{-1}           upper-neighbour lower factor   (w in U_v)
        U_{vw}   = D_v^{-1} A_{vw}           upper-neighbour upper factor   (w in U_v)
        A_{w,w'} -= A_{wv} D_v^{-1} A_{vw'}  Schur fill over the clique U_v (both orientations)

    distribute (back-substitute, high -> low order):
        G_{wv}   = - sum_{u in U_v} G_{wu} L_{uv}        (lower selected blocks, row w > col v)
        G_{vw}   = - sum_{u in U_v} U_{vu} G_{uw}        (upper selected blocks, row v < col w)
        G_vv     =   D_v^{-1} - sum_{u in U_v} U_{vu} G_{uv}

When ``A`` is symmetric (``A_{vw} = A_{wv}^T``), one gets ``U_{vw} = L_{wv}^T`` and ``G_{vw} =
G_{wv}^T``, and every line reduces block-for-block to :mod:`gabp_sparse_inv.junction`; on a tree
(zero fill) it reduces to :func:`~gabp_sparse_inv.selected_inverse_nonsym_tree`. Both reductions
are asserted in the tests.

**No pivoting (the static-pattern regime).** The factorization eliminates in the fixed
symbolic order with **no pivoting**, which is what keeps the pattern static and the cost
front-dependent -- but requires every block pivot ``D_v`` to stay non-singular (e.g. block
diagonal dominance / the well-scaled regime; ``docs/ROADMAP.md`` "Non-symmetric stability").
Pivoting would make the symbolic pattern data-dependent; it is not handled here.

**Differentiability.** The core is functional (the block inverses are ``torch.linalg.inv``),
so reverse-mode autograd through it is the ``S``-local adjoint-up-to-transpose schedule
-- the non-symmetric analogue of the §8.4 argument. Formally the inversion derivative obeys
``L_A^* = L_{A^T}``; the VJP is not itself a selected inverse of ``A^T``
(``docs/derivations.md`` §10). The hand-written analytic backward is shipped in
:mod:`gabp_sparse_inv.junction_autodiff`.

**Solve sibling.** The same block ``LDU`` factorization yields a linear *solve*
``x = A^{-1} b`` (and the transpose ``A^{-T} b``) via standard block triangular substitution:
:func:`nonsym_junction_solve`, the non-symmetric counterpart of
:func:`~gabp_sparse_inv.junction_solve`. The transpose solve is the implicit-differentiation
adjoint a fixed-point / DEQ backward needs (``demos/deq_fixedpoint.py``), reusing the *same*
factors with no refactorization.

**Scope.** A minimal reference: the dependency-free min-degree order from
:mod:`gabp_sparse_inv.junction` and a per-node Python loop. No supernodal engineering, no
pivoting, no external ordering library -- the same deliberate non-goals as the symmetric kernel.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .junction import _normalize_edge_index, _symbolic

__all__ = [
    "selected_inverse_nonsym_junction",
    "selinv_nonsym_junction",
    "nonsym_junction_solve",
]


def _factor_nonsym(diag, edge_index, edge_lower, edge_upper, order):
    """Block ``LDU`` factorization on the filled pattern (the collect pass), functional.

    Returns ``(order, pos, U, S_pairs, Dinv, Lf, Uf)`` where, keyed by elimination-position
    orientation ``(hi, lo)`` with ``pos[hi] > pos[lo]``: ``Dinv[v] = D_v^{-1}``,
    ``Lf[(w,v)] = L_{wv} = A_{wv} D_v^{-1}`` (lower factor) and ``Uf[(w,v)] = U_{vw} =
    D_v^{-1} A_{vw}`` (upper factor).
    """
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    m = edge_index.shape[1]

    order, pos, U, S_pairs = _symbolic(n, edge_index, order)

    # Working blocks on S (pos-oriented): WL[(hi,lo)] = A_{hi,lo}, WU[(hi,lo)] = A_{lo,hi}.
    Wd = {v: diag[..., v, :, :] for v in range(n)}
    WL = {key: diag.new_zeros((*batch, b, b)) for key in S_pairs}
    WU = {key: diag.new_zeros((*batch, b, b)) for key in S_pairs}
    ij = edge_index.tolist()
    for k in range(m):
        i, j = ij[0][k], ij[1][k]                      # node index, i > j
        lo_blk = edge_lower[..., k, :, :]              # A_{i,j}
        up_blk = edge_upper[..., k, :, :]              # A_{j,i}
        if pos[i] > pos[j]:                            # pos-pair (hi=i, lo=j)
            WL[(i, j)] = WL[(i, j)] + lo_blk           # A_{hi,lo} = A_{i,j}
            WU[(i, j)] = WU[(i, j)] + up_blk           # A_{lo,hi} = A_{j,i}
        else:                                          # pos-pair (hi=j, lo=i)
            WL[(j, i)] = WL[(j, i)] + up_blk           # A_{hi,lo} = A_{j,i}
            WU[(j, i)] = WU[(j, i)] + lo_blk           # A_{lo,hi} = A_{i,j}

    Dinv, Lf, Uf = {}, {}, {}
    for v in order:
        Dv_inv = torch.linalg.inv(Wd[v])
        Dinv[v] = Dv_inv
        Uv = U[v]
        for w in Uv:
            Lf[(w, v)] = WL[(w, v)] @ Dv_inv           # L_{w,v}
            Uf[(w, v)] = Dv_inv @ WU[(w, v)]           # U_{v,w}
        # Schur over the clique: block(row w, col w') -= A_{w,v} D_v^{-1} A_{v,w'}.
        for w in Uv:
            AwvDinv = WL[(w, v)] @ Dv_inv
            for wp in Uv:
                term = AwvDinv @ WU[(wp, v)]            # A_{w,v} D_v^{-1} A_{v,w'}
                if w == wp:
                    Wd[w] = Wd[w] - term
                elif pos[w] > pos[wp]:
                    WL[(w, wp)] = WL[(w, wp)] - term    # lower block (row w > col w')
                else:
                    WU[(wp, w)] = WU[(wp, w)] - term    # upper block (row w < col w')

    return order, pos, U, S_pairs, Dinv, Lf, Uf


def _selinv_nonsym_junction_core(diag, edge_index, edge_lower, edge_upper, order, return_factors=False):
    """Functional forward. Returns ``(G_diag, S_index, G_lower, G_upper)`` (node-index ``i>j``).

    With ``return_factors=True`` also returns ``(order, pos, U, S_pairs, Dinv, Lf, Uf)`` (the
    analytic backward reuses them; see :mod:`gabp_sparse_inv.junction_autodiff`).
    """
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]

    order, pos, U, S_pairs, Dinv, Lf, Uf = _factor_nonsym(diag, edge_index, edge_lower, edge_upper, order)

    G_d, G_L, G_U = {}, {}, {}                          # G_L[(hi,lo)]=G_{hi,lo}; G_U[(hi,lo)]=G_{lo,hi}

    def Gget(a, c):                                      # G_{a,c}
        if a == c:
            return G_d[a]
        return G_L[(a, c)] if pos[a] > pos[c] else G_U[(c, a)]

    zero = diag.new_zeros((*batch, b, b))
    for v in reversed(order):
        Uv = U[v]
        for w in Uv:
            accL = None                                 # G_{w,v} = -sum_u G_{w,u} L_{u,v}
            for u in Uv:
                t = Gget(w, u) @ Lf[(u, v)]
                accL = t if accL is None else accL + t
            G_L[(w, v)] = -accL if accL is not None else zero
            accU = None                                 # G_{v,w} = -sum_u U_{v,u} G_{u,w}
            for u in Uv:
                t = Uf[(u, v)] @ Gget(u, w)
                accU = t if accU is None else accU + t
            G_U[(w, v)] = -accU if accU is not None else zero
        acc = None                                      # G_vv = D_v^{-1} - sum_u U_{v,u} G_{u,v}
        for u in Uv:
            t = Uf[(u, v)] @ G_L[(u, v)]
            acc = t if acc is None else acc + t
        G_d[v] = Dinv[v] if acc is None else Dinv[v] - acc

    G_diag = torch.stack([G_d[v] for v in range(n)], dim=-3)
    if S_pairs:
        idx_list, lo_list, up_list = [], [], []
        for (w, v) in S_pairs:
            i, j = (w, v) if w > v else (v, w)          # node-index orientation (i > j)
            idx_list.append((i, j))
            lo_list.append(Gget(i, j))                  # G_{i,j}
            up_list.append(Gget(j, i))                  # G_{j,i}
        S_index = torch.tensor(list(zip(*idx_list)), dtype=torch.long)
        G_lower = torch.stack(lo_list, dim=-3)
        G_upper = torch.stack(up_list, dim=-3)
    else:
        S_index = torch.zeros((2, 0), dtype=torch.long)
        G_lower = diag.new_zeros((*batch, 0, b, b))
        G_upper = diag.new_zeros((*batch, 0, b, b))
    if return_factors:
        return G_diag, S_index, G_lower, G_upper, (order, pos, U, S_pairs, Dinv, Lf, Uf)
    return G_diag, S_index, G_lower, G_upper


def selected_inverse_nonsym_junction(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_lower: Tensor,
    edge_upper: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Selected inverse of a general non-symmetric sparse block matrix on its filled pattern.

    Parameters
    ----------
    diag:
        Node diagonal blocks ``A_vv``: ``[..., n, b, b]`` (general square blocks).
    edge_index:
        ``[2, m]`` long tensor of off-diagonal positions ``(i, j)`` with ``i > j`` (node
        index). The pattern is structurally symmetric: each ``(i, j)`` carries *both*
        ``A_{i,j}`` and ``A_{j,i}`` (independent). Shared across leading batch dims.
    edge_lower:
        Lower blocks ``edge_lower[..., k, :, :] = A_{i_k, j_k}`` (``i_k > j_k``): ``[..., m, b, b]``.
    edge_upper:
        Upper blocks ``edge_upper[..., k, :, :] = A_{j_k, i_k}``: ``[..., m, b, b]``. For
        symmetric ``A`` this equals ``edge_lower[k].mT``.
    order:
        Optional explicit elimination order (a permutation of ``range(n)``); default greedy
        min-degree (``docs/ROADMAP.md`` non-goals: no AMD/METIS).
    check:
        If ``True``, validate shapes and the ``i > j`` / in-range index convention first.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast back).

    Returns
    -------
    (G_diag, S_index, G_lower, G_upper):
        ``G_diag`` ``[..., n, b, b]`` holds ``G_vv``. ``S_index`` ``[2, m_S]`` is the filled
        pattern (``i > j``); ``G_lower[..., k, :, :] = G_{i_k, j_k}`` and ``G_upper[..., k, :, :]
        = G_{j_k, i_k}`` (both orientations, since ``A^{-1}`` is non-symmetric).
    """
    edge_index = _normalize_edge_index(edge_index)
    if check:
        n, b = diag.shape[-3], diag.shape[-1]
        if edge_lower.shape[-3:] != (edge_index.shape[1], b, b) or edge_upper.shape[-3:] != (edge_index.shape[1], b, b):
            raise ValueError("edge_lower/edge_upper must have shape [..., m, b, b] matching edge_index")
        ij = edge_index.tolist()
        for i, j in zip(ij[0], ij[1]):
            if not (n > i > j >= 0):
                raise ValueError(f"edge_index must be lower-triangular in range (n>i>j>=0); got ({i},{j})")
        if len({(i, j) for i, j in zip(ij[0], ij[1])}) != edge_index.shape[1]:
            raise ValueError("duplicate edges in edge_index")

    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_lower = edge_lower.to(compute_dtype)
        edge_upper = edge_upper.to(compute_dtype)
    G_diag, S_index, G_lower, G_upper = _selinv_nonsym_junction_core(diag, edge_index, edge_lower, edge_upper, order)
    if compute_dtype is not None and compute_dtype != in_dtype:
        G_diag = G_diag.to(in_dtype)
        G_lower = G_lower.to(in_dtype)
        G_upper = G_upper.to(in_dtype)
    return G_diag, S_index, G_lower, G_upper


def _solve_nonsym_junction_core(diag, edge_index, edge_lower, edge_upper, rhs, order, transpose):
    """Functional (autograd-traceable) non-symmetric solve on the filled pattern.

    Uses the shared block ``LDU`` factorization (:func:`_factor_nonsym`) and a standard
    block triangular solve in elimination order. With ``A = L D U`` (unit lower ``L``, unit
    upper ``U``, block-diagonal ``D``), ``A^{-1} b`` is ``L y = b`` (increasing order),
    ``D z = y`` (per node), ``U x = z`` (decreasing order). For ``transpose=True`` it solves
    ``A^T x = b`` via ``A^T = U^T D^T L^T`` -- the same factors transposed, with the roles of
    ``Lf`` and ``Uf`` swapped -- which is exactly the implicit-differentiation adjoint solve.
    ``rhs`` is ``[..., n, b, k]``; the result matches.
    """
    n = diag.shape[-3]
    order, pos, U, S_pairs, Dinv, Lf, Uf = _factor_nonsym(diag, edge_index, edge_lower, edge_upper, order)

    # incoming[w] = nodes u eliminated before w with w in U_u (Lf/Uf share these keys).
    incoming: list[list[int]] = [[] for _ in range(n)]
    for (w, v) in Lf:
        incoming[w].append(v)

    # A = L D U: fwd factor blocks are Lf (lower), back factor blocks are Uf (upper), pivot D.
    # A^T = U^T D^T L^T: swap them and transpose every block (incl. the pivot).
    Ffac, Bfac, Dpiv = (Uf, Lf, Dinv) if transpose else (Lf, Uf, Dinv)

    # ---- forward solve  (L y = b, or U^T y = b)   increasing elimination order ----------
    y = {}
    for v in order:
        acc = rhs[..., v, :, :]
        for u in incoming[v]:
            blk = Ffac[(v, u)].mT if transpose else Ffac[(v, u)]
            acc = acc - blk @ y[u]
        y[v] = acc

    # ---- diagonal solve  (D z = y, or D^T z = y)   per node -----------------------------
    z = {v: (Dpiv[v].mT if transpose else Dpiv[v]) @ y[v] for v in range(n)}

    # ---- back solve  (U x = z, or L^T x = z)   decreasing elimination order --------------
    x = {}
    for v in reversed(order):
        acc = z[v]
        for w in U[v]:
            blk = Bfac[(w, v)].mT if transpose else Bfac[(w, v)]
            acc = acc - blk @ x[w]
        x[v] = acc

    return torch.stack([x[v] for v in range(n)], dim=-3)           # [..., n, b, k]


def nonsym_junction_solve(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_lower: Tensor,
    edge_upper: Tensor,
    b: Tensor,
    order: Tensor | Sequence[int] | None = None,
    *,
    transpose: bool = False,
    check: bool = False,
    compute_dtype: torch.dtype | None = None,
):
    """Differentiable general non-symmetric sparse linear solve on the filled pattern.

    Solves ``A x = b`` (or ``A^T x = b`` when ``transpose=True``) for a general
    non-symmetric block matrix ``A`` with a structurally-symmetric pattern, via the same
    symbolic min-degree elimination + block ``LDU`` factorization as
    :func:`selected_inverse_nonsym_junction`. The factorization path is shared (one
    :func:`_factor_nonsym`), so this is the **solve sibling** of the non-symmetric selected
    inverse and the non-symmetric counterpart of :func:`~gabp_sparse_inv.junction_solve`. On
    symmetric input (``edge_upper = edge_lower.mT``) it matches ``junction_solve``
    block-for-block.

    The ``transpose`` flag is the implicit-differentiation adjoint solve: backpropagating
    through a fixed-point / DEQ layer ``z* = A^{-1} b`` requires one ``A^T`` solve (the
    vector-Jacobian product of the implicit function theorem), and it is the *same* factors
    transposed -- no refactorization.

    Parameters
    ----------
    diag, edge_index, edge_lower, edge_upper, order:
        As in :func:`selected_inverse_nonsym_junction` (``diag`` ``[..., n, b, b]``;
        ``edge_index`` ``[2, m]`` with ``i > j``; ``edge_lower``/``edge_upper`` the two
        independent off-diagonal orientations ``[..., m, b, b]``; ``order`` an optional
        elimination order, default greedy min-degree).
    b:
        Right-hand side ``[..., n, b]`` (one vector per node) or ``[..., n, b, k]``
        (``k`` right-hand sides). The result has the matching shape.
    transpose:
        If ``True``, solve ``A^T x = b`` instead of ``A x = b``.
    check:
        If ``True``, validate the matrix inputs / index convention first.
    compute_dtype:
        If given, all linear algebra runs in this dtype (inputs upcast, result cast back).

    Returns
    -------
    x:
        ``A^{-1} b`` (or ``A^{-T} b``) shaped like ``b``.

    Notes
    -----
    Functional / autograd-traceable: gradients flow to ``diag``, ``edge_lower``,
    ``edge_upper`` and ``b`` by reverse-mode (no custom backward). No pivoting -- the
    static-pattern regime (every block pivot ``D_v`` must stay non-singular).
    """
    edge_index = _normalize_edge_index(edge_index)
    if check:
        n, bs = diag.shape[-3], diag.shape[-1]
        if edge_lower.shape[-3:] != (edge_index.shape[1], bs, bs) or edge_upper.shape[-3:] != (edge_index.shape[1], bs, bs):
            raise ValueError("edge_lower/edge_upper must have shape [..., m, b, b] matching edge_index")
        ij = edge_index.tolist()
        for i, j in zip(ij[0], ij[1]):
            if not (n > i > j >= 0):
                raise ValueError(f"edge_index must be lower-triangular in range (n>i>j>=0); got ({i},{j})")
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
        edge_lower = edge_lower.to(compute_dtype)
        edge_upper = edge_upper.to(compute_dtype)
        rhs = rhs.to(compute_dtype)
    x = _solve_nonsym_junction_core(diag, edge_index, edge_lower, edge_upper, rhs, order, transpose)
    if compute_dtype is not None and compute_dtype != in_dtype:
        x = x.to(in_dtype)
    return x.squeeze(-1) if vector_rhs else x


def selinv_nonsym_junction(
    diag: Tensor,
    edge_index: Tensor | Sequence[Sequence[int]],
    edge_lower: Tensor,
    edge_upper: Tensor,
    order: Tensor | Sequence[int] | None = None,
):
    """Autograd-facing general non-symmetric selected inverse (gradients to the inputs).

    Identical result to :func:`selected_inverse_nonsym_junction`; the functional forward is
    differentiable, so gradients of any loss over the selected blocks flow to ``diag``,
    ``edge_lower`` and ``edge_upper`` by reverse-mode. For the inversion derivative,
    ``L_A^*(X) = -A^{-T} X A^{-T} = L_{A^T}(X)``; this is a derivative map, not the
    selected-inverse value at ``A^T`` (``docs/derivations.md`` §10). No custom backward.
    """
    edge_index = _normalize_edge_index(edge_index)
    return _selinv_nonsym_junction_core(diag, edge_index, edge_lower, edge_upper, order)
