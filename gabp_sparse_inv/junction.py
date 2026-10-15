"""General sparse SPD selected inverse via the elimination tree (junction tree).

This is the general-pattern version of the tree kernel (:mod:`gabp_sparse_inv.tree`):
the input SPD block matrix may have an **arbitrary** sparsity pattern, not just a
tree. The implementation picks an elimination order, symbolically completes the pattern to its
**chordal (filled)** closure ``S = pattern(L + L^T)`` (``docs/derivations.md`` §2.1),
and compute every block of ``A^{-1}`` on ``S`` -- the node diagonals plus the filled
off-diagonal blocks -- without ever forming the dense inverse.

Trees are the zero-fill special case (``S`` = the input pattern, ``U_v`` =
``{parent}``); there the recurrences below reduce *block-for-block* to ``tree.py``
eqs. (5)-(6), which the tests assert.

Algorithm (the multi-neighbour Takahashi / Erisman-Tinney recurrence; its reverse
schedule is derived in ``docs/derivations.md`` §8.4). Number the nodes by their
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
``S``-local reverse schedule of §8.4 -- no custom backward needed for correct
gradients. :func:`selinv_junction` is that autograd-facing entry point. A
hand-written analytic clique backward is shipped in
:mod:`gabp_sparse_inv.junction_autodiff`; it avoids an operation-by-operation
autograd tape and is first-order only.

**Scope.** A minimal, dependency-free reference: a greedy **min-degree** elimination
order and a per-node Python loop with dict-of-blocks sparse storage. If
``w_v = |U_v|``, numeric storage is ``Theta(sum_v(1+w_v) b^2)`` and arithmetic is
``Theta(sum_v(1+w_v^2) b^3)``. The checked level-set path additionally
pre-materializes ``Theta(sum_v(1+w_v^2))`` symbolic indices. There is no
supernodal/multifrontal engineering or external ordering library (AMD/METIS);
a better ordering changes the factor pattern and work, not the mathematical result.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ._linalg import cholesky_spd, inv_via_chol
from .layout import _as_edge_index as _normalize_edge_index
from .layout import _check_edge_index, _check_junction

__all__ = [
    "JunctionAnalysis",
    "SparseGaussianFactor",
    "analyze_junction",
    "selected_inverse_junction",
    "selinv_junction",
    "junction_solve",
    "junction_logdet",
    "elimination_order_min_degree",
    "elimination_order_nested_dissection",
]


def _rhs_is_vector(b: Tensor, diag_shape) -> bool:
    """Read ``b`` as ``[..., n, b]`` vectors (``True``) or ``[..., n, b, k]`` columns.

    A shape that fits both layouts is read as vectors only when it equals
    ``diag.shape[:-1]``, the ``torch.linalg.solve`` convention; ``b.unsqueeze(-1)``
    always selects the column layout.
    """
    n, bs = diag_shape[-3], diag_shape[-1]
    vector = b.ndim >= 2 and tuple(b.shape[-2:]) == (n, bs)
    matrix = b.ndim >= 3 and tuple(b.shape[-3:-1]) == (n, bs)
    if not (vector or matrix):
        raise ValueError(
            f"b must have shape [..., n, b] or [..., n, b, k] with n={n}, b={bs}; "
            f"got {tuple(b.shape)}"
        )
    if vector and matrix:
        return tuple(b.shape) == tuple(diag_shape[:-1])
    return vector


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


class JunctionAnalysis:
    """Values-independent symbolic plan for repeated SPD junction factorizations.

    Construct with :func:`analyze_junction`, then call :meth:`factor` whenever the
    numeric diagonal/off-diagonal blocks change.  The input pattern, elimination order,
    filled pattern, and clique schedule are computed exactly once and shared by every
    resulting :class:`SparseGaussianFactor`.

    The counters are deliberately explicit instrumentation for amortization studies;
    they are not timing surrogates.  ``symbolic_analyses`` is always one for an analysis
    object, while ``numeric_factorizations`` increases only after a factorization
    succeeds.
    """

    def __init__(self, n: int, edge_index, order=None):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        ei = _normalize_edge_index(edge_index).clone()
        _check_edge_index(ei, n)
        symbolic = _symbolic(n, ei, order)
        self.n = n
        self.edge_index = ei
        self.order, self.pos, self.U, self.S_pairs = symbolic
        self._symbolic_data = symbolic
        self.symbolic_analyses = 1
        self.numeric_factorizations = 0
        self.tensor_schedules = 0
        self._batched_symbolic_by_device = {}

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_selected_edges(self) -> int:
        return len(self.S_pairs)

    def factor(
        self,
        diag: Tensor,
        edge_val: Tensor,
        *,
        check: bool = False,
        compute_dtype: torch.dtype | None = None,
        batched: bool = False,
    ) -> "SparseGaussianFactor":
        """Numerically factor values on this plan and return a reusable factor object.

        ``batched=True`` uses the cached level-set tensor schedule for the numeric
        collect pass and every downstream factor query.  The reference dictionary
        factor remains the default and the correctness oracle.
        """
        if diag.ndim < 3 or diag.shape[-3] != self.n or diag.shape[-2] != diag.shape[-1]:
            raise ValueError(
                f"diag must have shape [..., {self.n}, b, b]; got {tuple(diag.shape)}"
            )
        b = diag.shape[-1]
        if (edge_val.ndim < 3 or edge_val.shape[-3] != self.num_edges
                or edge_val.shape[-2:] != (b, b)):
            raise ValueError(
                f"edge_val must have shape [..., {self.num_edges}, {b}, {b}]; "
                f"got {tuple(edge_val.shape)}"
            )
        if diag.device != edge_val.device:
            raise ValueError(
                f"diag and edge_val must share a device; got {diag.device} and {edge_val.device}"
            )
        if check:
            from .layout import BlockSparseSym

            BlockSparseSym(
                diag=diag, edge_index=self.edge_index, edge_val=edge_val
            ).validate()
        result_dtype = diag.dtype
        if compute_dtype is not None and compute_dtype != result_dtype:
            diag = diag.to(compute_dtype)
            edge_val = edge_val.to(compute_dtype)
        batched_symbolic = None
        if batched:
            key = (diag.device.type, diag.device.index)
            batched_symbolic = self._batched_symbolic_by_device.get(key)
            if batched_symbolic is None:
                batched_symbolic = _batched_symbolic(
                    self.n, self.edge_index, self.order, diag.device
                )
                self._batched_symbolic_by_device[key] = batched_symbolic
                self.tensor_schedules += 1
            factors = _factor_junction_batched_numeric(
                diag, edge_val, batched_symbolic, check_spd=True
            )
        else:
            factors = _factor_junction(
                diag,
                self.edge_index,
                edge_val,
                None,
                symbolic=self._symbolic_data,
            )
        self.numeric_factorizations += 1
        return SparseGaussianFactor(
            analysis=self,
            diag=diag,
            factors=factors,
            batched_symbolic=batched_symbolic,
            result_dtype=result_dtype,
            factorization_id=self.numeric_factorizations,
        )


def analyze_junction(
    n: int,
    edge_index: Tensor | Sequence[Sequence[int]],
    order: Tensor | Sequence[int] | None = None,
) -> JunctionAnalysis:
    """Analyze an SPD sparse pattern once for repeated numeric factorizations.

    This is the persistent counterpart of the symbolic work performed inside the
    functional junction operators.  The returned plan owns a CPU copy of ``edge_index``;
    its numeric factors may live on any device supported by PyTorch.
    """
    return JunctionAnalysis(n, edge_index, order)


def _factor_junction(diag, edge_index, edge_val, order, *, symbolic=None):
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

    if symbolic is None:
        order, pos, U, S_pairs = _symbolic(n, edge_index, order)
    else:
        order, pos, U, S_pairs = symbolic
        if len(order) != n:
            raise ValueError(
                f"symbolic plan has n={len(order)} but numeric diagonal has n={n}"
            )

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


def _selected_inverse_from_factors(diag, factors):
    """Distribute a stored junction factor into selected inverse blocks."""
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    order, pos, U, S_pairs, chol, ell = factors

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
    return G_diag, S_index, G_lower


def _solve_from_factors(diag, rhs, factors):
    """Solve ``A x = rhs`` from a stored junction factor (rhs ``[..., n, b, k]``)."""
    n = diag.shape[-3]
    order, _pos, U, _S_pairs, chol, ell = factors

    incoming: list[list[int]] = [[] for _ in range(n)]
    for (w, v) in ell:
        incoming[w].append(v)

    y = {}
    for v in order:
        acc = rhs[..., v, :, :]
        for u in incoming[v]:
            acc = acc - ell[(v, u)] @ y[u]
        y[v] = acc

    z = {v: torch.cholesky_solve(y[v], chol[v]) for v in range(n)}

    x = {}
    for v in reversed(order):
        acc = z[v]
        for w in U[v]:
            acc = acc - ell[(w, v)].mT @ x[w]
        x[v] = acc
    return torch.stack([x[v] for v in range(n)], dim=-3)


def _logdet_from_factors(diag, factors):
    """Return ``log det A`` from stored pivot Cholesky factors."""
    n = diag.shape[-3]
    chol = factors[4]
    total = None
    for v in range(n):
        ld = 2.0 * torch.log(torch.diagonal(chol[v], dim1=-2, dim2=-1)).sum(-1)
        total = ld if total is None else total + ld
    return total


def _sample_transform_from_factors(diag, z, factors):
    """Apply ``M^{-T}`` from stored factors to standard-normal columns ``z``."""
    n = diag.shape[-3]
    order, _pos, U, _S_pairs, chol, ell = factors
    u = {
        v: torch.linalg.solve_triangular(chol[v].mT, z[..., v, :, :], upper=True)
        for v in range(n)
    }
    x = {}
    for v in reversed(order):
        acc = u[v]
        for w in U[v]:
            acc = acc - ell[(w, v)].mT @ x[w]
        x[v] = acc
    return torch.stack([x[v] for v in range(n)], dim=-3)


class SparseGaussianFactor:
    """One differentiable numeric SPD factor reused across Gaussian operations.

    Instances are created by :meth:`JunctionAnalysis.factor`.  They retain the PyTorch
    graph of exactly one numeric ``LDL^T`` factorization, so a combined loss over
    :meth:`logdet`, :meth:`solve`, and :meth:`selected_inverse` differentiates through
    the same pivots and factor blocks rather than refactorizing per operation.

    The default dictionary backend remains the correctness reference.  A checked
    level-set backend stores the same factor in tensor buffers so all downstream
    queries reuse the accelerator-friendly schedule without refactorizing.
    """

    def __init__(
        self,
        *,
        analysis,
        diag,
        factors,
        batched_symbolic,
        result_dtype,
        factorization_id,
    ):
        self.analysis = analysis
        self._diag = diag
        self._factors = factors
        self._batched_symbolic = batched_symbolic
        self.backend = "level_set" if batched_symbolic is not None else "reference"
        self.result_dtype = result_dtype
        self.compute_dtype = diag.dtype
        self.device = diag.device
        self.factorization_id = factorization_id
        self._query_counts = {
            "logdet": 0,
            "solve": 0,
            "selected_inverse": 0,
            "marginal_diag": 0,
            "sample": 0,
        }

    @property
    def num_nodes(self) -> int:
        return self.analysis.n

    @property
    def block_size(self) -> int:
        return int(self._diag.shape[-1])

    def operation_counts(self) -> dict[str, int]:
        """Return auditable analysis/factor/query counts for this object."""
        return {
            "symbolic_analyses": 1,
            "numeric_factorizations": 1,
            "analysis_factorizations_created": self.analysis.numeric_factorizations,
            **self._query_counts,
        }

    def _cast_result(self, value):
        return value.to(self.result_dtype) if value.dtype != self.result_dtype else value

    def logdet(self):
        """Return ``log det A`` without another symbolic or numeric factorization."""
        self._query_counts["logdet"] += 1
        if self._batched_symbolic is None:
            value = _logdet_from_factors(self._diag, self._factors)
        else:
            value = _logdet_from_batched_factors(self._factors)
        return self._cast_result(value)

    def solve(self, b: Tensor):
        """Return ``A^{-1} b`` without another symbolic or numeric factorization.

        ``b`` is ``[..., n, b]`` or ``[..., n, b, k]``, read as in :func:`junction_solve`.
        """
        vector_rhs = _rhs_is_vector(b, self._diag.shape)
        if b.device != self.device:
            raise ValueError(f"b must be on {self.device}; got {b.device}")
        rhs = b.unsqueeze(-1) if vector_rhs else b
        if rhs.dtype != self.compute_dtype:
            rhs = rhs.to(self.compute_dtype)
        self._query_counts["solve"] += 1
        if self._batched_symbolic is None:
            x = _solve_from_factors(self._diag, rhs, self._factors)
        else:
            x = _solve_from_batched_factors(
                self._diag, rhs, self._batched_symbolic, self._factors
            )
        x = self._cast_result(x)
        return x.squeeze(-1) if vector_rhs else x

    def selected_inverse(self):
        """Return inverse blocks on the filled pattern without refactorizing."""
        self._query_counts["selected_inverse"] += 1
        if self._batched_symbolic is None:
            G_diag, S_index, G_lower = _selected_inverse_from_factors(
                self._diag, self._factors
            )
        else:
            G_diag, G_lower = _selected_inverse_from_batched_factors(
                self._diag, self._batched_symbolic, self._factors
            )
            S_index = self._batched_symbolic[3]
        return self._cast_result(G_diag), S_index, self._cast_result(G_lower)

    def marginal_diag(self):
        """Return component-wise marginal variances ``diag(A^{-1})``."""
        self._query_counts["marginal_diag"] += 1
        if self._batched_symbolic is None:
            G_diag = _selected_inverse_from_factors(self._diag, self._factors)[0]
        else:
            G_diag = _selected_inverse_from_batched_factors(
                self._diag, self._batched_symbolic, self._factors
            )[0]
        variances = torch.diagonal(G_diag, dim1=-2, dim2=-1)
        return self._cast_result(variances)

    def sample(self, num_samples: int, *, generator: torch.Generator | None = None):
        """Draw reparameterized samples ``x ~ N(0, A^{-1})`` from stored factors."""
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
            raise ValueError(f"num_samples must be a positive integer; got {num_samples!r}")
        n, b = self.num_nodes, self.block_size
        batch = self._diag.shape[:-3]
        z = torch.randn(
            (*batch, n, b, num_samples),
            generator=generator,
            dtype=self.compute_dtype,
            device=self.device,
        )
        self._query_counts["sample"] += 1
        if self._batched_symbolic is None:
            x = _sample_transform_from_factors(self._diag, z, self._factors)
        else:
            x = _sample_transform_from_batched_factors(
                self._diag, z, self._batched_symbolic, self._factors
            )
        return self._cast_result(x).movedim(-1, 0)


def _selinv_junction_core(diag, edge_index, edge_val, order, return_factors=False):
    """Functional (autograd-traceable) forward. Returns ``(G_diag, S_index, G_lower)``.

    Storage is a dict of ``b x b`` blocks keyed by elimination-position orientation
    ``(hi, lo)`` (so ``hi`` is eliminated after ``lo``); ``G_diag``/``G_lower`` are
    re-emitted in *node-index* orientation (``i > j``) to match the input layout.

    With ``return_factors=True`` also returns the collect-pass factors
    ``(order, pos, U, S_pairs, chol, ell)`` (the analytic backward reuses them; see
    :mod:`gabp_sparse_inv.junction_autodiff`).
    """
    factors = _factor_junction(diag, edge_index, edge_val, order)
    result = _selected_inverse_from_factors(diag, factors)
    if return_factors:
        return (*result, factors)
    return result


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
        "ell_dest", "ell_lo", "ell_hi",           # ell_{wv}: slot, pivot v, higher w
        "sf_dest", "sf_src", "sf_ell",            # solve collect: y_w -= ell_{wv} y_v
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
        ell_dest, ell_lo, ell_hi = [], [], []
        sf_dest, sf_src, sf_ell = [], [], []
        sc_ellw, sc_ellwp, sc_v, sc_dest = [], [], [], []
        cr_dest, cr_ell, cr_gidx, cr_gT = [], [], [], []
        dg_v, dg_ell = [], []
        for v in nodes:
            for u in lower[v]:                         # forward solve dependencies into v
                sf_dest.append(v); sf_src.append(u); sf_ell.append(slot[(v, u)])
            Uv = U[v]
            for w in Uv:                              # ell_{wv} = A_{wv} D_v^{-1}
                ell_dest.append(slot[(w, v)]); ell_lo.append(v); ell_hi.append(w)
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
            ell_dest=lt(ell_dest), ell_lo=lt(ell_lo), ell_hi=lt(ell_hi),
            sf_dest=lt(sf_dest), sf_src=lt(sf_src), sf_ell=lt(sf_ell),
            sc_ellw=lt(sc_ellw), sc_ellwp=lt(sc_ellwp), sc_v=lt(sc_v), sc_dest=lt(sc_dest),
            cr_dest=lt(cr_dest), cr_ell=lt(cr_ell), cr_gidx=lt(cr_gidx), cr_gT=bt(cr_gT),
            dg_v=lt(dg_v), dg_ell=lt(dg_ell),
        ))

    return nS, lt(init_dest), bt(init_mask), S_index, bt(emit_mask), levels


def _factor_junction_batched_numeric(diag, edge_val, symbolic, *, check_spd=True):
    """Level-set collect pass returning reusable tensor factor buffers."""
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    nS, init_dest, init_mask, _S_index, _emit_mask, levels = symbolic

    def sym(x):
        return 0.5 * (x + x.mT)

    def chol(block, *, name):
        if check_spd:
            return cholesky_spd(block, name=name)
        return torch.linalg.cholesky_ex(block, check_errors=False)[0]

    # ---- unified working block array: [ D (n node diagonals) | W (nS off-blocks) ] ----
    Mbuf = torch.cat([diag, diag.new_zeros((*batch, nS, b, b))], dim=-3)
    if init_dest.numel() > 0:
        contrib = torch.where(init_mask[:, None, None], edge_val.mT, edge_val)
        Mbuf = Mbuf.index_add(-3, init_dest, contrib)

    chol_buf = diag.new_zeros((*batch, n, b, b))
    Dsym_buf = diag.new_zeros((*batch, n, b, b))
    ell_buf = diag.new_zeros((*batch, nS, b, b))

    # ---- collect: sparse block Cholesky + clique Schur fill, increasing level ----------
    for h, lv in enumerate(levels):
        Dl = sym(Mbuf.index_select(-3, lv.nodes))
        chol_l = chol(Dl, name=f"D(level {h})")
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

    return chol_buf, ell_buf


def _selected_inverse_from_batched_factors(diag, symbolic, factors):
    """Distribute reusable level-set factor buffers into selected inverse blocks."""
    n, b = diag.shape[-3], diag.shape[-1]
    batch = diag.shape[:-3]
    nS, _init_dest, _init_mask, _S_index, emit_mask, levels = symbolic
    chol_buf, ell_buf = factors

    def sym(x):
        return 0.5 * (x + x.mT)

    # ---- distribute: Takahashi back-substitution restricted to S, decreasing level -----
    Gbuf = diag.new_zeros((*batch, n + nS, b, b))
    for lv in reversed(levels):
        if lv.cr_dest.numel() > 0:
            Gsrc = Gbuf.index_select(-3, lv.cr_gidx)
            Gsrc = torch.where(lv.cr_gT[:, None, None], Gsrc.mT, Gsrc)
            Gbuf = Gbuf.index_add(
                -3,
                lv.cr_dest,
                -(Gsrc @ ell_buf.index_select(-3, lv.cr_ell)),
            )
        Dinv_l = inv_via_chol(chol_buf.index_select(-3, lv.nodes))
        if lv.dg_v.numel() > 0:
            elluv = ell_buf.index_select(-3, lv.dg_ell)
            Guv = Gbuf.index_select(-3, lv.dg_ell + n)
            acc = diag.new_zeros((*batch, n, b, b)).index_add(
                -3, lv.dg_v, elluv.mT @ Guv
            )
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
    return G_diag, G_lower


def _solve_from_batched_factors(diag, rhs, symbolic, factors):
    """Solve from reusable level-set factors; ``rhs`` is ``[..., n, b, k]``."""
    n = diag.shape[-3]
    levels = symbolic[5]
    chol_buf, ell_buf = factors

    ybuf = torch.zeros_like(rhs)
    for lv in levels:
        if lv.sf_dest.numel() > 0:
            contributions = ell_buf.index_select(-3, lv.sf_ell) @ ybuf.index_select(
                -3, lv.sf_src
            )
            accumulated = torch.zeros_like(rhs).index_add(
                -3, lv.sf_dest, contributions
            )
            y_l = rhs.index_select(-3, lv.nodes) - accumulated.index_select(
                -3, lv.nodes
            )
        else:
            y_l = rhs.index_select(-3, lv.nodes)
        ybuf = ybuf.index_add(-3, lv.nodes, y_l)

    zbuf = torch.cholesky_solve(ybuf, chol_buf)
    xbuf = torch.zeros_like(rhs)
    for lv in reversed(levels):
        if lv.ell_dest.numel() > 0:
            contributions = ell_buf.index_select(-3, lv.ell_dest).mT @ xbuf.index_select(
                -3, lv.ell_hi
            )
            accumulated = torch.zeros_like(rhs).index_add(
                -3, lv.ell_lo, contributions
            )
            x_l = zbuf.index_select(-3, lv.nodes) - accumulated.index_select(
                -3, lv.nodes
            )
        else:
            x_l = zbuf.index_select(-3, lv.nodes)
        xbuf = xbuf.index_add(-3, lv.nodes, x_l)
    return xbuf


def _logdet_from_batched_factors(factors):
    """Return ``log det A`` from level-set pivot Cholesky buffers."""
    chol_buf = factors[0]
    return 2.0 * torch.log(
        torch.diagonal(chol_buf, dim1=-2, dim2=-1)
    ).sum(dim=(-1, -2))


def _sample_transform_from_batched_factors(diag, z, symbolic, factors):
    """Apply the stored level-set ``M^{-T}`` sampling transform."""
    levels = symbolic[5]
    chol_buf, ell_buf = factors
    ubuf = torch.linalg.solve_triangular(chol_buf.mT, z, upper=True)
    xbuf = torch.zeros_like(z)
    for lv in reversed(levels):
        if lv.ell_dest.numel() > 0:
            contributions = ell_buf.index_select(-3, lv.ell_dest).mT @ xbuf.index_select(
                -3, lv.ell_hi
            )
            accumulated = torch.zeros_like(z).index_add(
                -3, lv.ell_lo, contributions
            )
            x_l = ubuf.index_select(-3, lv.nodes) - accumulated.index_select(
                -3, lv.nodes
            )
        else:
            x_l = ubuf.index_select(-3, lv.nodes)
        xbuf = xbuf.index_add(-3, lv.nodes, x_l)
    return xbuf


def _selinv_junction_batched_numeric(
    diag,
    edge_val,
    symbolic,
    *,
    check_spd=True,
):
    """Level-set factor-plus-selected-inverse from a symbolic tensor schedule.

    Separating the values-dependent body from :func:`_batched_symbolic` makes the
    shortest accelerator route testable without tracing Python graph construction or
    ``Tensor.tolist()``.  The public path keeps ``check_spd=True`` and therefore retains
    informative pivot failure checks.  A compiler feasibility probe may set it to
    ``False`` only after an eager checked execution on the same inputs.

    Returns ``(G_diag, G_lower)``.  The values-independent ``S_index`` remains in the
    symbolic record and is attached by :func:`_selinv_junction_batched`.
    """
    factors = _factor_junction_batched_numeric(
        diag, edge_val, symbolic, check_spd=check_spd
    )
    return _selected_inverse_from_batched_factors(diag, symbolic, factors)


def _selinv_junction_batched(diag, edge_index, edge_val, order):
    """Level-set batched forward (functional / autograd-traceable).

    Mathematically identical to the per-node reference :func:`_selinv_junction_core`
    (validated block-for-block in ``tests/test_junction_batched.py``), but the Python loop
    runs over *elimination levels* -- ``O(elimination-tree height)``: ``~log n`` balanced,
    ``~sqrt(n)`` on a 2-D grid -- instead of over nodes, with each level a handful of
    batched block ops. The unified working array ``Mbuf`` (and result ``Gbuf``) stack the
    ``n`` node diagonals over the ``nS`` off-diagonal slots, so the variable-size clique
    updates become flat ``index_select`` / ``index_add`` scatters. Reverse-mode autograd
    through it is the §8.4 reverse schedule, so ``selinv_junction(..., batched=True)``
    gets exact gradients with no custom backward -- exactly as the per-node path does.
    """
    dev = diag.device
    symbolic = _batched_symbolic(diag.shape[-3], edge_index, order, dev)
    G_diag, G_lower = _selinv_junction_batched_numeric(
        diag, edge_val, symbolic, check_spd=True
    )
    S_index = symbolic[3]
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
        ``[2, m]`` integer tensor of off-diagonal block positions ``(i, j)`` with
        ``i > j`` (node index), each block listed once; the upper triangle is implied by
        symmetry. A list holding both orientations or a repeated block raises (convert a
        both-direction list with ``keep = edge_index[0] > edge_index[1]``). Shared across
        leading batch dims (not batched).
    edge_val:
        Off-diagonal blocks ``edge_val[..., k, :, :] = A_{i_k, j_k}``: ``[..., m, b, b]``.
    order:
        Optional explicit elimination order (a permutation of ``range(n)``). Defaults
        to a greedy minimum-degree order computed from the pattern.
    check:
        If ``True``, also validate values (dtype / finiteness / block symmetry). Structural
        checks (shapes, integer in-range ``i > j`` indices without duplicates) always run.
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
    edge_index = _check_junction(diag, edge_index, edge_val=edge_val)
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()
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
    factors = _factor_junction(diag, edge_index, edge_val, order)
    return _solve_from_factors(diag, rhs, factors)


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
        (``k`` right-hand sides). The result has the matching shape. A shape that fits
        both layouts (only possible when the trailing dimensions are ``(n, n, n)`` with
        ``n == b``) is read as vectors only when it equals ``diag.shape[:-1]``, as in
        :func:`torch.linalg.solve`; pass ``b.unsqueeze(-1)`` to force the column layout.
    check:
        If ``True``, also validate values (dtype / finiteness / block symmetry). Structural
        checks (shapes, integer in-range ``i > j`` indices without duplicates) always run.
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
    edge_index = _check_junction(diag, edge_index, edge_val=edge_val)
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()
    vector_rhs = _rhs_is_vector(b, diag.shape)
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
    edge_index = _check_junction(diag, edge_index, edge_val=edge_val)
    if check:
        from .layout import BlockSparseSym

        BlockSparseSym(diag=diag, edge_index=edge_index, edge_val=edge_val).validate()
    in_dtype = diag.dtype
    if compute_dtype is not None and compute_dtype != in_dtype:
        diag = diag.to(compute_dtype)
        edge_val = edge_val.to(compute_dtype)
    factors = _factor_junction(diag, edge_index, edge_val, order)
    total = _logdet_from_factors(diag, factors)
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
    reverse-mode through the ``S``-local schedule of ``docs/derivations.md`` §8.4. The
    hand-written, tape-free analytic backward is
    :func:`~gabp_sparse_inv.selinv_junction_analytic`.

    ``batched=True`` selects the level-set batched forward (see
    :func:`selected_inverse_junction`); the result and its gradients are identical (the
    batched path is likewise functional, so reverse-mode gives the same backward), and
    only the scheduling -- per elimination level vs per node -- differs.
    """
    edge_index = _check_junction(diag, edge_index, edge_val=edge_val)
    forward = _selinv_junction_batched if batched else _selinv_junction_core
    return forward(diag, edge_index, edge_val, order)
