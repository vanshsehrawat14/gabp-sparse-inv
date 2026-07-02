"""Matched-compute B2 baselines and a structure-exact FLOP counter.

**B2** is the *same* global operator ``A`` with its inverse applied by a **truncated
iterative solver** -- global but inexact (the operator is never restricted; only the
iteration count is truncated, so ``K -> infinity`` recovers the exact solve). This module
provides the SPD iterative solvers (Jacobi, CG) and a FLOP counter that prices the exact
selected-inverse / structured-solve path and the iterative path under **one convention**,
so the iteration budget ``N`` can be matched to the exact budget (:func:`n_match`) and swept
past it to find break-even.

Per-op FLOP convention (one place; applied to **both** sides so the ratio that sets
``N_match`` is robust to the constant):

* matmul ``(b x b)(b x c)`` = ``2 b^2 c``; Cholesky = ``b^3 / 3``; ``cholesky_solve`` /
  block diag-solve with ``c`` RHS = ``2 b^2 c``; ``inv_via_chol`` = ``2 b^3``.

The exact-path counts are derived **structurally** from the kernel's own elimination
(:func:`gabp_sparse_inv.junction._symbolic`), so the *scaling* with fill is exact; only the
per-op constant is modeled. **Conservatism:** where a modeling choice exists the exact path
is rounded up and the iterative path down, so every approximation leans *against* the
"exactness wins" hypothesis (larger ``N_match``, cheaper iterations).

Note: for an SPD operator, diagonally-preconditioned **Neumann == damped Jacobi** (same
iteration), so the two distinct SPD methods here are ``jacobi`` and ``cg``. Neumann is the
genuinely distinct method in the *non-symmetric* DEQ backward; its term cost is counted here
(:func:`neumann_term_flops`) and the solver itself lives in ``demos/deq_fixedpoint.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..junction import _symbolic

__all__ = [
    "spd_matvec",
    "jacobi_solve",
    "cg_solve",
    "neumann_series",
    "matvec_flops",
    "factor_flops",
    "exact_solve_flops",
    "exact_solve_adjoint_flops",
    "exact_selinv_flops",
    "iter_flops",
    "iter_solve_adjoint_flops",
    "n_match",
    "neumann_term_flops",
    "nonsym_backward_flops_chain",
    "k_match_neumann",
]


# --------------------------------------------------------------------------- #
# SPD iterative solvers (global, truncated) on the scalar-node grid rep:
#   diag [..., n], edge (i_idx, j_idx) [m], edge_val [..., m], rhs [..., n].
# spd_matvec matches demos/maze_grid._jacobi_solve block-for-block (b=1), so the
# maze's shipped causal control and this module cannot drift.
# --------------------------------------------------------------------------- #
def spd_matvec(diag: Tensor, i_idx: Tensor, j_idx: Tensor, edge_val: Tensor, x: Tensor) -> Tensor:
    """``y = A x`` for ``A = diag + sum_edges`` on the symmetric scalar pattern (reads all edges)."""
    ax = diag * x
    ax = ax.index_add(-1, i_idx, edge_val * x.index_select(-1, j_idx))
    ax = ax.index_add(-1, j_idx, edge_val * x.index_select(-1, i_idx))
    return ax


def jacobi_solve(diag: Tensor, i_idx: Tensor, j_idx: Tensor, edge_val: Tensor, b: Tensor,
                 K: int, *, omega: float = 1.0) -> Tensor:
    """``K`` damped-Jacobi sweeps for ``A x = b`` from ``x0 = 0`` (one matvec / sweep = one hop)."""
    x = torch.zeros_like(b)
    for _ in range(K):
        ax = spd_matvec(diag, i_idx, j_idx, edge_val, x)
        x = x + omega * (b - ax) / diag
    return x


def cg_solve(diag: Tensor, i_idx: Tensor, j_idx: Tensor, edge_val: Tensor, b: Tensor,
             K: int) -> Tensor:
    """``K`` iterations of diagonal-preconditioned CG for SPD ``A x = b`` (out-of-place / autograd-safe)."""
    x = torch.zeros_like(b)
    r = b - spd_matvec(diag, i_idx, j_idx, edge_val, x)
    minv = 1.0 / diag
    z = minv * r
    p = z
    rz = (r * z).sum(-1, keepdim=True)
    for _ in range(K):
        Ap = spd_matvec(diag, i_idx, j_idx, edge_val, p)
        alpha = rz / (p * Ap).sum(-1, keepdim=True).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = minv * r
        rz_new = (r * z).sum(-1, keepdim=True)
        beta = rz_new / rz.clamp_min(1e-30)
        p = z + beta * p
        rz = rz_new
    return x


def neumann_series(matvec_t, b: Tensor, K: int) -> Tensor:
    """``sum_{k=0}^{K-1} M^k b`` using caller-provided transpose/global matvec ``M``.

    This is the truncated global Neumann/Richardson adjoint used by the DEQ
    harness. The callable keeps the solver independent of the matrix storage
    convention while still making the B2 budget a first-class parameter.
    """
    if K < 1:
        raise ValueError("Neumann K must be >= 1")
    out = b
    term = b
    for _ in range(K - 1):
        term = matvec_t(term)
        out = out + term
    return out


# --------------------------------------------------------------------------- #
# FLOP model.
# --------------------------------------------------------------------------- #
def matvec_flops(n: int, m: int, *, b: int = 1, k: int = 1) -> float:
    """One sparse block matvec: ``n`` diagonal applies + ``2m`` edge applies, ``2 b^2 k`` each."""
    return (2 * n + 4 * m) * b * b * k


def _clique_sizes(n: int, edge_index, order=None):
    """``(per-node clique sizes d_v, number of off-diagonal fill slots nS)`` from the symbolic factor."""
    _order, _pos, U, S_pairs = _symbolic(n, edge_index, order)
    return [len(U[v]) for v in range(n)], len(S_pairs)


def factor_flops(n: int, edge_index, order=None, *, b: int = 1) -> float:
    """Collect-pass (LDL^T) cost: per node ``b^3/3`` (Cholesky) + ``2 d`` ell-solves + ``2 d(d+1)`` Schur."""
    d, _nS = _clique_sizes(n, edge_index, order)
    b3 = b ** 3
    return sum(b3 / 3 + 2 * dv * b3 + 2 * dv * (dv + 1) * b3 for dv in d)


def exact_solve_flops(n: int, edge_index, order=None, *, b: int = 1, k: int = 1) -> float:
    """Exact structured solve ``A^{-1} b``: factorization + triangular solve (the maze anchor)."""
    _d, nS = _clique_sizes(n, edge_index, order)
    return factor_flops(n, edge_index, order, b=b) + (4 * nS + 2 * n) * b * b * k


def exact_selinv_flops(n: int, edge_index, order=None, *, b: int = 1) -> float:
    """Exact selected inverse: factorization + Takahashi distribute (reported for transparency)."""
    d, _nS = _clique_sizes(n, edge_index, order)
    b3 = b ** 3
    distribute = sum(2 * dv * dv * b3 + 2 * b3 + 2 * dv * b3 for dv in d)
    return factor_flops(n, edge_index, order, b=b) + distribute


def exact_solve_adjoint_flops(n: int, edge_index, order=None, *, b: int = 1, k: int = 1) -> float:
    """Exact solve plus its training adjoint, rounded high for matched-compute anchoring.

    The forward is one structured solve. The adjoint needs transpose solves and
    factor-adjoint work on the same filled pattern; using the full selected-inverse
    distribute cost for that reverse pass is deliberately conservative.
    """
    return exact_solve_flops(n, edge_index, order, b=b, k=k) + exact_selinv_flops(n, edge_index, order, b=b)


def iter_flops(n: int, m: int, K: int, *, b: int = 1, k: int = 1, method: str = "jacobi") -> float:
    """Total FLOPs for ``K`` iterations of ``method`` (matvec + small per-iter overhead)."""
    per = matvec_flops(n, m, b=b, k=k)
    if method == "jacobi":
        per += 2 * n * b * b * k                 # diagonal solve + axpy
    elif method == "cg":
        per += 5 * n * b * k                      # 2 inner products + 3 axpys (lower order)
    else:
        raise ValueError(f"unknown SPD method {method!r}; known: jacobi, cg")
    return K * per


def iter_solve_adjoint_flops(n: int, m: int, K: int, *, b: int = 1, k: int = 1,
                             method: str = "jacobi") -> float:
    """Forward iterative solve plus reverse iterative adjoint at the same budget ``K``."""
    return 2 * iter_flops(n, m, K, b=b, k=k, method=method)


def n_match(n: int, edge_index, order=None, *, b: int = 1, k: int = 1,
            method: str = "jacobi", target: str = "solve_adjoint") -> int:
    """Smallest iteration budget whose FLOPs reach the exact budget (>= 1).

    ``target`` selects the exact anchor: ``"solve_adjoint"`` (default, the
    training path: structured solve plus adjoint), ``"solve"`` (forward only), or
    ``"selinv"`` (the full selected inverse). Conservative: exact FLOPs are
    rounded up and per-iteration FLOPs down before the ceiling division, which
    gives the truncated baseline extra matched iterations.
    """
    m = edge_index.shape[1]
    if target == "solve_adjoint":
        c_exact = exact_solve_adjoint_flops(n, edge_index, order, b=b, k=k)
        per = iter_solve_adjoint_flops(n, m, 1, b=b, k=k, method=method)
    elif target == "solve":
        c_exact = exact_solve_flops(n, edge_index, order, b=b, k=k)
        per = iter_flops(n, m, 1, b=b, k=k, method=method)
    elif target == "selinv":
        c_exact = exact_selinv_flops(n, edge_index, order, b=b)
        per = iter_flops(n, m, 1, b=b, k=k, method=method)
    else:
        raise ValueError(f"unknown exact target {target!r}; known: solve_adjoint, solve, selinv")
    return max(1, math.ceil(math.ceil(c_exact) / max(1, math.floor(per))))


# --------------------------------------------------------------------------- #
# Non-symmetric (DEQ) backward accounting. The exact backward is the nonsym
# selected-inverse adjoint solve (LU on fill + transpose solve); its FLOPs are
# rho-independent (set by fill, not spectral radius) -- the "exact stays flat" backbone.
# --------------------------------------------------------------------------- #
def neumann_term_flops(n: int, m: int, *, b: int = 1, k: int = 1) -> float:
    """One Neumann term = one transpose spmv (``n`` diag + ``2m`` edge matmuls) + axpy."""
    return (2 * n + 4 * m) * b * b * k + n * b * k


def nonsym_backward_flops_chain(n: int, *, b: int = 1, k: int = 1) -> float:
    """Exact nonsym backward on a chain (zero-fill LU + transpose solve + VJP-through-f).

    rho-independent by construction. Chain has ``m = n - 1`` edges and no fill, so the LU
    factor is ``O(n)``; constants rounded up (conservative against the exact-wins claim).
    """
    b3 = b ** 3
    factor = n * (b3 + 2 * b3)                          # per-pivot LU inverse + Schur (chain)
    tri = (4 * (n - 1) + 2 * n) * b * b * k             # transpose triangular solve
    vjp = (2 * n + 4 * (n - 1)) * b * b * k             # one VJP-through-f spmv
    return factor + tri + vjp


def k_match_neumann(n: int, m: int, *, b: int = 1, k: int = 1) -> int:
    """Neumann terms whose FLOPs reach the exact chain backward budget (>= 1, backward-only match)."""
    c_exact = nonsym_backward_flops_chain(n, b=b, k=k)
    return max(1, round(c_exact / neumann_term_flops(n, m, b=b, k=k)))
