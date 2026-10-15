"""Shared batched linear-algebra helpers.

These back every selected-inverse kernel (chain, star, ...): a batched Cholesky
with an informative SPD-failure message, an explicit SPD-block inverse from a
Cholesky factor, and the element-growth guard of the static no-pivot
non-symmetric eliminations. Keeping them in one place means there is a single,
tested path regardless of the matrix structure being inverted.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "cholesky_spd",
    "inv_via_chol",
    "block_amax",
    "block_norm",
    "local_scales",
    "check_pivot_growth",
    "refuse_double_backward",
]


def cholesky_spd(block: Tensor, *, name: str) -> Tensor:
    """Batched Cholesky via ``cholesky_ex`` with an informative SPD failure message.

    ``cholesky_ex`` reports failure through an ``info`` code per batch element instead
    of raising inside the kernel. Inspecting ``info`` here still costs one host
    synchronization on accelerators.

    ``name`` identifies the failing pivot in the error message (e.g. ``"D_3"`` or
    ``"S (center Schur complement)"``).
    """
    chol, info = torch.linalg.cholesky_ex(block)
    if torch.any(info != 0):
        where = (
            f"batch indices {torch.nonzero(info != 0, as_tuple=False).tolist()}"
            if info.ndim > 0
            else "an unbatched input"
        )
        raise torch.linalg.LinAlgError(
            f"pivot block {name} lost positive-definiteness during factorization "
            f"(cholesky_ex info != 0 for {where}); matrix is not SPD or is too "
            f"ill-conditioned"
        )
    return chol


def inv_via_chol(chol: Tensor) -> Tensor:
    """Explicit inverse of an SPD block from its Cholesky factor."""
    eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device)
    eye = eye.expand_as(chol)
    return torch.cholesky_solve(eye, chol)


def block_amax(x: Tensor, dims: int = 2) -> Tensor:
    """Largest absolute entry over the trailing ``dims`` dimensions, detached."""
    return x.detach().abs().amax(dim=tuple(range(-dims, 0)))


def local_scales(diag: Tensor, row: Tensor, col: Tensor, lower: Tensor, upper: Tensor):
    """Largest input entry of every block row and block column, per batch element.

    ``diag`` is ``[..., n, b, b]``; off-diagonal block ``k`` has ``lower[..., k]`` at block
    position ``(row[k], col[k])`` and ``upper[..., k]`` at ``(col[k], row[k])``. Returns
    ``(row_scale, col_scale)``, each ``[..., n]`` and detached.
    """
    d = block_amax(diag)
    n = d.shape[-1]
    if row.numel() == 0:
        return d, d
    lo, up = block_amax(lower), block_amax(upper)
    m = lo.shape[-1]
    batch = torch.broadcast_shapes(d.shape[:-1], lo.shape[:-1], up.shape[:-1])
    d, lo, up = d.expand(*batch, n), lo.expand(*batch, m), up.expand(*batch, m)
    r = row.to(d.device).expand(*batch, m)
    c = col.to(d.device).expand(*batch, m)
    rows = d.scatter_reduce(-1, r, lo, "amax").scatter_reduce(-1, c, up, "amax")
    cols = d.scatter_reduce(-1, c, lo, "amax").scatter_reduce(-1, r, up, "amax")
    return rows, cols


def block_norm(x: Tensor) -> Tensor:
    """Infinity norm (largest absolute row sum) of each trailing ``b x b`` block, detached."""
    return x.detach().abs().sum(-1).amax(-1)


SOLVE_GROWTH_POWER = 0.5
SELINV_GROWTH_POWER = 0.25


def check_pivot_growth(growth: Tensor, dtype: torch.dtype, power: float) -> None:
    """Refuse a static no-pivot elimination whose local element growth is unsafe.

    ``growth`` holds, per batch element, the largest ratio between a quantity formed during
    the elimination and the larger of the largest input entries in its block row and its
    block column (:func:`local_scales`). The quantities are the reduced pivots and reduced
    off-diagonal blocks, and for every Schur update ``L_wv D_v U_vw'`` the product of norms
    ``||L_wv|| ||D_v|| ||U_vw'||``, which bounds the entries of ``|L| |D| |U|`` in the usual
    backward error bound ``|dA| <~ c u |L| |D| |U|`` of elimination without pivoting
    (block multipliers can be large while their product cancels, so the Schur entries
    alone understate it). Measuring each position against its own row and column means a
    large block elsewhere, a disconnected component or a global rescaling cannot hide
    growth.

    The limit is ``eps ** -power`` of the working dtype. A solve's backward error grows
    like ``u * growth``, so solves use ``power = 1/2``. The Takahashi recurrence of the
    selected inverse amplifies errors by the multipliers a second time, so its error grows
    like ``u * growth**2`` and selected inverses use ``power = 1/4``. Either way an accepted
    elimination can lose up to about half the working digits, more when ``A`` is ill
    conditioned. Non-finite growth (overflow, or ``inf``/``nan`` input) is refused as
    well. This is a heuristic test, not a stability proof.
    """
    limit = torch.finfo(dtype).eps ** -power
    if not bool(torch.all(growth <= limit)):
        raise torch.linalg.LinAlgError(
            f"local element growth {float(growth.max()):.3g} exceeds {limit:.3g} = eps**-{power:g} "
            f"for {dtype} during static no-pivot elimination (the factors outgrew the input "
            "entries of their block rows and columns); the result would be numerically "
            "unreliable. Use an elimination order that keeps the leading pivots well "
            "conditioned, compute in float64 (compute_dtype=torch.float64), or use a "
            "pivoted LU."
        )


def refuse_double_backward(name: str, alternative: str) -> None:
    """Raise if autograd asks a first-order analytic backward for a differentiable gradient.

    Autograd runs a custom backward with grad mode on exactly when the caller passed
    ``create_graph=True`` (double backward, Hessian-vector products,
    ``torch.autograd.functional.hessian``). These backwards read saved factors that are not
    connected to the inputs, so a second derivative through them would be wrong or silently
    zero; ``once_differentiable`` alone still lets ``torch.autograd.functional.hessian``
    return zeros. Refusing here makes every such request fail loudly.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            f"{name} has a first-order analytic backward and cannot build a differentiable "
            "gradient (create_graph=True: double backward, Hessian-vector products, "
            f"torch.autograd.functional.hessian). Use {alternative} for higher derivatives."
        )
