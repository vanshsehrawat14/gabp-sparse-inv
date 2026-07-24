"""DEQ / fixed-point layer: the non-symmetric selected inverse as the exact backward.

This demonstration covers deep-equilibrium / fixed-point models where a fast differentiable
inverse is useful. A deep-equilibrium model defines its output as a fixed point
``z* = f(z*, x)``; the
backward pass needs, by the implicit function theorem,

    (I - J)^T u = dL/dz* ,        J = df/dz |_{z*}      (the adjoint/VJP solve)

a **non-symmetric** linear solve with the equilibrium Jacobian. When ``f``'s coupling is
sparse on a low-treewidth graph, that solve is exactly
:func:`~gabp_sparse_inv.nonsym_junction_solve` on ``A = I - J``: the block ``LDU`` of
``docs/derivations.md`` §10, one front-dependent factorization, and the *transpose* of the
same factors gives the adjoint with no refactorization. For fixed block size and bounded
front width the work is linear in the number of graph nodes; in general it is
``Theta(sum_v (1 + w_v**2) b**3)``.

Two things this module demonstrates, both honestly scoped:

1. **Exactness.** The implicit-diff backward through the structured solve matches a dense
   implicit-diff oracle (and autograd through an unrolled solver) to machine precision -- for
   an affine fixed point it is literally autograd through :func:`nonsym_junction_solve`; for a
   nonlinear DEQ it is the custom IFT backward in :func:`deq_fixed_point`.

2. **A finite stiff-regime mechanism check.** Standard DEQ backprop solves ``(I-J)^T u =
   g`` *iteratively* (Neumann / Richardson: ``u_{k+1} = J^T u_k + g``), which converges like
   ``rho(J)^k`` in the normal/contraction construction used here. The structured solve is
   one-shot. :func:`backward_accuracy_sweep` checks four finite ``rho`` values and shows the
   direct backward agreeing closely with the dense oracle while the ``K``-step Neumann
   truncation worsens as stiffness increases. This finite grid neither proves
   condition-independent floating-point accuracy nor compares against Krylov,
   preconditioned, or accelerated implicit-gradient methods.

**Honest scope.** The advantage is for fixed-point layers whose Jacobian is **graph-
structured / low-treewidth**; for a dense Jacobian the ``LDU`` is ``O(n^3)`` and an iterative
solver is the right tool. ``rho(J) < 1`` is assumed (a contraction -- the fixed point exists
and is unique), the non-symmetric analogue of the maze's SPD-by-construction. Accelerated
backward solvers (Anderson/Broyden) cut the iteration count but not the ``rho -> 1`` scaling;
the structured solve removes it. This is a capability/mechanism result on controlled problems,
not a SOTA claim -- the same honesty bar as ``docs/MAZE.md``.

Run ``python -m gabp_sparse_inv.demos.deq_fixedpoint`` for the rho-sweep table.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..nonsym_junction import nonsym_junction_solve

__all__ = [
    "assemble_blocks",
    "spmv",
    "random_coupling",
    "im_minus_jac_blocks",
    "affine_fixed_point",
    "deq_fixed_point",
    "neumann_adjoint",
    "backward_accuracy_sweep",
    "fit_affine_fixed_point",
    "run_demo",
]


# --------------------------------------------------------------------------- #
# Block <-> dense helpers and the sparse block mat-vec (operator, no inverse).
# --------------------------------------------------------------------------- #
def assemble_blocks(diag: Tensor, edge_index: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Dense ``[..., n*b, n*b]`` from node-diagonal + two-orientation edge blocks.

    Convention (matching :mod:`gabp_sparse_inv.nonsym_junction`): edge ``k = (i, j)`` with
    ``i > j`` carries ``lower[k] = A_{i,j}`` (row ``i``, col ``j``) and ``upper[k] = A_{j,i}``.
    Differentiable -- used both for spectral scaling and as the dense gradient oracle.
    """
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    A = diag.new_zeros((*diag.shape[:-3], N, N))
    for v in range(n):
        A[..., v * b:(v + 1) * b, v * b:(v + 1) * b] = diag[..., v, :, :]
    ij = edge_index.tolist()
    for k in range(edge_index.shape[1]):
        i, j = ij[0][k], ij[1][k]
        A[..., i * b:(i + 1) * b, j * b:(j + 1) * b] = lower[..., k, :, :]
        A[..., j * b:(j + 1) * b, i * b:(i + 1) * b] = upper[..., k, :, :]
    return A


def spmv(diag: Tensor, edge_index: Tensor, lower: Tensor, upper: Tensor, v: Tensor,
         *, transpose: bool = False) -> Tensor:
    """Sparse block mat-vec ``y = A v`` (or ``A^T v``) for ``v`` shaped ``[..., n, b]``.

    The forward operator (no inverse), used for the fixed-point iteration and the Neumann
    adjoint baseline. The block transpose is ``(A^T)_{ab} = (A_{ba})^T``.
    """
    D = diag.mT if transpose else diag
    y = (D @ v.unsqueeze(-1)).squeeze(-1)
    if edge_index.shape[1]:
        i, j = edge_index[0], edge_index[1]
        blk_i = upper.mT if transpose else lower          # block sent to row i (from col j)
        blk_j = lower.mT if transpose else upper          # block sent to row j (from col i)
        vj = v.index_select(-2, j)
        vi = v.index_select(-2, i)
        y = y.index_add(-2, i, (blk_i @ vj.unsqueeze(-1)).squeeze(-1))
        y = y.index_add(-2, j, (blk_j @ vi.unsqueeze(-1)).squeeze(-1))
    return y


def _spectral_radius(diag: Tensor, edge_index: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """``max |eig|`` of the assembled (small, demo-sized) block matrix."""
    A = assemble_blocks(diag, edge_index, lower, upper).to(torch.float64)
    return torch.linalg.eigvals(A).abs().max(dim=-1).values


def random_coupling(n: int, edge_index: Tensor, b: int, *, seed: int, rho: float,
                    dtype: torch.dtype = torch.float64):
    """Random non-symmetric block coupling ``W`` on ``edge_index``, scaled to ``rho(W) = rho``.

    Returns ``(Wd, Wl, Wu)`` (node-diagonal blocks and the two independent edge orientations).
    The fixed point of ``z = W z + b`` exists and is unique iff ``rho(W) < 1``; scaling all
    blocks by one scalar sets the spectral radius exactly while keeping the pattern.
    """
    g = torch.Generator().manual_seed(seed)
    m = edge_index.shape[1]
    Wd = 0.3 * torch.randn(n, b, b, generator=g, dtype=dtype)
    Wl = 0.5 * torch.randn(m, b, b, generator=g, dtype=dtype)
    Wu = 0.5 * torch.randn(m, b, b, generator=g, dtype=dtype)
    cur = _spectral_radius(Wd, edge_index, Wl, Wu)
    scale = rho / float(cur)
    return Wd * scale, Wl * scale, Wu * scale


def im_minus_jac_blocks(Wd: Tensor, edge_index: Tensor, Wl: Tensor, Wu: Tensor,
                        s: Tensor | None = None):
    """Blocks of ``A = I - diag(s) W`` (the equilibrium Jacobian ``I - J``).

    ``s`` is the per-coordinate activation derivative ``[..., n, b]`` (left mat-row scaling);
    ``s = None`` is the affine case ``J = W`` (``s = 1``). Returns ``(Adiag, Alower, Aupper)``
    in the :func:`nonsym_junction_solve` layout.
    """
    n, b = Wd.shape[-3], Wd.shape[-1]
    eye = torch.eye(b, dtype=Wd.dtype, device=Wd.device)
    if s is None:
        Adiag = eye - Wd
        Alo, Aup = -Wl, -Wu
    else:
        si = s.unsqueeze(-1)                              # [..., n, b, 1] row scaling
        Adiag = eye - si * Wd
        i, j = edge_index[0], edge_index[1]
        Alo = -s.index_select(-2, i).unsqueeze(-1) * Wl   # row i scaled
        Aup = -s.index_select(-2, j).unsqueeze(-1) * Wu   # row j scaled
    return Adiag, Alo, Aup


# --------------------------------------------------------------------------- #
# Affine fixed point: z* = (I - W)^{-1} b. Autograd through the structured solve
# *is* the exact implicit-differentiation gradient (no unrolling).
# --------------------------------------------------------------------------- #
def affine_fixed_point(Wd: Tensor, edge_index: Tensor, Wl: Tensor, Wu: Tensor, binj: Tensor,
                       order=None, *, transpose: bool = False) -> Tensor:
    """Equilibrium of the linear map ``z = W z + b``: ``z* = (I - W)^{-1} b``.

    One structured non-symmetric solve on ``A = I - W``. Differentiable: reverse-mode through
    :func:`nonsym_junction_solve` is exactly the implicit-function-theorem gradient.
    ``transpose=True`` returns ``(I - W)^{-T} b`` (the adjoint solve).
    """
    Adiag, Alo, Aup = im_minus_jac_blocks(Wd, edge_index, Wl, Wu)
    return nonsym_junction_solve(Adiag, edge_index, Alo, Aup, binj, order, transpose=transpose)


def neumann_adjoint(Wd: Tensor, edge_index: Tensor, Wl: Tensor, Wu: Tensor, g: Tensor,
                    K: int) -> Tensor:
    """``K``-term Neumann approximation of the adjoint solve ``(I - W^T) u = g``.

    The standard iterative DEQ backward: ``u_{k+1} = W^T u_k + g``, i.e. ``u_K = sum_{k<K}
    (W^T)^k g``. Converges like ``rho(W)^K`` -- the baseline the structured solve beats as
    ``rho -> 1``.
    """
    u = g
    term = g
    for _ in range(K - 1):
        term = spmv(Wd, edge_index, Wl, Wu, term, transpose=True)
        u = u + term
    return u


# --------------------------------------------------------------------------- #
# Nonlinear DEQ: forward by iteration, backward by one IFT solve (the op).
# --------------------------------------------------------------------------- #
class _DEQFixedPoint(torch.autograd.Function):
    """``z* = sigma(W z* + b)`` with the implicit-function-theorem backward.

    Forward iterates the (detached) Picard map to the equilibrium. Backward solves
    ``(I - J)^T u = grad`` once with ``J = diag(sigma'(p)) W`` via the structured non-symmetric
    transpose solve, then propagates to the parameters as the VJP ``u^T df/dtheta`` of a single
    differentiable application of ``f`` at the fixed point. No unrolling.
    """

    @staticmethod
    def forward(ctx, Wd, edge_index, Wl, Wu, binj, order, max_iter, tol):
        with torch.no_grad():
            z = torch.zeros_like(binj)
            for _ in range(max_iter):
                z_next = torch.tanh(spmv(Wd, edge_index, Wl, Wu, z) + binj)
                if (z_next - z).abs().max() < tol:
                    z = z_next
                    break
                z = z_next
        ctx.save_for_backward(Wd, Wl, Wu, binj, z)
        ctx.edge_index = edge_index
        ctx.order = order
        return z

    @staticmethod
    def backward(ctx, grad_z):
        Wd, Wl, Wu, binj, z = ctx.saved_tensors
        edge_index, order = ctx.edge_index, ctx.order
        # Jacobian at the fixed point: J = diag(sigma'(p)) W,  p = W z* + b,  sigma' = 1-tanh^2.
        with torch.no_grad():
            p = spmv(Wd, edge_index, Wl, Wu, z) + binj
            s = 1.0 - torch.tanh(p) ** 2
            Adiag, Alo, Aup = im_minus_jac_blocks(Wd, edge_index, Wl, Wu, s)
            u = nonsym_junction_solve(Adiag, edge_index, Alo, Aup, grad_z, order, transpose=True)
        # Parameter grads = u^T df/dtheta: VJP of one differentiable f-application at z* (fixed).
        with torch.enable_grad():
            Wd_ = Wd.detach().requires_grad_(True)
            Wl_ = Wl.detach().requires_grad_(True)
            Wu_ = Wu.detach().requires_grad_(True)
            b_ = binj.detach().requires_grad_(True)
            fz = torch.tanh(spmv(Wd_, edge_index, Wl_, Wu_, z) + b_)
            gWd, gWl, gWu, gb = torch.autograd.grad(fz, (Wd_, Wl_, Wu_, b_), grad_outputs=u)
        return gWd, None, gWl, gWu, gb, None, None, None


def deq_fixed_point(Wd: Tensor, edge_index: Tensor, Wl: Tensor, Wu: Tensor, binj: Tensor,
                    order=None, *, max_iter: int = 200, tol: float = 1e-10) -> Tensor:
    """Nonlinear deep-equilibrium layer ``z* = tanh(W z* + b)`` with the IFT backward.

    A genuine fixed-point layer (Picard forward, implicit backward) whose only non-trivial
    gradient computation is the structured non-symmetric transpose solve on ``I - J``. The
    coupling ``W`` is sparse on ``edge_index``; gradients flow to ``Wd``, ``Wl``, ``Wu``,
    ``binj`` exactly. Assumes a contraction (so the equilibrium is unique).
    """
    return _DEQFixedPoint.apply(Wd, edge_index, Wl, Wu, binj, order, max_iter, tol)


# --------------------------------------------------------------------------- #
# The result: backward-gradient accuracy vs rho(J), exact solve vs iterative.
# --------------------------------------------------------------------------- #
def _grad_blocks(u: Tensor, z: Tensor, edge_index: Tensor) -> Tensor:
    """Flattened parameter gradient ``dL/dW`` on the pattern from adjoint ``u`` and ``z*``.

    For ``L`` over ``z* = (I - W)^{-1} b`` the gradient is ``dL/dW_{ab} = u_a z*_b^T`` on the
    selected pattern (diagonal blocks + both edge orientations). Linear in ``u`` -- so the
    relative error of this vector is the relative error of the parameter gradient.
    """
    diag = u.unsqueeze(-1) @ z.unsqueeze(-2)                       # [n, b, b]: u_v z_v^T
    parts = [diag.reshape(-1)]
    if edge_index.shape[1]:
        i, j = edge_index[0], edge_index[1]
        gl = u.index_select(-2, i).unsqueeze(-1) @ z.index_select(-2, j).unsqueeze(-2)
        gu = u.index_select(-2, j).unsqueeze(-1) @ z.index_select(-2, i).unsqueeze(-2)
        parts += [gl.reshape(-1), gu.reshape(-1)]
    return torch.cat(parts)


def backward_accuracy_sweep(edge_index: Tensor, n: int, b: int, *, rhos=(0.5, 0.9, 0.99, 0.999),
                            Ks=(8, 16, 32), seed: int = 0):
    """Relative error of the affine-DEQ parameter gradient vs a dense implicit-diff oracle.

    For each ``rho``: build a coupling with ``rho(W) = rho``, a random injection and target,
    and compare the gradient from (a) the exact structured adjoint solve and (b) the ``K``-step
    Neumann backward, both against the dense oracle. Returns ``{rho: {"exact": e, "neumannK":
    e, ...}}``. The exact solve is flat at ``~1e-12``; Neumann tracks ``rho^K``.
    """
    out = {}
    for rho in rhos:
        Wd, Wl, Wu = random_coupling(n, edge_index, b, seed=seed, rho=rho)
        gbatch = torch.Generator().manual_seed(seed + 1)
        binj = torch.randn(n, b, generator=gbatch, dtype=torch.float64)
        target = torch.randn(n, b, generator=gbatch, dtype=torch.float64)

        # Forward equilibrium (exact) and the loss cotangent g = dL/dz* for L = 1/2||z*-t||^2.
        z = affine_fixed_point(Wd, edge_index, Wl, Wu, binj).detach()
        g_z = z - target

        # Dense oracle adjoint and gradient.
        A = assemble_blocks(torch.eye(b, dtype=torch.float64) - Wd, edge_index, -Wl, -Wu)
        u_oracle = torch.linalg.solve(A.T, g_z.reshape(-1)).reshape(n, b)
        grad_oracle = _grad_blocks(u_oracle, z, edge_index)
        ref = grad_oracle.norm()

        # (a) exact structured adjoint solve.
        u_exact = affine_fixed_point(Wd, edge_index, Wl, Wu, g_z, transpose=True).detach()
        row = {"exact": float((_grad_blocks(u_exact, z, edge_index) - grad_oracle).norm() / ref)}
        # (b) K-step iterative (Neumann) backward.
        for K in Ks:
            u_K = neumann_adjoint(Wd, edge_index, Wl, Wu, g_z, K)
            row[f"neumann{K}"] = float((_grad_blocks(u_K, z, edge_index) - grad_oracle).norm() / ref)
        out[rho] = row
    return out


def fit_affine_fixed_point(edge_index: Tensor, n: int, b: int, *, rho: float = 0.9,
                           steps: int = 300, lr: float = 0.05, seed: int = 0):
    """Learnability gate: fit a coupling so ``(I - W)^{-1} b`` matches a target, via the op.

    The whole gradient path runs through :func:`nonsym_junction_solve` (the exact IFT
    backward). Returns ``(mse0, mse_final)``; a working backward drives the MSE down by orders
    of magnitude. A minimal end-to-end check -- the affine analogue of the maze learnability
    test -- not a tuned benchmark.
    """
    Wd0, Wl0, Wu0 = random_coupling(n, edge_index, b, seed=seed, rho=rho)
    gen = torch.Generator().manual_seed(seed + 7)
    binj = torch.randn(n, b, generator=gen, dtype=torch.float64)
    # Target field from a *different* coupling the model must match through the inverse.
    WdT, WlT, WuT = random_coupling(n, edge_index, b, seed=seed + 99, rho=rho)
    target = affine_fixed_point(WdT, edge_index, WlT, WuT, binj).detach()

    Wd = (0.5 * Wd0).clone().requires_grad_(True)
    Wl = (0.5 * Wl0).clone().requires_grad_(True)
    Wu = (0.5 * Wu0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Wd, Wl, Wu], lr=lr)
    mse0 = None
    for t in range(steps):
        opt.zero_grad()
        z = affine_fixed_point(Wd, edge_index, Wl, Wu, binj)
        loss = torch.mean((z - target) ** 2)
        loss.backward()
        opt.step()
        if t == 0:
            mse0 = float(loss.detach())
    return mse0, float(loss.detach())


def run_demo(rows: int = 3, cols: int = 3, b: int = 2, **kw):
    """Print the rho-sweep backward-accuracy table on a loopy grid coupling (genuine fill)."""
    from ..generators import grid_edges

    edge_index = grid_edges(rows, cols)
    n = rows * cols
    res = backward_accuracy_sweep(edge_index, n, b, **kw)
    Ks = [k for k in next(iter(res.values())) if k.startswith("neumann")]
    header = f"{'rho(J)':>8} {'exact':>11} " + " ".join(f"{k:>11}" for k in Ks)
    print(f"DEQ backward gradient error vs dense oracle  (grid {rows}x{cols}, b={b}, fill)")
    print(header)
    print("-" * len(header))
    for rho, row in res.items():
        line = f"{rho:>8.3f} {row['exact']:>11.2e} " + " ".join(f"{row[k]:>11.2e}" for k in Ks)
        print(line)
    mse0, mse1 = fit_affine_fixed_point(edge_index, n, b)
    print(f"\nlearnability (fit through the exact backward): MSE {mse0:.2e} -> {mse1:.2e}")
    return res


if __name__ == "__main__":
    run_demo()
