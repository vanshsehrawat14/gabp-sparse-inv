"""DeltaNet / linear-attention chunk inverse as a hot-swappable differentiable op.

The capability rung of the roadmap (``docs/ROADMAP.md`` -- the M-JOSS "impact" demo that
needs *no new kernel*): the differentiable triangular chunk inverse ``T = (I - A)^{-1}``
already shipped in :mod:`gabp_sparse_inv.nonsym` (``selinv_tril``) drops, unchanged, into a
real chunked linear-attention layer and trains identically to the stock autograd baseline.

**Background.** DeltaNet / gated linear attention parallelise the *delta rule* over a chunk of
``C`` tokens. Writing token ``i`` changes the state that token ``j > i`` reads, so the
within-chunk corrected writes ``W`` solve a lower-triangular system (Yang et al. 2024): with
``delta_i = beta_i (v_i - k_i^T S_prev)`` and the strictly-lower coupling
``L = tril(diag(beta) K K^T, -1)``,

    (I + L) W = delta        i.e.   W = (I - A)^{-1} delta,   A = -L

-- exactly the chunk inverse ``selinv_tril`` computes. :func:`delta_chunk_attention` is a
minimal, faithful chunked delta-rule layer built around that one inverse; with a strictly
causal intra-chunk read it reproduces the token-by-token delta rule **exactly** (the chunking
is an algebraic identity, validated against :func:`sequential_delta_reference`).

**What this demonstrates (the drop-in equivalence).** :func:`chunk_inverse_apply` forms ``T``
two ways inside the same layer:

  * ``method="selinv"`` -- ``selinv_tril`` (the analytic transpose-form VJP
    ``bar_A = tril(T^T bar_T T^T, -1)``, ``docs/derivations.md`` §9.4, no autograd tape over
    the triangular solve), and
  * ``method="solve"`` -- the stock baseline ``torch.linalg.solve_triangular(I - A, I)`` with
    autograd through it.

The forward is identical by construction (both call ``solve_triangular``); the content is that
the **analytic backward equals autograd's** through the whole multi-chunk layer (forward to
``~1e-12``, gradients to ``~1e-9``), and that a layer **trains identically** through either
(:func:`fit_teacher_student`). So the op is genuinely hot-swappable.

**Scope.** This is a *capability / drop-in-equivalence* result, **not** a DeltaNet
reimplementation or a performance claim. The dense chunk inverse is ``O(C^3)`` with
no sparsity win. The analytic transpose-form backward avoids taping the triangular solve, but
forming the full inverse and its dense VJP is still cubic in ``C`` here. The *same*
selected-inverse op serves both the sparse kernels and this dense triangular case. The toy
training task is a synthetic teacher-student fit (its job is to show the op trains, not to beat
a recall benchmark) -- the same honesty bar as ``docs/MAZE.md`` / ``docs/DEQ.md``.

Run ``python -m gabp_sparse_inv.demos.deltanet_chunk`` for the equivalence + training table.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..nonsym import selinv_tril

__all__ = [
    "chunk_inverse_apply",
    "delta_chunk_attention",
    "sequential_delta_reference",
    "init_params",
    "forward_from_inputs",
    "equivalence_check",
    "fit_teacher_student",
    "run_demo",
]


# --------------------------------------------------------------------------- #
# The drop-in point: T = (I - A)^{-1} applied to RHS, two interchangeable ways.
# --------------------------------------------------------------------------- #
def chunk_inverse_apply(A: Tensor, rhs: Tensor, *, method: str = "selinv") -> Tensor:
    """Apply the chunk inverse ``(I - A)^{-1} @ rhs``; ``A`` strictly-lower ``[..., C, C]``.

    ``method="selinv"`` forms ``T`` with :func:`~gabp_sparse_inv.selinv_tril` (analytic
    transpose-form VJP); ``method="solve"`` forms it with
    ``torch.linalg.solve_triangular(I - A, I)`` and lets autograd differentiate the solve.
    Both compute the same ``T`` (only ``A.tril(-1)`` is used) and then the same ``T @ rhs`` --
    the single swapped line is how ``T`` is obtained, so this is a literal drop-in.
    """
    if method == "selinv":
        T = selinv_tril(A)
    elif method == "solve":
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        T = torch.linalg.solve_triangular(eye - A.tril(-1), eye, upper=False, unitriangular=True)
    else:
        raise ValueError(f"method must be 'selinv' or 'solve'; got {method!r}")
    return T @ rhs


# --------------------------------------------------------------------------- #
# Minimal chunked delta-rule linear attention (the layer the inverse lives in).
# --------------------------------------------------------------------------- #
def delta_chunk_attention(
    Q: Tensor, K: Tensor, V: Tensor, beta: Tensor, chunk_size: int, *, method: str = "selinv"
) -> Tensor:
    """Chunked delta-rule linear attention; the within-chunk mixing is ``(I - A)^{-1}``.

    Parameters
    ----------
    Q, K, V:
        Queries / keys / values ``[..., L, d]`` (already projected).
    beta:
        Per-token write strengths ``[..., L]`` in ``(0, 1)``.
    chunk_size:
        Tokens per chunk ``C``; need not divide ``L`` (the last chunk may be shorter). Larger
        ``C`` means a larger triangular inverse but the **same** output (chunking is exact).
    method:
        How to apply the chunk inverse (see :func:`chunk_inverse_apply`).

    Returns
    -------
    O:
        Layer output ``[..., L, d]``. With the strictly-causal intra-chunk read this equals
        the token-by-token delta rule exactly (:func:`sequential_delta_reference`).
    """
    d = Q.shape[-1]
    L = Q.shape[-2]
    batch = Q.shape[:-2]
    K = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # L2-normalised keys (as in DeltaNet)
    S = Q.new_zeros((*batch, d, d))                       # state: maps a key to its value
    outs = []
    for c0 in range(0, L, chunk_size):
        Kc = K[..., c0:c0 + chunk_size, :]                # [..., C, d]
        Vc = V[..., c0:c0 + chunk_size, :]
        Qc = Q[..., c0:c0 + chunk_size, :]
        bc = beta[..., c0:c0 + chunk_size].unsqueeze(-1)  # [..., C, 1]

        A = -(bc * (Kc @ Kc.mT)).tril(-1)                 # -tril(diag(beta) K K^T, -1)
        delta = bc * (Vc - Kc @ S)                        # targets to write, relative to S
        W = chunk_inverse_apply(A, delta, method=method)  # corrected within-chunk writes
        Oc = Qc @ S + (Qc @ Kc.mT).tril(-1) @ W           # read state + strictly-causal intra
        outs.append(Oc)
        S = S + Kc.mT @ W                                 # state update: S += sum_i k_i w_i^T
    return torch.cat(outs, dim=-2)


def sequential_delta_reference(Q: Tensor, K: Tensor, V: Tensor, beta: Tensor) -> Tensor:
    """Token-by-token delta rule -- the ground-truth oracle the chunked layer must match.

    An ``O(L)`` Python loop with no chunking and no inverse: each token reads the running
    state (strictly causal), then writes ``beta_t (v_t - k_t^T S) k_t^T`` into it. The chunked
    :func:`delta_chunk_attention` is an exact algebraic reformulation of this, so it must agree
    block-for-block at every chunk size.
    """
    d = Q.shape[-1]
    L = Q.shape[-2]
    batch = Q.shape[:-2]
    K = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # match delta_chunk_attention
    S = Q.new_zeros((*batch, d, d))
    outs = []
    for t in range(L):
        k = K[..., t, :].unsqueeze(-2)                    # [..., 1, d]
        v = V[..., t, :].unsqueeze(-2)
        q = Q[..., t, :].unsqueeze(-2)
        bt = beta[..., t].unsqueeze(-1).unsqueeze(-1)     # [..., 1, 1]
        outs.append((q @ S).squeeze(-2))                  # read BEFORE writing (causal)
        w = bt * (v - k @ S)                              # [..., 1, d]
        S = S + k.mT @ w                                  # S += k w^T
    return torch.stack(outs, dim=-2)


# --------------------------------------------------------------------------- #
# A thin parameterisation, so the layer can be trained (teacher-student).
# --------------------------------------------------------------------------- #
def init_params(d: int, *, seed: int, scale: float = 1.0, dtype: torch.dtype = torch.float64):
    """Learnable projections ``W_Q, W_K, W_V`` (``d x d``) and ``w_beta`` (``d``)."""
    g = torch.Generator().manual_seed(seed)

    def mk(*shape):
        return scale * torch.randn(*shape, generator=g, dtype=dtype) / d ** 0.5

    return {"WQ": mk(d, d), "WK": mk(d, d), "WV": mk(d, d), "wb": mk(d)}


def forward_from_inputs(X: Tensor, params: dict, chunk_size: int, *, method: str = "selinv") -> Tensor:
    """Project inputs ``X`` ``[..., L, d]`` and run :func:`delta_chunk_attention`."""
    Q = X @ params["WQ"]
    K = X @ params["WK"]
    V = X @ params["WV"]
    beta = torch.sigmoid(X @ params["wb"])                # [..., L] in (0, 1)
    return delta_chunk_attention(Q, K, V, beta, chunk_size, method=method)


# --------------------------------------------------------------------------- #
# The drop-in equivalence: same forward, analytic backward == autograd backward.
# --------------------------------------------------------------------------- #
def equivalence_check(d: int = 8, L: int = 12, chunk_size: int = 4, *, batch: int = 4, seed: int = 0):
    """Max forward and gradient discrepancy between the ``selinv`` and ``solve`` paths.

    Returns ``(fwd_err, grad_err)`` over ``(Q, K, V, beta)``. The forward is identical by
    construction; ``grad_err`` is the substantive number -- the analytic ``selinv_tril``
    backward vs autograd through ``solve_triangular``, composed across all chunks.
    """
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(batch, L, d, generator=g, dtype=torch.float64)
    K = torch.randn(batch, L, d, generator=g, dtype=torch.float64)
    V = torch.randn(batch, L, d, generator=g, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(batch, L, generator=g, dtype=torch.float64))

    O1 = delta_chunk_attention(Q, K, V, beta, chunk_size, method="selinv")
    O2 = delta_chunk_attention(Q, K, V, beta, chunk_size, method="solve")
    fwd_err = float((O1 - O2).abs().max())

    def grads(method):
        leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, beta)]
        O = delta_chunk_attention(leaves[0], leaves[1], leaves[2], leaves[3], chunk_size, method=method)
        O.pow(2).sum().backward()
        return [t.grad for t in leaves]

    grad_err = max(float((a - b).abs().max()) for a, b in zip(grads("selinv"), grads("solve")))
    return fwd_err, grad_err


def fit_teacher_student(
    d: int = 8, L: int = 12, chunk_size: int = 4, *, batch: int = 16, steps: int = 500,
    lr: float = 0.1, seed: int = 0, method: str = "selinv",
):
    """Learnability gate: fit a student layer to a fixed teacher layer's outputs, via the op.

    The whole gradient path runs through the chunk inverse (``method``). Returns
    ``(mse0, mse_final)``; a working backward drives the MSE down by orders of magnitude. A
    minimal end-to-end check (the DeltaNet analogue of the maze / DEQ learnability gates), not
    a tuned benchmark. Identical ``(mse0, mse_final)`` for ``method`` in ``{selinv, solve}``
    (same init / data) is the end-to-end drop-in statement.
    """
    teacher = init_params(d, seed=seed + 100, scale=1.5)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch, L, d, generator=g, dtype=torch.float64)
    with torch.no_grad():
        target = forward_from_inputs(X, teacher, chunk_size, method=method)

    student = init_params(d, seed=seed, scale=1.0)
    leaves = {k: v.clone().requires_grad_(True) for k, v in student.items()}
    opt = torch.optim.Adam(list(leaves.values()), lr=lr)
    mse0 = None
    for t in range(steps):
        opt.zero_grad()
        loss = torch.mean((forward_from_inputs(X, leaves, chunk_size, method=method) - target) ** 2)
        loss.backward()
        opt.step()
        if t == 0:
            mse0 = float(loss.detach())
    return mse0, float(loss.detach())


def run_demo(d: int = 8, L: int = 12, chunk_size: int = 4):
    """Print the drop-in equivalence numbers and a train-both-ways table."""
    fwd_err, grad_err = equivalence_check(d=d, L=L, chunk_size=chunk_size)
    print("DeltaNet chunk inverse (I - A)^{-1}: selinv_tril vs solve_triangular (drop-in)")
    print(f"  L={L}, chunk_size={chunk_size}, d={d}")
    print(f"  forward  max|diff| = {fwd_err:.2e}   (identical by construction)")
    print(f"  gradient max|diff| = {grad_err:.2e}   (analytic backward == autograd)")
    print("  teacher-student fit (MSE start -> end), trained through each path:")
    for method in ("selinv", "solve"):
        mse0, mse1 = fit_teacher_student(d=d, L=L, chunk_size=chunk_size, method=method)
        print(f"    {method:>7}:  {mse0:.2e} -> {mse1:.2e}")
    return fwd_err, grad_err


if __name__ == "__main__":
    run_demo()
