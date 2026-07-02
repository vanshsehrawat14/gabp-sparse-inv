# Demonstration: the non-symmetric selected inverse as the exact DEQ backward

This note records the **deep-equilibrium / fixed-point** demonstration: a controlled case where
the non-symmetric selected inverse is the exact implicit-differentiation backward for a
graph-structured fixed-point layer. Code:
`gabp_sparse_inv/demos/deq_fixedpoint.py`; tests: `tests/test_deq_fixedpoint.py`.

## The op's role: implicit differentiation is a non-symmetric solve

A deep-equilibrium model defines its output as a fixed point `z* = f(z*, x)`. Differentiating
the equilibrium condition (the implicit function theorem) gives the backward pass as a single
linear solve with the **equilibrium Jacobian** `J = ∂f/∂z |_{z*}`:

    (I − J)ᵀ u = ∂L/∂z* ,        then   ∂L/∂θ = uᵀ ∂f/∂θ .

`I − J` is **non-symmetric**. When `f`'s coupling is sparse on a graph - the low-treewidth
regime this whole project targets - that adjoint solve is exactly
`nonsym_junction_solve(..., transpose=True)` on `A = I − J`: the block `LDU` of
[derivations.md](derivations.md) §10, computed once, `O(fill)`, with the *transpose* of the
same factors giving the adjoint at no extra cost. This is the piece the general non-symmetric
kernel (Track C) unblocked.

The module realizes both the affine equilibrium `z* = (I − W)⁻¹ b` (where autograd through the
structured solve *is* the exact implicit-diff gradient, no unrolling) and a genuine nonlinear
DEQ `z* = tanh(W z* + b)` (`deq_fixed_point`: Picard forward, a custom IFT backward whose only
non-trivial step is the structured transpose solve). The nonlinear backward is validated equal
to autograd through a fully unrolled solver to `~1e-15`, and `gradcheck` passes.

## The result: exact and robust as the equilibrium stiffens (ρ(J) → 1)

Standard DEQ backprop solves `(I − J)ᵀ u = g` **iteratively** (Neumann / Richardson:
`u_{k+1} = Jᵀ u_k + g`), which converges like `ρ(J)ᵏ` - arbitrarily slow as the fixed point
stiffens (`ρ(J) → 1`). The structured solve is one-shot and stays machine-accurate at every
`ρ`. `backward_accuracy_sweep` builds a non-symmetric coupling on a loopy 3×3 grid (genuine
fill), scales it to a target spectral radius, and reports the **parameter-gradient relative
error vs a dense implicit-diff oracle** for the exact solve and for the `K`-step iterative
backward.

CPU, fp64; `python -m gabp_sparse_inv.demos.deq_fixedpoint`:

| ρ(J) | exact (selected inverse) | Neumann-8 | Neumann-16 | Neumann-32 |
|-----:|-------------------------:|----------:|-----------:|-----------:|
| 0.500 | 1.8e-16 | 1.2e-02 | 5.0e-05 | 7.6e-10 |
| 0.900 | 5.2e-16 | 6.7e-01 | 3.1e-01 | 5.6e-02 |
| 0.990 | 1.6e-14 | 9.7e-01 | 9.1e-01 | 7.7e-01 |
| 0.999 | 1.1e-12 | 1.0e+00 | 9.9e-01 | 9.8e-01 |

The exact selected-inverse backward is at machine precision everywhere - the `1e-12` floor at
`ρ = 0.999` is just `κ·u` from the (now nearly singular) solve, not iteration error. The
iterative backward tracks `ρᴷ`: at `ρ = 0.99` even 32 iterations leaves the gradient **77%
wrong**. A linear layer also trains cleanly through the exact backward (learnability check:
MSE `~11 → ~1e-11`).

## Scope and honest caveats

- **The advantage is for graph-structured (low-treewidth) Jacobians** - this project's regime.
  For a *dense* Jacobian the `LDU` is `O(n³)` and an iterative solver is the right tool; the
  structured solve wins precisely when `I − J` is sparse on a low-treewidth graph, which is the
  same assumption the maze and the rest of the program make.
- **`ρ(J) < 1` is assumed** (a contraction - the equilibrium exists and is unique), the
  non-symmetric analogue of the maze's SPD-by-construction. The result is *about* the approach
  to that boundary, where iterative backprop is known to struggle.
- **Accelerated solvers (Anderson/Broyden)** reduce the iteration count of the iterative
  backward but not its `ρ → 1` scaling; the structured solve removes it. Neumann is shown as the
  transparent baseline, not a claim that practitioners use the worst iterative method.
- This is a **capability / mechanism** result on controlled problems (exact `O(fill)` implicit
  gradients, robust as `ρ → 1`), not a SOTA claim on a task - the same honesty bar as
  [MAZE.md](MAZE.md). The machine-independent facts are gated in `tests/test_deq_fixedpoint.py`
  (exact backward `< 1e-9` at every `ρ ≤ 0.99`; iterative `≫` exact and growing toward `ρ → 1`;
  nonlinear IFT backward `==` unrolled autograd; end-to-end learnability).
