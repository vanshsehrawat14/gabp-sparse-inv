# Project status

This is the authoritative public implementation and scope record.

**Last audited:** 2026-07-23 PDT
**Version:** 0.3.3

## Implemented kernels

All selected-inverse APIs return requested or filled-pattern blocks of `A^{-1}` without
materializing the dense inverse.

### Symmetric positive-definite

| Structure | API | Differentiation |
|---|---|---|
| Block tridiagonal chain | `selected_inverse_chain` | functional/autograd |
| Block arrowhead star | `selected_inverse_star` | functional/autograd |
| General tree | `selected_inverse_tree` | functional/autograd |
| Tree analytic path | `selinv_tree` | first-order custom backward |
| General sparse SPD / junction | `selected_inverse_junction`, `selinv_junction` | functional/autograd |
| Junction analytic path | `selinv_junction_analytic` | first-order custom backward |

The junction result is on the filled Cholesky pattern. Reference and level-set
`batched=True` paths are tested for agreement.

### Non-symmetric

| Structure | API | Boundary |
|---|---|---|
| Lower bidiagonal | `selected_inverse_bidiag`, `selinv_bidiag` | structured, no pivoting |
| Triangular chunk `(I-A)^{-1}` | `selected_inverse_tril`, `selinv_tril` | dense triangular |
| General tree | `selected_inverse_nonsym_tree` | zero fill, nonsingular pivots |
| Filled-pattern LU | `selected_inverse_nonsym_junction`, `selinv_nonsym_junction` | static structurally symmetric pattern, no pivoting |
| Analytic filled-pattern LU | `selinv_nonsym_junction_analytic` | first-order custom backward |
| Transpose/non-transpose solve | `nonsym_junction_solve` | same static no-pivot regime |

For the inversion derivative `L_A`, the non-symmetric VJP is `L_A^* = L_{A^T}`. It is a
same-pattern reverse schedule with independent lower/upper cotangents, not the
selected-inverse value of `A^T`.

## Factor-derived operations

| Capability | API / object | Release status |
|---|---|---|
| SPD solve | `tree_solve`, `junction_solve` | public |
| SPD log determinant | `tree_logdet`, `junction_logdet` | public |
| Gaussian sampling | `sample_gaussian_tree`, `sample_gaussian_junction` | public |
| Tree/grid GMRF | `gabp_sparse_inv.gmrf`, `gabp_sparse_inv.gmrf_grid` | public |

The public API exposes separate function calls. Cross-call factor reuse is not part of
version 0.3.3, and separate calls may refactorize.

## Complexity

For a fixed elimination order, let `w_v` be the number of later neighbours of block `v`:

```
F = sum_v (1 + w_v)
W = sum_v (1 + w_v^2)
```

- numeric factor/selected-block storage: `Theta(F b^2)`;
- factorization plus selected inversion: `Theta(W b^3)`; and
- current checked level-set symbolic metadata: `Theta(W)`.

For fixed block size and bounded fronts, chains/trees and bounded-treewidth families are linear
in `n`. General arithmetic is not `O(fill)`. On a 2-D nested-dissection grid,
`F = Theta(n log n)` and `W = Theta(n^(3/2))`.

## Numerical scope

- SPD paths are algebraically exact up to floating-point rounding and need no pivoting.
- “Direct” removes convergence truncation; it does not remove `kappa(A) u` sensitivity.
- Finite equal-precision diagnostics show no systematic selected-inverse precision penalty and
  no robust advantage against a well-ordered dense factorization.
- The local primitive error bounds do not prove a global height- or condition-independent
  structured backward-error theorem.
- Non-symmetric paths require nonsingular, numerically usable no-pivot Schur pivots.

## Demonstrations

| Demonstration | File | What it establishes |
|---|---|---|
| Tree routed field | `demos/maze_tree.py` | solve is the only global operator in a controlled proxy |
| Grid routed field | `demos/maze_grid.py` | same mechanism on a loopy graph/fill pattern |
| DEQ / fixed point | `demos/deq_fixedpoint.py` | exact sparse-direct implicit gradient vs finite Neumann bias |
| DeltaNet chunk | `demos/deltanet_chunk.py` | triangular chunk identity and gradient equivalence |

These are capability/mechanism checks, not production speed or state-of-the-art learning claims.
The “maze” task is a routed-field proxy, not a full learned discrete maze solver.

## Verification surface

The suite covers, where applicable:

- dense fp64 value oracles and on-pattern residual identities;
- `gradcheck`, analytic-versus-functional VJPs, and `gradgradcheck` for documented functional
  higher-order paths;
- singleton/empty-edge/batch/dtype cases;
- chain/tree/star/junction and symmetric/non-symmetric reductions;
- order invariance and selected trace identities;
- maze/DEQ/DeltaNet mechanism invariants; and
- benchmark helper invariants.

Run:

```powershell
python -m pytest -q
```

Higher-order support for a functional implementation does not prove arbitrary-order
same-pattern closure. Custom analytic backwards are first-order unless explicitly documented
otherwise.

## Packaging

- Version 0.3.3 builds an sdist and universal wheel for Python `>=3.12`, requiring
  `torch>=2.2`.
- Public CI covers Python 3.12/3.13 on Linux and Windows plus Python 3.12 on macOS.
- The release gate includes metadata validation and an isolated wheel smoke test.
- Public GPU performance is not published.
- The corrected Paper 1 source and its tracked data are included under
  `paper/attainability/`; the JOSS paper remains in preparation.

## Deliberately out of scope

- singular PSD pseudoinverse/gauge handling;
- pivoted non-symmetric selected inversion;
- arbitrary directed sparsity without the documented structural pattern;
- indefinite or complex-Hermitian systems;
- approximate/iterative loopy GaBP as a package feature;
- heterogeneous block sizes;
- arbitrary off-pattern inverse entries;
- second-order autograd through first-order custom analytic backwards; and
- performance competition with mature sparse-direct systems without a measured user workload.
