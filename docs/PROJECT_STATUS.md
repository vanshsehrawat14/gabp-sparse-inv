# Project Status

This document records the public package status: what is implemented, tested, and deliberately
out of scope. It is not a benchmark report; hardware-dependent numbers remain diagnostics unless
called out as tests.

## Implemented

`gabp-sparse-inv` provides PyTorch selected-inverse operators for sparse block-structured
matrices. Each kernel returns the blocks of `A^{-1}` on the input pattern or on the filled
factor pattern, without forming the dense inverse.

### Symmetric Positive-Definite Kernels

| Structure | Public API | Status |
|---|---|---|
| Block tridiagonal chain | `selected_inverse_chain` | Implemented and tested |
| Block arrowhead star | `selected_inverse_star` | Implemented and tested |
| General tree | `selected_inverse_tree` | Implemented and tested |
| Differentiable tree | `selinv_tree` | Implemented and tested, first-order analytic backward |
| General sparse SPD / junction tree | `selected_inverse_junction`, `selinv_junction` | Implemented and tested |
| Tape-free junction backward | `selinv_junction_analytic` | Implemented and tested |

The junction kernel symbolically completes the input pattern to the filled Cholesky pattern and
returns selected inverse blocks on that filled pattern. It also has a level-set `batched=True`
path that matches the reference path block-for-block.

### Non-Symmetric Kernels

| Structure | Public API | Status |
|---|---|---|
| Block lower-bidiagonal | `selected_inverse_bidiag`, `selinv_bidiag` | Implemented and tested |
| Dense triangular chunk inverse `(I - A)^{-1}` | `selected_inverse_tril`, `selinv_tril` | Implemented and tested |
| Non-symmetric tree | `selected_inverse_nonsym_tree` | Implemented and tested |
| General sparse non-symmetric / LU | `selected_inverse_nonsym_junction`, `selinv_nonsym_junction` | Implemented and tested |
| Non-symmetric sparse solve | `nonsym_junction_solve` | Implemented and tested |
| Tape-free non-symmetric junction backward | `selinv_nonsym_junction_analytic` | Implemented and tested |

The general non-symmetric kernels use a static symbolic pattern and no pivoting. They are intended
for regimes where the chosen elimination order has nonsingular, numerically usable pivots.

### Solves, Log-Determinants, And Sampling

| Capability | Public API | Status |
|---|---|---|
| SPD sparse solve | `junction_solve`, `tree_solve` | Implemented and tested |
| Log determinant | `junction_logdet`, `tree_logdet` | Implemented and tested |
| Exact Gaussian sampling | `sample_gaussian_tree`, `sample_gaussian_junction` | Implemented and tested |
| Tree GMRF learning | `gabp_sparse_inv.gmrf` | Implemented and tested |
| Grid / loopy GMRF learning | `gabp_sparse_inv.gmrf_grid` | Implemented and tested |

### Demonstrations

| Demonstration | File | Status |
|---|---|---|
| Tree maze | `gabp_sparse_inv/demos/maze_tree.py` | Implemented and tested |
| Grid maze | `gabp_sparse_inv/demos/maze_grid.py` | Implemented and tested |
| DEQ / fixed-point backward | `gabp_sparse_inv/demos/deq_fixedpoint.py` | Implemented and tested |
| DeltaNet chunk inverse | `gabp_sparse_inv/demos/deltanet_chunk.py` | Implemented and tested |

The demonstrations are capability and mechanism checks. They are not claims of state-of-the-art
model performance.

## Verification Bar

The test suite covers the supported kernels with dense fp64 oracle checks and structure-specific
invariants. Where applicable, tests also cover:

- `torch.autograd.gradcheck` and analytic-vs-autograd adjoints.
- `gradgradcheck` for functional kernels that support second-order derivatives.
- Edge cases such as `n=1`, scalar blocks, empty-edge patterns, explicit topologies, and leading
  batch dimensions.
- Cross-kernel reductions such as chain-as-tree, star-as-tree, junction-as-tree, and symmetric
  non-symmetric reductions.
- `compute_dtype` paths and low-precision storage sanity checks.
- Order invariance for junction solve and selected inverse outputs.
- On-pattern residual identities and trace identities.

Run:

```bash
pytest -q
```

## Numerical Characterization

The precision harness compares selected inversion with dense solves at equal precision, scored
against fp64 dense oracles on the returned pattern. Current diagnostic reading:

- Selected inversion shows no systematic precision penalty versus a dense SPD solve.
- It also shows no robust precision advantage; accuracy is generally comparable to a dense
  factorization in a good elimination order.
- The non-symmetric no-pivot path is at parity with pivoted dense LU while pivots remain well
  conditioned, and degrades when the static no-pivot assumption breaks.

These are diagnostics, not CI assertions. See `gabp_sparse_inv.bench.precision` and
`gabp_sparse_inv.bench.nonsym_stability`.

## Packaging Status

- Package metadata, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, changelog, CI, and
  JOSS paper draft are present.
- The version is `0.3.1`.
- PyPI publication is still pending.
- GPU timing is not yet published. The CUDA-compatible level-set paths exist, but the public
  package does not ship measured GPU benchmark claims.

## Deliberately Out Of Scope

The package does not currently claim support for:

- Pivoted non-symmetric / LU selected inversion.
- Iterative or approximate loopy Gaussian belief propagation.
- Indefinite, complex Hermitian, or heterogeneous-block-size kernels.
- Second-order autograd through the custom analytic `selinv_tree`, `selinv_bidiag`, or
  `selinv_tril` backwards.
- Raw CPU-performance competition with mature sparse-direct selected-inversion packages.

Use the functional kernels when higher-order derivatives are needed.
