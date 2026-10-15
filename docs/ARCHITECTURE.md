# Architecture

A map of the package for contributors: what each module owns, the pattern every kernel
follows, and the conventions the tests enforce. The mathematics is in
[`derivations.md`](derivations.md); the supported scope is in the
[README](../README.md).

## What the package computes

The **selected inverse** of a sparse block matrix `A`: the blocks of `A⁻¹` on the pattern
of `A`, or on its filled pattern after symbolic elimination, without forming the dense
inverse, together with derivatives with respect to the stored input blocks: first order
through every kernel, second order through the functional ones.
On a tree the forward is the two-pass collect/distribute of Gaussian belief propagation,
which is the Takahashi recurrence on a zero-fill elimination order; on a general pattern
it is the same recurrence on the filled (chordal) pattern. The reverse-mode derivative is
a two-pass schedule on the same elimination structure. All of this is classical
(`derivations.md` §7); the package is a PyTorch implementation of it.

For a fixed elimination order let `w_v` be the number of neighbours of node `v` that are
eliminated after it in the filled graph (the pattern of `A` plus the fill elimination
creates), not in the graph of `A`:

- `F = sum_v (1 + w_v)` counts factor and selected blocks, so numeric storage is
  `Theta(F b^2)`;
- `W = sum_v (1 + w_v^2)` counts dense clique work, so factorization and selected
  inversion cost `Theta(W b^3)`.

Bounded front width makes them proportional; it is sufficient, not necessary. When no
order is given, the default greedy minimum-degree order is computed in Python with
`O(n^2)` node scans and is recomputed by every functional call; pass an order or reuse an
`analyze_junction` plan on large patterns.

## Module map (`gabp_sparse_inv/`)

| module | what it owns |
|--|--|
| `layout.py` | storage containers `BlockTridiag`, `BlockBidiag`, `BlockStar`, `BlockTree`, `BlockSparseSym`, each with `to_dense` / `from_dense` and `validate`; tree orderings |
| `_linalg.py` | batched Cholesky with an informative failure message, inverse from a Cholesky factor, the local element-growth guard of the non-symmetric eliminations, and the refusal of second derivatives through the analytic backwards |
| `generators.py` | seeded test matrices with a monotone conditioning knob, `grid_edges`, `condition_number` |
| `chain.py`, `star.py`, `tree.py` | SPD selected inverse on paths, stars and general trees |
| `junction.py` | SPD selected inverse, solve and log-determinant on the filled pattern; ordering helpers; the level-set `batched=True` schedule; `analyze_junction` / `SparseGaussianFactor` for reusing one factorization |
| `nonsym.py` | non-symmetric lower-bidiagonal and tree selected inverses; the dense `(I - A)⁻¹` chunk inverse with an analytic backward |
| `nonsym_junction.py` | non-symmetric selected inverse and (transpose) solve on the filled pattern, fixed order, no pivoting |
| `autodiff.py` | `SelInvTree` / `selinv_tree`: analytic, level-set batched tree backward, first order only |
| `junction_autodiff.py` | tape-free analytic backwards for the SPD and non-symmetric junction kernels, first order only |
| `gmrf.py`, `gmrf_grid.py` | tree and grid GMRF precisions, marginal likelihood, posterior marginal variances, fitting helpers |
| `sampling.py` | exact Gaussian sampling on tree and junction layouts |
| `bench/` | accuracy, stability and timing harnesses (`python -m gabp_sparse_inv.bench.run`) |

## Adding a kernel

Every supported structure has the same pieces; `junction.py` with `BlockSparseSym`,
`random_spd_graph` and `tests/test_junction.py` is the reference example.

1. **Container** in `layout.py`: structured storage, `to_dense` / `from_dense`, and a
   `validate` that raises `TypeError` / `ValueError` (never a bare `assert`).
2. **Generator** in `generators.py`: a PSD base on the exact pattern plus `diag_load * I`,
   so `diag_load` controls conditioning monotonically.
3. **Kernel**: a functional forward (below).
4. **Tests** at the bar described below.
5. **Docs**: the recurrence in `derivations.md`, a README entry, and the export in
   `__init__.py`.

## The differentiable pattern

Write the forward functionally: out-of-place arithmetic with `torch.linalg.cholesky`,
`torch.cholesky_solve`, matrix products and `.mT`, never in-place writes to tensors that
require gradients. Reverse-mode autograd through such a forward is the reverse schedule
of `derivations.md` §8.4, with exact gradients checked by `gradcheck`, and second
derivatives checked by `gradgradcheck`. Hand-written analytic backwards (`autodiff.py`,
`junction_autodiff.py`) are first-order, tape-free versions validated against the
functional paths. They read saved factors that autograd cannot differentiate, so they call
`_linalg.refuse_double_backward` and raise on `create_graph=True` (`once_differentiable`
alone still lets `torch.autograd.functional.hessian` return zeros). Add one only after the
functional version is tested.

`gradcheck` compares a gradient with finite differences of the same function, so it cannot
detect a wrong forward. The dense oracle checks the forward; `gradcheck` checks the
backward. Both are required.

## Conventions

- Blocks are `[..., b, b]` and batched over leading dimensions; the structural axis is
  processed sequentially (or by elimination level).
- Off-diagonal storage is lower-triangular, `i > j`, each block once: `edge_val[k] =
  A[i, j]` and `A[j, i] = A[i, j]ᵀ`. Filled-pattern outputs use the same orientation. Tree
  storage is `edge[v] = A[parent(v), v]`.
- Structural checks run on every call, whatever `check` says: integer indices and parents,
  in-range `i > j` edges without duplicates (so both orientations of an edge are refused),
  one rooted tree, and block shapes and counts (`layout._check_junction`,
  `layout._check_tree`). `check=True` adds the value checks: finite blocks, each diagonal
  block symmetric relative to its own entries, a zero root slot.
- Schur-updated SPD pivots are symmetrized before their Cholesky, and selected diagonal
  outputs are symmetrized (`derivations.md` §6).
- SPD failures raise `torch.linalg.LinAlgError` naming the failing node. The
  non-symmetric eliminations also raise when their local element growth exceeds
  `eps^{-1/2}` (solves) or `eps^{-1/4}` (selected inverses); `derivations.md` §10.4.
  Nothing continues silently on bad input.
- Right-hand sides are `[..., n, b]` or `[..., n, b, k]`; an ambiguous shape is read as
  vectors only when it equals `diag.shape[:-1]`.
- Tolerances on uncontrolled inputs scale with the condition number, `C * kappa * eps`;
  absolute tolerances are used only on controlled topologies. bf16 and fp16 are storage
  formats: they need `compute_dtype=torch.float32`, and without it the CPU backend raises
  because LAPACK has no half-precision Cholesky or inverse.
- Values stay on the device of the input tensors; topology arguments (`edge_index`,
  `parent`, orders) live on the CPU. CI is CPU-only; `tests/test_cuda.py` compares the main
  entry points on a CUDA device with the CPU results and is skipped without one.

## The test bar

A kernel is tested when it has: agreement with `torch.linalg.inv` of the dense matrix;
on-pattern residual identities; symmetry and definiteness of outputs where they apply;
reductions to the other kernels on shared special cases (chain as a path tree, star as a
depth-one tree, junction on a tree or path); for differentiable kernels `gradcheck` in
fp64 and agreement of the analytic backward with a dense adjoint for general cotangents;
second derivatives (`gradgradcheck` or a dense Hessian-vector product for functional paths,
a refusal for analytic ones);
edge cases (`n = 1`, `b = 1`, explicit orders, empty or disconnected patterns where
meaningful); leading batch dimensions; and fp32 and bf16 storage.
