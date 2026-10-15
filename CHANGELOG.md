# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 0.3.0 were internal development milestones with no published
release.

## [Unreleased]

## [0.4.1] - 2026-10-15

A corrective release. An independent review of 0.4.0 found several cases where the package
returned wrong numbers without raising, although every first-order test passed. Each case
below now returns a result checked against a dense reference or raises a clear error.

### Upgrading from 0.4.0

Code written for 0.4.0 can behave differently in these cases.

- Edge lists are checked on every call, not only with `check=True`: integer indices in
  range, `i > j`, each block once. Unchecked 0.4.0 calls summed a list that held both
  orientations of an edge (the PyTorch Geometric convention) or a repeated block into a
  different matrix, and accepted an `i < j` listing that `check=True` rejected. Convert a
  both-direction list with `keep = edge_index[0] > edge_index[1]; edge_index, edge_val =
  edge_index[:, keep], edge_val[keep]` (the non-symmetric kernels take the dropped `(j, i)`
  blocks as `edge_upper`), and an upper-triangle list with `edge_index.flip(0)` and
  `edge_val.mT`.
- Tree parent arrays must hold integers and have one entry per node. Floating-point parents
  raise `TypeError` (0.4.0 truncated them, even with `check=True`), and a parent array
  shorter than `diag` raises instead of leaving uninitialized output blocks. Block tensors
  with the wrong number of blocks raise `ValueError` instead of being ignored or broadcast.
- The non-symmetric kernels refuse more eliminations. The growth guard now measures growth
  locally, and selected inverses use a stricter limit than solves: `eps^{-1/4}` (about `54`
  in float32 and `8.2e3` in float64) instead of `eps^{-1/2}`. Float32 problems that are not
  diagonally dominant can now raise where 0.4.0 returned blocks; `compute_dtype=torch.float64`
  or a better elimination order usually avoids it.
- `selinv_tree`, `selinv_junction_analytic` and `selinv_nonsym_junction_analytic` raise
  `RuntimeError` when asked for a differentiable gradient (`create_graph=True`: double
  backward, Hessian-vector products, `torch.autograd.functional.hessian`). 0.4.0 returned
  Hessian-vector products off by about 60%. `selected_inverse_tree`, `selinv_junction` and
  `selinv_nonsym_junction` give correct second derivatives.
- `junction_posterior_marginal_variances` returns per-coordinate variances of shape
  `[..., n, b]` for block nodes (`b > 1`); 0.4.0 returned the sum of the `b` variances of
  each node. Scalar nodes still give `[..., n]`.
- `check=True` and `validate()` judge each diagonal block's symmetry against its own
  entries. 0.4.0 compared against `max(1, largest entry anywhere)`, so an asymmetric block
  of small entries, or one next to a much larger block, passed.
- Failing SPD pivots are named by their 0-based node index in every path; the chain counted
  from 1 and the batched paths named only the elimination level.

### Fixed

- Non-symmetric growth guard. 0.4.0 divided the growth by the largest input entry of the
  whole matrix, so one large block anywhere, even a disconnected one, switched the guard off:
  `[[1e-18, 1], [2, 1]]` plus an isolated node `1e12` gave `G_00 = 128` instead of `-0.5`
  from `selected_inverse_nonsym_junction(check=True)`, `selinv_nonsym_junction`,
  `selinv_nonsym_junction_analytic`, `nonsym_junction_solve` and
  `selected_inverse_nonsym_tree`, and in float32 an isolated node of `1e5` was enough. The
  guard now compares each reduced block and each factor product `||L|| ||D|| ||U||` with the
  largest input entries of its own block row and column. Selected-inverse errors grow like
  `u * growth^2` rather than `u * growth`: on a matrix with condition number 5.6 that 0.4.0
  accepted at growth `1e7`, the selected blocks were wrong by `2.5e-3` while a solve was
  accurate to `1e-9`, hence the separate limits. The guard is a heuristic;
  `docs/derivations.md` §10.4 states what it does and does not guarantee.
- Second derivatives through the analytic backwards were silently wrong (see above).
  `once_differentiable` alone would still let `torch.autograd.functional.hessian` return
  zeros, so these backwards refuse every `create_graph=True` request.
- Edge lists, parents and block counts were validated only with `check=True`, or not at all
  (see above). Negative or out-of-range indices and empty inputs now raise `ValueError` with
  a message instead of a `KeyError` or `RuntimeError` from inside the kernel.
- `selected_inverse_tree` with the default `batched=False` wrote into preallocated buffers
  in place, so `.backward()` raised although the README documented every operation as
  differentiable. It now gives exact first and second derivatives. Its forward time is
  unchanged; its peak memory is higher (about 136 MiB instead of 55 MiB above the inputs
  for `n = 20000`, `b = 8`), and `batched=True`, unchanged, remains the route for large
  trees.
- `junction_posterior_marginal_variances` returned the wrong quantity for `b > 1` (see
  above).
- `grid_gmrf_precision` built its degree vector on the CPU whatever the device of `kappa`
  and `a`; its values now follow the device of the hyperparameters. The tree and grid fit
  helpers create their parameters on the device of `y`. GPU use is still untested.
- `tree_logdet` reported a non-SPD pivot through a bare backend error; it now names the
  node like the other kernels.

### Documentation

- `derivations.md` §6 quotes Higham's Cholesky bound correctly, `|ΔD| <= gamma_{b+1}
  |R̂^T||R̂|` (Theorem 10.3; it said `(b+1) gamma_{b+1}`), and gives the normwise constants,
  which grow like `b^2`, `b^{3/2}` and `b^2` for Cholesky, triangular solves and products
  rather than like `b`. A computed triangular inverse has a columnwise backward error, not
  one nearby matrix. §8.4 no longer calls the reverse schedule self-adjoint, and the §8
  arguments are labelled as derivations checked against autograd. §10.4 describes the new
  guard and a randomized check of it.
- `bench/stability.py` no longer asserts the retracted height- and condition-independent
  backward-error theorem in its docstring and command-line help.
- The cost formulas state that `w_v` counts later neighbours in the filled graph, and the
  README and `ARCHITECTURE.md` note the cost of the default minimum-degree ordering.
- The README states that the condition number alone does not control the non-symmetric
  kernels, that float16 and bfloat16 inputs need `compute_dtype`, that GPU use is untested,
  and which gradient route to use when, with labelled CPU timings. References to removed
  research demonstrations are gone from the package and documentation.

### Added

- `tests/test_cuda.py`, a CUDA smoke test that compares the main entry points with the CPU
  results; it is skipped without a CUDA device and has not been run on GPU hardware for
  this release.
- Regression tests for each defect above.

## [0.4.0] - 2026-10-14

### Upgrading from 0.3.4

Code written for 0.3.4 can behave differently in these cases.

- The non-symmetric kernels raise `torch.linalg.LinAlgError` instead of returning blocks
  when element growth exceeds `1 / sqrt(eps)`: about `2.9e3` in float32 and `6.7e7` in
  float64. Float32 problems with moderate growth can now raise; eliminating in float64
  avoids that.
- `check=True` rejects infinite or NaN blocks with `ValueError` before factorizing, where
  0.3.4 could fail later with `torch.linalg.LinAlgError`, and floating-point edge indices
  with `TypeError`, where 0.3.4 truncated them.
- When `n == b`, a right-hand side ending in `[n, n, n]` is read as columns unless it
  equals `diag.shape[:-1]`; 0.3.4 read it as a batch of vectors. Passing
  `rhs.unsqueeze(-1)` and squeezing the result keeps the 0.3.4 reading.
- The Gaussian marginal likelihoods and their `sigma2` gradients change only where 0.3.4
  lost accuracy at small observation noise. At `sigma2 >= 1e-2` on unit-scale test
  problems they agree with 0.3.4 to about `1e-12`.
- `gabp_sparse_inv.demos` and `gabp_sparse_inv.bench.matched_compute` are removed.

The rest of the 0.3.4 API keeps its signatures; `junction_marginal_log_likelihood` and
`junction_posterior_marginal_variances` gain an optional `analysis` argument.

### Added

- `analyze_junction`, `JunctionAnalysis` and `SparseGaussianFactor`. Symbolic analysis
  of an SPD pattern runs once and is reused across numeric factorizations; one numeric
  factor serves `logdet`, `solve`, `selected_inverse`, `marginal_diag` and `sample`,
  differentiably, without refactorizing.
- CI jobs that run the suite with the minimum supported PyTorch and against the
  installed wheel, and a test that runs the README examples in a fresh interpreter.

### Fixed

- `junction_solve`, `nonsym_junction_solve` and `SparseGaussianFactor.solve` read an
  unbatched right-hand side of shape `[n, b, k]` with `n == b == k` as a batch of
  vectors and returned the solution of different systems. A shape that fits both
  layouts is now read as vectors only when it equals `diag.shape[:-1]`, as in
  `torch.linalg.solve`; `rhs.unsqueeze(-1)` selects the column layout.
- `marginal_log_likelihood` and `junction_marginal_log_likelihood` subtracted two terms
  of order `1 / sigma2`. At small observation noise this lost the value and the
  derivative with respect to `sigma2` (on a 3 x 3 grid the derivative was already off by
  a relative `3e-6` at `sigma2 = 1e-6`, and every digit was lost at `1e-12`). Both now
  factor `I + sigma2 Q` and use the posterior energy `mu' Q mu + sigma2 ||Q mu||^2`, and
  agree with dense references to about `1e-15` down to `sigma2 = 1e-18`. The tape-free
  gradient of `fit_grid_marginal_likelihood_analytic` uses the same form.
- The non-symmetric kernels eliminate in a fixed order without pivoting and returned
  wrong blocks when elimination amplified entries, even for well-conditioned matrices
  such as `[[1e-18, 1], [2, 1]]`. `selected_inverse_nonsym_junction`,
  `selinv_nonsym_junction`, `selinv_nonsym_junction_analytic`, `nonsym_junction_solve`
  and `selected_inverse_nonsym_tree` now raise `torch.linalg.LinAlgError` when element
  growth exceeds `1 / sqrt(eps)` of the working dtype.
- `check=True` rejects infinite and NaN blocks. Edge indices must be integers; floating
  indices were truncated toward zero.
- Importing `gabp_sparse_inv.bench` no longer requires setuptools. Under Python 3.12,
  torch 2.2 wheels need it for `torch.utils.benchmark` without declaring it, so the
  benchmark harness imports the timer only when it times something.
- Documentation: the stored off-diagonal cotangent in the derivations is split between the
  two symmetric slots (it was counted twice); the "log-det check" of the tree backward
  differentiates the trace of the inverse; a missing transpose in the bidiagonal adjoint;
  row and column slips in the tree proof; bounded front width is sufficient, not
  necessary, for linear work; and the derivative schedule is attributed to the prior work
  it reproduces. The README example that mixed float32 and float64 is fixed.
- `CITATION.cff` no longer names an unpublished article as the preferred citation.

### Removed

- The research demonstrations (`gabp_sparse_inv.demos`), the research benchmarks
  `gabp_sparse_inv.bench.matched_compute` and `gabp_sparse_inv.bench.deq_gmres_falsifier`,
  and the manuscript sources, figure data and archived experiment records. They served a
  research study rather than the library. They remain at the `research-archive` tag, and
  the demonstrations and `matched_compute` also ship in 0.3.4.

## [0.3.4] - 2026-07-28

### Fixed

- Seed the routed-field baseline models before construction, so a requested seed
  controls both data generation and model initialization. Evaluation now
  explicitly uses inference mode.
- Correct the public interpretation of the routed-field and DEQ diagnostics:
  cross-architecture parameter count is not a capacity match, `K` Jacobi
  sweeps from zero reach at most `K - 1` edges, the diagonal lift guarantees
  positive definiteness but does not fix conditioning, and the finite Neumann
  probe is not a comparison with accelerated or preconditioned solvers.
- Recompute the tracked maze diagnostics from the corrected seeded harness and
  add a claim verifier that checks manuscript numbers against committed records.
  Numerical library kernels and the public API are unchanged.

## [0.3.3] - 2026-07-23

### Changed

- Scientific documentation correction across `README.md`, `docs/derivations.md`,
  `docs/ARCHITECTURE.md`, `docs/DEQ.md`, `docs/MAZE.md`, `docs/DELTANET.md`, the JOSS paper,
  and package/module docstrings. Library numerics are unchanged; only prose and docstrings
  were corrected.
  - Complexity claims now separate numeric storage (proportional to the factor pattern) from
    arithmetic cost (proportional to squared front/clique sizes); the two coincide for bounded
    treewidth (including trees) but not in general.
  - The floating-point claim is bounded to the measured fp64 result instead of a general
    condition-independent stability statement.
  - The non-symmetric derivative identity is stated as the adjoint operator
    `L_A^*(X) = -A^{-T} X A^{-T}`, not identified with the selected inverse at `A^T`; wording
    changed from "self-adjoint backward" to "transpose-form backward".
  - Symmetric kernels are documented as requiring SPD pivots; singular PSD pseudoinverse
    handling is explicitly out of scope.
  - The maze and DEQ demonstrations are described as routed-field/solve-shaped proxies and a
    finite mechanism diagnostic, not a maze-solving benchmark or a sequence-model performance
    result.
  - The JOSS paper's statement of need adds the nearest 2025-2026 systems (Serinv, sTiles,
    Schwarz-Schur Involution, TMB, and Siden et al.) and no longer claims to be the only
    differentiable selected-inverse library or an unqualified linear-cost operator.
- Republished the corrected attainability manuscript and figures under `paper/attainability/`.
- Packaging metadata modernized to PEP 639 (`license = "MIT"` / `license-files`); the deprecated
  MIT license classifier was removed.

## [0.3.2] - 2026-07-02

### Fixed

- PyPI README links now use absolute GitHub URLs, so documentation, citation,
  contribution, and JOSS-paper links resolve from the PyPI project page. No
  library code changed.

## [0.3.1] - 2026-07-02

### Fixed

- PyPI packaging: the 0.3.0 sdist and wheel were built from the internal
  research tree and included internal experiment and benchmark modules that
  are not part of the public API and are not in the public source tree.
  0.3.1 is built from the public tree; the library code is unchanged. The
  0.3.0 release has been deleted from PyPI.
- JOSS paper: bibliography entries corrected against primary sources.

## [0.3.0] - 2026-06-27

First release, prepared for submission to the Journal of Open Source Software.

### Added

- Selected-inverse kernels on the input (or filled) pattern for block
  **chain**, **star**, **tree**, and general sparse SPD **junction-tree**
  matrices, plus the structured **non-symmetric** ladder (lower-bidiagonal,
  `selinv_tril` triangular chunk inverse, non-symmetric tree, and the general
  non-symmetric / Erisman-Tinney junction inverse and solve).
- **Differentiable** forms (`selinv_*`) with autograd plus, for the
  filled-pattern kernels, hand-written tape-free analytic backwards validated
  against autograd to machine precision.
- Statistical operators from the same factorization family: `junction_logdet` /
  `tree_logdet`, Gaussian sampling (`sample_gaussian_tree` /
  `sample_gaussian_junction`), and sparse solves (`junction_solve`,
  `nonsym_junction_solve` with the transpose / implicit-diff adjoint). Public
  function calls factor independently.
- Tree- and grid/loopy-GMRF learning applications, seeded SPD/non-symmetric
  generators, elimination-ordering helpers, and reproducible benchmark scripts.
- Demonstrations: tree and grid mazes, the DEQ fixed-point backward, and the
  DeltaNet chunk-inverse drop-in.
- Packaging and community health for JOSS: `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, `.zenodo.json`, issue/PR templates,
  expanded project URLs, a JOSS paper draft (`paper/joss/`), and CI across
  Ubuntu / Windows on Python 3.12 and 3.13, and macOS on Python 3.12.

[Unreleased]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vanshsehrawat14/gabp-sparse-inv/releases/tag/v0.3.0
