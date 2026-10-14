# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 0.3.0 were internal development milestones with no published
release.

## [Unreleased]

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

[Unreleased]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vanshsehrawat14/gabp-sparse-inv/releases/tag/v0.3.0
