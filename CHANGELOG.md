# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 0.3.0 were internal development milestones with no published
release.

## [Unreleased]

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
- Statistical operators from the shared factorization: `junction_logdet` /
  `tree_logdet`, Gaussian sampling (`sample_gaussian_tree` /
  `sample_gaussian_junction`), and sparse solves (`junction_solve`,
  `nonsym_junction_solve` with the transpose / implicit-diff adjoint).
- Tree- and grid/loopy-GMRF learning applications, seeded SPD/non-symmetric
  generators, elimination-ordering helpers, and reproducible benchmark scripts.
- Demonstrations: tree and grid mazes, the DEQ fixed-point backward, and the
  DeltaNet chunk-inverse drop-in.
- Packaging and community health for JOSS: `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, `.zenodo.json`, issue/PR templates,
  expanded project URLs, a JOSS paper draft (`paper/joss/`), and CI across
  Ubuntu / Windows on Python 3.12 and 3.13, and macOS on Python 3.12.

[Unreleased]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/vanshsehrawat14/gabp-sparse-inv/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vanshsehrawat14/gabp-sparse-inv/releases/tag/v0.3.0
