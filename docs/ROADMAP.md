# Roadmap

`PROJECT_STATUS.md` is the authoritative record of what is implemented and tested. This file
records the public forward direction for the package.

## Goal

The package aims to make selected inversion a reliable PyTorch primitive for sparse
block-structured matrices:

- exact selected inverse blocks on the input or filled factor pattern,
- exact first-order gradients at the same asymptotic cost as the forward pass,
- sparse solves, log-determinants, and Gaussian sampling from the same factorizations,
- symmetric and non-symmetric APIs that can be swapped without leaving PyTorch autograd.

The mathematical engine is classical selected inversion, Gaussian belief propagation, and the
Takahashi / Erisman-Tinney recurrences. The package contribution is the tested, differentiable,
uniform PyTorch implementation.

## Current Milestone: Software Release

Near-term release work is mechanical and packaging-focused:

1. Publish the `0.3.1` package to PyPI (0.3.0 was built from the internal tree and was deleted from PyPI).
2. Submit the JOSS software paper.
3. Archive the release snapshot and add the archive DOI when available.
4. Keep the public documentation limited to implemented, verified behavior.

No GPU timing claim is required for the software paper. The current kernels are reference-quality
and correctness-focused.

## Engineering Work Ahead

### Performance

The level-set `batched=True` paths are designed to reduce Python-loop overhead and amortize GPU
launch cost. The next performance work is to measure them on CUDA and report:

- junction forward and backward wall time,
- fill count and memory use,
- dense crossover diagnostics,
- sensitivity to graph shape, block size, and batch shape.

These numbers should be recorded as hardware-dependent diagnostics unless they are turned into
stable tests.

### Packaging

The package metadata is in place. Remaining packaging tasks:

- publish to TestPyPI or PyPI,
- verify install from the published artifact in a clean environment,
- tag the release,
- create an archival DOI for the tagged release.

### Documentation

Documentation should stay concise and factual:

- describe implemented APIs and tested behavior,
- keep limitations explicit,
- avoid claims based only on local benchmark artifacts,
- update `PROJECT_STATUS.md` whenever scope or verification changes.

## Research And Application Directions

The implemented primitive enables several application directions, all requiring separate
experiments before they should be claimed as results:

- graph-structured fixed-point / DEQ layers where the backward pass is a sparse non-symmetric
  solve,
- exact marginal variances and log-determinants for GMRF / SPDE-style models,
- structured covariance and uncertainty estimates in low-treewidth Gaussian models,
- size-extrapolation tasks where a scale-free exact solve is an intended inductive bias,
- sequence or attention mechanisms whose local chunk update contains a triangular inverse.

The repository already includes small, controlled demonstrations for mazes, DEQ-style fixed
points, DeltaNet-style chunks, and tree/grid GMRFs. They are mechanism checks, not broad model
benchmarks.

## Out Of Scope For Now

The following are intentionally not on the immediate roadmap:

- pivoted non-symmetric selected inversion,
- iterative or approximate loopy GaBP,
- CHOLMOD / SelInv / PEXSI CPU-performance competition,
- indefinite or complex Hermitian factorizations,
- heterogeneous block sizes,
- arbitrary extra off-pattern inverse entries without a downstream consumer.

These are valid future projects, but they should be added only when a concrete use case and
verification plan exist.
