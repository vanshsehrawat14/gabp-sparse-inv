# Roadmap

`PROJECT_STATUS.md` records the verified public surface. This file records the
public package roadmap.

**Last rewritten:** 2026-07-23 PDT

## Product principle

Maintain a correctness-first PyTorch reference for selected inverse blocks,
sparse solves, log determinants, selected marginals, and Gaussian sampling on
the documented matrix families. The underlying recurrences are classical.
Performance work requires a measured workload and comparison with mature
sparse-direct software.

## Release 0.3.3

Version 0.3.3 is a corrective release. Its scope is:

1. separate factor-pattern storage from clique/front arithmetic in every
   public complexity statement;
2. bound floating-point claims to the recorded finite diagnostics;
3. correct the non-symmetric inversion-adjoint identity;
4. state the SPD and static no-pivot assumptions explicitly;
5. describe the routed-field, DEQ, and DeltaNet examples as mechanism or
   identity demonstrations rather than performance results;
6. publish the corrected Paper 1 source and tracked evidence references; and
7. modernize packaging metadata to PEP 639.

The public numerical kernels are unchanged in this release.

## Near-term engineering

- Keep dense-oracle, residual, gradient, batching, and cross-structure tests
  green on the supported Python and operating-system matrix.
- Reduce Python overhead only after profiling identifies a consumer-relevant
  bottleneck.
- Evaluate external ordering or compiled sparse-direct backends when a
  concrete workload supplies graph family, block size, batch shape, dtype,
  conditioning, and requested outputs.
- Keep first-order custom backward paths and higher-order functional paths
  clearly separated in the API documentation.
- Add public APIs only with dense-oracle tests, documented scope, packaging
  coverage, and a migration note.

## Mathematical and numerical questions

- Structured backward-error bounds with explicit dependence on scaling,
  depth, degree, front size, conditioning, and factor growth.
- Memory-efficient checked schedules that avoid materializing all clique-pair
  metadata.
- Pivoted or dynamic-pattern differentiation for non-normal asymmetric
  problems.
- Comparison-matrix or pseudospectral conditions that predict no-pivot
  reliability.
- Measured fill, work, and memory curves across graph families and at least
  three problem sizes.

## Deliberately outside the current roadmap

- singular PSD pseudoinverse or gauge handling;
- arbitrary indefinite or complex-Hermitian support;
- a general C++/CUDA rewrite without a measured consumer;
- production-performance claims from a single device or problem size;
- approximate loopy GaBP under the exact junction API; and
- arbitrary off-pattern inverse entries.
