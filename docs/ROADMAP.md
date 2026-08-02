# Roadmap

`PROJECT_STATUS.md` records the verified public surface.

**Last rewritten:** 2026-07-30 PDT

The package is feature-complete for its documented 0.3.4 scope. Current work
is distribution, correctness maintenance, and reproducibility:

1. keep the supported Python/OS test matrix and claim verifier green;
2. correct reproducible bugs and inaccurate public documentation;
3. profile or add a backend only for a concrete consumer workload; and
4. add an API only with dense-oracle tests, explicit numerical scope,
   packaging coverage, and a migration note.

There is no active public feature campaign. In particular, the roadmap does
not promise singular PSD, pivoted or dynamic-pattern LU, indefinite or
complex-Hermitian support, approximate loopy GaBP, arbitrary off-pattern
entries, or a general C++/CUDA rewrite. Those require a concrete user,
independent novelty review, and measured target-hardware cost.
