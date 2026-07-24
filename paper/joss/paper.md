---
title: 'gabp-sparse-inv: Differentiable selected inversion for sparse block matrices in PyTorch'
tags:
  - Python
  - PyTorch
  - sparse linear algebra
  - selected inversion
  - Gaussian belief propagation
  - automatic differentiation
authors:
  - name: Vanshdeep Sehrawat
    orcid: 0009-0005-8062-1337
    affiliation: 1
affiliations:
  - name: University of Nevada, Las Vegas, USA
    index: 1
date: 26 June 2026
bibliography: paper.bib
---

# Summary

The *selected inverse* of a sparse matrix $A$ is the set of entries of $A^{-1}$ that lie on
$A$'s own sparsity pattern. When the block graph of $A$ has low treewidth, these entries are
computable in $\mathcal{O}(n)$ time (for trees) to $\mathcal{O}(n^{1.5})$ (for two-dimensional
grids) by the Takahashi recurrence [@takahashi1973; @erismantinney1975], which on a tree is
exactly Gaussian belief propagation [@weiss2001correctness] without ever forming the dense
inverse. `gabp-sparse-inv` provides this computation as a set of reliable, hot-swappable PyTorch
operators for both symmetric positive-definite (SPD) and structured non-symmetric matrices, and,
unlike existing selected-inversion libraries, makes every operator **differentiable**: gradients
of any loss over the on-pattern inverse blocks flow back to the matrix entries at the same
asymptotic cost as the forward pass [@baur1983complexity; @zhu2019sparse].

The package covers a ladder of structures behind one interface: block **chains**
(the Rauch-Tung-Striebel / Kalman-smoother form), **stars** (arrowheads), general **trees**,
**junction trees** for arbitrary sparse SPD matrices (via a symbolic min-degree elimination to the
filled pattern), and the general **non-symmetric** (LU / Erisman-Tinney [@erismantinney1975])
selected inverse and its solve. The companion statistical operations derived from the same
factorization machinery, the **log-determinant**, **Gaussian sampling**
$x \sim \mathcal{N}(0, A^{-1})$, and the sparse **linear solve** $x = A^{-1}b$, are included as
separate function APIs; the current release does not expose a persistent factor object for
cross-call reuse. Every numerical kernel is validated against a
dense $\mathcal{O}(n^3)$ oracle, and every differentiable kernel additionally with
`torch.autograd.gradcheck`.

# Statement of need

Matrix inversion is increasingly a primitive *inside* machine-learning models, not just a tool used
to build them: deep equilibrium models [@bai2019deep], linear attention with the delta rule
[@yang2024parallelizing], and other implicit / fixed-point layers all place a structured linear
solve in the forward pass and its gradient in the backward pass. For low-treewidth structure, the
*selected* inverse costs far less than the general $\mathcal{O}(n^3)$ inverse, but the surrounding
software landscape is split by purpose rather than absent. Recent accelerator systems target raw
throughput: `Serinv` for block-tridiagonal arrowhead matrices [@maillou2025serinv], `sTiles` for
tile-parallel selected inversion of structured matrices [@fattah2025stiles], and Schwarz--Schur
Involution for batched differentiable sparse subdomain solves on GPUs [@wang2025schwarz]; none is a
general-purpose autodiff-framework operator. `torch-sla` [@torchsla2026] ships a differentiable
sparse *solve* with a constant-memory adjoint inside PyTorch, but no selected inverse,
log-determinant gradient, marginal variances, or Gaussian sampling. TMB [@kristensen2016tmb] has
long combined sparse latent-Gaussian calculation, automatic differentiation, and the Laplace
approximation at scale, and approximate selected-covariance methods with estimable confidence
bounds exist for the large-sparse-precision setting [@siden2017covariance]; neither is a drop-in
PyTorch tensor operator.

Mature selected-inversion codes (SelInv [@lin2011selinv], PEXSI [@lin2013pexsi]) target
high-performance computing in compiled languages and are not written against an autodiff framework
or designed to be dropped into a neural network's forward pass. `gabp-sparse-inv` fills the
correctness-first, PyTorch-native niche in this landscape: drop-in tensor operators that return
exact on-pattern inverse blocks (or log-dets, samples, solves) and exact gradients as separate
function calls (the current release does not expose a persistent cross-call factor object), so a
practitioner can swap a symmetric inverse for a non-symmetric one, or a dense solve for a
structured selected inverse, without leaving the autodiff graph. The intended users are researchers
building differentiable models over graph-structured Gaussian fields, deep-equilibrium and
fixed-point layers, and structured-attention mechanisms, as well as anyone needing differentiable
marginal variances or log-determinants of a sparse Gaussian Markov random field at low treewidth.

The library is deliberately scoped as a correctness-first reference and differentiable
implementation, not a performance competitor to the compiled HPC or GPU-accelerator systems above:
its niche is the differentiable thread and the uniform symmetric / non-symmetric interface, not
out-engineering mature direct solvers on raw throughput.

# Functionality

- Selected inverse on the input (or filled) pattern: `selected_inverse_chain`,
  `selected_inverse_star`, `selected_inverse_tree`, `selected_inverse_junction`, and the
  non-symmetric `selected_inverse_bidiag`, `selinv_tril`, `selected_inverse_nonsym_tree`,
  and `selected_inverse_nonsym_junction`.
- Differentiable forms (`selinv_*`) with autograd support and, for the filled-pattern
  kernels, hand-written tape-free analytic backwards validated against autograd to machine
  precision.
- Solves and statistics: `junction_solve`, `nonsym_junction_solve` (with the transpose /
  implicit-differentiation adjoint), `junction_logdet` and `tree_logdet`, and
  `sample_gaussian_tree` / `sample_gaussian_junction`.
- Generators and tooling: seeded SPD and non-symmetric generators with a condition-number
  knob, a graph-Laplacian generator for the well-scaled high-condition regime,
  elimination-ordering helpers (min-degree, nested dissection), and reproducible benchmark
  scripts.
- Precision: native fp64/fp32 and a low-precision storage with higher-precision compute
  path. A characterization study shows no accuracy penalty versus a dense solve at equal
  precision, and no robust accuracy advantage either; the value is cost and
  differentiability, not precision.

# Acknowledgements

The author thanks Isaac Liao (Carnegie Mellon University) for scoping the selected-inverse operator and the
maze demonstration, and for feedback on the target regime.

# AI usage disclosure

I used AI assistance during software development and manuscript preparation, including coding
support, editing, and artifact checks. I reviewed and validated the assisted work and remain
responsible for the software, tests, data, citations, and paper.

# References
