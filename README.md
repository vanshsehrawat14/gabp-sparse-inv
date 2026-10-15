# gabp-sparse-inv

Differentiable selected inversion for sparse block matrices in PyTorch.

For a sparse symmetric positive-definite block matrix `A`, the package computes the
blocks of `A⁻¹` that lie on the sparsity pattern of `A`, or on its filled (chordal)
pattern, without forming the dense inverse. In a Gaussian Markov random field with
precision `A`, these are the marginal covariances of every node and the covariances of
neighbouring nodes. Every kernel gives first-order gradients with respect to the input
blocks, and the functional kernels also give second derivatives. Linear solves,
log-determinants and Gaussian samples come from the same block factorization, and a
non-symmetric variant covers structurally symmetric patterns eliminated in a fixed order.

| matrix family | operations |
|---|---|
| SPD block chain, star, tree | selected inverse; analytic first-order tree backward |
| SPD general sparse pattern | selected inverse on the filled pattern; solve; log-determinant; Gaussian sampling; a reusable factor shared by all of them; functional and analytic first-order backward |
| non-symmetric, fixed elimination order | lower-bidiagonal, tree, and filled-pattern selected inverse; solve and transpose solve; the dense unit-lower-triangular inverse `(I - A)⁻¹`; functional and analytic first-order backward |
| Gaussian models | tree and grid GMRF marginal likelihood and posterior marginal variances |

## Relation to other tools

The algorithms are classical: block elimination followed by the Takahashi and
Erisman–Tinney recurrences for the selected inverse, and reverse-mode differentiation of
those recurrences. For SPD matrices the derivative of the selected inverse is, up to sign,
the Hessian of the sparse log-determinant barrier, for which Andersen, Dahl and
Vandenberghe (2013) give factorization-cost algorithms implemented in
[CHOMPACK](https://chompack.readthedocs.io). This package implements them for PyTorch:
tensors and autograd, leading batch dimensions, dense `b × b` blocks, analytic backwards
that keep no autograd tape, and tests against dense references.

It is a reference implementation written with per-block Python loops, so on a single
large matrix it is much slower than compiled sparse-direct libraries such as CHOLMOD,
CHOMPACK or PEXSI. Use those when you need speed outside autograd. Use this package when
you need selected inverse entries and their gradients inside a PyTorch computation;
[Choosing a route](#choosing-a-route) gives measured sizes. Leading batch dimensions let
many matrices with one sparsity pattern share each block operation.

## Install

```bash
pip install gabp-sparse-inv
```

0.4.1 corrects several cases where 0.4.0 returned wrong numbers without raising; the
[changelog](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/CHANGELOG.md#upgrading-from-040)
lists what can behave differently.

## Examples

The blocks of the inverse of a block-tridiagonal SPD matrix:

```python
import torch
from gabp_sparse_inv import random_spd_chain, selected_inverse_chain

A = random_spd_chain(num_blocks=8, block_size=4, seed=0, diag_load=2.0)
G_diag, G_lower = selected_inverse_chain(A.diag, A.lower, check=True)

dense_inverse = torch.linalg.inv(A.to_dense())
torch.testing.assert_close(G_diag[0], dense_inverse[:4, :4])
```

A general sparse pattern takes node blocks plus a lower-triangular edge list that names
each off-diagonal block once, as `(i, j)` with `i > j`. The output follows the filled
pattern, and gradients reach every stored input block. An explicit elimination order
skips the default minimum-degree search, and `batched=True` runs one batched block
operation per elimination level:

```python
from gabp_sparse_inv import (
    elimination_order_nested_dissection,
    grid_edges,
    random_spd_graph,
    selinv_junction,
)

A = random_spd_graph(12, grid_edges(3, 4), 2, seed=0)
order = elimination_order_nested_dissection(3, 4)
diag = A.diag.clone().requires_grad_(True)
edge_val = A.edge_val.clone().requires_grad_(True)

G_diag, filled_edges, G_lower = selinv_junction(diag, A.edge_index, edge_val, order, batched=True)
total_variance = G_diag.diagonal(dim1=-2, dim2=-1).sum()
total_variance.backward()          # fills diag.grad and edge_val.grad
```

An edge list that holds both directions of every edge, as in PyTorch Geometric, raises
`ValueError`; keep one direction:

```python
both = torch.cat([A.edge_index, A.edge_index.flip(0)], dim=1)
both_val = torch.cat([A.edge_val, A.edge_val.mT])      # A[j, i] = A[i, j]ᵀ
keep = both[0] > both[1]
G_diag, _, _ = selinv_junction(A.diag, both[:, keep], both_val[keep], order)
```

Symbolic analysis runs once per pattern and is reused when the values change. One
numeric factor serves several queries, differentiably:

```python
from gabp_sparse_inv import analyze_junction

analysis = analyze_junction(A.num_nodes, A.edge_index, order)
factor = analysis.factor(A.diag, A.edge_val, check=True, batched=True)

logdet = factor.logdet()
x = factor.solve(torch.ones(A.num_nodes, 2, dtype=A.diag.dtype))
variances = factor.marginal_diag()
samples = factor.sample(4, generator=torch.Generator().manual_seed(0))

for shift in (0.0, 0.5):           # new values, same pattern: no new symbolic work
    d = (A.diag + shift * torch.eye(2, dtype=A.diag.dtype)).requires_grad_(True)
    G_diag = analysis.factor(d, A.edge_val, batched=True).selected_inverse()[0]
    G_diag.diagonal(dim1=-2, dim2=-1).sum().backward()
```

The marginal likelihood of a grid GMRF observed with Gaussian noise, differentiated with
respect to its hyperparameters:

```python
from gabp_sparse_inv import grid_gmrf_precision, junction_marginal_log_likelihood

kappa = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
sigma2 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)    # noise variance
y = torch.randn(16, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

diag, edge_index, edge = grid_gmrf_precision(4, 4, kappa, a)
log_evidence = junction_marginal_log_likelihood(diag, edge_index, edge, y, sigma2)
log_evidence.backward()            # fills kappa.grad, a.grad and sigma2.grad
```

## Choosing a route

The SPD selected inverse on a general pattern can be differentiated four ways. All four
give the same values and first-order gradients.

| route | call | derivatives | use it for |
|---|---|---|---|
| reference | `selinv_junction(diag, edge_index, edge_val, order)` | first and second order | checking results on small problems |
| batched | `selinv_junction(..., order, batched=True)` | first and second order | one evaluation on a new pattern |
| reused analysis | `analysis = analyze_junction(n, edge_index, order)` once, then `analysis.factor(diag, edge_val, batched=True)` | first and second order | repeated evaluations on one pattern, such as a fitting loop |
| analytic | `selinv_junction_analytic(diag, edge_index, edge_val, order)` | first order only | a backward that keeps no autograd tape |

Without `order`, each functional call and each `analyze_junction` call first computes a
greedy minimum-degree order in Python with `O(n²)` node scans. On a 10,000-node path that
takes 13.5 s against 0.03 s with an explicit order. For grids,
`elimination_order_nested_dissection(rows, cols)` gives a good order; for other patterns,
compute an order once and reuse it, or reuse the analysis.

Time for the value and gradient of the total marginal variance `Σ tr(G_vv)` on a 2-D grid
with scalar nodes (`b = 1`), nested-dissection order, float64. Measured on one laptop CPU
(AMD Ryzen 7 7730U, 8 cores, 8 PyTorch threads, Windows 11, PyTorch 2.14.1); each entry is
the median of three runs after a warm-up, or a single run where marked with \*. Other
hardware and block sizes will move the crossovers.

| n | reference | analytic | batched | reused analysis | dense Cholesky |
|---|---|---|---|---|---|
| 1024 | 33 s\* | 21 s\* | 0.75 s | 0.26 s | 0.19 s |
| 2304 | 177 s\* | 85 s\* | 2.8 s | 0.53 s | 1.3 s |
| 4096 | not run | not run | 4.4 s | 0.68 s | 5.5 s |

The dense baseline assembles the matrix with one differentiable scatter, factors it with
`torch.linalg.cholesky` and differentiates `cholesky_inverse` with autograd; it needs
`O(n²)` memory. The analytic backward avoids the autograd tape but runs its reverse sweep
as a per-node Python loop, so with small blocks it is slower than the batched route.
The reused-analysis column excludes the symbolic analysis, which is paid once per pattern:
0.03 s, 0.12 s and 0.23 s with the nested-dissection order at these sizes, against 0.14 s,
0.72 s and 1.6 s for the default minimum-degree order.

## Contracts and limits

- The SPD kernels require SPD input. A pivot that is not positive definite raises
  `torch.linalg.LinAlgError` naming the node. Singular PSD, indefinite and
  complex-Hermitian matrices are not supported.
- Every call checks the structure of its input: integer indices in range, `i > j` with each
  block listed once, a parent array that forms one tree, and block shapes and counts.
  `check=True` adds the value checks: finite blocks and symmetric diagonal blocks.
- The non-symmetric kernels eliminate in a fixed order without pivoting, so the condition
  number of `A` alone does not control their accuracy: `[[1e-18, 1], [2, 1]]` has
  condition number 2.6 and loses every digit when node 0 is eliminated first. They measure
  element growth locally, against the largest input entries of each block row and column,
  and raise `torch.linalg.LinAlgError` above `eps^(-1/2)` for solves and `eps^(-1/4)` for
  selected inverses, whose errors grow faster. The test is a heuristic, not a stability
  guarantee: an accepted elimination can lose up to about half the working digits, more
  when `A` is ill conditioned. On refusal, choose an order that eliminates
  well-conditioned pivots first, compute in float64 (`compute_dtype=torch.float64`), or use
  a pivoted LU. The
  [derivations](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/docs/derivations.md#104-the-no-pivot-static-pattern-regime)
  describe the criterion and a randomized check of it.
- In the tested regimes the forward error of the SPD kernels follows `κ(A)·u`; there is no
  proved structured bound for the full algorithm.
- The analytic backwards (`selinv_tree`, `selinv_junction_analytic`,
  `selinv_nonsym_junction_analytic`) are first order and raise when asked for a
  differentiable gradient (`create_graph=True`, as in Hessian-vector products). The
  functional kernels (`selected_inverse_tree`, `selinv_junction`, `selinv_nonsym_junction`,
  the solves, log-determinants and likelihoods) support second derivatives.
- Computation runs in float32 or float64. Float16 and bfloat16 work as storage formats
  with `compute_dtype=torch.float32`; without it the CPU backend raises, because LAPACK has
  no half-precision Cholesky.
- Results stay on the device of the input values; index tensors stay on the CPU. Only CPU
  execution is tested. `tests/test_cuda.py` checks the main entry points on a CUDA device
  against the CPU results, but it has not been run for this release.
- A right-hand side is `[..., n, b]` (vectors) or `[..., n, b, k]` (columns). A shape that
  fits both is read as vectors only when it equals `diag.shape[:-1]`, as in
  `torch.linalg.solve`; `rhs.unsqueeze(-1)` selects columns.
- `junction_posterior_marginal_variances` returns one variance per coordinate: `[..., n]`
  for scalar nodes and `[..., n, b]` for block nodes.
- On trees the cost is `O(n b³)` time and `O(n b²)` storage. On general patterns it depends
  on the elimination order: with `w_v` neighbours of node `v` eliminated after it in the
  filled graph, numeric storage is proportional to `Σ(1 + w_v) b²` and work to
  `Σ(1 + w_v²) b³`.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Each kernel is checked against dense inverses in double precision, on-pattern residual
identities, finite differences, `gradcheck` and `gradgradcheck`, and the other kernels on
shared special cases. CI runs the suite on Linux, Windows and macOS, with the minimum
supported PyTorch, and against the installed wheel. CI has no GPU.

## Documentation

- [Derivations](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/docs/derivations.md):
  the recurrences, derivative conventions, costs, numerical boundaries and prior work.
- [Architecture](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/docs/ARCHITECTURE.md):
  module map and conventions for contributors.
- [Changelog](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/CHANGELOG.md)
  and [contributing guide](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/CONTRIBUTING.md).
- [Research archive](https://github.com/vanshsehrawat14/gabp-sparse-inv/tree/research-archive):
  the research code, experiments and manuscripts that `main` carried before 0.4.0.

## Citation and license

Citation metadata is in
[CITATION.cff](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/CITATION.cff).
If the selected inverse matters to your work, please also cite the algorithms: Takahashi,
Fagan and Chen (1973), Erisman and Tinney (1975), Lin et al. (2011), and Andersen, Dahl and
Vandenberghe (2013). The software is MIT licensed; see
[LICENSE](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.4.1/LICENSE).
