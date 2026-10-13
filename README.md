# gabp-sparse-inv

Differentiable selected inversion for sparse block matrices in PyTorch.

For a sparse symmetric positive-definite block matrix `A`, the package computes the
blocks of `A⁻¹` that lie on the sparsity pattern of `A`, or on its filled (chordal)
pattern, without forming the dense inverse. In a Gaussian Markov random field with
precision `A`, these are the marginal covariances of every node and the covariances of
neighbouring nodes. All operations are differentiable with respect to the input blocks.
Linear solves, log-determinants and Gaussian samples come from the same block
factorization, and a non-symmetric variant covers structurally symmetric patterns
eliminated in a fixed order.

| matrix family | operations |
|---|---|
| SPD block chain, star, tree | selected inverse; analytic first-order tree backward |
| SPD general sparse pattern | selected inverse on the filled pattern; solve; log-determinant; Gaussian sampling; a reusable factor shared by all of them; functional and analytic first-order backward |
| non-symmetric, fixed elimination order | lower-bidiagonal, tree, and filled-pattern selected inverse; solve and transpose solve; dense `(I - A)⁻¹` chunk inverse; functional and analytic first-order backward |
| Gaussian models | tree and grid GMRF marginal likelihood and posterior marginal variances |

## Relation to other tools

The algorithms are classical: block elimination followed by the Takahashi and
Erisman–Tinney recurrences for the selected inverse, and reverse-mode differentiation of
those recurrences. For SPD matrices the derivative of the selected inverse is, up to sign,
the Hessian of the sparse log-determinant barrier, for which Andersen, Dahl and
Vandenberghe (2013) give factorization-cost algorithms implemented in
[CHOMPACK](https://chompack.readthedocs.io). This package implements them for PyTorch:
tensors and autograd, leading batch dimensions, dense `b × b` blocks, CPU or GPU tensors,
analytic backwards that keep no autograd tape, and tests against dense references.

It is a reference implementation written with per-block Python loops, so on a single
large matrix it is much slower than compiled sparse-direct libraries such as CHOLMOD,
CHOMPACK or PEXSI. Use those when you need speed outside autograd. Use this package when
you need selected inverse entries and their gradients inside a PyTorch computation at
moderate size. Leading batch dimensions let many matrices with one sparsity pattern share
each block operation.

## Install

```bash
pip install gabp-sparse-inv
```

PyPI serves 0.3.4. The reusable factor and the fixes listed under *Unreleased* in the
[changelog](CHANGELOG.md) are on `main`, and the changelog lists the behaviour changes
from 0.3.4:

```bash
pip install git+https://github.com/vanshsehrawat14/gabp-sparse-inv
```

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

A general sparse pattern takes node blocks plus a lower-triangular edge list. The output
follows the filled pattern, and gradients reach every stored input block:

```python
from gabp_sparse_inv import grid_edges, random_spd_graph, selinv_junction_analytic

A = random_spd_graph(12, grid_edges(3, 4), 2, seed=0)
diag = A.diag.clone().requires_grad_(True)
edge_val = A.edge_val.clone().requires_grad_(True)

G_diag, filled_edges, G_lower = selinv_junction_analytic(diag, A.edge_index, edge_val)
total_variance = G_diag.diagonal(dim1=-2, dim2=-1).sum()
total_variance.backward()          # fills diag.grad and edge_val.grad
```

One numeric factor serves several queries. Symbolic analysis runs once per pattern and
can be reused when the values change:

```python
from gabp_sparse_inv import analyze_junction

analysis = analyze_junction(A.num_nodes, A.edge_index)
factor = analysis.factor(A.diag, A.edge_val, check=True)

logdet = factor.logdet()
x = factor.solve(torch.ones(A.num_nodes, 2, dtype=A.diag.dtype))
variances = factor.marginal_diag()
samples = factor.sample(4, generator=torch.Generator().manual_seed(0))
```

The marginal likelihood of a grid GMRF observed with Gaussian noise, differentiated with
respect to its hyperparameters:

```python
from gabp_sparse_inv import grid_gmrf_precision, junction_marginal_log_likelihood

kappa = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
noise = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
y = torch.randn(16, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

diag, edge_index, edge = grid_gmrf_precision(4, 4, kappa, a)
log_evidence = junction_marginal_log_likelihood(diag, edge_index, edge, y, noise)
log_evidence.backward()            # fills kappa.grad, a.grad and noise.grad
```

## Contracts and limits

- The SPD kernels require SPD input. A pivot that is not positive definite raises
  `torch.linalg.LinAlgError` naming the pivot. Singular PSD, indefinite and
  complex-Hermitian matrices are not supported.
- The non-symmetric kernels eliminate in a fixed order without pivoting. They raise
  `torch.linalg.LinAlgError` when a pivot is singular or when element growth exceeds
  `1/sqrt(eps)` of the working dtype, instead of returning unreliable blocks; choose an
  order that eliminates well-conditioned pivots first, or use a pivoted LU.
- `check=True` validates shapes, block symmetry, finite values and integer edge indices.
- A right-hand side is `[..., n, b]` (vectors) or `[..., n, b, k]` (columns). A shape that
  fits both is read as vectors only when it equals `diag.shape[:-1]`, as in
  `torch.linalg.solve`; `rhs.unsqueeze(-1)` selects columns.
- On trees the cost is `O(n b³)` time and `O(n b²)` storage. On general patterns it depends
  on the elimination order: with `w_v` later neighbours of node `v`, numeric storage is
  proportional to `Σ(1 + w_v) b²` and work to `Σ(1 + w_v²) b³`.
- In the tested regimes the forward error follows `κ(A)·u`; there is no proved
  structured bound for the full algorithm.
- The analytic backwards are first order. Use the functional kernels (for example
  `selinv_junction`) for higher-order derivatives.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Each kernel is checked against dense inverses in double precision, on-pattern residual
identities, finite differences and `gradcheck`, and the other kernels on shared special
cases. CI runs the suite on Linux, Windows and macOS, with the minimum supported PyTorch,
and against the installed wheel.

## Documentation

- [Derivations](docs/derivations.md): the recurrences, derivative conventions, costs,
  numerical boundaries and prior work.
- [Architecture](docs/ARCHITECTURE.md): module map and conventions for contributors.
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md).
- [Research archive](https://github.com/vanshsehrawat14/gabp-sparse-inv/tree/research-archive):
  the research code, experiments and manuscripts that `main` carried before 0.4.0.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). If the selected inverse matters to
your work, please also cite the algorithms: Takahashi, Fagan and Chen (1973), Erisman and
Tinney (1975), Lin et al. (2011), and Andersen, Dahl and Vandenberghe (2013). The software
is MIT licensed; see [LICENSE](LICENSE).
