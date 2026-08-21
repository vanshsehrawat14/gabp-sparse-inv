# gabp-sparse-inv

PyTorch reference kernels for exact selected inversion and related sparse
Gaussian operations on block-structured matrices. The package returns inverse
blocks on the input or filled sparsity pattern without forming the dense
inverse, and supports differentiation through the implemented operators.

## Scope

| family | implemented operations |
|---|---|
| SPD chain, star, tree | selected inverse; analytic first-order tree backward |
| general sparse SPD | filled-pattern selected inverse, solve, log determinant, sampling, functional and analytic first-order backward |
| static no-pivot non-symmetric | lower-bidiagonal, tree, and filled-pattern selected inverse; solve and transpose solve; functional and analytic first-order backward |
| applications | tree/grid GMRFs, routed-field demonstrations, DEQ implicit backward, DeltaNet triangular chunk inverse |

Tree kernels cost `O(n b^3)` time and `O(n b^2)` storage for `n` blocks of
size `b`. General sparse cost is controlled by the filled elimination fronts;
the package does not claim linear complexity on arbitrary graphs.

The symmetric factorizations require SPD input. Singular PSD, indefinite,
complex-Hermitian, pivoted non-symmetric, and iterative loopy-GaBP cases are
not implemented. The analytic junction backwards are first-order only; use the
functional kernels for higher-order derivatives. This is a
correctness-oriented Python/PyTorch implementation, not a production GPU
kernel library.

## Install

```bash
pip install gabp-sparse-inv
```

PyPI currently serves `0.3.4`. For development:

```bash
git clone https://github.com/vanshsehrawat14/gabp-sparse-inv.git
cd gabp-sparse-inv
python -m venv .venv
pip install torch
pip install -e ".[dev]"
```

## Quickstart

```python
import torch
from gabp_sparse_inv import random_spd_chain, selected_inverse_chain

A = random_spd_chain(num_blocks=8, block_size=4, seed=0, diag_load=2.0)
G_diag, G_lower = selected_inverse_chain(A.diag, A.lower, check=True)

dense = torch.linalg.inv(A.to_dense())
torch.testing.assert_close(G_diag[0], dense[:4, :4])
```

General sparse SPD input uses node-diagonal blocks plus an undirected edge
list. The returned off-diagonal blocks follow the filled pattern:

```python
from gabp_sparse_inv import grid_edges, random_spd_graph, selected_inverse_junction

A = random_spd_graph(12, grid_edges(3, 4), 2, seed=0)
G_diag, filled_edges, G_lower = selected_inverse_junction(
    A.diag, A.edge_index, A.edge_val, check=True
)
```

## Verification

```bash
pytest -q
```

The suite compares every operator family with dense fp64 oracles and checks
applicable residuals, batching, dtypes, gradients, higher-order functional
paths, ordering/fill behavior, sampling transforms, and demonstrations. CI
covers Ubuntu, Windows, and macOS on supported Python versions.

## Documentation

- [Architecture and extension pattern](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/docs/ARCHITECTURE.md)
- [Derivations and algorithm boundaries](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/docs/derivations.md)
- [Contributing and support](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/CONTRIBUTING.md)
- [JOSS software-paper draft](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/paper/joss/paper.md)

## Citation and license

Citation metadata is in
[CITATION.cff](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/CITATION.cff).
The software is MIT licensed; see
[LICENSE](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/main/LICENSE).
