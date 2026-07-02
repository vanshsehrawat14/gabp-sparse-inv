# gabp-sparse-inv

Sparse selected inverse kernels for block-structured matrices (SPD and structured
non-symmetric). For each supported pattern the package computes only the blocks of `A^-1`
that lie on `A`'s own (or its filled) sparsity pattern, without forming the dense inverse.

The organizing principle: when the block structure of `A` is a **tree**, selected
inversion is a two-pass collect/distribute schedule that is exactly **Gaussian
Belief Propagation** and equals the Takahashi recurrence. See
[docs/derivations.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/derivations.md) for the theorem and proofs.

New to the codebase? [docs/ARCHITECTURE.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/ARCHITECTURE.md) is the module map, the
conventions, and the pattern for adding a kernel.

## Statement of need

Matrix inversion increasingly sits *inside* differentiable models: deep-equilibrium layers,
gated linear attention and the delta rule, and other implicit / fixed-point layers place a
structured linear solve in the forward pass and its adjoint in the backward pass. The dense
inverse is `O(N^3)`; the on-pattern *selected* inverse of a low-treewidth matrix is `O(n)`
(trees) to `O(n^1.5)` (2-D grids). Existing selected-inversion libraries (SelInv, PEXSI) target
compiled HPC and are not differentiable, and general sparse solvers return factorizations or
solves rather than the on-pattern inverse blocks with gradients. `gabp-sparse-inv` provides
drop-in PyTorch operators that return exact on-pattern inverse blocks, plus the log-determinant,
Gaussian samples, and solves that share the same factorization, with exact gradients at the same
asymptotic cost as the forward pass, across one uniform symmetric / non-symmetric interface. The
full statement of need is in [paper/joss/paper.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/paper/joss/paper.md).

Implemented and tested kernels:

- **Chain** (block-tridiagonal; a path). Block `LDL^T` factorization + Takahashi
  back-recursion. `O(L * b^3)` time, `O(L * b^2)` storage for `L` blocks of size
  `b` - linear in `N = L * b` at fixed block size.
- **Star** (block-arrowhead; a depth-1 tree). One center block coupled to `K`
  leaf blocks, no leaf-leaf coupling. Leaves are eliminated in parallel.
  `O(K * b^3)` time, `O(K * b^2)` storage.
- **Tree** (arbitrary rooted tree). The general kernel: a node `parent` array and
  one edge block per non-root node; collect (leaves→root) then distribute
  (root→leaves). Chain and star are its path and depth-1 special cases.
  `O(n * b^3)` time, `O(n * b^2)` storage for `n` nodes.

- **Differentiable tree** (`selinv_tree`). The selected inverse with a hand-written
  analytic backward: the reverse two-pass is itself a collect/distribute on the same
  elimination tree (selected inversion is *self-adjoint*), `O((|V|+|E|) b^3)` like the
  forward. Gradients flow to `diag` and `edge`; an optional level-set **batched** path
  mirrors the forward batching. Proved in [docs/derivations.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/derivations.md)
  §8; the gradient identity itself is folklore (Dwyer-Macphail / Giles); the
  contribution is the structure-preserving schedule + batched kernel.

- **Non-symmetric chain** (`selected_inverse_bidiag` / `selinv_bidiag`). The selected
  inverse of a general (non-symmetric) block **lower-bidiagonal** matrix `M` -- the
  non-symmetric analogue of the chain: `G_ii = M_ii^-1` and
  `G_{i+1,i} = -M_{i+1,i+1}^-1 M_{i+1,i} M_ii^-1` on `M`'s pattern, with a hand-written
  analytic backward (`selinv_bidiag`). Fully local -- no collect/distribute sweep -- so
  forward and backward are each one batched block op, `O(n * b^3)` time, `O(n * b^2)`
  storage. The first rung of the non-symmetric ladder; see
  [docs/derivations.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/derivations.md) §9.

- **Non-symmetric tree** (`selected_inverse_nonsym_tree`). The zero-fill non-symmetric rung
  between the bidiagonal case and the general LU: a general block matrix whose off-diagonal
  graph is a tree but whose two directed edge blocks are independent (`M_{p,v} != M_{v,p}^T`).
  A two-sided Takahashi recurrence returns each node diagonal and both cross blocks exactly;
  functional / autograd-traceable (first- and higher-order), and it reduces block-for-block to
  the SPD tree kernel in the symmetric case. See [docs/derivations.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/derivations.md) §9.5.

- **DeltaNet chunk inverse** (`selected_inverse_tril` / `selinv_tril`). The dense
  triangular instance `T = (I - A)^-1` for strictly-lower `A` -- the chunk inverse of
  DeltaNet / gated linear attention -- with the analytic self-adjoint backward
  `bar_A = tril(T^T bar_T T^T, -1)` (`docs/derivations.md` §9.4). Here a blocked
  triangular solve is the baseline, so the contribution is the explicit analytic backward.

- **Junction tree / general sparse** (`selected_inverse_junction` / `selinv_junction`).
  The general sparse SPD case: an **arbitrary** block sparsity pattern, symbolically
  completed to its chordal (filled) pattern `S = pattern(L + L^T)` by a min-degree
  elimination order, then the multi-neighbour Takahashi recurrence (sparse block Cholesky +
  clique back-substitution). Trees are the zero-fill special case. The functional forward is
  autograd-differentiable, so `selinv_junction` yields exact gradients by reverse-mode
  through the `S`-local self-adjoint schedule (`docs/derivations.md` §2.2, §8.4).
  `O((|V| + fill) * b^3)` time.

The **junction-tree** kernel (above) ships the general sparse SPD forward and its autograd
adjoint (the §8.4 schedule realized through reverse-mode AD). The **tape-free hand-written
analytic** junction backward is available as `selinv_junction_analytic`; the non-symmetric
counterpart is `selinv_nonsym_junction_analytic`
(`gabp_sparse_inv/junction_autodiff.py`). They run the explicit reverse clique recurrence
(`docs/derivations.md` §8.5 / §10.3) with no autograd tape and are validated against the
functional path. The **general non-symmetric** selected inverse (LU / Erisman-Tinney;
`selected_inverse_nonsym_junction` / `selinv_nonsym_junction`, forward + autograd adjoint,
no pivoting) and its **solve sibling** `nonsym_junction_solve` (`A⁻¹b` / `A⁻ᵀb`) are also
included. The fixed-point and maze demonstrations are documented in [docs/DEQ.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/DEQ.md)
and [docs/MAZE.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/MAZE.md).

## Install

From PyPI:

```bash
pip install gabp-sparse-inv
```

From source (Linux / macOS):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch pytest
pip install -e .
```

From source (Windows / PowerShell):

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch pytest
pip install -e .
```

GPU paths are used automatically when `torch.cuda.is_available()`.

## Quickstart

```python
import torch
from gabp_sparse_inv import random_spd_chain, selected_inverse_chain

bt = random_spd_chain(num_blocks=8, block_size=4, seed=0, diag_load=2.0)

G_diag, G_lower = selected_inverse_chain(bt.diag, bt.lower, check=True)

inv = torch.linalg.inv(bt.to_dense())
assert torch.allclose(G_diag[0], inv[0:4, 0:4], atol=1e-10)
```

Inputs support optional leading batch dimensions: `[..., L, b, b]`. The chain
dimension `L` is sequential; each `b x b` block operation is batched over the
leading dimensions.

The **star** kernel takes the center block, the stacked leaf blocks, and the
center->leaf couplings, and returns the center, leaf, and cross inverse blocks:

```python
from gabp_sparse_inv import random_spd_star, selected_inverse_star

st = random_spd_star(num_leaves=16, block_size=4, seed=0, diag_load=2.0)

# G_center = (A^-1)_00 ; G_leaf[j] = (A^-1)_jj ; G_cross[j] = (A^-1)_{j,0}
G_center, G_leaf, G_cross = selected_inverse_star(
    st.center, st.leaf_diag, st.coupling, check=True
)

inv = torch.linalg.inv(st.to_dense())
assert torch.allclose(G_center, inv[0:4, 0:4], atol=1e-10)
```

The off-pattern leaf-leaf inverse blocks `(A^-1)_{jk}` (`j != k`) are nonzero but
outside the selected pattern and are never formed.

The **tree** kernel takes the stacked node blocks, one edge block per non-root node
(`edge[v] = A_{parent(v), v}`), and the `parent` array:

```python
from gabp_sparse_inv import random_spd_tree, selected_inverse_tree

bt = random_spd_tree(num_nodes=12, block_size=3, seed=0, diag_load=2.0, kind="random")

# G_diag[v] = (A^-1)_vv ; G_edge[v] = (A^-1)_{parent(v), v}  (root slot is zero)
G_diag, G_edge = selected_inverse_tree(bt.diag, bt.edge, bt.parent, check=True)

inv = torch.linalg.inv(bt.to_dense())
assert torch.allclose(G_diag[0], inv[0:3, 0:3], atol=1e-10)
```

`kind` selects the topology (`random`/`path`/`star`/`balanced`), or pass an explicit
`parent` array. The path and depth-1 trees reproduce the chain and star kernels
block-for-block (a root-invariance check, since the chain roots the path at its last
node and the tree roots it at node 0).

### General sparse selected inverse (junction tree)

`selected_inverse_junction` handles an **arbitrary** block sparsity pattern. Pass the node
diagonals, the lower-triangular off-diagonal blocks (`edge_index` columns `(i, j)` with
`i > j`, `edge_val[k] = A_{i,j}`), and an optional elimination order (default: greedy
min-degree). It returns the selected inverse on the **filled** pattern `S`, a superset of
the input pattern wherever elimination creates fill:

```python
import torch
from gabp_sparse_inv import random_spd_graph, grid_edges, selected_inverse_junction

# A 3x3 grid is loopy (treewidth 3): elimination fills in, unlike a tree.
sp = random_spd_graph(num_nodes=9, edges=grid_edges(3, 3), block_size=2, seed=0)

G_diag, S_index, G_lower = selected_inverse_junction(
    sp.diag, sp.edge_index, sp.edge_val, check=True
)

inv = torch.linalg.inv(sp.to_dense())
assert torch.allclose(G_diag[0], inv[0:2, 0:2], atol=1e-10)   # node-0 diagonal block
assert S_index.shape[1] > sp.edge_index.shape[1]              # fill: S grew past the input
```

`selinv_junction` is the autograd-connected form: gradients of any loss over `G_diag` /
`G_lower` flow to `diag` and `edge_val` through the self-adjoint `S`-local schedule
(`docs/derivations.md` §8.4). It reduces block-for-block to the tree kernel at zero fill.
Pass `batched=True` (to either entry point) for the level-set path: the Python loop runs
over elimination *levels* (`O(tree height)`: `~√n` on a 2-D grid) instead of nodes, each
level a few batched `index_add` block ops, the junction analogue of
`selected_inverse_tree(batched=True)`. Identical result and gradients (it stays functional,
so autograd gives the same backward); it amortizes kernel-launch latency on GPU.

`junction_solve` is the differentiable sparse SPD linear solve `x = A⁻¹ b` on the same
pattern, sharing one `LDL^T` factorization with the selected inverse:

```python
from gabp_sparse_inv import random_spd_graph, grid_edges, junction_solve

sp = random_spd_graph(num_nodes=9, edges=grid_edges(3, 3), block_size=2, seed=0)
b = torch.randn(9, 2)                                  # [n, b] (or [n, b, k] for k RHS)
x = junction_solve(sp.diag, sp.edge_index, sp.edge_val, b, check=True)
assert torch.allclose(x.reshape(-1), torch.linalg.solve(sp.to_dense(), b.reshape(-1)))
```

`junction_logdet` returns `log det A` straight from the same `LDL^T` factorization
(differentiable, the junction sibling of `tree_logdet`):

```python
from gabp_sparse_inv import junction_logdet

ld = junction_logdet(sp.diag, sp.edge_index, sp.edge_val)
assert torch.allclose(ld, torch.logdet(sp.to_dense()))
```

### Differentiable selected inverse

`selinv_tree` is the autograd-connected tree kernel: gradients of any loss over the
selected blocks flow back to the input blocks `diag` and `edge` through the analytic
two-pass backward (no autograd tape over the per-node loop).

```python
import torch
from gabp_sparse_inv import random_spd_tree, selinv_tree

bt = random_spd_tree(num_nodes=64, block_size=3, seed=0, diag_load=2.0, kind="balanced")
diag = bt.diag.clone().requires_grad_(True)
edge = bt.edge.clone().requires_grad_(True)

G_diag, G_edge = selinv_tree(diag, edge, bt.parent)        # or batched=True
loss = torch.diagonal(G_diag, dim1=-2, dim2=-1).sum()       # sum of marginal variances
loss.backward()
# diag.grad, edge.grad are the exact on-pattern cotangents (gradcheck-verified).
```

**First-order only** for `selinv_tree` (the hand-written analytic backward). For
Hessian-vector products use the functional junction kernels or
`selected_inverse_tree(batched=True)`; both pass `gradgradcheck`
(`tests/test_double_backward.py`). `batched=True` uses the level-set path for forward
and backward; on CUDA it amortizes kernel-launch latency (timing not benchmarked here).

### Non-symmetric selected inverse

The non-symmetric ladder is hot-swappable with the SPD ops above. The block **lower-bidiagonal**
case (`selinv_bidiag`) is the non-symmetric analogue of the chain: `G_ii` and `G_{i+1,i}` on
`M`'s pattern, fully local:

```python
from gabp_sparse_inv import random_nonsym_bidiag, selinv_bidiag

M = random_nonsym_bidiag(num_blocks=8, block_size=3, seed=0, diag_load=2.0)
G_diag, G_lower = selinv_bidiag(M.diag, M.lower)          # general blocks, no SPD assumption
inv = torch.linalg.inv(M.to_dense())
assert torch.allclose(G_diag[0], inv[0:3, 0:3], atol=1e-10)
```

The dense triangular chunk inverse `T = (I − A)⁻¹` for strictly-lower `A` (the DeltaNet /
gated-linear-attention primitive) is `selinv_tril`, with the analytic self-adjoint backward
`bar_A = tril(Tᵀ bar_T Tᵀ, −1)`:

```python
from gabp_sparse_inv import selinv_tril

A = torch.tril(torch.randn(6, 6, dtype=torch.float64), -1)   # a chunk operator
T = selinv_tril(A)                                           # T = (I - A)^{-1}, differentiable
assert torch.allclose(T, torch.linalg.inv(torch.eye(6, dtype=torch.float64) - A), atol=1e-12)
```

`nonsym_junction_solve` is the general non-symmetric sparse solve `A⁻¹b` (and `A⁻ᵀb`, the
DEQ/implicit-differentiation adjoint) on the filled `L+U` pattern. Pass **independent** lower and
upper edge blocks. On symmetric input (`edge_upper = edge_lower.mT`) it matches `junction_solve`:

```python
from gabp_sparse_inv import random_spd_graph, grid_edges, junction_solve, nonsym_junction_solve

sp = random_spd_graph(num_nodes=9, edges=grid_edges(3, 3), block_size=2, seed=0)
rhs = torch.randn(9, 2, dtype=torch.float64)
x = nonsym_junction_solve(sp.diag, sp.edge_index, sp.edge_val, sp.edge_val.mT, rhs)
assert torch.allclose(x, junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs), atol=1e-10)

# transpose=True reuses the same LDU factors transposed -- the exact DEQ backward (A^T u = g).
u = nonsym_junction_solve(sp.diag, sp.edge_index, sp.edge_val, sp.edge_val.mT, rhs, transpose=True)
```

The full selected inverse on the `L+U` pattern is `selinv_nonsym_junction` (forward + adjoint), and
the zero-fill tree rung is `selected_inverse_nonsym_tree`; both keep the two directed edge blocks
independent. See [docs/derivations.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/derivations.md) §9-§10.

### Gaussian sampling

`sample_gaussian_tree` / `sample_gaussian_junction` draw `x ~ N(0, A⁻¹)` from any tree- or
junction-structured SPD **precision** `A`, reusing the same factorization (covariance is exactly
`A⁻¹`, verified by the deterministic transform on the standard basis):

```python
from gabp_sparse_inv import random_spd_tree, sample_gaussian_tree

bt = random_spd_tree(num_nodes=16, block_size=2, seed=0, diag_load=2.0, kind="balanced")
x = sample_gaussian_tree(bt.diag, bt.edge, bt.parent, num_samples=8)   # [num_samples, n, b]
```

`junction_logdet` / `tree_logdet` (above) and these samplers are the statistical ops that fall out
of the shared `LDL^T` factorization. See [docs/APPLICATIONS.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/APPLICATIONS.md).

### Application: hierarchical tree-GMRF learning

`gabp_sparse_inv/gmrf.py` builds on `selinv_tree` to learn the hyperparameters of a
hierarchical (tree-structured) Gaussian Markov random field by exact marginal
likelihood and a posterior-variance objective, all `O(n)`, where a dense-autograd
baseline is `O(N^3)` time / `O(N^2)` memory. The **batched** schedule (`batched=True`)
beats a naive dense-autograd baseline at every measured size on CPU (113× at n=1023 in one
fp64 / 1-thread run with 16 fields; a diagnostic, not CI-gated; see
[docs/APPLICATIONS.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/APPLICATIONS.md)). The per-node reference loop is slower than
dense at small n; batched is the path intended for scale.

```python
from gabp_sparse_inv import sample_tree_gmrf, fit_marginal_likelihood

parent = [-1, 0, 0, 1, 1, 2, 2]
y = sample_tree_gmrf(parent, a=0.7, kappa=1.5, root_prec=2.0, seed=0)[None]  # one field
theta = fit_marginal_likelihood(parent, y, steps=300)   # recovers a, kappa, root_prec, sigma2
```

```bash
python -m gabp_sparse_inv.bench.gmrf_scaling --values 127 255 511 1023 2047
```

The **loopy/grid** counterpart (`gabp_sparse_inv/gmrf_grid.py`) ports the same model to an
arbitrary graph via the junction kernel: a CAR precision `Q = kappa (I + a L)` with exact
marginal likelihood (`junction_marginal_log_likelihood`) and posterior marginal variances
(`junction_posterior_marginal_variances`). The cycles are handled *exactly*, not iteratively.

```python
from gabp_sparse_inv import grid_gmrf_precision, fit_grid_marginal_likelihood

diag, edge_index, edge_val = grid_gmrf_precision(rows=8, cols=8, kappa=1.5, a=0.6)
# ... sample/observe y of shape [..., n, 1] ...
theta = fit_grid_marginal_likelihood(8, 8, y, steps=200)   # recovers kappa, a, sigma2
```

### Demonstration: the tree-inverse as the only long-range operator (maze on trees)

`gabp_sparse_inv/demos/maze_tree.py` is the clean-room experiment behind the headline: a
source-routing task on trees where a single differentiable `tree_solve` layer is the *only*
operator that can move information across the graph. A model with that layer routes the
source near-exactly (test MSE `~1e-5`); an otherwise-identical model with only `K`-hop local
message passing cannot, and the gap widens with the tree diameter. The learned precision is
kept SPD and well-conditioned (`kappa ~ 200`) by construction, handling the maze-conditioning
risk. It is the tree proxy for the loopy grid maze (Phase 4). See [docs/MAZE.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/MAZE.md).

```bash
python -m gabp_sparse_inv.demos.maze_tree     # depth-sweep table: gabp vs local vs baseline
```

### Demonstration: the junction inverse on a loopy grid

`gabp_sparse_inv/demos/maze_grid.py` is the direct analogue: each cell is a node on a 2-D
lattice, the precision is a grid Laplacian built from learned local features, and a single
differentiable `junction_solve` layer is the only long-range operator (convolutions are
strictly local). The loopy graph needs the junction-tree kernel; a tree kernel cannot
represent cycles. Same clean-attribution story as the tree proxy; see [docs/MAZE.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/MAZE.md)
(grid section).

```bash
python -m gabp_sparse_inv.demos.maze_grid     # size-sweep table: junction vs local vs baseline
```

### Demonstration: the non-symmetric inverse as the exact DEQ backward (fixed-point layer)

`gabp_sparse_inv/demos/deq_fixedpoint.py` is the real-impact rung. A deep-equilibrium layer
`z* = f(z*, x)` has, by the implicit function theorem, a backward that is a **non-symmetric**
solve with the equilibrium Jacobian, `(I − J)ᵀ u = ∂L/∂z*`. When `J` is graph-structured this
is exactly `nonsym_junction_solve(…, transpose=True)` on `A = I − J`: one block `LDU`,
`O(fill)`, the transpose of the same factors. The structured backward is **exact** (matches a
dense implicit-diff oracle and autograd through an unrolled solver to `~1e-15`) and stays
machine-accurate as the equilibrium stiffens (`ρ(J) → 1`), where the standard iterative
(Neumann) DEQ backward's gradient error tracks `ρᴷ`. Honest scope: low-treewidth Jacobians,
`ρ(J) < 1`, a mechanism/impact result (not a SOTA claim). See [docs/DEQ.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/DEQ.md).

```bash
python -m gabp_sparse_inv.demos.deq_fixedpoint   # rho-sweep: exact backward vs iterative
```

### Demonstration: the DeltaNet chunk inverse as a hot-swappable op (linear attention)

`gabp_sparse_inv/demos/deltanet_chunk.py` shows the differentiable triangular chunk inverse
`selinv_tril` (`T = (I − A)⁻¹`) is a **drop-in** inside a real chunked linear-attention layer;
no new kernel. DeltaNet's within-chunk delta rule *is* the triangular solve `W = (I − A)⁻¹ delta`
(`A = −tril(diag(β) K Kᵀ, −1)`); the minimal layer built around it reproduces the token-by-token
delta rule **exactly** (validated vs an `O(L)` sequential oracle at every chunk size). Forming
`T` with `selinv_tril` (analytic self-adjoint backward) vs the stock `solve_triangular` baseline
(autograd) gives the same forward and **the same gradients** through the whole multi-chunk layer
(`~3e-15`), and a layer trains identically either way. A capability / drop-in result, not a
DeltaNet reimplementation or a SOTA claim. See [docs/DELTANET.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/docs/DELTANET.md).

```bash
python -m gabp_sparse_inv.demos.deltanet_chunk   # drop-in equivalence + train-both-ways table
```

## Benchmark

```bash
python -m gabp_sparse_inv.bench.run --problem chain --sweep L --b 8 --precisions fp64 fp32 bf16
python -m gabp_sparse_inv.bench.run --problem star  --sweep K --b 8 --precisions fp64 fp32
python -m gabp_sparse_inv.bench.run --problem tree  --sweep n --b 8 --tree-kind random
python -m gabp_sparse_inv.bench.run --problem tree  --sweep n --b 8 --grad   # fwd+bwd
```

`--grad` benchmarks the differentiable kernel: forward+backward time scaling vs `n`
(linear), loop-vs-batched timing, gradient correctness vs dense autograd (machine
precision), and the structured-vs-dense-autograd backward-memory ratio (`O(n b^2)` vs
`O((n b)^2)`).

The benchmark writes CSV and JSON records keyed by `(seed, config)`. It reports
log-log timing slope, dense crossover, forward error, independent residuals, and
analytic structured-vs-dense memory **as diagnostics** - these depend on the BLAS
backend, device, and thread count, so they are recorded and reported rather than
asserted. Measured CPU peak RSS is secondary and noisy.

### Precision study

```bash
python -m gabp_sparse_inv.bench.precision --problem tree --size 64 --b 4
python -m gabp_sparse_inv.bench.precision --problem chain --precisions fp32 bf16
python -m gabp_sparse_inv.bench.precision --compare-orders --size 64 --b 4   # ordering study
```

Puts the kernel head-to-head with a dense inverse at the **same** precision, both scored
against an fp64 oracle on the pattern, swept over condition number. Reports each method's
on-pattern error and the selinv/dense advantage ratio; a low-precision factorization
breakdown is recorded as `inf` rather than aborting the sweep. The honest bottom line
(diagnostics, not asserted): **no penalty, and no robust win**. The apparent fp32 edge on
ill-scaled random trees is modest (~1.5-2.3× median, heavy-tailed) and *mostly an
elimination-ordering effect* (`--compare-orders` shows a same-order dense Cholesky matches the
kernel to ~1× in the median); the well-scaled grid Laplacian shows parity at every κ. Precision
is **not** the differentiator; `O(n)` cost, `O(fill)` memory, and differentiability are.

### No-pivot stability boundary (non-symmetric)

```bash
python -m gabp_sparse_inv.bench.nonsym_stability     # sweep block-diagonal dominance
```

The non-symmetric kernel eliminates with **no pivoting**, so the static-pattern factorization is
only safe while block-diagonally dominant (`docs/derivations.md` §10.4). This sweeps the dominance
ratio and reports the Schur-pivot floor and the fp32 no-pivot error against a dense fp64 oracle,
alongside the fp32 **pivoted** dense LU on the same blocks. Diagnostic finding: at parity with
pivoted LU while dominant (`α ≳ 1`), the Schur floor collapsing and the no-pivot error departing
from `κ·u` below the dominance boundary, quantifying where the static-pattern regime ends.

## Tests

```bash
pytest -q
```

The test suite covers every kernel against dense fp64 oracles: chain, star, tree, the
differentiable tree (`selinv_tree`), junction tree, the non-symmetric bidiagonal / DeltaNet
chunk / non-symmetric tree, the tree- and grid-GMRF applications, Gaussian sampling, and the
elimination-ordering helpers. Per kernel: dense-oracle accuracy, condition-aware
ill-conditioned cases (where implemented), independent residual checks (SPD kernels),
SPD/symmetry properties, edge cases, leading batch dimensions, `compute_dtype` (strongest on
junction; bf16 sanity elsewhere), first-order autograd (`gradcheck` / analytic-vs-dense
adjoint), and, where applicable, second-order autograd (`gradgradcheck` on junction,
`selected_inverse_tree(batched=True)`, and `selected_inverse_nonsym_tree` only; **not** on
`selinv_tree`, `selinv_bidiag`, or `selinv_tril`). Also: order-invariance, fill, trace
identities (junction), and HVP checks (`tests/test_double_backward.py`). CI runs on Ubuntu,
Windows, and macOS with Python 3.12 and 3.13.

## Citation

If you use `gabp-sparse-inv` in your research, please cite it. Machine-readable metadata is in
[CITATION.cff](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/CITATION.cff) (GitHub's "Cite this repository" reads it), and a software paper is
in preparation for the Journal of Open Source Software
([paper/joss/paper.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/paper/joss/paper.md)).

## Contributing and support

Contributions, bug reports, and usage questions are welcome. [CONTRIBUTING.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/CONTRIBUTING.md)
covers how to contribute, report issues, and get support; [CODE_OF_CONDUCT.md](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/CODE_OF_CONDUCT.md)
sets the community standards. The package scope and deliberate exclusions are summarized above
and in the package docstring.

## License

MIT. See [LICENSE](https://github.com/vanshsehrawat14/gabp-sparse-inv/blob/v0.3.2/LICENSE).
