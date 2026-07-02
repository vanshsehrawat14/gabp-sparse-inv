# Architecture & orientation

The 60-second map of the codebase: what each piece does, the pattern every kernel follows,
and where the math / status / scope live. Read this alongside `PROJECT_STATUS.md` (what's
done) and `ROADMAP.md` (where it's going).

## What this is

A PyTorch package for the **selected inverse** of sparse block-structured matrices: the
blocks of `A⁻¹` that lie on `A`'s own (or its *filled*) sparsity pattern, computed **without
forming the dense inverse**, in `O(n)` (low treewidth) / `O(n + fill)`, and
**differentiably**. The organizing fact (proved in `derivations.md`): selected inversion on a
**tree** = Gaussian Belief Propagation = the Takahashi recurrence, and its **reverse-mode
adjoint is the same two-pass schedule on the same structure** - it is *self-adjoint*. The code
realizes this across a generality ladder: chain → star → tree → junction tree (general sparse
SPD); and a non-symmetric ladder: lower-bidiagonal → dense `(I−A)⁻¹` chunk → general LU.
The exact engine is classical; **the novelty is the differentiable thread.**

## Module map (`gabp_sparse_inv/`)

| module | what it owns |
|--|--|
| `layout.py` | storage containers - `BlockTridiag`, `BlockBidiag`, `BlockStar`, `BlockTree`, `BlockSparseSym` - each with `to_dense`/`from_dense` (the oracle bridge) and `validate`; plus `tree_orders` / `tree_levels` |
| `_linalg.py` | shared batched SPD helpers: `cholesky_spd` (informative SPD-failure message), `inv_via_chol` |
| `generators.py` | seeded test-matrix generators with a **monotone κ knob** (`random_spd_chain/star/tree/graph/laplacian`, `random_nonsym_bidiag`) + `grid_edges`, `condition_number` |
| `chain.py`, `star.py`, `tree.py` | symmetric selected-inverse kernels: path, depth-1 star, general tree (`tree.py` is the reference the others specialize) |
| `junction.py` | **general sparse SPD** selected inverse on the filled pattern: symbolic elimination + multi-neighbour Takahashi; `junction_solve`, `junction_logdet`; ordering helpers; level-set `batched=True` path; functional ⇒ autograd |
| `nonsym.py` | non-symmetric kernels: lower-bidiagonal + tree (autograd); dense `(I−A)⁻¹` chunk (`selected_inverse_tril` / `selinv_tril`, analytic backward) |
| `nonsym_junction.py` | general non-symmetric selected inverse and solve on the filled `L+U` pattern (static no-pivot regime) |
| `junction_autodiff.py` | tape-free analytic filled-pattern backwards for the SPD and non-symmetric junction kernels |
| `autodiff.py` | `SelInvTree` / `selinv_tree`: hand-written analytic, level-set **batched** backward - **first-order only** |
| `gmrf.py` | tree-GMRF application: `tree_logdet`, `tree_solve`, marginal likelihood, posterior variances |
| `gmrf_grid.py` | grid/loopy GMRF via junction kernels (CAR precision, marginal likelihood, posterior variances) |
| `sampling.py` | exact Gaussian sampling on tree and junction layouts |
| `demos/maze_tree.py`, `demos/maze_grid.py` | maze demonstrations - architectural ablation, not causal attribution |
| `bench/` | `metrics.py`, `run.py` (timing: chain/star/tree only), `precision.py`, `stability.py`, `gmrf_scaling.py` |

## Anatomy of a structure (how to add a kernel)

Every supported structure gets the same five pieces - **copy the most recent example**:
`junction.py` + `BlockSparseSym` (`layout.py`) + `random_spd_graph` (`generators.py`) +
`tests/test_junction.py`.

1. **Container** in `layout.py` - structured storage + `to_dense`/`from_dense` (so the dense
   oracle can score you) + `validate` (raise `TypeError`/`ValueError`, never bare `assert`).
2. **Generator** in `generators.py` - build a PSD base *on the exact pattern*, return
   `A = A_base + diag_load·I` so `diag_load` is a monotone condition-number knob.
3. **Kernel** module - the forward; differentiable (see below).
4. **Tests** `tests/test_<x>.py` - at the bar (below).
5. **Docs** - `derivations.md` (the math/recurrence), `PROJECT_STATUS.md` (status row +
   section), `README.md` (a runnable quickstart). Export public names in `__init__.py`
   `__all__`.

## The differentiable pattern

Write the forward **functionally** - out-of-place arithmetic, `torch.linalg.cholesky` /
`torch.cholesky_solve` / matmul / `.mT`, **never in-place on grad tensors**. Then reverse-mode
autograd *through it* IS the self-adjoint schedule (`derivations.md` §8.4): exact gradients for
free, validated by `gradcheck`. `junction.py` is the canonical example.

The **only** hand-written analytic backward is in `autodiff.py` (the tree kernel), as an
`O(n)` batched, tape-free optimization. Hand-deriving backwards for new kernels is **reserved**
(the adjoint proofs are the research) - functional forward + autograd + gradcheck is the
rigorous, sufficient path for everything else.

> Gotcha: `gradcheck` does **not** catch a wrong *forward* (it checks gradient-vs-numeric of
> the same function). The **dense oracle** gates the forward; `gradcheck` gates the backward.
> You need both.

## Conventions (so you match the code)

- Blocks are `[..., b, b]`, batched over leading dims; the structural axis (`n` nodes / `L`
  blocks) is sequential in the reference kernels.
- Off-diagonal storage is **lower-triangular, `i > j`**: `edge_val[k] = A_{i,j}` and
  `A_{j,i} = A_{i,j}ᵀ` (symmetric) - the junction kernel re-emits its filled pattern the same
  way. Tree storage: `edge[v] = A_{p(v),v}`.
- **Pivot symmetrization policy** (`derivations.md` §6): symmetrize every pivot `D ← (D+Dᵀ)/2`
  before its Cholesky, and the diagonal outputs `G_vv` too.
- SPD failures **raise** (`cholesky_spd` gives the failing pivot's name) - never silently
  continue.
- **Condition-aware tolerances** on uncontrolled inputs: `tol ~ C·κ·eps` (random trees/graphs);
  absolute floors (`~1e-10`) only on controlled topologies (path/balanced). fp32 is a loose
  gate; bf16 uses a store-low / compute-fp32 path (CPU LAPACK has no half Cholesky).

## The test bar (what "tested" means here)

Match `tests/test_tree.py` / `tests/test_junction.py`. A kernel is "tested" when it has:
dense-oracle accuracy (vs `torch.linalg.inv` on `to_dense()`); independent **on-pattern
residuals**; SPD/symmetry of outputs; **cross-kernel reductions** (chain ≡ tree-path, star ≡
tree-depth1, junction ≡ tree on a tree, junction ≡ chain on a path); for differentiable
kernels `gradcheck` (fp64) + analytic-vs-dense adjoint for *general* cotangents; edge cases
(`n=1`, scalar `b=1`, explicit topology, disconnected/empty where meaningful); leading batch
dims; fp32 + bf16.

Run (Windows, CPU torch): `.venv314/Scripts/python.exe -m pytest -q` - the full suite must stay
green after every change.

## Where everything else lives

- `derivations.md` - theorems, proofs, the **explicit recurrences** (source of truth for the
  math): §2 tree theorem, §2.1 filled-pattern closure, §2.2 junction forward, §6 stability
  (tracked-constant), §8 the adjoint, §8.4 filled-pattern adjoint, §9 non-symmetric.
- `PROJECT_STATUS.md` - what is done and tested (authoritative), incl. the "Known gaps" table.
- `ROADMAP.md` - the forward program, milestones, open questions, and priority-ordered
  "What's next".
- `APPLICATIONS.md`, `MAZE.md`, `DEQ.md`, `DELTANET.md` - application and demonstration notes.
