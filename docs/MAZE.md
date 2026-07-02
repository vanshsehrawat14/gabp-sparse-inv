# Demonstration: the tree-inverse as the only long-range operator (maze on trees)

This note records the **maze-on-trees** demonstration: a clean-room experiment where a
single differentiable tree-solve layer (the linear-system form of the same
collect/distribute selected-inversion machinery as `selinv_tree`) is the *only* operator
that can move information across the graph. It is the tree proxy for the roadmap's grid
maze (which is loopy, treewidth `~sqrt(n)`, and needs the junction-tree selected inverse):
it de-risks the learning and the conditioning on the kernels that already ship. Code:
`gabp_sparse_inv/demos/maze_tree.py`; tests: `tests/test_maze_tree.py`.

## Task: route a source through the tree

Each example draws a random **maze geometry**: positive edge weights `w` on a fixed tree, and
builds the precision

    A = L(w) + eps * I,    A_vv = eps + (weighted degree),    A_{p(v),v} = -w[v],

a graph Laplacian lifted by `eps * I`. The lift makes `A` strictly diagonally dominant,
hence SPD, and pins the condition number at `~deg/eps`; a *small* `eps` gives a long
correlation length, so `A^{-1}` (the tree Green's function) carries signal far across the
tree - the long-range regime. A random **source** node `s` is chosen and the label is the
field it induces,

    target = A^{-1} e_s,

the potential routed out from `s` along the unique tree paths. A node far from `s` still has
a small, nonzero potential that depends on the entire path back to `s`. The labels are the
*exact* tree solve (verified against a dense solve to `~1e-14`).

## Two models, identical local inputs

Both models see, per node, only **local** information: that node's own row of `A`
(`A_vv`, the parent-edge weight, the down-degree) and the source indicator `b_v`. No node is
handed anything about distant nodes.

| Model | Long-range operator | Can it route? |
|-------|---------------------|---------------|
| `gabp` | a per-node MLP -> positive edge weights -> **one `tree_solve`** of `A x = b` | yes - exact global field in `O(n)` |
| `local` (ablation) | `K` rounds of nearest-neighbour averaging (`K << diameter`) | no - blind past `K` hops |

The `gabp` model's encoder reads only the geometry features (not the source), and `softplus`
keeps the edge weights positive, so its precision `A = L(w_hat) + eps I` is **SPD by
construction** for any encoder output - the flagged maze-conditioning risk is handled by the
parameterization, not by hoping training stays in a good region.

## Result (depth sweep, CPU, fp64; `python -m gabp_sparse_inv.demos.maze_tree`)

Complete binary trees; 400 Adam steps; test MSE on held-out mazes/sources:

| depth | n | diameter | `gabp` | `local` (K=2) | predict-mean baseline | kappa (median/max) |
|------:|--:|---------:|-------:|--------------:|----------------------:|-------------------:|
| 3 | 15 | 6 | 4.7e-05 | 4.6e-01 | 4.6e-01 | 183 / 213 |
| 4 | 31 | 8 | 2.4e-05 | 1.2e-01 | 4.8e-01 | 198 / 220 |
| 5 | 63 | 10 | 1.5e-05 | 9.8e-02 | 3.3e-01 | 207 / 235 |

The tree-solve model routes the source **near-exactly** (`~1e-5`) at every depth; the local
ablation sits at or near the predict-mean baseline and the gap **widens with the diameter**,
exactly the signature of "inversion is the only long-range path." The condition number stays
bounded (`~200`) throughout, confirming the parameterization keeps `A` well within fp32/fp64
range.

## Scope and honest caveats

- **It is a *capability* demonstration, not a hard learning problem.** The `gabp` model is
  given the local rows of `A` and must (a) recover the edge weights and (b) apply the global
  solve; the point is the *architectural* separation under identical inputs, not that the map
  is hard to fit. The local model's failure is structural (range-limited receptive field), so
  the separation between global-solve and local-MP models is clean - but this is an **ablation**,
  not a causal attribution proof (no matched-capacity control without a global op).
- **Tree proxy.** The loopy grid version is shipped below; this tree section de-risked the
  mechanism and the conditioning control on the tree kernel first.
- Numbers are BLAS/thread-dependent diagnostics from the demo script; the **test**
  (`tests/test_maze_tree.py`) asserts the robust, machine-independent facts: the labels equal
  the dense solve, the learned precision is SPD and `kappa < 1e3`, and `gabp` beats `local` by
  a wide margin (`gabp < 0.1 * local`) at diameter 8.

---

# Demonstration: the junction-tree inverse on a loopy grid (maze on grids)

The grid version of the experiment above, now on a **loopy** 2-D lattice (treewidth ~ side
length) where one differentiable `junction_solve` layer is the only long-range operator.
Unlike the tree proxy, the grid genuinely needs the junction-tree (filled-pattern) selected
inverse - a tree kernel cannot represent a graph with cycles. Code:
`gabp_sparse_inv/demos/maze_grid.py`; tests: `tests/test_maze_grid.py`.

## Task: route a source through the grid

Each example draws a random maze geometry and builds the grid-graph Laplacian precision

    A = L(w) + eps * I,    A_vv = eps + (weighted degree),    A_{i,j} = -w_{ij},

with positive edge weights `w`. As on the tree, the `eps` lift makes `A` SPD and pins
`kappa ~ deg/eps`; a small `eps` gives a long correlation length so `A^{-1}` (the grid
Green's function) carries signal across the lattice. A random **source** `s` is chosen and
the label is the routed field `target = A^{-1} e_s`, computed by `junction_solve` on the
loopy pattern (verified against a dense solve to `~1e-10`).

Edge weights are factored through per-node potentials, `w_ij = phi_i * phi_j > 0`. A tree
hands each node its single parent-edge weight; on a grid an edge has no owning node, so this
factorization is what keeps the operator a function of strictly **local** node features that
the per-node encoder can reconstruct - making the local model's only handicap the global
solve, not missing inputs.

## Two models, identical local inputs

Both models see, per node, only local information: `[phi_v, weighted_degree_v, b_v]`.

| Model | Long-range operator | Can it route? |
|-------|---------------------|---------------|
| `gabp` | a per-node MLP -> positive potentials -> edge weights -> **one `junction_solve`** of `A x = b` | yes - exact global field on the loopy grid |
| `local` (ablation) | `K` rounds of nearest-neighbour averaging (`K << diameter`) | no - blind past `K` hops |

The `gabp` encoder reads only the geometry features (not the source); `softplus` keeps the
potentials positive, so `A = L(w_hat) + eps I` is **SPD by construction** for any encoder output
 -  the maze-conditioning risk handled by the parameterization.

## Result (size sweep, CPU, fp64; `python -m gabp_sparse_inv.demos.maze_grid`)

Square grids; 250 Adam steps; test MSE on held-out mazes/sources:

| grid | n | diameter | `gabp` | `local` (K=2) | predict-mean baseline | kappa (median/max) |
|------|--:|---------:|-------:|--------------:|----------------------:|-------------------:|
| 4×4 | 16 | 6 | 1.3e-05 | 5.0e-03 | 3.6e-02 | 157 / 203 |
| 5×5 | 25 | 8 | 2.0e-05 | 5.3e-03 | 3.2e-02 | 176 / 213 |
| 6×6 | 36 | 10 | 1.2e-05 | 4.3e-03 | 2.6e-02 | 188 / 213 |

The `junction_solve` model routes the source **near-exactly** (`~1e-5`) at every grid size; the
local ablation plateaus roughly **two orders of magnitude worse** (`~5e-3`), capturing only
near-source structure and never the global field. The condition number stays bounded (`~200`)
throughout, confirming the parameterization keeps `A` well within fp32/fp64 range.

## Scope and honest caveats

- A **capability** demonstration, not a hard learning problem: the separation is architectural
  under identical local inputs (the local model's range limit is structural). The grid demo is
  **easier than it looks**: `gen_dataset` puts the true per-node potential `phi` in
  `feats[..., 0]`, so the encoder mainly needs to apply `junction_solve`, not recover geometry
  from scratch. The `local` model does beat predict-mean - it fits the near-source structure,
  but cannot route globally, which is the whole point. Matched-capacity hardening with
  fair GNN/Transformer baselines at 280-340x the parameter count shows that the
  **size/diameter-extrapolation** gap survives capacity, depth, and training-fairness controls
  (~7000-9000x at 10x10 from a 6x6-trained baseline), while the in-distribution gap is partly
  an optimisation effect. Multi-seed curves
  (`extrapolation_curve`) and a **held-out effective-resistance task** - a non-linear functional
  of `A⁻¹` using its diagonal, not the routed field - broaden the result past the exact function
  the solve computes; the extrapolation gap holds there too.
- The junction kernel is a per-node Python loop (no batching yet), so grids are kept **modest**
  (≤ ~6×6 in the sweep, `b=1`); this is a capability demo, not a speed benchmark. Batching it
  (roadmap T6) would scale it up.
- Numbers are BLAS/thread-dependent diagnostics; the **test** (`tests/test_maze_grid.py`)
  asserts the machine-independent facts: labels equal the dense solve, the learned precision is
  SPD with `kappa < 1e3`, and `gabp` beats `local` by a wide margin (`gabp < 0.1 * local`).
