# Application: differentiable hierarchical tree GMRF (M2)

This note records the application that turns the differentiable selected inverse
(`docs/derivations.md` §8) into an end-to-end result: **hyperparameter learning for a
hierarchical (tree-structured) Gaussian Markov random field**, where every quantity is
computed in `O(n)` by this package's tree kernels and a dense-autograd baseline is
`O(N^3)` time / `O(N^2)` memory and becomes infeasible. Code: `gabp_sparse_inv/gmrf.py`;
scaling study: `gabp_sparse_inv/bench/gmrf_scaling.py`; tests: `tests/test_gmrf.py`.

## Model

A latent multiresolution Gaussian field on a rooted tree, with Gaussian observations:

    x_root ~ N(0, root_prec^{-1} I),
    x_v | x_{p(v)} ~ N(a x_{p(v)}, kappa_v^{-1} I)   (branching AR / innovation process),
    y_v = x_v + eps_v,   eps_v ~ N(0, sigma2 I).

The latent precision `Q(theta)` is **tree-structured** (sparse on the tree pattern):

    Q_vv = (kappa_v if v != root else root_prec) + sum_{c in ch(v)} a^2 kappa_c,
    Q_{p(v), v} = -a kappa_v,

with hyperparameters `theta = (a, kappa, root_prec, sigma2)`. `Q` is SPD by construction
(it is the `L D L^T` reconstruction of the directed model).

## What is computed, and why it is tree-tractable

For `y ~ N(0, Q^{-1} + sigma2 I)` with posterior precision `Q_post = Q + sigma2^{-1} I`
(also tree-structured), the evidence identity gives

    log p(y) = 0.5 log det Q + 0.5 log det R - 0.5 log det Q_post
               - 0.5 (y^T R y - (R y)^T Q_post^{-1} (R y)) - (N/2) log 2pi,   R = sigma2^{-1} I.

Every term is a tree log-determinant (`tree_logdet`, the SPD-pivot product) or a tree
linear solve (`tree_solve`, a two-pass forward-eliminate / back-substitute). The
posterior **marginal variances** `diag(Q_post^{-1})` come from the selected inverse
(`selinv_tree`).

The two learning objectives stress different parts of the theory:

| Objective | Needs | selinv case |
|-----------|-------|-------------|
| `marginal_log_likelihood` (type-II MLE) | `log det Q`, `log det Q_post`, one solve | log-det (classical) |
| `posterior_marginal_variances` calibration | `diag(Q_post^{-1})` and `d/dtheta` of it | **general** adjoint (§8) |

The second objective is the one that *requires* the general selinv adjoint: its gradient
w.r.t. `a` depends on the off-diagonal precision blocks through `G_edge`, which the
log-det gradient alone does not provide. In the validation its coupling gradient is
nonzero and matches dense autograd to ~5e-17.

## Validation (`tests/test_gmrf.py`)

All against dense oracles, fp64: `Q` SPD with the correct tree pattern; `tree_logdet`
vs `torch.logdet` (~1e-15); `tree_solve` vs `torch.linalg.solve` (~1e-15);
`marginal_log_likelihood` vs a dense multivariate-normal log-prob (~1e-15);
`posterior_marginal_variances` vs `diag(inv(Q_post))` (~1e-16); **gradients** of both
objectives vs dense autograd (marginal likelihood ~1e-14; posterior variance ~5e-17,
with a nonzero coupling gradient); and **hyperparameter recovery** on synthetic data.

## Scaling result (`bench/gmrf_scaling.py`)

Balanced binary tree, scalar nodes, 16 iid fields, one marginal-likelihood
forward+backward step, CPU / 1 thread (fp64). Two tree paths - the per-node reference
loop and the level-set **batched** schedule (`batched=True`) - vs a dense-autograd
`O(N^3)` baseline (assemble `Q`, `inv`, MVN log-prob, backprop):

| n | tree loop | tree batched | dense fwd+bwd | batched vs dense | grad error |
|------:|----------:|-------------:|--------------:|-----------------:|-----------:|
| 127 | 0.16 s | 0.013 s | 0.03 s | **2.2x** | 2e-12 |
| 255 | 0.31 s | 0.016 s | 0.08 s | **5.2x** | 3e-12 |
| 511 | 0.63 s | 0.021 s | 0.35 s | **17x** | 6e-12 |
| 1023 | 1.29 s | 0.028 s | 3.19 s | **113x** | 6e-12 |
| 2047 | 2.97 s | 0.046 s | infeasible | - | 3e-11 |

Log-log time slopes: **loop 1.05** (linear), **batched 0.45** (the Python loop is now
over the `O(log n)` height levels, not the `n` nodes, so launch latency is amortized),
**dense 2.24** (trending to the cubic `O(N^3)`). The **batched** path beats dense at every
size in this table; the per-node loop does **not** (n=127: loop 0.16 s vs dense 0.03 s).
The table used **`--samples 16`**; the script default is 64 - regenerate with that flag to
reproduce these numbers. The posterior-variance calibration objective (the general
dense baseline is both slower and, via its `O(N^2)` dense inverse + autograd tape, headed
for memory infeasibility. The posterior-variance calibration objective (the general
adjoint) runs in <1 s at `n = 2047`, a size the dense baseline cannot reach.

**Honest caveats.** The asymptotic wins are time `O(n)` vs `O(N^3)` and memory `O(n)` vs
`O(N^2)`. The per-node reference loop is launch-latency-bound on scalar blocks (slower than
dense below `n ~ 1000` in this run); the batched schedule removes that caveat on CPU and is
the intended scale path (GPU timing not benchmarked). The loop is retained as the correctness
reference (batched is validated block-for-block against it). Numbers are BLAS/thread/device
dependent diagnostics from one CPU / 1-thread fp64 run - recorded, not asserted in CI.

## Scope (M2) and outlook

M2 delivers the model, exact tree-tractable objectives, gradient correctness, recovery,
the scaling study, and the level-set batched objectives (`batched=True`) that remove the
small-`n` constant-factor caveat and are GPU-ready. Out of scope (named follow-ups):
CUDA-device timing of the batched path and larger block sizes `b > 1`. The loopy/grid GMRF
via the junction-tree generalization is now **done** (next section), and second-order
training (double-backward / HVPs) works for free through the functional junction kernels
(`tests/test_double_backward.py`).

## Loopy (grid) GMRF via the junction kernel

The junction-kernel counterpart of the tree GMRF (`gabp_sparse_inv/gmrf_grid.py`): the same
observation model and learning objectives on an **arbitrary (loopy) graph** - the
4-neighbour grid being the motivating case the tree kernel cannot handle. The latent
precision is a conditional-autoregression (CAR) model `Q(kappa, a) = kappa (I + a L)` with
`L` the graph Laplacian, `kappa > 0`, `a >= 0`; it is strictly block-diagonally dominant
(`Q_vv - sum_j |Q_vj| = kappa`), hence SPD, with `Q_vv = kappa (1 + a deg_v)` and edge
blocks `Q_{ij} = -kappa a`. Built by `grid_gmrf_precision(rows, cols, kappa, a)` in the
junction (`edge_index`/`edge_val`) layout.

| Objective | Junction kernels used |
|---|---|
| `junction_marginal_log_likelihood` (type-II MLE for `y ~ N(0, Q^{-1} + sigma2 I)`) | `junction_logdet` × 2 + one `junction_solve` |
| `junction_posterior_marginal_variances` (`diag(Q_post^{-1})`) | `selinv_junction` (the general filled-pattern adjoint) |

Both are written over the junction layout (any graph, not just grids) and take an optional
shared elimination `order` (e.g. `elimination_order_nested_dissection`); the result is
invariant to it (only the fill changes). `fit_grid_marginal_likelihood` learns
`(kappa, a, sigma2)` in softplus-unconstrained space. Validation
(`tests/test_gmrf_grid.py`), all against dense fp64 oracles on the loopy grid: `Q` SPD with
the correct CAR structure; the marginal likelihood vs a dense multivariate-normal log-prob
(~1e-9); posterior variances vs `diag(inv(Q_post))` (~1e-9); **gradients** of both
objectives vs dense autograd (~1e-7); order-invariance (default min-degree vs nested
dissection); and hyperparameter recovery from sampled data. This is the first *statistical*
payoff of the junction keystone - the loopy generalization of the M2 tree-GMRF.

## Exact Gaussian sampling from a structured precision (M-JOSS)

`sample_gaussian_tree(diag, edge, parent, num_samples)` and
`sample_gaussian_junction(diag, edge_index, edge_val, num_samples)` draw
`x ~ N(0, A⁻¹)` from any SPD **precision** `A` in the tree or junction layout, reusing the
same `LDL^T` factorization as the selected inverse / solve. With `A = L D L^T`,
`D_v = C_v C_vᵀ` and `M = L C`, the sampler applies `x = M⁻ᵀ z` for `z ~ N(0, I)` (a per-node
`C_v⁻ᵀ` solve then the `Lᵀ` back-solve), so `Cov(x) = (M Mᵀ)⁻¹ = A⁻¹` **exactly**. Samples
are shaped `[num_samples, …, n, b]` (the draws are the trailing axis, one vectorized pass).
The covariance is verified exactly by applying the deterministic transform to the standard
basis (`T Tᵀ = A⁻¹`, no Monte-Carlo noise); a many-draw empirical-covariance check is a
sanity gate; and the junction sampler reproduces the tree sampler block-for-block on a tree
pattern. This is the sampling sibling of the log-determinants (`tree_logdet` /
`junction_logdet`) - together the statistical ops the roadmap names for M-JOSS. Distinct
from `sample_tree_gmrf`, which samples a *specific* hierarchical GMRF; these sample from an
*arbitrary* precision given in the kernel layout.

## Traces of the inverse: `trace(A⁻¹)` and `trace(A⁻¹B)`

Both are read straight off the selected inverse - no extra kernel. With `(G_diag, …)` (and
`(S_index, G_lower)` for the junction kernel) the outputs of `selected_inverse_tree` /
`selected_inverse_junction`:

* **`trace(A⁻¹) = Σ_v tr(G_vv)`** - the trace touches only the diagonal blocks, which the
  selected inverse already returns:

  ```python
  tr_Ainv = torch.diagonal(G_diag, dim1=-2, dim2=-1).sum((-1, -2))
  ```

* **`trace(A⁻¹B)`** for any `B` supported on the (filled) selected pattern `S` is a masked
  dot product: `trace(A⁻¹B) = Σ_{i,j} tr((A⁻¹)_{ij} B_{ji})`, and only blocks on `S`
  survive - all of which are computed. For symmetric `A`, `B` on `S` (diagonal blocks
  `Bd`, lower blocks `Bl` on the same `S_index`, junction layout):

  ```python
  tr_AinvB = (G_diag * Bd).sum() + 2.0 * (G_lower * Bl).sum()   # diag once, each off-pair twice
  ```

  This is the building block for the Hutchinson-free score / Fisher terms `tr(A⁻¹ ∂A)` that
  appear in Gaussian marginal-likelihood gradients - here exact, not stochastic, because the
  derivative `∂A` lives on `A`'s own pattern. (It is the closed form behind the same
  log-likelihood gradients that the GMRF objectives above obtain via autograd.)
