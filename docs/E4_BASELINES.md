# Matched-capacity long-range benchmark (Track B / E4)

**Current scientific status (2026-07-22).** This file records exploratory controls that
motivated the preregistered study; it is not the final verdict. The matched-fit HOLDOUT
analysis found that the truncated comparators generally did not attain comparable training
fit in the hard cells. It therefore rejected the intended inductive-bias interpretation and
supports only a hedged attainability reading under the tested budgets. One preregistered
negative-control clause also failed. See `archive/docs/CONFIRMATORY_RESULTS.md` and Paper 1.

This note pre-registers and records the experiment that hardens the maze from a
*capability ablation* into a *matched-capacity* result. Code:
`gabp_sparse_inv/demos/maze_baselines.py`; tests: `tests/test_maze_baselines.py`.

## What it hardens

The grid maze ([MAZE.md](MAZE.md)) compares one global `junction_solve` against a fixed
`K=2`-hop averaging model. The local model is range-limited **by construction**, so its
failure is structural - an *architectural ablation*, not evidence the inverse buys a
capability a fair learner couldn't. The honest question:

> Does the exact selected inverse buy something a matched-or-over-matched **universal**
> long-range learner cannot get from data?

## Pre-registration (fixed before running)

- **Baselines**, both on the *same* strictly-local per-node inputs `[phi_v, deg_v, b_v]`:
  a deep residual **message-passing GNN** (depth ≥ in-distribution diameter - a fair,
  unrolled-iterative competitor) and a **full-attention Transformer** with non-parametric
  2-D sinusoidal positional encoding (global reach at every layer; the strongest fixed
  baseline). Both vastly **over-parameterised** vs the 65-param GaBP encoder.
- **Axes.** (1) *Extrapolation* (headline): train at one grid size, evaluate on larger
  ones; the GaBP encoder is size-independent and the solve is exact at any size, so its
  weights transfer and should stay accurate; a fixed-capacity learner trained small should
  degrade as test diameter grows. (2) *Capacity* (control): vary baseline width at a fixed
  size; the gap should not close with capacity.
- **Falsification bar.** The claim is **falsified** if a parity-or-greater-capacity baseline
  lands within **2×** of GaBP both in-distribution *and* across the extrapolation sweep
  (⇒ fold the maze into the methods paper as an efficiency/exactness demo). It **survives**
  if a sustained extrapolation gap persists and does not close as capacity grows.

## First read (CPU, fp64; medians over 3 seeds; `python -m gabp_sparse_inv.demos.maze_baselines`)

Train 6×6, evaluate 6×6 → 10×10. Params: GaBP 65, GNN 22 337 (344×), Transformer 18 305 (282×).

| test grid | diameter | gabp | gnn | transformer | predict-mean |
|----------:|---------:|-----:|----:|------------:|-------------:|
| 6×6 | 10 | **2.1e-4** | 1.7e-2 | 2.4e-2 | 2.5e-2 |
| 8×8 | 14 | **1.3e-4** | 5.1e-2 | 7.9e-2 | 8.1e-2 |
| 10×10 | 18 | **8.5e-5** | 1.0e-1 | 1.4e-1 | 1.4e-1 |

GaBP stays flat (~1e-4 at every size); both baselines degrade monotonically to the
predict-mean floor as the diameter grows; per-seed ranges never overlap.

**Capacity control** (in-distribution 6×6, width 16→128, up to ~350k params): the gap does
**not** close - GNN saturates at best `~4e-3`, Transformer at best `~1.4e-2`, neither
approaching GaBP's `~1e-4`. (The widest models drifting toward predict-mean is an
optimisation artifact, not "bigger is worse"; the honest reading is the *plateau* - capacity
is not the bottleneck.)

**Depth (receptive-field) control** (in-distribution, GNN depth = grid diameter - an unrolled
iterative solver whose reach spans the whole grid): still **234× / 266×** worse than GaBP at
8×8 / 10×10 (GNN `~2e-2`, 30-39k params, vs GaBP `~9e-5`). So the gap is **not** merely
"receptive field < diameter" - even with full reach a matched-depth learned solver falls 2+
orders short, pushing the reading from *efficiency* toward *capability*.

**Training-fairness control** (the main confound - deep nets are hard to train). Giving the
baselines 10× more steps and a second lr *does* narrow the **in-distribution** gap: the GNN
falls `4.9e-3 → 3.5e-4` (≈180× → ≈13× GaBP) over 300 → 3000 steps and is *still descending*. So
the in-distribution gap is **substantially an optimisation/efficiency effect** - not headlined.
But the same well-trained (2000-step) baselines, evaluated out-of-distribution, *still* collapse:
**2728× / 6879×** (GNN) and **3844× / 9325×** (Transformer) worse than GaBP at 8×8 / 10×10.
Better in-distribution training does not buy extrapolation.

**Exploratory read before HOLDOUT:** the largest observed gap is on the
**size/diameter-extrapolation** axis - an exact sparse-direct primitive with *size-independent*
parameters generalises across problem sizes (GaBP flat `~1e-5` from 6×6 to 10×10), while a
universal long-range learner - even at >250× capacity and 10× training - cannot (≈7000-9000×
worse at 10×10). The *in-distribution* gap is partly trainable away (efficiency), so the honest
headline is **length/size generalisation**, not raw in-distribution fit. This supports an
application-forward paper centred on extrapolation.

## Multi-seed curves + held-out task (the remaining controls, now done)

**Multi-seed extrapolation curve.** `extrapolation_curve` / `run_curve` run the
train-small/evaluate-larger sweep over several seeds and report `median [min, max]` per model
per grid, so the headline claim - GaBP flat and below every baseline, *spreads non-overlapping*
at the largest grid - is a seed-robust statement, not a lucky seed. The separation reproduces
the single-seed table above (GaBP ~`1e-4`-`1e-5` flat; baselines climbing to the predict-mean
floor), now with explicit per-seed ranges.

**Held-out task: effective resistance.** The routing label *is* `junction_solve`'s output, so a
fair worry is that the GaBP model merely *has* the exact function. To broaden past that, the
held-out task regresses the **effective resistance**
`R(v, s) = (A^{-1})_vv - 2 (A^{-1})_vs + (A^{-1})_ss` - the canonical global graph metric, a
*non-linear* functional of the inverse that also needs its **diagonal** (`selinv_junction`), not
a single solve column. The GaBP model (`GaBPResistance`) contains the two selected-inverse ops
and computes `R` from them; the same GNN/Transformer baselines regress it from the same local
features. The label is exact (matches a dense fp64 inverse to `~1e-15`; `R(s,s)=0`, `R≥0`,
correlation `~0.8` with grid distance - a genuine long-range quantity).

`python -m gabp_sparse_inv.demos.maze_baselines` prints both curves. CPU, fp64; median over 3
seeds; train 6×6, evaluate 6×6 → 10×10 (diagnostics):

Params: GaBP 65, GNN 22 337, Transformer 18 305 (same baselines as routing).

| test grid | diameter | gabp (median [min, max]) | gnn | transformer | predict-mean |
|----------:|---------:|-------------------------:|----:|------------:|-------------:|
| 6×6 | 10 | **2.1e-3** [5e-4, 1e-2] | 8.3e-2 | 7.5e-2 | 2.1e-1 |
| 8×8 | 14 | **2.0e-3** [4e-4, 1e-2] | 9.3e-2 | 9.9e-2 | 1.9e-1 |
| 10×10 | 18 | **2.2e-3** [4e-4, 1e-2] | 8.4e-2 | 1.2e-1 | 1.9e-1 |

GaBP stays **flat** (`~2e-3`) from 6×6 to 10×10 - the extrapolation signature - at ~40× below
the baselines, whose per-seed spreads never reach it; both baselines sit well above
predict-the-mean's `~2e-1` only because they fit near-source structure. The resistance gap is
**smaller** than routing (~40× vs ~1000×) - honestly, the baselines capture some of `R`'s smooth
distance structure and the non-linear functional is harder for the GaBP encoder to nail than the
linear field - but the **extrapolation** gap is just as flat. So the inductive-bias result is not
specific to the exact `A^{-1} e_s` the solve computes: it carries to a different, non-linear,
diagonal-using global functional.

## Matched-capacity control (within-architecture intervention)

The GNN/Transformer comparisons above match (over-match) *capacity* but differ in
*architecture*, so a residual worry is that the gap reflects architecture, not the inverse
specifically. The cleanest fix is an intervention that holds the model **identical** and varies
**only** the solve's globality. `causal_solve_sweep` (`demos/maze_grid.py`) does exactly this:
the same `GaBPMazeGrid` encoder and parameter count, with its exact `junction_solve` replaced by
`K` damped-Jacobi sweeps on the *same* learned precision `A` - one matvec per sweep, so `K` steps
reach exactly `K` hops and `K→∞` recovers the exact solve. Architecture,
capacity, inputs, data, optimiser, and steps are fixed; the intervened variable is reach.

Train 6×6, evaluate 6×6 (CPU, fp64, seed 0, 200 steps; the same model, only `K` varies):

| solve reach `K` | 1 | 2 | 4 | 8 | 16 | 32 | exact |
|----------------:|--:|--:|--:|--:|---:|---:|------:|
| test MSE | 3.0e-1 | 2.6e-1 | 2.1e-1 | 1.6e-1 | 1.0e-1 | 5.1e-2 | **4.0e-5** |

(predict-the-mean floor: `2.5e-2`.) The dose-response is **monotone** in reach, and the exact
(global) solve is **~10³× below the deepest truncation** and the *only* variant that beats the
predict-mean floor - every bounded-reach version is *worse* than trivial, because a truncated
solve produces a confidently-wrong near-source field. Within this intervention, the result
supports attributing the gap to the inverse's **globality**, not to architecture or parameter
count: the matched-capacity attribution claim the cross-architecture baselines could only
approximate.
Gated in `tests/test_maze_grid.py`
(`test_matched_capacity_causal_dose_response`); figure `maze_causal` in `archive/paper/make_figures.py`.

## Scope and honest caveats (what this does *not* yet show)

- **Claim scope.** This says the selected inverse is the right **inductive bias for
  solve-shaped tasks** - when the target is an exact structured global linear solve, a model
  that *contains* that solve nails it and generalises across sizes, while generic
  high-capacity learners approximate it poorly. It does **not** say GNNs/Transformers are
  weak models in general. Note the task favours the solve by construction: the label *is*
  `junction_solve`'s output and the GaBP model *contains* `junction_solve`, so it is not
  *learning* the solve - it *has* it. That is the point (inductive bias), but it must be
  framed as such, not as general superiority.
- **Baseline training fairness - now controlled.** A 10×-steps × 2-lr sweep narrows the
  *in-distribution* gap (GNN to ~13× GaBP, still descending) but leaves the *extrapolation*
  gap intact (~7000-9000× at 10×10 with 2000-step baselines). So the in-distribution gap is
  largely optimisation/efficiency; the extrapolation gap is training-invariant.
- **Depth/receptive-field - now controlled.** A GNN with depth = diameter (an unrolled
  iterative solver, 30-39k params) still loses by ~250× in-distribution at 8×8/10×10, so the
  gap is not explained by receptive field. What remains is the *optimisation* confound - deep
  GNNs are hard to train, so a fuller baseline step-count/lr sweep is still owed before this
  is paper-grade.
- **Beyond pure routing - now addressed.** The **effective-resistance** held-out task (below)
  broadens the claim past reconstructing `A^{-1} e_s`: a non-linear functional of the inverse
  that also uses its diagonal (`selinv_junction`), where the same extrapolation gap holds.

## Remaining controls before paper-grade

1. ~~Multi-seed extrapolation curves + a held-out long-range task beyond `A^{-1} e_s`~~:
   **done** (see "Multi-seed curves + held-out task" below): `extrapolation_curve` /
   `run_curve` aggregate median + per-seed spread, and the **effective-resistance** task
   broadens the claim past the exact function any single solve computes.
2. ~~A real fixed-point / DEQ impact demo where the inverse is load-bearing~~ - **done**
   ([DEQ.md](DEQ.md)): the non-symmetric sparse-direct solve as an exact DEQ backward,
   compared with finite Neumann truncations on a controlled sweep.
3. ~~The architecture confound (the gap could be architecture, not the inverse)~~ - **done**:
   the within-architecture **matched-capacity control** (above) holds the model identical
   and varies only the solve's reach, giving a monotone dose-response with only the exact global
   solve succeeding - evidence for attribution to the inverse's globality, not architecture.
4. Architecture variants (relative position encodings, more heads/layers) for completeness;
   width and training budget are already controlled. *(Still open - minor.)*
