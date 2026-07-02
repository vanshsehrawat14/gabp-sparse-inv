# Phase-1 confirmatory findings

Status: `CONFIRMATORY`. HOLDOUT seeds `1000-1029` (30 seeds). Date: 2026-06-30.

Frozen-preregistration results on data the exploration never touched. Design and
decision rules: [PREREGISTRATION_CONFIRMATORY.md](PREREGISTRATION_CONFIRMATORY.md).
Exploratory menu (DEV, hypothesis-generating only):
[`_phenomenon_menu.md`](_phenomenon_menu.md). Machine-readable results:
`results/CONFIRMATORY/confirmatory_analysis/confirmatory_results.json` (ignored tree).

Meaningful-effect threshold `δ = 1e-3`; a cell counts iff its 95% paired-bootstrap CI
lower bound (`reps = 2000`, `seed = 314159`) clears `δ`.

## Headline

**H3 (gradient-channel isolation) is CONFIRMED on HOLDOUT.** With the DEQ forward
solve held bit-identical in both arms, training with a biased Neumann-`K` implicit
adjoint yields a worse exact-forward held-out function than training with the exact
adjoint, at `ρ = 0.99`: all three pre-registered primary cells (`K ∈ {1,2,4}`) clear
`δ`, the learned functions provably differ, the low-`ρ` control is silent, and the
forward channel is bitwise-identical. The effect replicated and, if anything,
strengthened relative to DEV. **H1 (maze cost)** holds as supporting context, with its
large-`ε` control failing as pre-registered — a disclosed caveat (below).

## H3 — gradient-channel isolation (primary): CONFIRMED

Verdict: **`H3_CONFIRMED`** — 3/3 primary cells confirmed; low-`ρ` control silent;
forward bitwise-identical (worst `max_abs = 0`). 30 seeds.

Primary cells (`ρ = 0.99`). `Δtest = test_loss(Neumann-K) − test_loss(exact)`, both
evaluated under the exact forward (positive ⇒ exact-adjoint training generalizes
better); `fn rel-MSE` is the held-out function distance under the common exact forward:

| K | Δtest | 95% CI | fn rel-MSE | fn CI low | ≥ δ & functions differ? |
|---:|---:|---|---:|---:|:--:|
| 1 | +7.06e-2 | [+3.40e-2, +1.14e-1] | 7.75e-3 | 4.39e-3 | yes |
| 2 | +1.33e-1 | [+7.42e-2, +2.00e-1] | 5.71e-3 | 3.83e-3 | yes |
| 4 | +9.64e-2 | [+5.45e-2, +1.44e-1] | 2.90e-3 | 1.94e-3 | yes |

Every primary CI lower bound clears `δ = 1e-3` by ~1.5 orders, and every
function-distance CI excludes 0 — the learned functions genuinely differ under a
common exact forward, ruling out the "different weights, equal test loss" trap. The
DEV means were `+6.0e-2 / +1.1e-1 / +7.7e-2`; the HOLDOUT effect is the same shape and
slightly larger.

Low-`ρ` control (`ρ ∈ {0.5, 0.7}`, all `K`) — required silent: every cell's `Δtest`
CI lower bound is far below `δ`. The largest is `~2e-9` (at `ρ = 0.5`, ~6 orders below
`δ`); the `ρ = 0.7` cells straddle 0. The DEV red flag at `(ρ = 0.7, K = 1)` did **not**
recur (its HOLDOUT CI now includes 0). Control clean.

## H1 — maze cost / infeasibility (supporting context)

H1 has no headline gate; it inoculates against "the baseline was simply under-budgeted."

Exact − matched-B2 test-MSE gap (negative ⇒ exact better), 30 seeds:

| size | gap | 95% CI |
|---:|---:|---|
| 6×6 | -1.75e-1 | [-1.76e-1, -1.75e-1] |
| 12×12 | -1.90e-2 | [-1.92e-2, -1.88e-2] |
| 24×24 | -3.62e-3 | [-3.67e-3, -3.57e-3] |
| 48×48 | -8.22e-4 | [-8.37e-4, -8.08e-4] |

The `48×48` gap (`-8.22e-4`) matches DEV (`-8.21e-4`) almost exactly. Break-even
(smallest B2 Jacobi budget whose test MSE enters the exact band): shape
`diverging_candidate` — B2 hits the K-bank cap (256) at `6×6` without entering the
band, and only reaches the band at `K = 256` for the larger sizes; `C*` grows
`1.97e5 → 8.36e5 → 3.44e6 → 1.40e7` FLOPs across sizes. Matched-budget B2 cannot
cheaply match exact as the maze hardens.

**Large-`ε` control (`ε = 2.0`): `AUTO-FAIL`** at every size (exact still beats matched
B2 at `6/12/24/48`). This is the **pre-registered disclosed caveat**: a residual
exact-solve advantage survives even where propagation should be short-range, so the
maze cost story is not purely reach-limited — which is exactly why H1 is context, not a
headline. It has **no** bearing on the H3 verdict.

## Provenance & integrity

- The design was frozen before any HOLDOUT seed was touched
  ([PREREGISTRATION_CONFIRMATORY.md](PREREGISTRATION_CONFIRMATORY.md)); the analysis
  code was frozen and validated against the published DEV numbers (bit-identical CIs)
  before the run.
- A pre-run canary on seed `1000` revealed the confirmatory runner invoked the maze
  control regime with the positive `ε` (0.05) instead of the pre-registered `ε = 2.0`.
  The runner was corrected to match the prereg and the full HOLDOUT was run **fresh**
  under the corrected code. **No decision rule, threshold, or cell was changed**; the
  fix was driven by the config bug, not by any observed effect size.
- HOLDOUT run once, 30 seeds, across 6 single-threaded workers over disjoint seed
  partitions (deterministic per seed). **H2 was excluded** from confirmation (its DEV
  symmetric-swap control fails, so its operator-swap asymmetry is not exactness-specific).

## What this means

`H3_CONFIRMED` is the strong outcome: the paper's central empirical pillar — exactness
as an inductive bias, isolated to the gradient channel — holds on untouched data,
supported by the H1 cost story with its large-`ε` caveat disclosed. The at-scale
program (E1/E2/E3) remains gated on GPU timing (G2); this mechanism result stands
independently of it.
