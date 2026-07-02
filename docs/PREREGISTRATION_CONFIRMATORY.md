# Confirmatory preregistration — exactness as an inductive bias

**Status: `CONFIRMATORY`, FROZEN. Written before any HOLDOUT seed was touched.**
This document freezes the subset of the exploratory hypothesis menu
([PREREGISTRATION.md](PREREGISTRATION.md)) that we commit to testing on the
reserved HOLDOUT seeds `1000-1029`. It is the only file whose results may back a
paper claim. It supersedes nothing in the exploratory menu; it selects from it.

The exploratory ranking that motivates this subset is recorded, honestly and
with all caveats, in [`_phenomenon_menu.md`](_phenomenon_menu.md). Per the
handoff rule in [PREREGISTRATION.md](PREREGISTRATION.md) §5, that ranking was
built on DEV seeds `0-19` only; every number below that comes from DEV is used
solely to *set thresholds and select cells in advance*, never as evidence.

Date frozen: 2026-06-30.

## 1. Frozen subset

| Hypothesis | Role | Included |
|---|---|---|
| **H3** gradient-channel isolation | **Primary confirmatory claim** | Yes |
| **H1** maze cost / infeasibility | **Supporting context** (not a headline) | Yes |
| **H2** operator-swap co-adaptation | — | **No** |
| **H4** negative controls | **Required gates** on H3 and H1 | Yes |

**Why H2 is excluded.** H2 is the most novel-looking DEV effect, but its own
symmetric-swap negative control `AUTO-FAIL`s on DEV: two equal-residual
approximations (Jacobi-K vs CG-K′) each prefer their own operator, so the
operator-swap asymmetry may be generic solver-form co-adaptation rather than
anything specific to exactness. Confirming H2 in its current form would spend
HOLDOUT on a claim its own control does not support. It is deferred until the
control is repaired or the mechanism is understood (a future DEV effort), not
tested here.

## 2. Seed protocol

- **HOLDOUT seeds:** exactly `1000, 1001, …, 1029` (`gabp_sparse_inv.bench.seeds.HOLDOUT_SEEDS`,
  30 seeds). No subset selection, no seed dropping, no addition.
- **DEV seeds `0-19` are forbidden** in every confirmatory run. The two-way guard
  `assert_seed_for_run(seed, "CONFIRMATORY")` raises on any DEV seed and on any
  non-HOLDOUT seed.
- **Run tag:** `CONFIRMATORY` in every output path, JSON payload, and log line.
- **Output paths** (all under the ignored `results/` tree):
  - H3: `results/CONFIRMATORY/deq_gradient_isolation/seed{1000..1029}.json`
  - H1: `results/CONFIRMATORY/maze_extrapolation/{positive,control}_seed{1000..1029}.json`
  - Analysis: `results/CONFIRMATORY/confirmatory_analysis/`
  - Human-readable verdict: `docs/_phase1_findings.md` (committed).
- **One run.** Each probe is run **once** on the full HOLDOUT set. There is no
  re-run-and-pick. If a run aborts mid-way, the resumable per-seed files are
  completed with the identical config; nothing already written is recomputed.

## 3. Shared statistics (frozen)

- **Estimator:** the mean per-seed paired difference across the 30 HOLDOUT seeds.
- **Interval:** paired bootstrap, `reps = 2000`, `bootstrap_seed = 314159`,
  `alpha = 0.05` (a 95% percentile CI), exactly
  `gabp_sparse_inv.bench.phase1_analysis.paired_bootstrap_ci`. The confirmatory
  outputs relabel `ci_mode` to `CONFIRMATORY_PAIRED_BOOTSTRAP`; the arithmetic is
  identical to the exploratory function.
- **Meaningful-effect threshold `δ = 1e-3` (frozen).** A cell shows a *meaningful
  positive effect* iff its 95% CI lower bound `ci_low ≥ δ`. Rationale, fixed in
  advance from DEV: `δ` sits ~1.5 orders below the smallest DEV primary effect
  (`~6e-2` at `ρ=0.99`) and ~3–4 orders above the largest DEV low-`ρ` control gap
  (`~1.4e-7`), so the same single threshold cleanly separates the claimed effect
  from the control. `δ` is used identically for the H3 claim and for the H3
  low-`ρ` control; it is not re-tuned after looking at HOLDOUT.
- **Sign convention:** losses are lower-is-better. For H3 `delta_test_loss =
  neumann_test_loss − exact_test_loss` (positive ⇒ exact-adjoint training
  generalizes better ⇒ the H3 direction). For H1 `exact − b2_matched` (negative ⇒
  exact better).

## 4. H3 — primary confirmatory claim

**Hypothesis (directional).** With the DEQ forward solve held **bit-identical**
in both arms, training the scalar gain with a biased Neumann-`K` implicit adjoint
yields a **worse exact-forward held-out function** than training with the exact
adjoint, at high `ρ`. Gradient exactness alone — not any forward-solve
difference — changes the learned function.

**Probe.** `gabp_sparse_inv.bench.deq_gradient_isolation`, run on HOLDOUT with the
`CONFIRMATORY` tag. Config is **identical to the DEV run** (no re-tuning):
`n = 16`, `steps = 120`, `lr = 0.05`, `n_train = 32`, `n_test = 32`,
`input_scale = 0.2`, linear affine chain, Adam. `ρ ∈ {0.5, 0.7, 0.85, 0.9, 0.95,
0.98, 0.99}`, `K ∈ {1, 2, 4, 8, 16, 32}`.

**Arms.** Both arms use the exact forward solve `z = (I − gain·W0)⁻¹ b`
(`nonsym_junction_solve`). The only intervention is the backward rule used while
training the gain: `exact` (exact implicit adjoint) vs `neumann_K` (biased
K-term Neumann adjoint). Both trained gains are evaluated under the **same exact
forward** operator.

**Metrics.**
- Primary: `delta_test_loss` (defined above), under exact forward.
- Secondary (anti-trap): `test_function_rel_mse = mean_x ‖f_exact(x) −
  f_neumannK(x)‖² / Var(target)`, the held-out function distance under the common
  exact forward. This defeats the "different weights, equal test loss" trap.

**Pre-declared cells.**
- **Primary cells:** `ρ = 0.99`, `K ∈ {1, 2, 4}` — the DEV-robust positive cells.
- **Low-`ρ` control cells (H4):** `ρ ∈ {0.5, 0.7}`, all `K`.
- **Secondary / reported-only cells:** `ρ = 0.99, K ∈ {8, 16, 32}` and
  `ρ ∈ {0.85, 0.9, 0.95, 0.98}`, all `K`. DEV showed `K` is **non-monotone**
  (`K=8` uncertain, `K=16` negative), so these are reported for completeness and
  are **not** part of the decision rule. Declaring this split now prevents any
  post-hoc cell picking.

**Correctness gate (must pass or the probe is void, not a science result).**
The forward-identity guard must be **bitwise zero** for every cell: the exact and
Neumann backward modes must produce identical forward values at fixed parameters
(`FORWARD_NOT_BITWISE_IDENTICAL` must not fire).

**Decision rule (frozen).**
1. **Effect:** at each primary cell, `delta_test_loss` has `ci_low ≥ δ`
   (meaningful positive effect), **and** `test_function_rel_mse` has `ci_low > 0`
   (the learned functions genuinely differ).
2. **Control:** **no** low-`ρ` control cell (`ρ ∈ {0.5, 0.7}`, any `K`) shows a
   meaningful positive effect (i.e. every such cell has `delta_test_loss ci_low <
   δ`). The DEV red flag at `(ρ=0.7, K=1)` was `~1.4e-7`, far below `δ`; under
   this rule it does not count as a control failure, but the raw number is
   reported.
3. **Outcome ladder** (pre-declared):
   - **H3 CONFIRMED** iff the correctness gate passes, the control passes, and
     **all three** primary cells satisfy (1).
   - **H3 PARTIAL** (report, not a headline) iff exactly **two** primary cells
     satisfy (1) with the gate and control passing.
   - **H3 NOT CONFIRMED** otherwise (including any control or correctness-gate
     failure).

## 5. H1 — supporting context (not a headline)

**Role.** H1 is **reviewer inoculation**, not the paper's central claim: it shows
the truncated global baseline (B2) was not simply under-budgeted, and that
matching exact becomes expensive as the maze hardens. It is expected to be real
but unsurprising (it collides with graph diameter / conditioning / oversquashing),
which is exactly why it is context and H3 is the headline.

**Probe.** `gabp_sparse_inv.bench.maze_extrapolation`, `regime = positive`, on
HOLDOUT with the `CONFIRMATORY` tag. Config identical to DEV: train side `S = 6`,
`mults = (1,2,4,8)` ⇒ eval sizes `6/12/24/48`, `eps = 0.05`, `hidden = 16`,
`steps = 250`, `lr = 0.02`, `n_train = n_test = 48`, B2 = Jacobi, matched budget
`K_match = n_match(...)`, K-bank up to `256`, matched-compute target
`solve_adjoint`.

**Metric and directional predictions.**
- `exact − b2_matched` test-MSE gap at the hardest size (`48×48`): **negative**,
  95% CI excluding 0 (exact meaningfully better at matched compute).
- Break-even `K*(size)` / `C*(size)`: the smallest B2 budget whose test MSE
  enters the exact band (`exact_median + sd`). Prediction: `K*` **saturates at the
  K-bank cap** at the larger sizes (a `diverging_candidate` shape), i.e. matched-
  budget B2 cannot cheaply reach exact as size grows.

**H1 is descriptive, with no headline gate.** Because H1 is context, its
confirmatory content is the reported gap + break-even curve with CIs. There is no
pass/fail that can "fail the paper"; the paper's confirmed claim rests on H3.

**Required control (H4): large-`eps` maze.** Same probe, `regime = control`,
`eps = 2.0` (short correlation length, so globality should barely matter), on
HOLDOUT. The `_negative_control_check` `AUTO-FAIL` banner is computed and
reported. **Pre-declared honest interpretation:** on DEV this control
`AUTO-FAIL`ed (exact still beat matched B2 even at short range). We therefore
**predict it may `AUTO-FAIL` on HOLDOUT too**, and we commit **now** to reading
that failure as a *disclosed limitation* of the maze cost story — evidence that a
residual exact-solve advantage survives even where reach should be irrelevant —
**not** as support for H1 and **not** something to omit. This disclosed caveat is
the reason H1 stays context. It has **no** bearing on the H3 verdict.

## 6. H4 — required controls, summarized

| Control | Attached to | Rule | On failure |
|---|---|---|---|
| Forward bitwise identity | H3 | every cell `max_abs = 0` | H3 void (probe broken) |
| Low-`ρ` silence (`ρ≤0.7`) | H3 | no cell has `ci_low ≥ δ` | H3 NOT CONFIRMED |
| Large-`eps` maze (`eps=2.0`) | H1 | reported; `AUTO-FAIL` disclosed | H1 caveat (no headline effect) |

H2's symmetric-swap control is **out of scope** (H2 is excluded).

## 7. Analysis-integrity commitments

- The analysis code that implements §3–§6 is written and **unit-tested on
  synthetic and DEV-shaped records before the HOLDOUT run**, and is **not modified
  after inspecting any HOLDOUT output**. (DEV records may be re-analyzed freely;
  they are exploratory.)
- Thresholds (`δ`), cells (primary / control / secondary), bootstrap parameters,
  and the outcome ladder are all fixed above **before** the run.
- The HOLDOUT run is executed once; `docs/_phase1_findings.md` reports whatever it
  produces — CONFIRMED, PARTIAL, or NOT CONFIRMED — with all controls and the
  disclosed H1 caveat.
- No claim in the paper cites DEV numbers as evidence; DEV is acknowledged only as
  the source of the (pre-registered) design choices.

## 8. What the outcomes mean

- **H3 CONFIRMED (+ H1 context):** the paper's central empirical pillar holds on
  untouched data — gradient exactness alone changes the learned function at the
  edge of stability — supported by the maze cost story, with the large-`eps`
  caveat disclosed. This is the strong outcome.
- **H3 PARTIAL:** reported as suggestive, not headline; the paper's empirical
  claim is softened accordingly and the at-scale program (E1/E2/E3, gated on GPU
  timing G2) carries more weight.
- **H3 NOT CONFIRMED:** the inductive-bias claim is not supported on HOLDOUT and
  is **not** made. The library, theory, and demonstrations stand on their own
  (M-JOSS); the paper is reframed away from a confirmed inductive-bias effect.
  The HOLDOUT is then spent and is not reused.
