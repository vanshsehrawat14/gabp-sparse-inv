> **Provenance note (added 2026-07-06, not part of the frozen text).** This is
> the matched-fit confirmatory pre-registration as frozen at commit `00c0bf5`
> (`docs/PREREGISTRATION_CONFIRMATORY.md`, dated 2026-06-29). It is the file
> [`CONFIRMATORY_RESULTS.md`](CONFIRMATORY_RESULTS.md) was produced under; its
> §5/§6 decision rules are the ones that adjudicated that run. The 2026-07-04
> merge reused the same filename for a different (bootstrap) pre-registration,
> clobbering this one at `HEAD`; it is restored here verbatim under a distinct
> name. Disclosure: the HOLDOUT seeds `1000-1029` were consumed twice — first by
> the matched-fit run under this file (2026-06-29), then by the bootstrap run
> under [`PREREGISTRATION_CONFIRMATORY.md`](PREREGISTRATION_CONFIRMATORY.md)
> (2026-06-30) — so that file's "before any HOLDOUT seed was touched" holds
> within its branch but not in wall-clock terms across branches.

# Confirmatory preregistration — attainability / optimization, not inductive bias

**Status: CONFIRMATORY, frozen.** Written after the DEV exploratory pass and the
Task A matched-fit adjudication, and **before any HOLDOUT run**. It contains no
HOLDOUT results. Confirmatory runs use **HOLDOUT seeds `1000-1029` only**; the
seed guard (`assert_seed_for_run`) must raise if a HOLDOUT seed is requested
outside a confirmatory script or a DEV seed inside one. Outputs are tagged
`CONFIRMATORY`. Only results produced under this file and protocol may support
paper claims.

Date: 2026-06-29. Supersedes the handoff stub in
[PREREGISTRATION.md §5](PREREGISTRATION.md); reads against the DEV adjudication in
[`_matched_fit_taskA.md`](_matched_fit_taskA.md).

## 1. The claim under test

DEV exploration (Task A matched-fit) found that the exactness advantage in both
the DEQ gradient-channel probe and the confound-free maze cell is **attainability
/ optimization**: the approximate-operator arm cannot reach the exact arm's
training fit (it is unmatchable), and where it can be matched (near-exact
operator) the learned function collapses to the floor. The apparent
**inductive-bias** reading did not survive matched fit.

This file freezes a test that confirms or refutes that claim on HOLDOUT.

- **Primary (predicted) claim — ATTAINABILITY.** Exact differentiable solves
  change *which optimum is reachable*. The biased-backward / approximate-operator
  arm cannot attain the exact arm's training fit at the edge of stability (high
  `rho` DEQ) or in-distribution (maze), so the exactness advantage is an
  attainability/optimization effect, not a preference between equally-good minima.
- **Rival claim — INDUCTIVE BIAS (predicted to fail).** At matched training fit,
  the exact arm's learned function differs from the approximate arm's, above the
  optimization-noise floor, with the low-difficulty control silent. If this fires
  on HOLDOUT despite the DEV adjudication, the inductive-bias claim is confirmed
  instead.

The test is symmetric: the decision rules below adjudicate whichever claim the
HOLDOUT data supports. The directional prediction from DEV is ATTAINABILITY.

## 2. Probes, arms, knobs, metrics

Mirror the Task A protocol exactly; forward is held exact and bit-identical, only
the backward / operator is swapped. Matching uses the train-set loss under the
**common exact operator** for both arms.

- **DEQ gradient-channel (primary).** Arms: exact-adjoint vs Neumann-`K` adjoint
  on the linear affine chain DEQ, exact forward in both (fixed-parameter forward
  equality asserted bitwise). Knob: `rho`. `K` grid `{1,2,4,8,16,32}`. Primary
  metric: held-out function rel-MSE under the common exact forward. Secondary:
  delta test loss; per-arm train loss (for matchability).
- **Maze confound-free (secondary).** Arms: exact-trained vs `B2(K_match)`-trained
  GaBP encoder, **both evaluated under the exact solve**, at sizes 6/12/24/48.
  Knob: size (in-distribution vs extrapolation). Floor arm: `B2(K=256)` (~exact).
- **Compute** is operator FLOPs, as in the program; not a decision variable here.

## 3. HOLDOUT protocol

- Seeds: the fixed list **`1000-1029`** (30 HOLDOUT seeds), no subsetting, no
  post-hoc seed selection.
- Output paths tagged `CONFIRMATORY` (e.g. `results/CONFIRMATORY/...`).
- Confidence intervals: paired bootstrap, `alpha = 0.05`, 2000 reps, fixed seed —
  now **confirmatory** (not the exploratory label).
- The analysis is the Task A module run under the HOLDOUT seed set; no new degrees
  of freedom are introduced beyond the frozen rules in §4-§5.

## 4. Decision rules (with the two DEV-exposed fixes)

Frozen before any HOLDOUT run.

- **MATCH TOLERANCE = 5%** (unchanged): the approx arm is fit-matched when
  `|train_loss_approx - train_loss_exact| / train_loss_exact <= 0.05`, on the
  approx arm only; the exact arm is never weakened.
- **FIX 1 — minimum-matchability fraction `m_min = 0.60`.** A cell's
  COLLAPSE/SURVIVE verdict is adjudicated **only if at least 60% of the 30 HOLDOUT
  seeds are matchable**. Below `m_min` the cell is **UNMATCHABLE-dominant ->
  attainability/optimization** (a positive primary-claim finding), reported with
  the median relative train-loss gap that proves it. This stops verdicts being
  read off non-representative matchable minorities (the DEV `partial k/20` trap).
- **FIX 2 — optimization-noise floor.** The floor is `F = max(2 x M_null, F_abs)`
  with **`F_abs = 2e-6`**, where `M_null` is the near-exact null cell function
  distance (`K=32` DEQ / `K=256` maze). `F_abs` is the optimization-noise scale
  calibrated on DEV: the largest matched-fit function rel-MSE in the DEV silent
  cells was `1.46e-6` (`rho=0.5, K=1`), so `2e-6` sits just above the observed
  noise. This stops machine-zero null cells (low difficulty, where `M_null` is
  bit-identical) from manufacturing spurious SURVIVE. A per-cell stochastic
  `F_noise` from exact-arm replicas is **not** usable: DEQ training is
  deterministic given the problem seed, so exact replicas are bit-identical;
  `F_abs` is the calibrated stand-in.

## 5. Pre-declared outcomes

Let high difficulty = `rho in {0.98, 0.99}` (DEQ) and in-distribution size 6
(maze). The **silent low-difficulty control is `rho = 0.5`** (DEQ) and large-`eps`
(maze); `rho = 0.7` is reported as a non-silent intermediate, not the silent gate
-- on DEV `rho=0.7` still carries a moderate underfit gap (median rel train gap
`+0.45` to `+0.58`) and sits only ~2.6 orders below the high-`rho` effect, whereas
`rho=0.5` is ~3.6 orders below.

- **ATTAINABILITY confirmed (predicted)** if **all** hold:
  - **C1 (DEQ).** At high `rho`, the strong-effect cells (`K in {1,2,4}`) are
    UNMATCHABLE-dominant (matchable fraction `< m_min`) in the **underfit
    direction**: median relative train gap `>= 0.5` (Neumann `>= 1.5x` exact train
    loss -- cannot reach exact's fit, distinct from low-`rho` overshoot where the
    gap is negative). DEV strong cells are `+0.8` to `+7.1`.
  - **C2 (DEQ collapse / no bias).** Where matchable at `m_min` (near-exact
    operator, e.g. `K=32`), the held-out function distance is `<= F` (collapses to
    the noise floor).
  - **C3 (maze).** The confound-free cell is UNMATCHABLE-dominant
    (matchable fraction `< m_min`), and the as-trained `b2 - exact` gap **does not
    grow** with size: `gap(48) < gap(6)`.
- **INDUCTIVE BIAS confirmed (not predicted)** if, at matched fit with matchable
  fraction `>= m_min`, the high-`rho` held-out function distance has point
  estimate `> F` **and** CI lower bound `> F`, with the low-difficulty control
  silent (below).
- **AMBIGUOUS** (neither cleanly fires) resolves to the conservative
  ATTAINABILITY/optimization claim, consistent with the program default.

## 6. Required negative controls

A positive result of either kind is void unless these hold on HOLDOUT:

- **NC1 — low-difficulty DEQ silence.** At the silent control `rho = 0.5`, the
  Neumann arm shows no underfit attainability gap (median rel train gap `<= 0.1`
  in absolute value -- matchable or overshoot, not underfit) and the low-`rho`
  function-distance effect is **`>= 3` orders of magnitude below** the high-`rho`
  effect (DEV: ~3.6 orders). `rho = 0.7` is reported but not gated. (The Task B
  `>= 3`-orders bar, retained, applied at the genuinely silent `rho`.)
- **NC2 — near-exact operator collapse.** `K=32` (DEQ) / `K=256` (maze) collapses
  to exact: function/extrapolation distance `<= F`, and the maze `K=256`
  extrapolation gap is orders below the `K_match` gap (floor soundness, now using
  the FIX-2 noise floor so it is not a machine-zero artifact).
- **NC3 — forward bitwise identity (DEQ).** Exact-backward and Neumann-backward
  modes produce bitwise-identical forward values at fixed parameters
  (`max_abs == 0`); any nonzero voids the gradient-channel attribution.

## 7. Falsification / honesty conditions

- If C1/C3 fail (the approx arm *is* matchable at high difficulty) and the
  matched high-`rho` function distance is `<= F`, the effect is a pure
  optimization-path artifact, weaker than the attainability claim; report that,
  do not upgrade.
- If the inductive-bias branch fires on HOLDOUT against the DEV prediction, report
  it as the confirmed claim — the DEV adjudication does not override HOLDOUT.
- No canonical/paper-doc propagation (`paper.html`, `PROJECT_STATUS.md`,
  `E4_BASELINES.md`) until results produced under this file exist.

## 8. Out of scope

This file does not authorize the Task B non-scalar DEQ vectorization: it was gated
out because the DEV inductive-bias branch did not survive. A non-scalar
gradient-channel confirmatory probe may be added later **only** if HOLDOUT
surprisingly confirms the inductive-bias branch under §5.
