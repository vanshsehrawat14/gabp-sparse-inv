# Exploratory hypothesis menu - exactness as an inductive bias

**Status: EXPLORATORY only.** This is a hypothesis menu for DEV-seed exploration,
not a confirmatory preregistration and not a paper verdict. It records the
candidate phenomena we are allowed to look for while using DEV seeds. A paper
claim is backed only by confirmatory numbers from data this exploration never
touched.

Date: 2026-06-28. Scope: the probe-scale maze and DEQ studies for the
*exactness-as-inductive-bias* paper program. The scope guardrail from
[AGENTS.md](../AGENTS.md) still applies: this project claims exact selected
inversion / solves on the supported sparse filled patterns, including the
static no-pivot non-symmetric regime. It does not claim pivoted LU selected
inversion, approximate iterative loopy GaBP, indefinite or complex-Hermitian
support, or second-order autograd through the analytic junction backwards.

## 1. Exploration vs. confirmation

Exploration generates hypotheses. Confirmation tests a frozen subset of those
hypotheses on data exploration never touched.

- **EXPLORATORY runs** use DEV seeds `0-19` only. Their outputs live under a
  path tagged `EXPLORATORY`, and JSON/log records carry
  `run_tag: "EXPLORATORY"`.
- **CONFIRMATORY runs** use HOLDOUT seeds `1000-1029` only. The seed guard must
  raise if a holdout seed is requested outside a confirmatory script, and must
  also raise if a DEV seed is requested inside a confirmatory script.
- Exploratory outputs may rank candidate effects, record rough DEV sizes, and
  propose mechanisms. They must not contain confirmatory confidence intervals,
  confirmatory claims, or a findings/verdict document.
- Confirmatory hypotheses will be written **after exploration** as a frozen
  subset in `docs/PREREGISTRATION_CONFIRMATORY.md`, then tested on HOLDOUT
  seeds. Confirmatory confidence intervals are computed only on HOLDOUT data.

This document therefore replaces the older single-claim preregistration with a
menu of candidate phenomena. The confirmatory file, not this one, will define the
claims eligible for the paper.

## 2. Common setup

The central comparison remains the learned global operator under different
inverse/adjoint implementations:

- **Exact:** the structured exact solve / adjoint (`junction_solve` for maze,
  `nonsym_junction_solve(transpose=True)` for DEQ).
- **B2(K):** the same learned operator, truncated to `K` iterative steps
  (Jacobi/CG/Neumann/Richardson depending on the probe).
- **B1:** the local/globality ablation.
- **B3:** a learned iterative steelman, if and when built for the same protocol.

The difficulty knob is problem size for maze and `rho` for DEQ. Compute is
reported as operator FLOPs, not wall-clock. The main exploratory instrument is
the break-even compute `C*(knob)`: the smallest B2 budget whose downstream
metric enters the exact variant's DEV seed band. During exploration this is a
ranking instrument only, not a confirmatory decision rule.

## 3. Candidate hypotheses

### H1 - cost / infeasibility

**Directional hypothesis.** B2's break-even compute `C*` to match exact grows
without saturating as the difficulty knob hardens: maze size increases, or DEQ
`rho` approaches one.

**Mechanism.** A truncated global iteration has to propagate information across
the graph or through the fixed-point memory horizon. The required iteration
count should scale with graph diameter / conditioning in maze-like Green's
functions and with roughly `1 / (1 - rho)` in DEQ-style Neumann truncations.

**Collision with known results.** This is likely true but likely unsurprising.
For mazes it collides mostly with graph diameter, conditioning, and
oversquashing; for DEQ it collides with phantom-gradient theory. This is the
floor finding: useful for cost/infeasibility, but not by itself the novel
inductive-bias claim.

### H2 - co-adaptation / operator-swap asymmetry

**Directional hypothesis.** A model trained with B2(K) co-adapts to the
approximation, so evaluating that trained model under the exact operator can be
worse than evaluating it under B2(K). In short: swapping in the better operator
at evaluation can hurt.

**Mechanism.** The local encoder and learned precision can learn features whose
meaning depends on the approximate inverse used during training. Exactness then
changes the learned function, not merely the numerical accuracy of a fixed
function.

**Predicted shape.** The asymmetry should be mostly forward-invisible when
measured before training, and should grow with nonlinearity and stronger
truncation (smaller `K`).

**Exploratory instrument.** For each probe, train checkpoints under
`{Exact, B2(K)}` for a fixed K grid and cross-evaluate every checkpoint under
every eval operator in the same set. The H2 signal is the paired, per-seed
loss gap `B2-trained/B2-eval - B2-trained/Exact-eval`; because lower loss is
better, negative values are the co-adaptation-shaped direction. This matrix is
allowed on DEV seeds only and is hypothesis-generating.

**Collision with known results.** This is the novel candidate. It is not
explained by graph diameter, conditioning, oversquashing, or `1 / (1 - rho)`
alone, because those mechanisms say the approximation is worse, not that a
trained model can prefer its own approximation over the exact operator.

### H3 - gradient-channel isolation

**Directional hypothesis.** With the forward solve held bit-identical, swapping
only the backward rule (exact adjoint vs. Neumann-K adjoint) changes the learned
function, not just the transient optimization path.

**Mechanism.** In the DEQ probe, the forward fixed point can be identical while
the implicit-differentiation channel differs. If final functions differ under a
common evaluation operator, the learning effect is attributable to gradient
exactness alone.

**Exploratory instruments.** In DEQ cross-eval, eval-operator loss columns are
expected to be identical because the forward solve is held fixed. The relevant
DEV-only rows are the learned-function gaps under a common forward loss and the
gradient-channel gaps at frozen checkpoints; they should not be interpreted as
H2 eval-operator co-adaptation.

The stricter H3 instrument trains two linear affine DEQ arms with the exact
forward solve in both arms and swaps only the backward rule: exact adjoint vs
Neumann-K adjoint. At fixed parameters the exact-backward and Neumann-backward
modes must produce bit-identical forward values. Both trained arms are then
evaluated under the same exact forward operator. Primary DEV-only metrics are
`test_loss_neumannK - test_loss_exact` and the held-out function distance
`mean_x ||f_exact(x) - f_neumannK(x)||^2 / Var(target)`. If weights/functions
differ but held-out test loss does not, that is labeled as the equal-test
minimum trap, not as an inductive-bias result.

**Collision with known results.** This is the cleanest attribution test against
"the forward approximation did it." It still has to be separated from ordinary
optimizer-path sensitivity, so the confirmatory version must specify common
evaluation, frozen optimizer budgets, and no post-hoc seed selection.

### H4 - negative controls

**Directional hypothesis.** Exactness should provide no advantage in regimes
where globality should not matter, and symmetric-by-construction swaps should
show no asymmetry.

**Mechanisms.**

- In the maze large-`eps` regime, correlation length is short and local or
  low-iteration approximations should be enough.
- In the low-`rho` DEQ regime, the memory horizon is short and Neumann truncation
  should be adequate.
- For a swap constructed from two equal-quality approximations of different
  form, neither approximation should systematically degrade when evaluated under
  the other.

**Collision with known results.** These controls must hold or the main effects
are artifacts: capacity, data leakage, optimizer quirks, or a measurement bug.
They are not optional supporting plots.

## 4. Allowed exploratory outputs

Exploration may produce:

- rough DEV-seed effect sizes and ranked candidate phenomena;
- mechanism notes explaining why a candidate is surprising or unsurprising;
- sanity checks for output tagging, seed separation, and resumability;
- proposed confirmatory hypotheses to freeze later.

Exploration may not produce:

- confirmatory confidence intervals;
- paper claims;
- a findings/verdict document from DEV results;
- any HOLDOUT-seed run or aggregate;
- a confirmatory file written after looking at HOLDOUT data.

The exploratory menu is a menu. It is not a verdict.

## 5. Confirmation handoff

After the exploratory pass is complete, choose a frozen subset of H1-H4 and write
`docs/PREREGISTRATION_CONFIRMATORY.md` before any HOLDOUT run. That file must
state:

- the exact hypotheses and directional predictions;
- the included probes, arms, knobs, and metrics;
- the fixed HOLDOUT seed list;
- the output paths tagged `CONFIRMATORY`;
- the allowed confidence intervals and decision rules;
- the negative controls required to validate any positive result.

Only results produced under that confirmatory file and HOLDOUT seed protocol may
support claims in the paper.
