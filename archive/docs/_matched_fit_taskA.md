# Task A matched-fit adjudication (EXPLORATORY)

Status: `EXPLORATORY`, DEV seeds `0-19` only. Not a verdict document and not a
paper claim. HOLDOUT seeds `1000-1029` were not touched. Protocol and decision
rules: the matched-fit thresholds and the 2026-06-29 amendment in
[PREREGISTRATION.md](PREREGISTRATION.md); producer code untouched; analysis in
`gabp_sparse_inv/bench/matched_fit_taskA.py`.

This is the matched-training-fit control for the H2/H3
inductive-bias-vs-optimization question. **Arm A (strict, approx-arm-only) is the
verdict; Arm B (symmetric-milestone) is labeled diagnostic only**, compromised by
the converged-vs-transient asymmetry. Matching uses the train-set loss under the
common exact operator for both arms.

## Determinism (replay reproduces the recorded scalars)

Before any matched step is read, the replay regenerates the consumed cells and
asserts they reproduce the recorded EXPLORATORY numbers in
[`_phenomenon_menu.md`](_phenomenon_menu.md). **All 12 DEQ anchors passed** (mean
over DEV seeds 0-19), reproducing the menu to 4-5 significant figures, e.g.:

| anchor | regenerated | recorded |
|---|---:|---:|
| `delta_test_loss(rho=0.99,K=2)` | `+1.120e-1` | `+1.120e-1` |
| `delta_test_loss(rho=0.99,K=1)` | `+6.037e-2` | `+6.040e-2` |
| `delta_test_loss(rho=0.99,K=16)` | `-2.144e-3` | `-2.140e-3` |
| `test_function_rel_mse(rho=0.99,K=2)` | `+4.182e-3` | `+4.180e-3` |

The deterministic-replay assumption holds for the consumed cells.

## DEQ Arm A (VERDICT)

Primary metric: held-out function rel-MSE under the common exact forward.
`matchA` = DEV seeds (of 20) whose Neumann arm can be brought into the +/-5%
train-loss band of the exact arm. `relgap` = median Neumann/exact train-loss
gap. `floorF` = `2x` the K=32 matched function distance. Verdicts are computed
over matchable seeds only; `(partial k/20)` flags how few that is.

| rho | K | matchA | relgap | A_fdist mean[CI] | floorF | verdict |
|---:|---:|---:|---:|---|---:|---|
| 0.50 | 1 | 7/20 | -4.4e-1 | 1.46e-6[5.5e-7,2.4e-6] | 9.2e-25 | SURVIVE(partial 7/20) |
| 0.50 | 2 | 5/20 | +1.1e-2 | 3.92e-7[1.3e-11,1.2e-6] | 9.2e-25 | SURVIVE(partial 5/20) |
| 0.50 | 4 | 12/20 | +5.3e-2 | 5.47e-7[5.9e-8,1.2e-6] | 9.2e-25 | SURVIVE(partial 12/20) |
| 0.50 | 8 | 18/20 | +3.0e-2 | 6.70e-8[3.3e-12,2.0e-7] | 9.2e-25 | SURVIVE(partial 18/20) |
| 0.50 | 16 | 20/20 | +1.3e-4 | 7.82e-16[2.1e-16,1.5e-15] | 9.2e-25 | SURVIVE |
| 0.50 | 32 | 20/20 | +3.4e-9 | 4.60e-25[1.4e-25,8.5e-25] | 9.2e-25 | COLLAPSE |
| 0.70 | 1 | 9/20 | +5.8e-1 | 7.86e-6[2.3e-6,1.4e-5] | 2.1e-14 | SURVIVE(partial 9/20) |
| 0.70 | 2 | 8/20 | +7.1e-2 | 1.03e-5[4.9e-6,1.6e-5] | 2.1e-14 | SURVIVE(partial 8/20) |
| 0.70 | 4 | 2/20 | +3.0e-1 | 1.12e-5[9.7e-6,1.3e-5] | 2.1e-14 | SURVIVE(partial 2/20) |
| 0.70 | 8 | 10/20 | +4.5e-1 | 2.48e-6[8.8e-11,7.4e-6] | 2.1e-14 | SURVIVE(partial 10/20) |
| 0.70 | 16 | 16/20 | +3.1e-3 | 5.50e-7[1.6e-12,1.6e-6] | 2.1e-14 | SURVIVE(partial 16/20) |
| 0.70 | 32 | 20/20 | -4.1e-6 | 1.05e-14[3.8e-15,1.9e-14] | 2.1e-14 | COLLAPSE |
| 0.85 | 1 | 1/20 | +1.4e+0 | 2.75e-10[..] | 2.7e-9 | COLLAPSE(partial 1/20) |
| 0.85 | 2 | 6/20 | +2.4e-1 | 4.35e-5[5.2e-6,8.4e-5] | 2.7e-9 | SURVIVE(partial 6/20) |
| 0.85 | 4 | 7/20 | -3.4e-1 | 3.24e-5[1.2e-5,6.0e-5] | 2.7e-9 | SURVIVE(partial 7/20) |
| 0.85 | 8 | 10/20 | +3.3e+0 | 2.12e-5[3.9e-6,4.4e-5] | 2.7e-9 | SURVIVE(partial 10/20) |
| 0.85 | 16 | 12/20 | +6.6e-1 | 3.46e-6[5.4e-11,1.0e-5] | 2.7e-9 | AMBIGUOUS->retitle(partial 12/20) |
| 0.85 | 32 | 13/20 | +1.2e-4 | 1.35e-9[1.9e-14,4.1e-9] | 2.7e-9 | COLLAPSE(partial 13/20) |
| 0.90 | 1 | 3/20 | +5.9e+0 | 1.68e-4[2.3e-8,4.3e-4] | 4.0e-5 | AMBIGUOUS->retitle(partial 3/20) |
| 0.90 | 2 | 5/20 | +3.2e-1 | 1.48e-4[2.0e-5,2.9e-4] | 4.0e-5 | AMBIGUOUS->retitle(partial 5/20) |
| 0.90 | 4 | 6/20 | +1.5e+0 | 7.43e-6[6.7e-10,2.2e-5] | 4.0e-5 | COLLAPSE(partial 6/20) |
| 0.90 | 8 | 4/20 | -4.5e-1 | 1.69e-4[1.0e-9,3.4e-4] | 4.0e-5 | AMBIGUOUS->retitle(partial 4/20) |
| 0.90 | 16 | 11/20 | -5.1e-2 | 3.19e-9[3.8e-10,7.1e-9] | 4.0e-5 | COLLAPSE(partial 11/20) |
| 0.90 | 32 | 13/20 | +1.1e-3 | 1.99e-5[4.6e-12,6.0e-5] | 4.0e-5 | COLLAPSE(partial 13/20) |
| 0.95 | 1 | 4/20 | +3.6e+0 | 7.37e-4[2.6e-4,1.0e-3] | 9.2e-5 | SURVIVE(partial 4/20) |
| 0.95 | 2 | 2/20 | +2.9e+0 | 3.64e-8[2.3e-8,5.0e-8] | 9.2e-5 | COLLAPSE(partial 2/20) |
| 0.95 | 4 | 2/20 | +3.3e-1 | 5.13e-7[1.1e-8,1.0e-6] | 9.2e-5 | COLLAPSE(partial 2/20) |
| 0.95 | 8 | 9/20 | +4.4e-2 | 3.78e-4[1.2e-4,7.6e-4] | 9.2e-5 | SURVIVE(partial 9/20) |
| 0.95 | 16 | 13/20 | -1.6e-1 | 1.71e-4[1.7e-8,4.2e-4] | 9.2e-5 | AMBIGUOUS->retitle(partial 13/20) |
| 0.95 | 32 | 12/20 | -6.8e-1 | 4.59e-5[4.7e-11,1.4e-4] | 9.2e-5 | COLLAPSE(partial 12/20) |
| 0.98 | 1 | 2/20 | +7.1e+0 | 1.38e-3[3.0e-8,2.8e-3] | 1.7e-3 | COLLAPSE(partial 2/20) |
| 0.98 | 2 | 2/20 | +2.5e+0 | 2.00e-8[1.2e-8,2.8e-8] | 1.7e-3 | COLLAPSE(partial 2/20) |
| 0.98 | 4 | 4/20 | +8.0e-1 | 8.66e-4[5.6e-7,1.9e-3] | 1.7e-3 | COLLAPSE(partial 4/20) |
| 0.98 | 8 | 6/20 | +6.0e-1 | 7.39e-7[8.1e-8,1.6e-6] | 1.7e-3 | COLLAPSE(partial 6/20) |
| 0.98 | 16 | 12/20 | +4.5e-1 | 2.32e-4[4.4e-8,6.9e-4] | 1.7e-3 | COLLAPSE(partial 12/20) |
| 0.98 | 32 | 17/20 | +2.1e-4 | 8.57e-4[1.7e-4,1.5e-3] | 1.7e-3 | COLLAPSE(partial 17/20) |
| 0.99 | 1 | 3/20 | +3.1e+0 | 4.23e-3[4.0e-3,4.5e-3] | 1.8e-7 | SURVIVE(partial 3/20) |
| 0.99 | 2 | 2/20 | +2.0e+0 | 4.09e-3[3.9e-3,4.3e-3] | 1.8e-7 | SURVIVE(partial 2/20) |
| 0.99 | 4 | 2/20 | +1.0e+0 | 5.40e-7[6.1e-12,1.1e-6] | 1.8e-7 | AMBIGUOUS->retitle(partial 2/20) |
| 0.99 | 8 | 9/20 | +4.2e-1 | 1.02e-3[8.5e-7,2.3e-3] | 1.8e-7 | SURVIVE(partial 9/20) |
| 0.99 | 16 | 12/20 | -2.0e-1 | 3.45e-4[4.9e-8,1.0e-3] | 1.8e-7 | AMBIGUOUS->retitle(partial 12/20) |
| 0.99 | 32 | 14/20 | -1.3e-2 | 8.82e-8[8.9e-10,2.1e-7] | 1.8e-7 | COLLAPSE(partial 14/20) |

### Reading (settle, not sell)

Neither clean SURVIVE branch fires. Two robust, threshold-independent facts drive
the reading:

1. **The effect is the fit gap.** The cells carrying the original H3 signal (low
   K, high rho) are exactly the cells where the Neumann arm fits *worse* — median
   `relgap` is `+1` to `+7` (Neumann train loss 2-8x exact) at K=1/2/4 for
   rho>=0.95 — and matchability there is low (2-4/20). The biased gradient cannot
   attain the exact arm's fit. Per the amendment, that is the
   **attainability/optimization** reading.
2. **Where matchable, it collapses.** As K rises and the Neumann operator
   approaches exact (`relgap -> 0`), matchability rises and the matched function
   distance falls to/near the floor (rho=0.99, K=32: `8.82e-8` vs floor
   `1.76e-7` -> COLLAPSE).

Two wrinkles make several literal verdict labels untrustworthy; both are flagged
for the confirmatory file rather than patched post-hoc on DEV:

- **Partial-matchability subsample.** Almost every cell is `(partial)`; the
  SURVIVE/COLLAPSE label is computed over the matchable minority, which is
  non-representative (biased toward seeds where exact under-fit, so Neumann could
  reach it). A `SURVIVE(partial 2-3/20)` label is **not** evidence the effect
  survives. The confirmatory file must pre-commit a minimum-matchability fraction
  and adjudicate only high-matchability cells.
- **Low-rho floor is machine-zero.** At rho=0.5/0.7 the K=32 cell is
  bit-identical to exact (`fdist ~1e-25 / ~1e-14`), so `floor = 2x` that is
  ~machine zero and any optimization-noise `fdist (~1e-6)` trivially "exceeds"
  it, producing spurious low-rho SURVIVE labels. The confirmatory floor must add
  an optimization-noise floor, e.g. `max(2x K32, noise scale)`.

**Synthesis.** The big-effect cells are unmatchable (attainability/optimization);
the matchable near-exact cells collapse (optimization); the partial/noisy middle
resolves, under the preregistered conservative `AMBIGUOUS -> retitle` default, to
the optimization claim. Combined with the realizability asymmetry the amendment
anticipated, the DEQ H3 inductive-bias claim **does not survive matched fit** on
DEV. The mechanism reads as: the gradient bias changes *which minimum is
attainable*, not *which of two equally-good minima is preferred*.

## DEQ Arm B (diagnostic only)

Reading the exact arm earlier on its own trajectory (equal Neumann-plateau loss)
does show function differences, but with high transient counts at the strong
cells (rho>=0.98, K=1/2/4: 19-20/20 reads are mid-descent, not converged), so the
differences are the mid-descent-vs-attractor artifact the amendment warned about.
Arm B does not rescue the bias claim.

## Maze probe (confound-free cell)

Full 20-seed run; determinism **passed** (`exact - b2_matched` at size 6:
`-1.760e-1` vs recorded `-1.760e-1`, |d|=7.9e-6; size 48: `-8.207e-4` vs
`-8.210e-4`). Cell = exact-trained vs B2(K_match=7)-trained, both evaluated under
exact, at sizes 6/12/24/48; matched on the size-6 train loss under common exact
eval.

### Arm A (VERDICT): UNMATCHABLE for all 20 seeds -> attainability/optimization

The B2-trained encoder cannot bring its size-6 exact-eval train loss within 5% of
the exact arm's for **any** of the 20 seeds: median `L_exact = 9.27e-6` vs median
B2-best `2.90e-2` (rel gap `+3.1e3`, ~3000x worse). There is no matched-fit
regime for this cell; per the amendment that settles it as
attainability/optimization without Arm B. (Cleaner than DEQ, where matchability
was partial.)

As-trained confound-free gap (full-fit B2 - exact, both eval exact):

| size | kind | b2 - exact (mean[CI]) |
|---:|---|---|
| 6 | in-dist | `+8.44e-2 [+8.27e-2,+8.62e-2]` |
| 12 | extrap | `+2.22e-2 [+2.17e-2,+2.28e-2]` |
| 24 | extrap | `+5.33e-3 [+5.22e-3,+5.46e-3]` |
| 48 | extrap | `+1.30e-3 [+1.27e-3,+1.34e-3]` |

The raw gap is **largest in-distribution and decays with size** -- it does not
grow with extrapolation, so even before matching there is no
growing-extrapolation-bias signal in this operator-training cell (distinct from
the generic-learner-vs-solver extrapolation story elsewhere).

Floor (K=256 ~exact, eval exact): extrapolation gaps are `+3.37e-6` (size12),
`+1.96e-7` (size24), `-4.88e-8` (size48) -- 2-4 orders below the B2(K=7) gaps, so
the near-exact operator collapses to exact as intended. The binary `floor_sound`
flag reads False only because exact's in-distribution loss is ~`9e-6`, making the
5%-of-loss threshold tighter than the K=256 in-distribution gap (`+3.16e-5`); the
**extrapolation** floor itself is sound (orders below the effect). The
confirmatory noise-floor fix removes this flag artifact.

### Arm B (diagnostic): uninformative here

Within-tol `0/20` and transient `20/20`: the exact arm reaches the B2 plateau
only at step ~2/250, deeply mid-descent, so the symmetric milestone is never
actually achieved. The `exact_read - b2` gaps are negative (`-7.40e-2` at size 6
down to `-1.16e-3` at size 48) but meaningless given the read point. Arm B does
not bear on the verdict, exactly as the amendment's convergence-asymmetry caveat
predicted.

### Maze synthesis

The confound-free maze cell is **unmatchable across all 20 seeds**: B2-operator
training cannot attain exact-training's in-distribution fit, so the exact
advantage is an attainability/optimization effect, not a separable extrapolation
bias -- and the raw gap shrinks rather than grows with size. Same direction as
DEQ.

## Both probes

Neither probe yields a clean matched-fit SURVIVE. DEQ: the H3-effect cells are
where the biased arm fits worse and is mostly unmatchable, the matchable
near-exact cells collapse, and the partial/noisy middle retitles under the
conservative default. Maze: unmatchable for all 20 seeds. The exactness advantage
on DEV reads as **attainability/optimization** (the approximate operator changes
which minimum is reachable), not inductive bias. This is the conservative branch
the program was prepared to retitle to.

## Task B gate (H3 off-toy, non-scalar DEQ)

Task B was **gated OUT and not run**. Its pre-declared condition was "ONLY if Task
A's DEQ branch is SURVIVE." Task A's DEQ branch resolved to attainability /
optimization (effect cells unmatchable, matchable cells collapse, middle ->
retitle); the clean SURVIVE branch fired nowhere. Vectorizing to a non-scalar DEQ
would inherit the same confound, so per the conditional it is skipped. It may be
revisited only if HOLDOUT surprisingly confirms the inductive-bias branch.

## Next

The claim that survived is **attainability / optimization**, frozen for HOLDOUT in
[`PREREGISTRATION_CONFIRMATORY.md`](PREREGISTRATION_CONFIRMATORY.md) (seeds
`1000-1029`), which folds in the two DEV-exposed fixes: a minimum-matchability
fraction (`m_min = 0.60`) and an optimization-noise floor
(`F = max(2x M_null, F_noise)`). No canonical/paper-doc propagation until results
produced under that file exist.
