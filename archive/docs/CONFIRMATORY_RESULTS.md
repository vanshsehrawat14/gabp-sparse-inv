> **EDITORIAL NOTE (2026-07-22; the frozen result below is unchanged).** This
> matched-fit run was the first aggregate read of HOLDOUT seeds `1000-1029` and is the only
> confirmatory evidence used by the current Paper 1 manuscript. A different branch reused
> those seeds on 2026-06-30 for a bootstrap analysis after this read; that later analysis is
> not an independent confirmation and its verdict/effect-size claims are excluded from the
> paper. In the first-read `summary.json`, the top-level tag and path are `CONFIRMATORY`, but
> the nested DEQ/maze payloads say `EXPLORATORY`: `matched_fit_taskA.py` returned a
> module-level constant in those nested fields even though its seed guard and output routing
> used the confirmatory argument. This is a provenance-label defect, not a scalar
> recomputation. A strict final audit found one further ambiguity: the maze JSON records
> `floor_sound: false` at the in-distribution size (mean `K=256` minus exact evaluation-loss
> gap `2.70e-5`, 44% of exact's mean loss, versus the code's 5% diagnostic), while the frozen
> prose below marks NC2 pass using only the 2--4-orders-smaller extrapolation gaps. The
> preregistration's "function/extrapolation distance" wording does not settle which
> non-equivalent check controls. The current manuscript therefore reports maze NC2 as
> ambiguous and does not depend on a pass. See Paper 1 Appendix `app:integrity`.

> **EDITORIAL NOTE (2026-07-25; the frozen result below is unchanged).** A re-audit of
> `archive/bench/matched_fit_taskA.py` against this record found that the **C2 pass recorded
> below is uninformative**, and that the inductive-bias branch was unfalsifiable on these data.
> The preregistration's FIX 2 sets `F = max(2 x M_null, F_abs)` with `M_null` the near-exact
> cell's own matched function distance, and C2 then tests that same cell against `F`. Since
> `2 x M_null > F_abs = 2e-6` at both high-`rho` cells, C2 evaluates `mean <= 2 x mean`, which
> holds identically: the recorded values are exactly half their floors
> (`9.11e-4` vs `1.82e-3`; `1.84e-4` vs `3.68e-4`, ratio `2.000000`). The verdict rule is
> `SURVIVE` if `ci_low > F` and `COLLAPSE` if `mean <= F`, so with `F = 2 x mean` the
> `COLLAPSE` verdict is forced and `SURVIVE` is unreachable. `K=32` is also the **only**
> high-`rho` arm clearing `m_min = 0.60` (`22/30`, `19/30`; every `K in {1,2,4,8,16}` cell is
> below), so it is the only arm whose equal-fit comparison is adjudicable, and it is the arm
> that defines the threshold. Therefore
> `bias = any(verdict.startswith("SURVIVE") ...)` in `confirmatory_decision` is identically
> `False` by construction, independent of the measurements. Against the DEV-calibrated absolute
> scale `F_abs = 2e-6` (largest DEV silent-cell matched distance `1.46e-6`) the same two
> distances are `456x` and `92x` larger, and the `rho=0.98` CI lower bound `3.84e-4` also
> exceeds it. **No number below is retracted and the frozen rule is not amended**: the
> conservative attainability verdict stands, but its support is C1/C3 and NC1's failure, not
> C2. The correct reading is that the equal-fit question is *unresolved* at the near-exact
> cells rather than answered negatively. Propagated to Paper 1 Appendix `app:integrity`, the
> DynaFront cut, and the project's internal failure ledger and audit records.

> **EDITORIAL NOTE (2026-08-08; the frozen result below is unchanged, and so is the
> 2026-07-25 note above it).** The absolute-scale sensitivity introduced in that note
> (`456x` and `92x` above `F_abs = 2e-6`) is a **mean-only reading**, and the same records
> point the other way on the median. `armA_matched_fdist` stores both. At `rho=0.98, K=32`
> the mean is `9.1136e-4` and the median `5.4651e-8`; at `rho=0.99, K=32` the mean is
> `1.8386e-4` and the median `3.2554e-8`. The medians are `37x` and `61x` **below** the same
> `F_abs`, and the mean exceeds the median by `16,676x` and `5,648x` (4.22 and 3.75 orders).
> The `rho=0.99` interval `[7.75e-8, 5.51e-4]` straddles `F_abs` outright, from `26x` below
> to `276x` above. The skew is specific to these two adjudicable cells and is not a property
> of the metric: at the low-difficulty control cells `rho=0.5, K=1` and `K=2` the mean/median
> ratios are `2.6x` and `1.4x`. **Nothing is retracted and the frozen rule is still not
> amended.** The preregistration specified the mean, so the frozen verdict remains the
> mean-based one. What changes is the standing of the 2026-07-25 sensitivity: it does not
> establish that the near-exact arms fail to collapse, because the direction of that
> conclusion depends on the location statistic and `n=22`/`n=19` matched seeds cannot
> separate the two readings. The 2026-07-25 conclusion, that the equal-fit question is
> *unresolved* at the near-exact cells, is unchanged and now has a second, independent
> reason. Only `armA_matched_fdist` carries a recorded median, so this check does not extend
> to the `as_trained_fdist_mean` separations used by the NC1 function-distance clause; the
> per-seed HOLDOUT records were deliberately **not** reopened to extend it, because those
> seeds are retired under the project's holdout ledger. Propagated to Paper 1 Appendix
> `app:integrity`, the DynaFront cut §"The conservative reading remains hedged" and its
> control table, `paper/attainability/verify_claims.py` (which now asserts both statistics),
> and the internal failure ledger (D31). Found during an independent review of the frozen
> records.

> **EDITORIAL NOTE (2026-08-08, second of the day; the frozen result below is unchanged, and
> so are all three notes above it).** A fourth defect at the same `K=32` cells is independent
> of the three above and is structural rather than statistical. The DEQ probe trains on
> `chain_edges(n)` with `n=16` and block size `1`, so the system being solved is
> **16-dimensional**, and the adjudicable arm is `K=32`. A Krylov space
> `span{b, Jb, ..., J^(K-1) b}` has dimension at most `n` and contains `A^-1 b` once
> `K >= n`, so at `K=32` on a 16-dimensional system the residual-optimal `K`-matvec rule *is*
> an exact solve. The cell therefore cannot distinguish "the budget is too small" from
> "truncated Neumann is a poor polynomial", and could not have done so even if `F` had been
> sound and even if mean and median had agreed. **Each of the three reasons voids the cell on
> its own.** The same threshold applies to the exploratory gradient sweep behind Table
> `tab:deq`: `make_figures.py` uses `rows, cols, b = 3, 3, 2`, an **18-dimensional** system,
> at `Ks = (8, 16, 32)`, so its `K=32` column is past saturation and every column is past the
> `3x3` grid's 4-hop diameter. Those entries are now reported in the manuscripts as properties
> of the truncated-Neumann rule, not of the matvec budget. **Nothing below is retracted.** The
> predeclared hard cells `K in {1,2,4}` are at most a quarter of the dimension and reach at
> most 3 of the chain's 15 hops, and the maze `K_match = 7` on 36 nodes reaches 6 of 10 hops
> at the training size, so the conservative attainability verdict this record supports is
> untouched. The consequence for future work is that **any successor experiment comparing
> solve rules at a matched budget must run at `n >> K_max`**; at `n <= K_max` the strongest
> competitor degenerates into the exact arm and the comparison is vacuous by construction.
> Propagated to Paper 1 Appendix `app:reach` (new, carrying `Proposition 1` and the saturation
> remark), Appendix `app:integrity`, the DynaFront cut §2 and its Appendix B,
> `paper/attainability/verify_claims.py` (`verify_budget_window`, which recomputes
> `K_match = 7` from `n_match` and reads both probe sizes off their sources), and
> the internal failure ledger (D32).

# Confirmatory results (HOLDOUT)

**Status: CONFIRMATORY.** Produced under
[`PREREGISTRATION_MATCHED_FIT_CONFIRMATORY.md`](PREREGISTRATION_MATCHED_FIT_CONFIRMATORY.md)
on **HOLDOUT seeds `1000-1029`** with the frozen rules (committed before the
run). *(Link repointed 2026-07-06: the prereg this run was produced under was
frozen as `PREREGISTRATION_CONFIRMATORY.md` at commit `00c0bf5`; the 2026-07-04
merge reused that filename for the bootstrap prereg, so the original is restored
under the distinct name above. The §5/§6 rules cited below are its sections.)* Analysis:
`gabp_sparse_inv/bench/matched_fit_taskA.py --confirmatory`. The DEV menu-anchor
determinism check is not applicable (no recorded HOLDOUT reference); NC3 forward
bitwise identity holds by construction (the linear DEQ forward is the exact solve
in both backward modes).

Date: 2026-06-29.

## DEQ (primary)

30 HOLDOUT seeds. Frozen floor `F = max(2x M_null, 2e-6)`, gate `m_min = 0.60`.

**Decision: AMBIGUOUS -> conservative attainability/optimization.** The
inductive-bias (SURVIVE) branch did **not** fire anywhere.

| check | result | evidence |
|---|---|---|
| **C1** attainability (high-rho strong cells K∈{1,2,4} unmatchable + underfit) | **PASS** | matchable `0-23%` (< 0.60); rel train gap `+1.46` to `+13.2` |
| **C2** collapse (near-exact K=32, matchable) | **PASS** | ρ=0.98: `9.11e-4 <= 1.82e-3`; ρ=0.99: `1.84e-4 <= 3.68e-4` |
| **NC1** silent control ρ=0.5 | **FAIL** | underfit clause failed: ρ=0.5 K=1/2/4 rel gap `+1.33/+1.06/+0.97` (> 0.10); the function-distance clause passed (`4.2` orders below high-ρ) |

Selected cells (matchable/30, rel train gap, matched fdist, floor F, verdict):

| ρ | K | matchA | relgap | A_fdist | F | verdict |
|---:|---:|---:|---:|---|---:|---|
| 0.99 | 1 | 3/30 | +10.9 | 1.49e-3 | 3.7e-4 | UNMATCHABLE-dominant→attainability |
| 0.99 | 2 | 4/30 | +7.2 | 9.79e-4 | 3.7e-4 | UNMATCHABLE-dominant→attainability |
| 0.99 | 4 | 7/30 | +3.9 | 1.61e-3 | 3.7e-4 | UNMATCHABLE-dominant→attainability |
| 0.99 | 32 | 19/30 | -0.01 | 1.84e-4 | 3.7e-4 | COLLAPSE(partial 19/30) |
| 0.98 | 1 | 0/30 | +13.2 | n/a | 1.8e-3 | UNMATCHABLE-dominant→attainability |
| 0.50 | 1 | 7/30 | +1.33 | 6.24e-7 | 2.0e-6 | UNMATCHABLE-dominant→attainability |

### Honest reading

- **The inductive-bias thesis is not confirmed.** No cell shows a clean
  matched-fit SURVIVE at `>= m_min` matchability. The H3 effect cells are
  unmatchable (attainability), and the near-exact cells collapse.
- **The attainability/optimization reading is directionally confirmed (C1, C2)
  but not *cleanly* confirmed**, because the silent control NC1 did not pass as
  pre-specified. Per §6 (a positive result of either kind is void unless the
  negative controls hold) and §5 (AMBIGUOUS resolves to the conservative
  attainability/optimization claim), the HOLDOUT verdict is the **conservative
  attainability/optimization reading -- supported, not a pristine confirmation.**
- **Why NC1 failed (honest mechanism, not a rescue).** At ρ=0.5 the exact arm
  converges to `L≈5e-9` on HOLDOUT (vs `~1.3e-8` on DEV), so the Neumann arm's
  similar `~1e-8` loss reads as a `+1.0` rel gap rather than the `~0`/overshoot
  seen on DEV. At these near-machine-precision magnitudes the train-loss *ratio*
  is unstable, and there is also a small genuine Neumann-1 bias even at ρ=0.5
  (the menu already noted ρ=0.5/0.7 are "not perfectly silent"). The robust
  function-distance silence clause (`>= 3` orders) did hold (`4.2`). The frozen
  rule keyed on the relgap direction, which failed; **it is not relaxed
  post-hoc.**

The pre-registration worked: it withheld a triumphant "confirmed" when the
control was imperfect, and resolved conservatively. The directional result is
unchanged from DEV -- attainability/optimization, not inductive bias.

## Maze (secondary)

30 HOLDOUT seeds. Confound-free cell = exact-trained vs B2(K_match=7)-trained,
both evaluated under exact. **C3 PASS, NC2 (maze floor) PASS.**

Arm A is **UNMATCHABLE for all 30 seeds**: median `L_exact = 1.45e-5` vs B2-best
`2.96e-2` (rel gap `+2.0e3`, ~2000x worse), so no matched-fit regime exists. The
as-trained confound-free gap **shrinks** with size (`+8.17e-2` at size 6 ->
`+1.30e-3` at size 48), so there is no growing-extrapolation signal. The K=256
(~exact) extrapolation gaps (`+2.97e-6`, `+1.60e-7`, `-2.71e-8`) sit 2-4 orders
below the B2(K=7) gaps -> floor sound in the extrapolation sense (the binary flag
is False only from exact's near-machine-precision in-distribution loss, as on
DEV). Arm B uninformative (within-tol `0/30`, transient `30/30`).

## Combined verdict

| check | DEQ | maze |
|---|---|---|
| C1/C3 attainability (unmatchable at high difficulty) | **PASS** | **PASS** |
| C2 collapse (near-exact, matchable) | **PASS** | -- |
| NC1 silent control (rho=0.5) | **FAIL** (relgap clause) | -- |
| NC2 floor sound | **PASS** (K=32) | **PASS** (K=256) |

**Branch: AMBIGUOUS -> conservative attainability/optimization.** The
inductive-bias SURVIVE branch fired on neither probe. Every attainability
signature (C1, C2, C3) and both floor controls (NC2) pass on HOLDOUT; the sole
failure is NC1's relgap-direction clause at the near-machine-precision `rho=0.5`
control (its function-distance silence clause held at `4.2` orders). Per the
frozen rules a failed negative control voids a *pristine* positive (§6), so the
verdict resolves to the conservative attainability/optimization reading (§5).

(The combined verdict is composed from the two completed `--confirmatory` HOLDOUT
runs -- DEQ and maze -- each executed under the frozen rules; the DEQ result is
deterministic given the seed, so a `--probe both` re-derivation reproduces it.)

## Bottom line

On HOLDOUT, the gradient-channel / operator-training **inductive-bias claim is
refuted** (no SURVIVE on either probe). The **attainability/optimization** reading
is the supported, conservative verdict: the approximate operator changes *which
optimum is reachable* (it cannot attain the exact arm's fit at high difficulty),
not *which of two equally-good minima is preferred*. The confirmation is **hedged,
not pristine** -- NC1's low-difficulty control failed on its relgap clause at
machine-precision `rho=0.5` (function-distance silence held). A cleaner re-test
would key NC1 on the function-distance silence rather than the relgap direction;
this is logged as a future-prereg refinement and **not** applied to this frozen
run. No canonical/paper-doc propagation beyond this confirmatory record without a
clean NC1.
