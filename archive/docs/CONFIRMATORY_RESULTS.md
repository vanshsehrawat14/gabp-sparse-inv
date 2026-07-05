# Confirmatory results (HOLDOUT)

**Status: CONFIRMATORY.** Produced under
[`PREREGISTRATION_CONFIRMATORY.md`](PREREGISTRATION_CONFIRMATORY.md) on **HOLDOUT
seeds `1000-1029`** with the frozen rules (committed before the run). Analysis:
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
