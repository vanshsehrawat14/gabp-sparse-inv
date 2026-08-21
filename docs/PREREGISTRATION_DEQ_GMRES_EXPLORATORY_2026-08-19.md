# Exploratory DEQ residual-optimal falsifier

Date frozen: 2026-08-19 PDT. Status: **EXPLORATORY, DEVELOPMENT DATA ONLY**.
Campaign UUID: `d3d1654b-3f65-4272-8f16-a58e027668d3`.
The first commit containing this file, the runner, its tests, and the hardware
manifest is the freeze. The runner requires that commit as `--protocol-commit`
and refuses a dirty worktree. This protocol supersedes no confirmatory record
and authorizes no HOLDOUT read.

## Question and evidence boundary

Paper 1 shows that finite Neumann adjoints fail its preregistered comparable-fit
gate in the hard cells. It does not show that a residual-optimal rule fails at
the same operator-product budget. This study asks whether the training failure
is materially reduced by unrestarted GMRES on a system too large for the tested
budget to saturate.

The result will be reported as a post-hoc adversarial study. It cannot become a
new confirmation, erase the frozen result, or establish superiority over
preconditioned, multiscale, Jacobian-free, phantom-gradient, or benchmark-scale
alternatives.

## Frozen intervention

- Population: the existing affine scalar-gain DEQ generator on a non-symmetric
  scalar chain, enlarged to `n=256`.
- Seeds: the previously unread calibration block `4000-4019`; no HOLDOUT seed.
- Spectral radii: `rho in {0.5, 0.98, 0.99}`. The first is a control; the latter
  two are the hard cells.
- Product budgets: `P in {1, 2, 4, 8, 16, 32}`. Since `P_max=32<n=256`, a
  generic exact-termination argument cannot rescue GMRES in the tested range.
- Data/training: 32 train and 32 test right-hand sides, fp64, 120 Adam steps,
  learning rate `0.05`, input scale `0.2`, and the existing initialization.
- Arms: exact structured adjoint; `P+1` Neumann terms, which spend exactly `P`
  products by `J^T`; and zero-start unrestarted GMRES with exactly `P` products
  by `(I-J)^T`.
- The exact forward solve is bit-identical across arms and every learned scalar
  is evaluated through the common exact forward.

Operator products are the primary mechanistic currency. GMRES orthogonalization
and basis memory are not free: arm wall time is recorded, and the write-up will
not call the arms FLOP-matched. A same-label comparison of `K` Neumann terms and
`K` GMRES iterations would silently give GMRES one extra operator product; this
study does not use that convention.

## Predictions fixed before any scientific read

1. At each hard `rho` and `P=32`, GMRES should have at most half the Neumann
   relative residual on at least 15 of 20 paired seeds. Non-normality means this
   need not order adjoint error or training.
2. At `P in {16,32}`, the adversarial expectation is a median paired
   `log10(L_gmres/L_neumann) <= -1` in at least one hard cell. If GMRES opens
   comparable fit in hard cells where Neumann does not, the DEQ
   mechanism is rule-sensitive and the manuscript must say so prominently.
3. If both remain non-attaining, the outcome is `JOINT_NONATTAINMENT_UNRESOLVED`,
   not proof that every finite-budget rule fails.
4. At the `rho=0.5, P=32` control, both finite arms are expected to have median
   initial relative adjoint error below `1e-4`. If both attain, the original
   equal-fit question reopens only within this exploratory population.
5. If GMRES trains worse, that is a scientific outcome, not presumptive evidence
   of an implementation defect; the residual and gradient diagnostics decide
   whether the numerical instrument itself failed.

## Endpoints and terminal labels

For every paired seed, cell, and budget the runner records:

- initial relative adjoint error and linear-system residual for both finite arms;
- exact-forward training trajectories and final train/test losses;
- the paired `log10(L_gmres/L_neumann)` final training-loss contrast;
- the original 5% trajectory-match criterion against the exact arm's final
  training loss, retained only for comparability with the frozen study; and
- measured arm runtime.

At each cell the descriptive 60% bar gives exactly one label:

- `GMRES_OPENS_COMPARABLE_FIT`: GMRES clears and Neumann does not;
- `BOTH_ATTAIN`: both clear;
- `JOINT_NONATTAINMENT_UNRESOLVED`: neither clears; or
- `GMRES_WORSE_OR_UNSTABLE`: Neumann clears and GMRES does not.

These are descriptive exploratory labels, not hypothesis tests. The continuous
paired ratios and all 20 seeds remain visible even when a label is inconvenient.

## Hard gates

1. Unit tests must verify both arms' exact product counts, dense-solve termination at the
   dimension, non-increasing GMRES residual, exact-forward identity, and an
   exact-gradient limit.
2. The runner asserts the complete envelope, `n >= 8 P_max`, `P_max<n`, the
   exact seed block, the clean frozen commit, and absence of Arnoldi breakdown.
3. The sanitized target manifest is
   `docs/hardware_manifest_deq_gmres_local_2026-08-19.json`.
4. After freeze, a cost-only pilot times ten steps for every budget-reaching
   arm. It prints timing only and writes no scientific scalar. If its
   full-grid projection exceeds four hours, the campaign stops; the grid is not
   resized after seeing any outcome.
5. Per-seed JSON is written after every rho so interruption is recoverable. A
   resume rejects a changed commit or configuration.
6. Any failed control, breakdown, convention disagreement, or outcome favoring
   GMRES is reported rather than repaired in prose.

## Frozen commands

```text
python -m pytest tests/test_deq_gmres_falsifier.py -q
python -m gabp_sparse_inv.bench.deq_gmres_falsifier --protocol-commit <freeze> --protocol-sha256 <sha256> --campaign-uuid d3d1654b-3f65-4272-8f16-a58e027668d3 --cost-pilot
python -m gabp_sparse_inv.bench.deq_gmres_falsifier --protocol-commit <freeze> --protocol-sha256 <sha256> --campaign-uuid d3d1654b-3f65-4272-8f16-a58e027668d3 --resume
```

Multigrid and nonlocal preconditioning are outside this study. They change the
access class and require setup-cost accounting; Paper 1 will state that escape
boundary instead of presenting this GMRES run as evidence against them.
