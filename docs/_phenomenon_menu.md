# Phenomenon menu

Status: `EXPLORATORY`, DEV seeds `0-19` only. This is not a verdict document and
does not support paper claims. HOLDOUT seeds `1000-1029` were not used. All
intervals below are labeled
`EXPLORATORY_PAIRED_BOOTSTRAP_NOT_CONFIRMATORY` in the source artifacts and are
for menu-building only, not confirmation.

Source artifacts:
- `results/EXPLORATORY/phase1_analysis/`
- `results/EXPLORATORY/operator_cross_eval_analysis/`
- `results/EXPLORATORY/deq_gradient_isolation_analysis/`
- `results/EXPLORATORY/maze_symmetric_swap_equal_residual_analysis/`

Sign convention: losses are lower-is-better. For `exact_minus_b2` gaps,
negative means exact is better. For operator-swap own-minus-other gaps, negative
means the trained model prefers its own operator.

## H1 - Cost / infeasibility

Effect: B2 needs much more compute to match exact as the maze hardens by size.
The cleanest plain-number effect is maze positive at `48x48`: exact vs matched
B2 loss gap is `-8.21e-4`, exploratory CI `[-8.41e-4, -8.00e-4]`. Break-even
compute hits the K-bank cap at the positive maze start and then stays at
`K*=256` for `12/24/48`; the corresponding `C*` values are `>196608`,
`835584`, `3440640`, and `13959168` FLOPs. DEQ has real stiff-regime loss gaps
but does not give a clean diverging C* curve: linear `K* = [1,1,1,2,4,1]`;
tanh `K* = [2,1,2,2,4,4]`.

Exploratory intervals:

| probe | knob | effect | exploratory paired-bootstrap CI |
|---|---:|---:|---:|
| maze positive | size 6 | exact - B2 loss `-1.76e-1` | `[-1.77e-1, -1.75e-1]` |
| maze positive | size 48 | exact - B2 loss `-8.21e-4` | `[-8.41e-4, -8.00e-4]` |
| DEQ linear | rho 0.95 | exact - B2 loss `-4.03e-4` | `[-5.76e-4, -2.67e-4]` |
| DEQ linear | rho 0.99 | exact - B2 loss `-1.12e-1` | `[-1.90e-1, -4.82e-2]` |
| DEQ tanh | rho 0.95 | exact - B2 loss `-1.21e-4` | `[-1.55e-4, -8.80e-5]` |
| DEQ tanh | rho 0.99 | exact - B2 loss `-2.87e-4` | `[-3.77e-4, -1.99e-4]` |

REAL? Yes for the maze positive loss gap and for high-rho DEQ loss gaps; the
intervals are far from zero. The specific H1 divergence claim is real only for
maze positive under this apparatus. DEQ is mixed and nonmonotone, so it is not a
clean C*-divergence phenomenon yet.

SURPRISING? Maze magnitude is large, but the direction is not surprising. A
reviewer can predict that truncated global iterations struggle as grid diameter
and conditioning harden. DEQ high-rho gaps are also directionally expected from
phantom-gradient / Neumann-truncation theory.

COLLISION? Severe. Maze H1 collides directly with graph diameter, conditioning,
and oversquashing. DEQ H1 collides directly with the `1/(1-rho)` memory-horizon
story. This is the floor finding, not the novel one.

CLEAN? Moderately clean as a cost story, not clean as an inductive-bias story.
The operator is the only swapped block, and compute is measured with the same
FLOP convention, but the explanation can remain entirely numerical/reach-based.

Honest read: H1 is real enough to keep as reviewer inoculation: it shows the B2
baseline was not simply under-budgeted and that matching exact can become
expensive. It is also the least novel item. The maze C* curve is the useful one;
the DEQ C* curve is too mixed to carry a divergence claim. If confirmed, this
would support the cost/infeasibility background, not the paper's central
inductive-bias thesis.

## H2 - Co-adaptation / operator-swap asymmetry

Effect: in the maze cross-eval matrix, B2-trained models at extrapolated size
prefer their own B2 eval operator over exact eval. At `48x48`, the H2 gap
`B2-trained/B2-eval - B2-trained/Exact-eval` is negative for every K. The
strongest cell is `K=2`: mean `-9.66e-4`, exploratory CI
`[-1.00e-3, -9.34e-4]`. At train size `6x6`, the sign is positive for every K,
so the effect is not a trivial in-distribution preference. DEQ has no eval-loss
operator asymmetry because the forward solve is held identical by construction;
DEQ cross-eval belongs under H3-style learned-function / gradient-channel
questions.

Exploratory intervals, maze positive:

| size | K | H2 eval gap | exploratory paired-bootstrap CI |
|---:|---:|---:|---:|
| 6 | 2 | `+1.20e-1` | `[+1.17e-1, +1.23e-1]` |
| 48 | 1 | `-9.85e-5` | `[-1.20e-4, -7.38e-5]` |
| 48 | 2 | `-9.66e-4` | `[-1.00e-3, -9.34e-4]` |
| 48 | 4 | `-7.58e-4` | `[-7.89e-4, -7.32e-4]` |
| 48 | 16 | `-1.75e-4` | `[-1.84e-4, -1.67e-4]` |
| 48 | 64 | `-5.55e-6` | `[-6.30e-6, -4.82e-6]` |

REAL? Yes as a DEV matrix effect. The `48x48` intervals are far from zero and
the sign is stable across K. The effect is also structured: it is largest at
small K and decays by `K=64`.

SURPRISING? Yes in the narrow sense. The cost story predicts exact eval should
help or at least not hurt a fixed trained model. Seeing the "better" operator
hurt B2-trained checkpoints at extrapolated size is the interesting H2 shape.

COLLISION? Not reducible to diameter or conditioning alone, because those
explain why B2 is worse, not why a B2-trained model can prefer B2 over exact at
eval. However, the new symmetric-swap negative control creates a different
collision: two approximate operators of comparable residual quality also show
own-operator preference. That means the H2 shape may be solver-form
co-adaptation or optimizer-path dependence, not exactness as such.

CLEAN? No. The exact-vs-B2 matrix is real, but the equal-residual Jacobi-vs-CG
control fails. Until that failure is explained, the H2 maze asymmetry is
mechanistically dirty.

Honest read: H2 is the most paper-shaped phenomenon if viewed alone, but it is
not clean after H4. The positive result says trained representations are
operator-dependent; the failed symmetric control says this dependence is not
specific to exactness. It remains worth understanding because it could lead to a
better exactness-specific discriminator, but in the current apparatus it should
not be confirmed as the headline without revising the control or mechanism.

## H3 - Gradient-channel isolation

Effect: with the DEQ forward solve held exact and bit-identical in both arms,
changing only the backward rule changes the learned function and held-out
metrics at high rho. At `rho=0.99`, Neumann-K training has higher exact-forward
test loss than exact-adjoint training for `K=1/2/4/32`; for example, `K=2` gives
delta test loss `+1.12e-1`, exploratory CI `[+4.72e-2, +1.90e-1]`, and
held-out function relative MSE `4.18e-3`, CI `[1.95e-3, 7.54e-3]`. The K trend
is not monotone: `K=8` is uncertain and `K=16` is negative.

Exploratory intervals:

| rho | K | delta test loss, Neumann - exact | exploratory paired-bootstrap CI |
|---:|---:|---:|---:|
| 0.7 | 1 | `+1.37e-7` | `[+7.02e-9, +3.00e-7]` |
| 0.7 | 2 | `+5.50e-8` | `[-1.29e-8, +1.35e-7]` |
| 0.99 | 1 | `+6.04e-2` | `[+2.38e-2, +1.03e-1]` |
| 0.99 | 2 | `+1.12e-1` | `[+4.72e-2, +1.90e-1]` |
| 0.99 | 4 | `+7.70e-2` | `[+3.17e-2, +1.33e-1]` |
| 0.99 | 8 | `+1.09e-3` | `[-6.99e-4, +3.01e-3]` |
| 0.99 | 16 | `-2.14e-3` | `[-3.58e-3, -9.04e-4]` |
| 0.99 | 32 | `+3.58e-2` | `[+1.39e-2, +6.23e-2]` |

Function-distance check under the same exact forward:

| rho | K | test function relative MSE | exploratory paired-bootstrap CI |
|---:|---:|---:|---:|
| 0.99 | 1 | `5.36e-3` | `[1.86e-3, 1.11e-2]` |
| 0.99 | 2 | `4.18e-3` | `[1.95e-3, 7.54e-3]` |
| 0.99 | 4 | `2.06e-3` | `[9.27e-4, 3.57e-3]` |
| 0.99 | 32 | `3.27e-3` | `[1.40e-3, 5.15e-3]` |

REAL? Yes at high rho for several K values. The hard-rho generalization gaps
and function distances are larger than DEV-seed noise in the exploratory
bootstrap. The low-rho control is not perfectly silent: `rho=0.7, K=1` has a
tiny positive delta test-loss interval.

SURPRISING? Moderately. Biased Neumann gradients near `rho -> 1` are expected to
be bad, but this probe removes the forward channel entirely. The surprising
part is not that Neumann is biased; it is that gradient exactness alone changes
the learned function under a common exact-forward evaluation.

COLLISION? Partial. The mechanism collides with phantom-gradient theory in the
sense that phantom gradients already predict bias near `rho=1`. It does not
collapse to forward approximation, diameter, or conditioning of the eval
operator, because the forward solve is exact and common. It still lives in a
scalar-gain DEQ toy, so standard optimizer-path objections remain available.

CLEAN? This is the cleanest attribution built so far. Fixed-parameter forward
equality is bitwise zero, and both trained arms are evaluated under exact
forward. The remaining confounds are optimizer dynamics, the scalar-gain model,
nonmonotone K behavior, and the small low-rho red flag.

Honest read: H3 is the strongest mechanistic candidate. It does not say
"exactness beats approximation" in a vague way; it isolates the backward channel
and shows a held-out learned-function difference when the forward numbers are
identical. The caveats are real: K is nonmonotone, one low-rho cell is not
silent, and the probe is small. But compared with H2, the attribution is much
cleaner.

## H4 - Negative controls

Effect: the controls are mixed, and two of them fail loudly. Large-`eps` maze
fails: exact still beats matched B2 at every size, even though `K*` is flat at
`16`. Low-rho DEQ mostly passes in the Phase-1 break-even harness: linear
`rho=0.5` has exact - B2 loss `+1.47e-9`, exploratory CI
`[-4.38e-9, +8.12e-9]`; tanh `rho=0.7` has `+9.60e-9`, CI
`[-2.72e-8, +4.96e-8]`. The symmetric-swap control fails: equal-residual
Jacobi-vs-CG swaps show own-operator preference at every size.

Exploratory intervals:

| control | knob | effect | exploratory paired-bootstrap CI |
|---|---:|---:|---:|
| large-eps maze | size 6 | exact - B2 loss `-6.43e-7` | `[-6.85e-7, -6.02e-7]` |
| large-eps maze | size 48 | exact - B2 loss `-9.65e-9` | `[-1.01e-8, -9.20e-9]` |
| low-rho DEQ linear | rho 0.5 | exact - B2 loss `+1.47e-9` | `[-4.38e-9, +8.12e-9]` |
| low-rho DEQ tanh | rho 0.7 | exact - B2 loss `+9.60e-9` | `[-2.72e-8, +4.96e-8]` |
| symmetric swap | size 6 | CG-trained own-minus-Jacobi `-2.37e-5` | `[-2.45e-5, -2.30e-5]` |
| symmetric swap | size 48 | CG-trained own-minus-Jacobi `-2.42e-7` | `[-2.50e-7, -2.35e-7]` |
| symmetric swap | size 48 | Jacobi-trained own-minus-CG `-1.79e-7` | `[-1.86e-7, -1.71e-7]` |

REAL? Yes, the failures are real under the DEV metrics: large-eps maze exact
advantage is bigger than bootstrap noise, and symmetric-swap own-operator
preference is stable. Low-rho DEQ break-even controls pass at the current
cutoff, aside from the H3 low-rho `rho=0.7, K=1` tiny red flag.

SURPRISING? The failures are surprising because these controls were designed to
be boring. Large-eps maze should have made globality mostly irrelevant.
Symmetric swap should have removed the special status of exactness.

COLLISION? H4 is not a collision with known theory; it is a collision with our
own story. The large-eps failure suggests the maze control still contains an
exact-solve advantage even when propagation should be short-range. The
symmetric-swap failure suggests operator-form co-adaptation may be generic, not
specific to exactness.

CLEAN? Clean as a warning signal, not as a phenomenon to sell. The controls use
the same DEV-seed discipline and the symmetric swap was corrected to use
comparable residual quality, but a failed control is evidence that the positive
story is confounded.

Honest read: H4 says the current Phase-1 story is not confirmatory-ready. Low-rho
DEQ behaves as hoped in the break-even harness, but both maze controls create
problems. The symmetric-swap failure is especially important: it directly
weakens the H2 claim that exactness is the distinctive cause of operator-swap
asymmetry. H4 should be treated as a blocking caveat for any H2-style
confirmatory design.

## Ranked recommendation for confirmation design

This is not a paper-pick decision. It is a recommendation about which 1-2
candidate phenomena look most worth freezing for a later HOLDOUT test, subject
to the Phase 2.5 human decision.

1. **H3 gradient-channel isolation.** It is the cleanest mechanistic candidate:
   the forward channel is held exact and bit-identical, the high-rho effects are
   larger than DEV-seed noise, and the held-out function-distance check argues
   against the "different weights, same function" trap. A confirmatory version
   should simplify the K grid around the robust cells, predeclare the low-rho
   control rule, and keep exact-forward evaluation for both arms.

2. **H1 maze cost/infeasibility as supporting context, not headline.** The maze
   positive cost curve is the most stable numerical effect, and it inoculates
   against "you undertrained the baseline." It is also unsurprising and collides
   badly with diameter / conditioning / oversquashing, so it should confirm only
   as context if the human decision wants a cost-story backbone.

H2 is not recommended for immediate confirmation in its current form, despite
being the most novel-looking positive effect, because H4's symmetric-swap
failure makes the exactness-specific attribution suspect. H4 itself should be
part of any confirmatory package as a required gate, not as a paper phenomenon.
