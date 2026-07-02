# Demonstration: the DeltaNet chunk inverse as a hot-swappable differentiable op

This note records the **DeltaNet / linear-attention** demonstration. It shows the differentiable
triangular chunk inverse `T = (I − A)⁻¹` already shipped as `selinv_tril`
(`gabp_sparse_inv/nonsym.py`, [derivations.md](derivations.md) §9.4) drops, unchanged, into a
real chunked linear-attention layer and trains identically to the stock autograd baseline. Code:
`gabp_sparse_inv/demos/deltanet_chunk.py`; tests:
`tests/test_deltanet_chunk.py`.

## The op's role: the within-chunk delta rule is a triangular solve

DeltaNet / gated linear attention parallelise the *delta rule* over a chunk of `C` tokens.
Within a chunk, writing token `i` changes the state that token `j > i` reads, so the corrected
writes `W` solve a lower-triangular system (Yang et al. 2024). With `delta_i = β_i (v_i − k_iᵀ
S_prev)` and the strictly-lower coupling `L = tril(diag(β) K Kᵀ, −1)`,

    (I + L) W = delta        i.e.   W = (I − A)⁻¹ delta,   A = −L,

which is exactly the chunk inverse `selinv_tril` computes. `delta_chunk_attention` is a minimal,
faithful chunked delta-rule layer built around that one inverse (L2-normalised keys, a
strictly-causal intra-chunk read, a carried state across chunks). With the strictly-causal read
the chunked layer reproduces the token-by-token delta rule **exactly** - the chunking is an
algebraic identity, validated against `sequential_delta_reference` (an `O(L)` token loop) at
every chunk size. The larger the chunk, the larger the genuine triangular inverse, and the answer
is invariant - so the inverse is doing real, correct work, not bookkeeping.

## The result: same forward, analytic backward == autograd, trains identically

`chunk_inverse_apply` forms `T` two interchangeable ways inside the same layer:

- `method="selinv"` - `selinv_tril`, whose **analytic self-adjoint backward** is
  `bar_A = tril(Tᵀ bar_T Tᵀ, −1)` ([derivations.md](derivations.md) §9.4) with **no autograd tape**
  over the triangular solve; and
- `method="solve"` - the stock baseline `torch.linalg.solve_triangular(I − A, I)` with autograd
  differentiating the solve.

The forward is identical by construction (both call `solve_triangular`); the content is that the
**analytic backward equals autograd's** through the whole multi-chunk layer, and that the layer
**trains identically** either way. CPU, fp64; `python -m gabp_sparse_inv.demos.deltanet_chunk`
(`L=12`, `chunk_size=4`, `d=8`):

| quantity | value |
|---|---|
| forward `max|selinv − solve|` | `0` (identical by construction) |
| gradient `max|selinv − solve|` over `(Q,K,V,β)` | `~3e-15` (analytic == autograd) |
| teacher-student fit, MSE start → end (`selinv`) | `5.2 → 4.7e-5` |
| teacher-student fit, MSE start → end (`solve`) | `5.2 → 4.7e-5` (bit-identical) |

So the op is genuinely hot-swappable: the differentiable selected-inverse primitive drops into
the layer unchanged and is indistinguishable, forward and backward, from the autograd baseline.

## Scope and honest caveats

- **This is a drop-in-equivalence / capability result, not a DeltaNet reimplementation or a SOTA
  claim** ([ROADMAP.md](ROADMAP.md): "the contribution must be the inversion primitive's role, not
  a re-derivation of DeltaNet"). The toy task is a synthetic teacher-student fit - its job is to
  show the op trains, not to beat a recall benchmark - the same honesty bar as [MAZE.md](MAZE.md)
  / [DEQ.md](DEQ.md).
- **The dense chunk inverse is `O(C³)` with no sparsity win.** Here the value is the analytic
  `O(C²)` self-adjoint backward (it never tapes the triangular solve) and that the *same*
  selected-inverse op serves both the sparse SPD/non-symmetric kernels and this dense triangular
  case - one primitive, many layers. `selinv_tril` is first-order (the custom-`Function`
  contract); use the functional kernels if higher-order derivatives are needed.
- The machine-independent facts are gated in `tests/test_deltanet_chunk.py`: the chunked layer
  `==` the sequential delta rule at every chunk size, chunk-size invariance, both methods agree
  forward (`<1e-12`) and in gradients (`<1e-9`), `gradcheck`, and an orders-of-magnitude
  learnability drop trained through the op.
