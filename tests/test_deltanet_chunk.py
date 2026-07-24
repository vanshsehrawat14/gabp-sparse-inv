"""Gates for the DeltaNet / linear-attention chunk-inverse drop-in (``demos/deltanet_chunk.py``).

Two things are gated. (1) **Forward correctness**: the chunked ``(I - A)^{-1}`` delta-rule layer
equals the token-by-token sequential delta rule exactly, at every chunk size (the chunking is an
algebraic identity, and the larger the chunk the larger the genuine triangular inverse). (2) The
**drop-in equivalence** that is the point of the demo: computing the chunk inverse with
``selinv_tril`` (analytic transpose-form VJP) vs the stock ``solve_triangular`` baseline
(autograd) gives the same forward *and the same gradients* through the whole multi-chunk layer,
and a layer trains identically through either. Capability / equivalence result, not SOTA -- see
the module docstring and ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv.demos.deltanet_chunk import (
    chunk_inverse_apply,
    delta_chunk_attention,
    equivalence_check,
    fit_teacher_student,
    sequential_delta_reference,
)

torch.manual_seed(0)


def _qkvb(batch, L, d, *, seed):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(*batch, L, d, generator=g, dtype=torch.float64)
    K = torch.randn(*batch, L, d, generator=g, dtype=torch.float64)
    V = torch.randn(*batch, L, d, generator=g, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(*batch, L, generator=g, dtype=torch.float64))
    return Q, K, V, beta


# --------------------------------------------------------------------------- #
# Forward correctness: chunked (I - A)^{-1} layer == sequential delta rule (oracle).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["selinv", "solve"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 6, 12])
def test_chunked_matches_sequential_reference(method, chunk_size):
    Q, K, V, beta = _qkvb((), 12, 5, seed=1)
    O = delta_chunk_attention(Q, K, V, beta, chunk_size, method=method)
    ref = sequential_delta_reference(Q, K, V, beta)
    assert torch.allclose(O, ref, atol=1e-10), f"chunk_size={chunk_size} method={method}"


def test_chunk_size_invariance():
    # Same output for every chunking of the same sequence -> the within-chunk inverse is doing
    # real (and correct) work; a 7-token chunk inverts a 7x7 triangular system, a 2-token one a
    # 2x2, both giving the identical delta-rule result.
    Q, K, V, beta = _qkvb((), 12, 4, seed=2)
    base = delta_chunk_attention(Q, K, V, beta, 12)
    for c in (1, 2, 3, 4, 6):
        assert torch.allclose(delta_chunk_attention(Q, K, V, beta, c), base, atol=1e-10)


def test_leading_batch_matches_per_item():
    Q, K, V, beta = _qkvb((4,), 9, 3, seed=3)
    O = delta_chunk_attention(Q, K, V, beta, 3)
    for k in range(4):
        Ok = delta_chunk_attention(Q[k], K[k], V[k], beta[k], 3)
        assert torch.allclose(O[k], Ok, atol=1e-12)


# --------------------------------------------------------------------------- #
# chunk_inverse_apply: both methods == the dense (I - A)^{-1} @ rhs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["selinv", "solve"])
def test_chunk_inverse_apply_matches_dense(method):
    g = torch.Generator().manual_seed(5)
    C, k = 6, 3
    A = torch.randn(C, C, generator=g, dtype=torch.float64).tril(-1)
    rhs = torch.randn(C, k, generator=g, dtype=torch.float64)
    out = chunk_inverse_apply(A, rhs, method=method)
    eye = torch.eye(C, dtype=torch.float64)
    ref = torch.linalg.inv(eye - A) @ rhs
    assert torch.allclose(out, ref, atol=1e-10)


def test_chunk_inverse_apply_rejects_bad_method():
    A = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        chunk_inverse_apply(A, torch.zeros(3, 2, dtype=torch.float64), method="nope")


# --------------------------------------------------------------------------- #
# The drop-in equivalence: selinv_tril (analytic backward) == solve_triangular (autograd).
# --------------------------------------------------------------------------- #
def test_forward_and_gradient_methods_agree():
    fwd_err, grad_err = equivalence_check(d=8, L=12, chunk_size=4, batch=4, seed=0)
    assert fwd_err < 1e-12, fwd_err            # identical by construction
    assert grad_err < 1e-9, grad_err           # analytic transpose-form VJP == autograd


def test_gradcheck_layer_selinv():
    # gradcheck the full multi-chunk layer through the analytic selinv_tril backward.
    Q, K, V, beta = _qkvb((2,), 6, 3, seed=7)
    leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, beta)]

    def f(Q, K, V, beta):
        return delta_chunk_attention(Q, K, V, beta, 3, method="selinv")

    assert torch.autograd.gradcheck(f, tuple(leaves), atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# End-to-end: the layer trains through the op, identically either way.
# --------------------------------------------------------------------------- #
def test_learnability_through_chunk_inverse():
    mse0, mse1 = fit_teacher_student(seed=0)        # defaults: steps=500, lr=0.1
    assert mse1 < mse0 * 1e-3                        # MSE drops by orders of magnitude


def test_training_identical_both_methods():
    # Same init / data / optimiser; only the (mathematically identical, equal-gradient) inverse
    # differs -> identical training. A short run keeps fp drift negligible.
    a0, a1 = fit_teacher_student(steps=40, seed=0, method="selinv")
    b0, b1 = fit_teacher_student(steps=40, seed=0, method="solve")
    assert a0 == pytest.approx(b0, rel=1e-9)
    assert a1 == pytest.approx(b1, rel=1e-6)
