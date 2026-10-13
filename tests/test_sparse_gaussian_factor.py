"""Gates for the persistent, differentiable SPD junction factor interface."""

from __future__ import annotations

import pytest
import torch

from gabp_sparse_inv import (
    SparseGaussianFactor,
    analyze_junction,
    grid_edges,
    junction_logdet,
    junction_marginal_log_likelihood,
    junction_solve,
    random_spd_graph,
    sample_gaussian_junction,
    selected_inverse_junction,
)
from gabp_sparse_inv.gmrf_grid import grid_gmrf_precision


def _problem(seed=0, *, batch_shape=()):
    n, b = 9, 2
    return random_spd_graph(
        n,
        grid_edges(3, 3),
        b,
        seed=seed,
        diag_load=2.0,
        batch_shape=batch_shape,
    )


def test_one_factor_reuses_logdet_solve_selinv_and_sampling():
    sp = _problem()
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)
    factor = analysis.factor(sp.diag, sp.edge_val, check=True)
    assert isinstance(factor, SparseGaussianFactor)
    assert analysis.symbolic_analyses == 1
    assert analysis.numeric_factorizations == 1

    rhs = torch.randn(
        sp.num_nodes,
        sp.block_size,
        3,
        dtype=sp.diag.dtype,
        generator=torch.Generator().manual_seed(7),
    )
    ld = factor.logdet()
    x = factor.solve(rhs)
    G_diag, S_index, G_lower = factor.selected_inverse()
    marginal_diag = factor.marginal_diag()

    ref_ld = junction_logdet(sp.diag, sp.edge_index, sp.edge_val, analysis.order)
    ref_x = junction_solve(sp.diag, sp.edge_index, sp.edge_val, rhs, analysis.order)
    ref_diag, ref_index, ref_lower = selected_inverse_junction(
        sp.diag, sp.edge_index, sp.edge_val, analysis.order
    )
    assert torch.allclose(ld, ref_ld, atol=1e-12)
    assert torch.allclose(x, ref_x, atol=1e-12)
    assert torch.allclose(G_diag, ref_diag, atol=1e-12)
    assert torch.equal(S_index, ref_index)
    assert torch.allclose(G_lower, ref_lower, atol=1e-12)
    assert torch.equal(marginal_diag, torch.diagonal(G_diag, dim1=-2, dim2=-1))

    generator = torch.Generator().manual_seed(11)
    samples = factor.sample(5, generator=generator)
    ref_generator = torch.Generator().manual_seed(11)
    ref_samples = sample_gaussian_junction(
        sp.diag,
        sp.edge_index,
        sp.edge_val,
        5,
        order=analysis.order,
        generator=ref_generator,
    )
    assert torch.allclose(samples, ref_samples, atol=1e-12)

    assert factor.operation_counts() == {
        "symbolic_analyses": 1,
        "numeric_factorizations": 1,
        "analysis_factorizations_created": 1,
        "logdet": 1,
        "solve": 1,
        "selected_inverse": 1,
        "marginal_diag": 1,
        "sample": 1,
    }


def test_combined_loss_gradient_matches_three_functional_factorizations():
    sp = _problem(seed=3)
    rhs = torch.randn(
        sp.num_nodes,
        sp.block_size,
        dtype=sp.diag.dtype,
        generator=torch.Generator().manual_seed(5),
    )

    d_reuse = sp.diag.clone().requires_grad_(True)
    e_reuse = sp.edge_val.clone().requires_grad_(True)
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)
    factor = analysis.factor(d_reuse, e_reuse)
    gd_reuse, _si, gl_reuse = factor.selected_inverse()
    loss_reuse = (
        0.3 * factor.logdet()
        + 0.1 * factor.solve(rhs).square().sum()
        + 0.01 * gd_reuse.square().sum()
        + 0.02 * gl_reuse.square().sum()
    )
    grad_reuse = torch.autograd.grad(loss_reuse, (d_reuse, e_reuse))

    d_ref = sp.diag.clone().requires_grad_(True)
    e_ref = sp.edge_val.clone().requires_grad_(True)
    gd_ref, _si_ref, gl_ref = selected_inverse_junction(
        d_ref, sp.edge_index, e_ref, analysis.order
    )
    loss_ref = (
        0.3 * junction_logdet(d_ref, sp.edge_index, e_ref, analysis.order)
        + 0.1 * junction_solve(d_ref, sp.edge_index, e_ref, rhs, analysis.order).square().sum()
        + 0.01 * gd_ref.square().sum()
        + 0.02 * gl_ref.square().sum()
    )
    grad_ref = torch.autograd.grad(loss_ref, (d_ref, e_ref))

    assert analysis.numeric_factorizations == 1
    assert torch.allclose(loss_reuse, loss_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(grad_reuse[0], grad_ref[0], atol=1e-10, rtol=1e-9)
    assert torch.allclose(grad_reuse[1], grad_ref[1], atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_level_set_factor_reuses_every_query_and_matches_reference(batch_shape):
    sp = _problem(seed=6, batch_shape=batch_shape)
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)
    reference = analysis.factor(sp.diag, sp.edge_val)
    level_set = analysis.factor(sp.diag, sp.edge_val, check=True, batched=True)

    rhs = torch.randn(
        *batch_shape,
        sp.num_nodes,
        sp.block_size,
        3,
        dtype=sp.diag.dtype,
        generator=torch.Generator().manual_seed(17),
    )
    assert level_set.backend == "level_set"
    assert analysis.tensor_schedules == 1
    assert torch.allclose(level_set.logdet(), reference.logdet(), atol=1e-12)
    assert torch.allclose(level_set.solve(rhs), reference.solve(rhs), atol=1e-12)

    gd_ref, si_ref, gl_ref = reference.selected_inverse()
    gd_got, si_got, gl_got = level_set.selected_inverse()
    assert torch.equal(si_got, si_ref)
    assert torch.allclose(gd_got, gd_ref, atol=1e-12)
    assert torch.allclose(gl_got, gl_ref, atol=1e-12)
    assert torch.equal(
        level_set.marginal_diag(), torch.diagonal(gd_got, dim1=-2, dim2=-1)
    )

    ref_generator = torch.Generator().manual_seed(23)
    got_generator = torch.Generator().manual_seed(23)
    assert torch.allclose(
        level_set.sample(4, generator=got_generator),
        reference.sample(4, generator=ref_generator),
        atol=1e-12,
    )

    # A second numeric factor on the same device reuses the values-independent tensor plan.
    analysis.factor(sp.diag, sp.edge_val, batched=True)
    assert analysis.tensor_schedules == 1
    assert level_set.operation_counts()["numeric_factorizations"] == 1
    assert level_set.operation_counts()["selected_inverse"] == 1


def test_level_set_factor_combined_gradient_matches_reference_factor():
    sp = _problem(seed=9)
    rhs = torch.randn(
        sp.num_nodes,
        sp.block_size,
        dtype=sp.diag.dtype,
        generator=torch.Generator().manual_seed(29),
    )
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)

    def evaluate(*, batched):
        diag = sp.diag.clone().requires_grad_(True)
        edge = sp.edge_val.clone().requires_grad_(True)
        factor = analysis.factor(diag, edge, batched=batched)
        gd, _si, gl = factor.selected_inverse()
        loss = (
            0.3 * factor.logdet()
            + 0.1 * factor.solve(rhs).square().sum()
            + 0.01 * gd.square().sum()
            + 0.02 * gl.square().sum()
        )
        return loss, torch.autograd.grad(loss, (diag, edge))

    ref_loss, ref_grad = evaluate(batched=False)
    got_loss, got_grad = evaluate(batched=True)
    assert torch.allclose(got_loss, ref_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(got_grad[0], ref_grad[0], atol=1e-10, rtol=1e-9)
    assert torch.allclose(got_grad[1], ref_grad[1], atol=1e-10, rtol=1e-9)


def test_symbolic_plan_refactors_new_values_without_reanalysis():
    sp0 = _problem(seed=0)
    sp1 = _problem(seed=1)
    analysis = analyze_junction(sp0.num_nodes, sp0.edge_index)
    f0 = analysis.factor(sp0.diag, sp0.edge_val)
    f1 = analysis.factor(sp1.diag, sp1.edge_val)

    assert analysis.symbolic_analyses == 1
    assert analysis.numeric_factorizations == 2
    assert f0.factorization_id == 1 and f1.factorization_id == 2
    assert torch.allclose(
        f1.logdet(),
        junction_logdet(sp1.diag, sp1.edge_index, sp1.edge_val, analysis.order),
        atol=1e-12,
    )


def test_grid_likelihood_reuses_qpost_factor_for_logdet_and_solve():
    rows, cols = 3, 3
    diag, edge_index, edge_val = grid_gmrf_precision(rows, cols, 1.2, 0.7)
    analysis = analyze_junction(rows * cols, edge_index)
    y = torch.randn(rows * cols, 1, dtype=diag.dtype)
    value = junction_marginal_log_likelihood(
        diag, edge_index, edge_val, y, 0.4, analysis=analysis
    )
    assert torch.isfinite(value)
    # One factor for Q and one for Q_post.  Q_post serves both logdet and solve.
    assert analysis.symbolic_analyses == 1
    assert analysis.numeric_factorizations == 2


@pytest.mark.parametrize("batched", [False, True])
def test_batch_and_compute_dtype_contract(batched):
    sp = _problem(seed=8, batch_shape=(3,))
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)
    d32, e32 = sp.diag.float(), sp.edge_val.float()
    factor = analysis.factor(
        d32, e32, compute_dtype=torch.float64, batched=batched
    )
    rhs = torch.randn(3, sp.num_nodes, sp.block_size, dtype=torch.float32)
    x = factor.solve(rhs)
    md = factor.marginal_diag()
    assert factor.compute_dtype == torch.float64
    assert factor.result_dtype == torch.float32
    assert x.shape == rhs.shape and x.dtype == torch.float32
    assert md.shape == (3, sp.num_nodes, sp.block_size) and md.dtype == torch.float32


def test_analysis_and_factor_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="positive integer"):
        analyze_junction(0, torch.zeros((2, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="lower-triangular"):
        analyze_junction(2, torch.tensor([[0], [1]]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_junction(2, torch.tensor([[1, 1], [0, 0]]))

    sp = _problem()
    analysis = analyze_junction(sp.num_nodes, sp.edge_index)
    with pytest.raises(ValueError, match="diag must have shape"):
        analysis.factor(sp.diag[:-1], sp.edge_val)
    factor = analysis.factor(sp.diag, sp.edge_val)
    with pytest.raises(ValueError, match="b must have shape"):
        factor.solve(torch.randn(3))
    with pytest.raises(ValueError, match="positive integer"):
        factor.sample(0)


@pytest.mark.parametrize("batched", [False, True])
def test_factor_solve_reads_an_ambiguous_rhs_as_columns(batched):
    diag = torch.tensor([[[4.0, 0.1], [0.1, 3.0]], [[5.0, 0.2], [0.2, 6.0]]], dtype=torch.float64)
    edge = torch.tensor([[[0.2, 0.1], [-0.1, 0.3]]], dtype=torch.float64)
    A = torch.zeros(4, 4, dtype=torch.float64)
    A[:2, :2], A[2:, 2:], A[2:, :2], A[:2, 2:] = diag[0], diag[1], edge[0], edge[0].T
    rhs = torch.arange(1.0, 9.0, dtype=torch.float64).reshape(2, 2, 2)
    factor = analyze_junction(2, torch.tensor([[1], [0]])).factor(diag, edge, batched=batched)
    ref = torch.linalg.solve(A, rhs.reshape(4, 2)).reshape(2, 2, 2)
    torch.testing.assert_close(factor.solve(rhs), ref)
    torch.testing.assert_close(factor.solve(rhs[..., 0]), ref[..., 0])
