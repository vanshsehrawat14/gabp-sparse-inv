import torch

from gabp_sparse_inv.bench.deq_gmres_falsifier import (
    chain_edges,
    fixed_point_with_backward,
    fixed_product_gmres,
    neumann_adjoint_products,
)
from gabp_sparse_inv.demos.deq_fixedpoint import random_coupling


def test_fixed_product_gmres_counts_products_and_terminates_at_dimension():
    A = torch.tensor(
        [[4.0, 1.0, 0.0], [-2.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[[1.0], [2.0], [-1.0]], [[2.0], [-1.0], [3.0]]], dtype=torch.float64)
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        return A @ value

    got, diagnostic = fixed_product_gmres(matvec, rhs, 3)
    expected = torch.linalg.solve(A, rhs)
    assert calls == diagnostic["products"] == 3
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)


def test_gmres_residual_is_nonincreasing_with_products():
    A = torch.tensor(
        [[3.0, 2.0, 0.0, 0.0], [-1.0, 2.0, 1.0, 0.0],
         [0.0, -2.0, 3.0, 1.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[[1.0], [0.5], [-2.0], [3.0]]], dtype=torch.float64)
    residuals = []
    for products in (1, 2, 3, 4):
        _, diagnostic = fixed_product_gmres(lambda value: A @ value, rhs, products)
        residuals.append(diagnostic["relative_residual_max"])
    assert all(left >= right - 1e-13 for left, right in zip(residuals, residuals[1:]))


def test_backward_mode_does_not_change_exact_forward():
    n = 8
    edge_index = chain_edges(n)
    Wd0, Wl0, Wu0 = random_coupling(n, edge_index, 1, seed=17, rho=1.0)
    rhs = torch.randn(4, n, 1, generator=torch.Generator().manual_seed(18), dtype=torch.float64)
    outputs = []
    for mode in ("exact", "neumann", "gmres"):
        raw = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        outputs.append(
            fixed_point_with_backward(
                raw, Wd0, edge_index, Wl0, Wu0, rhs, mode=mode, products=4
            )
        )
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_full_dimension_gmres_backward_matches_exact_gradient():
    n = 6
    edge_index = chain_edges(n)
    Wd0, Wl0, Wu0 = random_coupling(n, edge_index, 1, seed=23, rho=1.0)
    rhs = torch.randn(3, n, 1, generator=torch.Generator().manual_seed(24), dtype=torch.float64)
    target = torch.randn(3, n, 1, generator=torch.Generator().manual_seed(25), dtype=torch.float64)
    gradients = []
    for mode in ("exact", "gmres"):
        raw = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
        pred = fixed_point_with_backward(
            raw, Wd0, edge_index, Wl0, Wu0, rhs, mode=mode, products=n
        )
        torch.mean((pred - target) ** 2).backward()
        gradients.append(raw.grad.detach())
    torch.testing.assert_close(gradients[1], gradients[0], rtol=1e-10, atol=1e-12)


def test_product_budget_reaches_both_finite_backward_arms():
    n = 9
    edge_index = chain_edges(n)
    Wd0, Wl0, Wu0 = random_coupling(n, edge_index, 1, seed=31, rho=1.0)
    rhs = torch.randn(4, n, 1, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    target = torch.randn(4, n, 1, generator=torch.Generator().manual_seed(33), dtype=torch.float64)
    for mode in ("neumann", "gmres"):
        gradients = []
        for products in (1, 4):
            raw = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
            pred = fixed_point_with_backward(
                raw, Wd0, edge_index, Wl0, Wu0, rhs, mode=mode, products=products
            )
            torch.mean((pred - target) ** 2).backward()
            gradients.append(float(raw.grad))
        assert gradients[0] != gradients[1]


def test_neumann_product_counter_matches_budget():
    n = 7
    edge_index = chain_edges(n)
    Wd, Wl, Wu = random_coupling(n, edge_index, 1, seed=41, rho=0.8)
    rhs = torch.randn(2, n, 1, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    _, diagnostic = neumann_adjoint_products(Wd, edge_index, Wl, Wu, rhs, 5)
    assert diagnostic == {"products": 5, "terms": 6}
