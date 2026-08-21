"""EXPLORATORY residual-optimal adversary for the Paper 1 DEQ result.

The forward model is bit-identical across arms.  Only the implicit adjoint used
during scalar-gain training changes:

* ``exact`` uses the structured transpose solve;
* ``neumann`` uses ``P + 1`` terms and therefore exactly ``P`` products by
  ``J.T``; and
* ``gmres`` uses exactly ``P`` Arnoldi products by ``(I - J).T``.

The experiment is deliberately development-only.  It records paired raw
trajectories and continuous loss ratios; it does not issue a confirmatory
verdict or consume a HOLDOUT seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from ..demos.deq_fixedpoint import assemble_blocks, random_coupling, spmv
from ..nonsym_junction import nonsym_junction_solve


PROTOCOL = Path("docs/PREREGISTRATION_DEQ_GMRES_EXPLORATORY_2026-08-19.md")
RESULTS = Path("results/EXPLORATORY/deq_gmres_falsifier")
CAMPAIGN_UUID = "d3d1654b-3f65-4272-8f16-a58e027668d3"
CALIBRATION_SEEDS = tuple(range(4000, 4020))
DEFAULT_RHOS = (0.5, 0.98, 0.99)
DEFAULT_PRODUCTS = (1, 2, 4, 8, 16, 32)
MAX_GAIN = 0.999
MATCH_TOL = 0.05
MATCHABILITY_BAR = 0.60
PREDICTIONS = {
    "mechanism": (
        "At rho in {0.98,0.99} and P=32, GMRES will have at most half the "
        "Neumann relative residual on at least 15/20 paired seeds."
    ),
    "training": (
        "At rho in {0.98,0.99} and P in {16,32}, the median paired "
        "log10(L_gmres/L_neumann) will be <= -1 for at least one cell."
    ),
    "control": (
        "At rho=0.5 and P=32, both finite arms will have median initial "
        "relative adjoint error below 1e-4."
    ),
}


def chain_edges(n: int) -> Tensor:
    """Lower-triangular path edges ``(i, i-1)``."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.stack([torch.arange(1, n), torch.arange(n - 1)])


def gain_from_raw(raw_gain: Tensor) -> Tensor:
    """Map an unconstrained scalar to a contraction gain below one."""
    return MAX_GAIN * torch.sigmoid(raw_gain)


def raw_from_gain(gain: float) -> float:
    p = min(max(float(gain) / MAX_GAIN, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def im_minus_blocks(Wd: Tensor, edge_index: Tensor, Wl: Tensor, Wu: Tensor):
    eye = torch.eye(Wd.shape[-1], dtype=Wd.dtype, device=Wd.device)
    return eye - Wd, -Wl, -Wu


def fixed_product_gmres(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    products: int,
) -> tuple[Tensor, dict]:
    """Unrestarted batched zero-start GMRES using exactly ``products`` matvecs.

    Two-pass modified Gram--Schmidt controls loss of orthogonality.  The
    returned residual is the Arnoldi least-squares residual, so computing it
    spends no hidden operator product.
    """
    if products < 1:
        raise ValueError("GMRES products must be >= 1")
    if rhs.ndim < 2:
        raise ValueError("rhs must have vector axes")

    vector_shape = rhs.shape[-2:]
    batch_shape = rhs.shape[:-2]
    flat_rhs = rhs.reshape(*batch_shape, -1)
    beta = torch.linalg.vector_norm(flat_rhs, dim=-1)
    if bool(torch.any(beta == 0)):
        raise RuntimeError("GMRES received a zero right-hand side")

    q: list[Tensor] = [flat_rhs / beta.unsqueeze(-1)]
    H = rhs.new_zeros((*batch_shape, products + 1, products))
    product_count = 0
    min_relative_arnoldi_norm = math.inf

    for j in range(products):
        w = matvec(q[j].reshape(*batch_shape, *vector_shape)).reshape(*batch_shape, -1)
        product_count += 1
        Q = torch.stack(q, dim=-1)
        h1 = torch.einsum("...dk,...d->...k", Q, w)
        w = w - torch.einsum("...dk,...k->...d", Q, h1)
        h2 = torch.einsum("...dk,...d->...k", Q, w)
        w = w - torch.einsum("...dk,...k->...d", Q, h2)
        h = h1 + h2
        H[..., : j + 1, j] = h

        h_next = torch.linalg.vector_norm(w, dim=-1)
        relative = h_next / beta
        min_relative_arnoldi_norm = min(min_relative_arnoldi_norm, float(relative.min()))
        threshold = 100.0 * torch.finfo(rhs.dtype).eps
        breakdown = bool(torch.any(relative <= threshold))
        if breakdown and j + 1 < products:
            raise RuntimeError(
                f"GMRES lucky/numerical breakdown before product {j + 1}; "
                f"min relative Arnoldi norm={float(relative.min()):.3e}"
            )
        H[..., j + 1, j] = h_next
        if j + 1 < products:
            q.append(w / h_next.unsqueeze(-1))

    least_squares_rhs = rhs.new_zeros((*batch_shape, products + 1, 1))
    least_squares_rhs[..., 0, 0] = beta
    coeff = torch.linalg.lstsq(H, least_squares_rhs).solution[..., :products, 0]
    Q = torch.stack(q[:products], dim=-1)
    solution = torch.einsum("...dk,...k->...d", Q, coeff).reshape(
        *batch_shape, *vector_shape
    )
    residual = H @ coeff.unsqueeze(-1) - least_squares_rhs
    relative_residual = torch.linalg.vector_norm(residual[..., 0], dim=-1) / beta
    if product_count != products:
        raise AssertionError(f"charged {product_count} products for budget {products}")
    return solution, {
        "products": product_count,
        "relative_residual_max": float(relative_residual.max()),
        "relative_residual_median": float(relative_residual.median()),
        "min_relative_arnoldi_norm": min_relative_arnoldi_norm,
    }


def gmres_adjoint(
    Wd: Tensor,
    edge_index: Tensor,
    Wl: Tensor,
    Wu: Tensor,
    grad: Tensor,
    products: int,
) -> tuple[Tensor, dict]:
    """Approximate ``(I-W)^{-T} grad`` at a fixed product budget."""

    def matvec_t(v: Tensor) -> Tensor:
        return v - spmv(Wd, edge_index, Wl, Wu, v, transpose=True)

    return fixed_product_gmres(matvec_t, grad, products)


def neumann_adjoint_products(
    Wd: Tensor,
    edge_index: Tensor,
    Wl: Tensor,
    Wu: Tensor,
    grad: Tensor,
    products: int,
) -> tuple[Tensor, dict]:
    """Return ``products + 1`` Neumann terms after exactly ``products`` products."""
    if products < 1:
        raise ValueError("Neumann products must be >= 1")
    out = grad
    term = grad
    count = 0
    for _ in range(products):
        term = spmv(Wd, edge_index, Wl, Wu, term, transpose=True)
        out = out + term
        count += 1
    if count != products:
        raise AssertionError(f"charged {count} Neumann products for budget {products}")
    return out, {"products": count, "terms": products + 1}


class _ScalarGainAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_gain, Wd0, edge_index, Wl0, Wu0, binj, mode, products):
        gain = gain_from_raw(raw_gain)
        Wd, Wl, Wu = gain * Wd0, gain * Wl0, gain * Wu0
        Ad, Al, Au = im_minus_blocks(Wd, edge_index, Wl, Wu)
        z = nonsym_junction_solve(Ad, edge_index, Al, Au, binj)
        ctx.save_for_backward(raw_gain, Wd0, Wl0, Wu0, z)
        ctx.edge_index = edge_index
        ctx.mode = str(mode)
        ctx.products = int(products)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        raw_gain, Wd0, Wl0, Wu0, z = ctx.saved_tensors
        edge_index = ctx.edge_index
        gain = gain_from_raw(raw_gain)
        Wd, Wl, Wu = gain * Wd0, gain * Wl0, gain * Wu0
        Ad, Al, Au = im_minus_blocks(Wd, edge_index, Wl, Wu)

        with torch.no_grad():
            if ctx.mode == "exact":
                u = nonsym_junction_solve(Ad, edge_index, Al, Au, grad_z, transpose=True)
            elif ctx.mode == "neumann":
                u, diagnostic = neumann_adjoint_products(
                    Wd, edge_index, Wl, Wu, grad_z, ctx.products
                )
                if diagnostic["products"] != ctx.products:
                    raise AssertionError("Neumann product counter drifted")
            elif ctx.mode == "gmres":
                u, diagnostic = gmres_adjoint(
                    Wd, edge_index, Wl, Wu, grad_z, ctx.products
                )
                if diagnostic["products"] != ctx.products:
                    raise AssertionError("GMRES product counter drifted")
            else:
                raise ValueError(f"unknown backward mode {ctx.mode!r}")

            df_dgain = spmv(Wd0, edge_index, Wl0, Wu0, z)
            grad_gain = (u * df_dgain).sum()
            sig = torch.sigmoid(raw_gain)
            grad_raw = grad_gain * MAX_GAIN * sig * (1 - sig)
        return grad_raw.reshape_as(raw_gain), None, None, None, None, None, None, None


def fixed_point_with_backward(
    raw_gain: Tensor,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
    *,
    mode: str,
    products: int,
) -> Tensor:
    return _ScalarGainAdjoint.apply(
        raw_gain, Wd0, edge_index, Wl0, Wu0, binj, mode, products
    )


def exact_forward(
    raw_value: float | Tensor,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
) -> Tensor:
    raw = torch.as_tensor(raw_value, dtype=torch.float64)
    gain = gain_from_raw(raw)
    Ad, Al, Au = im_minus_blocks(gain * Wd0, edge_index, gain * Wl0, gain * Wu0)
    return nonsym_junction_solve(Ad, edge_index, Al, Au, binj)


def teacher_targets(
    rho: float,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
) -> Tensor:
    Ad, Al, Au = im_minus_blocks(rho * Wd0, edge_index, rho * Wl0, rho * Wu0)
    return nonsym_junction_solve(Ad, edge_index, Al, Au, binj).detach()


def eval_loss(
    raw_value: float,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
    target: Tensor,
) -> dict:
    with torch.no_grad():
        pred = exact_forward(raw_value, Wd0, edge_index, Wl0, Wu0, binj)
        loss = torch.mean((pred - target) ** 2)
    return {"loss": float(loss), "pred": pred}


def train_gain(
    raw_init: float,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
    target: Tensor,
    *,
    mode: str,
    products: int,
    steps: int,
    lr: float,
) -> dict:
    raw = torch.tensor(raw_init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    trajectory = [float(raw.detach())]
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        pred = fixed_point_with_backward(
            raw, Wd0, edge_index, Wl0, Wu0, binj, mode=mode, products=products
        )
        loss = torch.mean((pred - target) ** 2)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
        trajectory.append(float(raw.detach()))
    final = eval_loss(
        trajectory[-1], Wd0, edge_index, Wl0, Wu0, binj, target
    )["loss"]
    losses.append(final)
    return {
        "mode": mode,
        "products": products,
        "neumann_terms": products + 1 if mode == "neumann" else None,
        "raw": trajectory[-1],
        "gain": float(gain_from_raw(raw).detach()),
        "trajectory_raw": trajectory,
        "trajectory_exact_forward_loss": losses,
        "seconds": time.perf_counter() - t0,
    }


def matchability(trajectory_losses: list[float], exact_final_loss: float) -> dict:
    denom = max(exact_final_loss, torch.finfo(torch.float64).tiny)
    relative = [abs(value - exact_final_loss) / denom for value in trajectory_losses]
    in_band = [index for index, value in enumerate(relative) if value <= MATCH_TOL]
    return {
        "matched": bool(in_band),
        "first_step": in_band[0] if in_band else None,
        "best_relative_gap": min(relative),
    }


def adjoint_diagnostic(
    raw_init: float,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    binj: Tensor,
    target: Tensor,
    products: int,
    *,
    verify_dense: bool,
) -> dict:
    with torch.no_grad():
        raw = torch.tensor(raw_init, dtype=torch.float64)
        gain = gain_from_raw(raw)
        Wd, Wl, Wu = gain * Wd0, gain * Wl0, gain * Wu0
        pred = exact_forward(raw, Wd0, edge_index, Wl0, Wu0, binj)
        grad = 2.0 * (pred - target) / pred.numel()
        Ad, Al, Au = im_minus_blocks(Wd, edge_index, Wl, Wu)
        exact = nonsym_junction_solve(Ad, edge_index, Al, Au, grad, transpose=True)
        neumann, neumann_meta = neumann_adjoint_products(
            Wd, edge_index, Wl, Wu, grad, products
        )
        gmres, gmres_meta = gmres_adjoint(
            Wd, edge_index, Wl, Wu, grad, products
        )
        exact_vs_dense = None
        if verify_dense:
            dense_A = assemble_blocks(Ad, edge_index, Al, Au)
            dense_first = torch.linalg.solve(
                dense_A.mT, grad[0].reshape(-1, 1)
            ).reshape_as(exact[0])
            exact_vs_dense = float(
                torch.linalg.vector_norm(exact[0] - dense_first)
                / torch.linalg.vector_norm(dense_first)
            )

        def score(value: Tensor) -> dict:
            residual = value - spmv(
                Wd, edge_index, Wl, Wu, value, transpose=True
            ) - grad
            return {
                "relative_adjoint_error": float(
                    torch.linalg.vector_norm(value - exact)
                    / torch.linalg.vector_norm(exact)
                ),
                "relative_residual": float(
                    torch.linalg.vector_norm(residual)
                    / torch.linalg.vector_norm(grad)
                ),
            }

    return {
        "products": products,
        "structured_exact_vs_dense_relative_error_first_rhs": exact_vs_dense,
        "neumann": score(neumann),
        "neumann_internal": neumann_meta,
        "gmres": score(gmres),
        "gmres_internal": gmres_meta,
    }


def _cell(
    seed: int,
    rho: float,
    Wd0: Tensor,
    edge_index: Tensor,
    Wl0: Tensor,
    Wu0: Tensor,
    b_train: Tensor,
    b_test: Tensor,
    *,
    products_grid: tuple[int, ...],
    steps: int,
    lr: float,
) -> dict:
    y_train = teacher_targets(rho, Wd0, edge_index, Wl0, Wu0, b_train)
    y_test = teacher_targets(rho, Wd0, edge_index, Wl0, Wu0, b_test)
    raw_init = raw_from_gain(min(0.2, max(0.05, 0.5 * rho)))

    exact = train_gain(
        raw_init,
        Wd0,
        edge_index,
        Wl0,
        Wu0,
        b_train,
        y_train,
        mode="exact",
        products=1,
        steps=steps,
        lr=lr,
    )
    exact_train = eval_loss(
        exact["raw"], Wd0, edge_index, Wl0, Wu0, b_train, y_train
    )
    exact_test = eval_loss(
        exact["raw"], Wd0, edge_index, Wl0, Wu0, b_test, y_test
    )

    arms = {}
    for products in products_grid:
        diagnostic = adjoint_diagnostic(
            raw_init,
            Wd0,
            edge_index,
            Wl0,
            Wu0,
            b_train,
            y_train,
            products,
            verify_dense=products == max(products_grid),
        )
        paired = {}
        for mode in ("neumann", "gmres"):
            trained = train_gain(
                raw_init,
                Wd0,
                edge_index,
                Wl0,
                Wu0,
                b_train,
                y_train,
                mode=mode,
                products=products,
                steps=steps,
                lr=lr,
            )
            train_eval = eval_loss(
                trained["raw"], Wd0, edge_index, Wl0, Wu0, b_train, y_train
            )
            test_eval = eval_loss(
                trained["raw"], Wd0, edge_index, Wl0, Wu0, b_test, y_test
            )
            paired[mode] = {
                "training": trained,
                "matchability": matchability(
                    trained["trajectory_exact_forward_loss"], exact_train["loss"]
                ),
                "final_train_loss_exact_forward": train_eval["loss"],
                "final_test_loss_exact_forward": test_eval["loss"],
                "train_loss_ratio_to_exact": train_eval["loss"] / exact_train["loss"],
                "test_loss_delta_from_exact": test_eval["loss"] - exact_test["loss"],
            }
        arms[str(products)] = {
            "adjoint_at_initialization": diagnostic,
            "neumann": paired["neumann"],
            "gmres": paired["gmres"],
            "paired_log10_train_loss_gmres_over_neumann": math.log10(
                paired["gmres"]["final_train_loss_exact_forward"]
                / paired["neumann"]["final_train_loss_exact_forward"]
            ),
        }
        print(f"seed {seed} rho {rho:g} P {products} complete", flush=True)
    return {
        "seed": seed,
        "rho": rho,
        "raw_init": raw_init,
        "exact": {
            "training": exact,
            "final_train_loss": exact_train["loss"],
            "final_test_loss": exact_test["loss"],
        },
        "arms": arms,
    }


def _median(values) -> float:
    return float(statistics.median(float(value) for value in values))


def summarize(records: list[dict], rhos, products_grid) -> dict:
    rows = []
    for rho in rhos:
        cells = [cell for record in records for cell in record["cells"] if cell["rho"] == rho]
        for products in products_grid:
            key = str(products)
            neu = [cell["arms"][key]["neumann"] for cell in cells]
            gmr = [cell["arms"][key]["gmres"] for cell in cells]
            neu_fraction = sum(row["matchability"]["matched"] for row in neu) / len(neu)
            gmr_fraction = sum(row["matchability"]["matched"] for row in gmr) / len(gmr)
            if gmr_fraction >= MATCHABILITY_BAR and neu_fraction < MATCHABILITY_BAR:
                classification = "GMRES_OPENS_COMPARABLE_FIT"
            elif gmr_fraction < MATCHABILITY_BAR and neu_fraction < MATCHABILITY_BAR:
                classification = "JOINT_NONATTAINMENT_UNRESOLVED"
            elif gmr_fraction >= MATCHABILITY_BAR and neu_fraction >= MATCHABILITY_BAR:
                classification = "BOTH_ATTAIN"
            else:
                classification = "GMRES_WORSE_OR_UNSTABLE"
            rows.append(
                {
                    "rho": rho,
                    "products": products,
                    "n": len(cells),
                    "neumann_matchable_fraction": neu_fraction,
                    "gmres_matchable_fraction": gmr_fraction,
                    "median_log10_train_loss_gmres_over_neumann": _median(
                        cell["arms"][key]["paired_log10_train_loss_gmres_over_neumann"]
                        for cell in cells
                    ),
                    "median_neumann_train_loss_ratio_to_exact": _median(
                        row["train_loss_ratio_to_exact"] for row in neu
                    ),
                    "median_gmres_train_loss_ratio_to_exact": _median(
                        row["train_loss_ratio_to_exact"] for row in gmr
                    ),
                    "median_neumann_initial_relative_adjoint_error": _median(
                        cell["arms"][key]["adjoint_at_initialization"]["neumann"]
                        ["relative_adjoint_error"]
                        for cell in cells
                    ),
                    "median_gmres_initial_relative_adjoint_error": _median(
                        cell["arms"][key]["adjoint_at_initialization"]["gmres"]
                        ["relative_adjoint_error"]
                        for cell in cells
                    ),
                    "median_neumann_initial_relative_residual": _median(
                        cell["arms"][key]["adjoint_at_initialization"]["neumann"]
                        ["relative_residual"]
                        for cell in cells
                    ),
                    "median_gmres_initial_relative_residual": _median(
                        cell["arms"][key]["adjoint_at_initialization"]["gmres"]
                        ["relative_residual"]
                        for cell in cells
                    ),
                    "classification": classification,
                }
            )
    return {"rows": rows}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def validate_freeze(args) -> None:
    if args.campaign_uuid != CAMPAIGN_UUID:
        raise RuntimeError("campaign UUID does not match the frozen protocol")
    if args.protocol_sha256 != _protocol_hash():
        raise RuntimeError("protocol SHA-256 does not match the frozen file")
    if _git_head() != args.protocol_commit:
        raise RuntimeError("--protocol-commit must equal the current HEAD")
    if not _git_clean():
        raise RuntimeError("run requires a clean worktree at the frozen commit")


def validate_envelope(args) -> None:
    if tuple(args.seeds) != CALIBRATION_SEEDS:
        raise ValueError(f"full run requires the frozen seeds {CALIBRATION_SEEDS}")
    if tuple(args.rhos) != DEFAULT_RHOS:
        raise ValueError(f"full run requires the frozen rhos {DEFAULT_RHOS}")
    if tuple(args.products) != DEFAULT_PRODUCTS:
        raise ValueError(f"full run requires the frozen products {DEFAULT_PRODUCTS}")
    if args.n != 256 or args.steps != 120 or args.n_train != 32 or args.n_test != 32:
        raise ValueError("full run envelope drifted from n=256, steps=120, train=test=32")
    if max(args.products) >= args.n:
        raise ValueError("product ladder must remain below algebraic saturation")
    if args.n < 8 * max(args.products):
        raise ValueError("frozen n/P margin failed")
    validate_freeze(args)


def cost_pilot(args) -> dict:
    """Time every budget-reaching arm without emitting scientific outcomes."""
    torch.set_num_threads(1)
    seed, rho = 4999, 0.99
    edge_index = chain_edges(args.n)
    Wd0, Wl0, Wu0 = random_coupling(args.n, edge_index, 1, seed=seed, rho=1.0)
    generator = torch.Generator().manual_seed(seed + 100)
    b_train = args.input_scale * torch.randn(
        args.n_train, args.n, 1, generator=generator, dtype=torch.float64
    )
    target = teacher_targets(rho, Wd0, edge_index, Wl0, Wu0, b_train)
    raw_init = raw_from_gain(0.2)
    exact_measured = train_gain(
        raw_init, Wd0, edge_index, Wl0, Wu0, b_train, target,
        mode="exact", products=1, steps=args.pilot_steps, lr=args.lr,
    )
    exact_seconds = exact_measured["seconds"] / args.pilot_steps * args.steps
    timings = []
    for products in args.products:
        row = {"products": products}
        for mode in ("neumann", "gmres"):
            measured = train_gain(
                raw_init,
                Wd0,
                edge_index,
                Wl0,
                Wu0,
                b_train,
                target,
                mode=mode,
                products=products,
                steps=args.pilot_steps,
                lr=args.lr,
            )
            row[f"projected_seconds_{mode}_arm"] = (
                measured["seconds"] / args.pilot_steps * args.steps
            )
        diagnostic_t0 = time.perf_counter()
        adjoint_diagnostic(
            raw_init, Wd0, edge_index, Wl0, Wu0, b_train, target, products,
            verify_dense=products == max(args.products),
        )
        row["seconds_diagnostic"] = time.perf_counter() - diagnostic_t0
        timings.append(row)
    arms_per_seed = 1 + 2 * len(args.products)
    projected = len(args.seeds) * len(args.rhos) * (
        exact_seconds
        + sum(row["projected_seconds_neumann_arm"] for row in timings)
        + sum(row["projected_seconds_gmres_arm"] for row in timings)
        + sum(row["seconds_diagnostic"] for row in timings)
    )
    return {
        "pilot_steps": args.pilot_steps,
        "projected_seconds_per_exact_arm": exact_seconds,
        "budget_timings": timings,
        "arms_per_seed_rho": arms_per_seed,
        "projected_full_grid_seconds": projected,
    }


def run(args) -> dict:
    validate_envelope(args)
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir)
    config = {
        "seeds": args.seeds,
        "rhos": args.rhos,
        "products": args.products,
        "n": args.n,
        "steps": args.steps,
        "lr": args.lr,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "input_scale": args.input_scale,
    }
    records = []
    t0 = time.perf_counter()
    for seed in args.seeds:
        path = out_dir / f"seed{seed}.json"
        if path.exists() and args.resume:
            record = json.loads(path.read_text())
            if record["source_commit"] != args.protocol_commit or record["config"] != config:
                raise RuntimeError(f"resume envelope mismatch in {path}")
            records.append(record)
            continue

        edge_index = chain_edges(args.n)
        Wd0, Wl0, Wu0 = random_coupling(
            args.n, edge_index, 1, seed=seed, rho=1.0
        )
        gen_train = torch.Generator().manual_seed(100 + seed)
        gen_test = torch.Generator().manual_seed(200 + seed)
        b_train = args.input_scale * torch.randn(
            args.n_train, args.n, 1, generator=gen_train, dtype=torch.float64
        )
        b_test = args.input_scale * torch.randn(
            args.n_test, args.n, 1, generator=gen_test, dtype=torch.float64
        )
        record = {
            "run_tag": "EXPLORATORY",
            "campaign_uuid": CAMPAIGN_UUID,
            "source_commit": args.protocol_commit,
            "protocol_sha256": _protocol_hash(),
            "config": config,
            "seed": seed,
            "cells": [],
        }
        for rho in args.rhos:
            record["cells"].append(
                _cell(
                    seed,
                    rho,
                    Wd0,
                    edge_index,
                    Wl0,
                    Wu0,
                    b_train,
                    b_test,
                    products_grid=tuple(args.products),
                    steps=args.steps,
                    lr=args.lr,
                )
            )
            _write_json(path, record)
        records.append(record)
        print(f"completed seed {seed}", flush=True)

    summary = {
        "run_tag": "EXPLORATORY",
        "status": "post-hoc adversarial study; not confirmatory",
        "campaign_uuid": CAMPAIGN_UUID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.protocol_commit,
        "protocol": str(PROTOCOL).replace("\\", "/"),
        "protocol_sha256": _protocol_hash(),
        "config": config,
        "predictions_frozen_before_run": PREDICTIONS,
        "seconds": time.perf_counter() - t0,
        **summarize(records, args.rhos, args.products),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--campaign-uuid", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--rhos", type=float, nargs="+", default=list(DEFAULT_RHOS))
    parser.add_argument("--products", type=int, nargs="+", default=list(DEFAULT_PRODUCTS))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--input-scale", type=float, default=0.2)
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cost-pilot", action="store_true")
    parser.add_argument("--pilot-steps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cost_pilot:
        validate_freeze(args)
        print(json.dumps(cost_pilot(args), indent=2))
        return 0
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
