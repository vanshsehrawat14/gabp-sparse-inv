"""Task A matched-fit adjudication (EXPLORATORY, DEV seeds only).

This is the matched-training-fit control for the H2/H3 inductive-bias-vs-optimization
question, governed by the matched-fit thresholds and the 2026-06-29 amendment in
``docs/PREREGISTRATION.md``. It does not modify the producers; it deterministically
replays their training with per-step trajectory logging.

Two arms, per the amendment:

* **Arm A (strict, approx-arm-only) -> verdict.** Matching is achieved on the approx
  arm only; the exact arm is held at full fit and never touched. A cell where the
  approx arm cannot be brought into the +/-5% train-loss band (its best fit stays
  worse than exact) is reported as **fit-unmatchable -> attainability/optimization**,
  with the train losses that prove it. This is a positive finding, not a defaulted
  retitle.
* **Arm B (symmetric-milestone) -> labeled diagnostic.** Both arms read at the step
  where they share equal train loss, which means reading the better-fitting (usually
  exact) arm earlier on its own unchanged trajectory. Compromised by the
  converged-vs-transient asymmetry, so it is diagnostic color only. Every B cell
  discloses the exact-arm read step and its converged/transient status.

Before any matched step is read, the replay asserts that the regenerated as-trained
aggregates reproduce the recorded EXPLORATORY numbers in ``docs/_phenomenon_menu.md``
to numerical tolerance; a divergence aborts at the top.

All matching uses the train-set loss under the **common exact operator** for both
arms (the DEQ producer already trains both arms under the exact forward; for the maze
the size-6 train-set loss is measured under the exact solve for both arms).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..demos.deq_fixedpoint import random_coupling
from ..demos.maze_grid import GaBPMazeGrid, gen_dataset
from ..generators import grid_edges
from .deq_breakeven import chain_edges, raw_from_gain
from .deq_cross_eval import fixed_point_with_backward_beta
from .deq_gradient_isolation import (
    _eval_loss as _deq_eval_loss,
    _function_distance as _deq_fdist,
    _teacher_targets as _deq_teacher,
)
from .matched_compute import n_match
from .maze_extrapolation import _eval_arm, _predict
from .phase1_analysis import paired_bootstrap_ci
from .seeds import (
    RUN_CONFIRMATORY,
    RUN_EXPLORATORY,
    assert_seed_for_run,
    dev_seeds,
    holdout_seeds,
    tagged_outdir,
)

RUN_TAG = RUN_EXPLORATORY
RESULTS = Path("results") / RUN_TAG / "matched_fit_taskA"

MATCH_TOL = 0.05          # registered MATCH TOLERANCE (relative train loss)
FLOOR_MULT = 2.0          # registered floor multiplier (2x the null cell)

# Confirmatory frozen values (docs/PREREGISTRATION_CONFIRMATORY.md, 2026-06-29).
M_MIN_CONF = 0.60         # FIX 1: minimum-matchability fraction to adjudicate a cell
F_ABS_CONF = 2e-6         # FIX 2: optimization-noise floor, F = max(2x M_null, F_abs)
C1_RELGAP_MIN = 0.5       # C1 underfit-direction magnitude gate
NC1_ORDERS = 3.0          # NC1 low-rho silence: >= 3 orders below high-rho
NC1_SILENT_RHO = 0.5      # the genuinely silent control rho

# Recorded EXPLORATORY anchors from docs/_phenomenon_menu.md. The replay must
# reproduce these (mean over DEV seeds 0-19) before any matched step is read.
DEQ_ANCHORS = {
    # (rho, K): expected mean delta_test_loss (neumann - exact)
    "delta_test_loss": {
        (0.7, 1): 1.37e-7, (0.7, 2): 5.50e-8,
        (0.99, 1): 6.04e-2, (0.99, 2): 1.12e-1, (0.99, 4): 7.70e-2,
        (0.99, 8): 1.09e-3, (0.99, 16): -2.14e-3, (0.99, 32): 3.58e-2,
    },
    # (rho, K): expected mean test_function_rel_mse
    "test_function_rel_mse": {
        (0.99, 1): 5.36e-3, (0.99, 2): 4.18e-3, (0.99, 4): 2.06e-3, (0.99, 32): 3.27e-3,
    },
}
# Maze positive H1: mean (exact_eval_exact - b2_matched_eval_b2) per size.
MAZE_ANCHORS = {"exact_minus_b2_matched": {6: -1.76e-1, 48: -8.21e-4}}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _agg(diffs: list[float]) -> dict:
    ci = paired_bootstrap_ci(diffs)
    return {"n": ci["n"], "mean": ci["mean"], "median": ci["median"],
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}


def _assert_anchor(report: list[str], name: str, got: float, expected: float,
                   *, rtol: float = 0.03, atol: float = 3e-5) -> bool:
    tol = atol + rtol * abs(expected)
    ok = abs(got - expected) <= tol
    report.append(
        f"  [{'OK ' if ok else 'FAIL'}] {name}: got {got:+.3e} vs recorded {expected:+.3e} "
        f"(|d|={abs(got - expected):.2e}, tol={tol:.2e})"
    )
    return ok


def _sci(x) -> str:
    return "n/a" if x is None else f"{x:+.3e}"


# --------------------------------------------------------------------------- #
# DEQ probe (primary).
# --------------------------------------------------------------------------- #
def _deq_train_traj(raw_init, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr, *, mode, K,
                    steps, lr):
    """Mirror deq_gradient_isolation._train_gain, recording the raw trajectory.

    Returns ``(raws, final_raw)`` where ``raws[t]`` is the scalar after ``t`` Adam
    steps (raws[0] = init). Logging does not touch the optimized tensor, so the
    final raw reproduces the producer bit-for-bit.
    """
    raw = torch.tensor(float(raw_init), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    raws = [float(raw.detach())]
    for _ in range(steps):
        opt.zero_grad()
        pred = fixed_point_with_backward_beta(
            raw, Wd0, edge_index, Wl0, Wu0, b_tr, kind="linear", beta=1.0, mode=mode, K=K
        )
        loss = torch.mean((pred - y_tr) ** 2)
        loss.backward()
        opt.step()
        raws.append(float(raw.detach()))
    return raws, raws[-1]


def _deq_cell(seed, rho, *, k_grid, n, steps, lr, n_train, n_test, input_scale):
    """Train exact + Neumann-K arms for one (seed, rho); return trajectories + metrics."""
    edge_index = chain_edges(n)
    Wd0, Wl0, Wu0 = random_coupling(n, edge_index, 1, seed=seed, rho=1.0)
    gen_tr = torch.Generator().manual_seed(100 + seed)
    gen_te = torch.Generator().manual_seed(200 + seed)
    b_tr = input_scale * torch.randn(n_train, n, 1, generator=gen_tr, dtype=torch.float64)
    b_te = input_scale * torch.randn(n_test, n, 1, generator=gen_te, dtype=torch.float64)
    y_tr = _deq_teacher(rho, Wd0, edge_index, Wl0, Wu0, b_tr)
    y_te = _deq_teacher(rho, Wd0, edge_index, Wl0, Wu0, b_te)
    raw_init = raw_from_gain(min(0.2, max(0.05, 0.5 * rho)))

    ex_raws, ex_final = _deq_train_traj(
        raw_init, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr, mode="exact", K=1, steps=steps, lr=lr
    )
    ex_tr = _deq_eval_loss(ex_final, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr)
    ex_te = _deq_eval_loss(ex_final, Wd0, edge_index, Wl0, Wu0, b_te, y_te)

    def train_loss_at(raw):
        return _deq_eval_loss(raw, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr)["loss"]

    L_exact = ex_tr["loss"]
    ex_traj_loss = [train_loss_at(r) for r in ex_raws]

    arms = {}
    for K in k_grid:
        nu_raws, nu_final = _deq_train_traj(
            raw_init, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr,
            mode="b2_neumann", K=int(K), steps=steps, lr=lr,
        )
        nu_tr = _deq_eval_loss(nu_final, Wd0, edge_index, Wl0, Wu0, b_tr, y_tr)
        nu_te = _deq_eval_loss(nu_final, Wd0, edge_index, Wl0, Wu0, b_te, y_te)
        nu_traj_loss = [train_loss_at(r) for r in nu_raws]

        # As-trained metrics (reproduce the recorded menu numbers).
        delta_test = nu_te["loss"] - ex_te["loss"]
        fdist_test = _deq_fdist(ex_te["pred"], nu_te["pred"], y_te)["rel_mse_to_target_variance"]

        # Arm A: match the Neumann arm into L_exact +/-5% (approx-arm-only).
        lo, hi = (1 - MATCH_TOL) * L_exact, (1 + MATCH_TOL) * L_exact
        in_band = [(abs(l - L_exact), i) for i, l in enumerate(nu_traj_loss) if lo <= l <= hi]
        if in_band:
            _, mstep = min(in_band)
            matched_raw = nu_raws[mstep]
            matched_te = _deq_eval_loss(matched_raw, Wd0, edge_index, Wl0, Wu0, b_te, y_te)
            a_fdist = _deq_fdist(ex_te["pred"], matched_te["pred"], y_te)["rel_mse_to_target_variance"]
            a_delta = matched_te["loss"] - ex_te["loss"]
            a_match = {"matched": True, "step": mstep, "fdist_test": a_fdist, "delta_test": a_delta}
        else:
            a_match = {"matched": False, "step": None, "fdist_test": None, "delta_test": None}

        # Arm B (diagnostic): symmetric milestone at the Neumann plateau; read EXACT
        # earlier on its own trajectory. Nearest-neighbour step, with a within-tol
        # flag (fast descent can step over the +/-5% band).
        milestone = nu_tr["loss"]
        bstep = min(range(len(ex_traj_loss)), key=lambda i: abs(ex_traj_loss[i] - milestone))
        within_tol = bool(abs(ex_traj_loss[bstep] - milestone) <= MATCH_TOL * milestone)
        ex_read = _deq_eval_loss(ex_raws[bstep], Wd0, edge_index, Wl0, Wu0, b_te, y_te)
        b_fdist = _deq_fdist(ex_read["pred"], nu_te["pred"], y_te)["rel_mse_to_target_variance"]
        # Transient if exact keeps descending well past this read point.
        transient = bool(L_exact < 0.9 * ex_traj_loss[bstep])
        b_match = {"matched": True, "within_tol": within_tol, "exact_read_step": bstep,
                   "n_steps": steps, "exact_loss_at_read": ex_traj_loss[bstep],
                   "exact_final_loss": L_exact, "transient": transient, "fdist_test": b_fdist}

        arms[int(K)] = {
            "as_trained": {"delta_test_loss": delta_test, "test_function_rel_mse": fdist_test,
                           "neumann_train_loss": nu_tr["loss"]},
            "arm_a": a_match,
            "arm_b": b_match,
        }

    return {"rho": rho, "L_exact_train": L_exact, "exact_test_loss": ex_te["loss"], "arms": arms}


def run_deq(seeds, *, rhos, k_grid, n=16, steps=120, lr=0.05, n_train=32, n_test=32,
            input_scale=0.2, skip_assert=False, run_tag=RUN_TAG, check_anchors=True,
            m_min=0.0, floor_abs=0.0) -> dict:
    cells: dict[tuple[int, float], dict] = {}
    for seed in seeds:
        assert_seed_for_run(seed, run_tag)
        for rho in rhos:
            cells[(seed, rho)] = _deq_cell(
                seed, rho, k_grid=k_grid, n=n, steps=steps, lr=lr,
                n_train=n_train, n_test=n_test, input_scale=input_scale,
            )

    # --- determinism assertion (DEV menu reproduction; skipped for HOLDOUT) ---
    if check_anchors:
        report = ["DEQ determinism check vs docs/_phenomenon_menu.md:"]
        ok = True
        for (rho, K), exp in DEQ_ANCHORS["delta_test_loss"].items():
            if rho not in rhos or K not in k_grid:
                continue
            vals = [cells[(s, rho)]["arms"][K]["as_trained"]["delta_test_loss"] for s in seeds]
            ok &= _assert_anchor(report, f"delta_test_loss(rho={rho},K={K})", _agg(vals)["mean"], exp)
        for (rho, K), exp in DEQ_ANCHORS["test_function_rel_mse"].items():
            if rho not in rhos or K not in k_grid:
                continue
            vals = [cells[(s, rho)]["arms"][K]["as_trained"]["test_function_rel_mse"] for s in seeds]
            ok &= _assert_anchor(report, f"test_function_rel_mse(rho={rho},K={K})", _agg(vals)["mean"], exp)
        print("\n".join(report))
        if not ok and not skip_assert:
            raise RuntimeError("DEQ replay diverges from recorded menu numbers; determinism broken. Aborting.")
    else:
        ok = None
        print("DEQ determinism: menu-anchor check skipped (no recorded reference for this seed pool).")

    return _deq_summary(cells, seeds, rhos, k_grid, steps=steps, determinism_ok=ok,
                        m_min=m_min, floor_abs=floor_abs)


def _deq_summary(cells, seeds, rhos, k_grid, *, steps: int, determinism_ok, m_min=0.0,
                 floor_abs=0.0) -> dict:
    # Floor: 2x the K=32 function distance (matched if available, else as-trained, flagged).
    floors = {}
    for rho in rhos:
        matched32 = [cells[(s, rho)]["arms"][32]["arm_a"]["fdist_test"]
                     for s in seeds if 32 in k_grid and cells[(s, rho)]["arms"][32]["arm_a"]["matched"]]
        if matched32:
            floors[rho] = {"value": FLOOR_MULT * _agg(matched32)["mean"], "basis": "matched_K32", "n": len(matched32)}
        elif 32 in k_grid:
            astr = [cells[(s, rho)]["arms"][32]["as_trained"]["test_function_rel_mse"] for s in seeds]
            floors[rho] = {"value": FLOOR_MULT * _agg(astr)["mean"], "basis": "as_trained_K32_FLAGGED", "n": len(astr)}
        else:
            floors[rho] = {"value": None, "basis": "no_K32", "n": 0}

    rows = []
    nseed = len(seeds)
    for rho in rhos:
        F0 = floors[rho]["value"]
        F = max(F0 if F0 is not None else 0.0, floor_abs)   # FIX 2: optimization-noise floor
        for K in k_grid:
            a_matched = [cells[(s, rho)]["arms"][K]["arm_a"] for s in seeds]
            n_match = sum(1 for a in a_matched if a["matched"])
            frac = n_match / nseed
            L_ex = _agg([cells[(s, rho)]["L_exact_train"] for s in seeds])["median"]
            L_nu = _agg([cells[(s, rho)]["arms"][K]["as_trained"]["neumann_train_loss"] for s in seeds])["median"]
            astr_fdist = _agg([cells[(s, rho)]["arms"][K]["as_trained"]["test_function_rel_mse"] for s in seeds])
            astr_delta = _agg([cells[(s, rho)]["arms"][K]["as_trained"]["delta_test_loss"] for s in seeds])

            matched_stat = _agg([a["fdist_test"] for a in a_matched if a["matched"]]) if n_match else None
            if frac < m_min or n_match == 0:   # FIX 1: minimum-matchability gate
                verdict = "UNMATCHABLE-dominant->attainability" if m_min > 0 else "UNMATCHABLE->optimization/attainability"
                if 0 < n_match < nseed and m_min > 0:
                    verdict += f"(matchable {n_match}/{nseed}<{m_min:g})"
            else:
                if matched_stat["ci_low"] > F:
                    verdict = "SURVIVE"
                elif matched_stat["mean"] <= F:
                    verdict = "COLLAPSE"
                else:
                    verdict = "AMBIGUOUS->retitle"
                if n_match < nseed:
                    verdict += f"(partial {n_match}/{nseed})"

            # Arm B diagnostic.
            b = [cells[(s, rho)]["arms"][K]["arm_b"] for s in seeds]
            b_matched = [x for x in b if x["matched"]]
            b_within = sum(1 for x in b_matched if x.get("within_tol"))
            b_stat = _agg([x["fdist_test"] for x in b_matched]) if b_matched else None
            b_steps = sorted(x["exact_read_step"] for x in b_matched) if b_matched else []
            b_transient = sum(1 for x in b_matched if x["transient"])

            rows.append({
                "rho": rho, "K": K, "n_matchable_A": n_match, "n_seeds": len(seeds),
                "matchable_frac": frac,
                "median_L_exact_train": L_ex, "median_L_neumann_train": L_nu,
                "median_rel_train_gap": (L_nu - L_ex) / L_ex if L_ex else None,
                "as_trained_delta_test_mean": astr_delta["mean"],
                "as_trained_fdist_mean": astr_fdist["mean"],
                "floor_F": F,
                "armA_matched_fdist": matched_stat,
                "armA_verdict": verdict,
                "armB_n_matched": len(b_matched), "armB_n_within_tol": b_within, "armB_fdist": b_stat,
                "armB_exact_read_step_median": (b_steps[len(b_steps) // 2] if b_steps else None),
                "armB_n_transient": b_transient,
            })
    return {"probe": "deq", "run_tag": RUN_TAG, "determinism_ok": determinism_ok,
            "deq_steps": steps, "m_min": m_min, "floor_abs": floor_abs,
            "floors": floors, "rows": rows, "rhos": list(rhos),
            "k_grid": list(k_grid), "seeds": list(seeds)}


# --------------------------------------------------------------------------- #
# Maze probe (secondary, confound-free cell).
# --------------------------------------------------------------------------- #
def _maze_train_traj(S, operator, K, seed, eps, hidden, steps, lr, feats_tr, y_tr):
    """Mirror maze_extrapolation._train_arm; log per-step size-6 train loss under EXACT."""
    torch.manual_seed(seed)
    model = GaBPMazeGrid(S, S, eps, hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    exact_train_losses = []
    for _ in range(steps):
        opt.zero_grad()
        torch.mean((_predict(model, feats_tr, operator, K) - y_tr) ** 2).backward()
        opt.step()
        with torch.no_grad():
            exact_train_losses.append(float(torch.mean((_predict(model, feats_tr, "exact", None) - y_tr) ** 2)))
    return model, exact_train_losses


def _maze_train_to(S, operator, K, seed, eps, hidden, n_steps, lr, feats_tr, y_tr):
    """Deterministic re-train to ``n_steps`` (reproduces the trajectory state)."""
    torch.manual_seed(seed)
    model = GaBPMazeGrid(S, S, eps, hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        torch.mean((_predict(model, feats_tr, operator, K) - y_tr) ** 2).backward()
        opt.step()
    return model


def _maze_evals(model, sizes, operator, K, eps, hidden, testsets):
    """Eval one trained model under ``operator`` at every size (cached test sets)."""
    out = {}
    for L in sizes:
        feats_te, y_te = testsets[L]
        out[L] = _eval_arm(model, L, operator, K, eps, hidden, feats_te, y_te)
    return out


def _maze_seed(seed, *, S, sizes, eps, hidden, steps, lr, n_train, K_match, K_floor):
    b2_op = "b2_jacobi"
    _ei, feats_tr, y_tr = gen_dataset(S, S, n_train, seed=seed, eps=eps)
    # Deterministic test sets, built once per (seed, size) and reused across all evals.
    testsets = {}
    for L in sizes:
        _eil, feats_te, y_te = gen_dataset(L, L, 48, seed=200 + seed, eps=eps)
        testsets[L] = (feats_te, y_te)

    m_exact, ex_traj = _maze_train_traj(S, "exact", None, seed, eps, hidden, steps, lr, feats_tr, y_tr)
    m_b2, b2_traj = _maze_train_traj(S, b2_op, K_match, seed, eps, hidden, steps, lr, feats_tr, y_tr)
    m_floor = _maze_train_to(S, b2_op, K_floor, seed, eps, hidden, steps, lr, feats_tr, y_tr)  # trajectory unused

    ex_eval = _maze_evals(m_exact, sizes, "exact", None, eps, hidden, testsets)         # train_exact__eval_exact
    b2_evalexact = _maze_evals(m_b2, sizes, "exact", None, eps, hidden, testsets)        # train_b2__eval_exact
    b2_matched = _maze_evals(m_b2, sizes, b2_op, K_match, eps, hidden, testsets)         # train_b2__eval_b2 (H1 anchor)
    floor_evalexact = _maze_evals(m_floor, sizes, "exact", None, eps, hidden, testsets)  # K=256 eval exact

    L_exact = ex_traj[-1]                       # size-6 exact-eval train loss, exact arm (= its training loss)
    lo, hi = (1 - MATCH_TOL) * L_exact, (1 + MATCH_TOL) * L_exact
    in_band = [(abs(l - L_exact), i) for i, l in enumerate(b2_traj) if lo <= l <= hi]
    if in_band:
        _, mstep = min(in_band)
        m_b2_matched_fit = _maze_train_to(S, b2_op, K_match, seed, eps, hidden, mstep + 1, lr, feats_tr, y_tr)
        b2_matchedfit_evalexact = _maze_evals(m_b2_matched_fit, sizes, "exact", None, eps, hidden, testsets)
        a_match = {"matched": True, "step": mstep, "eval_exact": b2_matchedfit_evalexact}
    else:
        a_match = {"matched": False, "step": None, "b2_best_exact_train": min(b2_traj)}

    # Arm B (diagnostic): read exact earlier at the B2 plateau (symmetric milestone),
    # nearest-neighbour step with a within-tol flag.
    milestone = b2_traj[-1]
    bstep = min(range(len(ex_traj)), key=lambda i: abs(ex_traj[i] - milestone))
    within_tol = bool(abs(ex_traj[bstep] - milestone) <= MATCH_TOL * milestone)
    m_ex_read = _maze_train_to(S, "exact", None, seed, eps, hidden, bstep + 1, lr, feats_tr, y_tr)
    ex_read_evalexact = _maze_evals(m_ex_read, sizes, "exact", None, eps, hidden, testsets)
    transient = bool(L_exact < 0.9 * ex_traj[bstep])
    b_match = {"matched": True, "within_tol": within_tol, "exact_read_step": bstep, "n_steps": steps,
               "transient": transient, "exact_read_eval_exact": ex_read_evalexact, "b2_eval_exact": b2_evalexact}

    return {"seed": seed, "L_exact_train": L_exact, "b2_final_exact_train": b2_traj[-1],
            "b2_best_exact_train": min(b2_traj),
            "ex_eval": ex_eval, "b2_evalexact": b2_evalexact, "b2_matched": b2_matched,
            "floor_evalexact": floor_evalexact, "arm_a": a_match, "arm_b": b_match}


_MAZE_SIZE_KEYED = ("ex_eval", "b2_evalexact", "b2_matched", "floor_evalexact")


def _coerce_loaded_seed(d: dict) -> dict:
    """JSON stringifies int size keys; convert them back after a cache reload."""
    for k in _MAZE_SIZE_KEYED:
        d[k] = {int(kk): vv for kk, vv in d[k].items()}
    if d["arm_a"].get("matched"):
        d["arm_a"]["eval_exact"] = {int(kk): vv for kk, vv in d["arm_a"]["eval_exact"].items()}
    for k in ("exact_read_eval_exact", "b2_eval_exact"):
        d["arm_b"][k] = {int(kk): vv for kk, vv in d["arm_b"][k].items()}
    return d


def run_maze(seeds, *, S=6, sizes=(6, 12, 24, 48), eps=0.05, hidden=16, steps=250, lr=0.02,
             n_train=48, k_floor=256, skip_assert=False, run_tag=RUN_TAG, check_anchors=True,
             m_min=0.0, floor_abs=0.0) -> dict:
    K_match = n_match(S * S, grid_edges(S, S), b=1, k=1, method="jacobi", target="solve_adjoint")
    cache_dir = Path("results") / run_tag / "matched_fit_taskA" / (
        "maze_seeds" if eps == 0.05 else f"maze_seeds_eps{eps:g}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for i, seed in enumerate(seeds):
        assert_seed_for_run(seed, run_tag)
        cpath = cache_dir / f"seed{seed}.json"
        if cpath.exists():
            per_seed[seed] = _coerce_loaded_seed(json.loads(cpath.read_text()))
            print(f"[maze] seed {seed} loaded from cache ({i + 1}/{len(seeds)})", flush=True)
            continue
        t0 = time.perf_counter()
        res = _maze_seed(seed, S=S, sizes=sizes, eps=eps, hidden=hidden, steps=steps,
                         lr=lr, n_train=n_train, K_match=K_match, K_floor=k_floor)
        cpath.write_text(json.dumps(res))
        per_seed[seed] = res
        print(f"[maze] seed {seed} done ({i + 1}/{len(seeds)}, {time.perf_counter() - t0:.0f}s)", flush=True)

    # --- determinism assertion (DEV menu reproduction; skipped for HOLDOUT) ---
    if check_anchors:
        report = [f"Maze determinism check vs docs/_phenomenon_menu.md (K_match={K_match}):"]
        ok = True
        for L, exp in MAZE_ANCHORS["exact_minus_b2_matched"].items():
            if L not in sizes:
                continue
            vals = [per_seed[s]["ex_eval"][L] - per_seed[s]["b2_matched"][L] for s in seeds]
            ok &= _assert_anchor(report, f"exact_minus_b2_matched(L={L})", _agg(vals)["mean"], exp)
        print("\n".join(report))
        if not ok and not skip_assert:
            raise RuntimeError("Maze replay diverges from recorded menu numbers; determinism broken. Aborting.")
    else:
        ok = None
        print("Maze determinism: menu-anchor check skipped (no recorded reference for this seed pool).")

    return _maze_summary(per_seed, seeds, sizes, K_match, k_floor, determinism_ok=ok,
                         m_min=m_min, floor_abs=floor_abs)


def _maze_summary(per_seed, seeds, sizes, K_match, k_floor, *, determinism_ok, m_min=0.0,
                  floor_abs=0.0) -> dict:
    # Floor soundness: K=floor (~exact) in-distribution distance to exact, eval-exact.
    floor_indist = _agg([per_seed[s]["floor_evalexact"][sizes[0]] - per_seed[s]["ex_eval"][sizes[0]] for s in seeds])
    extrap = [L for L in sizes if L != sizes[0]]
    floor_gap = {L: _agg([per_seed[s]["floor_evalexact"][L] - per_seed[s]["ex_eval"][L] for s in seeds]) for L in extrap}
    # F per size = max(2x |null extrapolation gap| of the K=floor arm vs exact, F_abs).
    F = {L: max(FLOOR_MULT * abs(floor_gap[L]["mean"]), floor_abs) for L in extrap}  # FIX 2
    floor_sound = abs(floor_indist["mean"]) <= MATCH_TOL * abs(
        _agg([per_seed[s]["ex_eval"][sizes[0]] for s in seeds])["mean"]) + 1e-12

    # Train losses that prove (un)matchability of the B2 arm under common exact eval.
    med_L_exact = _agg([per_seed[s]["L_exact_train"] for s in seeds])["median"]
    med_b2_best = _agg([per_seed[s]["b2_best_exact_train"] for s in seeds])["median"]
    proving = {"median_L_exact_train": med_L_exact, "median_b2_best_exact_train": med_b2_best,
               "median_rel_gap": (med_b2_best - med_L_exact) / med_L_exact if med_L_exact else None}
    # As-trained confound-free gap (full-fit B2 vs exact, both eval exact): the raw effect.
    astrained = {L: _agg([per_seed[s]["b2_evalexact"][L] - per_seed[s]["ex_eval"][L] for s in seeds]) for L in sizes}

    n_match = sum(1 for s in seeds if per_seed[s]["arm_a"]["matched"])
    frac = n_match / len(seeds)
    gap_shrinks = astrained[sizes[-1]]["mean"] < astrained[sizes[0]]["mean"]
    if frac < m_min or n_match == 0:   # FIX 1: minimum-matchability gate
        maze_verdict = ("UNMATCHABLE-dominant->attainability" if m_min > 0
                        else "UNMATCHABLE->attainability") + f" (gap_shrinks_with_size={gap_shrinks})"
    else:
        maze_verdict = "matchable->see armA_rows"
    rows = []
    if n_match > 0:
        ms = [s for s in seeds if per_seed[s]["arm_a"]["matched"]]
        for L in sizes:
            gap = _agg([per_seed[s]["arm_a"]["eval_exact"][L] - per_seed[s]["ex_eval"][L] for s in ms])
            tag = "in_distribution" if L == sizes[0] else "extrapolation"
            verdict = "-"
            if tag == "extrapolation":
                FL = F[L]
                verdict = ("SURVIVE" if gap["ci_low"] > FL else
                           "COLLAPSE" if gap["mean"] <= FL else "AMBIGUOUS->retitle")
            rows.append({"size": L, "kind": tag, "matched_gap_b2_minus_exact": gap,
                         "floor_F": (F[L] if tag == "extrapolation" else None), "verdict": verdict})

    # Arm B diagnostic.
    b_ms = [s for s in seeds if per_seed[s]["arm_b"]["matched"]]
    b_rows = []
    for L in sizes:
        if not b_ms:
            break
        gap = _agg([per_seed[s]["arm_b"]["exact_read_eval_exact"][L] - per_seed[s]["arm_b"]["b2_eval_exact"][L]
                    for s in b_ms])
        b_rows.append({"size": L, "exact_read_minus_b2_eval_exact": gap})
    b_steps = sorted(per_seed[s]["arm_b"]["exact_read_step"] for s in b_ms) if b_ms else []
    b_transient = sum(1 for s in b_ms if per_seed[s]["arm_b"]["transient"])
    b_within = sum(1 for s in b_ms if per_seed[s]["arm_b"].get("within_tol"))

    return {
        "probe": "maze", "run_tag": RUN_TAG, "determinism_ok": determinism_ok,
        "K_match": K_match, "k_floor": k_floor, "sizes": list(sizes), "seeds": list(seeds),
        "n_matchable_A": n_match, "n_seeds": len(seeds), "matchable_frac": frac,
        "armA_verdict": maze_verdict, "gap_shrinks_with_size": gap_shrinks,
        "m_min": m_min, "floor_abs": floor_abs, "proving": proving,
        "as_trained_gap_by_size": astrained,
        "floor_F_by_size": F, "floor_gap_by_size": floor_gap, "floor_indistribution_gap": floor_indist,
        "floor_sound": floor_sound, "armA_rows": rows,
        "armB": {"n_matched": len(b_ms), "n_within_tol": b_within,
                 "exact_read_step_median": (b_steps[len(b_steps) // 2] if b_steps else None),
                 "n_transient": b_transient, "n_steps": 250, "rows": b_rows},
    }


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def print_deq_table(summary: dict) -> None:
    nseed = len(summary["seeds"])
    print("\n=== DEQ Arm A (VERDICT) -- matched-fit, approx-arm-only ===")
    print("PRIMARY metric: held-out function rel-MSE under common exact forward. "
          "Lower train loss = better fit.")
    hdr = (f"{'rho':>5} {'K':>4} {'matchA':>7} {'L_exact':>11} {'L_neu':>11} "
           f"{'relgap':>9} {'floorF':>10} {'A_fdist(mean[CI])':>26} {'verdict':>22}")
    print(hdr)
    for r in summary["rows"]:
        a = r["armA_matched_fdist"]
        af = "n/a" if a is None else f"{a['mean']:.2e}[{a['ci_low']:.1e},{a['ci_high']:.1e}]"
        rg = "n/a" if r["median_rel_train_gap"] is None else f"{r['median_rel_train_gap']:+.1e}"
        print(f"{r['rho']:>5} {r['K']:>4} {r['n_matchable_A']:>3}/{r['n_seeds']:<3} "
              f"{r['median_L_exact_train']:>11.3e} {r['median_L_neumann_train']:>11.3e} {rg:>9} "
              f"{_sci(r['floor_F']):>10} {af:>26} {r['armA_verdict']:>22}")
    print("\n=== DEQ Arm B (DIAGNOSTIC ONLY -- compromised by converged-vs-transient asymmetry) ===")
    print("equal-loss read of the EXACT arm earlier on its own trajectory; B_inTol = "
          "seeds whose read step is within +/-5% of the Neumann plateau.")
    print(f"{'rho':>5} {'K':>4} {'B_inTol':>8} {'read_step(med)/N':>18} "
          f"{'#transient':>11} {'B_fdist(mean[CI])':>26}")
    for r in summary["rows"]:
        b = r["armB_fdist"]
        bf = "n/a" if b is None else f"{b['mean']:.2e}[{b['ci_low']:.1e},{b['ci_high']:.1e}]"
        step = "n/a" if r["armB_exact_read_step_median"] is None else f"{r['armB_exact_read_step_median']}/{summary['deq_steps']}"
        print(f"{r['rho']:>5} {r['K']:>4} {r['armB_n_within_tol']:>5}/{nseed:<2} "
              f"{step:>18} {r['armB_n_transient']:>11} {bf:>26}")


def print_maze_table(summary: dict) -> None:
    p = summary["proving"]
    print(f"\n=== MAZE Arm A (VERDICT) -- confound-free cell, both eval under exact (K_match={summary['K_match']}) ===")
    print(f"matchable A: {summary['n_matchable_A']}/{summary['n_seeds']} seeds. "
          f"Match quantity = size-{summary['sizes'][0]} train loss under common exact eval.")
    print(f"  proving fit: median L_exact={p['median_L_exact_train']:.3e}  "
          f"median B2-best(exact-eval)={p['median_b2_best_exact_train']:.3e}  "
          f"rel gap={_sci(p['median_rel_gap'])}")
    if summary["n_matchable_A"] == 0:
        print("  UNMATCHABLE -> the operator-training bias is an attainability/optimization effect "
              "(B2's best size-6 exact-eval fit stays >5% worse than exact; cannot match by handicapping B2).")
    print("  as-trained confound-free gap (full-fit B2 - exact, both eval exact):")
    for L in summary["sizes"]:
        g = summary["as_trained_gap_by_size"][L]
        tag = "in-dist" if L == summary["sizes"][0] else "extrap "
        print(f"    size {L:>3} [{tag}] b2-exact = {g['mean']:+.3e} [{g['ci_low']:+.2e},{g['ci_high']:+.2e}]")
    print(f"  floor soundness: K={summary['k_floor']} (~exact) gaps vs exact, eval exact "
          f"(sound={summary['floor_sound']}; in-dist {_sci(summary['floor_indistribution_gap']['mean'])}):")
    for L, g in summary["floor_gap_by_size"].items():
        print(f"    size {L:>3} K{summary['k_floor']}-exact = {g['mean']:+.3e} [{g['ci_low']:+.2e},{g['ci_high']:+.2e}]"
              f"  -> floorF(2x|gap|)={_sci(summary['floor_F_by_size'][L])}")
    if summary["armA_rows"]:
        print("  matched-fit gap (only over matchable seeds):")
        for r in summary["armA_rows"]:
            g = r["matched_gap_b2_minus_exact"]
            print(f"    size {r['size']:>3} [{r['kind']:>15}] gap(b2-exact)={g['mean']:+.3e} "
                  f"[{g['ci_low']:+.2e},{g['ci_high']:+.2e}]  floorF={_sci(r['floor_F'])}  -> {r['verdict']}")
    b = summary["armB"]
    print(f"\n=== MAZE Arm B (DIAGNOSTIC) -- read EXACT earlier at B2 plateau; within-tol {b['n_within_tol']}/{b['n_matched']}, "
          f"read step median {b['exact_read_step_median']}/{b['n_steps']}, transient {b['n_transient']}/{b['n_matched']} ===")
    for r in b["rows"]:
        g = r["exact_read_minus_b2_eval_exact"]
        print(f"  size {r['size']:>3} exact_read - b2 (eval exact) = {g['mean']:+.3e} [{g['ci_low']:+.2e},{g['ci_high']:+.2e}]")


def confirmatory_decision(deq: dict, maze: dict | None = None) -> dict:
    """Evaluate the frozen C1/C2/C3 + NC1/NC2 rules (PREREGISTRATION_CONFIRMATORY.md)."""
    rows = {(r["rho"], r["K"]): r for r in deq["rows"]}
    high_rho = [x for x in (0.98, 0.99) if x in deq["rhos"]]
    m_min = deq["m_min"]
    checks = []

    # C1: high-rho strong cells (K in 1,2,4) unmatchable-dominant in the underfit direction.
    c1 = []
    for rho in high_rho:
        for K in (1, 2, 4):
            r = rows.get((rho, K))
            if not r:
                continue
            g = r["median_rel_train_gap"]
            ok = (r["matchable_frac"] < m_min) and (g is not None and g >= C1_RELGAP_MIN)
            c1.append((rho, K, round(r["matchable_frac"], 2), None if g is None else round(g, 2), ok))
    c1_ok = bool(c1) and all(x[-1] for x in c1)
    checks.append(("C1 DEQ attainability (strong cells unmatchable + underfit relgap>=0.5)", c1_ok, c1))

    # C2: near-exact K=32 at high rho, where matchable, collapses to <= F.
    c2 = []
    for rho in high_rho:
        r = rows.get((rho, 32))
        if r and r["matchable_frac"] >= m_min and r["armA_matched_fdist"]:
            ok = r["armA_matched_fdist"]["mean"] <= r["floor_F"]
            c2.append((rho, 32, r["armA_matched_fdist"]["mean"], r["floor_F"], ok))
    c2_ok = bool(c2) and all(x[-1] for x in c2)
    checks.append(("C2 DEQ collapse (near-exact K=32, matchable, fdist<=F)", c2_ok, c2))

    # NC1: silent control rho=0.5 -> no underfit (relgap<=0.1) and >=3 orders below high-rho.
    sr = NC1_SILENT_RHO
    nc1_gap = [(sr, K, round(rows[(sr, K)]["median_rel_train_gap"], 3),
                rows[(sr, K)]["median_rel_train_gap"] <= 0.1)
               for K in (1, 2, 4) if (sr, K) in rows and rows[(sr, K)]["median_rel_train_gap"] is not None]
    nc1_underfit_ok = bool(nc1_gap) and all(x[-1] for x in nc1_gap)
    hi_vals = [rows[(rho, K)]["as_trained_fdist_mean"] for rho in high_rho for K in (1, 2) if (rho, K) in rows]
    lo_vals = [rows[(sr, K)]["as_trained_fdist_mean"] for K in (1, 2, 4) if (sr, K) in rows]
    orders = math.log10(max(hi_vals) / max(lo_vals)) if hi_vals and lo_vals and max(lo_vals) > 0 else None
    nc1_ok = nc1_underfit_ok and orders is not None and orders >= NC1_ORDERS
    checks.append((f"NC1 silence rho={sr} (no underfit & >=3 orders below high-rho; orders="
                   f"{None if orders is None else round(orders, 2)})", nc1_ok, nc1_gap))

    # C3 + maze floor (if maze provided).
    c3_ok = None
    if maze is not None:
        c3_ok = (maze["matchable_frac"] < maze["m_min"]) and maze["gap_shrinks_with_size"]
        checks.append((f"C3 maze attainability (unmatchable-dominant frac={round(maze['matchable_frac'], 2)} "
                       f"& gap shrinks={maze['gap_shrinks_with_size']})", c3_ok, None))
        fl = [(L, fg["mean"], maze["as_trained_gap_by_size"][L]["mean"],
               abs(fg["mean"]) < 0.1 * abs(maze["as_trained_gap_by_size"][L]["mean"]))
              for L, fg in maze["floor_gap_by_size"].items()]
        checks.append(("NC2 maze floor sound (K=256 extrap gap << B2 gap)", all(x[-1] for x in fl), fl))

    bias = any(r["armA_verdict"].startswith("SURVIVE")
               for r in deq["rows"] if r["matchable_frac"] >= m_min and r["rho"] in high_rho)
    core_attain = c1_ok and nc1_ok and (c3_ok is not False)
    if bias and not core_attain:
        branch = "INDUCTIVE_BIAS (unexpected -> confirm bias)"
    elif core_attain and not bias:
        branch = "ATTAINABILITY/OPTIMIZATION (predicted)"
    else:
        branch = "AMBIGUOUS -> conservative attainability/optimization"
    return {"branch": branch, "checks": checks, "bias_branch_fired": bias}


def print_confirmatory_decision(dec: dict) -> None:
    print("\n=== CONFIRMATORY DECISION (frozen rules) ===")
    for name, ok, detail in dec["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            for d in detail:
                print(f"        {d}")
    print(f"  -> BRANCH: {dec['branch']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Task A matched-fit adjudication (EXPLORATORY).")
    ap.add_argument("--probe", choices=["deq", "maze", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=None, help="seed count (default 20 DEV / 30 HOLDOUT)")
    ap.add_argument("--deq-rhos", type=float, nargs="+", default=[0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99])
    ap.add_argument("--deq-k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--maze-eps", type=float, default=0.05, help="maze eps (0.05 positive / 2.0 large-eps control)")
    ap.add_argument("--confirmatory", action="store_true", help="HOLDOUT seeds + CONFIRMATORY rules (one-shot)")
    ap.add_argument("--dev-confirm", action="store_true", help="DEV seeds with confirmatory rules (dry-run)")
    ap.add_argument("--skip-assert", action="store_true", help="smoke only: do not abort on anchor mismatch")
    args = ap.parse_args(argv)

    if args.confirmatory:
        run_tag, check_anchors, m_min, floor_abs = RUN_CONFIRMATORY, False, M_MIN_CONF, F_ABS_CONF
        seeds = holdout_seeds(args.seeds if args.seeds is not None else 30)
    elif args.dev_confirm:
        run_tag, check_anchors, m_min, floor_abs = RUN_EXPLORATORY, True, M_MIN_CONF, F_ABS_CONF
        seeds = dev_seeds(args.seeds if args.seeds is not None else 20)
    else:
        run_tag, check_anchors, m_min, floor_abs = RUN_EXPLORATORY, True, 0.0, 0.0
        seeds = dev_seeds(args.seeds if args.seeds is not None else 20)

    outdir = tagged_outdir(Path("results") / run_tag / "matched_fit_taskA", run_tag)
    suffix = "" if args.maze_eps == 0.05 else f"_eps{args.maze_eps:g}"
    out = {"run_tag": run_tag, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "seeds": seeds, "match_tolerance": MATCH_TOL, "floor_multiplier": FLOOR_MULT,
           "m_min": m_min, "floor_abs": floor_abs, "maze_eps": args.maze_eps}

    deq = maze = None
    if args.probe in ("deq", "both"):
        t0 = time.perf_counter()
        deq = run_deq(seeds, rhos=tuple(args.deq_rhos), k_grid=tuple(args.deq_k),
                      skip_assert=args.skip_assert, run_tag=run_tag, check_anchors=check_anchors,
                      m_min=m_min, floor_abs=floor_abs)
        deq["seconds"] = time.perf_counter() - t0
        out["deq"] = deq
        print_deq_table(deq)
        _write_json(outdir / f"deq_matched_fit{suffix}.json", deq)

    if args.probe in ("maze", "both"):
        t0 = time.perf_counter()
        maze = run_maze(seeds, eps=args.maze_eps, skip_assert=args.skip_assert, run_tag=run_tag,
                        check_anchors=check_anchors, m_min=m_min, floor_abs=floor_abs)
        maze["seconds"] = time.perf_counter() - t0
        out["maze"] = maze
        print_maze_table(maze)
        _write_json(outdir / f"maze_matched_fit{suffix}.json", maze)

    if m_min > 0 and deq is not None:
        dec = confirmatory_decision(deq, maze)
        print_confirmatory_decision(dec)
        out["confirmatory_decision"] = dec
        _write_json(outdir / f"confirmatory_decision{suffix}.json", dec)

    _write_json(outdir / f"summary{suffix}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
