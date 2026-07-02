"""Precision study: is structured selected inversion more accurate than a dense inverse?

Two algorithms that return the *same* result in exact arithmetic can disagree in floating
point. This harness puts the kernel
head-to-head with a dense inverse computed at the *same* precision, both scored against
an fp64 dense oracle, and only on the matrix's own pattern (the entries the kernel
actually returns).

Per (structure, dtype, condition number) it records, as relative error vs the oracle:

  * ``selinv_*``     -- the kernel,
  * ``dense_chol_*`` -- an SPD dense solve (``cholesky_solve``) at the same dtype: the
                        fair baseline, since it exploits SPD exactly like the kernel,
  * ``dense_inv_*``  -- a general dense inverse (``torch.linalg.inv``),
  * ``advantage``    -- ``dense_chol_normwise / selinv_normwise`` (``> 1`` => the kernel
                        is the more accurate algorithm at that precision and conditioning).

Condition number is swept through the generators' ``diag_load`` knob (smaller load =>
larger kappa) and *measured*, not assumed. The ``grid`` problem is the junction-tree kernel
on a loopy graph-Laplacian (``random_spd_laplacian``), scored on the kernel's *filled*
pattern ``S``; there ``diag_load`` is the Laplacian lift ``eps``.

Each record also reports the **output dynamic range** ``max_off |A^-1| / max_on |A^-1|`` (the
inverse mass off the kernel's returned pattern vs on it). It was originally conjectured to
*predict* the kernel's accuracy edge (more off-pattern mass => bigger edge). **That law is
refuted**: the well-scaled grid Laplacian reaches dynamic range ~0.999 yet shows *no* edge at
high kappa (~0.5x), while the ill-scaled ``random_spd_tree`` shows an edge at dynamic range
~0.43 -- the opposite ordering. The dynamic range is kept as a diagnostic; it does **not**
predict the advantage.

So what *does* drive the apparent edge on ill-scaled random trees? Mostly the **elimination
order**: the kernel implicitly factorizes in a fill-reducing (leaf-first / perfect-elimination)
order, so most of its "edge" is over a *naively-ordered* dense solve. ``tree_order_sensitivity``
(below, ``--compare-orders``) commits the experiment that pins this down -- run on the *same*
matrices it scores the kernel, a NATURAL-order dense ``cholesky_solve``, and a LEAF-FIRST-order
one. The honest, properly-seeded read (fp32, ``random_spd_tree``, 16 seeds; ratios are
**heavy-tailed**, so report medians WITH spread, never a single ratio):

  * advantage over *naive* dense is **modest in the median (~1.5-2.3x), not the ~7x** an early
    3-seed ratio-of-medians suggested; per-seed it ranges from ~0.3x (kernel worse) to ~40x;
  * a dense Cholesky in the kernel's **own elimination order** is ~0.9-1.4x of the kernel in
    the median -- i.e. **ordering explains the bulk of the median gap** -- though a residual,
    instance-dependent edge survives on the hardest trees (tail to ~12x);
  * the well-scaled **grid Laplacian** (large kappa, O(1) entries)
    shows **no edge at any kappa** (~0.5x, i.e. parity / slightly worse).

Net: **no precision penalty, and no *robust* precision win** -- accuracy is, in the median,
that of a well-ordered dense factorization. The structural value is ``O(n)`` time, ``O(fill)``
memory (never forming the dense inverse), and differentiability -- not accuracy. (See
``docs/ROADMAP.md`` Thread D for the current interpretation.)
Like the rest of the bench these are BLAS/dtype/device-dependent diagnostics -- recorded, not
asserted.

    python -m gabp_sparse_inv.bench.precision --problem tree --size 64 --b 4
    python -m gabp_sparse_inv.bench.precision --problem grid --size 8 --precisions fp32
    python -m gabp_sparse_inv.bench.precision --compare-orders --size 64 --b 4 \
        --diag-loads 1e-3 1e-5 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from ..chain import selected_inverse_chain
from ..generators import (
    condition_number,
    grid_edges,
    random_spd_chain,
    random_spd_laplacian,
    random_spd_star,
    random_spd_tree,
)
from ..junction import selected_inverse_junction
from ..layout import _as_parent_tensor, tree_orders
from ..star import selected_inverse_star
from ..tree import selected_inverse_tree
from . import metrics
from .harness import PrecisionSpec, resolve_precision

__all__ = ["PrecisionRecord", "precision_one", "main"]

_TINY = 1e-300  # guards the advantage ratio when the kernel is exact (e.g. fp64)


@dataclass
class _Adapter:
    """Per-structure plumbing for the head-to-head.

    The dense baseline is projected onto the pattern with the *same* extractor the oracle
    uses (``metrics._selected_from_dense*``), so the kernel and the dense inverse are
    scored on exactly the same entries by the same ``forward_error`` against the same
    oracle -- the only differences are the algorithm and the rounding. ``ferr`` and
    ``project`` take the shared fp64 ``oracle64`` / kernel ``blocks`` so the junction
    pattern (which is data-dependent fill) is threaded through uniformly. ``on_index``
    returns the off-diagonal on-pattern block pairs for the output-dynamic-range measure;
    ``None`` (star) skips that measure.
    """

    build: Callable[..., object]                          # (size, b, seed, diag_load, kind) -> layout
    selinv: Callable[[object, PrecisionSpec], tuple]      # layout, spec -> selected blocks
    ferr: Callable[[tuple, object, Tensor], metrics.ForwardError]  # blocks, layout, oracle64 -> error
    project: Callable[[Tensor, object, tuple], tuple]     # dense inverse, layout, blocks -> selected blocks
    nblocks: Callable[[object], int]                      # layout -> number of node blocks
    on_index: Callable[[object, tuple], list[tuple[int, int]]] | None


def _cd(spec: PrecisionSpec) -> torch.dtype | None:
    """compute_dtype to pass the kernel: ``None`` when compute == storage (native path)."""
    return spec.compute if spec.compute != spec.input_storage else None


# -- chain -------------------------------------------------------------------- #
def _build_chain(size, b, seed, dl, kind):
    return random_spd_chain(size, b, seed=seed, diag_load=dl)


def _selinv_chain(layout, spec):
    diag = layout.diag.to(spec.input_storage)
    lower = layout.lower.to(spec.input_storage)
    return selected_inverse_chain(diag, lower, compute_dtype=_cd(spec))


def _ferr_chain(blocks, layout, oracle64):
    return metrics.forward_error(blocks[0], blocks[1], layout)


def _project_chain(inv, layout, blocks):
    return metrics._selected_from_dense(inv, layout.num_blocks, layout.block_size)


def _on_chain(layout, blocks):
    return [(i + 1, i) for i in range(layout.num_blocks - 1)]


# -- star --------------------------------------------------------------------- #
def _build_star(size, b, seed, dl, kind):
    return random_spd_star(size, b, seed=seed, diag_load=dl)


def _selinv_star(layout, spec):
    center = layout.center.to(spec.input_storage)
    leaf = layout.leaf_diag.to(spec.input_storage)
    coupling = layout.coupling.to(spec.input_storage)
    return selected_inverse_star(center, leaf, coupling, compute_dtype=_cd(spec))


def _ferr_star(blocks, layout, oracle64):
    return metrics.forward_error_star(blocks[0], blocks[1], blocks[2], layout)


def _project_star(inv, layout, blocks):
    return metrics._selected_from_dense_star(inv, layout.num_leaves, layout.block_size)


# -- tree --------------------------------------------------------------------- #
def _build_tree(size, b, seed, dl, kind):
    return random_spd_tree(size, b, seed=seed, diag_load=dl, kind=kind)


def _selinv_tree(layout, spec):
    diag = layout.diag.to(spec.input_storage)
    edge = layout.edge.to(spec.input_storage)
    return selected_inverse_tree(diag, edge, layout.parent, compute_dtype=_cd(spec))


def _ferr_tree(blocks, layout, oracle64):
    return metrics.forward_error_tree(blocks[0], blocks[1], layout)


def _project_tree(inv, layout, blocks):
    return metrics._selected_from_dense_tree(inv, layout.parent.tolist(), layout.block_size)


def _on_tree(layout, blocks):
    par = layout.parent.tolist()
    return [(v, par[v]) for v in range(layout.num_nodes) if par[v] >= 0]


# -- grid (junction tree on a loopy graph) ------------------------------------ #
def _build_grid(size, b, seed, dl, kind):
    # ``size`` is the grid side length; eps = diag_load is the condition-number knob.
    edges = grid_edges(size, size)
    return random_spd_laplacian(size * size, edges, b, eps=dl, seed=seed)


def _selinv_grid(layout, spec):
    diag = layout.diag.to(spec.input_storage)
    edge_val = layout.edge_val.to(spec.input_storage)
    # store-low/compute path mirrors the other kernels; a low-precision Cholesky breakdown
    # still surfaces as a LinAlgError -> inf in scoring.
    return selected_inverse_junction(diag, layout.edge_index, edge_val, compute_dtype=_cd(spec))


def _ferr_grid(blocks, layout, oracle64):
    G_diag, S_index, G_lower = blocks
    return metrics.forward_error_junction(G_diag, S_index, G_lower, oracle64)


def _project_grid(inv, layout, blocks):
    _G_diag, S_index, _G_lower = blocks
    gd, gl = metrics._selected_from_dense_junction(inv, layout.num_nodes, S_index, layout.block_size)
    return (gd, S_index, gl)


def _on_grid(layout, blocks):
    S_index = blocks[1]
    if S_index.numel() == 0:
        return []
    return list(zip(S_index[0].tolist(), S_index[1].tolist()))


_ADAPTERS = {
    "chain": _Adapter(_build_chain, _selinv_chain, _ferr_chain, _project_chain,
                      lambda L: L.num_blocks, _on_chain),
    "star": _Adapter(_build_star, _selinv_star, _ferr_star, _project_star,
                     lambda L: L.num_leaves + 1, None),
    "tree": _Adapter(_build_tree, _selinv_tree, _ferr_tree, _project_tree,
                     lambda L: L.num_nodes, _on_tree),
    "grid": _Adapter(_build_grid, _selinv_grid, _ferr_grid, _project_grid,
                     lambda L: L.num_nodes, _on_grid),
}


def _dense_inverse(dense64: Tensor, spec: PrecisionSpec, method: str) -> Tensor:
    """Dense inverse at the spec's precision: store low, compute, cast back, symmetrize.

    Mirrors the kernel's store-low/compute policy so the comparison is apples-to-apples:
    the matrix is rounded to the storage dtype, the solve runs in the compute dtype, and
    the result is cast to the output dtype before scoring. ``method`` is ``"chol"`` (SPD
    ``cholesky_solve``) or ``"inv"`` (general ``torch.linalg.inv``).
    """
    A = dense64.to(spec.input_storage).to(spec.compute)
    if method == "chol":
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)
        inv = torch.cholesky_solve(eye, torch.linalg.cholesky(A))
    else:
        inv = torch.linalg.inv(A)
    inv = 0.5 * (inv + inv.mT)
    return inv.to(spec.output)


def _safe_score(fn: Callable[[], metrics.ForwardError]) -> tuple[float, float, float]:
    """Score a method, mapping a low-precision linalg breakdown (lost positive-
    definiteness / singular factor) to ``inf`` error so the sweep continues and the
    failure point is visible in the data rather than aborting the run."""
    try:
        fe = fn()
        return fe.normwise, fe.componentwise, fe.worst_block
    except torch.linalg.LinAlgError:
        return float("inf"), float("inf"), float("inf")


@dataclass
class PrecisionRecord:
    problem: str
    precision: str
    note: str            # PrecisionSpec.note: "native" vs store-low/compute-fp32
    size: int
    b: int
    seed: int
    diag_load: float
    kappa: float
    dyn_range: float     # output dynamic range max_off|A^-1| / max_on|A^-1| (nan if unmeasured)
    selinv_normwise: float
    selinv_componentwise: float
    selinv_worst_block: float
    dense_chol_normwise: float
    dense_chol_componentwise: float
    dense_inv_normwise: float
    dense_inv_componentwise: float
    advantage_chol: float    # dense_chol_normwise / selinv_normwise  (> 1 => kernel wins)
    advantage_inv: float     # dense_inv_normwise  / selinv_normwise


def precision_one(
    problem: str,
    *,
    size: int,
    b: int,
    seed: int,
    diag_load: float,
    precision: str,
    kind: str = "random",
    device: torch.device | str = "cpu",
) -> PrecisionRecord:
    """One head-to-head record for a single (structure, dtype, conditioning, seed)."""
    device = torch.device(device)
    spec = resolve_precision(precision, device)
    if not spec.runnable:
        raise RuntimeError(f"precision {precision!r} not runnable: {spec.note}")
    adapter = _ADAPTERS[problem]

    layout = adapter.build(size, b, seed, diag_load, kind)
    kappa = float(condition_number(layout))

    dense64 = layout.to_dense().to(torch.float64)
    oracle64 = _dense_inverse(dense64, resolve_precision("fp64", device), "chol")

    # The kernel runs once; its (possibly filled) pattern feeds both the projection of the
    # dense baselines and the output-dynamic-range measure, so all three see the same S.
    try:
        sel_blocks = adapter.selinv(layout, spec)
        sel_fe = adapter.ferr(sel_blocks, layout, oracle64)
        sel_n, sel_c, sel_w = sel_fe.normwise, sel_fe.componentwise, sel_fe.worst_block
    except torch.linalg.LinAlgError:
        sel_blocks = None
        sel_n = sel_c = sel_w = float("inf")

    if adapter.on_index is not None and sel_blocks is not None:
        n = adapter.nblocks(layout)
        on_pairs = adapter.on_index(layout, sel_blocks)
        dyn_range = metrics.output_dynamic_range(oracle64, on_pairs, n, layout.block_size)
    else:
        dyn_range = float("nan")

    pattern_blocks = sel_blocks if sel_blocks is not None else adapter.selinv(layout, resolve_precision("fp64", device))
    chol_n, chol_c, _ = _safe_score(
        lambda: adapter.ferr(adapter.project(_dense_inverse(dense64, spec, "chol"), layout, pattern_blocks), layout, oracle64))
    inv_n, inv_c, _ = _safe_score(
        lambda: adapter.ferr(adapter.project(_dense_inverse(dense64, spec, "inv"), layout, pattern_blocks), layout, oracle64))

    denom = max(sel_n, _TINY)
    return PrecisionRecord(
        problem=problem, precision=precision, note=spec.note, size=size, b=b, seed=seed,
        diag_load=diag_load, kappa=kappa, dyn_range=dyn_range,
        selinv_normwise=sel_n, selinv_componentwise=sel_c, selinv_worst_block=sel_w,
        dense_chol_normwise=chol_n, dense_chol_componentwise=chol_c,
        dense_inv_normwise=inv_n, dense_inv_componentwise=inv_c,
        advantage_chol=chol_n / denom, advantage_inv=inv_n / denom,
    )


DEFAULT_DIAG_LOADS = [10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# Field names of the ForwardError metric swept in the order comparison.
_ORDER_METRICS = ("normwise", "componentwise", "worst_block")


def tree_order_sensitivity(
    *,
    size: int,
    b: int,
    diag_load: float,
    precision: str,
    seeds: list[int],
    kind: str = "random",
    device: torch.device | str = "cpu",
) -> list[tuple[float, dict[str, tuple[float, float]]]]:
    """Is the tree kernel's accuracy edge over a dense solve an *elimination-ordering* artifact?

    This commits the experiment behind the Thread-D "ordering" claim (an earlier planning run
    was never committed -- this is its reproducible form). For each seed, on the *same*
    ill-conditioned ``random_spd_tree``, score the on-pattern blocks (harness metric
    :func:`metrics.forward_error_tree`, vs an fp64 oracle) of three algorithms:

      * the ``selinv`` tree kernel -- factorizes leaf-first / perfect-elimination, zero fill;
      * a dense ``cholesky_solve`` in NATURAL node order (what a naive dense baseline does);
      * a dense ``cholesky_solve`` in the kernel's own LEAF-FIRST order (``P A Pᵀ``).

    Returns ``[(kappa, {metric: (nat/ker, leaf/ker)}), ...]``, one tuple per seed. **Read it as
    a median WITH spread:** the ratios are heavy-tailed (per-seed they span ~0.3x..40x at high
    kappa), so a single ratio -- or a 3-seed median-of-medians -- is misleading. The honest
    summary (16 seeds): ``nat/ker`` median ~1.5-2.3x, ``leaf/ker`` median ~0.9-1.4x. So ordering
    explains the bulk of the (modest) median gap, with a residual edge only on the worst trees.
    """
    spec = resolve_precision(precision, device)
    out: list[tuple[float, dict[str, tuple[float, float]]]] = []
    for seed in seeds:
        bt = random_spd_tree(size, b, seed=seed, diag_load=diag_load, kind=kind)
        plist = bt.parent.tolist()
        kappa = float(condition_number(bt))
        _root, _children, collect = tree_orders(_as_parent_tensor(bt.parent))
        # leaf-first block permutation: node `collect[r]` occupies block-row r of P A Pᵀ.
        perm = torch.tensor([i for node in collect for i in range(node * b, (node + 1) * b)])

        A64 = bt.to_dense().to(torch.float64)
        A_lo = A64.to(spec.input_storage).to(spec.compute)         # store-low, compute dtype
        N = A64.shape[-1]
        eye = torch.eye(N, dtype=A_lo.dtype, device=A_lo.device)

        def dense_chol(reorder: bool):
            """Selected tree blocks from a dense cholesky_solve, optionally in leaf-first order."""
            M = A_lo[perm][:, perm] if reorder else A_lo
            inv = torch.cholesky_solve(eye, torch.linalg.cholesky(M)).to(torch.float64)
            if reorder:                                            # undo the permutation
                full = torch.empty(N, N, dtype=torch.float64)
                full[perm.unsqueeze(1), perm.unsqueeze(0)] = inv
                inv = full
            return metrics._selected_from_dense_tree(0.5 * (inv + inv.mT), plist, b)

        diag = bt.diag.to(spec.input_storage)
        edge = bt.edge.to(spec.input_storage)
        Gd, Ge = selected_inverse_tree(diag, edge, bt.parent, compute_dtype=_cd(spec))
        fk = metrics.forward_error_tree(Gd, Ge, bt)
        fn = metrics.forward_error_tree(*dense_chol(reorder=False), bt)
        fl = metrics.forward_error_tree(*dense_chol(reorder=True), bt)
        ratios = {
            m: (getattr(fn, m) / max(getattr(fk, m), _TINY),
                getattr(fl, m) / max(getattr(fk, m), _TINY))
            for m in _ORDER_METRICS
        }
        out.append((kappa, ratios))
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Precision study: structured selected inverse vs a dense inverse at "
                    "matched precision, swept over condition number.")
    ap.add_argument("--problem", choices=["chain", "star", "tree", "grid"], default="tree",
                    help="chain: size = length L; star: leaves K; tree: nodes n; "
                         "grid: side length (n = size^2, junction tree on a loopy graph)")
    ap.add_argument("--size", type=int, default=64, help="primary size (L / K / n / grid side)")
    ap.add_argument("--b", type=int, default=4, help="block size")
    ap.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--diag-loads", type=float, nargs="+", default=DEFAULT_DIAG_LOADS,
                    help="condition-number knob; smaller => larger kappa")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tree-kind", default="random", help="tree topology (--problem tree)")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--out", default="precision_results", help="output path prefix")
    ap.add_argument("--compare-orders", action="store_true",
                    help="run the tree elimination-order sensitivity study "
                         "(kernel vs natural- vs leaf-first-order dense) instead of the sweep; "
                         "use many --seeds, the ratios are heavy-tailed")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.num_threads)

    if args.compare_orders:
        prec = args.precisions[0]
        print(f"\nelimination-order sensitivity  problem=tree size={args.size} b={args.b} "
              f"[{prec}], {len(args.seeds)} seeds")
        print("  ratios vs the kernel, median [min, max] over seeds; heavy-tailed -> read the "
              "spread, not the median alone\n")
        for dl in args.diag_loads:
            recs = tree_order_sensitivity(size=args.size, b=args.b, diag_load=dl,
                                          precision=prec, seeds=args.seeds, kind=args.tree_kind)
            kappa = _median([k for k, _ in recs])
            print(f"  diag_load={dl:.0e}  kappa~{kappa:.2e}")
            for m in _ORDER_METRICS:
                nat = sorted(r[m][0] for _, r in recs)
                leaf = sorted(r[m][1] for _, r in recs)
                fmt = lambda xs: f"{_median(xs):5.2f} [{min(xs):.2f}, {max(xs):.2f}]"  # noqa: E731
                print(f"    {m:14}: NAT/ker {fmt(nat):24}  LEAF/ker {fmt(leaf)}")
            print()
        print("read: NAT/ker>1 => kernel beats a *naive*-order dense solve; LEAF/ker~1 => a "
              "dense solve in the kernel's own order matches it (ordering, not structure, is the "
              "bulk of the edge). No robust win; no penalty.  (diagnostic, not asserted)\n")
        return 0

    records: list[dict] = []
    for prec in args.precisions:
        for dl in args.diag_loads:
            for seed in args.seeds:
                rec = precision_one(args.problem, size=args.size, b=args.b, seed=seed,
                                    diag_load=dl, precision=prec, kind=args.tree_kind)
                records.append(asdict(rec))

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(records[0].keys())
        for r in records:
            w.writerow(r.values())

    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    print(f"problem={args.problem} size={args.size} b={args.b}  "
          f"(median over seeds; on-pattern normwise error vs fp64 oracle; not asserted)\n")
    for prec in args.precisions:
        sub = [r for r in records if r["precision"] == prec]
        if not sub:
            continue
        print(f"[{prec}] {sub[0]['note']}")
        print(f"  {'kappa':>10}  {'dyn_range':>10}  {'selinv':>10}  {'dense_chol':>10}  "
              f"{'dense_inv':>10}  {'sel/chol':>9}")
        by_dl: dict[float, list[dict]] = {}
        for r in sub:
            by_dl.setdefault(r["diag_load"], []).append(r)
        rows = sorted(
            (
                _median([r["kappa"] for r in rs]),
                _median([r["dyn_range"] for r in rs]),
                _median([r["selinv_normwise"] for r in rs]),
                _median([r["dense_chol_normwise"] for r in rs]),
                _median([r["dense_inv_normwise"] for r in rs]),
                _median([r["advantage_chol"] for r in rs]),
            )
            for rs in by_dl.values()
        )
        for kappa, dr, sel, chol, inv, adv in rows:
            print(f"  {kappa:10.2e}  {dr:10.2e}  {sel:10.2e}  {chol:10.2e}  {inv:10.2e}  {adv:8.2f}x")
        wins = sum(1 for r in rows if r[5] > 1.0)
        finite_adv = [r[5] for r in rows if math.isfinite(r[5])]
        med = _median(finite_adv) if finite_adv else float("nan")
        print(f"  selinv more accurate than dense-chol at {wins}/{len(rows)} conditions; "
              f"median advantage = {med:.2f}x  (any edge is an elimination-ordering artifact, "
              f"not a dyn_range law; inf = method broke down)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
