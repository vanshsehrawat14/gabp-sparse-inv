"""Backward-stability study: verify the tracked-constant theorem of derivations.md Sec 6.

The Sec 6 theorem (eq. 9a) says the computed tree selected inverse is the exact selected
inverse of ``A + E`` with ``||E||_2 <= c(b, Delta) * u * ||A||_2`` -- a constant that is a
low-degree polynomial in the block size ``b`` and node degree ``Delta``, and is
**independent of the tree height and of kappa(A)** (no growth factor). The forward error
then tracks ``kappa`` (eq. 9): ``||Ghat - G|| / ||G|| <= c(b, Delta) * kappa(A) * u``.

This harness measures both, sweeping the two axes the theorem says decouple -- tree
**height** and **kappa** -- on balanced trees (controlled height, fixed degree Delta=3):

  * ``backward_over_u = ||E||_2 / (u ||A||_2)``, where ``E = Lhat Dhat Lhat^T - A`` is built
    from the kernel's own fp32 ``LDL^T`` factors -- the literal eq.-(9a) object. The claim:
    this stays ~ ``c(b, Delta)`` (a small constant), **flat across height and kappa**.
  * ``forward_over_kappa_u = (||Ghat - G|| / ||G||) / (kappa * u)``: the claim is this is
    also ~ ``c(b, Delta)`` (flat), i.e. the *forward* error grows like ``kappa * u`` while
    the *backward* constant does not.

Like the rest of ``bench/`` these are BLAS/dtype/device-dependent **diagnostics** -- recorded
and printed, not asserted as thresholds (the theorem is about the *form*, not a tuned const).

    python -m gabp_sparse_inv.bench.stability
    python -m gabp_sparse_inv.bench.stability --depths 3 4 5 6 --b 1 2 4
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..generators import condition_number, random_spd_tree
from ..gmrf import _tree_collect_batched
from ..layout import _as_parent_tensor, tree_orders
from ..tree import selected_inverse_tree
from . import metrics

__all__ = ["StabilityRecord", "stability_one", "main"]


@dataclass
class StabilityRecord:
    depth: int
    n: int
    height: int
    b: int
    seed: int
    diag_load: float
    kappa: float
    storage: str
    backward_norm: float          # ||E||_2 / ||A||_2
    backward_over_u: float        # ||E||_2 / (u ||A||_2)  -- ~ c(b, Delta), height/kappa-free
    forward_norm: float           # ||Ghat - G|| / ||G||  (on the pattern)
    forward_over_kappa_u: float   # forward_norm / (kappa u) -- ~ c(b, Delta)


_STORAGE = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16}


def _balanced_n(depth: int) -> int:
    return 2 ** (depth + 1) - 1


def _assemble_ldlt(chol_D, ell, parent_t, root, n, b) -> torch.Tensor:
    """Dense A_hat = L D L^T from the (fp32) tree factors, upcast to fp64.

    ``L`` is unit lower-triangular with ``L_{p(v), v} = ell[v]``; ``D`` is block-diagonal
    with ``D_v = C_v C_v^T`` (``C_v = chol_D[v]``). On a tree this reproduces ``A`` exactly
    in exact arithmetic, so ``A_hat - A`` is precisely the factorization backward error.
    """
    plist = parent_t.tolist()
    N = n * b
    L = torch.eye(N, dtype=torch.float64)
    D = torch.zeros((N, N), dtype=torch.float64)
    for v in range(n):
        Cv = chol_D[..., v, :, :].to(torch.float64)
        D[v * b:(v + 1) * b, v * b:(v + 1) * b] = Cv @ Cv.mT
        if v != root:
            p = plist[v]
            L[p * b:(p + 1) * b, v * b:(v + 1) * b] = ell[..., v, :, :].to(torch.float64)
    return L @ D @ L.mT


def stability_one(
    *,
    depth: int,
    b: int,
    seed: int,
    diag_load: float,
    storage: str = "fp32",
) -> StabilityRecord:
    """One backward/forward stability record for a balanced tree at a given precision."""
    n = _balanced_n(depth)
    dtype = _STORAGE[storage]
    u = float(torch.finfo(dtype).eps)

    layout = random_spd_tree(n, b, seed=seed, diag_load=diag_load, kind="balanced")  # fp64
    parent_t = _as_parent_tensor(layout.parent)
    root, _children, _collect = tree_orders(parent_t)

    A = layout.to_dense().to(torch.float64)
    normA = float(torch.linalg.matrix_norm(A, ord=2))
    kappa = float(condition_number(layout))

    # ---- backward error E = Lhat Dhat Lhat^T - A from the kernel's own low-precision factors.
    diag_s = layout.diag.to(dtype)
    edge_s = layout.edge.to(dtype)
    chol_D, ell, _ld, _root, _lvl = _tree_collect_batched(diag_s, edge_s, parent_t)
    A_hat = _assemble_ldlt(chol_D, ell, parent_t, root, n, b)
    normE = float(torch.linalg.matrix_norm(A_hat - A, ord=2))
    backward_norm = normE / normA
    backward_over_u = backward_norm / u

    # ---- forward error of the selected inverse vs the fp64 dense oracle.
    G_diag, G_edge = selected_inverse_tree(diag_s, edge_s, layout.parent)
    fe = metrics.forward_error_tree(G_diag, G_edge, layout)
    forward_norm = fe.normwise
    forward_over_kappa_u = forward_norm / (kappa * u)

    return StabilityRecord(
        depth=depth, n=n, height=depth, b=b, seed=seed, diag_load=diag_load,
        kappa=kappa, storage=storage,
        backward_norm=backward_norm, backward_over_u=backward_over_u,
        forward_norm=forward_norm, forward_over_kappa_u=forward_over_kappa_u,
    )


DEFAULT_DEPTHS = [3, 4, 5, 6]
DEFAULT_DIAG_LOADS = [1.0, 1e-1, 1e-2, 1e-3]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Backward-stability study: verify ||E|| <= c(b,Delta) u ||A|| is "
                    "height- and kappa-free (derivations.md Sec 6, eq. 9a).")
    ap.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS,
                    help="balanced-tree depths (height); n = 2^(depth+1)-1")
    ap.add_argument("--b", type=int, nargs="+", default=[2], help="block size(s)")
    ap.add_argument("--diag-loads", type=float, nargs="+", default=DEFAULT_DIAG_LOADS,
                    help="condition-number knob; smaller => larger kappa")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--storage", default="fp32", choices=list(_STORAGE))
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--out", default="stability_results", help="output path prefix")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.num_threads)

    records: list[dict] = []
    for b in args.b:
        for depth in args.depths:
            for dl in args.diag_loads:
                for seed in args.seeds:
                    rec = stability_one(depth=depth, b=b, seed=seed, diag_load=dl,
                                        storage=args.storage)
                    records.append(asdict(rec))

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(records[0].keys())
        for r in records:
            w.writerow(r.values())

    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    print(f"storage={args.storage}  (median over seeds; diagnostics, not asserted)")
    print("Claim (Sec 6, eq. 9a): backward/u is ~constant -- flat across height AND kappa;")
    print("forward error tracks kappa*u, so forward/(kappa*u) is ~constant too.\n")
    for b in args.b:
        print(f"[b={b}]  {'depth':>5} {'n':>5} {'kappa':>10} {'backward/u':>11} "
              f"{'forward':>11} {'fwd/(k*u)':>11}")
        sub = [r for r in records if r["b"] == b]
        by_key: dict[tuple, list[dict]] = {}
        for r in sub:
            by_key.setdefault((r["depth"], r["diag_load"]), []).append(r)
        for (depth, _dl), rs in sorted(by_key.items()):
            print(
                f"        {depth:>5} {rs[0]['n']:>5} {_median([r['kappa'] for r in rs]):>10.2e} "
                f"{_median([r['backward_over_u'] for r in rs]):>11.2f} "
                f"{_median([r['forward_norm'] for r in rs]):>11.2e} "
                f"{_median([r['forward_over_kappa_u'] for r in rs]):>11.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
