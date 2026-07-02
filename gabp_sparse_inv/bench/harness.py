"""One benchmark rig: time + memory + accuracy, regenerable from ``(seed, config)``.

Timing uses ``torch.utils.benchmark`` (warmup + repeats, CUDA-synced on GPU). Memory is
reported analytically as the primary result; a measured peak is reported as a secondary,
explicitly-noisy number. Accuracy uses :mod:`gabp_sparse_inv.bench.metrics`.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field

try:
    import resource  # Unix-only
except ImportError:  # Windows
    resource = None

import torch
import torch.utils.benchmark as tbench

from ..chain import selected_inverse_chain
from ..generators import (
    condition_number,
    random_spd_chain,
    random_spd_star,
    random_spd_tree,
)
from ..autodiff import selinv_tree
from ..star import selected_inverse_star
from ..tree import selected_inverse_tree
from . import metrics

__all__ = [
    "PrecisionSpec",
    "resolve_precision",
    "BenchRecord",
    "GradBenchRecord",
    "bench_one",
    "bench_one_star",
    "bench_one_tree",
    "bench_grad_tree",
    "DEFAULT_PRECISIONS",
]

# Requested low-precision name -> storage dtype.
_STORAGE_DTYPE = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

DEFAULT_PRECISIONS = ("fp64", "fp32", "bf16")


@dataclass
class PrecisionSpec:
    """Fully-specified precision configuration (the 4-tuple)."""

    name: str
    input_storage: torch.dtype
    factor_storage: torch.dtype
    compute: torch.dtype
    output: torch.dtype
    runnable: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in asdict(self).items()}
        return d


def _native_cholesky_ok(dtype: torch.dtype, device: torch.device) -> bool:
    """Probe whether ``cholesky_ex`` runs natively in ``dtype`` on ``device``."""
    try:
        a = torch.eye(2, dtype=dtype, device=device)
        torch.linalg.cholesky_ex(a)
        return True
    except (RuntimeError, torch.linalg.LinAlgError):
        return False


def resolve_precision(name: str, device: torch.device) -> PrecisionSpec:
    """Resolve a precision name to a concrete, probed :class:`PrecisionSpec`.

    Low-precision dtypes with no native Cholesky kernel use a store-low / compute-fp32
    path: storage stays low, the factorization runs in fp32, output is cast back.
    """
    if name not in _STORAGE_DTYPE:
        raise ValueError(f"unknown precision {name!r}; known: {sorted(_STORAGE_DTYPE)}")
    storage = _STORAGE_DTYPE[name]

    if _native_cholesky_ok(storage, device):
        return PrecisionSpec(name, storage, storage, storage, storage, runnable=True,
                             note="native")

    # No native kernel (e.g. bf16/fp16 on CPU): upcast compute to fp32.
    if _native_cholesky_ok(torch.float32, device):
        return PrecisionSpec(
            name, storage, torch.float32, torch.float32, storage, runnable=True,
            note="store-low/compute-fp32 (no native half Cholesky on this backend)",
        )
    return PrecisionSpec(name, storage, storage, storage, storage, runnable=False,
                         note="no Cholesky backend available")


def analytic_memory(L: int, b: int) -> dict:
    """Element counts for structured storage vs the dense matrix."""
    structured = (L + max(L - 1, 0)) * b * b   # diag + lower blocks
    dense = (L * b) ** 2
    return {"structured_elems": structured, "dense_elems": dense,
            "ratio_dense_over_structured": dense / structured}


def _peak_rss_bytes() -> int:
    """Process peak RSS in bytes (noisy; secondary metric). Cross-platform best-effort."""
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage * 1024 if platform.system() == "Linux" else usage  # Linux: KiB, macOS: bytes
    if platform.system() == "Windows":  # no resource module; query peak working set via psapi
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    return 0


@dataclass
class BenchRecord:
    seed: int
    L: int                 # primary size: chain length L (chain) or num leaves K (star)
    b: int
    precision: dict
    device: str
    num_threads: int
    time_median_s: float
    time_iqr_s: float
    dense_inv_time_s: float | None
    dense_chol_time_s: float | None
    kappa: float
    forward_normwise: float
    forward_componentwise: float
    forward_worst_block: float
    factorization_residual: float
    selected_inverse_residual: float
    analytic: dict
    peak_rss_bytes_noisy: int
    problem: str = "chain"   # "chain" or "star"
    extra: dict = field(default_factory=dict)


def _time(fn, *, min_run_time: float = 0.2, num_threads: int = 1) -> tuple[float, float]:
    """Median and IQR (seconds) via ``torch.utils.benchmark``."""
    timer = tbench.Timer(stmt="fn()", globals={"fn": fn}, num_threads=num_threads)
    m = timer.blocked_autorange(min_run_time=min_run_time)
    return float(m.median), float(m.iqr)


def bench_one(
    *,
    L: int,
    b: int,
    seed: int,
    precision: str = "fp64",
    diag_load: float = 1.0,
    device: torch.device | str = "cpu",
    num_threads: int = 1,
    dense_baseline_max_L: int = 64,
    time_min_run_s: float = 0.2,
) -> BenchRecord:
    """Run the full time+memory+accuracy rig for one ``(seed, config)``."""
    device = torch.device(device)
    torch.set_num_threads(num_threads)
    spec = resolve_precision(precision, device)

    # Build the problem in fp64 then cast to the storage dtype.
    bt64 = random_spd_chain(L, b, seed=seed, diag_load=diag_load, dtype=torch.float64, device=device)
    diag = bt64.diag.to(spec.input_storage)
    lower = bt64.lower.to(spec.input_storage)
    compute_dtype = spec.compute if spec.compute != spec.input_storage else None

    def call():
        return selected_inverse_chain(diag, lower, compute_dtype=compute_dtype)

    if not spec.runnable:
        raise RuntimeError(f"precision {precision!r} not runnable: {spec.note}")

    # Accuracy (single run) + timing (many runs).
    G_diag, G_lower = call()
    fe = metrics.forward_error(G_diag, G_lower, bt64)
    _, _, factors = selected_inverse_chain(
        bt64.diag, bt64.lower, return_factors=True
    )  # factors in fp64 for the factorization residual
    fac_res = metrics.factorization_residual(bt64, factors)
    si_res = metrics.selected_inverse_residual(bt64, G_diag, G_lower)
    kappa = float(condition_number(bt64))

    t_med, t_iqr = _time(call, min_run_time=time_min_run_s, num_threads=num_threads)

    dense_inv_t = dense_chol_t = None
    if L <= dense_baseline_max_L:
        dense = bt64.to_dense().to(spec.input_storage if spec.runnable and spec.compute == spec.input_storage else torch.float64)
        eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=device)
        dense_inv_t, _ = _time(lambda: torch.linalg.inv(dense), min_run_time=time_min_run_s, num_threads=num_threads)
        dense_chol_t, _ = _time(
            lambda: torch.cholesky_solve(eye, torch.linalg.cholesky(dense)),
            min_run_time=time_min_run_s, num_threads=num_threads,
        )

    return BenchRecord(
        seed=seed, L=L, b=b, precision=spec.as_dict(), device=str(device),
        num_threads=num_threads, time_median_s=t_med, time_iqr_s=t_iqr,
        dense_inv_time_s=dense_inv_t, dense_chol_time_s=dense_chol_t, kappa=kappa,
        forward_normwise=fe.normwise, forward_componentwise=fe.componentwise,
        forward_worst_block=fe.worst_block, factorization_residual=fac_res,
        selected_inverse_residual=si_res, analytic=analytic_memory(L, b),
        peak_rss_bytes_noisy=_peak_rss_bytes(), problem="chain",
    )


def _analytic_memory_star(K: int, b: int) -> dict:
    """Element counts for star structured storage (center + K leaf + K coupling) vs dense."""
    structured = (1 + 2 * K) * b * b
    dense = ((K + 1) * b) ** 2
    return {"structured_elems": structured, "dense_elems": dense,
            "ratio_dense_over_structured": dense / structured}


def bench_one_star(
    *,
    K: int,
    b: int,
    seed: int,
    precision: str = "fp64",
    diag_load: float = 1.0,
    device: torch.device | str = "cpu",
    num_threads: int = 1,
    dense_baseline_max_K: int = 64,
    time_min_run_s: float = 0.2,
) -> BenchRecord:
    """Run the full time+memory+accuracy rig for one star ``(seed, config)``.

    The primary size axis is the number of leaves ``K`` (stored in the ``L`` field
    of :class:`BenchRecord`, with ``problem="star"``).
    """
    device = torch.device(device)
    torch.set_num_threads(num_threads)
    spec = resolve_precision(precision, device)

    st64 = random_spd_star(K, b, seed=seed, diag_load=diag_load, dtype=torch.float64, device=device)
    center = st64.center.to(spec.input_storage)
    leaf_diag = st64.leaf_diag.to(spec.input_storage)
    coupling = st64.coupling.to(spec.input_storage)
    compute_dtype = spec.compute if spec.compute != spec.input_storage else None

    if not spec.runnable:
        raise RuntimeError(f"precision {precision!r} not runnable: {spec.note}")

    def call():
        return selected_inverse_star(center, leaf_diag, coupling, compute_dtype=compute_dtype)

    G_c, G_l, G_x = call()
    fe = metrics.forward_error_star(G_c, G_l, G_x, st64)
    _, _, _, factors = selected_inverse_star(
        st64.center, st64.leaf_diag, st64.coupling, return_factors=True
    )  # factors in fp64 for the factorization residual
    fac_res = metrics.factorization_residual_star(st64, factors)
    si_res = metrics.selected_inverse_residual_star(st64, G_c, G_l, G_x)
    kappa = float(condition_number(st64))

    t_med, t_iqr = _time(call, min_run_time=time_min_run_s, num_threads=num_threads)

    dense_inv_t = dense_chol_t = None
    if K <= dense_baseline_max_K:
        dense = st64.to_dense().to(
            spec.input_storage if spec.runnable and spec.compute == spec.input_storage else torch.float64
        )
        eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=device)
        dense_inv_t, _ = _time(lambda: torch.linalg.inv(dense), min_run_time=time_min_run_s, num_threads=num_threads)
        dense_chol_t, _ = _time(
            lambda: torch.cholesky_solve(eye, torch.linalg.cholesky(dense)),
            min_run_time=time_min_run_s, num_threads=num_threads,
        )

    return BenchRecord(
        seed=seed, L=K, b=b, precision=spec.as_dict(), device=str(device),
        num_threads=num_threads, time_median_s=t_med, time_iqr_s=t_iqr,
        dense_inv_time_s=dense_inv_t, dense_chol_time_s=dense_chol_t, kappa=kappa,
        forward_normwise=fe.normwise, forward_componentwise=fe.componentwise,
        forward_worst_block=fe.worst_block, factorization_residual=fac_res,
        selected_inverse_residual=si_res, analytic=_analytic_memory_star(K, b),
        peak_rss_bytes_noisy=_peak_rss_bytes(), problem="star",
    )


@dataclass
class GradBenchRecord:
    """Forward+backward timing, gradient correctness, and memory for the diff. kernel."""

    seed: int
    n: int
    b: int
    kind: str
    device: str
    num_threads: int
    grad_time_loop_s: float
    grad_time_batched_s: float
    fwdbwd_dense_autograd_s: float | None
    adjoint_err_vs_dense: float | None
    grad_mem: dict
    problem: str = "tree-grad"


def _analytic_grad_memory_tree(n: int, b: int) -> dict:
    """Backward memory: structured ``O((|V|+|E|) b^2)`` vs taping a dense inverse.

    The analytic backward holds a constant number of ``[n, b, b]`` accumulators and
    reuses the forward factors -- ``O((2n-1) b^2)``, the same order as the forward
    storage. Autograd through a *dense* ``inv`` must tape the ``(n b)^2`` inverse (and
    its graph). The honest asymptotic win is vs. the dense tape, not vs. taping the
    structured loop (which is also ``O(n b^2)``). See derivations.md §8 / B2.
    """
    structured = (2 * n - 1) * b * b
    dense_autograd = (n * b) ** 2
    return {"structured_grad_elems": structured, "dense_autograd_elems": dense_autograd,
            "ratio_dense_over_structured": dense_autograd / structured}


def _dense_selinv_grads(bt64, wd, we):
    """Reference (diag, edge) gradients via autograd through a symmetrized dense inverse."""
    n, b = bt64.diag.shape[-3], bt64.diag.shape[-1]
    plist = bt64.parent.tolist()
    diag = bt64.diag.clone().requires_grad_(True)
    edge = bt64.edge.clone().requires_grad_(True)
    N = n * b
    mat = diag.new_zeros((N, N))
    for v in range(n):
        rv = slice(v * b, (v + 1) * b)
        mat[rv, rv] = diag[v]
        p = plist[v]
        if p != -1:
            rp = slice(p * b, (p + 1) * b)
            mat[rp, rv] = edge[v]
            mat[rv, rp] = edge[v].mT
    mat = 0.5 * (mat + mat.mT)
    G = torch.linalg.inv(mat)
    loss = sum((G[v * b:(v + 1) * b, v * b:(v + 1) * b] * wd[v]).sum() for v in range(n))
    loss = loss + sum(
        (G[plist[v] * b:(plist[v] + 1) * b, v * b:(v + 1) * b] * we[v]).sum()
        for v in range(n) if plist[v] != -1
    )
    loss.backward()
    return diag.grad, edge.grad


def bench_grad_tree(
    *,
    n: int,
    b: int,
    seed: int,
    diag_load: float = 1.0,
    kind: str = "random",
    device: torch.device | str = "cpu",
    num_threads: int = 1,
    dense_baseline_max_n: int = 64,
    time_min_run_s: float = 0.1,
) -> GradBenchRecord:
    """Time forward+backward (loop vs batched), check gradient vs dense, report memory."""
    device = torch.device(device)
    torch.set_num_threads(num_threads)
    bt64 = random_spd_tree(n, b, seed=seed, diag_load=diag_load, kind=kind,
                           dtype=torch.float64, device=device)
    diag, edge, parent = bt64.diag, bt64.edge, bt64.parent
    g = torch.Generator().manual_seed(seed)
    wd = torch.randn(n, b, b, generator=g, dtype=torch.float64, device=device)
    we = torch.randn(n, b, b, generator=g, dtype=torch.float64, device=device)

    def fwdbwd(batched: bool):
        d = diag.clone().requires_grad_(True)
        e = edge.clone().requires_grad_(True)
        Gd, Ge = selinv_tree(d, e, parent, batched=batched)
        ((Gd * wd).sum() + (Ge * we).sum()).backward()
        return d.grad

    t_loop, _ = _time(lambda: fwdbwd(False), min_run_time=time_min_run_s, num_threads=num_threads)
    t_batched, _ = _time(lambda: fwdbwd(True), min_run_time=time_min_run_s, num_threads=num_threads)

    dense_t = None
    adj_err = None
    if n <= dense_baseline_max_n:
        dense_t, _ = _time(lambda: _dense_selinv_grads(bt64, wd, we),
                           min_run_time=time_min_run_s, num_threads=num_threads)
        d = diag.clone().requires_grad_(True)
        e = edge.clone().requires_grad_(True)
        Gd, Ge = selinv_tree(d, e, parent)
        ((Gd * wd).sum() + (Ge * we).sum()).backward()
        gd_o, ge_o = _dense_selinv_grads(bt64, wd, we)
        adj_err = float(max((d.grad - gd_o).abs().max(), (e.grad - ge_o).abs().max()))

    return GradBenchRecord(
        seed=seed, n=n, b=b, kind=kind, device=str(device), num_threads=num_threads,
        grad_time_loop_s=t_loop, grad_time_batched_s=t_batched,
        fwdbwd_dense_autograd_s=dense_t, adjoint_err_vs_dense=adj_err,
        grad_mem=_analytic_grad_memory_tree(n, b),
    )


def _analytic_memory_tree(n: int, b: int) -> dict:
    """Element counts for tree structured storage (n node + (n-1) edge blocks) vs dense."""
    structured = (2 * n - 1) * b * b
    dense = (n * b) ** 2
    return {"structured_elems": structured, "dense_elems": dense,
            "ratio_dense_over_structured": dense / structured}


def bench_one_tree(
    *,
    n: int,
    b: int,
    seed: int,
    precision: str = "fp64",
    diag_load: float = 1.0,
    kind: str = "random",
    device: torch.device | str = "cpu",
    num_threads: int = 1,
    dense_baseline_max_n: int = 64,
    time_min_run_s: float = 0.2,
) -> BenchRecord:
    """Run the full time+memory+accuracy rig for one tree ``(seed, config)``.

    The primary size axis is the number of nodes ``n`` (stored in the ``L`` field of
    :class:`BenchRecord`, with ``problem="tree"``). ``kind`` selects the topology
    family (``random``/``path``/``star``/``balanced``).
    """
    device = torch.device(device)
    torch.set_num_threads(num_threads)
    spec = resolve_precision(precision, device)

    bt64 = random_spd_tree(n, b, seed=seed, diag_load=diag_load, kind=kind,
                           dtype=torch.float64, device=device)
    diag = bt64.diag.to(spec.input_storage)
    edge = bt64.edge.to(spec.input_storage)
    parent = bt64.parent
    compute_dtype = spec.compute if spec.compute != spec.input_storage else None

    if not spec.runnable:
        raise RuntimeError(f"precision {precision!r} not runnable: {spec.note}")

    def call():
        return selected_inverse_tree(diag, edge, parent, compute_dtype=compute_dtype)

    G_diag, G_edge = call()
    fe = metrics.forward_error_tree(G_diag, G_edge, bt64)
    _, _, factors = selected_inverse_tree(
        bt64.diag, bt64.edge, parent, return_factors=True
    )  # factors in fp64 for the factorization residual
    fac_res = metrics.factorization_residual_tree(bt64, factors)
    si_res = metrics.selected_inverse_residual_tree(bt64, G_diag, G_edge)
    kappa = float(condition_number(bt64))

    t_med, t_iqr = _time(call, min_run_time=time_min_run_s, num_threads=num_threads)

    dense_inv_t = dense_chol_t = None
    if n <= dense_baseline_max_n:
        dense = bt64.to_dense().to(
            spec.input_storage if spec.runnable and spec.compute == spec.input_storage else torch.float64
        )
        eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=device)
        dense_inv_t, _ = _time(lambda: torch.linalg.inv(dense), min_run_time=time_min_run_s, num_threads=num_threads)
        dense_chol_t, _ = _time(
            lambda: torch.cholesky_solve(eye, torch.linalg.cholesky(dense)),
            min_run_time=time_min_run_s, num_threads=num_threads,
        )

    return BenchRecord(
        seed=seed, L=n, b=b, precision=spec.as_dict(), device=str(device),
        num_threads=num_threads, time_median_s=t_med, time_iqr_s=t_iqr,
        dense_inv_time_s=dense_inv_t, dense_chol_time_s=dense_chol_t, kappa=kappa,
        forward_normwise=fe.normwise, forward_componentwise=fe.componentwise,
        forward_worst_block=fe.worst_block, factorization_residual=fac_res,
        selected_inverse_residual=si_res, analytic=_analytic_memory_tree(n, b),
        peak_rss_bytes_noisy=_peak_rss_bytes(), problem="tree", extra={"kind": kind},
    )
