"""Accuracy metrics for the supported selected-inverse kernels.

Each problem has independent checks plus a forward error against a dense oracle:

1. ``forward_error`` -- compares the computed selected blocks to the dense inverse
   restricted to the pattern (needs an oracle).
2. ``factorization_residual`` -- reconstructs ``A`` from the factors ``(D, ell)`` and
   compares to the input. Independent of the back-recursion.
3. ``selected_inverse_residual`` -- the diagonal block-rows of ``A G = I``. For a
   block-tridiagonal ``A`` these rows involve only on-pattern blocks of ``G``, giving a
   genuine ``A G = I`` check that needs no oracle and is far stronger than verifying the
   recurrence against itself.

The star residual uses the analogous on-pattern center, cross, and leaf equations;
off-pattern leaf-leaf inverse blocks are generally nonzero and are not checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..chain import ChainFactors
from ..layout import BlockStar, BlockTree, BlockTridiag, tree_orders
from ..star import StarFactors
from ..tree import TreeFactors

__all__ = [
    "ForwardError",
    "forward_error",
    "factorization_residual",
    "selected_inverse_residual",
    "dense_oracle_star",
    "forward_error_star",
    "factorization_residual_star",
    "selected_inverse_residual_star",
    "dense_oracle_tree",
    "forward_error_tree",
    "factorization_residual_tree",
    "selected_inverse_residual_tree",
    "forward_error_junction",
    "output_dynamic_range",
]


@dataclass
class ForwardError:
    """Forward-error summary against the dense oracle."""

    normwise: float            # ||G_hat - G|| / ||G|| over all selected blocks
    componentwise: float       # max |G_hat - G| / (|G| + eps) entrywise
    worst_block: float         # worst per-block normwise relative error


def _selected_from_dense(inv: Tensor, L: int, b: int) -> tuple[Tensor, Tensor]:
    """Extract the on-pattern diagonal and sub-diagonal blocks of a dense inverse."""
    diag = inv.new_empty((*inv.shape[:-2], L, b, b))
    lower = inv.new_empty((*inv.shape[:-2], max(L - 1, 0), b, b))
    for i in range(L):
        r = slice(i * b, (i + 1) * b)
        diag[..., i, :, :] = inv[..., r, r]
        if i < L - 1:
            c = slice((i + 1) * b, (i + 2) * b)
            lower[..., i, :, :] = inv[..., c, r]
    return diag, lower


def dense_oracle(bt: BlockTridiag) -> tuple[Tensor, Tensor]:
    """Selected blocks of ``A^{-1}`` via a dense fp64 Cholesky solve (the oracle)."""
    dense = bt.to_dense().to(torch.float64)
    eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=dense.device).expand_as(dense)
    chol = torch.linalg.cholesky(dense)
    inv = torch.cholesky_solve(eye, chol)
    inv = 0.5 * (inv + inv.mT)
    return _selected_from_dense(inv, bt.num_blocks, bt.block_size)


def forward_error(
    G_diag: Tensor, G_lower: Tensor, bt: BlockTridiag, *, eps: float = 1e-30
) -> ForwardError:
    """Forward error of computed selected blocks vs the dense oracle."""
    ref_diag, ref_lower = dense_oracle(bt)
    g_diag = G_diag.to(torch.float64)
    g_lower = G_lower.to(torch.float64)

    num_sq = (g_diag - ref_diag).pow(2).sum() + (g_lower - ref_lower).pow(2).sum()
    den_sq = ref_diag.pow(2).sum() + ref_lower.pow(2).sum()
    normwise = float((num_sq.sqrt() / den_sq.sqrt().clamp_min(eps)))

    comp = max(
        float(((g_diag - ref_diag).abs() / (ref_diag.abs() + eps)).amax()),
        float(((g_lower - ref_lower).abs() / (ref_lower.abs() + eps)).amax()) if g_lower.numel() else 0.0,
    )

    # Per-block normwise relative error, worst case over all blocks.
    bd_num = (g_diag - ref_diag).flatten(-2).norm(dim=-1)
    bd_den = ref_diag.flatten(-2).norm(dim=-1).clamp_min(eps)
    worst = float((bd_num / bd_den).amax())
    if g_lower.numel():
        bl_num = (g_lower - ref_lower).flatten(-2).norm(dim=-1)
        bl_den = ref_lower.flatten(-2).norm(dim=-1).clamp_min(eps)
        worst = max(worst, float((bl_num / bl_den).amax()))

    return ForwardError(normwise=normwise, componentwise=comp, worst_block=worst)


def factorization_residual(bt: BlockTridiag, factors: ChainFactors) -> float:
    """Max abs residual of ``A`` reconstructed from ``(D, ell)``.

    Checks ``A_ii == D_i + ell_i D_{i-1} ell_i^T`` (with ``A_11 == D_1``) and
    ``A_{i+1,i} == ell_{i+1} D_i``. Independent of the Takahashi back-recursion.
    """
    chol_D, ell = factors.chol_D, factors.ell
    diag, lower = bt.diag.to(torch.float64), bt.lower.to(torch.float64)
    chol_D = chol_D.to(torch.float64)
    ell = ell.to(torch.float64)
    L = bt.num_blocks

    D = chol_D @ chol_D.mT  # reconstruct pivots D_i = L_i L_i^T

    res = 0.0
    # Diagonal: A_11 = D_1.
    res = max(res, float((diag[..., 0, :, :] - D[..., 0, :, :]).abs().amax()))
    for i in range(1, L):
        ell_i = ell[..., i - 1, :, :]
        D_prev = D[..., i - 1, :, :]
        recon = D[..., i, :, :] + ell_i @ D_prev @ ell_i.mT
        res = max(res, float((diag[..., i, :, :] - recon).abs().amax()))
        # Sub-diagonal: A_{i+1,i} (== lower[i-1]) = ell_i D_{i-1}.
        recon_lo = ell_i @ D_prev
        res = max(res, float((lower[..., i - 1, :, :] - recon_lo).abs().amax()))
    return res


def selected_inverse_residual(bt: BlockTridiag, G_diag: Tensor, G_lower: Tensor) -> float:
    """Max abs residual of the diagonal block-rows of ``A G = I``.

    For block-tridiagonal ``A`` the ``i``-th block-row of ``A G`` restricted to the
    diagonal block is

        A_{i,i-1} G_{i-1,i} + A_ii G_ii + A_{i,i+1} G_{i+1,i} = I_b,

    and every ``G`` block appearing is on the pattern:
    ``G_{i-1,i} = G_lower[i-1]^T``, ``G_{i+1,i} = G_lower[i]``.
    """
    diag, lower = bt.diag.to(torch.float64), bt.lower.to(torch.float64)
    G_diag = G_diag.to(torch.float64)
    G_lower = G_lower.to(torch.float64)
    L, b = bt.num_blocks, bt.block_size
    eye = torch.eye(b, dtype=torch.float64, device=diag.device)

    res = 0.0
    for i in range(L):
        acc = diag[..., i, :, :] @ G_diag[..., i, :, :]
        if i > 0:
            # A_{i,i-1} = lower[i-1] ; G_{i-1,i} = G_lower[i-1]^T
            acc = acc + lower[..., i - 1, :, :] @ G_lower[..., i - 1, :, :].mT
        if i < L - 1:
            # A_{i,i+1} = lower[i]^T ; G_{i+1,i} = G_lower[i]
            acc = acc + lower[..., i, :, :].mT @ G_lower[..., i, :, :]
        res = max(res, float((acc - eye).abs().amax()))
    return res


# --------------------------------------------------------------------------- #
# Star (block-arrowhead) metrics.
# --------------------------------------------------------------------------- #
def _selected_from_dense_star(inv: Tensor, K: int, b: int) -> tuple[Tensor, Tensor, Tensor]:
    """Extract on-pattern blocks of a dense star inverse (center, leaf diag, cross G_{j,0})."""
    c0 = slice(0, b)
    G_center = inv[..., c0, c0].clone()
    G_leaf = inv.new_empty((*inv.shape[:-2], K, b, b))
    G_cross = inv.new_empty((*inv.shape[:-2], K, b, b))
    for j in range(K):
        r = slice((j + 1) * b, (j + 2) * b)
        G_leaf[..., j, :, :] = inv[..., r, r]
        G_cross[..., j, :, :] = inv[..., r, c0]      # G_{j,0}
    return G_center, G_leaf, G_cross


def dense_oracle_star(st: BlockStar) -> tuple[Tensor, Tensor, Tensor]:
    """Selected blocks of ``A^{-1}`` via a dense fp64 Cholesky solve (the oracle)."""
    dense = st.to_dense().to(torch.float64)
    eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=dense.device).expand_as(dense)
    chol = torch.linalg.cholesky(dense)
    inv = torch.cholesky_solve(eye, chol)
    inv = 0.5 * (inv + inv.mT)
    return _selected_from_dense_star(inv, st.num_leaves, st.block_size)


def forward_error_star(
    G_center: Tensor, G_leaf: Tensor, G_cross: Tensor, st: BlockStar, *, eps: float = 1e-30
) -> ForwardError:
    """Forward error of computed star selected blocks vs the dense oracle."""
    ref_c, ref_l, ref_x = dense_oracle_star(st)
    g_c = G_center.to(torch.float64)
    g_l = G_leaf.to(torch.float64)
    g_x = G_cross.to(torch.float64)

    num_sq = (g_c - ref_c).pow(2).sum() + (g_l - ref_l).pow(2).sum() + (g_x - ref_x).pow(2).sum()
    den_sq = ref_c.pow(2).sum() + ref_l.pow(2).sum() + ref_x.pow(2).sum()
    normwise = float(num_sq.sqrt() / den_sq.sqrt().clamp_min(eps))

    comp = max(
        float(((g_c - ref_c).abs() / (ref_c.abs() + eps)).amax()),
        float(((g_l - ref_l).abs() / (ref_l.abs() + eps)).amax()),
        float(((g_x - ref_x).abs() / (ref_x.abs() + eps)).amax()),
    )

    worst = 0.0
    for g, ref in ((g_c.unsqueeze(-3), ref_c.unsqueeze(-3)), (g_l, ref_l), (g_x, ref_x)):
        num = (g - ref).flatten(-2).norm(dim=-1)
        den = ref.flatten(-2).norm(dim=-1).clamp_min(eps)
        worst = max(worst, float((num / den).amax()))
    return ForwardError(normwise=normwise, componentwise=comp, worst_block=worst)


def factorization_residual_star(st: BlockStar, factors: StarFactors) -> float:
    """Max abs residual of ``A`` reconstructed from the star factors ``(chol_leaf, chol_S, B)``.

    Checks ``A_jj = L_j L_j^T``, ``A_{j,0} = A_jj B_j``, and the center
    ``A_00 = S + sum_j B_j^T A_jj B_j``. Independent of the distribute pass.
    """
    chol_leaf = factors.chol_leaf.to(torch.float64)
    chol_S = factors.chol_S.to(torch.float64)
    B = factors.B.to(torch.float64)
    center = st.center.to(torch.float64)
    leaf_diag = st.leaf_diag.to(torch.float64)
    coupling = st.coupling.to(torch.float64)

    leaf_recon = chol_leaf @ chol_leaf.mT
    res = float((leaf_diag - leaf_recon).abs().amax())

    # A_{j,0} = A_jj B_j ; stored coupling is A_0j = A_{j,0}^T.
    Aj0_recon = leaf_recon @ B
    res = max(res, float((coupling.mT - Aj0_recon).abs().amax()))

    S_recon = chol_S @ chol_S.mT
    center_recon = S_recon + (B.mT @ leaf_recon @ B).sum(dim=-3)   # sum_j A_0j B_j
    res = max(res, float((center - center_recon).abs().amax()))
    return res


def selected_inverse_residual_star(
    st: BlockStar, G_center: Tensor, G_leaf: Tensor, G_cross: Tensor
) -> float:
    """Max abs residual of the three on-pattern ``A G`` equations for a star.

    Only equations whose ``G`` blocks are *all* on the selected pattern are checked
    (the off-pattern leaf--leaf inverse blocks are nonzero but never computed):

        center : A_00 G_00 + sum_j A_0j G_{j,0} = I
        cross  : A_{j,0} G_00 + A_jj G_{j,0}    = 0
        leaf   : A_{j,0} G_{0,j} + A_jj G_jj    = I
    """
    center = st.center.to(torch.float64)
    leaf_diag = st.leaf_diag.to(torch.float64)
    coupling = st.coupling.to(torch.float64)               # A_0j
    G_center = G_center.to(torch.float64)
    G_leaf = G_leaf.to(torch.float64)
    G_cross = G_cross.to(torch.float64)                    # G_{j,0}
    b = st.block_size
    eye = torch.eye(b, dtype=torch.float64, device=center.device)

    # center equation (sum over leaves of A_0j G_{j,0}).
    center_eq = center @ G_center + (coupling @ G_cross).sum(dim=-3) - eye
    res = float(center_eq.abs().amax())

    G_center_K = G_center.unsqueeze(-3)
    # cross equation, batched over leaves: A_{j,0} = coupling^T.
    cross_eq = coupling.mT @ G_center_K + leaf_diag @ G_cross
    res = max(res, float(cross_eq.abs().amax()))

    # leaf equation, batched over leaves: G_{0,j} = G_cross^T.
    leaf_eq = coupling.mT @ G_cross.mT + leaf_diag @ G_leaf - eye
    res = max(res, float(leaf_eq.abs().amax()))
    return res


# --------------------------------------------------------------------------- #
# General tree metrics.
# --------------------------------------------------------------------------- #
def _selected_from_dense_tree(inv: Tensor, parent: list[int], b: int) -> tuple[Tensor, Tensor]:
    """Extract on-pattern blocks of a dense tree inverse (node diag, parent-child edge)."""
    n = len(parent)
    G_diag = inv.new_empty((*inv.shape[:-2], n, b, b))
    G_edge = inv.new_zeros((*inv.shape[:-2], n, b, b))     # root slot stays zero
    for v in range(n):
        rv = slice(v * b, (v + 1) * b)
        G_diag[..., v, :, :] = inv[..., rv, rv]
        pv = parent[v]
        if pv != -1:
            rp = slice(pv * b, (pv + 1) * b)
            G_edge[..., v, :, :] = inv[..., rp, rv]        # G_{p(v),v}
    return G_diag, G_edge


def dense_oracle_tree(bt: BlockTree) -> tuple[Tensor, Tensor]:
    """Selected blocks of ``A^{-1}`` via a dense fp64 Cholesky solve (the oracle)."""
    dense = bt.to_dense().to(torch.float64)
    eye = torch.eye(dense.shape[-1], dtype=dense.dtype, device=dense.device).expand_as(dense)
    chol = torch.linalg.cholesky(dense)
    inv = torch.cholesky_solve(eye, chol)
    inv = 0.5 * (inv + inv.mT)
    return _selected_from_dense_tree(inv, bt.parent.tolist(), bt.block_size)


def forward_error_tree(
    G_diag: Tensor, G_edge: Tensor, bt: BlockTree, *, eps: float = 1e-30
) -> ForwardError:
    """Forward error of computed tree selected blocks vs the dense oracle.

    The root edge slot is excluded (it is identically zero in both the computed and
    reference tensors and carries no information).
    """
    ref_diag, ref_edge = dense_oracle_tree(bt)
    g_diag = G_diag.to(torch.float64)
    g_edge = G_edge.to(torch.float64)

    # Mask selecting the non-root edge slots (root edge is identically zero).
    root = bt.root
    keep = [v for v in range(bt.num_nodes) if v != root]
    idx = torch.tensor(keep, dtype=torch.long, device=g_edge.device)
    g_edge_nr = g_edge.index_select(-3, idx)
    ref_edge_nr = ref_edge.index_select(-3, idx)

    num_sq = (g_diag - ref_diag).pow(2).sum() + (g_edge_nr - ref_edge_nr).pow(2).sum()
    den_sq = ref_diag.pow(2).sum() + ref_edge_nr.pow(2).sum()
    normwise = float(num_sq.sqrt() / den_sq.sqrt().clamp_min(eps))

    comp = max(
        float(((g_diag - ref_diag).abs() / (ref_diag.abs() + eps)).amax()),
        float(((g_edge_nr - ref_edge_nr).abs() / (ref_edge_nr.abs() + eps)).amax())
        if g_edge_nr.numel() else 0.0,
    )

    worst = 0.0
    for g, ref in ((g_diag, ref_diag), (g_edge_nr, ref_edge_nr)):
        if not g.numel():
            continue
        num = (g - ref).flatten(-2).norm(dim=-1)
        den = ref.flatten(-2).norm(dim=-1).clamp_min(eps)
        worst = max(worst, float((num / den).amax()))
    return ForwardError(normwise=normwise, componentwise=comp, worst_block=worst)


def factorization_residual_tree(bt: BlockTree, factors: TreeFactors) -> float:
    """Max abs residual of ``A`` reconstructed from the tree factors ``(chol_D, ell)``.

    Checks ``A_vv = D_v + sum_{c in ch(v)} ell_c D_c ell_c^T`` and the edge
    ``A_{p(v),v} = ell_v D_v``. Independent of the distribute pass.
    """
    chol_D = factors.chol_D.to(torch.float64)
    ell = factors.ell.to(torch.float64)
    diag = bt.diag.to(torch.float64)
    edge = bt.edge.to(torch.float64)
    root, children, _ = tree_orders(bt.parent)

    D = chol_D @ chol_D.mT                                  # reconstruct pivots D_v

    res = 0.0
    for v in range(bt.num_nodes):
        recon = D[..., v, :, :].clone()
        for c in children[v]:
            ell_c = ell[..., c, :, :]
            recon = recon + ell_c @ D[..., c, :, :] @ ell_c.mT
        res = max(res, float((diag[..., v, :, :] - recon).abs().amax()))
        if v != root:
            recon_e = ell[..., v, :, :] @ D[..., v, :, :]   # A_{p(v),v} = ell_v D_v
            res = max(res, float((edge[..., v, :, :] - recon_e).abs().amax()))
    return res


def selected_inverse_residual_tree(bt: BlockTree, G_diag: Tensor, G_edge: Tensor) -> float:
    """Max abs residual of the diagonal block-rows of ``A G = I`` for a tree.

    For block-tree ``A`` the ``v``-th block-row restricted to the diagonal block uses
    only on-pattern blocks (parent and children of ``v``):

        A_vv G_vv  +  A_{v,p(v)} G_{p(v),v}  +  sum_{c in ch(v)} A_{v,c} G_{c,v}  =  I,

    where ``A_{v,p(v)} = edge[v]^T``, ``G_{p(v),v} = G_edge[v]``, ``A_{v,c} = edge[c]``,
    and ``G_{c,v} = G_edge[c]^T``.
    """
    diag = bt.diag.to(torch.float64)
    edge = bt.edge.to(torch.float64)
    G_diag = G_diag.to(torch.float64)
    G_edge = G_edge.to(torch.float64)
    root, children, _ = tree_orders(bt.parent)
    b = bt.block_size
    eye = torch.eye(b, dtype=torch.float64, device=diag.device)

    res = 0.0
    for v in range(bt.num_nodes):
        acc = diag[..., v, :, :] @ G_diag[..., v, :, :]
        if v != root:
            # A_{v,p(v)} G_{p(v),v} = edge[v]^T @ G_edge[v]
            acc = acc + edge[..., v, :, :].mT @ G_edge[..., v, :, :]
        for c in children[v]:
            # A_{v,c} G_{c,v} = edge[c] @ G_edge[c]^T
            acc = acc + edge[..., c, :, :] @ G_edge[..., c, :, :].mT
        res = max(res, float((acc - eye).abs().amax()))
    return res


# --------------------------------------------------------------------------- #
# Junction-tree (general sparse SPD) metrics, scored on the kernel's *filled* pattern.
# --------------------------------------------------------------------------- #
def _selected_from_dense_junction(
    inv: Tensor, n: int, S_index: Tensor, b: int
) -> tuple[Tensor, Tensor]:
    """On-pattern blocks of a dense inverse for the junction kernel's filled pattern ``S``.

    Returns ``(G_diag [..., n, b, b], G_lower [..., m_S, b, b])`` with ``G_lower[k] =
    inv[hi_k, lo_k]`` for ``S_index = [[hi], [lo]]`` (``hi > lo`` by elimination position),
    matching the layout :func:`gabp_sparse_inv.junction.selected_inverse_junction` returns.
    """
    G_diag = inv.new_empty((*inv.shape[:-2], n, b, b))
    for v in range(n):
        rv = slice(v * b, (v + 1) * b)
        G_diag[..., v, :, :] = inv[..., rv, rv]
    m = int(S_index.shape[1]) if S_index.numel() else 0
    G_lower = inv.new_zeros((*inv.shape[:-2], m, b, b))
    hi = S_index[0].tolist() if m else []
    lo = S_index[1].tolist() if m else []
    for k in range(m):
        rh = slice(hi[k] * b, (hi[k] + 1) * b)
        rl = slice(lo[k] * b, (lo[k] + 1) * b)
        G_lower[..., k, :, :] = inv[..., rh, rl]
    return G_diag, G_lower


def forward_error_junction(
    G_diag: Tensor, S_index: Tensor, G_lower: Tensor, oracle64: Tensor, *, eps: float = 1e-30
) -> ForwardError:
    """Forward error of junction selected blocks vs a dense fp64 oracle, on the kernel's ``S``.

    ``oracle64`` is the dense ``A^{-1}`` in fp64. Both the kernel's blocks and any dense
    baseline are projected onto the *same* filled pattern ``S_index`` with
    :func:`_selected_from_dense_junction`, so they are scored on identical entries.
    """
    n = G_diag.shape[-3]
    b = G_diag.shape[-1]
    ref_diag, ref_lower = _selected_from_dense_junction(oracle64, n, S_index, b)
    g_diag = G_diag.to(torch.float64)
    g_lower = G_lower.to(torch.float64)

    num_sq = (g_diag - ref_diag).pow(2).sum() + (g_lower - ref_lower).pow(2).sum()
    den_sq = ref_diag.pow(2).sum() + ref_lower.pow(2).sum()
    normwise = float((num_sq.sqrt() / den_sq.sqrt().clamp_min(eps)))

    comp = max(
        float(((g_diag - ref_diag).abs() / (ref_diag.abs() + eps)).amax()),
        float(((g_lower - ref_lower).abs() / (ref_lower.abs() + eps)).amax()) if g_lower.numel() else 0.0,
    )

    bd_num = (g_diag - ref_diag).flatten(-2).norm(dim=-1)
    bd_den = ref_diag.flatten(-2).norm(dim=-1).clamp_min(eps)
    worst = float((bd_num / bd_den).amax())
    if g_lower.numel():
        bl_num = (g_lower - ref_lower).flatten(-2).norm(dim=-1)
        bl_den = ref_lower.flatten(-2).norm(dim=-1).clamp_min(eps)
        worst = max(worst, float((bl_num / bl_den).amax()))

    return ForwardError(normwise=normwise, componentwise=comp, worst_block=worst)


def output_dynamic_range(
    inv64: Tensor, on_pairs: list[tuple[int, int]], n: int, b: int, *, eps: float = 1e-30
) -> float:
    """Output dynamic range ``max_off |A^{-1}| / max_on |A^{-1}|`` (block-max, fp64 oracle).

    ``on_pairs`` lists the on-pattern block index pairs ``(i, j)`` (diagonal included or not;
    the diagonal is always treated as on-pattern). Both orientations are marked on. The ratio
    measures how much inverse mass lives *off* the kernel's returned pattern relative to *on*
    it. It was conjectured to predict the kernel's precision edge; Thread D / T3 **refuted**
    that (it does not predict the advantage -- see ``bench/precision.py``), so this is kept as
    a descriptive diagnostic only. ``inv64`` must be a single (unbatched) ``[n*b, n*b]`` dense
    inverse.
    """
    on = torch.zeros(n, n, dtype=torch.bool, device=inv64.device)
    on.fill_diagonal_(True)
    for i, j in on_pairs:
        on[i, j] = True
        on[j, i] = True
    blk = inv64.abs().reshape(n, b, n, b).amax(dim=(1, 3))     # [n, n] block max-abs
    max_on = blk[on].amax().clamp_min(eps)
    off = blk[~on]
    max_off = off.amax() if off.numel() else torch.zeros((), dtype=blk.dtype, device=blk.device)
    return float(max_off / max_on)
