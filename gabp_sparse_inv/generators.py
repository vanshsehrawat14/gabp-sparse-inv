"""Seeded generators for supported SPD block test matrices.

For chains, a base SPD matrix is built as ``A0 = M M^T`` with a
*block-bidiagonal* ``M`` (diagonal + one sub-diagonal of blocks), which keeps
``A0`` on the block-tridiagonal pattern. The returned matrix is
``A = A0 + c * I``: diagonal loading. Because

    kappa(A0 + cI) = (lambda_max + c) / (lambda_min + c)

is strictly monotone *decreasing* in ``c >= 0``, the ``diag_load`` knob gives a
deterministic condition-number control for a fixed seed. The star generator uses
the same ``A_base + cI`` idea while preserving the arrowhead pattern.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .layout import BlockBidiag, BlockSparseSym, BlockStar, BlockTree, BlockTridiag, tree_orders

__all__ = [
    "random_spd_chain",
    "random_spd_star",
    "random_spd_tree",
    "random_nonsym_bidiag",
    "random_spd_graph",
    "random_spd_laplacian",
    "grid_edges",
    "condition_number",
]


def random_spd_chain(
    num_blocks: int,
    block_size: int,
    *,
    seed: int = 0,
    diag_load: float = 1.0,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockTridiag:
    """Generate a seeded SPD block-tridiagonal matrix ``A = M M^T + diag_load * I``.

    Parameters
    ----------
    num_blocks, block_size:
        Chain length ``L`` and block dimension ``b``.
    seed:
        Seed for a private ``torch.Generator`` (does not touch global RNG state).
    diag_load:
        Diagonal-loading constant ``c >= 0``. Larger ``c`` lowers the condition number.
    batch_shape:
        Optional leading batch dimensions.
    dtype, device:
        Output dtype/device.

    Returns
    -------
    BlockTridiag with ``diag`` ``[*batch, L, b, b]`` and ``lower`` ``[*batch, L-1, b, b]``.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1; got {num_blocks}")
    if diag_load < 0:
        raise ValueError(f"diag_load must be >= 0; got {diag_load}")

    L, b = num_blocks, block_size
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        # Generate on CPU (Generator is CPU) then move; keeps results device-independent.
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)

    # Block-bidiagonal M: diagonal blocks M_diag, sub-diagonal blocks M_sub.
    # Scale the diagonal so M_ii is comfortably full rank.
    M_diag = randn(*batch_shape, L, b, b)
    M_sub = randn(*batch_shape, max(L - 1, 0), b, b) if L > 1 else torch.empty(
        (*batch_shape, 0, b, b), dtype=dtype, device=device
    )

    eye = torch.eye(b, dtype=dtype, device=device)
    diag = M_diag @ M_diag.mT + diag_load * eye
    if L > 1:
        # A_ii += M_{i,i-1} M_{i,i-1}^T for i >= 1 (0-based: blocks 1..L-1).
        diag[..., 1:, :, :] = diag[..., 1:, :, :] + M_sub @ M_sub.mT
        # A_{i+1,i} = M_{i+1,i} M_{i,i}^T  (sub-diagonal of A0).
        lower = M_sub @ M_diag[..., :-1, :, :].mT
    else:
        lower = torch.empty((*batch_shape, 0, b, b), dtype=dtype, device=device)

    # Numerically symmetrize the diagonal blocks (kills round-off asymmetry).
    diag = 0.5 * (diag + diag.mT)
    return BlockTridiag(diag=diag, lower=lower)


def random_spd_star(
    num_leaves: int,
    block_size: int,
    *,
    seed: int = 0,
    diag_load: float = 1.0,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockStar:
    """Generate a seeded SPD block-arrowhead (star) matrix ``A = A_base + diag_load * I``.

    A fixed, ``diag_load``-independent PSD arrowhead ``A_base`` is built so that the
    pattern (zero leaf--leaf blocks) is preserved and the condition number has a
    deterministic knob, exactly as for :func:`random_spd_chain`:

    * leaf base blocks ``R_j = L_j L_j^T / b + I`` are SPD and well-conditioned
      (the ``+ I`` bounds ``R_j^{-1}`` so the center stays well-scaled);
    * with random couplings ``W_j = A_0j`` and ``Q_0`` SPD, the center base is
      ``A_00^base = sum_j W_j R_j^{-1} W_j^T + Q_0 Q_0^T``.

    The Schur complement of the leaf block-diagonal is then exactly ``Q_0 Q_0^T``,
    so ``A_base`` is PSD on the arrowhead pattern. Returning ``A = A_base +
    diag_load * I`` keeps the pattern and gives

        kappa(A_base + cI) = (lambda_max + c) / (lambda_min + c),

    strictly decreasing in ``c = diag_load >= 0``.

    Parameters
    ----------
    num_leaves, block_size:
        Number of leaves ``K`` and block dimension ``b`` (center and leaves share ``b``).
    seed:
        Seed for a private ``torch.Generator`` (does not touch global RNG state).
    diag_load:
        Diagonal-loading constant ``c >= 0``. Larger ``c`` lowers the condition number.
    batch_shape, dtype, device:
        Optional leading batch dimensions and output dtype/device.

    Returns
    -------
    BlockStar with ``center`` ``[*batch, b, b]`` and ``leaf_diag`` / ``coupling``
    ``[*batch, K, b, b]``.
    """
    if num_leaves < 1:
        raise ValueError(f"num_leaves must be >= 1; got {num_leaves}")
    if diag_load < 0:
        raise ValueError(f"diag_load must be >= 0; got {diag_load}")

    K, b = num_leaves, block_size
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        # Build in fp64 on CPU (the Generator is CPU); move to dtype/device at the end.
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    eye = torch.eye(b, dtype=torch.float64)

    # Leaf base blocks R_j = L_j L_j^T / b + I (SPD, well-conditioned) and
    # couplings W_j = A_0j.
    L_leaf = randn(*batch_shape, K, b, b)
    R = L_leaf @ L_leaf.mT / b + eye
    W = randn(*batch_shape, K, b, b)
    Q0 = randn(*batch_shape, b, b)

    # Center base = sum_j W_j R_j^{-1} W_j^T + Q0 Q0^T  (Schur complement = Q0 Q0^T).
    chol_R = torch.linalg.cholesky(R)
    Rinv_WT = torch.cholesky_solve(W.mT, chol_R)           # R_j^{-1} A_{j,0}
    schur = (W @ Rinv_WT).sum(dim=-3)
    center_base = schur + Q0 @ Q0.mT

    # A = A_base + diag_load * I  (loading touches only the diagonal blocks).
    center = 0.5 * (center_base + center_base.mT) + diag_load * eye
    leaf_diag = R + diag_load * eye
    leaf_diag = 0.5 * (leaf_diag + leaf_diag.mT)
    coupling = W

    return BlockStar(
        center=center.to(dtype=dtype, device=device),
        leaf_diag=leaf_diag.to(dtype=dtype, device=device),
        coupling=coupling.to(dtype=dtype, device=device),
    )


def _build_parent(num_nodes: int, kind: str, gen: torch.Generator) -> list[int]:
    """Construct a parent array for a tree of ``num_nodes`` nodes (root = node 0)."""
    n = num_nodes
    if kind == "path":
        return [-1] + [v - 1 for v in range(1, n)]
    if kind == "star":
        return [-1] + [0] * (n - 1)
    if kind == "balanced":            # complete binary tree: parent = (v-1)//2
        return [-1] + [(v - 1) // 2 for v in range(1, n)]
    if kind == "random":
        # Recursive random tree: parent[v] uniform in [0, v-1]. Always acyclic.
        parent = [-1]
        for v in range(1, n):
            parent.append(int(torch.randint(0, v, (1,), generator=gen).item()))
        return parent
    raise ValueError(f"unknown tree kind {kind!r}; known: path, star, balanced, random")


def random_spd_tree(
    num_nodes: int,
    block_size: int,
    *,
    seed: int = 0,
    diag_load: float = 1.0,
    kind: str = "random",
    parent: "list[int] | Tensor | None" = None,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockTree:
    """Generate a seeded SPD block-tree matrix ``A = A_base + diag_load * I``.

    The tree topology is either an explicit ``parent`` array or one of
    ``kind in {"random", "path", "star", "balanced"}`` (root = node 0). ``A_base`` is
    built directly from explicit tree ``L D L^T`` factors so the arrowhead/tridiagonal
    *pattern is exact* and SPD is guaranteed, mirroring :func:`random_spd_chain` /
    :func:`random_spd_star`:

    * node base pivots ``D_v = M_v M_v^T / b + I`` are SPD and well-conditioned;
    * for each non-root ``v`` a random factor ``ell_v`` defines the edge
      ``edge[v] = A_{p(v),v} = ell_v D_v`` and the parent contribution
      ``ell_v D_v ell_v^T`` added to ``A_{p(v),p(v)}``.

    Then ``A_vv^base = D_v + sum_{c in ch(v)} ell_c D_c ell_c^T`` is exactly the
    block ``L D L^T`` reconstruction, PSD on the tree pattern, and
    ``A = A_base + diag_load * I`` keeps the pattern with the same monotone
    ``kappa(A_base + cI)`` knob as the chain/star generators.

    Parameters
    ----------
    num_nodes, block_size:
        Number of node blocks ``n`` and block dimension ``b``.
    seed, diag_load, batch_shape, dtype, device:
        As in :func:`random_spd_chain`.
    kind:
        Topology family when ``parent`` is not given.
    parent:
        Explicit parent array (overrides ``kind``); ``parent[root] = -1``.

    Returns
    -------
    BlockTree with ``diag``/``edge`` ``[*batch, n, b, b]`` and the ``parent`` array.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    if diag_load < 0:
        raise ValueError(f"diag_load must be >= 0; got {diag_load}")

    n, b = num_nodes, block_size
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    if parent is None:
        parent_list = _build_parent(n, kind, gen)
    else:
        parent_list = (parent.tolist() if isinstance(parent, Tensor) else list(parent))
    # Validate topology (raises) and get the leaves-to-root order for accumulation.
    root, _children, collect_order = tree_orders(parent_list)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    eye = torch.eye(b, dtype=torch.float64)

    # Node base pivots D_v (SPD, well-conditioned) and per-edge factors ell_v.
    M = randn(*batch_shape, n, b, b)
    D = M @ M.mT / b + eye                                   # [*batch, n, b, b]
    ell = randn(*batch_shape, n, b, b)                       # ell[root] unused

    # edge[v] = A_{p(v),v} = ell_v D_v ; diag base accumulates children Schur terms.
    edge = torch.zeros((*batch_shape, n, b, b), dtype=torch.float64)
    diag = D.clone()
    for v in collect_order:
        if v == root:
            continue
        Dv = D[..., v, :, :]
        ell_v = ell[..., v, :, :]
        edge[..., v, :, :] = ell_v @ Dv
        # A_{p,p}^base += ell_v D_v ell_v^T
        pv = parent_list[v]
        diag[..., pv, :, :] = diag[..., pv, :, :] + ell_v @ Dv @ ell_v.mT

    diag = 0.5 * (diag + diag.mT) + diag_load * eye
    return BlockTree(
        diag=diag.to(dtype=dtype, device=device),
        edge=edge.to(dtype=dtype, device=device),
        parent=torch.tensor(parent_list, dtype=torch.long),
    )


def random_nonsym_bidiag(
    num_blocks: int,
    block_size: int,
    *,
    seed: int = 0,
    diag_load: float = 1.0,
    off_scale: float = 0.5,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockBidiag:
    """Generate a seeded general (non-symmetric) block lower-bidiagonal matrix ``M``.

    Blocks are ``M_ii = off_scale * N(0,1) + (diag_load + 1) I`` and
    ``M_{i+1,i} = off_scale * N(0,1)`` (general, *not* symmetric). Larger ``diag_load``
    makes the diagonal blocks dominate, lowering the condition number -- a monotone knob
    in spirit, mirroring :func:`random_spd_chain`. ``M`` is invertible whenever every
    diagonal block is; the ``+ I`` shift keeps the diagonal blocks well away from
    singular at the default ``diag_load``.

    Parameters
    ----------
    num_blocks, block_size:
        Chain length ``n`` and block dimension ``b``.
    seed:
        Seed for a private ``torch.Generator`` (does not touch global RNG state).
    diag_load:
        Diagonal-dominance constant ``c >= 0``. Larger ``c`` lowers the condition number.
    off_scale:
        Scale of the random (off-identity) part of every block.
    batch_shape, dtype, device:
        Optional leading batch dimensions and output dtype/device.

    Returns
    -------
    BlockBidiag with ``diag`` ``[*batch, n, b, b]`` and ``lower`` ``[*batch, n-1, b, b]``.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1; got {num_blocks}")
    if diag_load < 0:
        raise ValueError(f"diag_load must be >= 0; got {diag_load}")

    n, b = num_blocks, block_size
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    eye = torch.eye(b, dtype=torch.float64)
    diag = off_scale * randn(*batch_shape, n, b, b) + (diag_load + 1.0) * eye
    if n > 1:
        lower = off_scale * randn(*batch_shape, n - 1, b, b)
    else:
        lower = torch.empty((*batch_shape, 0, b, b), dtype=torch.float64)

    return BlockBidiag(
        diag=diag.to(dtype=dtype, device=device),
        lower=lower.to(dtype=dtype, device=device),
    )


def grid_edges(rows: int, cols: int) -> Tensor:
    """Edge index ``[2, m]`` (``i > j``) of a 2-D 4-neighbour lattice.

    Node ``(r, c)`` has id ``r * cols + c`` and connects to its right and down
    neighbours. This is the loopy graph behind the grid maze (treewidth ``~min(rows,
    cols)``), the first structure that genuinely needs the junction-tree selected
    inverse rather than a tree.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1; got {rows}x{cols}")
    pairs: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                w = r * cols + (c + 1)
                pairs.append((max(v, w), min(v, w)))
            if r + 1 < rows:
                w = (r + 1) * cols + c
                pairs.append((max(v, w), min(v, w)))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(list(zip(*pairs)), dtype=torch.long)


def random_spd_graph(
    num_nodes: int,
    edges: Tensor | list[tuple[int, int]],
    block_size: int,
    *,
    seed: int = 0,
    diag_load: float = 1.0,
    off_scale: float = 0.5,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockSparseSym:
    """Generate a seeded SPD block matrix on an arbitrary (general) graph pattern.

    Off-diagonal blocks are random; each diagonal block is built **block diagonally
    dominant** so the matrix is provably SPD on the *exact* input pattern (no fill is
    baked in -- fill appears only when the kernel eliminates). Concretely, for each
    node ``i`` with incident off-diagonal blocks summing to ``s_i`` in Frobenius norm,

        A_ii = G_i G_i^T / b + (s_i + diag_load + 1) I,

    so ``lambda_min(A_ii) > s_i >= sum_j ||A_ij||_2`` -- the Feingold-Varga block
    diagonal-dominance condition for positive definiteness. Larger ``diag_load`` lifts
    every ``lambda_min`` and lowers the condition number (a monotone knob, as in
    :func:`random_spd_chain`).

    Parameters
    ----------
    num_nodes, block_size:
        Number of node blocks ``n`` and block dimension ``b``.
    edges:
        ``[2, m]`` edge index or a list of ``(i, j)`` pairs (any orientation; stored as
        ``i > j`` and de-duplicated). Self-loops are rejected.
    seed, diag_load, off_scale, batch_shape, dtype, device:
        As in the other generators (``off_scale`` scales the random off-diagonals).

    Returns
    -------
    BlockSparseSym with ``diag`` ``[*batch, n, b, b]`` and ``edge_val`` ``[*batch, m, b, b]``.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    if diag_load < 0:
        raise ValueError(f"diag_load must be >= 0; got {diag_load}")

    n, b = num_nodes, block_size
    if isinstance(edges, Tensor):
        ei = edges.to(device="cpu", dtype=torch.long)
        pair_list = [(int(a), int(c)) for a, c in zip(ei[0].tolist(), ei[1].tolist())]
    else:
        pair_list = [(int(a), int(c)) for a, c in edges]
    # Canonicalize to i > j and de-duplicate (deterministic order).
    canon: list[tuple[int, int]] = []
    seen = set()
    for a, c in pair_list:
        if a == c:
            raise ValueError(f"self-loop at node {a}")
        key = (max(a, c), min(a, c))
        if key not in seen:
            seen.add(key)
            canon.append(key)
    m = len(canon)
    edge_index = (
        torch.tensor(list(zip(*canon)), dtype=torch.long) if m else torch.zeros((2, 0), dtype=torch.long)
    )

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    eye = torch.eye(b, dtype=torch.float64)

    edge_val = off_scale * randn(*batch_shape, m, b, b) if m else torch.empty(
        (*batch_shape, 0, b, b), dtype=torch.float64
    )

    # s_i = sum of Frobenius norms of node i's incident off-diagonal blocks.
    s = torch.zeros((*batch_shape, n), dtype=torch.float64)
    if m:
        fro = torch.linalg.matrix_norm(edge_val, ord="fro")            # [*batch, m]
        for k, (i, j) in enumerate(canon):
            s[..., i] = s[..., i] + fro[..., k]
            s[..., j] = s[..., j] + fro[..., k]

    G = randn(*batch_shape, n, b, b)
    base = G @ G.mT / b
    load = (s + diag_load + 1.0)[..., None, None]                       # [*batch, n, 1, 1]
    diag = base + load * eye
    diag = 0.5 * (diag + diag.mT)

    return BlockSparseSym(
        diag=diag.to(dtype=dtype, device=device),
        edge_index=edge_index,
        edge_val=edge_val.to(dtype=dtype, device=device),
    )


def random_spd_laplacian(
    num_nodes: int,
    edges: Tensor | list[tuple[int, int]],
    block_size: int,
    *,
    eps: float = 1e-2,
    seed: int = 0,
    weight_floor: float = 0.1,
    batch_shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BlockSparseSym:
    """Block graph-Laplacian SPD matrix ``A = L(W) + eps*I`` on a given pattern.

    Well-scaled, high-conditioning generator: small ``eps`` -> large ``kappa``
    (``~ lambda_max(L)/eps``), ``O(1)`` entries, a graph-Green's-function inverse. This
    is the graph-Laplacian / maze regime, complementary to :func:`random_spd_graph`
    (block diagonally dominant, ``kappa`` capped low): here ``eps`` is the monotone
    condition-number knob in the *other* direction -- smaller ``eps`` raises ``kappa``.

    Each edge ``(i, j)`` gets an SPD weight ``W_ij = V V^T / b + weight_floor * I``
    (``O(1)``, SPD); then ``A_ii = eps*I + sum_{j~i} W_ij`` and ``A_ij = -W_ij``. The
    block Laplacian ``L(W)`` is PSD; ``+ eps*I`` makes ``A`` SPD for ``eps > 0``.

    Parameters
    ----------
    num_nodes, block_size:
        Number of node blocks ``n`` and block dimension ``b``.
    edges:
        ``[2, m]`` edge index or a list of ``(i, j)`` pairs (any orientation; stored as
        ``i > j`` and de-duplicated). Self-loops are rejected.
    eps:
        Diagonal lift ``eps > 0`` (the monotone condition-number knob: smaller -> larger
        ``kappa``).
    seed:
        Seed for a private ``torch.Generator`` (does not touch global RNG state).
    weight_floor:
        Additive ``weight_floor * I`` keeping each edge weight strictly SPD.
    batch_shape, dtype, device:
        Optional leading batch dimensions and output dtype/device.

    Returns
    -------
    BlockSparseSym with ``diag`` ``[*batch, n, b, b]`` and ``edge_val`` ``[*batch, m, b, b]``.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0 for SPD; got {eps}")
    n, b = num_nodes, block_size
    if isinstance(edges, Tensor):
        ei = edges.to(device="cpu", dtype=torch.long)
        pair_list = [(int(a), int(c)) for a, c in zip(ei[0].tolist(), ei[1].tolist())]
    else:
        pair_list = [(int(a), int(c)) for a, c in edges]
    canon: list[tuple[int, int]] = []
    seen = set()
    for a, c in pair_list:
        if a == c:
            raise ValueError(f"self-loop at node {a}")
        key = (max(a, c), min(a, c))
        if key not in seen:
            seen.add(key)
            canon.append(key)
    m = len(canon)
    edge_index = (
        torch.tensor(list(zip(*canon)), dtype=torch.long) if m else torch.zeros((2, 0), dtype=torch.long)
    )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    eye = torch.eye(b, dtype=torch.float64)
    diag = (eps * eye).expand(*batch_shape, n, b, b).clone()
    if m:
        V = torch.randn(*batch_shape, m, b, b, generator=gen, dtype=torch.float64)
        W = V @ V.mT / b + weight_floor * eye                       # [*batch, m, b, b] SPD
        i_idx = torch.tensor([i for i, _ in canon], dtype=torch.long)
        j_idx = torch.tensor([j for _, j in canon], dtype=torch.long)
        diag.index_add_(-3, i_idx, W)
        diag.index_add_(-3, j_idx, W)
        edge_val = -W
    else:
        edge_val = torch.empty((*batch_shape, 0, b, b), dtype=torch.float64)
    diag = 0.5 * (diag + diag.mT)
    return BlockSparseSym(
        diag=diag.to(dtype=dtype, device=device),
        edge_index=edge_index,
        edge_val=edge_val.to(dtype=dtype, device=device),
    )


def condition_number(bt: BlockTridiag | BlockStar | BlockTree | BlockSparseSym) -> Tensor:
    """2-norm condition number of the dense matrix (test/diagnostic use; dense).

    Accepts any container exposing ``to_dense`` (chain, star, tree, or general sparse).
    Assumes a *symmetric* matrix (uses ``eigvalsh``); for the non-symmetric
    :class:`~gabp_sparse_inv.layout.BlockBidiag` use ``torch.linalg.cond`` directly.
    """
    dense = bt.to_dense().to(torch.float64)
    eig = torch.linalg.eigvalsh(dense)
    return eig[..., -1] / eig[..., 0]
