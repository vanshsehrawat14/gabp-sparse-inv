"""Storage containers and dense bridges for supported block layouts.

The :class:`BlockTridiag` and :class:`BlockStar` containers hold symmetric sparse
block matrices in structured storage and never materialize the dense ``N x N``
matrix on the hot path. ``to_dense`` / ``from_dense`` exist only for tests and
the dense oracle.

Tensors carry optional leading batch dimensions so a single call can process a
stack of independent problems (seeds / devices).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

__all__ = [
    "BlockTridiag",
    "BlockBidiag",
    "BlockStar",
    "BlockTree",
    "BlockSparseSym",
    "tree_orders",
    "tree_levels",
]


def _as_parent_tensor(parent: Tensor | Sequence[int]) -> Tensor:
    """Normalize a parent array to a 1-D CPU ``long`` tensor (topology is not batched)."""
    if isinstance(parent, Tensor):
        p = parent.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    else:
        p = torch.tensor(list(parent), dtype=torch.long, device="cpu")
    return p


def tree_orders(parent: Tensor | Sequence[int]) -> tuple[int, list[list[int]], list[int]]:
    """Validate a ``parent`` array and return ``(root, children, collect_order)``.

    ``parent[v]`` is the parent index of node ``v``; the unique root has
    ``parent[root] == -1``. ``collect_order`` lists every node *children-before-parents*
    (a leaves-to-root / Kahn leaf-peeling order); ``reversed(collect_order)`` is the
    parents-before-children distribute order. Raises ``ValueError`` if ``parent`` is
    not a single rooted tree (wrong number of roots, out-of-range or self-parent
    entries, or a cycle / disconnection, detected when peeling fails to reach all
    nodes).
    """
    p = _as_parent_tensor(parent)
    n = int(p.numel())
    if n < 1:
        raise ValueError(f"need at least one node; got n={n}")
    plist = p.tolist()

    root = -1
    nchild = [0] * n
    for v in range(n):
        pv = plist[v]
        if pv == -1:
            if root != -1:
                raise ValueError(f"multiple roots: nodes {root} and {v} both have parent -1")
            root = v
            continue
        if pv < 0 or pv >= n:
            raise ValueError(f"parent[{v}]={pv} out of range [0,{n - 1}] (or -1 for root)")
        if pv == v:
            raise ValueError(f"node {v} is its own parent")
        nchild[pv] += 1
    if root == -1:
        raise ValueError("no root: every node has a parent (parent array contains a cycle)")

    children: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        if plist[v] != -1:
            children[plist[v]].append(v)

    # Kahn leaf-peeling: emit a node once all its children are emitted.
    pending = nchild[:]
    queue = deque(v for v in range(n) if pending[v] == 0)   # leaves first
    collect_order: list[int] = []
    while queue:
        v = queue.popleft()
        collect_order.append(v)
        pv = plist[v]
        if pv != -1:
            pending[pv] -= 1
            if pending[pv] == 0:
                queue.append(pv)
    if len(collect_order) != n:
        raise ValueError(
            "parent array is not a single acyclic tree "
            f"(peeled {len(collect_order)} of {n} nodes; contains a cycle or is disconnected)"
        )
    return root, children, collect_order


def tree_levels(parent: Tensor | Sequence[int]) -> list[list[int]]:
    """Group nodes into height antichains for level-set (batched) processing.

    Returns ``levels`` where ``levels[h]`` lists every node whose *height* (longest
    edge-distance to a descendant leaf) is ``h``. Because a parent's height strictly
    exceeds each child's, every level is an antichain (mutually independent nodes), and
    processing ``levels`` in increasing ``h`` is a valid collect (children-before-
    parents) schedule; ``reversed(levels)`` is the distribute schedule. The root is the
    unique node of maximal height, alone in the last level.

    All nodes in a level can be factored with a single batched Cholesky, and their
    Schur terms scattered into parents with one ``index_add_`` -- the batching the
    sequential per-node loop defers. See :func:`gabp_sparse_inv.tree.selected_inverse_tree`
    (``batched=True``).
    """
    p = _as_parent_tensor(parent)
    _root, _children, collect_order = tree_orders(p)
    n = int(p.numel())
    plist = p.tolist()
    height = [0] * n
    for v in collect_order:                      # children before parents
        pv = plist[v]
        if pv != -1:
            height[pv] = max(height[pv], height[v] + 1)
    levels: list[list[int]] = [[] for _ in range(max(height) + 1)]
    for v in range(n):
        levels[height[v]].append(v)
    return levels


@dataclass
class BlockTridiag:
    """Symmetric block-tridiagonal matrix.

    Attributes
    ----------
    diag:
        Diagonal blocks ``A_ii`` with shape ``[..., L, b, b]`` (each block symmetric).
    lower:
        Sub-diagonal blocks ``A_{i+1,i}`` with shape ``[..., L-1, b, b]``. The
        super-diagonal is implied by symmetry: ``A_{i,i+1} = lower[..., i, :, :].mT``.
    """

    diag: Tensor
    lower: Tensor

    # -- basic shape accessors -------------------------------------------------
    @property
    def num_blocks(self) -> int:
        """Number of diagonal blocks ``L``."""
        return self.diag.shape[-3]

    @property
    def block_size(self) -> int:
        """Block dimension ``b``."""
        return self.diag.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions (everything before ``L, b, b``)."""
        return self.diag.shape[:-3]

    # -- validation ------------------------------------------------------------
    def validate(self, *, symmetry_tol: float | None = None) -> "BlockTridiag":
        """Validate shapes, dtype/device consistency, and block symmetry.

        Raises ``TypeError`` / ``ValueError`` (never bare ``assert``) so validation
        survives ``python -O``.
        """
        if not isinstance(self.diag, Tensor) or not isinstance(self.lower, Tensor):
            raise TypeError("diag and lower must be torch.Tensor instances")
        if self.diag.dtype != self.lower.dtype:
            raise TypeError(
                f"dtype mismatch: diag={self.diag.dtype}, lower={self.lower.dtype}"
            )
        if self.diag.device != self.lower.device:
            raise ValueError(
                f"device mismatch: diag={self.diag.device}, lower={self.lower.device}"
            )
        if self.diag.ndim < 3:
            raise ValueError(
                f"diag must have shape [..., L, b, b]; got ndim={self.diag.ndim}"
            )
        L, b, b2 = self.diag.shape[-3], self.diag.shape[-2], self.diag.shape[-1]
        if b != b2:
            raise ValueError(f"diag blocks must be square; got {b}x{b2}")
        if L < 1:
            raise ValueError(f"need at least one block; got L={L}")

        if L == 1:
            # lower may be an empty [..., 0, b, b] tensor.
            if self.lower.shape[-3] != 0:
                raise ValueError(
                    f"L=1 requires zero sub-diagonal blocks; got {self.lower.shape[-3]}"
                )
        else:
            expected = (*self.diag.shape[:-3], L - 1, b, b)
            if tuple(self.lower.shape) != expected:
                raise ValueError(
                    f"lower must have shape {expected}; got {tuple(self.lower.shape)}"
                )

        tol = symmetry_tol
        if tol is None:
            tol = 1e-6 if self.diag.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-10
        asym = (self.diag - self.diag.mT).abs().amax()
        scale = self.diag.abs().amax().clamp_min(1.0)
        if asym > tol * scale:
            raise ValueError(
                f"diagonal blocks are not symmetric within tol={tol} "
                f"(max asymmetry {float(asym):.3e}, scale {float(scale):.3e})"
            )
        return self

    # -- dense bridges (tests / oracle only) -----------------------------------
    def to_dense(self) -> Tensor:
        """Materialize the dense ``[..., N, N]`` matrix (``N = L*b``). Test/oracle use only."""
        L, b = self.num_blocks, self.block_size
        N = L * b
        out = self.diag.new_zeros((*self.batch_shape, N, N))
        for i in range(L):
            r = slice(i * b, (i + 1) * b)
            out[..., r, r] = self.diag[..., i, :, :]
            if i < L - 1:
                c = slice((i + 1) * b, (i + 2) * b)
                lo = self.lower[..., i, :, :]
                out[..., c, r] = lo            # A_{i+1,i}
                out[..., r, c] = lo.mT         # A_{i,i+1}
        return out

    @classmethod
    def from_dense(cls, mat: Tensor, block_size: int) -> "BlockTridiag":
        """Extract block-tridiagonal storage from a dense ``[..., N, N]`` matrix.

        Off-tridiagonal entries are *ignored* (this is a projection onto the pattern).
        Test/oracle use only.
        """
        N = mat.shape[-1]
        if mat.shape[-2] != N:
            raise ValueError(f"matrix must be square; got {tuple(mat.shape[-2:])}")
        if N % block_size != 0:
            raise ValueError(f"N={N} not divisible by block_size={block_size}")
        L = N // block_size
        b = block_size
        diag = mat.new_zeros((*mat.shape[:-2], L, b, b))
        lower = mat.new_zeros((*mat.shape[:-2], max(L - 1, 0), b, b))
        for i in range(L):
            r = slice(i * b, (i + 1) * b)
            diag[..., i, :, :] = mat[..., r, r]
            if i < L - 1:
                c = slice((i + 1) * b, (i + 2) * b)
                lower[..., i, :, :] = mat[..., c, r]
        return cls(diag=diag, lower=lower)


@dataclass
class BlockBidiag:
    """General (non-symmetric) block lower-bidiagonal matrix ``M``.

    The non-symmetric analogue of :class:`BlockTridiag`: ``M`` is block
    lower-triangular with bandwidth one (diagonal blocks ``M_ii`` and one sub-diagonal
    of blocks ``M_{i+1,i}``; the super-diagonal is **zero**, not the transpose).
    No symmetry is assumed on any block. The selected inverse of ``M`` on its own
    pattern (diagonal + first sub-diagonal) is computed by
    :func:`gabp_sparse_inv.selected_inverse_bidiag`; the off-pattern blocks of
    ``M^{-1}`` (which are nonzero) are never formed.

    Attributes
    ----------
    diag:
        Diagonal blocks ``M_ii`` with shape ``[..., n, b, b]`` (each a general square
        block).
    lower:
        Sub-diagonal blocks ``M_{i+1,i}`` with shape ``[..., n-1, b, b]``.
    """

    diag: Tensor
    lower: Tensor

    # -- basic shape accessors -------------------------------------------------
    @property
    def num_blocks(self) -> int:
        """Number of diagonal blocks ``n``."""
        return self.diag.shape[-3]

    @property
    def block_size(self) -> int:
        """Block dimension ``b``."""
        return self.diag.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions (everything before ``n, b, b``)."""
        return self.diag.shape[:-3]

    # -- validation ------------------------------------------------------------
    def validate(self) -> "BlockBidiag":
        """Validate shapes and dtype/device consistency (no symmetry requirement).

        Raises ``TypeError`` / ``ValueError`` (never bare ``assert``) so validation
        survives ``python -O``.
        """
        if not isinstance(self.diag, Tensor) or not isinstance(self.lower, Tensor):
            raise TypeError("diag and lower must be torch.Tensor instances")
        if self.diag.dtype != self.lower.dtype:
            raise TypeError(
                f"dtype mismatch: diag={self.diag.dtype}, lower={self.lower.dtype}"
            )
        if self.diag.device != self.lower.device:
            raise ValueError(
                f"device mismatch: diag={self.diag.device}, lower={self.lower.device}"
            )
        if self.diag.ndim < 3:
            raise ValueError(
                f"diag must have shape [..., n, b, b]; got ndim={self.diag.ndim}"
            )
        n, b, b2 = self.diag.shape[-3], self.diag.shape[-2], self.diag.shape[-1]
        if b != b2:
            raise ValueError(f"diag blocks must be square; got {b}x{b2}")
        if n < 1:
            raise ValueError(f"need at least one block; got n={n}")

        if n == 1:
            if self.lower.shape[-3] != 0:
                raise ValueError(
                    f"n=1 requires zero sub-diagonal blocks; got {self.lower.shape[-3]}"
                )
        else:
            expected = (*self.diag.shape[:-3], n - 1, b, b)
            if tuple(self.lower.shape) != expected:
                raise ValueError(
                    f"lower must have shape {expected}; got {tuple(self.lower.shape)}"
                )
        return self

    # -- dense bridges (tests / oracle only) -----------------------------------
    def to_dense(self) -> Tensor:
        """Materialize the dense ``[..., N, N]`` matrix (``N = n*b``). Test/oracle use only.

        Lower-bidiagonal: only the diagonal and first sub-diagonal blocks are nonzero;
        the super-diagonal is zero (this is a non-symmetric, triangular matrix).
        """
        n, b = self.num_blocks, self.block_size
        N = n * b
        out = self.diag.new_zeros((*self.batch_shape, N, N))
        for i in range(n):
            r = slice(i * b, (i + 1) * b)
            out[..., r, r] = self.diag[..., i, :, :]
            if i < n - 1:
                c = slice((i + 1) * b, (i + 2) * b)
                out[..., c, r] = self.lower[..., i, :, :]   # M_{i+1,i}
        return out

    @classmethod
    def from_dense(cls, mat: Tensor, block_size: int) -> "BlockBidiag":
        """Extract block lower-bidiagonal storage from a dense matrix (projection).

        Off-pattern entries (super-diagonal and deeper sub-diagonals) are *ignored*.
        Test/oracle use only.
        """
        N = mat.shape[-1]
        if mat.shape[-2] != N:
            raise ValueError(f"matrix must be square; got {tuple(mat.shape[-2:])}")
        if N % block_size != 0:
            raise ValueError(f"N={N} not divisible by block_size={block_size}")
        n = N // block_size
        b = block_size
        diag = mat.new_zeros((*mat.shape[:-2], n, b, b))
        lower = mat.new_zeros((*mat.shape[:-2], max(n - 1, 0), b, b))
        for i in range(n):
            r = slice(i * b, (i + 1) * b)
            diag[..., i, :, :] = mat[..., r, r]
            if i < n - 1:
                c = slice((i + 1) * b, (i + 2) * b)
                lower[..., i, :, :] = mat[..., c, r]
        return cls(diag=diag, lower=lower)


@dataclass
class BlockStar:
    """Symmetric block-arrowhead (star-graph) matrix.

    A star is the smallest *branching* tree: one center block coupled to ``K``
    leaf blocks, with no leaf--leaf coupling. The matrix is the block arrowhead

        [ center      A_01  A_02  ...  A_0K ]
        [ A_01^T   leaf_1     0   ...   0   ]
        [ A_02^T      0    leaf_2 ...   0   ]
        [   :                        \\      ]
        [ A_0K^T      0       0   ... leaf_K]

    stored in ``O(K * b**2)`` (the dense ``N x N`` matrix, ``N = (K+1)*b``, is
    materialized only by :meth:`to_dense` for tests / the oracle).

    Attributes
    ----------
    center:
        Center diagonal block ``A_00`` with shape ``[..., b, b]`` (symmetric).
    leaf_diag:
        Leaf diagonal blocks ``A_jj`` with shape ``[..., K, b, b]`` (each symmetric).
    coupling:
        Center->leaf coupling blocks ``A_0j`` with shape ``[..., K, b, b]``. The
        transpose is implied by symmetry: ``A_{j,0} = coupling[..., j, :, :].mT``.
    """

    center: Tensor
    leaf_diag: Tensor
    coupling: Tensor

    # -- basic shape accessors -------------------------------------------------
    @property
    def num_leaves(self) -> int:
        """Number of leaf blocks ``K``."""
        return self.leaf_diag.shape[-3]

    @property
    def block_size(self) -> int:
        """Block dimension ``b``."""
        return self.center.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions (everything before the block structure)."""
        return self.center.shape[:-2]

    # -- validation ------------------------------------------------------------
    def validate(self, *, symmetry_tol: float | None = None) -> "BlockStar":
        """Validate shapes, dtype/device consistency, and block symmetry.

        Raises ``TypeError`` / ``ValueError`` (never bare ``assert``) so validation
        survives ``python -O``.
        """
        tensors = (self.center, self.leaf_diag, self.coupling)
        if not all(isinstance(t, Tensor) for t in tensors):
            raise TypeError("center, leaf_diag and coupling must be torch.Tensor instances")
        if not (self.center.dtype == self.leaf_diag.dtype == self.coupling.dtype):
            raise TypeError(
                f"dtype mismatch: center={self.center.dtype}, "
                f"leaf_diag={self.leaf_diag.dtype}, coupling={self.coupling.dtype}"
            )
        if not (self.center.device == self.leaf_diag.device == self.coupling.device):
            raise ValueError(
                f"device mismatch: center={self.center.device}, "
                f"leaf_diag={self.leaf_diag.device}, coupling={self.coupling.device}"
            )
        if self.center.ndim < 2:
            raise ValueError(
                f"center must have shape [..., b, b]; got ndim={self.center.ndim}"
            )
        b, b2 = self.center.shape[-2], self.center.shape[-1]
        if b != b2:
            raise ValueError(f"center block must be square; got {b}x{b2}")
        if self.leaf_diag.ndim < 3:
            raise ValueError(
                f"leaf_diag must have shape [..., K, b, b]; got ndim={self.leaf_diag.ndim}"
            )
        batch = self.center.shape[:-2]
        K = self.leaf_diag.shape[-3]
        if K < 1:
            raise ValueError(f"need at least one leaf; got K={K}")
        expected = (*batch, K, b, b)
        if tuple(self.leaf_diag.shape) != expected:
            raise ValueError(
                f"leaf_diag must have shape {expected}; got {tuple(self.leaf_diag.shape)}"
            )
        if tuple(self.coupling.shape) != expected:
            raise ValueError(
                f"coupling must have shape {expected}; got {tuple(self.coupling.shape)}"
            )

        tol = symmetry_tol
        if tol is None:
            tol = 1e-6 if self.center.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-10
        for label, blocks in (("center", self.center), ("leaf_diag", self.leaf_diag)):
            asym = (blocks - blocks.mT).abs().amax()
            scale = blocks.abs().amax().clamp_min(1.0)
            if asym > tol * scale:
                raise ValueError(
                    f"{label} blocks are not symmetric within tol={tol} "
                    f"(max asymmetry {float(asym):.3e}, scale {float(scale):.3e})"
                )
        return self

    # -- dense bridges (tests / oracle only) -----------------------------------
    def to_dense(self) -> Tensor:
        """Materialize the dense ``[..., N, N]`` matrix (``N = (K+1)*b``). Test/oracle use only.

        Block 0 is the center; blocks ``1..K`` are the leaves.
        """
        K, b = self.num_leaves, self.block_size
        N = (K + 1) * b
        out = self.center.new_zeros((*self.batch_shape, N, N))
        c0 = slice(0, b)
        out[..., c0, c0] = self.center
        for j in range(K):
            r = slice((j + 1) * b, (j + 2) * b)
            out[..., r, r] = self.leaf_diag[..., j, :, :]
            cpl = self.coupling[..., j, :, :]          # A_0j
            out[..., c0, r] = cpl                       # center row -> leaf col
            out[..., r, c0] = cpl.mT                    # A_{j,0}
        return out

    @classmethod
    def from_dense(cls, mat: Tensor, block_size: int) -> "BlockStar":
        """Extract block-arrowhead storage from a dense ``[..., N, N]`` matrix.

        Block 0 is the center; blocks ``1..K`` are the leaves. Off-pattern
        (leaf--leaf) entries are *ignored* (projection onto the pattern).
        Test/oracle use only.
        """
        N = mat.shape[-1]
        if mat.shape[-2] != N:
            raise ValueError(f"matrix must be square; got {tuple(mat.shape[-2:])}")
        if N % block_size != 0:
            raise ValueError(f"N={N} not divisible by block_size={block_size}")
        b = block_size
        K = N // b - 1
        if K < 1:
            raise ValueError(f"need N >= 2*block_size for a star; got N={N}, b={b}")
        c0 = slice(0, b)
        center = mat[..., c0, c0].clone()
        leaf_diag = mat.new_zeros((*mat.shape[:-2], K, b, b))
        coupling = mat.new_zeros((*mat.shape[:-2], K, b, b))
        for j in range(K):
            r = slice((j + 1) * b, (j + 2) * b)
            leaf_diag[..., j, :, :] = mat[..., r, r]
            coupling[..., j, :, :] = mat[..., c0, r]    # A_0j
        return cls(center=center, leaf_diag=leaf_diag, coupling=coupling)


@dataclass
class BlockTree:
    """Symmetric SPD matrix whose block graph is an arbitrary rooted tree.

    Generalizes :class:`BlockTridiag` (path) and :class:`BlockStar` (depth-1 tree):
    ``n`` node blocks on the diagonal and one edge block per non-root node, stored in
    ``O(n * b**2)`` (the dense ``N x N`` matrix, ``N = n*b``, is materialized only by
    :meth:`to_dense` for tests / the oracle).

    Attributes
    ----------
    diag:
        Node diagonal blocks ``A_vv`` with shape ``[..., n, b, b]`` (each symmetric).
    edge:
        Edge blocks ``edge[v] = A_{p(v), v}`` (parent-row, child-col) with shape
        ``[..., n, b, b]``. The slot for the root is unused and must be zero;
        ``A_{v, p(v)} = edge[v].mT`` by symmetry.
    parent:
        1-D ``long`` parent array of length ``n`` (``parent[root] = -1``). The tree
        topology is shared across the leading batch dims and is *not* batched.
    """

    diag: Tensor
    edge: Tensor
    parent: Tensor

    def __post_init__(self) -> None:
        self.parent = _as_parent_tensor(self.parent)
        self._orders: tuple[int, list[list[int]], list[int]] | None = None

    # -- topology (validated + cached) ----------------------------------------
    def _ensure_orders(self) -> tuple[int, list[list[int]], list[int]]:
        if self._orders is None:
            self._orders = tree_orders(self.parent)
        return self._orders

    @property
    def root(self) -> int:
        return self._ensure_orders()[0]

    @property
    def children(self) -> list[list[int]]:
        return self._ensure_orders()[1]

    @property
    def collect_order(self) -> list[int]:
        """Nodes children-before-parents (leaves-to-root)."""
        return self._ensure_orders()[2]

    @property
    def distribute_order(self) -> list[int]:
        """Nodes parents-before-children (root-to-leaves)."""
        return list(reversed(self._ensure_orders()[2]))

    # -- basic shape accessors -------------------------------------------------
    @property
    def num_nodes(self) -> int:
        """Number of node blocks ``n``."""
        return self.diag.shape[-3]

    @property
    def block_size(self) -> int:
        """Block dimension ``b``."""
        return self.diag.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions (everything before ``n, b, b``)."""
        return self.diag.shape[:-3]

    # -- validation ------------------------------------------------------------
    def validate(self, *, symmetry_tol: float | None = None) -> "BlockTree":
        """Validate shapes, dtype/device, tree topology, symmetry, and zero root edge."""
        if not isinstance(self.diag, Tensor) or not isinstance(self.edge, Tensor):
            raise TypeError("diag and edge must be torch.Tensor instances")
        if self.diag.dtype != self.edge.dtype:
            raise TypeError(f"dtype mismatch: diag={self.diag.dtype}, edge={self.edge.dtype}")
        if self.diag.device != self.edge.device:
            raise ValueError(f"device mismatch: diag={self.diag.device}, edge={self.edge.device}")
        if self.diag.ndim < 3:
            raise ValueError(f"diag must have shape [..., n, b, b]; got ndim={self.diag.ndim}")
        n, b, b2 = self.diag.shape[-3], self.diag.shape[-2], self.diag.shape[-1]
        if b != b2:
            raise ValueError(f"diag blocks must be square; got {b}x{b2}")
        if n < 1:
            raise ValueError(f"need at least one node; got n={n}")
        if tuple(self.edge.shape) != tuple(self.diag.shape):
            raise ValueError(
                f"edge must have the same shape as diag {tuple(self.diag.shape)}; "
                f"got {tuple(self.edge.shape)}"
            )
        if int(self.parent.numel()) != n:
            raise ValueError(f"parent must have length n={n}; got {int(self.parent.numel())}")

        # Topology (raises on invalid parent array).
        root, _, _ = self._ensure_orders()

        tol = symmetry_tol
        if tol is None:
            tol = 1e-6 if self.diag.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-10
        asym = (self.diag - self.diag.mT).abs().amax()
        scale = self.diag.abs().amax().clamp_min(1.0)
        if asym > tol * scale:
            raise ValueError(
                f"diagonal blocks are not symmetric within tol={tol} "
                f"(max asymmetry {float(asym):.3e}, scale {float(scale):.3e})"
            )
        root_edge = self.edge[..., root, :, :].abs().amax()
        if root_edge > tol * scale:
            raise ValueError(
                f"edge slot for the root (node {root}) must be zero; "
                f"got max abs {float(root_edge):.3e}"
            )
        return self

    # -- dense bridges (tests / oracle only) -----------------------------------
    def to_dense(self) -> Tensor:
        """Materialize the dense ``[..., N, N]`` matrix (``N = n*b``). Test/oracle use only."""
        n, b = self.num_nodes, self.block_size
        plist = self.parent.tolist()
        N = n * b
        out = self.diag.new_zeros((*self.batch_shape, N, N))
        for v in range(n):
            rv = slice(v * b, (v + 1) * b)
            out[..., rv, rv] = self.diag[..., v, :, :]
            pv = plist[v]
            if pv != -1:
                rp = slice(pv * b, (pv + 1) * b)
                e = self.edge[..., v, :, :]          # A_{p,v}
                out[..., rp, rv] = e                  # parent row -> child col
                out[..., rv, rp] = e.mT               # A_{v,p}
        return out

    @classmethod
    def from_dense(cls, mat: Tensor, block_size: int, parent: Tensor | Sequence[int]) -> "BlockTree":
        """Extract block-tree storage from a dense ``[..., N, N]`` matrix and a ``parent`` array.

        Off-pattern entries are *ignored* (projection onto the tree pattern).
        Test/oracle use only.
        """
        N = mat.shape[-1]
        if mat.shape[-2] != N:
            raise ValueError(f"matrix must be square; got {tuple(mat.shape[-2:])}")
        if N % block_size != 0:
            raise ValueError(f"N={N} not divisible by block_size={block_size}")
        b = block_size
        n = N // b
        p = _as_parent_tensor(parent)
        if int(p.numel()) != n:
            raise ValueError(f"parent must have length n={n}; got {int(p.numel())}")
        plist = p.tolist()
        diag = mat.new_zeros((*mat.shape[:-2], n, b, b))
        edge = mat.new_zeros((*mat.shape[:-2], n, b, b))
        for v in range(n):
            rv = slice(v * b, (v + 1) * b)
            diag[..., v, :, :] = mat[..., rv, rv]
            pv = plist[v]
            if pv != -1:
                rp = slice(pv * b, (pv + 1) * b)
                edge[..., v, :, :] = mat[..., rp, rv]   # A_{p,v}
        return cls(diag=diag, edge=edge, parent=p)


@dataclass
class BlockSparseSym:
    """Symmetric SPD block matrix with an arbitrary (general) sparsity pattern.

    The input layout for the junction-tree / general-treewidth selected inverse
    (:func:`gabp_sparse_inv.selected_inverse_junction`). Stores the node diagonal
    blocks and the lower-triangular off-diagonal blocks on the pattern; the upper
    triangle is implied by symmetry. The dense ``N x N`` matrix (``N = n*b``) is
    materialized only by :meth:`to_dense` for tests / the oracle.

    Attributes
    ----------
    diag:
        Node diagonal blocks ``A_vv`` with shape ``[..., n, b, b]`` (each symmetric).
    edge_index:
        ``[2, m]`` long tensor of off-diagonal block positions ``(i, j)`` with
        ``i > j`` (node index). Shared across leading batch dims (not batched).
    edge_val:
        Off-diagonal blocks ``edge_val[..., k, :, :] = A_{i_k, j_k}`` (row ``i``, col
        ``j``) with shape ``[..., m, b, b]``.
    """

    diag: Tensor
    edge_index: Tensor
    edge_val: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.edge_index, Tensor):
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_index = self.edge_index.to(device="cpu", dtype=torch.long)

    # -- basic shape accessors -------------------------------------------------
    @property
    def num_nodes(self) -> int:
        """Number of node blocks ``n``."""
        return self.diag.shape[-3]

    @property
    def num_edges(self) -> int:
        """Number of stored off-diagonal blocks ``m``."""
        return self.edge_index.shape[1]

    @property
    def block_size(self) -> int:
        """Block dimension ``b``."""
        return self.diag.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions (everything before ``n, b, b``)."""
        return self.diag.shape[:-3]

    # -- validation ------------------------------------------------------------
    def validate(self, *, symmetry_tol: float | None = None) -> "BlockSparseSym":
        """Validate shapes, dtype/device, index range/orientation, and diag symmetry."""
        if not isinstance(self.diag, Tensor) or not isinstance(self.edge_val, Tensor):
            raise TypeError("diag and edge_val must be torch.Tensor instances")
        if self.diag.dtype != self.edge_val.dtype:
            raise TypeError(f"dtype mismatch: diag={self.diag.dtype}, edge_val={self.edge_val.dtype}")
        if self.diag.device != self.edge_val.device:
            raise ValueError(f"device mismatch: diag={self.diag.device}, edge_val={self.edge_val.device}")
        if self.diag.ndim < 3:
            raise ValueError(f"diag must have shape [..., n, b, b]; got ndim={self.diag.ndim}")
        n, b, b2 = self.diag.shape[-3], self.diag.shape[-2], self.diag.shape[-1]
        if b != b2:
            raise ValueError(f"diag blocks must be square; got {b}x{b2}")
        if n < 1:
            raise ValueError(f"need at least one node; got n={n}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, m]; got {tuple(self.edge_index.shape)}")
        m = self.edge_index.shape[1]
        expected = (*self.diag.shape[:-3], m, b, b)
        if tuple(self.edge_val.shape) != expected:
            raise ValueError(f"edge_val must have shape {expected}; got {tuple(self.edge_val.shape)}")
        if m > 0:
            ii, jj = self.edge_index[0], self.edge_index[1]
            if int(ii.max()) >= n or int(jj.max()) >= n or int(ii.min()) < 0 or int(jj.min()) < 0:
                raise ValueError(f"edge_index entries out of range [0, {n - 1}]")
            if bool((ii <= jj).any()):
                raise ValueError("edge_index must store the lower triangle: every column needs i > j")
            pairs = [(int(a), int(c)) for a, c in zip(ii.tolist(), jj.tolist())]
            if len(set(pairs)) != len(pairs):
                raise ValueError("edge_index contains duplicate (i, j) blocks")

        tol = symmetry_tol
        if tol is None:
            tol = 1e-6 if self.diag.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-10
        asym = (self.diag - self.diag.mT).abs().amax()
        scale = self.diag.abs().amax().clamp_min(1.0)
        if asym > tol * scale:
            raise ValueError(
                f"diagonal blocks are not symmetric within tol={tol} "
                f"(max asymmetry {float(asym):.3e}, scale {float(scale):.3e})"
            )
        return self

    # -- dense bridges (tests / oracle only) -----------------------------------
    def to_dense(self) -> Tensor:
        """Materialize the dense ``[..., N, N]`` matrix (``N = n*b``). Test/oracle use only."""
        n, b = self.num_nodes, self.block_size
        N = n * b
        out = self.diag.new_zeros((*self.batch_shape, N, N))
        for v in range(n):
            rv = slice(v * b, (v + 1) * b)
            out[..., rv, rv] = self.diag[..., v, :, :]
        ij = self.edge_index.tolist()
        for k in range(self.num_edges):
            i, j = ij[0][k], ij[1][k]
            ri, rj = slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b)
            blk = self.edge_val[..., k, :, :]
            out[..., ri, rj] = blk            # A_{i,j}
            out[..., rj, ri] = blk.mT         # A_{j,i}
        return out

    @classmethod
    def from_dense(cls, mat: Tensor, block_size: int, edge_index: Tensor) -> "BlockSparseSym":
        """Extract sparse-symmetric storage from a dense matrix on a given pattern.

        Off-pattern entries are *ignored* (projection onto the pattern). Test/oracle use.
        """
        N = mat.shape[-1]
        if mat.shape[-2] != N:
            raise ValueError(f"matrix must be square; got {tuple(mat.shape[-2:])}")
        if N % block_size != 0:
            raise ValueError(f"N={N} not divisible by block_size={block_size}")
        b = block_size
        n = N // b
        ei = edge_index.to(device="cpu", dtype=torch.long)
        m = ei.shape[1]
        diag = mat.new_zeros((*mat.shape[:-2], n, b, b))
        for v in range(n):
            rv = slice(v * b, (v + 1) * b)
            diag[..., v, :, :] = mat[..., rv, rv]
        edge_val = mat.new_zeros((*mat.shape[:-2], m, b, b))
        ij = ei.tolist()
        for k in range(m):
            i, j = ij[0][k], ij[1][k]
            ri, rj = slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b)
            edge_val[..., k, :, :] = mat[..., ri, rj]
        return cls(diag=diag, edge_index=ei, edge_val=edge_val)
