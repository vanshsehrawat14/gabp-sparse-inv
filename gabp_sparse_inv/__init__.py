"""Selected inversion for sparse block-structured matrices.

Returns the blocks of ``A^{-1}`` on a chosen sparsity pattern without ever forming the dense
inverse. Trees cost ``O(n b^3)`` work and ``O(n b^2)`` storage. For a general elimination
order with later-neighbour counts ``w_v``, numeric storage is
``Theta(sum_v(1+w_v) b^2)`` and clique work is
``Theta(sum_v(1+w_v^2) b^3)``. Kernels (all validated against dense fp64
oracles; see ``docs/PROJECT_STATUS.md`` for the authoritative scope):

SPD
    ``selected_inverse_chain`` (block-tridiagonal), ``selected_inverse_star`` (arrowhead),
    ``selected_inverse_tree`` (general tree), ``selinv_tree`` (the differentiable tree kernel
    with an analytic backward), and ``selected_inverse_junction`` / ``selinv_junction`` (general
    sparse SPD via the filled / chordal pattern, handling loopy graphs *exactly*).
Non-symmetric
    ``selected_inverse_bidiag`` (block lower-bidiagonal, DeltaNet chunk),
    ``selected_inverse_nonsym_tree`` (tree pattern, ``M_{uv} != M_{vu}``), and
    ``selected_inverse_nonsym_junction`` / ``selinv_nonsym_junction`` (general sparse
    non-symmetric on the filled pattern, via block LDU + the two-sided Takahashi /
    Erisman-Tinney recurrence; no pivoting, the static-pattern regime), with
    ``nonsym_junction_solve`` the sibling linear solve ``A^{-1} b`` / ``A^{-T} b`` (the
    fixed-point / DEQ implicit-differentiation primitive).
Applications
    differentiable tree- and grid/loopy-GMRF learning (:mod:`gabp_sparse_inv.gmrf`,
    :mod:`gabp_sparse_inv.gmrf_grid`) and Gaussian sampling (:mod:`gabp_sparse_inv.sampling`).

Out of scope unless noted otherwise here: iterative / loopy GaBP, pivoting for the
non-symmetric kernels (the static-pattern regime is assumed), and indefinite or
complex-Hermitian matrices.
"""

from __future__ import annotations

from .chain import ChainFactors, selected_inverse_chain
from .generators import (
    condition_number,
    grid_edges,
    random_nonsym_bidiag,
    random_spd_chain,
    random_spd_graph,
    random_spd_laplacian,
    random_spd_star,
    random_spd_tree,
)
from .autodiff import SelInvTree, selinv_tree
from .junction import (
    elimination_order_min_degree,
    elimination_order_nested_dissection,
    junction_logdet,
    junction_solve,
    selected_inverse_junction,
    selinv_junction,
)
from .nonsym import (
    SelInvBidiag,
    SelInvTril,
    selected_inverse_bidiag,
    selected_inverse_nonsym_tree,
    selected_inverse_tril,
    selinv_bidiag,
    selinv_tril,
)
from .nonsym_junction import (
    nonsym_junction_solve,
    selected_inverse_nonsym_junction,
    selinv_nonsym_junction,
)
from .junction_autodiff import (
    selinv_junction_analytic,
    selinv_nonsym_junction_analytic,
)
from .gmrf import (
    fit_marginal_likelihood,
    marginal_log_likelihood,
    posterior_marginal_variances,
    sample_tree_gmrf,
    tree_gmrf_precision,
    tree_logdet,
    tree_solve,
)
from .gmrf_grid import (
    fit_grid_marginal_likelihood,
    grid_gmrf_precision,
    grid_node_degrees,
    junction_marginal_log_likelihood,
    junction_posterior_marginal_variances,
)
from .layout import BlockBidiag, BlockSparseSym, BlockStar, BlockTree, BlockTridiag
from .sampling import sample_gaussian_junction, sample_gaussian_tree
from .star import StarFactors, selected_inverse_star
from .tree import TreeFactors, selected_inverse_tree

__all__ = [
    "BlockTridiag",
    "BlockBidiag",
    "BlockStar",
    "BlockTree",
    "BlockSparseSym",
    "ChainFactors",
    "StarFactors",
    "TreeFactors",
    "selected_inverse_chain",
    "selected_inverse_star",
    "selected_inverse_tree",
    "selinv_tree",
    "SelInvTree",
    "selected_inverse_junction",
    "selinv_junction",
    "junction_solve",
    "junction_logdet",
    "elimination_order_min_degree",
    "elimination_order_nested_dissection",
    "selected_inverse_bidiag",
    "selinv_bidiag",
    "SelInvBidiag",
    "selected_inverse_tril",
    "selinv_tril",
    "SelInvTril",
    "selected_inverse_nonsym_tree",
    "selected_inverse_nonsym_junction",
    "selinv_nonsym_junction",
    "nonsym_junction_solve",
    "selinv_junction_analytic",
    "selinv_nonsym_junction_analytic",
    "tree_gmrf_precision",
    "tree_logdet",
    "tree_solve",
    "marginal_log_likelihood",
    "posterior_marginal_variances",
    "sample_tree_gmrf",
    "sample_gaussian_tree",
    "sample_gaussian_junction",
    "fit_marginal_likelihood",
    "grid_gmrf_precision",
    "grid_node_degrees",
    "junction_marginal_log_likelihood",
    "junction_posterior_marginal_variances",
    "fit_grid_marginal_likelihood",
    "random_spd_chain",
    "random_spd_star",
    "random_spd_tree",
    "random_nonsym_bidiag",
    "random_spd_graph",
    "random_spd_laplacian",
    "grid_edges",
    "condition_number",
]

__version__ = "0.3.3"
