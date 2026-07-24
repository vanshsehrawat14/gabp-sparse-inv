# Selected inversion on trees: derivations

This note states and proves the result that underlies every kernel in
`gabp-sparse-inv`: for an SPD block matrix whose graph is a **tree**, the
*selected inverse* (the inverse blocks on the matrix's own sparsity pattern) is
computed **exactly** by a two-pass collect/distribute schedule, which is precisely
**Gaussian Belief Propagation** and coincides with the Takahashi recurrence.
The block-tridiagonal **chain** (Phase 1) and the block-arrowhead **star**
(Phase 2) are corollaries.

Notation follows the code: blocks are `b x b`; `A.mT` is the (batched) transpose;
`D^{-1}` denotes a block inverse realized in practice through a Cholesky factor.

---

## 1. Problem statement

Let `G = (V, E)` be an undirected graph on `|V| = n` nodes and let `A` be a
symmetric positive-definite (SPD) matrix block-partitioned by `V`,

    A = [A_{uv}]_{u,v in V},   A_{uv} in R^{b x b},   A_{uv} = A_{vu}^T,

whose **block sparsity pattern equals `G`**:

    A_{uv} = 0   whenever   u != v   and   {u,v} not in E.        (1)

The **selected inverse** of `A` is the set of blocks of `G := A^{-1}` that lie on
the pattern (diagonal blocks and the blocks of edges of `G`):

    SelInv(A) = { G_{uv} : u = v } union { G_{uv} : {u,v} in E }.     (2)

Off-pattern blocks `G_{uv}` (with `{u,v}` not an edge) are generally **nonzero**
- `A^{-1}` is dense even when `A` is sparse - but are *not* part of the target and
are never formed. (Section 5 exhibits them explicitly for the star.)

The goal is to compute (2) in time and memory proportional to the *structured*
size of `A` (`O(|V| + |E|)` blocks), never the `(nb)^2` of the dense inverse.

---

## 2. The tree theorem

> **Theorem.** Let `G` be a tree, rooted arbitrarily at `r`, and let `A` be SPD
> with pattern `G`. Order the nodes so that every node precedes its parent
> (any leaves-to-root order). Then:
>
> **(a) Zero fill.** The block `LDL^T` factorization `A = L D L^T` in this order
> has `L` *unit lower triangular with the same block pattern as `A`* - eliminating
> a node creates no new edges. Every pivot `D_v` is SPD.
>
> **(b) Collect (leaves -> root).** Defining, leaves first,
>
>     D_v = A_{vv} - sum_{c in children(v)} A_{vc} D_c^{-1} A_{cv},      (3)
>
> the root pivot is the root's marginal precision and
>
>     G_{rr} = D_r^{-1}.                                                (4)
>
> **(c) Distribute (root -> leaves).** With the factor blocks
> `ell_v = A_{p(v), v} D_v^{-1}` for each non-root `v` (parent `p(v)`),
>
>     G_{p(v), v} = - G_{p(v), p(v)} ell_v,                             (5)
>     G_{vv}      = D_v^{-1} + ell_v^T G_{p(v), p(v)} ell_v.            (6)
>
> Equations (3)-(6) produce every block in `SelInv(A)` exactly, in
> `O(sum_v (1 + |children(v)|) b^3) = O((|V|+|E|) b^3)` time and
> `O((|V|+|E|) b^2)` memory.

### Proof

**(a) Zero fill.** A vertex is *simplicial* if its neighbors form a clique.
A leaf has exactly one neighbor, so it is trivially simplicial; eliminating a
simplicial vertex introduces no fill edges (the standard fact that the elimination
graph gains edges only among the eliminated vertex's higher-ordered neighbors, of
which a leaf has none below its parent). Removing a leaf from a tree leaves a
tree, so by induction the leaves-to-root order is a *perfect elimination ordering*
and `L` inherits the pattern of `A`: `L_{uv} != 0` only for `u = v` or the
parent-child edge `{u, p(u)}`. SPD-ness of `A` makes each Schur complement (pivot)
SPD, so the factorization never breaks down. QED(a)

**(b) Collect.** Eliminating node `v` after all of its children is block Gaussian
elimination: the children contribute Schur terms `A_{vc} D_c^{-1} A_{cv}` to `v`'s
pivot, giving (3). For the root, `D_r` is the Schur complement of everything else,
i.e. the marginal precision of `r`; the marginal covariance is its inverse, so
`G_{rr} = D_r^{-1}`, which is (4). (Equivalently: `A = L D L^T` with `L` unit
lower-triangular puts `D_r` in the last pivot, and the last diagonal block of
`A^{-1} = L^{-T} D^{-1} L^{-1}` is `D_r^{-1}` because the last row of `L^{-1}` is
`e_r^T`.) QED(b)

**(c) Distribute.** From `A = L D L^T`, `G = A^{-1} = L^{-T} D^{-1} L^{-1}`. The
Takahashi identity rearranges this into a back-substitution that, by `(I - L^T)`
being strictly upper with the tree pattern, expresses each on-pattern block of `G`
through *already-computed* on-pattern blocks closer to the root:

    G = D^{-1} L^{-1} + (I - L^T) G.                                  (7)

Read (7) block-row by block-row from the root downward. On a tree the only nonzero
off-diagonal of `L` in row `v` is the parent edge, with factor block `ell_v`
(defined so that `A_{p(v),v} = ell_v D_v`, i.e. `ell_v = A_{p(v),v} D_v^{-1}`).
Specializing (7) to the cross block `(p(v), v)` and the diagonal block `(v, v)`
gives (5) and (6); every term on the right is a block at or above `v`'s parent,
already produced by the time node `v` is reached. Symmetry `G_{v,p(v)} = G_{p(v),v}^T`
closes the recurrence. A direct verification that (5)-(6) satisfy the on-pattern
rows of `A G = I` is given for the star in Section 5 and is what
`selected_inverse_residual*` checks numerically. QED(c)

**Complexity.** Each node does a constant number of `b x b` solves/products plus
one per child; summing `1 + |children(v)|` over the tree gives `|V| + |E|` block
operations, hence `O((|V|+|E|) b^3)` time and `O((|V|+|E|) b^2)` storage. QED

### 2.1 Pattern-closure lemma (stated in pattern-of-`L` terms)

Part (c) read (7) "block-row by block-row … every term on the right is already
produced." That reading rests on a closure property which, stated in the pattern of
the factor `L` rather than of `A`, is exactly what the general-treewidth case
(Phase 4) needs. Let `chol(A)` denote the lower block-Cholesky factor `L` of `A` in
the chosen elimination order, and define the **selected set**

    S  :=  pattern(L + L^T)                                           (8)

 -  the diagonal blocks together with every block where `L` (equivalently `L^T`) is
nonzero. (`S` contains the edges of `A` **plus any fill edges** created during
elimination.)

> **Lemma (closure).** The Takahashi recurrence (7) for the blocks `G_{uv}` with
> `(u,v) in S` references, on its right-hand side, only blocks `G_{u'v'}` with
> `(u',v') in S`. Hence the selected blocks `{G_{uv} : (u,v) in S}` are computable
> among themselves, with no off-`S` block ever required.

*Proof.* Row `v` of `(7)` couples `G_{·,v}` to `{G_{·,w} : L_{wv} != 0, w > v}`
through `(I - L^T)`. Each such `w` is, by definition of `S`, a neighbour of `v` in
`S`, and the standard Takahashi/Erisman-Tinney argument (inverse entries on the
factor pattern depend only on factor-pattern entries) closes the index set under the
recurrence. ∎

**Tree instance (this phase).** On a tree the leaves-to-root order is a perfect
elimination ordering with **zero fill** (part (a)), so `pattern(L) = pattern(A)` and
`S = SelInv(A)`: the selected set of the lemma is exactly the on-pattern target (2),
and "neighbour in `S`" means "parent or child." This is why the tree kernel needs
only the parent/child blocks. **For Phase 4 the same lemma holds verbatim with `S`
the *filled* pattern** - the chordal completion of `A` - which is the correct
statement to inherit there; the tree case is its zero-fill specialization.

### 2.2 The filled-pattern forward (junction tree)

The two-pass schedule (3)-(6) generalizes *verbatim* to an **arbitrary** sparsity
pattern once it is read on the filled set `S` (eq. 8) instead of the tree. This is the
Phase-4 forward, implemented in `gabp_sparse_inv/junction.py`. Number the nodes by an
elimination order; for node `v` let

    U_v := { w : w eliminated after v, (w, v) in S }

be its higher-ordered neighbours. By chordality of the filled graph `U_v` is a
**clique** in `S` - the single fact that closes the recurrence (every block it touches
is itself on `S`). With pivots `D_v` and factor blocks `ell_{wv} = A_{wv} D_v^{-1}`:

**Collect (sparse block Cholesky; eliminate `v` in increasing order).** The pivot `D_v`
is `A_vv` after the Schur updates of `v`'s earlier-eliminated neighbours have landed in
it; factor it, solve for the factor blocks, and apply the **clique (Schur) update**

    ell_{wv} = A_{wv} D_v^{-1},                            w in U_v
    A_{w,w'} <- A_{w,w'} - ell_{wv} D_v ell_{w'v}^T,       w, w' in U_v         (J-collect)

whose last line *is* the fill - eliminating `v` couples every pair of its higher
neighbours. On a tree `|U_v| <= 1`, so the update is empty (zero fill, recovering (3)).

**Distribute (Takahashi back-substitution; decreasing order).** Restricting (7) to row
`v`, where the only nonzero off-diagonals of `L` are the `ell_{uv}` (`u in U_v`),

    G_{wv} = - sum_{u in U_v} G_{wu} ell_{uv},             w in U_v             (J-cross)
    G_vv   =   D_v^{-1} - sum_{u in U_v} ell_{uv}^T G_{uv}.                     (J-diag)

Every `G_{wu}` on the right of (J-cross) has both indices in the clique `U_v ⊆ S` and
both eliminated after `v`, so it is already computed - the §2.1 closure Lemma in force.

**Tree reduction.** When `G` is a tree and the order is leaves-to-root, `U_v = {p(v)}`:
(J-cross) is `G_{p(v),v} = -G_{p(v),p(v)} ell_v` = (5), and substituting it into (J-diag)
gives `G_vv = D_v^{-1} - ell_v^T(-G_{p(v),p(v)} ell_v) = D_v^{-1} + ell_v^T G_{p(v),p(v)}
ell_v` = (6). `tests/test_junction.py` asserts this block-for-block against `tree.py`.

**Cost and storage.** These are different structural quantities. Write
`w_v := |U_v|` and define

    F := sum_v (1 + w_v),       W := sum_v (1 + w_v^2).                (J-cost)

`F` is the number of diagonal-plus-lower blocks in the factor pattern (up to
orientation), while `W` is the scalar-block structural work proxy for the dense
clique updates and Takahashi sums. Thus numeric factor/selected-block storage is
`Theta(F b^2)` and factorization plus selected inversion costs `Theta(W b^3)`.
If `omega := 1 + max_v w_v`, then

    F <= W <= omega F,          W >= F^2 / (2 |V|).

The first two inequalities follow from `w <= w^2` for integer `w >= 1` and
`w^2 <= (omega-1)w`; the last follows from Cauchy--Schwarz and
`sum_v w_v^2 >= (sum_v w_v)^2 / |V|`. Consequently `W = Theta(F)` only when
the front width is bounded. On a tree both are `Theta(n)`. On a 2-D grid under
nested dissection, `F = Theta(n log n)` but `W = Theta(n^{3/2})`.

The shipped level-set implementation also pre-materializes flattened
clique-pair index tensors, so its symbolic metadata is `Theta(W)`, not
`Theta(F)`. A streaming or supernodal implementation could retain
`Theta(F)` numeric storage while generating clique work on demand. The shipped
reference uses a min-degree order and a per-node loop; neither implementation
is claimed memory-optimal.

---

## 3. Gaussian Belief Propagation reading

Interpret `A` as the precision (information) matrix of a Gaussian Markov random
field `x ~ N(0, A^{-1})` whose conditional-independence graph is `G`. The collect
pass (3) is exactly the **upward precision message** of Gaussian BP on the tree:
child `c` sends `A_{vc} D_c^{-1} A_{cv}` to parent `v`, and `D_v` is `v`'s marginal
precision after absorbing its subtree. The distribute pass (5)-(6) is the
**downward pass** that turns marginal precisions into the joint pairwise marginals
`Cov(x_{p(v)}, x_v)` and `Var(x_v)` - i.e. the on-pattern covariance blocks.
Gaussian BP is exact on trees (Weiss & Freeman, 2001; Bickson, 2008), which is the
probabilistic restatement of the Theorem. Loopy graphs (treewidth > 1) are handled **exactly**
on the filled pattern by the junction-tree kernel (`junction.py`, §2.2); iterative loopy BP
is a separate approximate route and is **not** implemented here.

---

## 4. Corollary: the chain (Phase 1)

A path `0 - 1 - ... - (L-1)` rooted at the last node makes each node `i` the single
child of `i+1`. Then `children(i) = {i-1}` (or none), and (3)-(6) reduce to

    D_0 = A_{00},   D_i = A_{ii} - ell_i D_{i-1} ell_i^T,   ell_i = A_{i,i-1} D_{i-1}^{-1},
    G_{L-1,L-1} = D_{L-1}^{-1},
    G_{i+1,i} = - G_{i+1,i+1} ell_{i+1},   G_{ii} = D_i^{-1} + ell_{i+1}^T G_{i+1,i+1} ell_{i+1},

which is precisely the block `LDL^T` + Takahashi back-recursion implemented in
`gabp_sparse_inv/chain.py`. (This is also the matrix form of the RTS / Kalman
smoother two-pass.)

---

## 5. Corollary: the star (Phase 2)

A star roots at the **center** `0`; every leaf `j = 1..K` is a child of `0` with no
children of its own, so `D_j = A_{jj}`. Write `A_{0j}` for the center->leaf
coupling and `B_j := A_{jj}^{-1} A_{j,0} = ell_j^T`. The Theorem gives, with the
**center Schur complement** `S`,

    S      = A_{00} - sum_j A_{0j} B_j,                     (collect, eq. 3)
    G_{00} = S^{-1},                                        (eq. 4)
    G_{j,0} = - B_j G_{00},                                 (eq. 5, transposed)
    G_{jj}  = A_{jj}^{-1} + B_j G_{00} B_j^T.               (eq. 6)

These are the formulas in `gabp_sparse_inv/star.py`. Because the leaves are
mutually independent, the collect is a single sum over `j` and the distribute is
batched over `j` - there is **no Python loop over leaves**.

**On-pattern residual equations.** The blocks above satisfy exactly the rows of
`A G = I` whose `G`-columns are on the pattern:

    center:  A_{00} G_{00} + sum_j A_{0j} G_{j,0} = I,
    cross :  A_{j,0} G_{00} + A_{jj} G_{j,0}      = 0,
    leaf  :  A_{j,0} G_{0,j} + A_{jj} G_{jj}      = I.

`metrics.selected_inverse_residual_star` checks these three (and only these - the
mixed center<->leaf-`k` and leaf-`j`<->leaf-`k` rows involve off-pattern blocks).

**Off-pattern blocks are nonzero.** For `j != k` the path between leaves passes
through the center, and

    (A^{-1})_{jk} = B_j G_{00} B_k^T  !=  0   in general,

so `A^{-1}` is dense on the leaf-leaf pairs even though `A` is not. The implementation never forms
these; they are outside `SelInv(A)`.

**`K = 1` reduces to a 2-block chain.** A single leaf is the path `0 - 1`, so the
star and chain kernels must agree block-for-block - verified to ~1e-12 in
`tests/test_star.py::test_K1_matches_chain`.

---

## 5.5 The general tree kernel (Phase 3)

`gabp_sparse_inv/tree.py` implements (3)-(6) for an **arbitrary** rooted tree given
by a `parent` array (`parent[root] = -1`). Two conventions fix the orientation:

- **Storage.** One edge block per non-root node, `edge[v] = A_{p(v), v}`
  (parent-row, child-col). In the notation of (3)-(6) this is `U_v` with
  `A_{p(v),v} = ell_v D_v`, so `ell_v = edge[v] D_v^{-1} = U_v D_v^{-1}`. The
  selected cross output is stored the same way, `G_edge[v] = G_{p(v), v}` (and
  `G_{v,p(v)} = G_edge[v]^T`).
- **Schedule.** *Collect* visits nodes children-before-parents (Kahn leaf-peeling),
  accumulating each child's Schur term `ell_v U_v^T = U_v D_v^{-1} U_v^T` into its
  parent pivot (3). *Distribute* visits parents-before-children (the reverse),
  applying (5)-(6).

The chain and star kernels are the path and depth-1 instances and are cross-checked
against `tree.py` to ~1e-12 (`tests/test_tree.py`). The path check is in fact a
**root-invariance** statement: `chain.py` roots the path at its *last* node while the
path-tree roots it at node 0, and the selected inverse - being just blocks of
`A^{-1}` - must agree block-for-block regardless of elimination root.

This kernel is a **correctness reference**: its `O(n)` per-node Python loop is
launch-latency-bound on GPU. Batching siblings/level-sets (and the differentiable
backward) is deferred to the differentiable-selinv work (see §8 and PROJECT_STATUS).

---

## 6. Floating point and stability

**Pivot-symmetrization policy.** A pivot/Schur block `D_v` is symmetric in exact
arithmetic, but the computed Schur update (a sum of `ell_v U_v^T` terms, or
`A_{ii} - C D^{-1} C^T`) is not exactly symmetric in floating point. All kernels
adopt one policy: **symmetrize every pivot, `D <- (D + D^T)/2`, immediately before
its Cholesky**, and likewise symmetrize the selected diagonal outputs `G_{vv}`. This
is `O(b^2)` per block, removes a backend-dependent dependence on which triangle the
Cholesky reads, and gives a consistent story for the error analysis below.
(`star.py`, `tree.py`, and `chain.py` all follow it.)

**Local primitive error bounds (not a global stability theorem).** Let `u` be the unit roundoff and
`gamma_k := k u / (1 - k u)`. The schedule is built from three `b x b` primitives,
each with a standard backward-error bound (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed.):

- **(C) SPD Cholesky** of a pivot `D_v`: the computed factor satisfies
  `R̂^T R̂ = D_v + ΔD_v` with `|ΔD_v| <= (b+1) gamma_{b+1} |R̂^T||R̂|`, hence
  `||ΔD_v||_2 <= c_C · b · u · ||D_v||_2 + O(u^2)` (Thm 10.3-10.4). Crucially there is
  **no element-growth factor**: SPD needs no pivoting and `||R̂||_2^2 = ||D_v||_2`.
- **(S) Triangular solve** (each of the two in `cholesky_solve` for `ell_v`):
  `(T + ΔT) x̂ = y` with `|ΔT| <= gamma_b |T|`, so `||ΔT||_2 <= c_S · b · u · ||T||_2`
  (Thm 8.5).
- **(P) Block product** with inner dimension `b`: `|Ĉ - AB| <= gamma_b |A||B|`, so
  `||Ĉ - AB||_2 <= c_P · b · u · ||A||_2 ||B||_2` (§3.5).

**What these local bounds do and do not prove.** Composing (C), (S), and (P)
shows that every local factorization, solve, product, and accumulation has the
standard backward-error form. SPD Cholesky avoids pivot-growth factors, and the
first-order perturbation of the inverse still predicts forward sensitivity
proportional to `kappa(A) u`.

Those local facts do **not**, by themselves, prove that the complete two-pass
algorithm is the exact selected inverse of one globally perturbed matrix
`A + E` with a normwise constant independent of tree height or conditioning.
Turning local perturbations into a single structured `E` requires controlling
how factor-block magnitudes and cancellation accumulate along the elimination
tree. The previous version of this note asserted that global theorem without
such a proof. It is retracted here.

An honest open theorem is therefore: under which scaling and structural
assumptions can one bound a structured backward error

    ||E|| / ||A|| <= C(A, G, b) u + O(u^2),                              (9a)

and how must `C` depend on elimination depth, degree, front width, and factor
growth? If such a bound is available, the usual first-order inverse
perturbation gives

    ||G_hat - G|| / ||G|| <= C(A, G, b) kappa(A) u + O(u^2).             (9)

The absence of pivot growth for SPD Cholesky is useful, but it is not a license
to call the selected inverse condition-independent.

**Empirical check (`bench/stability.py`).** The harness reassembles the matrix
represented by the kernel's fp32 factors and reports
`||A_hat - A||_2 / (u ||A||_2)` while sweeping tree height and `kappa(A)`.
On one measured `b = 2` grid the statistic is about `0.3--0.55` and shows no
visible growth with either axis. This is a finite-grid diagnostic, not evidence
of height or condition-number independence; `tests/test_stability.py` checks
the harness, not a universal theorem. Forward error increases with
conditioning on the same grid, as ordinary perturbation theory predicts.
Values are BLAS- and dtype-dependent.

Block inverses are realized through Cholesky factors (`cholesky_solve`) rather than
explicit inverses where it matters. The finite test grids are consistent with a small
effective constant in (9), but do not establish that bound outside those grids. The tests
therefore use a conservative condition-aware tolerance (`tol ~ 1e3 * kappa * eps` in
fp64), not a claimed theorem. Well-conditioned fp64 problems match the dense oracle to
`~1e-10`; fp32 holds a loose `~1e-4`; bf16 is a coarse storage format exercised through
a store-low / compute-fp32 path (CPU LAPACK has no half Cholesky). Note that for an
*uncontrolled*
random tree the factor blocks `ell_v` compound along root-to-leaf paths (expected
height `~ e*log n`), so `kappa(A)` is not pinned by `b` alone the way it is for a
fixed-depth chain or star; the tests therefore use condition-aware tolerances
(`C*kappa*eps`) on random trees and reserve absolute floors for controlled
(path/balanced) topologies.

### 6.1 Non-symmetric kernels (§9)

The non-symmetric kernels lose SPD-ness, so the analysis splits into the benign and the
conditional case.

- **Lower-bidiagonal (§9.2).** Each selected block is *one* `b x b` inverse plus at most two
  products (15), with **no accumulation sweep**. Its backward error is therefore purely
  local - `||ΔM_vv||_2 <= c·b·u·||M_vv||_2` per diagonal block and similarly for the
  couplings - with no degree or height factor at all, and the forward error of each block is
  governed by `kappa(M_vv)` (resp. the two endpoint diagonals) rather than a global
  `kappa(M)`. This locality is why §9 needs no symmetrization and matches the dense oracle to
  `~1e-9`.
- **Dense triangular `(I - A)^{-1}` (§9.4).** Computed by backward-stable triangular solves
  (S): the computed `T̂` solves `(I - A + ΔM) T̂ = I` with `||ΔM||_2 <= c·N·u·||I - A||_2`
  (Higham ch. 8). Without SPD the forward error is governed by `kappa(I - A)`; for the
  strictly-lower (nilpotent) `A` of a DeltaNet chunk the unit diagonal keeps `I - A`
  well away from singular, so the **static, pivot-free** pattern is stable precisely in the
  well-conditioned-triangular regime - the explicit assumption the roadmap flags for the
  general non-symmetric thread (no pivoting keeps `O(n)`/zero-fill but needs the conditioning
  to cooperate).

---

## 7. Relation to prior work

- **Takahashi, Fagan & Chen (1973)** and **Erisman & Tinney (1975)** - the
  Takahashi equations and the observation that inverse entries on the pattern of
  the Cholesky factor are computable without the full inverse. The tree theorem is
  the zero-fill (treewidth-1) special case where the factor pattern equals `A`'s.
- **Selected inversion / SelInv, Lin, Lu, Ying & E (2011)** and **PEXSI** - selected
  inversion for general sparse SPD/Hermitian matrices via the elimination tree; the
  block-tree case here is the fill-free instance and the natural building block for
  the junction-tree generalization (§2.2, `junction.py`).
- **Gaussian Belief Propagation - Weiss & Freeman (2001); Bickson (2008)** - exactness
  of Gaussian BP on trees; Section 3 is the precise correspondence to (3)-(6).
- **Matrix-derivative adjoints - Dwyer & Macphail (1948); Giles (2008)** - the
  reverse-mode adjoint of the matrix inverse/determinant/product, i.e. eq. (10) and
  the log-det gradient. These are the *folklore* primitives §8 builds on and claims
  nothing for; the contribution of §8 is the structure-preserving (self-adjoint)
  *schedule*, not the identity. (See also Murray's `chol-rev` for the Cholesky
  adjoint.)

---

## 8. The adjoint of selected inversion (the differentiable thread)

This section **proves**, on trees, the result the differentiable kernel is built
around: the reverse-mode adjoint of selected inversion is *itself* a two-pass
collect/distribute on the same elimination tree, with the same `O((|V|+|E|) b^3)`
cost as the forward pass. The explicit reverse recurrences derived here are exactly
what `gabp_sparse_inv/autodiff.py` (`SelInvTree`) implements.

### 8.0 What is and is not new (honest accounting)

The matrix-level identity is *folklore* and this document claims nothing for it. For `G = A^{-1}`
and a cotangent `bar_G` (the same-shaped tensor `bar_G_{uv} = ∂f/∂G_{uv}`),

    bar_A  =  - A^{-1} bar_G A^{-1}                                   (10)

is the Dwyer-Macphail (1948) / Giles (2008) adjoint of the matrix inverse. Likewise,
differentiating `log det A` needs only the *diagonal* selected blocks (`∂/∂A log det
A = A^{-1}`), which is the classical GMRF / sparse-Cholesky log-det-gradient result,
and the path instance is the differentiable RTS/Kalman smoother. **The contribution
here is structural, not the formula:** that the *full on-pattern* cotangent - every
diagonal and edge block, not just the log-det diagonal - is produced by a schedule
with the *same shape and cost* as the forward selected inverse (a "self-adjoint"
computational structure), and is realized as a single batched kernel (§ level sets).

### 8.1 Setup and parametrization convention (the factor-of-2 is here)

`A` is **not** a free dense matrix; its independent parameters are exactly the stored
blocks of §5.5: the symmetric node diagonals `A_vv = diag[v]` and one edge block per
non-root node `U_v = edge[v] = A_{p(v),v}` (with `A_{v,p(v)} = U_v^T` a *second*
appearance of the same parameter). The selected inverse returns `G_vv = G_diag[v]`
(symmetric) and `G_{p(v),v} = G_edge[v]` (one representative per edge;
`G_{v,p(v)} = G_edge[v]^T`).

For a scalar loss `f`, the derivation therefore works with the cotangents **of the stored blocks**,

    barGd[v] = ∂f/∂G_vv,   barGe[v] = ∂f/∂G_{p(v),v},                 (11a)
    barAd[v] = ∂f/∂A_vv,   barAe[v] = ∂f/∂U_v,                        (11b)

i.e. `barAd`, `barAe` are the gradients in the **autograd sense** w.r.t. the kernel's
input tensors `diag`, `edge`. This is the convention `torch.autograd.gradcheck`
verifies and the only one the implementation needs. The dense identity (10) is
recovered as a *remark*: because `U_v` occupies the two matrix slots `(p,v)` and
`(v,p)`, the stored-block gradient carries the bookkeeping

    barAe[v] = M_{p(v),v} + (M_{v,p(v)})^T,   M := -A^{-1} bar_G A^{-1},   (12)

where `bar_G` is the *symmetric* matrix cotangent with `bar_G_{p,v} = bar_G_{v,p}^T =
barGe[v]` (and `bar_G_{vv}` the symmetrized `barGd[v]`). The factor-of-2 / transpose
in (12) is the classic source of error in selected-inverse adjoints; the implementation sidesteps it
entirely in validation by differentiating a *parametrized dense oracle*
(`to_dense(diag, edge) → inv → extract → f`), which carries the duplication through
autograd automatically (see `tests/test_autodiff.py`).

### 8.2 The adjoint theorem (proved on trees)

> **Theorem (self-adjoint schedule).** Let `A` be SPD with tree pattern `G`, rooted
> as in §2, with forward factors `D_v`, `ell_v = U_v D_v^{-1}` and selected blocks
> `G_vv`, `G_{p(v),v}` from (3)-(6). For any loss `f` of the selected blocks, the
> stored-block gradients (11b) are computed exactly by the **reverse two-pass
> schedule (13)-(14)** below: a *reverse-distribute* sweep leaves→root followed by a
> *reverse-collect* sweep root→leaves on the **same** tree, reusing `D_v`, `ell_v`.
> Its cost is `O((|V|+|E|) b^3)` time and `O((|V|+|E|) b^2)` memory - identical to
> the forward pass.

**Proof.** The forward map `(diag, edge) ↦ (G_diag, G_edge)` is the finite composition
of the differentiable primitives (3)-(6) (block products, a symmetric solve realized
through Cholesky, the inverse `D_v^{-1}`, and the symmetrizations of §6) scheduled on
the tree. Reverse-mode AD of a composition is the reversed composition of the local
vector-Jacobian products (VJPs); applied here it yields *exactly* (13)-(14), so the
schedule computes the exact gradient `(barAd, barAe)`. The two claims that make it a
*tree* schedule of forward cost are: (i) each forward node touches only its own blocks
and its parent's, so each local VJP touches only the same neighbours - the reverse
sweep has the same `1 + |children(v)|` block-op count per node; and (ii) the data
dependencies reverse cleanly - a quantity computed in the forward collect (resp.
distribute) has all its consumers in later collect/distribute steps, so its cotangent
is complete by the time the reversed sweep reaches it (made explicit after (14)). ∎

**Remark (why the naive "off-`S`" argument fails).** It is tempting to argue from (10)
that `bar_A|_S` "never references an off-`S` block of `A^{-1}`." That is **false**:
expanding a diagonal block,

    barA_vv = -(A^{-1} bar_G A^{-1})_vv = - Σ_{(s,t) ∈ S} G_{vs} bar_G_{st} G_{tv},

and a diagonal cotangent `bar_G_{ss}` at a node `s` that is *not* a neighbour of `v`
contributes `-G_{vs} bar_G_{ss} G_{sv}` with `G_{vs}` an **off-pattern** block of
`A^{-1}`. The dense product genuinely involves off-`S` blocks. What is true - and what
the proof above uses - is that the *reverse schedule* never forms them: it propagates
cotangents along tree edges, exactly as the forward pass computes `G_vv` without ever
forming the dense inverse. The §2.1 closure lemma is about the Takahashi recurrence on
the factor pattern, a different object; it does **not** assert (10) restricts to `S`.

### 8.3 The explicit reverse recurrences (what `SelInvTree.backward` runs)

Write `s(X) = (X + X^T)/2`, and let `Dinv_v = D_v^{-1}` (reused from the forward
factor `chol_D`). Initialize accumulators `bG[v] = barGd[v]` and `bDinv[v] = bell[v]
= bU[v] = 0`. All blocks below are `b x b` and batched over leading dims.

**Pass 1 - reverse-distribute (leaves → root).** For each `v` in collect order,
with `M = s(bG[v])`:

    bDinv[v] += M                                                     (13a)
    if v != root, parent p:
        bell[v] += G_pp ell_v (M + M^T) - G_pp barGe[v]            (13b)
        bG[p]   += ell_v M ell_v^T - barGe[v] ell_v^T             (13c)

where `G_pp = G_diag[p]` is the (already symmetric) parent block. Leaves→root order
guarantees every child of `p` has executed (13c) before `p` is read in (13a), so
`bG[p]` is complete - this is dependency (ii) of the proof for the distribute pass.

**Pass 2 - reverse-collect (root → leaves).** For each `v` in distribute order:

    if v != root:                                                    (14a)
        bU[v]    += bell[v] Dinv_v
        bDinv[v] += U_v^T bell[v]
    bP = s( -Dinv_v bDinv[v] Dinv_v )                                (14b)
    barAd[v] = bP                                                    (14c)
    for c in children(v):                                            (14d)
        bell[c] += -(bP U_c)
        bU[c]   += -(bP ell_c)

then `barAe[v] = bU[v]` for non-root `v` (and `barAe[root] = 0`). Root→leaves order
guarantees `bell[v]` is complete (it gets its last contribution from the parent's
(14d)) before (14a) consumes it; `bDinv[v]` is complete after (14a) before (14b)
inverts it; and `bU[v]` is complete after (14a) and the parent's (14d). The
symmetrization in (14b) is the adjoint of the forward pivot/diagonal symmetrizations
(§6) - omitting it breaks agreement with `gradcheck`, which perturbs `diag[v]` in
antisymmetric directions too.

**Degenerate check (log-det).** Putting `barGd[v] = I`, `barGe[v] = 0` makes (13)-(14)
reproduce `∂ log det A / ∂A_vv` and `∂/∂U_v` on the pattern; the `n=1` case gives
`barAd = -Dinv s(I) Dinv = -G_vv^2`, matching `∂ log det / ∂A = A^{-1}` differentiated
once more. This is the case the Phase-3 smoke test
(`test_adjoint_logdet_matches_autograd`) already exercises; `tests/test_autodiff.py`
extends validation to general `barGd`, `barGe` against the parametrized dense oracle.

### 8.4 Filled pattern (junction tree): the self-adjoint schedule

Sections 8.1-8.3 are the zero-fill (tree) case. The same construction extends to the
**filled pattern** - the Phase-4 generalization. This section proves that the *adjoint* extends:
the self-adjoint schedule earlier drafts only conjectured is a theorem, with the §2.1
closure lemma (and its transpose) controlling the index set. Only the explicit recurrences
and the implementation are deferred.

**Setup.** Let `A` be SPD with an arbitrary sparsity pattern, fix an elimination order, and
let `L` be the block-Cholesky factor and `S := pattern(L + L^T)` the **filled** (chordal)
pattern (§2.1, eq. (8)) - the edges of `A` plus all fill. The forward selected inverse on
`S` is the sparse block-Cholesky `A|_S ↦ (D, L|_S)` followed by the Takahashi
back-substitution (7), `(D, L|_S) ↦ G|_S`; by the closure Lemma (§2.1) it references only
`S`-blocks.

> **Theorem (self-adjoint, filled pattern).** For any scalar loss `f` of the selected
> blocks `{G_{uv} : (u,v) ∈ S}`, the cotangents `{∂f/∂A_{uv} : (u,v) ∈ S}` are computed by
> a **reverse two-pass schedule on the same elimination structure** - a reverse-Takahashi
> sweep followed by a reverse-elimination sweep - that touches only `S`-blocks, reuses the
> forward factors `(D, L|_S)`, and costs the same `O(·)` as the forward selected inverse.
> Selected inversion on the filled pattern is self-adjoint.

**Proof.** The forward map `A|_S ↦ G|_S` is a finite composition of block primitives, each
**`S`-local**, scheduled in elimination order:

- *Elimination (collect).* Eliminating `v` forms `chol(D_v)`, the factor blocks
  `ell_{wv} = A_{wv} D_v^{-1}` for the higher-ordered `S`-neighbours `w` of `v`, and the
  **clique update** `A_{ww'} ← A_{ww'} - ell_{wv} D_v ell_{w'v}^T` over pairs `w, w'` of
  those neighbours. By chordality every such pair `(w,w')` is itself in `S`, so the step
  reads and writes only blocks `(v,v), (w,v), (w,w') ∈ S`.
- *Back-substitution (distribute).* Step `v` of (7) forms `G_{wv}` and `G_{vv}` from `D_v`,
  the `ell_{·v}`, and `G`-blocks at the already-processed (higher) `S`-neighbours of `v`;
  it is `S`-local by the same lemma.

Reverse-mode AD is the reversed composition of the local vector-Jacobian products (VJPs).
Each elementary VJP touches exactly the operand blocks of its forward op (the VJP of
`C = X Y` touches `X, Y, C`; of `E = X^{-1}`, `X, E`); since every forward op is `S`-local,
**every VJP is `S`-local** - in particular the clique-update VJP couples the same `S`-pairs
`(w,w')`. The cotangents therefore never leave `S`. (As in the §8.2 remark, the *closed-form*
dense adjoint `bar_A = - A^{-1} bar_G A^{-1}` genuinely involves off-`S` blocks of `G`; the
point is that the *schedule* - reverse-mode through the elementary sparse operations - never
forms them, exactly as the forward Takahashi computes `G|_S` without forming off-`S` blocks.)

Finally the schedule is two clean sweeps. The forward is a monotone increasing sweep
(elimination) then a monotone decreasing sweep (back-substitution); reverse-mode visits
operations in reverse creation order, giving a reverse-Takahashi sweep (increasing order)
then a reverse-elimination sweep (decreasing order) on the **same** structure, each cotangent
complete before it is consumed (its forward consumers are later in creation order, hence
earlier in reverse) and the forward `(D, L|_S)` reused. With `w_v = |U_v|`, both
directions visit `Theta(1 + w_v^2)` block operations at node `v`, including the
clique-pair updates. Thus both cost `Theta(W b^3)` for
`W = sum_v (1 + w_v^2)`. ∎

**Clique-tree view.** Grouping eliminated variables into the cliques of the chordal
completion makes `S` the pattern of a **tree of cliques** (junction tree), and the schedule
above is the §8.2 two-pass theorem *lifted to super-nodes* - clique residuals as diagonal
blocks, separators as the couplings - which is why the tree argument generalizes verbatim in
structure (the §8.2 proof is block-size agnostic). The within-clique selected blocks are
dense but all lie in `S`.

**Scope.** This establishes the **existence, `S`-closure, self-adjoint structure, and cost**
of the filled-pattern adjoint. The **forward** is now shipped (§2.2,
`gabp_sparse_inv/junction.py`), and because that forward is written as a finite
composition of differentiable `S`-local block ops, **reverse-mode autograd through it is
exactly the self-adjoint schedule of this theorem** - i.e. the adjoint is realized
constructively, validated by `gradcheck` in `tests/test_junction.py`. The *hand-written*
analytic clique recurrence - the explicit analog of §8.3 eqs. (13)-(14) - is now also
shipped (§8.5, `gabp_sparse_inv/junction_autodiff.py`); the tree case (§8.2-8.3) is its
zero-fill specialization. It stores `Theta(F b^2)` numeric factor/selected blocks, but the
current checked level-set implementation also pre-materializes `Theta(W)` symbolic
clique-pair indices. It is therefore tape-free, not a claim of `Theta(F)` total memory for
the shipped path.

### 8.5 The explicit clique recurrences (what `SelInvJunction.backward` runs)

The §8.3 tree recurrences generalize to the filled pattern by replacing the single `parent`
with the clique `U_v` of later-eliminated `S`-neighbours. Write `s(X) = (X+X^T)/2`; reuse the
forward `D_v`, `Dinv_v`, `ell_{wv} = A_{wv} D_v^{-1}` (`w ∈ U_v`) and selected blocks
`G_{vv}, G_{wv}`. `Gget(a,c)` returns the `S`-block `G_{ac}` (transposing the stored lower
orientation when `pos[a] < pos[c]`); `addG(a,c,·)` is its adjoint accumulator. All cotangents
start at zero except `bG_{vv} = barGd[v]` and `bG_{wv} = barGl` (seeded from the output
cotangents). All blocks are `b×b`, batched over leading dims.

**Pass 1 - reverse-distribute (increasing elimination order).** For each `v`, with
`M = s(bG_{vv})`:

    bDinv[v] += M                                                          (15a)
    for u ∈ U_v:   bell[u,v] -= G_{uv} M^T ;   bG_{uv} -= ell_{uv} M       (15b)
    for w ∈ U_v, with c = bG_{wv}, for u ∈ U_v:
        bell[u,v] -= Gget(w,u)^T c ;   addG(w, u, -c · ell_{uv}^T)         (15c)

Increasing order makes every `bG_{wv}` complete before (15c) reads it; its forward consumers,
the higher-clique reads, are reverse-earlier.

**Pass 2 - reverse-collect (decreasing elimination order).** For each `v`:

    # reverse clique-Schur  A_{ww'} -= ell_{wv} D_v ell_{w'v}^T  (pos[w] ≥ pos[w']):
    bt = -(bWd[w] if w==w' else bW[w,w'])
    bell[w,v]  += bt · ell_{w'v} D_v^T                                     (16a)
    bD[v]      += ell_{wv}^T · bt · ell_{w'v}
    bell[w',v] += bt^T · ell_{wv} D_v
    # reverse factor  ell_{wv} = W_{wv} Dinv_v  (W_{wv} = ell_{wv} D_v):
    bW[w,v]  += bell[w,v] Dinv_v ;   bDinv[v] += W_{wv}^T bell[w,v]        (16b)
    # reverse pivot  Dinv_v = D_v^{-1},  D_v = s(Wd_v):
    bD[v]  += -Dinv_v bDinv[v] Dinv_v ;   bWd[v] += s(bD[v])               (16c)
    barAd[v] = bWd[v]

Decreasing order makes each `bWd[w]`, `bW[w,w']` complete before the lower clique that wrote
them (forward) reads them in (16a). Finally `barAe` reads `bW[hi,lo]` at each **input** edge
(fill blocks are not inputs, so their `bW` is internal only). The clique-Schur reversal (16a)
is the one new ingredient over §8.3; on a tree `U_v = {parent}` and (15)-(16) collapse
block-for-block to (13)-(14). Cost is the forward's `Theta(W b^3)` work and
`Theta(F b^2)` numeric block storage, with `F,W` defined in §2.2, and no autograd tape.
The current level-set schedule additionally stores `Theta(W)` symbolic indices.
Validated block-for-block against the
autograd path and by `gradcheck` (`tests/test_junction_autodiff.py`).

## 9. Non-symmetric selected inverse: the lower-bidiagonal case

Sections 1-8 are SPD. The first non-symmetric instance is the **block lower-bidiagonal**
matrix - the non-symmetric analogue of the chain (§4). It is the "easy" triangular case
that validates the non-symmetric forward + adjoint machinery and bridges to the DeltaNet
chunk inverse `(I − A)^{-1}` (§9.4). Implemented in `gabp_sparse_inv/nonsym.py`
(`selected_inverse_bidiag`, `selinv_bidiag`).

### 9.0 What is and is not new (honest accounting)

Two facts are classical and this document claims nothing for them: (i) the inverse of a block
lower-bidiagonal matrix is computed by forward substitution, and (ii) the differential of
a matrix inverse is `dG = -G (dM) G` (Dwyer-Macphail 1948 / Giles 2008, the same identity
(10) used in §8, now without symmetry). The contribution is the same *structural* one as
§8: that the **on-pattern** cotangent of the **selected** inverse is produced by a schedule
of the same shape and `O(n)` cost as the forward - here so directly that both passes are
*local* (no sequential sweep at all), realized as a single batched kernel. It is the
non-symmetric, triangular endpoint of the same-schedule thesis: the reverse has the same
shape and `O(n)` cost, while the inversion derivative obeys
`L_M^*(X) = -M^{-T} X M^{-T} = L_{M^T}(X)` (§10.3). This is an identity between
derivative maps, not a selected-inverse value at the transpose.

### 9.1 Setup

Let `M` be block lower-bidiagonal with diagonal blocks `D_i := M_ii` (general, invertible,
*not* symmetric) and sub-diagonal blocks `C_i := M_{i+1,i}` (`i = 0..n-1`, `C` indexed
`0..n-2`); the super-diagonal is **zero** (`M` is lower-triangular, not symmetric). Write
`G := M^{-1}`, which is block lower-triangular and **dense** below the diagonal. The
selected pattern is `M`'s own - the diagonal and first sub-diagonal:

    G_diag[i]  := G_ii ,        G_lower[i] := G_{i+1,i} .

(The relabeling `M = I − A` makes this `(I − A)^{-1}`; for a *unit*-diagonal `M`
(`D_i = I`) the selected blocks are trivial - `G_ii = I`, `G_{i+1,i} = -C_i` - so the
non-trivial selected inverse is exactly the general-diagonal `M` here, contrast §9.4.)

### 9.2 Forward (fill-free, local)

From `M G = I`, row `i` reads `C_i G_{i-1,j} + D_i G_{ij} = δ_{ij} I`. Because `G` is lower
triangular (`G_{i-1,j} = 0` for `j ≥ i`), the diagonal and first sub-diagonal close on
themselves with **zero fill**:

    G_ii      = D_i^{-1}                                              (15a)
    G_{i+1,i} = - D_{i+1}^{-1} C_i D_i^{-1} = - G_{i+1,i+1} C_i G_ii  (15b)

Every selected block depends only on its endpoint diagonal(s) and the one coupling between
them: there is **no collect/distribute sweep** (contrast the SPD chain, where symmetry
couples the whole path through the Takahashi back-recurrence). The deeper blocks
`G_{ij}` (`i − j ≥ 2`) are nonzero but off-pattern and never formed. Cost: `n` block
inverses and `n−1` pairs of `b×b` products, all batched - `O(n b^3)` time, `O(n b^2)`
storage.

### 9.3 Adjoint (local, `O(n)`)

For a scalar loss `f` of the selected blocks, with cotangents `bGd[i] = ∂f/∂G_ii` and
`bGl[i] = ∂f/∂G_{i+1,i}`, the goal is `∂f/∂D_i` and `∂f/∂C_i` (the autograd-sense gradients
of the kernel's inputs `diag`, `lower`). Reverse-mode through (15) is local. Let
`E_i := G_ii = D_i^{-1}`. Accumulate the cotangent on `E_i`:

    bE[i] = bGd[i]
            - C_i^T E_{i+1}^T bGl[i]          (from G_{i+1,i}, right factor E_i; i ≤ n-2)
            - bGl[i-1] E_{i-1}^T C_{i-1}^T    (from G_{i,i-1}, left factor E_i;  i ≥ 1)    (16a)

then push through the inverse VJP `D_i ↦ E_i = D_i^{-1}`, whose adjoint is
`bar_D = -E^T bar_E E^T` (the non-symmetric (10), no symmetrization - `D_i` is a free
block):

    ∂f/∂D_i = - E_i^T bE[i] E_i^T = - G_ii^T bE[i] G_ii                                    (16b)
    ∂f/∂C_i = - E_{i+1}^T bGl[i] E_i^T = - G_{i+1,i+1}^T bGl[i] G_ii^T                      (16c)

These are exactly the three accumulations and two contractions in
`SelInvBidiag.backward`. The backward touches the same blocks as the forward with the same
`O(n)` cost and is, like the forward, fully local/batched. Its transpose-form VJP is not a
self-adjoint non-symmetric inversion map. There is *no* pivot symmetrization (§6/§8.1): the dense
oracle assembles `M` exactly and inverts it, so analytic and autograd gradients agree to
`~1e-9` (`tests/test_nonsym.py`), without the factor-of-2/transpose bookkeeping that the
symmetric stored-block convention forces.

### 9.4 Dense triangular instance (DeltaNet chunk)

The same construction with a **dense** strictly-lower `A` (so `M = I − A` is unit
lower-triangular, `D_i = I`) yields the chunk inverse `T = (I − A)^{-1}`, the DeltaNet /
gated-DeltaNet primitive. There the selected set is the *full* lower triangle (no sparsity
win) and the adjoint is the same `dT = T (dA) T` restricted to the strictly-lower pattern,

    bar_A = tril(T^T bar_T T^T, -1) .                                                      (17)

It is the explicit bridge to the demonstration ladder, implemented as
`selected_inverse_tril` / `selinv_tril` (`gabp_sparse_inv/nonsym.py`). The
autograd-differentiable triangular solve is its baseline, so the contribution is the
analytic transpose-form VJP (17), not the inverse itself; it is validated against the
dense inverse and against autograd-through-`solve_triangular` to `~1e-15`
(`tests/test_nonsym.py`).

### 9.5 Non-symmetric *tree* instance (zero fill)

The zero-fill non-symmetric rung between the lower-bidiagonal case (§9.2) and the general
LU selected inverse: `M` is a block matrix whose off-diagonal graph is a rooted tree, but
the two directed blocks per edge are **independent**, `M_{p,v} != M_{v,p}^T`. A tree is a
perfect elimination order with no fill, so a block `LDU` factorization in collect order
keeps the tree pattern and the two-sided Takahashi recurrence (the non-symmetric
generalization of the §2/§5.5 SPD-tree passes) gives every selected block exactly. With
pivots `D_v`, lower factor `ellL_v = M_{p,v} D_v^{-1}` and upper factor
`ellU_v = D_v^{-1} M_{v,p}`:

    collect (leaves -> root):  D_v = M_vv − sum_{c in ch(v)} M_{v,c} D_c^{-1} M_{c,v}     (18)
    distribute (root -> leaves):  G_{p,v} = − G_pp ellL_v,    G_{v,p} = − ellU_v G_pp     (19)
                                  G_vv = D_v^{-1} + ellU_v G_pp ellL_v .                  (20)

(18)-(20) follow from the `2x2` block-inverse of the parent-child Schur system and reduce
block-for-block to the SPD-tree kernel when `M_{v,p} = M_{p,v}^T` (then `ellU_v = ellL_v^T`
and `G_{v,p} = G_{p,v}^T`). Each `D_v` must be invertible (general blocks; no SPD/Cholesky
assumption - the pivots use `torch.linalg.inv`). Implemented as
`selected_inverse_nonsym_tree` (`gabp_sparse_inv/nonsym.py`); functional / autograd-
traceable, so first- and higher-order gradients flow by reverse mode with no custom
backward. Validated against the dense `torch.linalg.inv` oracle across path/star/balanced/
random topologies, against the SPD-tree kernel in the symmetric special case, and by
`gradcheck`/`gradgradcheck` and an analytic-vs-dense adjoint gate (`tests/test_nonsym_tree.py`).
This is still a *structured* (zero-fill, tree-pattern) case; it does **not** imply the
general non-symmetric / pivoted LU (Erisman-Tinney) selected inverse - that is §10.

---

## 10. General non-symmetric (filled-pattern LU) selected inverse

The headline non-symmetric rung: `A` is a general block matrix with an **arbitrary
structurally-symmetric** pattern (`(i,j)` present iff `(j,i)` is) but independent directed
blocks (`A_{ij} != A_{ji}^T`). This generalizes the junction kernel (§2.2, §8.4) from SPD to
non-symmetric, and the non-symmetric *tree* (§9.5) from zero-fill to general fill.
Implemented as `selected_inverse_nonsym_junction` (`gabp_sparse_inv/nonsym_junction.py`).

### 10.0 What is and is not new (honest accounting)

The forward is **classical**: the block `LDU` selected inverse on the filled pattern is the
Erisman-Tinney (1975) recurrence (the non-symmetric Takahashi). The contribution is the same
as the rest of the program - the **differentiable** thread: its reverse-mode closes on a
same-pattern schedule. Because `A` is non-symmetric, the inversion derivative is not
self-adjoint: its linear adjoint is the
derivative at `Aᵀ` (the adjoint map is `P_S L_{Aᵀ} ι_S`, not
`P_S L_A ι_S`), with the same `Theta(W b^3)` structural work. This
is why §10.3 propagates **independent** lower/upper cotangents (the `L` of `A` is the `U` of
`Aᵀ`): that independence is the fingerprint of "up to transpose." Established for the
non-symmetric case in §10.3 (the non-symmetric analogue of §8.4). The proof technique
(locality of each block op's VJP + chordality) is standard, which is exactly why it carries
over from the SPD case. (On a structurally-symmetric pattern the selected blocks of `Aᵀ` are
just the transposes of the forward blocks, already in hand - so the adjoint costs no more than
the SPD case.)

### 10.1 Setup

Number nodes by an elimination order; `pos[v]` is `v`'s rank, `U_v` its later-eliminated
neighbours. Because the pattern is structurally symmetric, the chordal completion `S` (the
fill) is **identical** to the SPD case, and `U_v` is a clique in the filled graph (§2.1). The
selected set is all of `S` in **both** orientations (`A^{-1}` is non-symmetric): node
diagonals `G_vv`, and for each `(i,j) in S` both `G_{ij}` and `G_{ji}`.

### 10.2 Forward (block `LDU` + two-sided Takahashi)

With pivots `D_v`, lower factors `L_{wv} = A_{wv} D_v^{-1}` and upper factors
`U_{vw} = D_v^{-1} A_{vw}` (`w in U_v`), after earlier Schur updates have landed in the
working blocks:

    collect (low -> high order):
        D_v       = A_vv  (updated)                                                   (21)
        L_{wv}    = A_{wv} D_v^{-1},   U_{vw} = D_v^{-1} A_{vw}        (w in U_v)       (22)
        A_{w,w'} -= A_{wv} D_v^{-1} A_{vw'}            (w, w' in U_v; both orientations) (23)

    distribute (high -> low order, w in U_v):
        G_{wv}    = − sum_{u in U_v} G_{wu} L_{uv}                                     (24)
        G_{vw}    = − sum_{u in U_v} U_{vu} G_{uw}                                     (25)
        G_vv      =   D_v^{-1} − sum_{u in U_v} U_{vu} G_{uv}                          (26)

Every block (24)-(26) read - `G_{wu}`, `G_{uw}` for `u, w in U_v` - is on `S` because `U_v`
is a clique, and already computed because `u, w` are eliminated after `v` (processed earlier
in the reverse sweep): the same chordality argument as §2.2/§8.4. When `A` is symmetric
(`A_{vw} = A_{wv}^T`) one gets `U_{vw} = L_{wv}^T` and `G_{vw} = G_{wv}^T`, and (21)-(26)
collapse block-for-block to the SPD junction kernel; on a tree (no fill) they collapse to
(18)-(20). Both reductions are asserted in `tests/test_nonsym_junction.py`, alongside a dense
`torch.linalg.inv` oracle on loopy grids/random graphs and an oracle-free `A G = I` residual.

### 10.3 The adjoint is a same-pattern reverse schedule (the inversion derivative at the transpose)

The matrix differential of the inverse is `dG = − G (dA) G` (folklore). Restricting a scalar
loss `f(G|_S)` to the selected blocks, its cotangents `Ā` on `S` are obtained by a **reverse
sweep that touches only `S`-blocks**, reusing the saved `{D_v^{-1}, L, U, G|_S}` in the
reverse of the (21)-(26) schedule. With `w_v = |U_v|`, numeric factor/selected-block
storage is `Theta(F b^2)` and arithmetic is `Theta(W b^3)`; the current checked level-set
path additionally stores `Theta(W)` symbolic metadata. No new fill blocks are formed. The argument is
the §8.4 one verbatim: each line of (21)-(26) is a local block map whose VJP reads and writes
only blocks incident to the clique `U_v ∪ {v}` (all on `S` by chordality), so the adjoint
closes on `S`. The **only** structural difference from the symmetric case is that the lower
and upper factors now carry *independent* cotangents (no `L = U^T` symmetry to fold the two
into one), so the reverse sweep propagates both `L̄` and `Ū` - a constant-factor change, not a
complexity or closure change. At bounded front width and fixed block size this is a
belief-propagation-like `O(n)` adjoint: a two-pass collect/distribute schedule on the same
elimination structure. In general its cost is front-dependent, not linear in the number of
input nonzeros.

In the implementation the forward (21)-(26) is written functionally (the pivots are
`torch.linalg.inv`), so reverse-mode autograd **is** this `S`-local schedule, `gradcheck`-validated
in `tests/test_nonsym_junction.py`. The *hand-written* analytic version
(the tape-free §8.5 analogue) is now also shipped
(`gabp_sparse_inv/junction_autodiff.py`, `SelInvNonsymJunction` / `selinv_nonsym_junction_analytic`):
the reverse two-sweep of §8.5 with the lower/upper factors carrying **independent** cotangents
`bL`, `bU` (no `L = U^T` symmetry to fold them), the pivot reversed by the general inverse VJP
`bD = -Dinv^T bDinv Dinv^T` and no diagonal symmetrization (general blocks). It reduces
block-for-block to the symmetric §8.5 recurrence on symmetric input and is validated against the
autograd path to machine precision and by `gradcheck` (`tests/test_junction_autodiff.py`).

### 10.4 The no-pivot (static-pattern) regime

(21)-(26) eliminate in the fixed symbolic order with **no pivoting**. This keeps the
pattern static and the work `Theta(W b^3)`, but requires every pivot `D_v` (a Schur
complement) to stay nonsingular and numerically usable. The package's block-dominant
generators provide a controlled empirical regime; no general claim is made here that the
particular singular-value dominance diagnostic used by the harness is inherited by every
Schur complement. Partial / threshold pivoting would
restore stability for indefinite or badly-scaled `A`, but it changes the elimination pattern
*dynamically* - destroying the static-`S`, bounded-fill story. So pivoting is deliberately out
of scope and flagged as a research question (`docs/ROADMAP.md`, "Non-symmetric stability"),
not a gap closed here.

The boundary is **measured** in `gabp_sparse_inv/bench/nonsym_stability.py`: sweeping the
block-diagonal-dominance ratio `α = σ_min(A_vv) / Σ_j ||A_vj||` (Feingold-Varga boundary at
`α = 1`), it records the Schur-pivot floor `min_v σ_min(D_v)/||A||` and the fp32 no-pivot
selected-inverse error against a dense fp64 oracle, alongside the fp32 **pivoted** dense LU on
the same blocks. The reading: while dominant (`α ≳ 1`) the no-pivot kernel is at parity with
pivoted LU and tracks `κ·u`; as dominance is lost the pivot floor collapses and the no-pivot
error departs from `κ·u` while pivoted LU does not - the no-pivot penalty, growing past the
dominance boundary (a diagnostic, the §10.4 tradeoff made concrete).

---

## Roadmap pointer

Implemented and tested: **chain** (Phase 1), **star** (Phase 2), **general tree**
(Phase 3 - `gabp_sparse_inv/tree.py`), the **differentiable, level-set-batched adjoint of
selected inversion** on trees (§8 - `gabp_sparse_inv/autodiff.py`, `SelInvTree`/
`selinv_tree`; the §8.2 theorem proved, backward validated to machine precision), and the
**non-symmetric lower-bidiagonal** selected inverse plus the dense `(I − A)^{-1}` DeltaNet
chunk and the **non-symmetric tree** selected inverse (§9 - `gabp_sparse_inv/nonsym.py`),
and the **general-sparse (junction-tree)** selected
inverse on the filled pattern (§2.2, §8.4 - `gabp_sparse_inv/junction.py`,
`selected_inverse_junction`/`selinv_junction`): forward + an autograd adjoint (the §8.4
self-adjoint schedule realized by reverse-mode, `gradcheck`-validated), and the **general
non-symmetric (LU / Erisman-Tinney)** selected inverse on the filled pattern (§10,
`gabp_sparse_inv/nonsym_junction.py`, `selected_inverse_nonsym_junction`/
`selinv_nonsym_junction`): forward + an autograd adjoint evaluated through
`L_A^* = L_{A^T}` (§10.3), no pivoting (the static-pattern regime),
dense-oracle/`gradcheck`-gated.
The hand-written analytic filled-pattern backwards are shipped for both
symmetric and non-symmetric junction kernels; see
[PROJECT_STATUS.md](PROJECT_STATUS.md).
