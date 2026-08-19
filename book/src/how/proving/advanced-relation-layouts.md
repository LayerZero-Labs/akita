# Advanced relation layouts

The [semantic relations in an Akita fold](./akita-fold.md) start with one
commitment group, one witness chunk, and one common ring dimension, while the
[realizations page](./akita-fold-realizations.md) turns those relations into
physical rows. This page develops the multi-group extension. It preserves the
four semantic relation families and explains how group-local rows and witness
segments combine with one level-owned D relation.

The physical opening-commitment relation remains distinct from the
field-valued evaluation trace. The canonical pages for chunks and mixed ring
dimensions are linked in [Related layouts](#related-layouts).

## Multiple commitment groups

### Why multiple commitment groups

In zkVM applications, polynomials such as execution traces, advice, and
preprocessed data may be committed at different times. When one commitment is
formed, the prover may not yet know which other commitments it will later be
opened with, the opening point, or the root schedule that will combine them.
Akita calls the polynomials held under one independently formed outer
commitment a **commitment group**. The layout associates one complete opening
point with each group, shared by all claims in that group. Different groups may
be opened at different points. One group is the **final/new group**, while
commitments formed earlier enter as **precommitted groups** with already-fixed
parameters.

Proving each commitment group separately would repeat the entire recursive
opening protocol. Akita instead batches the groups in one root transition and
then resumes the ordinary single-opening recursion. The batching preserves
the separate group commitments and folded responses: each group has its own
$\mathbf z_g$, $\hat{\mathbf t}_g$, and group-local `consistency | A | B`
relations. On the opening side, however, every group contributes an
$\hat{\mathbf e}_g$ segment to one logically concatenated vector. One
$\mathbf D$ matrix binds that entire vector, and the field-level evaluation
trace separately batches the claimed evaluations at their group-local points.

### Group-local folded responses and relations

To show only what changes from the basic setting, assume one polynomial claim
per group and add a group index $g$ to the previous notation. Group $g$ has
its own blocks $\mathbf s_{g,b}$, fold challenges $c_{g,b}$, and folded response

$$
\boxed{
\mathbf z_g
=
\sum_b c_{g,b}\mathbf s_{g,b}.
}
$$

There is no sum over $g$: the folded responses $\mathbf z_g$ remain separate.
The basic consistency, $\mathbf A$, and $\mathbf B$ relations are repeated
independently for every group. In the semantic relation, the $\mathbf B_g$
rows bind $\hat{\mathbf t}_g$ to that group's outer commitment $\mathbf u_g$.
Each group also produces opening digits $\hat{\mathbf e}_g$, which become one
segment of the shared opening vector.

The fold and outer-commitment parts remain group-local because each commitment
fixes its own $\mathbf A_g$ and $\mathbf B_g$ matrices, decomposition
parameters, and semantic target $\mathbf u_g$. Its folded response
$\mathbf z_g$ is formed with that group's challenges and must be checked
against those fixed parameters. Combining the responses across groups would
lose these group-specific commitment bindings.

### The shared opening-commitment relation

The $\mathbf D$ relation can be shared for a different reason. Unlike
$\mathbf A_g$ and $\mathbf B_g$, $\mathbf D$ is owned by the fold level rather
than by an individual commitment group. Every $\hat{\mathbf e}_g$ segment uses
the same opening-role ring dimension and decomposition basis, so the segments
can occupy disjoint column ranges of the single opening-commitment matrix
$\mathbf D$ and form one input vector:

$$
\hat{\mathbf e}_{\mathrm{all}}
=
\big\Vert_{g\in\mathrm{relation\ order}}\hat{\mathbf e}_g,
\qquad
\mathbf D\hat{\mathbf e}_{\mathrm{all}}
=
\mathbf v_D.
$$

Here the concatenation is logical and follows relation order: the final/new
group first, then the precommitted groups in public order. It combines only the
opening-digit inputs to $\mathbf D$; the original group witnesses, folded
responses, and outer-commitment digits remain group-local. The fixed column
ranges record which coordinates came from each group, and the group-local
consistency rows prove what each $\hat{\mathbf e}_g$ segment represents at that
group's own opening point.

### Semantic relations and physical realizations

In the canonical semantic-row order, the final/new group is placed first,
followed by the precommitted groups. The shared $\mathbf D$ rows follow all
group-local rows:

$$
\begin{aligned}
{}&
[\mathrm{consistency}_{\mathrm{final}}
 \mid \mathbf A_{\mathrm{final}}
 \mid \mathbf B_{\mathrm{final}}]
\\[-2pt]
&\quad\Vert
\big\Vert_{g\in\mathrm{precommitted}}
[\mathrm{consistency}_g\mid\mathbf A_g\mid\mathbf B_g]
\quad\Vert\quad
\mathbf D.
\end{aligned}
$$

The semantic target vector is therefore

$$
\mathbf y
=
\big\Vert_{g\in\mathrm{relation\ order}}
[0\mid\mathbf 0_{\mathbf A_g}\mid\mathbf u_g]
\quad\Vert\quad
\mathbf v_D.
$$

The semantic layout is block sparse: a group's consistency, $\mathbf A_g$, and
$\mathbf B_g$ rows touch only that group's witness segment, whereas the shared
$\mathbf D$ rows touch the $\hat{\mathbf e}_g$ segments from every group.
Stage 2 batches the realized physical rows into one sumcheck, but this batching
does not merge the group-local relations.

The corresponding logical witness layout is

$$
\mathbf w_0
=
\big\Vert_{g\in\mathrm{relation\ order}}
[\hat{\mathbf z}_g\mid\hat{\mathbf e}_g\mid\hat{\mathbf t}_g].
$$

The displayed rows, target vector, and $\mathbf w_0$ describe only the semantic
relation. Raw and compressed payloads realize it with different physical rows
and flat witness suffixes.

Both realizations use the row-wise quotient lift from [Raw and compressed
physical realizations](./akita-fold-realizations.md#lift-the-physical-ring-relations-before-sumcheck).
Every physical row has one quotient polynomial in that row's native ring. Its
digit decomposition becomes part of the flat witness, and evaluation at the
ring-switch point multiplies it by the row-specific denominator
$\alpha^{d_i}+1$. The construction below changes only which rows and witness
segments are present, so the quotient derivation is not repeated here.

#### Raw realization

Raw mode keeps exactly the semantic row order displayed above and places the
full $\mathbf u_g$ targets on the group-local $\mathbf B_g$ rows and
$\mathbf v_D$ on the shared $\mathbf D$ rows. It adds no compression rows.
After digit-decomposing the quotient of every ordinary row, the flat witness is

$$
\boxed{
\mathbf w_{\mathrm{raw}}
=
\mathbf w_0
\;\Vert\;
\hat{\mathbf r}_{\mathrm{ord}}.
}
$$

Here $\hat{\mathbf r}_{\mathrm{ord}}$ follows the same canonical row order:
each group's `consistency | A | B` quotient digits first, followed by the
quotient digits for the shared $\mathbf D$ rows. It is one contiguous physical
suffix, but each quotient-row span retains that row's native ring dimension.

#### Compressed realization

Compressed mode replaces each semantic $\mathbf u_g$ by its own
$\mathbf F_g$ compression chain and replaces the single shared
$\mathbf v_D$ by one $\mathbf H$ chain. The ordinary $\mathbf B_g$ and
$\mathbf D$ right-hand sides become zero because their outputs recompose from
the first compression-digit layer. For each compression layer, the physical
row order is every group-local $\mathbf F_g$ row in relation order followed by
the one shared $\mathbf H$ row. Only the terminal rows carry the public
compressed payloads. The chain equations are the same as in the basic case and
are not expanded again here.

Suppressing the derived zero-alignment ranges, the flat witness layout for the
two compression layers is

$$
\boxed{
\begin{aligned}
\mathbf w_{\mathrm{comp}}
={}&
\mathbf w_0
\Vert
\hat{\mathbf r}_{\mathrm{ord}}
\\
&\Vert
\mathop{\Big\Vert}_{\ell=1}^{2}
\left[
\left(
\big\Vert_{g\in\mathrm{relation\ order}}
\boldsymbol\xi_{F_g,\ell}
\right)
\Vert
\boldsymbol\xi_{H,\ell}
\right.
\\[-2pt]
&\hspace{92pt}\left.
\Vert
\left(
\big\Vert_{g\in\mathrm{relation\ order}}
\hat{\mathbf r}_{F_g,\ell}
\right)
\Vert
\hat{\mathbf r}_{H,\ell}
\right].
\end{aligned}
}
$$

Thus one layer stores all group-local $\mathbf F$ digits, the shared
$\mathbf H$ digits, the corresponding group-local $\mathbf F$ quotient digits,
and the shared $\mathbf H$ quotient digits, in that order. When there is only
one group, suppressing alignment reduces this expression exactly to the basic
one-group compressed layout. The ordering is by relation family and compression
layer rather than by a global sort on ring dimension: every quotient and
compression span still retains its own native dimension.

### Return to the single-group recursion

The root fold consumes this multi-group semantic structure. After the semantic
rows have been realized and lifted as described on the previous page, their
witness coordinates form one complete flat ring-switch witness. That witness
is aligned to the successor commitment ring dimension and committed once.
Stage 2 evaluates it at its sumcheck-derived opening point, producing one
witness-side opening claim for the next fold.

Thus the contribution descended from the original root batch is one committed
witness and one opening claim. Its coordinates retain the original group ranges
inside the flat layout, but those root groups no longer define separate folded
responses or relation rows. If setup offloading is active, the next fold may
also receive an independent setup-prefix group; that extra group does not
preserve or recreate the original root grouping.

## Related layouts

This page owns the multi-group relation layout. The other layout axes have
canonical explanations elsewhere:

- [Chunks and fold challenges](./opening-points-layout.md#chunks-and-fold-challenges)
  defines exact chunk ranges and opening-point coordinates.
- [Canonical walk](../verifying/matrix_evaluation.md#canonical-walk) defines
  chunk-major witness order and the shared quotient and compression suffix.
- [Setup roles and mixed rings](../verifying/matrix_evaluation.md#setup-roles-and-mixed-rings)
  explains native A, B, and D dimensions. The current address authority is
  `RelationAddressGeometry`, including
  `relation_coefficient_block_len()` and
  `outgoing_witness_ring_dimension()` in
  `crates/akita-types/src/proof/relation_address.rs`.
