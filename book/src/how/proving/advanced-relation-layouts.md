# Advanced relation layouts

The [semantic relations in an Akita fold](./akita-fold.md) start with one
commitment group, one witness chunk, and one common ring dimension, while the
[realizations page](./akita-fold-realizations.md) turns those relations into
physical rows. This page develops the multi-group and multi-chunk extensions,
changing one layout axis at a time. Both preserve the four semantic relation
families while changing how their witnesses and matrix columns are organized.

The physical opening-commitment relation remains distinct from the
field-valued evaluation trace. The canonical page for mixed ring dimensions is
linked in [Related layouts](#related-layouts).

## Contents

- [Multiple commitment groups](#multiple-commitment-groups)
  - [Why multiple commitment groups](#why-multiple-commitment-groups)
  - [Group-local folded responses and relations](#group-local-folded-responses-and-relations)
  - [The shared opening-commitment relation](#the-shared-opening-commitment-relation)
  - [Semantic relations and witness layout](#semantic-relations-and-witness-layout)
  - [Return to the single-group recursion](#return-to-the-single-group-recursion)
- [Multiple witness chunks](#multiple-witness-chunks)
  - [Block ownership and partial folded responses](#block-ownership-and-partial-folded-responses)
  - [Semantic relations and chunked witness layout](#semantic-relations-and-chunked-witness-layout)
  - [Raw realization and recursive transition](#raw-realization-and-recursive-transition)
- [Related layouts](#related-layouts)

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

To isolate the group axis, assume one witness chunk, one common ring dimension,
and one polynomial claim per group, then add a group index $g$ to the previous
notation. Group $g$ has its own blocks $\mathbf s_{g,b}$, fold challenges
$c_{g,b}$, and folded response

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

### Semantic relations and witness layout

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

The displayed rows, target vector, and $\mathbf w_0$ are the complete semantic
multi-group relation under this section's single-chunk, common-ring assumptions.
They specify the algebraic variables, row support, and group ordering without
choosing a public payload representation.

#### Raw realization

Raw mode realizes these rows directly: the full $\mathbf u_g$ targets occupy
the group-local $\mathbf B_g$ rows, and $\mathbf v_D$ occupies the shared
$\mathbf D$ rows. Applying the ordinary row-wise quotient lift gives

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
quotient digits for the shared $\mathbf D$ rows. The quotient construction is
unchanged from the [basic realization](./akita-fold-realizations.md#lift-the-physical-ring-relations-before-sumcheck).
The implemented root transition uses compressed payloads, but compression does
not change the semantic multi-group relation above; its $\mathbf F/\mathbf H$
chains are the same realization step already owned by the
[realizations page](./akita-fold-realizations.md) and are not expanded here.

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

## Multiple witness chunks

To isolate the chunk axis, return to one commitment group and one common ring
dimension, but split its live blocks among $C$ witness chunks. Chunking does
not introduce new opening points, fold challenges, relation rows, or public
targets. It changes the semantic witness and the corresponding relation
columns so that each ownership unit covers a disjoint block range.

### Block ownership and partial folded responses

For $B$ live blocks, chunk $c$, where $0\le c<C$, owns the exact range

$$
I_c
=
\left[
\left\lfloor\frac{cB}{C}\right\rfloor,
\left\lfloor\frac{(c+1)B}{C}\right\rfloor
\right).
$$

These ranges partition $[0,B)$ without padding. The current implementation
requires $C$ to be a power of two. If $C>B$, some ranges are empty.

Restrict the original block-indexed opening digits and outer-commitment digits
to $I_c$, giving $\hat{\mathbf e}_c$ and
$\hat{\mathbf t}_c$. The same restriction of the group's fold challenges
produces a partial folded response

$$
\boxed{
\mathbf z_c
=
\sum_{b\in I_c}c_b\mathbf s_b,
\qquad
\mathbf z
=
\sum_{c=0}^{C-1}\mathbf z_c.
}
$$

Every $\mathbf z_c$ lives in the full folded-response space even though it
uses only one block range. Thus each chunk has a full-size
$\hat{\mathbf z}_c$ digit segment, while the
$\hat{\mathbf e}_c$ and $\hat{\mathbf t}_c$ segments partition the original
block-indexed data. An empty chunk retains its full-size
$\hat{\mathbf z}_c$ segment, filled with zero, and has empty opening and
outer-commitment segments.

### Semantic relations and chunked witness layout

Let $\mathbf M_c$ denote the columns of the basic relation assigned to chunk
$c$. Its opening and outer-commitment columns are restricted to $I_c$,
while its consistency and $\mathbf A$ rows reconstruct that chunk's partial
response $\mathbf z_c$. The relation is the horizontal combination

$$
\boxed{
\begin{bmatrix}
\mathbf M_0&\mathbf M_1&\cdots&\mathbf M_{C-1}
\end{bmatrix}
\begin{bmatrix}
\mathbf w_0\\
\mathbf w_1\\
\vdots\\
\mathbf w_{C-1}
\end{bmatrix}
=
\mathbf y,
\qquad
\mathbf w_c
=
[\hat{\mathbf z}_c\mid\hat{\mathbf e}_c\mid\hat{\mathbf t}_c].
}
$$

The row families and public target remain those of the basic single-group
relation:

$$
[\mathrm{consistency}\mid\mathbf A\mid\mathbf B\mid\mathbf D],
\qquad
\mathbf y
=
[0\mid\mathbf 0_{\mathbf A}\mid\mathbf u\mid\mathbf v_D].
$$

For example, the $\mathbf B$ and $\mathbf D$ targets are unchanged because
their chunk-restricted column contributions sum to the original commitments.
Likewise, linearity and $\sum_c\mathbf z_c=\mathbf z$ recover the original
consistency and $\mathbf A$ equations. Chunking therefore preserves the
algebraic statement, but it does not leave the matrix and witness literally
unchanged: both acquire chunk-indexed column segments.

### Raw realization and recursive transition

In raw mode the complete flat witness is chunk-major:

$$
\boxed{
\mathbf w_{\mathrm{raw}}
=
\big\Vert_{c=0}^{C-1}
[\hat{\mathbf z}_c\mid\hat{\mathbf e}_c\mid\hat{\mathbf t}_c]
\;\Vert\;
\hat{\mathbf r}_{\mathrm{ord}}.
}
$$

The quotient tail is shared across all chunks. Each quotient belongs to one
complete semantic row after that row's contributions from every chunk have
been added; there is no quotient copy per chunk. Compressed mode keeps the same
chunked semantic body and likewise has one shared quotient-and-compression
suffix, whose construction is already covered by the
[realizations page](./akita-fold-realizations.md#compressed-realization).

The chunks are columns of one relation and coordinates of one committed
witness, not independent proof executions. Ring switching and Stage 2 reduce
that witness to one opening claim for the next fold. The configured chunked
layout is active only for its selected leading fold levels; later levels return
to the ordinary single-chunk layout.

## Related layouts

This page owns the semantic multi-group and multi-chunk relation layouts. More
detailed address geometry and the remaining layout axis are explained
elsewhere:

- [Chunks and fold challenges](./opening-points-layout.md#chunks-and-fold-challenges)
  records the exact chunk ranges and opening-point coordinates.
- [Canonical walk](../verifying/matrix_evaluation.md#canonical-walk) defines
  chunk-major witness order and the shared quotient and compression suffix.
- [Setup roles and mixed rings](../verifying/matrix_evaluation.md#setup-roles-and-mixed-rings)
  explains native A, B, and D dimensions. The current address authority is
  `RelationAddressGeometry`, including
  `relation_coefficient_block_len()` and
  `outgoing_witness_ring_dimension()` in
  `crates/akita-types/src/proof/relation_address.rs`.
