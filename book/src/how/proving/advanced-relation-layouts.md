# Advanced relation layouts

The [semantic relations in an Akita fold](./akita-fold.md) start with one
commitment group, one witness chunk, and one common ring dimension, while the
[realizations page](./akita-fold-realizations.md) turns those relations into
physical rows. This page extends that basic case along three independent layout
axes:

1. multiple commitment groups add group-local relation rows and witness
   segments to the root fold;
2. multiple witness chunks partition the work and physical witness columns
   within each group; and
3. different ring dimensions let each physical row retain the native ring of
   its matrix role.

None of these extensions changes the four semantic relation families. They
change which physical rows and columns belong to each group, chunk, or native
ring. The physical opening-commitment relation remains distinct from the
field-valued evaluation trace, just as in the basic setting.

## Contents

- [Multiple commitment groups](#multiple-commitment-groups)
  - [Why multiple commitment groups](#why-multiple-commitment-groups)
  - [Group-local folded responses and relations](#group-local-folded-responses-and-relations)
  - [The shared opening-commitment relation](#the-shared-opening-commitment-relation)
  - [Semantic row and witness layout](#semantic-row-and-witness-layout)
  - [Return to the single-group recursion](#return-to-the-single-group-recursion)
- [Multiple witness chunks](#multiple-witness-chunks)
- [Different ring dimensions](#different-ring-dimensions)

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

### Semantic row and witness layout

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
layout. In the compressed realization, each semantic group commitment
$\mathbf u_g$ is compressed through its own $\mathbf F_g$ chain. That chain
adds distinct group-local compression digits and quotient coordinates to the
eventual flat witness; it does not create a separate committed witness for the
group. The single shared commitment $\mathbf v_D$ is compressed once through
one $\mathbf H$ chain, whose compression digits and quotient coordinates form
one level-shared segment in the eventual flat witness. These are the same
constructions as in the basic case, so their expanded rows, right-hand sides,
and witness suffix are omitted here; see the
[realizations page](./akita-fold-realizations.md).

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

> **Status:** stub. This section will derive how each group's live blocks are
> partitioned among witness chunks, how chunk-local segments contribute to the
> same group-level rows, and why all groups and chunks share one quotient tail.

**Sources to fold in**

- `crates/akita-types/src/witness.rs` (`WitnessLayout`, `WitnessUnitLayout`).
- `crates/akita-types/src/proof/ring_relation.rs` (`RingRelationInstance::segment_layout`).
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (group- and chunk-ordered witness emission).
- [Opening points and digit-innermost layout](./opening-points-layout.md#chunks-and-tensor-challenges).
- [The distributed prover](./distributed-prover.md).

## Different ring dimensions

> **Status:** stub. This section will distinguish each relation row's native
> ring dimension from the common carrier dimension, then derive the mixed-row
> quotient and ring-switch evaluation without assuming a common denominator
> $X^D+1$.

**Sources to fold in**

- `crates/akita-types/src/proof/relation.rs` (`RelationRhsLayout::row_ring_dims`).
- `crates/akita-types/src/proof/relation_address.rs` (`RelationAddressGeometry`).
- `crates/akita-types/src/layout/params.rs` (`relation_witness_carrier_ring_dimension`).
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`.
- `crates/akita-verifier/src/protocol/ring_switch/prepared_relation_point.rs`.
- `crates/akita-verifier/src/protocol/ring_switch/relation_evaluation.rs`.

With the logical relation layout established, [Opening points and
digit-innermost layout](./opening-points-layout.md) specifies how its segments
are flattened into the physical source and committed witness order.
