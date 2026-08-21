# Advanced relation layouts

The [semantic relations in an Akita fold](./akita-fold.md) start with one
commitment group, one witness chunk, and one common ring dimension, while the
[realizations page](./akita-fold-realizations.md) turns those relations into
physical rows. This page develops the multi-group and multi-chunk extensions
as two independent axes. Both preserve the four semantic relation families.
Multiple groups add group-local rows around one level-owned D relation, whereas
multiple chunks divide one group's witness columns across block ranges without
duplicating those rows.

The physical opening-commitment relation remains distinct from the
field-valued evaluation trace. This page derives the algebraic chunk layout;
the canonical pages for exact chunk addresses, their physical order, and mixed
ring dimensions are linked in [Related layouts](#related-layouts).

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

### Why multiple witness chunks

The basic fold immediately reduces all live blocks to one folded response

$$
\mathbf z
=
\sum_b c_b\mathbf s_b.
$$

This is algebraically compact, but it creates a synchronization point when the
source blocks are distributed across GPU workgroups, devices, or machines.
Each worker already owns a contiguous block range together with the semantic
inner rows retained in the commitment hint. Forming one outgoing
$\mathbf z$ would require the workers to reduce their full-width responses
before that response could enter the next witness.

A multi-chunk fold retains those local responses instead. A worker can derive
its opening digits and folded-response digits from the same local blocks and
contribute one unit

$$
[\hat{\mathbf z}^{(j)}
 \mid\hat{\mathbf e}^{(j)}
 \mid\hat{\mathbf t}^{(j)}]
$$

to the outgoing witness. The smaller outer- and opening-commitment images can
still be combined across chunks. Linearity then lets the relation sum all
chunk contributions in the original rows. Thus multi-chunking changes the
witness columns, not the semantic row families or their public targets.

The protocol defines chunks as witness block ranges, not as trusted parties or
a required execution schedule. An implementation may map a chunk to one GPU
workgroup, device, or machine. The current prover also retains the aggregate
$\mathbf z$ internally where useful; the protocol-level change is that the
committed witness carries the local responses rather than one global response.

This locality has a cost. The $\hat{\mathbf e}$ and
$\hat{\mathbf t}$ segments are partitioned across chunks and do not grow in
total, but every chunk carries a full-width $\hat{\mathbf z}^{(j)}$ segment.
For $C$ chunks, the live ordinary witness therefore gains
$(C-1)|\hat{\mathbf z}|$ coordinates relative to the single-chunk layout.

### Chunk ranges and commitment-side data

To isolate the chunk axis, assume one commitment group, one polynomial claim,
one common ring

$$
R=F[X]/(X^D+1),
$$

and one unsliced $\mathbf B$ matrix. Let $N$ be the number of live blocks and
let $C\ge 1$ be the chunk count. For this derivation, assume $C\mid N$ and set
$L=N/C$. Chunk $j\in[C]$ owns the equal-sized range

$$
\mathcal I_j
=
\{jL,jL+1,\ldots,(j+1)L-1\}.
$$

These ranges partition the live blocks without padding. The production layout
also supports unequal and empty ranges through the [canonical proportional
partition](./opening-points-layout.md#chunks-and-fold-challenges); that physical
generality does not change the relations below. The transcript still samples
one challenge $c_b$ for every global live block; it does not sample an
independent challenge family for each chunk.

Before the opening query, the incoming commitment has already fixed the
commitment-side data. For every $b\in\mathcal I_j$, the commitment computation
viewed the $b$-th polynomial block as one vector $\mathbf s_b$ and formed its
inner image

$$
\mathbf t_b
=
\mathbf A\mathbf s_b.
$$

Here $\mathbf s_b$ contains all inner-digit ring elements in block $b$; no
additional coordinate indices are needed for this commitment path. Let
$\mathbf G_{\mathrm{out}}$ denote the outer gadget-recomposition map applied
coordinatewise to an inner image. Its bounded digit representation satisfies

$$
\mathbf t_b
=
\mathbf G_{\mathrm{out}}\hat{\mathbf t}_b.
$$

The commitment hint retains the recomposed inner images $\mathbf t_b$, not
materialized outer digits. During the current fold, chunk $j$ decomposes the
images belonging to its range and collects the recovered digits as

$$
\hat{\mathbf t}^{(j)}
=
\mathop{\Vert}_{b\in\mathcal I_j}\hat{\mathbf t}_b.
$$

Let $\mathbf B^{(j)}$ denote the columns of $\mathbf B$ belonging to that
segment. The chunk-local outer image

$$
\mathbf u^{(j)}
=
\mathbf B^{(j)}\hat{\mathbf t}^{(j)}
$$

is an algebraic partial contribution, not a separate payload. The semantic
commitment is

$$
\boxed{
\mathbf u
=
\sum_{j=0}^{C-1}\mathbf u^{(j)}
=
\sum_{j=0}^{C-1}
\mathbf B^{(j)}\hat{\mathbf t}^{(j)}.
}
$$

In compressed mode, this aggregate $\mathbf u$ is the hidden source of one
$\mathbf F$ chain. Multi-chunking does not create one compression payload per
chunk.

### Chunk-local partial evaluations and opening commitment

The opening point supplies the same position weights $Q_p$ as in the basic
derivation. Where this evaluation requires coordinates, write $s_{b,p,a}$ for
the entry of $\mathbf s_b$ at position $p$ and inner digit $a$. For every block
$b\in\mathcal I_j$, chunk $j$ computes

$$
F_{p,b}
=
\sum_aG_a^{\mathrm{in}}s_{b,p,a},
\qquad
E_b
=
\sum_pQ_pF_{p,b},
$$

and decomposes the result as

$$
E_b
=
\sum_hG_h^{\mathrm{open}}\hat e_{b,h}.
$$

Write $\hat{\mathbf e}_b$ for the opening digits of $E_b$ and collect the
blocks belonging to chunk $j$ into $\hat{\mathbf e}^{(j)}$. The complete
opening-digit vector is their concatenation:

$$
\boxed{
\hat{\mathbf e}
=
\hat{\mathbf e}^{(0)}
\Vert\cdots\Vert
\hat{\mathbf e}^{(C-1)}.
}
$$

The partial evaluations $E_b$ and the digit segments
$\hat{\mathbf e}^{(j)}$ remain local to their chunks. Partition the columns of
the opening-commitment matrix along the same ranges,

$$
\mathbf D
=
[\mathbf D^{(0)}\mid\cdots\mid\mathbf D^{(C-1)}].
$$

Chunk $j$ can then compute its opening-commitment contribution directly:

$$
\mathbf v_D^{(j)}
=
\mathbf D^{(j)}\hat{\mathbf e}^{(j)}.
$$

By linearity, the semantic opening commitment is obtained by reducing only
these smaller images:

$$
\boxed{
\mathbf v_D
=
\sum_{j=0}^{C-1}\mathbf v_D^{(j)}
=
\sum_{j=0}^{C-1}
\mathbf D^{(j)}\hat{\mathbf e}^{(j)}.
}
$$

Thus a distributed prover does not aggregate the block-indexed partial
evaluations or their digit segments. Each worker first applies its local
columns of $\mathbf D$, and the workers aggregate only the resulting
opening-commitment images. Like $\mathbf u^{(j)}$, the partial image
$\mathbf v_D^{(j)}$ is not a public payload. Raw mode exposes the aggregate
$\mathbf v_D$; compressed mode uses that same aggregate as the hidden source
of one $\mathbf H$ chain. Multi-chunking does not create one compression chain
per chunk.

### Chunk-local folded responses and witness relations

The payload binding $\mathbf u$ and $\mathbf v_D$ is fixed before the
transcript samples the fold challenges $c_b$. Computing the basic global
response at this point would require the workers to reduce a full-width
vector. Multi-chunking instead keeps the response local.

Chunk $j$ then folds only its own source blocks:

$$
\boxed{
\mathbf z^{(j)}
=
\sum_{b\in\mathcal I_j}c_b\mathbf s_b.
}
$$

Each $\mathbf z^{(j)}$ has the same full ambient width as the single response
$\mathbf z$. It is a partial sum over blocks, not a width slice. Decompose it
with the ordinary fold gadget. Writing $\mathbf G_{\mathrm{fold}}$ for its
coordinatewise recomposition map gives

$$
\mathbf z^{(j)}
=
\mathbf G_{\mathrm{fold}}\hat{\mathbf z}^{(j)}.
$$

Together with the local opening and outer-commitment digits, this gives the
chunk-local witness unit

$$
\boxed{
\mathbf w^{(j)}
=
[\hat{\mathbf z}^{(j)}
 \mid\hat{\mathbf e}^{(j)}
 \mid\hat{\mathbf t}^{(j)}].
}
$$

Within the ordinary witness prefix, the three segments of each unit are
contiguous, and the units themselves are concatenated in chunk order:

$$
\boxed{
\mathbf w_{0,\mathrm{chunk}}
=
\big\Vert_{j=0}^{C-1}\mathbf w^{(j)}.
}
$$

This is the portion of the outgoing witness contributed by the chunks. When
the complete current-level witness is committed as the next level's
polynomial, these units remain contiguous source ranges inside the flat
witness. The shared quotient and compression suffixes are appended afterward,
so a current chunk unit need not coincide with exactly one next-level block.

The local responses still define the same global folded response
algebraically:

$$
\boxed{
\mathbf z
=
\sum_{j=0}^{C-1}\mathbf z^{(j)}
=
\sum_{j=0}^{C-1}
\mathbf G_{\mathrm{fold}}\hat{\mathbf z}^{(j)}.
}
$$

The global $\mathbf z$ is therefore determined by the chunk-local witnesses,
but it is not an additional committed coordinate and need not be materialized
through a full-width reduction before constructing the outgoing witness. If a
production chunk range is empty, its E and T segments are empty and the honest
prover puts zero in its full-width Z segment.

For an honestly constructed chunk, the same linear derivation as in the basic
case can be performed over its local block range. Evaluation within each block
and then folding gives the recomposed identity, where $z_{p,a}^{(j)}$ denotes
the corresponding coordinate of $\mathbf z^{(j)}$:

$$
\begin{aligned}
\sum_{b\in\mathcal I_j}c_bE_b
&=
\sum_{b\in\mathcal I_j,p,a}
c_bQ_pG_a^{\mathrm{in}}s_{b,p,a}
\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}^{(j)}.
\end{aligned}
$$

Likewise, applying $\mathbf A$ before or after the local fold gives the second
recomposed identity

$$
\sum_{b\in\mathcal I_j}c_b\mathbf t_b
=
\mathbf A\mathbf z^{(j)}.
$$

These local identities explain how an honest distributed prover constructs
each witness unit. The protocol does not add a separate consistency row or set
of $\mathbf A$ rows for every chunk. Instead, it proves their aggregate in the
same rows as the basic relation. Only at this aggregate step do we substitute
the digit decompositions carried by the witness.

To keep the aggregate equations at the block-vector level, let
$\mathbf G_{\mathrm{in}}$ recompose the inner digits of every position, let
$\mathbf Q$ apply the position weights $Q_p$, and let
$\mathbf G_{\mathrm{open}}$ recompose one block's opening digits. The
$\mathbf G$ maps are the matrix forms of the gadget weights already used
above, while $\mathbf Q$ is the row map defined by the opening weights. The
fold-evaluation consistency relation is then

$$
\boxed{
\sum_{j=0}^{C-1}
\sum_{b\in\mathcal I_j}
c_b\mathbf G_{\mathrm{open}}\hat{\mathbf e}_b
=
\sum_{j=0}^{C-1}
\mathbf Q\mathbf G_{\mathrm{in}}\mathbf G_{\mathrm{fold}}
\hat{\mathbf z}^{(j)}.
}
$$

In vector form, inner-commitment consistency is

$$
\boxed{
\sum_{j=0}^{C-1}
\sum_{b\in\mathcal I_j}
c_b\mathbf G_{\mathrm{out}}\hat{\mathbf t}_b
=
\sum_{j=0}^{C-1}
\mathbf A\mathbf G_{\mathrm{fold}}
\hat{\mathbf z}^{(j)}.
}
$$

The other two families are the aggregate commitment relations already defined
above:

$$
\boxed{
\sum_{j=0}^{C-1}
\mathbf B^{(j)}\hat{\mathbf t}^{(j)}
=
\mathbf u,
}
$$

$$
\boxed{
\sum_{j=0}^{C-1}
\mathbf D^{(j)}\hat{\mathbf e}^{(j)}
=
\mathbf v_D.
}
$$

Thus the multi-chunk construction preserves one consistency row, the original
$\mathbf A$, $\mathbf B$, and $\mathbf D$ row families, and the original
semantic target. It does not claim that each chunk-local identity is proved
independently; chunks are column ranges inside one prover's aggregate relation.

### Logical witness and physical realization

The chunk-major prefix defined above is the chunk-local core of the outgoing
witness. After the shared quotient and, in compressed mode, compression
segments described below are appended, the next fold views the complete flat
outgoing witness as a new polynomial and partitions it into new block vectors
$\mathbf s'_0,\ldots,\mathbf s'_{N'-1}$. A next-level chunk owns a range
$\mathcal I'_j$ of those blocks; one current unit $\mathbf w^{(j)}$ need not be
exactly one next-level block.

If chunked execution continues, each worker can commit its assigned next-level
blocks by the same path that produced the current commitment-side data:

$$
\mathbf t'_b
=
\mathbf A'\mathbf s'_b,
\qquad
\mathbf t'_b
=
\mathbf G'_{\mathrm{out}}\hat{\mathbf t}'_b,
\qquad b\in\mathcal I'_j.
$$

Collecting the local outer digits into $\hat{\mathbf t}'^{(j)}$, the worker
forms a partial outer image and the workers reduce only those smaller images:

$$
\mathbf u'^{(j)}
=
\mathbf B'^{(j)}\hat{\mathbf t}'^{(j)},
\qquad
\boxed{
\mathbf u'
=
\sum_{j=0}^{C-1}\mathbf u'^{(j)}.
}
$$

Thus the current chunk witness becomes distributed source data for the next
level without first reducing the full-width $\hat{\mathbf z}^{(j)}$ segments.
Only the smaller commitment images are combined. The primes distinguish the
next level's blocks, matrices, digits, and commitment from the current level's
objects.

Let $\mathbf M^{(j)}$ be the full-height relation column block acting on
$\mathbf w^{(j)}$. The $\hat{\mathbf e}^{(j)}$ and
$\hat{\mathbf t}^{(j)}$ columns restrict the original block-indexed
coefficients to $\mathcal I_j$, while the same folded-response operators act
on every full-width $\hat{\mathbf z}^{(j)}$ segment. The complete matrix is the
horizontal concatenation

$$
\mathbf M_{\mathrm{chunk}}
=
[\mathbf M^{(0)}\mid\cdots\mid\mathbf M^{(C-1)}],
$$

and its semantic statement is

$$
\boxed{
\mathbf M_{\mathrm{chunk}}\mathbf w_{0,\mathrm{chunk}}
=
\sum_{j=0}^{C-1}\mathbf M^{(j)}\mathbf w^{(j)}
=
\mathbf y,
}
$$

with the same raw target as the basic relation:

$$
\mathbf y
=
[0\mid\mathbf 0_{\mathbf A}\mid\mathbf u\mid\mathbf v_D].
$$

Raw mode appends one ordinary quotient family after all chunk units:

$$
\boxed{
\mathbf w_{\mathrm{raw}}
=
\mathbf w_{0,\mathrm{chunk}}
\;\Vert\;
\hat{\mathbf r}_{\mathrm{ord}}.
}
$$

There is one quotient polynomial per physical row, not one per chunk. Under
this section's common-ring assumption, row $i$ satisfies

$$
\sum_{j=0}^{C-1}
\widetilde{\mathbf M}^{(j)}_i(X)
\widetilde{\mathbf w}^{(j)}(X)
-
\widetilde y_i(X)
=
(X^D+1)r_i(X).
$$

Compressed mode keeps the same chunked ordinary prefix and shared ordinary
quotients. It then adds the single $\mathbf F$ chain for the aggregate
$\mathbf u$ and the single $\mathbf H$ chain for the aggregate
$\mathbf v_D$, using the physical order defined on the realizations and
opening-layout pages. Neither compression chain is replicated per chunk.

Finally, chunking does not turn the scalar evaluation claim into a physical
ring row. For the one `EvaluationTrace` claim in this section, the virtual
relation is simply distributed over the chunk-local E segments:

$$
v_{\mathrm{tr}}
=
\sum_{j=0}^{C-1}
\sum_{b\in\mathcal I_j,h,\ell}
\hat e_{b,h,\ell}B_bG_h^{\mathrm{open}}J_\ell.
$$

It remains one field-valued Stage-2 relation with no ring-switch quotient. When
$C=1$, the sole range contains every live block, the single local response is
$\mathbf z$, and every equation and witness layout above reduces to the basic
single-chunk construction.

## Related layouts

This page owns the multi-group relation layout and the single-group
multi-chunk relation derivation. Exact physical geometry and the other layout
axes have canonical explanations elsewhere:

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
