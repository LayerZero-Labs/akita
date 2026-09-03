# Advanced relation layouts

The [semantic relations in an Akita fold](./akita-fold.md) start with one
commitment group, one witness chunk, and one common ring dimension, while the
[realizations page](./akita-fold-realizations.md) turns those relations into
physical rows. This page develops three independent extensions: multiple
commitment groups, multiple witness chunks, and mixed commitment-ring
dimensions. All three preserve the four semantic relation families. Multiple
groups add group-local rows around one level-owned D relation, multiple chunks
divide one group's witness columns across block ranges without duplicating
those rows, and mixed rings assign each row family its role-native dimension.

The physical opening-commitment relation remains distinct from the
field-valued evaluation trace. Each section isolates one extension under its
own simplifying assumptions before explaining how it composes with the other
axes.

## Contents

- [Multiple commitment groups](#multiple-commitment-groups)
  - [Why multiple commitment groups](#why-multiple-commitment-groups)
  - [Group-local folded responses and relations](#group-local-folded-responses-and-relations)
  - [The shared opening-commitment relation](#the-shared-opening-commitment-relation)
  - [Semantic relations and witness layout](#semantic-relations-and-witness-layout)
  - [Return to the single-group recursion](#return-to-the-single-group-recursion)
- [Multiple witness chunks](#multiple-witness-chunks)
  - [Why multiple witness chunks](#why-multiple-witness-chunks)
  - [Chunk ranges and commitment-side data](#chunk-ranges-and-commitment-side-data)
  - [Chunk-local partial evaluations and opening commitment](#chunk-local-partial-evaluations-and-opening-commitment)
  - [Chunk-local folded responses and witness relations](#chunk-local-folded-responses-and-witness-relations)
  - [Semantic relations remain unchanged](#semantic-relations-remain-unchanged)
- [Mixed commitment-ring dimensions](#mixed-commitment-ring-dimensions)
  - [Why use different ring dimensions](#why-use-different-ring-dimensions)
  - [Role-native projection and decomposition](#role-native-projection-and-decomposition)
  - [The four relations in their native rings](#the-four-relations-in-their-native-rings)
  - [Lift and switch the native rows](#lift-and-switch-the-native-rows)
  - [Relation to compressed realization](#relation-to-compressed-realization)
  - [Composition with groups and chunks](#composition-with-groups-and-chunks)

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

Raw and compressed modes realize this semantic layout as described on the
[realizations page](./akita-fold-realizations.md). Multi-grouping changes the
group-local row blocks and the input layout of the shared $\mathbf D$ relation,
but not the ordinary quotient or compression machinery.

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

| Object | Effect of multi-chunking |
|---|---|
| $\hat{\mathbf z}$ | Replaced by $C$ full-width local responses $\hat{\mathbf z}^{(j)}$ |
| $\hat{\mathbf e},\hat{\mathbf t}$ | Partitioned by block range; total width is unchanged |
| $\mathbf u,\mathbf v_D$ | Remain aggregate commitment images |
| Semantic rows and targets | Remain unchanged |

### Chunk ranges and commitment-side data

To isolate the chunk axis, assume one commitment group, one polynomial claim,
the `EvaluationTrace` opening method, one unsliced $\mathbf B$ matrix, and one
common ring

$$
R=F[X]/(X^D+1).
$$

Commitment slicing is a separate axis. It uses the same proportional partition,
as described in [B slices and chunks](./opening-points-layout.md#b-slices-and-chunks).

Let $N$ be the number of live blocks. Let $C$ be a supported power-of-two chunk
count and, for this derivation, assume $C\mid N$. For $j=0,\ldots,C-1$, chunk
$j$ owns the equal-sized range

$$
\mathcal I_j
=
\left\{\frac{jN}{C},\frac{jN}{C}+1,\ldots,
\frac{(j+1)N}{C}-1\right\}.
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
\big\Vert_{b\in\mathcal I_j}\hat{\mathbf t}_b.
$$

The complete outer-digit vector and the corresponding column partition of
$\mathbf B$ are

$$
\hat{\mathbf t}
=
\hat{\mathbf t}^{(0)}\Vert\cdots\Vert\hat{\mathbf t}^{(C-1)},
\qquad
\mathbf B
=
[\mathbf B^{(0)}\mid\cdots\mid\mathbf B^{(C-1)}].
$$

The chunk-local outer image

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
\mathbf B^{(j)}\hat{\mathbf t}^{(j)}
=
\mathbf B\hat{\mathbf t}.
}
\tag{1}
$$

Raw mode exposes the aggregate $\mathbf u$. In compressed mode, the
chunk-local images are first reduced to this single $\mathbf u$; only then is
$\mathbf u$ used as the hidden source of one shared $\mathbf F$ chain.
Multi-chunking does not create one compression payload per chunk.

### Chunk-local partial evaluations and opening commitment

The opening point supplies the same opening vector $(Q_p)_p$ as in the basic
derivation. Let $\mathbf G_{\mathrm{in}}$ recompose the inner digits at every
position of a block, and let $\mathbf Q$ be the row map induced by the weights
$Q_p$. For every block $b\in\mathcal I_j$, chunk $j$ computes the partial
evaluation directly from its block vector and decomposes the result:

$$
E_b
=
\mathbf Q\mathbf G_{\mathrm{in}}\mathbf s_b
=
\mathbf G_{\mathrm{open}}\hat{\mathbf e}_b.
$$

Here $\mathbf G_{\mathrm{open}}$ recomposes the opening digits
$\hat{\mathbf e}_b$ of $E_b$. Collect the blocks belonging to chunk $j$ into
$\hat{\mathbf e}^{(j)}$. The complete opening-digit vector is their
concatenation:

$$
\hat{\mathbf e}
=
\hat{\mathbf e}^{(0)}
\Vert\cdots\Vert
\hat{\mathbf e}^{(C-1)}.
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
\mathbf D^{(j)}\hat{\mathbf e}^{(j)}
=
\mathbf D\hat{\mathbf e}.
}
\tag{2}
$$

Thus a distributed prover does not aggregate the block-indexed partial
evaluations or their digit segments. Each worker first applies its local
columns of $\mathbf D$, and the workers aggregate only the resulting
opening-commitment images. Like $\mathbf u^{(j)}$, the partial image
$\mathbf v_D^{(j)}$ is not a public payload. Raw mode exposes the aggregate
$\mathbf v_D$. In compressed mode, the workers likewise first reduce their
partial images to the single $\mathbf v_D$; only then is $\mathbf v_D$ used as
the hidden source of one shared $\mathbf H$ chain. Multi-chunking does not
create one compression chain per chunk.

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
\tag{3}
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
\mathbf w^{(j)}
=
[\hat{\mathbf z}^{(j)}
 \mid\hat{\mathbf e}^{(j)}
 \mid\hat{\mathbf t}^{(j)}].
$$

Within the logical witness for the four ordinary semantic relation families,
the three segments of each unit are contiguous, and the units themselves are
concatenated in chunk order:

$$
\mathbf w
=
\big\Vert_{j=0}^{C-1}\mathbf w^{(j)}.
$$

Thus $\mathbf w$ is the complete chunk-major logical witness for those four
families, not one separately proved witness per chunk. The physical
realizations extend it with the shared ordinary quotients and, in compressed
mode, the compression data described on the [realizations
page](./akita-fold-realizations.md).

The local responses still define the same global folded response
algebraically:

$$
\mathbf z
=
\sum_{j=0}^{C-1}\mathbf z^{(j)}
=
\sum_{j=0}^{C-1}
\mathbf G_{\mathrm{fold}}\hat{\mathbf z}^{(j)}.
$$

The global $\mathbf z$ is therefore determined by the chunk-local witnesses,
but it is not an additional committed coordinate and need not be materialized
through a full-width reduction before constructing the outgoing witness.
The [distributed prover](./distributed-prover.md#ring-switch-lift-and-next-level-commitment)
explains how the workers combine partial commitments to that witness.

The basic $\hat{\mathbf z}$ is not a concatenation or reordering of the local
$\hat{\mathbf z}^{(j)}$ segments. Recovering that digit vector would require
materializing the global $\mathbf z$ and decomposing it again; the multi-chunk
relation avoids that full-width reduction by acting on the local digit vectors
directly.

This aggregation must also be reflected in weak binding. Stage 1 checks every
$\hat{\mathbf z}^{(j)}$ against the same balanced digit interval, but the shared
A rows bind $\mathbf z_\Sigma=\sum_j\mathbf z^{(j)}$. For basis $b$ and fold
depth $\delta$, one accepted chunk has exact difference diameter
$b^\delta-1$; $C$ accepted chunks therefore have aggregate diameter
$C(b^\delta-1)$. The A-role SIS row is sized for this aggregate envelope, not
for a single chunk. Honest chunk norms may differ because the block ranges are
ragged, but every chunk keeps the same basis, depth, and full ambient Z width.

Because the ranges $\mathcal I_j$ partition the live blocks, this derived
$\mathbf z$ still satisfies the two recomposed identities from the basic
setting:

$$
\begin{gathered}
\sum_{j=0}^{C-1}
\sum_{b\in\mathcal I_j}
c_bE_b
=
\mathbf Q\mathbf G_{\mathrm{in}}\mathbf z,
\\
\sum_{j=0}^{C-1}
\sum_{b\in\mathcal I_j}
c_b\mathbf t_b
=
\mathbf A\mathbf z.
\end{gathered}
$$

Substituting the digit decompositions of $E_b$, $\mathbf t_b$, and the derived
$\mathbf z$ gives the aggregate fold-evaluation consistency relation

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
\tag{4}
$$

The same substitution gives aggregate inner-commitment consistency:

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
\tag{5}
$$

### Semantic relations remain unchanged

For the complete chunked witness $\mathbf w$, the four proved semantic
relations can be read together in canonical row-family order:

$$
\begin{aligned}
\sum_{j=0}^{C-1}
\left(
\sum_{b\in\mathcal I_j}
c_b\mathbf G_{\mathrm{open}}\hat{\mathbf e}_b
-
\mathbf Q\mathbf G_{\mathrm{in}}\mathbf G_{\mathrm{fold}}
\hat{\mathbf z}^{(j)}
\right)
&=0,
&&\text{fold-evaluation consistency},
\\
\sum_{j=0}^{C-1}
\left(
\sum_{b\in\mathcal I_j}
c_b\mathbf G_{\mathrm{out}}\hat{\mathbf t}_b
-
\mathbf A\mathbf G_{\mathrm{fold}}\hat{\mathbf z}^{(j)}
\right)
&=\mathbf 0_{\mathbf A},
&&\text{inner-commitment consistency},
\\
\sum_{j=0}^{C-1}
\mathbf B^{(j)}\hat{\mathbf t}^{(j)}
&=\mathbf u,
&&\text{outer-commitment consistency},
\\
\sum_{j=0}^{C-1}
\mathbf D^{(j)}\hat{\mathbf e}^{(j)}
&=\mathbf v_D,
&&\text{opening-commitment consistency}.
\end{aligned}
$$

These are the same four semantic relations as in the basic setting. Their
semantic target remains

$$
\mathbf y
=
[0\mid\mathbf 0_{\mathbf A}\mid\mathbf u\mid\mathbf v_D].
$$

Each relation sums the relevant chunk-local witness contributions inside the
original rows. Multi-chunking does not create a separately proved row family
for every chunk; it changes the witness layout and the column support of those
rows.

Raw and compressed modes realize these same semantic relations as described on
the [realizations page](./akita-fold-realizations.md). Chunking does not
replicate the ordinary quotient family or the shared $\mathbf F/\mathbf H$
compression chains. The field-valued evaluation trace likewise remains one
virtual relation over the chunk-local opening-digit ranges, not an additional
physical ring row; its construction is described in [Field-to-ring evaluation
reduction](./field-ring-reduction.md#express-the-direct-relation-as-a-sumcheck-claim).

When $C=1$, the sole range contains every live block, the single local response
is $\mathbf z$, and every equation and witness layout above reduces to the
basic single-chunk construction.

## Mixed commitment-ring dimensions

### Why use different ring dimensions

The common-ring derivation gives $\mathbf A$, $\mathbf B$, and $\mathbf D$ one
dimension because that is the simplest setting in which to see the four
semantic relations. The three matrices perform different jobs, however, and
need not have the same best dimension. The $\mathbf A$ matrix carries the
source and folded witness, the $\mathbf B$ matrix binds the outer-commitment
digits, and the $\mathbf D$ matrix binds the opening digits. A larger
$\mathbf A$ ring may improve fold geometry or shorten the recursive schedule,
while smaller $\mathbf B$ or $\mathbf D$ rings may give better commitment
ranks, quotient sizes, setup cost, or verifier work.

To isolate this axis, return to one commitment group, one witness chunk, and
the `EvaluationTrace` opening method, but allow the three commitment matrices
to use

$$
R_A=F[X]/(X^{d_A}+1),
\qquad
R_B=F[X]/(X^{d_B}+1),
\qquad
R_D=F[X]/(X^{d_D}+1).
$$

The supported dimensions are powers of two and satisfy

$$
d_B\mid d_A,
\qquad
d_D\mid d_A.
$$

There is no ordering requirement between $d_B$ and $d_D$. The uniform setting
$d_A=d_B=d_D$ is the projection-ratio-one instance of the same protocol, not a
separate relation path.

Mixed rings preserve the meaning of the four semantic relations. They change
the native ring in which each physical row is represented:

| Semantic relation family | Native ring |
|---|---|
| fold-evaluation consistency | $R_A$ |
| inner-commitment consistency | $R_A$ |
| outer-commitment consistency | $R_B$ |
| opening-commitment consistency | $R_D$ |

Thus the four families do not each receive an arbitrary ring. The two
consistency families remain $\mathbf A$-native, while the two commitment-image
families use the native rings of $\mathbf B$ and $\mathbf D$.

### Role-native projection and decomposition

The source block $\mathbf s_b$, its inner image
$\mathbf t_b=\mathbf A\mathbf s_b$, the partial evaluation $E_b$, and the
folded response $\mathbf z$ are first formed over $R_A$. Before
$\mathbf t_b$ is decomposed for $\mathbf B$, or $E_b$ is decomposed for
$\mathbf D$, the implementation splits the $\mathbf A$-native value into exact
role-native coefficient subcolumns.

Let $r$ be either $d_B$ or $d_D$, and let $q=d_A/r$. For
$y(X)\in R_A$, define

$$
y_s(X)
=
\sum_{k=0}^{r-1}y_{sr+k}X^k
\in R_r,
\qquad
0\le s<q.
$$

Taking the canonical representative of degree less than $d_A$, $y$ has the
exact decomposition

$$
\boxed{
y(X)
=
\sum_{s=0}^{q-1}\sum_h
X^{sr}G_h\hat y_{s,h}(X),
\qquad
\hat y_{s,h}\in R_r.
}
\tag{6}
$$

Here $y_s=\sum_hG_h\hat y_{s,h}$ is digit-decomposed inside its native role
ring. Equation (6) is a coefficient identity, not an embedding of $R_r$ into
$R_A$ and not padding to an $\mathbf A$-sized carrier. The physical digit order
is

```text
[semantic value][role subcolumn][digit][native coefficient].
```

The $\mathbf B$ representation of an $\mathbf A$-native value therefore has
$d_A/d_B$ subcolumns, and its $\mathbf D$ representation has $d_A/d_D$
subcolumns. Write $\operatorname{Rec}_B$ and $\operatorname{Rec}_D$ for the
corresponding recomposition maps, including the shifts $X^{s d_B}$ and
$X^{s d_D}$ from Equation (6). They reconstruct $\mathbf A$-native values from
the role-native digits used by the consistency relations.

### The four relations in their native rings

The fold still computes

$$
\mathbf z
=
\sum_b c_b\mathbf s_b
=
\mathbf G_{\mathrm{fold}}\hat{\mathbf z}
\qquad\text{over }R_A.
$$

With role-native recomposition made explicit, the four semantic relations are

$$
\boxed{
\begin{aligned}
\sum_b c_b\operatorname{Rec}_D(\hat{\mathbf e}_b)
&=
\mathbf Q\mathbf G_{\mathrm{in}}
\mathbf G_{\mathrm{fold}}\hat{\mathbf z}
&&\text{over }R_A,
\\
\sum_b c_b\operatorname{Rec}_B(\hat{\mathbf t}_b)
&=
\mathbf A\mathbf G_{\mathrm{fold}}\hat{\mathbf z}
&&\text{over }R_A,
\\
\mathbf B\hat{\mathbf t}
&=
\mathbf u
&&\text{over }R_B,
\\
\mathbf D\hat{\mathbf e}
&=
\mathbf v_D
&&\text{over }R_D.
\end{aligned}
}
\tag{7}
$$

The first two rows compare values that originate from the $\mathbf A$-native
source, so they recompose the $\mathbf B$- or $\mathbf D$-native digits back
into $R_A$. The last two rows operate directly on those role-native digits.
The physical columns of $\mathbf B$ and $\mathbf D$ include the subcolumn axis,
so neither matrix pads its input back to $R_A$.

The canonical row-family order and targets remain

$$
[\mathrm{consistency}\mid\mathbf A\mid\mathbf B\mid\mathbf D],
\qquad
[0\mid\mathbf 0_{\mathbf A}\mid\mathbf u\mid\mathbf v_D].
$$

This notation records the row order and semantic targets; in the mixed setting
it is not one vector equation over a common ring. Each component is interpreted
in the native ring shown in Equation (7).

The ordinary witness is stored as one flat coefficient vector. Its
$\hat{\mathbf z}$ segment is $\mathbf A$-native, its
$\hat{\mathbf t}$ and $\hat{\mathbf e}$ segments use the role-native
subcolumn layout, and every quotient row added by the physical realization is
stored at that row's exact native dimension. No batch-wide carrier ring is
introduced for witness storage.

### Lift and switch the native rows

Rows over different quotient rings cannot be combined directly. The physical
realization first lifts each row from its native quotient ring to an exact
polynomial identity. If row $i$ has native dimension $d_i$ and semantic form

$$
L_i(X)=y_i(X)
\qquad\text{in }F[X]/(X^{d_i}+1),
$$

then the prover supplies a native quotient $r_i(X)$ such that

$$
L_i(X)
-(X^{d_i}+1)r_i(X)
=y_i(X).
\tag{8}
$$

Consistency and $\mathbf A$ rows therefore have $R_A$-native quotients,
$\mathbf B$ rows have $R_B$-native quotients, and $\mathbf D$ rows have
$R_D$-native quotients. These quotient polynomials are digit-decomposed and
included in the physical witness.

Ring switching then samples one extension-field element $\alpha$ and evaluates
every lifted row at that same point:

$$
\boxed{
L_i(\alpha)
-(\alpha^{d_i}+1)r_i(\alpha)
=y_i(\alpha)
\qquad\text{for every physical row }i.
}
\tag{9}
$$

After Equation (9), all rows are scalar identities over the same extension
field even though they originated in different cyclotomic rings. Stage 2 can
therefore batch them with its row challenge $\tau_1$.

This is the precise role of ring switching in the mixed-ring protocol. It does
not first convert all relations into one common quotient ring. Role-native
projection makes each relation well formed in its own ring; the native
quotient lift and evaluation at $\alpha$ then place all row checks in one
common field. The [realizations page](./akita-fold-realizations.md#lift-the-physical-ring-relations-before-sumcheck)
derives this lift for the complete physical witness.

### Relation to compressed realization

Compressed realization uses the same native-row quotient and ring-switch
mechanism. In raw mode, the semantic commitment images
$\mathbf u=\mathbf B\hat{\mathbf t}$ and
$\mathbf v_D=\mathbf D\hat{\mathbf e}$ are public. In compressed mode they are
private intermediate values, and additional $\mathbf F$ and $\mathbf H$ rows
bind them to smaller public payloads. Those compression rows have their own
native ring dimensions, quotients, and instances of Equation (9).

The difference is structural. Mixed $\mathbf A$/$\mathbf B$/$\mathbf D$
dimensions assign native rings to the existing four semantic relation
families. Compression adds new $\mathbf F$/$\mathbf H$ physical row families.
Once the rows have been formed, both constructions use the same native
quotient lift and ring-switch evaluation before Stage 2.

Choosing a smaller $d_B$ or $d_D$ is not automatically cheaper. The physical
column counts expand by the projection ratios

$$
q_B=\frac{d_A}{d_B},
\qquad
q_D=\frac{d_A}{d_D},
$$

and, for fixed digit depths, splitting a value does not reduce the total number
of its $\hat{\mathbf t}$ or $\hat{\mathbf e}$ coefficients. The benefit instead
comes from the complete interaction among native commitment ranks, payload and
quotient sizes, setup geometry, verifier arithmetic, and later fold levels.
The planner therefore prices complete $(d_A,d_B,d_D)$ schedules rather than
minimizing each dimension independently.

### Composition with groups and chunks

Mixed dimensions compose with the two earlier axes without changing their
ownership rules. Each commitment group owns its $\mathbf A$ and $\mathbf B$
dimensions, while the consuming level owns the shared $\mathbf D$ dimension.
Each chunk retains the native layouts of its group's
$[\hat{\mathbf z}\mid\hat{\mathbf e}\mid\hat{\mathbf t}]$ unit; chunking changes
the block ranges and column support, not the ring assigned to a relation row.

With multiple groups and chunks, the exact unit order is

$$
\big\Vert_{j=0}^{C-1}
\big\Vert_{g\in\mathrm{relation\ order}}
[\hat{\mathbf z}^{(j)}_g
 \mid\hat{\mathbf e}^{(j)}_g
 \mid\hat{\mathbf t}^{(j)}_g].
$$

The resulting physical witness remains one chunk-major flat coefficient vector
followed by the shared native quotient rows and any compression suffix. Stage 2
uses each row's own dimension in its powers of $\alpha$ and denominator
$\alpha^{d_i}+1$, so group, chunk, and mixed-ring layouts can be combined
without introducing a common carrier ring.
