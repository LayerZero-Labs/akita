# Payload and ring-relation realizations of an Akita fold

This page starts from the [four semantic relation
families](./akita-fold.md#the-four-semantic-relation-families) derived on the
previous page and explains how they become physical proof rows. Three schedule
choices are involved:

| Choice | Alternatives | What it changes |
|---|---|---|
| opening method | `EvaluationTrace`, `SubringCoefficientPacking` | opening-digit geometry, source-consistency realization, scalar opening row |
| payload mode | raw, compressed | how $\mathbf B\hat{\mathbf t}$ and $\mathbf D\hat{\mathbf e}$ are publicly bound |
| ring-relation mode | `QuotientLift`, `ReducedEvaluation` | how physical ring equations become field equations for Stage 2 |

Raw mode transmits the semantic commitments directly. Compressed mode keeps
those values private, binds them to smaller terminal payloads through two-map
commitment chains, and adds the corresponding witness segments and physical
rows. Neither payload choice changes which opening method was scheduled.

Quotient lifting adds private polynomial-modulus quotient digits. Reduced
evaluation instead transposes negacyclic reduction into public coefficient
weights and omits those digits. Relation mode is independent of raw versus
compressed payload where schedule validation permits the combination;
production coefficient-packing folds remain quotient-lift-only.

After defining the payload modes, the page explains the method-dependent
consistency geometry and both ring-relation realizations before evaluation at
the ring-switch challenge. It then separates those physical rows from the
field-valued virtual opening row consumed by Stage 2.

## Contents

- [From semantic relations to physical realizations](#from-semantic-relations-to-physical-realizations)
  - [Raw realization](#raw-realization)
  - [Compressed realization](#compressed-realization)
  - [Planner-selected realization transition](#planner-selected-realization-transition)
- [Commitment compression realization](#commitment-compression-realization)
  - [Why recommit?](#why-recommit)
  - [One recommitment step](#one-recommitment-step)
  - [The two-map commitment chains](#the-two-map-commitment-chains)
  - [Additional physical relations and witness](#additional-physical-relations-and-witness)
- [Ring-relation realization before sumcheck](#ring-relation-realization-before-sumcheck)
  - [Quotient lifting](#quotient-lifting)
  - [Reduced evaluation](#reduced-evaluation)
- [The scalar opening claim is a method-dependent virtual row](#the-scalar-opening-claim-is-a-method-dependent-virtual-row)
- [Code reference](#code-reference)

## From semantic relations to physical realizations

The [four relations derived on the previous
page](./akita-fold.md#the-four-semantic-relation-families) are semantic: they
state which algebraic constraints a valid fold must satisfy, without
prescribing which commitment values are sent in the proof. A **physical
realization** turns those constraints into the native-ring matrix equations
used by a particular payload mode. It determines the public payload, any
additional compression witness, and the right-hand side of each equation.

When these equations are assembled as a matrix, each coordinate equation is a
**physical row**. For example, the vector equation
$\mathbf B\hat{\mathbf t}=\mathbf u$ contributes $n_B$ physical rows, one for
each coordinate of $\mathbf u$. A physical row is therefore an individual
native-ring equation implementing a semantic relation, not a new semantic
claim.

Only the two commitment relations depend on the **payload-mode** choice. Their semantic
values are

$$
\mathbf u=\mathbf B\hat{\mathbf t},
\qquad
\mathbf v_D=\mathbf D\hat{\mathbf e}.
$$

The fold-evaluation and inner-commitment relations have the same physical form
in raw and compressed payload modes. Their form may still differ between the
two opening methods.

### Raw realization

In raw mode, the public payload *is* the semantic commitment. The prover
computes $\mathbf u$ and $\mathbf v_D$ and transmits their complete ring
vectors. They appear directly as the right-hand sides of the ordinary
$\mathbf B$ and $\mathbf D$ rows:

$$
\mathbf B\hat{\mathbf t}=\mathbf u,
\qquad
\mathbf D\hat{\mathbf e}=\mathbf v_D.
$$

The logical witness contains only the three segments already used by the four
semantic relations:

$$
\mathbf w_0
=
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}.
\tag{16}
$$

For the basic `EvaluationTrace` geometry, writing the four row families as one
conceptual matrix gives

$$
\boxed{
\mathbf M_0\mathbf w_0=\mathbf y
\quad\text{over }R.
}
\tag{17}
$$

Let $n_A$, $n_B$, and $n_D$ denote the row counts of $\mathbf A$,
$\mathbf B$, and $\mathbf D$. In that one-group layout, the raw realization
has $1+n_A+n_B+n_D$ physical rows:

| Physical rows | Count | Meaning | Right-hand side |
|---|---:|---|---|
| `consistency` | $1$ | Equation (12a) | $0$ |
| $\mathbf A$ rows | $n_A$ | Equation (13) | $\mathbf 0$ |
| $\mathbf B$ rows | $n_B$ | Equation (14) | $\mathbf u$ |
| $\mathbf D$ rows | $n_D$ | Equation (15) | $\mathbf v_D$ |

Consequently,

$$
\mathbf y
=
0
\;\Vert\;
\mathbf 0_{\mathbf A}
\;\Vert\;
\mathbf u
\;\Vert\;
\mathbf v_D.
\tag{18}
$$

The verifier therefore receives the values defined by the semantic relations
and places them directly in the raw relation instance. No compression digits
or $\mathbf F/\mathbf H$ rows are present. The matrix $\mathbf M_0$ need not
be materialized densely: its entries are generated from the fold challenges,
opening weights, gadget weights, and the setup matrices $\mathbf A$,
$\mathbf B$, and $\mathbf D$.

With `SubringCoefficientPacking`, raw mode still transmits the same semantic
$\mathbf B$ and $\mathbf D$ commitments, but Equation (12b) is realized by
the packed coordinate-plane relations described below. It therefore does not
reuse the single ordinary `consistency` row counted in this example.

Equation (17) is still a relation in the native cyclotomic ring; it is not yet
the exact field identity consumed by sumcheck. The
[ring-relation realization](#ring-relation-realization-before-sumcheck) later
on this page explains the two schedule-selected ways to obtain that field
identity. This step is deferred until after the compressed rows have also been
defined.

### Compressed realization

In compressed mode, the prover still computes the same semantic commitments,
but does not transmit $\mathbf u$ or $\mathbf v_D$. Instead, it decomposes each
one, recommits the resulting blocks through a two-map chain, and sends one
fixed 128-byte terminal payload for each chain. In the illustrative q128
example developed below, this replaces a 1024-byte semantic outer commitment
by the 128-byte payload $p_F$. The two commitment relations follow parallel
chains:

$$
\mathbf B\hat{\mathbf t}=\mathbf u
\xrightarrow{\text{compression }\mathbf F\text{ chain}}
p_F,
$$

$$
\mathbf D\hat{\mathbf e}=\mathbf v_D
\xrightarrow{\text{compression }\mathbf H\text{ chain}}
p_H.
$$

The prover stores the two layers of $\mathbf F$ and $\mathbf H$ compression
digits as additional private witness coordinates. These arrows are not a
decoding procedure: the verifier does not recover $\mathbf u$ from $p_F$ or
$\mathbf v_D$ from $p_H$. Instead, additional $\mathbf F/\mathbf H$ equations,
represented by physical rows, prove that the hidden semantic commitments
recompose from the first digit layer and that the two compression maps lead
to the transmitted terminal payloads. The
$\mathbf B$ and $\mathbf D$ right-hand sides are therefore zero in compressed
mode; $p_F$ and $p_H$ appear only on the terminal $\mathbf F_2$ and
$\mathbf H_2$ rows.

From the verifier's perspective, the schedule already determines the payload
mode. For a raw level, it assembles the ordinary relation right-hand side from
$\mathbf u$ and $\mathbf v_D$. For a compressed level, it assembles the larger
row layout from $p_F$ and $p_H$ and verifies the compression chains together
with the ordinary relations. The proof does not carry a separate mode tag.

| View | Raw realization | Compressed realization |
|---|---|---|
| Public payload | $\mathbf u,\mathbf v_D$ | $p_F,p_H$ |
| Compression-digit witness | none | $\boldsymbol\xi_{F,1},\boldsymbol\xi_{F,2},\boldsymbol\xi_{H,1},\boldsymbol\xi_{H,2}$ |
| $\mathbf B/\mathbf D$ right-hand sides | $\mathbf u,\mathbf v_D$ | zero |
| Additional compression rows | none | $\mathbf F_1,\mathbf F_2,\mathbf H_1,\mathbf H_2$ |
| What binds the payload | the ordinary commitment equations directly | the ordinary equations followed by the compression chains |

### Planner-selected realization transition

Compression saves public payload bytes but adds digit witnesses, relation
rows, quotient witnesses, and restricted range-check work. The planner prices
both sides of this tradeoff for the complete recursive schedule; it does not
choose a mode from the payload size alone.

Commitment groups created separately before recursive proving—for example,
groups later supplied as precommitted root inputs—always use compressed
payloads and are not part of the recursive mode choice. The protocol also
requires compressed payloads for the root fold and the first recursive fold,
when that fold exists. Any later fold that consumes a setup prefix must
likewise remain compressed. At a later level that does not consume a setup
prefix, the planner may either continue the compressed prefix or begin a raw
suffix. Once it selects raw mode, every later recursive level remains raw:

$$
\underbrace{\text{compressed}\;\longrightarrow\;\cdots\;\longrightarrow\;
\text{compressed}}_{\text{planner-selected prefix}}
\longrightarrow
\underbrace{\text{raw}\;\longrightarrow\;\cdots\;\longrightarrow\;
\text{raw}}_{\text{raw suffix}}.
$$

The prefix length is schedule-dependent. Some current generated schedules cut
over immediately after the required first recursive fold, whereas deeper
recursive schedules keep several early recursive levels compressed. Thus
"root and early folds are compressed" describes a planner-selected prefix,
not a globally fixed transition point.

The next section expands the two commitment relations into their compressed
physical rows.

## Commitment compression realization

The [outer-commitment](./akita-fold.md#3-outer-commitment-consistency) and
[opening-commitment](./akita-fold.md#4-opening-commitment-consistency)
semantic relations, Equations (14) and (15), have the same structure: a
commitment matrix maps a short witness to a semantic commitment consisting of
a vector of ring elements. The raw realization places that complete
commitment in the proof. The compressed realization instead recommits it using
rank-one matrices over progressively smaller rings, producing a commitment
chain whose terminal payload is exactly 128 bytes.

### Why recommit?

The serialized size of an uncompressed semantic outer commitment is

$$
|\mathbf u|_{\mathrm{bytes}}
=
n_Bd_Bb_F,
$$

where $d_B$ is the $\mathbf B$ ring dimension and $b_F$ is the canonical byte
width of one field element. In the q128 profile, one field element occupies
16 bytes. For a simple rank-one example, take $n_B=1$ and $d_B=64$. Its
uncompressed payload then occupies

$$
1\cdot64\cdot16=1024\ \text{bytes}.
$$

The goal of commitment compression is to recommit this value in a smaller ring
and repeat the process until the public payload has the desired size. This
reduces the bytes occupied by the public commitment payload, but it also
introduces new witness coordinates and relation rows. The planner accounts for
both effects when it chooses between compressed and raw recursive payloads.

### One recommitment step

At a high level, one step does not map a large-ring element directly into a
smaller quotient ring. It first applies the balanced decomposition introduced
above with base $2$, whose digit range is $\{-1,0\}$. It then groups those
digits into short coefficient blocks, each representing an element of the
smaller ring. A rank-one setup matrix over that same smaller ring commits the
resulting vector to one new ring element. The recomposition identity is the
bridge from these smaller-ring blocks back to the original large-ring element.

For intuition, first suppose that the semantic commitment is a single ring
element

$$
u(X)
=
\sum_{\ell=0}^{d-1}u_\ell X^\ell
\in R_d=F[X]/(X^d+1).
$$

The first operation is a coefficientwise balanced base-$2$ decomposition. For
each $u_\ell\in F$, choose digits
$\xi_{k,\ell}\in\{-1,0\}$ such that

$$
u_\ell
=
\sum_{k=0}^{\kappa-1}2^k\xi_{k,\ell}
\qquad\text{in }F,
$$

where $\kappa$ is the field-modulus bit width. Put the $k$-th digit of every
coefficient into one digit polynomial

$$
\xi_k(X)
=
\sum_{\ell=0}^{d-1}\xi_{k,\ell}X^\ell.
$$

Then

$$
u(X)
=
\sum_{k=0}^{\kappa-1}2^k\xi_k(X),
\qquad
[\xi_k]_\ell\in\{-1,0\},
$$

so every coefficient of every decomposed polynomial is either $-1$ or $0$.

The second operation repacks these digits into a smaller ring. Choose
$d'\mid d$ and divide the length-$d$ coefficient vector of each $\xi_k$ into
consecutive blocks of length $d'$. The $j$-th block becomes the coefficient
vector of

$$
\xi'_{k,j}(X)
=
\sum_{\ell=0}^{d'-1}
\xi_{k,jd'+\ell}X^\ell
\in R_{d'}=F[X]/(X^{d'}+1).
$$

Equivalently, if $\widetilde{\xi'_{k,j}}(X)$ denotes the canonical
degree-less-than-$d'$ representative of this small-ring element, then

$$
\xi_k(X)
=
\sum_{j=0}^{d/d'-1}
X^{jd'}\widetilde{\xi'_{k,j}}(X)
\qquad\text{in }R_d.
$$

Collect all of the small-ring blocks into a vector

$$
\boldsymbol\xi
=
(\xi'_{0,0},\ldots,\xi'_{\kappa-1,d/d'-1})
\qquad
\text{over }R_{d'}.
$$

Akita uses bit-major order: all coefficient blocks for one bit position are
contiguous before the blocks for the next bit position. In this unpadded
single-element example, the vector has width $w=\kappa d/d'$ over $R_{d'}$.

The fixed recomposition map restores both the coefficient positions and the
powers of two:

$$
\operatorname{Rec}_{d\leftarrow d'}(\boldsymbol\xi)
=
\sum_{k=0}^{\kappa-1}2^k
\sum_{j=0}^{d/d'-1}
X^{jd'}\widetilde{\xi'_{k,j}}(X)
=u(X).
$$

This is the bridge between $R_{d'}$ and $R_d$. It is a fixed $F$-linear
coefficient-recomposition map, not a ring embedding or a homomorphism between
the two quotient rings.

Finally, a rank-one matrix over the smaller ring recommits the packed blocks:

$$
\mathbf F\in R_{d'}^{1\times w},
\qquad
u'=\mathbf F\boldsymbol\xi\in R_{d'}.
$$

Together, the two equalities

$$
u=\operatorname{Rec}_{d\leftarrow d'}(\boldsymbol\xi),
\qquad
u'=\mathbf F\boldsymbol\xi
$$

link the original commitment in $R_d$ to a new commitment in $R_{d'}$ through
the shared digit vector $\boldsymbol\xi$. Conceptually, one recommitment step
therefore performs three operations: decompose the source coefficients,
repack the digits as small-ring elements, and commit those elements to one
small-ring image.

### The two-map commitment chains

The smaller ring cannot be chosen solely to minimize the payload. Decreasing
its dimension makes the output shorter, but it also increases the number of
input columns presented to the rank-one compression matrix. The pair
consisting of the ring dimension and input width must remain within Akita's
Module-SIS security bounds.

The current protocol therefore uses a fixed, profile-specific ladder of
exactly two certified rank-one maps. The ladder accepts a complete source of
at most 8 KiB and terminates at exactly 128 bytes.

For the semantic commitment $\mathbf u\in R_{d_B}^{n_B}$, flatten its
$n_B$ ring coordinates into $n_Bd_B$ field coefficients. Balanced base-$2$
decomposition produces $\kappa n_Bd_B$ digits in $\{-1,0\}$. Pack these digits
in bit-major order into
$
w_1
$
elements of $R_{d_1}$, filling any unused coefficients of the last element
with zero. These smaller-ring elements form
$\boldsymbol\xi_{F,1}\in R_{d_1}^{w_1}$. The first rank-one map commits this
vector to $u^{(1)}\in R_{d_1}$; the second layer repeats the same decomposition,
repacking, and recommitment at dimension $d_2$:

$$
\underbrace{\mathbf u\in R_{d_B}^{n_B}}_{\text{semantic commitment}}
\xrightarrow{}
\underbrace{\boldsymbol\xi_{F,1}\in R_{d_1}^{w_1}}_{\text{small-ring digit blocks}}
\overset{\mathbf F_1}{\longrightarrow}
\underbrace{u^{(1)}\in R_{d_1}}_{\text{intermediate image}}
\xrightarrow{}
\underbrace{\boldsymbol\xi_{F,2}\in R_{d_2}^{w_2}}_{\text{second-layer digit blocks}}
\overset{\mathbf F_2}{\longrightarrow}
\underbrace{p_F\in R_{d_2}}_{\text{terminal payload}}.
$$

From the prover's perspective, the chain consists of five concrete steps:

1. Compute the semantic commitment
   $$
   \mathbf u=\mathbf B\hat{\mathbf t}
   \in R_{d_B}^{n_B}.
   $$
2. Decompose and repack $\mathbf u$ into $\boldsymbol\xi_{F,1}$ so that
   $$
   \operatorname{Rec}_{d_B\leftarrow d_1}(\boldsymbol\xi_{F,1})
   =\mathbf u.
   $$
3. Apply the first rank-one map:
   $$
   u^{(1)}=\mathbf F_1\boldsymbol\xi_{F,1}
   \in R_{d_1}.
   $$
4. Decompose and repack $u^{(1)}$ into $\boldsymbol\xi_{F,2}$ so that
   $$
   \operatorname{Rec}_{d_1\leftarrow d_2}(\boldsymbol\xi_{F,2})
   =u^{(1)}.
   $$
5. Apply the terminal rank-one map:
   $$
   p_F=\mathbf F_2\boldsymbol\xi_{F,2}
   \in R_{d_2}.
   $$

Only $p_F$ is transmitted. The semantic commitment $\mathbf u$ is computed
but omitted from the compressed payload. The intermediate image $u^{(1)}$ is
also not transmitted and is not stored as an independent witness segment;
the two digit vectors $\boldsymbol\xi_{F,1}$ and
$\boldsymbol\xi_{F,2}$ are the additional witness material. The physical
relations in the next subsection bind this private chain all the way back to
$\hat{\mathbf t}$. The opening-side $\mathbf H$ chain applies the same five
steps to $\mathbf v_D=\mathbf D\hat{\mathbf e}$ and terminates at $p_H$.

The production compression dimensions are fixed by the modulus profile:

| Profile | First ring $d_1$ | First image | Terminal ring $d_2$ | Terminal payload |
|---|---:|---:|---:|---:|
| q128 | $16$ | 256 bytes | $8$ | 128 bytes |
| q64 | $32$ | 256 bytes | $16$ | 128 bytes |
| q32 | $64$ | 256 bytes | $32$ | 128 bytes |

For the q128 example above, the complete chain is

$$
\underbrace{u\in R_{64}}_{1024\ \text{bytes}}
\xrightarrow{\text{}}
\underbrace{\boldsymbol\xi_{F,1}\in R_{16}^{512}}_{\text{base-2 digit blocks}}
\overset{\mathbf F_1}{\longrightarrow}
\underbrace{u^{(1)}\in R_{16}}_{256\ \text{bytes}}
\xrightarrow{\text{}}
\underbrace{\boldsymbol\xi_{F,2}\in R_8^{256}}_{\text{base-2 digit blocks}}
\overset{\mathbf F_2}{\longrightarrow}
\underbrace{p_F\in R_8}_{128\ \text{bytes}}.
$$

The widths count the small-ring elements needed to hold all balanced base-$2$
digits: $w_1=64\cdot128/16=512$ and
$w_2=16\cdot128/8=256$.

The schedule's payload mode determines whether these chains are present at a
particular recursive level, according to the planner-selected transition
described above.

### Additional physical relations and witness

The equalities in a compression chain do not all live in the same ring. Each
physical row is interpreted in the native ring displayed beside it, and each
$\operatorname{Rec}$ is the fixed linear coefficient-recomposition map defined
above. Suppressing component indices inside that map, the outer
$\mathbf B/\mathbf F$ chain gives three physical relation equations. Their
labels extend Equation (14) to emphasize that the entire chain realizes that
one semantic commitment relation:

$$
\boxed{
\mathbf B\hat{\mathbf t}
-
\operatorname{Rec}_{d_B\leftarrow d_1}(\boldsymbol\xi_{F,1})
=
\mathbf 0
}
\qquad\text{in }R_{d_B}^{n_B},
\tag{14a}
$$

$$
\boxed{
\mathbf F_1\boldsymbol\xi_{F,1}
-
\operatorname{Rec}_{d_1\leftarrow d_2}(\boldsymbol\xi_{F,2})
=0
}
\qquad\text{in }R_{d_1},
\tag{14b}
$$

$$
\boxed{
\mathbf F_2\boldsymbol\xi_{F,2}=p_F
}
\qquad\text{in }R_{d_2}.
\tag{14c}
$$

The first equation is the compressed realization of the existing
$\mathbf B$ rows; the latter two add the rank-one $\mathbf F_1$ and
$\mathbf F_2$ rows. The semantic opening commitment follows the same
construction, with an $\mathbf H$ chain:

$$
\boxed{
\mathbf D\hat{\mathbf e}
-
\operatorname{Rec}_{d_D\leftarrow d_1}(\boldsymbol\xi_{H,1})
=
\mathbf 0
}
\qquad\text{in }R_{d_D}^{n_D},
\tag{15a}
$$

$$
\boxed{
\mathbf H_1\boldsymbol\xi_{H,1}
-
\operatorname{Rec}_{d_1\leftarrow d_2}(\boldsymbol\xi_{H,2})
=0
}
\qquad\text{in }R_{d_1},
\tag{15b}
$$

$$
\boxed{
\mathbf H_2\boldsymbol\xi_{H,2}=p_H
}
\qquad\text{in }R_{d_2}.
\tag{15c}
$$

For the basic one-group `EvaluationTrace` case, compressed mode therefore has
$1+n_A+n_B+n_D+4$ physical rows:

| Physical rows | Count | Right-hand side |
|---|---:|---|
| `consistency` | $1$ | $0$ |
| $\mathbf A$ | $n_A$ | $\mathbf 0$ |
| $\mathbf B$ | $n_B$ | $\mathbf 0$ |
| $\mathbf D$ | $n_D$ | $\mathbf 0$ |
| $\mathbf F_1$ | $1$ | $0$ |
| $\mathbf H_1$ | $1$ | $0$ |
| $\mathbf F_2$ | $1$ | $p_F$ |
| $\mathbf H_2$ | $1$ | $p_H$ |

For packing, the payload-mode suffix is identical, but the method-specific
coordinate-plane relations replace the ordinary `consistency` row just as in
raw mode.

Before adding quotient digits and alignment, the logical compressed witness
has the following layer order:

$$
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}
\;\Vert\;
\boldsymbol\xi_{F,1}
\;\Vert\;
\boldsymbol\xi_{H,1}
\;\Vert\;
\boldsymbol\xi_{F,2}
\;\Vert\;
\boldsymbol\xi_{H,2}.
$$

The next section explains when the witness adds one quotient-digit row for
every physical ring row and when reduced evaluation omits all such rows. The
restricted $\{-1,0\}$ check on the compression digits is described with the
Stage 2 sumcheck rather than as another physical ring row.

## Ring-relation realization before sumcheck

The schedule stores one `RingRelationMode` per nonterminal fold. It is bound by
the effective schedule descriptor before the outgoing witness commitment and
the ring-switch challenge `alpha`. The terminal fold has no mode because it
checks its clear response directly.

The payload and ring-relation axes produce four witness shapes:

| Payload | Quotient lifting | Reduced evaluation |
|---|---|---|
| raw | Z/E/T and ordinary quotient digits | Z/E/T |
| compressed | Z/E/T, F/H digits, ordinary quotients, and F/H quotients | Z/E/T and F/H digits |

`WitnessLayout` is the authority for these ranges. An omitted quotient is not a
zero-width placeholder and does not enter Stage 1, Stage 2, response sizing, or
the successor commitment.

### Quotient lifting

The payload mode determines which commitment rows exist; the opening method
determines how the first semantic family enters this lift.

| Opening method | Consistency realization | Source-fold geometry |
|---|---|---|
| `EvaluationTrace` | one ordinary relation in $R_D$ | the same $R_D$ challenge acts on $\mathbf z$ and $\hat{\mathbf e}$ |
| `SubringCoefficientPacking` | one logical $C=E[U]/(U^s+1)$ relation over $k$ base-field coordinate planes; packed E/Q use the common relation events and the folded source uses the packing-Z term | $c(U)$ is embedded as $c(X^{k\eta})$ in the A ring |

For packing, each coordinate plane has modulus $U^s+1$. The complete relation
has physical width $ks$; that does **not** make it one ring of dimension $ks$.
The logical consistency-row slot remains in the row domain, so its $\tau_1$
weight also batches the packed relation. The packed E/Q coordinate-plane
events join the common relation-weight factorization; only the folded-source
side is supplied as the separate packing-Z structured term. These coefficients
replace the legacy `EvaluationTrace` consistency formula, and all planes
together realize the single semantic Equation (12b).

The ordinary physical equations are congruences in cyclotomic rings, whereas
sumcheck needs exact field identities. In the raw basic case, every ordinary
row uses the common ring $R_D$ for `EvaluationTrace`. Packing omits the
legacy `EvaluationTrace` consistency coefficients; its A, B, and D rows remain
in their scheduled native rings, while its consistency-row slot uses $k$
coordinate planes of dimension $s$. Compressed mode retains those scheduled
dimensions and adds two
compression-only dimensions: the $\mathbf F_1$ and $\mathbf H_1$ rows lie in
$R_{d_1}$, while the $\mathbf F_2$ and $\mathbf H_2$ rows lie in $R_{d_2}$.
There is therefore no single denominator $X^D+1$ that applies to every row in
the general physical layout.

Instead, lift each physical row in its own ring. Index the rows by $i$, let
$d_i$ be the native dimension of row $i$, and choose the canonical
degree-less-than-$d_i$ representative of each ring element used by that row.
If $\mathbf M_i$ and $y_i$ denote its matrix coefficients and right-hand side,
then the ring equality is equivalent to

$$
\sum_j
\widetilde M_{i,j}(X)\widetilde w_j(X)
-
\widetilde y_i(X)
=
(X^{d_i}+1)r_i(X).
\tag{19}
$$

Thus every physical row owns one quotient polynomial $r_i$ in the same native
dimension. Digit-decompose it with the quotient gadget:

$$
r_i(X)
=
\sum_{g=0}^{L_r-1}
G_g^{(r)}\hat r_{i,g}(X),
\qquad
\hat r_{i,g}\in R_{d_i}.
\tag{20}
$$

Logically, these quotient digits extend the witness in the same way in both
payload modes. Before any compression-only suffix, the witness layout places
the quotient digits for every relation-row family in one shared segment
$\hat{\mathbf r}_{\mathrm{ord}}$. In raw `EvaluationTrace` mode this segment
contains the quotients for the `consistency`, $\mathbf A$, $\mathbf B$, and
$\mathbf D$ rows in canonical row order:

$$
\boxed{
\mathbf w_{\mathrm{raw}}
=
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}
\;\Vert\;
\hat{\mathbf r}_{\mathrm{ord}}.
}
\tag{21a}
$$

For packing, the same shared segment still includes the consistency-row
quotient slot. That slot stores the digit-decomposed $k$ coordinate planes of
$Q_{\mathrm{pack}}$; the A, B, and D quotients follow their normal row layout.
$Q_{\mathrm{pack}}$ is therefore not stored in a separate method-dependent
witness span. Its packed Q events, together with the packed E events, enter the
common relation-weight factorization. The packing-Z and direct-opening terms
are the separate structured Stage-2 sources.

Compressed mode keeps that ordinary quotient segment, then stores each
compression layer's balanced base-$2$ digits beside the quotient digits for
the same $\mathbf F/\mathbf H$ maps. Suppressing derived zero-alignment ranges,
the basic one-group layout is

$$
\boxed{
\begin{aligned}
\mathbf w_{\mathrm{comp}}
={}&
\hat{\mathbf z}
\Vert\hat{\mathbf e}
\Vert\hat{\mathbf t}
\Vert\hat{\mathbf r}_{\mathrm{ord}}
\\
&\Vert\boldsymbol\xi_{F,1}
\Vert\boldsymbol\xi_{H,1}
\Vert\hat{\mathbf r}_{F,1}
\Vert\hat{\mathbf r}_{H,1}
\\
&\Vert\boldsymbol\xi_{F,2}
\Vert\boldsymbol\xi_{H,2}
\Vert\hat{\mathbf r}_{F,2}
\Vert\hat{\mathbf r}_{H,2}.
\end{aligned}
}
\tag{21b}
$$

The implementation derives zero padding before the first compression layer,
between layers when required, and at the end of the witness. Raw mode has no
compression spans or compression-alignment padding.

Substituting Equation (20) into Equation (19) and moving the denominator term
to the left gives one exact polynomial identity per row:

$$
\sum_j
\widetilde M_{i,j}(X)\widetilde w_j(X)
-
(X^{d_i}+1)
\sum_{g=0}^{L_r-1}G_g^{(r)}\hat r_{i,g}(X)
=
\widetilde y_i(X).
\tag{22}
$$

Call the row operator on the left $\mathbf M_{\mathrm{ext},i}(X)$. Then

$$
\boxed{
\mathbf M_{\mathrm{ext},i}(X)\widetilde{\mathbf w}(X)
=
\widetilde y_i(X).
}
\tag{23}
$$

Equation (19) uses the undecomposed quotient $r_i$, whereas Equation (23)
already includes its digits inside the appropriate raw or compressed witness
layout. The denominator term must not also be added to the right-hand side.

Ring switching samples one field element $\alpha$ and evaluates every row. A
row of dimension $d_i$ uses the powers
$1,\alpha,\ldots,\alpha^{d_i-1}$ and its own denominator
$\alpha^{d_i}+1$:

$$
\boxed{
\mathbf M_{\mathrm{ext},i}(\alpha)\mathbf w(\alpha)
=
y_i(\alpha)
\qquad\text{for every physical row }i.
}
\tag{24}
$$

After evaluation, all rows are scalar identities over the same extension
field even though they originated in different cyclotomic rings. Equation
(24) is therefore the field relation that Stage 2 can batch with $\tau_1$. The
[Sumcheck stages](./sumcheck-stages.md#stage-2-fused-relation-sumcheck) page
explains how $\tau_1$ batches its physical rows and how the resulting relation
is proved over the flat witness address.

### Reduced evaluation

Reduced evaluation checks the same native-ring equation without introducing
$r_i$. For a public multiplier
$A(X)=\sum_{k=0}^{d-1}a_kX^k$ and private witness coefficients $w_j$, define

$$
\kappa_{A,\alpha}(j)
=\left(A(X)X^j\bmod(X^d+1)\right)(\alpha).
$$

Then

$$
(A\circledast W)(\alpha)
=\sum_{j=0}^{d-1}w_j\kappa_{A,\alpha}(j),
$$

so reduction is moved into public weights rather than a private quotient. The
weights are prepared in linear time:

$$
\kappa_{A,\alpha}(0)=A(\alpha),
\qquad
\kappa_{A,\alpha}(j+1)
=\alpha\kappa_{A,\alpha}(j)
-(\alpha^d+1)a_{d-1-j}.
$$

This recurrence never divides by $\alpha^d+1$; an evaluation point that is a
root of the modulus remains valid. Each physical row uses its own native
dimension, including the F/H compression rows.

At the verifier's final multilinear point, an exact physical coefficient
window has equality weights $e_j$. The transposed terminal functional obeys

$$
H_0=\sum_je_j\alpha^j,
\qquad
H_{k+1}=\alpha H_k-(\alpha^d+1)e_{d-1-k}.
$$

The verifier uses this $O(d)$ functional state for structured terms and for
the existing fused A/B/D setup traversal. Compression F/H maps use their
canonical compression program with the same reduced semantics. The verifier
does not materialize a witness-sized table or rescan A, B, and D separately.
The prover currently materializes one ephemeral dense Stage-2 weight table;
that table is folded by sumcheck but is neither committed nor serialized.

Production schedules admit reduced evaluation only as a monotone,
setup-direct `EvaluationTrace` suffix beginning at absolute level 2. It composes
with raw or compressed payloads and with Linf or selective L2 security. An
incoming setup prefix, a later setup-offload edge, coefficient packing, or a
return to quotient lifting is rejected by `FoldSchedule::validate_structure`.
These are the supported scope of the current implementation, not algebraic
claims that reduced evaluation could never support a broader protocol.

## The scalar opening claim is a method-dependent virtual row

Two different statements involve $\hat e$, and they should not be conflated:

| Statement | Form | Physical ring row? | Ring-switch quotient? |
|---|---|---:|---:|
| opening commitment | $\mathbf D\hat{\mathbf e}=\mathbf v_D$ | yes | only in `QuotientLift` |
| evaluation correctness | $\sum_xw(x)T_{\mathrm{open}}(x)=v_{\mathrm{open}}$ | no | no |

The scheduled opening method prepares the second statement. `EvaluationTrace`
uses the trace weight derived in [Field-to-ring evaluation
reduction](./field-ring-reduction.md#express-the-direct-relation-as-a-sumcheck-claim).
`SubringCoefficientPacking` uses the [direct packed scalar
row](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials).
The direct scalar opening in either method reuses the same row-batching
challenge $\tau_1$, but it is absent from the physical ring-row layout, its
public right-hand side, and the quotient polynomials $r_i$. Coefficient packing
also changes the physical consistency realization through the packed E/Q
events in the common relation-weight factorization and the separate packing-Z
term described above. Both are distinct from the direct scalar opening.

[Sumcheck stages](./sumcheck-stages.md#stage-2-fused-relation-sumcheck)
continues from Equation (24) and fuses the physical relation, the
method-selected opening terms, and the range-image binding into one Stage-2
sumcheck.

## Code reference

The implementation follows the same semantic relations in both payload modes,
then selects different public right-hand sides and witness suffixes. The
functions below also support multiple groups, chunks, and role-specific ring
dimensions; in the basic setting they reduce to the construction on this
page.

### Prover flow

1. **Create and retain the commitment-side material.** The standalone/root
   commitment paths in
   [`commitment.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/api/commitment.rs)
   compute the semantic outer commitment
   $\mathbf u=\mathbf B\hat{\mathbf t}$ and compress it. The recursive
   [`commit_w`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/commit.rs)
   computes the same semantic value, then follows the payload mode selected
   for that level. A raw recursive level exposes $\mathbf u$ directly. A
   compressed commitment passes it to
   [`execute_compression_chains`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/compute/compression.rs),
   exposes only $p_F$, and retains the two packed $\mathbf F$ digit layers and
   their quotient rows in the commitment hint.
2. **Build the fold-side objects.**
   [`RingRelationProver::new`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_relation.rs)
   decomposes the position-folded values into $\hat{\mathbf e}$, computes
   $\mathbf v_D=\mathbf D\hat{\mathbf e}$, samples the fold challenges, and
   builds $\mathbf z$. In compressed mode,
   [`materialize_compression_witness`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_relation/compression_witness.rs)
   combines the retained $\mathbf F$ material with a newly computed
   $\mathbf H$ chain for $\mathbf v_D$, producing the terminal payload $p_H$
   and storing both chains as `CompressionWitnessMaterialization`.
3. **Assemble the mode-selected public statement.** Raw mode calls
   [`assemble_relation_rhs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/relation.rs)
   with $\mathbf u$ and $\mathbf v_D$. Compressed mode instead calls
   [`assemble_compressed_relation_rhs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/relation.rs)
   with $p_F$ and $p_H$. The latter emits zero ordinary $\mathbf B/\mathbf D$
   and first-map right-hand sides, followed by the terminal payloads on the
   $\mathbf F_2/\mathbf H_2$ rows.
4. **Construct the committed relation witness.**
   [`ring_switch_build_w`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/coeffs.rs)
   derives $\hat{\mathbf t}$ from the semantic inner rows stored in the hint,
   and emits $\hat{\mathbf z}$, $\hat{\mathbf e}$, and $\hat{\mathbf t}$. In
   quotient-lift mode it invokes
   [`compute_multi_group_relation_quotient`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs)
   to compute one quotient in every physical row's native ring. In compressed
   mode it also emits the two $\mathbf F/\mathbf H$ digit layers. In reduced
   mode it dispatches before quotient construction and uses negacyclic-only D
   and compression products. `WitnessLayout` determines which ranges and
   alignment are live.
5. **Prepare the Stage 2 relation evaluators.** Quotient lifting uses the
   factored
   [`build_relation_weight_events`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/relation_weights.rs)
   path for the `consistency`, $\mathbf A$, $\mathbf B$, and $\mathbf D$
   contributions. Reduced evaluation uses the semantic compiler in
   `ring_switch/relation_weights/compiler.rs` to build one dense Stage-2
   weight oracle. Compressed mode additionally uses
   [`build_compression_relation_weights`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/compression_relation_weights.rs)
   or its reduced counterpart for the recomposition, $\mathbf F/\mathbf H$,
   and mode-selected compression-quotient contributions, while
   [`NegativeBinarySupport`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/compression_relation_weights.rs)
   restricts the separate $w(w+1)=0$ check to the compression-digit spans.
   The scheduled opening method contributes separate $\tau_1$-weighted Stage-2
   terms after the payload-mode relation terms.

The mode split can be summarized as follows:

```text
semantic inner rows
        |
        v
derive t_hat and u = B t_hat
        |
        +---------------- raw ----------------> public u
        |
        `-- compressed --> F chain ----------> public p_F
                             |
                             `--> retained F digits

position-folded values E_b
        |
        v
e_hat and v_D = D e_hat
        |
        +---------------- raw ----------------> public v_D
        |
        `-- compressed --> H chain ----------> public p_H

raw:        assemble_relation_rhs(u, v_D)
compressed: assemble_compressed_relation_rhs(p_F, p_H)
                         |
                         v
             RingRelationInstance + prover witness
                         |
                         v
                ring_switch_build_w
                  |              |
           quotient lift    reduced evaluation
             add R digits      no R digits
                         |
                         v
          factored or dense relation evaluator
          + optional compression evaluator
          + optional {-1, 0} support evaluator
          + method-selected opening terms
                         |
                         v
                       Stage 2
```

### Public statement: `RingRelationInstance`

[`RingRelationInstance`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/ring_relation.rs)
is the common relation-statement carrier constructed independently by the
prover and verifier. Its statement fields are verifier-reconstructible; the
prover may additionally retain a private intermediate needed while preparing
its witness:

| Field or accessor | Mathematical meaning |
|---|---|
| `group_challenges()[0]` | fold challenges $c_b$ |
| `group_ring_multiplier_point(0)` | ring multipliers used by the physical consistency row |
| `opening_batch()` | authenticated group and claim geometry |
| `role_dims()` | native $\mathbf A/\mathbf B/\mathbf D$ ring dimensions |
| `rhs()` in raw mode | $[0\mid\mathbf 0_A\mid\mathbf u\mid\mathbf v_D]$ in the basic setting |
| `rhs()` in compressed mode | zero ordinary and first-map targets, followed by terminal $p_F,p_H$ targets |
| `v()` in raw mode | public $\mathbf v_D=\mathbf D\hat{\mathbf e}$ |
| `v()` in the compressed prover instance | the privately computed $\mathbf v_D$, retained locally after constructing the $\mathbf H$ chain |
| `v()` in compressed verifier replay | empty; the verifier uses $p_H$ rather than reconstructing $\mathbf v_D$ |

Full opening points are not owned by `RingRelationInstance`. They are prepared
separately by the scheduled opening method. The verifier consumes evaluation
trace points through
[`evaluation_trace.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/evaluation_trace.rs)
and coefficient-packing points through its compact packing relation path. Only
the projections needed by the physical consistency relation remain in this
instance.

### Prover witness: `RingRelationWitness`

[`RingRelationWitness`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_relation_witness.rs)
is the prover-only aggregate witness. In the basic setting, its `groups`
vector contains one
`RingRelationGroupWitness`:

| Field | Mathematical meaning |
|---|---|
| `z_folded_rings` | folded response $\mathbf z$, before decomposition into $\hat z$ |
| `e_folded` | recomposed position-folded rings $E_b$ |
| `e_hat` | opening digits $\hat{\mathbf e}$ |
| `hint` | semantic inner rows, plus retained $\mathbf F$ stages and quotients when the incoming payload is compressed |

The
[`AkitaCommitmentHint`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/hints.rs)
does not store a materialized $\hat{\mathbf t}$ or a separate copy of
$\mathbf u$. `ring_switch_build_w` derives $\hat{\mathbf t}$ from its semantic
inner rows. At the aggregate level, `RingRelationWitness::compression` holds
the optional materialized $\mathbf F/\mathbf H$ chains used by this fold. The
quotient output is computed afterward only in quotient-lift mode. Reduced
evaluation places only the ordinary and optional compression digits according
to `WitnessLayout`.

### Verifier reconstruction

The verifier does not receive a serialized `RingRelationInstance` or any
`RingRelationWitness`. In
[`verify_fold`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/core/fold/mod.rs),
it reconstructs the public instance from the schedule-selected payload and
transcript data, then calls
[`ring_switch_verifier`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/ring_switch.rs):

```text
schedule-selected payload mode
              |
       +------+------+
       |             |
      raw        compressed
    u, v_D         p_F, p_H
       |             |
assemble_relation_rhs
                 assemble_compressed_relation_rhs
       |             |
       +------+------+
              |
     RingRelationInstance::new
              |
      authenticated relation mode
       |                     |
 quotient lift       reduced evaluation
 quotient tail       terminal residue kernels
 common alpha        fused direct setup scan
       |                     |
       +----------+----------+
                  |
     optional compact F/H evaluator
     + optional {-1, 0} support evaluator
              |
          Stage 2 verifier
```

Compressed replay deliberately sets `RingRelationInstance::v()` to an empty
carrier: the zero $\mathbf D$ rows and terminal $p_H$ row are already encoded
in the compressed right-hand side. The separate method-selected opening
preparation retains the full point data needed to bind the claimed evaluation.
[Opening points and digit-innermost
layout](./opening-points-layout.md#witness-order) describes the canonical
physical source and witness order.
