# Ring relation in an Akita fold

This page describes the ring-valued relations proved by one non-terminal
Akita fold. The presentation starts with one polynomial group, one opening
claim, and one common ring dimension:

$$
R=F[X]/(X^D+1).
$$

The current implementation also supports multiple polynomial groups,
multiple claims, chunked witnesses, and different ring dimensions for
different relation rows. Those extensions change the layout, but not the four
core relations developed below.

The Akita paper presents a more general matrix with additional compression
relations. Its basic Greyhound relation motivates the four row families here;
the current implementation is the source of truth for the rows and witness
layout documented on this page.

The goal of the fold is to replace the current polynomial blocks by a smaller
digit witness while proving that the new witness is consistent with:

1. the opening data computed from the old polynomial;
2. the inner and outer commitments; and
3. the random fold of the old polynomial blocks.

These statements form the **physical ring relation**. The scalar evaluation
claim from [Field-to-ring evaluation
reduction](./field-ring-reduction.md) is a separate field-valued relation. It
is fused with the physical rows later, but it is not a row of the ring matrix
described on this page.

## Contents

- [Objects entering the fold](#objects-entering-the-fold)
  - [Why the witness is digit-decomposed](#why-the-witness-is-digit-decomposed)
  - [Polynomial blocks and inner digits](#polynomial-blocks-and-inner-digits)
  - [Partial evaluations and opening digits](#partial-evaluations-and-opening-digits)
  - [The folded response and its digitization](#the-folded-response-and-its-digitization)
- [The four physical relation families](#the-four-physical-relation-families)
  - [Fold-evaluation consistency](#1-fold-evaluation-consistency)
  - [Inner-commitment consistency](#2-inner-commitment-consistency)
  - [Outer-commitment consistency](#3-outer-commitment-consistency)
  - [Opening-commitment consistency](#4-opening-commitment-consistency)
- [Assemble the ring relation](#assemble-the-ring-relation)
- [Relation layouts beyond the basic setting](#relation-layouts-beyond-the-basic-setting)
  - [Multiple polynomial groups](#multiple-polynomial-groups)
  - [Multiple witness chunks](#multiple-witness-chunks)
- [Lift the ring relation before sumcheck](#lift-the-ring-relation-before-sumcheck)
- [The scalar opening claim is a virtual row](#the-scalar-opening-claim-is-a-virtual-row)
- [Code reference](#code-reference)

## Objects entering the fold

### Why the witness is digit-decomposed

The witness committed for the next level must have bounded coefficients. This
shortness condition is essential for the Module-SIS binding argument: two
different bounded openings of the same linear commitment would give a short,
nonzero vector in the kernel of its commitment matrix.

Gadget decomposition provides the bounded representation. For a power-of-two
base $g$ and digit depth $\delta$, define

$$
\mathbf G_{g,n}
=
\mathbf I_n\otimes(1,g,\ldots,g^{\delta-1}).
$$

A balanced decomposition of $\mathbf x$ is a digit vector $\hat{\mathbf x}$
such that

$$
\mathbf x=\mathbf G_{g,n}\hat{\mathbf x},
\qquad
\hat x_i\in
\{-g/2,\ldots,g/2-1\}.
$$

The decomposition specifies this small-digit representation; the protocol's
range check proves that the committed coordinates really lie in the required
range. Akita may use different bases and depths for the inner, outer, opening,
fold-response, and quotient decompositions. Below, $G_a^{\mathrm{in}}$,
$G_h^{\mathrm{out}}$, $G_h^{\mathrm{open}}$, and $G_f^{\mathrm{fold}}$ denote
the scalar gadget weights used to decompose $F_{p,b}$, $\mathbf t_b$, $E_b$,
and $\mathbf z$, respectively. They are entries of the corresponding gadget
rows defined above. We omit the role-specific base and depth from this notation
for simplicity.

### Polynomial blocks and inner digits

As in the previous page, split the ring-valued polynomial table into blocks.
Let $b$ index a live block and $p$ a position inside the block. Write the ring
at that location as

$$
F_{p,b}(X)\in R.
$$

Digit-decompose each ring with public inner gadget weights
$G_a^{\mathrm{in}}$:

$$
F_{p,b}(X)
=
\sum_a G_a^{\mathrm{in}}s_{b,p,a}(X).
\tag{1}
$$

For one block, collect all digit rings $s_{b,p,a}$ into a vector
$\mathbf{s}_b$. The inner commitment matrix $\mathbf A$ maps that vector to

$$
\mathbf t_b
=
\mathbf A\mathbf s_b.
\tag{2}
$$

Each coordinate of $\mathbf t_b$ is decomposed again, now with the outer
gadget weights $G_h^{\mathrm{out}}$:

$$
t_{b,\rho}(X)
=
\sum_hG_h^{\mathrm{out}}\hat t_{b,\rho,h}(X),
\tag{3}
$$

where $\rho$ selects a row of $\mathbf A$. Stack the $\hat t$ digits from all
blocks. The public outer commitment is

$$
\mathbf u
=
\mathbf B\hat{\mathbf t}.
\tag{4}
$$

The matrices $\mathbf A$ and $\mathbf B$ therefore serve different purposes:
$\mathbf A$ creates an inner image for each block, while $\mathbf B$ commits
the digit-decomposed inner images across all blocks.

### Partial evaluations and opening digits

Let $Q_p$ be the position weight derived from the opening point. For the
base-field setting of the previous page, $Q_p\in F$ acts as a constant in
$R$. Evaluate the position coordinate inside each block:

$$
E_b(X)
=
\sum_pQ_pF_{p,b}(X).
\tag{5}
$$

Digit-decompose each $E_b$ with the opening gadget weights
$G_h^{\mathrm{open}}$:

$$
E_b(X)
=
\sum_hG_h^{\mathrm{open}}\hat e_{b,h}(X).
\tag{6}
$$

The fold witness contains the digit rings $\hat e$, not a second copy of the
recomposed $E_b$. To bind those digits, Akita computes an opening commitment

$$
\mathbf v_D
=
\mathbf D\hat{\mathbf e}.
\tag{7}
$$

The subscript in $\mathbf v_D$ distinguishes this ring vector from the scalar
opening target $v_{\mathrm{tr}}$. Equation (7) is a commitment relation; it
does not prove that the multilinear evaluation equals
$v_{\mathrm{tr}}$.

### The folded response and its digitization

After the relevant data is fixed, the transcript samples one sparse
ring-valued fold challenge $c_b(X)$ for each live block. The prover folds the
original block digits:

$$
z_{p,a}(X)
=
\sum_b c_b(X)s_{b,p,a}(X).
\tag{8}
$$

The response $\mathbf z$ no longer carries a live-block index. This is the
fold's main reduction toward a smaller next-level witness, but combining the
blocks increases coefficient magnitudes. Let $\sigma_\infty$ bound the
coefficient norm of every digit block $\mathbf s_b$, and let
$\omega=\max_b\lVert c_b\rVert_1$. Negacyclic multiplication gives

$$
\begin{aligned}
\lVert\mathbf z\rVert_{\infty,\mathrm{coef}}
&\le
\sum_b
\lVert c_b\mathbf s_b\rVert_{\infty,\mathrm{coef}}\\
&\le
|\mathcal B|\,\omega\,\sigma_\infty,
\end{aligned}
$$

where $\mathcal B$ is the set of live blocks. The schedule fixes the challenge
family, its relevant norm bounds, an admissible fold-response bound
$\beta_{\mathrm{fold}}$, and a digit depth large enough to represent the
accepted response. The implementation may resample the transcript nonce until
the resulting $\mathbf z$ fits that scheduled bound. This grinding helps the
honest prover find a compact response; the range check on its committed digits
is what certifies the bound in the protocol.

Before digitizing $\mathbf z$, the two challenge-dependent relations already
follow directly from linearity. For the partial evaluations,

$$
\begin{aligned}
\sum_b c_bE_b
&=
\sum_b c_b\sum_pQ_pF_{p,b}\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}
\left(\sum_bc_bs_{b,p,a}\right)\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}.
\end{aligned}
$$

Using the opening digits from Equation (6), this is

$$
\sum_{b,h}c_bG_h^{\mathrm{open}}\hat e_{b,h}
=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}.
$$

Similarly, $\mathbf t_b=\mathbf A\mathbf s_b$ implies

$$
\sum_bc_b\mathbf t_b
=
\mathbf A\left(\sum_bc_b\mathbf s_b\right)
=
\mathbf A\mathbf z.
$$

For row $\rho$ of $\mathbf A$, and using the outer digits from Equation (3),
this becomes

$$
\sum_{b,h}c_bG_h^{\mathrm{out}}\hat t_{b,\rho,h}
=
\sum_{p,a}A_{\rho,(p,a)}z_{p,a}.
$$

These identities explain the relations in terms of the raw folded response.
The next-level committed witness, however, must again consist of bounded
digits so that its shortness is certified for the Module-SIS binding argument.
Akita therefore decomposes $\mathbf z$ once more:

$$
z_{p,a}(X)
=
\sum_fG_f^{\mathrm{fold}}\hat z_{p,a,f}(X).
\tag{9}
$$

The three main digit segments produced so far are therefore

$$
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}.
\tag{10}
$$

They have different origins:

| Segment | What it digit-decomposes | Why it is needed |
|---|---|---|
| $\hat{\mathbf z}$ | the challenge-folded block digits $\mathbf z$ | becomes the smaller folded response |
| $\hat{\mathbf e}$ | the position-folded rings $E_b$ | carries the opening data into the fold |
| $\hat{\mathbf t}$ | the inner images $\mathbf t_b$ | binds the folded response to the existing commitment |

## The four physical relation families

The verifier must check that the three segments in Equation (10) describe the
same original polynomial and commitment. Akita expresses the checks as four
families of linear equations over $R$.

### 1. Fold-evaluation consistency

Fold the recomposed partial evaluations using the same challenges as in
Equation (8):

$$
\sum_{b,h}
c_bG_h^{\mathrm{open}}\hat e_{b,h}.
\tag{11}
$$

Alternatively, first fold the original digit blocks into $\hat z$, recompose
them with Equation (9), and then apply the position weights:

$$
\sum_{p,a,f}
Q_pG_a^{\mathrm{in}}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
\tag{12}
$$

Both expressions equal $\sum_bc_bE_b$. The first physical row therefore
checks

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{open}}\hat e_{b,h}
=
\sum_{p,a,f}
Q_pG_a^{\mathrm{in}}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{13}
$$

This is called the `consistency` row in the code. It binds the partial
evaluation digits $\hat e$ to the folded response $\hat z$. It does **not**
contain the scalar opening target $v_{\mathrm{tr}}$.

Notice that this row uses the random fold challenges $c_b$, not the block
opening weights $B_b$ from the previous page. The $B_b$ weights belong to the
separate evaluation-correctness relation on $\hat e$.

### 2. Inner-commitment consistency

For every row $\rho$ of $\mathbf A$, fold the corresponding recomposed inner
images:

$$
\sum_{b,h}
c_bG_h^{\mathrm{out}}\hat t_{b,\rho,h}.
\tag{14}
$$

By linearity of $\mathbf A$, this must equal row $\rho$ of $\mathbf A$ applied
to the folded response:

$$
\sum_{p,a,f}
A_{\rho,(p,a)}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
\tag{15}
$$

Thus every $\mathbf A$ row checks

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{out}}\hat t_{b,\rho,h}
=
\sum_{p,a,f}
A_{\rho,(p,a)}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{16}
$$

There is no factor $G_a^{\mathrm{in}}$ on the right of Equation (16):
$\mathbf A$ already acts on the inner digit vector $\mathbf s_b$, whose
columns are indexed by $(p,a)$.

### 3. Outer-commitment consistency

The $\hat t$ segment must still open the public commitment that entered this
fold:

$$
\boxed{
\mathbf B\hat{\mathbf t}
=
\mathbf u.
}
\tag{17}
$$

Unlike Equations (13) and (16), this relation does not use the fold
challenges. It checks the existing outer commitment directly.

### 4. Opening-commitment consistency

The $\hat e$ segment is bound by the opening commitment from Equation (7):

$$
\boxed{
\mathbf D\hat{\mathbf e}
=
\mathbf v_D.
}
\tag{18}
$$

This relation also does not use the fold challenges. It prevents the prover
from changing the partial-evaluation digits after $\mathbf v_D$ has been
absorbed.

## Assemble the ring relation

Define the pre-switch witness

$$
\mathbf w_0
=
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}.
\tag{19}
$$

The four relation families can be written as one matrix equation

$$
\boxed{
\mathbf M_0\mathbf w_0=\mathbf y
\quad\text{over }R.
}
\tag{20}
$$

For one group, the physical row order and right-hand side are:

| Physical rows | Meaning | Right-hand side |
|---|---|---|
| `consistency` | Equation (13) | $0$ |
| $\mathbf A$ rows | Equation (16) | $\mathbf 0$ |
| $\mathbf B$ rows | Equation (17) | $\mathbf u$ |
| $\mathbf D$ rows | Equation (18) | $\mathbf v_D$ |

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
\tag{21}
$$

The matrix is usually not materialized as one dense object. Its entries come
from the fold challenges, opening weights, gadget weights, and the setup
matrices $\mathbf A$, $\mathbf B$, and $\mathbf D$. The code generates these
contributions directly from the canonical witness layout.

## Relation layouts beyond the basic setting

The relation above uses one polynomial group and one witness chunk. The code
also supports multiple groups and multiple chunks. These cases change how the
physical rows or witness columns are arranged, while preserving the basic
relations above.

### Multiple polynomial groups

In zkVM applications, polynomials such as execution traces, advice, and
preprocessed data may be committed at different times. When one commitment is
formed, the prover may not yet know which other commitments it will later be
opened with, the opening point, or the root schedule that will combine them.
Akita calls the polynomials held under one independently formed outer
commitment a **commitment group**. All claims in one group share an opening
point. In the current implementation, all groups are opened at one shared root
point. One group is the **final/new group**, while commitments formed earlier
enter as **precommitted groups** with already-fixed parameters.

Proving each commitment group separately would repeat the entire recursive
opening protocol. Akita instead batches the groups in one root transition and
then resumes the ordinary single-opening recursion. The batching does not
merge the group witnesses or commitments: each group keeps its own folded
response $\mathbf z_g$, digit witnesses $\hat{\mathbf e}_g$ and
$\hat{\mathbf t}_g$, and group-local `consistency | A | B` relations. Within
the physical ring matrix, only the opening-commitment relation spans the
groups: one $\mathbf D$ matrix acts on their concatenated opening digits. The
field-level evaluation trace separately batches their claimed evaluations.

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

There is no sum over $g$. Each group keeps a separate response
$\mathbf z_g$, opening digits $\hat{\mathbf e}_g$, and inner-commitment digits
$\hat{\mathbf t}_g$. The basic consistency, $\mathbf A$, and $\mathbf B$
relations are repeated independently for every group. In particular, the
$\mathbf B_g$ rows bind $\hat{\mathbf t}_g$ to that group's public commitment
$\mathbf u_g$.

Among the four physical relation families, only the $\mathbf D$ relation
combines witness data from different groups. It acts once on the concatenated
opening digits:

$$
\hat{\mathbf e}_{\mathrm{all}}
=
\big\Vert_{g\in\mathrm{relation\ order}}\hat{\mathbf e}_g,
\qquad
\mathbf D\hat{\mathbf e}_{\mathrm{all}}=\mathbf v_D.
$$

In the canonical physical row order, the final/new group is placed first,
followed by the precommitted groups. The shared $\mathbf D$ rows remain at the
end:

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

Consequently, the full right-hand side is

$$
\mathbf y
=
\big\Vert_{g\in\mathrm{relation\ order}}
[0\mid\mathbf 0_{\mathbf A_g}\mid\mathbf u_g]
\quad\Vert\quad
\mathbf v_D.
$$

This relation is block sparse: a group's consistency, $\mathbf A_g$, and
$\mathbf B_g$ rows touch only that group's witness segment, whereas the
shared $\mathbf D$ rows touch the $\hat{\mathbf e}_g$ segments from every
group. Stage 2 batches all physical rows into one sumcheck, but this batching
does not merge the group-local relations.

For a single chunk per group, the corresponding pre-switch witness layout is

$$
\mathbf w_0
=
\big\Vert_{g\in\mathrm{relation\ order}}
[\hat{\mathbf z}_g\mid\hat{\mathbf e}_g\mid\hat{\mathbf t}_g].
$$

The quotient digits for all physical rows are stored once, in one shared
$\hat{\mathbf r}$ tail after these group segments.

The root fold consumes this multi-group structure. After ring switching, the
group segments and the shared quotient tail form one witness

$$
\mathbf w^{\mathrm{next}}
=
\mathbf w_0\Vert\hat{\mathbf r},
$$

which is committed once for the next level. The output of the multi-group root
is therefore the basic recursive object: one polynomial group, one committed
witness, and one opening claim at one point. The original root groups remain
only as ranges inside the flat witness; they no longer define separate folded
responses or relation rows.

### Multiple witness chunks

The chunked or distributed layout further partitions each group's live blocks
into disjoint ranges $I_{g,k}$. Chunk $k$ computes the partial folded response

$$
\mathbf z_{g,k}
=
\sum_{b\in I_{g,k}}c_{g,b}\mathbf s_{g,b},
\qquad
\mathbf z_g=\sum_k\mathbf z_{g,k}.
$$

The canonical layout is group-major and then chunk-minor:

$$
\boxed{
\mathbf w
=
\big\Vert_g\big\Vert_k
[\hat{\mathbf z}_{g,k}
 \mid\hat{\mathbf e}_{g,k}
 \mid\hat{\mathbf t}_{g,k}]
\quad\Vert\quad
\hat{\mathbf r}.
}
$$

Every chunk has a full-shaped local $\hat z$ segment, while its $\hat e$ and
$\hat t$ segments contain only the live blocks owned by that chunk. Chunking
adds witness columns, not relation rows: the chunk matrices are stacked
horizontally and contribute to the same group-level `consistency | A | B`
rows. The $\mathbf D$ rows and the quotient tail $\hat{\mathbf r}$ are shared
across all groups and chunks.

## Lift the ring relation before sumcheck

Equation (20) is an equality modulo $X^D+1$. Sumcheck, however, needs a field
identity. Choose the canonical representatives of degree less than $D$ for
all ring elements. There is then one quotient polynomial for every physical
row:

$$
\widetilde{\mathbf M}_0(X)\widetilde{\mathbf w}_0(X)
-
\widetilde{\mathbf y}(X)
=
(X^D+1)\mathbf r(X).
\tag{22}
$$

Digit-decompose the quotient vector:

$$
\mathbf r(X)
=
\mathbf G_r\hat{\mathbf r}(X),
\tag{23}
$$

and append its digits to the committed witness:

$$
\boxed{
\mathbf w
=
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}
\;\Vert\;
\hat{\mathbf r}.
}
\tag{24}
$$

Move the denominator term to the left and define

$$
\mathbf M_{\mathrm{ext}}(X)
=
\left[
\widetilde{\mathbf M}_0(X)
\;\middle|\;
-(X^D+1)\mathbf G_r
\right].
\tag{25}
$$

The extended relation is the exact polynomial identity

$$
\boxed{
\mathbf M_{\mathrm{ext}}(X)\widetilde{\mathbf w}(X)
=
\widetilde{\mathbf y}(X).
}
\tag{26}
$$

This distinction is important: Equation (22) uses the quotient
$\mathbf r$, while Equation (26) already includes the quotient digits
$\hat{\mathbf r}$ inside $\mathbf w$. The denominator term must not be added
to the right-hand side a second time.

Ring switching now samples $\alpha$ and evaluates Equation (26):

$$
\mathbf M_{\mathrm{ext}}(\alpha)\mathbf w(\alpha)
=
\mathbf y(\alpha).
\tag{27}
$$

Equation (27) is the field relation consumed by Stage 2. The
[Sumcheck stages](./sumcheck-stages.md#stage-2-fused-relation-sumcheck) page
explains how $\tau_1$ batches its physical rows and how the resulting relation
is proved over the flat witness address.

## The scalar opening claim is a virtual row

Two different statements involve $\hat e$, and they should not be conflated:

| Statement | Form | Physical ring row? | Ring-switch quotient? |
|---|---|---:|---:|
| opening commitment | $\mathbf D\hat{\mathbf e}=\mathbf v_D$ | yes | yes |
| evaluation correctness | $\sum_xw(x)T(x)=v_{\mathrm{tr}}$ | no | no |

The second statement is derived in [Field-to-ring evaluation
reduction](./field-ring-reduction.md#express-the-direct-relation-as-a-sumcheck-claim).
It is already a linear equation over the field coefficients of the committed
witness. Akita therefore treats it as an `EvaluationTrace` virtual row after
the physical rows. It reuses the same row-batching challenge $\tau_1$, but it
is absent from $\mathbf M_0$, $\mathbf y$, and the quotient vector
$\mathbf r$.

[Sumcheck stages](./sumcheck-stages.md#stage-2-fused-relation-sumcheck)
continues from Equation (27) and fuses the physical relation, the virtual
evaluation row, and the range-image binding into one Stage-2 sumcheck.

## Code reference

The current prover follows the construction above:

1. **Build the partial-evaluation and fold witnesses.**
   [`RingRelationProver::new`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation.rs#L433-L760)
   decomposes $E_b$ into $\hat e$, computes
   $\mathbf v_D=\mathbf D\hat{\mathbf e}$, samples the fold challenges, and
   builds $\mathbf z$.
2. **Assemble the public relation statement.**
   [`assemble_relation_rhs`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-types/src/proof/relation.rs#L286-L353)
   lays out $\mathbf y$ as
   `consistency | A | B | D`, while
   [`RingRelationInstance`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-types/src/proof/ring_relation.rs#L82-L220)
   carries the public challenges, points, and right-hand side.
3. **Prepare the digit segments.**
   [`ring_switch_build_w`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_switch/coeffs.rs#L253-L455)
   extracts $\hat t$ from the commitment hint and prepares
   $\hat z\Vert\hat e\Vert\hat t$ in the canonical witness layout.
4. **Compute the row quotients.**
   [`compute_multi_group_relation_quotient`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs#L412-L690)
   computes one quotient for each `consistency`, $\mathbf A$, $\mathbf B$, and
   $\mathbf D$ row. `ring_switch_build_w` decomposes them and appends
   $\hat r$.
5. **Evaluate the extended relation.**
   [`build_relation_weight_events`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_switch/relation_weights.rs#L398-L870)
   emits the contributions of all four row families and the quotient columns
   after evaluation at $\alpha$ and batching by $\tau_1$.

The main data flow is:

```text
old polynomial blocks and commitment hints
                  |
                  v
RingRelationProver::new
|-- position-folded rings E_b --> e_hat
|-- inner-image hints ----------> t_hat
|-- fold challenges ------------> z
|-- D * e_hat ------------------> v_D
`-- [0 | 0_A | u | v_D] -------> relation rhs y
                  |
                  v
ring_switch_build_w
|-- compute_multi_group_relation_quotient --> r
|-- decompose z ----------------------------> z_hat
|-- decompose r ----------------------------> r_hat
`-- emit [z_hat | e_hat | t_hat | r_hat] ---> committed witness w
                  |
                  v
build_relation_weight_events
`-- M_ext(alpha), row-batched by tau_1 ------> Stage 2
```

### Public statement: `RingRelationInstance`

[`RingRelationInstance`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-types/src/proof/ring_relation.rs#L82-L220)
contains the public relation statement. It contains only values that the
verifier can reconstruct:

| Field | Mathematical meaning |
|---|---|
| `group_challenges()` | fold challenges $c_b$ |
| `group_opening_point()` | ordinary opening weights, including $Q_p$ and $B_b$ |
| `group_ring_multiplier_point()` | ring multipliers used by the physical consistency row |
| `rhs()` | $\mathbf y=[0\mid\mathbf 0_A\mid\mathbf u\mid\mathbf v_D]$ in the basic setting |
| `v()` | $\mathbf v_D=\mathbf D\hat{\mathbf e}$ |
| `role_dims()` | the $\mathbf A$-, $\mathbf B$-, and $\mathbf D$-row ring dimensions |

### Prover witness: `RingRelationWitness`

[`RingRelationWitness`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation_witness.rs#L141-L220)
is the prover-only aggregate witness. It holds the fold-grinding nonce and one
[`RingRelationGroupWitness`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation_witness.rs#L8-L140)
per polynomial group. In the basic setting, the vector contains one group:

| Field | Mathematical meaning |
|---|---|
| `z_folded_rings` | folded response $\mathbf z$, before decomposition into $\hat z$ |
| `z_folded_centered_per_chunk` | chunk-local folded responses $\mathbf z_k$ |
| `e_folded` | recomposed position-folded rings $E_b$ |
| `e_hat` | opening digits $\hat{\mathbf e}$ |
| `hint` | commitment hint containing $\hat{\mathbf t}$ |

The quotient output $\mathbf r$ is computed after these structures are built.
Its digits are appended when `ring_switch_build_w` emits the flat
$\hat z\Vert\hat e\Vert\hat t\Vert\hat r$ witness.

### Verifier reconstruction

The verifier does not receive a serialized `RingRelationInstance`. In
[`verify_fold`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-verifier/src/protocol/core/fold.rs#L646-L741),
it reconstructs the public instance from the transcript and public proof data:

```text
public commitment rows, opening points, v_D, and transcript
                            |
                            v
rederive the per-group fold challenges
                            |
                            v
assemble_relation_rhs
                            |
                            v
RingRelationInstance::new
                            |
                            v
ring_switch_verifier --------------------------------------> Stage 2 verifier
```

Only the public instance is reconstructed on the verifier. The
`RingRelationWitness` and its group witnesses remain prover-only.

The canonical multi-group and multi-chunk physical layout is described in
[Opening points and digit-innermost
layout](./opening-points-layout.md#witness-order).
