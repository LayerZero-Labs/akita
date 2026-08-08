# Basic relations in an Akita fold

This page describes the ring-valued relations proved by one non-terminal
Akita fold. The presentation starts with one polynomial group, one opening
claim, and one common ring dimension:

$$
R=F[X]/(X^D+1).
$$

The current implementation also supports more elaborate physical layouts —
commitment groups, witness chunks, and different ring dimensions — but those
extensions do not change the four core relations developed below. This page
establishes only the basic case; advanced layouts are outside its scope.

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

- [Inputs and objects derived in the fold](#inputs-and-objects-derived-in-the-fold)
  - [The committed polynomial and opening query](#the-committed-polynomial-and-opening-query)
  - [Balanced digit representations](#balanced-digit-representations)
  - [Polynomial blocks and commitment hint](#polynomial-blocks-and-commitment-hint)
  - [Partial evaluations and opening digits](#partial-evaluations-and-opening-digits)
  - [The folded response and its digitization](#the-folded-response-and-its-digitization)
- [The four physical relation families](#the-four-physical-relation-families)
  - [Fold-evaluation consistency](#1-fold-evaluation-consistency)
  - [Inner-commitment consistency](#2-inner-commitment-consistency)
  - [Outer-commitment consistency](#3-outer-commitment-consistency)
  - [Opening-commitment consistency](#4-opening-commitment-consistency)
- [Assemble the ring relation](#assemble-the-ring-relation)
- [Lift the ring relation before sumcheck](#lift-the-ring-relation-before-sumcheck)
- [The scalar opening claim is a virtual row](#the-scalar-opening-claim-is-a-virtual-row)
- [Code reference](#code-reference)

## Inputs and objects derived in the fold

A fold starts from a public opening claim

$$
\widetilde f(r)=v
$$

for a polynomial whose commitment $\mathbf u$ is already fixed. At a recursive
level, $\mathbf u$ is the next-witness commitment produced by the preceding
level; at the root, the original polynomial commitment plays the same role.

The prover and verifier have different views of these inputs. The prover holds
the polynomial blocks, their inner-digit representation, the commitment hint
generated when $\mathbf u$ was formed, and the public commitment itself. The
verifier knows $\mathbf u$ and the opening claim, but receives neither the
blocks nor the hint.

From these inputs, the current fold derives two new representations of the
committed polynomial. The opening point determines partial evaluations $E_b$
inside each block, while fresh transcript challenges fold the old block digits
into a response $\mathbf z$. Both are digit-decomposed before entering the next
committed witness. The four relation families later prove that these derived
objects are consistent with the same hidden opening of $\mathbf u$.

### The committed polynomial and opening query

As in the previous page, split the ring-valued polynomial table into blocks.
Let $b$ index a live block and $p$ a position inside that block. Pack the inner
coefficient axis into the ring element

$$
F_{p,b}(X)\in R.
$$

The [field-to-ring evaluation
reduction](./field-ring-reduction.md#the-evaluation-problem) splits $r$ into
inner, position, and block coordinates and defines their interpolation weights
$I_\ell$, $Q_p$, and $B_b$. We reuse those definitions here rather than
deriving them again. The position weights $Q_p$ produce $E_b$ below, while the
block weights $B_b$ and inner weights $I_\ell$ belong to the field-valued
evaluation relation. The opening claim enters this page with target $v$. The
evaluation reduction later writes its trace-form target as
$v_{\mathrm{tr}}$; for the single base-field claim considered here, the valid
relation has $v_{\mathrm{tr}}=v$.

### Balanced digit representations

Both the existing commitment opening and the next committed witness must have
bounded coefficients. This shortness condition is what lets commitment binding
reduce to Module-SIS: two distinct bounded openings of the same linear
commitment would yield a short, nonzero kernel vector.

Akita obtains these bounded representations by decomposing ring coefficients
into balanced base-$g$ digits. Let $g$ be an even power of two, let $\delta$ be
the digit depth, and let
$\mathbf x=(x_0,\ldots,x_{n-1})\in R^n$. A balanced decomposition of
$\mathbf x$ consists of digit rings $\hat x_{i,h}(X)\in R$, indexed by
$0\le i<n$ and $0\le h<\delta$, such that

$$
x_i(X)
=
\sum_{h=0}^{\delta-1}g^h\hat x_{i,h}(X),
\qquad
[\hat x_{i,h}]_\ell
\in
\{-g/2,\ldots,g/2-1\}
$$

for every coefficient position $0\le \ell<D$. Stack the digit rings with $h$
innermost into $\hat{\mathbf x}\in R^{n\delta}$. Define the gadget row and its
block-diagonal recomposition matrix by

$$
\mathbf g_{g,\delta}
=
(1,g,\ldots,g^{\delta-1}),
\qquad
\mathbf G_{g,n}
=
\mathbf I_n\otimes\mathbf g_{g,\delta}
\in R^{n\times n\delta}.
$$

The coefficientwise identities then become the vector equation

$$
\boxed{
\mathbf x
=
\mathbf G_{g,n}\hat{\mathbf x}.
}
$$

The entries of $\mathbf G_{g,n}$ are public scalars embedded as constant ring
elements. Thus $\mathbf G_{g,n}$ is a deterministic **recomposition matrix**,
not a commitment matrix such as $\mathbf A$, $\mathbf B$, or $\mathbf D$.
Digit decomposition produces $\hat{\mathbf x}$; multiplication by
$\mathbf G_{g,n}$ reconstructs $\mathbf x$. The protocol's range check
certifies that the committed coefficients of $\hat{\mathbf x}$ lie in the
balanced digit range.

Akita chooses separate bases and depths for different witness roles. To keep
the derivation readable, write $G_a^{\mathrm{in}}$,
$G_h^{\mathrm{out}}$, $G_h^{\mathrm{open}}$, and
$G_f^{\mathrm{fold}}$ for the corresponding scalar gadget weights. The four
recomposition identities used on this page are

$$
\begin{aligned}
F_{p,b}(X)
&=
\sum_a G_a^{\mathrm{in}}s_{b,p,a}(X),
\\
t_{b,\rho}(X)
&=
\sum_h G_h^{\mathrm{out}}\hat t_{b,\rho,h}(X),
\\
E_b(X)
&=
\sum_h G_h^{\mathrm{open}}\hat e_{b,h}(X),
\\
z_{p,a}(X)
&=
\sum_f G_f^{\mathrm{fold}}\hat z_{p,a,f}(X).
\end{aligned}
$$

The first two identities describe commitment-side data already fixed by
$\mathbf u$: $\mathbf s_b$ is the incoming inner-digit representation, and
$\hat{\mathbf t}$ is reconstructed from the incoming hint. The latter two
digit families, $\hat{\mathbf e}$ and $\hat{\mathbf z}$, are newly derived from
the opening point and fold challenges. We now place each identity in its
protocol context.

### Polynomial blocks and commitment hint

The commitment-side inputs are fixed before the polynomial is queried. For
each block, the prover has the inner digit rings $s_{b,p,a}$ and can therefore
recompose the polynomial rings as

$$
F_{p,b}(X)
=
\sum_a G_a^{\mathrm{in}}s_{b,p,a}(X).
\tag{1}
$$

For one block, collect the digit rings $s_{b,p,a}$ into a vector
$\mathbf{s}_b$. The inner commitment matrix $\mathbf A$ maps this block vector
to an inner image

$$
\mathbf t_b
=
\mathbf A\mathbf s_b.
\tag{2}
$$

Each coordinate of $\mathbf t_b$ is itself represented by balanced outer
digits:

$$
t_{b,\rho}(X)
=
\sum_h G_h^{\mathrm{out}}\hat t_{b,\rho,h}(X),
\tag{3}
$$

where $\rho$ selects a row of $\mathbf A$. Stack these digits over all blocks
to obtain $\hat{\mathbf t}$. The outer commitment matrix $\mathbf B$ then gives
the public commitment

$$
\mathbf u
=
\mathbf B\hat{\mathbf t}.
\tag{4}
$$

The two matrices have distinct roles: $\mathbf A$ forms one inner image per
block, whereas $\mathbf B$ commits the digit-decomposed inner images across all
blocks. The prover's commitment hint stores the recomposed inner images
$\mathbf t_b$; this fold decomposes them to recover $\hat{\mathbf t}$. At a
recursive level, the polynomial blocks, hint, and $\mathbf u$ were produced by
the preceding level. At the root, they come from the original commitment. The
verifier receives only $\mathbf u$ from this commitment-side data. Equations
(1)--(4) describe the hidden opening that the later relation rows bind to that
public value.

### Partial evaluations and opening digits

The first new witness object derived in this fold comes from the opening point.
Use its position weights $Q_p$ to evaluate the position coordinate inside each
block. In the base-field setting of the previous page, $Q_p\in F$ acts as a
constant in $R$:

$$
E_b(X)
=
\sum_p Q_pF_{p,b}(X).
\tag{5}
$$

Digit-decompose each $E_b$ with the opening gadget weights
$G_h^{\mathrm{open}}$:

$$
E_b(X)
=
\sum_h G_h^{\mathrm{open}}\hat e_{b,h}(X).
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

The subscript in $\mathbf v_D$ distinguishes this ring-vector commitment from
the scalar opening target $v$ and its trace-form counterpart
$v_{\mathrm{tr}}$. Equation (7) binds the newly derived opening digits; it does
not by itself prove the scalar evaluation claim.

### The folded response and its digitization

The opening commitment $\mathbf v_D$ binds the prover to
$\hat{\mathbf e}$, but it does not by itself show that the corresponding
partial evaluations were computed from the incoming polynomial blocks. For
every block $b$, correctness requires

$$
\boxed{
E_b
=
\sum_p Q_pF_{p,b}
=
\sum_{p,a}Q_pG_a^{\mathrm{in}}s_{b,p,a}.
}
\tag{8}
$$

Equation (8) connects the partial evaluation derived in this fold to the
incoming block witness $\mathbf s_b$. The public commitment $\mathbf u$
creates a second consistency requirement. Equation (4),
$\mathbf u=\mathbf B\hat{\mathbf t}$, binds the outer digits
$\hat{\mathbf t}$, which recompose the inner images $\mathbf t_b$ through
Equation (3). However, this commitment relation does not by itself show that
the inner images were computed from the incoming block witness. The missing
link is the blockwise relation $\mathbf t_b=\mathbf A\mathbf s_b$ from
Equation (2).

Checking both relations separately for every live block would retain the block
index in the next proof. Instead, after $\mathbf u$ and $\mathbf v_D$ have
been fixed, the transcript samples one sparse ring-valued challenge $c_b(X)$
for each live block. These challenges are separate from the query weights
$B_b$. The prover folds the incoming block witnesses into one response:

$$
z_{p,a}(X)
=
\sum_b c_b(X)s_{b,p,a}(X).
\tag{9}
$$

The folded response $\mathbf z$ no longer carries a live-block index. Because
both blockwise relations are linear in $\mathbf s_b$, the same challenges
batch them into relations on this single response. For the partial
evaluations, Equations (8) and (9) give

$$
\begin{aligned}
\sum_b c_bE_b
&=
\sum_{b,p,a}c_bQ_pG_a^{\mathrm{in}}s_{b,p,a}\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}
\left(\sum_b c_bs_{b,p,a}\right)\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}.
\end{aligned}
\tag{9a}
$$

For the inner commitments, Equations (2) and (9) give the vector relation

$$
\begin{aligned}
\sum_b c_b\mathbf t_b
&=
\mathbf A\left(\sum_b c_b\mathbf s_b\right)\\
&=
\mathbf A\mathbf z.
\end{aligned}
\tag{9b}
$$

Equation (9a) says that evaluating within each block and then folding gives the
same result as first folding the block witnesses into $\mathbf z$ and then
applying the evaluation weights. Equation (9b) similarly connects
$\mathbf z$ to the inner images bound through $\mathbf u$. If any blockwise
relation is incorrect, its error is unlikely to disappear in the corresponding
random combination. Thus the challenges remove the block index while
preserving the two links from the incoming witness: one to the partial
evaluations created in this fold, and one to the commitment that entered it.

This compression has an arithmetic cost: combining the blocks increases
coefficient magnitudes. Let $\sigma_\infty$ bound the coefficient norm of
every digit block $\mathbf s_b$, and let
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

Equations (9a) and (9b) are identities among the recomposed values; they do not
yet use the opening digits $\hat{\mathbf e}$, the outer digits
$\hat{\mathbf t}$, or bounded digits for $\mathbf z$. The next-level committed
witness contains digit rings rather than $\mathbf z$ itself so that its
shortness is certified for the Module-SIS binding argument. Akita therefore
decomposes $\mathbf z$ once more:

$$
z_{p,a}(X)
=
\sum_f G_f^{\mathrm{fold}}\hat z_{p,a,f}(X).
\tag{10}
$$

The three main digit segments assembled for the next witness are therefore

$$
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}.
\tag{11}
$$

They have different origins:

| Segment | What it digit-decomposes | Why it is needed |
|---|---|---|
| $\hat{\mathbf z}$ | the challenge-folded block digits $\mathbf z$ | becomes the smaller folded response |
| $\hat{\mathbf e}$ | the position-folded rings $E_b$ | carries the opening data into the fold |
| $\hat{\mathbf t}$ | the inner images $\mathbf t_b$ | binds the folded response to the existing commitment |

## The four physical relation families

Equation (11) specifies how the three digit segments are assembled, but it
does not impose any algebraic relation among them. Substituting the balanced
recompositions from Equations (3), (6), and (10) into the recomposed identities
(9a) and (9b) gives two relations among the private witness segments.
Equations (4) and (7) provide two additional relations that anchor those
segments to the public commitments. Together, these give four families of
linear equations over

$$
R=F[X]/(X^D+1).
$$

Every sum, product, and equality in this section is computed in $R$; vector
equations are interpreted coordinatewise in $R$. The first two families
connect the private witness segments to one another. The last two anchor those
segments to the public commitments $\mathbf u$ and $\mathbf v_D$.

### 1. Fold-evaluation consistency

Equation (9a) is the fold-evaluation identity among the recomposed values. To
express it in terms of the next-fold witness, substitute the balanced digit
representations of $E_b$ from Equation (6) and $\mathbf z$ from Equation (10).
This gives the fold-evaluation consistency relation

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{open}}\hat e_{b,h}
=
\sum_{p,a,f}
Q_pG_a^{\mathrm{in}}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{12}
$$

This relation uses the random fold challenges $c_b$, not the block-opening weights
$B_b$. The latter belong to the separate field-valued evaluation relation on
$\hat{\mathbf e}$. In particular, this ring relation does not contain the
scalar target $v_{\mathrm{tr}}$.

### 2. Inner-commitment consistency

Equation (9b) is the inner-commitment identity among the recomposed values.
The next-fold witness stores their balanced digits instead. Substitute
Equation (3) for $\mathbf t_b$ and Equation (10) for $\mathbf z$. For every
row $\rho$ of $\mathbf A$, this gives

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{out}}\hat t_{b,\rho,h}
=
\sum_{p,a,f}
A_{\rho,(p,a)}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{13}
$$

There is no factor $G_a^{\mathrm{in}}$ on the right of Equation (13):
$\mathbf A$ already acts on the inner digit vector $\mathbf s_b$, whose
columns are indexed by $(p,a)$.

### 3. Outer-commitment consistency

The first two families compare private witness segments but do not yet tie
them to the commitment seen by the verifier. The outer-commitment relation
provides that public anchor by requiring $\hat{\mathbf t}$ to open the
commitment that entered this fold:

$$
\boxed{
\mathbf B\hat{\mathbf t}
=
\mathbf u.
}
\tag{14}
$$

This is a direct commitment check and therefore does not use the fold
challenges.

### 4. Opening-commitment consistency

Finally, the opening-commitment relation anchors $\hat{\mathbf e}$ to the
public ring vector $\mathbf v_D$ that was absorbed before the fold challenges
were sampled:

$$
\boxed{
\mathbf D\hat{\mathbf e}
=
\mathbf v_D.
}
\tag{15}
$$

Because $\mathbf v_D$ is fixed before the fold challenges are sampled, this
relation, together with the boundedness of $\hat{\mathbf e}$ and Module-SIS
binding, prevents the prover from adapting the partial-evaluation digits after
learning those challenges. Like the other three families, it is a relation
over $R$; it is distinct from the field-valued scalar evaluation claim.

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
\tag{16}
$$

The four relation families can be written as one matrix equation

$$
\boxed{
\mathbf M_0\mathbf w_0=\mathbf y
\quad\text{over }R.
}
\tag{17}
$$

Let $n_A$, $n_B$, and $n_D$ denote the row counts of $\mathbf A$,
$\mathbf B$, and $\mathbf D$, respectively. In the basic one-group layout,
the four relation families occupy $1+n_A+n_B+n_D$ physical rows. Their order
and right-hand sides are:

| Physical rows | Count | Meaning | Right-hand side |
|---|---:|---|---|
| `consistency` | $1$ | Equation (12) | $0$ |
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

The matrix is usually not materialized as one dense object. Its entries come
from the fold challenges, opening weights, gadget weights, and the setup
matrices $\mathbf A$, $\mathbf B$, and $\mathbf D$. The code generates these
contributions directly from the canonical witness layout.

## Lift the ring relation before sumcheck

Equation (17) is an equality modulo $X^D+1$. Sumcheck, however, needs a field
identity. Choose the canonical representatives of degree less than $D$ for
all ring elements. There is then one quotient polynomial for every physical
row:

$$
\widetilde{\mathbf M}_0(X)\widetilde{\mathbf w}_0(X)
-
\widetilde{\mathbf y}(X)
=
(X^D+1)\mathbf r(X).
\tag{19}
$$

Digit-decompose the quotient vector:

$$
\mathbf r(X)
=
\mathbf G_r\hat{\mathbf r}(X),
\tag{20}
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
\tag{21}
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
\tag{22}
$$

The extended relation is the exact polynomial identity

$$
\boxed{
\mathbf M_{\mathrm{ext}}(X)\widetilde{\mathbf w}(X)
=
\widetilde{\mathbf y}(X).
}
\tag{23}
$$

This distinction is important: Equation (19) uses the quotient
$\mathbf r$, while Equation (23) already includes the quotient digits
$\hat{\mathbf r}$ inside $\mathbf w$. The denominator term must not be added
to the right-hand side a second time.

Ring switching now samples $\alpha$ and evaluates Equation (23):

$$
\mathbf M_{\mathrm{ext}}(\alpha)\mathbf w(\alpha)
=
\mathbf y(\alpha).
\tag{24}
$$

Equation (24) is the field relation consumed by Stage 2. The
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
continues from Equation (24) and fuses the physical relation, the virtual
evaluation row, and the range-image binding into one Stage-2 sumcheck.

## Code reference

The current prover uses canonical entry points that also support more general
layouts. With one group, one chunk, and one common ring dimension, they reduce
to the construction above:

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
   $\mathbf D$ row. Despite its general name, this is also the canonical
   single-group path. `ring_switch_build_w` decomposes the quotients and
   appends $\hat r$.
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
|-- compute relation quotients ----------> r
|-- decompose z -------------------------> z_hat
|-- decompose r -------------------------> r_hat
`-- emit [z_hat | e_hat | t_hat | r_hat] --> committed witness w
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
| `group_challenges()[0]` | fold challenges $c_b$ |
| `group_opening_point(0)` | ordinary opening weights, including $Q_p$ and $B_b$ |
| `group_ring_multiplier_point(0)` | ring multipliers used by the physical consistency row |
| `rhs()` | $\mathbf y=[0\mid\mathbf 0_A\mid\mathbf u\mid\mathbf v_D]$ in the basic setting |
| `v()` | $\mathbf v_D=\mathbf D\hat{\mathbf e}$ |

### Prover witness: `RingRelationWitness`

[`RingRelationWitness`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation_witness.rs#L141-L220)
is the prover-only aggregate witness. In the basic setting, its `groups`
vector contains one
[`RingRelationGroupWitness`](https://github.com/LayerZero-Labs/akita/blob/eea8443841ed4a701bf84a9f6415aa9415d6250d/crates/akita-prover/src/protocol/ring_relation_witness.rs#L8-L140):

| Field | Mathematical meaning |
|---|---|
| `z_folded_rings` | folded response $\mathbf z$, before decomposition into $\hat z$ |
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
rederive the fold challenges
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
`RingRelationWitness` remains prover-only. [Opening points and digit-innermost
layout](./opening-points-layout.md#witness-order) specifies the canonical
physical source and digit order used by the implementation.
