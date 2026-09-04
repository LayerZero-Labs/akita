# Sumcheck stages

Every non-terminal Akita fold runs a short sumcheck cascade over the fold
witness:

1. **Stage 1 — digit range check.** Proves that every witness entry is a valid
   balanced digit and outputs one evaluation of the virtual range-image table.
   A fold on the `L2` route also proves the complete physical response norm in
   the final Stage 1 substage.
2. **Stage 2 — fused relation sumcheck.** Proves the ring-switched fold
   relation and binds both Stage 1's virtual range-image value and the opening
   claim carried into the fold to the committed witness; the resulting witness
   evaluation becomes the next opening claim.
3. **Stage 3 — setup product sumcheck.** When recursive setup contribution is
   selected, proves the deferred A/B/D setup contribution and leaves one
   setup-prefix opening claim for the next fold.

This chapter explains the Stage 1 range protocol and the Stage 2 fused relation
protocol in detail. Stage 3 is summarized at the end. The terminal fold runs
none of these sumchecks. When it uses an L2 route, the verifier computes the
clear response norm directly. See [The proving protocol](./proving.md).

## Contents

- [Stage 1: digit range check](#stage-1-digit-range-check)
  - [What it certifies](#what-it-certifies)
  - [The simplest sound design](#the-simplest-sound-design)
  - [Reduce the degree with the range image](#reduce-the-degree-with-the-range-image)
- [The complete Stage-1 protocol](#the-complete-stage-1-protocol)
  - [Quartic leaves and product substages](#quartic-leaves-and-product-substages)
  - [One product substage](#one-product-substage)
  - [The final leaf](#the-final-leaf)
  - [Optional physical norm term](#optional-physical-norm-term)
  - [Domain and challenge order](#domain-and-challenge-order)
  - [The verifier](#the-verifier)
- [Stage 2: fused relation sumcheck](#stage-2-fused-relation-sumcheck)
  - [Start with the ordinary ring relation](#start-with-the-ordinary-ring-relation)
  - [Batch the matrix rows](#batch-the-matrix-rows)
  - [Expand the ring elements into one flat witness](#expand-the-ring-elements-into-one-flat-witness)
  - [Raw and compressed relation terms](#raw-and-compressed-relation-terms)
  - [Add the range-image binding](#add-the-range-image-binding)
  - [Add the opening claim consistency](#add-the-opening-claim-consistency)
  - [The fused Stage-2 claim](#the-fused-stage-2-claim)
  - [Sumcheck rounds and the final point](#sumcheck-rounds-and-the-final-point)
- [Stage 3: recursive setup contribution](#stage-3-recursive-setup-contribution)

## Stage 1: digit range check

### What it certifies

Let

$$
w:\{0,1\}^{n}\rightarrow\mathbb{F}
$$

be the balanced-digit table of the newly committed witness. The level chooses
one basis

$$
b\in\{4,8,16,32,64\},
$$

and every Boolean entry must lie in

$$
\mathcal{A}_b
=
\left\{-\frac b2,\ldots,\frac b2-1\right\}.
$$

This range bound keeps the recursive witness norm under control.

### The simplest sound design

The direct vanishing polynomial for the balanced alphabet is

$$
D_b(W)
=
\prod_{a\in\mathcal{A}_b}(W-a).
$$

A Boolean entry is valid exactly when $D_b(w(x))=0$. Checking only the
unweighted sum of these values would not be sound, because nonzero violations
could cancel. Instead, the protocol anchors the table at a random equality
point $\tau$ and proves

$$
0
=
\sum_{x\in\{0,1\}^{n}}
\operatorname{eq}(\tau,x)\,D_b(w(x)).
$$

The right-hand side is a random evaluation of the multilinear extension of
the Boolean violation table. An equality-factored sumcheck proves this identity
one variable at a time; see
[Equality-factored sum-check](../../foundations/eq-factored-sumcheck.md).

This design is simple, but $D_b$ has degree $b$. Akita represents the same
condition with a degree-$b/2$ polynomial before deciding whether a product
tree is needed.

### Reduce the degree with the range image

Pair the positive digit $k$ with the negative digit $-(k+1)$:

$$
(W-k)(W+k+1)
=
W(W+1)-k(k+1).
$$

Define the pointwise **range image**

$$
S(x)
=
\operatorname{range\_image}(w(x))
=
w(x)\bigl(w(x)+1\bigr)
$$

and roots

$$
c_k=k(k+1),
\qquad
0\le k<\frac b2.
$$

The direct polynomial factors as

$$
D_b(W)
=
\prod_{k=0}^{b/2-1}\left(W(W+1)-c_k\right)
=
R_b\bigl(W(W+1)\bigr),
$$

where

$$
R_b(T)
=
\prod_{k=0}^{b/2-1}(T-c_k).
$$

Thus $w(x)\in\mathcal{A}_b$ exactly when $R_b(S(x))=0$. Stage 1 starts from
the anchored zero claim

$$
0
=
\sum_{x\in\{0,1\}^{n}}
\operatorname{eq}(\tau_0,x)\,R_b(S(x)).
$$

The table $S$ is virtual: it is not committed and is not appended to the
recursive witness. Stage 1 proves the range identity for $S$; Stage 2 later
proves that its final evaluation comes from $w(x)(w(x)+1)$ on the committed
witness.

## The complete Stage-1 protocol

### Quartic leaves and product substages

For basis $4$ or $8$, $R_b$ has degree at most four and one
equality-factored sumcheck proves the anchored identity directly.

For larger bases, the protocol partitions the roots into consecutive groups of
at most four:

$$
L_\ell(T)
=
\prod_{k=4\ell}^{\min(4\ell+3,\,b/2-1)}
(T-c_k).
$$

Their product is $R_b$. Each $L_\ell$ is quartic except at basis $4$, where
the only leaf is quadratic. Product substages prove how these leaves combine,
using only arity-$2$ or arity-$4$ products. The topology is fixed by
[`DigitRangePlan`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/stage1.rs):

| Basis | Product substages | Final leaf |
|---:|---|---|
| 4 | none | one quadratic leaf |
| 8 | none | one quartic leaf |
| 16 | arity 2, emitting 2 child claims | batch of 2 quartic leaves |
| 32 | arity 4, emitting 4 child claims | batch of 4 quartic leaves |
| 64 | arity 2, emitting 2 claims; then arity 4, emitting 8 claims | batch of 8 quartic leaves |

### One product substage

Suppose the current substage has parent tables $P_i$, and each parent is the
pointwise product of $a$ child tables:

$$
P_i(x)
=
\prod_{j=0}^{a-1}C_{i,j}(x),
\qquad a\in\{2,4\}.
$$

Let $\xi$ be the current equality point and $\lambda_i$ the current parent
weights. The carried claim is

$$
v
=
\sum_i\lambda_i\,\widetilde{P_i}(\xi).
\tag{1}
$$

The product substage proves

$$
v
=
\sum_{x\in\{0,1\}^{n}}
\operatorname{eq}(\xi,x)
\sum_i\lambda_i
\prod_{j=0}^{a-1}C_{i,j}(x).
$$

At the sumcheck's sampled point $r$, the prover supplies the child evaluations

$$
u_{i,j}
=
\widetilde{C_{i,j}}(r)
$$

in canonical order. The verifier closes the substage against

$$
\operatorname{eq}(\xi,r)
\sum_i\lambda_i
\prod_{j=0}^{a-1}u_{i,j}.
$$

The protocol then:

1. absorbs the child evaluations in canonical order;
2. samples a fresh interstage challenge $\gamma$;
3. assigns weights $1,\gamma,\gamma^2,\ldots$ in that same order;
4. batches the child evaluations into the next carried claim; and
5. uses $r$ as the equality point for the next substage.

Let the canonical order of the $m$ child nodes be
$C_0,C_1,\ldots,C_{m-1}$, and write their evaluations at $r$ as
$u_h=\widetilde{C_h}(r)$. The child node in position $h$ receives weight
$\gamma^h$, so both parties derive the next claim from the absorbed child
claims and the transcript challenge:

$$
v_{\mathsf{next}}
=
\sum_{h=0}^{m-1}\gamma^h u_h
=
u_0+\gamma u_1+\gamma^2u_2+\cdots.
$$

For the next substage, these canonically ordered child nodes become the new
parent nodes. Set its equality point to $r$ and its parent weights to
$\lambda_h=\gamma^h$. The carried claim is therefore

$$
v_{\mathsf{next}}
=
\sum_{h=0}^{m-1}\lambda_h\,\widetilde{C_h}(r).
\tag{2}
$$

Equation (2) has the same form as Equation (1), the carried claim proved by
each product substage. In other words,
the handoff substitutes $P_i\leftarrow C_h$, $\xi\leftarrow r$, and
$\lambda_i\leftarrow\gamma^h$ in the product-substage claim above.

At the root there is one parent with weight $1$ and claim $0$. Each product
substage expands the current parents into their children; the fresh powers of
$\gamma$ compress those child claims back into one claim for the next
substage. The prover and verifier follow the same transcript order
([`digit_range/mod.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/sumcheck/digit_range/mod.rs),
[`stage1.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/stages/stage1.rs)).

### The final leaf

After all product substages, let $\xi$ be the current equality point, $v$ the
current batched claim, and $\lambda_\ell$ the current leaf weights. Define

$$
B(T)
=
\sum_\ell\lambda_\ell L_\ell(T).
$$

The final equality-factored sumcheck proves

$$
v
=
\sum_{x\in\{0,1\}^{n}}
\operatorname{eq}(\xi,x)\,B(S(x)).
$$

For bases $8$, $16$, $32$, and $64$, $B$ is quartic. For basis $4$, it is
quadratic. When there is no product substage, $v=0$, $\xi=\tau_0$, and
$B=R_b$.

At the final sampled point $r_{\mathsf{range}}$, the proof carries

$$
\mathsf{range\_image\_evaluation}
=
\widetilde S(r_{\mathsf{range}})
=
\sum_{x\in\{0,1\}^{n}}
\operatorname{eq}(r_{\mathsf{range}},x)\,
w(x)\bigl(w(x)+1\bigr).
$$

The verifier closes the leaf against

$$
\operatorname{eq}(\xi,r_{\mathsf{range}})
B\bigl(\mathsf{range\_image\_evaluation}\bigr).
$$

The distinction between the Boolean table and its MLE matters:

$$
\widetilde S(r)
\neq
\widetilde w(r)\bigl(\widetilde w(r)+1\bigr)
$$

in general. The equality $S(x)=w(x)(w(x)+1)$ holds at Boolean vertices, but
multilinear extension does not commute with the quadratic map away from those
vertices. The proof therefore carries the independent
`range_image_evaluation` field
([`levels.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/levels.rs)).

### Optional physical norm term

An `L2` selected fold proves the squared norm of the complete physical folded
response. The physical response domain contains every live Z coefficient once
and assigns zero to padding addresses. `WitnessLayout` defines this domain for
the prover, verifier, and proof size code.

For a large field, the final Stage 1 substage batches the range leaf with

$$
S = \sum_x z(x)^2.
$$

The proof carries the public integer value $S$, one additional coefficient in
each final sumcheck round, and the final virtual evaluation of $z$. The verifier
checks $S$ against the cap stored in the schedule.

For a small field, the allowed digit range may make the full square sum wrap
the base field. The schedule then divides the physical domain into fixed
blocks and proves bounded limb inner products. Each subclaim has a unique
centered integer lift because its public bound is below half the modulus. The
verifier reconstructs $S$ with checked integer arithmetic and rejects a wrong
subclaim count, an overflow, or a value above the cap.

The proof stream has no route header. The schedule determines whether the norm
proof exists and gives its exact shape. An `L∞` fold uses the ordinary Stage 1
shape and carries no norm values.

### Domain and challenge order

Stage 1 views the digits as one flat Boolean table. The live witness occupies a
prefix; every remaining address is public zero padding. Zero is a valid
balanced digit, so padded entries also satisfy the range polynomial.

Ring switching supplies $\tau_0$ in column-then-ring order, while the flat
table binds variables in increasing physical-address-bit order. The protocol
reorders the point so that ring-slot coordinates come first, followed by
column coordinates
([`stage1.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/stage1.rs)).

### The verifier

The verifier replays the same product substages, derives the same interstage
challenges and weights, checks each child-product claim, and closes the final
leaf at `range_image_evaluation`
([`stage1.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/stages/stage1.rs)).

Passing Stage 1 proves that the range-tree claims are internally consistent and
reduces the final leaf to `range_image_evaluation`. It does **not** by itself
tie that virtual value to the committed witness.


## Stage 2: fused relation sumcheck

Stage 2 always proves three statements about the same committed digit witness $w$:

1. $w$ satisfies the ring-switched fold relation;
2. Stage 1's virtual range-image value is correctly derived from $w(w+1)$; and
3. the opening claim for this fold is consistent with $w$.

The protocol fuses all three statements into one sumcheck. The range-image and
opening-consistency terms are shared by both payload modes and both
ring-relation modes. Raw payload uses only the ordinary physical rows;
compressed payload adds compact $\mathbf F/\mathbf H$ relation weights and a
support-restricted $\{-1,0\}$ constraint for the compression digits. Separately,
quotient lifting uses a factored relation weight over explicit quotient digits,
whereas reduced evaluation uses a dense prover weight table over a witness with
no quotient ranges. Neither axis adds a second Stage-2 sumcheck.

When Stage 1 also proves a physical norm, Stage 2 adds the Z virtualization
relation derived from the schedule. It proves that each final physical response
or limb evaluation is the balanced basis recomposition of the committed Z digit
plane evaluations at the same point. The transcript samples the batching
challenge after it has absorbed all Stage 1 claims, including the ordinary
range image. This order prevents one false virtual relation from canceling
another.

### Start with the schedule-selected ring relation

The physical row families and both ring-relation realizations are derived in
[Payload and ring-relation realizations of an Akita
fold](./akita-fold-realizations.md). We first derive the existing quotient-lift
factorization, then state the reduced-evaluation replacement.

Let $w_j(X)$ be the $j$-th ring element encoded by the digit witness, and let
row $i$ of the ordinary extended fold relation be

$$
\sum_j M_{i,j}(X)w_j(X)=h_i(X).
\tag{3}
$$

In quotient-lift mode, the extended relation includes the quotient witness, so
Equation (3) is an
equality rather than a congruence modulo a ring polynomial. The public right
side $h_i$ is zero for some rows and contains the public data
for the remaining rows.

Ring switching samples one field element $\alpha$ and evaluates every ring
polynomial at that point:

$$
\sum_j M_{i,j}(\alpha)w_j(\alpha)=h_i(\alpha).
\tag{4}
$$

The scalar $\alpha$ removes the ring-coefficient dimension from the algebraic
relation. It is not a Boolean sumcheck point.

### Batch the matrix rows

Suppose the ordinary relation has rows indexed by $i$. Its method-independent
rows are $\mathbf A$, $\mathbf B$, and $\mathbf D$, with their quotient terms.
For `EvaluationTrace`, it also includes the legacy fold-consistency row and its
quotient. `SubringCoefficientPacking` keeps the consistency-row slot and its
batching weight, but replaces the legacy coefficients: its packed E/Q
coordinate-plane events join the common relation events, and its
folded-source contribution is added later as the packing-Z structured term.
The protocol samples a Boolean MLE point $\tau_1$ of sufficient width for the
complete physical row domain and defines

$$
\beta_i=\operatorname{eq}(\tau_1,i).
$$

It takes the random linear combination of Equation (4) over all rows. Define

$$
m_j^{\mathrm{ord}}
=
\sum_i\beta_iM_{i,j}(\alpha)
\qquad\text{and}\qquad
h_{\tau}^{\mathrm{raw}}
=
\sum_i\beta_i h_i(\alpha).
$$

The batched relation is the single scalar claim

$$
\sum_j m_j^{\mathrm{ord}}w_j(\alpha)=h_{\tau}^{\mathrm{raw}}.
\tag{5}
$$

This is where the matrix-row dimension goes. It is contracted by $\tau_1$
before Stage 2 starts its witness-address sumcheck. Consequently, the Stage-2
sumcheck point has no matrix-row coordinate.


Stage 2 proves the relation for this batched row. Soundness comes from the fact
that $\tau_1$ was sampled after the witness was committed: a false collection
of rows cannot generally arrange for its random multilinear combination to
vanish.

### Expand the ring elements into one flat witness

For the moment, suppose every witness ring element has $D$ coefficients:

$$
w_j(X)=\sum_{k=0}^{D-1}w_{j,k}X^k.
$$

Evaluating at $\alpha$ gives

$$
w_j(\alpha)=\sum_{k=0}^{D-1}w_{j,k}\alpha^k.
$$

Substituting this into Equation (5) yields

$$
h_{\tau}^{\mathrm{raw}}
=
\sum_{j,k}w_{j,k}\,\alpha^k m_j^{\mathrm{ord}}.
\tag{6}
$$

Now view $(j,k)$ as one flat Boolean address $x$. The low bits select the
coefficient $k$ and the remaining bits select the witness lane $j$. Define

$$
A(k)=\alpha^k,
\qquad
L_{\mathrm{ord}}(j)=m_j^{\mathrm{ord}}.
$$

The native matrix relation weight factors as

$$
R_{\mathrm{ord}}(k,j)=A(k)L_{\mathrm{ord}}(j),
$$

so Equation (6) becomes

$$
h_{\tau}^{\mathrm{raw}}
=
\sum_{x\in\{0,1\}^n}w(x)R_{\mathrm{ord}}(x).
\tag{7}
$$

The Boolean domain is padded with zeros when the live witness length is not a
power of two.

The production protocol also permits different ring dimensions in different
parts of the native relation. It chooses the largest power-of-two coefficient
block common to every role and to the outgoing witness. The low address still
selects a coefficient inside that common block. Any remaining high power of
$\alpha$, together with the matrix entry and its $\beta_i$ row weight, is
absorbed into the lane weight $L_{\mathrm{ord}}$. Thus the exact factorization
$R_{\mathrm{ord}}=A\cdot L_{\mathrm{ord}}$ continues to hold without adding
another sumcheck dimension. The packed E/Q events also preserve this
common-alpha factorization and are included in $R_{\mathrm{ord}}$. The
packing-Z and direct-opening contributions are ordered structured linear terms
that are not forced into it.

Reduced evaluation starts from the same physical rows but replaces each
unreduced product coefficient by the public signed-wrap kernel
$\kappa_{A,\alpha}$. After row batching and flattening, define
$R_{\mathrm{red}}(x)$ to be the complete coefficient multiplying witness entry
$w(x)$. It satisfies the same scalar claim

$$
h_{\tau}^{\mathrm{raw}}
=\sum_x w(x)R_{\mathrm{red}}(x),
$$

but it does not generally factor as one coefficient-power table times one lane
table. The prover therefore materializes and folds $R_{\mathrm{red}}$ as one
ephemeral dense Stage-2 oracle. The verifier evaluates its final MLE from
terminal residue kernels and the fused setup scan without materializing this
table.

For the rest of this chapter, let

$$
R_{\mathrm{rel}}
=
\begin{cases}
R_{\mathrm{ord}}, & \text{quotient lifting},\\
R_{\mathrm{red}}, & \text{reduced evaluation}.
\end{cases}
$$

### Raw and compressed relation terms

Raw mode has no payload-compression term $C_{\mathrm{comp}}$. The mode-selected
$R_{\mathrm{rel}}$ is its complete relation component, and its $\mathbf B$ and $\mathbf D$
right-hand sides contain the transmitted semantic commitments. The scheduled
opening method still contributes $C_{\mathrm{method}}$ below.

Compressed mode retains the method-selected ordinary relation and, in
quotient-lift mode, its quotients, but sets the $\mathbf B$ and $\mathbf D$
right-hand sides to zero.
Their rows now also contain the source-recomposition terms, and the physical
row domain additionally contains the $\mathbf F_1$, $\mathbf H_1$,
$\mathbf F_2$, and $\mathbf H_2$ rows. Thus $R_{\mathrm{rel}}$ remains one
component of the compressed relation, but it is not a complete relation
identity by itself.

To derive the missing component directly, process each compressed physical row
in its native ring using the selected relation realization. Define
$C_{\mathrm{comp}}(x)$ to be the coefficient multiplying the flat witness
entry $w(x)$ in the $\tau_1$-weighted sum of all linear terms not already in
$R_{\mathrm{rel}}$.

In quotient-lift mode, the intuition is simpler than the expanded formula. Its
first line is only the increment to the existing $\mathbf B$ rows:
$\mathbf B\hat{\mathbf t}$ and their ordinary quotient terms are already in
$R_{\mathrm{ord}}$, so $C_{\mathrm{comp}}$ adds the missing negative
recomposition term. The second and third lines are the two newly added
physical rows: $\mathbf F_1$ links the two compression layers, while
$\mathbf F_2$ binds the last layer to the terminal payload $p_F$. Their
quotient terms lift those native-ring congruences to exact identities.
Suppressing component indices, the quotient-lift contribution of one
$\mathbf F$ chain is

$$
\begin{aligned}
{}&-\sum_{i\in I_B}\beta_i
  \operatorname{Rec}_{d_B\leftarrow d_1}
  (\boldsymbol\xi_{F,1})_i(\alpha)\\
&+\beta_{F,1}\Bigl(
  \mathbf F_1\boldsymbol\xi_{F,1}
  -\operatorname{Rec}_{d_1\leftarrow d_2}(\boldsymbol\xi_{F,2})
  -(X^{d_1}+1)q_{F,1}\Bigr)(\alpha)\\
&+\beta_{F,2}\Bigl(
  \mathbf F_2\boldsymbol\xi_{F,2}
  -(X^{d_2}+1)q_{F,2}\Bigr)(\alpha).
\end{aligned}
\tag{7a}
$$

Here $I_B$ is the set of ordinary $\mathbf B$ rows, $\beta_{F,\ell}$ is the
row weight of $\mathbf F_\ell$, and $q_{F,\ell}$ is the quotient ring element
that lifts that compression row. The $\mathbf H$ chain contributes the
analogous terms with $B,F$ replaced by $D,H$. Summing these coefficient
weights over every $\mathbf F$ chain and the shared $\mathbf H$ chain gives
the single compact Boolean table $C_{\mathrm{comp}}$.

Reduced-evaluation mode has the same row ownership and batching, but replaces
each parenthesized lifted identity by its terminal-residue kernel. It has no
$q_{F,\ell}$ or $q_{H,\ell}$ witness spans. The verifier evaluates the
$\mathbf F/\mathbf H$ residue kernels from their separately owned canonical
compression-relation program; they are not columns of the fused public
$\mathbf A/\mathbf B/\mathbf D$ setup tensor. Define the complete compressed
right-hand side in either mode by

$$
h_{\tau}^{\mathrm{comp}}
=
\sum_i\beta_i h_i(\alpha).
$$

The nonzero new targets in $h_{\tau}^{\mathrm{comp}}$ are carried by the
terminal $p_F$ and $p_H$ rows. Adding the directly collected compression
weights to $R_{\mathrm{rel}}$ gives the full compressed relation identity

$$
h_{\tau}^{\mathrm{comp}}
=
\sum_x w(x)
\bigl(R_{\mathrm{rel}}(x)+C_{\mathrm{comp}}(x)\bigr).
\tag{7b}
$$

This follows the same lift-evaluate-batch-flatten procedure as the raw
relation, but the resulting compact table does not need to share the
$A\cdot L_{\mathrm{ord}}$ factorization because its rows use different native
dimensions.

The compression digit spans must also use the stricter alphabet $\{-1,0\}$.
Let $I_{\mathrm{comp}}$ be exactly the union of those $\mathbf F/\mathbf H$
digit intervals and define the Boolean table

$$
B_{\mathrm{comp}}(r_1,x)
=
\mathbf 1_{I_{\mathrm{comp}}}(x)\operatorname{eq}(r_1,x).
$$

With a separate batching challenge $\rho$, compressed mode adds the zero claim

$$
0
=
\sum_x
\rho B_{\mathrm{comp}}(r_1,x)
w(x)\bigl(w(x)+1\bigr).
\tag{7c}
$$

This restriction does not apply to compression quotient digits or alignment
padding. At a non-Boolean sumcheck point, the verifier evaluates the MLE of the
single restricted table $B_{\mathrm{comp}}$; it does not multiply separately
extended support and equality tables.

### Add the range-image binding

Let $r_1$ be the final point produced by Stage 1, and let $s_1$ be the
`range_image_evaluation` carried by its proof. Stage 1 established a claim
about a virtual table. Stage 2 must connect that table to the committed
witness by proving

$$
s_1
=
\sum_x
\operatorname{eq}(r_1,x)w(x)\bigl(w(x)+1\bigr).
\tag{8}
$$

This is a sum over Boolean addresses. It does not claim that
$s_1=\widetilde w(r_1)(\widetilde w(r_1)+1)$, which is generally false.

After absorbing $s_1$, the transcript samples a fresh scalar $\gamma$. The
protocol uses $\gamma$ to batch Equation (8) with the relation claim.

### Add the opening claim consistency

Stage 2 also verifies the incoming opening claim $v_{\mathrm{open}}$ against
the committed witness $w$. The schedule selects how the public opening weight
$T_{\mathrm{open}}(x)$ is prepared. `EvaluationTrace` uses the trace weights
derived in [Field-to-ring evaluation reduction](./field-ring-reduction.md).
`SubringCoefficientPacking` uses the direct coefficient-packing weights
derived in [Field-to-ring evaluation
reduction](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials).
Both methods establish the same linear evaluation-consistency shape

$$
v_{\mathrm{open}}=\sum_x w(x)T_{\mathrm{open}}(x).
\tag{9}
$$

Here we focus only on how this relation is fused into Stage 2. Equation (9)
has the same linear form in $w$ as a row of the ring-switched relation, so the
protocol treats it as a **virtual row** placed immediately after the physical
relation rows. If $i_{\mathrm{open}}$ is that row's index in the padded row
domain, the shared row challenge $\tau_1$ assigns it the weight

$$
\beta_{\mathrm{open}}
=
\operatorname{eq}(\tau_1,i_{\mathrm{open}}).
$$

The virtual row is not inserted into the physical matrix or its
relation-weight factorization. Only its batching weight
$\beta_{\mathrm{open}}$ comes from the row domain. Stage 2 directly fuses

$$
\beta_{\mathrm{open}}v_{\mathrm{open}}
=
\sum_x\beta_{\mathrm{open}}w(x)T_{\mathrm{open}}(x)
$$

into the sumcheck over the flat witness address $x$. In this way, the opening
claim reuses the same row randomness $\tau_1$ as the ring-switched relation
without becoming a physical matrix row.

The implementation does not materialize a dense
$T_{\mathrm{open}}$ table. Define the method-dependent weight
$C_{\mathrm{method}}(x)$ as follows. For `EvaluationTrace`, it is the compact
trace opening weight $\beta_{\mathrm{open}}T_{\mathrm{open}}(x)$; its legacy
fold-consistency row is already in $R_{\mathrm{rel}}$. For
`SubringCoefficientPacking`, which production schedules pair only with
quotient lifting, $R_{\mathrm{rel}}=R_{\mathrm{ord}}$ contains the packed E/Q
relation events in place of the legacy consistency coefficients. Its
$C_{\mathrm{method}}$ is the ordered sum of only the packing-Z and
direct-opening terms. Each contribution is therefore included exactly once.
Stage 2 folds each structured linear term under the same challenges and sums
their final values. This preserves the native relation fast path without
building one dense weight table. The
[physical packing
realization](./root-fold-ring-switch.md#the-packing-consistency-quotient)
explains the $U^s+1$ quotient and the two evaluations of the shared challenge.

### The fused Stage-2 claim

In raw mode, combining Equations (7), (8), and (9) gives the input claim

$$
C_0^{\mathrm{raw}}
=
\gamma s_1+h_{\tau}^{\mathrm{raw}}
+\beta_{\mathrm{open}}v_{\mathrm{open}}.
$$

Stage 2 proves

$$
\begin{aligned}
C_0^{\mathrm{raw}}
=\sum_x \bigl[{}
&\gamma\operatorname{eq}(r_1,x)
  w(x)\bigl(w(x)+1\bigr)\\
&+w(x)R_{\mathrm{rel}}(x)\\
&+w(x)C_{\mathrm{method}}(x)
\bigr].
\end{aligned}
\tag{10}
$$

In compressed mode, the stricter binary claim in Equation (7c) has target zero,
while Equation (7b) supplies the complete compressed relation claim. Therefore

$$
C_0^{\mathrm{comp}}
=
\gamma s_1
+h_{\tau}^{\mathrm{comp}}
+\beta_{\mathrm{open}}v_{\mathrm{open}},
$$

and the sumcheck polynomial gains two terms:

$$
\begin{aligned}
C_0^{\mathrm{comp}}
=\sum_x \bigl[{}
&\gamma\operatorname{eq}(r_1,x)
  w(x)\bigl(w(x)+1\bigr)\\
&+w(x)R_{\mathrm{rel}}(x)\\
&+w(x)C_{\mathrm{comp}}(x)\\
&+\rho B_{\mathrm{comp}}(r_1,x)
  w(x)\bigl(w(x)+1\bigr)\\
&+w(x)C_{\mathrm{method}}(x)
\bigr].
\end{aligned}
\tag{10c}
$$

Every term uses the same flat witness address $x$. This is why each payload
mode still needs only one Stage-2 sumcheck.

### Sumcheck rounds and the final point

Let $P(X)$ denote the mode-selected polynomial expression inside the brackets
in Equation (10) or Equation (10c). In round $t$, after challenges
$r_{2,0},\ldots,r_{2,t-1}$ have been sampled, the prover sends

$$
g_t(Z)
=
\sum_{x_{t+1},\ldots,x_{n-1}\in\{0,1\}}
P(r_{2,0},\ldots,r_{2,t-1},Z,x_{t+1},\ldots,x_{n-1}).
$$

The verifier checks

$$
g_t(0)+g_t(1)=C_t,
$$

samples the next coordinate $r_{2,t}$, and sets

$$
C_{t+1}=g_t(r_{2,t}).
$$

After all $n$ rounds, these coordinates form the flat witness point

$$
r_2=(r_{2,0},\ldots,r_{2,n-1}).
$$

Split it according to the flat address as

$$
r_2=(r_{\mathrm{coeff}},r_{\mathrm{lane}}).
$$

In raw mode, the verifier closes the sumcheck against

$$
\begin{aligned}
C_n^{\mathrm{raw}}={}
&\gamma\operatorname{eq}(r_1,r_2)
  \widetilde w(r_2)\bigl(\widetilde w(r_2)+1\bigr)\\
&+\widetilde w(r_2)\widetilde R_{\mathrm{rel}}(r_2)\\
&+\widetilde w(r_2)\widetilde C_{\mathrm{method}}(r_2).
\end{aligned}
\tag{11}
$$

In compressed mode, the ordinary factors are evaluated for the compressed
layout, and the verifier closes the sumcheck against

$$
\begin{aligned}
C_n^{\mathrm{comp}}={}
&\gamma\operatorname{eq}(r_1,r_2)
  \widetilde w(r_2)\bigl(\widetilde w(r_2)+1\bigr)\\
&+\widetilde w(r_2)\widetilde R_{\mathrm{rel}}(r_2)\\
&+\widetilde w(r_2)\widetilde C_{\mathrm{comp}}(r_2)\\
&+\rho\widetilde B_{\mathrm{comp}}(r_1,r_2)
  \widetilde w(r_2)\bigl(\widetilde w(r_2)+1\bigr)\\
&+\widetilde w(r_2)\widetilde C_{\mathrm{method}}(r_2).
\end{aligned}
\tag{11c}
$$

The value $\widetilde w(r_2)$ becomes the next-witness opening claim in both
modes. The shared range term and the restricted compression-binary term have
degree at most three in each sumcheck variable. The ordinary, compression, and
method-dependent structured terms have degree at most two. Therefore Stage 2
keeps the same degree-three bound in both modes.

The random objects have separate jobs:

| Object | Shape | What it removes or binds |
|---|---|---|
| $\alpha$ | one field element | evaluates the ring-polynomial variable $X$ |
| $\tau_1$ | a point over matrix rows | batches all relation rows into one virtual row |
| $\gamma$ | one field element | batches range-image consistency with the linear claims |
| $\rho$ | one field element in compressed mode | batches the zero-valued $\{-1,0\}$ constraint |
| $r_1$ | Stage-1 output point over flat witness addresses | identifies the carried range-image evaluation |
| $r_2$ | Stage-2 output point over flat witness addresses | reduces the complete fused claim to one witness evaluation |

In particular, $\alpha$, $\tau_1$, and $r_2$ are not coordinates of one larger
point. They contract three different domains.

## Stage 3: recursive setup contribution

Stage 3 is controlled by the setup-contribution mode, not by the raw or
compressed payload equation above. Payload mode determines whether Stage 2 has
the extra $C_{\mathrm{comp}}$ and $B_{\mathrm{comp}}$ terms. Setup mode
determines whether the verifier scans the public A/B/D setup directly or
defers that contribution to Stage 3. Current schedules permit recursive setup
offloading only inside the compressed prefix, but the Stage-3 algebra remains
an A/B/D setup product and does not include the $\mathbf F/\mathbf H$ maps.
Reduced evaluation is admitted only with direct setup contribution, so any
fold that reaches Stage 3 necessarily uses quotient lifting. Schedule
validation rejects a reduced fold with an incoming setup prefix or deferred
Stage 3 before transcript replay.

After Stage 2 fixes its final relation-address point, let $S[\lambda,y]$ be
coefficient $y$ of shared setup ring $\lambda$, and let $W(\lambda)$ collect
all A/B/D row, witness-address, gadget, and mixed-dimension projection weights
for that setup ring. The deferred claim used by Stage 2 is

$$
\sigma_S
=
\sum_{\lambda,y}S[\lambda,y]W(\lambda)\alpha^y.
\tag{12}
$$

Pad the setup-index domain to a power of two and set $W(\lambda)=0$ outside
the active A/B/D footprint. If $\widetilde S$, $\widetilde W$, and
$\widetilde A_\alpha$ are the MLEs of the setup table, the setup-index weight,
and $A_\alpha(y)=\alpha^y$, Stage 3 runs a degree-two sumcheck on

$$
F(Y,X)
=
\widetilde S(Y,X)\widetilde W(X)\widetilde A_\alpha(Y),
\qquad
\sigma_S=
\sum_{Y\in\{0,1\}^{n_y}}
\sum_{X\in\{0,1\}^{n_{\mathrm{setup}}}}F(Y,X).
\tag{13}
$$

The implementation binds coefficient variables first and setup-index
variables second. At its final point
$\rho_3=(\rho_y,\rho_{\mathrm{setup}})$, the verifier checks

$$
C_{\mathrm{final}}
=
\widetilde S(\rho_y,\rho_{\mathrm{setup}})
\widetilde W(\rho_{\mathrm{setup}})
\widetilde A_\alpha(\rho_y).
\tag{14}
$$

This is a setup-only sumcheck. It does not batch Stage 2's next-witness opening.
When the next level selects an authenticated setup prefix, it leaves the
independent setup-prefix opening

$$
\bigl(\rho_3,\widetilde S(\rho_3)\bigr).
$$

The next fold then receives that opening as one precommitted polynomial group
and receives the Stage-2 next-witness opening as a second group at $r_2$. Its
grouped opening protocol later authenticates both claims against their own
commitments. Under direct setup contribution, Stage 3 is absent and the
verifier computes $\sigma_S$ by scanning the required public setup prefix.
See [Setup contribution and Stage 3](../verifying/setup_contribution.md) for
the mixed-dimension plan and setup-prefix checks.
