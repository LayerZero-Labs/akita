# Sumcheck stages

Every non-terminal Akita fold runs a short sumcheck cascade over the fold
witness:

1. **Stage 1 — digit range check.** Proves that every witness entry is a valid
   balanced digit and outputs one evaluation of the virtual range-image table.
2. **Stage 2 — fused relation sumcheck.** Proves the ring-switched fold
   relation and binds both Stage 1's virtual range-image value and the opening
   claim carried into the fold to the committed witness; the resulting witness
   evaluation becomes the next opening claim.
3. **Stage 3 — setup product sumcheck.** Optionally carries a recursive setup
   contribution together with the next opening.

The required terminal fold takes a separate direct path: it reveals the
remaining witness and runs none of these sumchecks
([`fold.rs:582`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/core/fold.rs#L582-L606)).

This chapter explains the Stage-1 range protocol and the Stage-2 fused
relation protocol in detail. Stage 3 is summarized at the end.

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
[`DigitRangePlan`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-types/src/proof/stage1.rs#L179-L400):

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
([`digit_range/mod.rs:230`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/sumcheck/digit_range/mod.rs#L230-L299),
[`stage1.rs:167`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-verifier/src/stages/stage1.rs#L167-L267)).

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
([`levels.rs:20`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-types/src/proof/levels.rs#L20-L25)).

### Domain and challenge order

Stage 1 views the digits as one flat Boolean table. The live witness occupies a
prefix; every remaining address is public zero padding. Zero is a valid
balanced digit, so padded entries also satisfy the range polynomial.

Ring switching supplies $\tau_0$ in column-then-ring order, while the flat
table binds variables in increasing physical-address-bit order. The protocol
reorders the point so that ring-slot coordinates come first, followed by
column coordinates
([`stage1.rs:19`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-types/src/proof/stage1.rs#L19-L170)).

### The verifier

The verifier replays the same product substages, derives the same interstage
challenges and weights, checks each child-product claim, and closes the final
leaf at `range_image_evaluation`
([`stage1.rs:167`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-verifier/src/stages/stage1.rs#L167-L267)).

Passing Stage 1 proves that the range-tree claims are internally consistent and
reduces the final leaf to `range_image_evaluation`. It does **not** by itself
tie that virtual value to the committed witness.


## Stage 2: fused relation sumcheck

Stage 2 proves three statements about the same committed digit witness $w$:

1. $w$ satisfies the ring-switched fold relation;
2. Stage 1's virtual range-image value is correctly derived from $w(w+1)$; and
3. the opening claim for this fold is consistent with $w$.

The protocol fuses all three statements into one sumcheck. We first isolate
the fold-relation term to make its matrix-row dimension explicit, then
incorporate the range-image and opening-consistency terms.

### Start with the ring relation

Let $w_j(X)$ be the $j$-th ring element encoded by the digit witness, and let
row $i$ of the extended fold relation be

$$
\sum_j M_{i,j}(X)w_j(X)=h_i(X).
\tag{3}
$$

The extended relation includes the quotient witness, so Equation (3) is an
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

Suppose the relation has rows indexed by $i$. The protocol samples a Boolean
MLE point $\tau_1$ of sufficient width for that row domain and defines

$$
\beta_i=\operatorname{eq}(\tau_1,i).
$$

It takes the random linear combination of Equation (4) over all rows. Define

$$
m_j
=
\sum_i\beta_iM_{i,j}(\alpha)
\qquad\text{and}\qquad
h_{\tau}
=
\sum_i\beta_i h_i(\alpha).
$$

The batched relation is the single scalar claim

$$
\sum_j m_jw_j(\alpha)=h_{\tau}.
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
h_{\tau}
=
\sum_{j,k}w_{j,k}\,\alpha^k m_j.
\tag{6}
$$

Now view $(j,k)$ as one flat Boolean address $x$. The low bits select the
coefficient $k$ and the remaining bits select the witness lane $j$. Define

$$
A(k)=\alpha^k,
\qquad
L(j)=m_j.
$$

The relation weight factors as

$$
R(k,j)=A(k)L(j),
$$

so Equation (6) becomes

$$
h_{\tau}
=
\sum_{x\in\{0,1\}^n}w(x)R(x).
\tag{7}
$$

The Boolean domain is padded with zeros when the live witness length is not a
power of two.

The production protocol also permits different ring dimensions in different
parts of the relation. It chooses the largest power-of-two coefficient block
common to every role and to the outgoing witness. The low address still
selects a coefficient inside that common block. Any remaining high power of
$\alpha$, together with the matrix entry and its $\beta_i$ row weight, is
absorbed into the lane weight $L$. Thus the exact factorization $R=A\cdot L$
continues to hold without adding another sumcheck dimension.

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

The protocol also has an incoming opening target $v_{\mathrm{tr}}$.
[Multilinear evaluation reduction](./trace-open-reduction.md) derives how
evaluating the
outer-folded polynomial ring against the inner opening point becomes a public
linear function of the $\hat e$ digits. Write that function as $T(x)$. Then

$$
v_{\mathrm{tr}}=\sum_x w(x)T(x).
\tag{9}
$$

The opening trace is **not** inserted into the ring-relation matrix. The
physical matrix and its relation-weight factorization contain only the
consistency, $A$, $B$, and $D$ rows.

The protocol nevertheless reserves the next index $i_{\mathrm{tr}}$ in the
padded row domain used by $\tau_1$. Its batching weight is

$$
\beta_{\mathrm{tr}}
=
\operatorname{eq}(\tau_1,i_{\mathrm{tr}}).
$$

Only this scalar comes from the row domain. The trace function $T(x)$ is built
separately from the matrix weights, and
$\beta_{\mathrm{tr}}w(x)T(x)$ is fused directly into the Stage-2 sumcheck over
the flat witness address $x$. Thus the trace reuses $\tau_1$ without becoming
an evaluated matrix row.

### The fused Stage-2 claim

Combining Equations (7), (8), and (9), the input claim is

$$
C_0
=
\gamma s_1+h_{\tau}+\beta_{\mathrm{tr}}v_{\mathrm{tr}}.
$$

Stage 2 proves

$$
\begin{aligned}
C_0
=\sum_x \bigl[{}
&\gamma\operatorname{eq}(r_1,x)
  w(x)\bigl(w(x)+1\bigr)\\
&+w(x)A(k(x))L(j(x))\\
&+\beta_{\mathrm{tr}}w(x)T(x)
\bigr].
\end{aligned}
\tag{10}
$$

All three terms use the same flat witness address $x$. This is why they can be
proved by one sumcheck.

### Sumcheck rounds and the final point

Let $P(X)$ denote the multilinear-polynomial expression inside the brackets in
Equation (10). In round $t$, after challenges
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

The verifier closes the sumcheck against

$$
\begin{aligned}
C_n={}
&\gamma\operatorname{eq}(r_1,r_2)
  \widetilde w(r_2)\bigl(\widetilde w(r_2)+1\bigr)\\
&+\widetilde w(r_2)
  \widetilde A(r_{\mathrm{coeff}})
  \widetilde L(r_{\mathrm{lane}})\\
&+\beta_{\mathrm{tr}}\widetilde w(r_2)\widetilde T(r_2).
\end{aligned}
\tag{11}
$$

The value $\widetilde w(r_2)$ becomes the next-witness opening claim. The
range term has degree three in each sumcheck variable: one degree from the
equality polynomial and two from $w(w+1)$. The relation and trace terms each
have degree two. Therefore Stage 2 sends degree-three round polynomials.

The random objects have separate jobs:

| Object | Shape | What it removes or binds |
|---|---|---|
| $\alpha$ | one field element | evaluates the ring-polynomial variable $X$ |
| $\tau_1$ | a point over matrix rows | batches all relation rows into one virtual row |
| $\gamma$ | one field element | batches range-image consistency with the linear claims |
| $r_1$ | Stage-1 output point over flat witness addresses | identifies the carried range-image evaluation |
| $r_2$ | Stage-2 output point over flat witness addresses | reduces the complete fused claim to one witness evaluation |

In particular, $\alpha$, $\tau_1$, and $r_2$ are not coordinates of one larger
point. They contract three different domains.

### Implementation map

The implementation builds the row weights with
`eq_tau1.eval_at(row)`, factors the resulting flat relation table into the
common alpha and lane factors, and evaluates that factorization at $r_2$
([`relation_weights.rs:442`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/ring_switch/relation_weights.rs#L442-L520),
[`relation_weights.rs:234`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/ring_switch/relation_weights.rs#L234-L310),
[`ring_switch.rs:627`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-verifier/src/protocol/ring_switch.rs#L627-L657)).
Those relation weights stop at the physical fold-consistency, $A$, $B$, and
$D$ rows. The trace index is the next reserved padded-domain index, but no
matrix row is created for it. The prover builds its weight function separately
and adds it directly to the relation weight during the shared witness scan
([`relation_range_image/mod.rs:281`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs#L281-L300)).
The fused identity is implemented by
[`relation_range_image/mod.rs`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs#L1-L77).

## Stage 3: recursive setup contribution

When the level uses recursive setup contribution, Stage 3 batches the public
setup-product claim with Stage 2's next-witness opening. One sumcheck proves
both claims and returns their evaluations at the resulting projected points.
Under direct setup contribution, Stage 3 is absent
([`fold.rs:1079`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/core/fold.rs#L1079-L1141)).
