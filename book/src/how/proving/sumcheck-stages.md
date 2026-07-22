# Sumcheck stages

Every non-terminal Akita fold runs a short sumcheck cascade over the fold
witness:

1. **Stage 1 — digit range check.** Proves that every witness entry is a valid
   balanced digit and outputs one evaluation of the virtual range-image table.
2. **Stage 2 — fused relation sumcheck.** Binds that virtual evaluation to the
   committed witness while proving the fold relation, then produces the next
   opening claim.
3. **Stage 3 — setup product sumcheck.** Optionally carries a recursive setup
   contribution together with the next opening.

The required terminal fold takes a separate direct path: it reveals the
remaining witness and runs none of these sumchecks
([`fold.rs:582`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/core/fold.rs#L582-L606)).

This chapter explains the Stage-1 protocol in detail. It starts with the
simplest sound range check, then derives the complete product-tree protocol.
Stages 2 and 3 are summarized at the end.

## Stage 1: the digit range check

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

### Scope

This protocol checks every position against the common balanced alphabet
$\mathcal{A}_b$. Some compression spans use the narrower alphabet
$\{-1,0\}$; fusing a separate binariness check for those positions remains
outside the current Stage-1 protocol.

## Stage 2: bind the range image and prove the fold relation

After Stage 1, the transcript absorbs `range_image_evaluation` and samples a
fresh batching coefficient. Stage 2 includes the claim
$\widetilde S(r_{\mathsf{range}})$ in its fused relation sumcheck and checks it
against the actual witness term $w(x)(w(x)+1)$. This binds Stage 1's virtual
range-image table to the committed witness. The same sumcheck proves the fold
relation and produces the next-witness opening claim
([`fold.rs:666`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/core/fold.rs#L666-L861)).

## Stage 3: recursive setup contribution

When the level uses recursive setup contribution, Stage 3 batches the public
setup-product claim with Stage 2's next-witness opening. One sumcheck proves
both claims and returns their evaluations at the resulting projected points.
Under direct setup contribution, Stage 3 is absent
([`fold.rs:1079`](https://github.com/LayerZero-Labs/akita/blob/8a4ec9b140e23514b8e53e61b885774f8d8397d7/crates/akita-prover/src/protocol/core/fold.rs#L1079-L1141)).
