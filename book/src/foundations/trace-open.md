# Trace openings

Akita commits base-field data in the cyclotomic ring

$$
R=F[X]/(X^D+1),
$$

but an opening claim is a scalar in a field $E$ that may be an extension of
$F$. A **trace opening** is the public linear map that converts an
outer-folded ring element into that scalar.

The construction has two inputs:

- $Y\in R$, derived from the polynomial and the outer coordinates of its
  opening point; and
- $P\in R$, derived from the inner coordinates of the opening point.

It is defined so that

$$
\operatorname{TraceOpen}_P(Y)=\widetilde f(r).
$$

This page first reduces multilinear evaluation to a ring pairing, then derives
the extension-field form of that pairing.

## From multilinear evaluation to two rings

Split a polynomial table into three indices:

$$
f[\ell,p,b],
$$

where $\ell$ is an inner index, $p$ is a position inside a block, and $b$ is a
block index. Split the opening point in the same way:

$$
r=(r_{\mathrm{in}},r_{\mathrm{pos}},r_{\mathrm{blk}}).
$$

Write the corresponding interpolation weights as

$$
I_\ell,\qquad Q_p,\qquad B_b.
$$

For a multilinear opening in the Lagrange basis, these are equality weights;
for example,

$$
I_\ell=\operatorname{eq}(r_{\mathrm{in}},\ell).
$$

The evaluation claim is

$$
\widetilde f(r)
=
\sum_{\ell,p,b}I_\ell Q_pB_bf[\ell,p,b].
\tag{1}
$$

Pack each length-$D$ inner slice into a ring element:

$$
F_{p,b}(X)
=
\sum_{\ell=0}^{D-1}f[\ell,p,b]X^\ell.
$$

Evaluate the position and block axes while staying in the ring:

$$
E_b(X)=\sum_pQ_pF_{p,b}(X),
$$

$$
Y(X)=\sum_bB_bE_b(X).
\tag{2}
$$

The coefficient of $X^\ell$ in $Y$ is

$$
[Y]_\ell
=
\sum_{p,b}Q_pB_bf[\ell,p,b].
$$

Equation (1) therefore becomes

$$
\widetilde f(r)
=
\sum_{\ell=0}^{D-1}I_\ell[Y]_\ell.
\tag{3}
$$

Thus $Y$ contains the polynomial after evaluating the outer point, while $P$
will encode the remaining inner weights.

## Base-field trace opening

First suppose $E=F$. Pack the inner weights as

$$
P(X)=\sum_{\ell=0}^{D-1}I_\ell X^\ell.
$$

Let $\sigma_{-1}$ be the ring automorphism

$$
\sigma_{-1}(X)=X^{-1}.
$$

The matching terms in $Y\sigma_{-1}(P)$ are

$$
[Y]_\ell X^\ell\cdot I_\ell X^{-\ell}
=[Y]_\ell I_\ell.
$$

They all contribute to the constant coefficient. The relation
$X^D=-1$ supplies exactly the signs needed when negative exponents are written
in the standard coefficient range. Therefore

$$
\left[Y\sigma_{-1}(P)\right]_0
=
\sum_\ell[Y]_\ell I_\ell.
$$

Combining this with Equation (3) gives the base-field definition

$$
\boxed{
\operatorname{TraceOpen}_P(Y)
=
\left[Y\sigma_{-1}(P)\right]_0.
}
\tag{4}
$$

## Why an extension needs more than coefficient zero

Now let

$$
E/F,qquad K=[E:F]>1,
$$

and put

$$
m=\frac DK.
$$

The supported parameters make $D$ a power of two and require $K$ to divide
$D/2$, so $m$ is an integer and the subgroup below is well defined.

An answer in $E$ has $K$ base-field coordinates. Reading only the constant
coefficient of a ring product would recover only one coordinate. Instead,
Akita identifies $E$ with a $K$-dimensional fixed subring of $R$ and projects
the product into that subring.

Every suitable odd exponent $a$ modulo $2D$ defines an automorphism

$$
\sigma_a(X)=X^a.
$$

Akita uses the subgroup

$$
H=\langle\sigma_{-1},\sigma_{4K+1}\rangle,
$$

which has size $m=D/K$. Its fixed subring is

$$
R^H=\{z\in R:\sigma(z)=z\text{ for every }\sigma\in H\}.
$$

The protocol uses an explicit isomorphism between $E$ and this fixed subring.
Write it as

$$
\iota:E\longrightarrow R^H.
$$

The implementation represents an element of $E$ in the ring-subfield basis

$$
1,e_1,\ldots,e_{K-1},
$$

where

$$
e_j=X^{jD/(2K)}+X^{-jD/(2K)}.
$$

The `FpExtEncoding` contract ensures that an extension field's coordinate
basis agrees with this embedding.

## The subgroup trace

Define the unnormalized subgroup trace

$$
\operatorname{Tr}_H(Z)
=
\sum_{\sigma\in H}\sigma(Z).
\tag{5}
$$

This is not the field trace $\operatorname{Tr}_{E/F}:E\to F$. It maps the
whole ring into the fixed subring:

$$
\operatorname{Tr}_H:R\longrightarrow R^H\simeq E.
$$

Because every automorphism in $H$ fixes $\iota(a)$, the map is $E$-linear:

$$
\operatorname{Tr}_H(\iota(a)Z)
=
\iota(a)\operatorname{Tr}_H(Z).
$$

It is unnormalized, so a fixed element is multiplied by the subgroup size:

$$
\operatorname{Tr}_H(\iota(a))
=
m\iota(a).
\tag{6}
$$

## Pack several extension elements into one ring

The dimensions agree:

$$
\dim_F(E^m)=Km=D=\dim_F(R).
$$

Akita therefore uses an invertible packing map

$$
\psi:E^m\longrightarrow R.
$$

Choose the $m$ shifts

$$
T=
\left\{0,\ldots,\frac{D}{2K}-1\right\}
\cup
\left\{\frac D2,\ldots,\frac D2+\frac{D}{2K}-1\right\}
$$

and enumerate them as $t_0,\ldots,t_{m-1}$. Conceptually,

$$
\psi(y_0,\ldots,y_{m-1})
=
\sum_{i=0}^{m-1}X^{t_i}\iota(y_i).
\tag{7}
$$

The shifts are trace-orthogonal:

$$
\operatorname{Tr}_H
\left(X^{t_i}\sigma_{-1}(X^{t_j})\right)
=
m\delta_{i,j}.
\tag{8}
$$

Equation (8) is the extension-field analogue of selecting the constant
coefficient in Equation (4).

## Derive the extension-field definition

At the ring-subfield boundary, view the outer-folded data and inner opening
weights as vectors in $E^m$:

$$
y=(y_0,\ldots,y_{m-1}),
\qquad
q=(q_0,\ldots,q_{m-1}).
$$

Pack them as

$$
Y=\psi(y),
\qquad
P=\psi(q).
$$

Since $\sigma_{-1}\in H$, it fixes every embedded extension value. Expanding
the product gives

$$
Y\sigma_{-1}(P)
=
\sum_{i,j}X^{t_i-t_j}\iota(y_iq_j).
$$

Apply $\operatorname{Tr}_H$ and use Equation (8). Every cross term with
$i\ne j$ vanishes, while every diagonal term is multiplied by $m$:

$$
\operatorname{Tr}_H
\left(Y\sigma_{-1}(P)\right)
=
m\iota\left(\sum_i y_iq_i\right).
\tag{9}
$$

Normalize by $m$ and decode the fixed-subring element back into $E$:

$$
\boxed{
\operatorname{TraceOpen}_P(Y)
=
\iota^{-1}\left(
\frac1m
\operatorname{Tr}_H
\left(Y\sigma_{-1}(P)\right)
\right).
}
\tag{10}
$$

Equation (9) immediately gives

$$
\operatorname{TraceOpen}_P(Y)
=
\sum_{i=0}^{m-1}y_iq_i.
\tag{11}
$$

So the extension-field construction computes the same inner product as the
base-field construction. The difference is that the subgroup trace preserves
all $K$ coordinates of the answer instead of selecting one coefficient.

## Example: $D=8$ and $K=2$

Here $m=4$. The fixed subring has basis

$$
1,e,
\qquad
e=X^2+X^{-2}=X^2-X^6,
$$

and $e^2=2$. Thus it represents a quadratic extension compatible with
$F[\sqrt 2]$.

The packing shifts are

$$
T=\{0,1,4,5\}.
$$

For $y_i,q_i\in E$, construct

$$
Y=\sum_{i=0}^3X^{t_i}\iota(y_i),
\qquad
P=\sum_{i=0}^3X^{t_i}\iota(q_i).
$$

The subgroup has four automorphisms, and the trace identity is

$$
\operatorname{Tr}_H
\left(Y\sigma_{-1}(P)\right)
=
4\iota(y_0q_0+y_1q_1+y_2q_2+y_3q_3).
$$

Dividing by four and decoding returns the desired $E$-valued inner product.

## From `TraceOpen` to the Stage-2 trace claim

The Stage-2 witness does not contain $Y$ directly. It contains digit
decompositions of the position-folded block rings:

$$
E_b(X)=\sum_hG_h\hat e_{b,h}(X).
$$

The block fold from Equation (2) makes $Y$ the virtual ring

$$
Y(X)=\sum_{b,h}B_bG_h\hat e_{b,h}(X).
$$

By linearity,

$$
\operatorname{TraceOpen}_P(Y)
=
\sum_{b,h,\ell}
\hat e_{b,h,\ell}B_bG_hJ_\ell,
$$

where

$$
J_\ell=\operatorname{TraceOpen}_P(X^\ell).
$$

The public factors $B_bG_hJ_\ell$, together with claim-batching coefficients,
form the transparent Stage-2 trace-weight function $T$. It is supported only
on the $\hat e$ segment of the committed witness. The fused sumcheck proves

$$
v_{\mathrm{tr}}=\sum_xw(x)T(x)
$$

without transmitting or committing the virtual ring $Y$. The trace is not
inserted into the physical ring-relation matrix. Its weight function is built
separately and fused directly into Stage 2, scaled by the equality weight of a
reserved index in the padded $\tau_1$ domain. See
[Sumcheck stages](../how/proving/sumcheck-stages.md#add-the-opening-trace) for
the complete fused claim.

## Implementation map

- [`SubfieldParams`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L73-L176)
  defines and validates $H$.
- [`psi_embed`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L194-L272)
  implements $\psi$.
- [`trace_h`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L178-L192)
  implements Equation (5).
- [`embed_subfield`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L880-L913)
  implements $\iota$.
- [`recover_ring_subfield_inner_product`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L554-L615)
  implements Equation (10), including normalization and decoding.
- [`trace_open_ring_row`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/field_reduction.rs#L621-L665)
  computes all $J_\ell$ values used by Stage 2.
