# Multilinear evaluation reduction

This page considers one base-field evaluation claim:

$$
f:\{0,1\}^n\rightarrow F,
\qquad
r\in F^n,
\qquad
\widetilde f(r)=v.
$$

Both the polynomial table and the opening point are defined over the base
field $F$. Akita commits the table through the cyclotomic ring

$$
R=F[X]/(X^D+1).
$$

The goal is to turn the multilinear evaluation into a multiplication of two
ring elements, define that multiplication as a `TraceOpen` operation, and then
write the same evaluation claim directly as a linear relation on the committed
fold witness.

Base-field polynomials evaluated at extension-field points are left as a stub
at the end of the page.

## The evaluation problem

Choose a ring dimension $D=2^d$ and a power-of-two number of positions per
block. Re-index the polynomial table as

$$
f[\ell,p,b],
$$

where:

- $\ell\in[D]$ is an inner index that will become a ring coefficient;
- $p$ is a position inside a block; and
- $b$ is a block index.

Missing entries in a partial final block are public zeros.

Split the opening point in the same order:

$$
r=(r_{\mathrm{in}},r_{\mathrm{pos}},r_{\mathrm{blk}}).
$$

Write the corresponding interpolation weights as

$$
I_\ell,
\qquad
Q_p,
\qquad
B_b.
$$

For a multilinear opening in the Lagrange basis, these are equality weights:

$$
I_\ell=\operatorname{eq}(r_{\mathrm{in}},\ell),
\qquad
Q_p=\operatorname{eq}(r_{\mathrm{pos}},p),
\qquad
B_b=\operatorname{eq}(r_{\mathrm{blk}},b).
$$

The evaluation claim is therefore

$$
\widetilde f(r)
=
\sum_{\ell,p,b}I_\ell Q_pB_bf[\ell,p,b].
\tag{1}
$$

Akita evaluates the three axes in the order

$$
\text{position}\longrightarrow\text{block}\longrightarrow\text{inner}.
$$

## From multilinear evaluation to two rings

### Pack the inner axis

For each position $p$ and block $b$, pack the inner slice into a ring:

$$
F_{p,b}(X)
=
\sum_{\ell=0}^{D-1}f[\ell,p,b]X^\ell
\in R.
\tag{2}
$$

This is only a change of representation. The table entry
$f[\ell,p,b]$ becomes the coefficient of $X^\ell$.

### Fold the position and block axes

First evaluate the position coordinate independently inside every block:

$$
E_b(X)
=
\sum_pQ_pF_{p,b}(X).
\tag{3}
$$

The coefficient of $X^\ell$ in $E_b$ is

$$
[E_b]_\ell
=
\sum_pQ_pf[\ell,p,b].
$$

Next evaluate the block coordinate:

$$
Y(X)
=
\sum_bB_bE_b(X).
\tag{4}
$$

Now

$$
[Y]_\ell
=
\sum_{p,b}Q_pB_bf[\ell,p,b].
\tag{5}
$$

Thus $Y$ contains the polynomial after evaluating the position and block
parts of $r$. Only the inner coordinate remains.

### Pack the inner opening weights

Pack the remaining weights into a second ring:

$$
P(X)
=
\sum_{\ell=0}^{D-1}I_\ell X^\ell.
\tag{6}
$$

The two rings have different sources:

| Ring | Derived from | Meaning |
|---|---|---|
| $Y$ | $f$, $r_{\mathrm{pos}}$, and $r_{\mathrm{blk}}$ | the polynomial after the two outer folds |
| $P$ | $r_{\mathrm{in}}$ | the weights for the remaining inner fold |

Using Equation (5), the original evaluation can already be written as

$$
\widetilde f(r)
=
\sum_{\ell=0}^{D-1}I_\ell[Y]_\ell.
\tag{7}
$$

## Base-field trace opening

Let $\sigma_{-1}$ be the ring automorphism

$$
\sigma_{-1}(X)=X^{-1}.
$$

For any $Z\in R$, define

$$
\boxed{
\operatorname{TraceOpen}_P(Z)
:=
\left[Z(X)\sigma_{-1}(P(X))\right]_0,
}
\tag{8}
$$

where $[\cdot]_0$ denotes the constant coefficient in
$F[X]/(X^D+1)$.

If

$$
Z(X)=\sum_\ell[Z]_\ell X^\ell,
$$

then the matching terms in $Z\sigma_{-1}(P)$ are

$$
[Z]_\ell X^\ell\cdot I_\ell X^{-\ell}
=
[Z]_\ell I_\ell.
$$

They contribute to the constant coefficient, giving

$$
\operatorname{TraceOpen}_P(Z)
=
\sum_{\ell=0}^{D-1}[Z]_\ell I_\ell.
\tag{9}
$$

Applying this definition to $Y$ and using Equation (7),

$$
\operatorname{TraceOpen}_P(Y)
=
\sum_\ell[Y]_\ell I_\ell
=
\widetilde f(r).
\tag{10}
$$

Therefore:

$$
\boxed{
\widetilde f(r)=v
\quad\Longleftrightarrow\quad
\operatorname{TraceOpen}_P(Y)=v.
}
\tag{11}
$$

`TraceOpen` is a coefficient pairing. It is not the univariate evaluation
$Y(\alpha)$ used to reduce ring-valued relations to the field.

## Eliminate the virtual ring $Y$

A direct protocol could expose $Y$ and check two statements:

$$
Y=\sum_bB_bE_b,
\tag{12}
$$

and

$$
\operatorname{TraceOpen}_P(Y)=v.
\tag{13}
$$

But Equation (12) already determines $Y$ from the folded rings $E_b$.
Sending $Y$ would introduce a redundant ring element and an extra interface
between the two checks.

Akita instead composes the two linear maps. The witness committed for this fold
contains digit decompositions of the position-folded rings:

$$
E_b(X)
=
\sum_hG_h\hat e_{b,h}(X),
\tag{14}
$$

where $G_h$ are public gadget weights. Consequently,

$$
Y(X)
=
\sum_{b,h}B_bG_h\hat e_{b,h}(X).
\tag{15}
$$

The ring $Y$ is now only a convenient name for this public linear
combination. It does not need to appear in the proof.

Apply `TraceOpen` directly to Equation (15):

$$
\begin{aligned}
v
&=
\operatorname{TraceOpen}_P(Y)\\
&=
\sum_{b,h}B_bG_h
\operatorname{TraceOpen}_P(\hat e_{b,h}).
\end{aligned}
\tag{16}
$$

Write each digit ring as

$$
\hat e_{b,h}(X)
=
\sum_{\ell=0}^{D-1}\hat e_{b,h,\ell}X^\ell
$$

and define the public inner trace weight

$$
J_\ell
:=
\operatorname{TraceOpen}_P(X^\ell).
\tag{17}
$$

By linearity,

$$
\operatorname{TraceOpen}_P(\hat e_{b,h})
=
\sum_\ell\hat e_{b,h,\ell}J_\ell.
$$

Equation (16) becomes the direct evaluation-consistency relation

$$
\boxed{
v
=
\sum_{b,h,\ell}
\hat e_{b,h,\ell}B_bG_hJ_\ell.
}
\tag{18}
$$

In the base-field setting, Equation (9) gives

$$
J_\ell
=
\operatorname{TraceOpen}_P(X^\ell)
=
I_\ell.
\tag{19}
$$

Thus every factor in Equation (18) has a simple role:

- $G_h$ recomposes the digit planes;
- $B_b$ evaluates across blocks; and
- $J_\ell=I_\ell$ evaluates inside the packed ring.

The two possible protocol views are:

```text
Expose Y:

committed ê  ──recompose──>  E_b  ──block fold──>  Y
                                                    │
                                                 TraceOpen
                                                    │
                                                    v

Eliminate Y:

committed ê  ───────composed public linear map──────>  v
```

## Write the relation as a sumcheck claim

The committed fold witness is stored as one flat table $w$. Flatten the
indices $(b,h,\ell)$ into a Boolean address $x$, and define the public
weight function

$$
T(x)
=
\begin{cases}
B_bG_hJ_\ell,
&\text{if }x\text{ addresses the coefficient }\hat e_{b,h,\ell},\\
0,
&\text{if }x\text{ lies outside the }\hat e\text{ segment.}
\end{cases}
\tag{20}
$$

Then Equation (18) is

$$
\boxed{
v
=
\sum_{x\in\{0,1\}^{\mu}}w(x)T(x).
}
\tag{21}
$$

This is the evaluation-correctness relation consumed by the later sumcheck
protocol. It is already a field-valued linear relation on the committed
witness. It therefore needs neither evaluation at $\alpha$ nor a ring-switch
quotient.

[Sumcheck stages](./sumcheck-stages.md#add-the-opening-trace) explains how this
claim is row-batched and fused with the other Stage-2 terms.

## Code reference

The base-field path follows the reduction above:

1. [`prepare_opening_point`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/proof/batch.rs#L687-L750)
   constructs $Q_p$, $B_b$, and $P$.
2. [`evaluate_claims_at_prepared_point`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/core/fold_kernels.rs#L61-L89)
   returns the position-folded rings $E_b$ and the virtual ring $Y$.
3. [`scalar_opening_from_folded_ring`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/core/fold_kernels.rs#L224-L274)
   computes $\operatorname{TraceOpen}_P(Y)$.
4. [`build_evaluation_trace_weights`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/evaluation_trace.rs#L101-L168)
   constructs $T(x)$ on the committed $\hat e$ segment.
5. [`accumulate_fused_relation_trace`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs#L281-L300)
   hands the relation to the fused Stage-2 sumcheck.

The main values are:

| Code value | Mathematical object |
|---|---|
| `PreparedOpeningPoint::ring_opening_point.position_weights` | $Q_p$ |
| `PreparedOpeningPoint::ring_opening_point.live_block_weights` | $B_b$ |
| `PreparedOpeningPoint::packed_inner_point` | $P$ |
| `OpeningFoldOutput::folded` | $E_0,E_1,\ldots$ |
| `OpeningFoldOutput::eval` | $Y$ |
| `trace_eval_target` | $\operatorname{TraceOpen}_P(Y)$ |
| `EvaluationTraceWeights` | $T$ |

## Base-field polynomial at an extension-field point

> **Status:** stub.
