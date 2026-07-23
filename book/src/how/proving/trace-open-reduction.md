# Multilinear evaluation reduction

This page considers the base case in which both the multilinear polynomial and
its opening point are defined over the base field \(F\):

$$
f:\{0,1\}^n\rightarrow F,
\qquad
r\in F^n.
$$

Akita commits data through the cyclotomic ring

$$
R=F[X]/(X^D+1).
$$

The protocol must therefore reduce the field claim

$$
\widetilde f(r)=v
$$

to a claim about committed ring elements. The reduction has three steps:

1. fold two outer parts of the evaluation point to obtain a ring \(Y\);
2. encode the remaining inner part of the point as a ring \(P\); and
3. recover the scalar evaluation with
   \(\operatorname{TraceOpen}_P(Y)\).

In this base-field setting, `TraceOpen` is a constant-coefficient pairing.

## The evaluation problem

The multilinear extension of \(f\) is

$$
\widetilde f(r)
=
\sum_{x\in\{0,1\}^n}
\operatorname{eq}(r,x)f(x).
\tag{1}
$$

Choose a ring dimension \(D=2^d\) and a power-of-two number \(M\) of positions
per block. Re-index the table as

$$
f[\ell,p,b],
$$

where:

- \(\ell\in[D]\) is the inner index that becomes a ring coefficient;
- \(p\in[M]\) is a position inside a block; and
- \(b\) is a block index.

Missing entries in a partial final block are public zeros.

Split the evaluation point in the same order:

$$
r=(r_{\mathrm{in}},r_{\mathrm{pos}},r_{\mathrm{blk}}).
$$

Let

$$
q_\ell=\operatorname{eq}(r_{\mathrm{in}},\ell),
\qquad
a_p=\operatorname{eq}(r_{\mathrm{pos}},p),
\qquad
c_b=\operatorname{eq}(r_{\mathrm{blk}},b).
$$

Then Equation (1) becomes

$$
\widetilde f(r)
=
\sum_{\ell,p,b}q_\ell a_pc_bf[\ell,p,b].
\tag{2}
$$

Akita evaluates the three axes in the order

$$
\text{position}\longrightarrow\text{block}\longrightarrow\text{inner}.
$$

## From multilinear evaluation to two rings

### Pack the inner axis

For each position \(p\) and block \(b\), pack the inner slice into

$$
F_{b,p}(X)
=
\sum_{\ell=0}^{D-1}f[\ell,p,b]X^\ell
\in R.
\tag{3}
$$

This is a change of representation: \(f[\ell,p,b]\) becomes the coefficient
of \(X^\ell\).

### First outer fold

Apply the position weights inside each block:

$$
E_b(X)
=
\sum_p a_pF_{b,p}(X)
=
\langle a,F_b\rangle.
\tag{4}
$$

Its coefficients are

$$
[E_b]_\ell=\sum_p a_pf[\ell,p,b].
$$

### Second outer fold

Apply the block weights:

$$
Y(X)
=
\sum_b c_bE_b(X)
=
\langle c,E\rangle.
\tag{5}
$$

Now

$$
[Y]_\ell
=
\sum_{p,b}a_pc_bf[\ell,p,b].
\tag{6}
$$

Thus \(Y\) is derived from \(f\) after evaluating every coordinate except
\(r_{\mathrm{in}}\).

### Encode the inner fold

Pack the remaining inner weights into a second ring:

$$
P(X)
=
\sum_{\ell=0}^{D-1}q_\ell X^\ell.
\tag{7}
$$

The two rings have different sources:

| Ring | Derived from | Meaning |
|---|---|---|
| \(Y\) | \(f\), \(r_{\mathrm{pos}}\), and \(r_{\mathrm{blk}}\) | the polynomial after the two outer folds |
| \(P\) | \(r_{\mathrm{in}}\) | the weights for the remaining inner fold |

## Base-field `TraceOpen`

Let \(\sigma_{-1}\) be the ring automorphism

$$
\sigma_{-1}(X)=X^{-1}.
$$

Akita defines

$$
\boxed{
\operatorname{TraceOpen}_P(Y)
=
\left[Y(X)\sigma_{-1}(P(X))\right]_0,
}
\tag{8}
$$

where \([\cdot]_0\) means the constant coefficient in
\(F[X]/(X^D+1)\).

The term \([Y]_\ell X^\ell\) pairs with
\(q_\ell X^{-\ell}\), so

$$
\operatorname{TraceOpen}_P(Y)
=
\sum_{\ell=0}^{D-1}[Y]_\ell q_\ell.
\tag{9}
$$

Substituting Equation (6) gives

$$
\begin{aligned}
\operatorname{TraceOpen}_P(Y)
&=
\sum_\ell q_\ell
\left(\sum_{p,b}a_pc_bf[\ell,p,b]\right)\\
&=
\sum_{\ell,p,b}q_\ell a_pc_bf[\ell,p,b]\\
&=
\widetilde f(r).
\end{aligned}
\tag{10}
$$

Therefore

$$
\boxed{
\widetilde f(r)=v
\quad\Longleftrightarrow\quad
\operatorname{TraceOpen}_P(Y)=v.
}
\tag{11}
$$

This is not the univariate evaluation \(Y(\alpha)\) used during ring
switching. `TraceOpen` is a coefficient pairing determined by the multilinear
opening point.

## Fuse the trace claim into Stage 2

The committed Stage-2 witness does not contain \(Y\) directly. It contains
digit decompositions of the first-fold rings:

$$
E_b(X)=\sum_hG_h\widehat e_{b,h}(X),
\tag{12}
$$

where \(G_h\) are public gadget weights. By Equation (5) and linearity of
`TraceOpen`,

$$
\begin{aligned}
v_{\mathrm{tr}}
&=
\operatorname{TraceOpen}_P(Y)\\
&=
\sum_{b,h,\ell}
\widehat e_{b,h,\ell}c_bG_hq_\ell.
\end{aligned}
\tag{13}
$$

After flattening the digit witness into Boolean addresses \(x\), the public
factors in Equation (13) define a transparent weight function \(T(x)\):

$$
v_{\mathrm{tr}}=\sum_xw(x)T(x).
\tag{14}
$$

This trace equation is **not** added to the ring-relation matrix. The physical
matrix and its relation-weight factorization contain only the consistency,
\(A\), \(B\), and \(D\) rows.

The protocol reserves the next index \(i_{\mathrm{tr}}\) in the padded
\(\tau_1\) domain and derives the scalar

$$
\beta_{\mathrm{tr}}
=
\operatorname{eq}(\tau_1,i_{\mathrm{tr}}).
\tag{15}
$$

Only this scalar comes from the relation-row domain. Akita builds \(T\)
separately and directly fuses

$$
\beta_{\mathrm{tr}}v_{\mathrm{tr}}
=
\sum_x\beta_{\mathrm{tr}}w(x)T(x)
\tag{16}
$$

into the Stage-2 sumcheck over the same flat witness address used by the
relation and range-image terms. The reserved index lets the trace reuse the
existing row-batching randomness without becoming a physical or evaluated
matrix row.

See [Sumcheck stages](./sumcheck-stages.md#add-the-opening-trace) for the full
fused identity.

## Code reference

The base-field execution path follows the reduction above:

1. [`prepare_opening_point`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/proof/batch.rs#L687-L750)
   splits \(r\), constructs the outer weights, and packs \(P\).
2. [`evaluate_claims_at_prepared_point`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/core/fold_kernels.rs#L61-L89)
   returns both the first-fold rings \(E_b\) and the final ring \(Y\).
3. [`scalar_opening_from_folded_ring`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/core/fold_kernels.rs#L224-L274)
   computes the constant coefficient of \(Y\sigma_{-1}(P)\) for \(E=F\).
4. [`build_evaluation_trace_weights`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/evaluation_trace.rs#L101-L168)
   constructs \(T\) independently of the ring-relation weights.
5. [`accumulate_fused_relation_trace`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs#L281-L300)
   adds the trace weight directly to the relation weight during the shared
   Stage-2 witness scan.
6. [`PreparedEvaluationTrace::evaluate_at_point`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-verifier/src/protocol/evaluation_trace.rs#L45-L104)
   evaluates \(T\) at the final Stage-2 point.

The layout API makes the matrix separation explicit:
[`relation_matrix_row_count`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/layout/params.rs#L1126-L1150)
counts the physical rows, while
[`evaluation_trace_row_index`](https://github.com/LayerZero-Labs/akita/blob/b104dae6c672f406b676b04c47e00f4249669ba5/crates/akita-types/src/layout/params.rs#L1152-L1170)
returns the next index used only to derive \(\beta_{\mathrm{tr}}\).

The main code values are:

| Code value | Mathematical object |
|---|---|
| `OpeningFoldOutput::folded` | \(E_0,E_1,\ldots\) |
| `OpeningFoldOutput::eval` / `folded_ring` | \(Y\) |
| `PreparedOpeningPoint::packed_inner_point` | \(P\) |
| `trace_eval_target` | \(\operatorname{TraceOpen}_P(Y)\) |

## Base-field polynomial at an extension-field point

> **Status:** stub.
