# Field-to-ring evaluation reduction

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

## Reduce to a ring-valued evaluation

### Pack the inner axis into ring coefficients

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

Equivalently, the values $F_{p,b}$ form a ring-valued multilinear table

$$
f_R:\{0,1\}^{n-d}\rightarrow R,
\qquad
f_R[p,b]:=F_{p,b}.
$$

This is the same underlying table under a lossless coefficient packing, not a
new witness. The ring polynomial has $d=\log_2D$ fewer variables and is opened
at

$$
r_R=(r_{\mathrm{pos}},r_{\mathrm{blk}}),
$$

whose base-field coordinates act as constant elements of $R$.

### Evaluate the ring polynomial

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

Thus Equation (4) is the ring-based evaluation claim

$$
\boxed{
\widetilde f_R(r_R)=Y.
}
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

## Recover the evaluation with `TraceOpen`

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

## Eliminate the intermediate ring evaluation $Y$

### Hachi: expose $Y$
The baseline Hachi protocol exposes $Y$ and checks two statements:

$$
Y=\sum_bB_bE_b,
\tag{12}
$$

and

$$
\operatorname{TraceOpen}_P(Y)=v.
\tag{13}
$$

The first statement proves that $Y$ is the correct evaluation of the
ring-valued polynomial. In this fold, each $E_b(X)$ is digit-decomposed into
the committed partial-evaluation witness:

$$
E_b(X)
=
\sum_hG_h\hat e_{b,h}(X),
\tag{14}
$$

where $\hat e_{b,h}$ are the digit rings and $G_h$ are public gadget weights.
Substituting this decomposition into Equation (12) gives

$$
\boxed{
Y(X)
=
\sum_{b,h}B_bG_h\hat e_{b,h}(X).
}
\tag{15}
$$

Equation (15) is a relation over the ring that enforces consistency between
the ring element $Y$ and the witness polynomials $\hat e_{b,h}(X)$. Hachi
sends $Y$ to the verifier, which checks Equation (13) directly. The prover
then proves Equation (15) using the same ring-relation machinery as the other
constraints that bind the previous witness to the next witness, as described
in Section 2.5.2.

### Akita: compose the two checks

Akita's crucial observation is that $Y$ is already determined linearly by the
committed partial-evaluation witness and that
$\operatorname{TraceOpen}_P$ is itself a linear map. Sending $Y$ would
therefore introduce a redundant ring element, an extra interface between the
two checks, and additional verifier work. Akita instead composes the two
linear maps and applies `TraceOpen` directly to Equation (15):

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

This row acts on the committed partial-evaluation digits $\hat e$. The other
fold relations bind those digits back to the original committed polynomial.

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

## Express the direct relation as a sumcheck claim

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


[Sumcheck stages](./sumcheck-stages.md#add-the-opening-claim-consistency)
explains how this claim is row-batched and fused with the other Stage-2 terms.

## Code reference

The base-field path follows the reduction above:

1. **Prepare the opening weights.**
   [`prepare_opening_point`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/batch.rs)
   constructs $Q_p$, $B_b$, and $P$.
2. **Evaluate the ring polynomial.**
   [`evaluate_claims_at_prepared_point`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/core/fold_kernels.rs)
   returns the position-folded rings $E_b$ and the temporary ring $Y$.
3. **Recover the scalar evaluation.**
   [`scalar_opening_from_folded_ring`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/core/fold_kernels.rs)
   computes $\operatorname{TraceOpen}_P(Y)$.
4. **Prepare the trace factors.**
   [`prepare_evaluation_trace_group_parameters`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/trace_weight/evaluation_trace.rs)
   prepares the block point underlying $B_b$, the gadget weights $G_h$, and
   the inner trace weights $J_\ell$.
5. **Construct the trace weights.**
   [`build_evaluation_trace_weights`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/sumcheck/relation_range_image/evaluation_trace.rs)
   combines those factors with the claim coefficients and physical $\hat e$
   locations to construct $T(x)$.
6. **Fuse the Stage-2 relation.**
   [`accumulate_fused_relation_trace`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs)
   adds the trace relation to the fused Stage-2 sumcheck.

The main data flow is:

```text
opening point r
      |
      v
PreparedOpeningPoint { Q_p, B_b, P }
      |
      v
OpeningFoldOutput
|-- folded: [E_b] -- digit decomposition --> e_hat in witness w
`-- eval: Y ------- TraceOpen_P ----------> v_tr
                                                  |
                                                  v
PreparedFold
|-- evaluation_trace_claim: v_tr
|-- evaluation_trace_points: prepared opening points
|-- evaluation_trace_claim_coefficients: c_q
`-- witness: contains E_b and e_hat
      |
      v
prepare_evaluation_trace_group_parameters
      |
      `-- public factors B_b, G_h, J_l
                         |
                         v
build_evaluation_trace_weights
      |
      `-- T(x) on the committed e_hat segment
                         |
                         v
accumulate_fused_relation_trace
      |
      `-- Stage 2 proves v_tr = sum_x w(x) T(x)
```

The main values are:

| Code value | Mathematical object |
|---|---|
| `PreparedOpeningPoint::ring_opening_point.position_weights` | $Q_p$ |
| `PreparedOpeningPoint::ring_opening_point.live_block_weights` | $B_b$ |
| `PreparedOpeningPoint::packed_inner_point` | $P(X)$ |
| `OpeningFoldOutput::folded` | $E_0,E_1,\ldots$ |
| `OpeningFoldOutput::eval` | temporary $Y(X)$ |
| `PreparedEvaluationTraceClaim::claimed_evaluation` | $v_{\mathrm{tr}}=\operatorname{TraceOpen}_P(Y)$ |
| `PreparedEvaluationTraceClaim::claim_coefficients` | claim-batching coefficients $c_q$ |
| `RingRelationGroupWitness::e_folded` | position-folded rings $E_b$ |
| `RingRelationGroupWitness::e_hat` | digit rings $\hat e_{b,h}(X)$ |
| `PreparedFold::evaluation_trace_claim` | $v_{\mathrm{tr}}$ carried into Stage 2 |
| `PreparedFold::evaluation_trace_points` | prepared $P$, $Q$, and $B$ for each group |
| `PreparedFold::evaluation_trace_claim_coefficients` | $c_q$ carried into trace-weight construction |
| `EvaluationTraceGroupParameters::block_opening_point` | block point from which $B_b$ is evaluated |
| `EvaluationTraceGroupParameters::opening_digit_weights` | $G_h$ |
| `EvaluationTraceGroupParameters::inner_trace` | $J_\ell$, equal to $I_\ell$ in the base-field case |
| `EvaluationTraceWeights` | $T(x)$ |

The temporary ring $Y$ is used only to compute $v_{\mathrm{tr}}$; it is not
stored in `PreparedFold` or sent to the verifier. Stage 2 instead proves

$$
v_{\mathrm{tr}}
=
\sum_x w(x)T(x)
$$

directly from the committed digit witness.

## Base-field polynomial at an extension-field point

> **Status:** stub.
