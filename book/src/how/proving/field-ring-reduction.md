# Field-to-ring evaluation reduction

This page explains the first correctness obligation of an Akita opening:
given the partial evaluations produced by the scheduled opening method, how do
public weights recombine them into the scalar claim carried by the protocol?

Start with the base-field specialization

$$
f:\{0,1\}^n\rightarrow F,
\qquad
r\in F^n,
\qquad
\widetilde f(r)=v.
$$

Here both the polynomial table and the opening point are defined over the base
field $F$. Akita commits the table through the cyclotomic ring

$$
R=F[X]/(X^D+1).
$$

The schedule chooses one of two opening methods. `EvaluationTrace` keeps one
full element $E_b(X)\in R$ per live block and finishes the inner contraction
with `TraceOpen`. `SubringCoefficientPacking` contracts part of the inner axis
earlier and keeps a shorter polynomial $e_b(U)$ over the opening field. In
both cases the result is a field-valued linear relation on the committed
opening digits:

| Opening method | Partial carried into the fold | Scalar target bound here |
|---|---|---|
| `EvaluationTrace` | full-ring $E_b(X)$ | the trace target; it equals $v$ in the base-field case |
| `SubringCoefficientPacking` | shorter $e_b(U)$ | the original extension-valued $v$ |

For a proper extension-field point, `EvaluationTrace` first uses
[extension-opening reduction](./extension-opening-reduction.md), or EOR, and
the derivation below applies to the reduced trace target. Coefficient packing
instead binds the original extension-valued opening directly. The
[fold-path overview](./fold-path.md) states where the schedule uses each
method.

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
in [Semantic relations in an Akita fold](./akita-fold.md).

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


## Subring coefficient packing: shorter partials

The evaluation-trace partial $E_b(X)$ retains all $D$ inner coefficients.
Subring coefficient packing shortens this partial by contracting one part of
the inner axis before the opening digits are formed.

Return now to the general case in which the opening point lies in a field $E$
containing $F$; the interpolation weights $I$, $Q$, and $B$ are then
$E$-valued. The case $E=F$ is included. Let

$$
k=[E:F].
$$

Choose a retained-axis dimension $s$—later also the challenge-subring
dimension—and write

$$
D=k\eta s.
\tag{22}
$$

Here $\eta$ is the packing factor. The implementation and specification call
this value $h$; this page uses $\eta$ so that it cannot be confused with the
opening-digit index already used in the evaluation-trace derivation. Every
inner coefficient index has a unique form

$$
\ell=u+k\eta j,
\qquad
u\in[k\eta],
\qquad
j\in[s].
\tag{23}
$$

Split the inner opening point in the same order:

$$
r_{\mathrm{in}}
=
(r_{\mathrm{pack}},r_{\mathrm{tail}}).
$$

The tensor-product interpolation weight therefore factors as

$$
I_{u+k\eta j}
=
I_u^{\mathrm{pack}}I_j^{\mathrm{tail}}.
\tag{24}
$$

The index $u$ is contracted now; $j$ remains explicit. It helps to see this
as a two-dimensional coefficient table rather than one flat list. For a fixed
block $b$, first apply the position weights and write

$$
\phi_{b,j,u}
:=
\sum_p Q_p f[u+k\eta j,p,b]
\in E.
$$

The rows are indexed by the coefficient $j$ that will remain in the packed
partial. The columns are indexed by $u$, which is consumed now:

| retained row | $u=0$ | $u=1$ | $\cdots$ | $u=k\eta-1$ | after contracting the row |
|---|---|---|---|---|---|
| $j=0$ | $\phi_{b,0,0}$ | $\phi_{b,0,1}$ | $\cdots$ | $\phi_{b,0,k\eta-1}$ | $e_{b,0}$ |
| $j=1$ | $\phi_{b,1,0}$ | $\phi_{b,1,1}$ | $\cdots$ | $\phi_{b,1,k\eta-1}$ | $e_{b,1}$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | $\vdots$ |
| $j=s-1$ | $\phi_{b,s-1,0}$ | $\phi_{b,s-1,1}$ | $\cdots$ | $\phi_{b,s-1,k\eta-1}$ | $e_{b,s-1}$ |

The table is not a new witness. It is only a view of the position-folded
source coefficients. Contracting one row gives

$$
e_{b,j}
=
\sum_u I_u^{\mathrm{pack}}\phi_{b,j,u}
=
\sum_{p,u}
Q_p I_u^{\mathrm{pack}}
f[u+k\eta j,p,b]
\in E.
\tag{25}
$$

For example, take $D=8$, $k=2$, $\eta=2$, and $s=2$. Then $k\eta=4$, so the
flat inner coefficients become two rows:

~~~text
        j = 0:  ell = 0  1  2  3  -> e[b,0]
        j = 1:  ell = 4  5  6  7  -> e[b,1]
~~~

Each row is contracted to one element of the degree-two field $E$. The packed
partial therefore contains two elements of $E$, or four base-field
coordinates, instead of all eight source coefficients.

Substituting Equations (24) and (25) into the original multilinear evaluation
gives

$$
\begin{aligned}
v
&=
\sum_{u,j,p,b}
I_u^{\mathrm{pack}}I_j^{\mathrm{tail}}
Q_pB_bf[u+k\eta j,p,b]\\
&=
\sum_{b,j}B_bI_j^{\mathrm{tail}}e_{b,j}.
\end{aligned}
\tag{26}
$$

Thus the early contraction loses no part of the claimed evaluation. It only
splits the old inner contraction into two stages: $u$ is consumed while the
partial is formed, and $j$ is consumed in the final scalar relation.

Package the retained coefficients as

$$
e_b(U)
=
\sum_{j=0}^{s-1}e_{b,j}U^j
\in
C:=E[U]/(U^s+1).
\tag{27}
$$

The ring $C$ has $s$ coefficients in $E$. Fix the canonical $F$-basis
$\beta_0,\ldots,\beta_{k-1}$ of $E$ and write

$$
e_{b,j}
=
\sum_{t=0}^{k-1}\beta_t e_{b,t,j},
\qquad
e_{b,t,j}\in F.
\tag{28}
$$

Consequently one packed partial has $ks=D/\eta$ base-field coordinates,
rather than the $D$ coordinates of $E_b(X)$. The basis elements $\beta_t$ are
fixed by the canonical extension-field representation; they are not transcript
challenges or schedule choices.

The next witness stores balanced digits of these base-field coordinates. Use
$d$ for the opening-digit index:

$$
e_{b,t,j}
=
\sum_dG_d^{\mathrm{open}}\hat e_{b,d,t,j}.
\tag{29}
$$

Substituting Equations (28) and (29) into Equation (26) gives the direct packed
opening relation

$$
\boxed{
v
=
\sum_bB_b
\sum_jI_j^{\mathrm{tail}}
\sum_t\beta_t
\sum_dG_d^{\mathrm{open}}\hat e_{b,d,t,j}.
}
\tag{30}
$$

Read Equation (30) from the inside out. The opening gadget weights rebuild
each base-field coordinate $e_{b,t,j}$ from its digits. The basis elements
$\beta_t$ rebuild the extension-field coefficient $e_{b,j}$. The tail weights
consume the retained $j$ rows, and the block weights combine the live blocks.
The result is the original scalar opening $v$.

This is an $E$-valued virtual row on the committed digit witness. It is not a
ring-matrix row and has no cyclotomic quotient. Claim batching and Stage-2 row
batching multiply the displayed weights by additional public scalars, without
changing the single-claim identity.

At this scalar layer, the two methods have the same high-level output: a public
linear map from the method-selected opening digits to the scalar target. They
use different digit geometries and different public maps.

## What scalar correctness does not prove

Equations (21) and (30) answer the scalar question under one assumption: the
partials $E_b$ or $e_b$ are the correct partial evaluations of the incoming
committed polynomial. The equations do not establish that assumption. A prover
could otherwise choose unrelated partials that satisfy the scalar equation.

The next chapter supplies the missing source-consistency statement. Rather
than check every block independently, Akita samples random fold challenges and
batches the blockwise equations into one relation on the folded response
$\mathbf z$. The two method-dependent shapes are

$$
\text{EvaluationTrace:}
\qquad
\sum_b c_bE_b
=
\operatorname{Eval}(\mathbf z),
$$

and

$$
\text{SubringCoefficientPacking:}
\qquad
\sum_b c_be_b
=
L(\mathbf z).
$$

[Semantic relations in an Akita fold](./akita-fold.md#the-folded-response-and-its-digitization)
derives both relations. In particular, it explains why the packing challenges
must act only on the retained $j$ axis. Quotients and evaluation at the
ring-switch challenge come later, after this semantic relation is established.
[Sumcheck stages](./sumcheck-stages.md#add-the-opening-claim-consistency)
explains how the method-selected scalar relation is row-batched and fused with
the other Stage-2 terms.

## Opening-method code reference

### Evaluation trace

This subsection applies when the schedule selects `EvaluationTrace`. A fold
that selects `SubringCoefficientPacking` uses the coefficient-packing flow in
the following subsection instead.

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
   [`accumulate_fused_relation_linear`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/sumcheck/relation_range_image/mod.rs)
   adds the prepared linear relation to the fused Stage-2 sumcheck.

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
accumulate_fused_relation_linear
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

### Subring coefficient packing

The implementation follows the same derivation in three stages:

1. **Prepare the point.**
   [`PreparedSubringCoefficientPackingPoint`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/subring_coefficient_packing.rs)
   checks $D=k\eta s$ and prepares the packing, tail, position, and block
   weights.
2. **Pack and bind the partials.**
   The prover contracts the $p$ and $u$ axes to obtain the coordinates
   $e_{b,t,j}$ in `[block][extension coordinate][subring coefficient]` order.
   It recombines those coordinates as in Equation (30), then binds their digit
   decompositions before sampling the fold challenges.
3. **Fold and verify.**
   [`fold_coefficient_packing_group`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/coefficient_packing.rs)
   folds the packed partials and produces $Q_{\mathrm{pack}}$. Stage 2 then
   checks both the packing relation and the direct scalar opening, using
   $\beta_t I_j^{\mathrm{tail}}$ to rebuild each extension-valued coefficient.

The reference tests in
[`subring_coefficient_packing_reference_tests.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/subring_coefficient_packing_reference_tests.rs)
compare the direct partial and scalar formulas with the flat factorization.
The Stage-2 tests in
[`coefficient_packing_relation_tests.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/coefficient_packing_relation_tests.rs)
compare the expanded prover terms with the verifier's compact evaluation and
check that every extension-coordinate plane is bound.

## Base-field polynomial at an extension-field point

The scheduled opening method determines this case.

With `EvaluationTrace`, Akita does not treat a base-field multilinear
polynomial as an ordinary extension-valued opening of the same arity. It first
runs the [extension-opening reduction](./extension-opening-reduction.md). The
low $\log_2[\mathbb E:\mathbb F]$ variables are packed into extension-field
coefficients, and one degree-two tensor sumcheck reduces the original claim to
a claim on that packed polynomial with fewer variables. The reduced claim then
enters the evaluation trace. The trace represents its extension weights by
canonical subfield coordinates, and the verifier rejects invalid shapes,
coordinate counts, and noncanonical images before using them. Configurations
with $[\mathbb E:\mathbb F]=1$ skip this reduction.

With `SubringCoefficientPacking`, Akita keeps one coefficient axis in the
challenge subring, contracts the other axes over the extension field, and
binds the original scalar opening directly in Stage 2 without EOR. The
derivation is in [Subring coefficient packing: shorter
partials](#subring-coefficient-packing-shorter-partials); [Fold path and field
geometry](./fold-path.md) states the schedule boundary.
