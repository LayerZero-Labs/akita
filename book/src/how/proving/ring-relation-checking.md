# Checking ring relations over a field

A fold has already produced the [physical ring
equations](./akita-fold-realizations.md) that connect its witness to the
commitment payloads. This chapter explains how Akita turns those equations
into field claims for sumcheck.

The difficulty is ring multiplication. In $F[X]/(X^D+1)$, a term $X^{D+j}$
becomes $-X^j$. Ordinary polynomial evaluation does not apply that rule.
Akita has two ways to include its effect in the field check:

| Method | Main idea | Effect on the witness |
|---|---|---|
| Quotient lifting | Record the reduction in a private quotient polynomial | Add its digits to the next witness |
| Quotient-free checking, called `ReducedEvaluation` in code | Include the reduction in public weights on the existing coefficients | Add no quotient digits |

Both methods check the same ring equations. Raw or compressed payloads
determine which physical equations exist. The ring-relation mode determines
how to check them. We first explain both methods for one equation, then apply
them to the complete fold.

## Contents

- [Why ordinary evaluation is not enough](#why-ordinary-evaluation-is-not-enough)
- [Quotient lifting](#quotient-lifting)
- [Quotient-free checking by reduced evaluation](#quotient-free-checking-by-reduced-evaluation)
- [From one equation to the full relation](#from-one-equation-to-the-full-relation)
- [What enters the next witness](#what-enters-the-next-witness)
- [Costs and current schedule choices](#costs-and-current-schedule-choices)
- [Code reference](#code-reference)

## Why ordinary evaluation is not enough

Consider one relation

$$
AW=Y
\qquad\text{in }R_D=F[X]/(X^D+1),
\tag{1}
$$

where $A$ is a public multiplier, $W$ is private, and $Y$ is a public target.
Use the same letters for their canonical polynomial representatives. Each
representative has degree less than $D$.

Write $\operatorname{red}_D(P)$ for the remainder of a polynomial $P$ after
division by $X^D+1$. Equation (1) says

$$
\operatorname{red}_D(AW)=Y.
$$

For example, let $D=4$, $A=X^3$, and $W=X$. Their ordinary product is $X^4$.
Their ring product is $-1$, because $X^4=-1$ in $R_4$. Thus $Y=-1$ is the
correct target. At a general field point $\alpha$, however,

$$
A(\alpha)W(\alpha)=\alpha^4\ne -1=Y(\alpha).
$$

The missing step is the reduction. To **account for reduction modulo
$X^D+1$** means to include each replacement $X^{D+j}\mapsto -X^j$ in the
field check. This replacement is also called a signed wrap. Multiplication
with this rule is called negacyclic multiplication.

The evaluation point $\alpha$ lies in the protocol's challenge field $E$,
which contains $F$. It need not be a root of $X^D+1$.

## Quotient lifting

Quotient lifting keeps ordinary polynomial multiplication and adds the missing
reduction explicitly. Equation (1) holds exactly when a quotient $r(X)$
satisfies

$$
\boxed{
A(X)W(X)-Y(X)=(X^D+1)r(X)
\quad\text{in }F[X].
}
\tag{2}
$$

The canonical representatives determine this quotient uniquely. It has degree
less than $D$. In the example, $r=1$, because

$$
X^4-(-1)=X^4+1.
$$

The quotient is private because it depends on $W$. The prover decomposes it
into bounded digit polynomials using public gadget weights $G_g^{(r)}$:

$$
r(X)=\sum_{g=0}^{L_r-1}G_g^{(r)}\hat r_g(X).
\tag{3}
$$

Here $L_r$ is the quotient digit depth. These digits join the next witness.
After that witness is bound, the transcript samples $\alpha$.
The field check is

$$
A(\alpha)W(\alpha)
-(\alpha^D+1)\sum_gG_g^{(r)}\hat r_g(\alpha)
=Y(\alpha).
\tag{4}
$$

The modulus factor $\alpha^D+1$ multiplies the quotient contribution. This
check does not divide by that factor.

To see the public weights, write $W(X)=\sum_{j=0}^{D-1}w_jX^j$. Then

$$
A(\alpha)W(\alpha)
=\sum_{j=0}^{D-1}w_j\,A(\alpha)\alpha^j.
\tag{5}
$$

The weight factors into one multiplier evaluation and one coefficient power.
The quotient digits have weights
$-(\alpha^D+1)G_g^{(r)}\alpha^j$. If $W$ is also stored through digits,
substitute its public gadget recomposition as well. The entire check is
linear in the committed digits.

Fixing the witness before $\alpha$ is essential. For a false ring equation,
no allowed quotient can make Equation (2) a polynomial identity. Once the
witness and quotient are fixed, its nonzero polynomial residual has only a
bounded number of roots. Random evaluation detects that residual except when
$\alpha$ is one of those roots. This is the algebraic check; commitment
binding and the later sumcheck must still bind it to the same witness.

## Quotient-free checking by reduced evaluation

Reduced evaluation applies the signed wrap before evaluating at $\alpha$.
It uses the fact that multiplication by a public $A$ is linear in the private
coefficients $w_j$:

$$
\operatorname{red}_D(AW)
=\sum_{j=0}^{D-1}w_j\operatorname{red}_D(AX^j).
$$

Define the public weight for coefficient $j$ by

$$
\kappa_{A,\alpha}^{(D)}(j)
:=\operatorname{red}_D(AX^j)(\alpha).
\tag{6}
$$

This list of weights is called the **residue kernel**. The superscript records
the native ring dimension. Evaluating the reduced equation now gives

$$
\boxed{
\sum_{j=0}^{D-1}w_j\kappa_{A,\alpha}^{(D)}(j)=Y(\alpha).
}
\tag{7}
$$

Every weight is public. There is no private quotient to commit, range-check,
or carry into later folds.

In the same example, $A=X^3$ and the only nonzero coefficient of $W=X$ is
$w_1=1$. Its weight is

$$
\kappa_{X^3,\alpha}^{(4)}(1)
=\operatorname{red}_4(X^4)(\alpha)=-1.
$$

Equation (7) therefore checks $-1=Y(\alpha)$ directly.

### Why this is transpose-convolution

Let $C_A$ be the public matrix that multiplies a coefficient vector by $A$
with signed wraparound. Let $\mathbf w_W$ be the coefficient vector of $W$
and let $\mathbf p_\alpha=(1,\alpha,\ldots,\alpha^{D-1})^\top$.
Multiplication followed by evaluation is

$$
\mathbf p_\alpha^\top C_A\mathbf w_W
=\bigl(C_A^\top\mathbf p_\alpha\bigr)^\top\mathbf w_W.
\tag{8}
$$

The vector $C_A^\top\mathbf p_\alpha$ is exactly the residue kernel.
This explains the name **transpose-convolution**: transpose the public
multiplication map onto the evaluation weights. The implementation computes
these weights without building $C_A$.

The reduction correction has not disappeared. For each public basis product,
write

$$
AX^j=\operatorname{red}_D(AX^j)+(X^D+1)r_{A,j}(X).
$$

The polynomial $r_{A,j}$ is public because $A$ and $j$ are public. Hence

$$
\kappa_{A,\alpha}^{(D)}(j)
=A(\alpha)\alpha^j-(\alpha^D+1)r_{A,j}(\alpha).
$$

The same correction that quotient lifting stores through private digits is
already part of the public coefficient weights in Equation (7).

### Compute the weights in linear time

Write $A(X)=\sum_{k=0}^{D-1}a_kX^k$. Expanding Equation (6) gives

$$
\kappa_{A,\alpha}^{(D)}(j)
=\sum_{k=0}^{D-1}a_k
\begin{cases}
\alpha^{k+j},&k+j<D,\\
-\alpha^{k+j-D},&k+j\ge D.
\end{cases}
\tag{9}
$$

This formula shows every signed wrap, but computing it separately for all
$j$ takes quadratic work. A recurrence avoids that cost.

The first weight is $\kappa_{A,\alpha}^{(D)}(0)=A(\alpha)$. To get the next
reduced basis product, multiply the current one by $X$. Its top coefficient
$a_{D-1-j}$ crosses the modulus boundary. Ordinary evaluation would give
$a_{D-1-j}\alpha^D$ for that term. The ring requires $-a_{D-1-j}$ instead.
Subtracting their difference gives

$$
\boxed{
\kappa_{A,\alpha}^{(D)}(j+1)
=\alpha\kappa_{A,\alpha}^{(D)}(j)
-(\alpha^D+1)a_{D-1-j},
\qquad 0\le j<D-1.
}
\tag{10}
$$

For $A=X^3$ in $R_4$, the first step is
$\alpha\alpha^3-(\alpha^4+1)=-1$, as required.
All $D$ weights for one multiplier take $O(D)$ field operations. The
recurrence uses no division, so it also works when $\alpha^D+1=0$.

Here soundness concerns the **reduced residual**
$\operatorname{red}_D(AW)-Y$. A false ring equation gives a nonzero
polynomial of degree less than $D$. The committed witness fixes this residual
before $\alpha$. Random evaluation checks that polynomial, rather than the
unreduced product with its quotient term removed.

## From one equation to the full relation

Both methods extend by linearity to a row with several public multipliers.
For one native dimension $d_i$, write its ordinary product sum as
$P_i(X)=\sum_c A_{i,c}(X)W_{i,c}(X)$ and its target as $y_i(X)$.
The index $c$ selects a term in row $i$. Each $W_{i,c}$ uses the coefficient
slice required by that row's layout. The ring claim is
$\operatorname{red}_{d_i}(P_i)=y_i$.

Quotient lifting checks

$$
P_i(\alpha)-(\alpha^{d_i}+1)r_i(\alpha)=y_i(\alpha).
\tag{11}
$$

Reduced evaluation instead checks

$$
\sum_c\sum_{j=0}^{d_i-1}
[W_{i,c}]_j\,\kappa_{A_{i,c},\alpha}^{(d_i)}(j)=y_i(\alpha),
$$

where $[W_{i,c}]_j$ is coefficient $j$ of the row's witness term.
It uses $\kappa^{(d_i)}$, not a kernel for a larger storage ring.
In either method, the resulting row is a scalar claim over $E$.
The public multipliers, targets, layout, and witness are fixed before
$\alpha$; the coefficient weights are then derived from that challenge.

For `EvaluationTrace`, the consistency and $\mathbf A$ rows use the native A
dimension. The $\mathbf B$ and $\mathbf D$ rows use their own dimensions.
Compressed payloads also add $\mathbf F/\mathbf H$ rows in the native ring of
each compression map. Fixed coefficient-recomposition terms keep the source
positions defined by the payload relation. They do not change into ring
embeddings between these dimensions.

`SubringCoefficientPacking` has a separate consistency geometry. Its logical
relation is in $C=E[U]/(U^s+1)$ and uses $k=[E:F]$ coordinate planes of
length $s$. In the current implementation it uses quotient lifting. Its
modulus factor is $\alpha^s+1$, even though the stored planes have total
width $ks$. The [packing derivation](./root-fold-ring-switch.md#the-packing-consistency-quotient)
explains how its E/Q terms and folded-source term jointly realize that one
claim.

### Batch the rows and hand them to Stage 2

Express each evaluated row in the stored digits. Let $w(x)$ be the complete
flat digit-witness table.
Its address $x$ ranges over a padded Boolean domain $\{0,1\}^{\mu}$.
Each evaluated row has public coefficient weights $K_i(x)$ such that

$$
\sum_xw(x)K_i(x)=y_i(\alpha).
$$

In quotient-lift mode, $w$ includes the quotient digits and $K_i$ includes
their negative modulus factors. In reduced mode, $K_i$ includes the signed
wraps and $w$ has no quotient coordinates.

The transcript samples the row point $\tau_1$. Define the row weights
$\beta_i=\operatorname{eq}(\tau_1,i)$ and the combined public function

$$
K(x)=\sum_i\beta_iK_i(x),
\qquad
y_\tau=\sum_i\beta_i y_i(\alpha).
$$

Both methods therefore supply the same form of field claim:

$$
\boxed{
\sum_xw(x)K(x)=y_\tau.
}
\tag{12}
$$

This combines all physical rows. The [Stage 2
derivation](./sumcheck-stages.md#stage-2-fused-relation-sumcheck) separates their
ordinary and compression contributions for efficient evaluation. It also
adds the range-image binding and the scalar opening claim. Sumcheck reduces
the combined claim to an opening of the same witness at its final point
$r_2$. The next fold authenticates that opening.

The three random objects have different jobs: $\alpha$ evaluates the ring
variable, $\tau_1$ batches rows, and $r_2$ addresses the flat witness.
Range checking remains necessary in both modes. The ring relation checks
algebraic consistency; the range check certifies small witness digits.

The [EvaluationTrace scalar row](./field-ring-reduction.md#express-the-direct-relation-as-a-sumcheck-claim)
is already a field-valued claim. It needs no ring-switch quotient in either
mode. This is distinct from using residue kernels to check the physical ring
rows without quotients. In particular, $\mathbf D\hat{\mathbf e}=\mathbf v_D$
is still a physical ring relation.

## What enters the next witness

The two choices now combine without changing their roles. Payload mode
determines the commitment rows and compression digits. Relation mode
determines whether those rows also need quotient digits.

| Payload | Quotient lifting | Reduced evaluation |
|---|---|---|
| raw | $\hat z,\hat e,\hat t$ and ordinary quotient digits | $\hat z,\hat e,\hat t$ |
| compressed | $\hat z,\hat e,\hat t$, compression digits, ordinary quotients, and compression quotients | $\hat z,\hat e,\hat t$ and compression digits |

In the basic one-group `EvaluationTrace` layout, write
$\hat{\mathbf r}_{\mathrm{ord}}$ for the quotient digits of the consistency,
$\mathbf A$, $\mathbf B$, and $\mathbf D$ rows, in that order. Quotient
lifting gives the raw witness

$$
\mathbf w_{\mathrm{raw}}
=\hat{\mathbf z}\Vert\hat{\mathbf e}\Vert\hat{\mathbf t}
\Vert\hat{\mathbf r}_{\mathrm{ord}}.
\tag{13}
$$

For compressed payloads, retain the notation $\boldsymbol\xi_{F,\ell}$ and
$\boldsymbol\xi_{H,\ell}$ for the two chains' compression digits at layer
$\ell$. Write $\hat{\mathbf r}_{F,\ell}$ and
$\hat{\mathbf r}_{H,\ell}$ for that layer's quotient digits. The physical
order, with zero-alignment ranges suppressed, is

$$
\begin{aligned}
\mathbf w_{\mathrm{comp}}={}&
\hat{\mathbf z}\Vert\hat{\mathbf e}\Vert\hat{\mathbf t}
\Vert\hat{\mathbf r}_{\mathrm{ord}}\\
&\Vert\boldsymbol\xi_{F,1}\Vert\boldsymbol\xi_{H,1}
\Vert\hat{\mathbf r}_{F,1}\Vert\hat{\mathbf r}_{H,1}\\
&\Vert\boldsymbol\xi_{F,2}\Vert\boldsymbol\xi_{H,2}
\Vert\hat{\mathbf r}_{F,2}\Vert\hat{\mathbf r}_{H,2}.
\end{aligned}
\tag{14}
$$

Reduced evaluation removes every $\hat r$ segment from these layouts.
It keeps the compression digits and their restricted $\{-1,0\}$ check when
the payload is compressed. `WitnessLayout` derives the live ranges and any
required alignment for the selected mode. Missing quotients have no placeholder
coordinates. They contribute no range-check work or successor commitment
input. Raw mode has no compression-alignment ranges.

For coefficient packing, the shared ordinary quotient segment includes the
consistency-row slot for the digit-decomposed coordinate planes of
$Q_{\mathrm{pack}}$. It has no separate method-specific quotient span.
The [witness-order page](./opening-points-layout.md#witness-order) gives the
group, chunk, and coefficient order used by both sides.

## Costs and current schedule choices

Quotient lifting gives the simple weights in Equation (5). The prover can
keep a factored relation representation. Its cost includes computing the
quotients and carrying their digits through commitment, range checking,
Stage 2, and later folds.

Reduced evaluation removes that witness material. Its public weights are
more involved. The recurrence makes one multiplier's kernel linear in its
native dimension, but this is not a bound on the whole fold. The current
prover builds one temporary dense Stage-2 relation-weight table. Sumcheck
folds that table; the proof neither commits nor serializes it.

The verifier uses a different calculation. At the final sumcheck point it
prepares compact weights for each required native coefficient window. It then
uses the existing combined A/B/D setup scan. It does not build the prover's
dense table. [Matrix evaluation at a
point](../verifying/matrix_evaluation.md#reduced-evaluation-at-the-final-point)
derives this calculation. Compression F/H maps are evaluated separately.

This explains the tradeoff across recursion. Large early folds benefit from
factored weights. In smaller later folds, quotient digits can form a larger
share of the next witness. Removing them can reduce later work. These facts
do not give a universal runtime winner or a fixed transition level.

Setup offloading is a separate choice. It replaces a direct public-setup scan
with Stage 3 and a setup-prefix opening carried to the next fold. Quotient
lifting supplies the factored weights used by the current offloading path.
The [offloading chapter](../setup-offloading.md) explains that handoff.

The current schedule permits `ReducedEvaluation` only at absolute level 2 or
later. That fold and every later nonterminal fold must use `EvaluationTrace`,
consume no incoming setup prefix, and perform setup evaluation directly.
Once the schedule switches, it cannot return to quotient lifting. The suffix
can use raw or compressed payloads, subject to the separate payload rules.
Coefficient-packing folds remain quotient-lift-only.

These are implementation restrictions. They do not show that other algebraic
combinations are impossible. The schedule binds the mode before the
next-witness binding and $\alpha$; the proof carries no extra mode tag or challenge.
The terminal fold has no ring-relation mode. It checks its clear response
directly instead of producing another committed witness.

On an ordinary recursive edge, the outgoing binding is the next witness's
commitment payload. On the [last edge](../recursion.md#the-last-edge-and-terminal),
it consists of the successor's inner commitment images. Both are fixed before
the predecessor samples $\alpha$. The last edge does not add a duplicate
outer commitment payload.

## Code reference

- [`ring_relation_mode.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/ring_relation_mode.rs)
  defines both modes and the allowed transition. `FoldSchedule::validate_structure`
  enforces the complete schedule restrictions.
- [`ring_switch/coeffs.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/coeffs.rs)
  constructs the mode-selected witness. Reduced evaluation branches before
  quotient construction. `WitnessLayout` owns the actual ranges.
- [`ring_switch/finalize.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/finalize.rs)
  samples $\alpha$ after the caller binds the outgoing witness. It prepares
  factored or dense relation weights for Stage 2.
- [`ring/residue.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-algebra/src/ring/residue.rs)
  implements the recurrence. Its tests compare the result with explicit signed
  multiplication and ring products, including points where $\alpha^D+1=0$.
- [`ring_switch/relation_evaluation.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/ring_switch/relation_evaluation.rs)
  selects the verifier calculation from the authenticated mode. Reduced mode
  uses the complete coefficient functional and rejects deferred setup.

The [prover ring-switch tests](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/ring_switch/tests.rs)
check that raw and compressed reduced witnesses avoid quotient construction.
The [schedule tests](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/schedule_tests/relation_mode.rs)
check the transition, opening-method, and setup-prefix restrictions.
