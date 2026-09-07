# Extension-opening reduction

An opening claim gives a point $r$ and a value $v$, and asks the verifier to
check that the committed polynomial evaluates to $v$ at $r$. The verifier
knows the commitment, not the full polynomial table.

Extension-opening reduction, or EOR, handles a base-field table opened at an
extension-field point. It packs several base-field entries into each
extension-field entry, then reduces the original opening claim to a claim
about that packed table. The number of table entries falls, but the underlying
base-field data stays the same.

The main idea is to compare two ways of reading a two-dimensional table.
One path evaluates its columns first. The other packs its rows first.
A small table of partial evaluations connects the two paths. Sumcheck checks
that connection, and Akita's subsequent opening checks bind its result to the
committed witness.

This page starts with one polynomial and one opening. The
[implementation page](../how/proving/extension-opening-reduction.md) explains
how Akita combines several openings and selects EOR in a recursive schedule.

## A small example

Work over $\mathbb F=\mathbb F_5$, so base-field arithmetic is modulo $5$.
Use the quadratic extension $\mathbb E=\mathbb F_5[\beta]/(\beta^2-2)$ with
basis $1,\beta$. Every extension-field value has two base-field coordinates.
These small fields are only for the example, not a secure parameter choice.

Consider a two-variable table $f(y,w)$. The head bit $y$ selects a column and
the tail bit $w$ selects a row:

| Tail $w$ | Head $y=0$ | Head $y=1$ | Pack across the row |
|---|---:|---:|---|
| $0$ | $1$ | $2$ | $g(0)=1+2\beta$ |
| $1$ | $4$ | $3$ | $g(1)=4+3\beta$ |

Packing stores each pair as the two coordinates of one extension-field value.
It produces a table $g$ with one variable instead of two.

Now open the original table at $r=(2,\beta)$. Multilinear evaluation uses
weights $1-t,t$ for a bit opened at $t$. Evaluating down the two columns at
the tail point $\beta$ gives the column partials

$$
\begin{aligned}
S_0&=(1-\beta)\cdot1+\beta\cdot4=1+3\beta,\\
S_1&=(1-\beta)\cdot2+\beta\cdot3=2+\beta.
\end{aligned}
$$

The head point $2$ then gives the original opening value

$$
v=(1-2)S_0+2S_1=3+4\beta.
$$

Packing uses the fixed weights $1,\beta$, whereas the head evaluation uses
$1-2,2$. These are different operations. Indeed, evaluating the packed table
at the same tail point gives

$$
\widetilde g(\beta)
=(1-\beta)(1+2\beta)+\beta(4+3\beta)=3,
$$

which is not $v$. The tilde denotes the multilinear polynomial defined by a
Boolean table. EOR must prove a relation between these two representations;
it cannot simply replace $f$ by $g$ and keep the opening value.

## The general input and packed table

Let $\mathbb E$ have degree $k=2^\kappa>1$ over $\mathbb F$. Fix the canonical
$\mathbb F$-basis $(\beta_y)_{y\in\{0,1\}^\kappa}$ used by the field
implementation. The basis is public and does not depend on a challenge.

The prover holds an $n$-variable base-field table $f$, where $n\geq\kappa$.
Split its indices and opening point into a head of length $\kappa$ and a tail
of length $m=n-\kappa$:

$$
f:\{0,1\}^\kappa\times\{0,1\}^m\rightarrow\mathbb F,
\qquad
r=(r_{\mathrm{head}},r_{\mathrm{tail}})\in\mathbb E^n.
$$

The incoming claim is $\widetilde f(r)=v$. Both parties know $r$, $v$, and the
commitment before EOR begins. Knowing $v$ does not mean that the claim is
already verified.

For equally sized tuples $a,b$, define the equality polynomial

$$
\operatorname{eq}(a,b)
=\prod_i\bigl((1-a_i)(1-b_i)+a_i b_i\bigr).
$$

Its values at Boolean indices are the interpolation weights. Thus

$$
\widetilde f(r)
=\sum_{y\in\{0,1\}^\kappa}
 \sum_{w\in\{0,1\}^m}
 \operatorname{eq}(r_{\mathrm{head}},y)
 \operatorname{eq}(r_{\mathrm{tail}},w)f(y,w).
\tag{1}
$$

The prover packs the head entries at each Boolean tail position:

$$
g(w)=\sum_y f(y,w)\beta_y,
\qquad
g:\{0,1\}^m\rightarrow\mathbb E.
\tag{2}
$$

The table $g$ has $2^m$ extension-field entries in place of $2^n$ base-field
entries. It is a lossless representation of $f$, not a separately chosen
witness. Akita uses little-endian Boolean order, so the flat entry at
$kw+y$ is $f(y,w)$ when the bit strings are read as integers.
The prover does not send the full table $f$ or $g$ in the EOR proof.

## Send column partials

The prover first computes one partial for each head index:

$$
S_y
=\widetilde f(y,r_{\mathrm{tail}})
=\sum_w\operatorname{eq}(r_{\mathrm{tail}},w)f(y,w)
\in\mathbb E.
\tag{3}
$$

It sends these $k$ extension-field values. The verifier checks their count and
tests whether they recover the incoming opening:

$$
v\stackrel{?}{=}
\sum_y\operatorname{eq}(r_{\mathrm{head}},y)S_y.
\tag{4}
$$

The verifier can perform this check without knowing $f$. But it only checks
consistency with $v$. A dishonest prover could choose unrelated partials that
pass Equation (4). The rest of EOR must connect these particular partials to
the packed witness.

## Derive row partials

Both parties decompose each received column partial in the fixed basis:

$$
S_y=\sum_u S_{u,y}\beta_u,
\qquad S_{u,y}\in\mathbb F.
\tag{5}
$$

The index $u\in\{0,1\}^\kappa$ selects a basis coordinate; $y$ still selects a
head column. They have the same range but different roles.
The coefficients $S_{u,y}$ form a $k$ by $k$ table.

For the example, $S_0=1+3\beta$ and $S_1=2+\beta$ give

| Basis coordinate $u$ | Column $y=0$ | Column $y=1$ | Repack this row |
|---|---:|---:|---|
| $0$ | $1$ | $2$ | $\operatorname{row}_0=1+2\beta$ |
| $1$ | $3$ | $1$ | $\operatorname{row}_1=3+\beta$ |

In general, repacking the $u$-th row means computing

$$
\operatorname{row}_u=\sum_y S_{u,y}\beta_y\in\mathbb E.
\tag{6}
$$

This basis transpose has no new prover message. The verifier derives every
$\operatorname{row}_u$ from the received $S_y$. A row partial is one
extension-field value, not a polynomial. Its value is now known, but its
claimed relation to the witness remains to be checked.

## Build the public factors

To express the row claims in terms of $g$, decompose each tail equality weight
in the same basis:

$$
\operatorname{eq}(r_{\mathrm{tail}},w)
=\sum_u A_u(w)\beta_u,
\qquad
A_u(w)=\operatorname{coord}_u
\bigl(\operatorname{eq}(r_{\mathrm{tail}},w)\bigr)\in\mathbb F.
\tag{7}
$$

Here $w$ is Boolean and $\operatorname{coord}_u$ extracts coordinate $u$ in
the fixed basis. Every $A_u$ is a public Boolean table. The verifier can
compute its entries from $r_{\mathrm{tail}}$ alone, without any witness data.
This is why these factors are called transparent.

Substitute Equation (7) into Equation (3). Since $f(y,w)\in\mathbb F$,
multiplication by $f(y,w)$ scales each basis coordinate without mixing them:

$$
S_{u,y}=\sum_w A_u(w)f(y,w).
\tag{8}
$$

Now repack across $y$:

$$
\begin{aligned}
\operatorname{row}_u
&=\sum_y\left(\sum_w A_u(w)f(y,w)\right)\beta_y\\
&=\sum_w A_u(w)\left(\sum_y f(y,w)\beta_y\right)\\
&=\sum_w A_u(w)g(w).
\end{aligned}
$$

The relation to be proved is therefore

$$
\boxed{\operatorname{row}_u=\sum_w A_u(w)g(w)}
\qquad\text{for every }u.
\tag{9}
$$

The verifier derives the left side from the received partials. On the right,
the factor $A_u$ is public and the table $g$ is the prover's packed witness.
For honest partials, the identity follows by exchanging two sums. In the
protocol, it is an obligation to check, not an assumption the verifier makes.

The example makes the exchange visible. The tail weights are $1-\beta$ and
$\beta$, so their coordinate tables are

| Coordinate | $w=0$ | $w=1$ |
|---|---:|---:|
| $A_0(w)$ | $1$ | $0$ |
| $A_1(w)$ | $-1$ | $1$ |

Consequently,

$$
\operatorname{row}_0=g(0)=1+2\beta,
\qquad
\operatorname{row}_1=-g(0)+g(1)=3+\beta.
$$

Taking a coordinate after evaluating the columns gives the same result as
using that coordinate of the weights on the packed rows.

## Batch the row claims

Rather than prove all $k$ row relations separately, the protocol combines them
with fresh random weights. After the original claim values and all column
partials enter the transcript, it derives a point
$\eta\in\mathbb E^\kappa$. Both parties compute

$$
\lambda_u=\operatorname{eq}(\eta,u),
\qquad
c_\eta=\sum_u\lambda_u\operatorname{row}_u,
\qquad
A_\eta(w)=\sum_u\lambda_u A_u(w).
\tag{10}
$$

Neither $c_\eta$ nor $A_\eta$ is sent. The verifier derives $c_\eta$ from the
received partials and derives $A_\eta$ from the public tail point and $\eta$.
Equation (9) now gives the single sumcheck claim

$$
c_\eta=\sum_w A_\eta(w)g(w).
\tag{11}
$$

For the degree-two example, $\eta$ has one coordinate. The row weights are
$1-\eta,\eta$, giving

$$
A_\eta(0)=1-2\eta,
\qquad A_\eta(1)=\eta,
\qquad
c_\eta=(1-\eta)\operatorname{row}_0+\eta\operatorname{row}_1.
$$

The challenge must follow the partials. Otherwise a prover could choose row
errors to cancel under already known weights. With the partials fixed first,
row errors define a multilinear polynomial in $\eta$; a false collection of
rows can still pass at an unlucky challenge, but cannot choose its errors
after seeing that challenge.

### From a public table to a polynomial

Sumcheck evaluates away from the Boolean cube, so it needs polynomials, not
just table entries. Write $\widetilde A_u$, $\widetilde A_\eta$, and
$\widetilde g$ for the multilinear extensions of their respective tables.
For example,

$$
\widetilde A_\eta(X)
=\sum_{w\in\{0,1\}^m}\operatorname{eq}(X,w)A_\eta(w)
=\sum_u\lambda_u\widetilde A_u(X).
\tag{12}
$$

The polynomial $\widetilde A_u$ has coefficients in $\mathbb F$.
The polynomials $\widetilde A_\eta$ and $\widetilde g$ have coefficients in
$\mathbb E$. Each has degree at most one in every tail variable.
For the example, Equation (12) is just

$$
\widetilde A_\eta(X)=(1-X)(1-2\eta)+X\eta.
$$

At an extension-field point $\rho$, evaluate the multilinear extension of the
coordinate table. Do not replace $\widetilde A_u(\rho)$ by
$\operatorname{coord}_u(\operatorname{eq}(r_{\mathrm{tail}},\rho))$.
Coordinate extraction is only $\mathbb F$-linear, so it cannot in general
move past the extension-field interpolation weights.
The verifier computes $\widetilde A_\eta(\rho)$ without materializing the
full Boolean table.

## Reduce the sum to one point

Define

$$
P(X)=\widetilde A_\eta(X)\widetilde g(X).
$$

Each factor is multilinear, so $P$ has degree at most two in each variable.
Sumcheck reduces Equation (11) over the $m$ tail variables.
It starts with the known target $C_0=c_\eta$.

In round $j$, after challenges $\rho_0,\ldots,\rho_{j-1}$, the honest prover
forms the univariate polynomial

$$
q_j(T)=
\sum_{w_{j+1},\ldots,w_{m-1}\in\{0,1\}}
P(\rho_0,\ldots,\rho_{j-1},T,w_{j+1},\ldots,w_{m-1}).
\tag{13}
$$

The prover sends the round polynomial in the protocol's compressed encoding.
The verifier reconstructs a polynomial of degree at most two and checks
$q_j(0)+q_j(1)=C_j$. After absorbing that message, the transcript derives
$\rho_j\in\mathbb E$, and both parties set $C_{j+1}=q_j(\rho_j)$.
In Akita, challenges come from the shared transcript rather than a separate
verifier message; the schedule also specifies
[transcript grinding](../how/transcript.md).

After $m$ rounds, the challenge point is $\rho=(\rho_0,\ldots,\rho_{m-1})$.
The prover sends a terminal product claim $h$. The verifier checks
$h=C_m$ and computes the public factor
$\theta=\widetilde A_\eta(\rho)$. The remaining obligation is

$$
h\stackrel{?}{=}\theta\widetilde g(\rho).
\tag{14}
$$

The sumcheck messages alone do not verify this final witness evaluation.
They reduce the original sum claim to Equation (14).
If $m=0$, there are no round messages and $\rho$ is empty; the same final
product relation still applies.

## Bind the result to the committed witness

Akita keeps $h$ and $\theta$ separate. It does not send an independent
$\widetilde g(\rho)$ in the EOR payload, and it does not divide $h$ by
$\theta$. The next opening checks must prove Equation (14) for the packed
representation of the same incoming committed table.
The product relation does not recover an unscaled value when $\theta=0$.

For a nonterminal fold, `EvaluationTrace` supplies a public linear relation
on the fold's opening digits. Akita multiplies its opening weights by
$\theta$ and uses $h$ as the target. The fold's source-consistency and
commitment relations connect those digits back to the incoming witness.
[Stage 2](../how/proving/sumcheck-stages.md#add-the-opening-claim-consistency)
checks the resulting relations together.
The terminal path checks the corresponding trace relation directly against
its terminal response and has no Stage-2 sumcheck.

Passing the column check, deriving the rows, and replaying sumcheck are not
independent proofs of the original opening. They are successive reductions
whose final witness relation must also be checked.

## What the prover sends

The following table lists the algebraic data for one opening. The input value
$v$ may itself be a claim carried from an earlier protocol stage; it is not a
new field in the EOR payload.

| Object | Meaning | How the verifier obtains it |
|---|---|---|
| $r,v$ and the commitment | Incoming opening claim | Already available at the EOR boundary |
| $f,g$ | Original and packed witness tables | The prover does not send these full tables in EOR |
| $S_y$ | Claimed column partials | Sent by the prover |
| $S_{u,y},\operatorname{row}_u$ | Coordinates and repacked rows | Derived from the received $S_y$ |
| $A_u$ | Public coordinate tables | Derived from $r_{\mathrm{tail}}$ and the fixed basis |
| $\eta,\rho$ | Row and sumcheck challenges | Derived from the transcript in order |
| $A_\eta,\theta$ | Batched public factor and its final evaluation | Derived from $r_{\mathrm{tail}},\eta,\rho$ |
| $c_\eta$ | Sumcheck input target | Derived from the rows and $\eta$ |
| $q_j$ | Degree-two round polynomials | Sent in compressed form |
| $h$ | Claimed final product | Sent by the prover and checked against the sumcheck output |

In particular, a derived value need not be a verified witness value.
The verifier knows $\operatorname{row}_u$ after reading the partials, but
Equation (9) is still a claim about the hidden table $g$.

The message flow is:

```text
Prover                                         Verifier
  holds f; derives packed table g                knows r, v, commitment
       |
       +---------------- S_y ------------------> checks recombination to v
                                                 derives row_u from S_y

       both derive eta after absorbing the partials
       both compute c_eta and the public factor A_eta

       +------------ round polynomial ---------> checks the round relation
       both derive the next rho coordinate       repeat for each tail bit

       +----------------- h -------------------> checks h = final round claim
                                                 computes theta

       remaining claim: h = theta * g_tilde(rho)
       Akita's opening and commitment checks bind it to the witness
```

## Where Akita uses this reduction

A scheduled fold uses EOR exactly when its opening method is
`EvaluationTrace` and its claim field is a proper extension of its base
field. A single-field configuration needs no EOR. A
`SubringCoefficientPacking` fold binds the original extension-valued opening
directly and also omits EOR. See
[Fold path and field geometry](../how/proving/fold-path.md).

With multiple claims, Akita runs one shared EOR sumcheck and sends one
terminal product $h_i$ for each opening. It uses an early set of coefficients
inside EOR and a separate set after the fold's opening payload is fixed.
Groups retain their own public points and native tail lengths.
The [multi-group description](../how/proving/extension-opening-reduction.md#multi-group-openings)
explains those two batching steps and the common sumcheck domain.

## Code reference

- [Shared tensor algebra](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/extension_opening_reduction.rs)
  defines the head split, basis packing, column-to-row transpose, and public
  factor evaluation. In particular, `tensor_equality_factor_eval_at_point`
  evaluates Equation (12), not coordinates of an already evaluated equality.
- [Prover orchestration](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/src/protocol/core/extension_opening_reduction.rs)
  absorbs claims and partials before sampling $\eta$, runs sumcheck, and
  records the terminal products. The
  [proof type](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/proof/levels.rs)
  carries `partials`, `sumcheck`, and `final_claims`.
- [Verifier replay](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/core/fold/extension_claim.rs)
  checks the required payload and shape, recombines partials to $v$, and
  reconstructs the same row claims and public factors.
  [Fold claim preparation](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-verifier/src/protocol/core/fold/mod.rs)
  applies the factors to the later evaluation-trace weights.

The [tensor and sumcheck tests](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-prover/tests/extension_opening_reduction.rs)
compare partial recombination with direct evaluation, the row claim with the
dense witness-factor sum, and the verifier's factor evaluation with the
multilinear extension of the factor table. The verifier replay tests also
reject changed partials, missing terminal claims, and EOR payloads that do not
match the scheduled method. These tests check the implementation's algebra
and rejection paths; they do not replace a soundness analysis of the composed
protocol.
