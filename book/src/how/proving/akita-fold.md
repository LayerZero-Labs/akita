# Semantic relations in an Akita fold

This page derives the semantic relations proved by one non-terminal Akita
fold. The central question is the second correctness obligation from the
previous page: **are the method-selected partial evaluations consistent with
the same incoming polynomial that is being folded?**

The presentation starts with one polynomial group and one opening claim. The
commitment relations use the ordinary source ring

$$
R=F[X]/(X^D+1).
$$

The opening-consistency relation has two method-dependent geometries:

- `EvaluationTrace` keeps a full ring partial and proves the relation in $R$.
- `SubringCoefficientPacking` keeps $s$ extension-field coefficients, proves
  the corresponding relation in the challenge subring, and embeds the same
  sparse challenge into $R$ for the source fold.

The other three semantic roles—inner-commitment, outer-commitment, and
opening-commitment consistency—are unchanged by that choice.

The current implementation also supports more elaborate physical layouts.
These include commitment groups, witness chunks, and different ordinary
$\mathbf A$, $\mathbf B$, and $\mathbf D$ ring dimensions. These extensions do
not change the four core relations developed below. This page establishes only the
basic case; advanced layouts are outside its scope. The [advanced relation
layouts](./advanced-relation-layouts.md) page develops the multi-group case.
The [raw and compressed realizations](./akita-fold-realizations.md) use the same
semantic relations;
the compression realization introduces its own smaller ring dimensions.

The four families below are the semantic source relations. The current
implementation realizes the $\mathbf B$ and $\mathbf D$ commitment relations
either by transmitting their semantic commitments as raw payloads or by
binding those commitments to smaller terminal payloads through compression
relations. This choice changes their physical realization, not the four
semantic constraints derived here.

The goal of the fold is to replace the current polynomial blocks by a smaller
digit witness while proving that the new witness is consistent with:

1. the opening data computed from the old polynomial;
2. the inner and outer commitments; and
3. the random fold of the old polynomial blocks.

These statements form the semantic core of the **physical ring relation**.
The scalar evaluation claim from [Field-to-ring evaluation
reduction](./field-ring-reduction.md) is a separate field-valued relation.
Each opening method supplies its own public linear map from opening digits to
that scalar. This page instead proves the source-consistency relation that
justifies those digits. The scalar row and physical rows are fused later.

## Contents

- [Inputs and objects derived in the fold](#inputs-and-objects-derived-in-the-fold)
  - [The committed polynomial and opening query](#the-committed-polynomial-and-opening-query)
  - [Balanced digit representations](#balanced-digit-representations)
  - [Polynomial blocks and commitment hint](#polynomial-blocks-and-commitment-hint)
  - [Partial evaluations and opening digits](#partial-evaluations-and-opening-digits)
  - [The folded response and its digitization](#the-folded-response-and-its-digitization)
- [The four semantic relation families](#the-four-semantic-relation-families)
  - [Fold-evaluation consistency](#1-fold-evaluation-consistency)
  - [Inner-commitment consistency](#2-inner-commitment-consistency)
  - [Outer-commitment consistency](#3-outer-commitment-consistency)
  - [Opening-commitment consistency](#4-opening-commitment-consistency)

## Inputs and objects derived in the fold

A fold starts from a public opening claim

$$
\widetilde f(r)=v
$$

for a polynomial whose commitment payload is already fixed. We call
$\mathbf u=\mathbf B\hat{\mathbf t}$ the semantic commitment behind that
payload. In raw mode, $\mathbf u$ itself is transmitted; in compressed mode,
the payload is the smaller terminal commitment $p_F$, which is bound to
$\mathbf u$ by the compression relations on the [realizations
page](./akita-fold-realizations.md#compressed-realization).

The prover and verifier have different views of these inputs. The prover holds
the polynomial blocks, their inner-digit representation, the commitment hint
generated when the commitment was formed, and the public commitment payload.
The verifier knows that payload and the opening claim, but receives neither the
blocks nor the hint.

From these inputs, the current fold derives two new representations of the
committed polynomial. The opening point determines partial evaluations $E_b$
inside each block, while fresh transcript challenges fold the old block digits
into a response $\mathbf z$. Both are digit-decomposed before entering the next
committed witness. The four relation families later prove that these derived
objects are consistent with the same hidden opening of $\mathbf u$.

### The committed polynomial and opening query

As in the previous page, split the ring-valued polynomial table into blocks.
Let $b$ index a live block and $p$ a position inside that block. Pack the inner
coefficient axis into the ring element

$$
F_{p,b}(X)\in R.
$$

The [field-to-ring evaluation
reduction](./field-ring-reduction.md#the-evaluation-problem) splits $r$ into
inner, position, and block coordinates and defines their interpolation weights
$I_\ell$, $Q_p$, and $B_b$. We reuse those definitions here rather than
deriving them again. The position weights $Q_p$ produce $E_b$ below, while the
block weights $B_b$ and inner weights $I_\ell$ belong to the field-valued
evaluation relation. The opening claim enters this page with target $v$. An
`EvaluationTrace` fold later writes its reduced target as $v_{\mathrm{tr}}$;
a `SubringCoefficientPacking` fold binds the original scalar target directly.
For the single base-field claim considered here, the valid method-selected
target equals $v$.

### Balanced digit representations

Both the existing commitment opening and the next committed witness must have
bounded coefficients. This shortness condition is what lets commitment binding
reduce to Module-SIS: two distinct bounded openings of the same linear
commitment would yield a short, nonzero kernel vector.

Akita obtains these bounded representations by decomposing ring coefficients
into balanced base-$g$ digits. Let $g$ be an even power of two, let $\delta$ be
the digit depth, and let
$\mathbf x=(x_0,\ldots,x_{n-1})\in R^n$. A balanced decomposition of
$\mathbf x$ consists of digit rings $\hat x_{i,h}(X)\in R$, indexed by
$0\le i<n$ and $0\le h<\delta$, such that

$$
x_i(X)
=
\sum_{h=0}^{\delta-1}g^h\hat x_{i,h}(X),
\qquad
[\hat x_{i,h}]_\ell
\in
\{-g/2,\ldots,g/2-1\}
$$

for every coefficient position $0\le \ell<D$. Stack the digit rings with $h$
innermost into $\hat{\mathbf x}\in R^{n\delta}$. Define the gadget row and its
block-diagonal recomposition matrix by

$$
\mathbf g_{g,\delta}
=
(1,g,\ldots,g^{\delta-1}),
\qquad
\mathbf G_{g,n}
=
\mathbf I_n\otimes\mathbf g_{g,\delta}
\in R^{n\times n\delta}.
$$

The coefficientwise identities then become the vector equation

$$
\boxed{
\mathbf x
=
\mathbf G_{g,n}\hat{\mathbf x}.
}
$$

The entries of $\mathbf G_{g,n}$ are public scalars embedded as constant ring
elements. Thus $\mathbf G_{g,n}$ is a deterministic **recomposition matrix**,
not a commitment matrix such as $\mathbf A$, $\mathbf B$, or $\mathbf D$.
Digit decomposition produces $\hat{\mathbf x}$; multiplication by
$\mathbf G_{g,n}$ reconstructs $\mathbf x$. The protocol's range check
certifies that the committed coefficients of $\hat{\mathbf x}$ lie in the
balanced digit range.

Akita chooses separate bases and depths for different witness roles. To keep
the derivation readable, write $G_a^{\mathrm{in}}$,
$G_h^{\mathrm{out}}$, $G_h^{\mathrm{open}}$, and
$G_f^{\mathrm{fold}}$ for the corresponding scalar gadget weights. The four
recomposition identities used on this page are

$$
\begin{aligned}
F_{p,b}(X)
&=
\sum_a G_a^{\mathrm{in}}s_{b,p,a}(X),
\\
t_{b,\rho}(X)
&=
\sum_h G_h^{\mathrm{out}}\hat t_{b,\rho,h}(X),
\\
E_b(X)
&=
\sum_h G_h^{\mathrm{open}}\hat e_{b,h}(X),
\\
z_{p,a}(X)
&=
\sum_f G_f^{\mathrm{fold}}\hat z_{p,a,f}(X).
\end{aligned}
$$

The first two identities describe commitment-side data already fixed by the
incoming commitment payload: $\mathbf s_b$ is the incoming inner-digit
representation, and $\hat{\mathbf t}$ is reconstructed from the incoming hint.
The latter two digit families, $\hat{\mathbf e}$ and $\hat{\mathbf z}$, are
newly derived from the opening point and fold challenges. We now place each
identity in its protocol context.

### Polynomial blocks and commitment hint

The commitment-side inputs are fixed before the polynomial is queried. For
each block, the prover has the inner digit rings $s_{b,p,a}$ and can therefore
recompose the polynomial rings as

$$
F_{p,b}(X)
=
\sum_a G_a^{\mathrm{in}}s_{b,p,a}(X).
\tag{1}
$$

For one block, collect the digit rings $s_{b,p,a}$ into a vector
$\mathbf{s}_b$. The inner commitment matrix $\mathbf A$ maps this block vector
to an inner image

$$
\mathbf t_b
=
\mathbf A\mathbf s_b.
\tag{2}
$$

Each coordinate of $\mathbf t_b$ is itself represented by balanced outer
digits:

$$
t_{b,\rho}(X)
=
\sum_h G_h^{\mathrm{out}}\hat t_{b,\rho,h}(X),
\tag{3}
$$

where $\rho$ selects a row of $\mathbf A$. Stack these digits over all blocks
to obtain $\hat{\mathbf t}$. The outer commitment matrix $\mathbf B$ then gives
the semantic commitment

$$
\mathbf u
=
\mathbf B\hat{\mathbf t}.
\tag{4}
$$

The two matrices have distinct roles: $\mathbf A$ forms one inner image per
block, whereas $\mathbf B$ commits the digit-decomposed inner images across all
blocks. The prover's commitment hint stores the recomposed inner images
$\mathbf t_b$; this fold decomposes them to recover $\hat{\mathbf t}$. At a
recursive level, the polynomial blocks, hint, and public payload were produced
by the preceding level. At the root, they come from the original commitment.
The semantic commitment $\mathbf u$ fixes this commitment-side data.
Equations (1)--(4) describe its hidden opening, which the later relation rows
bind to $\mathbf u$.

### Partial evaluations and opening digits

The first new witness object derived in this fold comes from the opening point.
Use its position weights $Q_p$ to evaluate the position coordinate inside each
block. In the base-field setting of the previous page, $Q_p\in F$ acts as a
constant in $R$:

$$
E_b(X)
=
\sum_p Q_pF_{p,b}(X).
\tag{5}
$$

Digit-decompose each $E_b$ with the opening gadget weights
$G_h^{\mathrm{open}}$:

$$
E_b(X)
=
\sum_h G_h^{\mathrm{open}}\hat e_{b,h}(X).
\tag{6}
$$

The fold witness contains the digit rings $\hat e$, not a second copy of the
recomposed $E_b$. Their semantic opening commitment is

$$
\mathbf v_D
=
\mathbf D\hat{\mathbf e}.
\tag{7}
$$

Here $\mathbf D$ commits to the digit-decomposed partial evaluations.

Equations (5) to (7) show the `EvaluationTrace` shape, where each partial is a
full ring element. With `SubringCoefficientPacking`, the previous page instead
splits the source coefficient index as

$$
\ell=u+k\eta j,
\qquad
u\in[k\eta],
\qquad
j\in[s],
$$

contracts $u$, and retains $j$. The resulting partial is

$$
e_b(U)=\sum_{j=0}^{s-1}e_{b,j}U^j
\in C:=E[U]/(U^s+1).
$$

Writing each $E$-coefficient in the canonical basis
$\beta_0,\ldots,\beta_{k-1}$ gives $ks=D/\eta$ base-field coordinates:

$$
e_{b,j}=\sum_{t=0}^{k-1}\beta_t e_{b,t,j},
\qquad
e_{b,t,j}=\sum_dG_d^{\mathrm{open}}\hat e_{b,d,t,j}.
$$

Thus the two methods create different opening-digit layouts, but in both cases
$\mathbf D\hat{\mathbf e}=\mathbf v_D$ binds the newly derived digits. The
[scalar derivation](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials)
already showed how the packed coordinates recombine into the original opening
claim. The next subsection proves that they came from the folded source.

The subscript in $\mathbf v_D$ distinguishes this ring-vector commitment from
the scalar opening target $v$ and its method-selected field relation. Equation
(7) binds the newly derived opening digits; it does not by itself prove the
scalar evaluation claim.

### The folded response and its digitization

The semantic opening commitment $\mathbf v_D$ commits the prover to
$\hat{\mathbf e}$. It does not by itself show that the corresponding partial
evaluations were computed from the incoming polynomial blocks. That missing
link is method-dependent.

#### Evaluation-trace consistency

For `EvaluationTrace`, correctness in every block $b$ requires

$$
\boxed{
E_b
=
\sum_p Q_pF_{p,b}
=
\sum_{p,a}Q_pG_a^{\mathrm{in}}s_{b,p,a}.
}
\tag{8}
$$

Equation (8) connects the partial evaluation derived in this fold to the
incoming block witness $\mathbf s_b$. The semantic commitment $\mathbf u$
creates a second consistency requirement. Equation (4),
$\mathbf u=\mathbf B\hat{\mathbf t}$, binds the outer digits
$\hat{\mathbf t}$, which recompose the inner images $\mathbf t_b$ through
Equation (3). However, this commitment relation does not by itself show that
the inner images were computed from the incoming block witness. The missing
link is the blockwise relation $\mathbf t_b=\mathbf A\mathbf s_b$ from
Equation (2).

Checking both relations separately for every live block would retain the block
index in the next proof. Instead, after the public payloads binding
$\mathbf u$ and $\mathbf v_D$ have been fixed, the transcript samples one
sparse ring-valued challenge $c_b(X)$ for each live block. These challenges
are separate from the query weights $B_b$. The prover folds the incoming block
witnesses into one response:

$$
z_{p,a}(X)
=
\sum_b c_b(X)s_{b,p,a}(X).
\tag{9}
$$

The folded response $\mathbf z$ no longer carries a live-block index. Because
both blockwise relations are linear in $\mathbf s_b$, the same challenges
batch them into relations on this single response. For the partial
evaluations, Equations (8) and (9) give

$$
\begin{aligned}
\sum_b c_bE_b
&=
\sum_{b,p,a}c_bQ_pG_a^{\mathrm{in}}s_{b,p,a}\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}
\left(\sum_b c_bs_{b,p,a}\right)\\
&=
\sum_{p,a}Q_pG_a^{\mathrm{in}}z_{p,a}.
\end{aligned}
\tag{9a}
$$

For the inner commitments, Equations (2) and (9) give the vector relation

$$
\begin{aligned}
\sum_b c_b\mathbf t_b
&=
\mathbf A\left(\sum_b c_b\mathbf s_b\right)\\
&=
\mathbf A\mathbf z.
\end{aligned}
\tag{9b}
$$

Equation (9a) says that evaluating within each block and then folding gives the
same result as first folding the block witnesses into $\mathbf z$ and then
applying the evaluation weights. Equation (9b) similarly connects
$\mathbf z$ to the inner images bound through $\mathbf u$. If any blockwise
relation is incorrect, its error is unlikely to disappear in the corresponding
random combination. Thus the challenges remove the block index while
preserving the two links from the incoming witness: one to the partial
evaluations created in this fold, and one to the commitment that entered it.

#### Subring-coefficient-packing consistency

Recall the coefficient-table index split

$$
\ell=u+k\eta j,
\qquad
u\in[k\eta],
\qquad
j\in[s].
$$

Here $\ell$ (read "ell") is only the flat coefficient index in the A ring:
$u$ selects a column and $j$ selects a row. The packed partial has already
taken the same weighted sum across the $u$ columns of every row. It keeps the
$j$ rows, so the fold challenge must act on those rows without mixing the
already-contracted column axis. This is why it is sampled from

$$
S=F[U]/(U^s+1).
$$

We use two linear maps. The first symbol below is $\iota$ (Greek "iota",
not the index $\ell$). It embeds a row operation from $S$ into the source ring
$R$:

$$
\iota:S\hookrightarrow R,
\qquad
\iota(U)=X^{k\eta}.
$$

The second map, $L$, is the public packing map from one source-block vector to
$C=E[U]/(U^s+1)$. For any vector
$\mathbf y=(y_{p,a})_{p,a}$ with the same shape as $\mathbf s_b$, define

$$
L(\mathbf y)(U)
=
\sum_{j=0}^{s-1}
\left(
\sum_{p,a,u}
Q_pG_a^{\mathrm{in}}I_u^{\mathrm{pack}}
[y_{p,a}]_{u+k\eta j}
\right)U^j.
$$

Thus $L$ first recomposes the inner digits with $G_a^{\mathrm{in}}$, then
contracts the position axis $p$ and the column axis $u$, while leaving one
output coefficient for every row $j$. Together with the definition of $e_b$
above, Equation (1) gives $L(\mathbf s_b)=e_b$. The intuition is now simple:
$\iota$ moves rows, while $L$ compresses each row.
For example, take $D=8$, $k=2$, $\eta=2$, and $s=2$. The source table has two
rows of width four. In this picture, $a_u$ and $b_u$ already include the
position sum over $p$ and the inner-digit recomposition over $a$:

~~~text
       u = 0   1   2   3       after applying L
j = 0     a0  a1  a2  a3  ----------------------> e0
j = 1     b0  b1  b2  b3  ----------------------> e1
~~~

Both rows use the same packing weights, so

$$
e_0=\sum_{u=0}^{3}I_u^{\mathrm{pack}}a_u,
\qquad
e_1=\sum_{u=0}^{3}I_u^{\mathrm{pack}}b_u.
$$

Since $\iota(U)=X^4$, multiplying in $R$ moves the first row to the second and
wraps the second row back with a minus sign. Thus $(a,b)$ becomes $(-b,a)$.
Applying $L$ after this move gives $-e_1+e_0U$. Applying $L$ first gives
$e_0+e_1U$, and multiplying by $U$ in $C$ gives the same result because
$U^2=-1$:

$$
L(\iota(U)\mathbf s)
=
-e_1+e_0U
=
U(e_0+e_1U)
=
UL(\mathbf s).
$$

Because any $c(U)\in S$ is a linear combination of these row shifts, the same
commuting identity holds for every challenge:

$$
\boxed{
L\!\left(\iota(c)\mathbf s\right)=cL(\mathbf s).
}
\tag{9c}
$$

The transcript draws one $c_b\in S$ per live block and the prover forms

$$
\mathbf z
=
\sum_b\iota(c_b)\mathbf s_b
\tag{9d}
$$

Linearity and Equation (9c) then give the packed source-consistency identity

$$
\boxed{
L(\mathbf z)
=
\sum_b c_be_b
\quad\text{in }C.
}
\tag{9e}
$$

The same embedded challenges $\iota(c_b)$ are used in the inner-commitment
identity $\sum_b\iota(c_b)\mathbf t_b=\mathbf A\mathbf z$. There is no
second challenge draw for the A relation. This shared challenge is what makes
the packed partial and the source fold describe the same random combination
of blocks.

This compression has an arithmetic cost: combining the blocks increases
coefficient magnitudes. Let $\sigma_\infty$ bound the coefficient norm of
every digit block $\mathbf s_b$, and let
$\omega=\max_b\lVert c_b\rVert_1$, using $c_b$ itself for
`EvaluationTrace` and its sparse embedding $\iota(c_b)$ for packing.
Negacyclic multiplication gives

$$
\begin{aligned}
\lVert\mathbf z\rVert_{\infty,\mathrm{coef}}
&\le
\sum_b
\lVert c_b\mathbf s_b\rVert_{\infty,\mathrm{coef}}\\
&\le
|\mathcal B|\,\omega\,\sigma_\infty,
\end{aligned}
$$

where $\mathcal B$ is the set of live blocks. The schedule fixes the challenge
family, its relevant norm bounds, an admissible fold-response bound
$\beta_{\mathrm{fold}}$, and a digit depth large enough to represent the
accepted response. The implementation may resample the transcript nonce until
the resulting $\mathbf z$ fits that scheduled bound. This grinding helps the
honest prover find a compact response; the range check on its committed digits
is what certifies the bound in the protocol.

Equations (9a), (9b), and (9e) are identities among recomposed values; they do
not yet use the opening digits $\hat{\mathbf e}$, the outer digits
$\hat{\mathbf t}$, or bounded digits for $\mathbf z$. The next-level committed
witness contains digit rings rather than $\mathbf z$ itself so that its
shortness is certified for the Module-SIS binding argument. Akita therefore
decomposes $\mathbf z$ once more:

$$
z_{p,a}(X)
=
\sum_f G_f^{\mathrm{fold}}\hat z_{p,a,f}(X).
\tag{10}
$$

The three main digit segments assembled for the next witness are therefore

$$
\hat{\mathbf z}
\;\Vert\;
\hat{\mathbf e}
\;\Vert\;
\hat{\mathbf t}.
\tag{11}
$$

They have different origins:

| Segment | What it digit-decomposes | Why it is needed |
|---|---|---|
| $\hat{\mathbf z}$ | the challenge-folded block digits $\mathbf z$ | becomes the smaller folded response |
| $\hat{\mathbf e}$ | the position-folded rings $E_b$ | carries the opening data into the fold |
| $\hat{\mathbf t}$ | the inner images $\mathbf t_b$ | binds the folded response to the existing commitment |

## The four semantic relation families

Equation (11) specifies how the three digit segments are assembled, but it
does not impose any algebraic relation among them. Substituting the balanced
recompositions into the identities above gives two relations among the private
witness segments. Equations (4) and (7) provide two additional relations that
define the semantic commitments $\mathbf u$ and $\mathbf v_D$ computed by
$\mathbf B$ and $\mathbf D$, respectively. Together, these give four semantic
roles. Three are ordinary ring relations over

$$
R=F[X]/(X^D+1).
$$

The opening-consistency family is also over $R$ for `EvaluationTrace`; for
packing, it is over $C=E[U]/(U^s+1)$ and is later realized through coordinate
planes. Vector equations are interpreted coordinatewise in their stated ring.
The first two families connect private witness segments to one another. The
last two define the semantic commitments $\mathbf u$ and $\mathbf v_D$.

### 1. Fold-evaluation consistency

For `EvaluationTrace`, Equation (9a) is the fold-evaluation identity among the
recomposed values. Substitute the balanced digit representations of $E_b$
from Equation (6) and $\mathbf z$ from Equation (10) to obtain

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{open}}\hat e_{b,h}
=
\sum_{p,a,f}
Q_pG_a^{\mathrm{in}}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{12a}
$$

This relation uses the random fold challenges $c_b$, not the block-opening weights
$B_b$. The latter belong to the separate field-valued evaluation relation on
$\hat{\mathbf e}$. In particular, this ring relation does not contain the
scalar target $v_{\mathrm{tr}}$.

For `SubringCoefficientPacking`, substitute the packed opening-digit
recomposition and Equation (10) into Equation (9e):

$$
\boxed{
\sum_b c_b(U)
\sum_{j,t,d}\beta_tG_d^{\mathrm{open}}
\hat e_{b,d,t,j}U^j
=
L\!\left(
\left(
\sum_fG_f^{\mathrm{fold}}
\hat z_{p,a,f}
\right)_{p,a}
\right)(U).
}
\tag{12b}
$$

The outer $(\cdot)_{p,a}$ is essential: it reconstructs each coordinate
$z_{p,a}$ separately before $L$ applies the position weights $Q_p$, the inner
gadget weights $G_a^{\mathrm{in}}$, and the packing weights
$I_u^{\mathrm{pack}}$. Equivalently, the right-hand side is

$$
\sum_{j,p,a,u,f}
Q_pG_a^{\mathrm{in}}I_u^{\mathrm{pack}}G_f^{\mathrm{fold}}
[\hat z_{p,a,f}]_{u+k\eta j}U^j.
$$

This is the same semantic obligation in the smaller geometry: the packed
partials on the left and the packed folded source on the right must agree.
The expression is one logical $C$-valued relation. Its physical realization
uses $k$ extension-coordinate planes. The packed E/Q contributions enter the
common relation-weight factorization, while the folded-source contribution is
a separate packing-Z Stage-2 term. Together they realize one protocol claim,
not $k$ independent claims.

### 2. Inner-commitment consistency

Equation (9b) is the inner-commitment identity among the recomposed values.
The next-fold witness stores their balanced digits instead. Substitute
Equation (3) for $\mathbf t_b$ and Equation (10) for $\mathbf z$. For every
row $\rho$ of $\mathbf A$, this gives

$$
\boxed{
\sum_{b,h}
c_bG_h^{\mathrm{out}}\hat t_{b,\rho,h}
=
\sum_{p,a,f}
A_{\rho,(p,a)}G_f^{\mathrm{fold}}\hat z_{p,a,f}.
}
\tag{13}
$$

For packing, every $c_b$ in Equation (13) denotes the embedded element
$\iota(c_b)\in R$. Thus Equations (12b) and (13) use the same
transcript challenge in their respective native geometries.

There is no factor $G_a^{\mathrm{in}}$ on the right of Equation (13):
$\mathbf A$ already acts on the inner digit vector $\mathbf s_b$, whose
columns are indexed by $(p,a)$.

### 3. Outer-commitment consistency

The first two families compare private witness segments but do not yet define
the commitment produced by the outer commitment matrix. The semantic
outer-commitment relation is

$$
\boxed{
\mathbf B\hat{\mathbf t}
=
\mathbf u.
}
\tag{14}
$$

Here $\mathbf u\in R^{n_B}$, where $n_B$ is the output rank of $\mathbf B$.
Equation (14) binds the private outer digits $\hat{\mathbf t}$ to the semantic
commitment $\mathbf u$ and does not use the fold challenges. In the raw
realization, the prover proves this equation directly against the public
$\mathbf u$. In the compressed realization, $\mathbf u$ remains the semantic
intermediate value: the prover recommits it through an $\mathbf F$ chain and
proves that the chain ends at the smaller public payload $p_F$.

### 4. Opening-commitment consistency

Likewise, the semantic opening-commitment relation defines the commitment to
the opening digits under $\mathbf D$:

$$
\boxed{
\mathbf D\hat{\mathbf e}
=
\mathbf v_D.
}
\tag{15}
$$

Here $\mathbf v_D\in R^{n_D}$, where $n_D$ is the output rank of $\mathbf D$.
Equation (15) similarly binds the private opening digits
$\hat{\mathbf e}$ to the semantic opening commitment $\mathbf v_D$. In the
raw realization, the prover proves this equation directly against the public
$\mathbf v_D$. In the compressed realization, $\mathbf v_D$ remains the
semantic intermediate value: the prover recommits it through an $\mathbf H$
chain and proves that the chain ends at the smaller public payload $p_H$.

The public value binding this relation is absorbed before the fold challenges
are sampled: raw mode absorbs $\mathbf v_D$, whereas compressed mode absorbs
$p_H$. Together with the boundedness of $\hat{\mathbf e}$ and Module-SIS
binding, this prevents the prover from adapting the partial-evaluation digits
after learning those challenges. Like the other three families, Equation (15)
is a ring-valued relation distinct from the field-valued scalar evaluation
claim.

The four families above determine the algebraic constraints of a valid fold,
but they do not yet determine which commitment values are transmitted, which
additional witness segments are required, or which native-ring rows appear in
the proved matrix relation. [Raw and compressed realizations of an Akita
fold](./akita-fold-realizations.md) makes those choices explicit while
preserving the method-selected Equation (12a) or (12b) together with
Equations (13)--(15) as their semantic source.
