# Spec: Heterogeneous-basis common folded witness

| Field         | Value      |
|---------------|------------|
| Author(s)     | RadNi      |
| Created       | 2026-06-30 |
| Status        | proposed   |
| PR            |            |
| Supersedes    |            |
| Superseded-by |            |
| Book-chapter  |            |

## Summary

This spec records the first step toward batching several Hachi/Akita opening
claims whose committed polynomials may have different arities and different
commitment decomposition bases. The goal is to place all commitment-opening
digits into one compatibility-keyed coordinate system so the root fold can use
one logical folded response witness $z_\star$, while preserving the ability to
reconstruct each local polynomial block through its original gadget basis.

This document also specifies the next setup-repacking step: a canonical setup
grid from which the local $A$ coefficients are selected so the shared relation

$$
A_\star J_\star \hat z_\star = c \cdot G_{n_A}\hat t
$$

can be formed over the common folded witness. The construction below specifies
the proposed common-$z$ coordinate system, the algebra it must preserve, and the
coordinate rule for choosing compatible $A$ coefficients. Coordinates are shared
only when their full $A$-column semantics match. When two sources have the same
vector coordinate and gadget exponent but different $A$ row semantics, the
planner keeps them in separate namespaces inside the same logical $z_\star$
witness rather than rejecting the batch.

## Intent

### Goal

Define a canonical embedding from each local Hachi opening witness layout into a
single logical folded-witness layout, allowing heterogeneous polynomial sizes
and heterogeneous `log_basis` values to contribute to one random-linear folded
response $z_\star$. The layout maximally shares compatible coordinates and
automatically splits incompatible coordinates into separate namespaces.

### Invariants

1. **Exponent preservation.** A local digit at gadget exponent $b_i^p$ must map
   to a shared slot whose common-basis exponent is equal to $b_i^p$.
2. **Coordinate preservation.** A local vector coordinate $k$ must map to the
   same semantic coordinate $k$ in the shared layout, with zero padding for
   polynomials with smaller $m_i$.
3. **A-profile preservation.** A local digit may share a folded-response
   coordinate with another local digit only if both induce the same full
   $A$-column profile after embedding into the batched row space. Equal
   $(k,e)$ alone is not sufficient.
4. **Linear folding.** The shared folded response is the transcript-challenge
   linear combination of embedded local witnesses:

   $$
   z_\star = \sum_{i,a} c_{i,a} E_i\!\left(s^{(i)}_a\right).
   $$

5. **Local reconstruction.** For every local block, applying the common gadget
   matrix after embedding must recover the same ring vector as applying the
   local gadget matrix before embedding.
6. **Transcript binding.** The batch descriptor must bind every layout parameter
   that affects the embedding: $(m_i, r_i, \texttt{log\_basis}_i, \delta_i)$, the common
   layout, the compatibility namespace map, the padding convention, and the
   claim/block ordering.

### Non-Goals

- This spec does not define the `B`, `D`, `w_hat`, or `t_hat` batching layout.
- This spec only defines the deterministic setup allocation order for additional
  `B`, `D`, or `F` coefficients; it does not define those matrices' batching
  relations.
- This spec does not claim that arbitrary independently sampled local $A_i$
  matrices can share a folded-response coordinate; compatible local $A_i$
  columns must have identical descriptor-bound $A$ profiles. Incompatible
  columns stay in separate namespaces inside $z_\star$.
- This spec does not optimize the norm bound or SIS rank pricing for the
  batched relation.
- This spec does not require backward compatibility with existing proof bytes or
  descriptor bytes.

## Design

### Local Opening Layout

Consider a batch of polynomial-opening claims indexed by $i \in [h]$. Claim $i$
has a multilinear polynomial with

$$
2^{m_i + r_i}
$$

ring coefficients. As in the Hachi split-and-fold layout, write its coefficient
vector as $2^{r_i}$ blocks:

$$
f^{(i)} = \left(f^{(i)}_a\right)_{a \in \{0,1\}^{r_i}},
\qquad
f^{(i)}_a \in R_q^{N_i},
\qquad
N_i := 2^{m_i}.
$$

Claim $i$ uses commitment decomposition base

$$
b_i := 2^{\ell_i}
$$

and digit depth

$$
\delta_i := \left\lceil \log_{b_i} q \right\rceil.
$$

Let

$$
G_i := G_{b_i,N_i}
     := I_{N_i} \otimes
        \begin{bmatrix}
        1 & b_i & b_i^2 & \cdots & b_i^{\delta_i - 1}
        \end{bmatrix}.
$$

Each block is decomposed as

$$
f^{(i)}_a = G_i s^{(i)}_a,
\qquad
s^{(i)}_a \in R_q^{N_i\delta_i}.
$$

We write the local digit coordinates as

$$
s^{(i)}_{a,k,p}
$$

where $0 \le k < N_i$ is the vector coordinate and
$0 \le p < \delta_i$ is the local base-$b_i$ digit index. Thus

$$
\left(G_i s^{(i)}_a\right)_k
  = \sum_{p=0}^{\delta_i - 1} b_i^p s^{(i)}_{a,k,p}.
$$

### Common Exponent Grid

We need a common basis whose powers contain every local gadget exponent. Since
$b_i = 2^{\ell_i}$, the largest possible common `log_basis` with this property is

$$
\ell_\star := \gcd(\ell_1, \ell_2, \ldots, \ell_h),
\qquad
b_\star := 2^{\ell_\star}.
$$

For each claim define

$$
g_i := \frac{\ell_i}{\ell_\star}.
$$

Then every local exponent has an exact representation in the common basis:

$$
b_i^p
  = \left(2^{\ell_i}\right)^p
  = \left(2^{\ell_\star}\right)^{p g_i}
  = b_\star^{p g_i}.
$$

The semantic vector coordinate dimension is

$$
N_\star := 2^{m_\star},
\qquad
m_\star := \max_i m_i.
$$

The common digit-slot depth must cover every embedded local exponent. A minimal
slot depth for the sparse embedding is

$$
\Delta_\star := 1 + \max_i \left((\delta_i - 1)g_i\right).
$$

Equivalently, the implementation may use the full common-base depth

$$
\delta_\star := \left\lceil \log_{b_\star} q \right\rceil
$$

as long as $\Delta_\star \le \delta_\star$ and unused slots are zero. The full-depth choice
is simpler to bind in descriptors and aligns with a canonical $G_{b_\star,N_\star}$.

The exponent grid alone is not enough to decide sharing. Each local digit also
has an $A$-role meaning: it contributes a column vector to the batched
$A$-relation after the local rows have been embedded into the common row space.
Let

$$
\chi_i(k,p)
$$

denote the descriptor-bound $A$-column profile of the local digit
$s^{(i)}_{a,k,p}$. The profile is the full embedded column semantics: the
common row embedding, the active row domain, and the setup-coordinate handle
read by each active row. Inactive rows are part of the profile as zeros.

Two digits are allowed to share one folded-response coordinate only when their
profiles are equal. This handles heterogeneous ranks generally: if one source
has rows $0,\ldots,3$ active and another has rows $0,\ldots,5$ active, their
profiles are different even if the first four coefficient handles coincide, so
the planner allocates two coordinates. A later optimization may deliberately
lift the smaller source to the larger row profile, committing the extra
$\hat t$ rows, in order to make the profiles equal and recover sharing.

Define the compatibility-keyed coordinate set

$$
\mathcal{J}_\star
  :=
  \left\{(k,e,\chi_i(k,p)):
    i \in [h],\
    0 \le k < N_i,\
    0 \le p < \delta_i,\
    e = p g_i
  \right\}.
$$

The common reconstruction gadget is the linear map

$$
G_\star : R_q^{\mathcal{J}_\star} \to R_q^{N_\star},
\qquad
\left(G_\star z\right)_k
  :=
  \sum_{\substack{(k',e,\chi)\in\mathcal{J}_\star\\ k'=k}}
    b_\star^e z_{k,e,\chi}.
$$

Thus $G_\star$ ignores the compatibility namespace when recomposing field
values, but the $A$ relation does not: $A_\star$ has one column for every
coordinate in $\mathcal{J}_\star$.

### Sparse Exponent- and Profile-preserving Embedding

For every claim $i$, define a linear embedding

$$
E_i : R_q^{N_i\delta_i} \longrightarrow R_q^{\mathcal{J}_\star}
$$

by

$$
\left(E_i(s)\right)_{k,e,\chi}
=
\begin{cases}
s_{k,p} &
\text{if } 0 \le k < N_i,\ e = p g_i,\ \chi=\chi_i(k,p)
  \text{ for some } 0 \le p < \delta_i, \\
0 &
\text{otherwise.}
\end{cases}
$$

Equivalently, each local digit `s^{(i)}_{a,k,p}` is placed at shared slot

$$
\operatorname{slot}_i(k,p) := (k, p g_i, \chi_i(k,p)).
$$

All coordinates $k \ge N_i$ are padding coordinates and are zero for claim $i$.

This embedding is exponent-preserving:

$$
\begin{aligned}
\left(G_\star E_i(s)\right)_k
  &= \sum_{\substack{(k',e,\chi)\in\mathcal{J}_\star\\ k'=k}}
     b_\star^e \left(E_i(s)\right)_{k,e,\chi} \\
  &= \sum_{p=0}^{\delta_i - 1} b_\star^{p g_i} s_{k,p} \\
  &= \sum_{p=0}^{\delta_i - 1} b_i^p s_{k,p} \\
  &= \left(G_i s\right)_k.
\end{aligned}
$$

for $0 \le k < N_i$, and $\left(G_\star E_i(s)\right)_k = 0$ for padded coordinates
$N_i \le k < N_\star$.

Thus

$$
G_\star E_i\!\left(s^{(i)}_a\right)
  = \operatorname{pad}_i\!\left(f^{(i)}_a\right),
$$

where $\operatorname{pad}_i : R_q^{N_i} \to R_q^{N_\star}$ appends zero coordinates.

### Shared Folded Response

After the prover commits to the relevant first-round messages, the transcript
samples folding challenges

$$
c_{i,a} \in \mathcal{C}
$$

for every claim $i$ and block $a \in \{0,1\}^{r_i}$. The proposed shared folded
response is

$$
z_\star
  := \sum_{i=1}^{h} \sum_{a \in \{0,1\}^{r_i}}
       c_{i,a} E_i\!\left(s^{(i)}_a\right)
  \in R_q^{\mathcal{J}_\star}.
$$

At a concrete coordinate $(k,e,\chi)$,

$$
\left(z_\star\right)_{k,e,\chi}
  =
  \sum_{\substack{i,a,p:\\ k < N_i,\ e = p g_i,\\ \chi_i(k,p)=\chi}}
      c_{i,a} s^{(i)}_{a,k,p}.
$$

This is the intended "shared slot" behavior: two local digits contribute to the
same coordinate of $z_\star$ exactly when they represent the same vector
coordinate $k$, the same power of the common basis $b_\star^e$, and the same
embedded $A$-column profile $\chi$.

The common reconstruction is

$$
\begin{aligned}
G_\star z_\star
  &= \sum_{i,a} c_{i,a} G_\star E_i\!\left(s^{(i)}_a\right) \\
  &= \sum_{i,a} c_{i,a} \operatorname{pad}_i\!\left(f^{(i)}_a\right).
\end{aligned}
$$

This identity is the core algebraic reason for constructing $z_\star$ on the
compatibility-keyed common exponent grid.

### Relation to $\hat z$

The object $z_\star$ is the folded response. As in Hachi, the relation sent to the
next recursive layer should not expose $z_\star$ as an unrestricted field vector.
Instead, it is decomposed into a short digit witness:

$$
z_\star = J_\star \hat z_\star.
$$

The decomposition gadget $J_\star$ may use the common basis $b_\star$ or another
protocol-selected fold decomposition basis. This choice is separate from the
embedding basis, but it must be transcript-bound and priced in the norm/SIS
analysis.

The target relation for the next recursive layer is therefore of the form

$$
A_\star J_\star \hat z_\star = \operatorname{batched\_t\_side},
$$

where $\operatorname{batched\_t\_side}$ will be a challenge-weighted combination of the embedded
inner commitment images. The setup grid below fixes the compatible $A_\star$
coefficient coordinates; the exact $\operatorname{batched\_t\_side}$ layout remains
separate from this section.

### Setup Repacking and A Profiles

During setup generation, the planner receives the number of committed
polynomials and their sizes, along with the local decomposition bases used by
their commitment openings. From these inputs it derives the common-basis layout
from the previous sections:

$$
D_\star := \delta_\star = \left\lceil \log_{b_\star} q \right\rceil,
\qquad
L_\star := N_\star = 2^{m_\star}.
$$

The planner also derives a common row domain $\mathcal{R}_\star$ for the
batched $A$ relation. This row domain may be a disjoint union of local row
domains, a shared lifted row domain, or a mixture chosen by the planner. The
only invariant is algebraic:

$$
A_\star E_i = \bar A_i
$$

for every source $i$, where $\bar A_i$ is the local $A_i$ matrix embedded into
the common row domain and zero on rows that do not belong to source $i$.

For every local digit $(i,k,p)$ the planner defines the profile

$$
\chi_i(k,p): \mathcal{R}_\star \to \Lambda_\star \cup \{\bot\}.
$$

Here $\Lambda_\star$ is the set of logical setup-coordinate handles, and
$\bot$ means the row is inactive for this local digit. If
$\chi_i(k,p)(\rho)=\lambda$, then row $\rho$ of the embedded local $A$ column
uses setup handle $\lambda$.

The shared coordinate set $\mathcal{J}_\star$ from the previous section uses
the whole profile $\chi_i(k,p)$, not only the active-row count. Therefore the
default setup-repacking rule is:

1. if two local digits share one folded-response coordinate, their complete
   $A$ profiles are byte-identical and they use the same setup handles on every
   active row;
2. if their profiles differ, they are separate $z_\star$ coordinates and their
   setup handles are independent unless a later proof explicitly justifies a
   correlated-column construction.

This rule is what makes heterogeneous row ranks safe. A one-hot source with
four active $A$ rows and a full-field source with six active $A$ rows do not
share the same profile merely because rows $0,\ldots,3$ use the same formula.
Sharing would incorrectly make the one-hot contribution appear in rows $4,5$.
The planner can still choose to lift the one-hot source to the six-row profile,
commit those extra $\hat t$ rows, and thereby make the profiles equal. That is
a cost/performance choice, not a soundness requirement.

The setup object is the sorted set of all nonzero logical handles appearing in
the descriptor-bound profiles:

$$
\mathcal{U}_A
  :=
  \left\{\lambda \in \Lambda_\star:
    \exists (k,e,\chi)\in \mathcal{J}_\star,\ \exists \rho\in\mathcal{R}_\star
    \text{ with } \chi(\rho)=\lambda
  \right\}.
$$

The physical setup coefficient order is the canonical lexicographic order of
the handles in $\mathcal{U}_A$. A simple default handle format is

$$
\lambda=(\mathtt{A},\operatorname{profile\_id},\rho,k,e),
$$

where `profile_id` is the canonical digest or table index of $\chi$. With this
default, incompatible profiles receive independent setup coefficients even when
they have the same $(\rho,k,e)$ on some rows. If the planner wants two sources
to reuse setup coefficients, it must make their full profiles equal by
construction.

The shared matrix is then

$$
A_\star[\rho,(k,e,\chi)]
  :=
  \begin{cases}
  \operatorname{SETUP}[\chi(\rho)] & \text{if } \chi(\rho)\neq \bot,\\
  0 & \text{if } \chi(\rho)=\bot.
  \end{cases}
$$

For claim $i$, recall that $g_i = \ell_i / \ell_\star$. A local digit
$s^{(i)}_{a,k,p}$ embeds into coordinate $(k,pg_i,\chi_i(k,p))$, and the
definition above gives exactly the embedded local column:

$$
\left(A_\star E_i(s)\right)_\rho
  =
  \left(\bar A_i s\right)_\rho.
$$

Consequently the batched $A$ relation is honest:

$$
A_\star z_\star
  =
  \sum_{i,a} c_{i,a}\,\bar A_i s^{(i)}_a.
$$

The right-hand side is represented by the corresponding challenge-weighted
combination of embedded inner commitment images. Its `t_hat` layout must bind
the same row domain $\mathcal{R}_\star$ and the same source-to-row embedding.

The auxiliary allocation order for `B`, `D`, `F`, or related matrices first uses
the setup handles selected by the $A$ rule above. If more setup coefficients are
needed, the extension order scans the same descriptor-bound handle namespace in
canonical order:

$$
(\mathtt{B},0),(\mathtt{B},1),\ldots,\quad
(\mathtt{D},0),(\mathtt{D},1),\ldots,\quad
(\mathtt{F},0),(\mathtt{F},1),\ldots.
$$

The batch descriptor must bind $D_\star$, $L_\star$, $\mathcal{R}_\star$, the
canonical profile table, the handle ordering convention, every local tuple
$(m_i,r_i,\ell_i,\delta_i)$, and the claim/block ordering that determines which
local $A$ profiles are used.

### Setup-Coefficient Accounting Examples

The setup footprint is the number of distinct nonzero handles in the selected
$A$ profiles:

$$
\mathcal{U}_A
  :=
\left\{\lambda:
  \exists (k,e,\chi)\in\mathcal{J}_\star,\ \exists \rho\in\mathcal{R}_\star
  \text{ with } \chi(\rho)=\lambda
\right\}.
$$

The number of distinct setup coefficients consumed by $A$ is

$$
|\mathcal{U}_A|.
$$

Here $\delta^{A}_i$ is the digit depth of the witness actually committed by the
$A$ matrix. For current one-hot root commitments, $\delta^{A}_i = 1$. For current
recursive commitments, $\delta^{A}_i = 1$ as well because the recursive committed
witness is a balanced-digit witness whose commit bound collapses to its
`log_basis`. For a full-field root commitment over a 128-bit field,

$$
\delta^{A}_i = \left\lceil \frac{128}{\ell_i} \right\rceil.
$$

#### One-hot plus full-field, same 25-variable size

Using the current `fp128_d64_onehot` and `fp128_d64_full` generated root layouts
for `num_vars = 25`, both roots use $\ell = 3$, so $\ell_\star = 3$ and
$g_i = 1$.

The one-hot root has

$$
L_{\mathsf{oh}} = 2^{12},\qquad R_{\mathsf{oh}} = 4,\qquad
\delta^{A}_{\mathsf{oh}} = 1,
$$

so it selects

$$
1 \cdot 4 \cdot 2^{12} = 16{,}384
$$

setup coefficients, all at common exponent row $e = 0$.

The full-field root has

$$
L_{\mathsf{full}} = 2^{10},\qquad R_{\mathsf{full}} = 6,\qquad
\delta^{A}_{\mathsf{full}} = \left\lceil \frac{128}{3} \right\rceil = 43,
$$

so it selects

$$
43 \cdot 6 \cdot 2^{10} = 264{,}192
$$

setup coefficients, at common exponent rows $e = 0,1,\ldots,42$.

Under the compatibility-keyed rule, the apparent overlap at

$$
\{(0,r,k): 0 \le r < 4,\ 0 \le k < 2^{10}\},
$$

is not shareable by default. The one-hot source has four active $A$ rows, while
the full-field source has six. Their embedded $A$ profiles differ because rows
$4,5$ are zero for the one-hot source and nonzero for the full-field source.
Therefore the default safe layout puts those low-exponent digits in separate
compatibility namespaces and consumes

$$
16{,}384 + 264{,}192 = 280{,}576
$$

distinct $A$ setup coefficients.

The planner has another safe option: lift the one-hot source to the six-row
profile and commit the two additional $\hat t$ rows in its `B` relation. Then
the e=0 profile can be made identical on the overlap

$$
\{(0,r,k): 0 \le r < 6,\ 0 \le k < 2^{10}\}.
$$

In that lifted layout, the one-hot source selects

$$
1\cdot 6\cdot 2^{12}=24{,}576
$$

coefficients, the overlap has size $6\cdot 2^{10}=6{,}144$, and the distinct
$A$ setup footprint becomes

$$
24{,}576 + 264{,}192 - 6{,}144 = 282{,}624.
$$

This is larger in setup footprint than the default split in this concrete
example, but it may reduce the folded-witness and relation-layout cost by
sharing actual $z_\star$ coordinates. The planner should compare these choices
using proof-size and verifier-time costs, not only setup coefficient count.

#### Two one-hot roots, same 40-variable size

Using the current `fp128_d64_onehot` generated root layout for `num_vars = 40`,
each one-hot root has

$$
\ell = 2,\qquad
L = 2^{21},\qquad
R = 7,\qquad
\delta^{A} = 1.
$$

Each root individually selects

$$
1 \cdot 7 \cdot 2^{21} = 14{,}680{,}064
$$

$A$ setup coefficients, all at $e = 0$. If two such roots use the same row
embedding and the same profile table, their selected profiles are identical:

$$
\chi_1(k,0)=\chi_2(k,0)
\qquad\text{for every }0\le k<2^{21}.
$$

Thus the proposed grid consumes

$$
14{,}680{,}064
$$

distinct $A$ setup coefficients in total, which equals the maximum individual
footprint:

$$
\max(14{,}680{,}064,\ 14{,}680{,}064) = 14{,}680{,}064.
$$

These examples isolate coordinate reuse for $A$. A final planner may still choose
a different batched SIS rank once norm accounting for the shared folded witness
is finalized.

## Compatibility Condition for $A$

The common-$z$ embedding is algebraically useful only if the linear rows that
act on $z_\star$ are compatible with the local rows they replace. Abstractly,
if claim $i$ previously used a local matrix $A_i$, then the shared matrix
$A_\star$ must satisfy

$$
A_\star E_i = \bar A_i
$$

for every source, where $\bar A_i$ is $A_i$ embedded into the common row domain.
The compatibility namespace is the mechanism that enforces this equation
without rejecting heterogeneous layouts.

Equivalently, whenever two local digits land in the same shared slot, every
batched row that touches that slot must assign the same coefficient to both
digits and every inactive row must be inactive for both. If this is not true,
the digits are not rejected; they simply receive different profile identifiers
and therefore different columns of $z_\star$.

Partial coefficient reuse across incompatible profiles is deliberately outside
the default rule. It creates correlated $A_\star$ columns that are not covered
by the ordinary Module-SIS argument for a uniform role matrix. A future design
may allow such reuse only with an explicit correlation-aware binding proof.

## Norm Accounting

Let local opening digits for claim $i$ satisfy

$$
\left\|s^{(i)}_a\right\|_\infty \le B_i.
$$

For balanced base-$b_i$ digits, typically $B_i = b_i / 2$ up to the exact
rounding convention. Let

$$
\omega_{i,a} := \left\|c_{i,a}\right\|_1.
$$

A conservative coordinate-wise bound for $z_\star$ is

$$
\left\|z_\star\right\|_\infty
  \le
  \max_{k,e,\chi}
  \sum_{\substack{i,a,p:\\ k < N_i,\ e = p g_i,\\ \chi_i(k,p)=\chi}}
       \omega_{i,a} B_i.
$$

A simpler implementation bound is

$$
\left\|z_\star\right\|_\infty
  \le \sum_{i,a} \omega_{i,a} B_i.
$$

The tighter bound should eventually account for both exponent-slot sparsity and
profile sparsity: a claim with $g_i > 1$ occupies only every $g_i$-th exponent
slot, and a claim with a distinct $A$ profile does not contribute to other
profile namespaces. The first implementation may use the simpler bound if the
resulting SIS rank and proof-size cost are acceptable.

## Open Questions

- Should $J_\star$ use the common embedding basis $b_\star$, the maximum local basis, or
  a fold-specific basis optimized for proof size?
- Can the norm bound exploit the sparse exponent occupancy without complicating
  verifier-reachable layout derivation?
- How should different $r_i$ values be represented in the transcript challenge
  schedule so the extractor can isolate each local block?
- When is lifting a smaller source to a larger shared $A$ profile worth the
  extra `t_hat` and `B`-binding cost?

## Evaluation

### Acceptance Criteria

- [ ] A formal $E_i$ embedding is implemented or otherwise encoded in the batch
      descriptor, and prover/verifier derive identical slot maps.
- [ ] For every local claim, a unit test checks
      $G_\star E_i(s) = \operatorname{pad}_i(G_i s)$ on randomized digit witnesses.
- [ ] The transcript binds all local and common layout parameters that affect
      slot placement, including the canonical profile table and row-domain
      embedding.
- [ ] A planner or verifier check confirms that every shared coordinate has one
      byte-identical $A$ profile and that incompatible profiles are allocated to
      distinct coordinates.
- [ ] Norm-bound tests cover heterogeneous `log_basis` values, heterogeneous
      $m_i$, sparse occupancy where $g_i > 1$, and profile splitting where two
      sources share $(k,e)$ but not $\chi$.

### Testing Strategy

The first tests should be algebraic and deterministic:

- randomized local digit vectors for several $(m_i, \texttt{log\_basis}_i)$ pairs;
- exact slot-map tests for $\ell_\star = \gcd_i \ell_i$;
- padding tests where $m_i < m_\star$;
- split-namespace tests where incompatible $A$ profiles at the same $(k,e)$
  become distinct coordinates;
- negative tests where a descriptor attempts to map two non-identical profiles
  to the same coordinate.

Once the $A_\star$ design is added, end-to-end prover/verifier tests should cover at
least two polynomials with different sizes and different `log_basis` values in
one shared root fold.

### Performance

The common layout may increase witness length when $\ell_\star$ is much smaller than
some local $\ell_i$, because

$$
\delta_\star = \left\lceil \log_{2^{\ell_\star}} q \right\rceil
$$

can be larger than every local $\delta_i$. The planner should compare one shared
batch against batching by `log_basis` cohort:

- fixed-$\ell_i$ cohorts, which produce several smaller witnesses $z_g$;
- one gcd-basis batch, which produces one larger witness $z_\star$.

The shared batch is worthwhile only when the proof-size and verifier-time savings
from one root relation exceed the extra common-layout and norm-bound cost.

## Documentation

This is an in-flight protocol design record. If implemented, the durable
explanation should be folded into the Akita Book sections that describe
same-point batching, root folding, and the Hachi relation layout.

## Execution

This spec should evolve in two stages:

1. Finalize the common-$z$ layout and descriptor binding.
2. Finalize the $A_\star$ setup-repacking design and update the remaining
   `B`, `D`, `w_hat`, and `t_hat` batching layouts.

No implementation should expose this batch shape until the second stage is
complete.

## References

- Hachi paper, Section 4.1: inner and outer commitment.
- Hachi paper, Section 4.2: polynomial evaluation as a quadratic equation and
  folded witness relation.
- [`multi-group-batching.md`](multi-group-batching.md)
- [`single-point-opening-batch.md`](single-point-opening-batch.md)
