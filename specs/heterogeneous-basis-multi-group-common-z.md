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
digits into one shared coordinate system so the root fold can use one folded
response witness $z_\star$, while preserving the ability to reconstruct each local
polynomial block through its original gadget basis.

This document also specifies the next setup-repacking step: a canonical setup
grid from which the local $A$ coefficients are selected so the shared relation

$$
A_\star J_\star \hat z_\star = c \cdot G_{n_A}\hat t
$$

can be formed over the common folded witness. The construction below specifies
the proposed common-$z$ coordinate system, the algebra it must preserve, and the
coordinate rule for choosing compatible $A$ coefficients.

## Intent

### Goal

Define a canonical embedding from each local Hachi opening witness layout into a
single shared folded-witness layout, allowing heterogeneous polynomial sizes and
heterogeneous `log_basis` values to contribute to one random-linear folded
response $z_\star$.

### Invariants

1. **Exponent preservation.** A local digit at gadget exponent $b_i^p$ must map
   to a shared slot whose common-basis exponent is equal to $b_i^p$.
2. **Coordinate preservation.** A local vector coordinate $k$ must map to the
   same semantic coordinate $k$ in the shared layout, with zero padding for
   polynomials with smaller $m_i$.
3. **Linear folding.** The shared folded response is the transcript-challenge
   linear combination of embedded local witnesses:

   $$
   z_\star = \sum_{i,a} c_{i,a} E_i\!\left(s^{(i)}_a\right).
   $$

4. **Local reconstruction.** For every local block, applying the common gadget
   matrix after embedding must recover the same ring vector as applying the
   local gadget matrix before embedding.
5. **Transcript binding.** The batch descriptor must bind every layout parameter
   that affects the embedding: $(m_i, r_i, \texttt{log\_basis}_i, \delta_i)$, the common
   layout, the padding convention, and the claim/block ordering.

### Non-Goals

- This spec does not define the `B`, `D`, `w_hat`, or `t_hat` batching layout.
- This spec only defines the deterministic setup allocation order for additional
  `B`, `D`, or `F` coefficients; it does not define those matrices' batching
  relations.
- This spec does not claim that arbitrary independently sampled local $A_i$
  matrices can share $z_\star$; compatible local $A_i$ matrices must be derived
  from the setup-repacking grid.
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

The shared coordinate dimension is

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

Define the common gadget matrix

$$
G_\star := G_{b_\star,N_\star}
     := I_{N_\star} \otimes
        \begin{bmatrix}
        1 & b_\star & b_\star^2 & \cdots & b_\star^{\delta_\star - 1}
        \end{bmatrix}.
$$

### Sparse Exponent-preserving Embedding

For every claim $i$, define a linear embedding

$$
E_i : R_q^{N_i\delta_i} \longrightarrow R_q^{N_\star\delta_\star}
$$

by

$$
\left(E_i(s)\right)_{k,e}
=
\begin{cases}
s_{k,p} &
\text{if } 0 \le k < N_i,\ e = p g_i\ \text{for some } 0 \le p < \delta_i, \\
0 &
\text{otherwise.}
\end{cases}
$$

Equivalently, each local digit `s^{(i)}_{a,k,p}` is placed at shared slot

$$
\operatorname{slot}_i(k,p) := k\delta_\star + p g_i.
$$

All coordinates $k \ge N_i$ are padding coordinates and are zero for claim $i$.

This embedding is exponent-preserving:

$$
\begin{aligned}
\left(G_\star E_i(s)\right)_k
  &= \sum_{e=0}^{\delta_\star - 1} b_\star^e \left(E_i(s)\right)_{k,e} \\
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
  \in R_q^{N_\star\delta_\star}.
$$

At a concrete coordinate $(k,e)$,

$$
\left(z_\star\right)_{k,e}
  =
  \sum_{\substack{i,a,p:\\ k < N_i,\ e = p g_i}}
      c_{i,a} s^{(i)}_{a,k,p}.
$$

This is the intended "shared slot" behavior: two local digits contribute to the
same coordinate of $z_\star$ exactly when they represent the same vector coordinate
$k$ and the same power of the common basis $b_\star^e$.

The common reconstruction is

$$
\begin{aligned}
G_\star z_\star
  &= \sum_{i,a} c_{i,a} G_\star E_i\!\left(s^{(i)}_a\right) \\
  &= \sum_{i,a} c_{i,a} \operatorname{pad}_i\!\left(f^{(i)}_a\right).
\end{aligned}
$$

This identity is the core algebraic reason for constructing $z_\star$ on the common
exponent grid.

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

### Setup Repacking Grid

During setup generation, the planner receives the number of committed
polynomials and their sizes, along with the local decomposition bases used by
their commitment openings. From these inputs it derives the common-basis layout
from the previous sections:

$$
D_\star := \delta_\star = \left\lceil \log_{b_\star} q \right\rceil,
\qquad
L_\star := N_\star = 2^{m_\star}.
$$

The planner also derives the setup rank

$$
R_\star
$$

required by the batched $A$ relation. The generated setup is viewed as a
rectangle with

$$
D_\star R_\star
$$

rows and

$$
L_\star
$$

columns. It is indexed by logical coordinates

$$
\operatorname{SETUP}[e,r,k],
\qquad
0 \le e < D_\star,\quad
0 \le r < R_\star,\quad
0 \le k < L_\star.
$$

Equivalently, the physical row is $eR_\star + r$ and the column is $k$. Define
one physical setup row by

$$
\operatorname{row}(e,r)
  :=
  \begin{bmatrix}
  \operatorname{SETUP}[e,r,0] &
  \operatorname{SETUP}[e,r,1] &
  \cdots &
  \operatorname{SETUP}[e,r,L_\star - 1]
  \end{bmatrix}.
$$

Then the canonical row-major order is the following stack of row vectors, grouped
first by the common exponent coordinate $e$ and then by the local row coordinate
$r$:

$$
\begin{bmatrix}
\operatorname{row}(0,0) \\
\operatorname{row}(0,1) \\
\vdots \\
\operatorname{row}(0,R_\star - 1) \\[2pt]
\operatorname{row}(1,0) \\
\vdots \\
\operatorname{row}(D_\star - 1,R_\star - 1)
\end{bmatrix}.
$$

For claim $i$, recall that $g_i = \ell_i / \ell_\star$. A local digit
$s^{(i)}_{a,k,p}$ embeds into common exponent slot $e = pg_i$. The corresponding
local $A$ coefficient is selected from the same shared coordinate:

$$
A^{(i)}[r,k,p]
  := \operatorname{SETUP}[pg_i,r,k],
$$

for all

$$
0 \le r < \operatorname{rank}(A^{(i)}),\qquad
0 \le k < N_i,\qquad
0 \le p < \delta_i.
$$

Thus the vector coordinate $k$ remains the same semantic coordinate in the
shared grid, while only the local digit index $p$ is converted into the common
exponent coordinate $pg_i$.

The shared $A_\star$ coefficients are therefore

$$
A_\star[r,k,e] := \operatorname{SETUP}[e,r,k],
$$

and the local matrices are exactly restrictions of $A_\star$ along the embedding
map $E_i$.

The auxiliary allocation order for `B`, `D`, `F`, or related matrices first uses
the coordinates selected by the $A$ rule above. If more setup coefficients are
needed, the extension order scans the same setup grid in canonical order:

$$
(0,0,0), (0,0,1), \ldots, (0,0,L_\star - 1),
(0,1,0), \ldots, (D_\star - 1,R_\star - 1,L_\star - 1).
$$

The batch descriptor must bind $D_\star$, $R_\star$, $L_\star$, the row-major
ordering convention, every local tuple $(m_i,r_i,\ell_i,\delta_i)$, and the
claim/block ordering that determines which local $A$ restrictions are used.

## Compatibility Condition for $A$

The common-$z$ embedding is algebraically useful only if the linear rows that act
on $z_\star$ are compatible with the local rows they replace. Abstractly, if claim
$i$ previously used a local matrix $A_i$, then a shared matrix $A_\star$ must satisfy

$$
A_\star E_i = A_i
$$

for every claim whose digits are embedded into shared slots, or else the batch
must use disjoint slots for the incompatible columns.

Equivalently, whenever two local digits land in the same shared slot, every
batched row that touches that slot must assign the same coefficient to both
digits. The setup-repacking grid enforces this condition by assigning one
coefficient to each shared coordinate $(e,r,k)$ and deriving each local
coefficient from the embedded coordinate $(pg_i,r,k)$.

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
  \max_{k,e}
  \sum_{\substack{i,a,p:\\ k < N_i,\ e = p g_i}}
       \omega_{i,a} B_i.
$$

A simpler implementation bound is

$$
\left\|z_\star\right\|_\infty
  \le \sum_{i,a} \omega_{i,a} B_i.
$$

The tighter bound should eventually account for exponent-slot sparsity: a claim
with $g_i > 1$ occupies only every $g_i$-th slot, so it does not contribute to
all common digit slots. The first implementation may use the simpler bound if
the resulting SIS rank and proof-size cost are acceptable.

## Open Questions

- If two claims use incompatible $A$ coefficients at the same $(k,e)$ slot,
  should the planner split them into disjoint slot namespaces or separate root
  batches?
- Should $J_\star$ use the common embedding basis $b_\star$, the maximum local basis, or
  a fold-specific basis optimized for proof size?
- Can the norm bound exploit the sparse exponent occupancy without complicating
  verifier-reachable layout derivation?
- How should different $r_i$ values be represented in the transcript challenge
  schedule so the extractor can isolate each local block?

## Evaluation

### Acceptance Criteria

- [ ] A formal $E_i$ embedding is implemented or otherwise encoded in the batch
      descriptor, and prover/verifier derive identical slot maps.
- [ ] For every local claim, a unit test checks
      $G_\star E_i(s) = \operatorname{pad}_i(G_i s)$ on randomized digit witnesses.
- [ ] The transcript binds all local and common layout parameters that affect
      slot placement.
- [ ] A planner or verifier check rejects batches whose $A$ rows are not derived
      from the setup-repacking grid at the embedded coordinates $(pg_i,r,k)$.
- [ ] Norm-bound tests cover heterogeneous `log_basis` values, heterogeneous
      $m_i$, and sparse occupancy where $g_i > 1$.

### Testing Strategy

The first tests should be algebraic and deterministic:

- randomized local digit vectors for several $(m_i, \texttt{log\_basis}_i)$ pairs;
- exact slot-map tests for $\ell_\star = \gcd_i \ell_i$;
- padding tests where $m_i < m_\star$;
- negative tests where incompatible slot maps or incompatible $A$ coefficients
  are rejected before proof construction.

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
