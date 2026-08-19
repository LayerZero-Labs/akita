# Spec: Compact Role-Native Witness Layout

| Field | Value |
|---|---|
| Author(s) | Quang Dao; Codex assistant |
| Created | 2026-07-31 |
| Revised | 2026-07-31 |
| Status | active |
| PR | #337 |
| Supersedes | Projected-digit and outgoing-witness storage rules in `archive/2026-Q3/digit-innermost-layout.md`, the deleted mixed-ring experiment, `archive/2026-Q3/distributed-setup-offloading.md`, and `archive/2026-Q3/relation-range-image-sumcheck.md` |
| Superseded-by | |
| Book-chapter | book/src/how/proving/opening-points-layout.md |
| Related-chapter | book/src/how/verifying/matrix_evaluation.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals, as described in RFC 2119 and
RFC 8174.

## Authority

This document is the normative source for all of the following:

- persistent commitment-hint storage;
- projected B and D digit order;
- physical Z, E, T, and R witness coefficient order;
- group and chunk nesting in the outgoing witness;
- exact live witness length and its one zero suffix;
- the common relation-coefficient block used by Stage 2 and Stage 3; and
- the validation and performance requirements of those layouts.

The related specs continue to own source-polynomial block order, transcript
group authentication, relation algebra, setup-prefix packing, challenge
sampling, and planner policy. If another live document describes an outgoing
ring-element carrier, per-unit carrier padding, group-major witness units, or
a different projected-digit order, this document takes precedence.

## Decision

The outgoing witness is one compact vector of field coefficients. It is not a
vector of rings in a batch-wide dimension.

For every chunk, the prover emits every group in authenticated relation order:

```text
chunk 0: [group 0: Z | E | T] [group 1: Z | E | T] ...
chunk 1: [group 0: Z | E | T] [group 1: Z | E | T] ...
...
shared:  [native R rows]
suffix:  [zeros, if required by the successor Boolean domain]
```

Every live segment uses its native ring dimension. No maximum A dimension is
computed for witness storage. No live segment contains padding for a different
group or role.

E and T use the same split-before-decompose rule:

```text
[semantic value][native role subcolumn][digit][native coefficient]
```

Uniform dimensions are the `q = 1` instance of this representation. They do
not select another protocol path.

This is a breaking cutover. Implementations **MUST NOT** retain a layout mode,
compatibility decoder, conversion wrapper, or parallel uniform implementation.

## Notation

Let

\[
R_d = F[X]/(X^d+1).
\]

Let groups be enumerated in authenticated relation order by positions
`p = 0, ..., G - 1`. Let `g(p)` be the corresponding stable group index. At a
root this is the final/new group first, followed by precommitted groups in their
authenticated order.

For group `g`, define:

```text
a_g       A ring dimension
b_g       B ring dimension
d_g       D ring dimension
H_g       number of claims
F_g       total live source blocks
M_g       positions per source block
n_A,g     A matrix row count
delta_Z,g inner/witness digit count
delta_F,g fold digit count
delta_B,g outer digit count
delta_D,g opening digit count
beta_B,g  outer gadget basis
beta_D,g  opening gadget basis
```

For the relation quotient, let rows be indexed by `rho`, let `r_rho` be the
native ring dimension of row `rho`, and let `delta_R` be the quotient digit
count. `RelationRhsLayout::row_ring_dims()` is the single source of truth for
the ordered sequence `(r_rho)`.

All role dimensions **MUST** be nonzero powers of two and **MUST** satisfy

\[
b_g\mid a_g,\qquad d_g\mid a_g.
\]

Define

\[
q_{B,g}=a_g/b_g,\qquad q_{D,g}=a_g/d_g.
\]

There is no witness-storage quantity `C`, `max_g a_g`, `C/b_g`, or `C/d_g`.

## Role-native split and decomposition

For `y in R_a` and a native role dimension `r | a`, define `q = a/r` and

\[
y_s(X)=\sum_{k=0}^{r-1} y_{sr+k}X^k\in R_r,
\qquad 0\le s<q.
\]

Then

\[
y(X)=\sum_{s=0}^{q-1}X^{sr}y_s(X).
\]

With gadget basis `beta` and `delta` digits, decompose each native subring:

\[
y_s(X)=\sum_{o=0}^{\delta-1}\beta^o\widehat y_{s,o}(X).
\]

Therefore

\[
\boxed{
y(X)=\sum_{s=0}^{q-1}\sum_{o=0}^{\delta-1}
X^{sr}\beta^o\widehat y_{s,o}(X)
}.
\]

The implementation **MUST** split into native role subrings before gadget
decomposition. It **MUST** store `s` outside `o`:

\[
\operatorname{flat}(s,o,k)=((s\delta+o)r+k).
\]

It **MUST NOT** produce A-wide digit planes and transpose them during witness
emission. Recomposition, relation evaluation, setup evaluation, and commitment
kernels **MUST** consume this same order.

When `q = 1`, implementations **MUST** treat the projection as structural
identity: no projection-power allocation, no multiplication by one, and no
empty-factor traversal.

## Persistent commitment hints

`AkitaCommitmentHint<F>` contains one A-native `RingVec<F>` per polynomial in
claim order and one shared A dimension.

For polynomial `h`, its row stores

```text
[source block][A row][A coefficient]
```

and has coefficient length

\[
F_g n_{A,g} a_g.
\]

All polynomial rows in one hint **MUST** have equal coefficient length and the
same nonzero A dimension. The serialized form **MUST** be:

```text
polynomial_count
A_ring_dimension
for each polynomial:
    coefficient_count
    coefficients in stored order
```

The hint **MUST NOT** persist B digits, packed digits, per-block wrappers, or a
recomposition cache. Proof preparation derives one temporary `t_hat` from the
stored A rows. The A relation consumes the stored A rows directly.

Packed digits are not a persistent hint representation because their packing
depends on the consuming kernel, basis, and native role dimension. Persisting
them would duplicate the semantic source and create a format-specific
conversion boundary.

## Chunk partition

Every group is partitioned into the same positive `W` chunk indices. For group
`g`, define

\[
S_{g,c}=\left\lfloor\frac{cF_g}{W}\right\rfloor,
\qquad
F_{g,c}=S_{g,c+1}-S_{g,c}.
\]

Chunk `c` owns `[S_{g,c},S_{g,c+1})`. The intervals **MUST** be adjacent, and
their union **MUST** be `[0,F_g)`. When `W > F_g`, repeated boundaries produce
empty intervals. The layout still retains all `W` chunk indices.

`ChunkedWitnessCfg` chooses `W`; it is not resolved address geometry.
`WitnessLayout` owns the resolved block intervals and coefficient ranges.

## Physical unit order

The `WitnessLayout::units()` sequence **MUST** use this nesting:

```text
for chunk c in 0..W:
    for relation-order position p in 0..G:
        emit unit (group = g(p), chunk = c)
```

The physical unit index is

\[
u(c,p)=cG+p.
\]

Groups **MUST NOT** be sorted by dimension. A group's units are generally
strided by `G` in the unit table; callers that need all units of one group
**MUST** select them by group index and preserve increasing chunk index.

For `G = 1` or `W = 1`, chunk-major and group-major nesting coincide. The
dominant single-final-group case therefore incurs no ordering penalty.

## Exact coefficient ranges

All `WitnessUnitLayout` ranges are ranges of field coefficients, not ranges of
implicit ring slots.

For group `g` and chunk `c`, define:

\[
\begin{aligned}
L_Z(g) &= M_g\delta_{Z,g}\delta_{F,g}a_g,\\
L_E(g,c) &= H_gF_{g,c}q_{D,g}\delta_{D,g}d_g
          = H_gF_{g,c}\delta_{D,g}a_g,\\
L_T(g,c) &= H_gF_{g,c}n_{A,g}q_{B,g}\delta_{B,g}b_g
          = H_gF_{g,c}n_{A,g}\delta_{B,g}a_g.
\end{aligned}
\]

Starting from `cursor = 0`, each physical unit receives adjacent ranges:

```text
z_range = cursor .. cursor + L_Z(g)
e_range = z_range.end .. z_range.end + L_E(g,c)
t_range = e_range.end .. e_range.end + L_T(g,c)
cursor  = t_range.end
```

Every chunk contains a complete copy of group `g`'s Z segment. E and T contain
only the source blocks owned by that chunk. An empty chunk therefore has an
empty E range and an empty T range, but its Z range remains nonempty.

### Z order

Z uses

```text
[position][inner digit][fold digit][A coefficient]
```

For `0 <= k < a_g`,

\[
j_Z=Z_{g,c}+((p\delta_{Z,g}+z)\delta_{F,g}+f)a_g+k.
\]

The fold digit and then the native coefficient are contiguous. No A padding
follows a Z value.

### E order

E uses

```text
[claim][local block][D subcolumn][opening digit][D coefficient]
```

Let `ell = global_block - S_g,c`. For `0 <= k < d_g`,

\[
j_E=E_{g,c}+
(((hF_{g,c}+\ell)q_{D,g}+s)\delta_{D,g}+o)d_g+k.
\]

The valid subcolumns are exactly `0 <= s < q_D,g`. No other subcolumns exist.

### T order

T uses

```text
[claim][local block][A row][B subcolumn][outer digit][B coefficient]
```

For `0 <= k < b_g`,

\[
j_T=T_{g,c}+
((((hF_{g,c}+\ell)n_{A,g}+i)q_{B,g}+s)
\delta_{B,g}+o)b_g+k.
\]

The valid subcolumns are exactly `0 <= s < q_B,g`. No other subcolumns exist.

## Native quotient tail

After the final Z/E/T unit, the witness contains one shared R tail. Relation
rows use the canonical order

```text
[consistency_g | A_g | B_g] for each relation-order group, with B rows in
slice-major then physical-row order
[shared D rows]
```

For row `rho`, allocate an adjacent range of length `delta_R * r_rho`. Within
that row the order is

```text
[quotient digit][native row coefficient]
```

and

\[
j_R=R_\rho+o r_\rho+k.
\]

The exact live R length is

\[
L_R=\delta_R\sum_\rho r_\rho.
\]

`WitnessLayout` **MUST** retain enough checked row-range information to answer
an R coefficient address without assuming a uniform row stride.
`relation_rows * delta_R` is not a valid storage length for mixed native rows.

## Complete live length and zero suffix

The exact live coefficient length is

\[
L=\sum_{c=0}^{W-1}\sum_{p=0}^{G-1}
(L_Z(g(p))+L_E(g(p),c)+L_T(g(p),c))+L_R.
\]

Let `d_next` be the successor commitment's A ring dimension. Define

\[
N_{next}=\left\lceil L/d_{next}\right\rceil,
\qquad
N_{cube}=2^{\lceil\log_2 N_{next}\rceil},
\qquad
P=N_{cube}d_{next}.
\]

The committed and Stage-2 multilinear source is the coefficient vector of
length `P` formed by the exact live prefix `[0,L)` followed by one zero suffix
`[L,P)`. This suffix simultaneously supplies any partial successor ring and
the Boolean-domain padding. There **MUST NOT** be zero gaps inside `[0,L)`.

The prover **SHOULD** avoid materializing the suffix when a downstream kernel
accepts an exact live prefix plus an implicit-zero domain. Serialization and
commitment semantics are nevertheless defined by the length-`P` vector.

## Common relation coefficient block

Let `D_all` contain every `a_g`, `b_g`, `d_g`, and every quotient row dimension
`r_rho`. Define

\[
m=\min D_{all}.
\]

Because all supported dimensions are powers of two, `m` divides every member
of `D_all`. Each Z, E, T, and R segment length is therefore a multiple of `m`.
Since the first segment starts at zero and every next segment starts at the
previous end, every live segment base is `m`-aligned without sorting or
padding.

`RelationAddressGeometry` **MUST** own:

- `m`;
- exact live coefficient length `L`;
- committed coefficient length `P`;
- the checked Boolean domain over `P`; and
- per-role lane counts `d_role / m`.

It **MUST NOT** own a batch carrier dimension or derive `L` by multiplying a
ring-slot count by `max_g a_g`.

For any live coefficient address `j`, write

\[
j=m\lambda+\kappa,
\qquad 0\le\kappa<m.
\]

At relation point `(r_low,r_high)`, its equality weight factorizes as

\[
\operatorname{eq}(j,r)
=\operatorname{eq}(\kappa,r_{low})
 \operatorname{eq}(\lambda,r_{high}).
\]

A native role ring of dimension `r` occupies `r/m` adjacent high-coordinate
lanes. This is the only mixed-dimension difference in flat relation addressing.
No transpose, carry split, dimension sort, or dense fallback is required.

## Relation algebra is unchanged

Compact storage changes addresses, not equations. For each native relation row
of dimension `r_rho`, the prover and verifier still enforce

\[
M_\rho(\widehat z,\widehat e,\widehat t)
=y_\rho+(X^{r_\rho}+1)\widehat r_\rho.
\]

E and T contributions **MUST** be recomposed at their A semantics with

\[
\sum_s\sum_o X^{s d_g}\beta_{D,g}^o\widehat e_{s,o}
\quad\text{and}\quad
\sum_s\sum_o X^{s b_g}\beta_{B,g}^o\widehat t_{s,o}.
\]

The consistency and A rows consume A-native values. B rows consume B-native T
digits. Shared D rows consume D-native E digits. Alpha powers for projection
belong to the subcolumn tensor factor `alpha^(s r)`; gadget powers belong to
the digit factor `beta^o`. Moving from a larger to a smaller native ring shifts
which equality coordinates and alpha powers are explicit, but does not create
a different relation.

## Setup and evaluation-trace columns

Setup B and D columns **MUST** use the same semantic axes as witness storage:

\[
\begin{aligned}
\operatorname{col}_D(h,b,s,o)
  &=(((hF_g+b)q_{D,g}+s)\delta_{D,g}+o),\\
\operatorname{col}_B(h,b,i,s,o)
  &=((((hF_g+b)n_{A,g}+i)q_{B,g}+s)\delta_{B,g}+o).
\end{aligned}
\]

Chunks map local block `ell` back to global block `S_g,c + ell`; setup
matrices are not copied per chunk. Group offsets use checked prefix sums in
authenticated relation order.

The evaluation trace **MUST** address E with the same E formula. It may fold
contiguous coefficient runs or factor the common block, but it **MUST NOT**
materialize padded projected planes. For `q_D,g = 1`, it **MUST** use the
contiguous identity case without projection powers or multiplication by one.

## Single sources of truth

The implementation **MUST** evolve the existing authorities:

- `WitnessLayout` owns chunk-major units and all exact coefficient ranges;
- `WitnessUnitLayout` owns Z/E/T address validation;
- `RelationRhsLayout::row_ring_dims()` owns quotient row dimensions and order;
- `RelationAddressGeometry` owns the common block and complete flat domain;
- the existing projected-decomposition kernel owns split-before-decompose; and
- `AkitaCommitmentHint` owns persistent A-native hint rows.

Code **MUST NOT** introduce a second compact layout type, a mixed-only witness
type, a uniform wrapper, a `_for_level` forwarding helper, or duplicated length
arithmetic in planner, prover, setup, and verifier code. Those consumers call
the canonical authorities directly.

`RelationRangeImagePlan` **MUST NOT** encode each group's units as one
contiguous `Range<usize>`. It **MAY** store the authenticated group order and
derive unit index `cG+p`, or store validated per-group unit indices. Either
representation **MUST** preserve chunk-major physical order.

## Validation

At a trusted schedule or verifier boundary, construction **MUST** reject:

- zero, non-power-of-two, or non-dividing role dimensions;
- an empty group, chunk, segment, or quotient depth;
- chunk count above the repository cap or above any group's live block count;
- units not in exact chunk-major/authenticated-group order;
- chunk block ranges that overlap, gap, reorder, or fail to cover `[0,F_g)`;
- a Z/E/T range whose length differs from the formulas above;
- an R row range whose length differs from `delta_R * r_rho`;
- any internal range gap or overlap;
- a live length, padding length, or address computation that overflows;
- a committed domain shorter than `L`, not divisible by `d_next`, or not a
  power-of-two number of successor rings; and
- any prover-supplied index outside its checked semantic range.

Verifier-reachable code **MUST** return `AkitaError` or `SerializationError`.
It **MUST NOT** panic, assert, unwrap, index unchecked, or allocate from an
unvalidated proof-controlled size.

## Performance requirements

The compact representation is an optimization contract, not only a byte
contract.

- Witness emission **MUST** copy native contiguous coefficient runs directly.
- No Z/E/T emitter may zero-fill per-role or per-group padding.
- No relation, setup, trace, or commitment consumer may transpose an entire
  projected segment.
- Mixed dimensions **SHOULD** use common-block tensors and batched affine
  recurrences rather than one scalar equality expansion per coefficient.
- The `q = 1` specialization **MUST** avoid projection allocation and
  multiplication by one inside the shared operation.
- The single-group path **MUST** retain contiguous sequential units and must not
  pay for group lookup or gather buffers in its hot loop.
- Recursive setup offloading **MUST** preserve chunk ownership: one machine's
  complete multi-group chunk is contiguous, while the setup prefix remains a
  shared semantic matrix rather than a chunk copy.

Performance acceptance is relative to PR 337's merge base `a0b436dc5`, not to
an intermediate commit on the branch. The benchmark matrix **MUST** include:

- uniform single-group single-chunk;
- uniform single-group multi-chunk;
- mixed single-group;
- mixed multi-group single-chunk;
- mixed multi-group multi-chunk; and
- recursive setup offloading with multi-group multi-chunk geometry.

Uniform single-group performance **MUST NOT** materially regress. Any mixed
regression **MUST** be explained by measured arithmetic or memory work, not by
layout conversion, padding, or a broad dispatch boundary.

## Required tests

Independent address-oracle tests **MUST** enumerate Z, E, T, and R coefficients
from the formulas in this document and compare them with `WitnessLayout`.
Coverage **MUST** cross:

- `G in {1, 2, 3}`;
- `W in {1, 2, 4, 8}` where block counts permit;
- uniform and mixed A/B/D dimensions;
- final A smaller than a precommitted A;
- unequal group block counts and claim counts;
- `q_B = 1`, `q_D = 1`, and both greater than one;
- native quotient rows of at least two dimensions;
- exact successor alignment and nontrivial Boolean suffix padding; and
- malformed ranges, dimensions, chunk partitions, and point lengths.

Algebra tests **MUST** compare native split/decompose/recompose against a dense
A-ring oracle for E and T. Relation, setup contribution, evaluation trace, and
Stage 2 tests **MUST** compare compact mixed layouts with dense coefficient
oracles. Serialization tests **MUST** pin the A-native hint format.

An implementation slice is complete only after formatting, repository
guardrails, all three release Clippy feature graphs, the CI Nextest target set,
and the benchmark matrix pass.

## Required deletion surface

The cutover **MUST** delete or evolve away:

- batch `max A` witness carrier computation;
- `carrier_ring_dimension` accessors and fields in relation geometry;
- carrier-scaled `total_len` and outgoing-source-length formulas;
- carrier subcolumn counts and live-versus-padding subcolumn branches;
- group-major unit construction and contiguous per-group `unit_range` metadata;
- uniform-stride R indexing;
- padded projected plane construction or zero filling;
- recomposition and recursive hint wrappers; and
- stale docs or tests that present any of those as current behavior.

## References

- [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119)
- [RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)
- [`archive/2026-Q3/digit-innermost-layout.md`](archive/2026-Q3/digit-innermost-layout.md)
- The deleted mixed-ring experiment and its planner evidence are superseded by this layout and the flat public matrix spec.
- [`archive/2026-Q3/distributed-setup-offloading.md`](archive/2026-Q3/distributed-setup-offloading.md)
- [`relation-range-image-sumcheck.md`](archive/2026-Q3/relation-range-image-sumcheck.md)
- [`archive/2026-Q3/setup-layout-repack.md`](archive/2026-Q3/setup-layout-repack.md)
