# Spec: Group-local opening points and reusable verifier preparation

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-22 |
| Status        | active |
| PR            | #322 |
| Supersedes    | Point-model portions of [`shared-opening-claims-api.md`](shared-opening-claims-api.md), [`multi-group-batching.md`](multi-group-batching.md), and the shared-point witness carry in [`batched-stage3-setup-opening.md`](batched-stage3-setup-opening.md) |
| Superseded-by | |
| Book-chapter  | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita's multi-group root protocol already prepares and retains one ring opening
point per group, but its public claims model reconstructs those points by routing
prefixes or suffixes out of one shared point. This restriction is not required by
the proof relation and currently provides no verifier multiplication saving.

This spec makes one complete opening point per polynomial group the canonical
model. The verifier uses one preparation pipeline for all groups, with internal
reuse when two groups happen to require identical material. Point equality or
nesting is an optimization opportunity, not a protocol precondition and not
caller-supplied routing metadata. The existing layout and schedule model remains
the single source of structural truth.

Independent group-local points also remove the reason for Stage 3 to carry the
recursive-witness claim through the setup-product sumcheck. Under recursive
setup offloading, the successor opens the witness at the point already produced
by Stage 2 and opens the setup prefix at the independent point produced by a
setup-only Stage 3. This removes the Stage 3 witness reduction, reduces prover
and verifier work, and shortens the Stage 3 proof whenever the padded setup
prefix domain is smaller than the padded recursive-witness domain.

The first implementation PR also changes full-table multilinear Lagrange and
equality-table expansion from two multiplications per parent to one
multiplication and one subtraction. That independent improvement removes almost
half of the opening point preparation multiplications in the representative
production schedule. The final implementation MUST also consolidate the
duplicated serial builders: `akita-algebra` owns one parent-split primitive and
one full-table serial traversal, and opening-point preparation calls that
implementation after applying its verifier sequence bound. Optimizing several
copies independently is an interim arithmetic repair, not the approved
single-source-of-truth design. The claims, transcript, and descriptor cutover is
larger follow-up work governed by the acceptance criteria below.

## Status and decision boundary

This record distinguishes current code from the approved target so that readers
do not mistake design work for shipped behavior.

| Concern | State before this record | Approved target |
|---------|--------------------------|-----------------|
| Public point input | One shared point plus `PointVariableSelection` per group | One complete point stored by each group |
| Group preparation | One `prepare_opening_point` call per group | The same canonical call per group, backed by internal exact reuse |
| Nested points | Prefix/suffix routing is representable | Arbitrary points are valid; nesting is optional optimization metadata derived internally |
| Layout and schedules | Ordered `(num_vars, num_polys)` groups | Unchanged |
| Recursive Stage 3 | Fused setup-product and witness-carry sumcheck so both successor groups use projections of one point | Setup-product sumcheck only; carry the Stage 2 witness claim and point unchanged |
| Lagrange/equality full-table expansion | Three independent serial recurrences, each using two multiplications per parent; a separate parallel recurrence is already optimal | One canonical parent split and one canonical full-table serial traversal, using one multiplication and one subtraction per parent |

The analysis and concrete opening-preparation counts in this record were checked
against `main` at commit `a1c8782e9b2f3d4fa35e78918c64e4c3c0a6d94d`.
The Stage 3 analysis was checked against the complete diff and implementation at
open PR #320 head `6652c08a21cb418a845adf65891b276dc1e81816`; this
specification remains based directly on `main`, not on that PR.

## Terminology

- A **group-local point** is the complete ordered point at which every
  polynomial in one commitment group is opened.
- **Opening preparation** converts a field point into its padded point, packed
  inner factor, position weights, live block weights, and ring-multiplier view.
- **Exact reuse** shares a prepared factor when its complete semantic cache key
  is equal.
- **Nested reuse** constructs a larger tensor-product factor from a smaller
  prefix or suffix factor plus additional coordinates.
- **Recursive setup offloading** proves a setup-prefix opening in Stage 3 and
  carries that committed prefix into a successor fold instead of scanning the
  corresponding setup contribution there.
- **Witness-claim reduction** is PR #320's Stage 3 term that reopens the
  recursive witness from its Stage 2 point at a projection of the Stage 3
  challenge. It is required by shared-point routing, not by the setup-product
  relation.

## Intent

### Goal

Represent every polynomial group by its own opening point and prepare all group
points through one canonical pipeline, while preserving transcript soundness,
layout-driven scheduling, and efficient internal reuse.

### Invariants

1. Each `PolynomialGroupClaims` MUST contain one non-empty evaluation list, one
   commitment, and one complete opening point.
2. Every polynomial in a group MUST have the same arity and MUST be claimed at
   that group's point. Different groups MAY have arbitrary point values and
   arbitrary supported arities.
3. `OpeningClaimsLayout` MUST derive each ordered
   `PolynomialGroupLayout(num_vars, num_polys)` from the group-local point and
   evaluation count. It MUST NOT store a second point-shape representation.
4. Setup capacity and schedule selection MUST continue to use the maximum
   supported group arity and the ordered group layouts, not the sum of point
   lengths.
5. Claims MUST NOT contain parallel `points` and `groups` vectors. A point and
   its evaluations MUST be owned by the same group object.
6. The target model MUST remove `PointVariableSelection`,
   `OpeningClaims::group_point_vars`, and constructors whose purpose is custom
   routing from a shared point.
7. Prover and verifier MUST use the same checked geometry split
   `[inner | position | block]` and the same basis-weight primitives.
8. Caches MUST be derived implementation state. Cache identity MUST include all
   data that affects a factor: basis, ring dimension, coordinate values, factor
   role, and relevant position/block geometry.
9. Transcript and descriptor commitments MUST bind group order, each group's
   arity, polynomial count, point, commitment, and evaluation claims before the
   batching challenge is sampled.
10. Padded zeros, prepared factors, and cache hits MUST NOT be absorbed as new
    statement data. They are deterministic derivatives of the group-local
    points and selected schedule.
11. Groups at different points MUST NOT be collapsed into a synthetic common
    point or a single scalar opening relation. Root batching combines their
    separate relations using transcript-derived coefficients.
12. For a fixed ordered layout, this cutover MUST NOT change setup dimensions,
    schedule eligibility, witness partition, or SIS pricing. The planner MUST
    reprice the intentionally smaller Stage 3 proof before comparing schedules.
13. The verifier MUST NOT materialize a full table of size `2^num_vars` merely
    to discover or exploit nested points. It prepares the inner, position, and
    live-block factors required by the selected geometry.
14. In recursive setup-offload mode, the successor witness group MUST use the
    Stage 2 point and evaluation unchanged. The setup-prefix group MUST use the
    independently sampled Stage 3 setup point and its verified evaluation.
15. Stage 3 MUST prove only the setup-product claim. It MUST NOT scan, validate,
    contract, fold, rerandomize, or serialize a second opening of the recursive
    witness.
16. `SetupSumcheckProof` MUST contain exactly the setup-product claim, the
    setup-prefix evaluation, and the setup-only sumcheck. The Stage 2 proof
    remains the single source of the recursive-witness evaluation.
17. All verifier-reachable point validation and cache lookups MUST satisfy the
    repository no-panic contract.
18. `akita-algebra` MUST own the canonical Lagrange/equality parent split and
    the canonical full-table serial traversal. `lagrange_weights` and
    `EqPolynomial` MUST NOT maintain sibling full-table expansion loops.
19. Every serial, cached, and parallel Lagrange/equality expansion MUST derive
    each pair of children from the same arithmetic invariant:
    `right = value * point` and `left = value - right`. No implementation MAY
    restore independent multiplication by `1 - point`.
20. The all-layers cached builder MAY retain a distinct storage traversal
    because its output contract differs from a full table, but it MUST call the
    canonical parent-split primitive rather than restating the recurrence.

### Non-Goals

- Opening different polynomials within one commitment group at different
  points. Such claims belong in separate groups.
- Extending dense, extension-opening-reduction, root-terminal, or recursive
  suffix protocols to more groups where their schedules currently require one.
- Changing the setup-product relation, offload eligibility policy, witness
  partition, or generated commitment geometry. Exact proof-byte repricing is
  in scope.
- Adding a persistent or cross-proof cache.
- Shipping a general prefix/suffix tensor DAG in the first group-local-point
  implementation.
- Preserving serialized API compatibility. Akita makes no backward-
  compatibility guarantee.

## Cost model

### What is prepared per group

For ring dimension `D`, position count `M`, and live block count `B`, let
`alpha = log2(D)`, `p = log2(M)`, and
`q = log2(next_power_of_two(B))`. The canonical point order is:

```text
[ alpha inner coordinates | p position coordinates | q block coordinates ]
```

`prepare_opening_point` produces:

- `D` inner basis weights packed as one ring element;
- `M` position weights;
- the first `B` block weights from a Boolean domain of size `2^q`;
- a ring-multiplier representation of the outer weights; and
- the checked, padded point used by later evaluation-trace preparation.

These values are needed whether setup contributions are evaluated locally or
offloaded. Offloading can remove the verifier's setup-contribution scan or
evaluation, but it does not remove the relation opening, ring-switch, or
evaluation-trace inputs derived from each group point.

### Lagrange expansion

#### Current defect

The code before this record has four paths for the same mathematical
operation, and they have drifted:

- `akita_types::layout::opening_point::lagrange_weights` owns a serial
  full-table expansion loop;
- `EqPolynomial::evals_serial` owns a second serial full-table loop;
- `EqPolynomial::evals_cached_with_scaling` owns a third serial recurrence
  while retaining every intermediate layer; and
- `EqPolynomial::evals_parallel` owns a fourth recurrence.

The first three compute both children by multiplication. The parallel path
already computes the right child once and obtains the left child by
subtraction. This is therefore not a missing mathematical identity. It is
implementation drift caused by duplicating one arithmetic primitive across
several APIs.

The drift has concrete production consequences. `lagrange_weights` bypasses
`EqPolynomial`, so it never receives the parallel path's optimal recurrence.
`EqPolynomial::evals_with_scaling` selects its parallel implementation only
above its variable-count threshold, while `evals_prefix` normally builds two
smaller split tables. Those split tables consequently use the inefficient
serial recurrence in the relevant schedules.

Value-equality tests cannot detect this defect because all four paths return
the same field elements. The compiler also cannot repair it: `FieldCore`
multiplication and subtraction are opaque trait operations, and Rust does not
encode the field distributive law needed to replace `value * (1 - point)` with
`value - value * point`. The implementation must state and test the cost
invariant explicitly.

This defect is computational, not cryptographic. It does not change the table,
proof, transcript, or verifier decision. It wastes one field multiplication
per expanded parent and makes later performance fixes prone to the same drift.

#### Optimal recurrence

Before this record, a full Lagrange table over `s` variables computed
both children independently:

```text
left  = value * (1 - p)
right = value * p
```

Its multiplication count is:

```text
C_full_old(s) = 2 * (2^s - 1)
```

The equivalent recurrence approved here is:

```text
right = value * p
left  = value - right
```

It preserves table order and values, and changes the count to:

```text
C_full_new(s) = 2^s - 1
```

The new recurrence performs `2^s - 1` subtractions in place of the removed
multiplications. Field subtraction is substantially cheaper than field
multiplication in the base and extension fields used here. This count is
independent of group-local points, point nesting, or cross-group reuse.

The live block prefix uses `EqPolynomial::evals_prefix`. Before this record, its
normal serial split path required:

```text
C_block(q, B) = 2(2^floor(q/2) - 1)
              + 2(2^ceil(q/2) - 1)
              + B
```

The prefix routine evaluates two split full tables and multiplies their entries
for the `B` requested outputs. Applying the same recurrence rewrite to its
serial full-table builder changes the count to:

```text
C_block_new(q, B) = (2^floor(q/2) - 1)
                  + (2^ceil(q/2) - 1)
                  + B
```

### Representative production schedule

The generated `fp128_d64_onehot_recursive_multi_chunk_w8r2` entry with a
32-variable final group has:

| Group | Arity | `D` | `M` | `B` | Old full/prefix multiplications | After full-table rewrite |
|-------|------:|----:|----:|----:|--------------------------------:|-------------------------:|
| Final | 32 | 64 | 65,536 | 1,024 | 132,344 | 66,684 |
| Precommitted 0 | 16 | 64 | 32 | 32 | 240 | 136 |
| Precommitted 1 | 16 | 64 | 32 | 32 | 240 | 136 |
| **Total** | | | | | **132,824** | **66,956** |

Thus the generic recurrence improvement removes 65,868 base-field
multiplications for this root before any cross-group caching.

If both 16-variable precommitted points are nested in the final point, the
absolute upper bound from reusing all of their old preparation work is only 480
multiplications, about 0.36% of the old total. That upper bound overstates what
whole-point reuse can achieve because a 16-coordinate prefix crosses different
`inner | position | block` boundaries in the two geometries.

After the recurrence rewrite, the corresponding upper bound is 272
multiplications, about 0.41% of the new total.

The current short Stage 2 evaluation-trace preparation for the same three
groups costs 353 extension-field multiplications in the analyzed path; sharing
the two equal precommitted preparations can save at most 146. This is separate
from the much larger generic Lagrange-table saving, and later direct equality-
MLE evaluation can reduce the absolute importance of that trace reuse.

### Stage 3 witness-reduction elimination

PR #320's recursive Stage 3 combines two terms over a padded common cube:

```text
setup product at the Stage 3 setup point
+ eta * witness carry from the Stage 2 point to the Stage 3 witness point
```

The second term exists to make the successor's setup-prefix point and witness
point suffix projections of one shared challenge. It is not needed to establish
the setup-product claim. Once the successor accepts arbitrary group-local
points, it can carry:

```text
witness group:      (stage2_point, stage2_next_w_eval)
setup-prefix group: (stage3_setup_point, stage3_setup_prefix_eval)
```

Stage 3 then runs only the setup-product sumcheck. On the prover, this removes:

- the balanced-digit validation performed specifically for the witness carry;
- both full passes over the compact recursive witness in
  `WitnessClaimReductionTerm`;
- witness equality-table construction and folding;
- witness round-polynomial accumulation and batching with `eta`; and
- computation and serialization of `W(stage3_witness_point)`.

On the verifier, this removes the witness lift scale, the
`eq(stage2_point, stage3_witness_point)` evaluation, the `eta` batching
challenge, the Stage 3 witness final-relation term, and the second witness
evaluation absorption. Stage 2, the setup-product term, the setup-prefix
evaluation, and the successor's ordinary per-group opening verification remain.

Let:

```text
w = log2(D_w) + log2(next_power_of_two(witness_field_len / D_w))
s = log2(D_setup) + log2(next_power_of_two(setup_prefix_field_len / D_setup))
c = serialized challenge-field bytes
```

PR #320 prices the fused Stage 3 payload as:

```text
bytes_fused = 3c + 2c * max(w, s)
```

The three scalars are the setup-product claim, setup-prefix evaluation, and
rerandomized witness evaluation. A degree-two compressed sumcheck contributes
two field elements per round. The setup-only target costs:

```text
bytes_setup_only = 2c + 2c * s
saving           = c + 2c * max(0, w - s)
```

For a 128-bit challenge field, the saving is 16 bytes plus 32 bytes for every
witness round beyond the setup domain. A naturally smaller setup prefix reduces
rounds only when its padded power-of-two domain is also smaller. If `w <= s`,
the proof still saves one field element and removes all witness-term prover and
verifier work, but the sumcheck round count does not decrease.

The first 32-variable production entry in PR #320's generated
`fp128_d64_onehot_recursive_multi_chunk_w8r2` table makes the difference
concrete:

| Offloaded edge | Live witness ring elements | Padded witness rounds `w` | Padded setup field length | Setup rounds `s` | Fused bytes | Setup-only bytes | Saving |
|----------------|---------------------------:|--------------------------:|--------------------------:|-----------------:|------------:|-----------------:|-------:|
| Root to recursive fold 0 | 2,647,068 | 28 | `2^25` | 25 | 944 | 832 | 112 |
| Recursive fold 0 to 1 | 722,408 | 26 | `2^25` | 25 | 880 | 832 | 48 |
| **Total** | | | | | **1,824** | **1,664** | **160** |

The removed prover witness passes traverse up to 169,412,352 and 46,234,112
compact digits on these two edges, respectively. These are structural counts
from the generated geometry at PR #320 head `6652c08a`, not wall-clock benchmark
results; the implementation must measure the realized speedup.

### Work that point nesting does not remove

Point nesting does not reduce:

- challenge sampling or claim-coefficient generation;
- `eq_tau1` and relation-matrix row work;
- setup-contribution planning, local scans, or offloaded contribution checks;
- commitment-row assembly and ring-switch relation evaluation; or
- contractions whose dimensions are fixed by the selected schedule.

The verifier therefore MUST NOT require nested points to claim a broad
`2^m` verifier speedup. At most, nesting reuses already identified tensor
factors; it does not shrink the relation or setup geometry.

This limitation concerns tensor-factor reuse between related points. It does
not apply to the Stage 3 witness reduction above: that entire reduction becomes
unnecessary because independent group-local points remove its protocol purpose.

## Design

### Canonical claims model

The target public model is logically:

```rust,ignore
pub struct PolynomialGroupClaims<'a, F, C> {
    point: &'a [F],
    evaluations: Vec<F>,
    commitment: C,
}

pub struct OpeningClaims<'a, F, C> {
    groups: Vec<PolynomialGroupClaims<'a, F, C>>,
}
```

The concrete implementation MAY use the repository's existing owned or
borrowed point type, but it MUST preserve this ownership relation: there is one
complete point per group and no separate routing layer.

The current recursive suffix construction's setup-prefix and witness points
become the complete points of their respective groups. It MUST NOT rebuild them
as selections over a concatenated ambient point.

`OpeningClaimsLayout` remains the canonical field-free structural model. It is
the ordered list of `PolynomialGroupLayout` values and continues to drive setup,
planner, schedule lookup, and relation layout. Aggregate counts and maximum
arity are derived accessors, not separately serialized fields.

### Descriptor and transcript

The opening descriptor MUST commit to the ordered group count and, for every
group, `(num_vars, num_polys)`, the basis mode, and the existing domain or
protocol separator. The layout digest MUST use these canonical ordered fields;
it MUST NOT retain selection indices or a second scalar "shared arity" field.

The transcript order for a multi-group root MUST be canonical and identical on
the prover and verifier:

1. protocol and descriptor data;
2. ordered group layout;
3. commitments in group order;
4. each complete group-local point in group order;
5. claimed evaluations in group and polynomial order;
6. batching challenges; and
7. proof messages.

Existing transcript helper functions SHOULD be extended directly. The cutover
MUST NOT introduce a second claims-absorption wrapper or retain old routing
absorption alongside the new path.

For recursive setup offloading, Stage 2 continues to absorb its
`next_w_eval`. Setup-only Stage 3 then absorbs its setup-product claim, samples
only its own sumcheck challenges, and checks the setup-prefix evaluation at the
resulting point. It MUST NOT sample `CHALLENGE_SUMCHECK_BATCH` for an absent
witness term or absorb `ABSORB_STAGE3_NEXT_W_EVAL`. The successor transcript
binds the unchanged Stage 2 witness point and the Stage 3 setup point when it
absorbs the two ordered group-local claims.

### One preparation pipeline

Every supported group passes through the same checked
`prepare_opening_point(point, basis, M, B, alpha)` function. A one-group fold is
the `G = 1` case, not a sibling preparation algorithm. Extension-opening
reduction MAY retain its mathematically distinct point-conversion boundary, but
must call the same preparation primitive after conversion.

The verifier preparation owner SHOULD retain prepared objects by shared
ownership where several later consumers need them. It SHOULD avoid cloning the
entire `RingOpeningPoint` merely to construct a base-field
`RingMultiplierOpeningPoint`; representation sharing is preferable when the
type cutover makes that possible.

### Canonical Lagrange expansion ownership

`akita-algebra` MUST own one inlinable parent-split arithmetic primitive with
the following semantic contract:

```text
split(value, point) = (value - value * point, value * point)
```

The implementation MUST compute `value * point` exactly once. It MUST return or
write the left child before the right child in the repository's existing
little-endian table order.

`akita-algebra` MUST also own one serial full-table traversal built from that
primitive. `EqPolynomial::evals_serial` uses this traversal directly.
Opening-point preparation applies `basis_weight_len` first to enforce the
verifier sequence bound, then calls the same traversal. That boundary check is
meaningful policy and MAY remain in `akita-types`; the expansion loop may not.
The implementation SHOULD remove `lagrange_weights` if callers can use the
canonical function without losing the sequence-bound contract. If a named
opening-point boundary remains, it MUST contain validation and delegation only,
not a second recurrence or a compatibility-only alias.

`EqPolynomial::evals_cached_with_scaling` has a genuinely different output
contract because it retains all layers. It MAY keep its layer-allocation and
layer-order logic, but each parent MUST be expanded by the canonical split
primitive. The parallel traversal MUST use that primitive as well unless a
benchmark demonstrates that abstraction prevents inlining or materially
regresses the hot loop; any specialized parallel spelling must still be pinned
to the same one-multiplication operation-count test.

This ownership boundary is part of the implementation, not optional cleanup.
After the cutover, adding another Lagrange or equality-table builder instead of
extending the canonical primitive is non-conforming.

### Exact reuse

The first group-local-point implementation MUST reuse exact duplicate factors
within one proof. Reuse belongs behind the preparation pipeline and MUST be
invisible to callers and transcripts.

The implementation SHOULD cache factor-level outputs rather than only complete
prepared points. A factor cache key contains:

```text
(basis, ring dimension, factor role, coordinates, relevant geometry)
```

This permits safe reuse of:

- equal inner factors across groups with the same `D`;
- equal position factors across groups with the same `M`;
- equal live-block factors across groups with the same padded block domain and
  live prefix; and
- a complete prepared point when all components match.

The cache is bounded by the number of groups and factors in one proof. It MUST
NOT accept caller-provided cache keys.

### Nested reuse

Nested prefix/suffix reuse is OPTIONAL and MUST be benchmark-gated after exact
reuse lands. If implemented, it constructs a factor by tensoring an already
validated factor with the missing coordinates. It MUST NOT change accepted
claims, transcript bytes, or schedule selection.

Prefix nesting is only directly useful when the reused coordinate interval is
also a complete semantic factor under both groups' geometry. Suffix nesting is
usually less useful because changing arity or `M` shifts the
`inner | position | block` boundaries. Comparing raw point prefixes or suffixes
is therefore insufficient; reuse must be decided from factor keys after the
geometry split.

### Setup offloading

Setup-contribution preparation remains a schedule concern. The same
`RelationRangeImagePlan`, contribution identifiers, and checked setup geometry
are used for local and offloaded evaluation. Group-local points neither add nor
remove contribution materials.

The verifier always prepares the point factors required by the root opening and
evaluation trace. An offloaded contribution result MAY skip the corresponding
local contribution scan exactly as it does today, but MUST NOT skip point
validation or any prepared factor consumed by another relation.

This rule does not retain PR #320's witness-claim reduction. That reduction is
neither setup-contribution material nor group opening preparation; it only moves
the already-proved recursive-witness claim onto the setup challenge. The target
offloaded flow is:

1. Stage 2 proves `W(stage2_point) = stage2_next_w_eval` as in direct mode.
2. Setup-only Stage 3 proves the setup product and returns
   `(stage3_setup_point, stage3_setup_prefix_eval)`.
3. The successor creates two `PolynomialGroupClaims` values directly, one from
   each point/evaluation pair.
4. The ordinary multi-group opening pipeline verifies both commitments at their
   respective points.

`BatchedStage3Geometry::shared_suffix_point`,
`BatchedStage3Geometry::setup_prefix_point_vars`, and
`WitnessClaimReductionTerm` MUST be removed if they have no consumer after this
cutover. The implementation MUST extend the setup-product prover directly; it
MUST NOT preserve the fused driver as a wrapper around a one-term sumcheck.

The target proof shape is:

```rust,ignore
pub struct SetupSumcheckProof<E> {
    pub claim: E,
    pub setup_prefix_eval: E,
    pub sumcheck: SumcheckProof<E>,
}
```

The planner's `stage3_setup_product_bytes` MUST use the setup domain alone and
MUST match actual serialization. It no longer accepts `output_witness_len`
unless another setup-only sizing rule genuinely requires it.

## Evaluation

### Acceptance Criteria

#### This specification PR

- [x] This normative record passes repository documentation guardrails and
  links the superseded point and Stage 3 models from all predecessor specs.
- [x] `lagrange_weights` and the serial `EqPolynomial` table builders use one
  multiplication and one subtraction per expanded parent while preserving
  exact values and order.
- [x] This record identifies the independently optimized loops as an interim
  state and makes canonical serial ownership a required implementation outcome.
- [x] Focused `akita-types` tests pass with default and no-default features.
- [x] The branch has `origin/main` as its merge base and contains no commits
  from another open PR.

#### Group-local claims cutover

- [ ] `PolynomialGroupClaims` stores its complete point and
  `PointVariableSelection` is removed from public and internal APIs.
- [ ] Constructors reject empty groups, inconsistent dimensions, unsupported
  arities, and layout/schedule mismatches without panicking.
- [ ] Existing one-group and multi-group proofs round-trip through the new model.
- [ ] A multi-group end-to-end test opens at least two groups at unrelated point
  values and different supported arities.
- [ ] Prefix-related, suffix-related, equal, and unrelated group points produce
  the same verification result with reuse enabled and disabled.
- [ ] Setup generation and schedule lookup use only
  `OpeningClaimsLayout`; no fake points or duplicate shape types are introduced.

#### Transcript and descriptor cutover

- [ ] Descriptor tests show that changing one group arity, count, or order
  changes the digest.
- [ ] Transcript-smell tests show that changing one group point, commitment, or
  evaluation changes all subsequently sampled batching challenges.
- [ ] Prover and verifier transcript event logs are byte-identical for equal,
  nested, and unrelated points.
- [ ] Old routing fields are removed in one breaking cutover; there is no dual
  encoding or compatibility wrapper.

#### Setup-only Stage 3 cutover

- [ ] Recursive setup offloading carries the Stage 2 witness point and
  `stage2_next_w_eval` unchanged into the successor witness group.
- [ ] Stage 3 produces only the setup-prefix point and evaluation; its prover
  does not receive the compact recursive witness or construct a witness term.
- [ ] `SetupSumcheckProof` contains only `claim`, `setup_prefix_eval`, and the
  setup-only sumcheck, and its serializer and shape descriptors agree.
- [ ] The verifier's Stage 3 final relation contains only the setup-product
  term and rejects tampering with the claim, setup-prefix evaluation, point, or
  round polynomial.
- [ ] Recursive-mode transcript tests confirm removal of the Stage 3 batching
  challenge and second witness-evaluation absorption while preserving exact
  prover/verifier event parity.
- [ ] The old fused geometry, witness reduction, routing helpers, labels, and
  dead tests are deleted rather than retained behind adapters.
- [ ] Planner proof accounting matches actual serialization for `w < s`,
  `w = s`, and `w > s`, including the exact saving
  `c + 2c * max(0, w - s)`.
- [ ] Direct setup mode remains byte-identical and does not create a Stage 3
  proof.
- [ ] An end-to-end recursive-offload test opens the setup prefix and witness at
  unrelated points in the successor two-group fold.

#### Verifier preparation and performance

- [ ] There is exactly one serial full-table Lagrange/equality traversal in
  `akita-algebra`; opening-point preparation validates its sequence bound and
  calls that traversal instead of owning another loop.
- [ ] Serial full-table, cached all-layers, and parallel builders all use one
  canonical parent-split primitive, with no duplicated two-child arithmetic.
- [ ] An operation-count test over `s` variables observes exactly `2^s - 1`
  field multiplications for each full-table serial entry point. Output-parity
  tests cover empty, scaled, base-field, and extension-field tables and preserve
  little-endian order.
- [ ] Exact duplicate factor reuse is covered by hit/miss tests for inner,
  position, block, and complete-point keys.
- [ ] Cache-key negative tests vary basis, `D`, `M`, `B`, coordinate order, and
  live block length independently.
- [ ] A preparation benchmark reports base- and extension-field multiplication
  counts separately for equal, nested, and unrelated points.
- [ ] Arbitrary unrelated points do not add asymptotic work beyond independent
  per-group preparation and do not change group-opening or setup size. Recursive
  Stage 3 proof size changes only by the setup-only formula above.
- [ ] Any nested-factor DAG is merged only if the representative production
  profile shows a material wall-clock benefit after exact reuse and the generic
  Lagrange rewrite.

### Testing Strategy

For this PR, run:

```bash
rtk cargo test -p akita-types layout::opening_point
rtk cargo test -p akita-types --no-default-features layout::opening_point
./scripts/check-doc-guardrails.sh
```

The group-local cutover must additionally run focused prover/verifier transcript
tests, multi-group end-to-end tests, and the repository preflight commands from
`AGENTS.md`. Validation must cover default features and the CI no-default
feature graph. Malformed claims tests must exercise verifier-reachable APIs and
confirm `AkitaError` rather than panics. The Stage 3 cutover must also run the
planner's exact-byte tests and recursive setup-offload end-to-end suite inherited
from PR #320.

### Performance

The recurrence changes preserve allocations and table order while replacing
exactly one multiplication per expanded parent with a subtraction. The first
implementation should confirm output equivalence and exact multiplication
counts in tests; a wall-clock benchmark is not required to accept this
algebraic rewrite. Code review MUST also confirm that the serial full-table loop
has one owner and that the cached builder uses the same parent split.

The later claims cutover must record the preparation-only counts described
above and run the representative end-to-end profile:

```bash
cargo run -p akita-pcs --release --no-default-features \
  --features parallel,profile-onehot-fp128-d64 \
  --example profile
```

For a fixed layout, setup bytes and commitment geometry remain unchanged. Stage
3 proof bytes decrease by the formula above, so generated schedule totals and
possibly the planner-selected suffix MUST be regenerated from the canonical
proof-size helper. Exact reuse should allocate at most one stored factor per
distinct cache key. A nested-factor implementation is optional unless profiling
shows a material improvement beyond exact reuse.

## Alternatives Considered

### Generalize the shared point into a point arena

A point arena plus per-group selection indices can express arbitrary points,
but it preserves the routing abstraction, complicates transcript
canonicalization, and makes callers describe an implementation optimization.
It is rejected in favor of group ownership and derived internal reuse.

### Store a parallel vector of group points

This is mechanically small but permits points and groups to drift out of sync.
It duplicates the association already represented by `PolynomialGroupClaims`
and is rejected by the repository's single-source-of-truth policy.

### Require prefix- or suffix-related points

This retains a protocol restriction for a small and geometry-dependent verifier
optimization. It prevents valid independent openings and does not reduce setup
or relation-matrix costs. It is rejected.

### Implement a general tensor DAG immediately

A DAG can reuse nested factors, but exact reuse and the full-table recurrence
capture the clearer savings first. The production example bounds whole-point
nested reuse at 0.36% of old opening preparation. A DAG is deferred until a
benchmark demonstrates value.

### Optimize every existing expansion loop independently

Changing each current loop to the one-multiplication recurrence gives the right
local arithmetic count, but it preserves the defect that caused the paths to
drift. A later edit could optimize one path, restore the old recurrence in
another, or apply a safety fix inconsistently. This alternative also conflicts
with the repository's one-canonical-function policy. It is accepted only as the
interim patch in this specification PR and rejected as the completed design.

### Combine groups into one synthetic opening

For distinct points there is no single evaluation vector that turns the claims
into the existing one-point relation without changing the protocol. Transcript
batching of separate per-group relations is the correct general construction.

### Keep the fused Stage 3 witness carry

The fused construction cryptographically works with group-local points, but its
witness term no longer serves a protocol need. It scans the prover's compact
witness twice, adds verifier arithmetic and transcript state, serializes a
duplicate witness evaluation, and may extend the sumcheck to the larger witness
domain. Retaining it would preserve dead complexity solely for historical wire
shape, which Akita does not guarantee. It is rejected.

## Security and failure behavior

Binding complete ordered group points before batching challenges prevents a
prover from adapting point routing after seeing coefficients. Layout and point
dimensions are validated before allocation or indexing. Cache equality is an
implementation optimization only: a false miss costs time, while a false hit
would be a soundness bug, so semantic cache keys are complete and tested by
negative cases.

Removing the Stage 3 witness carry does not remove the witness opening proof.
Stage 2 already binds `stage2_next_w_eval` to the recursive witness commitment
at `stage2_point`; the successor verifies that exact claim through its witness
group. Setup-only Stage 3 independently binds the setup-product claim to
`stage3_setup_prefix_eval` at `stage3_setup_point`. Both ordered claims are
absorbed before the successor samples its group-batching coefficient. No
cross-group equality or prefix/suffix relation is assumed for soundness.

Malformed serialized points, excessive group counts, overflowing dimensions,
and schedule mismatches return typed errors at existing verifier boundaries.
No new unchecked indexing, assertion, unbounded allocation, or
attacker-controlled persistent cache is permitted.

## Compatibility

The claims and descriptor cutover is intentionally breaking. Callers must move
from one shared point plus selections to one complete point per group. Because
the descriptor and transcript statement change, proofs produced by the old and
new APIs are not cross-compatible.

The Stage 3 cutover is also a proof-wire and transcript break:
`SetupSumcheckProof.next_w_eval`, the fused witness rounds, and their transcript
events disappear. Setup artifacts, commitment dimensions, and SIS pricing
remain valid, but planner proof totals and generated schedule choices must be
recomputed. The Lagrange recurrence change is purely computational and produces
byte-identical field values.

## Documentation

This active spec is the implementation record until the claims cutover ships.
The point-model sections of `shared-opening-claims-api.md` and
`multi-group-batching.md`, plus the shared-point witness carry in
`batched-stage3-setup-opening.md`, link here as superseded guidance. When
implementation is complete, durable user-facing behavior belongs in
`book/src/usage/commitment-api.md`; verifier preparation and failure behavior
belong in `book/src/how/verification.md`. At that point this spec should be
marked `implemented` and later folded or archived according to
[`PRUNING.md`](PRUNING.md).

## Execution

1. Land this decision record and the interim one-multiplication
   Lagrange/equality recurrence on a branch based directly on `main`.
2. Consolidate the serial full-table builders in `akita-algebra`, route
   opening-point preparation through that owner after its sequence-bound check,
   and make cached and parallel expansion use the canonical parent split.
3. Change claims, prover data, descriptor absorption, and transcript absorption
   in one breaking cutover; remove point selection and all pass-through routing
   APIs.
4. In that same cutover, simplify recursive Stage 3 to the setup-product term,
   carry the Stage 2 witness claim unchanged, and delete fused witness-routing
   machinery.
5. Reprice Stage 3 from the setup domain alone, regenerate affected schedules,
   and add exact serialization tests.
6. Route every group through the canonical preparation pipeline and add bounded
   exact factor reuse.
7. Add unrelated-point end-to-end tests and preparation benchmarks, including a
   recursive setup-offload successor whose two group points are unrelated.
8. Implement nested-factor reuse only if the measured result justifies its
   complexity.
9. Fold the shipped behavior into the Akita Book and update this spec's
   lifecycle fields.

## References

- [`shared-opening-claims-api.md`](shared-opening-claims-api.md)
- [`multi-group-batching.md`](multi-group-batching.md)
- [`batched-stage3-setup-opening.md`](batched-stage3-setup-opening.md)
- [`distributed-setup-offloading.md`](distributed-setup-offloading.md)
- [`setup-offloading-planner.md`](setup-offloading-planner.md)
- [PR #320 at inspected head `6652c08a`](https://github.com/LayerZero-Labs/akita/pull/320)
- [`book/src/how/verification.md`](../book/src/how/verification.md)
- [`book/src/usage/profiling.md`](../book/src/usage/profiling.md)
