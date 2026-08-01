# Spec: Group-local opening points and reusable verifier preparation

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-22 |
| Status        | archived |
| PR            | #322 |
| Supersedes    | Point-model portions of [`shared-opening-claims-api.md`](../../shared-opening-claims-api.md), [`multi-group-batching.md`](../../multi-group-batching.md), and the shared-point witness carry in [`batched-stage3-setup-opening.md`](../../batched-stage3-setup-opening.md) |
| Superseded-by | |
| Book-chapter  | book/src/how/architecture.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Current `main` already carries vectors of prepared ring opening points through
the multi-group prover, verifier, extension-opening reduction (EOR), and ring
relation. The remaining public boundary is older: `OpeningClaims` owns one
ambient point, while each `PolynomialGroupClaims` owns a
`PointVariableSelection` into that point. Root and recursive-suffix code
materialize each selected point into a new `Vec` before entering the otherwise
group-local protocol.

This spec completes that cutover. Each polynomial group owns its complete
opening point. `OpeningClaims` owns only the ordered groups, and
`OpeningClaimsLayout` remains the field-free source of setup and schedule
geometry. Equal, nested, and unrelated points are all ordinary inputs. Any reuse
is derived inside preparation and is never public routing metadata.

Independent group-local points also remove the reason for Stage 3 to carry the
recursive-witness claim through the setup-product sumcheck. Under recursive
setup offloading, the successor opens the witness at the point already produced
by Stage 2 and opens the setup prefix at the independent point produced by a
setup-only Stage 3. This removes the Stage 3 witness reduction, reduces prover
and verifier work, and shortens the Stage 3 proof whenever the padded setup
prefix domain is smaller than the padded recursive-witness domain.

PR #322 also implements the independent arithmetic slice: `akita-algebra` owns
one Lagrange/equality parent split, and opening-point preparation delegates to
the canonical equality-table traversal after enforcing its verifier sequence
bound. Each parent expansion uses one multiplication and one subtraction.
That arithmetic slice is independent of the claims and Stage 3 wire cutovers.

Current `main` supports dense and one-hot multi-group roots, batched multi-group
EOR, mixed per-group and per-role ring dimensions, recursive setup offloading,
and a planner/runtime schedule split. This record preserves all of those
capabilities. It changes point ownership and removes dead Stage 3 witness
routing; it does not roll the protocol back to the narrower July 22 feature
matrix.

## Status and decision boundary

This record distinguishes current code from the approved target so that readers
do not mistake design work for shipped behavior.

| Concern | Current `main` (`af770e129`) | Approved target |
|---------|------------------------------|-----------------|
| Public point input | One ambient point plus `PointVariableSelection` per group | One complete point stored by each group |
| Internal group points | `OpeningClaims::group_point` allocates selected coordinates; prover/verifier then carry per-group vectors | Borrow the point directly from its owning group |
| Multi-group EOR | One batched EOR already accepts a vector of materialized group points | Same protocol, sourced directly from group-owned points |
| Mixed ring dimensions | Per-group `d_a` and per-role `d_a/d_b/d_d` already drive preparation and relations | Unchanged |
| Source versus opening arity | Final-group source arity and maximum opening/EOR arity are distinct | Unchanged; no ambient point is needed to represent the maximum |
| Layout and schedules | Ordered `(num_vars, num_polys)` groups | Unchanged |
| Recursive Stage 3 | Fused setup-product and witness-carry sumcheck so successor claims are projections of one challenge | Setup-product sumcheck only; carry the Stage 2 witness claim and point unchanged |
| Lagrange/equality expansion | Serial paths duplicate a two-multiplication recurrence; the parallel loop is already optimal | One canonical parent split and one canonical serial full-table traversal, implemented on this branch |
| Preparation reuse | No protocol-visible cache | Optional, per-proof, benchmark-gated derived state |

This revision was checked against `main` at `af770e129`, including merged PR
#320 (`9c4f3e645`) and the mixed-D/multi-group composition in PR #331
(`af770e129`). Historical counts from the July 22 draft are retained only as
motivation and are not acceptance baselines.

## Terminology

- A **group-local point** is the complete ordered point at which every
  polynomial in one commitment group is opened.
- **Opening preparation** converts a field point into its padded point, packed
  inner factor, position weights, live block weights, and ring-multiplier view.
- **Exact reuse** is an optional preparation optimization that shares material
  when its complete semantic key is equal.
- **Nested reuse** constructs a larger tensor-product factor from a smaller
  prefix or suffix factor plus additional coordinates.
- **Recursive setup offloading** proves a setup-prefix opening in Stage 3 and
  carries that committed prefix into a successor fold instead of scanning the
  corresponding setup contribution there.
- **Witness-claim reduction** is the Stage 3 term introduced by PR #320 that reopens the
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
8. If a preparation cache is implemented, it MUST be derived per-proof state.
   Its identity MUST include all data that affects a factor: basis, the
   group's A-role ring dimension, coordinate values, factor role, and relevant
   position/block geometry.
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
    The separately versioned Fold-l∞ snap policy is not part of this point-model
    invariant: formula tag `4` binds an Fp32-only `3/4 · t*` floor and a `1/2 · t*`
    floor for other fields, so affected fold digit plans and pricing are
    intentionally regenerated under that independent policy.
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
17. All verifier-reachable point validation and any cache lookups MUST satisfy
    the repository no-panic contract.
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
21. For every recursive edge, let `n` be the predecessor-derived Stage 2 point
    arity and `N` be the successor's scheduled opening capacity. Schedule
    validation MUST accept `n <= N` and reject `n > N`. The raw Stage 2 vector
    of length `n` remains the claim, EOR, and transcript object. Any zero
    extension to `N` is a deterministic derivative permitted only inside
    `prepare_opening_point` for scheduled-width evaluation state.

### Non-Goals

- Opening different polynomials within one commitment group at different
  points. Such claims belong in separate groups.
- Adding new dense, one-hot, EOR, mixed-D, or recursive schedule families. The
  implementations already supported on `main` MUST keep working.
- Implementing immediately-terminal multi-group roots or tiered multi-group
  commitments, which remain separately guarded.
- Changing the setup-product relation, offload eligibility policy, witness
  partition, or generated commitment geometry. Exact proof-byte repricing is
  in scope.
- Adding a persistent or cross-proof cache.
- Shipping a point-factor cache or general prefix/suffix tensor DAG as part of
  the correctness cutover. Both require post-cutover benchmark evidence.
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

### Production measurement contract

The July 22 draft used one generated fp128 D64 W8R2 row to show that the
one-multiplication recurrence dominates any plausible cross-group reuse. Since
then the planner, runtime schedule resolver, generated-table ownership,
root-basis policy, setup-prefix planning, supported ring dimensions, and
multi-group protocol have changed. Those old row counts are not current
acceptance numbers.

The implementation benchmark MUST obtain its schedule through the same runtime
resolution path as production. It MUST report, per group:

| Input | Meaning |
|-------|---------|
| `num_vars` | Group-owned point arity |
| `d_a` | A-role ring dimension used by `prepare_opening_point` |
| `M` | `num_positions_per_block` |
| `B` | `num_live_blocks` |
| basis | Lagrange or monomial |
| field | Base or extension point field |
| expansion multiplications | Count before and after canonicalization |
| preparation wall time | Isolated point-preparation time |

At minimum, profile the current production fp128 multi-group recursive W8R2
case, one mixed-D case from `mixed-ring-dimension-per-level`, and one
extension-field multi-group case. Cache or nesting numbers MUST NOT be quoted
until measured on these current paths.

### Stage 3 witness-reduction elimination

The Stage 3 implementation merged in PR #320 combines two terms over a padded
common cube:

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

Current `stage3_setup_product_bytes` prices the fused Stage 3 payload as:

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

The formula is the invariant; concrete schedule savings are derived outputs.
After changing it, the implementation MUST regenerate schedule tables and
report every planner-selected edge whose suffix or setup-prefix slot changes.
It MUST also report the removed compact-witness scan length and realized Stage
3 wall time for the current production W8R2 profile. Historical July 22 byte
and digit counts are deliberately not carried forward as targets.

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
pub struct PolynomialGroupClaims<'a, F: Clone, C> {
    point: OpeningPoints<'a, F>,
    evaluations: Vec<F>,
    commitment: C,
}

pub struct OpeningClaims<'a, F: Clone, C> {
    groups: Vec<PolynomialGroupClaims<'a, F, C>>,
}
```

The concrete implementation SHOULD reuse the existing owned-or-borrowed
`OpeningPoints` carrier. The required API shape is:

```rust,ignore
impl<'a, F: Clone, C> PolynomialGroupClaims<'a, F, C> {
    pub fn new(
        point: impl Into<OpeningPoints<'a, F>>,
        evaluations: Vec<F>,
        commitment: C,
    ) -> Result<Self, AkitaError>;

    pub fn point(&self) -> &[F];
    pub fn num_vars(&self) -> usize;
    pub fn evaluations(&self) -> &[F];
    pub fn commitment(&self) -> &C;
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    pub fn from_groups(
        groups: Vec<PolynomialGroupClaims<'a, F, C>>,
    ) -> Result<Self, AkitaError>;

    pub fn group_point(&self, group: usize) -> Result<&[F], AkitaError>;
    pub fn max_num_vars(&self) -> usize;
}
```

The batch-level `point()` and `num_vars()` accessors are removed. There is no
replacement ambient point, point arena, parallel point vector, or custom-routing
constructor. `group_point` changes from allocation to borrowing.

The current recursive suffix construction's setup-prefix and witness points
become the complete points of their respective groups. It MUST NOT rebuild them
as selections over a concatenated ambient point.

`OpeningClaimsLayout` remains the canonical field-free structural model. It is
the ordered list of `PolynomialGroupLayout` values and continues to drive setup,
planner, schedule lookup, and relation layout. Aggregate counts and maximum
arity are derived accessors, not separately serialized fields.

The final group continues to define the newly committed source arity.
`AkitaScheduleLookupKey::max_num_vars()` continues to define the maximum
opening/EOR capacity across all groups. Group-local ownership removes only the
ambient-value representation; it does not collapse these distinct structural
quantities.

### Existing implementation seams

Current `main` already has the correct downstream shape:

- `verify_multi_group_root_inner` and its prover counterpart build a
  `group_points` vector and dispatch preparation with each group's `d_a`, `M`,
  and `B`;
- `verify_fold_eor` accepts all group points and returns prepared points in
  group order;
- `PreparedFoldReplay` and `RingRelationInstance` store per-group opening and
  multiplier points;
- grouped EOR uses one batched sumcheck while retaining a distinct equality
  factor for each group point; and
- recursive suffix code is the remaining place that concatenates Stage 2 and
  Stage 3 points and reconstructs routing selections.

The cutover MUST simplify these seams in place. It MUST NOT add a second
group-point carrier beside `PolynomialGroupClaims`, or a second relation
statement beside `RingRelationInstance`.

### Descriptor and transcript

The opening descriptor MUST commit to the ordered group count and, for every
group, `(num_vars, num_polys)`, plus the existing basis/domain/protocol fields
owned by the surrounding descriptor. The layout digest MUST use these canonical
ordered fields; it MUST NOT retain selection indices or a second scalar
"shared arity" field.

The transcript order for a multi-group root MUST be canonical and identical on
the prover and verifier:

1. protocol and descriptor data;
2. ordered group layout;
3. commitments in group order;
4. every coordinate of each complete group-local point in group order;
5. claimed evaluations in group and polynomial order;
6. batching challenges; and
7. proof messages.

Existing transcript helper functions SHOULD be extended directly. The cutover
MUST NOT introduce a second claims-absorption wrapper or retain old routing
absorption alongside the new path.

For recursive setup offloading, Stage 2 continues to absorb
`ABSORB_STAGE2_NEXT_W_EVAL`. Setup-only Stage 3 then absorbs its setup-product
claim, samples only `CHALLENGE_SUMCHECK_ROUND`, and checks the setup-prefix
evaluation at the resulting point. It MUST NOT sample the Stage 3 use of
`CHALLENGE_SUMCHECK_BATCH` or absorb `ABSORB_STAGE3_NEXT_W_EVAL`. Those labels
remain available to unrelated sumchecks that still use them. The successor
transcript binds the unchanged Stage 2 witness point and the Stage 3 setup point
when it absorbs the two ordered group-local claims.

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

### Optional exact reuse

The correctness cutover does not require a cache. Current preparation already
operates once per group, and the dominant proven win is canonicalizing the
full-table recurrence. Exact reuse may be added afterward only if the
post-cutover benchmark shows material wall-clock value.

If implemented, reuse belongs behind the preparation pipeline and MUST be
invisible to callers and transcripts. Factor-level outputs are preferable to
only complete prepared points. A factor cache key contains:

```text
(basis, d_a, factor role, coordinates, relevant geometry)
```

This permits safe reuse of:

- equal inner factors across groups with the same `D`;
- equal position factors across groups with the same `M`;
- equal live-block factors across groups with the same padded block domain and
  live prefix; and
- a complete prepared point when all components match.

The cache is bounded by the number of groups and factors in one proof. It MUST
NOT accept caller-provided cache keys or persist across proofs. A linear scan
over the small group set is preferable to adding a public hashability contract
to field types.

### Nested reuse

Nested prefix/suffix reuse is OPTIONAL and MUST be benchmark-gated after the
canonical expansion path is measured. If implemented, it constructs a factor
by tensoring an already validated factor with the missing coordinates. It MUST
NOT change accepted claims, transcript bytes, or schedule selection.

Prefix nesting is only directly useful when the reused coordinate interval is
also a complete semantic factor under both groups' geometry. Suffix nesting is
usually less useful because changing arity or `M` shifts the
`inner | position | block` boundaries. Comparing raw point prefixes or suffixes
is therefore insufficient; reuse must be decided from factor keys after the
geometry split.

### Setup offloading

Setup-contribution preparation remains a schedule concern. The same
`RelationAddressGeometry`, `SetupContributionPlan`, contribution identifiers,
outgoing-aware spans, and checked setup geometry are used for local and
offloaded evaluation. Group-local points neither add nor remove contribution
materials.

The verifier always prepares the point factors required by the root opening and
evaluation trace. An offloaded contribution result MAY skip the corresponding
local contribution scan exactly as it does today, but MUST NOT skip point
validation or any prepared factor consumed by another relation.

This rule does not retain the witness-claim reduction from PR #320. That reduction is
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

`BatchedStage3Geometry`, `WitnessClaimReductionTerm`, and
`witness_claim_reduction.rs` MUST be removed. Setup Stage 3 has one native
domain, so its sumcheck challenge vector is already the setup-prefix point; no
two-domain projection type remains. The implementation MUST simplify the
setup-product prover directly and MUST NOT preserve the fused driver as a
wrapper around a one-term sumcheck.

The Stage 3 orchestration signatures lose all witness-only inputs:

```text
stage2_next_w_eval
logical_w
live_x_cols
col_bits
ring_bits
eta
```

Stage 2 challenges remain an input only where the setup-contribution plan needs
the outgoing relation point. `Stage3ProveOutput` contains the proof and
setup-prefix point/evaluation only. Verifier Stage 3 returns the setup-prefix
opening only. The enclosing fold output always takes its successor witness
point and evaluation from Stage 2.

The target proof shape is:

```rust,ignore
pub struct SetupSumcheckProof<E> {
    pub claim: E,
    pub setup_prefix_eval: E,
    pub sumcheck: SumcheckProof<E>,
}
```

The canonical sizing API becomes:

```rust,ignore
pub fn stage3_setup_product_bytes(
    challenge_field_bits: u32,
    setup_ring_dimension: usize,
    setup_ring_len: usize,
) -> usize;
```

It MUST match actual serialization and use only the setup domain. Planner and
runtime schedule callers remove `output_witness_len`. Generated schedules and
required setup-prefix slot registries MUST be regenerated from the new totals.
The transparent setup matrix and group-local SIS geometry are unchanged, but the
separately bound Fp32 snap policy may change fold digit plans and A-role pricing;
those affected tables are regenerated. Serialized setup caches containing a
different prefix-slot registry are not assumed compatible.

## Evaluation

### Acceptance Criteria

#### This specification PR

- [x] This normative record passes repository documentation guardrails and
  links the superseded point and Stage 3 models from all predecessor specs.
- [x] Serial, cached, and parallel `EqPolynomial` table builders use one
  canonical parent split with one multiplication and one subtraction per
  expanded parent.
- [x] `lagrange_weights` enforces the verifier sequence bound and delegates to
  the canonical equality-table traversal.
- [x] Focused `akita-types` tests pass with default and no-default features.
- [x] The branch includes `main` through `af770e129` and describes the merged
  PR #320 and #331 implementations rather than their earlier development heads.
- [x] Dense, one-hot, EOR, mixed-D, runtime-schedule, and setup-prefix behavior
  added since the original draft is represented in the target and diff plan.

#### Group-local claims cutover

- [x] `PolynomialGroupClaims` stores its complete point and
  `PointVariableSelection` is removed from public and internal APIs.
- [x] Constructors reject empty groups, inconsistent dimensions, unsupported
  arities, and layout/schedule mismatches without panicking.
- [x] Existing one-group and multi-group proofs round-trip through the new model.
- [x] A multi-group end-to-end test opens at least two groups at unrelated point
  values and different supported arities.
- [x] Dense, one-hot, base-field, extension-field/EOR, uniform-D, mixed-D, and
  recursive-setup multi-group tests retain their current support matrix.
- [x] Prefix-related, suffix-related, equal, and unrelated group points produce
  the same verification result. If reuse exists, results are identical with it
  enabled and disabled.
- [x] Setup generation and schedule lookup use only
  `OpeningClaimsLayout`; no fake points or duplicate shape types are introduced.

#### Transcript and descriptor cutover

- [x] Descriptor tests show that changing one group arity, count, or order
  changes the digest.
- [x] Transcript-smell tests show that changing one group point, commitment, or
  evaluation changes all subsequently sampled batching challenges.
- [x] Prover and verifier transcript event logs are byte-identical for equal,
  nested, and unrelated points.
- [x] Old routing fields are removed in one breaking cutover; there is no dual
  encoding or compatibility wrapper.

#### Setup-only Stage 3 cutover

- [x] Recursive setup offloading carries the Stage 2 witness point and
  `stage2_next_w_eval` unchanged into the successor witness group.
- [x] Stage 3 produces only the setup-prefix point and evaluation; its prover
  does not receive the compact recursive witness or construct a witness term.
- [x] `SetupSumcheckProof` contains only `claim`, `setup_prefix_eval`, and the
  setup-only sumcheck, and its serializer and shape descriptors agree.
- [x] The verifier's Stage 3 final relation contains only the setup-product
  term and rejects tampering with the claim, setup-prefix evaluation, point, or
  round polynomial.
- [x] Recursive-mode transcript tests confirm removal of the Stage 3 batching
  challenge and second witness-evaluation absorption while preserving exact
  prover/verifier event parity.
- [x] The old fused geometry, witness reduction, routing helpers, and dead
  tests are deleted rather than retained behind adapters.
  `ABSORB_STAGE3_NEXT_W_EVAL` is deleted;
  `CHALLENGE_SUMCHECK_BATCH` remains available to unrelated protocols but is no
  longer emitted by Stage 3.
- [x] Planner proof accounting matches actual serialization for `w < s`,
  `w = s`, and `w > s`, including the exact saving
  `c + 2c * max(0, w - s)`.
- [x] Direct setup mode remains byte-identical and does not create a Stage 3
  proof.
- [x] An end-to-end recursive-offload test opens the setup prefix and witness at
  unrelated points in the successor two-group fold.
- [x] Setup cache tests demonstrate the intended policy explicitly: unchanged
  required prefix-slot registries round-trip, while changed registries are
  regenerated rather than silently treated as compatible.

#### Verifier preparation and performance

- [x] There is exactly one serial full-table Lagrange/equality traversal in
  `akita-algebra`; opening-point preparation validates its sequence bound and
  calls that traversal instead of owning another loop.
- [x] Serial full-table, cached all-layers, and parallel builders all use one
  canonical parent-split primitive, with no duplicated two-child arithmetic.
- [x] An operation-count test over `s` variables observes exactly `2^s - 1`
  field multiplications for each full-table serial entry point. Output-parity
  tests cover empty, scaled, base-field, and extension-field tables and preserve
  little-endian order.
- [x] A preparation benchmark reports base- and extension-field multiplication
  counts separately for equal, nested, and unrelated points under uniform-D,
  mixed-D, and EOR profiles.
- [x] Arbitrary unrelated points do not add asymptotic work beyond independent
  per-group preparation and do not change group-opening or setup size. Recursive
  Stage 3 proof size changes only by the setup-only formula above.
- [x] Any exact cache or nested-factor DAG is a separate benchmark-gated change.
  If added, hit/miss and negative-key tests vary basis, `d_a`, `M`, `B`,
  coordinate order, and live block length independently.

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
from merged PR #320, plus the dense, extension-opening, mixed-D, and generated
schedule suites added afterward.

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

For a fixed schedule, setup matrix dimensions and commitment geometry remain
unchanged. Stage 3 proof bytes decrease by the formula above, so generated
schedule totals and possibly the planner-selected suffix and required
setup-prefix registry MUST be regenerated from the canonical proof-size helper.
Any preparation cache or nested-factor implementation remains optional unless
profiling after the correctness cutover shows a material improvement.

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
with the repository's one-canonical-function policy. An earlier revision of
this branch used that interim shape; the canonical implementation replaces it.

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
dimensions are validated before allocation or indexing. If preparation reuse
is implemented, cache equality remains an optimization only: a false miss costs
time, while a false hit would be a soundness bug. Its semantic keys therefore
must be complete and covered by negative tests.

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
events disappear. Commitment dimensions and the transparent setup matrix remain
valid for the group-local cutover. The separately bound Fp32 snap policy can
change fold pricing, so planner proof totals and generated schedule choices must
be recomputed, and serialized setup caches must be regenerated if their
required setup-prefix slot registry changes. The Lagrange recurrence change is
purely computational and produces byte-identical field values.

## Documentation

This active spec is the implementation record until the claims cutover ships.
The point-model sections of `shared-opening-claims-api.md` and
`multi-group-batching.md`, plus the shared-point witness carry in
`batched-stage3-setup-opening.md`, link here as superseded guidance. When
implementation is complete, durable user-facing behavior belongs in
`book/src/usage/commitment-api.md`; verifier preparation and failure behavior
belong in `book/src/how/verification.md`. At that point this spec should be
marked `implemented` and later folded or archived according to
[`PRUNING.md`](../../PRUNING.md).

## Implementation slices

The slices below describe review and ownership boundaries. Slices 2 through 5
form one protocol cutover: no branch state may expose both the ambient-point API
and the group-owned API, and no compatibility wrapper may survive the final
diff. They may be separate commits while developing, but the merged result is a
single protocol epoch.

### Slice 1: canonical Lagrange expansion

**Goal.** Canonicalize the independent arithmetic repair in PR #322.

**Status.** Implemented on this branch. The canonical parent split is shared by
the serial, cached, and parallel builders, and the operation-count test checks
the exact `2^s - 1` multiplication and subtraction count.

**Changes.**

1. Add one inlinable parent split and one serial full-table traversal to
   `akita-algebra`.
2. Make `EqPolynomial::evals_serial` delegate to the traversal.
3. Make `lagrange_weights` retain only `basis_weight_len` validation and
   delegation, or remove it after converting all callers.
4. Make cached-layer and parallel expansion use the canonical split.
5. Add output-parity and multiplication-count tests.

**Primary diff surface.**

- `crates/akita-algebra/src/eq_poly.rs`
- `crates/akita-types/src/layout/opening_point.rs`
- their unit tests and exports only if the public boundary changes

**Completion gate.** A grep/review finds one serial full-table traversal and
one two-child arithmetic primitive. Empty, scaled, base-field, extension-field,
serial, cached, and parallel outputs retain their existing order and values.

### Slice 2: group-owned public claims

**Goal.** Replace the ambient point plus routing selection with one point owned
by each `PolynomialGroupClaims`.

**Status.** Implemented on this branch, including the descriptor epoch cutover
and all public, test, benchmark, profile, prover, and verifier call sites.

**Changes.**

1. Move `OpeningPoints` into `PolynomialGroupClaims`.
2. Remove `PointVariableSelection`, the batch point, batch `num_vars`,
   `group_point_vars`, `from_groups_allow_custom_routing`, and shared-point
   padding constructors.
3. Change `group_point` to return a borrowed slice.
4. Derive `OpeningClaimsLayout` from group point lengths and evaluation counts.
5. Update capacity validation to compare `max_num_vars()` with the setup seed.
6. Update prover validation so each group's polynomial arity equals its owned
   point length.
7. Update descriptor/transcript tests for ordered group points.

**Primary diff surface.**

- `crates/akita-types/src/opening_claims.rs`
- `crates/akita-types/src/lib.rs`
- `crates/akita-types/src/proof/mod.rs`
- `crates/akita-prover/src/types/opening_data.rs`
- `crates/akita-prover/src/api/commitment.rs`
- public PCS re-exports, examples, benches, and claim-construction tests

**Mechanical migration surface.** Every
`PolynomialGroupClaims::new(PointVariableSelection::..., ...)` call becomes
`PolynomialGroupClaims::new(group_point, ...)`. Multi-group fixtures pass the
existing per-group point slices instead of first padding or concatenating them.

**Completion gate.**

```text
rg "PointVariableSelection|group_point_vars|from_groups_allow_custom_routing" crates
```

returns no matches, and no replacement arena/selection type exists.

### Slice 3: direct root and EOR consumption

**Goal.** Feed group-owned points into the per-group machinery already present
on `main`.

**Status.** Implemented on this branch. Root and EOR paths borrow group-owned
points directly, and end-to-end tests cover unrelated group points.

**Changes.**

1. Root prove/verify borrows each claim group's point directly.
2. Each group is validated against its scheduled `d_a`, position bits, and
   block bits before preparation.
3. Batched multi-group EOR receives the borrowed group-point vector directly;
   the single-group EOR path creates one ordinary group rather than a synthetic
   padded ambient claim.
4. `PreparedFoldReplay` and `RingRelationInstance` keep their existing
   per-group carriers and ordering.
5. Remove point materialization allocations and routing-only checks.

**Primary diff surface.**

- `crates/akita-prover/src/protocol/core/root_fold.rs`
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`
- `crates/akita-verifier/src/protocol/core/root_fold.rs`
- `crates/akita-verifier/src/protocol/core/fold.rs`
- `crates/akita-types/src/extension_opening_reduction.rs`
- multi-group dense, one-hot, EOR, and mixed-D tests

**Completion gate.** Equal, prefix-related, suffix-related, and unrelated group
points all reach the same per-group preparation and relation APIs. No root code
constructs a common value point.

### Slice 4: setup-only Stage 3 and recursive suffix

**Goal.** Remove the witness carry and let the successor consume the independent
Stage 2 and Stage 3 claims directly.

**Status.** Implemented on this branch. Stage 3 proves only the setup product,
while recursive successors consume the Stage 2 witness point and Stage 3 setup
point as independent group-owned claims.

**Changes.**

1. Simplify the Stage 3 prover to one setup-product term and one native setup
   domain.
2. Delete `witness_claim_reduction.rs`, `WitnessClaimReductionTerm`, the
   two-term batching driver, `eta`, lift scales, and witness digit scans.
3. Delete `BatchedStage3Geometry`; the Stage 3 sumcheck challenge is the setup
   point.
4. Remove `next_w_eval` from `SetupSumcheckProof`, its wire serializer,
   deserializer, size calculation, validity checks, dummy proofs, and reports.
5. Remove the Stage 3 sample of `CHALLENGE_SUMCHECK_BATCH` and delete
   `ABSORB_STAGE3_NEXT_W_EVAL`; keep `CHALLENGE_SUMCHECK_BATCH` for the other
   protocols that still use it.
6. Make recursive suffix construction create:

   ```text
   setup group   = (stage3_setup_point, stage3_setup_prefix_eval)
   witness group = (stage2_point, stage2_next_w_eval)
   ```

7. Remove shared-point concatenation, suffix selection, and setup-offset
   construction from prover and verifier suffix code.

**Primary diff surface.**

- `crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`
- delete
  `crates/akita-prover/src/protocol/sumcheck/akita_stage3/witness_claim_reduction.rs`
- `crates/akita-prover/src/protocol/core/fold.rs`
- `crates/akita-prover/src/protocol/core/suffix.rs`
- `crates/akita-prover/src/types/opening_data.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
- `crates/akita-verifier/src/protocol/core/fold.rs`
- `crates/akita-verifier/src/protocol/core/suffix.rs`
- delete `crates/akita-types/src/stage3_geometry.rs`
- `crates/akita-types/src/proof/levels.rs`
- `crates/akita-types/src/proof/wire.rs`
- `crates/akita-transcript/src/labels.rs`
- transcript-hardening, recursive setup, and malformed-proof tests

**Completion gate.**

```text
rg "BatchedStage3Geometry|WitnessClaimReductionTerm|ABSORB_STAGE3_NEXT_W_EVAL" crates
```

returns no matches. Recursive-offload E2Es demonstrate unrelated successor
points and exact prover/verifier transcript parity.

### Slice 5: proof pricing, schedules, and setup registries

**Goal.** Make all derived planning state reflect the smaller setup-only proof.

**Status.** Implemented on this branch. Proof pricing now depends only on the
challenge width and setup domain, generated schedules remain stable, and the
selected setup-prefix registry and cache round-trip checks pass.

**Changes.**

1. Remove `output_witness_len` from `stage3_setup_product_bytes`.
2. Update planner and runtime schedule callers to price only
   `(challenge bits, d_setup, setup ring length)`.
3. Update actual-serialization tests for `SetupSumcheckProof`.
4. Regenerate schedule tables and identify any changed planner-selected suffix.
5. Recompute required setup-prefix slot IDs for every supported capacity.
6. Regenerate serialized setup caches when the registry changes.
7. Update profile reporting to remove the Stage 3 witness-evaluation component.

**Primary diff surface.**

- `crates/akita-types/src/proof_size.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-schedules/src/runtime.rs`
- `crates/akita-schedules/src/generated/`
- `crates/akita-config/src/setup_prefix_slots.rs`
- `crates/akita-setup/src/recursive_prefixes.rs`
- `crates/akita-setup/src/lib.rs`
- `crates/akita-pcs/examples/profile/report.rs`
- generated-table, proof-size, setup-cache, and planner tests

**Completion gate.** Planned Stage 3 bytes equal serialized bytes for several
setup-domain sizes and fields. Schedule regeneration is clean on a second run,
and every schedule-selected prefix has a setup registry entry.

### Slice 6: documentation and performance record

**Goal.** Make the shipped protocol, not this transition spec, the durable
source of truth.

**Status.** Implemented on this branch. The durable claim, verification,
architecture, and EOR descriptions live in the Book chapters listed below.

**Changes.**

1. Update the Book commitment API with group-owned claims examples.
2. Update verifier and extension-opening chapters with the group-local dataflow.
3. Update architecture diagrams and proof-size descriptions.
4. Run the current production W8R2, mixed-D, and extension-field multi-group
   profiles.
5. Record canonical-expansion counts and Stage 3 proof/time changes.
6. Mark this spec implemented and fold or archive it under `PRUNING.md`.

**Validation record.** On 2026-07-29, the production recursive W8R2 profile
used two unrelated 16-variable precommitted points and an unrelated 32-variable
final point. The two setup-only Stage 3 payloads were 832 bytes each, compared
with 944 and 880 bytes before the cutover, a total reduction of 160 bytes.
Single-run traced Stage 3 prover times were 58.1 ms and 43.5 ms, compared with
186 ms and 74.2 ms before the cutover. These timings are indicative local
measurements, not stable performance thresholds.

The mixed-D one-hot, unrelated dense, and unrelated extension-field EOR
profiles all round-tripped. Canonical serial and cached expansion tests observe
exactly `2^s - 1` multiplications and `2^s - 1` subtractions. Point equality,
prefix relationships, and unrelated values do not affect that count; each
group's arity determines its independent expansion cost. No preparation cache
or nested-factor DAG was added.

**Primary diff surface.**

- `book/src/usage/commitment-api.md`
- `book/src/how/verification.md`
- `book/src/how/architecture.md`
- `book/src/how/proving/extension-opening-reduction.md`
- relevant profiling documentation and superseded specs

### Optional Slice 7: preparation reuse

Only after Slice 6 measurements, add bounded exact or nested factor reuse if it
has material wall-clock value. This slice is not part of protocol completion.
It changes neither public claims, transcript, descriptor, proof wire, schedule,
nor setup.

## Final state

After Slices 1 through 6, the opening path has one simple ownership tree:

```text
OpeningClaims
└── groups, in transcript order
    └── PolynomialGroupClaims
        ├── complete point
        ├── evaluations
        └── commitment
```

The structural and value paths are cleanly separated:

```text
group lengths/counts ──> OpeningClaimsLayout ──> setup / planner / schedule
group points ──────────> EOR / prepare_opening_point ──> RingRelationInstance
group evaluations ─────> transcript batching / relation target
```

Recursive setup offloading has no point-routing detour:

```text
Stage 2 ──> witness point + witness evaluation ─┐
                                               ├─> successor OpeningClaims
Stage 3 ──> setup point + setup evaluation ────┘
```

There is:

- no ambient shared point;
- no `PointVariableSelection`;
- no allocation to recover a group point;
- no custom-routing constructor;
- no fused Stage 3 witness term;
- no duplicate Stage 3 witness evaluation;
- no two-domain Stage 3 geometry;
- one canonical Lagrange parent split and serial table traversal;
- one existing multi-group EOR and ring-relation pipeline serving arbitrary
  points, dense/one-hot sources, extension fields, and mixed ring dimensions;
- unchanged layout-driven setup/SIS geometry; and
- smaller, exactly priced recursive Stage 3 proofs.

The resulting implementation is materially easier to explain and audit:
public ownership matches algebraic ownership, the prover and verifier consume
the same ordered group objects, schedule code sees only field-free layout, and
recursive offloading proves exactly the setup statement it is responsible for.

## References

- [`shared-opening-claims-api.md`](../../shared-opening-claims-api.md)
- [`multi-group-batching.md`](../../multi-group-batching.md)
- [`batched-stage3-setup-opening.md`](../../batched-stage3-setup-opening.md)
- [`distributed-setup-offloading.md`](../../distributed-setup-offloading.md)
- [`setup-offloading-planner.md`](../../setup-offloading-planner.md)
- [PR #320: Stage 3 setup products and witness reduction](https://github.com/LayerZero-Labs/akita/pull/320)
- [PR #331: mixed-D multi-group composition](https://github.com/LayerZero-Labs/akita/pull/331)
- [`mixed-ring-dimension-per-level.md`](../../mixed-ring-dimension-per-level.md)
- [`runtime-schedule-boundary.md`](../../runtime-schedule-boundary.md)
- [`book/src/how/verification.md`](../../../book/src/how/verification.md)
- [`book/src/usage/profiling.md`](../../../book/src/usage/profiling.md)
