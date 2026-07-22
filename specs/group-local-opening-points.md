# Spec: Group-local opening points and reusable verifier preparation

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-22 |
| Status        | active |
| PR            | |
| Supersedes    | Point-model portions of [`shared-opening-claims-api.md`](shared-opening-claims-api.md) and [`multi-group-batching.md`](multi-group-batching.md) |
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

The first implementation PR also changes full-table multilinear Lagrange and
equality-table expansion from two multiplications per parent to one
multiplication and one subtraction. That independent improvement removes almost
half of the opening point preparation multiplications in the representative
production schedule. The claims, transcript, and descriptor cutover is larger
follow-up work governed by the acceptance criteria below.

## Status and decision boundary

This record distinguishes current code from the approved target so that readers
do not mistake design work for shipped behavior.

| Concern | State before this record | Approved target |
|---------|--------------------------|-----------------|
| Public point input | One shared point plus `PointVariableSelection` per group | One complete point stored by each group |
| Group preparation | One `prepare_opening_point` call per group | The same canonical call per group, backed by internal exact reuse |
| Nested points | Prefix/suffix routing is representable | Arbitrary points are valid; nesting is optional optimization metadata derived internally |
| Layout and schedules | Ordered `(num_vars, num_polys)` groups | Unchanged |
| Lagrange/equality full-table expansion | Two multiplications per parent in serial paths | One multiplication and one subtraction per parent, landed with this record |

The analysis and concrete counts in this record were checked against `main` at
commit `a1c8782e9b2f3d4fa35e78918c64e4c3c0a6d94d`.

## Terminology

- A **group-local point** is the complete ordered point at which every
  polynomial in one commitment group is opened.
- **Opening preparation** converts a field point into its padded point, packed
  inner factor, position weights, live block weights, and ring-multiplier view.
- **Exact reuse** shares a prepared factor when its complete semantic cache key
  is equal.
- **Nested reuse** constructs a larger tensor-product factor from a smaller
  prefix or suffix factor plus additional coordinates.
- **Offloading** means distributed setup-contribution generation or evaluation.
  It does not move opening-point preparation out of the verifier.

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
    schedule selection, proof fields, proof size, or SIS pricing.
13. The verifier MUST NOT materialize a full table of size `2^num_vars` merely
    to discover or exploit nested points. It prepares the inner, position, and
    live-block factors required by the selected geometry.
14. All verifier-reachable point validation and cache lookups MUST satisfy the
    repository no-panic contract.

### Non-Goals

- Opening different polynomials within one commitment group at different
  points. Such claims belong in separate groups.
- Extending dense, extension-opening-reduction, root-terminal, or recursive
  suffix protocols to more groups where their schedules currently require one.
- Changing setup-contribution relations, offloading policy, witness partition,
  or generated schedule geometry.
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

## Evaluation

### Acceptance Criteria

#### This specification PR

- [x] This normative record passes repository documentation guardrails and
  links the point-model supersession from both predecessor specs.
- [x] `lagrange_weights` and the serial `EqPolynomial` table builders use one
  multiplication and one subtraction per expanded parent while preserving
  exact values and order.
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

#### Verifier preparation and performance

- [ ] Exact duplicate factor reuse is covered by hit/miss tests for inner,
  position, block, and complete-point keys.
- [ ] Cache-key negative tests vary basis, `D`, `M`, `B`, coordinate order, and
  live block length independently.
- [ ] A preparation benchmark reports base- and extension-field multiplication
  counts separately for equal, nested, and unrelated points.
- [ ] Arbitrary unrelated points do not add asymptotic work beyond independent
  per-group preparation and do not change proof or setup size.
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
confirm `AkitaError` rather than panics.

### Performance

The recurrence changes preserve allocations and table order while replacing
exactly one multiplication per expanded parent with a subtraction. The first
implementation should confirm output equivalence in tests; a benchmark is not
required to accept this algebraic rewrite.

The later claims cutover must record the preparation-only counts described
above and run the representative end-to-end profile:

```bash
cargo run -p akita-pcs --release --no-default-features \
  --features parallel,profile-onehot-fp128-d64 \
  --example profile
```

For a fixed layout, proof bytes, setup bytes, and generated schedule identity
must remain unchanged. Exact reuse should allocate at most one stored factor per
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

### Combine groups into one synthetic opening

For distinct points there is no single evaluation vector that turns the claims
into the existing one-point relation without changing the protocol. Transcript
batching of separate per-group relations is the correct general construction.

## Security and failure behavior

Binding complete ordered group points before batching challenges prevents a
prover from adapting point routing after seeing coefficients. Layout and point
dimensions are validated before allocation or indexing. Cache equality is an
implementation optimization only: a false miss costs time, while a false hit
would be a soundness bug, so semantic cache keys are complete and tested by
negative cases.

Malformed serialized points, excessive group counts, overflowing dimensions,
and schedule mismatches return typed errors at existing verifier boundaries.
No new unchecked indexing, assertion, unbounded allocation, or
attacker-controlled persistent cache is permitted.

## Compatibility

The claims and descriptor cutover is intentionally breaking. Callers must move
from one shared point plus selections to one complete point per group. Because
the descriptor and transcript statement change, proofs produced by the old and
new APIs are not cross-compatible.

For the same ordered group layout, setup artifacts, schedules, relation
dimensions, and proof structure remain valid. The Lagrange recurrence change is
purely computational and produces byte-identical field values.

## Documentation

This active spec is the implementation record until the claims cutover ships.
The point-model sections of `shared-opening-claims-api.md` and
`multi-group-batching.md` link here as superseded guidance. When implementation
is complete, durable user-facing behavior belongs in
`book/src/usage/commitment-api.md`; verifier preparation and failure behavior
belong in `book/src/how/verification.md`. At that point this spec should be
marked `implemented` and later folded or archived according to
[`PRUNING.md`](PRUNING.md).

## Execution

1. Land this decision record and the independent one-multiplication
   Lagrange/equality recurrence on a branch based directly on `main`.
2. Change the claims and prover-data types in one breaking cutover; remove point
   selection and all pass-through routing APIs.
3. Change descriptor and transcript absorption in the same cutover, then update
   generated fixtures and transcript-smell tests.
4. Route every group through the canonical preparation pipeline and add bounded
   exact factor reuse.
5. Add unrelated-point end-to-end tests and preparation benchmarks.
6. Implement nested-factor reuse only if the measured result justifies its
   complexity.
7. Fold the shipped behavior into the Akita Book and update this spec's
   lifecycle fields.

## References

- [`shared-opening-claims-api.md`](shared-opening-claims-api.md)
- [`multi-group-batching.md`](multi-group-batching.md)
- [`distributed-setup-offloading.md`](distributed-setup-offloading.md)
- [`setup-offloading-planner.md`](setup-offloading-planner.md)
- [`book/src/how/verification.md`](../book/src/how/verification.md)
- [`book/src/usage/profiling.md`](../book/src/usage/profiling.md)
