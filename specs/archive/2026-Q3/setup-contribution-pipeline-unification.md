# Spec: Unify Setup-Contribution Evaluation Around One Canonical Plan

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| Author(s)     | Quang Dao                                                   |
| Created       | 2026-07-22                                                   |
| Status        | archived                                                     |
| PR            | [#321](https://github.com/LayerZero-Labs/akita/pull/321)      |
| Supersedes    | Setup-evaluator ownership in `setup-layout-repack.md`         |
| Superseded-by |                                                              |
| Book-chapter  | book/src/how/proving/sumcheck-stages.md                       |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14
([RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) and
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)) when, and only when, they
appear in all capitals.

## Summary

The relation-matrix evaluator currently prepares one `SetupContributionPlan`
for Stage 2, but that plan mixes three kinds of state: canonical setup/witness
geometry, materialized structured-relation weights, and direct-scan execution
segments. Recursive setup offloading then evaluates the same mathematical
setup weight through a separate `SetupIndexWeightEvaluator`, with a
`RelationMatrixEvaluator` cache and reconstruction fallback connecting Stage 2
to Stage 3.

This design replaces those overlapping authorities with one immutable,
challenge-bound semantic plan. The plan describes each D, B, and A contribution
once as typed affine spans over relation rows, witness addresses, and packed
setup addresses. Structured relation evaluation, direct setup contraction,
recursive Stage 3 point evaluation, and prover weight materialization are
distinct contractions of those same spans.

Direct and offloaded verification remain two protocol execution paths, but not
two setup-weight pipelines. Both prepare the same compact plan and evaluate the
same structured relation terms. Only the direct path derives a scan backend;
only the offloaded path evaluates the setup-index weight at the Stage 3 point.

## Intent

### Goal

Make `SetupContributionPlan` the sole semantic source of setup-contribution
weights, with one common preparation and structured-relation pipeline followed
by either direct setup contraction or recursive Stage 3 point evaluation.

### Terminology

- **Semantic plan:** the immutable, checked description of the D, B, and A
  setup-contribution spans and the challenges that bind their weights.
- **Role span:** one typed affine mapping between a relation row range, a
  witness-opening address range, and a packed setup address range. D, B, and A
  spans are different variants because their dimensions and A-fold behavior
  differ.
- **Structured relation:** the E, T, and Z relation-column contribution that is
  evaluated in Stage 2 independently of whether the setup term is direct or
  offloaded.
- **Direct contraction:** the inner product between the expanded setup and the
  setup-index weight induced by the semantic plan.
- **Point contraction:** evaluation of the setup-index-weight multilinear
  extension at the Stage 3 setup-index challenge.
- **Materialization:** production of the dense setup-index-weight vector for a
  prover backend or a test oracle. Materialization is not verifier-plan
  preparation.
- **Compiled backend:** private, derived execution state such as direct-scan
  segments, tiles, or bounded equality lookup tables. It is not a second
  semantic representation.

### Scope

This spec changes the current verifier and shared setup-contribution
implementation. It covers:

- `akita-types` setup-contribution geometry, planning, and contractions;
- `RelationMatrixEvaluator` use of the plan during Stage 2;
- the Stage 2 to Stage 3 ownership handoff;
- recursive Stage 3 setup-index-weight evaluation;
- prover materialization of setup-index weights; and
- direct, recursive, singleton, multi-group, and multi-chunk setup layouts that
  are already accepted by the schedule and verifier.

### Invariants

- **One semantic representation.** Every production D, B, and A weight MUST be
  derived from the role spans stored in `SetupContributionPlan`. A consumer
  MUST NOT rederive witness or setup addresses from `CommittedGroupParams`,
  `OpeningClaimsLayout`, or `WitnessLayout` after the plan is prepared.
- **One plan for both paths.** Direct and offloaded Stage 2 verification MUST
  prepare the same plan from the same relation point and checked layout inputs.
  The plan MUST NOT contain a direct/offloaded mode or any other independent
  scheduling authority.
- **Common structured evaluation.** Both paths MUST evaluate E, T, and Z from
  the same role spans before selecting the setup-contribution execution path.
  Substituting an offloaded proof claim changes only how the setup term is
  discharged; it MUST NOT select a different structured-weight formula.
- **Path-local work.** Offloaded verification MUST NOT build direct-scan
  segments or scan the expanded setup to prepare relation weights. Direct
  verification MUST NOT prepare Stage 3 compact-pair state. A backend MAY
  compile private state only when its contraction is invoked.
- **No dense verifier preparation.** Verifier plan preparation MUST NOT
  allocate E, T, or Z weight vectors, copied A/B/D row-weight vectors, or a
  vector proportional to the active setup length. It MAY retain shared
  challenge vectors and bounded lookup state whose size is independent of the
  active E/T/Z column counts and setup prefix length.
- **Explicit stage ownership.** A successfully verified Stage 2 MUST return the
  exact plan used to compute its expected relation claim. Recursive Stage 3
  MUST consume that plan. It MUST NOT recover it from
  `RelationMatrixEvaluator`, rebuild it as a fallback, or accept a plan prepared
  from a different challenge point.
- **Pure relation evaluator.** `RelationMatrixEvaluator` MUST remain a checked,
  immutable description of the relation. It MUST NOT own a mutable
  cross-stage setup-plan cache.
- **Distinct operations, shared terms.** Structured evaluation, direct
  contraction, point contraction, and dense materialization are different
  mathematical operations and MAY have separate optimized implementations.
  Each implementation MUST iterate or compile the plan's canonical role spans;
  it MUST NOT maintain a parallel role-address formula.
- **Prover/verifier agreement.** Prover materialization, direct verifier
  contraction, and recursive verifier point contraction MUST define the same
  setup-index-weight multilinear polynomial, including role projection,
  alpha-scaling, row weights, fold-gadget weights, padding, and setup-prefix
  truncation.
- **No protocol change.** Proof serialization, transcript labels, challenge
  order, schedule selection, setup layout, setup-prefix identity, and Stage 2
  and Stage 3 equations MUST remain byte-for-byte unchanged.
- **Verifier rejection.** Invalid dimensions, overflowing affine spans,
  truncated setup prefixes, inconsistent group order, wrong challenge lengths,
  and missing Stage 2 plan state MUST return `AkitaError`. Verifier-reachable
  code MUST NOT panic, index unchecked, or allocate from an unvalidated proof
  length.

### Non-Goals

- Changing when the planner selects direct or recursive setup contribution.
- Adding a new setup mode, a runtime strategy selector, or a second protocol
  path.
- Changing proof objects, transcript events, setup-prefix commitments, or
  generated schedules.
- Fusing Stage 2 and Stage 3 or removing Stage 3.
- Generalizing layouts or ring-dimension combinations that the current
  verifier rejects.
- Requiring one execution kernel for all contractions. The single source of
  truth is the semantic span representation, not a universal hot loop.
- Preserving the current internal Rust API. Akita makes no source-level
  backward-compatibility guarantee.

## Design

### Canonical semantic plan

`SetupContributionPlan` is prepared once the Stage 2 relation point is known.
It owns or shares only the data required to define the setup weight:

- validated group order and role geometry;
- D, B, and A row ranges;
- typed role spans with checked affine setup and witness addresses;
- projection ratios between each role ring and the base setup ring;
- the active setup length and padding domain;
- the Stage 2 column challenges;
- the Stage 2 row-equality data (`eq_tau1` or an equivalent shared
  representation);
- the common fold gadget and per-group fold depths; and
- the alpha-independent projection geometry needed by every contraction.

The plan MUST NOT own:

- materialized `e_eq_slice`, `t_eq_slice`, or `z_eq_slice` vectors;
- copied A, B, or D row-weight vectors;
- `GroupSetupSegment` or another direct-scan schedule;
- setup coefficients or an expanded setup view;
- a dense setup-index-weight vector;
- `SetupContributionMode` or a boolean that selects direct versus recursive;
  or
- a second copy of the source layouts from which consumers can rederive
  addresses independently of the checked spans.

The plan's semantic identity is its checked role spans plus challenge and
projection data. A bounded equality window or a direct scan schedule MAY be
cached privately inside the contraction that derives it, but it MUST be
discardable and MUST NOT become an alternate source of addresses or weights.

### Typed role spans

Each role span represents a checked recurrence rather than one element per
column. At minimum, a span defines:

```text
role
relation row or row range
setup start, setup stride, and length
witness start, witness stride, and length
group and chunk ownership
role projection ratio
A-only fold range and gadget indexing, when applicable
```

Preparation performs all checked arithmetic required to prove that both affine
ranges fit their validated domains. Contractions can therefore traverse a span
without reconstructing layout formulas. Role-specific variants MAY carry
additional compact indices when required, but MUST NOT expand a span into
per-column field elements during common plan preparation.

### One pipeline, two execution paths

The verifier flow is:

```text
prepare SetupContributionPlan from the Stage 2 relation point
    |
    +-- evaluate structured E/T/Z relation terms from canonical role spans
    |
    +-- evaluate the common non-setup relation terms
    |
    +-- direct schedule
    |      derive a private direct backend
    |      contract the expanded setup
    |
    +-- recursive schedule
           substitute the transcript-bound setup claim in Stage 2
           pass the exact plan with the verified Stage 2 output
           evaluate the plan at the Stage 3 setup-index point
```

The existing schedule/proof shape is the only authority selecting the final
branch. The plan is identical up to that branch.

The production plan exposes one canonical implementation for each required
contraction. Exact names MAY follow local Rust conventions, but their semantic
responsibilities are:

```text
evaluate_structured_relation(plan, relation inputs)
evaluate_direct(plan, expanded setup, alpha)
evaluate_at_point(plan, rho_setup_index, alpha)
materialize_setup_index_weights(plan, alpha)
```

These operations are not pass-through wrappers. Each performs a distinct fold
over the same role spans. Shared arithmetic primitives, such as evaluating one
span's witness equality or role projection, MUST live below these operations
and be called directly.

### Structured relation evaluation

The E, T, and Z weights are evaluations of witness-address spans at the Stage 2
column point. They are not long-lived plan data.

`evaluate_structured_relation` evaluates the required span contractions into
the relation accumulator. It MAY use a shared bounded `OffsetEqWindow`, but it
MUST NOT first materialize complete E, T, or Z vectors. A role span's witness
address and length come from the plan; this operation does not call the old
`setup_e_col_weights`, `setup_t_col_weights`, or `setup_z_col_weights`
production materializers.

This operation is common to direct and recursive schedules. The recursive path
therefore still performs the work required to evaluate the structured relation,
but skips all direct-setup scan preparation.

### Direct contraction

`evaluate_direct` computes the setup contribution by contracting each live
setup coefficient with the weight induced by the role spans. It validates the
expanded setup envelope before reading coefficients.

The implementation MAY derive sorted segments, tiles, or fused multi-group
jobs inside this operation. Such state is a private compilation of the role
spans and MUST NOT be stored in the semantic plan. Direct contraction is the
only verifier operation allowed to derive direct-scan segments.

### Recursive point contraction

`evaluate_at_point` evaluates the multilinear extension of the setup-index
weight at `rho_setup_index`. It uses the canonical affine spans and a compact
pair/carry recurrence or an algebraically equivalent method. It MUST NOT scan
the active setup, build a dense setup-index equality table, or rederive role
addresses from the source layouts.

This operation replaces `SetupIndexWeightEvaluator`. There is no separately
constructed evaluator object with cloned layouts and a second role formula.
Stage 3 receives the plan used by Stage 2 and calls the plan's point
contraction directly.

The setup-prefix polynomial evaluation and alpha-coordinate evaluation remain
separate Stage 3 factors. This spec changes only the source and evaluation of
the setup-index weight.

### Prover materialization

The Stage 3 prover MAY require a dense setup-index-weight vector. It obtains
that vector only through `materialize_setup_index_weights`, which evaluates the
canonical role spans over the active setup-index domain.

Dense materialization is permitted in the prover path because it is an explicit
output of this contraction, not hidden plan-preparation state. If a later prover
backend consumes spans directly, it replaces this contraction without changing
the plan or verifier semantics.

### Stage 2 to Stage 3 ownership

Stage 2 verification returns an explicit value equivalent to:

```text
Stage2VerifyOutput {
    challenges,
    setup_plan,
}
```

The generic sumcheck verifier currently asks for
`expected_output_claim(&self)`. The Akita Stage 2 implementation MAY use a
single-assignment cell local to that verifier instance to retain the plan while
the generic verifier finishes. After successful verification, the Stage 2
orchestrator consumes the verifier and moves the plan into
`Stage2VerifyOutput`.

This cell is a trait-adaptation detail, not a cache:

- it accepts exactly the plan used by `expected_output_claim`;
- it is consumed immediately after the sumcheck succeeds;
- it is not owned by `RelationMatrixEvaluator`;
- it is not keyed for later lookup;
- it has no reconstruction fallback; and
- duplicate initialization or missing state returns `AkitaError`.

Direct verification drops the plan after Stage 2. Recursive verification moves
it into `SetupSumcheckVerifier`. Stage 3 construction MUST NOT accept a
`RelationMatrixEvaluator` as a way to reconstruct setup-contribution state.

### Errors and security boundary

Preparation validates all group, row, projection, fold-gadget, witness-domain,
and setup-domain geometry before constructing a plan. Checked spans are not a
license for unchecked indexing: every external setup or proof slice is still
validated at its consumption boundary.

Malformed verifier inputs return the repository's existing `AkitaError`
variants. Internal single-assignment failure also returns `AkitaError`; mutex
poisoning is avoided by using a non-poisoning single-assignment primitive where
available. No error path falls back to recomputing a plan, because fallback
would hide an ownership defect and recreate two preparation paths.

## Evaluation

### Acceptance Criteria

- [x] `SetupContributionPlan` stores canonical typed D/B/A spans and no
      materialized E/T/Z slices, copied row-weight vectors, or direct segments.
- [x] Direct and recursive Stage 2 call the same plan preparation and the same
      structured-relation contraction.
- [x] Recursive Stage 2 does not derive direct-scan segments or scan setup
      coefficients while preparing/evaluating structured relation weights.
- [x] Direct setup evaluation derives any scan backend privately from the
      canonical spans.
- [x] Stage 2 returns an explicit output containing its challenges and the
      exact plan used for expected-claim evaluation.
- [x] `RelationMatrixEvaluator::setup_plan_cache`,
      `CachedSetupContributionPlan`, cache take/store methods, and Stage 3
      reconstruction fallback are deleted.
- [x] `SetupIndexWeightEvaluator` is deleted; Stage 3 evaluates the plan's
      canonical spans directly.
- [x] Production `setup_e_col_weights`, `setup_t_col_weights`, and
      `setup_z_col_weights` materialization is deleted. A dense implementation
      MAY remain under `#[cfg(test)]` as a migration oracle.
- [x] The Stage 3 prover obtains setup-index weights from the canonical
      materialization contraction or consumes the canonical spans directly.
- [x] Direct and recursive proof bytes and logging-transcript event streams are
      unchanged for deterministic fixtures.
- [x] All verifier-reachable malformed-layout tests return errors without
      panicking.

### Equivalence Tests

The migration retains dense implementations under `#[cfg(test)]` until all
four equalities below are covered:

```text
materialize(plan, alpha)
    == dense_setup_index_weight_oracle(inputs, alpha)

evaluate_direct(plan, setup, alpha)
    == dot(setup, materialize(plan, alpha))

evaluate_at_point(plan, rho, alpha)
    == MLE(materialize(plan, alpha), rho)

evaluate_structured_relation(plan, relation_inputs)
    == dense_E_T_Z_relation_oracle(inputs)
```

Coverage MUST include:

- singleton and multi-group opening batches;
- one and multiple witness chunks;
- ordinary, sparse, and tensor challenge shapes accepted by the verifier;
- multiple fold depths and supported opening bases;
- uniform role dimensions and every mixed-role layout already accepted by the
  verifier;
- direct and recursive generated schedules;
- overlapping multi-group setup spans;
- zero-length legal roles and padded setup-index domains; and
- malformed group order, truncated layouts, overflow boundaries, and wrong
  point lengths.

End-to-end tests MUST compare proof serialization and `logging-transcript`
events before and after the refactor. Existing recursive setup, distributed
setup-offloading, mixed-relation, and direct PCS suites remain required.

### Implementation evidence

- `akita-types` setup-contribution tests compare structured, direct, point, and
  materialized contractions across singleton, multi-group, multi-chunk,
  non-power-of-two, carry, and mixed-role fixtures.
- `mixed_role_e2e` commits, proves, serializes, deserializes, and verifies the
  supported direct mixed-role schedule through the public PCS API.
- `recursive_setup_e2e` and `distributed_setup_offload_e2e` exercise uniform
  recursive setup contribution, including the multi-chunk path.
- `fold_protocol_epoch` pins deterministic direct and recursive proof bytes and
  logging-transcript event streams.

### Performance

The refactor is successful only if common preparation is compact and path-local
work remains path-local.

- Recursive verifier preparation allocates no vectors proportional to active E,
  T, Z, or setup-index lengths.
- Recursive Stage 3 setup-index-weight evaluation remains sublinear in the
  active setup length and retains the compact pair/carry asymptotic behavior.
- Direct verification retains a fused multi-group scan and does not scan the
  expanded setup more than once per relation evaluation.
- Prover materialization performs at most one dense setup-index-weight build per
  Stage 3 proof.
- Benchmarks and Perfetto spans distinguish common plan preparation,
  structured-relation evaluation, direct backend compilation/scan, prover
  materialization, and recursive point evaluation.

No proof-size change is expected. Runtime comparisons use the same direct and
recursive profiles as PRs #318 and #320; any material regression must be
explained by a changed kernel rather than hidden duplicate preparation.

## Execution

The implementation should migrate by changing the semantic authority first,
then deleting the old representations:

1. Introduce typed role spans and populate them from the existing checked
   preparation inputs.
2. Implement dense materialization from the spans and compare it with the
   current dense setup-weight oracle.
3. Move structured E/T/Z evaluation to the span representation.
4. Derive the direct scan backend from spans inside `evaluate_direct`.
5. Move the compact Stage 3 point recurrence onto the plan spans.
6. Return the plan explicitly from Stage 2 and pass it to Stage 3.
7. Delete materialized production slices, stored scan segments,
   `SetupIndexWeightEvaluator`, and the relation-evaluator cache/fallback.
8. Retune direct, structured, materialization, and point-contraction kernels
   without changing the role-span representation.

Intermediate commits MAY retain old implementations as test-only differential
oracles. No intermediate production state should add a wrapper or a third
semantic setup-weight representation.

## Alternatives Considered

### Keep the current plan and skip fields in recursive mode

A mode-aware constructor could omit direct segments when offloading. This
reduces some work but makes the plan depend on scheduling, leaves materialized
E/T/Z slices in common preparation, and preserves the separate Stage 3 role
formula. It does not achieve a single source of truth.

### Separate direct and recursive plan types

Two plan types make path-local state explicit, but duplicate common geometry
and invite drift in structured weights, role projection, and address formulas.
The chosen design shares one semantic plan and keeps only compiled execution
state path-local.

### Keep `SetupIndexWeightEvaluator` as a wrapper over the plan

A thin evaluator wrapper would add ownership and API surface without defining a
new concept. The point contraction belongs directly on the canonical plan.

### Store every possible backend in the plan

Eagerly storing dense slices, direct segments, and point-evaluation state makes
both paths pay for all backends and turns execution artifacts into competing
authorities. The chosen design derives only the backend that is invoked.

### Widen the generic sumcheck trait

Changing the workspace-wide verifier trait to return Akita-specific auxiliary
state would spread this refactor beyond its concept boundary. A Stage 2-local
single-assignment handoff preserves explicit ownership without imposing
setup-contribution semantics on generic sumcheck code.

## Documentation

When implemented, fold the durable one-plan/two-path architecture into
`book/src/how/verification.md` and update
`book/src/how/proving/sumcheck-stages.md` only where the Stage 2 to Stage 3
handoff is described. Archive this spec after those pages become authoritative.

The implementation PR also updates any surviving setup-layout and Stage 3 specs
that name deleted materialized slices, `SetupIndexWeightEvaluator`, or the
relation-evaluator cache.

## References

- [`setup-layout-repack.md`](setup-layout-repack.md)
- [`setup-product-sumcheck.md`](setup-product-sumcheck.md)
- [`setup-offloading-planner.md`](setup-offloading-planner.md)
- [`distributed-setup-offloading.md`](distributed-setup-offloading.md)
- [`relation-range-image-sumcheck.md`](relation-range-image-sumcheck.md)
