# Spec companion: Quotient-free tail implementation contract

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-29 |
| Status        | active |
| Parent        | [quotient-free-tail-ring-relations.md](quotient-free-tail-ring-relations.md) |
| PR            | [#466](https://github.com/LayerZero-Labs/akita/pull/466) |
| Book-chapter  | book/src/how/proving/akita-fold-realizations.md |

This is the normative implementation, evaluation, and risk companion to the
quotient-free tail protocol specification. The parent owns the protocol
decision, algebra, feature matrix, eligibility state machine, and compatibility
boundary; this companion owns the cross-crate architecture and executable
acceptance contract.

Acceptance status was re-audited against PR #466 on 2026-09-04. The exact
code-and-evidence head is `04111dedf`; merge `5fd356d0c` then incorporates the
documentation-only main commit `f9f7de87b` without changing protocol code or
measurements. Checked boxes below have a concrete implementation and regression
test, generated artifact, or pinned benchmark record. Remaining work, if any,
must be stated explicitly rather than inherited from the closed #445 review.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

### End-to-end data flow

```text
Planner decision
    |
    v
CommittedGroupParams.ring_relation_mode
    |----> canonical level/schedule descriptor
    |----> external catalog row and identity
    |----> FoldSchedule eligibility validation
    |
    v
RelationWitnessGeometry + WitnessLayout
    |---- QuotientLift: Z | E | T | R | compression digits/quotients
    `---- ReducedEvaluation: Z | E | T | compression digits
    |
    +----> exact successor witness length / source moments / proof sizing
    +----> prover relation-weight compiler
    `----> verifier relation evaluator
              |
              +---- structured challenge and trace terms
              +---- fused direct setup scan with H weights
              `---- no quotient-tail evaluation
```

### Protocol type and descriptor ownership

`RingRelationMode` belongs in `akita-types`. `CommittedGroupParams` SHOULD
store it beside `payload_mode`, because both values describe how one complete
fold realizes its relation and outgoing witness. It MUST NOT live on an
individual `GroupOpenPhaseParams`: one Stage-2 relation batches every group
and owns one shared quotient policy.

Required type-layer changes include:

1. Add `RingRelationMode` with stable descriptor tags:
   `QuotientLift = 1` and `ReducedEvaluation = 2`. The implementation MUST use an
   explicit `tag()` match, not a Rust discriminant cast.
2. Add `ring_relation_mode` to `CommittedGroupParams::try_new` and every
   canonical builder/materializer.
3. Append exactly one relation-mode tag immediately after the existing
   payload-mode tag in
   `CommittedGroupParams::append_descriptor_bytes_with_payload_mode`, before
   `source_encoding`. Root and recursive schedule descriptors already invoke
   this canonical parameter encoding; they MUST NOT append a second copy.
4. Bump the effective schedule descriptor epoch because the previous byte
   language did not contain this field.
5. Serialize the mode once in each fold's `CommittedGroupParams` inside the
   full `FoldSchedule` stored by the external artifact. Do not introduce a
   second generated row schema.
6. Include the tag in external catalog row identity and policy reports.
7. Keep the mode out of proof serialization. The verifier obtains it from the
   already authenticated schedule.

### Schedule validation

`FoldSchedule::validate_structure` is the canonical adjacency checker. It
SHOULD validate the complete nonterminal mode sequence in one pass while it
already checks payload phase and setup-prefix topology.

The pass keeps two booleans or typed phases:

```text
relation phase = QuotientPrefix
for each nonterminal level in absolute order:
    validate current mode against absolute level and incoming prefix
    if current mode is ReducedEvaluation:
        relation phase = ReducedEvaluationSuffix
        reject any future incoming prefix
    if relation phase is ReducedEvaluationSuffix:
        reject QuotientLift
```

The generated walker MUST exercise the same validator after expansion. The
planner MUST not have a private copy of this schedule rule.

### Mode-aware witness layout

`WitnessLayout` currently places the Z/E/T units, ordinary R rows, compression
digits, compression quotient rows, alignment, and zero suffix. The new mode
must alter that one construction:

```text
QuotientLift, raw:
    Z | E | T | ordinary R

ReducedEvaluation, raw:
    Z | E | T

QuotientLift, compressed:
    Z | E | T | ordinary R | F/H digits | F/H quotient rows

ReducedEvaluation, compressed:
    Z | E | T | F/H digits
```

The semantic `RelationRhsLayout::row_families()` remains complete in both
modes because the reduced-evaluation compiler and verifier still need every physical row.
Only `WitnessLayout::r_rows()` becomes empty in reduced-evaluation mode.

APIs that address an R coefficient SHOULD return a typed error when called on
a reduced-evaluation layout. They MUST NOT return zero, alias another range, or use an
unchecked optional value. The existing live-length, successor padding, and
Boolean-domain calculations then update automatically.

The normative native quotient-tail section of
[`role-native-projected-digit-layout.md`](role-native-projected-digit-layout.md)
continues to define `QuotientLift`. This specification is the additional
authority for the no-R `ReducedEvaluation` case until both designs are folded into
the Book.

### Shared algebra primitive

The residue recurrence belongs in `akita-algebra::ring`, where both prover and
verifier can test it without depending on schedule or proof types. The module
SHOULD expose one checked primitive for each actual concept, for example:

```rust
pub fn residue_kernel<F, E>(
    coefficients: &[F],
    alpha: E,
) -> Result<Vec<E>, AkitaError>;

pub fn terminal_residue_kernel<E>(
    equality_weights: &[E],
    alpha: E,
) -> Result<Vec<E>, AkitaError>;
```

Exact names may change during implementation. The ownership requirements do
not:

- one checked recurrence implementation;
- one independent quadratic reference under tests;
- no prover and verifier copies of the recurrence;
- no `_for_level` wrappers;
- power-of-two dimension and point-arity validation before allocation;
- no division by `alpha^d + 1`.

The caller SHOULD derive `e_j = eq(point, physical_start + j)` through the
existing checked offset-equality window authority, then pass those exact
weights to `terminal_residue_kernel`. The primitive MAY reuse that input
allocation for `H` when ownership makes this clear. Tests must still expose
both mathematical quantities to the oracle.

### Prover relation-weight representation

The current Stage-2 prover stores a rank-one
`RelationWeightFactorization<E>`:

```text
common alpha factor: d0 elements
relation lane weights: W / d0 elements
```

Generic reduced-evaluation weights are not rank one across lane and coefficient.
The implementation MUST not hide that fact behind an invalid
factorization.

The clean minimum is one canonical Stage-2 relation-weight state with two
realizations:

```rust
enum RelationWeightOracle<E> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
}
```

The exact type name is not normative. Its behavior is:

- `QuotientFactored` preserves the current optimized coefficient/lane path.
- `ReducedDense` contains the complete padded reduced-evaluation weight MLE over the
  existing Stage-2 witness domain and folds in place with sumcheck challenges.
- Dispatch occurs once per sumcheck round or construction phase, not once per
  witness coordinate.
- Both variants use the same witness state, range-image term, structured
  opening term, transcript, and terminal claim API.
- The dense reduced-evaluation table is ephemeral extension-field prover state. It is not
  committed, serialized, or counted as proof bytes.

This baseline path uses `O(W)` extension-field storage and `O(W)` folding work
for a Stage-2 domain of size `W`. The acceptance criteria prioritize verifier
cleanliness and proof size, while still requiring the prover cost to be
measured and reported. A future optimization may add streamed, checkpointed,
sparse-kernel, or rank-two prover variants behind the same semantic oracle
without changing this protocol mode.

The reduced-evaluation compiler MUST derive weights from semantic row families and
canonical witness ranges. It MUST NOT construct a second dense relation matrix
or replay setup rows in a new order.

### Prover construction path

In `QuotientLift`, `ring_switch_build_w` keeps the current order:

1. Prepare group Z/E/T values.
2. Compute ordinary and compression relation quotients.
3. Allocate the mode-aware witness.
4. Emit Z/E/T, quotient digits, and compression digits.

In `ReducedEvaluation`, it becomes:

1. Prepare the same group Z/E/T values and compression digits.
2. Construct the mode-aware `WitnessLayout` with no R rows.
3. Allocate and emit only the live non-quotient witness segments.
4. Skip `compute_multi_group_relation_quotient` and every R decomposition.
5. After `alpha` and `tau1`, compile the dense reduced-evaluation relation-weight oracle
   from public setup, public challenges, row weights, and canonical addresses.
6. Run the existing fused range-image/relation sumcheck through the reduced-evaluation
   oracle variant.

No empty `RelationQuotientOutput` SHOULD be constructed. The mode match should
occur before quotient computation.

The mode match MUST also occur before selecting product and transform backends.
Deleting R ranges is insufficient if the prover still computes values that are
discarded later. In reduced-evaluation mode:

- the D path MUST produce only its negacyclic reduced image, not a cyclic image
  and derived polynomial quotient;
- the compression path MUST retain negative-binary digits and terminal reduced
  images without constructing cyclic product or quotient images;
- A/B relation-quotient kernels MUST not run; and
- NTT/cache requirements MUST omit relation cyclic transforms, compression
  cyclic transforms, and centered quotient-tail caches that no live witness
  segment consumes.

These are typed backend requirements, not benchmark-only optimizations. Tests
and diagnostics MUST show zero quotient construction, decomposition, and
quotient-only NTT/cache preparation in reduced-evaluation mode.

### Compression path

Compression retains the F/H digit witness and its negative-binary restriction.
It removes only the polynomial-modulus quotient rows associated with the F/H
ring relations.

The current verifier keeps F/H relation weights in
`CompressionRelationWeights` and evaluates their quotient contribution through
the compression-specific path. Reduced-evaluation mode SHOULD reuse the same compression
map authority to compile reduced functional weights over the F/H digit ranges.
It MUST not reconstruct map coefficients from witness offsets.

The reduced-evaluation prover may merge these linear weights into `ReducedDense`. The
negative-binary pointwise term remains in `AdditionalRelationTerms`. This
separation keeps “ring reduction” distinct from “digit alphabet”.

### Verifier relation evaluator

`RelationMatrixEvaluator` remains the one verifier-side prepared object. It
SHOULD store the trusted `RingRelationMode` and dispatch once in
`evaluate_relation_at_point`:

```text
QuotientLift:
    prepare alpha-power coefficient functionals
    evaluate structured groups
    evaluate direct or deferred setup
    evaluate quotient tail
    multiply by common low alpha MLE factor

ReducedEvaluation:
    require deferred_setup_claim == None
    prepare H coefficient functionals
    evaluate structured groups with reduced evaluation weights
    evaluate fused direct setup with H
    do not evaluate a quotient tail
    return the already complete flat MLE
```

The reduced-evaluation branch MUST NOT call
`PreparedRelationPoint::common_alpha_evaluation`
as an outer factor. `H` already includes the coefficient equality
contraction. Multiplying by the old common factor would double-count it.

`PreparedRelationPoint` SHOULD be generalized around a checked native
coefficient functional rather than accreting optional alpha-power and H fields.
One possible internal shape is:

```rust
enum PreparedCoefficientFunctional<E> {
    LiftedPower { powers: Arc<[E]>, lane_powers: Arc<[E]> },
    ReducedEvaluation { terminal_kernel: Arc<[E]> },
}
```

The exact representation may differ. The important boundary is that
`SetupContributionPlan` and structured-term evaluators request the coefficient
weights they need without knowing planner policy or proof serialization.

### Fused setup scan

`SetupContributionPlan::evaluate_direct` currently receives native alpha-power
slices. It SHOULD evolve to consume a checked coefficient-functional view for
each role. The outer fused scan, setup bounds, segment scheduling, parallel job
partition, group fusion, and base-ring projection remain common.

The per-ring inner operation becomes:

```text
QuotientLift:  dot(setup_ring, [1, alpha, ..., alpha^(d-1)])
ReducedEvaluation: dot(setup_ring, H^(d)(r_coeff, alpha))
```

The common scanner MUST specialize the lifted power path where
`eval_ring_at_pows_fast` is faster. Sharing the scanner does not require
discarding its optimized inner product.

Because reduced evaluation is forbidden when setup is deferred, the
reduced-evaluation branch
does not cache a Stage-3 `SetupContributionPlan` and does not consume
`setup_prefix_eval`.

### Planner integration

Candidate materialization receives the selected per-level relation mode before
it computes:

- `WitnessLayout`;
- outgoing live and padded witness length;
- source moments and typed component counts;
- candidate A/B/D ranks that depend on the successor geometry;
- commitment payload bytes;
- EOR and Stage-2 domain sizes;
- terminal input length;
- complete suffix proof bytes.

The earlier diagnostic prototype that merely toggled quotient inclusion in a
scalar length formula is useful for proof-size intuition, but is not a valid
production architecture. Production planning MUST put the mode in the typed
candidate and replay the exact generated row through the same layout and proof
accounting as runtime.

The suffix DP SHOULD branch as follows:

1. In `QuotientPrefix`, materialize the existing quotient candidate.
2. If the current state is eligible, also materialize a reduced-evaluation
   candidate with the same independent geometry choices.
3. Recurse from the reduced-evaluation candidate with
   `RingRelationPhase::ReducedEvaluationSuffix` and no offloaded child.
4. In `ReducedEvaluationSuffix`, materialize only reduced-evaluation candidates and
   suppress setup-prefix search.
5. Retain candidates through the existing complete objective and
   parent-observable frontier.

The opening method, payload mode, security route, ring dimensions, split,
slice count, chunking, and relation mode remain independent decisions where
the feature matrix permits them. Candidate generation SHOULD enumerate the
relation mode outside low-level split loops when both modes share the same
geometry domain, so it does not duplicate expensive matrix derivation before
the witness length differs.

### External schedule artifacts and reports

Each external artifact row MUST store the exact relation mode for every root
and recursive fold as part of its complete `FoldSchedule`. Artifact encoding
MUST represent both `QuotientLift` and `ReducedEvaluation` explicitly; no
ambient config or compiled default may supply either mode.

`catalog_policy_report` SHOULD add `rel=quotient` or `rel=reduced-evaluation` to
each nonterminal level. It SHOULD report:

- selected cutover level or `none`;
- ordinary quotient coefficient count removed per level;
- compression quotient coefficient count removed per level;
- input and output witness lengths;
- payload mode;
- opening method;
- Linf or L2 route;
- incoming setup-prefix presence;
- setup-direct and Stage-3 proof bytes.

Dense fp32, fp64, and fp128 evidence MUST compare the baseline and regenerated
external artifacts row by row. An aggregate proof-size delta without per-level
witness geometry is insufficient.

## Transcript and serialization contract

### Effective schedule binding

The proof does not serialize `RingRelationMode`. The mode is public
configuration selected by the trusted schedule. `PlanSection` binds the digest
of `FoldSchedule::append_descriptor_bytes`, which in turn includes each
`CommittedGroupParams` descriptor. Adding the mode there binds it into the
transcript preamble before any commitment or challenge whose meaning depends
on it.

Changing only the mode MUST change:

- the level canonical descriptor;
- the complete schedule descriptor;
- the effective schedule digest;
- external catalog row identity and digest;
- the derived witness length and proof shape when quotient rows are nonempty.

It MUST NOT change a commitment to an already frozen source group whose
commitment profile is independent of the consuming fold’s relation mode.

### Challenge order

The order remains:

```text
bind public instance and effective schedule
bind current commitments and outgoing witness commitment
[grind at the scheduled ring-switch-alpha site, if enabled]
sample alpha
sample tau0
sample tau1
run Stage 1 and Stage 2 transcript events in their existing order
```

Reduced evaluation uses `alpha` only after the outgoing witness is fixed. The
dense prover table is derived after `alpha`; it is not another witness that
needs commitment.

No new transcript query label or domain separator is introduced. The existing
ring-switch `alpha`, `tau0`, and `tau1` query labels remain unchanged; mode
separation comes from the effective schedule digest already absorbed in the
instance preamble. Prover and verifier MUST use the same bumped descriptor
epoch and canonical parameter bytes before reaching those existing labels.

### Compatibility

This is a breaking protocol and artifact change. The implementation MUST
regenerate every affected schedule table and any trusted schedule artifact.
It MUST reject old rows under the new catalog identity. It MUST NOT add a
legacy decoder, descriptor fallback, or implicit default based on a missing
wire field.

The proof serialization schema need not gain a field, but exact proof bytes
change because recursive witness and commitment shapes change.

## Performance model

### Verifier

Let:

- `S` be the active direct setup coefficient count;
- `D` be the sum of distinct native role dimensions prepared at the terminal
  Stage-2 point;
- `C` be the number of non-setup public sparse coefficients actually queried.

Then the reduced-evaluation verifier target is:

\[
O(S+D+C)
\]

extension-field work and `O(D)` auxiliary extension-field storage. Current
quotient lifting is `O(S+D+C)` as well: it prepares alpha powers, scans the
same setup, evaluates structured terms, and reads quotient-tail MLE weights.
Reduced evaluation replaces quotient-tail work with `H` preparation and changes
the coefficient multiplier used during the scan.

Concrete benchmarking MUST separate:

- coefficient-functional preparation;
- structured group evaluation;
- direct setup scan;
- quotient-tail evaluation, which is zero in reduced-evaluation mode;
- complete Stage-2 verifier time;
- total verifier time.

The primary verifier acceptance condition is asymptotic and architectural: no
witness-sized functional table and no extra setup scan. Concrete regressions
must be reported before expanding eligibility earlier than this tail scope.

For the initial tail rollout, the quantitative acceptance budget is a maximum
25% increase in either single-threaded or multi-threaded median total verifier
time for every production profile case. The comparison MUST use the matching
merge-base and head feature graphs, one discarded warmup, three measured runs,
and interleaved executions on the same runner. Any case above that budget
blocks merge unless this specification records a reviewed exception. The phase
table is diagnostic evidence for that decision: reduced levels MUST report zero
quotient-tail time, and the setup-scan phase MUST remain the single fused scan.

### Prover

For Stage-2 witness domain `W`, the reduced-evaluation implementation may use:

| Mode | Relation-weight work | Extra extension state |
|---|---:|---:|
| Quotient lift | `O(W)` factored Stage 2 plus quotient construction/decomposition | `O(d0 + W/d0)` plus quotient witness |
| Reduced evaluation, dense | `O(W)` dense-table generation and folding | `O(W)` ephemeral |

This specification accepts the dense prover cost. It does not accept
accidentally computing both the quotient and reduced-evaluation table. Whole-fold
benchmarks MUST show quotient construction and quotient digit emission at zero
in reduced-evaluation mode.

Future optimizations may explore:

- streaming and recomputation;
- checkpointing after one or more coefficient rounds;
- one kernel per distinct sparse challenge;
- two carry-state factors for equality tensors;
- a rank-two Stage-3 setup product.

Those are alternatives behind the same reduced-evaluation semantics. They MUST
not add planner-visible proof modes unless they change proof bytes or verifier
behavior.

### Proof size

Reduced evaluation adds zero proof fields. Its structural saving at one fold is
the effect of removing the quotient digits from that fold’s outgoing witness:

```text
ordinary removed coefficients
    = quotient_depth * sum(native ordinary row dimensions)

compressed-only removed coefficients
    = quotient_depth * sum(native F/H quotient row dimensions)
```

The downstream byte saving is not just those coefficients times one byte. A
smaller witness can change:

- successor Boolean capacity and zero suffix;
- A/B/D ranks;
- digit depths;
- compression payloads;
- Linf/L2 route choice and norm-proof shape;
- fold count and terminal response;
- grinding nonce pricing, if enabled.

Therefore only the complete planner schedule estimate and serialized proof
benchmark count as proof-size evidence.

### Planner complexity

The implementation MUST record, for baseline and head:

- raw relation-mode transitions considered;
- reduced-evaluation transitions rejected by each eligibility rule;
- suffix calls and memo hits by relation phase;
- peak memo entries under the existing direct/prefixed quotas;
- frontier candidate counts;
- wall time and peak resident memory for dense fp32, fp64, and fp128 generation;
- the selected schedule descriptor and proof bytes.

The implementation MUST NOT increase `MAX_SUFFIX_SEARCH_CACHE_ENTRIES` as part
of enabling the feature. If generation no longer fits the existing bound, the
implementation must improve state sharing or candidate traversal before the
feature is accepted.

## Evaluation

### Acceptance criteria

#### Algebra and soundness

- [x] The linear residue recurrence matches literal negacyclic reduction for
      random powers-of-two dimensions, public multipliers, witnesses, and
      `alpha` values.
- [x] The terminal `H` recurrence matches an independently materialized
      residue-kernel MLE at random coefficient points.
- [x] The fused direct setup scan matches independent dense public-matrix
      oracles for A, B, and D across mixed dimensions, groups, chunks, and the
      configured extension field. The separately owned compression program
      matches an independent dense F/H oracle.
- [x] Quotient-lift and reduced-evaluation modes produce the same scalar relation
      claim on identical valid witnesses.
- [x] A nonzero reduced residual is rejected in reduced-evaluation mode without a quotient
      witness.
- [x] Tests cover `alpha^d + 1 == 0` in a field where such a test point is
      constructible, or explain why the test field lacks one. No division is
      used.

#### Eligibility and schedule binding

- [x] Reduced evaluation is rejected at root L0.
- [x] Reduced evaluation is rejected at recursive L1.
- [x] Reduced evaluation is accepted at L2 or later when the complete suffix has
      no setup prefix and uses evaluation trace.
- [x] A reduced-evaluation fold with an incoming setup prefix is rejected.
- [x] A schedule that returns from reduced evaluation to quotient lifting is
      rejected.
- [x] A schedule that adds an offloaded setup edge after the cutover is
      rejected.
- [x] A reduced-evaluation coefficient-packing fold is rejected.
- [x] Changing only the relation mode changes the effective schedule digest
      and transcript preamble.
- [x] Prover and verifier reject a proof replayed under the other mode’s
      schedule. The end-to-end cross-mode fixture independently plans valid
      quotient-only and reduced-suffix rows for the same committed profile,
      proves both, verifies both honestly, and rejects both replay directions.

#### Witness and proof shape

- [x] A raw reduced-evaluation layout contains exactly Z/E/T and no R range.
- [x] A compressed reduced-evaluation layout contains Z/E/T and F/H digits, but no
      ordinary or compression R range.
- [x] Stage 1 domain, Stage 2 domain, outgoing commitment length, response
      model, and proof estimate all equal values derived from the same
      mode-aware `WitnessLayout`.
- [x] Reduced-evaluation mode never calls quotient construction or quotient digit
      decomposition; an operation counter or focused mock test proves this.
- [x] Reduced-evaluation mode adds no serialized proof field or sumcheck round.

#### Feature combinations

- [x] Raw/Linf/evaluation-trace reduced-evaluation suffix.
- [x] Compressed/Linf/evaluation-trace reduced-evaluation suffix with F/H relations.
- [x] Raw/L2/evaluation-trace reduced-evaluation suffix.
- [x] Compressed/L2/evaluation-trace reduced-evaluation suffix.
- [x] Small-field evaluation trace with EOR. The fp32 dense end-to-end driver
      selects a shipped reduced suffix with EOR, verifies the honest proof, and
      rejects both an absent EOR proof and a tampered EOR partial.
- [x] Mixed A/B/D dimensions.
- [x] A multi-chunk eligible fold, if any production or focused fixture reaches
      level 2 with more than one chunk; otherwise a constructed type-level
      fixture covers it.
- [x] Each forbidden matrix cell has a negative schedule-validation test.

#### Planner

- [x] The planner may choose no cutover, an L2 cutover, or a later cutover from
      the same exact search engine.
- [x] Every complete candidate contains at most one quotient-to-reduced-evaluation
      transition.
- [x] A small exhaustive oracle enumerates all `m + 1` monotone cutovers and
      matches the suffix DP’s selected complete descriptor.
- [x] Reversing relation-mode traversal order does not change the selected
      descriptor. The regression runs the complete planner twice with fresh
      memo state and compares exact proof bytes and canonical descriptors.
- [x] Reduced-evaluation suffix states cannot invoke setup-prefix candidate search.
- [x] Existing suffix-cache quotas remain unchanged.
- [x] Artifact row replay recomputes the exact reduced-evaluation witness lengths and
      proof estimate.
- [x] Catalog identity and policy reports include relation mode and cutover.

#### Verifier performance and safety

- [x] The reduced-evaluation verifier allocates `O(d)` coefficient-functional state, not
      `O(W)` witness-sized state.
- [x] The reduced-evaluation verifier uses the existing fused setup traversal and scans
      each active setup coefficient once.
- [x] Benchmarks separately report coefficient-functional preparation,
      structured groups, direct setup scan, quotient-tail evaluation, complete
      Stage 2, and total verifier time. The profile harness performs one extra
      untimed honest replay after measured verification, captures the exact
      production spans by relation mode, and renders them beside the unchanged
      public total. The same tracing layer is shared with the focused phase
      benchmark so the two paths cannot silently redefine phase ownership.
- [x] A bounded deterministic property matrix rejects malformed mode,
      dimension, row, point, and setup combinations without panic or unbounded
      allocation. It covers both modes and every nonempty combination of the
      five mutation categories through public replay, preparation, and
      evaluation boundaries.

#### Non-normative performance evidence

- [x] Regenerate every affected schedule table with
      `scripts/generate-schedule-tables.sh`.
- [x] Record base/head proof-size deltas for dense fp32, fp64, and fp128 in the
      implementation PR, pinned to the compared commits. These measurements are
      review evidence, not compatibility baselines.
- [x] Serialize representative proofs and confirm that measured sizes remain
      within the generated proof estimates. Profile benchmark run
      [33823315475](https://github.com/LayerZero-Labs/akita/actions/runs/33823315475)
      completed all 13 production cases at review head `1d2800432`; the runtime
      harness serializes each proof and rejects any result above its planned
      byte count.
- [x] Report representative planner wall time, peak resident memory, and
      available search counters separately from compilation time. The PR
      evidence report compares release binaries for dense fp32/fp64/fp128 rows
      and records wall time, maximum RSS, per-phase suffix and memo counts,
      transition and rejection counts, peak memo occupancy, candidates, and
      selected cutovers at code-and-evidence head `04111dedf`.

For quotient-free-tail acceptance, exact generated rows, cutovers, witness
lengths, and proof byte counts MUST NOT be checked in as golden evidence. They
are planner outputs and may change when the protocol, objective, security
model, or search implementation improves. The checked-in external catalog
artifacts and their normal admission validation remain the source of truth.

### Testing strategy

#### Algebra tests

Add independent reference tests under `akita-algebra`:

1. Construct random `A` and `W` in `R_d`.
2. Compute `A circledast W` by the existing cyclotomic ring implementation.
3. Evaluate the result at `alpha`.
4. Generate `kappa` by recurrence and compute `sum_j kappa_j w_j`.
5. Compare both values.
6. Compare `H` recurrence against literal `sum_j eq(r,j) Phi(k,j)`.

Use every dispatch dimension exercised by fp32, fp64, and fp128 schedules. Add
small exhaustive dimensions when they make wrap and carry failures easier to
localize.

#### Shared-layout tests

Extend `akita-types` relation and witness tests to build the same geometry in
both modes. Assert exact segment ranges and live length. Cover raw and
compressed payloads, mixed row dimensions, alignment boundaries, and no-R
access errors.

#### Prover/verifier equivalence tests

Focused fixtures SHOULD expose both modes under the same public relation even
when production eligibility would reject the early level. Algebra equivalence
belongs in the relation engine; schedule eligibility belongs in separate
tests. End-to-end PCS tests MUST use only eligible L2+ tail schedules.

Tamper tests should modify:

- one Z/E/T digit;
- one retained compression digit;
- one public setup coefficient view;
- one sparse challenge coefficient;
- the row order or `tau1` point;
- the trusted mode or schedule digest.

Each tamper must reject under the verifier without relying on an absent
quotient range.

#### Planner tests

Add a small unpruned relation-cutover oracle beside the existing suffix-search
oracles. It should enumerate the monotone cutover index explicitly, construct
all feasible schedules through canonical materializers, and compare the exact
complete objective and descriptor with production suffix DP.

Property tests should vary:

- number of eligible tail folds;
- setup-prefix disappearance point;
- compressed-to-raw cutover;
- Linf/L2 route availability;
- role dimensions and bases;
- EOR availability;
- traversal order and memo capacity.

#### Repository gates

The implementation must run the CI preflight and feature-graph commands from
`AGENTS.md`. Protocol, Book, or spec edits additionally run:

```bash
./scripts/check-doc-guardrails.sh
scripts/check-spec-references.sh --all
```

Generated-table drift and profile-specific workflows are required when their
source files change.

### Performance evidence policy

The implementation PR SHOULD record the exact compared SHAs, commands,
aggregate proof-size deltas, representative serialized-proof agreement, and
representative planner wall/RSS measurements. Detailed per-row reports MAY be
attached to the PR or retained as local review output, but they MUST NOT become
repository compatibility fixtures. Performance evidence must distinguish
compilation time from execution time and state whether figures are estimates
or serialized measurements.

## Alternatives considered

### Keep quotient lifting everywhere

This preserves the current factored prover and mature implementation. It also
keeps paying quotient construction, decomposition, range checks, commitments,
and later folds when quotient rows dominate the remaining tail witness. It is
the retained baseline, not the selected new feature.

### Enable reduced evaluation at every fold

The verifier can perform reduced evaluation efficiently even for dense setup, but
the generic prover loses the current compact rank-one relation table. Early
folds also contain more lanes and challenges while quotient rows are a smaller
fraction of the recursive witness. This feature therefore starts at L2
and only in a setup-direct tail. Earlier activation requires new benchmarks and
a scope revision.

### Independent per-level mode bits

This permits `QuotientLift`/`ReducedEvaluation`/`QuotientLift` oscillation and creates `2^m` mode
sequences over an `m`-fold tail. No protocol advantage requires switching back
after quotient rows have been removed. The monotone cutover is easier to
validate, plan, report, and optimize.

### A fixed cutover after two or three folds

A fixed threshold is useful for diagnostics and produced the initial
proof-size preview. It is not a durable planner policy: witness geometry,
payload compression, L2 availability, and fold count vary by family and row.
The production planner searches the one cutover under its complete objective,
while eligibility fixes only the lower bound `level >= 2`.

### Treat reduced evaluation as a Boolean sizing toggle only

Subtracting R widths from a planner estimate does not update source moments,
downstream ranks, commitment geometry, terminal response, generated replay, or
verifier semantics. The mode must be a typed schedule and layout input.

### Retain compression quotient rows

This would reduce implementation work but makes reduced evaluation depend on
payload mode and leaves a material quotient tail precisely when commitment
compression is selected. Compression relations are public-linear ring
relations too. The implementation therefore covers them.

### Add packing-specific reduced evaluation now

The mathematics applies to the packing consistency relation over its smaller
modulus and coordinate planes. Current schedules use coefficient packing only
at L0 and L1, while reduced evaluation is forbidden there. Implementing the
combination would add untestable production surface and weaken the desired
scope boundary. It is deferred.

### Force setup offloading and use rank-two Stage 3

Keeping the setup coefficient as an independent Stage-3 axis admits an exact
two-carry-state factorization and avoids dense prover setup kernels. It is a
valuable future optimization when Stage 3 is already selected. Forcing a new
setup prefix in the tail adds a subproof and successor group solely to avoid
prover state, outside this feature’s proof-size and verifier-first scope.

### Stream or checkpoint the dense reduced-evaluation prover table

Streaming can reduce extension-field memory but may rescan public sources in
each coefficient round. Checkpointing gives intermediate time-memory points.
Neither changes proof bytes or verifier behavior. This implementation
uses one dense oracle behind an abstraction that can admit these optimizations
later.

### Add a prover-cost coordinate to the planner now

The current complete objective does not price prover time. Adding an
uncalibrated penalty would make catalog selection harder to audit and could
hide proof-size improvements. The implementation measures prover cost and
keeps the initial eligibility conservative. A future multi-objective policy
may add an explicit versioned coordinate with measured evidence.

## Open implementation risks

### Compression map transpose

The ordinary A/B/D setup path already exposes public ring coefficients in
canonical geometry. Compression F/H uses a separate compact map authority.
The implementation must prove that its reduced transpose uses the exact same
map, row order, native modulus, and witness digit addresses. This is the most
likely place for a correct ordinary path and an incorrect compressed path to
diverge.

Mitigation: land scalar reference oracles and compressed equivalence tests
before optimizing the dense reduced-evaluation table compiler.

### Common-factor assumptions in Stage 2

`RelationRangeImageProver`, `PreparedRelationPoint`, and verifier terminal
evaluation currently assume a common low alpha factor. Reduced evaluation breaks
that rank-one assumption. Leaving one outer multiplication or one lane-power
projection in the reduced-evaluation branch can produce a plausible but incorrect
relation.

Mitigation: make the relation-weight representation and prepared coefficient
functional typed enums, and compare full materialized tables in debug tests.

### Setup scanner duplication

A naïve reduced-evaluation implementation can accidentally build `H`, materialize one
kernel per setup lane, or rescan A/B/D separately. Any of these defeats the
verifier design even though the proof remains correct.

Mitigation: extend `SetupContributionPlan` first and benchmark its existing
fused traversal with power and H functionals before integrating the full
protocol.

### Hidden quotient work below witness construction

The current D and compression executors can compute cyclic and negacyclic
products together and derive quotients before witness emission decides what to
keep. A layout-only cutover would save proof bytes while retaining most of the
prover work and exact-NTT cache pressure.

Mitigation: introduce negacyclic-only reduced-mode product paths and make NTT
requirements depend on `RingRelationMode`. Assert that quotient-only kernels,
cyclic transforms, and centered quotient-tail caches are absent before
collecting prover benchmarks.

### Planner state widening

Removing quotient rows changes witness length, response moments, and later
candidate geometry. The relation phase itself is only monotone, but those new
lengths can expose suffix states that did not exist in the baseline.

Mitigation: preserve cache quotas, add phase-specific diagnostics, compare an
unpruned small oracle, and publish dense-family wall/RSS evidence.
