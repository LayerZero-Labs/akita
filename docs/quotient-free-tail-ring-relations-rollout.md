# Quotient-free tail implementation rollout

This is a non-normative implementation record for
[`quotient-free-tail-ring-relations.md`](../specs/quotient-free-tail-ring-relations.md)
and its
[`implementation contract`](../specs/quotient-free-tail-ring-relations-implementation.md).
It may record slice order, review routing, and exact dependency heads. Those
details guide the active PR but do not define protocol acceptance or security
semantics.

## Execution plan

### Slice 0: restack and correct the active specification

- Stack this PR on the exact #444 head, whose first parent stack contains the
  selected #448 transcript-grinding head.
- Refresh concurrent-PR references and mark this specification active.
- Record the offset-aware terminal equality-window requirement and the ban on
  hidden quotient-only product or NTT work.

Exit condition: the specification branch has the intended first parent and no
known code/spec mismatch remains before implementation.

### Slice 1: protocol type, binding, and generated schema

- Add `RingRelationMode` and the one-per-fold `CommittedGroupParams` field.
- Bind its stable tag into level and schedule descriptors.
- Bump the instance descriptor epoch.
- Carry the field through generated rows, expansion, emission, and catalog
  identity without changing the existing schedule choices.

Exit condition: descriptors and all shipped catalog identities distinguish the
two modes, while every existing row replays explicitly as `QuotientLift`.

### Slice 2: mode-aware witness layout authority

- Replace the implicit always-present quotient tail with a typed lifted or
  reduced relation layout.
- Remove ordinary and compression quotient ranges in reduced-evaluation mode.
- Route successor length, proof sizing, source moments, and range access through
  the same layout authority.
- Add complete `FoldSchedule` eligibility and monotone-suffix validation.

Exit condition: typed layout and schedule tests distinguish both modes, all
malformed sequences reject, and no planner-only quotient toggle remains.

### Slice 3: shared residue algebra

- Add the residue recurrence and offset-aware terminal-kernel recurrence.
- Consume exact checked equality weights for each physical native window.
- Add independent quadratic references and malformed-input tests.

Exit condition: algebra oracles agree across supported dimensions, offsets,
and mixed-window fixtures without prover/verifier copies.

### Slice 4: verifier coefficient functional and fused setup scan

- Generalize prepared native coefficient functionals.
- Extend `SetupContributionPlan` to evaluate power or terminal-kernel weights
  through the same fused scan.
- Add reduced structured-challenge and compression-map terminal evaluation.

Exit condition: verifier-focused dense oracles pass for raw, compressed,
mixed-dimension, and unaligned fixtures with one setup scan and bounded
auxiliary state.

### Slice 5: verifier protocol integration

- Add exhaustive mode dispatch to `RelationMatrixEvaluator`.
- Reject deferred setup claims in reduced-evaluation mode.
- Remove quotient-tail evaluation and the common-alpha outer factor from the
  reduced branch.
- Add transcript-order, schedule-digest, tamper, and no-panic tests.

Exit condition: the verifier accepts reduced scalar fixtures and rejects every
cross-mode or malformed replay without a proof-format field.

### Slice 6: zero-quotient prover substrate and NTT requirements

- Select negacyclic-only D and compression product paths before quotient work.
- Skip ordinary and compression quotient construction and emission.
- Remove relation-cyclic and quotient-tail-only transforms and caches from the
  reduced-mode NTT requirement set.

Exit condition: diagnostics prove zero quotient construction, decomposition,
cyclic-only transforms, and quotient-only cache preparation in reduced mode.

### Slice 7: dense Stage-2 prover oracle

- Introduce the canonical factored-or-dense relation-weight oracle.
- Compile all ordinary and compression reduced weights into the dense variant.
- Integrate it with the existing fused range-image/relation sumcheck.
- Preserve evaluation-trace/EOR structured terms and negative-binary terms.

Exit condition: quotient-lift and reduced-evaluation proofs agree on valid
relations and the declared feature matrix passes end to end.

### Slice 8: exact planner cutover

- Add `RingRelationPhase` to suffix state and memo keys.
- Enumerate the one-way cutover and suppress later setup-prefix search.
- Price exact mode-aware witness shapes, source moments, proof bytes, and
  grinding nonce streams.
- Add the small exhaustive cutover oracle and phase diagnostics.

Exit condition: traversal order does not change selection, cache quotas remain
unchanged, and generated replay matches planner estimates.

### Slice 9: generated schedules, evidence, and documentation

- Regenerate affected catalogs only after the planner and proof shapes settle.
- Produce dense fp32/fp64/fp128 proof-size and verifier-phase evidence.
- Record planner wall time, peak RSS, search counters, and prover quotient-work
  counters.
- Update the Book after behavior and evidence are stable.

Exit condition: checked evidence supports the proof-size, verifier architecture,
zero-quotient-work, and bounded-search claims.

Optional prover optimizations follow profiling and are not required for initial
acceptance. They MUST preserve the shared algebra oracle and verifier equation.

## Pull-request landscape and stacking plan

This section records the Akita branches refreshed on 2026-08-29. SHAs are
included so the recommendation does not silently apply to later rewrites.

### Transcript grinding PR 448

PR [#448](https://github.com/LayerZero-Labs/akita/pull/448), head
`303ddbca548c788e5ec7fbd74b5a679bd269d8cd`, is open. It
changes transcript query sites around `alpha`, `tau0`, and `tau1`; proof-level
nonce serialization; planner cost composition; suffix-DP state and frontiers;
schedule estimates; generated catalogs; and both ring-switch implementations.

This feature has no mathematical dependency on grinding, but it has a strong
code, transcript, and proof-cost dependency. The implementation branch is now
explicitly stacked in the intended merge order:

```text
#448 transcript grinding @ 303ddbca5
  -> #444 q128 SIS widening @ b5326abf8
    -> #445 quotient-free relations
```

The #445 spec replay commit has `b5326abf8d7311b13c9f1146d0a515e549c64e9a`
as its sole parent. Later slices MUST preserve this order or restack on the
corresponding newer exact heads after refreshing all three PRs.

### Suffix EOR and packed prover stack

The accepted packed recursive-witness cutover from
[#437](https://github.com/LayerZero-Labs/akita/pull/437) is already present in
the current stack as `4eb6b0128`. The accepted commitment-stage refactor from
[#441](https://github.com/LayerZero-Labs/akita/pull/441) is also present as
`4a6897c9b`. PR [#439](https://github.com/LayerZero-Labs/akita/pull/439), head
`fb4fa643b22953f90085d919e31c006105b5cf51`, adds packed dense prover storage.

The merged changes do not alter the reduced-evaluation verifier equation, but
they define the witness and commitment ownership model this implementation
MUST use. The open dense-storage PR intersects the baseline dense prover slice;
refresh it before that slice. The reduced-evaluation oracle may remain unpacked
extension-field scratch, while the compact recursive witness follows the
accepted packed ownership model.

### Trusted schedule artifacts PR 428

PR [#428](https://github.com/LayerZero-Labs/akita/pull/428), head
`d6499748e121851b1fcc5967256dff3403f59d0e`, is open, behind main, and
review-required. It changes runtime schedule authority, trusted artifacts,
schedule resolution, descriptors, generated catalogs, witness types, and
verifier setup boundaries.

The reduced-evaluation mode must be authenticated by whichever schedule authority
lands. It has no need to stack on the current behind head. If #428 lands first,
Slice 1 must add the mode to its trusted artifact schema and validation. If it
does not, current effective-schedule digest and generated-row authority remain
the integration target.

### Certified planner spec PR 434

PR [#434](https://github.com/LayerZero-Labs/akita/pull/434), head
`d7261d9167667ce71a38152a2f5d7d3867cdb621`, is a draft documentation PR. It
does not supply implementation code, but its audited-decision-domain rule is
directly relevant. This spec follows that rule: the cutover is an exact
decision, traversal order is guidance, and no candidate is removed without a
complete dominance proof.

There is no code-stack dependency. If #434 is approved, the implementation PR
should cite its planner architecture and add relation phase to the documented
state sufficiency and oracle tests.

### Grouped planner PRs 409 and 412

PR [#409](https://github.com/LayerZero-Labs/akita/pull/409), head
`5bef0c1a54d7ac7c8718e4a7aca803f64f83f24b`, plans exact precommit profiles.
PR [#412](https://github.com/LayerZero-Labs/akita/pull/412), head
`733efbea094e02756b88aa8662f37323198e9f9f`, changes grouped-root planner
scaling and several suffix candidate files.

Reduced evaluation is forbidden at the root, and current recursive suffixes contain no
frozen precommitted group, so these PRs are not semantic prerequisites. Their
planner file overlap argues for rebasing the planner slice after any accepted
planner stack, not for stacking the verifier or algebra work on them.

### Commitment-stage PR 441

PR [#441](https://github.com/LayerZero-Labs/akita/pull/441), head
`4fb5264b3be6e076a925ce88ba837932a2940ed9`, stacks a prover commitment-stage
refactor on the packed recursive witness branch. It may change where the
smaller reduced-evaluation witness is committed, but not how its relation is verified.
Treat it as a prover integration surface, not a protocol dependency.

### Recommended stack shape

```text
codex/quotient-free-tail-relations
  |-- #448 transcript grinding exact head
  |-- #444 q128 SIS widening exact head
  |-- specification and shared protocol/type slices
  |-- implement verifier, prover, and exact planner cutover on that base
  `-- generate catalogs, evidence, Book updates, and the review-ready PR
```

Keep the implementation as reviewable commits on this PR. Do not stack the
branch on every open feature branch. Re-evaluate the exact #448 and #444 heads
before later slices and restack only when either chosen lower dependency moves.

## Documentation plan

The active spec owns the in-flight design. It intentionally does not cite
an unpublished Akita paper or require a private research note. The Book must
explain the feature from code and approved specifications once implementation
lands.

Expected durable destinations are:

- `book/src/how/proving/akita-fold-realizations.md`: quotient-lift and
  reduced-evaluation realizations, witness shapes, and cutover;
- `book/src/how/verifying/matrix_evaluation.md`: terminal residue kernel and
  fused setup scan;
- `book/src/how/proving/sumcheck-stages.md`: Stage-2 equation in both modes;
- `book/src/how/configuration.md`: planner cutover and supported feature
  matrix;
- `book/src/how/security.md`: reduced-residual soundness statement and
  unchanged Linf/L2 boundary.

When the implementation and Book updates land, mark this spec `implemented`.
Archive it after the durable content is fully folded, following
[`specs/PRUNING.md`](PRUNING.md).

## Reviewer map

| Review concern | Primary current files |
|---|---|
| Protocol mode and schedule binding | `crates/akita-types/src/layout/params.rs`, `layout/params/descriptor.rs`, `schedule.rs`, `instance_descriptor/mod.rs` |
| Semantic rows and physical layout | `crates/akita-types/src/proof/relation_layout.rs`, `proof/relation.rs`, `witness.rs`, `witness/scalar_len.rs` |
| Shared residue algebra | `crates/akita-algebra/src/ring/` |
| Prover quotient removal | `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, `ring_switch/coeffs.rs` |
| Prover Stage-2 weights | `crates/akita-prover/src/protocol/ring_switch/relation_weights/`, `sumcheck/relation_range_image/` |
| Verifier terminal MLE | `crates/akita-verifier/src/protocol/ring_switch/prepared_relation_point.rs`, `relation_evaluation.rs` |
| Fused direct setup scan | `crates/akita-types/src/setup_contribution/plan/` |
| Compression reduced transpose | `crates/akita-types/src/proof/compression_relation_weights.rs`, prover/verifier ring-switch compression paths |
| Planner state and cutover | `crates/akita-planner/src/schedule_params/suffix_dp/`, recursive candidate materialization, response model |
| Generated rows and identity | `crates/akita-schedules/src/generated/`, `catalog_identity.rs`, planner emitter and reports |
| Transcript grinding interaction | PR #448 ring-switch query sites, packed proof cost, and grinding plan |
| End-to-end protocol tests | `crates/akita-pcs/src/scheme/tests/`, `crates/akita-pcs/tests/protocol_soundness.rs` |
