# Quotient-free tail implementation rollout

This is a non-normative implementation record for
[`quotient-free-tail-ring-relations.md`](../specs/quotient-free-tail-ring-relations.md)
and its
[`implementation contract`](../specs/quotient-free-tail-ring-relations-implementation.md).
It may record slice order, review routing, and exact dependency heads. Those
details guide the active PR but do not define protocol acceptance or security
semantics.

## Current audit status

As of 2026-09-06, slices 0 through 9 are present and re-audited in PR #466 at
code-and-evidence head `a890d7bfb`. The audit includes the post-`04111dedf`
planner relation-state simplification, schedule validation, witness-tail
construction, prover/verifier plumbing, relation-phase evidence checks, and
documentation guards. The mode is bound, layouts omit both ordinary and
compression quotients, shared residue algebra drives the prover and verifier,
production proofs exercise eligible reduced suffixes, and the exact planner
emits reduced external artifact rows. Cross-mode replay, small-field reduced EOR,
reversed traversal, production-profile verifier phase timing, the bounded
malformed-input matrix, serialized-proof agreement, and final-head planner
telemetry are complete. The external catalogs and durable Book explanations
are present. Aggregate base/head evidence belongs in the PR rather than a
checked compatibility snapshot.

This distinction is intentional: the protocol feature is implemented, while
the record remains `active` until its stated validation and evidence gates are
closed.

## Execution plan

### Slice 0: restack and correct the active specification

- Stack this PR on the exact #444 head, whose first parent stack contains the
  selected #448 transcript-grinding head.
- Refresh concurrent-PR references and mark this specification active.
- Record the offset-aware terminal equality-window requirement and the ban on
  hidden quotient-only product or NTT work.

Exit condition: the specification branch has the intended first parent and no
known code/spec mismatch remains before implementation.

### Slice 1: protocol type, binding, and external artifact schema

- Add `RingRelationMode` and the one-per-fold `CommittedGroupParams` field.
- Bind its stable tag into level and schedule descriptors.
- Bump the instance descriptor epoch.
- Carry the field through external rows, expansion, emission, and catalog
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

### Slice 9: external schedule artifacts, evidence, and documentation

- Regenerate affected catalogs only after the planner and proof shapes settle.
- Produce dense fp32/fp64/fp128 proof-size and verifier-phase evidence.
- Record planner wall time, peak RSS, search counters, and prover quotient-work
  counters.
- Update the Book after behavior and evidence are stable.

Exit condition: PR-attached evidence supports the proof-size, verifier
architecture, zero-quotient-work, and bounded-search claims without turning
exact planner outputs into repository compatibility fixtures.

Optional prover optimizations follow profiling and are not required for initial
acceptance. They MUST preserve the shared algebra oracle and verifier equation.

## Pull-request lineage

### Active implementation PR

PR [#466](https://github.com/LayerZero-Labs/akita/pull/466) replaces closed,
unmerged PR #445. Review checkpoint `1d2800432` contained the 127-commit
restack from merge base `26bdbac79`; the 2026-09-06 re-audit now covers every
later implementation and evidence change through `a890d7bfb` against merge
base `f9f7de87b`.

PR #466 targets `main` directly. It is not an active stack on #448, #444, or
#445, and those historical branch heads must not be used as its current
acceptance authority.

### Archived #445 stack

The original #445 rollout was assembled over the then-open #448 transcript
grinding and #444 q128 SIS widening branches. That ordering explained the
implementation history, but it was superseded when #466 restacked the complete
feature onto current main. The older exact heads remain recoverable from Git
history and the closed PR; repeating them here would make a stale stack look
normative.

Concurrent work such as certified planner documentation and grouped planner
changes remains an integration surface only. Refresh from main and re-run the
affected descriptor, planner, external-catalog, and verifier gates before those
changes land.

### Current branch shape

```text
main @ d1b224d80 (squash merge of #466)
  `-> codex/trusted-schedule-artifacts (PR #428)
      `-> external artifact ownership and regenerated `.aks` catalogs
```

### Trusted schedule artifacts PR 428

PR [#428](https://github.com/LayerZero-Labs/akita/pull/428) removes compiled
schedule rows in favor of explicitly supplied trusted artifacts. It is stacked
on the merged PR #466 result `d1b224d80`, serializes the full relation-aware
`FoldSchedule` in each external row, validates rows at admission, and uses one
scheme-owned catalog for setup, proving, and verification. No generated Rust
row schema or ambient resolver remains.

Do not reconstruct the obsolete #448 -> #444 -> #445 stack.

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
[`specs/PRUNING.md`](../specs/PRUNING.md).

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
| External rows and identity | `crates/akita-schedules/src/artifact.rs`, planner emitter and reports |
| Transcript grinding interaction | PR #448 ring-switch query sites, packed proof cost, and grinding plan |
| End-to-end protocol tests | `crates/akita-pcs/src/scheme/tests/`, `crates/akita-pcs/tests/protocol_soundness.rs` |
