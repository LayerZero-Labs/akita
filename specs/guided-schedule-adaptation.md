# Spec: Guided Schedule Adaptation

| Field         | Value                               |
|---------------|-------------------------------------|
| Author(s)     | Quang Dao                           |
| Created       | 2026-09-04                          |
| Status        | active                              |
| PR            | #472                                |
| Supersedes    |                                     |
| Superseded-by |                                     |
| Book-chapter  | book/src/how/configuration.md       |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

Akita's exhaustive schedule planner is an offline optimization tool. A
downstream application may nevertheless need to add a small set of exact
precommitted groups after it has selected a schedule for a much larger final
group. Guided schedule adaptation makes this operation bounded: it retains the
trusted scalar row's structural choices, rebuilds every group-dependent value,
and fails when that structure cannot support the grouped request.

The result is still an ordinary expanded schedule row. It becomes usable only
after the application admits its final row set through
`TrustedScheduleCatalog::try_new` and reprovisions setup for the resulting
catalog.

## Intent

### Goal

Provide a `Cfg`-free planner API that adapts one validated scalar
`ResolvedScheduleRow` to an exact `GroupedGenerationRequest` in bounded time
without weakening schedule audit, catalog ownership, or quotient-free
relation-mode constraints.

### Terms

- A **scalar row** has one final group and no precommitted groups.
- A **structural guide** is the subset of schedule choices retained from that
  scalar row.
- A **derived value** is a length, rank, bound, matrix width, relation shape,
  setup size, or proof cost that depends on the exact grouped request.
- An **adapted row** is the newly materialized grouped row before trusted-catalog
  admission.

### Invariants

- **Approved input.** Adaptation MUST accept only a scalar
  `ResolvedScheduleRow` that passes the canonical audit under the supplied
  `PlannerPolicy`.
- **Exact producers.** Each precommitted producer MUST bind one frozen
  `GroupCommitPhaseParams`, one `CommittedSourceContract`, and the matching
  `HonestFoldPolicySpec`. A mismatch MUST return a typed setup error.
- **Frozen main root.** The final group's root A/B matrices, blocks, slices,
  digit bases, and opening plan MUST match the scalar row. The fold-owned D
  matrix MUST retain its audited table and ring identity while its grouped input
  width and required rank are recomputed.
- **Frozen suffix structure.** Recursive depth, per-level dimensions, block
  splits, slice counts, digit bases, opening methods and challenges, payload
  modes, relation modes, source encodings, witness chunking, terminal shape, and
  direct-versus-offloaded setup topology MUST match the scalar guide.
- **Fresh derivation.** Witness lengths, live blocks, source moments, relation
  rows, matrix widths and ranks, setup-prefix lengths, terminal response bounds,
  grinding parameters, and proof/setup accounting MUST be rebuilt through the
  canonical planner primitives for the exact grouped key.
- **Absolute relation cutover.** A guide MUST retain the relation mode at each
  absolute fold level. Adaptation MUST NOT move, remove, or introduce a
  quotient-free cutover.
- **Fail closed.** If no candidate satisfies every guide constraint, adaptation
  MUST return `AkitaError::UnsupportedSchedule`. It MUST NOT fall back to an
  unconstrained topology or invoke exhaustive planning implicitly.
- **Bounded opening products.** Guided coefficient-packing search MUST reject
  more than 256 canonical precommit opening products before materializing the
  assignment vectors.
- **Final admission.** An adapted row MUST NOT bypass
  `TrustedScheduleCatalog::try_new`, challenge-hook validation, duplicate-key
  rejection, row identity, or catalog identity.
- **Oracle preservation.** `find_schedule` MUST remain the unconstrained full-DP
  correctness and proof-size oracle when no guide is supplied.
- **Offline only.** Setup restoration, commitment, proving, proof decoding,
  verification, and guest execution MUST NOT call either planner entry point.

### Non-goals

- A universal adapter that succeeds for every grouped request.
- Runtime schedule search or a process-global schedule registry.
- A fallback from guided search to full DP.
- A proof, transcript, statement, or `.aks` schema change.
- Authentication of a catalog subset or a new recursion membership proof.
- Selection of Jolt's reachable profile set or preprocessing representation.
- Preservation of a catalog digest after rows are added or replaced.

## Evaluation

### Acceptance criteria

- [x] `akita-planner` exposes `find_adapted_schedule` without depending on a
  concrete `CommitmentConfig`.
- [x] Plain-value producer construction validates the frozen descriptor and
  producer policy/contract agreement.
- [x] Tests prove that the main root and suffix structure remain frozen while
  grouped successor values are rebuilt.
- [x] Tests cover empty and mismatched requests, infeasible guides, recursive
  setup-prefix topology, and final trusted-catalog admission.
- [x] Guided opening-product limits are enforced before allocation and have a
  regression test.
- [x] The manual benchmark compares guided rows with full-DP rows, including
  28- through 50-variable final groups.
- [x] The implementation adds no verifier, transcript, proof-wire, or artifact
  schema path.

### Testing strategy

`crates/akita-planner/src/test/adapted_schedule.rs` owns end-to-end adapter
tests and the ignored quality benchmark. Candidate-level tests protect the
pre-allocation product bound. Existing unpruned-search and relation-order tests
continue to exercise the full planner with no guide.

The GitHub merge gates MUST run the two workspace nextest shards, all three
Clippy feature graphs, transcript modes, and external schedule-artifact drift.
Artifact drift MUST remain empty because the ordinary generated-family path
supplies no guide or product cap.

### Performance

Guided search has no platform-specific latency guarantee. It MUST bound the
precommit coefficient-packing product domain at 256 and MUST expose failures as
typed errors. The release benchmark records adaptation time and compares the
expanded proof-payload estimate against the full-DP oracle; proof-size equality
is measured evidence, not a correctness requirement.

## Design

### Architecture

`PrecommittedProducer::try_new` binds the producer declaration.
`GroupedGenerationRequest` derives the exact lookup key and producer fold
policies. `find_adapted_schedule` re-audits the scalar row and invokes the
canonical suffix DP with a root constraint and a schedule guide.

The guide narrows existing candidate domains rather than introducing a second
materializer. Root, recursive, setup-prefix, and terminal candidates continue
to flow through the same length, security, response, relation, and proof-cost
derivations as exhaustive planning. The output then flows through the existing
`ResolvedScheduleRow` and `TrustedScheduleCatalog` audits.

### Alternatives considered

- **Run full DP for every late grouped key.** This preserves global optimality
  but takes seconds for production-sized rows and defeats bounded build-time
  adaptation.
- **Copy the scalar schedule and patch its root groups.** This retains stale D
  width, witness, relation, setup, and response values and is therefore invalid.
- **Use one deliberately oversized generic row.** This avoids adaptation but
  pays recurring proof/setup overhead and obscures the exact producer profile.
- **Plan during runtime setup or verification.** This violates explicit
  external-catalog ownership and makes runtime acceptance depend on search.

## Documentation

The durable integrator contract lives in
[`book/src/how/configuration.md`](../book/src/how/configuration.md). This spec
remains active with PR #472 and can move to the archive after the implementation
lands and its remaining design value is folded into the Book.

`AGENTS.md` does not change because verifier reachability, feature flags, and CI
commands are unchanged. `docs/crate-graph.md` does not change because this PR
adds no workspace dependency edge.

## References

- [PR #472](https://github.com/LayerZero-Labs/akita/pull/472)
- [External schedule catalog ownership](external-schedule-catalog-ownership.md)
- [Quotient-free tail implementation](quotient-free-tail-ring-relations-implementation.md)
- [Setup offloading planner](setup-offloading-planner.md)
- `crates/akita-planner/src/planner.rs`
- `crates/akita-planner/src/schedule_params/suffix_dp/`
- `crates/akita-planner/src/test/adapted_schedule.rs`
