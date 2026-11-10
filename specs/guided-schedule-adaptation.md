# Spec: Guided Schedule Adaptation

| Field         | Value                               |
|---------------|-------------------------------------|
| Author(s)     | Quang Dao                           |
| Created       | 2026-09-04                          |
| Status        | active                              |
| PR            | #472; #373 (root opening ranking)    |
| Supersedes    |                                     |
| Superseded-by |                                     |
| Book-chapter  | book/src/how/configuration.md       |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

Akita's schedule planner is an offline optimization tool. A downstream
application may need to add a small set of exact precommitted groups after it
has selected a schedule for a much larger final group. Guided schedule
adaptation makes this operation fast in the common case:
it retains the trusted scalar row's structural choices and rebuilds every
group-dependent value. When that structure cannot support the grouped request,
adaptation falls back to the full planner for the same key. For a valid request,
adaptation fails only when both searches fail to return a schedule. Both use
the retained-assignment rule below, which can exclude a feasible or globally
optimal complete schedule.

The result is still an ordinary expanded schedule row. It becomes usable only
after the application admits its final row set through
`ValidatedScheduleCatalog::try_new`, binds it as `TrustedScheduleCatalog<Cfg>`,
and reprovisions setup for the resulting catalog.

## Intent

### Goal

Provide a `Cfg`-free planner API that adapts one validated scalar
`ResolvedScheduleRow` to an exact `GroupedGenerationRequest` without weakening
schedule audit, catalog ownership, or quotient-free relation-mode constraints.
Liveness takes priority over planning latency: a request that the full planner
can serve MUST NOT fail because the scalar row's structure does not fit it.

### Terms

- A **scalar row** has one final group and no precommitted groups.
- A **structural guide** is the subset of schedule choices retained from that
  scalar row.
- A **derived value** is a length, rank, bound, matrix width, relation shape,
  setup size, or proof cost that depends on the exact grouped request.
- An **adapted row** is the newly materialized grouped row before trusted-catalog
  admission.
- An **interchangeable class** is a set of precommitted groups with equal
  `GroupCommitPhaseParams` and equal `CommittedSourceContract`.

### Invariants

The frozen-structure invariants constrain the guided search. A row returned by
the full-search fallback is selected by the full planner and need not retain
them. Selection is subject to the retained-assignment rule; it does not imply
global optimality over discarded assignments.

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
- **Full-search fallback.** If no candidate satisfies every guide constraint,
  the guided search reports `AkitaError::UnsupportedSchedule`, and adaptation
  MUST then return the result of `find_schedule` for the same key, source
  contracts, and policy. Adaptation MUST NOT fall back for an invalid request:
  a request without precommitted groups, a final group that differs from the
  scalar row, or a scalar row that fails audit under the supplied policy.
- **One opening per class.** Both the guided and the full search MUST assign
  one root opening to each interchangeable class, and every group of the class
  MUST open with it. The precommit opening domain is the Cartesian product of
  the per-class domains; it MUST NOT depend on class multiplicity. Groups of one
  class therefore materialize to identical parameters.
- **No planner-local group limit.** Neither search limits the number of
  precommitted groups. The opening-layout and wire bounds that apply to every
  schedule bound the group count.
- **Bounded retained assignments.** Both searches MUST preserve the original
  enumeration for class-opening products of at most 256 assignments. For a
  larger product, including one whose count overflows `usize`, selection MUST
  wait until the root dimension tuple, final-group opening, and inner and
  opening decomposition bases are fixed. Each search MUST stream the full
  Cartesian product of admitted class openings for that context. It MUST size
  each assignment through the canonical combined D-rank, compression, and
  witness-layout calculations. The score MUST be the smallest live outgoing
  root witness length across the allowed final-group splits and slices, before
  setup-based slice pruning. It MUST count all groups, decomposition digits,
  chunks, compression and relation-quotient spans, and required alignment.
  Assignments with no valid root candidate MUST NOT be retained. Each search
  MUST retain all eligible assignments if there are at most 256. Otherwise it
  MUST retain the 256 assignments with the smallest scores.
  Equal scores MUST use the original class-opening order. Shared subrings MUST
  NOT receive reserved slots. Both searches MUST NOT discard an opening
  dimension merely because its product exceeds 256. Oversized products MUST NOT
  be allocated eagerly. The limit bounds retained assignments, not enumeration
  work.
- **Final admission.** An adapted row MUST NOT bypass
  `ValidatedScheduleCatalog::try_new`, challenge-hook validation, duplicate-key
  rejection, row identity, catalog identity, or final `TrustedScheduleCatalog<Cfg>`
  configuration binding.
- **Full-search domain.** With no guide, `find_schedule` MUST search the full
  policy domain subject to one opening per class and the retained-assignment
  rule. Complete schedules in that retained domain MUST use the canonical
  schedule objective. Next-witness ranking does not guarantee feasibility
  completeness or global proof/setup optimality over discarded assignments.
  The full-search fallback MUST apply the same rule. It does not restore a
  guarantee over discarded assignments.
- **Offline only.** Setup restoration, commitment, proving, proof decoding,
  verification, and guest execution MUST NOT call either planner entry point.

### Non-goals

- Runtime schedule search or a process-global schedule registry.
- Opening assignments that differ within an interchangeable class.
- Global feasibility completeness or proof/setup optimality over opening
  assignments discarded by the retained-assignment rule.
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
- [x] Adaptation falls back to the full search when the guided search reports
  `UnsupportedSchedule`, and a test checks that the fallback equals
  `find_schedule`.
- [x] Root precommit openings are enumerated once per interchangeable class; a
  regression test plans 259 precommitted producers of two classes.
- [x] Oversized opening products are streamed and ranked by exact combined next
  witness size. An exhaustive sorting test checks the retained 256 assignments,
  repeated groups, tie order, and the absence of shared-subring reservations.
- [x] The manual benchmark compares guided rows with full-DP rows, including
  28- through 50-variable final groups.
- [x] The implementation adds no verifier, transcript, proof-wire, or artifact
  schema path.

### Testing strategy

`crates/akita-planner/src/test/adapted_schedule.rs` owns end-to-end adapter
tests and the ignored quality benchmark. Candidate-level tests protect the
per-class opening domain and its independence from class multiplicity.
`crates/akita-planner/src/test/root_witness_ranking.rs` compares exact witness
ranking with an exhaustive 512-assignment sort. Existing unpruned-search and
relation-order tests continue to exercise the full planner with no guide.

The GitHub merge gates MUST run the two workspace nextest shards, all three
Clippy feature graphs, transcript modes, and external schedule-artifact drift.
For the current checked-in rows, catalog and planner-cost baseline checks MUST
pass and artifact drift MUST remain empty.

### Performance

Adaptation has no platform-specific latency guarantee. The guided search is the
fast path; a fallback costs one full search. For oversized products, both
searches scan every admitted opening assignment in each fixed root context.
Work grows with the full Cartesian product and can grow exponentially with
the number of distinct classes. The 256 retained assignments are not a latency
or total-work bound. Caching class admissions and final-group layouts avoids
repeated response sizing and A/B derivation, but does not remove that product.
Ranking uses parameters; it does not generate a witness or proof.

Additional groups in an existing class do not add opening choices. Their
materialization and witness-sizing work still grows with class multiplicity.
The retained-assignment rule can discard a feasible or better complete
schedule, and the fallback uses the same rule. The release benchmark records
guided-search time and compares the expanded proof-payload estimate against
the selected full-DP row. Proof-size equality is measured evidence, not a
correctness requirement. Each
benchmark case declares whether the guided search should succeed. Unexpected
successes and failures fail the test; successful rows must pass final catalog
validation, configuration binding, and exact-key resolution. A second manual
benchmark measures both searches as one interchangeable class grows.

## Design

### Architecture

`PrecommittedProducer::try_new` binds the producer declaration.
`GroupedGenerationRequest` derives the exact lookup key and producer fold
policies. `find_adapted_schedule` re-audits the scalar row and invokes the
canonical suffix DP with a root constraint and a schedule guide. If that search
reports `UnsupportedSchedule`, it calls `find_schedule` for the same key.

The guide narrows existing candidate domains rather than introducing a second
materializer. Root, recursive, setup-prefix, and terminal candidates continue
to flow through the same length, security, response, relation, and proof-cost
derivations as full planning. The output then flows through the existing
`ResolvedScheduleRow` audit, `ValidatedScheduleCatalog` semantic admission, and
`TrustedScheduleCatalog<Cfg>` configuration binding.

### Alternatives considered

- **Run full DP for every late grouped key.** This removes the scalar guide
  constraints, but still applies the retained-assignment rule. It can cost more
  than the guided fast path and does not guarantee global optimality over
  discarded assignments.
- **Fail closed when the guide is infeasible.** The original contract. It
  bounded latency but made liveness depend on the scalar row's structure: large
  producers on setup-offloaded rows had no adapted schedule even though the full
  search found one.
- **Enumerate every multiset of openings within a class.** This is
  C(d + n - 1, n) assignments for n groups over a d-candidate domain, which is
  intractable for hundreds of groups. No checked-in row uses a mixed class, and
  measured full searches with up to eight interchangeable groups selected the
  same schedule under both rules.
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
- [PR #373: root opening ranking](https://github.com/LayerZero-Labs/akita/pull/373)
- [External schedule catalog ownership](external-schedule-catalog-ownership.md)
- [Quotient-free tail implementation](quotient-free-tail-ring-relations-implementation.md)
- [Setup offloading planner](setup-offloading-planner.md)
- `crates/akita-planner/src/planner.rs`
- `crates/akita-planner/src/planner/root_openings.rs`
- `crates/akita-planner/src/schedule_params/suffix_dp/`
- `crates/akita-planner/src/test/adapted_schedule.rs`
