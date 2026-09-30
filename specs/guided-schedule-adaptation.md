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
group. Guided schedule adaptation makes this operation fast in the common case:
it retains the trusted scalar row's structural choices and rebuilds every
group-dependent value. When that structure cannot support the grouped request,
adaptation falls back to the full planner for the same key, so it fails only
when no schedule exists in the audited domain.

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
the full-search fallback is the full planner's optimum and need not retain
them.

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
- **Bounded class products.** Both searches MUST bound the per-dimension
  product of class opening domains at 256 before allocating it. A larger
  product MUST remove coefficient-packing root openings for that dimension
  rather than fail the search. The root only folds by coefficient packing, so
  that dimension then offers no root candidate, exactly as an unsupported
  dimension does; other dimensions are unaffected.
- **Final admission.** An adapted row MUST NOT bypass
  `ValidatedScheduleCatalog::try_new`, challenge-hook validation, duplicate-key
  rejection, row identity, catalog identity, or final `TrustedScheduleCatalog<Cfg>`
  configuration binding.
- **Oracle preservation.** `find_schedule` MUST remain the full-DP
  correctness and proof-size oracle when no guide is supplied. Its only
  restrictions on the precommit opening domain are one opening per class and
  the class-product bound.
- **Offline only.** Setup restoration, commitment, proving, proof decoding,
  verification, and guest execution MUST NOT call either planner entry point.

### Non-goals

- Runtime schedule search or a process-global schedule registry.
- Opening assignments that differ within an interchangeable class.
- Coefficient-packing root openings for requests whose distinct classes
  exceed the 256-combination product bound. Coordinate-wise search over classes
  is a follow-up.
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
- [x] The manual benchmark compares guided rows with full-DP rows, including
  28- through 50-variable final groups.
- [x] The implementation adds no verifier, transcript, proof-wire, or artifact
  schema path.

### Testing strategy

`crates/akita-planner/src/test/adapted_schedule.rs` owns end-to-end adapter
tests and the ignored quality benchmark. Candidate-level tests protect the
per-class opening domain and its independence from class multiplicity.
Existing unpruned-search and relation-order tests continue to exercise the full
planner with no guide.

The GitHub merge gates MUST run the two workspace nextest shards, all three
Clippy feature graphs, transcript modes, and external schedule-artifact drift.
Artifact drift MUST remain empty: no checked-in grouped row assigns different
openings within an interchangeable class, so the per-class opening rule selects
the same rows.

### Performance

Adaptation has no platform-specific latency guarantee. The guided search is the
fast path; a fallback costs one full search. Per candidate edge, both searches
materialize one group per interchangeable class and validate the root batch
once per group loop, so the cost of additional groups in an existing class is
linear and small. The release benchmark records guided-search time and
compares the expanded proof-payload estimate against the full-DP oracle;
proof-size equality is measured evidence, not a correctness requirement. Each
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
derivations as exhaustive planning. The output then flows through the existing
`ResolvedScheduleRow` audit, `ValidatedScheduleCatalog` semantic admission, and
`TrustedScheduleCatalog<Cfg>` configuration binding.

### Alternatives considered

- **Run full DP for every late grouped key.** This preserves global optimality
  but takes seconds for production-sized rows, which the guided fast path
  avoids whenever the scalar structure fits.
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
- [External schedule catalog ownership](external-schedule-catalog-ownership.md)
- [Quotient-free tail implementation](quotient-free-tail-ring-relations-implementation.md)
- [Setup offloading planner](setup-offloading-planner.md)
- `crates/akita-planner/src/planner.rs`
- `crates/akita-planner/src/schedule_params/suffix_dp/`
- `crates/akita-planner/src/test/adapted_schedule.rs`
