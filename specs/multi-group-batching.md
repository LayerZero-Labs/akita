# Spec: Multi-group commitment profiles and root batching

| Field | Value |
|-------|-------|
| Status | active |
| Updated | 2026-08-09 |
| PR | #355 |
| Historical record | `archive/2026-Q3/multi-group-batching-legacy.md` |

## Summary

Akita supports commitments that are created separately and later opened in one
root proof. Each prior group carries the exact public A and B geometry
that was used to commit it. The final root schedule is selected from the final
group layout and the ordered list of those exact profiles.

This file is the current contract. The archived record preserves earlier
designs based on `ConservativeCommitmentConfig`, reconstructed layouts, and
ordinary `batched_commit`. Those APIs have been removed.

## Canonical types

`CommittedGroupProfile` is the single source of truth for a prior group.
It freezes:

- the polynomial group layout;
- live ring and block geometry;
- the independent inner and outer bases and digit counts;
- the exact inner and outer SIS matrices, including ranks and bounds.

`AkitaScheduleLookupKey` contains one final group layout and an ordered list of
exact `prior_group_profiles`. Profile order is part of the schedule and
transcript identity.

The final root must use every frozen profile exactly. It may choose fresh root
opening and D geometry, but it must not reconstruct or change a prior group's A
or B relation.

## Standalone profile selection

Offline generation calls the canonical standalone planner for each supported
group layout. It searches the configured inner basis domain with the root
opening basis pinned to the minimum configured opening basis. It keeps a Pareto
frontier for diagnostics and selects by:

1. exact outgoing witness length;
2. padded A or B setup fields;
3. exact A or B setup fields;
4. canonical profile descriptor bytes.

There is no separate one-variant precommit selection policy. A future change to
this objective must use a real catalog identity and update this spec.

Generated profile lookup is strict. An unlisted layout returns
`UnsupportedSchedule`. The runtime does not run the offline planner.

## Commitment flow

The current staggered flow is:

1. Commit each early group with the unified `commit` method and
   `GroupPosition::Prior`; this resolves its exact generated P profile.
2. Build one ordered `PriorGroupProfiles` owner from those committed groups.
3. Commit the last group with `GroupPosition::Final`, borrowing that owner.
4. Build the self-describing `OpeningClaims`, then pass the same owner to
   `SelectedProverOpeningData::from_committed_claims`; batch assembly selects
   the exact generated G row before profiles are stripped.
5. Prove with that selected row and verify against its explicit row identity.

`GroupPosition::Sole` commits the only group in an opening batch under its
scalar S row. Multiple homogeneous polynomials may still belong to that one
group.

The commitment and opening claims must use the same ordered group profiles.
Malformed, missing, reordered, or altered profiles return `AkitaError`.

## Planning and runtime ownership

`akita-planner` owns standalone profile search and grouped root search.
`akita-schedules` owns generated catalogs, catalog identity checks, expansion,
and strict runtime resolution. Verifier crates do not depend on the planner.

Generated families carry both standalone profiles and selected grouped rows.
Generation holds a typed planning request with its key and honest fold policies
through one materialization pass. Unsupported requests are omitted. Runtime
table misses remain unsupported.

## Security and binding

Every matrix rank comes from the canonical role-aware SIS coverage and secure
rank lookup. The final planner cannot replace a frozen matrix with a smaller
one. Independent inner and outer bases are included in profile descriptor
bytes. The effective root schedule binds the final group, ordered precommitted
profiles, fold topology, terminal response shape, and challenge policy.

Verifier-reachable code rejects malformed data. It must not panic, allocate
from unchecked lengths, or invoke the planner.

## Acceptance criteria

- Standalone generation emits exact profiles for every supported layout.
- Profile descriptor changes alter lookup and effective schedule identity.
- `GroupPosition::Prior` uses the exact generated standalone P profile.
- `GroupPosition::Final` selects from the exact ordered profiles supplied by
  the caller and rejects an empty prefix.
- Batch assembly consumes the same `PriorGroupProfiles` allocation borrowed by
  final commitment and checks it against the ordered committed claims.
- Reordered, altered, unknown, or undersized profiles reject.
- Scalar keys normalize through `AkitaScheduleLookupKey::single`.
- Grouped roots hand one compact witness to the recursive suffix.
- Prover and verifier agree on group ordering, opening claims, and schedule
  descriptor bytes.
- Generated catalogs and their module wiring publish as one staged batch with
  rollback. Crash durability is outside this file's contract.

## Deferred work

Applications may add workload-specific profile and grouped rows to generated
catalogs. New source families, broader group grids, and new selection objectives
require explicit catalog identity and tests. They must not be inferred as
runtime fallbacks.
