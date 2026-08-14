# Verifier no-panic contract

Canonical narrative: [`book/src/how/verification.md`](../book/src/how/verification.md).
Agent hot-path summary: [`AGENTS.md`](../AGENTS.md).

Verifier-reachable execution is a **no-panic boundary**.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

## In scope

- `akita-verifier`
- Verifier-reachable code in `akita-types` (including SIS derivation and table materialization), `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths
- `akita-config` (every `CommitmentConfig` method reachable from `batched_verify`)
- `akita-schedules` generated-catalog identity, row resolution, and canonical
  resolved-row audit paths

The verifier never invokes planner search. It accepts only an explicit
`OpeningScheduleSelection` that resolves in the enabled generated catalog.
Before setup access or transcript replay, it validates catalog identity and
runtime hooks, resolves the public row digest, compares every ordered public
`CommittedGroupProfile`, re-audits every A/B/D/recursive/terminal SIS matrix,
checks challenge and full terminal L infinity or L2 cap geometry, and confirms the schedule fits
the setup field capacity. Private polynomial representations and honest-prover
witness models are not verifier inputs.

The accepted proof topology is structural: a root fold, at least one suffix
fold, and one terminal cleartext witness. The verifier rejects empty/one-fold
schedules and proof-shape mismatches before transcript replay. Every terminal
uses predecessor-bound inner `t` and the `consistency | A` relation; there is no
root-terminal fallback, final outer `u`, or terminal B/D block to validate.

## Rules

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.
2. Prefer strengthening validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
3. Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
4. Prover-only panics are acceptable for now if they are not reachable from verifier paths.

## Evidence

Historical audit checklist: [`docs/verifier-panic-audit.md`](verifier-panic-audit.md) (PR #81 snapshot; link, do not duplicate).
