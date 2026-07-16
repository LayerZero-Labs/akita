# Verifier no-panic contract

Canonical narrative: [`book/src/how/verification.md`](../book/src/how/verification.md).
Agent hot-path summary: [`AGENTS.md`](../AGENTS.md).

Verifier-reachable execution is a **no-panic boundary**.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

## In scope

- `akita-verifier`
- Verifier-reachable code in `akita-types` (including SIS derivation and table materialization), `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths
- `akita-config` (every `CommitmentConfig` method reachable from `batched_verify`)
- `akita-planner` (the schedule-search DP is verifier-reachable through `CommitmentConfig::runtime_schedule` table-miss fallback)

The verifier must validate `key.nuposition_bits` against setup capacity before invoking the DP so a malformed proof cannot blow up the search's bounded state space.

## Rules

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.
2. Prefer strengthening validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
3. Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
4. Prover-only panics are acceptable for now if they are not reachable from verifier paths.

## Evidence

Historical audit checklist: [`docs/verifier-panic-audit.md`](verifier-panic-audit.md) (PR #81 snapshot; link, do not duplicate).
