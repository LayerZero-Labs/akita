# Verifier no-panic contract

Canonical narrative: [`book/src/how/verification.md`](../book/src/how/verification.md).
Agent hot-path summary: [`AGENTS.md`](../AGENTS.md).

Verifier-reachable execution is a **no-panic boundary**.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.
`AkitaError` has one canonical definition in `akita-error`.

## In scope

- `akita-verifier`
- Verifier-reachable code in `akita-types` (including SIS derivation and table materialization), `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths
- `akita-config` policy validation and trusted artifact binding used by `batched_verify`
- `akita-schedules` trusted artifact decoding, row identity, row resolution,
  and canonical resolved-row audit paths

The verifier never invokes planner search. It accepts only an explicit
`OpeningScheduleSelection` that resolves in a caller supplied trusted catalog.
The catalog is setup or preprocessing input, never proof input. Before setup
access or transcript replay, Akita validates artifact identity and runtime
hooks, resolves the public row digest, compares every ordered public
`CommittedGroupProfile`, re-audits every A/B/D/recursive/terminal SIS matrix,
checks challenge and full terminal L infinity or L2 cap geometry, and confirms the schedule fits
the setup field capacity. Private polynomial representations and honest-prover
witness models are not verifier inputs. A proof cannot supply schedule bytes.

The accepted proof topology is structural: a root fold, zero or more recursive
folds, and one terminal cleartext witness. The verifier rejects proof-shape
mismatches before transcript replay. Every terminal uses predecessor-bound
inner `t` and the `consistency | A` relation. The root may be that predecessor;
there is no separate fallback proof form, final outer `u`, or terminal B/D
block to validate.

## Rules

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.
2. Use `akita_error::checked` for reusable exact `usize` formulas. These functions return `Option`; map failure to the appropriate `AkitaError` at the protocol boundary. Direct standard library `checked_*` calls remain appropriate for a single local operation.
3. Do not replace exact size or index arithmetic with wrapping or saturating arithmetic. Those operations hide malformed geometry instead of rejecting it.
4. Prefer strengthening validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
5. Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
6. Prover-only panics are acceptable for now if they are not reachable from verifier paths.

## Evidence

Historical audit checklist: [`docs/verifier-panic-audit.md`](verifier-panic-audit.md) (PR #81 snapshot; link, do not duplicate).
