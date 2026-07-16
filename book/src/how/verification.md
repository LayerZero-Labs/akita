# Verification

How the verifier replays the proof level by level, and the no-panic contract
that governs every verifier-reachable line.

## Per-level replay

`batched_verify` (in `crates/akita-verifier/src/protocol/core/verify.rs`) is
directly `<Cfg>`-generic: it calls `CommitmentConfig` hooks and
`bind_transcript_instance_descriptor` with no policy closure layer.

At a high level:

1. **Bind the instance** and absorb the opening batch shape into the transcript.
2. **Resolve the schedule** the prover used (`CommitmentConfig::runtime_schedule`), validating `nuposition_index_bits` against setup capacity before any DP fallback.
3. **For each level**, replay sumcheck stages, opening checks, and fold or direct paths as dictated by `LevelParams` (see `protocol/levels/`, `levels/root_fold.rs`, `levels/suffix.rs`, `src/stages/`).
4. **Evaluate the relation matrix** `M` at the derived point ([Matrix evaluation](./verifying/matrix_evaluation.md)).

The verifier never constructs prover-only polynomial backends or setup expansion
kernels.

## The verifier no-panic contract

Verifier-reachable execution is a **no-panic boundary**.
Malformed verifier-facing proof, setup, schedule, public claim, opening point,
commitment, direct witness, or transcript input must be rejected with
`AkitaError` or `SerializationError`, never by panicking.

### Crates in scope

- `akita-verifier`
- Verifier-reachable paths in `akita-types`, `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, and verifier-used `akita-field` code
- `akita-config` (every `CommitmentConfig` method reachable from `batched_verify`)
- `akita-planner` (schedule-search DP on table miss, reachable through `runtime_schedule`)

The verifier must validate `key.nuposition_index_bits` against setup capacity **before**
invoking the DP so a malformed proof cannot blow up the search state space.

### Rules for contributors

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier boundary has validated the invariant.
2. Strengthen validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points rather than sprinkling checks through hot loops.
3. Prover-only panics are acceptable when not reachable from verifier paths.

Maintainer mirror: [`docs/verifier-contract.md`](../../../docs/verifier-contract.md).
Historical audit evidence: [`docs/verifier-panic-audit.md`](../../../docs/verifier-panic-audit.md).
