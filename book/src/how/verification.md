# Verification

How the verifier replays the proof level by level, and the no-panic contract
that governs every verifier-reachable line.

## Per-level replay

`batched_verify` (in `crates/akita-verifier/src/protocol/core/verify.rs`) is
directly `<Cfg>`-generic: it calls `CommitmentConfig` hooks and
`bind_transcript_instance_descriptor` with no policy closure layer.

At a high level:

1. **Bind the instance** and absorb the opening batch shape into the transcript.
2. **Resolve the schedule** the prover used (`CommitmentConfig::runtime_schedule`), validating `num_vars` against setup capacity before any DP fallback.
3. **Replay the structural folds** in `protocol/core`: the root fold followed by
   every recursive fold, using the schedule-selected `LevelParams`.
4. **Check the terminal witness directly** against its predecessor-bound `t`
   state. The terminal relation is `consistency | A`; it has no outer `u`, B
   block, D block, or quotient sumcheck.

The terminal `A * z` check accepts exactly the signed-i16 coefficient class.
Decoded coefficients outside `[-32768, 32767]` are rejected before arithmetic;
there is no alternate i8 or balanced-radix verifier path. The exact
CRT-capability selector keeps the base profile when
`2 * width * D * floor(q/2) * 32768 < product(base primes)` and otherwise adds
the 12289 i16 tail. A schedule whose accumulation exceeds both profiles is
rejected as an invalid setup.

The verifier warms the strongest representation selected by the validated
terminal schedule before transcript replay. Prepared forms are derived from
the coefficient setup, keyed by ring dimension, and never serialized. Groups
share one base prefix; its optional tail is only as long as the largest
tail-requiring group. Thus a base-only schedule never constructs the tail, and
a larger base-only group cannot unnecessarily extend one required by a smaller
group. Shape and setup-prefix checks happen before either kernel indexes
prepared state.

The verifier never constructs prover-only polynomial backends or setup expansion
kernels.

## Setup-contribution replay

Stage 2 evaluates the relation matrix at its sampled row and column point. The
setup contribution to that evaluation has two execution paths, but both use one
semantic description.

`SetupContributionPlan::prepare` validates the opening batch, witness layout,
group order, role dimensions, setup envelope, and challenge lengths. It records
the D, B, and A contributions as typed affine spans. Each span fixes its
relation rows, witness-opening addresses, packed setup addresses, role
projection, and A-fold geometry. Consumers do not reconstruct those addresses
from the source layouts.

The verifier then follows this sequence:

1. Stage 2 evaluates the structured E, T, and Z relation terms from the spans.
2. A direct schedule derives a private fused scan from the spans and contracts
   the expanded setup once.
3. A recursive schedule substitutes the transcript-bound setup claim and
   captures the exact plan used by Stage 2.
4. After Stage 2 succeeds, the fold orchestrator moves that plan into the Stage
   3 verifier.
5. Stage 3 evaluates the plan's setup-index multilinear polynomial at the
   sampled setup point.

The plan does not store dense E, T, or Z weights, copied row weights, direct
scan segments, setup coefficients, or a direct-versus-recursive mode. Direct
scan state is compiled only by direct contraction. Recursive verification does
not scan the active setup or prepare the direct backend.

`AkitaStage2Verifier` uses a single-assignment cell only to adapt this ownership
handoff to the generic sumcheck verifier trait. The cell is consumed
immediately after successful Stage 2 verification. `RelationMatrixEvaluator`
does not cache the plan, and Stage 3 has no reconstruction fallback. Missing or
duplicate handoff state returns `AkitaError`.

This ownership refactor does not change proof serialization, transcript labels,
challenge order, or the Stage 2 and Stage 3 equations.

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

The verifier must validate `key.num_vars` against setup capacity **before**
invoking the DP so a malformed proof cannot blow up the search state space.

### Rules for contributors

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier boundary has validated the invariant.
2. Strengthen validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points rather than sprinkling checks through hot loops.
3. Prover-only panics are acceptable when not reachable from verifier paths.

Maintainer mirror: [`docs/verifier-contract.md`](../../../docs/verifier-contract.md).
Historical audit evidence: [`docs/verifier-panic-audit.md`](../../../docs/verifier-panic-audit.md).
