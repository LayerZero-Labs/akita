# Verifier-only integration

> **Status:** current integration outline. A complete standalone example remains
> part of the reader-path follow-up.

Consumers that only verify, such as the Jolt guest, should depend on
`akita-verifier`, `akita-types`, and `akita-config`. They can call
`batched_verify::<Cfg, T>` directly. This bypasses
`AkitaCommitmentScheme::batched_verify`, whose timing log uses `Instant::now()`.

The verifier accepts an explicit `OpeningScheduleSelection` and resolves it
through the enabled generated catalog in `akita-schedules`. It never invokes
the schedule search in `akita-planner`. The selected schedule also determines
whether each nonterminal fold uses direct setup evaluation or the Stage 3
setup-prefix proof. The caller does not select that mode independently.

All verifier-facing proof, setup, schedule, claim, and transcript data is
untrusted. Malformed input must return `AkitaError` or `SerializationError`
without panicking. See [Verification](../how/verification.md) for the complete
contract.

## Sources to fold in

- `crates/akita-verifier/src/lib.rs`
- `docs/verifier-contract.md`
- `scripts/check-crate-deps.sh`
