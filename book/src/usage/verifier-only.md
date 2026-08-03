# Verifier-only integration

For consumers that only **verify** (the Jolt guest is the canonical example),
there is a slim dependency path that avoids the umbrella `akita-pcs` crate and
its prover-facing APIs.

## When to go verifier-only

The umbrella `akita-pcs` package is convenient for examples and end-to-end use,
but it pulls in prover-facing APIs. Verifier-only consumers should depend on the
slim role crates directly and call the verifier entry point themselves. See the
crate map in [`how/architecture.md`](../how/architecture.md) for the dependency
boundaries.

## Dependencies

```toml
[dependencies]
akita-verifier   = { git = "..." }   # verifier replay
akita-types      = { git = "..." }   # proof / setup / claim shapes
akita-config     = { git = "..." }   # concrete schedule / config policy
```

The role-crate boundaries are enforced by `scripts/check-crate-deps.sh`
(`akita-verifier` must not reach prover-only polynomial backends).

## The verification entry point

Call `akita_verifier::batched_verify::<Cfg, T>` directly
(`crates/akita-verifier/src/protocol/core/verify.rs:217-248`):

```rust
pub fn batched_verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    basis: BasisMode,
) -> Result<(), AkitaError>
```

Prefer this over `AkitaCommitmentScheme::batched_verify`
(`crates/akita-pcs/src/scheme/mod.rs:264-273`): the umbrella wrapper measures
elapsed time with `Instant::now()` for its telemetry span
(`scheme/mod.rs:305-312`), a host-timer dependency that the recursion guest does
not provide. The direct call also has no caller-selected setup-contribution
mode — the verifier derives that behavior from the resolved schedule.

### Replay validation flow

Every validation failure returns `AkitaError::InvalidProof` — the verifier
never panics on malformed input (`verify.rs:236-253`):

1. `claims.validate(...)` — claim shapes against the setup seed
2. schedule resolution for the opening batch
3. `validate_schedule_ring_dims` — ring dimensions against the setup seed
4. `ensure_schedule_fits_setup` — schedule within setup capacity
5. `validate_proof_against_schedule` — proof structure matches the schedule

Only then does replay begin, after warming the terminal-NTT prefixes for the
resolved schedule (`verify.rs:255-259`).

## The no-panic contract

Verifier-reachable code must reject malformed input with `AkitaError` or
`SerializationError`, and must never panic. The full contract and the
contributor rules are in
[`docs/verifier-contract.md`](../../../docs/verifier-contract.md) and the
panic-audit record in
[`docs/verifier-panic-audit.md`](../../../docs/verifier-panic-audit.md).

Note that the schedule planner is reached **transitively** via `akita-config`
(the DP fallback on a generated-table miss), so the planner is also
verifier-reachable and subject to the same contract.

Verifier replay internals (per-level verifiers, ring-switch replay, the stage-2
verifier, prepared-claim shapes) are crate-private; only the entry points
actually consumed downstream are public
(`crates/akita-verifier/src/lib.rs`).
