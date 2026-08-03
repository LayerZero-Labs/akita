# Verifier-only integration

For consumers that only **verify** (the Jolt guest is the canonical example —
see [`profile/akita-recursion/README.md`](../../../profile/akita-recursion/README.md)
for the recursion integration), there is a slim dependency path that avoids the
umbrella `akita-pcs` crate and its prover-facing APIs.

## When to go verifier-only

The umbrella `akita-pcs` package is convenient for examples and end-to-end use,
but it pulls in prover-facing APIs. Verifier-only consumers should depend on the
slim role crates directly and call the verifier entry point themselves. See the
crate map in [`how/architecture.md`](../how/architecture.md) for the dependency
boundaries.

## Dependencies

```toml
[dependencies]
akita-verifier   = { git = "https://github.com/LayerZero-Labs/akita", rev = "<exact-commit>" }
akita-types      = { git = "https://github.com/LayerZero-Labs/akita", rev = "<exact-commit>" }
akita-config     = { git = "https://github.com/LayerZero-Labs/akita", rev = "<exact-commit>" }
akita-field      = { git = "https://github.com/LayerZero-Labs/akita", rev = "<exact-commit>" }      # AkitaError
akita-transcript = { git = "https://github.com/LayerZero-Labs/akita", rev = "<exact-commit>" }      # Transcript trait
```

`AkitaError` lives in `akita-field` and the `Transcript` trait lives in
`akita-transcript`; neither is re-exported by the role crates, so add both to
compile the `batched_verify` call below.

Pin a specific git revision: Akita makes **no backward-compatibility
guarantees**, so `rev` must be pinned exactly (`crates/akita-types` bumps
`AKITA_INSTANCE_DESCRIPTOR_VERSION` on protocol breaks). Pin an exact commit and
re-run the prove/verify integration tests when upgrading.

The role-crate boundaries are enforced by `scripts/check-crate-deps.sh`
(`akita-verifier` must not reach prover-only polynomial backends).

## The verification entry point

Call `akita_verifier::batched_verify::<Cfg, T>` directly
(`crates/akita-verifier/src/protocol/core/verify.rs`). The proof, setup,
and claim types — `AkitaBatchedProof`, `AkitaVerifierSetup`, `OpeningClaims`,
`BasisMode` — live in `akita-types` (of these four, the verifier re-exports
none; reference them via the `akita-types` dependency directly):

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
(`crates/akita-pcs/src/scheme/mod.rs`): the umbrella wrapper measures
elapsed time with `Instant::now()` for its telemetry span
(`scheme/mod.rs`), a host-timer dependency that the recursion guest does
not provide. The direct call also has no caller-selected setup-contribution
mode — the verifier derives that behavior from the resolved schedule.

### Replay validation flow

Every validation failure returns a typed `AkitaError` — the verifier never
panics on malformed input. The error variant distinguishes the failure class
(`verify.rs`):

1. `claims.validate(...)` — claim shapes against the setup seed
   → `InvalidProof`
2. schedule resolution for the opening batch → `InvalidProof`
3. `validate_schedule_ring_dims` — ring dimensions against the setup seed
   → `InvalidSetup`
4. `ensure_schedule_fits_setup` — schedule within setup capacity → `InvalidSetup`
5. `validate_proof_against_schedule` — proof structure matches the schedule
   → `InvalidProof`

Only then does replay begin, after warming the terminal-NTT prefixes for the
resolved schedule (`verify.rs`).

## The no-panic contract

Verifier-reachable code must reject malformed input with `AkitaError` or
`SerializationError`, and must never panic. The full contract and the
contributor rules are in
[`docs/verifier-contract.md`](../../../docs/verifier-contract.md) and the
panic-audit record in
[`docs/verifier-panic-audit.md`](../../../docs/verifier-panic-audit.md).

Note that the schedule planner is **not** pulled into the verifier path. Runtime
schedule resolution is strict: `akita_schedules::resolve_group_batch_schedule`
resolves only against the enabled generated catalog and **never invokes planner
search** — a missing catalog or row returns `AkitaError::UnsupportedSchedule`
(`crates/akita-schedules/src/resolve.rs`), and `akita-config`'s runtime
resolution rejects instead of falling back to the DP
(`crates/akita-config/src/lib.rs`). `akita-planner` is only a dev-dependency
of `akita-config`, and `akita-verifier` deliberately avoids planner search
(`crates/akita-verifier/src/lib.rs`). The strict resolution that *is*
verifier-reachable rejects unsupported requests with `AkitaError` rather than
panicking, so it is still covered by the no-panic contract.

Verifier replay internals (per-level verifiers, the stage-2 verifier,
prepared-claim shapes) are crate-private. A few replay primitives
(`RingSwitchReplay`, `RelationMatrixEvaluator`,
`prepare_relation_matrix_evaluator`, `AkitaStage1Verifier`) are kept public
solely so integration tests in `akita-pcs` can exercise them in isolation;
they are not part of the intended downstream API
(`crates/akita-verifier/src/lib.rs`).
