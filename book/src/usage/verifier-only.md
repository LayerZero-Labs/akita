# Verifier-only integration

> **Status:** stub. Part of the initial Akita Book scaffold.

For consumers that only verify (e.g. the Jolt guest): depend on `akita-verifier`
+ `akita-types` + `akita-config`, call `verify_batched::<Cfg, T, D>` directly
(bypassing `AkitaCommitmentScheme::batched_verify`, which uses `Instant::now()`),
and note that the planner is reached transitively via `akita-config` (the DP
fallback is verifier-reachable). State the no-panic contract expectation.

## Sources to fold in

- `crates/akita-verifier/src/lib.rs`
- `AGENTS.md` (Verifier No-Panic Contract; crate roles)
- `scripts/check-crate-deps.sh` (dependency hygiene)
