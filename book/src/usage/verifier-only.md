# Verifier-only integration

> **Status:** stub. Part of the initial Akita Book scaffold.

For consumers that only verify (e.g. the Jolt guest): depend on `akita-verifier`
+ `akita-types` + `akita-config`, call `batched_verify::<Cfg, T>` directly
(bypassing `AkitaCommitmentScheme::batched_verify`, which uses `Instant::now()`),
with no caller-selected setup-contribution mode. The verifier derives that
behavior from the resolved schedule. Note that the planner is reached
transitively via `akita-config` (the DP
fallback is verifier-reachable). State the no-panic contract expectation.

## Sources to fold in

- `crates/akita-verifier/src/lib.rs`
- `AGENTS.md` (Verifier No-Panic Contract; crate roles)
- `scripts/check-crate-deps.sh` (dependency hygiene)
