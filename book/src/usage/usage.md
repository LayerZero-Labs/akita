# Usage

> **Status:** stub. Part of the initial Akita Book scaffold.

Everything an application developer needs to use Akita as a PCS: build and test
it, pick a configuration, commit to polynomials, prove and verify evaluation
claims, profile, and integrate the verifier into Jolt.

This part covers:

- [Quickstart and configuration](./quickstart.md) — smallest end-to-end example,
  and how to choose among the `fp32` / `fp64` / `fp128` presets.
- [The commitment API](./commitment-api.md) — `commit` / `prove` / `verify`,
  including setup, caching, and transcript handling.
- [Verifier-only integration](./verifier-only.md) — the no-prover-backend path.
- [Feature flags](./feature-flags.md), [Profiling](./profiling.md),
  [Troubleshooting](./troubleshooting.md).
- [Jolt recursion](./jolt-recursion.md) — the standalone `profile/akita-recursion/`
  sub-workspace.

## Sources to fold in

- `crates/akita-pcs/src/lib.rs` (re-exports), `crates/akita-pcs/src/scheme/mod.rs`.
- Paper §7 `sec:evaluation` (implementation and evaluation).
