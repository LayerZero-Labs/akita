# Verification

> **Status:** stub. Part of the initial Akita Book scaffold.

How the verifier replays the proof level by level, and the no-panic contract
that governs every verifier-reachable line.

## Per-level replay

The verifier's structure: re-derive the schedule, replay each level's sumcheck
stages and opening checks, and evaluate the relation matrix `M` at a point. Pair
each prover stage with its verifier mirror.

**Sources to fold in**

- `crates/akita-verifier/src/lib.rs`, `protocol/batched.rs:633-717`.
- `crates/akita-verifier/src/protocol/levels.rs`, `levels/recursive.rs`, `src/stages/`.
- Paper §3.8 `sec:akita-full-pcs` (`Eval.V`), §4.1 `sec:verifier-cost-anatomy` (per-level verifier cost).
- `specs/optimized_verifier.md` (M-at-a-point evaluation — durable reference).

## The verifier no-panic contract

The rule that verifier-reachable execution is a no-panic boundary: malformed
proof/setup/schedule/claim input is rejected with `AkitaError` /
`SerializationError`, never by panicking. Which crates the contract spans and
where validation is enforced (deserialization, setup, schedule selection,
`LevelParams`, API entry).

**Sources to fold in**

- `AGENTS.md` (Verifier No-Panic Contract — canonical statement).
- `docs/verifier-panic-audit.md` (historical evidence — being archived; link, don't duplicate).
- `specs/security-hardening.md`.
