# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

## Crate Structure

Workspace members under `crates/`:

- `akita-field` — field traits, prime/extension fields, wide/packed helpers, FFT, parallel macros
- `akita-serialization` — serialization/validation/compression traits
- `akita-algebra` — modules/vectors, NTTs, cyclotomic rings, sparse challenges, polynomials
- `akita-transcript` — spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — sumcheck proofs, drivers, compact folding, batching, accumulation
- `akita-types` — proof, setup, schedule, layout, commitment, transcript-append, PRG shapes
- `akita-config` — runtime config presets and config-backed schedule/SIS policy
- `akita-setup` — config-backed setup construction + optional setup cache
- `akita-verifier` — verifier replay (no prover-only polynomial backends)
- `akita-prover` — commitment, proving, setup expansion, recursive/ring-switch witnesses, polynomial backends
- `akita-scheme` — end-to-end `AkitaCommitmentScheme` orchestration
- `akita-planner` — offline schedule search, proof-size/security planning
- `akita-pcs` — umbrella crate with examples, benches, integration tests, public re-exports

## Key Abstractions

- `AkitaCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` + `LevelParams` — recursion schedule, layout, per-level config
- `DensePoly`, `OneHotPoly`, `AkitaPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaLevelProof`, `AkitaProofStep` — serialized proof structure (singleton openings are the 1x1 special case of the batched proof)
- `AkitaTranscript`, `Transcript` — spongefish-backed Fiat-Shamir layer
- `AkitaInstanceDescriptor` — canonical transcript preamble binding algebra, setup, plan, and call shape

## Feature Flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence paths used by some commitment flows
- `logging-transcript` — enables `LoggingTranscript` schedule events and smell checks in transcript tests

## Transcript Hardening

The active transcript-hardening pillars are:

- P0: bind canonical `AkitaInstanceDescriptor` bytes through spongefish `DomainSeparator.instance(...)` before protocol replay.
- P2: use `AkitaTranscript` plus production-ZST labels; labels are diagnostics and must not enter production sponge bytes.
- P3: use `LoggingTranscript` tests for prover/verifier event-stream equality and wire-before-squeeze smell checks.

Deferred items are in [`specs/transcript-hardening.md`](specs/transcript-hardening.md): prover/verifier trait split, `Bound<T>`, algorithm-as-bytes digest, and NARG migration.

## Profiling

Canonical: `AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile`.

Knobs (`AKITA_MODE`, `AKITA_NUM_VARS`, `AKITA_PROFILE_TRACE`, `AKITA_PROFILE_LOG`, `AKITA_PROFILE_ANSI`, `AKITA_PROFILE_SPAN_CLOSES`, `AKITA_ALLOW_DEBUG_PROFILE`): defaults and details in `examples/profile.rs`. `RAYON_NUM_THREADS` caps Rayon threads; `--no-default-features` disables `parallel`. The `--release` guard can be bypassed with `AKITA_ALLOW_DEBUG_PROFILE=1`.

## Running the verifier inside Jolt

Standalone sub-workspace at `profile/akita-recursion/` (excluded from this workspace, pinned to Rust 1.94 + RISC-V, applies Jolt's `[patch.crates-io]` overrides for `arkworks-algebra`). Full runbook, knob reference, current cycle results, and open follow-ups: [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
