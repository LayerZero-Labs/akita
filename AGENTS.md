# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Hachi is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

## Crate Structure

Workspace members live under `crates/`.

- `akita-field` — field traits, concrete prime/extension fields, wide/packed helpers, field FFT helpers, parallel macros, and core errors
- `akita-serialization` — serialization, validation, and compression traits
- `akita-algebra` — module/vector containers, NTTs, cyclotomic rings, sparse challenges, polynomial utilities, and algebra backends over `akita-field` scalars
- `akita-transcript` — Fiat-Shamir transcript traits, hash transcript implementations, and labels
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — generic sumcheck proof types, traits, drivers, compact folding, batching, and accumulation helpers
- `akita-types` — shared proof, setup, schedule, layout, commitment, transcript-append, and PRG data shapes
- `akita-config` — concrete runtime config presets and config-backed schedule/SIS policy
- `akita-setup` — config-backed setup construction and optional setup cache persistence
- `akita-verifier` — verifier replay without prover-only polynomial backends
- `akita-prover` — commitment, proving, setup expansion, recursive witness construction, ring-switch witnesses, and polynomial backends
- `akita-scheme` — end-to-end `AkitaCommitmentScheme` orchestration
- `akita-planner` — offline schedule search and proof-size/security planning
- `akita-pcs` — umbrella package with examples, benches, integration tests, and broad public re-exports

## Key Abstractions

- `HachiCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` + `LevelParams` — recursion schedule, layout, and per-level configuration
- `DensePoly`, `OneHotPoly`, `HachiPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `HachiBatchedProof`, `HachiBatchedRootProof`, `HachiLevelProof`, `HachiProofStep` — serialized proof structure (singleton openings are the 1x1 special case of the batched proof)
- `Blake2bTranscript`, `Transcript` — Fiat-Shamir layer

## Feature Flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence paths used by some commitment flows

## Profiling

Canonical run:

```bash
HACHI_MODE=onehot HACHI_NUM_VARS=32 cargo run --release --example profile
```

Knobs:

- `HACHI_MODE=full|onehot|all|full_d128|onehot_d64|full_d32|onehot_d32`
- `HACHI_NUM_VARS=<n>` — number of variables, default `25`
- `HACHI_PROFILE_TRACE=0|1` — write a Perfetto JSON trace to `profile_traces/`, default `1`
- `HACHI_PROFILE_LOG=<filter>` — tracing filter, default `trace`
- `HACHI_PROFILE_ANSI=0|1` — ANSI log colors, default `1`
- `HACHI_PROFILE_SPAN_CLOSES=0|1` — emit close-span timing events, default `1`
- `HACHI_ALLOW_DEBUG_PROFILE=1` — bypass the `--release` guard for debugging only
- Default features enable `parallel`; use `RAYON_NUM_THREADS=<n>` to cap threads or `--no-default-features` to profile without Rayon
