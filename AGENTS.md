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

Two workspace members: `hachi-pcs` (root) and `derive` (proc macros).

- `src/primitives/` — field/module traits, multilinear representations, serialization, transcripts
- `src/algebra/` — concrete fields, rings, NTT, polynomial utilities (eq_poly, split_eq, uni_poly)
- `src/protocol/commitment/` — configs, layouts, schedules, commitments, onehot helpers, utilities
- `src/protocol/commitment_scheme.rs` — top-level `HachiCommitmentScheme` commit/prove/verify wiring
- `src/protocol/sumcheck/` — generic sumcheck plus `hachi_stage1`, `hachi_stage2`, `two_round_prefix`
- `src/protocol/proof.rs` — proof object layout and flattened proof/witness encodings
- `src/protocol/opening_point.rs` — field-to-ring opening reduction
- `src/protocol/ring_switch.rs` — ring-switch proof logic
- `docs/block-order.md` — root-vs-recursive block-order contract
- `src/protocol/quadratic_equation.rs` — quadratic equation handling
- `src/protocol/recursive_runtime.rs` — recursive level scheduling
- `src/protocol/hachi_poly_ops/` — dense and one-hot polynomial operations
- `src/protocol/dispatch.rs` — protocol orchestration helpers
- `src/protocol/challenges/` — sparse challenge sampling
- `src/protocol/transcript/` — Fiat-Shamir transcript helpers and labels
- `src/protocol/prg.rs` — protocol PRG utilities
- `src/error.rs` — error types
- `examples/profile.rs` — profiling and proof-size harness
- `scripts/` — Python estimation scripts, hook installer
- `tests/` — end-to-end protocol tests

## Key Abstractions

- `HachiCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` + `HachiCommitmentLayout` — recursion schedule and layout knobs
- `DensePoly`, `OneHotPoly`, `HachiPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `HachiProof`, `HachiLevelProof`, `HachiProofTail` — serialized proof structure
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

- `HACHI_MODE=full|onehot|logbasis|all|compare_onehot|compare_logbasis|compare_basis`
- `HACHI_NUM_VARS=<n>` — number of variables, default `25`
- `HACHI_PROFILE_TRACE=0|1` — write a Perfetto JSON trace to `profile_traces/`, default `1`
- `HACHI_PROFILE_LOG=<filter>` — tracing filter, default `trace`
- `HACHI_PROFILE_ANSI=0|1` — ANSI log colors, default `1`
- `HACHI_PROFILE_SPAN_CLOSES=0|1` — emit close-span timing events, default `1`
- `HACHI_ALLOW_DEBUG_PROFILE=1` — bypass the `--release` guard for debugging only
- Default features enable `parallel`; use `RAYON_NUM_THREADS=<n>` to cap threads or `--no-default-features` to profile without Rayon
