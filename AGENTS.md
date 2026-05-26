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
- `akita-types` — proof, setup, schedule, layout, commitment, transcript-append, PRG shapes; verifier-reachable proof-size formulas; pure layout helpers (`level_layout_from_params`, `recursive_level_layout_from_params`, `decomp_depths`)
- `akita-derive` — verifier-reachable parameter derivation: SIS-secure layout derivation (`sis_derived_root_params_for_layout`, `sis_derived_recursive_params_for_layout`, `derived_root_commitment_layout_from_params`, `sis_secure_level_params`) and schedule-table materialization (`schedule_plan_from_table`, `PlanPolicy`). Free-function API. Sits below `akita-config` in the dep graph
- `akita-config` — runtime config presets, the single `CommitmentConfig` trait, `WCommitmentConfig`, config-backed schedule adapters, and the canonical `bind_transcript_instance_descriptor` helper consumed by both prover and verifier. Depends on `akita-derive`; does NOT depend on `akita-planner`. `Cfg` defaults are table-only and error on schedule-table miss; runtime DP fallback is opt-in for tests via `akita_planner::test_utils::PlannerCfg<Cfg>` (feature-gated)
- `akita-setup` — config-backed setup construction + optional setup cache
- `akita-verifier` — verifier replay (no prover-only polynomial backends). Public surface is `<Cfg>`-generic: `verify_batched::<F, Cfg, T, D>` (lives in `protocol::batched`). Dep tree excludes `akita-planner`
- `akita-prover` — commitment, proving, setup expansion, recursive/ring-switch witnesses, polynomial backends. Public surface is `<Cfg>`-generic: `prove_batched` (in `protocol::flow`), `commit` / `batched_commit` (in `api::commitment`), `recursive_w_commit_layout_for_d` (in `protocol::ring_switch`), and `bind_transcript_instance_descriptor` re-exported from `akita-config`. Multi-`D` dispatch helpers live here (not in scheme). Dep tree excludes `akita-planner`
- `akita-scheme` — thin end-to-end `AkitaCommitmentScheme` glue; each `CommitmentProver`/`CommitmentVerifier` method is a one-line call into the prover/verifier `<Cfg>`-generic API. Dep tree excludes `akita-planner`
- `akita-planner` — offline schedule search (`find_optimal_schedule::<Cfg>(key, allow_table_fast_path)`) plus the `gen_schedule_tables` binary that emits the generated tables. The `test_utils::PlannerCfg<Cfg>` DP-routing wrapper is gated behind the `test-utils` feature so production callers never link it. Depends on `akita-config` (Cfg-generic API). Sits ABOVE `akita-config` and is excluded from the prover/verifier/scheme/setup dep tree
- `akita-pcs` — umbrella crate with examples, benches, integration tests, public re-exports

## Key Abstractions

- `AkitaCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` — single user-facing trait defining every per-config policy hook (algebra, SIS family, decomposition, layout, schedule table/key/plan, transcript bind, prove/commitment params). Replaces the previous `CommitmentConfig` + `ScheduleProvider` + `PlannerConfig` triad. Verifier-reachable hooks return `Result<_, AkitaError>` (`level_params_with_log_basis`, `log_basis_at_level`, `stage1_challenge_config`)
- `WCommitmentConfig<const D: usize, Cfg>` — derived recursive-w `CommitmentConfig` for ring-degree dispatch
- `LevelParams` — recursion schedule, layout, per-level config
- `PlanPolicy` — value-typed inputs to `akita_derive::schedule_plan_from_table` (table materialization). Planner search is `<Cfg>`-generic — no separate `SearchOptions`-shaped input
- `PlannerCfg<Cfg>` — test-only `Cfg` wrapper in `akita_planner::test_utils` (gated by the `test-utils` Cargo feature) that overrides the `CommitmentConfig` parameter accessors to route schedule-table misses through `find_optimal_schedule::<Self>`. Production presets do not need this wrapper and never link it
- `DensePoly`, `OneHotPoly`, `AkitaPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaLevelProof`, `AkitaProofStep` — serialized proof structure (singleton openings are the 1x1 special case of the batched proof)
- `AkitaTranscript`, `Transcript` — spongefish-backed Fiat-Shamir layer
- `AkitaInstanceDescriptor` — canonical transcript preamble binding algebra, setup, plan, and call shape

## Verifier No-Panic Contract

Verifier-reachable execution is a no-panic boundary.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

This applies to `akita-verifier` and any verifier-reachable code in `akita-types`, `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths, `akita-config` (every `CommitmentConfig` method reachable from `verify_batched`), and `akita-derive` (SIS derivation + table materialization). `akita-planner` (offline DP search) is not on the verifier path and is excluded from the verifier crate's dep graph.
Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.

Prefer strengthening existing validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
Prover-only panics are acceptable for now if they are not reachable from verifier paths.

## Feature Flags

- `parallel` — Rayon parallelization (default)
- `planner` — opt-in dep on `akita-planner` (still defined on `akita-pcs`, `akita-scheme`, `akita-setup` for downstream callers that want the DP search at runtime; orthogonal to the `Cfg` trait defaults, which are always table-only)
- `akita-planner/test-utils` — gates `akita_planner::test_utils::PlannerCfg`; enabled in dev-dependencies of `akita-pcs`, `akita-scheme`, `akita-setup` so their test fixtures can drive table-miss DP fallback. Never enabled in production builds
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

Standalone sub-workspace at `profile/akita-recursion/` (excluded from this workspace, pinned to Rust 1.95 + RISC-V, applies Jolt's `[patch.crates-io]` overrides for `arkworks-algebra`). Full runbook, knob reference, current cycle results, and open follow-ups: [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
