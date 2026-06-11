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

## CI test timing

Every PR gets an upserted timing comment (marker `<!-- akita-ci-test-timing -->`)
showing per-pass wall time vs a main baseline, critical-path wall time when passes
run in parallel, plus per-test outliers from the nextest JUnit output. CI runs the
non-zk and all-features nextest passes in parallel matrix jobs (`slice:index/total`
via `matrix.shard` / `strategy.job-total`; 1-based index, not `strategy.job-index`),
with [Swatinem/rust-cache](https://github.com/Swatinem/rust-cache) per pass (`cache: false`
on `setup-rust-toolchain` so the explicit shared-key step owns `target/`). The
`test-timing` job merges shard JUnit and uploads artifact `ci-test-timing-data`
containing `summary.json` and the rendered comment/report.

## Crate Structure

Workspace members under `crates/`:

- `akita-field` — field traits, prime/extension fields, unreduced/packed helpers, FFT, parallel macros
- `akita-serialization` — serialization/validation/compression traits
- `akita-algebra` — modules/vectors, NTTs, cyclotomic rings, sparse challenges, polynomials
- `akita-transcript` — spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — sumcheck proofs, drivers, compact folding, batching, accumulation
- `akita-types` — proof, setup, schedule, layout, commitment, transcript-append, PRG shapes; the verifier-reachable per-level proof-size formula (`level_proof_bytes`); the SIS security-floor tables (`akita_types::sis_floor`: `SisModulusFamily`, `min_rank_for_secure_width`, `ceil_supported_collision`); pure layout helpers (`level_layout_from_params`, `recursive_level_layout_from_params`, `decomp_depths`); SIS-secure layout derivation (`sis_derived_root_params_for_layout`, `sis_secure_level_params`). The generated schedule-table *representation*, shipped tables, and on-demand expansion live in `akita-planner` (not here)
- `akita-config` — runtime config presets, the single `CommitmentConfig` trait, config-backed schedule adapters, the `policy_of::<Cfg>()` bridge, the generated-table family list (`generated_families`) + `gen_schedule_tables` binary, and the canonical `bind_transcript_instance_descriptor` helper consumed by both prover and verifier. **Depends on `akita-planner`.** `CommitmentConfig::runtime_schedule` is a one-line delegation to `akita_planner::get_schedule(key, &policy_of::<Self>(), …)`; the planner owns table selection (`shipped_table`), so the trait has **no** `schedule_table()` / `resolve_schedule()` hooks. Runtime DP fallback on a table miss is the default for every preset (no opt-in wrapper, no `test-utils` feature)
- `akita-setup` — config-backed setup construction + optional setup cache
- `akita-verifier` — verifier replay (no prover-only polynomial backends). **Depends on `akita-config`** and is directly `<Cfg>`-generic: `verify_batched::<Cfg, T, D>` (in `protocol::batched`) calls `Cfg::…` and `bind_transcript_instance_descriptor` directly — there is no `_with_policy` closure layer. Reaches `akita-planner` transitively via `akita-config` (DP fallback is verifier-reachable)
- `akita-prover` — commitment, proving, setup expansion, recursive/ring-switch witnesses, polynomial backends. **Depends on `akita-config`** and is directly `<Cfg>`-generic: `prove_batched::<Cfg, T, P, B, D>` (in `protocol::flow`), `commit::<Cfg, D, P, B>` / `batched_commit::<Cfg, D, P, B>` (in `api::commitment`), `commit_next_w::<Cfg, B, D>` and `prove_suffix::<Cfg, T, B, D>` (in `protocol`), calling `Cfg::…` and `bind_transcript_instance_descriptor` directly with no `_with_policy` closures. The root tensor-projection transform and the multi-`D` dispatch helpers live here (not in the scheme)
- `akita-planner` — pure, **`Cfg`-free** schedule owner. It holds the generated schedule-table representation (`Generated*` types, `table_entry`, `generated_schedule_lookup_key`), the shipped `src/generated/*.rs` tables + `*_table()` constructors, the `policy → table` registry `shipped_table(&PlannerPolicy, root_fold_is_tensor)`, the on-demand compact→`LevelParams` expansion (`generated::expand`, `schedule_from_entry`), and the schedule-search DP `find_schedule(key, &PlannerPolicy, stage1, fold_shape)`. The single resolution entry point is `get_schedule(key, &PlannerPolicy, stage1, fold_shape)` — it selects the shipped table from the policy (and the level-0 fold shape, which disambiguates the tensor table), expands the compact entry on a hit, and runs the DP on a miss. It names no `CommitmentConfig` type and depends only on `akita-types` / `akita-challenges` / `akita-field`. Sits **BELOW** `akita-config` (the arrow is inverted): `akita-config::runtime_schedule` is a one-line delegation to `get_schedule`. The preset family list and the `gen_schedule_tables` binary live in `akita-config` (the only crate that can name presets); the binary writes its output into `crates/akita-planner/src/generated/`
- `akita-pcs` — umbrella crate with `AkitaCommitmentScheme` orchestration, examples, benches, integration tests, and public re-exports

## Key Abstractions

- `AkitaCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` — single user-facing trait defining every per-config policy hook (algebra, SIS family, decomposition, layout, schedule table/key/plan, transcript bind, prove/commitment params). Replaces the previous `CommitmentConfig` + `ScheduleProvider` + `PlannerConfig` triad. Verifier-reachable hooks return `Result<_, AkitaError>` (`level_params_with_log_basis`, `log_basis_at_level`, `ring_challenge_config`)
- `LevelParams` — recursion schedule, layout, per-level config
- `PlanPolicy` — value-typed inputs to `akita_types::schedule_plan_from_table` (table materialization)
- `PlannerPolicy` — the `Cfg`-free plain-value projection of a preset (`D`, decomposition, SIS family, norm bound, ext degrees, basis range) that `akita_planner::find_schedule` consumes. Derived from a preset via `akita_config::policy_of::<Cfg>()` — the single source of truth stays the `Cfg` impl, never hand-written literals
- `DensePoly`, `OneHotPoly`, `AkitaPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaLevelProof`, `AkitaProofStep` — serialized proof structure (singleton openings are the 1x1 special case of the batched proof)
- `AkitaTranscript`, `Transcript` — spongefish-backed Fiat-Shamir layer
- `AkitaInstanceDescriptor` — canonical transcript preamble binding algebra, setup, plan, and call shape

## Verifier No-Panic Contract

Verifier-reachable execution is a no-panic boundary.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

This applies to `akita-verifier` and any verifier-reachable code in `akita-types` (including SIS derivation + table materialization), `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths, `akita-config` (every `CommitmentConfig` method reachable from `verify_batched`), and `akita-planner` (the schedule-search DP). The DP is now verifier-reachable through `CommitmentConfig::runtime_schedule`'s table-miss fallback, so `find_schedule` and every helper it calls must reject malformed input with `AkitaError`, never panic. The verifier must still validate `key.num_vars` against setup capacity before invoking the DP so a malformed proof cannot blow up the search's bounded state space.
Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.

Prefer strengthening existing validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
Prover-only panics are acceptable for now if they are not reachable from verifier paths.

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

## Offline SIS table regen

`scripts/stitch_generated_sis_table.py` regenerates `generated_sis_table/` using Sage and the
pinned `third_party/lattice-estimator` checkout (`git submodule update --init`).
Reference replay: `sage -python scripts/sis_golden/check.py`. Rust CI does not
require Sage or an initialized submodule.

## Profiling

Canonical: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`. Under committed-fold A-role SIS pricing the planner's `total_bytes` optimum is **D=32 or D=64** (D32 is marginally smaller; D128 is ~20% larger). Use `akita_config::proof_optimized::fp128::best_onehot_schedule` / `best_full_schedule` to pick across D32/D64/D128. D128 has no shipped table and is resolved via runtime DP only.

Knobs (`AKITA_MODE`, `AKITA_NUM_VARS`, `AKITA_PROFILE_TRACE`, `AKITA_PROFILE_LOG`, `AKITA_PROFILE_ANSI`, `AKITA_PROFILE_SPAN_CLOSES`, `AKITA_ALLOW_DEBUG_PROFILE`): defaults and details in `examples/profile.rs`. `RAYON_NUM_THREADS` caps Rayon threads; `--no-default-features` disables `parallel`. The `--release` guard can be bypassed with `AKITA_ALLOW_DEBUG_PROFILE=1`.

## Running the verifier inside Jolt

Standalone sub-workspace at `profile/akita-recursion/` (excluded from this workspace, pinned to Rust 1.95 + RISC-V, applies Jolt's `[patch.crates-io]` overrides for `arkworks-algebra`). Full runbook, knob reference, current cycle results, and open follow-ups: [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
