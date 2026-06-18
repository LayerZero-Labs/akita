# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh   # when changing book, specs, or docs/
```

## Documentation guardrails

Canonical policy: [`docs/documentation.md`](docs/documentation.md). The [Akita Book](book/README.md)
is the narrative; `specs/` is the design-record library until folded
([`specs/PRUNING.md`](specs/PRUNING.md)).

- **Hard (CI):** `Documentation guardrails` workflow — dead symbols in live specs,
  `Book-chapter:` paths, `mdbook build` (`scripts/check-doc-guardrails.sh`).
- **Soft (PR comment):** blast-radius advisory (`<!-- akita-doc-blast-radius -->`),
  from `docs/doc-blast-radius.json` via `scripts/doc_blast_radius.py`.

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
- `akita-witness` — the single shared borrowed witness/polynomial view vocabulary (`PolynomialView` + the fallible `WitnessProvider` source trait) consumed by the sumcheck Tier-A kernel and the polyops standard views; depends only on `akita-field`, sits below `akita-sumcheck`/`akita-prover`
- `akita-serialization` — serialization/validation/compression traits
- `akita-algebra` — modules/vectors, NTTs, cyclotomic rings, sparse challenges, polynomials
- `akita-transcript` — spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — sumcheck proofs, drivers, compact folding, batching, accumulation
- `akita-types` — proof, setup, schedule, layout, commitment, transcript-append, PRG shapes; the verifier-reachable per-level proof-size formula (`level_proof_bytes`); the SIS security-floor tables (`akita_types::sis_floor`: `SisModulusFamily`, `min_rank_for_secure_width`, `ceil_supported_collision`); pure layout helpers (`level_layout_from_params`, `recursive_level_layout_from_params`, `decomp_depths`); SIS-secure layout derivation (`sis_derived_root_params_for_layout`, `sis_secure_level_params`). Generated schedule-table *types* and compact→`LevelParams` expansion live in `akita-planner`; shipped static table data lives in `akita-schedules`
- `akita-config` — runtime config presets, the single `CommitmentConfig` trait, config-backed schedule adapters, the `policy_of::<Cfg>()` bridge, the generated-table family list (`generated_families`) + `gen_schedule_tables` binary, and the canonical `bind_transcript_instance_descriptor` helper consumed by both prover and verifier. **Depends on `akita-planner` and optionally `akita-schedules`.** `CommitmentConfig::runtime_schedule` delegates to `akita_planner::resolve_schedule` with `Self::schedule_catalog()`; table miss falls back to the offline DP. Opt-in `schedules-*` features gate which preset catalogs link in; `schedules-default` preserves current dev/CI bundles
- `akita-schedules` — feature-gated static schedule table data + `*_table()` constructors; depends on `akita-planner` for generated types only (data-only, no preset types)
- `akita-setup` — config-backed setup construction + optional setup cache
- `akita-verifier` — verifier replay (no prover-only polynomial backends). **Depends on `akita-config`** and is directly `<Cfg>`-generic: `batched_verify::<Cfg, T, D>` (in `protocol::core::verify`) calls `Cfg::…` and `bind_transcript_instance_descriptor` directly — there is no `_with_policy` closure layer. Reaches `akita-planner` transitively via `akita-config` (DP fallback is verifier-reachable)
- `akita-prover` — commitment, proving, setup expansion, recursive/ring-switch witnesses, polynomial backends. **Depends on `akita-config`** and is directly `<Cfg>`-generic: `batched_prove::<Cfg, T, P, B, D>` (in `protocol::core::prove`), `commit::<Cfg, D, P, B>` / `batched_commit::<Cfg, D, P, B>` (in `api::commitment`), `commit_next_w::<Cfg, B, D>` and `prove_suffix::<Cfg, T, B, D>` (in `protocol`), calling `Cfg::…` and `bind_transcript_instance_descriptor` directly with no `_with_policy` closures. The root tensor-projection transform and the multi-`D` dispatch helpers live here (not in the scheme)
- `akita-planner` — pure, **`Cfg`-free** schedule engine. Holds generated schedule-table *types* (`GeneratedScheduleTable`, `GeneratedStep`, `table_entry`, `expand`, `schedule_from_entry`), catalog identity validation, and the offline DP `find_schedule(key, &PlannerPolicy, ring_challenge_config, fold_shape)`. The single resolution entry point is `resolve_schedule(key, &PlannerPolicy, ring_challenge_config, fold_shape, catalog: Option<&GeneratedScheduleTable>)`: validates catalog identity on a hit, expands the compact entry, or runs the DP on a miss. Shipped table *data* lives in `akita-schedules`, not here. Names no `CommitmentConfig` type; depends only on `akita-types` / `akita-challenges` / `akita-field`. Sits **below** `akita-config`: `CommitmentConfig::runtime_schedule` passes `Self::schedule_catalog()` into `resolve_schedule`. The preset family list and `gen_schedule_tables` binary live in `akita-config`; the binary writes into `crates/akita-schedules/src/generated/`
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
- `OpeningBatch` — normalized single-point batch descriptor for batched prove/verify (`crates/akita-types/src/proof/opening_batch.rs`). One shared opening point per call; multipoint removed. Production folded path: one commitment bundling `N` polynomials. See [`specs/single-point-opening-batch.md`](specs/single-point-opening-batch.md).
- `AkitaTranscript`, `Transcript` — spongefish-backed Fiat-Shamir layer
- `AkitaInstanceDescriptor` — canonical transcript preamble binding algebra, setup, plan, and call shape

## Verifier No-Panic Contract

Verifier-reachable execution is a no-panic boundary.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

This applies to `akita-verifier` and any verifier-reachable code in `akita-types` (including SIS derivation + table materialization), `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, verifier-used `akita-field` paths, `akita-config` (every `CommitmentConfig` method reachable from `batched_verify`), and `akita-planner` (the schedule-search DP). The DP is now verifier-reachable through `CommitmentConfig::runtime_schedule`'s table-miss fallback, so `find_schedule` and every helper it calls must reject malformed input with `AkitaError`, never panic. The verifier must still validate `key.num_vars` against setup capacity before invoking the DP so a malformed proof cannot blow up the search's bounded state space.
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

Canonical: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.

Under committed-fold A-role SIS pricing, **fp128** production is **D=64** (exact-shell; ~20% smaller than D128). Shipped tables: `fp128_d64_onehot`, `fp128_d64_full`, `fp128_d128_*`. **D32** presets always use planner DP (no shipped table). **fp32/fp64** D32/D64 are not securable; smallest secure choice is **D128 one-hot** (CI benches at nv=28). Use `akita_config::proof_optimized::fp128::best_onehot_schedule` / `best_full_schedule` to compare ring degrees. See `.github/workflows/profile-bench.yml` for the active CI matrix. CI profile builds use `--no-default-features --features parallel,profile-ci`; when adding a bench case, extend the mode→feature table in `scripts/check_profile_ci_features.sh`.

Knobs (`AKITA_MODE`, `AKITA_NUM_VARS`, `AKITA_PROFILE_TRACE`, `AKITA_PROFILE_LOG`, `AKITA_PROFILE_ANSI`, `AKITA_PROFILE_SPAN_CLOSES`, `AKITA_ALLOW_DEBUG_PROFILE`): defaults and details in `examples/profile.rs`. `RAYON_NUM_THREADS` caps Rayon threads; `--no-default-features` disables `parallel`. The `--release` guard can be bypassed with `AKITA_ALLOW_DEBUG_PROFILE=1`.

## Running the verifier inside Jolt

Standalone sub-workspace at `profile/akita-recursion/` (excluded from this workspace, pinned to Rust 1.95 + RISC-V, applies Jolt's `[patch.crates-io]` overrides for `arkworks-algebra`). Full runbook, knob reference, current cycle results, and open follow-ups: [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
