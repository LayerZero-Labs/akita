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

## Documentation

Canonical policy: [`docs/documentation.md`](docs/documentation.md).
Narrative docs live in the [Akita Book](book/README.md); design records in `specs/` until folded ([`specs/PRUNING.md`](specs/PRUNING.md)).

- **Hard (CI):** dead symbols in live specs/docs, `Book-chapter:` paths, `mdbook build` — [`scripts/check-doc-guardrails.sh`](scripts/check-doc-guardrails.sh).
- **Soft (PR comment):** blast-radius advisory — [`docs/doc-blast-radius.json`](docs/doc-blast-radius.json).

## Verifier no-panic contract

Verifier-reachable code must reject malformed input with `AkitaError` or `SerializationError`, never panic.
Do not add verifier-reachable `panic!`, `assert!`, `unwrap`, unchecked indexing, or unbounded allocation without prior validation at a boundary.
Full contract: [`book/src/how/verification.md`](book/src/how/verification.md) and [`docs/verifier-contract.md`](docs/verifier-contract.md).

## Crate structure

Workspace members under `crates/`:

- `akita-field` — field traits, prime/extension fields, unreduced/packed helpers, FFT, parallel macros
- `akita-witness` — the single shared borrowed witness/polynomial view vocabulary (`PolynomialView` + the fallible `WitnessProvider` source trait) consumed by the sumcheck Tier-A kernel and the polyops standard views; depends only on `akita-field`, sits below `akita-sumcheck`/`akita-prover`
- `akita-serialization` — serialization/validation/compression traits
- `akita-algebra` — modules/vectors, NTTs, cyclotomic rings, sparse challenges, polynomials
- `akita-transcript` — spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — sumcheck proofs, drivers, compact folding, batching, accumulation; the generic declarative descriptor algebra (`descriptor` module: `Source`/`Term`/`Expr`/`SumcheckInstanceDescriptor`/`InstanceKind`/`ClaimSlot`, generic over identifier types) plus the panic-free `Expr::try_evaluate` evaluator. Names no protocol-specific identifier or equation
- `akita-protocol` — pure protocol *description* layer composed from `akita-sumcheck` building blocks: the concrete identifier types (`AkitaOpeningId`/`AkitaPublicId`/`AkitaChallengeId`), the per-stage formula constructors (e.g. `stage2_descriptor`), and the per-level protocol plan (`LevelProtocolPlan`, `StagePlan`, `BatchingScheme`, gating via `ProtocolGates`, transcript schedule, `plan_level`). Depends only on `akita-sumcheck`/`akita-types`/`akita-field`/`akita-challenges`; holds no engine code; sits below `akita-prover`/`akita-verifier`. Verifier-reachable, so panic-free
- `akita-types` — proof, setup, schedule, layout, commitment, transcript-append, PRG shapes; the verifier-reachable per-level proof-size formula (`level_proof_bytes`); the SIS security-floor tables (`akita_types::sis_floor`: `SisModulusFamily`, `min_rank_for_secure_width`, `ceil_supported_collision`); pure layout helpers (`level_layout_from_params`, `recursive_level_layout_from_params`, `decomp_depths`); SIS-secure layout derivation (`sis_derived_root_params_for_layout`, `sis_secure_level_params`). Generated schedule-table *types* and compact→`LevelParams` expansion live in `akita-planner`; shipped static table data lives in `akita-schedules`
- `akita-config` — runtime config presets, the single `CommitmentConfig` trait, config-backed schedule adapters, the `policy_of::<Cfg>()` bridge, the generated-table family list (`generated_families`) + `gen_schedule_tables` binary, and the canonical `bind_transcript_instance_descriptor` helper consumed by both prover and verifier. **Depends on `akita-planner` and optionally `akita-schedules`.** `CommitmentConfig::runtime_schedule` delegates to `akita_planner::resolve_schedule` with `Self::schedule_catalog()`; table miss falls back to the offline DP. Opt-in `schedules-*` features gate which preset catalogs link in; `schedules-default` preserves current dev/CI bundles
- `akita-schedules` — feature-gated static schedule table data + `*_table()` constructors; depends on `akita-planner` for generated types only (data-only, no preset types)
- `akita-setup` — config-backed setup construction + optional setup cache
- `akita-verifier` — verifier replay (no prover-only polynomial backends). **Depends on `akita-config`** and is directly `<Cfg>`-generic: `batched_verify::<Cfg, T, D>` (in `protocol::core::verify`) calls `Cfg::…` and `bind_transcript_instance_descriptor` directly — there is no `_with_policy` closure layer. Reaches `akita-planner` transitively via `akita-config` (DP fallback is verifier-reachable)
- `akita-prover` — commitment, proving, setup expansion, recursive/ring-switch witnesses, polynomial backends. **Depends on `akita-config`** and is directly `<Cfg>`-generic: `batched_prove::<Cfg, T, P, B, D>` (in `protocol::core::prove`), `commit::<Cfg, D, P, B>` / `batched_commit::<Cfg, D, P, B>` (in `api::commitment`), `commit_next_w::<Cfg, B, D>` and `prove_suffix::<Cfg, T, B, D>` (in `protocol`), calling `Cfg::…` and `bind_transcript_instance_descriptor` directly with no `_with_policy` closures. The root tensor-projection transform and the multi-`D` dispatch helpers live here (not in the scheme)
- `akita-planner` — pure, **`Cfg`-free** schedule engine. Holds generated schedule-table *types* (`GeneratedScheduleTable`, `GeneratedStep`, `table_entry`, `expand`, `schedule_from_entry`), catalog identity validation, the reusable table emitter (`emit::EmitSpec`), and the offline DP `find_schedule(key, &PlannerPolicy, ring_challenge_config, fold_challenge_shape_at_level)`. The single resolution entry point is `resolve_schedule(key, &PlannerPolicy, ring_challenge_config, fold_challenge_shape_at_level, catalog: Option<GeneratedScheduleTable>)`: validates catalog identity on a hit, expands the compact entry, or runs the DP on a miss. Shipped table *data* lives in `akita-schedules`, not here. Names no `CommitmentConfig` type; depends only on `akita-types` / `akita-challenges` / `akita-field`. Sits **below** `akita-config`: `CommitmentConfig::runtime_schedule` passes `Self::schedule_catalog()` into `resolve_schedule`. The preset family list and `gen_schedule_tables` binary live in `akita-config`; the binary adapts families into `EmitSpec` and writes into `crates/akita-schedules/src/generated/` (full regen: non-zk emit, `zk` emit, then `--wiring-only`)
- `akita-pcs` — umbrella crate with `AkitaCommitmentScheme` orchestration, examples, benches, integration tests, and public re-exports

## Key abstractions

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

## Feature flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence for some commitment flows
- `logging-transcript` — `LoggingTranscript` schedule events and smell checks

Details: [`book/src/usage/feature-flags.md`](book/src/usage/feature-flags.md).

## Maintainer pointers

| Topic | Where |
|-------|-------|
| Crate map and dependency graph | [`docs/crate-graph.md`](docs/crate-graph.md), [`book/src/how/architecture.md`](book/src/how/architecture.md) |
| Core API types | [`book/src/how/architecture.md`](book/src/how/architecture.md#core-types) |
| CI test timing | [`docs/ci-test-timing.md`](docs/ci-test-timing.md) |
| Profiling harness | [`book/src/usage/profiling.md`](book/src/usage/profiling.md) |
| Transcript hardening | [`specs/transcript-hardening.md`](specs/transcript-hardening.md) |
| Offline SIS table regen | `scripts/stitch_generated_sis_table.py` (Sage + pinned `third_party/lattice-estimator`) |
| Jolt verifier bench | [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md) |
