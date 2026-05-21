# Spec: Planner and Schedule Consolidation

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Codex               |
| Created     | 2026-05-21                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Akita currently has one runtime schedule path and one older planner/reporting path that can answer similar questions with different models. This spec consolidates planner, generated-table, setup, profile, and runtime schedule selection around one canonical schedule-profile pipeline, so every tool that asks "what schedule will Akita use?" gets the same answer with explicit provenance.

## Intent

### Goal

Build one canonical schedule resolution pipeline from a public root schedule profile to a validated `AkitaSchedulePlan` and executable `Schedule`, then route runtime config, generated-table materialization, table generation, profile tooling, setup sizing, and the planner CLI through that pipeline.

The consolidation modifies these key surfaces:

- `akita-types`: schedule profile/key types, generated table materialization, runtime `Schedule`/`Step`, planned schedule data, proof-size and witness-size evaluators.
- `akita-config`: `CommitmentConfig` schedule resolution, proof-optimized preset selectors, planner fallback policy, setup matrix envelope sizing, batched root layout selection.
- `akita-planner`: config-backed search, table-regeneration APIs, CLI/reporting, and removal or quarantine of the legacy universal planner model.
- `akita-pcs/examples/profile`: full/onehot preset selection and proof-size reporting.
- Existing specs: this follows the schedule-provider boundary in `specs/akita-pcs-crate-decomposition.md` and should be coordinated with `specs/planner-incidence-generalization.md`.

The intended end state has three clearly named questions:

- Generated lookup: is there a shipped generated schedule for this profile?
- Runtime resolution: what effective schedule will this config use for this profile?
- Fresh search: what schedule does the config-backed offline optimizer compute without consulting generated tables?

These questions must not share ambiguous names or silently fall back without reporting provenance.

### Invariants

- For a fixed config and public claim incidence, schedule resolution is deterministic and yields exactly one effective runtime `Schedule` or a clear `AkitaError`.
- Prover and verifier use the same effective runtime `Schedule`; `PlanSection::from_schedule` continues to bind the final effective schedule digest into the transcript descriptor.
- Generated schedule rows are a cache/provider artifact, not the schedule abstraction itself. Materializing a generated row must validate it into one canonical `AkitaSchedulePlan`.
- `AkitaSchedulePlan::exact_proof_bytes`, `schedule_from_plan(...).total_bytes`, `proof.size()`, and actual uncompressed serialization length agree for generated-hit profile cases.
- Terminal-fold accounting remains distinct from intermediate-fold accounting. Terminal folds use terminal witness layout, terminal `next_w_len`, packed direct witness shape when applicable, and omit proof objects that are absent from the terminal protocol step.
- SIS sizing remains config-aware and table-driven. No planner or reporting code may carry independent SIS floors, challenge L1 masses, challenge infinity norms, field-bit assumptions, or q128-only assumptions when a config hook or generated SIS table is available.
- Setup matrix envelope sizing remains max-over-supported-shapes and must account for non-monotone planned ranks across `num_vars`, batched polynomial counts, and opening-point counts.
- Normal verifier and prover crates remain free of `akita-planner` search dependencies. Any dynamic planner fallback stays in `akita-config` or offline tooling until explicitly removed.
- No verifier-reachable code path may gain new panic, `unwrap`, `expect`, unchecked indexing, unchecked slicing, or unbounded allocation from the consolidation.
- No compatibility aliases or shims are required for renamed planner/profile APIs; this repo permits breaking changes.

### Non-Goals

- Do not introduce mixed-D runtime schedules. The legacy universal planner explores a multi-D ladder, but making mixed-D schedules real would affect setup matrices, verifier replay, transcript binding, generated tables, and Jolt recursion assumptions.
- Do not make generated tables enumerate every possible grouped or multipoint batch shape. Generated tables remain a shipped cache for supported preset shapes; dynamic or external providers can cover misses.
- Do not optimize for verifier cycles or Jolt cycle counts in this spec. The design should leave room for future objective selection, but this round remains proof-byte preserving.
- Do not change transcript byte layout, proof serialization format, field arithmetic, sparse challenge distributions, or SIS security assumptions except where needed to remove duplicated constants and route through existing canonical sources.
- Do not keep `search.rs` as a public authoritative planner API. If it survives, it must be clearly named and scoped as a non-runtime diagnostic or research model.

## Evaluation

### Acceptance Criteria

- [ ] There is one high-level schedule resolver that returns the lookup/profile, source provenance, canonical `AkitaSchedulePlan` when available, executable `Schedule`, stable digest/key, and clear miss/error information.
- [ ] `CommitmentConfig::get_params_for_prove`, `get_params_for_commitment`, `commitment_layout`, batched root layout selection, setup envelope sizing, profile mode selection, and planner CLI reporting use the resolver or prove structural equivalence to it.
- [ ] The config-backed planner in `schedule_params.rs` is the only protocol-aware optimizer used for table generation and dynamic fallback.
- [ ] `run_universal_planner` and its planner-local `Schedule` model are removed, made private to a clearly named diagnostic module, or rewritten as an adapter over the canonical resolver/search engine.
- [ ] `AkitaScheduleLookupKey` semantics are clarified. Either replace it with a `RootScheduleProfile` type or remove surprising constructors such as `new(num_vars, t, w, z)` that infer `num_points` from `num_z_vectors`.
- [ ] Generated-table regeneration for non-ZK and ZK outputs produces no diff unless the implementation intentionally changes schedules and documents the reason.
- [ ] Representative generated plans and fresh config-backed search results match structurally for generated-covered keys.
- [ ] Canonical profile modes report schedules and proof sizes that match runtime proof serialization, including onehot `nv=32`.
- [ ] The no-planner feature surface is explicit: generated hits work where expected; generated misses fail with `AkitaError`; there is no accidental silent fallback.
- [ ] `rg`/dependency checks confirm `akita-verifier` and `akita-prover` do not depend on `akita-planner`, and normal runtime modules do not import the legacy diagnostic planner model.

### Testing Strategy

Add or update tests in these layers:

- `akita-types`: generated-entry materialization, schedule digest, terminal/intermediate witness-size evaluator, proof-byte sums, and invalid generated-entry rejection.
- `akita-config`: resolver source provenance, generated-table selectors, setup envelope coverage, batched root layout selection, no-planner generated-hit and generated-miss behavior.
- `akita-planner`: generated-fast-path lookup, from-scratch search, generated-vs-from-scratch parity, CLI output source labels, and absence/quarantine of legacy model exports.
- `akita-pcs`: profile-selected D/preset, schedule digest, fold count, terminal direct shape, planned bytes, actual `proof.size()`, and serialized byte length.

P0 checks:

```bash
cargo test -p akita-types
cargo test -p akita-config --features planner
cargo test -p akita-planner
cargo test -p akita-pcs --features planner
```

Feature matrix checks:

```bash
cargo test -p akita-config --no-default-features
cargo test -p akita-config --features planner,zk
cargo test -p akita-pcs --no-default-features
cargo test -p akita-pcs --all-features
cargo clippy --all --message-format=short -q -- -D warnings
```

Generated table checks:

```bash
cargo run -p akita-config --features planner --bin gen_schedule_tables -- <tmpdir>
cargo run -p akita-config --features planner,zk --bin gen_schedule_tables -- <tmpdir>
diff -ru <tmpdir> crates/akita-types/src/generated
```

Profile smoke checks for schedule/proof-byte equality:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 AKITA_NUM_POLYS=1 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 \
  cargo run --release -q -p akita-pcs --example profile

AKITA_MODE=full AKITA_NUM_VARS=32 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 \
  cargo run --release -q -p akita-pcs --example profile
```

If the implementation touches batched or small-field selectors, also run:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 \
  cargo run --release -q -p akita-pcs --example profile

AKITA_MODE=onehot_fp32_d32 AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 \
  AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 \
  cargo run --release -q -p akita-pcs --example profile
```

### Performance

This is a consolidation and cleanup round, so proof size and runtime-selected schedules should remain byte-for-byte stable unless a deliberately documented planner bug fix is included.

Expected performance behavior:

- Generated-hit runtime paths should not run dynamic DP search.
- Planner CLI and profile reporting may become more accurate, but should not force release-profile proving to do extra search work for generated-covered shapes.
- From-scratch table regeneration may keep its current runtime complexity.
- Schedule/proof-size bytes for generated-covered fp128 onehot/full cases should be exact matches before and after the cleanup.

Any proof-byte change must include:

- the affected config/profile/key;
- old and new schedule summaries;
- whether generated table files changed;
- a profile command showing actual serialized proof size equals the new planned size;
- a security/SIS explanation if ranks or challenge bounds changed.

## Design

### Architecture

The target architecture is:

```text
ClaimIncidenceSummary
  -> RootScheduleProfile
  -> ScheduleResolver<Cfg>
       -> GeneratedTableProvider
       -> SearchProvider (offline/dev fallback only)
       -> DirectRootFallback
  -> ResolvedSchedule
       -> AkitaSchedulePlan
       -> Schedule
       -> digest/source/stable id
```

`RootScheduleProfile` should be the public language for schedule selection. It should make all shape axes explicit:

- `num_vars`
- `num_points`
- `num_commitment_groups`, if this remains distinct from points
- `num_t_vectors`
- `num_w_vectors`
- `num_z_vectors`
- `num_public_rows` or equivalent public opening-row count

If the implementation keeps `AkitaScheduleLookupKey`, it must still remove ambiguous constructors and update docs so the distinction between opening points, commitment groups, public rows, `t`, `w`, and `z` vectors is not implicit.

`ResolvedSchedule` should record provenance:

```rust
pub enum ScheduleSource {
    Generated,
    PlannerFallback,
    RootDirectFallback,
}

pub struct ResolvedSchedule {
    pub profile: RootScheduleProfile,
    pub source: ScheduleSource,
    pub plan: Option<AkitaSchedulePlan>,
    pub schedule: Schedule,
    pub stable_id: String,
    pub digest: DescriptorDigest,
}
```

The exact API names can differ, but the information must exist so logs, CLI, profile output, setup cache keys, and tests can distinguish generated rows from dynamic fallback.

`akita-types` should own shared, config-neutral evaluators:

- generated entry validation;
- planned-to-runtime schedule conversion;
- witness element counts for intermediate and terminal layouts;
- level proof-byte accounting;
- schedule digesting or canonical stable identity helpers.

`akita-config` should own config-specific policy:

- decomposition;
- stage-1 challenge config;
- SIS modulus family;
- root-rank policy;
- generated table selection for concrete presets;
- resolver wiring for generated vs fallback sources.

`akita-planner` should own offline search:

- fresh config-backed DP search, currently `schedule_params.rs`;
- table generation support;
- optional diagnostic/reporting tools;
- no independent protocol constants when a config hook or `akita-types` evaluator exists.

The planner CLI should become a view over this architecture:

- default: show generated/runtime schedules for supported presets;
- explicit `--from-scratch`: run config-backed DP ignoring generated rows;
- explicit `--diagnostic-model` only if a non-runtime research model remains;
- all output labels must include source provenance.

### Migration Plan

1. Introduce `RootScheduleProfile` or an equivalent explicit replacement for `AkitaScheduleLookupKey`.
2. Add `ResolvedSchedule` and a config-backed resolver API.
3. Move profile full/onehot selection onto resolver-backed preset selectors. Do not silently default to D32 on generated miss.
4. Route `get_params_for_prove`, commitment layout selection, batched root layout, setup envelope sizing, and profile reporting through resolver-backed code.
5. Centralize terminal/intermediate witness and proof-size accounting so generated materialization and from-scratch search share the same evaluator.
6. Rewrite `akita-planner` CLI around generated/runtime and from-scratch config-backed search.
7. Remove `search.rs` from public exports, delete it, or rename it as a private diagnostic/research module with explicit non-runtime labels.
8. Add structural schedule fixtures or helper assertions for representative generated and fallback keys.
9. Regenerate tables and verify no diff unless intentionally changing schedule behavior.
10. Update docs and specs.

### Alternatives Considered

Keep `search.rs` and patch constants.

This fixes individual drift bugs but leaves a second planner mental model with duplicated proof-size, terminal-witness, SIS, and challenge logic. It is not acceptable as a long-term authoritative planner.

Make generated tables the only schedule source.

This would simplify runtime, but it would make arbitrary grouped and multipoint shapes fail unless every shape is precomputed. The existing crate-decomposition spec explicitly treats generated tables as a finite cache, not the scheduling abstraction.

Move all planner/config/schedule code into `akita-types`.

This would collapse indirection, but it would drag offline search and config policy toward verifier-facing crates. `akita-types` should own inert data shapes and shared validation/evaluation helpers, not search algorithms or concrete config policy.

Introduce mixed-D runtime schedules now.

The universal model makes this tempting, but it is too broad for this cleanup. Mixed-D runtime schedules would require a separate protocol/setup/verifier/transcript design.

Use byte snapshots as the primary regression guard.

Exact byte equality is useful for selected profile smoke cases, but structural equivalence is the primary guardrail. Byte-only snapshots make legitimate protocol changes noisy and do not explain which schedule invariant changed.

## Documentation

Update:

- `crates/akita-planner/src/lib.rs` module docs to distinguish runtime resolver, fresh search, generated cache, and any diagnostic model.
- `crates/akita-planner/src/bin/akita-planner.rs` help/output text to label sources.
- `crates/akita-config/src/lib.rs` docs for schedule resolution and planner fallback.
- `crates/akita-pcs/examples/profile` docs or emitted profile summary to show selected preset, source, schedule digest, and planned-vs-actual bytes.
- `specs/akita-pcs-crate-decomposition.md` with a short note that this follow-up has superseded the planner-selector TODO.
- `specs/planner-incidence-generalization.md` if `RootScheduleProfile` lands before or alongside that incidence cleanup.

## Execution

Recommended implementation slices:

1. Add resolver types and provenance without changing behavior.
2. Cut profile and CLI reporting over to generated/runtime resolver output.
3. Cut runtime config helpers over to resolver internals.
4. Centralize witness/proof-size evaluators and update planner search to call them.
5. Remove/quarantine the legacy universal model.
6. Add fixtures/regeneration tests and feature-matrix checks.

Risks to resolve early:

- whether `RootScheduleProfile` should replace `AkitaScheduleLookupKey` in one full cutover or land as a new internal type first;
- whether setup cache keys should include a digest of `AkitaSchedulePlan` or final runtime `Schedule`;
- whether planner fallback remains enabled by default in `akita-pcs` or becomes an explicit developer feature;
- how much of `gen_schedule_tables` should move into `akita-planner` versus remain in `akita-config` while config presets live there.

## References

- `specs/akita-pcs-crate-decomposition.md`
- `specs/planner-incidence-generalization.md`
- `specs/terminal-fold-cutover.md`
- `specs/SPEC_REVIEW.md`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-types/src/layout/proof_size.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-config/src/schedule_policy.rs`
- `crates/akita-config/src/bin/gen_schedule_tables.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-planner/src/search.rs`
- `crates/akita-planner/src/bin/akita-planner.rs`
- `crates/akita-pcs/examples/profile`
- `profile/akita-recursion/README.md`
