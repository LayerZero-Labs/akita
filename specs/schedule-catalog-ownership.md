# Spec: Schedule catalog ownership and opt-in shipped tables

> **Pre-zk-strip historical.** This spec predates the zk-strip
> ([`akita-zk-strip-for-audit.md`](akita-zk-strip-for-audit.md)). References to
> `zk_enabled` or `feature = "zk"` describe removed catalog paths preserved on
> `zk-wip`.

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-18 |
| Status        | proposed |
| PR            | [#203](https://github.com/LayerZero-Labs/akita/pull/203) |
| Supersedes    | (partial) schedule-key scope in [`planner-incidence-generalization.md`](planner-incidence-generalization.md) |
| Superseded-by | |
| Book-chapter  | how/configuration.md |

## Summary

Akita precomputes optimal fold/commit **schedules** offline and ships them as large
static Rust tables (~1.8 MiB of generated source today). Runtime resolution consults
those tables first and falls back to an exhaustive DP search (`find_schedule`) on a
miss. Correctness never depends on the tables; they are a performance cache.

Today the cache is **centrally owned and globally linked**: every preset funnels
through `akita_planner::get_schedule` → `shipped_table()`, a hardcoded registry in
`resolve.rs` that imports **all** generated families. Any binary that resolves a
schedule (including the `profile` example used in CI benchmarks) links every shipped
table, even when it only exercises one preset at runtime. Downstream integrators
(e.g. Jolt) cannot own schedule rows for their batch shapes without upstreaming into
`akita-config::ALL_GENERATED_FAMILIES` and regenerating monolithic artifacts (see
LayerZero PR #198, wide same-point batches for D64 one-hot).

This spec **splits the schedule engine from the schedule catalog**, moves shipped
tables into a dedicated `akita-schedules` crate (still in the Akita repo, still on
`main`), and makes catalog attachment **opt-in per `CommitmentConfig` preset** via
`schedule_catalog()`. The selection logic that `shipped_table()` performs at runtime
is redundant: the caller (`Cfg::runtime_schedule`) already knows the preset
statically, so each preset names its own catalog and the global registry is deleted.
The planner keeps DP fallback as the universal correctness path.

Because a preset can now hand the planner an arbitrary catalog, every generated table
also carries a compact **catalog identity**. `resolve_schedule` validates that
identity against the caller's `PlannerPolicy`, root fold shape, ZK mode, and generated
ring-challenge digest before consulting the entries. A mismatched catalog is rejected
with `AkitaError`, not silently treated as a cache miss. Table absence still falls
back to DP.

Downstream repos ship their own catalogs (Jolt: D64 one-hot + Jolt-specific batch
widths) without depending on `akita-schedules`. The engine cutover includes the small
reusable emit surface needed to generate such catalogs from an explicit
`PlannerPolicy` + hook bundle, so downstream ownership is available when this design
lands rather than deferred to a later archaeology step.

Per-preset feature gating controls link stripping. Plain dev builds keep current
table-backed behavior through forwarded default schedule bundles on the top-level
crates that currently disable `akita-config` defaults. The CI `profile` binary is the
exception: it builds with `--no-default-features --features parallel,profile-ci`, so
it links only the schedule families whose `schedules-*` feature is enabled by
`profile-ci`. A hard CI guard reads the existing `AKITA_BENCH_CASES` list in
`profile-bench.yml` and asserts `profile-ci` enables the schedule feature each
benched mode needs. No new manifest source-of-truth, exporter, or three-way
reconciler is introduced.

## Intent

### Goal

Refactor schedule resolution so:

1. **`akita-planner` is engine-only** (DP + compact entry expansion + generic table
   lookup). No global preset→table registry.
2. **`akita-schedules` holds all Akita-shipped tables** on `main`, feature-gated per
   family, optional for consumers.
3. **Each `CommitmentConfig` preset opts in** to zero or one catalog via
   `schedule_catalog() -> Option<GeneratedScheduleTable>` (default `None` means DP
   only). The table is a `Copy` value (identity + `&'static` entry slice), returned
   by value as `shipped_table` already does.
4. **Catalog identity is verifier-reachable validation**, not just documentation.
   A table whose identity does not match the current preset and hooks is rejected
   before lookup.
5. **Downstream projects** (Jolt) split runtime policy from catalog data. A Jolt
   runtime config crate may depend on `akita-config` to implement `CommitmentConfig`;
   its generated `jolt-schedules` data crate depends only on `akita-planner` and does
   not depend on `akita-config` or `akita-schedules`. The Akita emitter exposes the
   minimal API needed to create those catalogs from an explicit emit spec.
6. **Default Akita crate builds preserve current table-backed behavior**, even for
   crates that depend on `akita-config` with `default-features = false`, by forwarding
   a `schedules-default` bundle from `akita-pcs`, `akita-prover`, `akita-verifier`,
   and `akita-setup`.
7. **CI `profile` builds with `--no-default-features --features parallel,profile-ci`**,
   linking only schedules for modes in the CI benchmark matrix, enforced by a minimal
   guard over the existing `AKITA_BENCH_CASES` (no new manifest SSOT).
8. **Rename planner misnomers** (`stage1` closure parameters) to match
   `CommitmentConfig` vocabulary exactly (`ring_challenge_config`).

Schedule lookup remains keyed by **same-point opening batches only** (see
[`single-point-opening-batch.md`](single-point-opening-batch.md)). Multipoint and
general incidence graphs are out of scope.

### Motivation and context

#### Why schedules exist

The Akita PCS recursion picks per-level parameters (`log_basis`, fold split `m/r`,
matrix ranks, optional tiering) to minimize proof size subject to SIS security
floors. An offline DP (`find_schedule`) searches this space per lookup key. The
search is deterministic but not cheap at scale (CI drift tests today spend hundreds
of seconds re-running DP across all shipped keys).

**Generated schedule tables** store the DP output in compact form
(`GeneratedFoldStep`: `ring_d`, `log_basis`, `m_vars`, `r_vars`, `n_a`, `n_b`,
`n_d`, optional tier fields). At runtime `schedule_from_entry` expands a table row
into full `LevelParams` and a `Schedule` with correct proof-size accounting.

#### Why the current design hurts

| Problem | Symptom |
|---------|---------|
| Global `shipped_table()` registry | Every consumer links all families (~1.8 MiB source, all presets) |
| `ALL_GENERATED_FAMILIES` in `akita-config` | Adding Jolt batch width (e.g. 38 polys) requires upstreaming into Akita core |
| `profile` example registers 17+ modes | CI bench runs one mode at a time but compiles/links all tables |
| Planner parameter named `stage1` | Collides with sumcheck "stage 1"; actually means ring fold challenge policy |
| Stale incidence spec | `planner-incidence-generalization.md` describes APIs removed by single-point cutover |

#### High-level intuition

Think of schedules as **a cache layer over the planner DP**, analogous to query-plan
caches in databases:

- **Engine** (`find_schedule`, `schedule_from_entry`): how to compute or expand a plan.
- **Catalog** (static `GeneratedScheduleTable`): precomputed plans for known keys.
- **Preset** (`CommitmentConfig`): algebra/policy hooks + *which catalog, if any*,
  to consult before falling back to the engine.

The refactor makes the cache **pluggable per preset** instead of **one global cache
everyone shares at link time**.

```text
                    ┌─────────────────────────────────────┐
  OpeningBatch      │  CommitmentConfig::runtime_schedule │
  (same-point)  ──► │    resolve_schedule(                 │
                    │      key, policy,                    │
                    │      ring_challenge_config,          │
                    │      fold_challenge_shape_at_level,  │
                    │      Self::schedule_catalog(),  ◄── opt-in
                    │    )                                 │
                    └──────────────┬──────────────────────┘
                                   │
                     catalog hit   │   miss
                         ▼         ▼
              schedule_from_entry   find_schedule (DP)
                         │         │
                         └────┬────┘
                              ▼
                          Schedule
```

Downstream (Jolt) ships `jolt-schedules` and sets `JoltD64OneHot::schedule_catalog()`
to that table. Akita benches enable only the families the benchmark matrix needs via
the `profile-ci` feature.

#### Canonical entry walker (implemented)

Compact table rows are expanded through a single walker,
[`walk_generated_schedule_entry`](../crates/akita-planner/src/generated/walk.rs),
with two modes:

- **`Validate`**: admissibility checks only (`validate_generated_schedule_entry`).
- **`Materialize`**: the same walk plus a runtime [`Schedule`]
  (`schedule_from_entry`).

Both paths audit SIS ranks, witness transitions, and proof-byte totals in one pass.
There is no second expand path on catalog hits: `resolve_schedule` validates catalog
identity, then calls `schedule_from_entry` once.

**Catalog identity cache.** `validate_catalog_identity` memoizes successful checks
per `(table pointer, policy digest, identity digest)` and re-checks only the
ring-challenge hook digest on cache hits (hooks can vary per caller without
re-walking every entry).

**CI drift dedup.** `generated_schedule_tables_match_find_schedule` validates each
shipped family once via `validate_generated_schedule_table`, then compares DP output
per key through `schedule_from_entry` (table hit) instead of re-running full
`resolve_schedule` validation on every key.

### Invariants

- **DP is the source of truth.** Any shipped catalog row MUST match
  `find_schedule(key, policy, …)` for the same policy and hooks. Drift guards enforce
  this per catalog (today: `generated_schedule_tables_match_find_schedule`; becomes
  per-family tests in `akita-schedules` and downstream catalogs).
- **Catalog identity is checked before lookup.** A catalog is not just an entry slice.
  It carries the generated policy identity, ZK mode, root fold shape, and a digest of
  the ring-challenge configs used for every ring dimension represented by the table.
  `resolve_schedule` compares that identity with the caller's `PlannerPolicy`,
  `ring_challenge_config`, and `fold_challenge_shape_at_level`. A mismatch returns
  `AkitaError::InvalidSetup`; it must not fall through to DP because that would hide
  a broken preset/cargo-feature wiring bug.
- **Verifier no-panic.** `resolve_schedule` and `find_schedule` return `Result`,
  never panic on malformed keys (existing contract).
- **Determinism.** Prover and verifier both call `Cfg::runtime_schedule` with the
  same `policy_of::<Cfg>()` and hook fns; transcript `PlanSection` digest unchanged
  for identical inputs. A table-backed prover and a DP-only verifier (different
  `schedules-*` feature sets) still agree **because** the drift guard enforces
  table ≡ DP for every shipped key; this is the load-bearing reason opt-out is safe.
- **Default features preserve current behavior at the crate boundary.** `akita-config`
  exposes `schedules-default`, and every Akita crate that currently suppresses
  `akita-config` defaults forwards an equivalent default schedule bundle. Plain
  `cargo test` / `cargo build` / CI resolve byte-identical schedules to today. Only
  explicit minimal/downstream consumers (Jolt) turn `default-features = false` and
  omit `schedules-*`.
- **Default is DP-only for non-opted-in presets.** `schedule_catalog()` default
  `None` must not change schedules for presets that do not opt in (modulo explicit
  preset feature enables).
- **ZK mode selects ZK table data for ZK-capable families.** When `zk` is enabled
  together with a schedule family that has ZK data, Cargo forwards `zk` into
  `akita-schedules` through a weak dependency feature, so the accessor cannot pair ZK
  planner semantics with non-ZK table data. Non-ZK-only families (currently tiered)
  are inert under `zk` and excluded from the ZK drift guard.
- **Same-point batching only.** Lookup keys derive from `OpeningBatch` /
  `OpeningBatch::new(num_vars, num_polys)` (and `new_from_opening_batch`).
  No multipoint keys, no `ClaimIncidenceSummary` schedule path (type not in tree).
- **Table miss falls back to DP**, never errors solely because a row is absent (unless
  DP itself rejects the key).
- **CI bench coverage.** Every mode in `AKITA_BENCH_CASES` must have its `schedules-*`
  feature enabled by `profile-ci`, so the bench measures the shipped table rather than
  the DP fallback (one-directional hard CI check). The benchmark binary is built with
  defaults disabled so `profile-ci` can actually reduce linked table data.

### Non-Goals

- **Multipoint or general incidence schedule keys.** Removed by
  [`single-point-opening-batch.md`](single-point-opening-batch.md). Do not revive
  `RootPlannerProfile` / `ClaimIncidenceSummary` for planner input in this refactor.
- **Mixed `natural_num_vars` per slot in one batch.** Future work if a caller exists;
  not part of this cutover.
- **Mandatory catalogs for library users.** Opt-in only; `None` + DP is supported.
- **Moving Akita shipped tables out of the Akita repo.** Tables stay on `main` in
  `akita-schedules`; only *subscription* is optional.
- **Per-case separate `profile` binaries in default CI** (Option A). We choose Option
  B: one `profile-ci` feature union. Optional strict footprint job may come later.
- **Renaming sumcheck stage modules** (`akita_stage1`, `stages/stage1.rs`). Those
  refer to a different protocol stage.

## Evaluation

### Acceptance Criteria

#### Engine and API

- [ ] `akita_planner::shipped_table` and `get_schedule` are removed, along with the
  global `crate::generated::*_table` import block and the `root_fold_is_tensor`
  disambiguation hack they required.
- [ ] `akita_planner::resolve_schedule(key, policy, ring_challenge_config, fold_challenge_shape_at_level, catalog: Option<GeneratedScheduleTable>)` is the single runtime entry point (catalog passed **by value** because `GeneratedScheduleTable` is `Copy`).
- [ ] `GeneratedScheduleTable` contains a validated identity, not only `sis_family`:
  generated policy fields, `zk_enabled`, root fold shape, ring dimensions covered by
  the table, and a deterministic digest of `ring_challenge_config(d)` for those
  dimensions.
- [ ] `resolve_schedule` validates the catalog identity against the caller before
  `table_entry`. A mismatch returns `AkitaError::InvalidSetup` with the family name
  and the first mismatched field. A missing row still falls back to DP.
- [ ] Add a negative unit test with a deliberately miswired catalog (e.g. D64 full
  preset given a D64 one-hot catalog) proving `resolve_schedule` rejects the catalog
  instead of falling back to DP.
- [ ] Planner public closures use `ring_challenge_config` (not `stage1`) and `fold_challenge_shape_at_level` (not `fold_shape`) in signatures and docs.
- [ ] Internal type `Stage1Fn` renamed to `RingChallengeConfigFn`.
- [ ] `estimate_proof_bytes` and `schedule_from_entry` use the renamed closure
  parameters so generated-table readers do not keep the old `stage1` terminology.

#### `akita-schedules` crate

- [ ] New workspace crate `crates/akita-schedules` contains all generated table
  modules moved from `akita-planner/src/generated/`.
- [ ] Root `Cargo.toml` adds `crates/akita-schedules` to `workspace.members`.
- [ ] Each family is behind a Cargo feature (e.g. `fp128-d64-onehot`, `fp128-d64-full`).
- [ ] `akita-planner` default build contains **no** generated table `.rs` files.
- [ ] `gen_schedule_tables` writes into `akita-schedules/src/generated/`, emits catalog
  identities, and updates family feature wiring (not `akita-planner`). The binary
  itself stays in `akita-config` (only crate that can name presets).
- [ ] Table **types** (`GeneratedScheduleTable`, `GeneratedStep`, `GeneratedFoldStep`,
  `GeneratedCommitmentGroupScheduleKey`, `table_entry`, `expand`, `schedule_from_entry`) stay in
  `akita-planner`; `akita-schedules` is data-only and depends on `akita-planner` for
  them. No dependency cycle (`akita-schedules → akita-planner`; `akita-config →
  akita-schedules`).
- [ ] `akita-schedules` has `default = []`, one feature per family, `all-schedules`,
  and `zk`. Family features control modules; `zk` selects `_zk` data for every enabled
  ZK-capable family. Enabling `zk` alone does not enable any family. Non-ZK-only
  family features compile to no table accessor under `zk`, not to a missing-module
  error.
- [ ] `ALL_GENERATED_FAMILIES` and the drift guard `generated_schedule_tables_match_find_schedule`
  **stay in `akita-config`** (they name preset `Cfg` types and call
  `Cfg::runtime_schedule`, which `akita-schedules` cannot do). Each family row is
  gated behind its `schedules-*` feature; a meta-feature runs the full cross-product.
- [ ] `akita-planner` default build contains no `src/generated/fp*.rs` table data
  modules. `akita_planner::generated` may remain as the type/expansion namespace.

#### Opt-in presets

- [ ] `CommitmentConfig::schedule_catalog() -> Option<GeneratedScheduleTable>`
  with default `None`.
- [ ] `runtime_schedule` delegates to `resolve_schedule(..., Self::schedule_catalog())`.
- [ ] Each production preset that today uses a shipped table opts in via
  `#[cfg(feature = "schedules-…")]` returning `Some(akita_schedules::…::table())`.
- [ ] Preset feature flags documented in `akita-config/Cargo.toml`; default preset
  features preserve current dev/CI behavior for enabled families.
- [ ] `akita-config` has optional dependency
  `akita-schedules = { path = "../akita-schedules", optional = true, default-features = false }`.
- [ ] `akita-config` schedule features use the exact shape
  `schedules-fp128-d64-onehot = ["dep:akita-schedules", "akita-schedules/fp128-d64-onehot"]`
  (modulo family name).
- [ ] `akita-config/zk` forwards `akita-schedules?/zk` so ZK mode reaches the schedule
  crate only when the optional schedule dependency is enabled.
- [ ] `akita-config` exposes `schedules-default` and `all-schedules`. Defaults include
  `schedules-default` unless a caller opts out with `default-features = false`.
- [ ] `akita-pcs`, `akita-prover`, `akita-verifier`, and `akita-setup` each expose and
  default-enable a `schedules-default` forwarding feature because they currently use
  `akita-config` with `default-features = false` somewhere in the graph. `akita-pcs`
  forwards it explicitly through its own default feature list.
- [ ] `akita-setup` changes its `akita-config` and `akita-prover` dependency edges to
  `default-features = false`; it forwards `parallel`, `zk`, `disk-persistence`, and
  `schedules-default` explicitly. This is required so `akita-pcs --no-default-features
  --features parallel,profile-ci` cannot accidentally pull all default schedules
  through `akita-setup`.
- [ ] Minimal consumers can still build DP-only by setting `default-features = false`
  and not enabling any `schedules-*` feature.

#### Same-point keys only

- [ ] Schedule lookup for production prove/verify paths uses
  `CommitmentGroupScheduleKey::new_from_opening_batch` / `OpeningBatch::new`.
- [ ] `GeneratedFamily` replaces the current hardcoded `[1, 4]` enumeration with a
  per-family `num_polys: &'static [usize]` list. Akita defaults use `[1, 4]`; Jolt can
  emit `[1, 38]` without changing Akita core.
- [ ] Spec [`planner-incidence-generalization.md`](planner-incidence-generalization.md)
  header notes schedule-key portions are **superseded** by this spec (file may remain
  for historical witness-layout notes until archived).

#### CI profile isolation (Option B, minimal guard)

- [ ] The head profile binary in CI uses
  `cargo build --release --example profile --no-default-features --features parallel,profile-ci`.
  `--no-default-features` is required; otherwise default schedule bundles would link
  all production tables and defeat profile isolation.
- [ ] `profile-ci` on `akita-pcs` enables the union of `schedules-*` features for the
  presets exercised by the benchmark matrix (and nothing else).
- [ ] `AKITA_BENCH_CASES` in `.github/workflows/profile-bench.yml` stays the single
  list of CI bench cases (no new manifest SSOT, no exporter script).
- [ ] The PR merge-base profile binary has a transition probe: if the merge-base
  checkout defines `profile-ci`, build it with the same `--no-default-features
  --features parallel,profile-ci`; otherwise build it with the pre-refactor default
  command. This keeps the introducing PR benchmarkable against old main.
- [ ] A **hard CI gate** `scripts/check_profile_ci_features.sh` parses
  `AKITA_BENCH_CASES` from the workflow, maps each `mode` to its required
  `schedules-*` feature, and fails if `profile-ci` does not enable that feature.
  This is the only drift check; it does not reconcile three separate sources.
- [ ] The same script **fails** when a bench case uses a `num_polys` value outside the
  generated family key list, because profile CI is required to measure table-backed
  schedules, not DP fallback.
- [ ] Add a hard link-isolation smoke check for the CI profile binary, such as
  `scripts/check_profile_ci_linkage.sh`, that fails if an obvious non-`profile-ci`
  schedule family symbol is present. This can be conservative; it only needs to catch
  accidental all-table linkage.

#### Downstream path (documented, Jolt-ready)

- [ ] Spec documents Jolt pattern: `jolt-schedules` crate +
  `JoltD64OneHot::schedule_catalog()` without `akita-schedules` dependency.
- [ ] A reusable emit API in `akita-planner` (`EmitSpec` accepting `PlannerPolicy` +
  hook fn pointers, key list, module name, const name, family name, and output
  directory) exists so Jolt can generate catalogs outside `ALL_GENERATED_FAMILIES`.
  The `akita-config` CLI may keep adapting `ALL_GENERATED_FAMILIES`; the reusable
  library surface must not require Akita preset types or `akita-config`.
- [ ] The `jolt-schedules` data crate depends on `akita-planner` only. The separate
  Jolt runtime config crate that implements `CommitmentConfig` depends on
  `akita-config` and `jolt-schedules`, but not on `akita-schedules`.

#### Correctness

- [ ] `cargo test --workspace` and `cargo test --workspace --features zk` pass.
- [ ] Drift guard (kept in `akita-config`): every enabled family's table-hit
  expansion matches the DP (`generated_schedule_tables_match_find_schedule`).
- [ ] The drift guard first asserts that `Cfg::schedule_catalog()` is `Some` for every
  enabled family and that `table_entry` hits for every enumerated key. It must not
  pass vacuously by comparing DP fallback against DP.
- [ ] Run the full schedule cross-product guard in a dedicated job:
  `cargo test -p akita-config --features all-schedules generated_schedule_tables_match_find_schedule`
  and the ZK variant:
  `cargo test -p akita-config --features zk,all-schedules generated_schedule_tables_match_find_schedule`.
  The ZK variant checks every ZK-capable family and excludes non-ZK-only tiered rows.
- [ ] `runtime_fallback` tests pass. Note: `tests/runtime_fallback.rs` currently
  calls `akita_planner::shipped_table` directly (line ~47); migrate it to
  `Cfg::schedule_catalog()` + `table_entry` since `shipped_table` is removed.
- [ ] `crates/akita-config/src/proof_optimized/tests.rs` migrates off direct
  `akita_planner::generated::*_table` imports. Tests that require table data either
  use `Cfg::schedule_catalog().expect(...)` under the matching schedule feature or
  import `akita_schedules::*_table` behind the same feature.
- [ ] Add feature-off tests for one table-backed preset proving `schedule_catalog()`
  is `None` and `runtime_schedule` still equals `find_schedule` through DP.
- [ ] Profile benchmark CI cases produce identical timing-quality results (median
  setup/prove/verify within noise of pre-refactor; no functional regression).

### Testing Strategy

| Area | Tests |
|------|-------|
| Engine | Existing `runtime_fallback.rs` (migrated off `shipped_table`), planner unit tests, negative mismatched-catalog rejection test |
| Catalog identity | Unit tests for identity match/mismatch, ZK mismatch, root fold shape mismatch, and ring-challenge digest mismatch |
| Per-family drift | `generated_schedule_tables_match_find_schedule` in `akita-config`, gated per `schedules-*` feature; `all-schedules` meta-feature for full cross-product |
| Preset opt-in | `akita-config` tests: with feature on, catalog is `Some` and table hit; with feature off, catalog is `None` and runtime uses DP |
| Direct table tests | `proof_optimized/tests.rs` migrated to feature-gated `schedule_catalog()` or `akita_schedules::*_table()` access |
| profile-ci coverage | `scripts/check_profile_ci_features.sh` in CI (parses `AKITA_BENCH_CASES`) |
| profile-ci linkage | Hard smoke check that profile-ci binary does not contain an obvious non-profile family symbol |
| Profile compile | CI builds head with `--no-default-features --features parallel,profile-ci`; merge-base build probes feature availability |
| E2E | `akita-pcs` e2e, `batched_aggregated_e2e`, tiered/tensor cases with appropriate features |

Feature combinations: run drift guards under default and `zk` for families that ship
`zk` tables. The required schedule-specific jobs are:

```bash
cargo test -p akita-config --features all-schedules generated_schedule_tables_match_find_schedule
cargo test -p akita-config --features zk,all-schedules generated_schedule_tables_match_find_schedule
```

The second command intentionally means "all ZK-capable schedules"; it must not require
the non-ZK-only tiered family to expose a `_zk` table.

### Performance

- **No proof-size or transcript-byte change** for identical `(preset, key)` pairs
  when the same catalog row is used.
- **Compile time / link size:** `profile` CI binary MUST NOT link `akita-schedules`
  families outside the `profile-ci` union (verify via `cargo tree` and a hard
  symbol-smoke check).
- **Runtime:** Table hit path unchanged (same `schedule_from_entry` code, different
  crate path). DP fallback unchanged.
- **CI drift test duration:** May improve when the full cross-product guard runs in a
  dedicated `akita-config --features all-schedules` job rather than on every workspace
  test pass (default-feature passes check only the enabled families).

## Design

### Architecture

#### Crate graph (after)

Arrows point from dependent to dependency (`A → B` = A depends on B).

```text
akita-types, akita-challenges, akita-field
        ▲
        │
   akita-planner  (engine + table TYPES, no table DATA)
        ▲                         ▲
        │                         │
   akita-schedules           akita-config
   (data only,               CommitmentConfig + policy_of + preset features
    feature-gated            ALL_GENERATED_FAMILIES + drift guard (name presets)
    generated/*.rs)          gen_schedule_tables binary (writes akita-schedules/)
        ▲                         ▲
        └─────────┬───────────────┘
                  │  akita-config → akita-schedules (for *_table())
                  │
   akita-prover / akita-verifier / akita-pcs
        → akita-config only (no direct akita-schedules)

Downstream (Jolt):
   jolt-schedules → akita-planner (types only)
   jolt runtime config → akita-config + jolt-schedules
        (no akita-schedules dependency)
```

#### `resolve_schedule` (replaces `get_schedule`)

```rust
pub fn resolve_schedule(
    key: CommitmentGroupScheduleKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError>
```

Behavior:

1. If `catalog` is `Some`, validate the catalog identity against `policy`,
   `ring_challenge_config`, and `fold_challenge_shape_at_level`.
2. If identity validation fails, return `AkitaError::InvalidSetup`.
3. If identity validation succeeds, look up `generated_schedule_lookup_key(key)` in
   the table; on hit, `schedule_from_entry`.
4. Otherwise (no catalog, or miss), `find_schedule`.

`resolve_schedule` no longer computes `root_fold_is_tensor` at all — that hack only
existed so the global registry could disambiguate the tensor table from the flat one.
`fp128::D64OneHot` and `tensor_verifier::fp128::D64OneHotTensor` are different presets
that each return their own `schedule_catalog()`, so the discriminator is deleted.

#### Catalog identity

`GeneratedScheduleTable` becomes:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleCatalogIdentity {
    pub family_name: &'static str,
    pub zk_enabled: bool,
    pub sis_family: SisModulusFamily,
    pub ring_dimension: usize,
    pub decomposition: DecompositionParams,
    pub ring_subfield_norm_bound: u32,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub basis_range: (u32, u32),
    pub onehot_chunk_size: usize,
    pub tiered: bool,
    pub root_fold_shape: TensorChallengeShape,
    pub ring_dimensions: &'static [usize],
    pub ring_challenge_config_digest: u64,
    pub key_count: usize,
    pub key_digest: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub identity: GeneratedScheduleCatalogIdentity,
    pub entries: &'static [GeneratedScheduleTableEntry],
}
```

Validation derives the expected identity as follows:

- Policy fields copy directly from `PlannerPolicy`.
- `zk_enabled` is `cfg!(feature = "zk")` in `akita-planner`.
- `root_fold_shape` is evaluated with `AkitaScheduleInputs { num_vars: key.num_vars,
  level: 0, current_w_len: 1usize.checked_shl(key.num_vars as u32)? }`.
- `ring_dimensions` are the distinct ring degrees represented by the compact table,
  including fold steps and root-direct commit payloads. The emitter stores them sorted
  and deduplicated.
- `ring_challenge_config_digest` is a deterministic non-cryptographic digest over
  `(d, ring_challenge_config(d)?)` for those ring dimensions. The digest is only a
  wiring guard, not a security primitive.
- `key_count` and `key_digest` cover the sorted generated keys, including `num_vars`,
  `num_t_vectors`, `num_w_vectors`, `num_z_vectors`, and generated
  `num_commitment_groups`. This prevents a same-policy `[1, 4]` catalog from looking
  identical to a Jolt `[1, 38]` catalog.

Digest inputs use a fixed little-endian byte format:

- `usize` / `u32` fields are encoded as `u64::to_le_bytes` after checked conversion.
- `bool` is one byte (`0` or `1`).
- `SparseChallengeConfig` is encoded by variant tag plus canonical fields:
  uniform configs encode weight and ordered `nonzero_coeffs` bytes; bounded-L1 /
  exact-shell configs encode their public scalar parameters in declaration order.
  Adding a new challenge variant must update this encoder.

The emitter rejects a table whose key envelope would produce mixed root fold shapes.
All current Akita families are constant across the envelope (flat, tensor, or tiered
flat), and a future dynamic-shape family should be split into separate catalogs.

If `checked_shl` or `ring_challenge_config(d)` fails while validating a `Some`
catalog, return the same `AkitaError` shape the DP would return for invalid setup.
Do not default to flat root shape on overflow.

Catalog identity does not replace the drift guard. It catches wrong-family and
wrong-feature wiring at runtime; the drift guard proves that every enabled table row
is still the DP optimum for the exact hooks on this branch.

#### `CommitmentConfig` hook

```rust
/// Application-owned precomputed schedules. Default: none (DP-only).
fn schedule_catalog() -> Option<GeneratedScheduleTable> {
    None
}

fn runtime_schedule(key: CommitmentGroupScheduleKey) -> Result<Schedule, AkitaError> {
    akita_planner::resolve_schedule(
        key,
        &policy_of::<Self>(),
        Self::ring_challenge_config,
        Self::fold_challenge_shape_at_level,
        Self::schedule_catalog(),
    )
}
```

Presets that ship tables enable a feature (the `#[cfg]`-gated override returns the
catalog; with the feature off it falls back to the `None` default → DP):

```rust
impl CommitmentConfig for D64OneHot {
    #[cfg(feature = "schedules-fp128-d64-onehot")]
    fn schedule_catalog() -> Option<GeneratedScheduleTable> {
        Some(akita_schedules::fp128_d64_onehot_table())
    }
}
```

`akita-config` features:

- `schedules-fp128-d64-onehot` →
  `["dep:akita-schedules", "akita-schedules/fp128-d64-onehot"]`
- `zk` → `["akita-planner/zk", "akita-types/zk", "akita-schedules?/zk"]`
- `schedules-default` → current production table set
- `all-schedules` → union of every `schedules-*` family, for drift CI

Default `akita-config` features include `schedules-default`. Since `akita-pcs`,
`akita-prover`, and `akita-verifier` currently depend on `akita-config` with
`default-features = false`, they must forward an equivalent `schedules-default`
feature from their own default feature lists. **Jolt omits all `schedules-*` features**
by depending with `default-features = false`.

#### Naming: `stage1` → `ring_challenge_config`

The planner param is renamed to **exactly** mirror the trait hook
(`CommitmentConfig::ring_challenge_config`), not a near-synonym, so a reader can
trace the same name end to end.

| Location | Today | After |
|----------|-------|-------|
| `CommitmentConfig` | `ring_challenge_config` | unchanged (already correct) |
| Planner fn params | `stage1` | `ring_challenge_config` |
| Planner type alias | `Stage1Fn` | `RingChallengeConfigFn` |
| Planner docs | "stage-1 sparse-challenge closure" | "ring challenge config closure (`SparseChallengeConfig` per ring degree `d`)" |

Meaning (from `CommitmentConfig` docs): the sparse ring element `c(X)` used in the
**weak-binding fold** before sumcheck stage 1. It is **not** a sumcheck round
challenge. The planner passes this closure to price folded-witness norms and ω-based
collision bounds during DP and entry expansion.

`fold_challenge_shape_at_level` (flat vs tensor root fold) stays aligned with the
existing `CommitmentConfig` hook name.

### Schedule lookup keys (same-point only)

Production folded path uses `OpeningBatch::new(padded_num_vars, num_polys)`:

```text
num_t_vectors = num_polys        (polynomials in the bundled commitment)
num_w_vectors = num_claims       (= num_polys for same-point: one claim per poly)
num_z_vectors = 1                (one commitment group)
num_commitment_groups = 1        (in generated key shape)
```

`CommitmentGroupScheduleKey::new_from_opening_batch` is the canonical projection.

**Generated table enumeration** (`family_keys` in emitter) crosses:

- `num_vars` in `[min_num_vars, max_num_vars]`
- `num_polys` in per-family list (e.g. `[1, 4]` default; Jolt may ship `[1, 38]`)

The current `family_keys` helper hardcodes `[1, 4]`. This refactor moves that list
onto `GeneratedFamily` so Akita and downstream catalogs can enumerate different batch
widths without changing planner code.

This replaces the stale incidence generalization plan for scalar same-bundle
batching. Multi-commitment same-point batching is tracked separately in
[`multi-group-batching.md`](multi-group-batching.md); until its grouped key and
descriptor shape land, scalar lookup must not collapse grouped inputs through
`new_from_opening_batch`.

### `akita-schedules` crate

**Location:** `crates/akita-schedules/`

**Contents:**

- `src/generated/*.rs` (moved from `akita-planner`), referencing the table types via
  `akita_planner::generated::…`
- `src/lib.rs` exposes `*_table()` constructors per family. Each constructor returns
  `GeneratedScheduleTable { identity, entries }`.
- Table **types stay in `akita-planner`** (`GeneratedScheduleTable`, `GeneratedStep`,
  `table_entry`, `expand`, `schedule_from_entry`); `akita-schedules` is data-only and
  depends on `akita-planner` for them

**Features (one per family, examples):**

| Feature | Module | Typical preset |
|---------|--------|----------------|
| `fp128-d64-onehot` | `fp128_d64_onehot.rs` | `fp128::D64OneHot` |
| `fp128-d64-full` | `fp128_d64_full.rs` | `fp128::D64Full` |
| `fp128-d32-onehot` | (new/emitted) | `fp128::D32OneHot` |
| `fp32-d128-onehot` | `fp32_d128_onehot.rs` | `fp32::D128OneHot` |
| … | … | … |
| `fp128-d64-onehot-tiered` | `fp128_d64_onehot_tiered.rs` | `fp128::D64OneHotTiered` |
| `fp128-d64-onehot-tensor` | `fp128_d64_onehot_tensor.rs` | `D64OneHotTensor` |

Meta-features:

- `all-schedules` — union of every feature-gated family (drift CI job, not default).
  Under `zk`, non-ZK-only features such as tiered are enabled but intentionally inert.
- `zk` — switches enabled family accessors from non-ZK generated modules to `_zk`
  generated modules when those modules exist. It does not enable any family by itself.

`akita-schedules/Cargo.toml` shape:

```toml
[features]
default = []
zk = ["akita-planner/zk"]
all-schedules = [
    "fp128-d64-onehot",
    "fp128-d64-full",
    "fp128-d64-onehot-tiered", # non-ZK-only; inert when feature = "zk"
    # every other family
]
fp128-d64-onehot = []
fp128-d64-full = []
```

`akita-config/Cargo.toml` shape:

```toml
[features]
default = ["schedules-default"]
zk = ["akita-planner/zk", "akita-types/zk", "akita-schedules?/zk"]
schedules-default = [
    "schedules-fp128-d64-onehot",
    "schedules-fp128-d64-full",
    "schedules-fp128-d128-onehot",
    "schedules-fp128-d128-full",
    "schedules-fp32-d128-onehot",
    "schedules-fp32-d256-onehot",
    "schedules-fp64-d128-onehot",
    "schedules-fp64-d128-full",
    "schedules-fp64-d256-onehot",
]
all-schedules = [
    "schedules-default",
    "schedules-fp128-d64-onehot-tensor",
    "schedules-fp128-d64-onehot-tiered", # non-ZK-only; gated out under feature = "zk"
]
schedules-fp128-d64-onehot = [
    "dep:akita-schedules",
    "akita-schedules/fp128-d64-onehot",
]

[dependencies]
akita-schedules = {
    version = "0.1.0",
    path = "../akita-schedules",
    optional = true,
    default-features = false,
}
```

`akita-pcs` keeps a smaller `profile-ci` feature:

```toml
[features]
default = ["parallel", "schedules-default"]
schedules-default = ["akita-config/schedules-default"]
profile-ci = [
    "akita-config/schedules-fp32-d128-onehot",
    "akita-config/schedules-fp64-d128-onehot",
    "akita-config/schedules-fp128-d64-onehot",
    "akita-config/schedules-fp128-d64-full",
]
```

`akita-prover`, `akita-verifier`, and `akita-setup` expose the same
`schedules-default = ["akita-config/schedules-default"]` forwarding feature and add
it to their defaults unless there is a concrete crate-specific reason not to. This is
what preserves current table-backed behavior for standalone builds of those packages.
`akita-setup` must also set `default-features = false` on its `akita-config` and
`akita-prover` dependency edges and forward those crates' features explicitly; otherwise
`akita-pcs --no-default-features` can still pull default schedule bundles through
`akita-setup`.

**`ALL_GENERATED_FAMILIES`** **stays in `akita-config`.** It is a list of
`regen::<Cfg>` / `table_backed::<Cfg>` function pointers that name preset `Cfg` types
and call `Cfg::runtime_schedule` — only `akita-config` can name presets, and
`akita-schedules` (data-only, below `akita-config`) cannot. The emitter and the drift
guard both live in `akita-config` and share this list. `akita-schedules` holds **only**
the static table data + `*_table()` constructors; it does not enumerate presets.

After this refactor, `GeneratedFamily` also records the schedule feature name, the
`family_name` string used in the catalog identity, and the per-family `num_polys`
list. Rows are gated with `#[cfg(feature = "schedules-...")]`. Drift tests that are
meant to exercise catalogs must assert that the enabled family list is non-empty, so
they cannot pass by silently checking an empty set.

### Emitter (`gen_schedule_tables`)

- Reusable emitter library code lives in `akita-planner`, because it needs only
  `PlannerPolicy`, generated table types, explicit keys, and hook function pointers.
  It must not depend on `CommitmentConfig`, `policy_of`, or `ALL_GENERATED_FAMILIES`.
- Binary **stays in `akita-config`** (only crate that names preset `Cfg` types). It
  adapts `ALL_GENERATED_FAMILIES` into `EmitSpec` values and calls the planner emitter.
- Output directory: `crates/akita-schedules/src/generated/`.
- Updates `akita-schedules` `lib.rs` feature wiring (not `resolve.rs` imports).
- Emits each table's `GeneratedScheduleCatalogIdentity`, including sorted distinct
  ring dimensions and ring-challenge digest.
- Runs `cargo fmt -p akita-planner -p akita-schedules -p akita-config` after regen.

**Jolt-facing emit API** (part of the engine cutover):

```rust
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub family_name: &'static str,
    pub policy: PlannerPolicy,
    pub keys: Vec<CommitmentGroupScheduleKey>,
    pub output_dir: PathBuf,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
    pub zk_enabled: bool,
}
```

The CLI path adapts `ALL_GENERATED_FAMILIES` into `EmitSpec` values. `EmitSpec` takes
concrete keys as the source of truth; helper constructors such as `ScheduleKeyEnvelope`
may build those keys from `num_vars: 18..=32` and `num_polys: [1, 38]`, but the emitter
itself consumes only the concrete key list. Jolt runs the same planner emitter library
with an explicit `PlannerPolicy` and hook functions, writing `jolt-schedules/`.

### CI profile isolation (minimal guard)

#### Problem

The benchmark cases live in `AKITA_BENCH_CASES` in
`.github/workflows/profile-bench.yml`, and the schedule features live in
`Cargo.toml`. An agent can add `onehot_fp128_d32:28:1` to the matrix without enabling
`schedules-fp128-d32-onehot`, so the bench silently measures the DP fallback instead
of the shipped table it is supposed to profile.

This is the *only* drift that matters for the refactor. Note what is **not** a
problem: link stripping already falls out of per-preset feature gating. The `profile`
example names preset *types* (`type Cfg = fp128::D64Full`), which are always
available; only the table *data* is feature-gated. So
`--no-default-features --features parallel,profile-ci` linking the schedule families
for the bench matrix needs no manifest — it is just a feature list. We do not need a
new source of truth, a TOML file, an exporter script, or a three-way reconciler.

#### Solution: one feature + one coverage guard

**`akita-pcs/Cargo.toml`** — `profile-ci` enables the `schedules-*` features for the
presets the benchmark matrix exercises. It is intentionally separate from
`schedules-default`:

```toml
[features]
profile-ci = [
    "akita-config/schedules-fp32-d128-onehot",
    "akita-config/schedules-fp64-d128-onehot",
    "akita-config/schedules-fp128-d64-onehot",
    "akita-config/schedules-fp128-d64-full",
]
```

**Workflow** keeps `AKITA_BENCH_CASES` as-is. The head build disables defaults so the
profile-ci union is the only linked schedule set:

```yaml
- name: Build profile binary
  run: cargo build --release --quiet --example profile --no-default-features --features parallel,profile-ci
```

The PR merge-base build has a compatibility probe because the introducing PR compares
against a checkout that does not yet define `profile-ci`:

```bash
if cargo metadata --no-deps --format-version 1 \
    | python3 scripts/cargo_feature_exists.py akita-pcs profile-ci; then
  cargo build --release --quiet --example profile \
    --no-default-features --features parallel,profile-ci
else
  cargo build --release --quiet --example profile
fi
```

The helper can be a tiny Python script or an inline Python snippet. It must inspect the
merge-base checkout, not the PR head.

**`scripts/check_profile_ci_features.sh`** (hard CI gate) is the single drift check:

1. Parse `AKITA_BENCH_CASES` from `profile-bench.yml` → the set of bench `mode`s.
2. Map each `mode` to its required `schedules-*` feature via a small literal table
   in the script (e.g. `onehot_fp128_d64 → schedules-fp128-d64-onehot`).
3. Parse the `profile-ci` feature list from `akita-pcs/Cargo.toml`.
4. **Fail** if any benched mode's required feature is not enabled by `profile-ci`.
5. Warn or fail if a bench case's `num_polys` is not generated for that family
   (currently the Akita tables emit `[1, 4]`; Jolt-owned catalogs may emit `[1, 38]`).

The check is one-directional (every benched mode must be covered); it does not force
exact set equality, so `profile-ci` may carry a small extra family without failing.
The mode→feature table is the one literal an agent edits when adding a bench case;
AGENTS.md documents this in one line. The same check fails if a bench case uses a
`num_polys` outside the family's emitted list, because that would hit DP rather than
the table.

Modes do **not** need `#[cfg]` gates in `modes.rs`: gating modes only saves compile
time of unused mode code (a weaker goal) and is what forced the rejected three-way
reconciler. Keeping all modes always-compiled keeps the guard to a single direction.

#### Linkage smoke check

Add `scripts/check_profile_ci_linkage.sh` after the head profile build:

1. Build with `--no-default-features --features parallel,profile-ci`.
2. Run `nm` (or `llvm-nm` if available) on `target/release/examples/profile`.
3. Fail if a known non-profile family symbol is present, for example a D128 full or
   D256 one-hot table that is not in `profile-ci`.

This check is intentionally conservative. It only needs to catch accidental all-table
linkage from default features or a stray dependency edge.

#### Why Option B (one binary, feature union)

| Approach | Pros | Cons |
|----------|------|------|
| **B: `profile-ci` union** | One compile per CI job; matches current interleaved PR/base workflow | Binary links union of ~4 families, not one |
| A: per-case compile | Strictest isolation | 5× compile cost; complicates interleaved bench |

Option B meets the practical goal: **do not link all 17+ families and ~1.8 MiB of
tables** when CI only exercises four presets. Union of four families is acceptable.

Optional follow-up: nightly job builds per-mode with single-family features to
measure binary footprint.

### `profile` example changes

- `modes.rs` is largely **unchanged**: all `ProfileMode` entries stay compiled (they
  name preset types, which are always available). No `#[cfg]` gating of modes.
- A mode whose `schedules-*` feature is off still runs — it resolves via the DP
  fallback. The `profile-ci` feature ensures the benched modes hit their tables.
- Local dev single-family isolation:
  `cargo run --release --example profile --no-default-features --features parallel,akita-config/schedules-fp128-d64-onehot`.
  Without a schedule feature, the same mode runs DP-backed.

### Downstream: Jolt

1. Define `JoltD64OneHot: CommitmentConfig` in a Jolt runtime config crate. This crate
   depends on `akita-config` because it implements the trait and calls
   `schedule_catalog()`.
2. Crate `jolt-schedules` is data-only and depends on `akita-planner` for generated
   table types. It does not depend on `akita-config` or `akita-schedules`.
3. Generate `jolt_fp128_d64_onehot.rs` with explicit keys:
   `num_vars` envelope × `num_polys` e.g. `[1, 38]`).
4. Jolt invokes the planner emitter library through `EmitSpec`, with an explicit
   `PlannerPolicy` and hooks, not `akita_config::generated_families::ALL_GENERATED_FAMILIES`.
5. `JoltD64OneHot::schedule_catalog()` returns
   `Some(jolt_schedules::…_table())`; **no** `akita-schedules` dependency.
6. Jolt CI drift test: `assert_catalog_matches_dp` on Jolt keys only, including an
   assertion that each generated Jolt key hits the Jolt catalog.
7. LayerZero PR #198 style changes (wide batch rows in Akita core) are **not**
   required for Jolt integration.

### Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Keep `shipped_table()` global registry | Cannot isolate link-time catalogs; forces downstream upstreaming |
| Trust per-preset catalog wiring without runtime identity | Too easy to silently attach a one-hot/tensor/tiered catalog to the wrong preset; verifier-reachable code should reject this as invalid setup |
| Move all tables out of Akita repo | User requirement: tables stay on `main`, grow as needed (e.g. D32) |
| Runtime plugin / dynamic loading | Rust static tables; unnecessary complexity |
| Per-case CI binaries (Option A) | Compile cost; user chose Option B |
| Revive incidence-based keys | Multipoint removed; spec stale; same-point suffices for Jolt mega-poly |
| `schedule_table()` on trait without features | Cargo features needed for link-time stripping |
| `ALL_GENERATED_FAMILIES` / drift guard in `akita-schedules` | Impossible: they name preset `Cfg` types and call `Cfg::runtime_schedule`; only `akita-config` can. Stay there |
| TOML manifest SSOT + exporter + three-way reconciler | Over-built; link stripping falls out of feature gating, so a one-directional coverage check over the existing `AKITA_BENCH_CASES` suffices (also resolves the Bugbot inconsistency between the three rule sets) |
| `&'static GeneratedScheduleTable` return | `GeneratedScheduleTable` is `Copy`, built by value across the zk cfg branch; no static to borrow. Return by value |

## Documentation

- Update [`AGENTS.md`](../AGENTS.md): crate graph (`akita-schedules`), opt-in
  catalogs, one-line "edit the mode→feature table in `check_profile_ci_features.sh`
  when adding a bench case" note, the `--no-default-features --features
  parallel,profile-ci` benchmark build, and the `stage1 → ring_challenge_config`
  rename.
- Fold into [`book/src/how/configuration.md`](../book/src/how/configuration.md) when
  implemented.
- Update [`docs/doc-blast-radius.json`](../docs/doc-blast-radius.json): add
  `akita-schedules`, `scripts/check_profile_ci_features.sh`,
  `scripts/check_profile_ci_linkage.sh`, and the merge-base feature-probe helper if it
  is added as a standalone script.
- Update CI docs/comments near `.github/workflows/profile-bench.yml`: the merge-base
  profile build intentionally probes for `profile-ci` because the introducing PR must
  benchmark against pre-refactor main.
- Add note to [`planner-incidence-generalization.md`](planner-incidence-generalization.md)
  `Status` / header: schedule-key portions superseded.
- [`crates/akita-planner/README.md`](../crates/akita-planner/README.md): engine vs
  schedules split.

## Execution

### Phase 1 — Engine cutover (no behavioral change)

1. Add workspace member `crates/akita-schedules`; create an empty data-only crate that
   depends on `akita-planner`.
2. Change `akita-setup` dependency edges to `default-features = false` for
   `akita-config` and `akita-prover`; forward `parallel`, `zk`, `disk-persistence`, and
   `schedules-default` explicitly before adding new default schedule features.
3. Move generated table **types and expansion only** to the stable planner namespace;
   keep table data temporarily where it is until the new crate compiles.
4. Add `GeneratedScheduleCatalogIdentity`, canonical digest encoders, and runtime
   identity validation tests for wrong family, wrong root fold shape, wrong ZK mode,
   wrong ring-challenge digest, and wrong key digest.
5. Add `resolve_schedule` with `catalog: Option<GeneratedScheduleTable>` parameter
   (by value); rename planner `stage1`/`Stage1Fn` to `ring_challenge_config`/
   `RingChallengeConfigFn`.
6. Add `CommitmentConfig::schedule_catalog()` default `None`; wire `runtime_schedule`
   to `resolve_schedule`.
7. Extract the planner-level reusable emitter API (`EmitSpec`) and make
   `gen_schedule_tables` in `akita-config` call it via `ALL_GENERATED_FAMILIES`.
8. Move generated data modules into `akita-schedules`; feature-gate families, including
   inert-under-ZK handling for non-ZK-only tiered.
9. Retarget generated output to `akita-schedules/src/generated/` and regenerate.
10. Point presets at catalogs via `#[cfg]`-gated `schedule_catalog()` overrides
    (default features = current behavior through forwarded `schedules-default`).
11. Delete `shipped_table`, `get_schedule`, the planner generated table-data imports,
    and the `root_fold_is_tensor` hack in `resolve.rs`.
12. Keep `ALL_GENERATED_FAMILIES` + drift guard in `akita-config`; gate each family row
    behind its `schedules-*` feature; add `all-schedules` meta-feature. Migrate
    `tests/runtime_fallback.rs` and `proof_optimized/tests.rs` off direct planner table
    constructors.
13. Add `schedules-default` forwarding features to `akita-pcs`, `akita-prover`,
    `akita-verifier`, and `akita-setup`.

### Phase 2 — `profile-ci` and the coverage guard

1. Add `profile-ci` feature on `akita-pcs` enabling the bench matrix's `schedules-*`.
2. Add `scripts/check_profile_ci_features.sh` (parses `AKITA_BENCH_CASES`).
3. Add the conservative linkage smoke check for the profile-ci binary.
4. Update `profile-bench.yml` head build to use `--no-default-features --features
   parallel,profile-ci`; add the merge-base compatibility probe.
5. Wire both guards into CI (doc-guardrails or the profile workflow).

### Phase 3 — D32 expansion

1. Emit D32 families if missing (user: "maybe even more").
2. Add their schedule features and include them in `all-schedules`; include them in
   `schedules-default` only if they are intended to preserve current default behavior.

### Phase 4 — Jolt (separate repo PR)

1. `JoltD64OneHot` + `jolt-schedules` as documented.

### Risks

| Risk | Mitigation |
|------|------------|
| Feature matrix explosion | One feature per family; meta `all-schedules` for full drift only |
| Bench case added without its schedule feature | `check_profile_ci_features.sh` hard gate parses `AKITA_BENCH_CASES`; one-line AGENTS.md note |
| Production preset silently degraded to DP by a missing feature | Forward `schedules-default` through top-level crates; drift guard asserts `schedule_catalog().is_some()` and table hits for enabled rows |
| Wrong catalog attached to a preset | Runtime catalog identity validation rejects the mismatch before lookup |
| `zk` / non-`zk` table split | `akita-config/zk` forwards `akita-schedules?/zk`; identity includes `zk_enabled`; non-ZK-only tiered is inert under `zk` |
| Tensor/tiered presets | Separate families (already separate tables); each names its own catalog, so the `root_fold_is_tensor` runtime hack is deleted |
| Profile-ci accidentally links defaults | Build with `--no-default-features`, set `akita-setup` dependency defaults to false, and run linkage smoke check |

## References

- [`specs/single-point-opening-batch.md`](single-point-opening-batch.md) — batching contract
- [`specs/archive/2026-Q2/planner-owns-schedule-expansion.md`](archive/2026-Q2/planner-owns-schedule-expansion.md) — planner owns expansion (landed)
- [`specs/archive/2026-Q2/planner-config-consolidation.md`](archive/2026-Q2/planner-config-consolidation.md) — prior `schedule_table()` opt-in sketch
- [`specs/planner-incidence-generalization.md`](planner-incidence-generalization.md) — **stale** for schedule keys; superseded in part by this spec
- LayerZero PR #198 — motivation for downstream-owned wide-batch tables
- [`crates/akita-config/src/lib.rs`](../crates/akita-config/src/lib.rs) — `ring_challenge_config` docs
- [`crates/akita-planner/src/resolve.rs`](../crates/akita-planner/src/resolve.rs) — current global registry (to remove)
- [`.github/workflows/profile-bench.yml`](../.github/workflows/profile-bench.yml) — CI benchmark workflow
- [`crates/akita-pcs/examples/profile/modes.rs`](../crates/akita-pcs/examples/profile/modes.rs) — profile mode registry
