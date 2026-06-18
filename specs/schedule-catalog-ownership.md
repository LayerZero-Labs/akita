# Spec: Schedule catalog ownership and opt-in shipped tables

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-18 |
| Status        | proposed |
| PR            | |
| Supersedes    | (partial) schedule-key scope in [`planner-incidence-generalization.md`](planner-incidence-generalization.md) |
| Superseded-by | |
| Book-chapter  | `book/src/how/configuration.md` |

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
`schedule_catalog()`. The planner keeps DP fallback as the universal correctness
path. Downstream repos ship their own catalogs (Jolt: D64 one-hot + Jolt-specific
batch widths) without depending on `akita-schedules`.

CI profile benchmarks adopt a single Cargo feature `profile-ci` that links only the
schedule families required by the active benchmark matrix. A **canonical manifest**
is the single source of truth for benchmark cases and their schedule subscriptions;
a hard CI guard prevents drift when the matrix changes.

## Intent

### Goal

Refactor schedule resolution so:

1. **`akita-planner` is engine-only** (DP + compact entry expansion + generic table
   lookup). No global preset→table registry.
2. **`akita-schedules` holds all Akita-shipped tables** on `main`, feature-gated per
   family, optional for consumers.
3. **Each `CommitmentConfig` preset opt-ins** to zero or one catalog via
   `schedule_catalog() -> Option<&'static GeneratedScheduleTable>` (default `None`
   → DP only).
4. **Downstream projects** (Jolt) depend on the engine + their own generated catalog
   crate, not on `akita-schedules`.
5. **CI `profile` builds with `--features profile-ci`**, linking only schedules for
   modes in the CI benchmark matrix, enforced by manifest + drift guard.
6. **Rename planner misnomers** (`stage1` closure parameters) to match
   `CommitmentConfig` vocabulary.

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
                    │      ring_fold_challenge_config,     │
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
to that table. Akita benches ship `akita-schedules` and enable only the families listed
in `docs/ci-profile-manifest.toml` via `profile-ci`.

### Invariants

- **DP is the source of truth.** Any shipped catalog row MUST match
  `find_schedule(key, policy, …)` for the same policy and hooks. Drift guards enforce
  this per catalog (today: `generated_schedule_tables_match_find_schedule`; becomes
  per-family tests in `akita-schedules` and downstream catalogs).
- **Verifier no-panic.** `resolve_schedule` and `find_schedule` return `Result`,
  never panic on malformed keys (existing contract).
- **Determinism.** Prover and verifier both call `Cfg::runtime_schedule` with the
  same `policy_of::<Cfg>()` and hook fns; transcript `PlanSection` digest unchanged
  for identical inputs.
- **Default is DP-only.** `schedule_catalog()` default `None` must not change
  schedules for presets that do not opt in (modulo explicit preset feature enables).
- **Same-point batching only.** Lookup keys derive from `OpeningBatch` /
  `OpeningBatch::same_point(num_vars, num_polys)` (and `new_from_opening_batch`).
  No multipoint keys, no `ClaimIncidenceSummary` schedule path (type not in tree).
- **Table miss falls back to DP**, never errors solely because a row is absent (unless
  DP itself rejects the key).
- **CI manifest consistency.** Benchmark cases in CI, `profile-ci` feature union, and
  `profile` mode `cfg` gates MUST stay aligned (hard CI check).

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

- [ ] `akita_planner::shipped_table` and `get_schedule` are removed.
- [ ] `akita_planner::resolve_schedule(key, policy, ring_fold_challenge_config, fold_challenge_shape_at_level, catalog: Option<&GeneratedScheduleTable>)` is the single runtime entry point.
- [ ] Planner public closures use `ring_fold_challenge_config` (not `stage1`) and `fold_challenge_shape_at_level` (not `fold_shape`) in signatures and docs.
- [ ] Internal type `Stage1Fn` renamed (e.g. `RingFoldChallengeConfigFn`).

#### `akita-schedules` crate

- [ ] New workspace crate `crates/akita-schedules` contains all generated table
  modules moved from `akita-planner/src/generated/`.
- [ ] Each family is behind a Cargo feature (e.g. `fp128-d64-onehot`, `fp128-d64-full`).
- [ ] `akita-planner` default build contains **no** generated table `.rs` files.
- [ ] `gen_schedule_tables` writes into `akita-schedules/src/generated/` and updates
  family feature wiring (not `akita-planner`).

#### Opt-in presets

- [ ] `CommitmentConfig::schedule_catalog() -> Option<&'static GeneratedScheduleTable>`
  with default `None`.
- [ ] `runtime_schedule` delegates to `resolve_schedule(..., Self::schedule_catalog())`.
- [ ] Each production preset that today uses a shipped table opts in via
  `#[cfg(feature = "schedules-…")]` returning `Some(akita_schedules::…::table())`.
- [ ] Preset feature flags documented in `akita-config/Cargo.toml`; default preset
  features preserve current dev/CI behavior for enabled families.

#### Same-point keys only

- [ ] Schedule lookup for production prove/verify paths uses
  `AkitaScheduleLookupKey::new_from_opening_batch` / `OpeningBatch::same_point`.
- [ ] Spec [`planner-incidence-generalization.md`](planner-incidence-generalization.md)
  header notes schedule-key portions are **superseded** by this spec (file may remain
  for historical witness-layout notes until archived).

#### CI profile isolation (Option B)

- [ ] `docs/ci-profile-manifest.toml` is the **only** authoritative list of CI
  benchmark cases and their schedule/mode feature mapping.
- [ ] `.github/workflows/profile-bench.yml` loads cases from the manifest (script
  export), not a hand-maintained duplicate block.
- [ ] `cargo build --release --example profile --features profile-ci` is what CI uses.
- [ ] `profile-ci` on `akita-pcs` enables exactly the union of `schedule_family` and
  `profile_mode` features declared in the manifest (no more, no less).
- [ ] `scripts/check_ci_profile_manifest.sh` (or Python equivalent) is a **hard CI
  gate** (Documentation guardrails or dedicated workflow step).

#### Downstream path (documented, Jolt follow-up)

- [ ] Spec documents Jolt pattern: `jolt-schedules` crate +
  `JoltD64OneHot::schedule_catalog()` without `akita-schedules` dependency.
- [ ] `akita-schedule-emit` (or relocated emitter) accepts `PlannerPolicy` + hook
  fn pointers so Jolt can generate catalogs outside `ALL_GENERATED_FAMILIES`.

#### Correctness

- [ ] `cargo test --workspace` and `cargo test --workspace --features zk` pass.
- [ ] Relocated drift tests: every enabled `akita-schedules` family matches DP.
- [ ] `runtime_fallback` tests still pass (table miss → DP parity).
- [ ] Profile benchmark CI cases produce identical timing-quality results (median
  setup/prove/verify within noise of pre-refactor; no functional regression).

### Testing Strategy

| Area | Tests |
|------|-------|
| Engine | Existing `runtime_fallback.rs`, planner unit tests (relocated) |
| Per-family drift | `akita-schedules` crate tests parameterized by `ALL_SHIPPED_FAMILIES` |
| Preset opt-in | `akita-config` tests: with feature on, table hit; with feature off, DP path |
| Manifest drift | `scripts/check_ci_profile_manifest.sh` in CI |
| Profile compile | CI builds `profile` with only `profile-ci` (no `--all-features`) |
| E2E | `akita-pcs` e2e, `batched_aggregated_e2e`, tiered/tensor cases with appropriate features |

Feature combinations: run drift guards under default and `zk` for families that ship
`zk` tables (separate modules or `*_zk` features).

### Performance

- **No proof-size or transcript-byte change** for identical `(preset, key)` pairs
  when the same catalog row is used.
- **Compile time / link size:** `profile` CI binary MUST NOT depend on
  `akita-schedules` families outside the manifest union (verify via `cargo tree` /
  optional `nm` symbol check documented in manifest checker).
- **Runtime:** Table hit path unchanged (same `schedule_from_entry` code, different
  crate path). DP fallback unchanged.
- **CI drift test duration:** May improve when full cross-product guard is scoped to
  `akita-schedules/all-families` job rather than every workspace test pass.

## Design

### Architecture

#### Crate graph (after)

```text
akita-types ─────────────────────────────────────────┐
akita-challenges ───────────────────────────────────┤
akita-field ────────────────────────────────────────┤
                                                    ▼
                                            akita-planner
                                         (engine, no tables)
                                                    ▲
akita-schedules (optional) ─────────────────────────┤
  feature-gated generated/*.rs                      │
                                                    │
akita-config ───────────────────────────────────────┘
  CommitmentConfig + policy_of + preset features
  gen_schedule_tables binary (writes akita-schedules)

akita-prover / akita-verifier / akita-pcs
  → akita-config only (no direct akita-schedules)

Downstream (Jolt):
  jolt-schedules → akita-planner + akita-config (Jolt preset only)
  (no akita-schedules)
```

#### `resolve_schedule` (replaces `get_schedule`)

```rust
pub fn resolve_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_fold_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<&GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError>
```

Behavior:

1. If `catalog` is `Some`, look up `generated_schedule_lookup_key(key)` in the
   table; on hit, `schedule_from_entry`.
2. Otherwise (no catalog, or miss), `find_schedule`.

`root_fold_is_tensor` disambiguation for tensor vs flat D64 one-hot is **not** a
global registry concern: `fp128::D64OneHot` and `tensor_verifier::fp128::D64OneHotTensor`
are different presets with different `schedule_catalog()` tables.

#### `CommitmentConfig` hook

```rust
/// Application-owned precomputed schedules. Default: none (DP-only).
fn schedule_catalog() -> Option<&'static GeneratedScheduleTable> {
    None
}

fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
    akita_planner::resolve_schedule(
        key,
        &policy_of::<Self>(),
        Self::ring_fold_challenge_config,
        Self::fold_challenge_shape_at_level,
        Self::schedule_catalog(),
    )
}
```

Presets that ship tables enable a feature:

```rust
impl CommitmentConfig for D64OneHot {
    #[cfg(feature = "schedules-fp128-d64-onehot")]
    fn schedule_catalog() -> Option<&'static GeneratedScheduleTable> {
        Some(akita_schedules::fp128_d64_onehot::table())
    }
}
```

`akita-config` features:

- `schedules-fp128-d64-onehot` → `akita-schedules/fp128-d64-onehot`
- Preset bundles (e.g. `fp128-d64-onehot-preset`) enable schedule + preset impl

Default `akita-config` features for dev can include common schedules; **Jolt omits
all `schedules-*` features**.

#### Naming: `stage1` → `ring_fold_challenge_config`

| Location | Today | After |
|----------|-------|-------|
| `CommitmentConfig` | `ring_challenge_config` | unchanged (already correct) |
| Planner fn params | `stage1` | `ring_fold_challenge_config` |
| Planner type alias | `Stage1Fn` | `RingFoldChallengeConfigFn` |
| Planner docs | "stage-1 sparse-challenge closure" | "ring fold challenge config closure (`SparseChallengeConfig` per ring degree `d`)" |

Meaning (from `CommitmentConfig` docs): the sparse ring element `c(X)` used in the
**weak-binding fold** before sumcheck stage 1. It is **not** a sumcheck round
challenge. The planner passes this closure to price folded-witness norms and
operator-norm bounds during DP and entry expansion.

`fold_challenge_shape_at_level` (flat vs tensor root fold) stays aligned with the
existing `CommitmentConfig` hook name.

### Schedule lookup keys (same-point only)

Production folded path uses `OpeningBatch::same_point(padded_num_vars, num_polys)`:

```text
num_t_vectors = num_polys        (polynomials in the bundled commitment)
num_w_vectors = num_claims       (= num_polys for same-point: one claim per poly)
num_z_vectors = 1                (one commitment group)
num_commitment_groups = 1        (in generated key shape)
```

`AkitaScheduleLookupKey::new_from_opening_batch` is the canonical projection.

**Generated table enumeration** (`family_keys` in emitter) crosses:

- `num_vars` in `[min_num_vars, max_num_vars]`
- `num_polys` in per-family list (e.g. `[1, 4]` default; Jolt may ship `[1, 38]`)

This replaces the stale incidence generalization plan. If multi-commitment same-point
(`OpeningBatch::from_commitment_groups`) is enabled on the folded path in a future
PR, key derivation may extend `new_from_opening_batch` without reviving multipoint.

### `akita-schedules` crate

**Location:** `crates/akita-schedules/`

**Contents:**

- `src/generated/*.rs` (moved from `akita-planner`)
- `src/lib.rs` re-exports `table()` constructors per family
- `src/family.rs` shared types re-exported from `akita_planner::generated` OR types
  move to planner and schedules only holds data modules (prefer: types stay in
  planner, schedules is data-only)

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

- `all-schedules` — union of every family (drift CI job, not default)
- Individual `zk` variants: either `zk` feature on crate mirroring planner, or
  separate `fp128-d64-onehot-zk` features (match current `cfg(zk)` split)

**`ALL_GENERATED_FAMILIES`** moves from `akita-config` to `akita-schedules` (or
`akita-config` re-exports it only for the `gen_schedule_tables` binary). The emitter
and drift guard share this list.

### Emitter (`gen_schedule_tables`)

- Binary stays in `akita-config` (only crate that names preset `Cfg` types) OR moves
  to thin wrapper calling `akita-schedule-emit`.
- Output directory: `crates/akita-schedules/src/generated/`.
- Updates `akita-schedules` `lib.rs` feature wiring (not `resolve.rs` imports).

**Future Jolt-facing emit API** (same cutover or immediate follow-up):

```rust
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub policy: PlannerPolicy,
    pub keys: Vec<AkitaScheduleLookupKey>,
    pub ring_fold_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
}
```

Jolt runs emit with `policy_of::<JoltD64OneHot>()` and a `ScheduleKeyEnvelope` (e.g.
`num_vars: 18..=32`, `num_polys: [1, 38]`), writing `jolt-schedules/`.

### CI profile manifest and drift prevention

#### Problem

If benchmark cases live only in `.github/workflows/profile-bench.yml` and schedule
features live only in `Cargo.toml`, an agent can add `onehot_fp128_d32:28:1` to CI
without enabling `schedules-fp128-d32-onehot`, silently benchmarking DP fallback or
failing to compile gated modes.

#### Solution: single manifest, three-way check

**Canonical file:** `docs/ci-profile-manifest.toml`

```toml
# Single source of truth for profile-bench CI cases and schedule subscription.
# Do not edit AKITA_BENCH_CASES in the workflow by hand; edit this file and run
# scripts/check_ci_profile_manifest.sh.

[[bench_case]]
mode = "onehot_fp32_d128"
num_vars = 28
num_polys = 1

[[bench_case]]
mode = "onehot_fp64_d128"
num_vars = 28
num_polys = 1

[[bench_case]]
mode = "dense_fp128_d64"
num_vars = 24
num_polys = 1

[[bench_case]]
mode = "onehot_fp128_d64"
num_vars = 32
num_polys = 1

[[bench_case]]
mode = "onehot_fp128_d64"
num_vars = 30
num_polys = 4

# Mode registry: maps AKITA_MODE string → features and preset metadata.
[profile_modes.onehot_fp128_d64]
schedule_family = "fp128-d64-onehot"      # akita-schedules feature
config_schedule_feature = "schedules-fp128-d64-onehot"  # akita-config feature
profile_mode_feature = "profile-mode-onehot-fp128-d64"  # akita-pcs feature

[profile_modes.dense_fp128_d64]
schedule_family = "fp128-d64-full"
config_schedule_feature = "schedules-fp128-d64-full"
profile_mode_feature = "profile-mode-dense-fp128-d64"

# ... one entry per distinct mode appearing in bench_case ...
```

**Workflow integration** (no duplicated case list):

```yaml
- name: Load benchmark cases from manifest
  run: python3 scripts/ci_profile_manifest.py export-bench-cases >> "$GITHUB_ENV"

- name: Build profile binary
  run: cargo build --release --quiet --example profile --features profile-ci

- name: Run profile benchmark cases
  run: python3 scripts/profile_bench_report.py run ...  # cases from env
```

`scripts/ci_profile_manifest.py`:

- `export-bench-cases` — emits `AKITA_BENCH_CASES` multiline env var
- `check` — validates manifest consistency (called by CI guard)

**`scripts/check_ci_profile_manifest.sh`** (hard gate) verifies:

1. **Manifest ↔ workflow:** If workflow still contains inline cases, they match
   manifest export (prefer workflow uses export only).
2. **Manifest ↔ `profile-ci`:** Let `M` = set of distinct `mode` in `bench_case`.
   Let `F` = union of `config_schedule_feature` + `profile_mode_feature` for each
   mode in `M`. Then `akita-pcs` feature `profile-ci` must enable exactly `F` (sorted
   set equality).
3. **Manifest ↔ `modes.rs`:** Each `profile_mode_feature` appears in
   `#[cfg(feature = "...")]` on the corresponding `ProfileMode` entry.
4. **Schedule coverage (soft warning or hard):** For each bench case, if
   `num_polys` is not in the family's emitted `num_polys` list, CI warns or fails
   (DP fallback would be used; bench should reflect production intent).

**`akita-pcs/Cargo.toml`:**

```toml
[features]
profile-ci = [
    "profile-mode-onehot-fp32-d128",
    "profile-mode-onehot-fp64-d128",
    "profile-mode-dense-fp128-d64",
    "profile-mode-onehot-fp128-d64",
]
profile-mode-onehot-fp128-d64 = [
    "akita-config/schedules-fp128-d64-onehot",
]
# ...
```

**Agent workflow (document in AGENTS.md):**

> When changing CI profile benchmark cases, edit `docs/ci-profile-manifest.toml` only.
> Run `scripts/check_ci_profile_manifest.sh`. Do not hand-edit `profile-ci` feature
> lists; the checker derives the required union from the manifest.

Optional: `scripts/sync_profile_ci_features.py --write` auto-updates `Cargo.toml`
`profile-ci` from manifest (reduces manual toil; checker still required in CI).

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

- Gate each `ProfileMode` in `modes.rs`:

```rust
#[cfg(feature = "profile-mode-onehot-fp128-d64")]
ProfileMode { name: "onehot_fp128_d64", run: run_profile_onehot_fp128_d64 },
```

- `PROFILE_MODES` is the union of enabled mode features; unknown `AKITA_MODE` at
  runtime errors clearly.
- Local dev: `cargo run --release --example profile --features profile-mode-onehot-fp128-d32`
  for a single mode + schedule.

### Downstream: Jolt

1. Define `JoltD64OneHot: CommitmentConfig` (policy hooks from `fp128::D64OneHot` or
   thin newtype).
2. Crate `jolt-schedules` with generated `jolt_fp128_d64_onehot.rs` (keys:
   `num_vars` envelope × `num_polys` e.g. `[1, 38]`).
3. `schedule_catalog()` returns `Some(&JOLT_…)`; **no** `akita-schedules` dependency.
4. Jolt CI drift test: `assert_catalog_matches_dp` on Jolt keys only.
5. LayerZero PR #198 style changes (wide batch rows in Akita core) are **not**
   required for Jolt integration.

### Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Keep `shipped_table()` global registry | Cannot isolate link-time catalogs; forces downstream upstreaming |
| Move all tables out of Akita repo | User requirement: tables stay on `main`, grow as needed (e.g. D32) |
| Runtime plugin / dynamic loading | Rust static tables; unnecessary complexity |
| Per-case CI binaries (Option A) | Compile cost; user chose Option B |
| Revive incidence-based keys | Multipoint removed; spec stale; same-point suffices for Jolt mega-poly |
| `schedule_table()` on trait without features | Cargo features needed for link-time stripping |

## Documentation

- Update [`AGENTS.md`](../AGENTS.md): crate graph (`akita-schedules`), opt-in
  catalogs, manifest workflow, rename note.
- Fold into [`book/src/how/configuration.md`](../book/src/how/configuration.md) when
  implemented.
- Update [`docs/doc-blast-radius.json`](../docs/doc-blast-radius.json): add
  `akita-schedules`, `docs/ci-profile-manifest.toml`, manifest checker script.
- Add note to [`planner-incidence-generalization.md`](planner-incidence-generalization.md)
  `Status` / header: schedule-key portions superseded.
- [`crates/akita-planner/README.md`](../crates/akita-planner/README.md): engine vs
  schedules split.

## Execution

### Phase 1 — Engine cutover (no behavioral change)

1. Add `resolve_schedule` with `catalog` parameter.
2. Add `CommitmentConfig::schedule_catalog()` default `None`; wire `runtime_schedule`.
3. Create `akita-schedules`; move generated modules; feature-gate families.
4. Point presets at catalogs via features (default features = current behavior).
5. Delete `shipped_table`, `get_schedule`, planner `generated/` imports in `resolve.rs`.
6. Relocate drift tests to `akita-schedules`.
7. Rename planner `stage1` / `Stage1Fn`.

### Phase 2 — CI manifest and `profile-ci`

1. Add `docs/ci-profile-manifest.toml` with current `AKITA_BENCH_CASES`.
2. Implement `scripts/ci_profile_manifest.py` + `check_ci_profile_manifest.sh`.
3. Gate `profile` modes; add `profile-ci` feature chain.
4. Update `profile-bench.yml` to manifest export + `--features profile-ci`.
5. Add manifest check to CI (doc-guardrails or profile workflow).

### Phase 3 — Emitter and D32 expansion

1. Retarget `gen_schedule_tables` output to `akita-schedules`.
2. Emit D32 families if missing (user: "maybe even more").
3. Expose `EmitSpec` for downstream (Jolt).

### Phase 4 — Jolt (separate repo PR)

1. `JoltD64OneHot` + `jolt-schedules` as documented.

### Risks

| Risk | Mitigation |
|------|------------|
| Feature matrix explosion | One feature per family; meta `all-schedules` for full drift only |
| Missed manifest update | Hard CI checker; AGENTS.md; manifest is only edit point for cases |
| `zk` / non-`zk` table split | Mirror current `cfg(zk)` module split in features |
| Tensor/tiered presets | Separate families (already separate tables) |

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
