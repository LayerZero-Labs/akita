# Spec: Schedule catalog ownership and opt-in shipped tables

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
The planner keeps DP fallback as the universal correctness path. Downstream repos
ship their own catalogs (Jolt: D64 one-hot + Jolt-specific batch widths) without
depending on `akita-schedules`.

Per-preset feature gating makes link stripping fall out for free: the CI `profile`
binary links table data only for the presets whose `schedules-*` feature is enabled.
CI builds with a single `profile-ci` feature that enables the union of schedule
families for the benchmark matrix. A **minimal CI guard** reads the existing
`AKITA_BENCH_CASES` list in `profile-bench.yml` and asserts `profile-ci` enables the
schedule feature each benched mode needs. No new manifest source-of-truth, exporter,
or three-way reconciler is introduced.

## Intent

### Goal

Refactor schedule resolution so:

1. **`akita-planner` is engine-only** (DP + compact entry expansion + generic table
   lookup). No global preset→table registry.
2. **`akita-schedules` holds all Akita-shipped tables** on `main`, feature-gated per
   family, optional for consumers.
3. **Each `CommitmentConfig` preset opt-ins** to zero or one catalog via
   `schedule_catalog() -> Option<GeneratedScheduleTable>` (default `None` → DP only).
   The table is a `Copy` value (tag + `&'static` entry slice), returned by value as
   `shipped_table` already does.
4. **Downstream projects** (Jolt) depend on the engine + their own generated catalog
   crate, not on `akita-schedules`.
5. **CI `profile` builds with `--features profile-ci`**, linking only schedules for
   modes in the CI benchmark matrix, enforced by a minimal guard over the existing
   `AKITA_BENCH_CASES` (no new manifest SSOT).
6. **Rename planner misnomers** (`stage1` closure parameters) to match
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

### Invariants

- **DP is the source of truth.** Any shipped catalog row MUST match
  `find_schedule(key, policy, …)` for the same policy and hooks. Drift guards enforce
  this per catalog (today: `generated_schedule_tables_match_find_schedule`; becomes
  per-family tests in `akita-schedules` and downstream catalogs).
- **Verifier no-panic.** `resolve_schedule` and `find_schedule` return `Result`,
  never panic on malformed keys (existing contract).
- **Determinism.** Prover and verifier both call `Cfg::runtime_schedule` with the
  same `policy_of::<Cfg>()` and hook fns; transcript `PlanSection` digest unchanged
  for identical inputs. A table-backed prover and a DP-only verifier (different
  `schedules-*` feature sets) still agree **because** the drift guard enforces
  table ≡ DP for every shipped key; this is the load-bearing reason opt-out is safe.
- **Default features preserve current behavior.** `akita-config` default features
  keep every production preset table-backed, so plain `cargo test` / `cargo build` /
  CI resolve byte-identical schedules to today. Only minimal/downstream consumers
  (Jolt) turn `schedules-*` off.
- **Default is DP-only for non-opted-in presets.** `schedule_catalog()` default
  `None` must not change schedules for presets that do not opt in (modulo explicit
  preset feature enables).
- **Same-point batching only.** Lookup keys derive from `OpeningBatch` /
  `OpeningBatch::same_point(num_vars, num_polys)` (and `new_from_opening_batch`).
  No multipoint keys, no `ClaimIncidenceSummary` schedule path (type not in tree).
- **Table miss falls back to DP**, never errors solely because a row is absent (unless
  DP itself rejects the key).
- **CI bench coverage.** Every mode in `AKITA_BENCH_CASES` must have its `schedules-*`
  feature enabled by `profile-ci`, so the bench measures the shipped table rather than
  the DP fallback (one-directional hard CI check).

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
- [ ] `akita_planner::resolve_schedule(key, policy, ring_challenge_config, fold_challenge_shape_at_level, catalog: Option<GeneratedScheduleTable>)` is the single runtime entry point (catalog passed **by value** — `GeneratedScheduleTable` is `Copy`).
- [ ] Planner public closures use `ring_challenge_config` (not `stage1`) and `fold_challenge_shape_at_level` (not `fold_shape`) in signatures and docs.
- [ ] Internal type `Stage1Fn` renamed to `RingChallengeConfigFn`.

#### `akita-schedules` crate

- [ ] New workspace crate `crates/akita-schedules` contains all generated table
  modules moved from `akita-planner/src/generated/`.
- [ ] Each family is behind a Cargo feature (e.g. `fp128-d64-onehot`, `fp128-d64-full`).
- [ ] `akita-planner` default build contains **no** generated table `.rs` files.
- [ ] `gen_schedule_tables` writes into `akita-schedules/src/generated/` and updates
  family feature wiring (not `akita-planner`). The binary itself stays in
  `akita-config` (only crate that can name presets); only its output dir changes.
- [ ] Table **types** (`GeneratedScheduleTable`, `GeneratedStep`, `GeneratedFoldStep`,
  `GeneratedScheduleKey`, `table_entry`, `expand`, `schedule_from_entry`) stay in
  `akita-planner`; `akita-schedules` is data-only and depends on `akita-planner` for
  them. No dependency cycle (`akita-schedules → akita-planner`; `akita-config →
  akita-schedules`).
- [ ] `ALL_GENERATED_FAMILIES` and the drift guard `generated_schedule_tables_match_find_schedule`
  **stay in `akita-config`** (they name preset `Cfg` types and call
  `Cfg::runtime_schedule`, which `akita-schedules` cannot do). Each family row is
  gated behind its `schedules-*` feature; a meta-feature runs the full cross-product.

#### Opt-in presets

- [ ] `CommitmentConfig::schedule_catalog() -> Option<GeneratedScheduleTable>`
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

#### CI profile isolation (Option B, minimal guard)

- [ ] `cargo build --release --example profile --features profile-ci` is what CI uses.
- [ ] `profile-ci` on `akita-pcs` enables the union of `schedules-*` features for the
  presets exercised by the benchmark matrix (and nothing else).
- [ ] `AKITA_BENCH_CASES` in `.github/workflows/profile-bench.yml` stays the single
  list of CI bench cases (no new manifest SSOT, no exporter script).
- [ ] A **hard CI gate** `scripts/check_profile_ci_features.sh` parses
  `AKITA_BENCH_CASES` from the workflow, maps each `mode` to its required
  `schedules-*` feature, and fails if `profile-ci` does not enable that feature.
  This is the only drift check; it does not reconcile three separate sources.

#### Downstream path (documented, Jolt follow-up)

- [ ] Spec documents Jolt pattern: `jolt-schedules` crate +
  `JoltD64OneHot::schedule_catalog()` without `akita-schedules` dependency.
- [ ] A reusable emit API (`EmitSpec` accepting `PlannerPolicy` + hook fn pointers)
  so Jolt can generate catalogs outside `ALL_GENERATED_FAMILIES`. This is **Phase 3+**
  follow-up, not required for the engine cutover.

#### Correctness

- [ ] `cargo test --workspace` and `cargo test --workspace --features zk` pass.
- [ ] Drift guard (kept in `akita-config`): every enabled family's table-hit
  expansion matches the DP (`generated_schedule_tables_match_find_schedule`).
- [ ] `runtime_fallback` tests pass. Note: `tests/runtime_fallback.rs` currently
  calls `akita_planner::shipped_table` directly (line ~47); migrate it to
  `Cfg::schedule_catalog()` + `table_entry` since `shipped_table` is removed.
- [ ] Profile benchmark CI cases produce identical timing-quality results (median
  setup/prove/verify within noise of pre-refactor; no functional regression).

### Testing Strategy

| Area | Tests |
|------|-------|
| Engine | Existing `runtime_fallback.rs` (migrated off `shipped_table`), planner unit tests |
| Per-family drift | `generated_schedule_tables_match_find_schedule` in `akita-config`, gated per `schedules-*` feature; meta-feature for full cross-product |
| Preset opt-in | `akita-config` tests: with feature on, table hit; with feature off, DP path |
| profile-ci coverage | `scripts/check_profile_ci_features.sh` in CI (parses `AKITA_BENCH_CASES`) |
| Profile compile | CI builds `profile` with only `profile-ci` (no `--all-features`) |
| E2E | `akita-pcs` e2e, `batched_aggregated_e2e`, tiered/tensor cases with appropriate features |

Feature combinations: run drift guards under default and `zk` for families that ship
`zk` tables (separate modules or `*_zk` features).

### Performance

- **No proof-size or transcript-byte change** for identical `(preset, key)` pairs
  when the same catalog row is used.
- **Compile time / link size:** `profile` CI binary MUST NOT link `akita-schedules`
  families outside the `profile-ci` union (verify via `cargo tree` / optional `nm`
  symbol check).
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
   jolt-schedules → akita-planner (types) + Jolt preset
        (no akita-schedules dependency)
```

#### `resolve_schedule` (replaces `get_schedule`)

```rust
pub fn resolve_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError>
```

Behavior:

1. If `catalog` is `Some`, look up `generated_schedule_lookup_key(key)` in the
   table; on hit, `schedule_from_entry`.
2. Otherwise (no catalog, or miss), `find_schedule`.

`resolve_schedule` no longer computes `root_fold_is_tensor` at all — that hack only
existed so the global registry could disambiguate the tensor table from the flat one.
`fp128::D64OneHot` and `tensor_verifier::fp128::D64OneHotTensor` are different presets
that each return their own `schedule_catalog()`, so the discriminator is deleted.

#### `CommitmentConfig` hook

```rust
/// Application-owned precomputed schedules. Default: none (DP-only).
fn schedule_catalog() -> Option<GeneratedScheduleTable> {
    None
}

fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
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

- `schedules-fp128-d64-onehot` → `akita-schedules/fp128-d64-onehot`
- Preset bundles (e.g. `fp128-d64-onehot-preset`) enable schedule + preset impl

Default `akita-config` features for dev can include common schedules; **Jolt omits
all `schedules-*` features**.

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

- `src/generated/*.rs` (moved from `akita-planner`), referencing the table types via
  `akita_planner::generated::…`
- `src/lib.rs` exposes `*_table()` constructors per family
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

- `all-schedules` — union of every family (drift CI job, not default)
- Individual `zk` variants: either `zk` feature on crate mirroring planner, or
  separate `fp128-d64-onehot-zk` features (match current `cfg(zk)` split)

**`ALL_GENERATED_FAMILIES`** **stays in `akita-config`.** It is a list of
`regen::<Cfg>` / `table_backed::<Cfg>` function pointers that name preset `Cfg` types
and call `Cfg::runtime_schedule` — only `akita-config` can name presets, and
`akita-schedules` (data-only, below `akita-config`) cannot. The emitter and the drift
guard both live in `akita-config` and share this list. `akita-schedules` holds **only**
the static table data + `*_table()` constructors; it does not enumerate presets.

### Emitter (`gen_schedule_tables`)

- Binary **stays in `akita-config`** (only crate that names preset `Cfg` types). The
  single change in the engine cutover is its output directory.
- Output directory: `crates/akita-schedules/src/generated/`.
- Updates `akita-schedules` `lib.rs` feature wiring (not `resolve.rs` imports).

**Future Jolt-facing emit API** (Phase 3+ follow-up, not the engine cutover):

```rust
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub policy: PlannerPolicy,
    pub keys: Vec<AkitaScheduleLookupKey>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
}
```

Jolt runs emit with `policy_of::<JoltD64OneHot>()` and a `ScheduleKeyEnvelope` (e.g.
`num_vars: 18..=32`, `num_polys: [1, 38]`), writing `jolt-schedules/`.

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
`--features profile-ci` linking the schedule families for the bench matrix needs no
manifest — it is just a feature list. We do not need a new source of truth, a TOML
file, an exporter script, or a three-way reconciler.

#### Solution: one feature + one coverage guard

**`akita-pcs/Cargo.toml`** — `profile-ci` enables the `schedules-*` features for the
presets the benchmark matrix exercises:

```toml
[features]
profile-ci = [
    "akita-config/schedules-fp32-d128-onehot",
    "akita-config/schedules-fp64-d128-onehot",
    "akita-config/schedules-fp128-d64-onehot",
    "akita-config/schedules-fp128-d64-full",
]
```

**Workflow** keeps `AKITA_BENCH_CASES` as-is and just builds with the feature:

```yaml
- name: Build profile binary
  run: cargo build --release --quiet --example profile --features profile-ci
```

**`scripts/check_profile_ci_features.sh`** (hard CI gate) is the single drift check:

1. Parse `AKITA_BENCH_CASES` from `profile-bench.yml` → the set of bench `mode`s.
2. Map each `mode` to its required `schedules-*` feature via a small literal table
   in the script (e.g. `onehot_fp128_d64 → schedules-fp128-d64-onehot`).
3. Parse the `profile-ci` feature list from `akita-pcs/Cargo.toml`.
4. **Fail** if any benched mode's required feature is not enabled by `profile-ci`.

The check is one-directional (every benched mode must be covered); it does not force
exact set equality, so `profile-ci` may carry a small extra family without failing.
The mode→feature table is the one literal an agent edits when adding a bench case;
AGENTS.md documents this in one line. Optionally the same check warns if a bench case
uses a `num_polys` outside the family's emitted list (would hit DP, not the table).

Modes do **not** need `#[cfg]` gates in `modes.rs`: gating modes only saves compile
time of unused mode code (a weaker goal) and is what forced the rejected three-way
reconciler. Keeping all modes always-compiled keeps the guard to a single direction.

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
- Local dev: `cargo run --release --example profile --features akita-config/schedules-fp128-d64-onehot`
  links one family; without it, the same mode runs DP-backed.

### Downstream: Jolt

1. Define `JoltD64OneHot: CommitmentConfig` (policy hooks from `fp128::D64OneHot` or
   thin newtype).
2. Crate `jolt-schedules` with generated `jolt_fp128_d64_onehot.rs` (keys:
   `num_vars` envelope × `num_polys` e.g. `[1, 38]`).
3. `schedule_catalog()` returns `Some(jolt_schedules::…_table())`; **no**
   `akita-schedules` dependency.
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
| `ALL_GENERATED_FAMILIES` / drift guard in `akita-schedules` | Impossible: they name preset `Cfg` types and call `Cfg::runtime_schedule`; only `akita-config` can. Stay there |
| TOML manifest SSOT + exporter + three-way reconciler | Over-built; link stripping falls out of feature gating, so a one-directional coverage check over the existing `AKITA_BENCH_CASES` suffices (also resolves the Bugbot inconsistency between the three rule sets) |
| `&'static GeneratedScheduleTable` return | `GeneratedScheduleTable` is `Copy`, built by value across the zk cfg branch; no static to borrow. Return by value |

## Documentation

- Update [`AGENTS.md`](../AGENTS.md): crate graph (`akita-schedules`), opt-in
  catalogs, one-line "edit the mode→feature table in `check_profile_ci_features.sh`
  when adding a bench case" note, and the `stage1 → ring_challenge_config` rename.
- Fold into [`book/src/how/configuration.md`](../book/src/how/configuration.md) when
  implemented.
- Update [`docs/doc-blast-radius.json`](../docs/doc-blast-radius.json): add
  `akita-schedules` and `scripts/check_profile_ci_features.sh`.
- Add note to [`planner-incidence-generalization.md`](planner-incidence-generalization.md)
  `Status` / header: schedule-key portions superseded.
- [`crates/akita-planner/README.md`](../crates/akita-planner/README.md): engine vs
  schedules split.

## Execution

### Phase 1 — Engine cutover (no behavioral change)

1. Add `resolve_schedule` with `catalog: Option<GeneratedScheduleTable>` parameter
   (by value); rename planner `stage1`/`Stage1Fn` → `ring_challenge_config`/
   `RingChallengeConfigFn`.
2. Add `CommitmentConfig::schedule_catalog()` default `None`; wire `runtime_schedule`.
3. Create `akita-schedules` (data-only, depends on `akita-planner` types); move
   generated `.rs` modules there; feature-gate families. Retarget `gen_schedule_tables`
   output dir.
4. Point presets at catalogs via `#[cfg]`-gated `schedule_catalog()` overrides
   (default features = current behavior).
5. Delete `shipped_table`, `get_schedule`, the planner `generated/` `*_table` import
   block, and the `root_fold_is_tensor` hack in `resolve.rs`.
6. Keep `ALL_GENERATED_FAMILIES` + drift guard in `akita-config`; gate each family row
   behind its `schedules-*` feature; add `all-schedules` meta-feature. Migrate
   `tests/runtime_fallback.rs` off `shipped_table` to `Cfg::schedule_catalog()`.

### Phase 2 — `profile-ci` and the coverage guard

1. Add `profile-ci` feature on `akita-pcs` enabling the bench matrix's `schedules-*`.
2. Add `scripts/check_profile_ci_features.sh` (parses `AKITA_BENCH_CASES`).
3. Update `profile-bench.yml` to build `--features profile-ci`; keep `AKITA_BENCH_CASES`.
4. Wire the coverage guard into CI (doc-guardrails or the profile workflow).

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
| Bench case added without its schedule feature | `check_profile_ci_features.sh` hard gate parses `AKITA_BENCH_CASES`; one-line AGENTS.md note |
| Production preset silently degraded to DP by a missing feature | `akita-config` default features keep production presets table-backed; drift guard runs under those defaults |
| `zk` / non-`zk` table split | Mirror current `cfg(zk)` module split in features |
| Tensor/tiered presets | Separate families (already separate tables); each names its own catalog, so the `root_fold_is_tensor` runtime hack is deleted |

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
