# Spec: CI Test Timing Telemetry and Redundant Schedule Test Prune

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | Quang Dao                                  |
| Created     | 2026-06-02                                 |
| Status      | in progress (spec-only landing)              |
| Branch      | `quang/ci-test-timing`                     |
| PR          | (single PR: prune + telemetry + CI wiring) |

## Summary

CI Test job wall time grew from roughly 16 minutes to roughly 21 minutes after the
planner refactor and follow-on preset/table growth. Today we only notice that in
the GitHub Actions UI; there is no per-test breakdown, no baseline comparison, and
no PR comment. Separately, `akita-config/tests/proof_size_comparison.rs` duplicates
almost all work of `generated_tables.rs` while checking a strictly weaker
invariant (`total_bytes` monotonicity vs full schedule equality).

This PR does three things in one cutover:

1. **Delete** the redundant `proof_size_comparison` integration test.
2. **Run** the schedule drift guard (`generated_tables`) only in the non-zk
   nextest pass (schedule expansion does not depend on the `zk` feature).
3. **Add** machine-readable CI test timing (JUnit → `summary.json` artifact → PR
   comment), mirroring the existing profile-benchmark reporting pipeline.

There is **no backward-compatibility guarantee** for CI comment shape or artifact
layout on first landing; the marker and JSON schema are new.

## Intent

### Goal

Make test-time regressions visible in every PR the same way profile benchmarks
already surface prove/verify regressions: structured data, stable baselines, one
upserted comment per PR.

### Invariants

- **`generated_schedule_tables_match_find_schedule` remains the sole schedule
  drift guard** on the default CI path. It must still compare fully resolved
  `Schedule` values (bytes + steps + expanded `LevelParams`) for every
  `(family, key)` in `ALL_GENERATED_FAMILIES`.
- **Timing telemetry must not change test selection or features.** It only
  observes nextest runs; it does not skip, filter, or reorder tests.
- **Comparisons are apples-to-apples:** non-zk pass vs non-zk baseline,
  `all-features` pass vs `all-features` baseline.
- **Profile bench comment stays separate.** Test timing uses its own HTML marker
  and artifact name; do not merge bodies with `<!-- akita-profile-bench-report -->`.

Protected by existing tests:

| Invariant | Guard |
|-----------|--------|
| Table expansion == DP | `crates/akita-config/tests/generated_tables.rs` |
| Manual schedule diff (offline) | `crates/akita-config/tests/regen_diff.rs` (`#[ignore]`) |
| Runtime DP fallback | `crates/akita-config/tests/runtime_fallback.rs` |

### Non-Goals

- **Blocking** CI on test-duration regression (hosted-runner jitter is too noisy).
  A **non-blocking** check annotation when pass wall time exceeds main × **1.35**
  is in scope (see Design).
- Merging test timing into the profile-bench PR comment.
- Replacing `cargo nextest` with a custom test runner.
- Tracking clippy/fmt job times (Test job only).
- Historical charts or Grafana; Actions artifacts + PR comments are enough for v1.
- Rewriting `profile_bench_report.py` into a shared framework in this PR (optional
  follow-up only).

## Background

### Why Test CI got slower

Analysis of run `26798445379` (PR #143, green) vs `26796574529` (planner #139 on
main):

| Nextest pass | Planner #139 | Post-#143 merge | Delta |
|--------------|-------------|-----------------|-------|
| `parallel,disk-persistence` (non-zk) | 478s / 798 tests | 672s / 812 tests | +194s |
| `--all-features` (zk) | 205s / 326 tests | 275s / 340 tests | +70s |

Dominant outliers (non-zk pass, per-test wall times from logs):

| Duration | Test |
|----------|------|
| ~491s | `akita-config::generated_tables::generated_schedule_tables_match_find_schedule` |
| ~487s | `akita-config::proof_size_comparison::refactor_does_not_increase_proof_sizes` |
| ~121s | `akita-pcs::single_poly_e2e::single_dense_nv20` |
| ~60–93s | `akita-pcs::setup::large_setup_*` (macro-expanded per preset) |

The two `akita-config` integration tests each walk ~1,268 keys × two full schedule
resolutions (`table_backed` + `regen`). They run in **both** CI passes, so they
account for roughly half of the Test job CPU.

### Why `proof_size_comparison` is redundant

`generated_tables.rs` renders each schedule as `total_bytes={} steps={:?}` and
requires **exact string equality**. That implies equal `total_bytes` and equal
steps. `proof_size_comparison.rs` only checks `regen.total_bytes <= table_backed.total_bytes`.

The monotonicity test was a **migration guard** during the planner refactor
(terminal-direct placeholder removal; see `specs/planner-refactor.md` issue 2).
After tables were regenerated, the correct contract is equality, not “DP is no
worse than stale tables.” `regen_diff.rs` remains the manual diagnostic when
iterating on DP before `gen_schedule_tables`.

## Evaluation

### Acceptance Criteria

- [ ] `crates/akita-config/tests/proof_size_comparison.rs` is **deleted**; no
  references remain in specs/workflows except historical notes pointing at this
  spec.
- [ ] `generated_schedule_tables_match_find_schedule` runs in CI **once** (non-zk
  pass only) and still passes.
- [ ] `.config/nextest.toml` defines a `ci` profile that writes JUnit XML per pass.
- [ ] `scripts/ci_test_timing_report.py` implements `merge`, `render`, and
  `failure-summary` subcommands (parallel structure to `profile_bench_report.py`).
- [ ] CI Test job uploads artifact `ci-test-timing-data` containing at least
  `summary.json`, `comment.md`, and metadata (`source_sha`, pass wall times).
- [ ] Workflow `.github/workflows/test-timing-comment.yml` upserts a PR comment
  with marker `<!-- akita-ci-test-timing -->` after CI completes (including on
  Test failure, with `failure-summary` when partial).
- [ ] PR comment includes: pass wall-time table vs **main** baseline, top 20
  slowest tests per pass, largest per-test regressions vs main, new tests ≥30s.
- [ ] When a main baseline exists and any pass has `wall_s > main_wall_s × 1.35`,
  `test-timing-comment.yml` posts a **non-blocking** GitHub check
  (`conclusion: neutral`) on `workflow_run.head_sha` summarizing which pass(es)
  crossed the threshold. PR comment also calls out the flag in pass summary.
- [ ] `push` to `main` uploads the same artifact so PRs can resolve a main
  baseline (same pattern as profile bench).
- [ ] Documented in `AGENTS.md` under Essential Commands or a short pointer to
  this spec.
- [ ] Expected CI effect: Test job wall time drops by roughly **8–12 minutes**
  from removing duplicate `proof_size_comparison` runs and running
  `generated_tables` only once (measure on first green PR).

### Testing Strategy

- Run locally:
  ```bash
  cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence
  cargo nextest run --profile ci --all-features
  python3 scripts/ci_test_timing_report.py merge ...
  python3 scripts/ci_test_timing_report.py render ... --main-baseline-dir ...
  ```
- Verify `generated_tables` still passes after moving it to non-zk-only CI.
- Dry-run comment render with a downloaded Actions artifact from a prior run.
- Full workspace `cargo test` / CI green on the PR.

Feature combinations: timing artifact must record `pass` = `non-zk` | `all-features`
separately. Zk and non-zk timings must never be merged into one ranking table.

### Performance

| Change | Expected effect |
|--------|-----------------|
| Delete `proof_size_comparison` | −2 full DP sweeps per CI Test job |
| `generated_tables` only in non-zk pass | −1 full DP sweep per CI Test job |
| JUnit + merge script | Small overhead (seconds) per pass |
| Comment workflow | No change to compile/test critical path |

Target: Test job wall time back toward **~15–17 minutes** on `ubuntu-latest`
(from ~21 minutes observed post-#143), subject to runner variance.

No change to profile benchmark workflow timing or artifacts.

## Design

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  .github/workflows/ci.yml  (job: test)                          │
│    nextest --profile ci  (non-zk)  → junit/non-zk.xml           │
│    nextest --profile ci  (all-features) → junit/all-features.xml │
│    ci_test_timing_report.py merge → summary.json                 │
│    upload-artifact: ci-test-timing-data                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ workflow_run completed
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  .github/workflows/test-timing-comment.yml                        │
│    download artifact + main/previous baselines (github-script)   │
│    ci_test_timing_report.py render → comment.md                    │
│    upsert PR comment (<!-- akita-ci-test-timing -->)             │
└─────────────────────────────────────────────────────────────────┘
```

Mirror `profile-bench.yml` + `profile-bench-comment.yml`; reuse the same baseline
resolution ideas (merge-base main artifact, previous PR run on same branch).

### CI Test job changes

Replace bare nextest invocations with profiled runs and JUnit output paths:

```yaml
- name: Run tests (all non-zk features)
  run: |
    cargo nextest run --profile ci \
      --no-default-features --features parallel,disk-persistence

- name: Run tests (all features)
  run: |
    cargo nextest run --profile ci --all-features

- name: Merge test timing report
  if: always()
  run: |
    python3 scripts/ci_test_timing_report.py merge \
      --output-dir "$RUNNER_TEMP/ci-test-timing-artifact" \
      --pass non-zk --junit target/nextest/non-zk/junit.xml \
      --pass all-features --junit target/nextest/all-features/junit.xml
    # record wall times from $GITHUB_STEP_* or date delta per pass

- name: Upload test timing artifact
  if: always()
  uses: actions/upload-artifact@...
  with:
    name: ci-test-timing-data
    path: ${{ runner.temp }}/ci-test-timing-artifact
```

**Schedule drift guard:** do **not** run `generated_tables` in the all-features
pass. Options (pick one in implementation):

- **A (preferred):** `#[cfg(not(feature = "zk"))]` on the `generated_tables` test
  module or test function so it is not built when `zk` is enabled.
- **B:** `cargo nextest run ... --exclude akita-config::generated_tables` on the
  zk pass only.

Option A is clearer: the test is meaningless under `zk` for expansion parity.

### Nextest profile (`.config/nextest.toml`)

```toml
[store]
dir = "target/nextest"

[profile.ci]
retries = { backoff = "fixed", count = 0 }

[profile.ci.junit]
path = "junit/{pass}.xml"   # implementation sets pass via env or separate profiles
```

Implementation detail: use two profile names (`ci-non-zk`, `ci-all-features`) if
a single profile cannot parameterize the JUnit path.

### `summary.json` schema (v1)

```json
{
  "schema_version": 1,
  "source_sha": "3735fc4f...",
  "source_branch": "quang/ci-test-timing",
  "workflow_run_id": 123,
  "passes": {
    "non-zk": {
      "wall_s": 672.0,
      "test_count": 812,
      "skipped": 2,
      "failed": 0,
      "tests": [
        {
          "id": "akita-config::generated_tables::generated_schedule_tables_match_find_schedule",
          "crate": "akita-config",
          "binary": "generated_tables",
          "test": "generated_schedule_tables_match_find_schedule",
          "duration_s": 491.2
        }
      ]
    },
    "all-features": { }
  }
}
```

`tests` is sorted by `duration_s` descending at merge time. Full list is stored;
render keeps top 20 for the comment.

Test `id` format: `{binary}::{test_name}` as reported by JUnit (match nextest
human output for grepability).

### PR comment layout

HTML marker: `<!-- akita-ci-test-timing -->`

Sections:

1. **Pass summary** — wall time this PR vs main, delta %, test count delta.
   Mark passes with `wall_s > main × 1.35` as **⚠ pass regression (non-blocking)**.
2. **Slowest tests** — top 20 per pass (table: rank, duration, test id).
3. **Regressions vs main** — tests where `delta_s >= max(5s, 10% of baseline)`;
   cap 15 rows.
4. **New slow tests** — in PR, not in main baseline, `duration_s >= 30`.
5. **Footer** — workflow link, SHA, disclaimer on `ubuntu-latest` fleet variance.

Also append a shortened pass summary to **`$GITHUB_STEP_SUMMARY`** in the Test
job (top 10 slow tests across both passes).

### Baseline resolution

Copy the approach from `profile-bench.yml` `Determine comparison baseline artifacts`:

| Baseline | Source |
|----------|--------|
| **main** | Successful `CI` workflow on merge-base SHA, artifact `ci-test-timing-data`; fallback to latest green `main` push |
| **previous PR** | Last completed `CI` run on same PR branch with artifact |

On `push` to `main`, always upload `ci-test-timing-data` (retention 30–90 days).

### Script: `scripts/ci_test_timing_report.py`

Subcommands (match `profile_bench_report.py` ergonomics):

| Command | Purpose |
|---------|---------|
| `merge` | Read JUnit XML files + pass metadata → write `summary.json` |
| `render` | Read current + optional baseline dirs → write `comment.md` (+ optional full `report.md`) |
| `failure-summary` | Test job failed or JUnit missing → write explanatory comment body |

Implementation notes:

- Parse JUnit with `xml.etree.ElementTree` (stdlib).
- Strip ANSI is unnecessary (JUnit is clean).
- `render --compact` optional for step summary only.

### Workflow: `test-timing-comment.yml`

```yaml
name: Akita CI Test Timing Comment

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

permissions:
  actions: read
  contents: read
  pull-requests: write
  issues: write

env:
  AKITA_TEST_TIMING_ARTIFACT_NAME: ci-test-timing-data
```

Job conditions:

- `github.event.workflow_run.event == 'pull_request'`
- `conclusion != 'cancelled'`

Steps: resolve PR number → download artifact from workflow run → resolve main /
previous baselines → `render` → upsert comment (same github-script pattern as
`profile-bench-comment.yml`).

**Pass wall-time annotation (non-blocking):** after `render`, if main baseline
exists and any pass satisfies `wall_s > main_wall_s × 1.35`, create a check run
via `github.rest.checks.create`:

| Field | Value |
|-------|--------|
| `name` | `Akita CI test timing` |
| `head_sha` | `workflow_run.head_sha` |
| `status` | `completed` |
| `conclusion` | `neutral` |
| `output.title` | e.g. `Test pass wall time above main (non-blocking)` |
| `output.summary` | Per-pass table: this PR wall_s, main wall_s, ratio |

Do **not** fail the comment job or the PR when the threshold is crossed. Omit the
check when main baseline is missing.

### Spec and doc updates

| File | Change |
|------|--------|
| `specs/planner-owns-schedule-expansion.md` | Remove `proof_size_comparison.rs` from required test list; point to `generated_tables` only |
| `specs/ci-test-timing.md` | This document |
| `AGENTS.md` | One bullet: PRs receive CI test timing comment; local repro command |

## Alternatives Considered

| Alternative | Why not |
|-------------|---------|
| Merge into profile-bench comment | Different cadence, failure modes, and metrics; comment becomes huge |
| Keep `proof_size_comparison` but `#[ignore]` | Still confuses contributors; weaker invariant has no long-term role |
| Log-parse nextest stdout | Fragile under ANSI/format changes |
| Fail CI if Test job > baseline × 1.2 | Runner jitter; use neutral check at ×1.35 instead |
| `cargo test -- -Z unstable-options` timing | Nextest is already canonical in CI |

## Execution

### Implementation checklist

1. Branch `quang/ci-test-timing` from current `main` (includes #143).
2. Delete `crates/akita-config/tests/proof_size_comparison.rs`.
3. Gate `generated_tables` to non-zk only.
4. Add `.config/nextest.toml` profiles + update `ci.yml` Test job.
5. Add `scripts/ci_test_timing_report.py`.
6. Add `.github/workflows/test-timing-comment.yml`.
7. Update `specs/planner-owns-schedule-expansion.md` testing list.
8. Update `AGENTS.md`.
9. Open PR with body linking this spec; paste first timing comment screenshot.

### Risks

| Risk | Mitigation |
|------|------------|
| No main baseline on first PR | Comment shows “main baseline unavailable”; still lists top slow tests |
| JUnit path mismatch across nextest versions | Pin nextest via `taiki-e/install-action`; document version in spec |
| Test count changes skew wall-time delta | Always show test counts beside pass durations |
| Duplicate work if someone re-adds monotonicity test | Spec + PR description state redundancy argument |
| Check annotation noise on first landing | Only emit when main baseline exists; `neutral` conclusion |

## Documentation

- This spec is the source of truth for the PR.
- `AGENTS.md`: add subsection “CI test timing” with artifact name, marker, and
  local repro one-liner.
- No change to `specs/profile-bench-coverage-matrix.md`.

## References

- `specs/planner-refactor.md` — placeholder-removal migration context for
  `proof_size_comparison`
- `specs/planner-owns-schedule-expansion.md` — drift guard requirements
- `.github/workflows/profile-bench.yml` — baseline artifact pattern
- `.github/workflows/profile-bench-comment.yml` — upsert comment pattern
- `scripts/profile_bench_report.py` — script structure template
- `crates/akita-config/tests/generated_tables.rs` — retained guard
- `crates/akita-config/tests/regen_diff.rs` — ignored manual diagnostic
- CI analysis: runs `26798445379` vs `26796574529` (2026-06-02)
