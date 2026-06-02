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
2. **Keep** the schedule drift guard (`generated_tables`) for both shipped table
   sets, but avoid the accidental duplicate work of the weaker monotonicity
   guard.
3. **Add** machine-readable CI test timing (JUnit → `summary.json` artifact →
   trusted renderer → PR comment), mirroring the existing profile-benchmark
   reporting pipeline while keeping the privileged comment workflow away from
   PR-controlled code execution.

There is **no backward-compatibility guarantee** for CI comment shape or artifact
layout on first landing; the marker and JSON schema are new.

## Intent

### Goal

Make test-time regressions visible in every PR the same way profile benchmarks
already surface prove/verify regressions: structured data, stable baselines, one
upserted comment per PR.

### Invariants

- **`generated_schedule_tables_match_find_schedule` remains the sole schedule
  drift guard** on the CI path. It must still compare fully resolved `Schedule`
  values (bytes + steps + expanded `LevelParams`) for every `(family, key)` in
  `ALL_GENERATED_FAMILIES` under the non-zk and zk generated-table modules.
- **Timing telemetry must not change test selection or features.** It observes
  nextest runs; it does not skip, filter, or reorder tests. Any CI-only pruning
  must be called out as a separate coverage tradeoff, not hidden inside the
  telemetry wiring.
- **Comparisons are apples-to-apples:** non-zk pass vs non-zk baseline,
  `all-features` pass vs `all-features` baseline.
- **Profile bench comment stays separate.** Test timing uses its own HTML marker
  and artifact name; do not merge bodies with `<!-- akita-profile-bench-report -->`.
- **Privileged workflow trust boundary stays narrow.** The `workflow_run`
  comment job may read artifacts and post comments/checks, but it must not run
  scripts or post pre-rendered Markdown from a PR branch. The authoritative PR
  comment is rendered from trusted `main` code over untrusted structured artifact
  data.

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
- [ ] `generated_schedule_tables_match_find_schedule` still runs in CI for the
  non-zk generated tables and for the zk generated tables. If future
  measurements prove the zk pass is too expensive, replace it with an explicit
  zk-table drift guard rather than silently removing coverage.
- [ ] `.config/nextest.toml` defines explicit `ci-non-zk` and `ci-all-features`
  profiles, each writing a fixed JUnit path (`target/nextest/<profile>/junit.xml`).
- [ ] `scripts/ci_test_timing_report.py` implements `merge`, `render`, and
  `failure-summary` subcommands (parallel structure to `profile_bench_report.py`).
- [ ] CI Test job uploads artifact `ci-test-timing-data` containing at least
  `summary.json`, optional debug `report.md`, and metadata (`source_sha`,
  `source_branch`, pass start/end timestamps, pass wall times, exit codes).
- [ ] Workflow `.github/workflows/test-timing-comment.yml` upserts a PR comment
  with marker `<!-- akita-ci-test-timing -->` after CI completes (including on
  Test failure, with `failure-summary` when partial), using a trusted checkout of
  `main` for `scripts/ci_test_timing_report.py` and without checking out or
  executing PR-controlled code.
- [ ] PR comment includes: pass wall-time table vs **main** baseline, top 20
  slowest tests per pass, largest per-test regressions vs main, new tests ≥30s.
- [ ] When a main baseline exists and any pass has `wall_s > main_wall_s × 1.35`,
  `test-timing-comment.yml` posts a **non-blocking** GitHub check
  (`conclusion: neutral`) on `workflow_run.head_sha` summarizing which pass(es)
  crossed the threshold. The workflow declares `checks: write`. PR comment also
  calls out the flag in pass summary.
- [ ] The renderer escapes Markdown/HTML for test ids, failure messages, branch
  names, commit subjects, and baseline labels before inserting artifact data into
  `comment.md`.
- [ ] `push` to `main` uploads the same artifact so PRs can resolve a main
  baseline (same pattern as profile bench).
- [ ] Documented in `AGENTS.md` under Essential Commands or a short pointer to
  this spec.
- [ ] Expected CI effect: Test job CPU work drops by two full DP sweeps from
  removing duplicate `proof_size_comparison` runs while retaining both
  schedule-table drift checks. Wall-time savings are measured on the first green
  PR rather than asserted up front, because nextest may overlap the two long
  schedule tests.

### Testing Strategy

- Run locally:
  ```bash
  cargo nextest run --profile ci-non-zk --no-default-features --features parallel,disk-persistence
  cargo nextest run --profile ci-all-features --all-features
  python3 scripts/ci_test_timing_report.py merge ...
  python3 scripts/ci_test_timing_report.py render ... --main-baseline-dir ...
  ```
- Verify `generated_tables` still passes in both feature combinations, so the
  non-zk and zk shipped tables are both checked against DP output.
- Dry-run comment render with a downloaded Actions artifact from a prior run.
- Add fixture tests for:
  - happy-path two-pass JUnit merge;
  - missing JUnit / failed first pass / skipped second pass;
  - duplicate test ids across binaries;
  - malicious Markdown or HTML in test names and failure text;
  - no main baseline;
  - baseline schema mismatch;
  - threshold-crossing neutral check metadata.
- Full workspace `cargo test` / CI green on the PR.

Feature combinations: timing artifact must record `pass` = `non-zk` | `all-features`
separately. Zk and non-zk timings must never be merged into one ranking table.

### Performance

| Change | Expected effect |
|--------|-----------------|
| Delete `proof_size_comparison` | −2 full DP sweeps per CI Test job |
| Keep `generated_tables` in both passes | preserves non-zk + zk table drift coverage |
| JUnit + merge script | Small overhead (seconds) per pass |
| Comment workflow | No change to compile/test critical path |

Initial wall-time target: improve from the observed **~21 minutes** on
`ubuntu-latest` without losing zk drift coverage; set the numeric target from the
first green timing artifact.
The old **~15–17 minute** target required removing the all-features
`generated_tables` run, which this revision defers until there is a replacement
zk-table guard.
If the retained zk drift guard still dominates after first telemetry lands, follow
up with a focused replacement that checks zk shipped tables without running the
full all-features workspace pass.

No change to profile benchmark workflow timing or artifacts.

## Design

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  .github/workflows/ci.yml  (job: test)                          │
│    nextest --profile ci-non-zk → target/nextest/ci-non-zk/...    │
│    nextest --profile ci-all-features → target/nextest/...        │
│    ci_test_timing_report.py merge → summary.json/report          │
│    upload-artifact: ci-test-timing-data                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ workflow_run completed
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  .github/workflows/test-timing-comment.yml                        │
│    download artifact from completed CI run                       │
│    checkout trusted main + render comment from summary.json      │
│    upsert PR comment + optional neutral check                    │
└─────────────────────────────────────────────────────────────────┘
```

Mirror `profile-bench.yml` + `profile-bench-comment.yml`, but keep the privileged
comment workflow narrower than the profile-bench producer: the `workflow_run`
workflow must not checkout or execute PR-controlled files, and it must not post
pre-rendered Markdown from PR artifacts. Baseline resolution may happen in either
workflow, but the authoritative PR comment body is rendered by code checked out
from the repository default branch. The untrusted artifact supplies only
structured data (`summary.json` plus raw JUnit-derived fields).

### CI Test job changes

Replace bare nextest invocations with profiled runs and JUnit output paths:

```yaml
- name: Run tests (all non-zk features)
  id: test-non-zk
  continue-on-error: true
  run: |
    start_epoch=$(date +%s)
    set +e
    cargo nextest run --profile ci-non-zk \
      --no-default-features --features parallel,disk-persistence
    status=$?
    set -e
    end_epoch=$(date +%s)
    {
      echo "AKITA_TEST_TIMING_NON_ZK_START=$start_epoch"
      echo "AKITA_TEST_TIMING_NON_ZK_END=$end_epoch"
      echo "AKITA_TEST_TIMING_NON_ZK_STATUS=$status"
    } >> "$GITHUB_ENV"
    exit "$status"

- name: Run tests (all features)
  id: test-all-features
  continue-on-error: true
  run: |
    start_epoch=$(date +%s)
    set +e
    cargo nextest run --profile ci-all-features --all-features
    status=$?
    set -e
    end_epoch=$(date +%s)
    {
      echo "AKITA_TEST_TIMING_ALL_FEATURES_START=$start_epoch"
      echo "AKITA_TEST_TIMING_ALL_FEATURES_END=$end_epoch"
      echo "AKITA_TEST_TIMING_ALL_FEATURES_STATUS=$status"
    } >> "$GITHUB_ENV"
    exit "$status"

- name: Merge test timing report
  if: always()
  run: |
    python3 scripts/ci_test_timing_report.py merge \
      --output-dir "$RUNNER_TEMP/ci-test-timing-artifact" \
      --source-sha "$GITHUB_SHA" \
      --source-branch "${{ github.head_ref || github.ref_name }}" \
      --pass non-zk \
      --junit target/nextest/ci-non-zk/junit.xml \
      --started-at "$AKITA_TEST_TIMING_NON_ZK_START" \
      --finished-at "$AKITA_TEST_TIMING_NON_ZK_END" \
      --exit-code "$AKITA_TEST_TIMING_NON_ZK_STATUS" \
      --pass all-features \
      --junit target/nextest/ci-all-features/junit.xml \
      --started-at "$AKITA_TEST_TIMING_ALL_FEATURES_START" \
      --finished-at "$AKITA_TEST_TIMING_ALL_FEATURES_END" \
      --exit-code "$AKITA_TEST_TIMING_ALL_FEATURES_STATUS"
    python3 scripts/ci_test_timing_report.py render \
      "$RUNNER_TEMP/ci-test-timing-artifact/summary.json" \
      --output-dir "$RUNNER_TEMP/ci-test-timing-artifact" \
      --compact
    cat "$RUNNER_TEMP/ci-test-timing-artifact/report.md" >> "$GITHUB_STEP_SUMMARY"

- name: Upload test timing artifact
  if: always()
  uses: actions/upload-artifact@...
  with:
    name: ci-test-timing-data
    path: ${{ runner.temp }}/ci-test-timing-artifact

- name: Fail if any test pass failed
  if: always()
  run: |
    if [ "${AKITA_TEST_TIMING_NON_ZK_STATUS:-1}" -ne 0 ]; then
      exit "${AKITA_TEST_TIMING_NON_ZK_STATUS:-1}"
    fi
    if [ "${AKITA_TEST_TIMING_ALL_FEATURES_STATUS:-1}" -ne 0 ]; then
      exit "${AKITA_TEST_TIMING_ALL_FEATURES_STATUS:-1}"
    fi
```

The final status step preserves the existing failing-CI behavior while still
uploading partial timing artifacts.

**Schedule drift guard:** keep `generated_tables` in both nextest passes for this
cutover. The generated schedule modules are selected by `#[cfg(feature = "zk")]`,
so the non-zk and all-features runs validate different shipped table arrays. Do
not add `#[cfg(not(feature = "zk"))]` to the integration test. If later telemetry
shows the all-features drift guard is still too expensive, design a separate
focused zk drift job/test that validates the zk table arrays before excluding it
from the all-features workspace pass.

### Nextest profiles (`.config/nextest.toml`)

```toml
[store]
dir = "target/nextest"

[profile.ci-non-zk]
retries = { backoff = "fixed", count = 0 }

[profile.ci-non-zk.junit]
path = "junit.xml"

[profile.ci-all-features]
retries = { backoff = "fixed", count = 0 }

[profile.ci-all-features.junit]
path = "junit.xml"
```

Nextest writes JUnit paths relative to `target/nextest/<profile>/`, so the merge
script reads `target/nextest/ci-non-zk/junit.xml` and
`target/nextest/ci-all-features/junit.xml`.

### `summary.json` schema (v1)

```json
{
  "schema_version": 1,
  "source_sha": "3735fc4f...",
  "source_branch": "quang/ci-test-timing",
  "workflow_run_id": 123,
  "generated_at": "2026-06-02T05:00:00Z",
  "passes": {
    "non-zk": {
      "profile": "ci-non-zk",
      "started_at_epoch": 1780370000,
      "finished_at_epoch": 1780370672,
      "wall_s": 672.0,
      "exit_code": 0,
      "test_count": 812,
      "skipped": 2,
      "failed": 0,
      "tests": [
        {
          "id": "generated_tables::generated_schedule_tables_match_find_schedule",
          "crate": "akita-config",
          "package": "akita-config",
          "binary": "generated_tables",
          "test": "generated_schedule_tables_match_find_schedule",
          "classname": "generated_tables",
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

Test `id` format: `{binary}::{test_name}`. The merge command reads the JUnit
`testsuite.name` / `testcase.classname` / `testcase.name` fields and preserves
raw fields separately. If two JUnit records collide on `id`, append a stable
`#{n}` suffix and retain the original fields so the comment remains
deterministic.

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

All artifact-derived strings must be escaped before they enter Markdown or HTML.
This includes test ids, JUnit class names, failure messages, branch names,
commit subjects, baseline labels, and workflow URLs.

### Baseline resolution

Copy the artifact lookup approach from `profile-bench.yml` `Determine comparison
baseline artifacts`, but run it in the trusted comment workflow so the renderer
can compare current PR data against downloaded baselines before posting:

| Baseline | Source |
|----------|--------|
| **main** | Successful `CI` workflow on merge-base SHA, artifact `ci-test-timing-data`; fallback to latest green `main` push |
| **previous PR** | Last completed `CI` run on same PR branch with artifact |

On `push` to `main`, always upload `ci-test-timing-data` (retention 30–90 days).
On the first landing PR, the comment workflow may not be active on the default
branch yet; that is acceptable. The implementation PR must still produce the
raw artifact and validate rendering locally or with a manual `workflow_dispatch`
dry run after merge.

### Script: `scripts/ci_test_timing_report.py`

Subcommands (match `profile_bench_report.py` ergonomics):

| Command | Purpose |
|---------|---------|
| `merge` | Read JUnit XML files + pass metadata → write `summary.json` |
| `render` | Read current + optional baseline dirs → write `comment.md` and `report.md` |
| `failure-summary` | Test job failed or JUnit missing → write explanatory comment body |

Implementation notes:

- Parse JUnit with `xml.etree.ElementTree` (stdlib).
- Strip ANSI is unnecessary (JUnit is clean).
- `merge` accepts missing JUnit paths and records the pass as `missing_junit`
  rather than failing before artifact upload.
- `render --compact` is for the CI step summary/debug artifact only. The
  authoritative PR comment render in `test-timing-comment.yml` uses the trusted
  default-branch copy of the script.
- Escaping helpers should mirror `md_text` / `code_text` in
  `scripts/profile_bench_report.py`.

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
  checks: write

env:
  AKITA_TEST_TIMING_ARTIFACT_NAME: ci-test-timing-data
```

Job conditions:

- `github.event.workflow_run.event == 'pull_request'`
- `conclusion != 'cancelled'`

Steps:

1. Resolve PR number from `github.event.workflow_run.pull_requests`.
2. Checkout the repository default branch (`ref:
   ${{ github.event.repository.default_branch }}`), not the PR head.
3. Download the current `ci-test-timing-data` artifact from
   `github.event.workflow_run.id`.
4. Resolve and download main / previous baseline artifacts.
5. Run the trusted default-branch `scripts/ci_test_timing_report.py render` over
   the downloaded `summary.json` files.
6. Upsert the rendered comment using the same marker-based github-script pattern
   as `profile-bench-comment.yml`.

Do not read `comment.md` from the current PR artifact for the posted PR comment.
The artifact may include a debug render for the CI summary, but the posted body is
always generated in this trusted workflow.

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
| Post `comment.md` rendered by the PR run | Lets PR-controlled code choose the privileged bot comment body |
| `#[cfg(not(feature = "zk"))]` the drift guard | Removes local and CI coverage for zk generated tables |
| Single `ci` nextest profile with dynamic JUnit path | Nextest JUnit paths are profile-relative and not template-expanded by pass name |

## Execution

### Implementation checklist

1. Branch `quang/ci-test-timing` from current `main` (includes #143).
2. Delete `crates/akita-config/tests/proof_size_comparison.rs`.
3. Keep `generated_tables` active in both nextest passes.
4. Add `.config/nextest.toml` `ci-non-zk` / `ci-all-features` profiles + update
   `ci.yml` Test job.
5. Add `scripts/ci_test_timing_report.py`.
6. Add `.github/workflows/test-timing-comment.yml` with default-branch checkout,
   `checks: write`, and no PR-head checkout.
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
| Privileged workflow posts attacker-controlled Markdown | Trusted workflow renders from escaped structured data; never posts PR artifact Markdown |
| Retained zk drift guard leaves less savings than expected | Measure first artifact, then design a focused zk drift replacement if needed |
| `continue-on-error` masks failing tests | Final status step replays captured pass exit codes after artifact upload |

### Further Investigation Before Implementation

- Confirm the exact JUnit XML shape produced by the pinned `taiki-e/install-action`
  nextest version, including `testsuite.name`, `testcase.classname`,
  `testcase.name`, skipped tests, failed tests, and whether package names are
  present.
- Measure a branch with only `proof_size_comparison.rs` deleted to separate the
  safe prune savings from any future schedule-drift coverage changes.
- Time `generated_tables` under non-zk and all-features separately after
  telemetry lands. If the zk run remains a large outlier, design a focused
  `akita-config` zk drift job before excluding it from the workspace pass.
- Dry-run the trusted `workflow_run` renderer after the workflow is present on
  `main`, because a newly added `workflow_run` file usually cannot self-test
  fully on the PR that introduces it.
- Verify `github.rest.checks.create` behavior on the repo with `checks: write`,
  especially whether repeated threshold crossings should create one check per run
  or update the newest check with the same name.
- Build script fixtures from real artifacts for run `26798445379` and
  `26796574529` so baseline matching and rendered deltas are checked against
  observed CI data, not synthetic-only XML.

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
