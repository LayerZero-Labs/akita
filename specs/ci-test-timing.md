# Spec: CI Test Timing Telemetry

| Field | Value |
|-------|-------|
| Author(s) | Quang Dao |
| Created | 2026-06-02 |
| Status | implemented in PR #218, v2 single-pass cutover |
| Branch | `quang/zk-strip-impl` |
| PR | [#218](https://github.com/LayerZero-Labs/akita/pull/218) |

## Summary

CI uploads a machine-readable test timing artifact for every PR and `main` push.
The artifact is rendered into one upserted PR comment with marker
`<!-- akita-ci-test-timing -->`. The comment shows run wall time, main-baseline
comparison, slowest tests, per-test regressions, and new slow tests.

After the ZK strip milestone in PR #218, Akita no longer has a live
`all-features` / `zk` CI test pass. Timing schema v2 therefore records one real
pass:

| Field | Value |
|-------|-------|
| Pass key | `ci` |
| Nextest profile | `ci` |
| Artifact | `ci-test-timing-data` |
| Schema | v2, `pass_layout = "single"` |
| Baseline bridge | v2 `ci` may compare against older v1 `non-zk` main baselines |

The old v1 dual-pass shape remains renderable only as historical artifact input.
It should be deleted after the repository has had a full artifact-retention
window of v2 `main` baselines.

## Intent

### Goal

Make test-time regressions visible in every PR the same way profile benchmarks
surface prove/verify regressions: structured data, stable baselines, one
upserted comment per PR.

### Invariants

- **Timing telemetry must not change test selection.** It observes the canonical
  CI nextest run; it does not skip, filter, or reorder tests.
- **The CI merge gate is the transparent pass.** `test` runs
  `cargo nextest run --profile ci --cargo-profile ci-test --no-default-features --features parallel,disk-persistence`
  sharded across matrix jobs.
- **Schedule drift remains explicit.** `test-all-schedules-drift` runs the
  `all-schedules` drift guard outside the timing artifact.
- **Trusted renderer boundary stays narrow.** The `workflow_run` comment job may
  read artifacts and post comments, but it must not execute PR-controlled code or
  post pre-rendered Markdown from a PR artifact. The authoritative PR comment is
  rendered from trusted default-branch code over untrusted structured artifact
  data.
- **Profile bench comment stays separate.** Test timing uses its own marker and
  artifact name; do not merge it with `<!-- akita-profile-bench-report -->`.

### Non-Goals

- Blocking CI on wall-time regression. Hosted-runner jitter is too noisy.
- Tracking clippy, fmt, doc, fuzz, or benchmark job times.
- Historical charts or external dashboards.
- Completing or preserving the `zk` feature on `main`. ZK work is preserved on
  `zk-wip` and is expected to return later through a new design.
- Keeping the v1 dual-pass renderer forever.

## Evaluation

### Acceptance Criteria

- [x] `.config/nextest.toml` defines `[profile.ci.junit]` with fixed JUnit path
  `target/nextest/ci/junit.xml`.
- [x] `.github/workflows/ci.yml` runs the `test` job as a sharded single pass
  with nextest profile `ci`.
- [x] Each shard uploads a shard artifact containing `junit-shard-N.xml` when
  available and `timing-shard-N.json` with start/end epoch, exit code, shard
  index, and shard total.
- [x] The `test-timing` job merges shard artifacts with
  `scripts/ci_test_timing_report.py prepare-shards`.
- [x] The timing artifact `ci-test-timing-data` contains `summary.json` and the
  debug renders `comment.md` / `report.md`.
- [x] `summary.json` uses schema v2 for the single pass and includes
  `pass_layout`, `pass_order`, `passes_sharded`, and `shard_count`.
- [x] `scripts/ci_test_timing_report.py render` can compare current v2 `ci`
  summaries against older v1 `non-zk` main baselines and shows a layout-mismatch
  banner while that bridge is active.
- [x] `.github/workflows/test-timing-comment.yml` upserts the PR timing comment
  from trusted default-branch code.
- [x] Renderer fixture tests cover single-pass merge/render, v1 baseline bridge,
  timing JSON input, shard merge, nested artifacts, and missing-shard failure.
- [x] The local documentation page points at this spec and the current local
  repro commands.

### Testing Strategy

Local script tests:

```bash
python3 -m unittest discover -s scripts/tests -p "test_ci_test_timing_report.py"
```

Local nextest repro for the timed pass:

```bash
cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence
```

CI should additionally verify:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```

### Performance

The v2 timing pipeline should add only small Python and artifact overhead. Wall
time is dominated by the sharded nextest run and the separate schedule drift job.
The timing comment is advisory; it should make slow tests visible without
turning transient hosted-runner variance into a merge blocker.

## Design

### Architecture

```text
.github/workflows/ci.yml
  test (matrix shards)
    cargo nextest run --profile ci --partition slice:N/T
    copy target/nextest/ci/junit.xml to junit-shard-N.xml
    write timing-shard-N.json
    upload ci-test-pass-shard-N

  test-timing
    download ci-test-pass-shard-*
    ci_test_timing_report.py prepare-shards
    ci_test_timing_report.py merge --pass ci --profile ci --passes-sharded
    ci_test_timing_report.py render --compact
    upload ci-test-timing-data

.github/workflows/test-timing-comment.yml
  workflow_run on CI completion
    resolve PR
    checkout trusted default branch
    download current ci-test-timing-data
    download optional main and previous-run baselines
    render comment from trusted script
    upsert marker comment
```

The PR run may include a debug `comment.md` in the artifact for step summaries
and diagnosis. The privileged `workflow_run` job must not post that file. It
always renders a fresh comment from `summary.json`.

### Current CI Layout

`test` is the merge-gate nextest job:

```bash
cargo nextest run --profile ci --cargo-profile ci-test \
  --no-default-features --features parallel,disk-persistence \
  --partition "slice:${SHARD_INDEX}/${SHARD_TOTAL}"
```

`test-all-schedules-drift` is separate from the timing artifact:

```bash
cargo test -p akita-config --features all-schedules generated_schedule_tables_match_find_schedule
cargo test -p akita-config --no-default-features --test schedule_catalog_feature_off
cargo test -p akita-config --features schedules-fp128-d64-onehot,schedules-fp128-d64-full --test schedule_catalog_miswire
```

`test-timing` should run even when `test` fails so the PR still receives a
failure or partial-data timing artifact. Its final status must fail if the test
job failed or if the timing merge/render itself failed.

### `summary.json` Schema v2

```json
{
  "schema_version": 2,
  "source_sha": "3735fc4f...",
  "source_branch": "quang/zk-strip-impl",
  "workflow_run_id": 123,
  "generated_at": "2026-06-26T00:00:00Z",
  "pass_layout": "single",
  "pass_order": ["ci"],
  "passes_sharded": true,
  "shard_count": 2,
  "passes": {
    "ci": {
      "profile": "ci",
      "started_at_epoch": 1780370000,
      "finished_at_epoch": 1780370672,
      "wall_s": 672.0,
      "exit_code": 0,
      "test_count": 812,
      "skipped": 2,
      "failed": 0,
      "missing_junit": false,
      "tests": [
        {
          "id": "generated_tables::generated_schedule_tables_match_find_schedule",
          "package": "",
          "crate": "",
          "binary": "generated_tables",
          "test": "generated_schedule_tables_match_find_schedule",
          "classname": "generated_tables",
          "duration_s": 491.2,
          "status": "passed"
        }
      ]
    }
  }
}
```

`tests` is sorted by `duration_s` descending at merge time. The full list is
stored; the renderer caps rows for comments. Test id format is
`{binary}::{test_name}`. If two JUnit records collide on id, append a stable
`#{n}` suffix and retain the raw fields so output stays deterministic.

### Historical v1 Compatibility

Schema v1 contained two passes, `non-zk` and `all-features`. During the cutover,
the renderer maps current `ci` to older `non-zk` baselines:

```text
current v2 pass: ci
baseline candidates: non-zk, ci
```

When current and main baseline pass layouts differ, the PR comment shows a
baseline-layout mismatch banner and compares only against the matched historical
`non-zk` pass. The `all-features` row is intentionally ignored because there is
no current ZK pass to compare against.

### PR Comment Layout

Marker: `<!-- akita-ci-test-timing -->`

Sections:

1. **Run summary** for single-pass v2, or **Pass summary** for historical
   multi-pass input.
2. **Slowest tests** for the current run.
3. **Regressions vs main** where `delta_s >= max(5s, 10% of baseline)`.
4. **New slow tests** in the current run and not in the main baseline where
   `duration_s >= 30`.

The compact render used in the CI step summary omits the regression sections and
limits slow-test output.

All artifact-derived strings must be escaped before they enter Markdown or HTML.
This includes test ids, JUnit class names, branch names, baseline labels, and
workflow links.

### Baseline Resolution

`test-timing-comment.yml` resolves baselines in the trusted workflow:

| Baseline | Source |
|----------|--------|
| Main | Latest successful default-branch `CI` push run with `ci-test-timing-data` |
| Previous PR | Previous completed PR `CI` run on the same head branch with artifact |

On the first PR containing renderer changes, the trusted workflow still uses the
old script from `main`, so comments may remain in the old shape until the PR
merges. This is expected and keeps the trust boundary intact. After merge, the
first green `main` push uploads a v2 baseline; until then PR comments may show
the v1 mismatch banner.

### Script Commands

| Command | Purpose |
|---------|---------|
| `prepare-shards` | Combine sharded JUnit and timing files into one JUnit file and one timing JSON |
| `prepare-pass` | Deprecated alias for `prepare-shards` |
| `merge` | Read JUnit plus pass metadata and write `summary.json` |
| `render` | Read current plus optional baseline dirs and write `comment.md` / `report.md` |
| `failure-summary` | Write an explanatory failed summary when the timing artifact is missing |
| `combine-junit` | Merge JUnit files for local/debug use |
| `read-timing` | Print `started_at finished_at exit_code` from timing JSON |

## Follow-Up

- After one artifact retention window of v2 `main` baselines, remove the v1
  dual-pass render branches and the `ci` -> `non-zk` baseline alias.
- Rename historical fixtures such as `sample-non-zk.xml` only if they become
  confusing during future test maintenance.
- When ZK returns, add a new timing schema deliberately. Do not resurrect the old
  `all-features` row as a compatibility shim.

## References

- [docs/ci-test-timing.md](../docs/ci-test-timing.md)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml)
- [.github/workflows/test-timing-comment.yml](../.github/workflows/test-timing-comment.yml)
- [scripts/ci_test_timing_report.py](../scripts/ci_test_timing_report.py)
- [scripts/tests/test_ci_test_timing_report.py](../scripts/tests/test_ci_test_timing_report.py)
