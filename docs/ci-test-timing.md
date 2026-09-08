# CI test timing

Design spec: [`specs/archive/2026-Q3/ci-test-timing.md`](../specs/archive/2026-Q3/ci-test-timing.md).

Every PR gets an upserted timing comment (marker `<!-- akita-ci-test-timing -->`) showing run wall time vs a main baseline and per-test outliers from the nextest JUnit output.

## How CI runs tests

- **`test`** runs the workspace nextest merge gate (`--profile ci --cargo-profile ci-test`, features `parallel,disk-persistence`), sharded across matrix jobs (`slice:index/total`).
- **Generic CI and Jolt smoke jobs** load tracked external schedule artifacts at runtime.
- **`test-schedule-artifact-drift`** is the only CI job that regenerates schedules. It writes every family artifact to a temporary directory and rejects any byte difference from `artifacts/schedules/`.
- **`test-timing`** merges shard JUnit into `summary.json` (schema v2, single pass `ci`) and uploads artifact `ci-test-timing-data`.

## Local repro

```bash
cargo nextest run --profile ci --cargo-profile ci-test --no-default-features --features parallel,disk-persistence
python3 -m unittest discover -s scripts/tests -p "test_ci_test_timing_report.py"
```

For timing artifact layout and renderer trust boundary, see the spec linked above.
