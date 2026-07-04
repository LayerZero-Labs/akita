# CI test timing

Design spec: [`specs/ci-test-timing.md`](../specs/ci-test-timing.md).

Every PR gets an upserted timing comment (marker `<!-- akita-ci-test-timing -->`) showing run wall time vs a main baseline and per-test outliers from the nextest JUnit output.

## How CI runs tests

- **`test`** runs the workspace nextest merge gate (`--profile ci --cargo-profile ci-test`, features `parallel,disk-persistence`), sharded across matrix jobs (`slice:index/total`).
- **`test-all-schedules-drift`** runs schedule drift outside the timing artifact (`cargo test -p akita-config --features all-schedules …`).
- **`test-timing`** merges shard JUnit into `summary.json` (schema v2, single pass `ci`) and uploads artifact `ci-test-timing-data`.

## Local repro

```bash
cargo nextest run --profile ci --cargo-profile ci-test --no-default-features --features parallel,disk-persistence
python3 -m unittest discover -s scripts/tests -p "test_ci_test_timing_report.py"
```

For timing artifact layout and renderer trust boundary, see the spec linked above.
