# CI test timing

Design spec: [`specs/archive/2026-Q3/ci-test-timing.md`](../specs/archive/2026-Q3/ci-test-timing.md).

Every pull request run of the workspace test pass gets an upserted timing comment (marker `<!-- akita-ci-test-timing -->`) showing run wall time vs a main baseline and per-test outliers from the nextest JUnit output. A pull request runs the pass only when it carries the `run-full-tests` label. The merge queue runs it before every merge into `main`.

## How CI runs tests

- **`test`** in `.github/workflows/full-tests.yml` runs the workspace nextest merge gate (`--profile ci --cargo-profile ci-test`). One job on the large runner compiles and runs the whole suite: compilation dominates the job, so shards would each repeat it. The runner's CPU also takes the AVX-512 paths that the kernels select at runtime. `Full tests gate` is the required status check.
- **Generic CI and Jolt smoke jobs** load tracked external schedule artifacts at runtime.
- **`test-schedule-artifact-drift`** is the only CI job that regenerates schedules. It writes every family artifact to a temporary directory and rejects any byte difference from `artifacts/schedules/`.
- **`test-timing`** merges the JUnit output into `summary.json` (schema v2, single pass `ci`) and uploads artifact `ci-test-timing-data`. The comment compares against the latest successful merge queue run into `main`.

## Local repro

Copy the `Run tests` invocation from `.github/workflows/full-tests.yml`, then check the report scripts:

```bash
python3 -m unittest discover -s scripts/tests -p "test_ci_test_timing_report.py"
```

For timing artifact layout and renderer trust boundary, see the spec linked above.
