# Spec: Profile Bench Coverage Matrix

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-26                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Extend the profile benchmark CI from a small fp128/fp32 sample into a compact cross-prime coverage matrix that times one-hot and dense workloads for fp16, fp32, fp64, and fp128. The workflow should reduce samples from 5 to 3, keep the existing fp128 batched one-hot coverage, fix the current dense fp64 profile failure, and replace the long per-case PR comment with a matrix-first report that remains readable as coverage grows.

## Intent

### Goal

Build a 9-case profile benchmark CI matrix covering fp16, fp32, fp64, and fp128 singleton one-hot and dense workloads, plus the existing fp128 batched one-hot workload, with compact markdown and machine-readable artifact outputs.

The benchmark matrix is:

| Field family | Workload | Variables | Polys | Notes |
| --- | --- | ---: | ---: | --- |
| fp16 | one-hot | 32 | 1 | Fixed generated small-field schedule. |
| fp16 | dense | 26 | 1 | Fixed generated small-field schedule. |
| fp32 | one-hot | 32 | 1 | Fixed generated small-field schedule. |
| fp32 | dense | 26 | 1 | Fixed generated small-field schedule. |
| fp64 | one-hot | 32 | 1 | Fixed generated small-field schedule. |
| fp64 | dense | 26 | 1 | Must complete without the current dense profile size panic. |
| fp128 | one-hot | 32 | 1 | Existing adaptive fp128 one-hot profile behavior. |
| fp128 | dense | 26 | 1 | Existing adaptive fp128 full/dense profile behavior, represented as dense in reports. |
| fp128 | one-hot batched | 30 | 4 | Preserve current same-point batched one-hot coverage. |

The implementation affects benchmark infrastructure only:

- `.github/workflows/profile-bench.yml`
- `scripts/profile_bench_report.py`
- `crates/akita-pcs/examples/profile/modes.rs`
- `crates/akita-pcs/examples/profile/workload.rs`
- `crates/akita-pcs/examples/profile/report.rs`

### Invariants

1. Benchmark coverage is explicit and reproducible. The CI workflow must list the benchmark cases directly or derive them from one checked-in matrix definition; it must not hide case expansion in ad hoc shell logic.
2. Each successful case emits setup, commit, prove, verify, proof-size, proof-accounting, proof-level, field-role, tail-shape, and RSS metrics. Missing required metrics remain a benchmark failure.
3. Proof-size regression checks stay case-local. A case compares only to a baseline with the same stable case id; absent baseline cases are skipped rather than compared across fields or modes.
4. The benchmark does not change proof, transcript, serialization, setup, schedule selection, or verifier semantics. Any code touched to fix dense fp64 must preserve existing prover/verifier consistency and pass the normal profile verification path.
5. Dense fp64 at `nv=26` must produce and verify a proof. It must not panic at `crates/akita-pcs/examples/profile/workload.rs` during `batched_prove`.
6. PR comments stay readable for the full matrix. Detailed per-level tables must remain available in artifacts, but they should not dominate the default PR comment.
7. The profile report has stable machine-readable output. `summary.json` remains the canonical artifact, and a flat `summary.csv` or equivalent tabular artifact must be emitted for spreadsheet-friendly inspection.
8. Benchmark mode naming in user-facing output must distinguish field family and workload. Dense workloads should be displayed as dense even when existing internal fp128 names use "full".

### Non-Goals

- No protocol optimization, proof-size tuning, schedule-table regeneration, or security-parameter change is part of this spec.
- No new Criterion benches are required.
- No compatibility aliases should be added solely to preserve old benchmark mode names. If mode names are changed, update all checked-in call sites and documentation in one pass.
- No hard wall-clock regression gate is introduced in this PR. The workflow may report elapsed time, but proof-size remains the only enforced benchmark regression threshold.
- No attempt is made to make hosted GitHub runner timings perfectly stable. The matrix is for trend visibility and cross-prime smoke coverage, not precise microbenchmarking.

## Evaluation

### Acceptance Criteria

- [ ] `.github/workflows/profile-bench.yml` sets `AKITA_BENCH_RUNS` to `3`.
- [ ] The profile benchmark workflow runs exactly the 9 matrix cases listed in this spec.
- [ ] `dense fp64 nv26` completes setup, commit, prove, verify, proof summary, and proof accounting without panicking.
- [ ] Every case has a stable case id containing field family, workload, variable count, and polynomial count.
- [ ] The rendered PR comment contains a compact matrix summary with one row per case and columns for status, setup, commit, prove, verify, max RSS, proof size, and baseline deltas when available.
- [ ] Per-level schedule and proof-size breakdowns remain available in uploaded artifacts for every successful case.
- [ ] `summary.json` preserves all existing fields needed by the proof-size threshold check.
- [ ] A flat tabular artifact, preferably `summary.csv`, is uploaded with one row per case.
- [ ] The proof-size regression threshold still compares matching case ids against the main baseline and ignores cases missing from older baselines.
- [ ] The profile report handles partial failures by naming the failing case and phase clearly in the generated artifact.

### Testing Strategy

Existing checks that should remain green:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`

Focused implementation checks:

- `cargo build --release --quiet --example profile`
- `python3 scripts/profile_bench_report.py run --binary ./target/release/examples/profile --output-dir <tmpdir> --runs 1` with the 9 target cases, or an explicitly documented subset plus a full GitHub Actions run.
- `python3 scripts/profile_bench_report.py render <tmpdir>/summary.json` to verify the compact report shape.
- A synthetic baseline render check with at least one missing baseline case, proving new cases do not break comparison against older artifacts.
- A parser/render fixture or unit-style script check proving that failed cases are represented with case id and phase instead of disappearing from the output.

The dense fp64 fix should include either a focused regression test for the shape that panics today or an implementation note explaining why the release profile run is the correct regression check. The failure to guard against is:

```text
dense_fp64_* nv26: InvalidSize { expected: 16777216, actual: 33554432 }
```

### Performance

The current PR #104 benchmark run took about 11 minutes end to end, with about 7 minutes in release build and about 3 minutes 20 seconds in benchmark execution for 3 cases x 5 samples.

Reducing to 3 samples makes the new 9-case matrix expected to land around 25-35 minutes end to end on GitHub-hosted Ubuntu runners once dense fp64 is fixed. This is acceptable for the profile benchmark workflow because it runs as a dedicated benchmark CI job and because the broader matrix gives useful cross-prime regression visibility.

Performance expectations:

- The workflow should not introduce additional release builds per case.
- The benchmark script should keep running cases sequentially in one job unless memory measurements or runner stability require splitting later.
- The default PR comment should summarize medians over 3 runs and show sample ranges only in the detailed artifact.
- The implementation PR should report the actual GitHub Actions runtime for the first successful full-matrix run.

## Design

### Architecture

`profile-bench.yml` remains the workflow orchestrator. It builds `target/release/examples/profile` once, then calls `scripts/profile_bench_report.py run` with the configured case list.

`scripts/profile_bench_report.py` should own the matrix representation and rendering:

- Parse the existing `MODE:NUM_VARS:NUM_POLYS` case form.
- Normalize each case into display fields: `field_family`, `workload`, `num_vars`, `num_polys`, and `config`.
- Preserve raw profile mode in `summary.json` for reproducibility.
- Emit `summary.json`, `summary.csv`, a compact `comment.md`, and a fuller `report.md`.
- Render the PR comment as a single matrix table first, followed by short notes about baselines, samples, and artifacts.
- Move verbose per-level tables to `report.md` or per-case markdown artifacts, linked or named from the comment.

`crates/akita-pcs/examples/profile/modes.rs` owns profile mode dispatch. The implementation should make dense/small-field naming consistent enough that report generation does not need fragile special cases. If the code keeps internal names such as `full_fp16_d64`, the report must still present them as dense workloads. If names are changed, perform a full cutover of checked-in mode lists and workflow references.

`crates/akita-pcs/examples/profile/workload.rs` owns the dense profile proof path. The dense fp64 panic should be fixed at the root cause of the shape mismatch, not by skipping verification, lowering `nv`, adding a special failure waiver, or excluding fp64 dense from the matrix.

### Alternatives Considered

1. Keep 5 samples.
   Rejected for now because the 9-case matrix would unnecessarily lengthen every profile benchmark PR run. Three samples preserve median reporting while keeping workflow time reasonable.

2. Run exhaustive coverage only on a nightly or manual workflow.
   Rejected for the first cut because the matrix is small enough to run in the existing dedicated benchmark workflow, and PR visibility is useful while fp16/fp32/fp64 support is still moving.

3. Keep the current long per-case markdown sections in the PR comment.
   Rejected because 9 cases would make the comment hard to scan. The detailed data should remain, but the comment should lead with the comparison matrix.

4. Drop dense fp64 until it is optimized.
   Rejected because the point of this matrix is cross-prime coverage. A current panic is a reason to fix the profile path before expanding CI, not a reason to omit the cell.

5. Add compatibility aliases for old mode names.
   Rejected under the repo's no-backward-compatibility policy. Benchmark mode names are internal developer tooling; checked-in references should be cut over directly.

## Documentation

Update benchmark-facing documentation where appropriate:

- `README.md` or a profile-specific README section if one exists for CI benchmark expectations.
- The PR body for the implementation should include the final matrix, run-count change, first full-matrix CI runtime, and any known runner variance.
- If dense/full naming is changed, document the new names in the profile example help/error output or nearby source comments.

No paper, protocol, serialization, transcript, or verifier documentation changes are required because this spec changes benchmark coverage and reporting only.

## Execution

Suggested implementation order:

1. Fix or normalize profile mode names and labels so field family and workload are available in summary data.
2. Fix the dense fp64 `nv26` profile failure and verify it through the normal prove/verify path.
3. Update the benchmark matrix and reduce `AKITA_BENCH_RUNS` from 5 to 3.
4. Extend `profile_bench_report.py` to emit compact matrix markdown and `summary.csv`.
5. Keep detailed per-level reports in artifacts rather than the default PR comment.
6. Re-run focused local release profile checks.
7. Let GitHub Actions produce the first full 9-case, 3-sample run and record the elapsed time in the implementation PR.

Risks to resolve first:

- The dense fp64 panic may indicate an extension-field dense opening shape bug rather than a reporting issue.
- Renaming `full` to `dense` can accidentally break mode dispatch if all checked-in call sites are not updated together.
- The proof-size baseline for newly added cases will be absent until main has a successful artifact with the new matrix; the threshold logic must continue to skip missing baseline cases.

## References

- `specs/TEMPLATE.md`
- `specs/SPEC_REVIEW.md`
- `CONTRIBUTING.md`
- `.github/workflows/profile-bench.yml`
- `scripts/profile_bench_report.py`
- `crates/akita-pcs/examples/profile/modes.rs`
- `crates/akita-pcs/examples/profile/workload.rs`
- `crates/akita-pcs/examples/profile/report.rs`
- PR #104 benchmark comment: `https://github.com/LayerZero-Labs/akita/pull/104#issuecomment-4527174043`
- PR #104 benchmark run: `https://github.com/LayerZero-Labs/akita/actions/runs/26428943234`
