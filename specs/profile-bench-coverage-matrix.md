# Spec: Profile Bench Coverage Matrix

| Field       | Value                                                  |
|-------------|--------------------------------------------------------|
| Author(s)   | Quang Dao                                             |
| Created     | 2026-05-26                                            |
| Status      | implemented, with long hosted-runner cells deferred   |
| PR          | https://github.com/LayerZero-Labs/akita/pull/107      |

> **Status note (2026-06-03, PR #146).** The committed-fold A-role reprice in
> [`specs/weak-binding-norm-fix.md`](weak-binding-norm-fix.md) made the small-D
> families non-securable (fp16 entirely; fp32/fp64 at D32/D64), so the **active**
> benchmark matrix was re-pointed at securable D128 profiles for the small prime
> fields. A later follow-up re-pointed the **fp128** cells to D64 after measuring
> that D64 is the fp128 proof-size optimum (~20% smaller than D128 for both
> dense and one-hot, while still folding securely); the small-field fp32/fp64
> cells remain at D128 because their D64 is non-securable. That matrix was later
> replaced by the active matrix below. Everything else (this Summary,
> and everything from "## Evaluation" onward: Acceptance Criteria, Validation,
> Performance, Design, Alternatives, Follow-Up) is the original **PR #107
> historical record**. Its fp16 / D32 / D64 cell references (e.g.
> `onehot_fp16_d32`, `dense_fp64_d32`, `dense_fp128_d32`, the "fp128 D32" report
> wording) describe the pre-reprice matrix and are superseded by the Active
> Benchmark Matrix; they are retained as PR #107's completed acceptance record,
> not the current shipping configuration.

> **Status update (2026-08-09, PR #355).** The active direct fp128 modes now use
> adaptive generated catalogs. The dense benchmark check runs fp32 D128, fp64
> D128, and adaptive fp128 at `nv=26`. For fp128 dense at `nv=26`, the root
> A/B/D dimensions are 256/64/64. Every recursive fold and the terminal use
> D64. The historical record below still describes the earlier uniform D64
> benchmark.

## Summary

This PR widens the profile benchmark workflow from a small fp128/fp32 sample
into a 7-case active D32 matrix across fp16, fp32, fp64, and fp128, reduces
samples from 5 to 3, and keeps the existing fp128 same-point batched one-hot
coverage. Two intended hosted-runner cells are documented but deferred:
`onehot_fp16_d32:32:1` is currently too expensive for this PR's active CI
budget, while `dense_fp64_d32:25:1` is kept as the next dense fp64 target but
is not re-enabled yet.
The workflow intentionally replaces the old adaptive fp128 profile selectors
with explicit D32 cases; the benchmark path does not choose D at runtime.

The PR also fully cuts over profile mode names and benchmark labels, adds
matrix-first benchmark reports with machine-readable CSV output, preserves
detailed per-level schedule/proof-size artifacts, hardens partial-failure and
missing-summary reporting, and slims regular debug tests that duplicated
benchmark-sized proof work. This PR does not claim hosted-runner support for
the currently long benchmark configurations; it records them as follow-up work
instead of making every PR update pay their cost.

## Intent

### Active Benchmark Matrix

The checked-in workflow currently runs:

| Mode | Field | Workload | Variables | Polys | Config | Setup mode | Notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `dense_fp32` | fp32 | dense | 26 | 1 | adaptive D128/D256 | `direct` | Catalog-selected adaptive fp32 dense schedule. |
| `dense_fp64` | fp64 | dense | 26 | 1 | adaptive D64/D128/D256 | `direct` | Catalog-selected adaptive fp64 dense schedule. |
| `dense_fp128` | fp128 | dense | 26 | 1 | adaptive | `direct` | Root A/B/D = 256/64/64; recursive folds and terminal use D64. |
| `onehot_fp32` | fp32 | 1-of-256 one-hot | 28 | 1 | adaptive D128/D256 | `direct` | Adaptive fp32 one-hot under honest pricing. Capped at nv=28: the ext-degree-4 challenge schedule keeps a large un-folded witness, so at nv>=30 the prover's eq-evaluation table exceeds the 1 GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` ceiling. |
| `onehot_fp64` | fp64 | 1-of-256 one-hot | 28 | 1 | adaptive D64/D128/D256 | `direct` | Adaptive fp64 one-hot under honest pricing. Capped at nv=28 for the same eq-table-budget reason as the fp32 cell. |
| `onehot_fp128` | fp128 | 1-of-256 one-hot | 32 | 1 | adaptive | `direct` | Canonical adaptive fp128 one-hot catalog. |
| `onehot_fp128_multi_group` | fp128 | 1-of-256 one-hot batched multi-group | 32 | 4 | adaptive | `direct` | Direct multi-group coverage using the canonical adaptive fp128 one-hot catalog. |
| `onehot_fp128_multi_group_recursive` | fp128 | 1-of-256 one-hot batched multi-group | 32 | 4 | adaptive recursive multi-group | `recursive` | Recursive setup-product coverage using the adaptive recursive companion catalog. |
| `onehot_fp128_multi_group_recursive_multi_chunk_w8r2` | fp128 | 1-of-256 one-hot batched multi-group W8R2 | 32 | 4 | adaptive recursive multi-group W8R2 | `recursive` | Distributed recursive setup-offload row: `8` chunks, `2` leading levels, with adaptive role dimensions. |
| `onehot_fp128_multi_chunk_w2r2` | fp128 | 1-of-256 one-hot distributed chunked relation | 32 | 1 | adaptive multi-chunk W2R2 | `direct` | `2` chunks, `2` leading levels. |
| `onehot_fp128_multi_chunk_w4r2` | fp128 | 1-of-256 one-hot distributed chunked relation | 32 | 1 | adaptive multi-chunk W4R2 | `direct` | `4` chunks, `2` leading levels. |
| `onehot_fp128_multi_chunk_w8r2` | fp128 | 1-of-256 one-hot distributed chunked relation | 32 | 1 | adaptive multi-chunk W8R2 | `direct` | Production direct distributed preset (`8` chunks, `2` leading levels). |

Every active cell folds securely under honest committed-fold A-role pricing.
The ring degree differs by field, for two distinct reasons:

- **Small prime fields (fp32/fp64):** each production preset searches its
  catalog-bound adaptive domain and replays the selected A/B/D tuple at runtime.
  fp32 uses D128/D256 with a D128 suffix. fp64 uses D64/D128/D256 with a D64
  suffix.
- **fp128:** the canonical direct dense and one-hot presets use their adaptive
  generated catalogs. Each scheduled A/B/D dimension is selected offline from
  the audited domain, and runtime code resolves that single canonical row.
  Runtime code does not compare candidate families. It loads the one generated
  row for the requested profile.

The cost figures after this section belong to the historical PR #107 record.
They do not describe the active workflow.

### Scope

The benchmark/reporting changes touch:

- `.github/workflows/profile-bench.yml`
- `scripts/profile_bench_report.py`
- `crates/akita-pcs/examples/profile/main.rs`
- `crates/akita-pcs/examples/profile/modes.rs`
- `crates/akita-pcs/examples/profile/workload.rs`
- `crates/akita-pcs/examples/profile/report.rs`
- `AGENTS.md`

The test-coverage cleanup touches:

- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `crates/akita-pcs/tests/batched_aggregated_e2e.rs`
- `crates/akita-pcs/tests/setup.rs`

### Invariants

1. Benchmark modes are fully cut over to explicit names. There are no
   compatibility aliases for old bare names such as `onehot`, `full`,
   `full_d32`, `onehot_d32`, or `full_fp16_d32`.
2. The benchmark path is pinned to explicit per-field D mode names, not adaptive
   D selection. `AKITA_BENCH_MODE`, `AKITA_BENCH_CASES`, and the default profile
   mode all spell out the selected D value.
3. Benchmark-facing labels expose field family, workload, and ring dimension.
   fp128 rows say `*_fp128_d128`; one-hot rows say `1-of-256 one-hot`.
4. Case IDs are semantic and stable for new artifacts:
   `{field}-{workload[-batched]}-nv{num_vars}-np{num_polys}-d{D}` for direct
   setup mode, with `-setup-recursive` appended for recursive setup mode.
   Loaded summaries are normalized from `(mode, nv, np, setup_mode)` using the
   new naming scheme. This intentionally does not preserve or compare legacy
   IDs.
5. Each successful case must emit setup, commit, prove, verify, proof-size,
   proof-accounting, proof-level, planned-level, field-role, tail-encoding, and
   RSS metrics. Missing required metrics turn that case into a benchmark
   failure.
6. The dense D32 runtime fallback path must still emit schedule summaries and
   proof-size accounting from the actual runtime `Schedule`, even when there is
   no generated `AkitaSchedulePlan`.
7. The benchmark runner keeps later cases after an earlier case fails, writes
   one row per attempted case to `summary.json` and `summary.csv`, and returns
   a failing exit status if any case failed.
8. If the benchmark step fails before writing `summary.json`, the render step
   writes a synthetic failed summary for the configured matrix and routes it
   through the same full and compact renderers.
9. Proof-size regression enforcement compares only matching semantic case IDs,
   skips failed current cases, and skips cases missing from older baselines.
10. GitHub API conveniences for baseline lookup and PR comment upsert must not
   erase benchmark artifacts. API failures are warnings; artifact upload and
   local rendering still proceed.
11. Regular debug tests must not duplicate the full fp128 batched one-hot
    `nv30 x np4` benchmark proof. That final-witness bound is covered by a
    schedule-level test, while recursive-suffix truncation rejection remains
    covered by a smaller E2E fixture.
12. Slimmed setup and aggregated E2E tests must keep non-vacuous folded-proof
    coverage through explicit `!proof.is_root_direct()` assertions.

### Non-Goals

- No protocol optimization, schedule-table regeneration, proof-size tuning, or
  security-parameter change is part of this PR.
- No new Criterion benches are required.
- No hard wall-clock regression gate is introduced. The workflow reports
  timing and memory, but proof size remains the only enforced benchmark
  regression threshold.
- No hosted-runner timing stability guarantee is attempted. The matrix is for
  trend visibility and cross-prime smoke coverage, not precise microbenchmarking.
- No profile-only workaround is added for deferred hosted-runner cells. The
  `onehot_fp16_d32:32:1` cell remains blocked on performance work, and
  `dense_fp64_d32:25:1` remains documented but inactive until a separate
  re-enable pass.

## Evaluation

> **PR #107 historical record below.** The acceptance criteria, validation, and
> design notes that follow document what PR #107 shipped and tested (the D32 /
> fp16 era matrix). They are superseded for the active matrix by the PR #146
> re-point (see the top status note and "Active Benchmark Matrix" above); fp16 /
> D32 / D64 cell mentions here are historical, not current targets.

### Acceptance Criteria

- [x] `.github/workflows/profile-bench.yml` sets `AKITA_BENCH_RUNS` to `3`.
- [x] The active workflow lists the currently supported hosted-runner matrix
      cases.
- [x] The known long hosted-runner offender `onehot_fp16_d32:32:1` is
      documented as deferred rather than active.
- [x] `dense_fp128_d64` remains active at `nv=24`, not the earlier `nv=26`
      hosted-runner offender size.
- [ ] `dense_fp64_d32:25:1` is re-enabled after a separate validation pass
      and completes setup, commit, prove, verify, proof summary, and proof
      accounting.
- [x] Every new case has a semantic case ID containing field family, workload,
      variable count, polynomial count, and D config.
- [x] Old benchmark mode names and checked-in call sites are fully cut over to
      explicit field/workload/D names.
- [x] fp128 report labels use explicit `fp128 D32` wording instead of
      `adaptive`.
- [x] One-hot report labels describe the `1-of-256` sparsity.
- [x] The default PR comment is a compact matrix with status, case, mode,
      setup, commit, prove, verify, max RSS, proof size, and baseline deltas
      when baselines are available.
- [x] The uploaded `report.md` artifact keeps detailed per-case schedule,
      proof-size, and sample-range sections.
- [x] `summary.json` remains the canonical artifact for threshold checks.
- [x] `summary.csv` is emitted with one row per case for spreadsheet-friendly
      inspection.
- [x] Failed cases stay visible in `summary.json`, `summary.csv`, the compact
      comment, and the full report with a failing phase and error message.
- [x] Missing `summary.json` is converted into a structured synthetic failure
      report instead of a raw one-line fallback.
- [x] Missing `exit_code` defaults consistently to success in both aggregation
      and display paths.
- [x] Proof-size threshold checks compare matching semantic IDs, skip missing
      baselines, and skip failed current cases.
- [x] Baseline lookup and PR comment upsert API failures are warnings rather
      than artifact/report blockers.
- [x] `akita-pcs::akita_e2e` no longer runs the full
      `batched_onehot_4x30_keeps_folding_past_oversized_tail` proof.
- [x] The fp128 batched one-hot `nv30 x np4` final-witness bound is covered by
      `batched_onehot_4x30_plan_keeps_terminal_witness_bounded`.
- [x] Recursive-suffix truncation rejection remains covered by the smaller
      `batched_onehot_same_point_round_trip` E2E fixture.
- [x] `setup.rs` uses `POLY_NV=18` and asserts folded proof coverage for the
      successful setup-capacity paths.
- [x] `batched_aggregated_e2e.rs` keeps singleton, irregular one-hot, dense,
      and mixed aggregation coverage while shrinking the heaviest dense/mixed
      shapes and asserting folded proof coverage on nontrivial cases.

### Validation Performed

Local checks performed during this PR:

- `git diff --check`
- `cargo fmt -q --check`
- `python3 -m py_compile scripts/profile_bench_report.py`
- workflow YAML parse for `.github/workflows/profile-bench.yml`
- `cargo check -q -p akita-pcs --example profile`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test -p akita-config proof_optimized::tests`
- `cargo test -p akita-pcs --test akita_e2e`
- `cargo test`
- `cargo test -q -p akita-pcs --test setup --no-default-features --features parallel,disk-persistence`
- `cargo test -q -p akita-pcs --test batched_aggregated_e2e --no-default-features --features parallel,disk-persistence`
- `cargo nextest run --no-default-features --features parallel,disk-persistence -p akita-pcs --test setup --test batched_aggregated_e2e`
- `cargo build --release --quiet --example profile`
- release smoke for the original 8-case candidate matrix, which identified the
  long hosted-runner offenders and motivated the slim fp128 dense active size
- D32 dense report-gate smoke for `dense_fp16_d32:26:1` and
  `dense_fp32_d32:26:1`
- D32 small-field smoke for `onehot_fp16_d32`, `dense_fp16_d32`,
  `onehot_fp32_d32`, `dense_fp32_d32`, and `onehot_fp64_d32`
- `dense_fp64_d32:26:1` reproduction of the known PR #105 eq-table sizing
  failure
- synthetic failure-continuation check proving multiple failed cases remain in
  `summary.json` and `summary.csv`
- synthetic full-cutover check proving legacy baseline IDs do not compare
  against new semantic IDs
- synthetic missing-summary render check
- shell simulation of the workflow fallback render block
- active workflow matrix parse check: exactly 7 active cases, with
  `onehot_fp16_d32:32:1` and `dense_fp64_d32:25:1`
  omitted until their respective follow-ups

The focused nextest slice for `setup` and `batched_aggregated_e2e` completed
with 31 passed tests in 50.948s. Nextest reported 3 non-failing `LEAK` labels.

### Performance

The reference PR #104 benchmark run took about 11 minutes end to end, with
about 7 minutes in release build and about 3 minutes 20 seconds in benchmark
execution for 3 cases x 5 samples.

This PR reduces per-case samples from 5 to 3 and expands the active matrix from
3 cases to 7 cases. The first 8-case candidate run was useful for finding
costly coverage, but `onehot_fp16_d32:32:1` and `dense_fp128_d32:26:1` are too
expensive for this PR's always-on hosted-runner budget. The active workflow
therefore keeps dense fp128 coverage at `nv=24` (now `dense_fp128_d64:24:1`)
and remains a smoke matrix, not an exhaustive benchmark suite.

One PR run completed all 8 candidate benchmark cases with status `ok`, but the job
failed later in GitHub API baseline/comment handling. This PR now treats those
API paths as warnings so benchmark artifacts can still be uploaded and reviewed.
The same run is the source of the deferred-offender timings above.

## Design

### Profile Modes

`crates/akita-pcs/examples/profile/modes.rs` owns profile mode dispatch. The
mode surface is now explicit:

- `dense_fp{16,32,64,128}_d{32,64}`
- `onehot_fp{16,32,64,128}_d{32,64}`

The old `full*` and bare `onehot*` names are removed. `AGENTS.md` now points the
canonical profiling command at `AKITA_MODE=onehot_fp128_d64`. This is an
explicit per-field D cutover, not a renamed adaptive selector.

### Benchmark Runner And Artifacts

`scripts/profile_bench_report.py run` parses repeated
`MODE:NUM_VARS:NUM_POLYS[:SETUP_MODE]` cases, runs them sequentially, and writes:

- `summary.json`: canonical structured summary
- `summary.csv`: flat tabular summary
- per-case `benchmark.log` files

The runner records failure phase and error details, continues after a failed
case, and exits nonzero if any case failed.

`scripts/profile_bench_report.py failure-summary` exists for workflow-level
failures that occur before `summary.json` is written. It emits structured
failed rows for the configured matrix so the normal renderers still work.

### Report Rendering

`scripts/profile_bench_report.py render` now renders a matrix first. With
`--compact`, it emits the PR-comment version; without `--compact`, it also emits
per-case details in collapsible sections.

The renderer normalizes loaded case summaries from `(mode, nv, np, setup_mode)`.
Direct setup mode keeps the existing semantic ID; recursive setup mode appends
`-setup-recursive`, so direct and recursive rows compare against matching
baselines instead of colliding.

### Schedule And Proof Accounting

Successful runs require both runtime proof-level data and planned/runtime
schedule-level data. For generated plans, the profile asserts that the observed
proof size matches `AkitaSchedulePlan::exact_proof_bytes`. For runtime fallback
schedules, it asserts against `Schedule::total_bytes` and emits the same
level-shaped summary.

The fp128 batched one-hot path now passes the generated plan into the workload
runner so the batched profile emits planned-level output too.

### Workflow Behavior

`profile-bench.yml` builds the profile example once per matrix group. On pull
requests, `scripts/profile_bench_merge_base_policy.py resolve` is the single
source of truth for merge-base baseline:

1. **Mode names** — merge-base must define every profile mode in the group
   (`PROFILE_CI_MODES`, or legacy `PROFILE_MODES` / `PROFILE_ALL_MODES`).
2. **Smoke** — when (1) passes, CI builds the merge-base profile binary and runs
   one smoke execution per case. If merge-base cannot complete a case (placeholder
   schedules, panic, missing tables), baseline is skipped for the group.
3. **Benchmark** — when (1) and (2) pass, PR head and merge-base run
   interleaved; otherwise PR head only (Main=n/a, job stays green).

The workflow:

- looks for previous PR artifacts, including failed benchmark runs that still
  uploaded useful artifacts;
- looks for a main baseline artifact when available;
- skips proof-size threshold checks for failed current cases and missing
  baseline cases;
- renders both full and compact reports;
- wraps only the compact report in the PR-comment marker;
- uploads the benchmark artifact even when benchmark or GitHub API steps fail;
- treats baseline lookup and PR comment upsert API errors as warnings.

### Test Cleanup

The old `batched_onehot_4x30_keeps_folding_past_oversized_tail` E2E test was a
large debug proof for a shape that benchmark CI now preserves directly. Its
schedule-size invariant moved to
`batched_onehot_4x30_plan_keeps_terminal_witness_bounded`, which checks the
generated fp128 D64 one-hot plan and final witness bound without building the
proof. Truncation rejection remains in a smaller `nv20 x np2` E2E fixture.

`setup.rs` now uses `POLY_NV=18` and asserts successful paths are folded rather
than root-direct. `batched_aggregated_e2e.rs` trims the largest one-hot,
dense, and mixed aggregate fixtures while preserving singleton baselines,
irregular batches, mixed dense/one-hot aggregation, serialization round trips,
verification, and folded-proof assertions.

## Alternatives Considered

1. Keep 5 samples.
   Rejected because the expanded matrix would unnecessarily lengthen every
   profile benchmark run. Three samples preserve median reporting while keeping
   workflow time reasonable.

2. Run exhaustive coverage only on a nightly or manual workflow.
   Partially accepted for this first cut: the active PR workflow keeps a
   smaller smoke matrix, while observed long rows are documented as follow-up
   benchmark targets rather than always-on PR coverage.

3. Keep the long per-case markdown report as the PR comment.
   Rejected because even 6 active cases make the comment hard to scan. The full report
   remains available as `report.md`.

4. Permanently drop the deferred long cells.
   Rejected because the point of the matrix is cross-prime and workload
   visibility. Temporarily disabling `onehot_fp16_d32:32:1` and
   `dense_fp64_d32:25:1` is acceptable, and reducing dense fp128 from `nv=26`
   to `nv=24` keeps the active Q128 dense path within hosted-runner budget.

5. Add compatibility aliases for old profile modes or old artifact IDs.
   Rejected under the repo's full-cutover policy. Checked-in call sites and
   report normalization are updated in one pass.

6. Keep all previous heavy debug E2E parameters.
   Rejected because the regular tests were duplicating benchmark-sized coverage.
   The replacement tests keep the relevant invariants explicit and non-vacuous.

## Documentation

Documentation changes in this PR:

- `AGENTS.md` updates the canonical profile command to
  `AKITA_MODE=onehot_fp128_d64`.
- This spec records the active matrix, deferred long hosted-runner cells,
  reporting format, test cleanup, and verification.
- The PR body must summarize the final active matrix, deferred long cells,
  report/CI behavior, validation, and known follow-up.

No paper, protocol, serialization, transcript, or verifier documentation changes
are required because this PR changes benchmark coverage, reporting, and test
cost only.

## Follow-Up

- Re-enable `onehot_fp16_d32:32:1` after small-field one-hot prover cost is
  reduced or the CI runner budget changes.
- Re-enable `dense_fp64_d32:25:1` after a separate dense fp64 validation pass.
- Record the first fully successful expanded workflow runtime after the
  deferred cells are re-enabled.
- Use the candidate matrix data to prioritize real dense/one-hot prover
  performance hotspots separately from this infrastructure PR.

## References

- `specs/TEMPLATE.md`
- `specs/SPEC_REVIEW.md`
- `CONTRIBUTING.md`
- `.github/workflows/profile-bench.yml`
- `scripts/profile_bench_report.py`
- `crates/akita-pcs/examples/profile/modes.rs`
- `crates/akita-pcs/examples/profile/workload.rs`
- `crates/akita-pcs/examples/profile/report.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `crates/akita-pcs/tests/batched_aggregated_e2e.rs`
- `crates/akita-pcs/tests/setup.rs`
- PR #104 benchmark comment:
  `https://github.com/LayerZero-Labs/akita/pull/104#issuecomment-4527174043`
- PR #104 benchmark run:
  `https://github.com/LayerZero-Labs/akita/actions/runs/26428943234`
- Dense fp64 eq-table sizing fix:
  `https://github.com/LayerZero-Labs/akita/pull/105`
