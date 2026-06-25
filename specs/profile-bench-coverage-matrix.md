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
> cells remain at D128 because their D64 is non-securable. The current matrix
> is the "Active Benchmark Matrix" section below. Everything else (this Summary,
> and everything from "## Evaluation" onward: Acceptance Criteria, Validation,
> Performance, Design, Alternatives, Follow-Up) is the original **PR #107
> historical record**. Its fp16 / D32 / D64 cell references (e.g.
> `onehot_fp16_d32`, `dense_fp64_d32`, `dense_fp128_d32`, the "fp128 D32" report
> wording) describe the pre-reprice matrix and are superseded by the Active
> Benchmark Matrix; they are retained as PR #107's completed acceptance record,
> not the current shipping configuration.

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
| `onehot_fp32_d128` | fp32 | 1-of-256 one-hot | 28 | 1 | D128 | `direct` | Smallest securable fp32 one-hot under honest pricing. Capped at nv=28: the ext-degree-4 challenge schedule keeps a large un-folded witness, so at nv>=30 the prover's eq-evaluation table exceeds the 1 GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` ceiling. |
| `onehot_fp64_d128` | fp64 | 1-of-256 one-hot | 28 | 1 | D128 | `direct` | Smallest securable fp64 one-hot under honest pricing. Capped at nv=28 for the same eq-table-budget reason as the fp32 cell. |
| `dense_fp128_d64` | fp128 | dense | 24 | 1 | D64 | `direct` | fp128 dense smoke at the proof-size-optimal ring dimension (D64 beats D128 by ~18-22%). |
| `onehot_fp128_d64` | fp128 | 1-of-256 one-hot | 32 | 1 | D64 | `direct` | Explicit fp128 one-hot mode at the proof-size-optimal ring dimension. fp128 folds aggressively enough to stay at nv=32 under the eq-table budget. |
| `onehot_fp128_d64` | fp128 | 1-of-256 one-hot | 32 | 1 | D64 | `recursive` | Same nv32 singleton as the direct row, but with `SetupContributionMode::Recursive`, so the report compares proof size, prover time, and verifier time for recursive setup-product checks. |
| `onehot_fp128_d64` | fp128 | 1-of-256 one-hot batched | 30 | 4 | D64 | `direct` | Preserves same-point batched one-hot coverage. |

Every active cell folds securely under honest committed-fold A-role pricing.
The ring degree differs by field, for two distinct reasons:

- **Small prime fields (fp32/fp64):** their D32/D64 schedules are no longer
  securable under the reprice — they degrade to a cleartext root-direct proof
  and stop exercising a real folding commitment — so the smallest secure ring
  degree is D128. Those cells (and all fp16 cells) use D128 one-hot.
- **fp128:** D64 is the actual proof-size optimum for both dense and one-hot.
  Measured against the runtime schedule's `total_bytes`, D64 produces ~18-23%
  smaller proofs than D128 across the matrix shapes (e.g. one-hot nv=32:
  133,000 B at D64 vs 163,968 B at D128; dense nv=24: 131,656 B vs 160,080 B),
  while still folding through 8-9 secure recursive levels. This is confirmed by
  `current_d64_onehot_schedule_stays_within_audited_sis_widths` (securability)
  and by the `best_full_schedule` / `best_onehot_schedule` selectors, which
  pick D64 (or D32), never D128. The earlier D128 fp128 cells were *not*
  proof-size optimal; D32/D64 are the planner optima (D32 is marginally
  smaller for fp128). The benchmark matrix tracks D64; use
  `best_onehot_schedule` / `best_full_schedule` to compare D32/D64/D128.

D32/D128 profile modes still exist for direct local comparisons, and `main`
adds a D64-only tensor-verifier profile mode, but neither the adaptive
`full`/`onehot` selectors nor those comparison modes are part of the active
benchmark matrix.

Deferred target cells:

| Mode | Field | Workload | Variables | Polys | Config | Re-enable condition |
| --- | --- | --- | ---: | ---: | --- | --- |
| `dense_fp64_d128` | fp64 | dense | 24 | 1 | D128 | Re-enable after dense small-field hosted-runner cost is validated (fp32 ships no dense family, so fp64 D128 is the only securable dense small-field cell). |

The first successful 8-case candidate run identified the two cost offenders:
`onehot_fp16_d32:32:1` spent about 210 seconds in proving and peaked around
6.2 GiB RSS, while `dense_fp128_d32:26:1` spent about 56 seconds in commit plus
38 seconds in prove and peaked around 8.4 GiB RSS (those figures are the
historical D32-era measurements). The always-on fp128 dense cell stays at
`nv=24` to keep CI tractable, and now runs at the proof-size-optimal
`dense_fp128_d64:24:1`; D64 commit/prove cost differs from the D32 numbers
above, so the timing baseline must be regenerated on the first post-swap run.

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
- [x] The active workflow lists exactly the 7 currently supported
      hosted-runner matrix cases.
- [x] The known long hosted-runner offender `onehot_fp16_d32:32:1` is
      documented as deferred rather than active.
- [x] `dense_fp128_d128` remains active at `nv=24`, not the earlier `nv=26`
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
therefore keeps dense fp128 coverage at `nv=24` (now `dense_fp128_d128:24:1`)
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

After merging `main`, the profile example also exposes
`onehot_fp128_d64_tensor` as a direct local comparison mode because the tensor
verifier preset and generated tables are D64-only. That mode is intentionally
excluded from the active CI benchmark matrix and from `AKITA_MODE=all`.

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

`profile-bench.yml` builds the profile example once and runs the configured
matrix with `AKITA_BENCH_RUNS=3`.

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
- Revisit `dense_fp128_d128` at `nv=26` after dense fp128 commit/prove cost is
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
