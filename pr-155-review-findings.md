# PR #155 Review Findings

PR: https://github.com/LayerZero-Labs/akita/pull/155

## Conclusion

Request changes before merge. The core SIS L2 cutover and op-norm rejection paths look largely sound, and GitHub CI is green, but there are generated-table and reproducibility issues that should be fixed before landing a crypto-parameterization PR.

## Findings

### High: Q64 SIS Table Contains Unreproducible Rank Entries

`scripts/stitch_generated_sis_table.py` defaults to `--max-rank 20`, but `crates/akita-types/src/sis/generated_sis_table/q64.rs` contains at least two rows with more than 20 rank entries:

- `crates/akita-types/src/sis/generated_sis_table/q64.rs`, row for `(32, 137438953472)`: 24 entries.
- `crates/akita-types/src/sis/generated_sis_table/q64.rs`, row for `(32, 35184372088832)`: 26 entries.

`min_secure_rank` trusts every entry in the returned slice, so ranks 21+ can become accepted from stale or non-regenerable table data:

```rust
let widths = sis_max_widths(sis_family, d, collision_l2_sq_rounded_up)?;
for (i, &max_w) in widths.iter().enumerate() {
    if width <= max_w {
        return Some(i + 1);
    }
}
```

Fix by regenerating or truncating the affected q64 rows and adding a row-length assertion/check so generated table slices cannot silently exceed the intended rank cap.

### Medium: Tiered ZK Generated Table Is Dead Code

`crates/akita-planner/src/generated/fp128_d64_onehot_tiered_zk.rs` is committed, but the planner does not compile or select tiered tables under `zk`.

The generator emits `_zk` modules for every generated family, including tiered, but `crates/akita-planner/src/generated/mod.rs` only declares `fp128_d64_onehot_tiered` under `#[cfg(not(feature = "zk"))]`, and `crates/akita-planner/src/resolve.rs` only returns the tiered table under `#[cfg(not(feature = "zk"))]`.

Fix by either wiring the tiered ZK table into planner compilation/selection and tests, or gating tiered out of ZK generation so the dead generated file is not committed.

### Low/Medium: Golden CSV Fails `git diff --check`

Local `git diff --check pr-base/main...pr/155` reports trailing whitespace on every row in `scripts/sis_golden/golden.csv`. This appears to come from Python CSV default CRLF line endings.

This does not affect runtime behavior, but it breaks a common repository preflight and creates noisy hygiene failures for future reviewers.

Fix by setting `lineterminator="\n"` in `scripts/sis_golden/refresh_golden.py` and normalizing the committed CSV.

### Low: Tests And Docs Still Reference Stale Bucket/Table Assumptions

`crates/akita-pcs/src/scheme/tests/fp32_ring_subfield.rs` hard-codes power-of-two buckets even though the canonical helper now prefers derived `d * B^2` keys when available.

Separately, docs still point at the removed monolithic `generated_sis_table.rs` instead of the split `generated_sis_table/` module directory. Examples include:

- `scripts/sis_golden/README.md`
- `AGENTS.md`
- `specs/sis-euclidean-estimator.md`

Fix the fixture comments/constants to use the canonical helper behavior, and update documentation to describe the split table layout.

## Non-Blocking Notes

The op-norm rejection path appears transcript-stable and deterministic. The new `operator_norm_cap()` / `effective_operator_norm_cap()` APIs are currently not wired into A-role SIS sizing; sizing still uses `omega = ||c||_1`. This is conservative and is explicitly documented in `specs/l2-msis-opnorm-folded-witness.md`, so it is not a blocker for this PR.

Prior PR comments about unsorted bucket handling, `SIGALRM` import portability, golden SHA pinning, and ZK/non-ZK schedule divergence appear addressed in final head.

## Verification Performed

- Refreshed PR body, comments, and checks with `gh`.
- GitHub checks were green: format, clippy, tests, fuzz, portability, audit/deny, CodeQL, and benchmarks.
- Ran `cargo test -p akita-types sis -q`: passed, 34 tests.
- Ran `cargo test -p akita-challenges op_norm -q`: passed, 11 executed tests, 2 ignored.
- Ran `git diff --check pr-base/main...pr/155`: failed due to `scripts/sis_golden/golden.csv` row endings.
- Could not run `scripts/sis_golden/check.py` because `sage` is not installed locally; the PR body also leaves this item unchecked.
