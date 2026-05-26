# Spec: Rust File Line Cap

| Field       | Value             |
|-------------|-------------------|
| Author(s)   | Quang Dao         |
| Created     | 2026-05-26        |
| Status      | proposed          |
| PR          | #110              |

## Summary

Akita has several hand-maintained Rust files that are large enough to hide
ownership boundaries and make future refactors harder to review. This feature
adds a CI-enforced 1500-line cap for tracked Rust files, with an explicit
ratchet for the current files that already exceed the cap on `main`.

## Intent

### Goal

Add a repository check that prevents new Rust files from exceeding 1500 lines
and prevents the existing over-cap files from growing while they are being
modularized.

### Invariants

- Every tracked `.rs` file not listed in the line-cap baseline must have at
  most 1500 physical lines.
- Every baseline entry must name a tracked Rust file that currently exceeds
  1500 lines.
- A baseline file may not grow beyond its recorded line count.
- A baseline entry becomes stale once the file reaches 1500 lines or fewer;
  stale entries must make CI fail so the baseline shrinks over time.
- Malformed baseline rows must fail clearly, including invalid counts,
  duplicate paths, non-Rust paths, absolute or parent-directory paths, and
  paths that are not tracked by Git.
- The check must include Rust files under `src`, `tests`, `benches`,
  `examples`, and generated source directories. Generated files are not
  special-cased unless a future generated file is explicitly baselined.
- The check must be filename-safe for ordinary repository paths, including
  spaces.

### Non-Goals

- This PR does not modularize the existing over-cap files.
- This PR does not enforce a 1000-line cap.
- This PR does not change Clippy lint policy or rely on Clippy's
  `too_many_lines` lint.
- This PR does not exempt generated Rust files as a category.

## Evaluation

### Acceptance Criteria

- [ ] A local script fails when a non-baselined tracked Rust file has more
  than 1500 physical lines.
- [ ] The script fails when a baselined file grows beyond its recorded line
  count.
- [ ] The script fails when a baseline entry no longer points at a tracked
  Rust file over the cap.
- [ ] The self-test script exercises those failure paths using tracked
  temporary Git fixtures, including a Rust path containing a space.
- [ ] The self-test script exercises malformed baseline validation for
  duplicate paths, untracked paths, invalid line counts, non-Rust paths,
  absolute paths, and parent-directory paths.
- [ ] The script passes on the current branch when the baseline matches the
  audited current offenders.
- [ ] GitHub CI runs both the self-test script and the repository line-cap
  script on pull requests and pushes to `main`.

### Testing Strategy

Run the line-cap script locally in its normal mode:

```bash
scripts/check-rust-file-lines.sh
```

Run the self-test script, which creates isolated temporary Git repositories
with tracked Rust fixtures and small line limits:

```bash
scripts/test-rust-file-lines.sh
```

The self-tests must cover at least: a new unbaselined offender, baseline
growth, stale baseline removal, a tracked Rust path containing a space,
duplicate baseline rows, untracked baseline paths, invalid line counts,
non-Rust paths, absolute paths, and parent-directory paths.
Existing format, Clippy, doc, and test jobs are unchanged.

### Performance

The check only scans tracked Rust files using `git ls-files` and line counts.
There is no formal performance gate for this policy check; it is expected to
remain lightweight checkout-only shell work. If runtime becomes suspect,
measure it with:

```bash
time scripts/check-rust-file-lines.sh
```

The CI job is intentionally checkout-only and does not install a Rust toolchain.

## Design

### Architecture

Add a shell script under `scripts/` that:

1. Finds the repository root.
2. Reads `scripts/rust-file-line-cap-baseline.tsv`.
3. Counts physical lines for tracked `.rs` files.
4. Reports all violations in one run.
5. Exits nonzero on any violation.

Add a second shell script under `scripts/` that self-tests the checker in
temporary Git repositories. The self-test must use tracked fixture files rather
than untracked scratch files, because the production checker intentionally
scans only tracked Rust files.

The baseline file is a TSV with recorded line count and path. The recorded
count is an upper bound for the current offender, not a permanent exemption.
Once a file is modularized below the cap, the script rejects the stale baseline
entry and the implementation PR must remove it.

GitHub Actions gets a dedicated lightweight job in `.github/workflows/ci.yml`
that first runs the self-test script and then scans the real repository. This
keeps line-cap failures visible independently from format, Clippy, and test
failures.

### Alternatives Considered

- **Strict all-files cap immediately.** Rejected for this PR because current
  `main` has 16 tracked Rust files over 1500 lines. A strict check would make
  the PR unmergeable unless it also performed a broad modularization pass.
- **Check only changed files.** Rejected because it would not prevent an
  existing offender from growing and would make the final zero-exception state
  less visible.
- **Category exemption for generated files.** Rejected for now because the
  current generated Rust files are below the cap and explicit baselining is
  clearer if a generated file ever exceeds it.

## Documentation

This spec is the developer-facing documentation for the policy. The CI failure
message should be self-contained enough that a contributor can either split the
file or, for current offenders only, understand why the baseline must shrink
rather than grow.

## Execution

- Add `scripts/check-rust-file-lines.sh`.
- Add `scripts/test-rust-file-lines.sh`.
- Add `scripts/rust-file-line-cap-baseline.tsv` with the current audited
  offenders.
- Add a `Rust file line cap` job to `.github/workflows/ci.yml`.
- Verify with `scripts/test-rust-file-lines.sh` and
  `scripts/check-rust-file-lines.sh`.

## References

- Initial crate audit: 229 Rust files under `crates/`, 16 files over 1500
  lines, largest offender `crates/akita-types/src/proof/mod.rs` at 3695
  lines.
- `scripts/rust-file-line-cap-baseline.tsv` is the authoritative current
  offender list for this PR.
- The CI script scans all tracked Rust files; on this branch that is 237 files.
