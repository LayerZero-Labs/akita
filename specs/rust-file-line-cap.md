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

The latest audit, after merging the transcript/replay and generated-table
updates through PRs #104 and #113, finds 18 tracked Rust files over the cap
across 255 tracked Rust files. The largest current offender is
`crates/akita-prover/src/protocol/flow.rs` at 4387 lines.

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
- This PR records a follow-up modularization plan, but it does not perform the
  splits.
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
- [ ] Each self-test scenario uses a fresh temporary Git repository; scenarios
  must not rely on or inherit files from earlier cases.
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

Each self-test scenario must run in its own temporary Git repository so
fixtures from earlier scenarios cannot make later success or failure checks pass
by coincidence. The self-tests must cover at least: a new unbaselined offender,
baseline growth, stale baseline removal, a tracked Rust path containing a
space, duplicate baseline rows, untracked baseline paths, invalid line counts,
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

## Current Offender Audit

Audited on 2026-05-27 after merging current `main` into
`quang/rust-file-line-cap` and refreshing the baseline. These counts are the
ratchet start point for this PR, not accepted long-term targets.

| Lines | File | Natural split boundary |
|-------|------|------------------------|
| 4387 | `crates/akita-prover/src/protocol/flow.rs` | Prover flow phases: root setup, recursive suffix, terminal fold, extension-opening reduction, ZK hiding, final proof assembly. |
| 4179 | `crates/akita-types/src/proof/mod.rs` | Proof data families: flat ring/digit containers, direct witness, hints, level/root/terminal proofs, shapes, and serialization. |
| 3367 | `crates/akita-prover/src/protocol/sumcheck/akita_stage2.rs` | Stage-2 prover state, compact accumulation, relation/norm rounds, and tests. |
| 3193 | `crates/akita-prover/src/backend/onehot.rs` | One-hot storage blocks, polynomial API, `AkitaPolyOps` implementation, folding, inner Ajtai, column sweep, and tests. |
| 3090 | `crates/akita-prover/src/kernels/linear.rs` | Linear kernels by operation family: decomposition, NTT matvec, digit matvec, block-parallel paths, single/cyclic paths, fused quotient kernels, and tests. |
| 2894 | `crates/akita-prover/src/protocol/sumcheck/two_round_prefix.rs` | Const lookup-table construction, prefix interpolation helpers, stage-1 state machines, stage-2 state machines, and tests. |
| 2725 | `crates/akita-prover/src/protocol/sumcheck/akita_stage1.rs` | Stage-1 range precomputation, compact coefficient accumulation, prover state, rounds, and tests. |
| 2668 | `crates/akita-field/src/fields/ext.rs` | Extension-field families: `Fp2`, power/tower `Fp4`, ring-subfield `Fp4`, ring-subfield `Fp8`, multiplication backends, and tests. |
| 2243 | `crates/akita-sumcheck/src/extension_opening_reduction.rs` | Tensor helpers, dense reduction prover, sparse/batched witness handling, verifier, sumcheck wrapper, validation, and tests. |
| 2137 | `crates/akita-scheme/src/tests.rs` | Scheme test suites: batched root/direct, standard verify failures, one-hot roundtrips, FP32 ring-subfield configs, and shared fixtures. |
| 2057 | `crates/akita-verifier/src/protocol/levels.rs` | Verifier replay phases: ZK hiding, root level, recursive level, terminal level, dispatch, and shared validation helpers. |
| 1921 | `crates/akita-field/src/fields/fp128.rs` | 128-bit prime field core, arithmetic trait impls, named prime configs, FFT config impls, and tests. |
| 1857 | `crates/akita-config/src/proof_optimized.rs` | Schedule/layout helpers, matrix-envelope helpers, and per-field config modules. |
| 1814 | `crates/akita-pcs/tests/algebra.rs` | Algebra integration-test fixtures and scenario groups. |
| 1771 | `crates/akita-prover/src/protocol/quadratic_equation.rs` | Decompose-fold validation, witness aggregation, V-row construction, high-half/cyclic products, `r_split_eq`, `generate_y`, and tests. |
| 1576 | `crates/akita-prover/src/protocol/ring_switch.rs` | Ring-switch transcript/finalization, commitment construction, eval builders, coefficient construction, and tests. |
| 1506 | `crates/akita-types/src/field_reduction.rs` | Field-reduction encodings, trace/embed helpers, ring-subfield validation/checks, and tests. |
| 1503 | `crates/akita-algebra/src/ring/cyclotomic.rs` | Cyclotomic ring core operations, balanced decomposition, wide ring helpers, serialization, and tests. |

## Resolution Strategy

This PR resolves the immediate policy gap by making the current over-cap files
an explicit ratchet rather than an implicit exception. The follow-up
modularization work resolves the ratchet itself. The endpoint is:

- Every tracked Rust file is at most 1500 physical lines.
- `scripts/rust-file-line-cap-baseline.tsv` contains zero active entries because
  no current offender needs a ratchet row.
- CI still runs the repository-wide checker, so the cap remains enforced after
  the baseline reaches zero entries.

Treat each baseline row as a concrete debt item. A split PR that brings a file
to at most 1500 lines must remove that file's baseline row in the same change;
the checker should fail if the row is left behind. Do not raise recorded counts
to make a split pass. If `main` changes an offender before a split lands, refresh
the row to the exact current line count only as part of a main-merge refresh,
not as a substitute for modularization.

Split by ownership boundary rather than by mechanical line chunks. Production
splits should move cohesive families into private sibling modules and expose
only the narrow `pub(crate)` surface needed by the existing module root. Keep an
existing public path stable only when the crate already exposes it as public
API; do not add deprecated aliases, compatibility wrappers, or duplicate
entrypoints. Update call sites in the same PR so each split is a full cutover.

Use a verification ladder matched to risk:

- Every split PR runs `scripts/check-rust-file-lines.sh` and removes stale
  baseline rows immediately.
- Test-only and config-only splits run the line-cap checks plus the relevant
  crate or integration tests.
- Type, field, sumcheck, prover, and verifier protocol splits run the standard
  workspace verification (`cargo fmt -q`,
  `cargo clippy --all --message-format=short -q -- -D warnings`, and
  `cargo test`) unless the PR explicitly narrows risk with a smaller targeted
  command set.
- Backend and kernel splits additionally run representative profile or bench
  commands before their baseline rows are removed.

## Follow-Up Modularization Plan

The split work should be done in small PRs that keep the public behavioral
surface unchanged and remove baseline entries as soon as each file falls to at
most 1500 lines. For production modules, prefer private submodules plus narrow
`pub(crate)` exports over compatibility wrappers. Re-export existing public
items from the existing module root only where the crate's current public API
requires that path to remain stable.

### Phase 1: Low-risk baseline shrink

Start with files that can fall under the cap without protocol surgery.

1. Move large test bodies out of production files.
   - `crates/akita-scheme/src/tests.rs` should become a `tests/` module tree
     split by scenario family.
   - `crates/akita-pcs/tests/algebra.rs` should split integration fixtures
     from scenario groups.
   - `crates/akita-types/src/field_reduction.rs` and
     `crates/akita-algebra/src/ring/cyclotomic.rs` are barely over the cap;
     moving tests or wide-ring-only tests should remove their baseline entries
     with minimal production risk.
2. Split config presets from config helpers in
   `crates/akita-config/src/proof_optimized.rs`.
   - Keep schedule/layout helper functions near the top-level module.
   - Move `fp16`, `fp32`, `fp64`, and `fp128` preset modules into sibling
     files under a `proof_optimized/` module directory.

### Phase 2: Type and field modules

Handle large type-definition files before protocol flow files.

1. Split `crates/akita-types/src/proof/mod.rs` into proof-family modules.
   Suggested first cut:
   - `containers.rs` for `PackedDigits`, `FlatRingVec`, and
     `FlatDigitBlocks`.
   - `direct_witness.rs` for direct witness proof/shape handling.
   - `hints.rs` for `AkitaCommitmentHint`.
   - `levels.rs` for level, terminal, root, batched, and step proof structs.
   - `shapes.rs` for shape structs/enums and shape derivation helpers.
   - `wire.rs` for serialization/deserialization helpers and impl blocks.
2. Split `crates/akita-field/src/fields/ext.rs` by extension family.
   - Keep `Ext2`/`Fp2` together.
   - Separate power-basis `Fp4`, tower-basis `Fp4`, ring-subfield `Fp4`, and
     ring-subfield `Fp8`.
   - Keep multiplication backend traits near the family that consumes them so
     the full cutover does not leave a catch-all extension module.
3. Split `crates/akita-field/src/fields/fp128.rs` only after `ext.rs`.
   The field file is mostly trait impl surface and named prime config; splitting
   it too early risks churn with less architectural payoff.

### Phase 3: Sumcheck and extension-opening internals

These files are algorithmically cohesive but have good internal seams.

1. Split `two_round_prefix.rs` first because both stage drivers depend on it.
   Suggested modules: `lookup_tables`, `interpolation`, `stage1_state`,
   `stage2_state`, and tests.
2. Split `akita_stage1.rs` into range/precompute helpers, compact coefficient
   accumulation, prover state, round execution, and tests.
3. Split `akita_stage2.rs` into compact accumulators, relation/norm helpers,
   prover state, round execution, and tests.
4. Split `extension_opening_reduction.rs` into tensor helpers, dense prover,
   sparse witness/factors, batched prover, verifier/sumcheck wrapper, and
   validation helpers.

### Phase 4: Prover and verifier protocol flow

Do this after the type and sumcheck splits so the protocol files can import
smaller concepts instead of moving huge blocks unchanged.

1. Split `crates/akita-prover/src/protocol/ring_switch.rs` first. It is only
   76 lines over the cap and has a clean boundary between finalization,
   commitment construction, eval builders, and coefficient construction.
2. Split `crates/akita-prover/src/protocol/quadratic_equation.rs` next.
   Separate witness aggregation/V-row construction from high-half/cyclic
   product helpers and public `compute_r_split_eq` / `generate_y` entrypoints.
3. Split `crates/akita-verifier/src/protocol/levels.rs` by replay phase while
   preserving the verifier no-panic contract. Keep validation helpers close to
   the API boundary rather than duplicating defensive checks in hot loops.
4. Split `crates/akita-prover/src/protocol/flow.rs` last.
   Suggested modules: `inputs`, `zk_hiding`, `proof_steps`,
   `recursive_suffix`, `recursive_level`, `terminal_level`,
   `root_extension`, and `root_fold`. This should be a move-only refactor
   whenever possible; avoid rewriting transcript ordering or compute-backend
   plumbing during the split.

### Phase 5: Hot backend and kernel files

These should come after correctness-sensitive protocol files because they are
performance-sensitive and need targeted benchmarks.

1. Split `crates/akita-prover/src/backend/onehot.rs` by storage representation
   and operation family: `blocks`, `poly`, `ops`, `fold`, `inner_ajtai`,
   `column_sweep`, and tests.
2. Split `crates/akita-prover/src/kernels/linear.rs` by kernel family:
   `decompose`, `ntt_matvec`, `digit_matvec`, `block_parallel`,
   `single_cyclic`, `fused_quotients`, and tests.
3. Run the existing unit tests plus representative profile/bench commands
   before removing the baseline entries for backend/kernel files.

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
  `main` has 18 tracked Rust files over 1500 lines. A strict check would make
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
- Current branch audit after the 2026-05-27 main merge: 255 tracked Rust
  files, 18 files over 1500 lines, largest offender
  `crates/akita-prover/src/protocol/flow.rs` at 4387 lines.
- `scripts/rust-file-line-cap-baseline.tsv` is the authoritative current
  offender list for this PR.
- The CI script scans all tracked Rust files; on this branch that is 255 files.
