# SIS infinity estimator implementation slices

| Field | Value |
|-------|-------|
| Status | superseded |
| Superseded-by | [`specs/sis-linf-table-cutover.md`](../../sis-linf-table-cutover.md) |
| Archived | 2026-Q2 |

This plan breaks `specs/sis-infinity-estimator-crate.md` into slices that can be
implemented and audited one at a time. Codex owns the branch and final review.
Composer 2.5 sidekicks may inspect, draft patch plans, or prototype narrow
pieces, but their output is advisory until Codex verifies it against source and
tests.

The target profile for Akita's first useful infinity estimator is:

```text
norm = infinity
reduction cost model = ADPS16
shape model = LGSA
zeta = full optimizer
```

The public API must stay generic enough to expose lattice-estimator style SIS
inputs. Infinity norm is the priority. Euclidean norm remains a later
compatibility path.

## Review rules for every slice

1. No production table may become looser without an explicit golden update and
   review note.
2. Existing Euclidean table generation must keep working until a later slice
   deliberately changes it.
3. If Rust and Sage disagree, assume Rust is wrong until the exact cause is
   found.
4. Fragile Sage cells must be recorded as fragile. They must not become hard
   parity requirements.
5. Verifier-reachable code must remain table-only. The estimator crate is for
   offline generation, examples, and tests.
6. Each slice must include a sidekick review prompt and a main-agent audit pass.

## Slice 1. Infinity Sage goldens only

Goal: add a trusted, isolated infinity-norm golden harness that can query
lattice-estimator on the upstream stability PR branch without changing Akita's
current Euclidean generated tables.

Preferred reference source:

```text
malb/lattice-estimator PR 217
head branch = quangvdao:quang/fix-amplify-tiny-success
head SHA = c667a48546f140c3a5454c7503c3ca44a264cce2
```

Patch surface:

1. Add a small infinity golden grid under `scripts/sis_golden/`, separate from
   the current Euclidean `grid.py`. Suggested name:
   `scripts/sis_golden/infinity_grid.py`.
2. Add `refresh_infinity_golden.py` that writes an infinity CSV plus metadata.
3. Add `check_infinity.py` that replays committed infinity cells and validates
   monotonicity and security brackets where the metadata says the cell is
   trustworthy.
4. Add an infinity-local helper module if shared code is needed. Suggested name:
   `scripts/sis_golden/infinity_core.py`.
5. Keep `scripts/gen_sis_table.py`, `scripts/stitch_generated_sis_table.py`,
   `scripts/sis_golden/golden.csv`, and `scripts/sis_golden/metadata.json`
   behavior unchanged.
6. Document the new manual commands in `scripts/sis_golden/README.md`.

Suggested output names:

```text
scripts/sis_golden/infinity_golden.csv
scripts/sis_golden/infinity_metadata.json
```

Isolation rule:

The infinity harness may import neutral utility functions such as estimator
location and remote URL normalization, but it must not import
`binary_search_max_width`, `estimate_bits`, or `assert_pinned_estimator` from
`scripts/gen_sis_table.py`. Those helpers encode the Euclidean `norm=2`,
`BDGL16`, `collision_l2_sq` contract and the old SHA pin. Slice 1 should have
its own infinity profile check and its own PR 217 SHA check.

Minimum golden dimensions:

```text
families = q32, q64, q128
d = 32, 64, 128, 256
rank = 1, 5, 20
width = fixed small values plus one width-search bracket per family
coeff_linf_bound = representative Akita buckets, including small and large cells
target_bits = 138
profile = ADPS16 + LGSA + norm infinity + full zeta optimizer
```

Done when:

1. The new infinity golden refresh can run against the PR 217 checkout.
2. The new infinity check can replay the committed CSV.
3. Existing Euclidean golden check still passes against the existing pinned
   checkout.
4. Documentation guardrails pass.
5. No Rust production code or generated table files change.

Fast verification:

```bash
python3 -m py_compile scripts/sis_golden/refresh_infinity_golden.py scripts/sis_golden/check_infinity.py
git diff --check
./scripts/check-doc-guardrails.sh
```

Sage verification:

```bash
sage -python scripts/sis_golden/check.py
sage -python scripts/sis_golden/refresh_infinity_golden.py \
  --estimator-path /path/to/lattice-estimator-pr217
sage -python scripts/sis_golden/check_infinity.py \
  --estimator-path /path/to/lattice-estimator-pr217
```

Sidekick use:

Ask Composer to inspect only `scripts/sis_golden/`, `scripts/gen_sis_table.py`,
`scripts/stitch_generated_sis_table.py`, and
`crates/akita-types/src/sis/ajtai_key.rs`. It should report the smallest patch
surface and the exact Sage commands. It must not edit.

Audit focus:

1. Confirm `norm=oo` is passed to `SIS.Parameters`.
2. Confirm `red_cost_model=RC.ADPS16` is passed to `SIS.lattice`.
3. Confirm `red_shape_model="lgsa"` or the exact lattice-estimator equivalent is
   passed on the infinity path.
4. Confirm zeta is not accidentally fixed unless the row says it is a fixed
   zeta cell.
5. Confirm the old Euclidean scripts do not import infinity grid state or change
   output.
6. Confirm the infinity harness keys cells by `coeff_linf_bound`, not
   `collision_l2_sq` or the derived `d * B^2` Euclidean key.

## Slice 2. Crate skeleton and type-level API

Goal: add `crates/akita-sis-estimator` as a workspace member with the generic
public API and no real estimator math yet.

Patch surface:

1. Add crate manifest and `src/lib.rs`.
2. Add `params`, `config`, `cost`, `error`, and `numeric` modules.
3. Define `SisParameters`, `SisNorm`, `Bound`, `EstimateConfig`,
   `ReductionCostModel`, `ShapeModel`, `OptimizerConfig`, `SearchMode`,
   `LatticeCost`, `CostValue`, and `EstimatorError`.
4. Add validation that rejects malformed inputs without panics.
5. Add tests for constructors, validation, serialization-free debug output, and
   enum coverage.

Do not implement lattice cost formulas in this slice.

Done when:

```bash
cargo fmt -q
cargo clippy -p akita-sis-estimator --all-targets -- -D warnings
cargo test -p akita-sis-estimator
```

Sidekick use:

Ask Composer to review API completeness against the spec and list any missing
lattice-estimator input fields. It must not write code.

Audit focus:

1. The API accepts infinity and Euclidean norms, even if only infinity is later
   implemented first.
2. The API has room for fixed beta, fixed zeta, full beta search, and full zeta
   search.
3. No Akita runtime crate depends on the estimator crate.

## Slice 3. Fixed infinity cost for ADPS16 plus LGSA

Goal: implement fixed-beta, fixed-zeta infinity cost for the first target
profile and compare it to Sage fixed cells.

Patch surface:

1. Implement exact integer dimension mapping from Akita ring inputs to scalar
   SIS inputs.
2. Implement the ADPS16 reduction cost model needed by the fixed path.
3. Implement the LGSA shape model needed by the fixed path.
4. Implement `cost_infinity_fixed`.
5. Add golden tests that load Slice 1 fixed cells.

Do not implement beta or zeta search in this slice.

Done when:

```bash
cargo fmt -q
cargo test -p akita-sis-estimator fixed_infinity
cargo clippy -p akita-sis-estimator --all-targets -- -D warnings
```

Audit focus:

1. Check every logarithm base and rounding direction.
2. Compare Rust intermediate values to Sage for a few rows.
3. Make sure unstable cells are skipped or marked fragile, not forced.

## Slice 4. Full beta and zeta optimizer

Goal: match lattice-estimator's infinity optimizer behavior for the target
profile.

Status: complete. The serial optimizer matches the trusted infinity goldens, and
Slice 4b adds deterministic `ExhaustiveParallel` search behind the `parallel`
feature, serial-vs-parallel parity tests, and a durable Criterion benchmark for
local-minimum, serial exhaustive, and parallel exhaustive optimizer profiles.

Patch surface:

1. Implement beta search.
2. Implement zeta search.
3. Add serial full-search parity tests against Slice 1 optimizer goldens.
4. Add optional parallel search behind the workspace `parallel` feature only
   after the serial implementation is correct.

Done when:

```bash
cargo test -p akita-sis-estimator optimizer
cargo test -p akita-sis-estimator --no-default-features
cargo test -p akita-sis-estimator --features parallel
```

Audit focus:

1. Search bounds match lattice-estimator.
2. Tie-breaking matches lattice-estimator.
3. Parallel search is deterministic.

## Slice 5. Width search and Akita infinity table artifacts

Goal: produce comparison-only infinity max-width tables without changing current
runtime SIS sizing.

Status: complete for comparison artifacts. The Rust-native generator emits the
planner-shaped infinity key `(family, ring_dimension, coeff_linf_bound)` and
supports planner-scale explicit scalar `m` with wide `u64` dimensions.
Generated rows record true search-cap hits explicitly as lower bounds. Width
search reports the secure prefix boundary, so later secure islands caused by
infinity-probability branch changes do not make the planner table optimistic.

Patch surface:

1. Add a Rust or script-based table generator for
   `InfinityAdps16LgsaFullZeta`.
2. Add a generated comparison artifact outside production lookup modules.
3. Add monotonicity and bracket checks for width by rank and bound.
4. Keep `generated_sis_table/` Euclidean behavior unchanged.

Audit focus:

1. Width monotonicity by rank.
2. Bound monotonicity by `coeff_linf_bound`.
3. Cap-hit rows are recorded as lower bounds.

## Slice 6. Full model surface

Goal: fill out lattice-estimator-compatible options after the target profile is
stable.

Patch surface:

1. Add remaining shape models: GSA, ZGSA, CN11, CN11_NQ.
2. Add remaining reduction models needed for SIS parity: BDGL16, MATZOV, GJ21,
   Kyber, and any additional lattice-estimator SIS models we choose to expose.
3. Add parity tests per model.

Audit focus:

1. Model names and default behavior match lattice-estimator.
2. Unsupported model combinations return typed errors.

## Slice 7. Production Akita integration

Goal: switch Akita's infinity-originating SIS table path to the new estimator
artifacts only after enough parity evidence exists.

Patch surface:

1. Add production infinity table modules.
2. Add `SisTableKey::CoeffInfinityBound`.
3. Switch selected L-infinity-originating lookups away from derived L2 keys.
4. Keep old Euclidean table support for compatibility until Euclidean Rust parity
   lands.

Audit focus:

1. Every changed layout is at least as secure as before, or has an explicit
   reviewed rationale.
2. Descriptor and verifier no-panic contracts remain intact.
3. Existing proof fixtures either remain byte-compatible or are deliberately
   regenerated with a clear reason.

## Slice 8. Euclidean parity path

Goal: port the existing Euclidean `norm=2`, `BDGL16` table profile after the
infinity path is stable.

Patch surface:

1. Add Euclidean cost path.
2. Reuse current `scripts/sis_golden/golden.csv` as the parity source.
3. Add Rust Euclidean table generation in comparison mode.
4. Only then consider replacing the old Sage generator.

Audit focus:

1. Current Akita generated table rows remain reproducible.
2. Derived `d * B^2` keys match the existing stitched table.
3. The Euclidean path does not influence infinity defaults.
