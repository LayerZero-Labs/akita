# Security analysis for fp128 fold presets

This note records the SIS sizing invariant used by the generated fp128 schedule
tables and the tensor-shaped `D64OneHotTensor` preset.

## Tensor extraction bound

Akita §5, Lemma "Tensor MSIS norm", gives the two-level tensor fold extraction
degradation for challenge-dependent rows as exactly `4 · omega`, where `omega`
is the sparse challenge family's L1 bound. The honest folded-witness bound still
uses the logical per-block tensor challenge mass `omega²`, because the block
challenge is the product `left[p] · right[q]`.

The implementation therefore uses two separate quantities:

- `LevelParams::challenge_l1_mass()` for honest fold-witness digit sizing.
  Flat mode returns `omega`; tensor mode returns `omega²`.
- `LevelParams::stage1_sis_extraction_report(raw)` for A-role SIS sizing.
  Flat mode returns the historical `raw · infinity_norm` collision value.
  Tensor mode returns `raw · infinity_norm · 4 · omega`, rounded up to a
  generated SIS collision bucket.

For the production fp128 `D=64` one-hot tensor root, the challenge family has
`omega = 54` and `infinity_norm = 2`, so the tensor extraction multiplier is
`4 · 54 = 216` and the A-role extraction coefficient bound is `432` before the
role-specific raw collision multiplier.

## Generated SIS buckets

The checked-in `sis_floor` table remains a strict superset for flat presets:
existing buckets through `2047` are unchanged, so flat rank lookups are
bit-identical. The `Q128, D=64` table additionally covers tensor buckets
`4095` and `8191`; these cover the widest tensor A-role buckets reached by the
`D64OneHotTensor` planner search while preserving the existing flat tables.

## Per-preset status

- `D32Full`, `D32OneHot`, `D64Full`, `D64OneHot`, `D128Full`, and
  `D128OneHot` remain flat-mode presets. Their A-role extraction bucket is the
  historical `raw · infinity_norm` bucket, and their generated schedule tables
  continue to validate against the unchanged flat SIS rows.
- `D64OneHotTensor` uses tensor extraction only at the root fold level and flat
  extraction at recursive levels. Its generated table was produced after the
  `4 · omega` A-role bucket was wired into `sis_derived_root_params_for_layout`
  and `sis_derived_recursive_params_for_layout`.
- Batched-root entries scale B/D widths and revalidate the stored ranks against
  the same collision buckets as singleton entries.

The generated-table validation tests exercise the flat fp128 tables and the
new `D64OneHotTensor` table against the live planner, so stale rank or layout
entries fail during schedule materialisation rather than at proof time.
