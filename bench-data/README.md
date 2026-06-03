# Field Microbench Artifacts

The committed artifacts are generated from Criterion baselines under `target/criterion/`:

- `field-microbench.csv`: complete machine-readable rows, each carrying its own capture provenance (`git_commit`, `captured_at_utc`).
- `field-microbench.md`: reader-facing reference with workload definitions, anonymized machine configuration, coverage, headline rows, and ring-subfield focus rows.
- `field-microbench-meta.json`: collector metadata, per-baseline machine metadata, a `row_provenance` summary (per-baseline `commit_row_counts` and captured-at range), and warnings.

Do not hand-edit the generated files.
Refresh them with `scripts/field_microbench_collect.py`.

`ns_per_op_or_lane` is taken from Criterion `estimates.json` medians.
Criterion `WallTime` stores those point estimates in nanoseconds.

## How To Read Rows

| Column | Meaning |
|--------|---------|
| `baseline` | Saved Criterion baseline name, such as `neon`, `avx2`, or `avx512` |
| `machine_config` | Non-identifying hardware/SIMD configuration label from the per-baseline metadata JSON |
| `workload` | `latency_chain` is a dependent critical path; `throughput_stream` is independent parallel streams |
| `vectorization` | `scalar` or `packed`; packed rows are normalized per SIMD lane |
| `width` | SIMD lane count for packed rows, repeated on scalar rows to identify the build cell |
| `unit` | `ns/op` for scalar rows, `ns/lane` for packed rows |
| `lower`, `upper` | Criterion confidence interval bounds for the median estimate |
| `label` | Short bench label from the Criterion group name |
| `git_commit`, `captured_at_utc` | Per-row provenance: the commit and UTC time that produced *this* measurement. Rows refreshed independently legitimately carry different commits, so the table is not assumed to be a single-commit snapshot. |

Each `(baseline, library, field, ext_degree, basis, op, workload, vectorization)` tuple is intended to be one row.
Duplicate rows usually mean an old long-label baseline is still present in `target/criterion/`; the collector resolves those in favor of the short labels and emits a warning.

## Per-Row Provenance And Targeted Refresh

Provenance is per row, not per machine: each measurement records the commit and time it was captured at.
This lets a single bench be refreshed without re-running the whole suite or misdating every other row.

`collect` merges with the existing table by default.
A freshly measured row is restamped at the current commit (or `--git-commit`) only when its value actually changed; every other row (other baselines, un-rerun benches) is carried forward verbatim with its original provenance.
So a targeted refresh is: re-run only the affected benches into the saved baseline, then `collect` for just that baseline; the merge updates only the rows whose value moved.

- `--git-commit COMMIT` stamps freshly changed rows at `COMMIT` (default: current `HEAD`).
- `--replace` rebuilds the table from collected rows only, discarding carried-forward rows and their provenance (use for a full re-capture).

`field-microbench-meta.json`'s `row_provenance` reports, per baseline, how many rows sit at each commit and the captured-at range, so a mixed-commit table is auditable at a glance.

## Criterion Directory Names

Criterion truncates each path component to 64 characters.
Long group names like `prime31_offset19_ring_subfield_fp4_w8` are cut off and will not parse reliably.

`ext4` benches use short labels (see `crates/akita-pcs/benches/field_arith/ext4.rs`):

| Label | Field | Basis |
|-------|-------|-------|
| `m31_rs_fp4` | `mersenne31`, `2^31 - 1` | `ring_subfield` |
| `m31_tw_fp4` | `mersenne31`, `2^31 - 1` | `tower` |
| `m31_pw_fp4` | `mersenne31`, `2^31 - 1` | `power` |
| `p31o19_rs_fp4` | `prime31_offset19`, `2^31 - 19` | `ring_subfield` |
| `p31o19_tw_fp4` | `prime31_offset19`, `2^31 - 19` | `tower` |
| `p31o19_pw_fp4` | `prime31_offset19`, `2^31 - 19` | `power` |
| `p32o99_rs_fp4` | `prime32_offset99`, `2^32 - 99` | `ring_subfield` |
| `p32o99_tw_fp4` | `prime32_offset99`, `2^32 - 99` | `tower` |
| `p32o99_pw_fp4` | `prime32_offset99`, `2^32 - 99` | `power` |

Before a canonical refresh, remove stale `target/criterion/field_arith_ext4_*_ring_subfield_fp4_*` directories from earlier long-label runs or start from a fresh `target/criterion`.
Otherwise the collector will warn and de-duplicate what it can.

## Refresh Workflow

Use one saved Criterion baseline per hardware/SIMD cell.
Capture metadata on the same machine that produced the baseline, but use a non-identifying `machine_config` label.
When aggregating copied remote baselines locally, pass the metadata JSON explicitly.

```bash
# NEON (aarch64 Apple M-series)
RUSTFLAGS="-Ctarget-cpu=native" \
  cargo +1.95 bench -p akita-pcs --bench field_arith -- \
  --save-baseline neon 'field_arith/(base|ext4|ext5)/'

python3 scripts/field_microbench_collect.py machine-info \
  --baseline neon \
  --arch aarch64 \
  --simd neon \
  --machine-config apple-m4-max-neon \
  --rustflags=-Ctarget-cpu=native \
  --target-cpu native \
  --bench-filter 'field_arith/(base|ext4|ext5)/' \
  --bench-command 'RUSTFLAGS="-Ctarget-cpu=native" cargo +1.95 bench -p akita-pcs --bench field_arith -- --save-baseline neon field_arith/(base|ext4|ext5)/' \
  --out bench-data/machines/neon.json

# x86_64 AVX2
RUSTFLAGS="-Ctarget-cpu=x86-64-v3" \
  cargo +1.95 bench -p akita-pcs --bench field_arith -- \
  --save-baseline avx2 'field_arith/(base|ext4|ext5)/'

python3 scripts/field_microbench_collect.py machine-info \
  --baseline avx2 \
  --arch x86_64 \
  --simd avx2 \
  --machine-config amd-ryzen-9950x-avx2 \
  --rustflags=-Ctarget-cpu=x86-64-v3 \
  --target-cpu x86-64-v3 \
  --bench-filter 'field_arith/(base|ext4|ext5)/' \
  --bench-command 'RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo +1.95 bench -p akita-pcs --bench field_arith -- --save-baseline avx2 field_arith/(base|ext4|ext5)/' \
  --out bench-data/machines/avx2.json

# x86_64 AVX-512
RUSTFLAGS="-Ctarget-cpu=native" \
  cargo +1.95 bench -p akita-pcs --bench field_arith -- \
  --save-baseline avx512 'field_arith/(base|ext4|ext5)/'

python3 scripts/field_microbench_collect.py machine-info \
  --baseline avx512 \
  --arch x86_64 \
  --simd avx512 \
  --machine-config amd-ryzen-9950x-avx512 \
  --rustflags=-Ctarget-cpu=native \
  --target-cpu native \
  --bench-filter 'field_arith/(base|ext4|ext5)/' \
  --bench-command 'RUSTFLAGS="-Ctarget-cpu=native" cargo +1.95 bench -p akita-pcs --bench field_arith -- --save-baseline avx512 field_arith/(base|ext4|ext5)/' \
  --out bench-data/machines/avx512.json
```

After all baselines and metadata are available in the local worktree:

```bash
python3 scripts/field_microbench_collect.py collect \
  --baseline neon:aarch64:neon \
  --baseline avx2:x86_64:avx2 \
  --baseline avx512:x86_64:avx512 \
  --metadata neon=bench-data/machines/neon.json \
  --metadata avx2=bench-data/machines/avx2.json \
  --metadata avx512=bench-data/machines/avx512.json
```

The collector prints warnings for missing baselines, stale/truncated Criterion groups, duplicate rows, and missing canonical headline rows.
Treat warnings as things to explain before citing the numbers.
