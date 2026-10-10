# NIST category 3 SIS policy evidence

These files compare the complete generated schedule catalog at `main`
`60b82bc30f4f8f1130ccd71fe10b0b9bed909ab9` (128-bit policy) with the
`Quantum169BitADPS16` branch.

- `base.tsv` is the tracked `artifacts/schedule-catalog.tsv` snapshot at the
  base commit.
- `head.tsv` is the regenerated snapshot, identical to the branch's
  `artifacts/schedule-catalog.tsv`.
- `comparison.tsv` is the `--catalog-report` union of both: lookup and row
  digests, setup fields, first-direct capacity, proof bytes, and fold levels.
- `catalog-security.tsv` is `catalog_security --check` on the branch: the
  cheapest modeled SIS attack in every row.

Regenerate the comparison with:

```text
gen_schedule_artifacts artifacts/schedules \
  --catalog-baseline base.tsv \
  --catalog-report comparison.tsv \
  --catalog-snapshot head.tsv
```

## Catalog summary

All 115 rows plan under the 169-bit policy; none became unsupported. Every
row's weakest SIS instance costs exactly 169.07 bits, the BKZ block size 638
boundary.

| Metric | Result |
| --- | --- |
| Proof bytes, 111 rows | +4.2% to +25.2%, median +14.9% (+10,368 bytes) |
| Proof bytes, 4 small recursive rows | +97% to +250% (see below) |
| Total setup fields | median x1.14, range x0.69 to x2.67 |
| Fold levels (111 rows) | unchanged in 64, one more in 43, fewer in 4 |

Per-family medians fall between +10% and +17%. Higher A, B, and D ranks
enlarge the witness that every later fold and the terminal response carry,
and 43 rows need one more fold. Each compressed level also carries two
256-byte terminal images instead of two 128-byte ones. In the measured fp128
dense nv28 proof, the terminal response grows by 3,453 bytes and the fold
levels by 8,032 bytes.

### Small recursive rows

`fp128_dense_recursive` at nv20 and `fp32_dense_recursive` at nv20, nv22,
and nv24 select three-level schedules with 191 KB to 317 KB proofs. Recursive
families minimize the padded setup envelope before proof bytes. Under the
169-bit tables the deep schedules no longer fit the base setup bucket, so the
planner keeps the bucket and accepts a shallow schedule. `fp32_dense_recursive`
nv20 already made the same trade on `main` (161 KB). A larger setup bucket
would restore a deep schedule; the catalog objective is unchanged here.

## Measured profile benchmark

The profile-bench workflow cases ran on one AMD Ryzen 9 9950X (32 threads,
`-C target-cpu=x86-64-v3`, features `parallel,profile-ci,transcript-blake2b`).
The `main` and branch binaries ran interleaved: one warmup and five measured
processes per side and case. The table shows medians, `main -> branch`. Every
case proved and verified on both sides.

| Case | Proof bytes | Commit (s) | Prove (s) | Verify, 32 threads (ms) | Verify, 1 thread (ms) | Setup fields | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 dense nv30 | 64,548 → 75,294 (+17%) | 0.73 → 0.92 (+26%) | 0.61 → 0.73 (+19%) | 9.3 → 8.9 (-4%) | 26.9 → 23.4 (-13%) | -9% | 7 → 7 |
| fp32 one-hot nv34 | 65,507 → 75,398 (+15%) | 0.45 → 0.45 (-2%) | 0.71 → 0.83 (+17%) | 10.7 → 10.1 (-5%) | 35.6 → 41.8 (+17%) | +100% | 7 → 7 |
| fp64 dense nv29 | 67,091 → 78,244 (+17%) | 0.95 → 0.95 (-0%) | 0.62 → 0.59 (-5%) | 6.2 → 6.3 (+2%) | 17.8 → 17.5 (-2%) | +0% | 7 → 7 |
| fp64 one-hot nv35 | 67,711 → 78,913 (+17%) | 0.74 → 1.05 (+42%) | 0.69 → 0.85 (+23%) | 9.0 → 10.0 (+11%) | 34.5 → 33.5 (-3%) | +0% | 7 → 8 |
| fp128 dense nv28 | 69,062 → 80,581 (+17%) | 0.92 → 0.92 (+1%) | 0.62 → 0.59 (-4%) | 7.8 → 7.3 (-6%) | 25.1 → 21.7 (-14%) | +0% | 7 → 7 |
| fp128 one-hot nv36 | 69,700 → 81,777 (+17%) | 2.78 → 2.81 (+1%) | 1.28 → 1.32 (+3%) | 18.1 → 17.7 (-2%) | 66.1 → 72.5 (+10%) | +34% | 7 → 8 |
| fp128 one-hot nv36, recursive setup | 73,154 → 84,962 (+16%) | 2.77 → 3.25 (+17%) | 1.52 → 1.48 (-3%) | 10.0 → 10.0 (-0%) | 28.1 → 22.2 (-21%) | +100% | 8 → 8 |
| fp128 multi-group 4 polys, nv34 | 69,619 → 81,624 (+17%) | 1.08 → 1.43 (+32%) | 0.66 → 0.76 (+15%) | 13.4 → 13.9 (+4%) | 53.8 → 53.2 (-1%) | +0% | 7 → 8 |
| fp128 multi-group nv34, recursive | 86,034 → 89,769 (+4%) | 1.40 → 1.41 (+1%) | 1.06 → 1.08 (+2%) | 9.2 → 9.0 (-2%) | 18.0 → 18.8 (+4%) | +0% | 8 → 8 |
| fp128 multi-group nv32, recursive W8R2 | 75,253 → 83,769 (+11%) | 0.46 → 0.32 (-30%) | 1.07 → 1.23 (+15%) | 12.3 → 13.3 (+8%) | 33.2 → 38.8 (+17%) | +0% | 8 → 8 |
| fp128 one-hot nv32 W2R2 | 69,515 → 81,554 (+17%) | 0.20 → 0.19 (-2%) | 0.33 → 0.34 (+3%) | 8.1 → 8.7 (+6%) | 22.1 → 22.6 (+2%) | +0% | 7 → 8 |
| fp128 one-hot nv32 W4R2 | 70,509 → 81,782 (+16%) | 0.19 → 0.20 (+3%) | 0.44 → 0.45 (+2%) | 9.0 → 9.5 (+6%) | 25.5 → 26.9 (+6%) | +0% | 7 → 8 |
| fp128 one-hot nv32 W8R2 | 71,887 → 82,329 (+15%) | 0.19 → 0.19 (+1%) | 0.62 → 0.62 (+1%) | 10.3 → 10.4 (+1%) | 31.7 → 31.2 (-2%) | +0% | 8 → 8 |

Across the 13 cases the median change is +16.5% in proof bytes, +2.7% in prove
time, +1% in commit time, and under 1.5% in either verify time. Sample ranges
are a few percent wide, so the larger prover deltas are real. The fp64 one-hot
and fp128 multi-group rows add a fold; fp32 dense (+19% prove, +26% commit)
and the recursive W8R2 row (+15% prove) keep their fold count and were not
profiled further. The fp128 one-hot nv36 and recursive-setup rows grow their
setup by 34% and 100%, which raises setup time and peak memory (+49% and +93%
RSS).
