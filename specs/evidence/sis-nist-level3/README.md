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
| Total setup fields, 111 rows | median x1.14, range x0.54 to x2.67 |
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

### Slice-pruning fix

Under the 169-bit tables, the oracle comparison for the fp128 one-hot W8 nv20
fixture failed. At every level and under both objectives, the planner kept
only the outer slice counts with the smallest padded setup. That discarded
the 256-position single-slice first fold that the exhaustive reference picks,
which cost 116 proof bytes. The pruning is exact only at the root under the
first-direct setup objective. This branch keeps every recursive slice count
and prunes root counts only under that objective. The comparison above
includes the fix.

On the 169-bit catalog the fix changes 10 artifact files. Comparing the exact
objective of all 123 planning calls (catalog rows and precommit preplans)
before and after the fix: 105 are identical and 11 improve. The other six
fall into two groups:

- `fp128_onehot_multi_chunk` and `fp128_onehot_multi_chunk_w8r3` at nv16, and
  the two W8 preplans with the same layouts, halve their first-direct setup
  capacity from 262,144 to 131,072. That is the leading objective
  coordinate. Their proofs grow by 1,633 and 2,106 bytes.
- `fp128_onehot_multi_chunk` nv34 and `fp64_dense` nv20, each with
  precommits, plan against a changed producer. A grouped row freezes the
  root of its standalone producer row as the precommit descriptor. When that
  row improves, the grouped row's lookup key changes. Replanning the pre-fix
  keys with the fixed planner reproduces the pre-fix schedules exactly.

On `main` alone the fix changes 18 rows. The same producer effect adds 4,644
bytes to `fp128_onehot_multi_chunk_w2r2` nv14 with one nv14 precommit. Its
producer row saves one nonce byte with a different root matrix. Replanning
`main`'s key with the fixed planner reproduces `main`'s schedule exactly.

## Jolt-scale planning coverage

Jolt plans its own schedule artifacts with this planner. It uses fp128 one-hot
policies with K=16 or K=256 sources, each with single, W2R2, W4R2, or W8R2
chunking and recursive setup planning, plus bounded and full dense producers
with W8R2 chunking. The ignored test
`jolt_scale_shapes_plan_for_every_chunk_profile` builds the same policies from
`policy_of::<RecursiveCommitmentConfig<OneHot>>()`, `DenseBounded`, and
`Dense`, swapping in the K=16 source contract and each chunk profile. It plans
190 shapes at trace lengths `2^30` and `2^31`. Every shape plans and passes
`validate_structure`:

```text
cargo test --release -p akita-planner --lib --features catalog-gen -- \
  jolt_scale_shapes_plan_for_every_chunk_profile --ignored --nocapture
```

| Shapes (every chunk profile) | Proof bytes | Folds | Setup field elements |
| --- | ---: | ---: | ---: |
| K=16 trace, nv34 (`2^30`), 51 to 64 polys | 92,034 to 95,546 | 7 to 8 | `2^27` to `2^28` |
| K=16 trace, nv35 (`2^31`), 51 to 64 polys | 92,070 to 95,822 | 8 | `2^27` to `2^28` |
| K=16, nv40 to nv44, 1 or 2 polys | 92,030 to 103,506 | 7 to 8 | `2^27` to `2^30` |
| K=256 trace, nv38 and nv39, 27 and 37 polys | 92,190 to 103,506 | 8 | `2^28` to `2^30` |
| K=256, nv43 and nv44, 1 or 2 polys | 92,186 to 103,730 | 8 | `2^28` to `2^30` |
| Dense bounded and full, nv34 and nv35 | 88,786 to 88,910 | 7 | at most 90,177,536 |

Each shape plans in 35 to 90 seconds. The check plans schedules only; it
does not prove at these sizes.

## Measured profile benchmark

The profile-bench workflow cases ran on one AMD Ryzen 9 9950X (32 threads,
`-C target-cpu=x86-64-v3`, features `parallel,profile-ci,transcript-blake2b`).
The `main` and branch binaries ran interleaved: one warmup and five measured
processes per side and case. The table shows medians, `main → branch`. Every
case proved and verified on both sides.

| Case | Proof bytes | Commit (s) | Prove (s) | Verify, 32 threads (ms) | Verify, 1 thread (ms) | Setup fields | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 dense nv30 | 64,548 → 75,294 (+17%) | 0.72 → 0.91 (+27%) | 0.62 → 0.72 (+18%) | 9.1 → 8.9 (-2%) | 27.2 → 22.3 (-18%) | -9% | 7 → 7 |
| fp32 one-hot nv34 | 65,507 → 75,398 (+15%) | 0.46 → 0.44 (-4%) | 0.71 → 0.81 (+14%) | 10.5 → 10.1 (-3%) | 35.1 → 41.9 (+19%) | +100% | 7 → 7 |
| fp64 dense nv29 | 67,091 → 78,244 (+17%) | 0.94 → 0.94 (+1%) | 0.65 → 0.59 (-9%) | 6.4 → 6.2 (-3%) | 18.6 → 16.7 (-10%) | +0% | 7 → 7 |
| fp64 one-hot nv35 | 67,711 → 78,913 (+17%) | 0.77 → 1.03 (+34%) | 0.70 → 0.85 (+23%) | 9.3 → 10.4 (+11%) | 34.6 → 32.7 (-5%) | +0% | 7 → 8 |
| fp128 dense nv28 | 69,062 → 80,581 (+17%) | 0.94 → 0.92 (-2%) | 0.61 → 0.59 (-5%) | 8.0 → 7.3 (-9%) | 25.9 → 21.8 (-16%) | +0% | 7 → 7 |
| fp128 one-hot nv36 | 69,700 → 81,777 (+17%) | 2.77 → 2.80 (+1%) | 1.28 → 1.31 (+2%) | 18.0 → 18.0 (-0%) | 65.6 → 71.5 (+9%) | +34% | 7 → 8 |
| fp128 one-hot nv36, recursive setup | 73,154 → 84,962 (+16%) | 2.78 → 3.29 (+18%) | 1.45 → 1.50 (+3%) | 10.1 → 9.9 (-2%) | 28.5 → 23.2 (-19%) | +100% | 8 → 8 |
| fp128 multi-group 4 polys, nv34 | 69,619 → 81,624 (+17%) | 1.08 → 1.42 (+31%) | 0.66 → 0.76 (+15%) | 13.4 → 14.0 (+4%) | 53.8 → 51.4 (-4%) | +0% | 7 → 8 |
| fp128 multi-group nv34, recursive | 86,034 → 89,769 (+4%) | 1.39 → 1.43 (+3%) | 1.06 → 1.07 (+1%) | 9.4 → 9.1 (-3%) | 17.6 → 18.2 (+3%) | +0% | 8 → 8 |
| fp128 multi-group nv32, recursive W8R2 | 75,253 → 83,746 (+11%) | 0.46 → 0.33 (-27%) | 1.06 → 1.21 (+13%) | 12.6 → 13.5 (+7%) | 33.8 → 36.9 (+9%) | +0% | 8 → 8 |
| fp128 one-hot nv32 W2R2 | 69,515 → 81,550 (+17%) | 0.19 → 0.19 (+1%) | 0.33 → 0.34 (+2%) | 8.5 → 8.3 (-2%) | 22.8 → 22.4 (-2%) | +0% | 7 → 8 |
| fp128 one-hot nv32 W4R2 | 70,509 → 81,782 (+16%) | 0.19 → 0.20 (+2%) | 0.43 → 0.45 (+3%) | 9.0 → 9.3 (+3%) | 25.4 → 25.8 (+1%) | +0% | 7 → 8 |
| fp128 one-hot nv32 W8R2 | 71,887 → 82,329 (+15%) | 0.20 → 0.20 (+0%) | 0.62 → 0.62 (+1%) | 10.5 → 10.2 (-3%) | 32.3 → 31.4 (-3%) | +0% | 8 → 8 |

Across the 13 cases the median change is +16.5% in proof bytes, +2.7% in prove
time, and +1.3% in commit time. Verify time is within noise: the medians are
-1.9% with 32 threads and -2.7% single-threaded. The spread of the five prove
and commit samples (maximum minus minimum, over the median) is 3.7% at the
median and at most 13%, so prover deltas above 15% are real. The fp64 one-hot
and fp128 multi-group rows add a fold. Two rows keep their fold count and were
not profiled further: fp32 dense (+18% prove, +27% commit) and the recursive
W8R2 row (+13% prove). The fp128 one-hot nv36 and recursive-setup rows grow
their setup by 34% and 100%, which raises setup time and peak memory (+50% and
+93% RSS).
