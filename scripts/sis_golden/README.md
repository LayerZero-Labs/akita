# SIS golden reference cells

Offline regression cells for `scripts/gen_sis_table.py` against the pinned
`third_party/lattice-estimator` checkout.

## Setup

```bash
git submodule update --init third_party/lattice-estimator
```

## Refresh golden

Regenerates `golden.csv` and updates `metadata.json` from the grid in `grid.py`:

```bash
sage -python scripts/sis_golden/refresh_golden.py
```

The grid covers q32/q64/q128, `d ∈ {32,64,128,256}`, ranks `{1,5,20}`, and
includes the degenerate knee at `(q32, d=32, collision_l2_sq=16384)`.

## Check

Replay golden cells, rank monotonicity, and secure/insecure brackets:

```bash
sage -python scripts/sis_golden/check.py
```

## Full table regen

Regenerate and stitch every SIS table row with the pinned
`third_party/lattice-estimator` checkout:

```bash
sage -python scripts/stitch_generated_sis_table.py --jobs 6
```

The stitcher uses `--max-rank 20`, passes `--estimator-path
third_party/lattice-estimator` to every shard, and rejects any estimator checkout
whose `HEAD` does not match `metadata.json`.

Manual workflow only. Rust CI does not require Sage or an initialized submodule.

## Infinity-norm goldens

Infinity-norm goldens use the same pinned `third_party/lattice-estimator`
checkout as the Euclidean table generator:

```text
c667a48546f140c3a5454c7503c3ca44a264cce2
```

(malb/lattice-estimator#217; strict descendant of malb#213 @ 27a581b)

Profile:

```text
norm = infinity
red_cost_model = ADPS16
red_shape_model = LGSA
zeta = full optimizer
target_bits = 138
```

Refresh:

```bash
sage -python scripts/sis_golden/refresh_infinity_golden.py
```

Replay:

```bash
sage -python scripts/sis_golden/check_infinity.py
```

For quick local smoke tests, use the same script with filters such as
`--families q32 --dims 32 --ranks 1 --limit 2`.

## Fixed infinity-cost goldens

Slice 3 uses a smaller fixed-beta, fixed-zeta fixture. These cells exercise
`SISLattice.cost_infinity(...)` directly and are separate from the full
optimizer CSV above.

Refresh:

```bash
sage -python scripts/sis_golden/refresh_fixed_infinity_golden.py \
  --estimator-path /path/to/lattice-estimator-pr217
```

Replay:

```bash
sage -python scripts/sis_golden/check_fixed_infinity.py \
  --estimator-path /path/to/lattice-estimator-pr217
```

Benchmark the Rust fixed-cell estimator:

```bash
cargo bench -p akita-sis-estimator --bench fixed_infinity
```

By default the bench runs a representative subset of fixed infinity cells. To
bench a custom fixed-cell grid, point `AKITA_SIS_FIXED_INFINITY_BENCH_CSV` at a
CSV with the fixed golden columns `family`, `d`, `rank`, `width`,
`coeff_linf_bound`, `beta_input`, and `zeta_input`. The committed fixture works
as a full trusted-cell input:

```bash
AKITA_SIS_FIXED_INFINITY_BENCH_CSV=scripts/sis_golden/fixed_infinity_golden.csv \
  cargo bench -p akita-sis-estimator --bench fixed_infinity
```

Benchmark the Rust optimizer paths with Criterion:

```bash
cargo bench -p akita-sis-estimator --bench infinity_optimizer
```

By default the optimizer bench uses a representative trusted-row ladder from
`scripts/sis_golden/infinity_golden.csv` and runs the serial local-minimum and
serial exhaustive profiles. With `--features parallel`, it also runs the
parallel exhaustive profile in the same Criterion group:

```bash
cargo bench -p akita-sis-estimator --features parallel --bench infinity_optimizer
```

The durable benchmark controls are environment variables:

| Variable | Values | Default |
|---|---|---|
| `AKITA_SIS_INFINITY_BENCH_SET` | `representative`, `exhaustive-ci`, `all-trusted` | `representative` |
| `AKITA_SIS_INFINITY_BENCH_PROFILES` | comma-separated `local-minimum`, `exhaustive-serial`, `exhaustive-parallel` | serial profiles, plus parallel when the feature is enabled |
| `AKITA_SIS_INFINITY_BENCH_CSV` | CSV with `family`, `d`, `rank`, `width`, `coeff_linf_bound` columns | committed infinity golden CSV |
| `AKITA_SIS_INFINITY_BENCH_SAMPLE_SIZE` | Criterion sample size, minimum 10 | Criterion default |
| `AKITA_SIS_INFINITY_BENCH_WARM_UP_MS` | Criterion warm-up milliseconds | Criterion default |
| `AKITA_SIS_INFINITY_BENCH_MEASUREMENT_MS` | Criterion measurement milliseconds | Criterion default |

The committed fixture works as a full trusted-cell input:

```bash
AKITA_SIS_INFINITY_BENCH_SET=all-trusted \
  cargo bench -p akita-sis-estimator --features parallel --bench infinity_optimizer
```

For Rust-vs-Sage single-shot timing, run:

```bash
sage -python scripts/sis_golden/bench_infinity.py \
  --estimator-path /path/to/lattice-estimator-pr217
```

Add `--case label:family:d:rank:width:coeff_linf_bound` to benchmark specific
trusted golden rows without Criterion's repeated sampling loop.
