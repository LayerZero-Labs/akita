# CRT/NTT Capacity Profile

This artifact pins the single-CRT-lift capacity of the prime profiles used by
the prover i8 kernels. Regenerate the table with:

```bash
python3 scripts/gen_crt_capacity_profile.py > docs/crt-ntt-capacity-profile.md
```

The bound is intentionally conservative:

```text
2 * width * D * floor(q / 2) * rhs_abs_bound < product(CRT primes)
```

`balanced32` is the maximum supported balanced i8 digit bound for
`log_basis = 6`. `raw128` is the raw signed-i8 recursive-witness bound.
`zpre32768` is included to document when fused split-eq must use its exact
fallback for centered `z_pre` values; zero means one centered term does not fit.

## Profiles

| Profile | Role | K | Limb | q | Primes | log2(P_crt) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Q32-reference/4xi16 | comparison only | 4 | i16 | 2^32 - 99 | `15361, 13313, 12289, 11777` | 54.72 |
| Q32/2xi32 | production | 2 | i32 | 2^32 - 99 | `1073707009, 1073698817` | 60.00 |
| Q64/3xi32 | production | 3 | i32 | 2^64 - 59 | `1073707009, 1073698817, 1073692673` | 90.00 |
| Q128/5xi32 | production | 5 | i32 | 2^128 - 275 | `1073707009, 1073698817, 1073692673, 1073682433, 1073668097` | 150.00 |

## Safe Widths

| Profile | K | Limb | D | balanced32 | raw128 | zpre32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q32-reference/4xi16 | 4 | i16 | 32 | 6,729 | 1,682 | 6 |
| Q32-reference/4xi16 | 4 | i16 | 64 | 3,364 | 841 | 3 |
| Q32-reference/4xi16 | 4 | i16 | 128 | 1,682 | 420 | 1 |
| Q32-reference/4xi16 | 4 | i16 | 256 | 841 | 210 | 0 |
| Q32/2xi32 | 2 | i32 | 32 | 262,125 | 65,531 | 255 |
| Q32/2xi32 | 2 | i32 | 64 | 131,062 | 32,765 | 127 |
| Q32/2xi32 | 2 | i32 | 128 | 65,531 | 16,382 | 63 |
| Q32/2xi32 | 2 | i32 | 256 | 32,765 | 8,191 | 31 |
| Q64/3xi32 | 3 | i32 | 32 | 65,528 | 16,382 | 63 |
| Q64/3xi32 | 3 | i32 | 64 | 32,764 | 8,191 | 31 |
| Q64/3xi32 | 3 | i32 | 128 | 16,382 | 4,095 | 15 |
| Q64/3xi32 | 3 | i32 | 256 | 8,191 | 2,047 | 7 |
| Q128/5xi32 | 5 | i32 | 32 | 4,095 | 1,023 | 3 |
| Q128/5xi32 | 5 | i32 | 64 | 2,047 | 511 | 1 |
| Q128/5xi32 | 5 | i32 | 128 | 1,023 | 255 | 0 |
| Q128/5xi32 | 5 | i32 | 256 | 511 | 127 | 0 |

## Q32 Experiment

`Q32/2xi32` is the production Q32 profile. A local release microbenchmark
compared it against the `Q32-reference/4xi16` profile used during design:

| Variant | Round trip ns/iter | i8 mul-lift ns/iter |
| --- | ---: | ---: |
| Q32-reference/4xi16 | 2,587.14 | 2,090.77 |
| Q32/2xi32 | 1,044.49 | 876.62 |

Both variants have the same per-coefficient CRT limb footprint (8 bytes),
but `Q32/2xi32` halves the prime count and has substantially larger capacity.
The reference `4xi16` row remains here only as experiment evidence.

The production profiles all have nonzero `balanced32` and `raw128` widths at
`D in {32, 64, 128, 256}`. The `zpre32768 = 0` entries are acceptable because
the fused split-eq path has an exact fallback for centered `z_pre`.
