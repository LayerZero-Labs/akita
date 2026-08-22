# CRT/NTT Capacity Profile

This artifact pins the single-CRT-lift capacity of Akita's commitment kernels
and records the evidence behind dense q128 kernel dispatch. It includes the
portable 30-bit profiles and the 50-bit AVX-512IFMA exact representation.
Regenerate the table with:

```bash
python3 scripts/gen_crt_capacity_profile.py > docs/crt-ntt-capacity-profile.md
```

The bound is intentionally conservative:

```text
2 * width * D * floor(q / 2) * rhs_abs_bound < product(CRT primes)
```

`balanced128` is the maximum supported balanced i8 digit bound for
`log_basis = 8`. `raw128` is the raw signed-i8 recursive-witness bound.
`zpre32768` is included to document when fused split-eq must use its exact
fallback for centered `z_pre` values; zero means one centered term does not fit.

## Profiles

| Profile | Role | K | Limb | q | Primes | log2(P_crt) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Q32-reference/4xi16 | comparison only | 4 | i16 | 2^32 - 99 | `15361, 13313, 12289, 11777` | 54.72 |
| Q32/2xi32 | production | 2 | i32 | 2^32 - 99 | `1073692673, 1073668097` | 60.00 |
| Q64/3xi32 | production | 3 | i32 | 2^64 - 59 | `1073692673, 1073668097, 1073707009` | 90.00 |
| Q128/5xi32 | portable production | 5 | i32 | 2^128 - 2^32 + 22537 | `1073692673, 1073668097, 1073707009, 1073738753, 1073732609` | 150.00 |
| Q128/3xu64-IFMA52 | AVX-512 exact cache | 3 | u64 | 2^128 - 2^32 + 22537 | `1125899906826241, 1125899906629633, 1125899905744897` | 150.00 |

## Safe Widths

| Profile | K | Limb | D | balanced128 | raw128 | zpre32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q32-reference/4xi16 | 4 | i16 | 64 | 841 | 841 | 3 |
| Q32-reference/4xi16 | 4 | i16 | 128 | 420 | 420 | 1 |
| Q32-reference/4xi16 | 4 | i16 | 256 | 210 | 210 | 0 |
| Q32/2xi32 | 2 | i32 | 64 | 32,764 | 32,764 | 127 |
| Q32/2xi32 | 2 | i32 | 128 | 16,382 | 16,382 | 63 |
| Q32/2xi32 | 2 | i32 | 256 | 8,191 | 8,191 | 31 |
| Q32/2xi32 | 2 | i32 | 512 | 4,095 | 4,095 | 15 |
| Q32/2xi32 | 2 | i32 | 1024 | 2,047 | 2,047 | 7 |
| Q32/2xi32 | 2 | i32 | 2048 | 1,023 | 1,023 | 3 |
| Q64/3xi32 | 3 | i32 | 64 | 8,190 | 8,190 | 31 |
| Q64/3xi32 | 3 | i32 | 128 | 4,095 | 4,095 | 15 |
| Q64/3xi32 | 3 | i32 | 256 | 2,047 | 2,047 | 7 |
| Q64/3xi32 | 3 | i32 | 512 | 1,023 | 1,023 | 3 |
| Q64/3xi32 | 3 | i32 | 1024 | 511 | 511 | 1 |
| Q128/5xi32 | 5 | i32 | 64 | 511 | 511 | 1 |
| Q128/5xi32 | 5 | i32 | 128 | 255 | 255 | 0 |
| Q128/5xi32 | 5 | i32 | 256 | 127 | 127 | 0 |
| Q128/5xi32 | 5 | i32 | 512 | 63 | 63 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 64 | 511 | 511 | 1 |
| Q128/3xu64-IFMA52 | 3 | u64 | 128 | 255 | 255 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 256 | 127 | 127 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 512 | 63 | 63 | 0 |

## Q128 Balanced-Digit Capacity

The base CRT products for portable and AVX-512 exact accumulation are both
about 150 bits. Their mathematical thresholds are therefore almost the same.
A row above the listed width needs the 14-bit tail prime `12289` if it is
accumulated exactly in one pass.

| Representation | D | log basis 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q128/5xi32 | 64 | 16,381 | 8,190 | 4,095 | 2,047 | 1,023 | 511 |
| Q128/5xi32 | 128 | 8,190 | 4,095 | 2,047 | 1,023 | 511 | 255 |
| Q128/5xi32 | 256 | 4,095 | 2,047 | 1,023 | 511 | 255 | 127 |
| Q128/5xi32 | 512 | 2,047 | 1,023 | 511 | 255 | 127 | 63 |
| Q128/3xu64-IFMA52 | 64 | 16,383 | 8,191 | 4,095 | 2,047 | 1,023 | 511 |
| Q128/3xu64-IFMA52 | 128 | 8,191 | 4,095 | 2,047 | 1,023 | 511 | 255 |
| Q128/3xu64-IFMA52 | 256 | 4,095 | 2,047 | 1,023 | 511 | 255 | 127 |
| Q128/3xu64-IFMA52 | 512 | 2,047 | 1,023 | 511 | 255 | 127 | 63 |

## Dense q128 Commitment Dispatch

Tail presence and kernel selection are separate decisions. The capacity formula
answers whether one exact accumulation needs the tail. It does not justify
materializing an exact cache for digits that already fit in i8.

For balanced digits with log basis 1 through 8:

- Scalar, AVX2, and NEON backends use the portable five-prime chunked i8
  accumulation.
- An AVX-512IFMA backend uses one exact accumulation for a q128 row when the
  three-prime IFMA product is too small but the product with 12289 fits.
- All other rows stay chunked. Each chunk reconstructs before the next chunk,
  so the complete row does not need the tail prime.
- The block-parallel kernel still exposes independent blocks to Rayon when the
  workload has enough parallel work.

Log bases 9 through 16 require exact i16 digits on every backend because i8
cannot represent those balanced digits. The tail is still added only when the
exact capacity check requires it.

### Production root shapes

| Variables | D | Log basis | Live blocks | Complete row width | Portable aligned chunk | Chunks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 26 | 256 | 7 | 256 | 4,864 | 247 | 20 |
| 28 | 512 | 7 | 512 | 9,728 | 114 | 86 |
| 30 | 512 | 7 | 1,024 | 19,456 | 114 | 171 |

All three complete rows exceed the base capacity and fit after adding 12289.
They therefore use exact IFMA52 accumulation on eligible AVX-512 hosts and the
listed bounded chunks on scalar, AVX2, and NEON hosts.

### Measurement evidence

Measurements were collected on 2026-08-21 from the PR 430 optimization branch.
The measurements use the same production q128 schedule before and after the
candidate exact-tail dispatch. Each current AVX-512 result is the median of
three interleaved samples after one warmup.

| Backend and workload | Chunked i8 commit | Exact commit | Change |
| --- | ---: | ---: | ---: |
| Apple NEON, q128 nv26 | 1.163 s | 2.489 s | 114% slower |
| Zen 5 AVX-512IFMA, q128 nv26 | 1.296 s | 1.052 s | 18.8% faster |
| Zen 5 AVX-512IFMA, q128 nv28 | 5.158 s | 3.562 s | 30.9% faster |
| Zen 5 AVX-512IFMA, q128 nv30 | 20.173 s | 14.359 s | 28.8% faster |
| Zen 5 AVX2, q128 nv26 to nv30 | baseline | candidate | 29% to 32% faster |
| Hosted 32-thread AVX2, q128 nv28 | baseline | candidate | 16.8% slower |

On the AVX-512 run, exact accumulation increased setup by 0.051 s, 0.101 s,
and 0.239 s at nv26, nv28, and nv30. Prepared-cache bytes increased by 29% to
32%, while peak RSS increased by 1.6% to 5.5%. The durable policy therefore
selects this trade only from AVX-512IFMA capability and the exact capacity
bound. It does not dispatch on a host name or a particular variable count.

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

The portable production profiles all have nonzero `balanced128` and `raw128` widths
at every supported commitment ring dimension. The `zpre32768 = 0` entries
are acceptable because the fused split-eq path has an exact fallback for
centered `z_pre`.
