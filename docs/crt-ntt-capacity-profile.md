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
| Q64/3xi32 | production | 3 | i32 | 2^64 - 59 | `1073692673, 1073668097, 1073655809` | 90.00 |
| Q128/6xi32 | portable production | 6 | i32 | 2^128 - 2^32 + 22537 | `1073707009, 1073698817, 1073692673, 1073682433, 1073668097, 1073655809` | 180.00 |
| Q128/3xu64-IFMA52 | AVX-512 base cache | 3 | u64 | 2^128 - 2^32 + 22537 | `1125899906826241, 1125899906629633, 1125899905744897` | 150.00 |
| Q128/3xu64+1xi32-IFMA52 | AVX-512 hybrid exact cache | 4 | 3xu64+1xi32 | 2^128 - 2^32 + 22537 | `1125899906826241, 1125899906629633, 1125899905744897, 1073707009` | 180.00 |

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
| Q64/3xi32 | 3 | i32 | 2048 | 255 | 255 | 0 |
| Q128/6xi32 | 6 | i32 | 64 | 549,578,630,967 | 549,578,630,967 | 2,146,791,527 |
| Q128/6xi32 | 6 | i32 | 128 | 274,789,315,483 | 274,789,315,483 | 1,073,395,763 |
| Q128/6xi32 | 6 | i32 | 256 | 137,394,657,741 | 137,394,657,741 | 536,697,881 |
| Q128/6xi32 | 6 | i32 | 512 | 68,697,328,870 | 68,697,328,870 | 268,348,940 |
| Q128/6xi32 | 6 | i32 | 1024 | 34,348,664,435 | 34,348,664,435 | 134,174,470 |
| Q128/3xu64-IFMA52 | 3 | u64 | 64 | 511 | 511 | 1 |
| Q128/3xu64-IFMA52 | 3 | u64 | 128 | 255 | 255 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 256 | 127 | 127 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 512 | 63 | 63 | 0 |
| Q128/3xu64-IFMA52 | 3 | u64 | 1024 | 31 | 31 | 0 |
| Q128/3xu64+1xi32-IFMA52 | 4 | 3xu64+1xi32 | 64 | 549,737,987,960 | 549,737,987,960 | 2,147,414,015 |
| Q128/3xu64+1xi32-IFMA52 | 4 | 3xu64+1xi32 | 128 | 274,868,993,980 | 274,868,993,980 | 1,073,707,007 |
| Q128/3xu64+1xi32-IFMA52 | 4 | 3xu64+1xi32 | 256 | 137,434,496,990 | 137,434,496,990 | 536,853,503 |
| Q128/3xu64+1xi32-IFMA52 | 4 | 3xu64+1xi32 | 512 | 68,717,248,495 | 68,717,248,495 | 268,426,751 |
| Q128/3xu64+1xi32-IFMA52 | 4 | 3xu64+1xi32 | 1024 | 34,358,624,247 | 34,358,624,247 | 134,213,375 |

## Q128 Balanced-Digit Capacity

The portable and hybrid AVX-512 exact products are both about 180 bits.
The hybrid retains three hot IFMA limbs and adds one 30-bit tail only for
rows that exceed the roughly 150-bit IFMA base product.

| Representation | D | log basis 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q128/6xi32 | 64 | 17,586,516,190,962 | 8,793,258,095,481 | 4,396,629,047,740 | 2,198,314,523,870 | 1,099,157,261,935 | 549,578,630,967 |
| Q128/6xi32 | 128 | 8,793,258,095,481 | 4,396,629,047,740 | 2,198,314,523,870 | 1,099,157,261,935 | 549,578,630,967 | 274,789,315,483 |
| Q128/6xi32 | 256 | 4,396,629,047,740 | 2,198,314,523,870 | 1,099,157,261,935 | 549,578,630,967 | 274,789,315,483 | 137,394,657,741 |
| Q128/6xi32 | 512 | 2,198,314,523,870 | 1,099,157,261,935 | 549,578,630,967 | 274,789,315,483 | 137,394,657,741 | 68,697,328,870 |
| Q128/6xi32 | 1024 | 1,099,157,261,935 | 549,578,630,967 | 274,789,315,483 | 137,394,657,741 | 68,697,328,870 | 34,348,664,435 |
| Q128/3xu64-IFMA52 | 64 | 16,383 | 8,191 | 4,095 | 2,047 | 1,023 | 511 |
| Q128/3xu64-IFMA52 | 128 | 8,191 | 4,095 | 2,047 | 1,023 | 511 | 255 |
| Q128/3xu64-IFMA52 | 256 | 4,095 | 2,047 | 1,023 | 511 | 255 | 127 |
| Q128/3xu64-IFMA52 | 512 | 2,047 | 1,023 | 511 | 255 | 127 | 63 |
| Q128/3xu64-IFMA52 | 1024 | 1,023 | 511 | 255 | 127 | 63 | 31 |
| Q128/3xu64+1xi32-IFMA52 | 64 | 17,591,615,614,720 | 8,795,807,807,360 | 4,397,903,903,680 | 2,198,951,951,840 | 1,099,475,975,920 | 549,737,987,960 |
| Q128/3xu64+1xi32-IFMA52 | 128 | 8,795,807,807,360 | 4,397,903,903,680 | 2,198,951,951,840 | 1,099,475,975,920 | 549,737,987,960 | 274,868,993,980 |
| Q128/3xu64+1xi32-IFMA52 | 256 | 4,397,903,903,680 | 2,198,951,951,840 | 1,099,475,975,920 | 549,737,987,960 | 274,868,993,980 | 137,434,496,990 |
| Q128/3xu64+1xi32-IFMA52 | 512 | 2,198,951,951,840 | 1,099,475,975,920 | 549,737,987,960 | 274,868,993,980 | 137,434,496,990 | 68,717,248,495 |
| Q128/3xu64+1xi32-IFMA52 | 1024 | 1,099,475,975,920 | 549,737,987,960 | 274,868,993,980 | 137,434,496,990 | 68,717,248,495 | 34,358,624,247 |

## Dense q128 Commitment Dispatch

Tail presence and kernel selection are separate decisions. The capacity formula
answers whether one exact accumulation needs the tail. It does not justify
materializing an exact cache for digits that already fit in i8.

For balanced digits with log basis 1 through 8:

- Scalar, AVX2, and NEON backends use the portable six-prime chunked i8
  accumulation.
- An AVX-512IFMA backend uses one exact accumulation for a q128 row when the
  three-prime IFMA product is too small but the product with the 30-bit
  tail prime 1073707009 fits.
- All other rows stay chunked. Each chunk reconstructs before the next chunk,
  so the complete row does not need the tail prime.
- The block-parallel kernel still exposes independent blocks to Rayon when the
  workload has enough parallel work.

Log bases 9 through 16 require exact i16 digits on every backend because i8
cannot represent those balanced digits. The tail is still added only when the
exact capacity check requires it.

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
