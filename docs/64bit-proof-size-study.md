# Hachi Proof Size Study: 64-bit Prime Field

This note is the corrected 64-bit study after the same `A`-role planner fix
used in the 128-bit and 32-bit analyses.

The earlier draft compared 64-bit schedules against optimistic 128-bit numbers.
That is no longer the right reference point. With the corrected `A` filter,
64-bit still wins over 128-bit, but by a smaller and more believable margin.

## Methodology

The 64-bit planner now uses:

- direct digit-difference bounds for `B` and `D`
- a challenge-aware collision proxy for `A`
- expanded SIS tables for the larger `A` buckets

The script is:

- [scripts/hachi_64bit_proof_planner.py](/Users/quang.dao/Documents/SNARKs/hachi/scripts/hachi_64bit_proof_planner.py)

## Corrected Summary

### Onehot

| nv | Total (B) | D schedule | Tail (B) |
| ---: | ---: | --- | ---: |
| 20 | 47,520 | `64->64->64->64` | 34,896 |
| 25 | 54,272 | `64->64->64->64->64->64` | 34,128 |
| 30 | 57,168 | `64->64->64->64->64->64` | 35,760 |
| 32 | 58,976 | `64->64->64->64->64->64->64` | 33,648 |
| 38 | 62,304 | `128->64->64->64->64->64->64` | 35,472 |
| 44 | 66,464 | `128->64->64->64->64->64->64->64` | 34,128 |

### Dense / Full-field

| nv | Total (B) | D schedule | Tail (B) |
| ---: | ---: | --- | ---: |
| 20 | 56,032 | `64->64->64->64->64->64` | 33,552 |
| 25 | 57,168 | `64->64->64->64->64->64` | 35,760 |
| 30 | 60,320 | `64->64->64->64->64->64->64` | 34,320 |
| 32 | 61,344 | `128->64->64->64->64->64->64` | 35,184 |

## Comparison to Corrected 128-bit Planner

### Onehot

| nv | 64-bit (B) | Corrected 128-bit (B) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 47,520 | 64,224 | 0.74x |
| 25 | 54,272 | 70,736 | 0.77x |
| 30 | 57,168 | 74,800 | 0.76x |
| 32 | 58,976 | 75,632 | 0.78x |
| 38 | 62,304 | 78,896 | 0.79x |
| 44 | 66,464 | 83,184 | 0.80x |

### Dense

| nv | 64-bit (B) | Corrected 128-bit (B) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 56,032 | 76,128 | 0.74x |
| 25 | 57,168 | 75,264 | 0.76x |
| 30 | 60,320 | 77,760 | 0.78x |
| 32 | 61,344 | 78,896 | 0.78x |

64-bit remains meaningfully smaller than corrected 128-bit, but it is no longer
the near-tie with 32-bit that the earlier optimistic comparison suggested.

## Detailed Breakdowns

### Onehot nv=32

```text
L0: D=64 lb=2 m=15 r=11 [D64-na2]
    na=2 nb=2 nd=2  do=33 df=10 dc=1
    w_ring=530,696  next_w=33,964,544  level=4,672B

L1: D=64 lb=2 m=12 r=8 [D64-na2]
    na=2 nb=2 nd=2  do=33 df=8 dc=1
    w_ring=42,200  next_w=2,700,800  level=4,352B

L2: D=64 lb=2 m=10 r=6 [D64-na2]
    na=2 nb=2 nd=1  do=33 df=7 dc=1
    w_ring=11,187  next_w=715,968  level=3,680B

L3: D=64 lb=2 m=9 r=5 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=7 dc=1
    w_ring=4,727  next_w=302,528  level=3,088B

L4: D=64 lb=2 m=9 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=2,997  next_w=191,808  level=3,008B

L5: D=64 lb=2 m=8 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=2,349  next_w=150,336  level=3,008B

L6: D=64 lb=2 m=8 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=2,103  next_w=134,592  level=3,008B

Tail: w_len=134,592  lb=2  tail=33,648B
Total: 58,976 B
```

### Dense nv=32

```text
L0: D=128 lb=2 m=12 r=13 [D128-na1]
    na=1 nb=1 nd=1  do=33 df=10 dc=33
    w_ring=1,892,517  next_w=242,242,176  level=5,344B

L1: D=64 lb=2 m=13 r=9 [D64-na2]
    na=2 nb=2 nd=2  do=33 df=9 dc=1
    w_ring=117,489  next_w=7,519,296  level=4,432B

L2: D=64 lb=2 m=11 r=6 [D64-na2]
    na=2 nb=2 nd=1  do=33 df=7 dc=1
    w_ring=19,419  next_w=1,242,816  level=3,760B

L3: D=64 lb=2 m=9 r=6 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=7 dc=1
    w_ring=6,517  next_w=417,088  level=3,088B

L4: D=64 lb=2 m=9 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=3,669  next_w=234,816  level=3,008B

L5: D=64 lb=2 m=8 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=2,601  next_w=166,464  level=3,008B

L6: D=64 lb=2 m=8 r=4 [D64-na1]
    na=1 nb=1 nd=1  do=33 df=6 dc=1
    w_ring=2,199  next_w=140,736  level=3,008B

Tail: w_len=140,736  lb=2  tail=35,184B
Total: 61,344 B
```

## Interpretation

The corrected 64-bit planner keeps almost all recursive work at `D=64`, with
an occasional `D=128` root for the larger cases. That is a more conservative
shape than the old draft, and it fits the corrected `A` filter much better.

The practical story is now:

- 64-bit is clearly better than corrected 128-bit
- 32-bit is still smaller than 64-bit
- the spread between field sizes is driven mostly by the tail and opening
  decomposition depth, not by dramatic changes in the number of levels

## Mixed Boolean-Step Follow-Up

The later boolean experiment gives 64-bit a smaller but still real win:

| Polynomial | nv | Balanced | Mixed boolean |
| --- | ---: | ---: | ---: |
| onehot | 20 | `46.4 KB` | `46.4 KB` |
| onehot | 25 | `53.0 KB` | `51.8 KB` |
| onehot | 30 | `55.8 KB` | `55.4 KB` |
| onehot | 32 | `57.6 KB` | `56.0 KB` |
| dense | 20 | `54.7 KB` | `54.7 KB` |
| dense | 25 | `55.8 KB` | `55.8 KB` |
| dense | 30 | `58.9 KB` | `58.7 KB` |
| dense | 32 | `59.9 KB` | `59.2 KB` |

The mixed schedules again use boolean only at the top. For onehot `nv = 32`,
the schedule stays all-`64`, but the top three levels become boolean before the
planner drops back to balanced `lb = 2`.

This follow-up confirms that boolean steps help most when the base field is
already fairly small. 64-bit improves, but it still sits above both `32b-bool`
and `16b-bool`, and also above the strongest packed threshold-prime variants
`k7-pack-bool` and `k6-pack-bool`.

## Reproducibility

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi
python3 scripts/hachi_64bit_proof_planner.py
python3 scripts/hachi_proof_planner.py --field 64 --poly both --nv 20,25,30,32 --include-exp-bool
```
