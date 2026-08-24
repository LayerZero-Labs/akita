# Hachi Proof Size Study: 32-bit Prime Field

This is the corrected 32-bit study after updating the planner to account for
challenge-aware `A`-role collisions.

Compared with the older draft, the qualitative conclusion survives:
moving to a 32-bit base field still helps. What changed is the framing:
the comparison should now be made against the corrected 128-bit planner, not
against the older optimistic 128-bit numbers.

## Methodology

The 32-bit planner now uses the same `A`-role correction as the 128-bit study:

- `B` and `D` use `collision_inf = 2^lb - 1`
- `A` uses `raw_digit_collision * max_abs_challenge_coeff`
- that requested `A` collision is rounded up to the next available SIS bucket

The script is:

- [scripts/hachi_32bit_proof_planner.py](/Users/quang.dao/Documents/SNARKs/hachi/scripts/hachi_32bit_proof_planner.py)

## Corrected Summary

| Polynomial | nv | Total (B) | D schedule | Tail (B) |
| --- | ---: | ---: | --- | ---: |
| onehot | 20 | 38,960 | `64->64->64->64` | 27,520 |
| onehot | 25 | 45,056 | `128->128->64->64->64->64` | 26,176 |
| onehot | 30 | 48,208 | `128->128->64->64->64->64` | 27,904 |
| onehot | 32 | 49,568 | `128->128->128->64->64->64->64` | 26,176 |
| dense-32bit | 20 | 45,360 | `128->128->64->64->64->64` | 26,080 |
| dense-32bit | 25 | 48,016 | `128->128->64->64->64->64` | 27,712 |
| dense-32bit | 30 | 50,192 | `128->128->128->64->64->64->64` | 26,560 |
| dense-32bit | 32 | 51,904 | `128->128->128->64->64->64->64` | 27,520 |

## Comparison to Corrected 128-bit Planner

### Onehot

| nv | 32-bit (B) | Corrected 128-bit (B) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 38,960 | 64,224 | 0.61x |
| 25 | 45,056 | 70,736 | 0.64x |
| 30 | 48,208 | 74,800 | 0.64x |
| 32 | 49,568 | 75,632 | 0.66x |

### Dense

| nv | 32-bit (B) | Corrected 128-bit (B) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 45,360 | 76,128 | 0.60x |
| 25 | 48,016 | 75,264 | 0.64x |
| 30 | 50,192 | 77,760 | 0.65x |
| 32 | 51,904 | 78,896 | 0.66x |

So the 32-bit field still wins decisively even after correcting the 128-bit
planner. The old draft understated that comparison because it was using
optimistic 128-bit schedules.

## Detailed Breakdowns

### Onehot nv=32

```text
L0: D=128 lb=2 m=14 r=11 [D128-na2]
    na=2 nb=2 nd=2  do=17 df=9 dc=1
    w_ring=252,040  next_w=32,261,120  level=4,592B

L1: D=128 lb=2 m=10 r=8 [D128-na1]
    na=1 nb=2 nd=2  do=17 df=8 dc=1
    w_ring=16,703  next_w=2,137,984  level=4,352B

L2: D=128 lb=2 m=9 r=6 [D128-na1]
    na=1 nb=1 nd=1  do=17 df=7 dc=1
    w_ring=4,088  next_w=523,264  level=3,088B

L3: D=64 lb=2 m=8 r=5 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=7 dc=1
    w_ring=3,560  next_w=227,840  level=2,752B

L4: D=64 lb=2 m=8 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=2,290  next_w=146,560  level=2,752B

L5: D=64 lb=2 m=8 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=1,816  next_w=116,224  level=2,672B

L6: D=64 lb=2 m=7 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=1,636  next_w=104,704  level=2,672B

Tail: w_len=104,704  lb=2  tail=26,176B
Total: 49,568 B
```

### Dense nv=32

```text
L0: D=128 lb=2 m=12 r=13 [D128-na2]
    na=2 nb=2 nd=2  do=17 df=10 dc=17
    w_ring=1,114,248  next_w=142,623,744  level=4,832B

L1: D=128 lb=2 m=12 r=9 [D128-na2]
    na=2 nb=2 nd=2  do=17 df=8 dc=1
    w_ring=43,664  next_w=5,588,992  level=4,432B

L2: D=128 lb=2 m=10 r=6 [D128-na2]
    na=2 nb=2 nd=1  do=17 df=7 dc=1
    w_ring=8,164  next_w=1,044,992  level=3,680B

L3: D=64 lb=2 m=8 r=6 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=7 dc=1
    w_ring=5,192  next_w=332,288  level=2,832B

L4: D=64 lb=2 m=9 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=2,902  next_w=185,728  level=2,752B

L5: D=64 lb=2 m=8 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=2,044  next_w=130,816  level=2,672B

L6: D=64 lb=2 m=7 r=4 [D64-na2]
    na=2 nb=2 nd=2  do=17 df=6 dc=1
    w_ring=1,720  next_w=110,080  level=2,672B

Tail: w_len=110,080  lb=2  tail=27,520B
Total: 51,904 B
```

## Interpretation

The corrected 32-bit planner still prefers large root rings (`D=128`) and then
drops to `D=64` for recursion. Two things stand out:

- the `A`-role correction did not erase the 32-bit advantage
- the tail still dominates the final proof, just as it does in the corrected
  128-bit and 64-bit planners

## Mixed Boolean-Step Follow-Up

The later boolean-gadget experiment improves the 32-bit profile again:

| Polynomial | nv | Corrected balanced | Mixed boolean |
| --- | ---: | ---: | ---: |
| onehot | 20 | `38.0 KB` | `38.0 KB` |
| onehot | 25 | `44.0 KB` | `42.8 KB` |
| onehot | 30 | `47.1 KB` | `45.7 KB` |
| onehot | 32 | `48.4 KB` | `46.4 KB` |
| dense | 20 | `44.3 KB` | `44.3 KB` |
| dense | 25 | `46.9 KB` | `45.7 KB` |
| dense | 30 | `49.0 KB` | `47.5 KB` |
| dense | 32 | `50.7 KB` | `48.9 KB` |

The planner only uses boolean steps at the top. For example, onehot
`nv = 32` switches from `128->128->128->64->64->64->64` to
`64->64->64->64->64->64->64`, with boolean in the first three levels and
balanced `lb = 2` below that.

This follow-up matters for the later threshold-prime sweep: once boolean steps
are enabled, the 32-bit profile still remains the overall minimum once
recursion is nontrivial. The original above-threshold `k = 5, 6, 7` sweep does
not beat `32b-bool` anywhere. The later tightly packed below-threshold sweep
does let `k7-pack-bool` beat `32b-bool` at the tiny `nv = 20` point
(`37.7 KB` versus `38.0 KB` onehot, `42.7 KB` versus `44.3 KB` dense), but
32-bit retakes the lead from `nv >= 25`.

## Reproducibility

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi
python3 scripts/hachi_32bit_proof_planner.py
python3 scripts/hachi_proof_planner.py --field 32 --poly both --nv 20,25,30,32 --include-exp-bool
```
