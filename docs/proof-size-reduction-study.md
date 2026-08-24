# Proof Size Reduction Study

This note replaces the earlier optimistic version of the study.

The planner and supporting scripts now model the `A`-role with a
challenge-aware collision proxy instead of the old raw-digit proxy:

- `B` and `D` still use `collision_inf = 2^lb - 1`
- `A` now uses `collision_inf ~= raw_digit_collision * max_abs_challenge_coeff`
- the requested `A` collision is rounded up to the next SIS table bucket

That change matters. Under the corrected model, the planner no longer prefers
the old aggressive `D=16` schedules, and the headline 128-bit proof sizes move
up materially.

## Methodology

The corrected planner is implemented in:

- [src/planner/search.rs](/Users/quang.dao/Documents/SNARKs/hachi/src/planner/search.rs)
- [src/planner/sis_security.rs](/Users/quang.dao/Documents/SNARKs/hachi/src/planner/sis_security.rs)
- [scripts/hachi_proof_size_planner.py](/Users/quang.dao/Documents/SNARKs/hachi/scripts/hachi_proof_size_planner.py)

The SIS width tables are still lattice-estimator tables keyed by
`(D, collision_inf, rank)`, but they now include the larger `A` buckets needed
by the corrected proxy.

## Corrected Headlines

Current baselines are the ones produced by the in-repo baseline planner in
`planner/`, not the older spreadsheet-style numbers from the previous draft.

| Polynomial | nv | Baseline (B) | Corrected Optimized (B) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| onehot | 20 | 79,477 | 64,224 | 19.2% |
| onehot | 25 | 90,469 | 70,736 | 21.8% |
| onehot | 30 | 95,821 | 74,800 | 21.9% |
| onehot | 32 | 97,277 | 75,632 | 22.3% |
| onehot | 38 | 99,133 | 78,896 | 20.4% |
| onehot | 44 | 104,005 | 83,184 | 20.0% |
| full | 25 | 164,053 | 75,264 | 54.1% |
| full | 30 | 169,757 | 77,760 | 54.2% |
| full | 32 | 170,637 | 78,896 | 53.8% |

Two takeaways survived the correction:

- the universal planner still beats the current baselines comfortably
- smaller recursive rings still help, but only down to `D=32` under the
  corrected `A` filter

## Corrected 128-bit Schedules

### Onehot

| nv | Total (B) | D schedule | Tail (B) |
| ---: | ---: | --- | ---: |
| 20 | 64,224 | `32->32->32->32->32` | 39,936 |
| 25 | 70,736 | `32->32->32->32->32->32` | 39,872 |
| 30 | 74,800 | `32->32->32->32->32->32->32` | 39,936 |
| 32 | 75,632 | `32->32->32->32->32->32->32` | 40,576 |
| 38 | 78,896 | `32->32->32->32->32->32->32->32` | 40,832 |
| 44 | 83,184 | `64->32->32->32->32->32->32->32` | 39,936 |

### Full / Dense

| nv | Total (B) | D schedule | Tail (B) |
| ---: | ---: | --- | ---: |
| 20 | 76,128 | `64->32->32->32->32->32` | 39,296 |
| 25 | 75,264 | `32->32->32->32->32->32->32` | 39,808 |
| 30 | 77,760 | `32->32->32->32->32->32->32` | 41,728 |
| 32 | 78,896 | `32->32->32->32->32->32->32->32` | 40,832 |

## Detailed Breakdowns

### Onehot nv=32

Corrected optimized size: `75,632 B` versus baseline `97,277 B`.

```text
L0: D=32 lb=2 m=16 r=11 [D32-na3]
    na=3 nb=2 nd=2  do=65 df=11 dc=1
    w_ring=1,253,961  next_w=40,126,752  level=4,672B

L1: D=32 lb=2 m=13 r=8 [D32-na2]
    na=2 nb=2 nd=2  do=65 df=10 dc=1
    w_ring=99,430  next_w=3,181,760  level=4,352B

L2: D=32 lb=3 m=11 r=6 [D32-na2]
    na=2 nb=2 nd=2  do=43 df=6 dc=1
    w_ring=17,924  next_w=573,568  level=4,832B

L3: D=32 lb=4 m=10 r=5 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=6,237  next_w=199,584  level=5,216B

L4: D=32 lb=4 m=9 r=4 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=3,798  next_w=121,536  level=5,072B

L5: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,956  next_w=94,592  level=5,072B

L6: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,536  next_w=81,152  level=5,072B

Tail: w_len=81,152  lb=4  tail=40,576B
Total: 75,632 B
```

The structural shift from the previous draft is visible immediately:
the corrected planner never finds it worthwhile to jump to `D=16`.

### Onehot nv=44

Corrected optimized size: `83,184 B` versus baseline `104,005 B`.

```text
L0: D=64 lb=2 m=21 r=17 [D64-na2]
    na=2 nb=2 nd=2  do=65 df=13 dc=1
    w_ring=52,822,536  next_w=3,380,642,304  level=7,712B

L1: D=32 lb=2 m=16 r=11 [D32-na3]
    na=3 nb=2 nd=2  do=65 df=11 dc=1
    w_ring=1,100,500  next_w=35,216,000  level=4,672B

L2: D=32 lb=3 m=13 r=8 [D32-na2]
    na=2 nb=2 nd=2  do=43 df=7 dc=1
    w_ring=63,461  next_w=2,030,752  level=4,944B

L3: D=32 lb=3 m=10 r=6 [D32-na2]
    na=2 nb=2 nd=2  do=43 df=6 dc=1
    w_ring=14,552  next_w=465,664  level=4,720B

L4: D=32 lb=4 m=9 r=5 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=5,707  next_w=182,624  level=5,216B

L5: D=32 lb=4 m=9 r=4 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=3,633  next_w=116,256  level=5,072B

L6: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,876  next_w=92,032  level=5,072B

L7: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,496  next_w=79,872  level=5,072B

Tail: w_len=79,872  lb=4  tail=39,936B
Total: 83,184 B
```

### Full nv=32

Corrected optimized size: `78,896 B` versus baseline `170,637 B`.

```text
L0: D=32 lb=2 m=14 r=13 [D32-na3]
    na=3 nb=2 nd=2  do=65 df=12 dc=65
    w_ring=14,910,025  next_w=477,120,800  level=4,912B

L1: D=32 lb=2 m=14 r=10 [D32-na2]
    na=2 nb=2 nd=2  do=65 df=11 dc=1
    w_ring=360,371  next_w=11,531,872  level=4,512B

L2: D=32 lb=2 m=12 r=7 [D32-na2]
    na=2 nb=2 nd=2  do=65 df=9 dc=1
    w_ring=50,824  next_w=1,626,368  level=4,272B

L3: D=32 lb=2 m=11 r=5 [D32-na2]
    na=2 nb=1 nd=1  do=65 df=8 dc=1
    w_ring=19,342  next_w=618,944  level=3,168B

L4: D=32 lb=4 m=10 r=5 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=6,457  next_w=206,624  level=5,216B

L5: D=32 lb=4 m=9 r=4 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=5 dc=1
    w_ring=3,868  next_w=123,776  level=5,072B

L6: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,992  next_w=95,744  level=5,072B

L7: D=32 lb=4 m=9 r=3 [D32-na2]
    na=2 nb=2 nd=2  do=33 df=4 dc=1
    w_ring=2,552  next_w=81,664  level=5,072B

Tail: w_len=81,664  lb=4  tail=40,832B
Total: 78,896 B
```

## What Changed Conceptually

The main design conclusion is different from the previous draft:

- `D=16` is no longer a planner winner once the `A` role sees the
  challenge-aware collision proxy
- `D=32` remains the sweet spot for the corrected 128-bit planner
- the tail is still the dominant term, but the corrected security filter now
  forces more `D=32` levels before the proof can get there

That is exactly the kind of correction we wanted from the audit: smaller proofs
still exist, but the planner is no longer claiming an unrealistically cheap
`A` layer.

## Mixed Boolean-Step Follow-Up

The later boolean-gadget experiment has only a marginal effect on the corrected
128-bit profile:

| Polynomial | nv | Corrected balanced | Mixed boolean |
| --- | ---: | ---: | ---: |
| onehot | 20 | `62.7 KB` | `62.7 KB` |
| onehot | 25 | `69.1 KB` | `69.1 KB` |
| onehot | 30 | `73.0 KB` | `72.8 KB` |
| onehot | 32 | `73.9 KB` | `73.8 KB` |
| dense | 20 | `74.3 KB` | `74.3 KB` |
| dense | 25 | `73.5 KB` | `73.5 KB` |
| dense | 30 | `75.9 KB` | `75.9 KB` |
| dense | 32 | `77.0 KB` | `76.8 KB` |

The planner still wants ordinary balanced digits in most lower levels. For
example, onehot `nv = 32` only uses a boolean root before dropping back into
the corrected `lb = 2, 3, 4` schedule. Dense `nv = 20` is even more rigid: it
still wants the old `lb = 7` root, so the mixed profile picks the exact same
schedule as the balanced one.

So the boolean follow-up does not change the main 128-bit conclusion. The
correction to the `A` filter remains the dominant structural change, and the
later threshold-prime sweep still leaves 128-bit as the largest regime in the
family.

## Reproducibility

Commands used for the corrected numbers:

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi/planner
cargo test --quiet
cargo run --quiet
cargo run --quiet -- --breakdown

cd /Users/quang.dao/Documents/SNARKs/hachi
python3 scripts/hachi_proof_size_planner.py
python3 scripts/hachi_proof_planner.py --field 128 --poly both --nv 20,25,30,32 --include-exp-bool
```
