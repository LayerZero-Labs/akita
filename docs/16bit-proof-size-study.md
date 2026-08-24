# Hachi Proof Size Study: 16-bit Prime Field

This note records the first 16-bit planner pass for Hachi using the largest
prime below `2^16` with `q ≡ 5 mod 8`:

- `q = 2^16 - 99 = 65437`

This is an exploratory regime, not a final strict 128-bit profile. The
sumcheck field uses degree-8 extension over `F_q`, which gives about
`8 * log2(65437) ≈ 127.98` bits of field soundness. We also only assume the
`k = 2` LS18 invertibility setting, which is the reason to stay with
2-splitting only.

The main result is simple: moving from 32-bit to 16-bit does **not** make the
proof smaller in this model. The 16-bit schedules come out about `2%` to `4%`
larger than the corrected 32-bit schedules.

## Methodology

The 16-bit planner uses:

- the same challenge-aware `A`-role correction as the corrected 128/64/32-bit
planners
- fresh MSIS width tables for `q = 65437`, estimated with BDGL16 + lgsa
- an extra `l2 < (q - 1) / 2` cutoff in the SIS sweep to avoid the trivial
low-`q` regime
- only shifted-up ring dimensions `D ∈ {128, 256, 512}`
- degree-8 sumchecks, so extension elements are still `16` bytes
- capped workloads: onehot up to `nv = 32`, dense up to `nv = 30`

The script is:

- [scripts/hachi_16bit_proof_planner.py](/Users/quang.dao/Documents/SNARKs/hachi/scripts/hachi_16bit_proof_planner.py)

## Summary


| Polynomial  | nv  | Total (B) | D schedule                          | Tail (B) |
| ----------- | --- | --------- | ----------------------------------- | -------- |
| onehot      | 20  | 40,704    | `128->128->128->128`                | 29,184   |
| onehot      | 25  | 46,528    | `256->256->128->128->128->128`      | 27,648   |
| onehot      | 30  | 49,872    | `256->256->128->128->128->128`      | 29,568   |
| onehot      | 32  | 51,120    | `256->256->256->128->128->128->128` | 27,648   |
| dense-16bit | 20  | 46,112    | `256->128->128->128->128`           | 29,760   |
| dense-16bit | 25  | 49,408    | `256->256->128->128->128->128`      | 29,184   |
| dense-16bit | 30  | 51,664    | `256->256->256->128->128->128->128` | 28,032   |


Two structural points stand out immediately:

- `D=512` is never selected by the planner
- the useful 16-bit schedules are mostly `D=256` at the top and `D=128` below

So the MSIS sweep was pointing in the right direction: `D=256` is the real
entry point for a viable 16-bit regime, while `D=512` is already overkill.

## Comparison to Corrected 32-bit Planner

### Onehot


| nv  | 16-bit (B) | Corrected 32-bit (B) | Ratio |
| --- | ---------- | -------------------- | ----- |
| 20  | 40,704     | 38,960               | 1.04x |
| 25  | 46,528     | 45,056               | 1.03x |
| 30  | 49,872     | 48,208               | 1.03x |
| 32  | 51,120     | 49,568               | 1.03x |


### Dense


| nv  | 16-bit (B) | Corrected 32-bit (B) | Ratio |
| --- | ---------- | -------------------- | ----- |
| 20  | 46,112     | 45,360               | 1.02x |
| 25  | 49,408     | 48,016               | 1.03x |
| 30  | 51,664     | 50,192               | 1.03x |


So the trend from `128 -> 64 -> 32` does **not** continue to `16`. In this
planner, 32-bit remains the minimum.

16-bit still looks much better than corrected 64-bit and corrected 128-bit at
the same shared data points, but it gives back the small remaining advantage
once the ring dimensions have to shift upward.

## Detailed Breakdowns

### Onehot nv=32

```text
L0: D=256 lb=2 m=13 r=11 [D256-na2]
    na=2 nb=2 nd=2  do=9 df=9 dc=1
    w_ring=129,096  next_w=33,048,576  level=4,592B

L1: D=256 lb=2 m=9 r=8 [D256-na1]
    na=1 nb=2 nd=2  do=9 df=8 dc=1
    w_ring=8,711  next_w=2,230,016  level=4,352B

L2: D=256 lb=2 m=8 r=6 [D256-na1]
    na=1 nb=1 nd=1  do=9 df=7 dc=1
    w_ring=2,156  next_w=551,936  level=3,168B

L3: D=128 lb=2 m=8 r=5 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=7 dc=1
    w_ring=1,881  next_w=240,768  level=2,752B

L4: D=128 lb=2 m=7 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=1,212  next_w=155,136  level=2,752B

L5: D=128 lb=2 m=7 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=960  next_w=122,880  level=2,672B

L6: D=128 lb=2 m=6 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=864  next_w=110,592  level=2,672B

Tail: w_len=110,592  lb=2  tail=27,648B
Total: 51,120 B
```

### Dense nv=30

```text
L0: D=256 lb=2 m=10 r=12 [D256-na2]
    na=2 nb=2 nd=2  do=9 df=10 dc=9
    w_ring=202,824  next_w=51,922,944  level=4,672B

L1: D=256 lb=2 m=10 r=8 [D256-na2]
    na=2 nb=2 nd=2  do=9 df=8 dc=1
    w_ring=13,328  next_w=3,411,968  level=4,352B

L2: D=256 lb=2 m=8 r=6 [D256-na1]
    na=1 nb=1 nd=1  do=9 df=7 dc=1
    w_ring=2,660  next_w=680,960  level=3,168B

L3: D=128 lb=2 m=8 r=5 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=7 dc=1
    w_ring=2,105  next_w=269,440  level=2,832B

L4: D=128 lb=2 m=8 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=1,296  next_w=165,888  level=2,752B

L5: D=128 lb=2 m=7 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=990  next_w=126,720  level=2,672B

L6: D=128 lb=2 m=6 r=4 [D128-na2]
    na=2 nb=2 nd=2  do=9 df=6 dc=1
    w_ring=876  next_w=112,128  level=2,672B

Tail: w_len=112,128  lb=2  tail=28,032B
Total: 51,664 B
```

## Interpretation

The 16-bit regime is a good sanity check for the planner because it breaks the
naive monotone story.

What still helps:

- committed ring vectors are cheaper because base-field coefficients are only
`2` bytes

What pushes back:

- MSIS security forces the ring dimensions up from the 32-bit regime
- the best schedules need `D=256` roots and `D=128` recursion
- the tail is packed digits, so it does not shrink automatically with field
size
- in practice the larger 16-bit rings leave a slightly larger final witness,
which makes the tail a bit bigger than 32-bit

So the practical story is now:

- `128 -> 64 -> 32` helps
- `32 -> 16` does not
- the floor seems to be around the corrected 32-bit regime, at least under
this proof model and these MSIS estimates

## Mixed Boolean-Step Follow-Up

The later boolean-gadget experiment does help the 16-bit regime:


| Polynomial | nv  | Balanced  | Mixed boolean |
| ---------- | --- | --------- | ------------- |
| onehot     | 20  | `39.8 KB` | `39.8 KB`     |
| onehot     | 25  | `45.4 KB` | `44.3 KB`     |
| onehot     | 30  | `48.7 KB` | `47.2 KB`     |
| onehot     | 32  | `49.9 KB` | `48.2 KB`     |
| dense      | 20  | `45.0 KB` | `45.0 KB`     |
| dense      | 25  | `48.2 KB` | `46.8 KB`     |
| dense      | 30  | `50.5 KB` | `48.8 KB`     |
| dense      | 32  | `51.2 KB` | `50.3 KB`     |


As with 32-bit, the planner only wants boolean steps near the top. For onehot
`nv = 32`, the mixed schedule switches from
`256->256->256->128->128->128->128` to `128->128->128->128->128->128->128`,
with boolean only in the first three levels.

The later threshold-prime follow-ups between 16-bit and 32-bit split into two
stories:

- with the original coordinate-packed model, `k6-bool` is best but still lands
above `16b-bool` once recursion is nontrivial
- with tightly packed `16`-byte extension serialization, the packed
`k6/k7` profiles mostly overtake 16-bit

In particular, `k7-pack-bool` beats `16b-bool` on all shared onehot points, and
on dense `nv = 20, 25, 30`; only dense `nv = 32` still slightly prefers
`16b-bool` over the packed candidates (`50.3 KB` versus `50.4 KB` for
`k6-pack-bool`). 32-bit still remains smaller from `nv >= 25`.

## Reproducibility

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi
python3 scripts/hachi_16bit_proof_planner.py
python3 scripts/hachi_proof_planner.py --field 16 --poly both --nv 20,25,30,32 --include-exp-bool
```

