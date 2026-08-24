# Hachi Proof Size Study: Threshold Primes Between 16-bit and 32-bit

This note sweeps three additional base fields between the earlier 16-bit and
32-bit studies.

It now records two related follow-ups:

- an initial threshold sweep using the smallest primes `p ≡ 5 mod 8` above
  `2^(128/k)`
- a later packed-extension sweep using the largest primes `p ≡ 5 mod 8` below
  `2^(128/k)`, so that a tightly packed degree-`k` extension element fits in
  `16` bytes

The primes are chosen as the smallest primes `p ≡ 5 mod 8` above `2^(128/k)`,
so that a degree-`k` extension field clears `2^128` while still supporting the
LS18 `k = 2` invertibility setting:

| Label | k | Prime `p` | `log2(p)` | Base bytes | Extension bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k5-prime` | 5 | `50,859,013` | `25.6000` | 4 | 20 |
| `k6-prime` | 6 | `2,642,333` | `21.3334` | 3 | 18 |
| `k7-prime` | 7 | `319,589` | `18.2859` | 3 | 21 |

Two points matter immediately:

- these are not pseudo-Mersenne choices; this note only filters by `p ≡ 5 mod 8`
- unlike the `16/32/64/128` studies, the odd extension degrees no longer land
  on a neat `16`-byte sumcheck element, so the sumcheck layer costs `18`, `20`,
  or `21` bytes per element

That second point is the main structural headwind for these regimes.

## Methodology

The sweep uses the unified planner in:

- [scripts/hachi_proof_planner.py](/Users/quang.dao/Documents/SNARKs/hachi/scripts/hachi_proof_planner.py)

with the following assumptions:

- the corrected challenge-aware `A`-role SIS filter
- fresh BDGL16 + lgsa MSIS tables for each new prime
- the same low-`q` guard as the 16-bit study: require `l2 < (q - 1) / 2`
- balanced-only search restricted to `lb ∈ {2, 3, 4, 5}`
- an experimental mixed boolean-step variant that adds the `{0,1}`-style
  boolean gadget on top of those balanced choices

Ring search spaces:

- `k5-prime`: `D ∈ {64, 128, 256}`
- `k6-prime`: `D ∈ {64, 128, 256}`
- `k7-prime`: `D ∈ {64, 128, 256, 512}`

## Summary

### Onehot

| nv | `k5-prime` | `k5-bool` | `k6-prime` | `k6-bool` | `k7-prime` | `k7-bool` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `44.6 KB` | `44.6 KB` | `40.5 KB` | `38.9 KB` | `41.4 KB` | `38.8 KB` |
| 25 | `51.3 KB` | `51.1 KB` | `47.6 KB` | `45.0 KB` | `47.6 KB` | `45.2 KB` |
| 30 | `54.7 KB` | `54.5 KB` | `51.3 KB` | `47.2 KB` | `51.5 KB` | `48.7 KB` |
| 32 | `55.5 KB` | `55.5 KB` | `52.6 KB` | `48.6 KB` | `53.5 KB` | `49.6 KB` |

### Dense

| nv | `k5-prime` | `k5-bool` | `k6-prime` | `k6-bool` | `k7-prime` | `k7-bool` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `50.7 KB` | `50.7 KB` | `46.0 KB` | `45.8 KB` | `46.0 KB` | `46.0 KB` |
| 25 | `54.9 KB` | `54.2 KB` | `48.8 KB` | `47.3 KB` | `50.0 KB` | `48.6 KB` |
| 30 | `57.4 KB` | `56.9 KB` | `53.2 KB` | `49.6 KB` | `54.8 KB` | `51.9 KB` |
| 32 | `58.3 KB` | `58.0 KB` | `54.5 KB` | `51.4 KB` | `55.9 KB` | `53.1 KB` |

The dominant pattern is simple:

- `k5-prime` is never competitive; the `20`-byte sumcheck field is too costly
- `k6-bool` is the best of the new regimes
- `k7-bool` helps a lot over balanced-only, but its `21`-byte sumcheck field
  keeps it behind `k6-bool`

## Comparison to 16-bit and 32-bit Boolean Profiles

The more relevant comparison is against the mixed boolean-step experiments on
the existing `16`-bit and `32`-bit profiles.

### Onehot

| nv | `32b-bool` | `16b-bool` | Best midrange | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `38.0 KB` | `39.8 KB` | `38.8 KB` (`k7-bool`) | `32b-bool` |
| 25 | `42.8 KB` | `44.3 KB` | `45.0 KB` (`k6-bool`) | `32b-bool` |
| 30 | `45.7 KB` | `47.2 KB` | `47.2 KB` (`k6-bool`) | `32b-bool` |
| 32 | `46.4 KB` | `48.2 KB` | `48.6 KB` (`k6-bool`) | `32b-bool` |

### Dense

| nv | `32b-bool` | `16b-bool` | Best midrange | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `44.3 KB` | `45.0 KB` | `45.8 KB` (`k6-bool`) | `32b-bool` |
| 25 | `45.7 KB` | `46.8 KB` | `47.3 KB` (`k6-bool`) | `32b-bool` |
| 30 | `47.5 KB` | `48.8 KB` | `49.6 KB` (`k6-bool`) | `32b-bool` |
| 32 | `48.9 KB` | `50.3 KB` | `51.4 KB` (`k6-bool`) | `32b-bool` |

So the midrange sweep does **not** overturn the earlier conclusion:

- no threshold-prime regime beats `32b-bool`
- `k6-bool` is the strongest new candidate
- for nontrivial recursion, even `16b-bool` usually stays below the best
  threshold-prime field

The only tiny exception is the very small onehot `nv = 20` case, where
`k7-bool` at `38.8 KB` slips below `16b-bool` at `39.8 KB`, but it still does
not beat `32b-bool`.

## Representative Schedules

### `k5-prime`: the extension field is just too expensive

At `nv = 32`, onehot stays entirely in `D = 128` and the boolean variant does
not help at all:

- onehot: `55.5 KB` both with and without boolean
- schedule: `128->128->128->128->128->128`

This is the cleanest sign that the `20`-byte sumcheck field is dominating the
would-be tail savings.

### `k6-bool`: the best new regime

At `nv = 32`, onehot drops from `52.6 KB` to `48.6 KB`, and dense drops from
`54.5 KB` to `51.4 KB`.

Structurally:

- onehot shifts from all-`128` recursion to `128->128->128->64->64->64->64`
- dense shifts from `256->128...` into a mixed `128` then `64` schedule
- both boolean schedules keep `bits = 1` in the terminal tail

This is the strongest evidence that the boolean gadget is doing real work, but
the `18`-byte sumcheck field still leaves the regime above both `32b-bool` and
the larger `16b-bool` cases.

### `k7-bool`: good tail, wrong extension size

At `nv = 32`, onehot improves from `53.5 KB` to `49.6 KB`, but the mixed
boolean schedule still trails `k6-bool`.

The shape is instructive:

- early levels remain in `D = 128`
- only the last two levels drop to `D = 64`
- the terminal witness is boolean-packed (`bits = 1`)

So `k7-bool` does recover some tail efficiency, but the `21`-byte sumcheck
field and heavier top-level openings undo most of the benefit.

## Interpretation

The threshold-prime sweep sharpens the emerging picture:

- smaller base fields are not enough by themselves
- once the extension degree becomes odd and the extension element stops fitting
  in `16` bytes under the current representation model, the sumcheck layer
  becomes materially more expensive
- among the new candidates, `k6-bool` is the closest to being interesting, but
  it still does not beat `32b-bool`

So the current floor in this first, coordinate-packed model still looks like:

- `32b-bool` overall
- `16b-bool` as the best non-32-bit regime once recursion is nontrivial

That turns out to depend heavily on extension-field serialization.

## Packed 16-Byte Follow-Up

To isolate the serialization effect, the later sweep replaces the above-threshold
primes with the largest primes `p ≡ 5 mod 8` below `2^(128/k)`, so that a
tightly packed `F_{p^k}` element fits in `16` bytes:

| Label | k | Prime `p` | `k * log2(p)` | Packed extension bytes |
| --- | ---: | ---: | ---: | ---: |
| `k5-pack` | 5 | `50,858,909` | `127.99999` | 16 |
| `k6-pack` | 6 | `2,642,173` | `127.99976` | 16 |
| `k7-pack` | 7 | `319,541` | `127.99949` | 16 |

### Onehot

| nv | `k5-pack` | `k5-pack-bool` | `k6-pack` | `k6-pack-bool` | `k7-pack` | `k7-pack-bool` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `43.2 KB` | `43.2 KB` | `39.8 KB` | `38.4 KB` | `38.8 KB` | `37.7 KB` |
| 25 | `49.5 KB` | `49.0 KB` | `46.5 KB` | `44.3 KB` | `44.2 KB` | `43.4 KB` |
| 30 | `52.4 KB` | `52.3 KB` | `49.6 KB` | `46.4 KB` | `47.7 KB` | `46.5 KB` |
| 32 | `53.1 KB` | `53.1 KB` | `51.0 KB` | `47.7 KB` | `49.4 KB` | `47.4 KB` |

### Dense

| nv | `k5-pack` | `k5-pack-bool` | `k6-pack` | `k6-pack-bool` | `k7-pack` | `k7-pack-bool` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `48.4 KB` | `48.4 KB` | `44.5 KB` | `44.5 KB` | `42.7 KB` | `42.7 KB` |
| 25 | `52.6 KB` | `52.0 KB` | `47.7 KB` | `46.5 KB` | `46.3 KB` | `46.3 KB` |
| 30 | `54.6 KB` | `54.3 KB` | `51.8 KB` | `48.7 KB` | `50.8 KB` | `48.7 KB` |
| 32 | `55.5 KB` | `55.4 KB` | `53.1 KB` | `50.4 KB` | `51.6 KB` | `50.6 KB` |

The packed sweep changes the picture substantially:

- `k5-pack` is still not competitive
- `k6-pack-bool` becomes a strong midrange option
- `k7-pack-bool` becomes the best small-instance threshold-prime regime

### Comparison to 16-bit and 32-bit Boolean Profiles

#### Onehot

| nv | `32b-bool` | `16b-bool` | Best packed threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `38.0 KB` | `39.8 KB` | `37.7 KB` (`k7-pack-bool`) | `k7-pack-bool` |
| 25 | `42.8 KB` | `44.3 KB` | `43.4 KB` (`k7-pack-bool`) | `32b-bool` |
| 30 | `45.7 KB` | `47.2 KB` | `46.4 KB` (`k6-pack-bool`) | `32b-bool` |
| 32 | `46.4 KB` | `48.2 KB` | `47.4 KB` (`k7-pack-bool`) | `32b-bool` |

#### Dense

| nv | `32b-bool` | `16b-bool` | Best packed threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `44.3 KB` | `45.0 KB` | `42.7 KB` (`k7-pack-bool`) | `k7-pack-bool` |
| 25 | `45.7 KB` | `46.8 KB` | `46.3 KB` (`k7-pack-bool`) | `32b-bool` |
| 30 | `47.5 KB` | `48.8 KB` | `48.7 KB` (`k6-pack-bool`, `k7-pack-bool`) | `32b-bool` |
| 32 | `48.9 KB` | `50.3 KB` | `50.4 KB` (`k6-pack-bool`) | `32b-bool` |

So the packed follow-up refines the earlier conclusion:

- `k7-pack-bool` is now the global minimum at the smallest `nv = 20` point,
  for both onehot and dense
- `32b-bool` still retakes the lead from `nv >= 25`
- packed threshold-prime fields now beat `16b-bool` on all shared onehot
  points, and on dense `nv = 20, 25, 30`
- at dense `nv = 32`, `16b-bool` and `k6-pack-bool` are essentially tied
  (`50.3 KB` versus `50.4 KB`)

### Representative Packed Schedules

At `nv = 32`, the packed leaders look like this:

- `k6-pack-bool` onehot: `47.7 KB`, schedule
  `128->128->128->64->64->64->64`
- `k7-pack-bool` onehot: `47.4 KB`, schedule
  `128->128->128->128->128->64->64`
- `k6-pack-bool` dense: `50.4 KB`
- `k7-pack-bool` dense: `50.6 KB`

The interesting structural point is that `k7-pack-bool` still likes `D = 128`
for most of the recursion. The gain comes almost entirely from having a smaller
base field without paying the old `21`-byte extension-field penalty.

### Why `k7-pack` has the smallest packed tail

The planner's raw terminal tail is

```text
tail_bytes = ceil(final_w_len * final_bits / 8).
```

So the packed tail is controlled only by:

- the final witness length `final_w_len`
- the terminal digit width `final_bits`

For the best-tail packed regime, the key fact is that the balanced `k7-pack`
schedule drives `final_w_len` down far enough that a `4`-bit tail is still
cheaper than the longer `1`-bit or `2`-bit tails in the boolean-oriented
profiles.

At onehot `nv = 32`, the winning `k7-pack` tail comes from the last two levels:

- `L4`: `D=128, lb=4, m=7, r=3, na=2, nb=2, nd=2, do=5, df=3, dc=1`,
  `next_w=62,336`
- `L5`: `D=128, lb=4, m=6, r=3, na=2, nb=2, nd=2, do=5, df=3, dc=1`,
  `next_w=43,904`
- terminal tail: `w=43,904`, `bits=4`, so `tail=21,952 B`

At dense `nv = 32`, the same pattern appears:

- `L5`: `D=128, lb=4, m=7, r=3, na=2, nb=2, nd=2, do=5, df=3, dc=1`,
  `next_w=58,112`
- `L6`: `D=128, lb=4, m=6, r=3, na=2, nb=2, nd=2, do=5, df=3, dc=1`,
  `next_w=42,368`
- terminal tail: `w=42,368`, `bits=4`, so `tail=21,184 B`

This is also why the above-threshold `k7-prime` and packed `k7-pack` profiles
tie on tail: they pick the same late schedule and end with the same
`final_w_len` and `final_bits`. The packed profile only wins on total proof
size because its extension elements cost `16` bytes instead of `21`.

The tail-only comparison at `nv = 32` makes the tradeoff visible:

| Profile | Onehot tail | Dense tail |
| --- | ---: | ---: |
| `k7-pack` | `21,952 B` | `21,184 B` |
| `k7-pack-bool` | `28,248 B` | `27,672 B` |
| `32b-bool` | `27,904 B` | `27,136 B` |

So the smallest tail is **not** the boolean profile. It is the balanced
degree-7 profile, because its late `lb = 4` levels shrink the witness much more
aggressively before the final packing step.

### Why `32b-bool` still wins the total proof

The total proof does not track the tail alone. It is

```text
total = recursive levels + tail.
```

At `nv = 32`, the best onehot totals decompose as:

| Profile | Total | Tail | Non-tail |
| --- | ---: | ---: | ---: |
| `32b-bool` | `47,488 B` | `27,904 B` | `19,584 B` |
| `k7-pack` | `50,624 B` | `21,952 B` | `28,672 B` |
| `k7-pack-bool` | `48,568 B` | `28,248 B` | `20,320 B` |

And the dense totals decompose as:

| Profile | Total | Tail | Non-tail |
| --- | ---: | ---: | ---: |
| `32b-bool` | `50,112 B` | `27,136 B` | `22,976 B` |
| `k7-pack` | `52,800 B` | `21,184 B` | `31,616 B` |
| `k7-pack-bool` | `51,848 B` | `27,672 B` | `24,176 B` |

This shows the main pattern:

- `k7-pack` saves about `6 KB` on the tail, but pays roughly `9 KB` more
  outside the tail in onehot and about `8.6 KB` more in dense
- `k7-pack-bool` fixes most of that recursive overhead, but its tail rises back
  above `32b-bool`, so it still loses overall

The reason is mostly the cost of the folding levels, not the number of levels.
The schedules are similar in depth:

- onehot `nv = 32`: `32b-bool` has `7` levels, `k7-pack` has `6`,
  `k7-pack-bool` has `7`
- dense `nv = 32`: `32b-bool` has `8`, `k7-pack` has `7`,
  `k7-pack-bool` has `8`

What changes materially is the byte cost per recursive level.

For onehot `nv = 32`:

- `32b-bool` uses `D = 64` at every level, with lower-level costs
  `2,832 + 2,752 + 2,752 + 2,672 = 11,008 B`
- `k7-pack` stays at `D = 128` throughout and pays
  `4,464 + 5,200 + 4,288 + 4,288 = 18,240 B`

So even though `k7-pack` has one fewer level, its bottom four levels alone cost
`7,232 B` more.

For dense `nv = 32`:

- `32b-bool` pays `2,832 + 2,752 + 2,672 + 2,672 = 10,928 B` in its last four
  levels
- `k7-pack` pays `3,968 + 3,856 + 4,288 + 4,288 = 16,400 B`

Again, the late recursion is much more expensive in the degree-7 profile.

So the clean summary is:

- `k7-pack` is the best current **tail technology**
- `32b-bool` is the best current **whole-proof technology**
- the gap is not that `32b-bool` folds dramatically more times, but that it
  keeps those folding levels much cheaper by living in `D = 64` and mostly
  using the lighter `lb = 2` lower recursion

## Reproducibility

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi

python3 scripts/hachi_proof_planner.py --profile k5 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k6 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k7 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k5pack --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k6pack --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k7pack --poly both --nv 20,25,30,32 --include-exp-bool
```
