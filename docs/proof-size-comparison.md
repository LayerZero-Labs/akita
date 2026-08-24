# Proof Size Comparison Across Field Sizes

This note compares the corrected planners for the three main field-size
regimes plus one exploratory smaller-field regime:

- 128-bit base field
- 64-bit base field with degree-2 extension for sumcheck
- 32-bit base field with degree-4 extension for sumcheck
- 16-bit base field with degree-8 extension for sumcheck

All numbers here already include the corrected challenge-aware `A`-role filter.
That is the main difference from the older version of this file.

The 16-bit row should be read slightly differently from the others:

- it uses `q = 2^16 - 99 = 65437`
- it assumes only the LS18 `k = 2` setting, so this is a 2-splitting-only
  regime
- degree-8 extension gives about `127.98` bits of field soundness, so this is
  an exploratory "128-bit-ish" profile rather than a final strict 128-bit one

A later follow-up also swept three threshold-prime fields between 16-bit and
32-bit, first with the smallest primes above `2^(128/k)` and then with packed
variants just below `2^(128/k)`:

- `k = 5`: `p = 50,859,013`
- `k = 6`: `p = 2,642,333`
- `k = 7`: `p = 319,589`

Those regimes are discussed in detail in
[midrange-threshold-prime-sweep.md](/Users/quang.dao/Documents/SNARKs/hachi/docs/midrange-threshold-prime-sweep.md).

## Onehot Comparison

| nv | 128-bit (B) | 64-bit (B) | 32-bit (B) | 16-bit (B) | 64/128 | 32/128 | 16/32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 64,224 | 47,520 | 38,960 | 40,704 | 0.74x | 0.61x | 1.04x |
| 25 | 70,736 | 54,272 | 45,056 | 46,528 | 0.77x | 0.64x | 1.03x |
| 30 | 74,800 | 57,168 | 48,208 | 49,872 | 0.76x | 0.64x | 1.03x |
| 32 | 75,632 | 58,976 | 49,568 | 51,120 | 0.78x | 0.66x | 1.03x |

For larger onehot instances where the 16-bit and 32-bit studies do not report
data:

| nv | 128-bit (B) | 64-bit (B) | 64/128 |
| ---: | ---: | ---: | ---: |
| 38 | 78,896 | 62,304 | 0.79x |
| 44 | 83,184 | 66,464 | 0.80x |

## Dense Comparison

For the shared dense data points:

| nv | 128-bit (B) | 64-bit (B) | 32-bit (B) | 16-bit (B) | 64/128 | 32/128 | 16/32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 76,128 | 56,032 | 45,360 | 46,112 | 0.74x | 0.60x | 1.02x |
| 25 | 75,264 | 57,168 | 48,016 | 49,408 | 0.76x | 0.64x | 1.03x |
| 30 | 77,760 | 60,320 | 50,192 | 51,664 | 0.78x | 0.65x | 1.03x |

For `nv = 32`, where the 16-bit study was intentionally capped at dense
`nv = 30`:

| nv | 128-bit (B) | 64-bit (B) | 32-bit (B) | 64/128 | 32/128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 78,896 | 61,344 | 51,904 | 0.78x | 0.66x |

## Main Takeaways

The corrected comparison is cleaner than the old one:

- among the strict corrected regimes, 32-bit is the smallest across every
  shared data point
- the exploratory 16-bit regime sits slightly above 32-bit, typically by only
  `2%` to `4%`
- 64-bit sits above both 32-bit and 16-bit
- corrected 128-bit remains the largest of the four

Another important correction is structural:

- the corrected 128-bit planner stays at `D=32` after the root instead of
  immediately dropping to `D=16`
- the corrected 64-bit planner is mostly `D=64`
- the corrected 32-bit planner is mostly `D=128` at the top and `D=64` below
- the exploratory 16-bit planner is mostly `D=256` at the top and `D=128`
  below, with `D=512` never selected

So the cross-field comparison is no longer “who can use the tiniest recursive
ring.” It is “who gets the best digit depth and tail cost once the `A` layer is
priced honestly.”

The new 16-bit row sharpens that point even further. The trend
`128 -> 64 -> 32` does not continue forever. Once `q` gets small enough, the
security side pushes the rings up, and the proof starts growing again.

## Representative Onehot nv=32 Totals

| Field regime | Total (B) | Tail (B) | Tail share |
| --- | ---: | ---: | ---: |
| 128-bit | 75,632 | 40,576 | 53.7% |
| 64-bit | 58,976 | 33,648 | 57.1% |
| 32-bit | 49,568 | 26,176 | 52.8% |
| 16-bit exploratory | 51,120 | 27,648 | 54.1% |

The tail is still the dominant term in all four regimes. The main difference
between fields is therefore not a radically shorter recursive chain, but how
expensive each ring element and each opening decomposition becomes on the way
to the tail.

The 16-bit line makes one more point: the tail does not shrink automatically
just because the base field is smaller. Packed digits care mostly about final
witness length and `lb`, and the larger rings needed by 16-bit security make
that terminal witness slightly longer than in the 32-bit regime.

## Mixed Boolean-Step Comparison

The later mixed boolean-step experiment improves every sub-128-bit regime, but
the size of the win varies a lot:

- 32-bit gets the strongest consistent improvement and remains the overall best
- 16-bit also improves materially, but still stays above 32-bit
- 64-bit improves modestly
- 128-bit barely moves

At `nv = 32`, the mixed boolean totals are:

| Regime | Onehot | Dense |
| --- | ---: | ---: |
| `128b-bool` | `73.8 KB` | `76.8 KB` |
| `64b-bool` | `56.0 KB` | `59.2 KB` |
| `32b-bool` | `46.4 KB` | `48.9 KB` |
| `16b-bool` | `48.2 KB` | `50.3 KB` |

So even after enabling boolean steps, the best profile among the four main
field-size ladders still sits at 32-bit.

## Threshold-Prime Sweeps Between 16-bit and 32-bit

The first threshold-prime follow-up, using the smallest primes above
`2^(128/k)`, does **not** produce a new winner. Among those profiles,
`k6-bool` is the best one, but it still stays above `32b-bool` on every shared
point.

### Onehot

| nv | `32b-bool` | `16b-bool` | Best threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `38.0 KB` | `39.8 KB` | `38.8 KB` (`k7-bool`) | `32b-bool` |
| 25 | `42.8 KB` | `44.3 KB` | `45.0 KB` (`k6-bool`) | `32b-bool` |
| 30 | `45.7 KB` | `47.2 KB` | `47.2 KB` (`k6-bool`) | `32b-bool` |
| 32 | `46.4 KB` | `48.2 KB` | `48.6 KB` (`k6-bool`) | `32b-bool` |

### Dense

| nv | `32b-bool` | `16b-bool` | Best threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `44.3 KB` | `45.0 KB` | `45.8 KB` (`k6-bool`) | `32b-bool` |
| 25 | `45.7 KB` | `46.8 KB` | `47.3 KB` (`k6-bool`) | `32b-bool` |
| 30 | `47.5 KB` | `48.8 KB` | `49.6 KB` (`k6-bool`) | `32b-bool` |
| 32 | `48.9 KB` | `50.3 KB` | `51.4 KB` (`k6-bool`) | `32b-bool` |

This first sweep clarifies why:

- `k5-prime` pays too much for its `20`-byte extension field
- `k7-prime` and `k7-bool` pay even more with `21`-byte extension elements
- `k6-bool` is the best compromise, but still not enough to beat the existing
  `16b-bool` and `32b-bool` baselines once recursion becomes nontrivial

Packing the extension elements tightly changes that story enough to deserve a
second comparison. Using the largest primes below `2^(128/k)` gives
`k5pack/k6pack/k7pack`, each with a packed `16`-byte extension field.

### Packed Onehot Comparison

| nv | `32b-bool` | `16b-bool` | Best packed threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `38.0 KB` | `39.8 KB` | `37.7 KB` (`k7-pack-bool`) | `k7-pack-bool` |
| 25 | `42.8 KB` | `44.3 KB` | `43.4 KB` (`k7-pack-bool`) | `32b-bool` |
| 30 | `45.7 KB` | `47.2 KB` | `46.4 KB` (`k6-pack-bool`) | `32b-bool` |
| 32 | `46.4 KB` | `48.2 KB` | `47.4 KB` (`k7-pack-bool`) | `32b-bool` |

### Packed Dense Comparison

| nv | `32b-bool` | `16b-bool` | Best packed threshold-prime | Winner |
| ---: | ---: | ---: | ---: | --- |
| 20 | `44.3 KB` | `45.0 KB` | `42.7 KB` (`k7-pack-bool`) | `k7-pack-bool` |
| 25 | `45.7 KB` | `46.8 KB` | `46.3 KB` (`k7-pack-bool`) | `32b-bool` |
| 30 | `47.5 KB` | `48.8 KB` | `48.7 KB` (`k6-pack-bool`, `k7-pack-bool`) | `32b-bool` |
| 32 | `48.9 KB` | `50.3 KB` | `50.4 KB` (`k6-pack-bool`) | `32b-bool` |

The packed follow-up refines the cross-field picture:

- `k7-pack-bool` is the overall minimum at the smallest shared point
  `nv = 20`
- `32b-bool` remains the best regime once `nv >= 25`
- packed threshold-prime fields overtake `16b-bool` on most shared points, so
  16-bit is no longer the strongest non-32-bit regime under a tight
  serialization model

## Reproducibility

```bash
cd /Users/quang.dao/Documents/SNARKs/hachi/planner
cargo run --quiet

cd /Users/quang.dao/Documents/SNARKs/hachi
python3 scripts/hachi_16bit_proof_planner.py
python3 scripts/hachi_64bit_proof_planner.py
python3 scripts/hachi_32bit_proof_planner.py
python3 scripts/hachi_proof_planner.py --field 128 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --field 64 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --field 32 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --field 16 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k5 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k6 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k7 --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k5pack --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k6pack --poly both --nv 20,25,30,32 --include-exp-bool
python3 scripts/hachi_proof_planner.py --profile k7pack --poly both --nv 20,25,30,32 --include-exp-bool
```
