# NTT Prime Analysis (Pow2Offset / Solinas Context)

This note records the current analysis for small NTT primes and CRT coverage targets.

## References

- NIST ML-KEM: `paper/standards/NIST.FIPS.203.pdf`
- NIST ML-DSA: `paper/standards/NIST.FIPS.204.pdf`
- Current small-prime table: `src/algebra/ntt/tables.rs`
- Labrador generator heuristic: `../labrador/data.py`

## Why does `2D` divide `p - 1`?

For negacyclic NTT on `Z_p[X]/(X^D + 1)`, we need a primitive `2D`-th root `psi` such that:

- `psi^D = -1 (mod p)`
- `psi^(2D) = 1 (mod p)`

Over prime fields, `F_p^*` is cyclic of size `p - 1`, so an element of order `2D` exists iff:

- `2D | (p - 1)`

So yes, the `128 | (p - 1)` condition is directly tied to `D = 64`.

## What if `D = 1024`?

Then requirement becomes:

- `2D = 2048`, so `2048 | (p - 1)`.

Under the current "small prime" cap (`p < 2^14`), this is extremely restrictive.

## Why `< 2^14` in current code?

This is a backend implementation constraint, not a hard NTT math requirement:

- Current small-prime NTT backend stores modulus/coefficients in signed 16-bit lanes (`i16`).
- It relies on centered signed arithmetic and butterfly add/sub before full normalization.
- Keeping `p < 2^14` leaves practical headroom in those 16-bit operations.
- Current CRT limb code is also radix-`2^14`, matching this design style.

So the `2^14` cap is about the present `i16` scalar kernel design. If we introduce an `i32` backend, this cap can be raised substantially.

## Exhaustive counts (for `p < 2^14`)

We classify exact Solinas as:

- `p = 2^x - 2^y + 1`.

Results:

- `D = 64` (`128 | p-1`)
  - all small NTT primes: **31**
  - exact Solinas NTT primes: **6**
  - all-prime set:
    - `257, 641, 769, 1153, 1409, 2689, 3329, 3457, 4481, 4993, 6529, 7297, 7681, 7937, 9473, 9601, 9857, 10369, 10753, 11393, 11777, 12161, 12289, 13313, 13441, 13697, 14081, 14593, 15233, 15361, 16001`
  - Solinas set: `257, 769, 7681, 7937, 12289, 15361`
- `D = 256` (`512 | p-1`)
  - all small NTT primes: **6**
  - exact Solinas NTT primes: **3**
  - Solinas set: `7681, 12289, 15361`
- `D = 1024` (`2048 | p-1`)
  - all small NTT primes: **1**
  - exact Solinas NTT primes: **1**
  - Solinas set: `12289`

Conclusion: for higher `D`, the small-prime pool shrinks rapidly.

## 30-bit exploration (`p < 2^30`) with NTT constraints

To assess a larger-prime backend direction, we scanned for primes under `2^30` with:

- `p ≡ 1 (mod 2D)`.

Below are the **full outputs of the bounded search run** (top 30 largest primes found by descending scan):

### `D = 64` (`2D = 128`)

- Top-30 list:
  - `1073741441, 1073739649, 1073738753, 1073736449, 1073735297, 1073734913, 1073732993, 1073732609, 1073731201, 1073731073, 1073730817, 1073728897, 1073727617, 1073726977, 1073722753, 1073719681, 1073717377, 1073716993, 1073713409, 1073712769, 1073712257, 1073710721, 1073708929, 1073707009, 1073703809, 1073702657, 1073702401, 1073698817, 1073696257, 1073693441`
- Coverage for `q = 2^128 - 275`:
  - `P > q`: **5** limbs
  - `P > 128*q^2`: **9** limbs

### `D = 1024` (`2D = 2048`)

- Top-30 list:
  - `1073707009, 1073698817, 1073692673, 1073682433, 1073668097, 1073655809, 1073651713, 1073643521, 1073620993, 1073600513, 1073569793, 1073563649, 1073551361, 1073539073, 1073522689, 1073510401, 1073508353, 1073479681, 1073453057, 1073442817, 1073440769, 1073430529, 1073412097, 1073391617, 1073385473, 1073354753, 1073350657, 1073330177, 1073299457, 1073268737`
- Coverage for `q = 2^128 - 275`:
  - `P > q`: **5** limbs
  - `P > 128*q^2`: **9** limbs

### Bit-estimate sanity check

- `ceil(128 / 30) = 5`
- `ceil(263 / 30) = 9` (for `128*q^2 ~ 2^263`)

This matches the concrete product counts above.

## CRT size targets for `q = 2^128 - 275`

Two common thresholds:

1. Minimal uniqueness target:
   - `P = prod(p_i) > q`
2. Labrador conservative heuristic:
   - `P > 128 * q^2` (from `data.py`, with `FIXME` comment)

### Limb counts at `D = 64` with current small-prime pool

- Using all small NTT primes (`31` available):
  - `P > q` achievable with **10** limbs
  - `P > 128*q^2` achievable with **20** limbs
- Using only exact Solinas NTT primes (`6` available):
  - `P > q`: **not achievable**
  - `P > 128*q^2`: **not achievable**
  - total product is only about `2^70`

### Limb counts at `D = 1024` with current small-prime pool

- Only one qualifying prime (`12289`) under `p < 2^14`, so neither threshold is achievable.

## What is Labrador's safety margin doing?

In Labrador code, prime selection stops at:

- `P > 128 * q^2`

Interpretation:

- `q^2` tracks product-scale growth,
- extra factor `128` gives additional headroom (for `N=64`, this is `2N`),
- but their own `FIXME` comment indicates this is a conservative engineering bound, not a tight proof.

So treat this as a robust heuristic rather than a formal minimum.

## Practical implication for Hachi

- If we stay with `D=64` and small i16-ish primes, we need non-Solinas primes in the CRT set.
- If we push to `D=1024`, we must either:
  - lift prime size beyond `<2^14`, or
  - change CRT strategy (fewer larger limbs / different backend), or
  - avoid strict small-prime CRT-NTT at that degree.
- A mixed backend model is sensible:
  - keep the current `i16` backend for small-prime kernels,
  - add an `i32`/wider backend for larger-prime kernels (e.g., up to ~30-bit).
