# Field Operations Performance

Benchmark results for scalar and packed (SIMD) field arithmetic across
platforms.  All numbers are **element throughput** (median) reported by
`criterion` over 4096-element arrays (`cargo bench --bench field_arith`).

## Prime selection

All primes (except M31) are pseudo-Mersenne: `q = 2^k − c` where `c` is
the **smallest positive offset** such that `q` is prime and `q ≡ 5
(mod 8)`.  The congruence `q ≡ 5 (mod 8)` is required so that the
cyclotomic ring `Z_q[X]/(X^d + 1)` splits fully via NTT when `d` is a
power of two (equivalently, `−1` is a quadratic residue but not a quartic
residue mod `q`).

The constraint `q ≡ 5 (mod 8)` forces `c ≡ 2^k − 5 (mod 8)`:

| k mod 8 | required c mod 8 | examples |
|---------|------------------|----------|
| 0       | 3                | k=24,32,40,48,56,64,128 |
| 6       | 1                | k=30 |
| 7       | 2                | k=31 |

For each `k`, smaller candidates were checked and found composite.  For
instance at `k = 31` (`c ≡ 3 mod 8`): `2^31 − 3 = 5 × 429496729`,
`2^31 − 11 = 3 × 715827879`, so the first prime is `2^31 − 19`.

**M31** (`2^31 − 1`, Mersenne prime, `q ≡ 7 mod 8`) is included for
comparison with plonky3 even though it does not satisfy `q ≡ 5 (mod 8)`.

## Primes benchmarked

| Label | Modulus | Offset | Rust type | SIMD width |
|-------|---------|--------|-----------|------------|
| fp32_24b | `2^24 − 3` | 3 | `Fp32` | AVX-512: 16, AVX2: 8, NEON: 4 |
| fp32_30b | `2^30 − 35` | 35 | `Fp32` | AVX-512: 16, AVX2: 8, NEON: 4 |
| fp32_31b | `2^31 − 19` | 19 | `Fp32` | AVX-512: 16, AVX2: 8, NEON: 4 |
| fp32_m31 | `2^31 − 1` | 1 | `Fp32` | AVX-512: 16, AVX2: 8, NEON: 4 |
| fp32_32b | `2^32 − 99` | 99 | `Fp32` | AVX-512: 16, AVX2: 8, NEON: 4 |
| fp64_40b | `2^40 − 195` | 195 | `Fp64` | AVX-512: 8, AVX2: 4, NEON: 2 |
| fp64_48b | `2^48 − 59` | 59 | `Fp64` | AVX-512: 8, AVX2: 4, NEON: 2 |
| fp64_56b | `2^56 − 27` | 27 | `Fp64` | AVX-512: 8, AVX2: 4, NEON: 2 |
| fp64_64b | `2^64 − 59` | 59 | `Fp64` | AVX-512: 8, AVX2: 4, NEON: 2 |
| fp128 | `2^128 − 275` | 275 | `Fp128` | AVX-512: 8 (SoA), AVX2: 4 (SoA), NEON: 2 (SoA) |

---

## AMD Zen 5 (Ryzen 9950X / leopard)

Backend: **AVX-512** (16-wide Fp32, 8-wide Fp64, 8-wide Fp128 SoA with
vectorized add/sub and scalar-per-lane mul).
`RUSTFLAGS='-C target-cpu=native'`, nightly toolchain.

### Scalar (`throughput/`)

| Field | mul | add |
|-------|-----|-----|
| fp32_24b | 1.224 Gelem/s | 2.050 Gelem/s |
| fp32_30b | 1.220 Gelem/s | 2.026 Gelem/s |
| fp32_31b | 1.212 Gelem/s | 1.866 Gelem/s |
| fp32_m31 | 1.355 Gelem/s | 1.993 Gelem/s |
| fp32_32b | 1.219 Gelem/s | 1.955 Gelem/s |
| fp64_40b | 1.018 Gelem/s | 2.074 Gelem/s |
| fp64_48b | 1.021 Gelem/s | 2.073 Gelem/s |
| fp64_56b | 0.945 Gelem/s | 2.060 Gelem/s |
| fp64_64b | 0.927 Gelem/s | 1.840 Gelem/s |
| fp128 | 0.452 Gelem/s | 1.127 Gelem/s |

### Packed (`packed_throughput/`)

| Field | mul | add | sub |
|-------|-----|-----|-----|
| fp32_24b | 5.362 Gelem/s | 12.76 Gelem/s | 12.74 Gelem/s |
| fp32_30b | 6.145 Gelem/s | 13.53 Gelem/s | 13.55 Gelem/s |
| fp32_31b | 6.187 Gelem/s | 13.53 Gelem/s | 13.54 Gelem/s |
| fp32_m31 | 6.943 Gelem/s | 13.56 Gelem/s | 13.50 Gelem/s |
| fp32_32b | 6.785 Gelem/s | 13.02 Gelem/s | 12.66 Gelem/s |
| fp64_40b | 1.961 Gelem/s | 5.847 Gelem/s | 5.861 Gelem/s |
| fp64_48b | 1.942 Gelem/s | 5.852 Gelem/s | 5.853 Gelem/s |
| fp64_56b | 1.937 Gelem/s | 5.847 Gelem/s | 5.796 Gelem/s |
| fp64_64b | 1.742 Gelem/s | 5.278 Gelem/s | 5.760 Gelem/s |
| fp128 | 0.284 Gelem/s | 2.314 Gelem/s | 3.175 Gelem/s |

### Packed speedup over scalar

| Field | mul | add |
|-------|-----|-----|
| fp32_24b | **4.4x** | **6.2x** |
| fp32_30b | **5.0x** | **6.7x** |
| fp32_31b | **5.1x** | **7.3x** |
| fp32_m31 | **5.1x** | **6.8x** |
| fp32_32b | **5.6x** | **6.7x** |
| fp64_40b | **1.9x** | **2.8x** |
| fp64_48b | **1.9x** | **2.8x** |
| fp64_56b | **2.0x** | **2.8x** |
| fp64_64b | **1.9x** | **2.9x** |
| fp128 | **0.6x** | **2.1x** |

### Sumcheck MACC (`packed_sumcheck_mix/`)

`acc += eq[i] * poly[i]` loop (dominant inner loop in sumcheck provers).

| Field | MACC | % of pure mul |
|-------|------|---------------|
| fp32_24b | 4.764 Gelem/s | 89% |
| fp32_30b | 5.352 Gelem/s | 87% |
| fp32_31b | 5.351 Gelem/s | 86% |
| fp32_m31 | 6.097 Gelem/s | 88% |
| fp32_32b | 3.409 Gelem/s | 50% |
| fp64_40b | 1.488 Gelem/s | 76% |
| fp64_48b | 1.492 Gelem/s | 77% |
| fp64_56b | 1.491 Gelem/s | 77% |
| fp64_64b | 1.141 Gelem/s | 65% |
| fp128 | 0.323 Gelem/s | 114% |

---

## Apple M4 Pro (macOS / aarch64)

Backend: **NEON** (4-wide Fp32, 2-wide Fp64, 2-wide Fp128 SoA).
`RUSTFLAGS='-C target-cpu=native'`, nightly toolchain.

### Scalar (`throughput/`)

| Field | mul | add |
|-------|-----|-----|
| fp32_24b | 1.129 Gelem/s | 1.426 Gelem/s |
| fp32_30b | 1.133 Gelem/s | 1.425 Gelem/s |
| fp32_31b | 1.043 Gelem/s | 1.433 Gelem/s |
| fp32_m31 | 1.319 Gelem/s | 1.435 Gelem/s |
| fp32_32b | 1.135 Gelem/s | 1.423 Gelem/s |
| fp64_40b | 0.871 Gelem/s | 1.446 Gelem/s |
| fp64_48b | 0.886 Gelem/s | 1.385 Gelem/s |
| fp64_56b | 0.891 Gelem/s | 1.442 Gelem/s |
| fp64_64b | 0.923 Gelem/s | 1.443 Gelem/s |
| fp128 | 0.444 Gelem/s | 0.938 Gelem/s |

### Packed (`packed_throughput/`)

| Field | mul | add | sub |
|-------|-----|-----|-----|
| fp32_24b | 3.717 Gelem/s | 5.272 Gelem/s | 5.278 Gelem/s |
| fp32_30b | 3.719 Gelem/s | 5.281 Gelem/s | 5.275 Gelem/s |
| fp32_31b | 3.719 Gelem/s | 5.283 Gelem/s | 5.268 Gelem/s |
| fp32_m31 | 3.720 Gelem/s | 5.263 Gelem/s | 5.263 Gelem/s |
| fp32_32b | 2.524 Gelem/s | 5.296 Gelem/s | 5.253 Gelem/s |
| fp64_40b | 1.253 Gelem/s | 2.648 Gelem/s | 2.645 Gelem/s |
| fp64_48b | 1.254 Gelem/s | 2.650 Gelem/s | 2.643 Gelem/s |
| fp64_56b | 1.255 Gelem/s | 2.632 Gelem/s | 2.650 Gelem/s |
| fp64_64b | 1.399 Gelem/s | 2.639 Gelem/s | 2.602 Gelem/s |
| fp128 | 0.480 Gelem/s | 1.724 Gelem/s | 2.107 Gelem/s |

### Packed speedup over scalar

| Field | mul | add |
|-------|-----|-----|
| fp32_24b | **3.3x** | **3.7x** |
| fp32_30b | **3.3x** | **3.7x** |
| fp32_31b | **3.6x** | **3.7x** |
| fp32_m31 | **2.8x** | **3.7x** |
| fp32_32b | **2.2x** | **3.7x** |
| fp64_40b | **1.4x** | **1.8x** |
| fp64_48b | **1.4x** | **1.9x** |
| fp64_56b | **1.4x** | **1.8x** |
| fp64_64b | **1.5x** | **1.8x** |
| fp128 | **1.1x** | **1.8x** |

### Sumcheck MACC (`packed_sumcheck_mix/`)

| Field | MACC | % of pure mul |
|-------|------|---------------|
| fp32_24b | 2.652 Gelem/s | 71% |
| fp32_30b | 2.660 Gelem/s | 72% |
| fp32_31b | 2.662 Gelem/s | 72% |
| fp32_m31 | 2.661 Gelem/s | 72% |
| fp32_32b | 1.991 Gelem/s | 79% |
| fp64_40b | 0.990 Gelem/s | 79% |
| fp64_48b | 0.991 Gelem/s | 79% |
| fp64_56b | 0.993 Gelem/s | 79% |
| fp64_64b | 0.795 Gelem/s | 57% |
| fp128 | 0.450 Gelem/s | 94% |

---

## Notes

### Zen 5 AVX-512 observations

- **Fp32 add/sub** saturate at ~13–13.5 Gelem/s, close to 1 cycle per
  16-wide vector at 5 GHz.
- **M31 is the fastest Fp32 prime** for packed mul (6.94 Gelem/s) and
  sumcheck MACC (6.10 Gelem/s), because `C = 1` minimizes the reduction
  chain.
- **Fp32 mul** is latency-bound by the 2-fold Solinas reduction chain
  (~18 cycles per vector).
- **Fp64 sub-word** primes (40b, 48b, 56b) show nearly identical packed
  mul throughput (~1.94 Gelem/s), since the vectorized schoolbook
  multiply + Solinas reduction dominates regardless of bit-width.
- **Fp64 64b** is slower than sub-word variants due to multi-stage
  overflow tracking in the Solinas reduction.
- **Fp32 32b sumcheck MACC** drops to 50% of pure mul, the worst ratio,
  because the carry-based add correction creates additional dependencies
  in the `acc += eq * poly` loop.
- **Fp128** packed backend now uses SoA layout (8-wide) with vectorized
  add/sub via `__m512i`.  Add improved **2.1x** (1.13 → 2.31 Gelem/s),
  sub improved **2.4x** (1.34 → 3.18 Gelem/s).  Mul remains scalar
  per-lane and regressed 0.6x (0.45 → 0.28 Gelem/s) due to SoA
  pack/unpack overhead.  Sumcheck MACC is -7% (0.35 → 0.32 Gelem/s);
  MACC exceeds pure-mul throughput (114%) because the accumulation loop
  avoids the SoA store overhead that the throughput benchmark incurs.

### M4 Pro NEON observations

- **NEON is 4-wide for Fp32, 2-wide for Fp64/Fp128**, so maximum
  speedup is 4x and 2x respectively (vs 16x/8x for AVX-512).
- **Fp32 packed mul** is uniform at ~3.72 Gelem/s for all sub-word
  primes (24b–31b, including M31), unlike Zen 5 where M31 is notably
  faster.  The 4-wide NEON `vmull_u32` + reduction is the bottleneck.
- **Fp32 32b packed mul** drops to 2.52 Gelem/s (carry-based path).
- **Fp64 packed mul** ~1.25 Gelem/s for all sub-word primes, ~1.40
  for 64b — the 64b prime is *faster* on NEON, opposite to Zen 5.
- **Fp128 packed add** 1.72 Gelem/s = **1.8x** scalar speedup; sub
  2.11 Gelem/s = **2.2x**, both close to the theoretical 2x from
  2-wide `uint64x2_t`.  Mul 0.48 Gelem/s ≈ **1.1x** (scalar per-lane).
- **Sumcheck MACC** is 71–72% of pure mul for sub-word Fp32, 79% for
  Fp64 sub-word, and 94% for Fp128 — higher than Zen 5 ratios.

### Reduction strategy by field width

| Field | Reduction method |
|-------|-----------------|
| Fp32, BITS ≤ 31 | `min(t, t−P)` — single unsigned compare + blend |
| Fp32, BITS = 32 | carry-based: detect overflow, conditionally add `C`, then subtract `P` if `≥ P` |
| Fp64, BITS ≤ 62 | 2-fold Solinas in u64: `(lo & mask) + c * (lo >> k)`, repeat |
| Fp64, BITS = 64 | vectorized schoolbook 64×64→128 + multi-stage Solinas with overflow tracking |
| Fp128 add/sub | vectorized 128-bit add/sub with carry/borrow propagation via `__m512i` |
| Fp128 mul | scalar per-lane: 9-limb Solinas via u64 decomposition |
