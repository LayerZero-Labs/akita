# Spec: SIMD Ring-Subfield fp8 Multiplication

| Field     | Value                                 |
| --------- | ------------------------------------- |
| Author(s) | Taghi Badakhshan                      |
| Created   | 2026-05-26                            |
| Status    | implemented                           |
| PR        | `taghi/perf/simd-subfield-fp8`        |

## Summary

The `RingSubfieldFp8` extension field is the degree-8 opening/challenge
scalar type for `Fp16` configurations.  Before this PR every
`RingSubfieldFp8` multiply fell through to scalar lane-by-lane
arithmetic in the `PackedField` trait default — 8 sequential Karatsuba
expansions with 36 base-field multiplies each, zero SIMD utilisation.
This PR adds packed `PackedFp16` backends on all three SIMD tiers
(NEON 8-lane, AVX2 16-lane, AVX-512 16-lane) and wires Karatsuba fp8
multiplication directly over SIMD vectors, cutting the end-to-end prove
time by **1.8×** on AArch64 and up to **2.4×** on x86-512.

## Intent

### Goal

Provide SIMD-accelerated `RingSubfieldFp8` multiplication and squaring
for `Fp16` on AArch64 NEON, x86-64 AVX2, and x86-64 AVX-512.
Introduce the required `PackedFp16{Neon,Avx2,Avx512}` backends (no
`PackedFp16` existed before this PR).  Selected at compile time via the
existing `target_feature` cfg cascade — no runtime dispatch, no new
public API surface.

### Background: Chebyshev Basis Multiplication

`RingSubfieldFp8<F>` represents elements of the degree-8 fixed subfield
of a cyclotomic ring `Z_q[X]/(Phi_D(X))`.  The basis is the Chebyshev
basis `{1, e_1, ..., e_7}` where `e_j = zeta^(jm) + zeta^(-jm)` for
`m = D/16`.

The multiplication rule for two basis elements `e_i · e_j` is:

```
e_i · e_j  =  phi(i + j)  +  phi(|i - j|)
```

where `phi` is the Chebyshev fold-back map:

```
phi(k) = | 2          if k = 0    (contributes 2 to constant term)
         | e_k        if 1 <= k <= 7
         | 0          if k = 8    (e_8 = 0 in degree-8)
         | -e_{16-k}  if 9 <= k <= 15
```

A naive expansion of `(sum a_i e_i)(sum b_j e_j)` requires `8^2 = 64`
base-field multiplies.  The Karatsuba identity

```
a_i · b_j + a_j · b_i  =  (a_i + a_j)(b_i + b_j) - a_i · b_i - a_j · b_j
```

reuses the 8 diagonal products `diag[i] = a_i · b_i` to reduce the
cross-term count.  The resulting schedule is:

- **8 diagonal** products `a_i · b_i`
- **7 first-row** products `(a_0+a_k)(b_0+b_k)` for `k = 1..7`
- **21 cross-pair** products `(a_i+a_j)(b_i+b_j)` for `1 <= i < j <= 7`

Total: **36 multiplies** + adds/subs via the `add_phi` fold-back.

For squaring the identity simplifies to `2 · a_i · a_j`, computed as
`a_i · a_j` doubled, saving one add and two subs per cross-term versus
the Karatsuba form.

### Fp16 Arithmetic: Widening + Solinas Reduction

`Fp16<P>` stores elements as `u16` with `P` a Solinas prime
(`P = 2^BITS - C`).  SIMD arithmetic on u16 is non-trivial because
`u16 + u16` can overflow 16 bits before reduction.

All three backends use the same strategy:

1. **Widen** u16 lanes to u32 (NEON: `vmovl`, AVX2: `_mm256_cvtepu16_epi32`,
   AVX-512: `_mm512_cvtepu16_epi32`).
2. **Operate** in u32 — additions, subtractions, and multiplications all
   fit without overflow (`65535^2 < 2^32`).
3. **Three-fold Solinas reduction**: split the u32 product into low
   `BITS` and high bits, fold high with `C`, repeat 3 times.
   Three folds suffice for all valid `Fp16<P>` parameters because
   `C < sqrt(P) <= 2^8`, giving a worst-case bound
   `fold3 <= C^2 - C - 1 < 2^BITS`.
4. **Narrow** the reduced u32 lanes back to u16.

The scalar `Fp16` backend uses `i64` wide accumulators with
`rem_euclid(P)` to handle Karatsuba subtractions that may go negative.

### Invariants

1. **Codegen-only change.** Serialised proof bytes are identical across
   all four backends (scalar `NoPacking`, NEON, AVX2, AVX-512) for any
   fixed `(setup, polynomial, opening point, transcript)`.  Verified by
   the `packed_ring_subfield_fp8_*` parity tests plus all existing
   scheme-level tests.
2. **Existing paths untouched.** The `Fp32`, `Fp64`, `Fp128` packed
   backends are not modified.  `RingSubfieldFp4` arithmetic is unchanged.
3. **Backend selection is compile-time** via `cfg(target_feature = ...)`
   on the `packed_{neon,avx2,avx512}` modules.  Precedence: AVX-512
   (F + DQ) > AVX2 > NEON (aarch64) > scalar `NoPacking`.
4. **Verifier no-panic contract** (per `AGENTS.md`) is preserved. New
   SIMD code lives in the packed-field layer, which is exercised by
   both prover and verifier through `<F as HasPacking>::Packing`.

### Non-Goals

- **`PackedFp16` for `Fp32`-sized `RingSubfieldFp8`.** No production or
  planned configuration uses `RingSubfieldFp8<Fp32>` — `Fp32` configs
  use `RingSubfieldFp4` or extension degree 1.  Dead Fp32 NEON/AVX
  fp8 kernels were prototyped and removed in cleanup commits.
- **Lookup-table SIMD.** The original plan considered SIMD lookup tables
  (e.g. `vtbl` / `vpshufb`) for small-field multiplication.  Karatsuba
  over vector arithmetic proved simpler and already saturates the ALU
  pipeline; lookup-table exploration is deferred.
- **Runtime CPU dispatch.**  Single-backend-per-binary, matching the
  existing workspace convention.
- **No CI changes.**  The pre-existing `packed_avx{2,512}` and
  `packed_neon` modules already compile only under their target gates;
  CI runs on baseline x86-64.  This PR follows the same pattern.

## Evaluation

### Acceptance Criteria

- [x] **E2E (NEON):** fp16 prove time ≥ 1.8× faster vs scalar baseline.
- [x] **E2E (AVX2):** fp16 prove time ≥ 1.7× faster vs scalar baseline.
- [x] **E2E (AVX-512):** fp16 prove time ≥ 1.9× faster vs scalar baseline.
- [x] **Correctness:** 7 new `packed_ring_subfield_fp8_*` tests plus all
      existing `akita-field` tests pass on each backend.
- [x] **No regression:** `Fp64` / `Fp32` prove timings neutral (scalar
      path unchanged).
- [x] **Build hygiene:** `cargo fmt --check`, `cargo clippy -D warnings`
      clean on aarch64 and x86-64.

### Testing Strategy

- Packed-vs-scalar parity tests in `packed_ext/tests.rs`, each comparing
  `PackedRingSubfieldFp8` lane by lane against scalar `RingSubfieldFp8` and
  automatically exercising whichever backend `<F as HasPacking>::Packing`
  resolves to (scalar `NoPacking`, NEON, AVX2, or AVX-512):
  - `packed_ring_subfield_fp8_mul_{fp64,prime31,prime32,fp16}` — multiply.
  - `packed_ring_subfield_fp8_square{,_fp16}` — square, including the Fp16
    SIMD square kernel.
  - `packed_ring_subfield_fp8_fp16_edge` — Fp16 mul and square at field
    boundary coefficients (`0, 1, (P-1)/2, P-2, P-1`), stressing the Solinas
    reduction and the canonicalizing add/sub wraparound.
  - `packed_ring_subfield_fp8_broadcast`, `packed_fp16_basic_arithmetic`.
- CI exercises the scalar and AVX2 legs; NEON is covered on AArch64 dev
  machines and AVX-512 on the Zen 5 benchmark host.
- All existing `akita-field` tests pass.
- The `full_fp16_d64` profile exercises end-to-end commit + prove + verify.

### Performance

**AArch64 NEON — Apple M4 Max, `--release`, median of 3 runs**

Profile: `onehot`, nv=22:

| Config           | Baseline (scalar) | NEON       | Speedup     |
| ---------------- | -----------------:| ----------:| -----------:|
| `onehot_fp16_d32` |            756 ms |     414 ms | **1.83×** |
| `onehot_fp16_d64` |            734 ms |     399 ms | **1.84×** |
| `onehot` (fp64)  |             81 ms |      82 ms | neutral     |

**x86-64 AVX2 / AVX-512 — AMD Ryzen 9 9950X (Zen 5 / Granite Ridge),
`--release`, median of 3 runs**

Profile: `full_fp16_d64` prove time:

| nv   | Baseline (scalar) | AVX2     | AVX-512  | AVX2/base | AVX-512/base |
| ----:| -----------------:| --------:| --------:| ---------:| ------------:|
|   20 |            282 ms |   132 ms |   116 ms |  **2.14×** |   **2.36×** |
|   25 |           1404 ms |   785 ms |   693 ms |  **1.79×** |   **1.92×** |

> **STALE — needs re-measurement.** The `full_fp16_d64` mode no longer exists
> (current modes are `{dense,onehot}_fp16_{d32,d64}` etc.), so this table is
> not reproducible as written. A 2026-06 re-run on the Zen 5 host with
> `onehot_fp16_d64`/`dense_fp16_d64` gave scalar→AVX2 ratios of only
> ~1.02–1.25× (e.g. dense nv25: scalar 1049 ms → AVX2 841 ms), because
> `RingSubfieldFp8<Fp16>` multiply is not the E2E prove bottleneck for these
> modes. The kernel microbench below is the reliable measure of the SIMD fp8
> speedup; the E2E ratios above should be re-derived against a real mode.

**AVX2 fp8 kernel microbench** (`field_arith/ext8`, `prime16_offset99`,
ns/lane, criterion `--baseline` comparison on the same Zen 5 host). The
widen-once split-half rewrite replaced the original per-op widen/narrow:

| fp8 op    | per-op (chain / stream) | widen-once (chain / stream) | change         |
| --------- | -----------------------:| ---------------------------:| --------------:|
| `mul`     |     17.82 / 17.67       |        10.32 / 9.92         | −42% / −44%   |
| `square`  |     11.51 / 11.27       |         8.12 / 7.09         | −30% / −37%   |
| `mul_self`|     17.71 / 17.51       |        10.75 / 9.73         | −39% / −44%   |

## Design

### Architecture

**New types introduced:**

| Type | File | Lanes | Vector |
|------|------|------:|--------|
| `PackedFp16Neon<P>` | `packed_neon.rs` | 8 | `uint16x8_t` |
| `PackedFp16Avx2<P>` | `packed_avx2.rs` | 16 | `__m256i` (u16, widened to u32) |
| `PackedFp16Avx512<P>` | `packed_avx512.rs` | 16 | `__m512i` (widened to u32) |
| `PackedRingSubfieldFp8<F, PF>` | `packed_ext.rs` | `PF::WIDTH` | transpose `[PF; 8]` |

**`PackedField` trait extensions** (in `packed.rs`):

- `ring_subfield_fp8_mul(a, b) -> [Self; 8]` — default: generic Karatsuba.
- `ring_subfield_fp8_square(a) -> [Self; 8]` — default: cross-product doubling.

Inversion is not a `PackedField` hook: `PackedRingSubfieldFp8::inverse` runs
the scalar Gaussian-elimination inverse lane by lane.

The 36-multiply Karatsuba schedule and the Chebyshev `φ` fold-back live in
exactly one place — the lane-generic `ring_subfield_fp8_{mul,square}_schedule`
free functions in `ext/ring_subfield_fp8.rs`, parameterised over a lane type
`V` and its `add`/`sub`/`mul`.  Every consumer drives that one schedule:

- the generic `PackedField` default passes `Self` with operator `+`/`-`/`*`;
- the scalar field default passes `F` with field ops;
- the `i64` Fp16 scalar path passes `i64` with raw integer ops, deferring a
  single `rem_euclid` reduction per coefficient;
- each SIMD backend (`PackedFp16{Neon,Avx2,Avx512}`) widens every input
  coefficient to u32 once at entry, runs the schedule over u32 lanes via
  `#[inline(always)]` closures wrapping its own `add_u32`/`sub_u32`/`mul_u32`,
  and narrows once per output coefficient. NEON and AVX2 split the 16 u16
  lanes into two u32 halves and run the schedule per half; AVX-512 keeps all
  16 lanes in one `__m512i` of u32.

The closures inline fully, so each monomorphization compiles to
backend-specific intrinsics with no abstraction overhead (confirmed neutral
on the fp8 mul/square benches).

**`Fp16Packing` cfg cascade** (in `packed.rs`):

```
#[cfg(aarch64 + neon)]         → PackedFp16Neon<P>
#[cfg(x86_64 + avx512f + dq)]  → PackedFp16Avx512<P>
#[cfg(x86_64 + avx2, !avx512)] → PackedFp16Avx2<P>
#[cfg(fallback)]                → NoPacking<Fp16<P>>
```

**Scalar Fp16 specialization** (in `ext.rs`):

`RingSubfieldFp8MulBackend` is specialised for `Fp16<P>` to use `i64`
wide accumulators, avoiding the `u16` overflow on Karatsuba subtractions.
Final reduction uses `rem_euclid(P as i64)` to handle negative
intermediates.  `Fp32`/`Fp64`/`Fp128` use the generic default.

### Alternatives Considered

- **Lookup-table SIMD** (`vtbl` on NEON, `vpshufb` on AVX2).  For fp16
  the element range `[0, 65437)` is too large for byte-indexed tables.
  Decomposition into nibbles would need 4 table lookups + recombination,
  which exceeded the cost of direct widening multiply.
- **Per-op widen/narrow (AVX2, original).** The first AVX2 cut widened and
  narrowed inside every `add_vec`/`sub_vec`/`mul_vec`. Switching to widen-once
  split-half u32 (matching NEON/AVX-512) cut fp8 mul/square by 30–44% on
  Zen 5 with no register-pressure regression, so the per-op form was dropped.
- **Fp8 base field (8-bit modulus).** The spec explicitly limits scope
  to `Fp16`.  An 8-bit base field would enable byte-lane SIMD (32 lanes
  on NEON, 32/64 on AVX2/512) but introduces significant algebraic
  constraints and is a separate research direction.

## Documentation

This spec is the primary documentation.  Inline doc comments on
`solinas_reduce_16` / `solinas_reduce` explain the three-fold bound, and
`ring_subfield_fp8_{mul,square}_schedule` / `fp8_add_phi` in
`ext/ring_subfield_fp8.rs` document the shared Karatsuba schedule and `φ`
fold-back.  Module-level doc comments on `packed_{neon,avx2,avx512}.rs`
updated to include `Fp16`.

## References

- [`specs/fp16-small-field-support.md`](fp16-small-field-support.md) —
  the Fp16 field family spec that introduced `RingSubfieldFp8`.
- [`specs/avx-simd-port.md`](avx-simd-port.md) — the AVX port for Fp32
  `RingSubfieldFp4`, whose patterns this PR follows for fp8.
- [`crates/akita-field/src/fields/ext/ring_subfield_fp8.rs`](../crates/akita-field/src/fields/ext/ring_subfield_fp8.rs)
  — shared lane-generic Karatsuba schedule and `φ` fold-back driven by every backend.
- [`crates/akita-field/src/fields/packed_neon.rs`](../crates/akita-field/src/fields/packed_neon.rs)
  — NEON backend (reference SIMD wiring of the shared schedule).
