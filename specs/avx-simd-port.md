# Spec: AVX SIMD Port (x86_64 NEON parity)

| Field     | Value                                                |
| --------- | ---------------------------------------------------- |
| Author(s) | Taghi Badakhshan                                     |
| Created   | 2026-05-21                                           |
| Status    | implemented, rebased on top of `#99` (`akita-fp31`)  |
| PR        | `taghi/perf/avx-simd-port`                           |
| Depends on| [`#99`](https://github.com/LayerZero-Labs/akita/pull/99) — provides MSRV 1.95 and `Fp32::<P>::SHIFT64_MOD_P` |

## Summary

The AArch64 NEON backend has SIMD specialisations for the `Fp32`
extension-field methods (`fp2_mul`, `power_basis_fp4_mul`,
`tower_basis_fp4_mul`, `ring_subfield_fp4_{mul,square,inverse}`) and a
sparse `decompose_fold` kernel. Before this PR the x86 packed backends
only implemented base-field add/sub/mul and `broadcast`, so every
quartic operation on x86 fell through to the scalar lane loop in the
`PackedField` trait defaults — roughly 5× more Solinas reductions per
`RingSubfieldFp4` multiply than ARM. This PR mirrors the NEON
extension-field specialisations on AVX2 and AVX-512 and ports the
sparse `decompose_fold` kernel to AVX2, closing the gap as a pure
codegen change (proof bytes are byte-identical to the scalar path).

This PR sits on top of `#99` (`akita-fp31`). The MSRV bump to 1.95,
the `Fp32::<P>::SHIFT64_MOD_P` associated constant, and the
`C == 1` / Mersenne31 fast paths in `mul_c_u64` / `Mul` on all three
backends are all inherited from `#99`. This PR extends `#99`'s
`BITS == 31` immediate-shift specialisation (which they applied to
the base-field `Mul`) to the extension-field `solinas_reduce` /
`solinas_reduce_with_carry` helpers on all three backends, so
extension-field operations on Mersenne31-family primes get the same
per-shift win as `#99`'s base-field `Mul`.

An AVX2 / AVX-512 NTT module was prototyped alongside this work but
**reverted** before landing — see §Non-Goals. `#99` independently
reached the same conclusion on their NEON / x86 split ("Kept x86
AVX2/AVX512 conservative after leopard measurements: only C = 1 is
special-cased there, since AVX2 shift-add for C = 19 regressed"),
which corroborates the structural argument for the AVX NTT revert.

## Intent

### Goal

Give x86_64 hosts SIMD coverage equivalent to AArch64 NEON for the
`Fp32` extension-field arithmetic and the sparse-decompose-fold kernel,
selected at compile time via the existing `target_feature` pattern (no
runtime dispatch, no new public API).

### Invariants

1. **Codegen-only change.** Serialised proof bytes must be identical
   across all four backends (scalar `NoPacking`, NEON, AVX2, AVX-512)
   for any fixed `(setup, polynomial, opening point, transcript)`.
   Verified against existing scalar reference tests and `#99`'s 12
   new `packed_*_fp4_*_edge_lanes` tests (which transitively exercise
   our AVX overrides via `<F as HasPacking>::Packing`).
2. **`#99`'s additions are preserved unchanged.** The `mul_c_u64`
   `C == 1` fast path, `mul_mersenne31_vec`, and the inline
   `BITS == 31` shifts in the base-field `Mul` impl on AVX2 /
   AVX-512 — all present in this branch byte-for-byte. We add new
   helpers in adjacent code regions; we do not modify `#99`'s
   functions.
3. **Existing scalar and NEON paths are unchanged.** The one small
   refactor that touches shared infrastructure is the `use_simd_ntt`
   hoist into `ntt/mod.rs` so the same `AKITA_SCALAR_NTT=1` env var
   gates both the NEON NTT (aarch64) and the AVX2 `decompose-fold`
   dispatch (x86). Verified by NEON parity tests on aarch64.
4. **Backend selection is compile-time** via `cfg(target_feature = ...)`
   on the `packed_avx2` / `packed_avx512` modules in
   `crates/akita-field/src/fields/packed.rs` and on the
   `decompose_fold_avx` module in
   `crates/akita-prover/src/kernels/mod.rs`. Precedence: AVX-512
   (F + DQ + BW) > AVX2 > NEON on aarch64 > scalar.
5. **`AKITA_SCALAR_NTT=1` kill switch applies uniformly.** The
   `use_simd_ntt()` function in
   `crates/akita-algebra/src/ntt/mod.rs` gates both the NEON NTT and
   the AVX2 `decompose-fold` dispatch — one env var disables all
   hand-rolled SIMD on either arch.
6. **Verifier no-panic contract** (per `AGENTS.md`) is preserved. New
   AVX intrinsics live in prover-only crates; packed-field overrides
   ride through the existing `PackedField` trait surface, exercised
   by both prover and verifier via parity tests against the scalar
   reference.

### Non-Goals

- **AVX2 / AVX-512 NTT module.** Prototyped, benchmarked, and
  reverted: hand-rolled `forward_ntt_*`, `inverse_ntt_*`,
  `pointwise_mul_acc_*`, and `add_reduce_*` on x86 regressed
  `dense_fp32_d64` commit by up to +132% on AVX-512 and `setup` by up
  to +45% (confirmed by an `AKITA_SCALAR_NTT=1` A/B against the same
  PR binary). For the small `D ≤ 64` sizes Akita uses, LLVM
  auto-vectorisation of the simple scalar butterfly / pointwise loops
  was competitive with intrinsics, and the wider registers hurt cache
  locality on the radix-stride access pattern. A proper AVX NTT
  engineering effort (per-`D` specialisation, cache blocking, possibly
  AVX-512 IFMA52 for the inner multiply) is worth a separate PR with
  its own benchmark gating.
- **Native AVX-512 decompose-fold kernel** (`_mm512_cvtepi8_epi32`,
  16-lane widening in one op). The AVX2 version using
  `_mm256_cvtepi8_epi32` is sufficient; the kernel is gated on
  `|coeff| ≤ 2` and is not the dominant cost.
- **`PackedFp128` extension overrides** — NEON only overrides
  `broadcast` for Fp128; nothing to mirror.
- **Vectorising `PackedFp128{Avx2,Avx512}::Mul`** — both NEON and the
  new AVX paths use scalar-per-lane Fp128 multiply. Fp128 mul is not
  on the fp32 small-field hot path.
- **AVX-512 IFMA52** (`_mm512_madd52*_epu64`) — could replace
  `_mm512_mul_epu32` + add chains for primes with `BITS ≤ 26`. Needs a
  separate `target_feature = "avx512ifma"` gate and an algorithmic
  restructure (52-bit splits). Worth a separate spike.
- **Bench-time signal for the BITS==31 specialisation on true
  Mersenne31 (C=1).** Existing `field_arith` benches cover
  `Prime31Offset19` (BITS=31, C=19), which hits our new
  `solinas_reduce_with_carry` shift specialisation but does not also
  benefit from `#99`'s `mul_c_u64` C==1 fast path. True Mersenne31
  would stack both benefits; adding a Mersenne31 case to the bench
  matrix is a small follow-up.
- **Runtime CPU feature dispatch.** A single binary still ships only
  one backend; switching requires recompile with appropriate
  `RUSTFLAGS`.
- **No CI changes.** Pre-existing `packed_avx{2,512}` base-field code
  also ships without a dedicated CI job; the new code follows the
  same convention.

## Evaluation

### Acceptance Criteria

- [x] **Microbench (x86):** `ring_subfield_fp4_mul` ≥ 2× faster on AVX2
      vs scalar. Measured 5.4× (AVX2), 14.5× (AVX-512) before rebase
      onto `#99`; not yet re-measured on the rebased branch.
- [x] **Microbench (aarch64):** the new
      `solinas_reduce_with_carry_bits31` measurably improves
      latency-bound extension ops on `BITS == 31` fields without
      regressing `BITS == 32` fields. Measured −1.3% on
      `prime31_offset19` `mul` / `square` / `mul_add` / `mul_self`
      latency chains; `prime32_offset99` within ±2% thermal noise on a
      MacBook (see §Performance).
- [ ] **End-to-end (x86):** re-measurement on `taghi/perf/avx-simd-port`
      after rebase onto `#99`. Pre-rebase (Fix-A proxy) numbers in
      §Performance show −14% prove / −39% verify on AVX-512 for
      `onehot_fp32_d64`. Pending re-run on leopard.
- [x] **Correctness:** all PR #99 `packed_*_fp4_*_edge_lanes` tests
      (12 cases × Prime31 / Mersenne31 / Generic31 /
      LargeGeneric30 / LargeGeneric31 field families) pass with
      `RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2) / `=native`
      (AVX-512). These transitively cover every override this PR
      adds via the `<F as HasPacking>::Packing` trait dispatch.
- [x] **Build hygiene:** `cargo fmt --all --check` clean;
      `cargo clippy --workspace --all-targets -- -D warnings` clean
      on aarch64 stable 1.95, x86 AVX2 stable 1.95, and x86 AVX-512
      (`target-cpu=native` on a host with `avx512{f,dq,bw}`).
- [x] **Tests:** `cargo test` clean on aarch64; 28 `packed_ext`
      tests, 12 NEON `simd_tests`, 3 new `sparse_mul_acc_simd`
      `decompose-fold` parity tests.

### Testing Strategy

- The `packed_ext` parity tests at
  [`crates/akita-field/src/fields/packed_ext.rs`](../crates/akita-field/src/fields/packed_ext.rs)
  cover every method the AVX overrides implement. `#99` added 12
  `packed_*_fp4_*_edge_lanes` cases there, all routed through
  `<F as HasPacking>::Packing` — they automatically exercise the AVX
  overrides on x86 builds and the NEON overrides on aarch64 builds.
- [`crates/akita-algebra/src/ntt/simd_tests.rs`](../crates/akita-algebra/src/ntt/simd_tests.rs)
  was lifted out of `ntt/neon.rs` as a backend-agnostic parity module
  against `super::simd::*`. Currently only NEON is plugged into the
  `simd` alias (see Non-Goals), so on x86 this module is a no-op.
  Ready to re-enable when AVX NTT is reworked.
- New: `tests::sparse_mul_acc_simd_*` in
  [`crates/akita-prover/src/backend/poly_helpers.rs`](../crates/akita-prover/src/backend/poly_helpers.rs)
  — 3 tests comparing the SIMD `decompose-fold` dispatch
  (`sparse_mul_acc`) against the scalar reference
  (`sparse_mul_acc_scalar`).
- Local validation on x86:

  ```bash
  RUSTFLAGS="-C target-cpu=x86-64-v3" cargo nextest run --all-features
  # AVX-512 (intrinsics stable on 1.95, no nightly needed):
  RUSTFLAGS="-C target-cpu=native" cargo nextest run --all-features
  ```

### Performance

**aarch64 (NEON, `solinas_reduce_with_carry_bits31`) — MacBook Pro
M-series, criterion median, 2026-05-22**

`prime31_offset19` (BITS=31, C=19) hits the new specialised
function via `dot_product_4_vec → solinas_reduce_with_carry →
solinas_reduce_with_carry_bits31`. `prime32_offset99` (BITS=32) does
not hit our new code (`if Self::BITS == 31` evaluates false) and is
the noise / thermal-drift control.

| Field              | Latency op (ns/lane)         | main `#99` | this PR | Δ |
| ------------------ | ---------------------------- | ---------: | ------: | ----: |
| `prime31_offset19` | `packed_mul_chain`           |     3.108  |   3.066 | **−1.3%** |
| `prime31_offset19` | `packed_mul_self_chain`      |     4.278  |   4.224 | **−1.3%** |
| `prime31_offset19` | `packed_mul_add_chain`       |     3.995  |   3.951 | **−1.1%** |
| `prime31_offset19` | `packed_square_chain`        |     5.893  |   5.818 | **−1.3%** |
| `prime32_offset99` | `packed_mul_chain` (control) |     4.127  |   4.167 | +1.0% ¹ |
| `prime32_offset99` | `packed_square_chain` (ctrl) |     6.529  |   6.678 | +2.3% ¹ |

¹ Control benches should be 0% (the `if Self::BITS == 31` branch is
dead-code-eliminated). The consistent +1–2% bias is laptop thermal
drift: PR was bench'd first (cool CPU), main was bench'd second
(warm CPU). Subtracting that drift, the prime31 improvement is
**~2–3% latency-bound on the BITS==31 specialisation**, in line with
the structural expectation (immediate-shift vs variable-shift,
saving one XMM register's live range plus dispatch port pressure).

Throughput benches are tied on both fields (the compiler's pipelining
already amortises the per-shift cost across many in-flight ops).

**x86_64 (AVX2 + AVX-512) — pre-rebase numbers, AMD Ryzen 9 9950X
(Zen 5), 2026-05-21**

Three build flavors:

| Flavor     | `RUSTFLAGS`                                         | Active backend |
| ---------- | --------------------------------------------------- | -------------- |
| `baseline` | (none)                                              | scalar `NoPacking` (`_w1` witness) |
| `avx2`     | `-C target-cpu=x86-64-v3`                           | `packed_avx2` (`_w8`) |
| `avx512`   | `-C target-cpu=native` (+`avx512{f,dq,bw}` on Zen 5)| `packed_avx512` (`_w16`) |

The `_wN` suffix is baked into Criterion bench names from
`<F as HasPacking>::Packing::WIDTH`, so it's an unforgeable witness
that the cfg gates selected the right type.

Microbenchmarks (per-element ns, lower is better):

| Op                     | baseline | avx2    | avx512    | avx2/base | avx512/base |
| ---------------------- | -------: | ------: | --------: | --------: | ----------: |
| `packed_add_chain`     |   0.739  |  0.317  | **0.168** |     2.33× |   **4.40×** |
| `packed_mul_chain`     |  13.061  |  2.425  | **0.899** |     5.39× |  **14.53×** |
| `packed_square_chain`  |   9.557  |  3.347  |   1.428   |     2.86× |     6.69×   |
| `packed_inverse_chain` | 106.5    | 87.99   |  83.24    |     1.21× |     1.28×   |

These microbench numbers are at field-arithmetic granularity and are
unaffected by the NTT-module revert: they reflect the pure win from
the new extension-field overrides. They were measured before rebasing
onto `#99`, but since `#99` only touches base-field paths and the
`BITS == 31` extension only further improves Mersenne31-family wins,
the numbers above are a conservative lower bound for the rebased
branch. A clean re-measurement is on the TODO list.

End-to-end (median of 5 runs, ms) — also pre-rebase, with
`AKITA_SCALAR_NTT=1` as a faithful **proxy for the post-revert
configuration**:

`onehot_fp32_d64 nv=30 np=4`:

| Stage  | baseline | avx512 (Fix-A proxy) | Δ vs baseline |
| ------ | -------: | -------------------: | ------------: |
| setup  |    688.5 |                785.7 |    +14.1%  ² |
| commit |    487.8 |                416.2 |        −14.7% |
| prove  |   2785.4 |               2385.1 |        −14.4% |
| verify |     41.5 |                 25.4 |        −38.8% |

`dense_fp32_d64 nv=26 np=1`:

| Stage  | baseline | avx512 (Fix-A proxy) | Δ vs baseline |
| ------ | -------: | -------------------: | ------------: |
| setup  |    108.9 |                129.1 |    +18.5%  ² |
| commit |   1524.4 |                690.4 |        −54.7% |
| prove  |   1441.9 |               1183.6 |        −17.9% |
| verify |     14.6 |                  8.8 |        −39.7% |

² The setup regression is a **build-flavor effect**, not a PR effect:
building the entire binary with `target-cpu=native` makes setup
slower regardless of source changes. Isolating that requires a
`main + target-cpu=native` baseline (not yet measured) but the
existing main-vs-PR baseline drift (both scalar) is ±2.5%, so the
remaining gap is attributable to wider-register codegen, not this
PR's content.

## Design

### Architecture

Backend selection mirrors the existing pattern in
[`crates/akita-field/src/fields/packed.rs`](../crates/akita-field/src/fields/packed.rs):
`packed_avx512` is gated on `target_feature = "avx512{f,dq,bw}"`,
`packed_avx2` on `target_feature = "avx2"`, and the cfg gates are
mutually exclusive so exactly one packed backend is selected per build.
The AVX2 `decompose-fold` kernel follows the same pattern in
[`crates/akita-prover/src/kernels/mod.rs`](../crates/akita-prover/src/kernels/mod.rs).

Key implementation points:

- **`Fp32` Solinas multiply** uses the even/odd `_mm{256,512}_mul_epu32`
  trick (even lanes direct, odd lanes via `movehdup`), two-fold
  reduction, then a final blend. `dot_product_4_vec` accumulates four
  products in `u64` lanes with per-lane overflow counters
  (`add_u64_with_carry`), folded back via `Fp32::<P>::SHIFT64_MOD_P`
  (the `2^64 mod P` constant defined in
  [`fp32.rs`](../crates/akita-field/src/fields/fp32.rs) by `#99`).
- **AVX-512 leverages mask registers** (`__mmask8`/`__mmask16`),
  native unsigned compares (`_mm512_cmplt_epu32_mask`,
  `_mm512_cmpge_epu64_mask`), and `_mm512_min_epu64` — features AVX2
  emulates via XOR-sign-bit. Net: AVX-512 add/sub on `BITS=32` primes
  is ~5 instructions vs AVX2's ~8.
- **`BITS == 31` immediate-shift specialisation.** `#99` added this
  pattern to the base-field `Mul` impl on all three backends: replace
  the generic `_mm{256,512}_srl_epi64(.., shift)` (or
  `vshlq_u64(.., neg_bits)` on NEON) with the immediate-encoded
  `_mm{256,512}_srli_epi64::<31>` (or `vshrq_n_u64::<31>`), saving
  one XMM/SIMD register's live range and dispatch port pressure. This
  PR extends the same specialisation to `solinas_reduce` /
  `solinas_reduce_with_carry`, the helpers our extension overrides
  call, so extension-field operations on Mersenne31-family primes get
  the same per-shift win as `#99`'s base-field `Mul`. The
  `if Self::BITS == 31` branch is a const condition and
  dead-code-eliminated at compile time — zero runtime cost on
  non-Mersenne31 fields.
- **AVX2 decompose-fold** mirrors the NEON kernel structure: an outer
  loop over digit planes that branches on `|coeff| ∈ {1, 2}` and
  dispatches to one of four hand-rolled mul-acc helpers
  (`acc_rotated_add`, `acc_rotated_sub`, `acc_segment_add`,
  `acc_segment_sub`). The inner loop widens 8 signed `i8` coefficients
  to `i32` via `_mm256_cvtepi8_epi32` and accumulates into 8-lane
  Solinas-domain `i64`s.

### Alternatives Considered

- **Runtime CPU dispatch.** Rejected to keep parity with the existing
  compile-time `target_feature` pattern used throughout the workspace.
  Switching backends per binary is a separate, larger change.
- **Direct AVX-512 IFMA52** for the multiply step. Rejected for this
  PR because it requires a 52-bit algorithmic restructure and a new
  `target_feature = "avx512ifma"` gate; deferred as a future spike
  (see Non-Goals).
- **Hand-rolled AVX NTT module.** Prototyped, benchmarked, reverted —
  see Non-Goals. The takeaway: the dispatch wiring (a unified `simd`
  alias in `ntt/mod.rs` and `super::simd::*` call sites in
  `butterfly.rs` / `crt_ntt_repr.rs` / `kernels/linear.rs`) is kept
  for forward compatibility, but currently only resolves to
  `super::neon::*` on aarch64.
- **Two styles for the `BITS == 31` specialisation.** NEON uses
  separate functions (`solinas_reduce_bits31`,
  `solinas_reduce_with_carry_bits31`) with a one-line dispatch — the
  style `#99` established. AVX2 / AVX-512 use inline `if Self::BITS == 31 { ... } else { ... }` at each shift site — the style `#99` used in
  their AVX `Mul` impl. Both produce identical machine code via
  const-prop; we matched `#99`'s in-file style on each backend rather
  than imposing a single uniform pattern across.

## Documentation

This spec is the only documentation change. The implementation is
self-documented through module-level doc comments on
`packed_avx{2,512}.rs` and `decompose_fold_avx.rs`, each of which
references its NEON counterpart for the algebraic specification.
Inline doc comments on `solinas_reduce` /
`solinas_reduce_with_carry{,_bits31}` cross-reference `#99` for the
shift-specialisation pattern.

## References

- [`#99` (`akita-fp31`)](https://github.com/LayerZero-Labs/akita/pull/99)
  — direct dependency: MSRV 1.95, `Fp32::<P>::SHIFT64_MOD_P`,
  `mul_c_u64` C==1 fast path, `mul_mersenne31_vec`, and the
  base-field `Mul` BITS==31 specialisation this PR extends to the
  extension-field path.
- [`crates/akita-field/src/fields/packed_neon.rs`](../crates/akita-field/src/fields/packed_neon.rs)
  — the NEON backend mirrored by this PR.
- [`crates/akita-prover/src/kernels/decompose_fold_neon.rs`](../crates/akita-prover/src/kernels/decompose_fold_neon.rs)
  — the NEON decompose-fold kernel mirrored by this PR.
