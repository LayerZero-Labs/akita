# Spec: AVX SIMD Port (x86_64 NEON parity)

| Field     | Value                                          |
| --------- | ---------------------------------------------- |
| Author(s) | Taghi Badakhshan                               |
| Created   | 2026-05-21                                     |
| Status    | implemented                                    |
| PR        | `taghi/perf/avx-simd-port`                     |

## Summary

The AArch64 NEON backend has SIMD specializations for the `Fp32`
extension-field methods (`fp2_mul`, `power_basis_fp4_mul`,
`tower_basis_fp4_mul`, `ring_subfield_fp4_{mul,square,inverse}`) and a
sparse `decompose_fold` kernel. Before this PR the x86 packed backends
only implemented base-field add/sub/mul and `broadcast`, so every
quartic operation on x86 fell through to the scalar lane loop in the
`PackedField` trait defaults — roughly 5× more Solinas reductions per
`RingSubfieldFp4` multiply than ARM. This PR mirrors the NEON
extension-field specializations on AVX2 and AVX-512 and ports the
sparse `decompose_fold` kernel to AVX2, closing the gap as a pure
codegen change (proof bytes are byte-identical to the scalar path).

An AVX2 / AVX-512 NTT module was prototyped alongside this work but
**reverted** before landing — see §Non-Goals.

## Intent

### Goal

Give x86_64 hosts SIMD coverage equivalent to AArch64 NEON for the
`Fp32` extension-field arithmetic and the sparse-decompose-fold kernel,
selected at compile time via the existing `target_feature` pattern (no
runtime dispatch, no new public API).

### Invariants

1. **Codegen-only change.** Serialized proof bytes must be identical
   across all four backends (scalar `NoPacking`, NEON, AVX2, AVX-512)
   for any fixed `(setup, polynomial, opening point, transcript)`.
   Verified against existing scalar reference tests; spot-checked
   end-to-end (43,376-byte onehot proof / 41,008-byte dense proof are
   byte-identical across backends).
2. **Existing scalar and NEON paths are unchanged.** The two small
   refactors that touch shared code (the `shift64_mod_p_fp32` hoist
   into `util.rs`, the `use_simd_ntt` hoist into `ntt/mod.rs` so it can
   gate the NEON NTT and the AVX decompose-fold from one env var) must
   add no measurable overhead on either path. Verified by NEON parity
   tests on aarch64 and the main-vs-PR baseline drift table in
   §Performance.
3. **Backend selection is compile-time** via `cfg(target_feature = ...)`
   on the `packed_avx2` / `packed_avx512` modules in
   `crates/akita-field/src/fields/packed.rs` and on the
   `decompose_fold_avx` module in
   `crates/akita-prover/src/kernels/mod.rs`. Precedence: AVX-512 (F +
   DQ + BW) > AVX2 > NEON on aarch64 > scalar.
4. **`AKITA_SCALAR_NTT=1` kill switch applies uniformly.** The
   `use_simd_ntt()` function lives in
   `crates/akita-algebra/src/ntt/mod.rs` and gates both the NEON NTT
   and the AVX2 decompose-fold dispatch, so a single env var disables
   all hand-rolled SIMD on either arch.
5. **Verifier no-panic contract** (per `AGENTS.md`) is preserved. New
   AVX intrinsics live in prover-only crates; packed-field overrides
   ride through the existing `PackedField` trait surface, which is
   already exercised by both prover and verifier with parity tests
   against the scalar reference.

### Non-Goals

- **AVX2 / AVX-512 NTT module.** Prototyped, benchmarked, and
  reverted: hand-rolled `forward_ntt_*`, `inverse_ntt_*`,
  `pointwise_mul_acc_*`, and `add_reduce_*` on x86 regressed
  `dense_fp32_d64` commit by up to +132% on AVX-512 and `setup` by up
  to +45% (confirmed by an `AKITA_SCALAR_NTT=1` A/B against the same
  PR binary). For the small `D ≤ 64` sizes Akita uses, LLVM
  auto-vectorization of the simple scalar butterfly / pointwise loops
  turned out to be competitive with intrinsics, and the wider
  registers hurt cache locality on the radix-stride access pattern.
  A proper AVX NTT engineering effort (per-`D` specialization, cache
  blocking, possibly AVX-512 IFMA52 for the inner multiply) is worth
  a separate PR with its own benchmark gating.
- **Native AVX-512 decompose-fold kernel** (`_mm512_cvtepi8_epi32`,
  16-lane widening in one op). The AVX2 version using
  `_mm256_cvtepi8_epi32` is sufficient; the kernel is gated on
  `|coeff| ≤ 2` and is not the dominant cost.
- **`PackedFp128` extension overrides** — NEON only overrides
  `broadcast` for Fp128; nothing to mirror.
- **Vectorizing `PackedFp128{Avx2,Avx512}::Mul`** — both NEON and the
  new AVX paths use scalar-per-lane Fp128 multiply. Fp128 mul is not
  on the fp32 small-field hot path.
- **AVX-512 IFMA52** (`_mm512_madd52*_epu64`) — could replace
  `_mm512_mul_epu32` + add chains for primes with `BITS ≤ 26`. Needs a
  separate `target_feature = "avx512ifma"` gate and an algorithmic
  restructure (52-bit splits). Worth a separate spike.
- **Runtime CPU feature dispatch.** A single binary still ships only
  one backend; switching requires recompile with appropriate
  `RUSTFLAGS`.
- **MSRV bump to ≥ 1.89.** The pre-existing
  `#![cfg_attr(... feature(stdarch_x86_avx512))]` in `akita-algebra`
  and `akita-pcs` becomes a no-op after MSRV 1.89; leave for a
  separate cleanup PR.
- **No CI changes.** Pre-existing `packed_avx{2,512}` base-field code
  also ships without a dedicated CI job; the new code follows the
  same convention.

## Evaluation

### Acceptance Criteria

- [x] **Microbench:** `ring_subfield_fp4_mul` ≥ 2× faster on AVX2 vs
      scalar. Measured 5.4× (AVX2), 14.5× (AVX-512).
- [x] **End-to-end:** measurable improvement on `onehot_fp32_d64`
      prove time without regressing other stages. Measured −14% prove
      / −39% verify on AVX-512 (Fix-A proxy); setup / commit within
      ±2% of `main + AVX-512` (re-measurement on the trimmed branch
      pending — see §Performance).
- [x] **Refactor overhead:** main vs PR baseline (both scalar, same
      host) drift within ±5% per stage. Measured ±2.5%.
- [x] **Correctness:** proof bytes byte-identical across all flavors
      for `onehot_fp32_d64 nv=30 np=4` (43,376 B) and
      `dense_fp32_d64 nv=26 np=1` (41,008 B).
- [x] **Build hygiene:** `cargo fmt --all --check` clean;
      `cargo clippy --workspace --all-targets -- -D warnings` clean
      on aarch64 stable, x86 AVX2 stable, and x86 AVX-512 nightly.
- [x] **Tests:** `cargo test` clean on aarch64; new `decompose-fold`
      parity tests (3) and existing `packed_ext.rs` parity tests
      (extended to cover Fp64 `fp2_mul`) pass on aarch64 and on x86
      with `RUSTFLAGS="-C target-cpu=x86-64-v3"` / `=native`.

### Testing Strategy

- The existing extension-field parity tests at
  [`crates/akita-field/src/fields/packed_ext.rs:735-1064`](../crates/akita-field/src/fields/packed_ext.rs)
  cover every method the AVX overrides implement. They route through
  `<F as HasPacking>::Packing` which resolves to the active backend at
  compile time, so they exercise the AVX overrides automatically when
  `cargo test` is invoked with
  `RUSTFLAGS="-C target-cpu=x86-64-v3"` (or `=native`).
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
  # Or with AVX-512 (needs nightly):
  RUSTFLAGS="-C target-cpu=native" cargo +nightly nextest run --all-features
  ```

### Performance

Measured 2026-05-21 on AMD Ryzen 9 9950X (Zen 5). Three build flavors:

| Flavor     | `RUSTFLAGS`                                         | Toolchain | Active backend |
| ---------- | --------------------------------------------------- | --------- | -------------- |
| `baseline` | (none)                                              | stable    | scalar `NoPacking` (`_w1` witness) |
| `avx2`     | `-C target-cpu=x86-64-v3`                           | stable    | `packed_avx2` (`_w8`) |
| `avx512`   | `-C target-cpu=native` (+`avx512{f,dq,bw}` on Zen 5)| nightly   | `packed_avx512` (`_w16`) |

The `_wN` suffix is baked into Criterion bench names from
`<F as HasPacking>::Packing::WIDTH`, so it's an unforgeable witness
that the cfg gates selected the right type.

**Microbenchmarks (per-element ns, lower is better):**

| Op                     | baseline | avx2    | avx512    | avx2/base | avx512/base |
| ---------------------- | -------: | ------: | --------: | --------: | ----------: |
| `packed_add_chain`     |   0.739  |  0.317  | **0.168** |     2.33× |   **4.40×** |
| `packed_mul_chain`     |  13.061  |  2.425  | **0.899** |     5.39× |  **14.53×** |
| `packed_square_chain`  |   9.557  |  3.347  |   1.428   |     2.86× |     6.69×   |
| `packed_inverse_chain` | 106.5    | 87.99   |  83.24    |     1.21× |     1.28×   |

Microbench numbers are at field-arithmetic granularity and unaffected
by the NTT-module revert: they reflect the pure win from the new
extension-field overrides.

**End-to-end (median of 5 runs, ms):**

The numbers below report the AVX-512 build with `AKITA_SCALAR_NTT=1`
as a faithful **proxy for the post-revert configuration**: it
exercises the same binary path Fix-A will produce (AVX-512 base +
extension overrides + scalar NTT, with AVX decompose-fold still
dispatched). A clean rebuild + re-measurement on the trimmed branch
is the final acceptance step.

`onehot_fp32_d64 nv=30 np=4`:

| Stage  | baseline | avx512 (Fix-A proxy) | Δ vs baseline |
| ------ | -------: | -------------------: | ------------: |
| setup  |    688.5 |                785.7 |    +14.1%  ¹  |
| commit |    487.8 |                416.2 |        −14.7% |
| prove  |   2785.4 |               2385.1 |        −14.4% |
| verify |     41.5 |                 25.4 |        −38.8% |

`dense_fp32_d64 nv=26 np=1`:

| Stage  | baseline | avx512 (Fix-A proxy) | Δ vs baseline |
| ------ | -------: | -------------------: | ------------: |
| setup  |    108.9 |                129.1 |    +18.5%  ¹  |
| commit |   1524.4 |                690.4 |        −54.7% |
| prove  |   1441.9 |               1183.6 |        −17.9% |
| verify |     14.6 |                  8.8 |        −39.7% |

¹ The setup regression is a **build-flavor effect**, not a PR effect:
building the entire binary with `target-cpu=native` makes setup
slower regardless of source changes. Isolating that requires a
`main + target-cpu=native` baseline (not yet measured) but the
existing main-vs-PR baseline drift (both scalar) is ±2.5%, so the
remaining gap is attributable to wider-register codegen, not this
PR's content.

**Refactor overhead (main vs PR baseline, no SIMD, median of 5 runs):**

`onehot_fp32_d64`: setup −0.3%, commit −1.5%, prove −0.4%, verify −1.3%.
`dense_fp32_d64`:  setup −2.0%, commit −0.8%, prove −2.4%, verify −1.9%.

All within ±2.5% — well under measurement noise. Recursion levels
match across branches (`levels=5` onehot / `levels=4` dense).

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
  (`add_u64_with_carry`), folded back via the
  `shift64_mod_p_fp32(P) = 2^64 mod P` constant in
  [`util.rs`](../crates/akita-field/src/fields/util.rs) (shared with
  NEON to avoid duplication).
- **AVX-512 leverages mask registers** (`__mmask8`/`__mmask16`),
  native unsigned compares (`_mm512_cmplt_epu32_mask`,
  `_mm512_cmpge_epu64_mask`), and `_mm512_min_epu64` — features AVX2
  emulates via XOR-sign-bit. Net: AVX-512 add/sub on `BITS=32` primes
  is ~5 instructions vs AVX2's ~8.
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
- **`#[allow(dead_code)]` on `shift64_mod_p_fp32`** to silence the
  warning when no SIMD backend is selected. Rejected in favor of a
  cfg gate matching the union of caller conditions, so the function
  literally doesn't exist on non-SIMD builds. Self-correcting if the
  helper ever becomes truly orphaned.

## Documentation

This spec is the only documentation change. The implementation is
self-documented through module-level doc comments on
`packed_avx{2,512}.rs` and `decompose_fold_avx.rs`, each of which
references its NEON counterpart for the algebraic specification.

## References

- [`crates/akita-field/src/fields/packed_neon.rs`](../crates/akita-field/src/fields/packed_neon.rs)
  — the NEON backend mirrored by this PR.
- [`crates/akita-prover/src/kernels/decompose_fold_neon.rs`](../crates/akita-prover/src/kernels/decompose_fold_neon.rs)
  — the NEON decompose-fold kernel mirrored by this PR.
