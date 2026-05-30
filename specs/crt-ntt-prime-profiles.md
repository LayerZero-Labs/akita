# Spec: CRT/NTT Prime Profiles (Q16, Reduced Q32/Q64)

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | Quang Dao                                  |
| Created     | 2026-05-30                                 |
| Status      | proposed                                   |
| Branch      | `quang/crt-ntt-prime-profiles`             |
| PR          | #140                                       |

## Summary

Dense small-field commitments still pay conservative CRT/NTT prime counts: fp16
falls through to the six-prime Q32 i16 profile, fp32 uses all six i16 primes, and
fp64 uses the five-prime Q64 i32 profile.
That inflates setup NTT caches and matvec work even though merged PR #134 already
makes accumulation exact via capacity-bounded chunking and field-domain partial
sums.

This spec has two deliverables.
The primary deliverable cuts over internal CRT parameter tables and dispatch: add
a first-class three-prime Q16 i16 profile for fp16, reduce Q32 to four i16 primes,
reduce Q64 to three i32 primes, and leave Q128 at five i32 primes.
The i16 profiles (Q16, Q32) are extended to cover `D <= 256` (not just `D <= 64`)
so that fp16 (security-ladder `D = 256`) and fp32 (`D = 128`) keep 2-byte CRT
limbs instead of falling back to the 4-byte i32 Q64 profile, roughly halving their
shared NTT-cache footprint.
This requires lowering `MAX_CRT_RING_DEGREE` from `1024` to `256` and removing the
unused `D = 512` and `D = 1024` ring-degree presets, dispatch arms, and generated
families/tables; `D > 256` is no longer a supported ring degree.
The secondary deliverable is a backend-prepared layout experiment for the affected
CRT/NTT-cache paths (Invariant 11, acceptance criteria, Execution slice 7): the
implementation must run at least one alternative layout and keep it only if it
wins the measured thresholds, otherwise record why the current CPU reference is
retained.
The layout experiment is mandatory to run but bounded; if it grows beyond a
backend-private storage swap it should be split into its own follow-up PR rather
than expanding this one.
Proof bytes, transcripts, schedules, serialization, and verifier behavior stay
unchanged for both deliverables.

Range chunking and the `safe_crt_chunk_width` / `max_safe_crt_accumulation_width`
machinery are **not** re-specified here; they are implemented on `main` in
`specs/crt-ntt-accumulation-safety.md` (PR #134).

## Intent

### Goal

Replace conservative Q32/Q64 CRT prime sets with smaller field-oriented profiles,
introduce `ProtocolCrtNttParams::Q16` for 16-bit-or-smaller prime fields, extend
the i16 Q16/Q32 profiles to `D <= 256`, cap the supported ring degree at
`D <= 256`, and route every `NttSlotCache` consumer through the reduced profiles
with the existing #134 chunking paths so results match today's schoolbook
reference.
The semantic contract is the named compute operation output, not the current
row-major `Vec<CyclotomicCrtNtt>` physical cache layout.

Primary surfaces:

- `crates/akita-algebra/src/ntt/tables.rs`: prime tables, `Q16_*` constants,
  reduced `Q32_*` / `Q64_*` counts.
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`: `CrtNttParamSet` users keyed
  by new `K`; current prime-major `limbs[K][D]` CPU reference layout.
- `crates/akita-prover/src/kernels/crt_ntt.rs`: `ProtocolCrtNttParams`,
  `NttSlotCache`, `select_crt_ntt_params`.
- `crates/akita-prover/src/compute.rs`: `CpuPreparedSetup`, backend prepared
  setup boundary, dense/recursive/ring-switch compute operation calls.
- `crates/akita-prover/src/kernels/linear/ntt_matvec.rs`,
  `single_cyclic.rs`, `fused_quotients.rs`: explicit `NttSlotCache` match arms
  that need Q16 and any backend-private layout hooks.
- `crates/akita-prover/src/kernels/linear/*`: negacyclic i8, cyclic i8, fused
  split-eq, digit, block-parallel, CRT matvec drivers, and chunk width policy.
- `crates/akita-pcs/tests/algebra/ntt_crt.rs` and prover linear tests under
  `crates/akita-prover/src/kernels/linear/tests/`: prime validity, Garner
  constants, capacity tables, regression against references.
- `crates/akita-pcs/benches/ring_ntt.rs` and profile benchmarks: literal prime
  counts, cache footprint, and layout-performance measurements.

### Target profiles

The supported ring-degree set is `D in {32, 64, 128, 256}`.
All profiles cover the full `D <= 256` range, so dispatch is purely modulus-based;
there is no width fallback keyed on `D`.

| Profile | Current | Target | Dispatch |
| --- | ---: | ---: | --- |
| Q16 | (none; 16-bit fields use Q32) | 3 × i16 | `q <= u16::MAX`, `D <= 256` |
| Q32 | 6 × i16 | 2 × i32 (measured winner over reference 4 × i16) | `u16::MAX < q <= 2^32-99`, `D <= 256` |
| Q64 | 5 × i32 | 3 × i32 | `2^32-99 < q <= 2^64-59`, `D <= 256` |
| Q128 | 5 × i32 | unchanged | fp128 and listed offset moduli, `D <= 256` |

"Q16" names the dispatch class for field moduli that fit in 16 bits, gated by
`Q16_MODULUS = u16::MAX = 65535`.
It is not the fp16 field modulus itself (the current `Prime16Offset99` preset is
`q = 2^16 - 99 = 65437`); any prime field with `q <= 65535` and `D <= 256` selects
this profile.

**Q16 default primes** (all prime, `< 2^14`, `512 | (p - 1)` so they support the
full negacyclic NTT for `D <= 256`):

```text
15361, 13313, 12289
```

Product ≈ `2^41.19`.
These are the three largest i16 primes below `2^14` whose order admits `D = 256`;
restricting Q16 to `D <= 64` would only raise the product to `2^41.77`
(`16001, 15361, 15233`), a `0.58`-bit difference that does not justify forcing
fp16 onto the 4-byte i32 fallback at `D in {128, 256}`.
Montgomery constants are derived with `NttPrime::compute` like existing entries.
Define `Q16_MODULUS = u16::MAX`.

**Q32 default primes** (the two largest reduced-Q64 i32 primes, so the profile
reuses existing i32 NTT twiddles and Garner data and covers `D <= 256`):

```text
1073707009, 1073698817
```

Product ≈ `2^60.00`.
The original design default was the four-prime i16 reference profile
`[15361, 13313, 12289, 11777]` with product ≈ `2^54.72`.
The local release microbenchmark measured `2 × i32` faster on both the CRT round
trip and i8 multiply-lift loops, with the same per-coefficient CRT limb footprint
(8 bytes), so Q32 production is `2 × i32`. The `4 × i16` row remains only in the
capacity artifact as comparison evidence.

**Q64 default primes** (three largest i32 primes from the existing raw-prime
table, same ordering; each satisfies `512 | (p - 1)` for `D <= 256`):

```text
1073707009, 1073698817, 1073692673
```

**Q128**: keep `q128_primes()` values unchanged (five i32 primes).

The `D1024_RAW_PRIMES` constant should be renamed away from the `D1024` label
(for example `I32_RAW_PRIMES`) since `D = 1024` is removed.

Dispatch must test `q <= Q16_MODULUS` before the generic Q32 branch.
Fields with `u16::MAX < q <= Q32_MODULUS` use Q32; fields with
`Q32_MODULUS < q <= Q64_MODULUS` use Q64; the listed fp128 moduli use Q128.
Every branch requires `D <= 256` and returns `AkitaError::InvalidSetup` for any
larger ring degree.

### Invariants

1. For every supported schedule, field, ring degree, and valid prover input, each
   changed matvec or quotient kernel must equal the result of independent
   schoolbook ring arithmetic in the target field (same contract as PR #134).
2. Proof bytes, Fiat-Shamir order, transcript labels, public claims, schedule
   selection, setup seed semantics, and verifier code paths are unchanged.
3. Capacity uses the merged #134 conservative rule implemented in
   `crates/akita-prover/src/kernels/linear/capacity.rs`:

   ```text
   2 * width * D * floor(q/2) * rhs_abs_bound < P_crt
   ```

   where `P_crt` is the product of the **active** profile's primes.
   Balanced-digit RHS uses `rhs_abs_bound = 2^(log_basis - 1)`.
   The fused split-eq `z_pre` leg uses
   `rhs_abs_bound = max(z_pre_max_abs, centered_rows_abs_bound(z_pre))` as
   implemented in `fused_split_eq_quotients_with_params`.
4. Reduced profiles must be valid for every compute operation that uses that
   profile: negacyclic i8 matvecs, cyclic i8 matvecs, fused split-eq quotients,
   digit matvecs, block-parallel paths, dense commitment B matvecs, recursive
   witness rows, and ZK B/D rows when `zk` is enabled.
   There is no parallel “large prime” cache for quotient-only call sites.
5. If an i8/digit operation on a supported `(field, D, log_basis, width)` tuple
   cannot satisfy invariant (3) even at chunk width 1, prepared setup validation
   must return `AkitaError::InvalidSetup` before hot kernels run.
   The validating boundary is `CpuBackend::prepare_expanded` (the
   `ComputeBackendSetup::prepare_expanded` impl in
   `crates/akita-prover/src/compute.rs`), which already selects the profile via
   `build_ntt_slot` and runs before any matvec/quotient kernel.
   It must walk every i8/digit role implied by the prepared schedule and reject
   the setup there, so no hot kernel reaches the
   `single i8 CRT term must fit supported parameters` `.expect` in
   `kernels/linear/single_cyclic.rs` (or its `i8_matvec.rs` sibling).
   The fused centered-`z_pre` path may keep the exact field-native fallback from
   PR #134 when a single centered term cannot fit the CRT lift range; that
   fallback is not a legacy-prime fallback and must stay covered by tests.
6. Q128 prime count must not decrease.
   Four 30-bit i32 primes give `P_crt ≈ 2^120`, so the signed CRT range is only
   `≈ ±2^119`, but a single centered q128 coefficient already has magnitude up to
   `floor(q/2) ≈ 2^127`.
   One coefficient does not fit four primes (`2^127 > 2^119`), independent of
   chunking, so q128 D32 reconstruction needs all five primes.
7. Setup serialization stays canonical: backend-prepared caches rebuild
   deterministically from the same field matrix and setup seed.
   Backend-private physical layouts must not enter setup bytes, transcript
   bytes, proof bytes, or verifier inputs.
8. Verifier no-panic contract is unchanged (prover-only arithmetic).
9. Full cutover: no runtime shim for six-prime Q32, five-prime Q64, the legacy
   `D <= 64`-only i16 tables, or the removed `D = 512` / `D = 1024` ring degrees
   after merge.
10. `NttSlotCache` remains the CPU reference layout, not a public ABI.
    A backend may prepare a semantically equivalent prime-flat, column-tiled,
    or structure-of-arrays cache behind `ComputeBackendSetup::PreparedSetup`.
11. The implementation must run at least one backend-prepared layout experiment
    for the affected dense/root CRT paths.
    If a layout wins under the evaluation rules below, keep it as the
    backend-private implementation rather than discarding it as a benchmark-only
    spike.
12. Backend layout changes are allowed only behind named compute operations.
    Protocol code must continue to request operations such as dense commit rows,
    digit rows, cyclic rows, and ring-switch relation rows, not inspect
    backend-specific buffers.
13. Production CRT profiles are homogeneous in limb width (`i16` or `i32`) unless
    a later measured experiment proves a mixed-width profile is worth the added
    representation complexity.
    The current `CyclotomicCrtNtt<W, K, D>` and `CrtNttParamSet<W, K, D>` model
    assume one `W` per profile, and SIMD kernels assume homogeneous lanes.
14. `MAX_CRT_RING_DEGREE = 256`; `D in {32, 64, 128, 256}` are the only supported
    ring degrees.
    Every i16 NTT prime must satisfy `512 | (p - 1)` (a primitive `2 * 256`-th
    root for the negacyclic NTT at `D = 256`), and every i32 prime must satisfy
    the same `512 | (p - 1)`; the reused i32 primes already satisfy the stronger
    `2048 | (p - 1)`.
    `D = 512` and `D = 1024` are removed from `SUPPORTED_RING_DIMS`, the
    `dispatch_ring_dim` / `dispatch_ring_dim_result` arms, the fp16/fp32
    `D512Full` / `D512OneHot` public config presets, and the generated
    family/table lists; no production path may instantiate them.
    The `D512*` config preset names are removed rather than left as dead public
    aliases, so downstream attempts to use them fail at compile time instead of
    routing to an unsupported setup.

### Non-Goals

1. Re-litigating range chunking design (merged PR #134).
2. Changing proof format, Fiat-Shamir, verifier behavior, or the public
   commitment/proof API.
   Removing the unused fp16/fp32 `D512*` config presets is intentionally in
   scope as part of the full `D <= 256` cutover.
3. Q128 prime-count reduction.
4. Runtime selection of legacy prime counts.
5. fp16 two-i16 or single-i32 default profiles (benchmark-only spikes may be
   noted in implementation notes but are out of scope unless this spec is
   amended).
6. Q32 two-i32 default profile (optional spike only).
7. Planner / SIS table / `SisModulusFamily::Q16` floor generation (already on
   `main` via fp16 support; orthogonal to CRT dispatch).
8. Rewriting #134 chunking or "fixing" merged `single_cyclic` driver args for
   the Bugbot false positive (tests and optional cosmetic clarity only).
9. Requiring Metal, AVX, or any accelerator backend for the baseline cutover.
   The layout experiment may be CPU/Rayon or AArch64 NEON only. x86 CRT/NTT
   SIMD is tracked below as a high-priority deferred follow-up that may still
   be implemented in this PR after the main cutover is verified.
10. Changing canonical setup layout, proof layout, transcript binding, or
    verifier-visible semantics to accommodate a backend cache layout.
11. Choosing one new physical cache layout without measurements.
12. Introducing a mixed `i16`/`i32` production CRT profile in this PR.
    Mixed-width profiles are mathematically possible but fight the current
    homogeneous `PrimeWidth` abstraction, make SIMD/layout specialization more
    complex, and should be a follow-up only if homogeneous profiles cannot meet
    the measured goals.
13. Supporting `D > 256`.
    The `D = 512` / `D = 1024` ring degrees are removed, not merely left on the
    i32 fallback; no production field uses them and keeping them would force i16
    primes through a `1024 | (p - 1)` / `2048 | (p - 1)` order with too few
    candidates below `2^14` to build a four-prime Q32 set.
14. Maximizing the Q16/Q32 product by capping i16 at `D <= 64`.
    The larger `D <= 64` triple (`16001, 15361, 15233`, `2^41.77`) is rejected
    because the `0.58`-bit gain does not justify dropping fp16/fp32 i16 coverage
    at `D in {128, 256}`.

## PR #134 Cursor Bugbot audit (merged accumulation safety)

Range chunking landed in [#134](https://github.com/LayerZero-Labs/akita/pull/134)
(`specs/crt-ntt-accumulation-safety.md`).
Cursor Bugbot left **nine** inline findings; the final summary on `87e3474`
flagged one remaining **Medium** item on `single_cyclic.rs`.
A disposition comment for future readers is on
[#134#issuecomment-4582547527](https://github.com/LayerZero-Labs/akita/pull/134#issuecomment-4582547527).

This section records whether each finding is still valid on current `main`
(`0a360113`) and what the **prime-profiles** follow-up must do (if anything).

| Bugbot finding | Severity | Valid on `main`? | Required for prime profiles? |
| --- | --- | --- | --- |
| `single_cyclic` "wrong" `safe_width` / `tile_width` args | Medium | **No** (false positive; see below) | **Regression tests only** |
| `fused_split_eq_quotients_prover_bounds` lacks `w_hat` check | Medium | **No** (`with_params` returns `InvalidInput`) | None (optional `debug_assert!`) |
| CRT capacity uses `q` not `floor(q/2)` | Medium | **No** (fixed before merge) | None |
| `.expect` in raw-i8 strided `Result` path | Medium | **No** (`ok_or_else`) | None |
| Hardcoded digit bound 32 | Low | **No** | None |
| Redundant `i32::MIN` branch | Low | **No** | None |
| Duplicate comment in `digits.rs` | Low | **No** | None |
| Duplicate `validate_i8_log_basis` | Low | Yes (hygiene) | Out of scope |
| Single-row chunked path lacks Rayon | Low | Yes (perf) | Out of scope |

### False positive: `single_cyclic` one-shot vs chunked gate

**What Bugbot claimed.** In `mat_vec_mul_single_i8_with_params` and
`mat_vec_mul_single_i8_cyclic_with_params`, both the 3rd argument (`safe_width`)
and 5th argument (`chunk_width`) to `drive_single_chunked_matvec` are set to
`safe_crt_chunk_width(params, vec_len, digit_bound)`.
Bugbot concluded the gate `inner_width <= safe_width` is always true, so the
chunked fallback never runs and CRT overflow protection is defeated.

**Why that is wrong.** `safe_crt_chunk_width` returns `min(max_safe, vec_len)`,
where `max_safe` is `max_safe_crt_accumulation_width` (columns safe in one CRT
accumulator before Garner reconstruction).
With `inner_width = vec_len`, the gate is:

```text
vec_len <= min(max_safe, vec_len)   ⟺   vec_len <= max_safe
```

| Case | `min(max_safe, vec_len)` | Gate | `drive_single_chunked_matvec` path |
| --- | ---: | --- | --- |
| `vec_len <= max_safe` | `vec_len` | true | One-shot: accumulate all columns, one reconstruct (safe) |
| `vec_len > max_safe` | `max_safe` | **false** | Chunked: reconstruct per chunk of `max_safe` columns, sum in field |

Example: `max_safe = 1023`, `vec_len = 2050` → gate `2050 <= 1023` is false →
chunked path runs (three chunks), matching the intent of #134.

Passing the clamped value as the 3rd argument is **not** unsound: it cannot
approve a one-shot wider than `max_safe`.
An optional cosmetic change is to pass raw `max_safe` as the 3rd arg (matching
`i8_matvec.rs`) while keeping `chunk_width = min(max_safe, vec_len)`; behavior
is unchanged.

**Implication for this spec.** Prime-profile work does **not** depend on fixing
`single_cyclic.rs` call sites.
Smaller `P_crt` lowers `max_safe` and increases chunking frequency; that is
expected and must stay correct under the same #134 driver.
This spec therefore requires **regression tests**, not a driver rewrite:

- Mirror `mat_vec_mul_ntt_i8_dense_single_row_chunks_q128` in the focused
  prover linear test modules for `mat_vec_mul_ntt_single_i8` and
  `mat_vec_mul_ntt_single_i8_cyclic` at a width with `vec_len > max_safe`.
- Run on Q128 before prime cutover; repeat on reduced Q16/Q32/Q64 after.

Do not change `drive_single_chunked_matvec` arguments unless a new test fails.

Closed spec PR [#108](https://github.com/LayerZero-Labs/akita/pull/108) is the
design predecessor for prime tables only; its Cursor Bugbot run reported no
issues because that PR was documentation-only.
The blocking items there were from the model-agnostic spec review (cyclic/fused
coverage, `z_pre` bounds); those are absorbed into the Invariants and Acceptance
Criteria sections above, with #134 providing the chunking implementation.

## Evaluation

### Acceptance Criteria

- [ ] `tables.rs` defines `Q16_NUM_PRIMES = 3`, `Q16_PRIMES = [15361, 13313,
      12289]`, `q16_garner()`, and unit tests that each Q16 prime is prime,
      `< 2^14`, and satisfies `512 | (p - 1)`.
- [ ] `Q32_NUM_PRIMES = 2` with the measured-winner table
      `[1073707009, 1073698817]`; tests mirror Q64 (`512 | (p - 1)` and i32
      Montgomery constants).
- [ ] `Q64_NUM_PRIMES = 3` with the three-prime subset above; tests verify
      `512 | (p - 1)` and Garner data for `D = 32, 64, 256`.
- [ ] `tables.rs` sets `MAX_CRT_RING_DEGREE = 256` and the i32 raw-prime constant
      is renamed away from `D1024` (e.g. `I32_RAW_PRIMES`).
- [ ] `tables.rs` defines `Q16_MODULUS = u16::MAX`.
- [ ] `select_crt_ntt_params` dispatches `q <= Q16_MODULUS` to `Q16` before the
      generic Q32 branch; `Q16_MODULUS < q <= Q32_MODULUS` to `Q32`;
      `Q32_MODULUS < q <= Q64_MODULUS` to `Q64`; the listed fp128 moduli to Q128.
      Every branch requires `D <= 256` (one of `{32, 64, 128, 256}`) and returns
      `AkitaError::InvalidSetup` for `D > 256`.
      There is no `D`-keyed width fallback (no "16-bit field with `D > 64` uses
      Q64").
- [ ] `D = 512` / `D = 1024` are removed from `SUPPORTED_RING_DIMS`, the
      `dispatch_ring_dim` / `dispatch_ring_dim_result` macro arms, the fp16/fp32
      `D512Full` / `D512OneHot` public config presets, `generated_families`, and
      any generated table/drift-guard list, with `cargo test -q` and the drift
      guard green.
      The `D512*` preset names are not kept as deprecated aliases.
- [ ] `ProtocolCrtNttParams` and `NttSlotCache` include a `Q16` variant; all
      match arms updated in `crt_ntt.rs`, `ntt_matvec.rs`, `single_cyclic.rs`,
      `fused_quotients.rs`, test helpers, and benches (full cutover, no
      `panic!` on fp16).
- [ ] Q32 implementation compares the reference `4 × i16` profile against the
      production `2 × i32` profile `[1073707009, 1073698817]`
      (the two largest i32 raw primes, product ≈ `2^60.00`).
      The comparison must include correctness, generated schedule capacity, setup
      cache bytes, selected profile metadata (`K`, limb width, prime list,
      `log2(P_crt)`), safe-width/chunk-count summaries, and the required profile
      timings from the performance protocol below.
      The local release microbenchmark selected `2 × i32`; keep it as the only
      production Q32 path unless later required-profile medians contradict the
      same-machine result.
- [ ] `max_safe_crt_accumulation_width` unit tests for Q16, reduced Q32, and
      reduced Q64 cover balanced-i8 and centered-i32 (`z_pre_max_abs`) RHS
      bounds at concrete `D` and `log_basis` values.
      Walk every committed generated schedule table entry for `fp16_d32_full`,
      `fp16_d32_onehot`, `fp16_d64_full`, `fp16_d64_onehot`, `fp32_d32`,
      `fp32_d32_onehot`, `fp32_d64`, `fp32_d64_onehot`, `fp64_d32`,
      `fp64_d32_onehot`, `fp64_d64`, and `fp64_d64_onehot` (these are the only
      committed tables; `D in {128, 256}` tables do not exist yet).
      Additionally add direct capacity unit tests for the i16 Q16/Q32 profiles at
      `D = 128` and `D = 256` using `Cfg::decomposition()` `log_basis` values, so
      the extended `D <= 256` coverage is guarded before any `D > 64` schedule
      table is generated.
      Assert every i8/digit role has a nonzero single-term safe width under the
      selected profile.
- [ ] Capacity tests pin at least one golden
      `max_safe_crt_accumulation_width` value for Q16, reduced Q32, and reduced
      Q64 at named `(field, D, log_basis, rhs_abs_bound)` tuples.
- [ ] Add a capacity-profile artifact, checked into the implementation PR body
      or a committed generated markdown file, that lists for every candidate
      profile:
      `K`, limb width, prime list, `log2(P_crt)`, supported `D` values,
      representative `rhs_abs_bound` values, and the resulting
      `max_safe_crt_accumulation_width`.
      It must include the exact tuples used by generated schedule capacity
      tests and the Q32 reference `4 × i16` vs production `2 × i32` comparison.
- [ ] Forced sub-full chunk tests (width above `max_safe_crt_accumulation_width`)
      for:
      - negacyclic `mat_vec_mul_ntt_single_i8`,
      - cyclic `mat_vec_mul_ntt_single_i8_cyclic`,
      - `fused_split_eq_quotients` including the `z_pre` leg,
      each compared against a scalar or wide-reference path (Q128 profile on
      `main`; repeat on reduced Q16/Q32/Q64 after prime cutover).
- [ ] Single-row forced-chunk tests (`vec_len > max_safe_crt_accumulation_width`)
      for `mat_vec_mul_ntt_single_i8` and `mat_vec_mul_ntt_single_i8_cyclic`,
      modeled on `mat_vec_mul_ntt_i8_dense_single_row_chunks_q128`, with
      schoolbook reference equality (Q128 first; reduced profiles after cutover).
- [ ] No change to `single_cyclic.rs` `drive_single_chunked_matvec` arguments
      unless a new test fails (Bugbot Medium on #134 is a false positive).
- [ ] Prepared setup validation rejects any generated schedule/profile tuple
      whose i8/digit CRT path cannot fit one term at chunk width 1.
      This validation must happen before the corresponding hot kernel would
      reach an internal `.expect`.
- [ ] A backend-prepared layout experiment compares the current CPU reference
      layout (`Vec<CyclotomicCrtNtt>` in row-major `NttSlotCache`) against at
      least one alternative layout for the affected root dense paths:
      prime-flat, column-tiled, or structure-of-arrays.
      Keep the alternative if it improves setup, commit, or prove wall-clock by
      at least 5% on one required profile without regressing another required
      profile by more than 5%, or if it reduces shared NTT cache bytes by at
      least 20% without a measured wall-clock regression above 5%.
      Apply the same benchmark protocol and PR-description result table used for
      the prime-profile comparison.
- [ ] If the layout experiment does not win, the implementation PR records the
      attempted layout, benchmark numbers, and reason for keeping the current
      CPU reference layout.
- [ ] Existing `akita-pcs` algebra NTT/CRT tests and `cargo test -q -p
      akita-prover kernels::linear` pass.
- [ ] `cargo test -q` and `cargo clippy --all -- -D warnings` pass.

### Testing Strategy

- Extend `crates/akita-algebra/src/ntt/tables.rs` tests for new prime-derived
  Montgomery/Garner constants.
- Extend `capacity.rs` tests with `Q16_PRIMES`, production `Q32_PRIMES`, and
  reduced `Q64_PRIMES`; assert expected safe widths for fp16/fp32/fp64 dense
  and onehot `log_basis` pairs used in generated root schedules (not merely
  `CommitmentConfig::decomposition()` defaults).
- Add a small deterministic capacity-table generator or test helper that computes
  the precise bound:

  ```text
  max_width = max w such that
    2 * w * D * floor(q/2) * rhs_abs_bound < P_crt
  ```

  for every candidate profile and every tested operation role.
  Do not hand-maintain copied capacity numbers without a recomputation path.
- Add a schedule-capacity test next to the generated schedule materialization
  tests in `crates/akita-config/src/proof_optimized/tests.rs`, or a helper with
  equivalent table coverage, so future generated-table changes cannot silently
  pick a tuple whose selected CRT profile cannot fit one i8/digit term.
- Add or extend `akita-pcs/tests/algebra/ntt_crt.rs` for round-trip NTT on Q16
  and reduced Q32/Q64 at `D in {32, 64, 128, 256}` where applicable (the i16
  Q16/Q32 round trips must pass at `D = 256`, exercising the `512 | (p - 1)`
  order).
- Reuse PR #134 adversarial patterns (large centered setup coeffs, wide matrices,
  forced chunk widths) with the **new** prime products.
- Keep prover linear-kernel tests split by topic under
  `crates/akita-prover/src/kernels/linear/tests/` rather than growing a single
  large `tests.rs` file. Suggested modules are API validation, fused quotient
  rows, CRT dense matvec, i8/digit matvec, chunking, and reduced-profile
  regressions.
- All existing E2E / `single_poly_e2e` tests must remain green (prove + verify).

### Performance

- Direction: lower setup NTT cache size and fewer CRT limbs per coefficient for
  fp16/fp32/fp64 dense paths.
- The implementation PR description is the central performance record.
  It must include a single before/after table for the required modes and metrics.
  If a generated markdown artifact is also committed or uploaded, link it from
  the PR description rather than scattering numbers across comments.
- Measure a `main`/merge-base baseline and the implementation head on the same
  machine, with the same release profile, feature flags, `RAYON_NUM_THREADS`,
  benchmark script, and profile arguments.
  Record the baseline/head commit SHAs, hardware/OS, Rust version, feature flags,
  thread count, and relevant environment variables.
- Record before/after on `crates/akita-pcs/examples/profile/` for at least the
  D32 dense and one-hot small-field matrix:
  - `dense_fp16_d32`,
  - `onehot_fp16_d32`,
  - `dense_fp32_d32`,
  - `onehot_fp32_d32`,
  - `dense_fp64_d32`,
  - `onehot_fp64_d32`.
- Record the commands used, e.g.
  `AKITA_MODE=dense_fp16_d32 cargo run --release --example profile`, with the
  corresponding modes above.
- Run each required profile at least five times and report median wall-clock
  timings plus a simple spread measure (`min`/`max` or median absolute
  deviation).
  If local conditions make five stable runs impractical, the PR description must
  say why and still report at least three runs.
- For every required mode, record setup, commit, prove, and verify wall-clock;
  setup vector bytes; shared setup NTT cache bytes; maximum RSS; proof bytes;
  selected CRT profile; `K`; limb width; and the relevant
  `max_safe_crt_accumulation_width` / observed chunk-count summary for changed
  kernels.
- No fixed “must win” threshold: post numbers in the implementation PR.
  Regressions above ~5% wall-clock on any required mode require an explicit note
  in the PR body with a hypothesis (e.g., more chunks on fp32 outer-B).
- Proof bytes must be exactly unchanged for matching benchmark shapes.
  Verifier wall-clock is expected to be unchanged within benchmark noise; any
  >5% verifier movement needs an explicit note because this is intended to be
  prover-only.
- The layout experiment additionally records shared NTT cache bytes and the
  chosen physical prepared-cache layout.

## Design

### Architecture

```text
select_crt_ntt_params(F, D)   // requires D in {32,64,128,256}, else InvalidSetup
        │
        ├─ q<=u16::MAX ──────────────► Q16 (3× i16, 512|(p-1))
        ├─ q<=Q32 ───────────────────► Q32 (4× i16 default, 512|(p-1); test 2×i32)
        ├─ q<=Q64 ───────────────────► Q64 (3× i32)
        └─ fp128 family ─────────────► Q128 (5× i32)
                                    │
                   backend-prepared CRT/NTT cache
                                    │
       named compute operations + #134 chunking / field partial sums
```

Const-generic `K` changes propagate through `CyclotomicCrtNtt<W, K, D>`,
`DigitMontLut<W, K>`, and prover `match` arms.
Prefer a single source of truth in `tables.rs` over duplicating prime arrays.

The current CPU reference cache is:

```text
NttSlotCache::{Q16,Q32,Q64,Q128}
  neg: Vec<CyclotomicCrtNtt<W,K,D>>   // row-major cells
  cyc: Vec<CyclotomicCrtNtt<W,K,D>>   // row-major cells

CyclotomicCrtNtt<W,K,D>
  limbs[k][d]                         // prime-major, D-contiguous
```

That layout is good for today's AArch64 NEON kernels, which vectorize across the
`D` dimension inside one CRT prime.
It is not necessarily optimal for backend-specific execution because fixed-column
multi-row access strides by `num_cols * K * D * sizeof(W)`, cyclic and
negacyclic domains double cache memory, and GPU-style backends generally prefer
flat structure-of-arrays buffers.

This PR may replace the CPU-prepared physical cache with a backend-private layout
if measurements justify it.
Candidate layouts include:

1. **Prime-flat row-major:** one flat buffer per domain and CRT prime, indexed by
   `(row, column, d)`.
   This keeps per-prime NTT friendliness while making upload and SIMD prefetching
   simpler.
2. **Column-tiled prime-flat:** one flat buffer per `(domain, prime, column
   tile)`, with rows for the tile contiguous.
   This targets fused relation rows and small-`n_a` dense paths that repeatedly
   touch fixed columns across roles.
3. **Structure-of-arrays prepared cache:** backend-specific buffers arranged for
   vector lanes or device coalescing, hidden behind `ComputeBackendSetup`.

Do not make `NttSlotCache` the long-term backend ABI.
The durable API is the compute operation surface in `crates/akita-prover/src/compute.rs`:
`dense_commit_rows`, `digit_rows`, `cyclic_digit_rows`,
`recursive_witness_commit_rows`, `ring_switch_relation_rows`, and
`ring_switch_quotient_rows`.
The CPU reference may keep `NttSlotCache` if no measured alternative wins.

Mixed-width CRT profiles such as `2 × i16 + 1 × i32` are intentionally not part
of this PR's production target.
They can reduce byte footprint for a desired `P_crt`, but they cut against the
current homogeneous `PrimeWidth` model and make SIMD lanes, twiddle tables,
Garner reconstruction, and backend-prepared buffers heterogeneous.
The first implementation compared homogeneous candidates (`4 × i16` reference
vs `2 × i32` production for Q32) before considering mixed-width profiles.

### Deferred Follow-Ups

1. **x86 CRT/NTT SIMD for `i16` and `i32` primes.** High priority.
   Today x86 SIMD support exists for packed field arithmetic (`Fp32`, `Fp64`,
   `Fp128`) and for the prover decompose-fold AVX2 kernel, but the small-prime
   CRT/NTT path only has AArch64 NEON kernels.
   Add AVX2 and/or AVX-512 equivalents for negacyclic and cyclic
   `forward_ntt` / `inverse_ntt` plus fused pointwise multiply-accumulate for
   both `i16` and `i32` profiles.
   The current prime-major, `D`-contiguous `limbs[K][D]` layout is a plausible
   SIMD starting point on x86; any alternative layout should stay
   backend-private and be justified by the same layout-performance rules above.
   If this remains deferred at merge time, record the absence and expected
   performance impact in the implementation notes.

### Alternatives Considered

1. **Keep six/five primes.** Safe but leaves performance on the table; rejected.
2. **First four legacy Q32 primes.** Product too small for fp32 dense nv26
   one-shot even with chunking; rejected.
3. **Two i16 primes for Q16.** Too many reconstruction rounds on real widths;
   rejected.
4. **Two i32 primes for Q32.** Selected.
   The measured release microbenchmark beat the four-prime i16 reference while
   keeping the same 8-byte per-coefficient CRT limb footprint.
5. **Mixed i16/i32 profiles.** Deferred.
   They may be mathematically attractive, but they require a heterogeneous CRT
   representation instead of the current `CrtNttParamSet<W, K, D>` shape.
6. **Separate large-profile cache for quotients only.** Violates global cutover
   and duplicates cache memory; rejected (spec review blocking question).
7. **Combine with range-chunking spec PR #108.** Split for review/merge order:
   #134 landed chunking; this PR lands primes only.
8. **Freeze the current row-major `NttSlotCache` layout as the implementation
   contract.** Rejected.
   It is a good CPU reference, but it would make backend-specific SIMD and
   future Metal work inherit an avoidable AoS-shaped cache.
9. **Land a benchmark-only alternative layout and discard it.** Rejected.
   If the experiment wins under the acceptance criteria, the implementation
   should keep the backend-private layout and use CPU parity tests as the guard.
10. **Keep i16 capped at `D <= 64` and route fp16/fp32 `D > 64` to i32 Q64.**
    Rejected.
    fp16 (security-ladder `D = 256`) and fp32 (`D = 128`) would pay 4-byte i32
    limbs at their production ring degrees; extending the i16 order to
    `512 | (p - 1)` keeps them on 2-byte limbs for `< 0.6` bits of product.
11. **Keep `MAX_CRT_RING_DEGREE = 1024` and the `D = 512` / `D = 1024` presets.**
    Rejected.
    No production field instantiates `D > 256`, the i16 pool below `2^14` cannot
    supply a four-prime Q32 set at `D = 512` (`1024 | (p - 1)` leaves only three
    such primes), and carrying the unused arms blocks the i16 extension.

## Documentation

- This spec file.
- Update module docs in `ntt/tables.rs` and `crt_ntt.rs` dispatch comment to
  describe Q16 and reduced counts.
- If a new backend-prepared layout is kept, document it in
  `crates/akita-prover/src/compute.rs` or the local prepared-cache module:
  physical ordering, domain coverage (`neg`, `cyc`, or both), alignment
  expectations, and why it remains backend-private.
- No paper or verifier doc changes required.

## Execution

Suggested implementation slices:

0. **Slice 0 (recommended before prime tables):** add
   `mat_vec_mul_ntt_single_i8` / `_cyclic` forced-chunk tests on Q128; document
   that #134 Bugbot "wrong arg order" is closed as false positive (see audit
   section and #134 comment).
1. Lower `MAX_CRT_RING_DEGREE` to `256`; remove `D = 512` / `D = 1024` from
   `SUPPORTED_RING_DIMS`, the dispatch macros, the fp16/fp32 `D512*` presets, and
   `generated_families`; confirm the drift guard and `cargo test -q` stay green.
2. Add Q16 table + tests (`512 | (p - 1)`); add reduced Q32/Q64 tables + tests;
   rename the i32 raw-prime constant off the `D1024` label.
3. Generate and review the capacity-profile artifact for Q16, Q32 reference
   `4 × i16`, Q32 production `2 × i32`, Q64, and Q128 at
   `D in {32, 64, 128, 256}`.
4. Extend `ProtocolCrtNttParams` / `NttSlotCache` / `select_crt_ntt_params`
   (D-aware, `D <= 256`, no width fallback on `D`).
5. Fix const-generic `K` throughout prover linear + setup NTT cache build,
   including `ntt_matvec.rs`, `single_cyclic.rs`, `fused_quotients.rs`,
   `compute.rs`, algebra tests, and benches.
6. Add capacity validation in `CpuBackend::prepare_expanded`, direct `D = 128`
   / `D = 256` capacity unit tests, and forced-chunk tests for cyclic/fused/
   `z_pre` on new `K`; keep the linear-kernel tests split into focused files
   under `crates/akita-prover/src/kernels/linear/tests/`.
7. Run the Q32 reference `4 × i16` vs production-candidate `2 × i32`
   experiment. Keep the winner if it satisfies capacity and performance
   criteria.
8. Run the backend-prepared layout experiment on the current row-major CPU
   reference versus at least one prime-flat, column-tiled, or SoA layout.
   Keep the new layout if it wins under the acceptance criteria.
9. Profile dense fp16/fp32/fp64; fix any unexpected chunk-count regressions.

Risks to resolve first:

- Confirm removing the `D = 512` presets does not break `generated_families`, the
  drift-guard test, or any committed generated table (none should reference
  `D > 64` today, but verify).
- Generated schedule tables for `D in {128, 256}` are **out of scope**; this PR
  only makes the profiles/dispatch ready for them. Today's table-only `Cfg`
  production stays at `D <= 64`, so the i16 cache win at `D in {128, 256}` is
  realized only when those tables land in a follow-up.
- Confirm fp16 outer-B nv32 widths still chunk correctly under Q16 (may be
  two chunks; acceptable if correct).
- Update every `Q32_NUM_PRIMES` / `Q64_NUM_PRIMES` literal in tests and benches.
- Keep the layout experiment scoped to backend-prepared storage and named compute
  operations.
  Do not leak physical cache ordering into protocol, setup serialization, or
  verifier-facing types.

## References

- `specs/crt-ntt-accumulation-safety.md` (implemented, PR #134): chunking and
  capacity contract on `main`.
- [#134#issuecomment-4582547527](https://github.com/LayerZero-Labs/akita/pull/134#issuecomment-4582547527):
  disposition of the final Bugbot Medium (`single_cyclic` false positive).
- Closed `specs/crt-ntt-range-chunking.md` on branch
  `quang/crt-ntt-range-chunking-spec` (PR #108): combined design predecessor;
  prime tables copied from its profile section.
- Closed PR [#133](https://github.com/LayerZero-Labs/akita/pull/133): superseded
  chunking implementation path.
- `specs/fp16-small-field-support.md`: SIS `Q16` family (orthogonal).
- `specs/akita-compute-backend-metal.md`: compute backend boundary and
  backend-prepared setup ownership.
- `specs/SPEC_REVIEW.md`: review workflow for this spec PR.
