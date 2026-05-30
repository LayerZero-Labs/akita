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

This spec cuts over internal CRT parameter tables and dispatch only: add a
first-class three-prime Q16 i16 profile for fp16, reduce Q32 to four i16 primes,
reduce Q64 to three i32 primes, and leave Q128 at five i32 primes.
Proof bytes, transcripts, schedules, serialization, and verifier behavior stay
unchanged.

Range chunking and the `safe_crt_chunk_width` / `max_safe_crt_accumulation_width`
machinery are **not** re-specified here; they are implemented on `main` in
`specs/crt-ntt-accumulation-safety.md` (PR #134).

## Intent

### Goal

Replace conservative Q32/Q64 CRT prime sets with smaller field-oriented profiles,
introduce `ProtocolCrtNttParams::Q16` for fp16 `D <= 64`, and route every
`NttSlotCache` consumer through the reduced profiles with the existing #134
chunking paths so results match today's schoolbook reference.

Primary surfaces:

- `crates/akita-algebra/src/ntt/tables.rs`: prime tables, `Q16_*` constants,
  reduced `Q32_*` / `Q64_*` counts.
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`: `CrtNttParamSet` users keyed
  by new `K`.
- `crates/akita-prover/src/kernels/crt_ntt.rs`: `ProtocolCrtNttParams`,
  `NttSlotCache`, `select_crt_ntt_params`.
- `crates/akita-prover/src/kernels/linear/*`: negacyclic i8, cyclic i8, fused
  split-eq, digit, block-parallel, and CRT matvec drivers (chunk width only;
  no new chunking algorithm).
- `crates/akita-pcs/tests/algebra/ntt_crt.rs` and prover linear tests: prime
  validity, Garner constants, capacity tables, regression against references.

### Target profiles

| Profile | Current | Target | Dispatch |
| --- | ---: | ---: | --- |
| Q16 | (none; fp16 uses Q32) | 3 × i16 | fp16, `D <= 64` |
| Q32 | 6 × i16 | 4 × i16 | fp32, `D <= 64`; other `q <= 2^32-99` small fields at `D <= 64` |
| Q64 | 5 × i32 | 3 × i32 | fp64; small fields with `D > 64` and `q <= 2^64-59` |
| Q128 | 5 × i32 | unchanged | fp128 and listed offset moduli, `D <= 1024` |

**Q16 default primes** (all prime, `< 2^14`, `128 | (p - 1)`):

```text
16001, 15361, 15233
```

Product ≈ `2^41.77`.
Montgomery constants are derived with `NttPrime::compute` like existing Q32
entries.

**Q32 default primes** (do not take the first four legacy primes; their product
is too small for current fp32 dense one-shot widths):

```text
16001, 15361, 15233, 14593
```

Product ≈ `2^55.60`.

**Q64 default primes** (three largest `D1024` i32 primes, same ordering as
today's `D1024_RAW_PRIMES`):

```text
1073707009, 1073698817, 1073692673
```

**Q128**: keep `D1024_RAW_PRIMES` / `q128_primes()` unchanged.

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
4. Reduced profiles must be valid for **every** `NttSlotCache` user on that
   profile: negacyclic i8 matvecs, cyclic i8 matvecs, fused split-eq quotients,
   digit matvecs, block-parallel paths, and dense commitment B matvecs.
   There is no parallel “large prime” cache for quotient-only call sites.
5. If a profile cannot satisfy invariant (3) for a supported `(field, D,
   log_basis, width)` tuple even at chunk width 1, `select_crt_ntt_params` or setup
   expansion must return `AkitaError::InvalidSetup` (no silent fallback to legacy
   prime counts).
6. Q128 prime count must not decrease (four 30-bit i32 primes are insufficient
   for one-column q128 D32 reconstruction).
7. Setup serialization stays canonical: caches rebuild deterministically from the
   same field matrix; only internal CRT tables and cache element width change.
8. Verifier no-panic contract is unchanged (prover-only arithmetic).
9. Full cutover: no runtime shim for six-prime Q32 or five-prime Q64 after merge.

### Non-Goals

1. Re-litigating range chunking design (merged PR #134).
2. Changing proof format, Fiat-Shamir, or public APIs.
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

- Mirror `mat_vec_mul_ntt_i8_dense_single_row_chunks_q128` in
  `kernels/linear/tests.rs` for `mat_vec_mul_ntt_single_i8` and
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

- [ ] `tables.rs` defines `Q16_NUM_PRIMES = 3`, `Q16_PRIMES`, `q16_garner()`, and
      unit tests that each Q16 prime is prime, `< 2^14`, and satisfies
      `128 | (p - 1)`.
- [ ] `Q32_NUM_PRIMES = 4` with the four-prime table above; tests mirror Q16.
- [ ] `Q64_NUM_PRIMES = 3` with the three-prime subset above; tests verify
      `2048 | (p - 1)` and Garner data for `D = 32, 64, 1024`.
- [ ] `select_crt_ntt_params` returns `Q16` for fp16 with `D <= 64`, `Q32` for
      fp32 with `D <= 64`, `Q64` for fp64 and small-field `D > 64`, `Q128`
      unchanged.
- [ ] `ProtocolCrtNttParams` and `NttSlotCache` include a `Q16` variant; all
      match arms updated (full cutover, no `panic!` on fp16).
- [ ] `max_safe_crt_accumulation_width` unit tests for Q16, reduced Q32, and
      reduced Q64 cover balanced-i8 and centered-i32 (`z_pre_max_abs`) RHS
      bounds at representative `D` and `log_basis` values from generated dense
      schedules.
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
- [ ] Existing `akita-pcs` algebra NTT/CRT tests and `cargo test -q -p
      akita-prover kernels::linear` pass.
- [ ] `cargo test -q` and `cargo clippy --all -- -D warnings` pass.

### Testing Strategy

- Extend `crates/akita-algebra/src/ntt/tables.rs` tests for new prime-derived
  Montgomery/Garner constants.
- Extend `capacity.rs` tests with `Q16_PRIMES`, reduced `Q32_PRIMES`, and
  reduced `q64_primes()`; assert expected safe widths for fp16/fp32/fp64 dense
  `log_basis` pairs used in generated root schedules (not merely
  `CommitmentConfig::decomposition()` defaults).
- Add or extend `akita-pcs/tests/algebra/ntt_crt.rs` for round-trip NTT on Q16
  and reduced Q32/Q64 at `D in {32, 64, 1024}` where applicable.
- Reuse PR #134 adversarial patterns (large centered setup coeffs, wide matrices,
  forced chunk widths) with the **new** prime products.
- All existing E2E / `single_poly_e2e` tests must remain green (prove + verify).

### Performance

- Direction: lower setup NTT cache size and fewer CRT limbs per coefficient for
  fp16/fp32/fp64 dense paths.
- Record before/after on `crates/akita-pcs/examples/profile/` for at least:
  - `dense_fp16_d32` (or nearest fp16 dense mode),
  - `dense_fp32_d32`,
  - `dense_fp64_d32`.
- No fixed “must win” threshold: post numbers in the implementation PR.
  Regressions above ~5% wall-clock on any of the three modes require an explicit
  note in the PR body with hypothesis (e.g., more chunks on fp32 outer-B).
- Proof size and verifier time must be unchanged (prover-only).

## Design

### Architecture

```text
select_crt_ntt_params(F, D)
        │
        ├─ fp16, D<=64 ──► Q16 (3× i16) ──► NttSlotCache::Q16
        ├─ q<=Q32, D<=64 ─► Q32 (4× i16) ──► NttSlotCache::Q32
        ├─ q<=Q64 ────────► Q64 (3× i32) ──► NttSlotCache::Q64
        └─ fp128 family ──► Q128 (5× i32) ─► NttSlotCache::Q128
                                    │
                    linear kernels (unchanged chunking driver)
                                    │
              safe_crt_chunk_width / field partial sum (#134)
```

Const-generic `K` changes propagate through `CyclotomicCrtNtt<W, K, D>`,
`DigitMontLut<W, K>`, and prover `match` arms.
Prefer a single source of truth in `tables.rs` over duplicating prime arrays.

### Alternatives Considered

1. **Keep six/five primes.** Safe but leaves performance on the table; rejected.
2. **First four legacy Q32 primes.** Product too small for fp32 dense nv26
   one-shot even with chunking; rejected.
3. **Two i16 primes for Q16.** Too many reconstruction rounds on real widths;
   rejected.
4. **Separate large-profile cache for quotients only.** Violates global cutover
   and duplicates cache memory; rejected (spec review blocking question).
5. **Combine with range-chunking spec PR #108.** Split for review/merge order:
   #134 landed chunking; this PR lands primes only.

## Documentation

- This spec file.
- Update module docs in `ntt/tables.rs` and `crt_ntt.rs` dispatch comment to
  describe Q16 and reduced counts.
- No paper or verifier doc changes required.

## Execution

Suggested implementation slices:

0. **Slice 0 (recommended before prime tables):** add
   `mat_vec_mul_ntt_single_i8` / `_cyclic` forced-chunk tests on Q128; document
   that #134 Bugbot "wrong arg order" is closed as false positive (see audit
   section and #134 comment).
1. Add Q16 table + tests; add reduced Q32/Q64 tables + tests.
2. Extend `ProtocolCrtNttParams` / `NttSlotCache` / `select_crt_ntt_params`.
3. Fix const-generic `K` throughout prover linear + setup NTT cache build.
4. Extend capacity and forced-chunk tests for cyclic/fused/`z_pre` on new `K`.
5. Profile dense fp16/fp32/fp64; fix any unexpected chunk-count regressions.

Risks to resolve first:

- Confirm fp16 outer-B nv32 widths still chunk correctly under Q16 (may be
  two chunks; acceptable if correct).
- Update every `Q32_NUM_PRIMES` / `Q64_NUM_PRIMES` literal in tests and benches.

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
- `specs/SPEC_REVIEW.md`: review workflow for this spec PR.
