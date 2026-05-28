# Spec: CRT NTT Accumulation Safety

| Field       | Value                    |
|-------------|--------------------------|
| Author(s)   | Quang Dao                |
| Created     | 2026-05-28               |
| Status      | implemented              |
| PR          | #134                     |

## Summary

Several prover linear kernels accumulate products modulo auxiliary CRT primes
and run Garner reconstruction only once at the end. That is machine-overflow
safe, but it is semantically exact only while every true integer coefficient
stays inside the signed CRT lift range. For fp128/Q128, the fixed CRT product
leaves little headroom after one full-width setup coefficient, so wide
commitment, ring-switch, and quotient rows can silently alias before they are
mapped back to the field.

This feature makes CRT/NTT accumulation exact by bounding each accumulation
before reconstruction, chunking wide small-RHS work at the largest safe width,
and tightening the public helper contracts that previously allowed unchecked
LUT access or later panics.

PR #134 implements the fix with a shared capacity helper in the linear-kernel
module, capacity-aware chunking for the affected fused, i8, digit, single-row,
cyclic, and block-parallel kernels, and local validation/hardening for i8
`log_basis`, digit lookup, centered lookup, and sparse signed-ring inputs. The
full-field generic quotient helper is no longer a production path; the
remaining dense CRT matvec helpers are test-only fixtures.

## Intent

### Goal

Make every supported CRT/NTT linear kernel produce the same field result as an
independent schoolbook ring computation for all valid Akita schedules and
inputs, with minimal production-code surface and minimal unavoidable
performance overhead.

### Invariants

- Every Garner reconstruction from a CRT accumulator must be preceded by an
  operation-specific capacity argument. If `P` is the product of the auxiliary
  CRT primes, `Q` is the native field modulus, `B` bounds the RHS coefficient
  magnitude, `D` is the ring degree, and `W` is the number of accumulated
  columns, the implemented conservative safety condition is:

  ```text
  W * D * Q * B < P / 2
  ```

  This intentionally uses `Q` rather than the tighter centered setup bound
  `Q / 2`. Adversarial fp128 rows at the half-modulus boundary can otherwise
  sit too close to the CRT/Garner lift boundary, and the extra factor preserves
  correctness with modest additional chunking.

- For quotient kernels that compute `(cyclic - negacyclic) / 2`, the cyclic
  and negacyclic intermediates must each satisfy the CRT lift bound before
  reconstruction. It is not sufficient for only the final high-half quotient to
  fit after cancellation.

- Chunking is valid only when a single term fits the CRT lift range. If
  `D * A * B >= P / 2`, chunking cannot make the CRT path exact; the
  implementation must either use a field-native exact algorithm for that path
  or reject the input at a checked boundary with `AkitaError`.

- Chunked results must be accumulated after reconstruction in native field
  rings, not by adding chunk residues back into the same CRT accumulator.

- Public safe APIs must not allow unchecked LUT out-of-bounds access, undefined
  behavior, or panics from malformed but type-correct inputs. In particular:
  `log_basis` values used with i8 decomposition must be checked as `1..=6`;
  digit lookup must cover the full `i8` domain or validate public digits before
  lookup; centered-i32 lookup must be bounds-safe even when a caller-provided
  max-abs bound is stale; sparse signed-ring coefficients must match the commit
  path's signed-unit assumption or the commit path must support the advertised
  range.

- The proof format, transcript order, Fiat-Shamir bytes, setup seed semantics,
  generated schedule tables, and verifier replay behavior must not change.

- The fix must preserve the existing public behavior for valid inputs. Invalid
  inputs should fail earlier and more explicitly, not reach unsafe indexing,
  `assert!`, `unwrap`, `expect`, or `unreachable!`.

- Performance overhead must be proportional to the extra reconstructions needed
  for correctness. Q32/Q64 and already-safe fp128 widths should remain on the
  existing one-reconstruction path. The implementation should use the largest
  safe chunk width for each operation and should not add per-column
  reconstruction or a slow schoolbook fallback to hot valid paths. Release-mode
  validation should stay O(1) per call where possible; coefficient scans are
  reserved for debug assertions or existing boundary validation.

- Diff surface is a first-class requirement. Production code should add one
  small shared capacity/chunking abstraction and reuse it across affected
  kernels. Avoid duplicated kernel copies, broad rewrites, planner changes, new
  dependencies, compatibility wrappers, and style-only churn. If the production
  diff grows beyond a small focused patch, the PR must explain why that broader
  change buys correctness or maintainability.

- Touched files must remain comfortably under the 1500-line cap. If focused
  tests would push a file near the cap, split the tests into an appropriate
  module rather than bloating an existing file.

### Non-Goals

- Do not widen the Q128 CRT table, add more auxiliary primes, or select a
  larger CRT product as the primary fix. The bug is unbounded pre-lift work;
  the fix is to never let work exceed the lift range.

- Do not reduce generated schedules, lower supported problem sizes, or change
  planner policy to avoid the bug.

- Do not change proof serialization, transcript labels or payload ordering,
  setup artifacts, commitment layout, or public verifier semantics.

- Do not replace all CRT/NTT kernels with schoolbook/native-field arithmetic.
  Native-field fallback is acceptable only where the per-term bound makes CRT
  chunking impossible or where a path is demonstrably cold and the simpler
  exact algorithm is the smaller, cleaner fix.

- Do not land tests that only compare one CRT fast path to another CRT fast
  path. The adversarial correctness tests need an independent field-arithmetic
  oracle.

## Evaluation

### Acceptance Criteria

- [x] The implementation introduces a shared way to compute or enforce the
      safe CRT accumulation width for each linear-kernel operation:
      `max_safe_crt_accumulation_width` and `safe_crt_chunk_width` in
      `crates/akita-prover/src/kernels/linear/capacity.rs`.
- [x] Q128 fused split-eq quotient tests compare against an independent
      schoolbook high-half oracle and cover widths that previously exceeded the
      one-shot reconstruction bound.
- [x] Q128 i8/digit matvec tests cover widths above the old single-accumulator
      capacity and pass by chunking before reconstruction.
- [x] Cyclic and negacyclic quotient tests cover the case where relying on
      final quotient cancellation would be wrong.
- [x] The generic unreduced quotient path is removed from production use rather
      than given a risky full-field CRT fallback; `crt_matvec.rs` now contains
      test-only dense helpers.
- [x] Block-parallel digit kernels apply the same effective width clamp and
      safety policy as the generic digit kernels, and dispatch only when the
      full effective width is safe.
- [x] `log_basis > 6` in commitment/prover paths fails with `AkitaError`
      before reaching i8 decomposition assertions.
- [x] Understated `z_pre_max_abs` / `z_pre_centered_inf_norm` cannot reach an
      unchecked centered LUT access. `CenteredMontLut::get` returns `Option`
      and exact conversion is used as fallback; release capacity still relies
      on the prover's validated `centered_inf_norm` bound, with debug
      assertions guarding local caller mistakes.
- [x] Public sparse signed-ring construction and commit agree on the allowed
      coefficient range: sparse coefficients are signed units only.
- [x] New adversarial tests include Q128/fp128 cases, Q32 capacity sanity, and
      Q64 dispatch sanity coverage.
- [x] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
      and `cargo test` pass.
- [x] Profile comparison has no unexplained material regression. CI accepted
      the benchmark matrix; proof sizes were unchanged. Affected fp128 one-hot
      profiles pay the expected chunking cost, with commit around +10-14% and
      prove around +5.7%.

### Testing Strategy

Existing checks that must remain green:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

New or updated tests live near the affected code:

- `crates/akita-prover/src/kernels/linear/tests.rs` for fused split-eq,
  i8/digit matvec, single-row, block-parallel clamp parity, and chunking
  behavior.
- `crates/akita-pcs/tests/algebra/ntt_crt.rs` for lower-level CRT
  reconstruction capacity and partial-split non-regression tests.
- `crates/akita-prover/src/api/commitment.rs` tests for `log_basis > 6`
  rejection before decomposition.
- `crates/akita-prover/src/backend/sparse_ring.rs` tests for the sparse
  signed-unit contract.

Adversarial fixtures should use deterministic, sign-aligned inputs near the
capacity boundary: setup rows with coefficients near `q/2`, RHS digit planes at
`-32` or `31`, and `z_pre` coefficients at explicitly chosen centered bounds.
Expected values must come from direct field/ring arithmetic, not from another
CRT kernel.

The test suite includes:

- Fused split-eq A quotient with fp128/Q128, `D=32`, centered `z_pre`, and
  enough columns to exceed the old one-shot reconstruction bound.
- i8 single-row, pair/triple or block-parallel paths with fp128/Q128, `D=64`,
  and a width that requires chunking.
- A cyclic-only case for D/B rows where no quotient cancellation exists.
- A quotient case that demonstrates cyclic and negacyclic intermediates must be
  safe independently.
- A near-capacity Q32 or Q64 test that remains under the bound and verifies the
  capacity formula is not overly pessimistic by orders of magnitude.
- Contract tests for invalid `log_basis`, centered LUT fallback, full-range
  public digit lookup, sparse non-signed-unit coefficients, and digit
  block-parallel width mismatch.

### Performance

The performance target is "pay only for necessary correctness." The
implementation should preserve the existing single-accumulator path whenever
the full operation width is safe, and should choose the largest safe chunk size
otherwise.

Measure the canonical profile before and after the implementation on the same
machine, preferably with repeated runs and median comparison:

```bash
AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 cargo run --release --example profile
```

If the dense representative profile remains flaky or fails on base `main`,
record that fact and use targeted kernel/profile spans instead of treating one
failed profile command as a regression. Single-run timing deltas are not enough
evidence for either approval or rejection.

PR #134 uses the CI benchmark matrix as the current review artifact. The latest
run passed. Proof sizes were unchanged. The notable accepted deltas were:

- fp128 one-hot D32 nv32: commit about +13.5%, prove about +5.7%;
- fp128 one-hot D32 nv30 np4: commit about +10.6%, prove about +5.5%;
- fp32 dense D32: prove about +8.1%.

These increases are consistent with chunking work that was previously unsafe.
The one-shot fast path remains in place when the full effective width fits the
CRT lift range.

Memory overhead should remain bounded by the existing accumulator shape plus
the final field result. Do not allocate per-column rings or materialize a full
chunk-result matrix when a row/block accumulator can be reused.

## Design

### Issue Inventory

| Area | Representative path | Issue | Implemented resolution |
|------|---------------------|-------|------------------------|
| Fused split-eq | `crates/akita-prover/src/kernels/linear/fused_quotients.rs` | D/B cyclic rows and A quotient rows reconstructed wide CRT accumulators once. | Keep the fused one-shot path only when every role is safe; otherwise chunk D and B cyclic rows by their digit bounds, chunk A quotient cyclic/negacyclic intermediates independently, reconstruct each chunk, and add native field rings. If one centered term is too large for CRT, use a field-native exact quotient path. |
| Generic quotient | `crates/akita-prover/src/kernels/linear/crt_matvec.rs` | A production full-field CRT quotient path would be unsafe for fp128/Q128 because even one full-field RHS term may not fit. | Remove this as a production path. The file now contains `cfg(test)` dense CRT helpers used only as fixtures. |
| i8/digit matvec | `i8_matvec.rs`, `digits.rs`, `single_cyclic.rs`, `block_parallel.rs` | Balanced i8 RHS is small but wide fp128 rows can exceed Q128 lift range. | Compute the safe width from `D`, field modulus, RHS bound, and CRT product; preserve one-shot accumulation when safe; otherwise chunk and add reconstructed native field results. |
| Block-parallel clamp | `digits.rs`, `block_parallel.rs` | Fast path could bypass the generic `inner_width = min(mat_width, data_width)` clamp. | Dispatch to block-parallel paths only when the full effective width is both present and safe; otherwise use the shared chunked generic path. |
| Centered LUT bound | `crt_ntt_repr.rs`, `fused_quotients.rs` | `z_pre_max_abs` sized a LUT that was later indexed unchecked by actual coefficients. | `CenteredMontLut::get` is bounds-checked and falls back to exact conversion on miss. Fused quotient code avoids giant LUTs and uses debug assertions to catch stale local bounds. |
| Digit LUT contract | `crt_ntt_repr.rs` and public digit kernels | Safe callers can pass arbitrary `i8`, but LUT covered only `[-32, 31]`. | `DigitMontLut` covers all 256 `i8` values, avoiding release-mode scans on hot public digit paths. |
| Commit log basis | `api/commitment.rs`, `protocol/ring_switch.rs`, `protocol/quadratic_equation.rs`, `kernels/linear/ntt_matvec.rs` | Commit validation accepted `1..=128`; i8 decomposition supports `1..=6`. | Centralize `MAX_I8_LOG_BASIS = 6` in `validation.rs` and reject invalid setup/input log bases before decomposition. |
| Sparse signed-ring contract | `backend/sparse_ring.rs` | Constructor accepted any nonzero `i8`; commit assumed signed units and used `unreachable!`. | Sparse ring construction now rejects all coefficients except `-1` and `1`, matching the commit path. |

### Architecture

The implemented shared capacity helper near the linear CRT kernels answers
two questions:

1. Is this operation safe to run as one CRT accumulation?
2. If not, what is the largest safe chunk width?

The helper is conservative, exact, and cheap. `capacity.rs` implements a local
`SmallNat` multi-limb integer type, avoiding new dependencies and dynamic big
integers in the hot loop. `max_safe_crt_accumulation_width` binary-searches the
largest `width` satisfying `2 * width * D * Q * B < P`, after first requiring a
single term to fit. `safe_crt_chunk_width` clamps that width to the operation's
full effective width. Quotient operations use the cyclic and negacyclic
intermediate bounds, not the final high-half bound.

The linear kernels share one chunking pattern:

1. determine the effective input width once;
2. compute the max safe chunk width for the operation;
3. if the full width is safe, keep the current fast path shape;
4. otherwise, process contiguous column chunks, reconstruct each chunk to
   `CyclotomicRing<F, D>`, and accumulate those chunk results in native field
   rings.

The block-parallel kernels do not fork a separate safety policy. They receive
or recompute the same effective-width safety decision and are used only for
full-width safe cases.

The generic full-field quotient helper is not a production path in the current
implementation. Since a full-field RHS can be too large even for a single Q128
CRT product, the production fix deliberately avoids adding a broad slow
fallback there. Exact field-native fallback is used only for centered
fused-quotient terms when the single-term CRT bound fails.

Validation fixes should stay local:

- commitment parameter validation rejects unsupported i8 `log_basis` values;
- centered LUT construction/use is bounds-safe and falls back exactly on miss;
- public digit-kernel entrypoints avoid unchecked LUT preconditions by covering
  the full `i8` domain;
- sparse signed-ring construction and commit agree that values are signed units
  only.

### Alternatives Considered

- **Add more Q128 CRT primes.** Rejected. It increases table/code size and only
  moves the boundary; it does not enforce the invariant that work must not
  exceed the lift range.

- **Change schedules to avoid wide rows.** Rejected. It hides a kernel
  correctness bug in planner policy and risks proof/performance tradeoffs
  unrelated to this fix.

- **Always use native field arithmetic.** Rejected for hot valid i8/digit
  paths because it discards the existing CRT/NTT performance design. It remains
  acceptable for paths where CRT cannot represent even one term exactly.

- **Add ad hoc chunk loops in each kernel.** Rejected unless the shared helper
  becomes more complex than the problem. Duplicated chunking logic would make
  future capacity bugs easier to reintroduce.

- **Trust existing generated schedules and add only tests.** Rejected. The bug
  is input- and schedule-dependent, and safe public APIs should not rely on
  unspoken generated-table assumptions.

## Documentation

This spec is the durable design note for the fix. PR #134 references it in the
PR body and summarizes:

- the capacity formula used by the shared helper;
- which paths chunk and which paths remain one-shot;
- the centered fused-quotient path that uses native field fallback when one
  centered term cannot fit the CRT lift range;
- the measured performance impact and the commands used.

No user-facing README changes are required. The public behavioral changes are
local error hardening: i8 `log_basis` must be `1..=6`, and sparse ring
coefficients must be signed units.

## Execution

Implemented order:

1. Added adversarial coverage for fused split-eq and i8/digit cases, using
   independent field arithmetic oracles.
2. Added the shared CRT accumulation capacity helper and unit-tested Q128/Q32
   boundary decisions.
3. Applied chunking to fused split-eq D/B cyclic rows and A quotient rows,
   preserving the fused one-shot path when all roles are safe.
4. Applied the same chunking policy to i8, digit, single-row, cyclic, and
   block-parallel kernels; block-parallel dispatch now requires full safe
   effective width.
5. Removed the generic unreduced full-field quotient path from production use
   instead of adding a broad fallback. Kept `crt_matvec.rs` as test-only dense
   helpers.
6. Tightened local input contracts: `log_basis`, centered LUT fallback, full
   `i8` digit lookup, and sparse signed-ring coefficient semantics.
7. Ran targeted tests, full format/clippy/test, line-cap checking, and release
   profile commands.
8. Audited and documented performance in the PR body.

Deviation policy:

- If a broader refactor is needed to avoid duplicated chunking logic, keep it
  inside the linear-kernel ownership boundary and document the reason in the PR
  body.
- If a proposed fix increases hot-path runtime materially, first check whether
  the full-width safe path still bypasses chunking and whether chunk size is
  maximal. Then compare repeated medians before accepting the regression.
- If a failing adversarial case cannot be reproduced, do not delete the issue;
  record the concrete bound or caller invariant that makes it safe and encode
  that invariant in a test.

## References

- `crates/akita-prover/src/kernels/linear/fused_quotients.rs`
- `crates/akita-prover/src/kernels/linear/capacity.rs`
- `crates/akita-prover/src/kernels/linear/common.rs`
- `crates/akita-prover/src/kernels/linear/crt_matvec.rs`
- `crates/akita-prover/src/kernels/linear/i8_matvec.rs`
- `crates/akita-prover/src/kernels/linear/digits.rs`
- `crates/akita-prover/src/kernels/linear/single_cyclic.rs`
- `crates/akita-prover/src/kernels/linear/block_parallel.rs`
- `crates/akita-prover/src/kernels/linear/ntt_matvec.rs`
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`
- `crates/akita-algebra/src/ring/partial_split_ntt.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/backend/sparse_ring.rs`
- `crates/akita-prover/src/validation.rs`
- `specs/SPEC_REVIEW.md`
