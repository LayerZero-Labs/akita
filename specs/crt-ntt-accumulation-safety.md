# Spec: CRT NTT Accumulation Safety

| Field       | Value                    |
|-------------|--------------------------|
| Author(s)   | Quang Dao                |
| Created     | 2026-05-28               |
| Status      | implemented              |
| PR          | #134                     |

## Summary

### 2026-07 large-basis extension

The original fix deliberately kept all balanced digits in i8. The inner
commitment decomposition now has an arithmetic capability for larger bases
without changing existing schedules: balanced base `2^L` digits use i8 for
`1 <= L <= 8` and signed i16 for `9 <= L <= 16`.

The exact coefficient bound is

```text
2 * W * D * floor(q / 2) * B < P
```

where `B = 2^(L-1)` for balanced digits. `select_crt_ntt_capability` evaluates
that strict inequality with overflow-safe multi-limb arithmetic. It retains the
base i32 profile when sufficient and otherwise appends exactly one residue
modulo 12289. The tail prime has `v2(p - 1) = 12`, hence supports every Akita
negacyclic degree through `D = 2048`. A schedule that fits the base product does
not construct tail twiddles, transforms, or matrix cache entries.

The terminal verifier deliberately uses the signed-i16 kernel for every
schedule, including schedules whose decomposition digits were produced by an
i8 prover kernel. It rejects decoded terminal coefficients outside i16 and
selects cache capability for the full bound `B = 32768`, so terminal cache
selection is independent of the schedule's decomposition storage type. For
the shipped q32 catalogs, the base profile's full-i16 safe widths are 63 at
D128 and 31 at D256; terminal widths are exactly 128 at D128 and 64--128 at
D256. Those schedules therefore require the 12289 tail as well.

At each field tier's maximum ring degree, the resulting safe matrix widths are:

| field profile | D | base 10 | base 10 + tail | base 11 | base 11 + tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q32/2xi32 | 2048 | 255 | 3,145,624 | 127 | 1,572,812 |
| Q64/3xi32 | 1024 | 127 | 1,572,760 | 63 | 786,380 |
| Q128/5xi32 | 512 | 15 | 196,592 | 7 | 98,296 |

The physical representation is mixed rather than pretending all primes have a
common machine width: the existing `CyclotomicCrtNtt<i32, K, D>` prefix and one
`CyclotomicCrtNtt<i16, 1, D>` tail. Mixed Garner reconstruction computes the
prefix digits with the existing table, one cross-prime tail digit, and Horner
accumulation directly in the target field. This adds exactly two bytes per
cached coefficient only when selected (25% Q32, 16.7% Q64, 10% Q128).
The cache is derived and contributes zero bytes to setup serialization.

A debug-profile construction diagnostic over 256 prepared D256 rings measured:

| profile | base construction / bytes | base + i16 tail / bytes |
| --- | ---: | ---: |
| Q64 | 31.38 ms / 786,432 | 45.98 ms / 917,504 |
| Q128 | 50.72 ms / 1,310,720 | 65.37 ms / 1,441,792 |

The byte deltas are exact (131,072 bytes = `256 rings * D256 * 2`); timings are
local construction diagnostics rather than cross-machine release claims.

Release Criterion measurements on Apple Silicon/NEON provide the current
backend baseline (the committed `ring_ntt` benchmark is reproducible on x86 to
collect AVX2/AVX-512 numbers). The end-to-end comparison uses the production
`Prime128OffsetA7F7` field, `D = 64`, and an 8-by-128 cached matvec; 128
accumulated columns cross the base-profile exactness boundary for base 10 and
therefore exercise a schedule for which the tail is actually relevant.

| operation | base | i16/mixed | delta |
| --- | ---: | ---: | ---: |
| one D64 forward+inverse residue | i32: 196.1 ns | i16/12289: 264.9 ns | 1.35x per residue |
| production Q128 D64 cached 8x128 matvec | 5xi32+i8: 281-360 us | 5xi32+1xi16+i16: 321-395 us | +9.8% to +14.0% |
| terminal Q128 D64 8x128, full kernel | two balanced-radix64 i8 passes: 281.4 us | one mixed i16 pass: 191.0 us | -32.1% |

The NEON implementation uses eight direct i16 lanes for independent pointwise
Montgomery products, but retains widening four-lane chains inside the
dependency-heavy NTT butterflies: applying the eight-lane `sqdmulh` form to the
whole transform regressed the D64 round trip by about 8%. Mixed reconstruction
precomputes the affine final Garner digit and performs one modular reduction
per coefficient instead of `K` dependent reductions; this improved the
diagnostic D32 mixed reconstruction by 10.5% and its cached matvec by 5.5%.
The remaining production overhead is paid only after the exactness selector
chooses the tail. These measurements do not establish x86 performance: the
AVX2 low/high-half transform and pointwise tests/benchmarks are present, but
must run on an x86 host before making an x86 throughput claim.

The cached-matvec row compares one already-transformed i8 RHS against one
already-transformed mixed RHS; it measures the marginal sixth residue, not the
terminal choice. The terminal row is the decision-relevant comparison and
includes signed-digit splitting, RHS NTTs, pointwise accumulation, inverse
NTTs, reconstruction, and radix scaling. The mixed path builds centered
Montgomery tables once per matvec and reuses them across all 128 RHS rings.

The D64 absolute timings were bimodal across macOS scheduler/core placement,
so the table reports two adjacent base/mixed trial pairs rather than treating
their absolute latency as stable. Their relative overhead is the useful local
signal; controlled pinned-core CI remains necessary for a release claim.

The i8 cutoff is `L = 8`, not `L = 6`: L7 digits are in `[-64, 63]` and L8
digits are in `[-128, 127]`. The fixed five-prime Montgomery LUT occupies about
5 KiB and initializes only its active range. On the same host, construction was
about 0.315 us for L6 and 0.713 us for L8. Keeping L7/L8 on this path avoids the
tail representation entirely; the planner may choose these bases once they are
Pareto-optimal even though the current generated schedules still top out at L6.

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
`log_basis`, digit lookup, centered lookup, and sparse signed-ring inputs.
Akita-owned predecomposed digit paths use the validated balanced
base-`2^log_basis` digit bound (`log_basis <= 8`) for both capacity planning
and the optimized allocation-free per-basis digit LUT. Direct non-LUT `i8`
conversion remains available for raw arbitrary-byte helpers. The full-field
generic quotient helper is no longer a production path; the remaining dense CRT
matvec helpers are test-only fixtures.

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

- Checked boundaries must reject malformed but type-correct inputs before they
  reach optimized kernels. In particular: `log_basis` values used with i8
  decomposition must be checked as `1..=8`; LUT-backed digit paths are for
  balanced `[-2^(log_basis-1), 2^(log_basis-1))` digits. Public APIs that
  receive caller-owned predecomposed digits scan those rows once and reject
  out-of-range digits with `AkitaError`; lower Akita-owned kernels use that
  boundary validation plus debug assertions before unchecked tight-LUT
  lookups. Centered-i32 capacity planning must remain correct even when a
  caller-provided max-abs hint is stale; sparse signed-ring coefficients must
  match the commit path's signed-unit assumption or the commit path must
  support the advertised range.

- Capacity planning and LUT sizing must distinguish digit provenance. Raw
  direct digit conversion can handle the full `i8` domain without using the
  balanced LUT. Prover-owned digit planes produced by Akita balanced
  decomposition use `B = 1 << (log_basis - 1)`, whose worst case is `128` at
  `log_basis = 8`; `DigitMontLut` has 256 fixed slots and initializes exactly
  the active `2^log_basis` entries. Release code may scan untrusted
  predecomposed inputs at public boundaries, but hot accumulation loops must
  not allocate or use checked table indexing on every coefficient.
  Recursive-witness rows with `num_digits_inner = 1` are direct signed-i8
  coefficients, not balanced gadget digits; the canonical block-major raw-i8
  path therefore plans capacity from the actual signed coefficient norm.

- The proof format, transcript order, Fiat-Shamir bytes, setup seed semantics,
  generated schedule tables, and verifier replay behavior must not change.

- The fix must preserve the existing public behavior for valid inputs. Invalid
  inputs should fail earlier and more explicitly, not reach unsafe indexing,
  `assert!`, `unwrap`, `expect`, or `unreachable!`.

- Performance overhead must be proportional to the extra reconstructions needed
  for correctness. Q32/Q64 and already-safe fp128 widths should remain on the
  existing one-reconstruction path. The implementation should use the largest
  safe chunk width for each operation and should not add per-column
  reconstruction or a slow schoolbook fallback to hot valid paths. Public
  predecomposed digit validation is a one-time boundary scan; the validated
  inner kernels use allocation-free fixed LUTs and unchecked indexing guarded
  by that boundary contract.

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

- Do not widen the Q128 CRT table unconditionally or use a larger CRT product
  as the fix for unbounded work. The large-basis extension above is a bounded,
  exactness-selected capability: it materializes one i16 residue only when a
  concrete `(q, D, W, B)` schedule cannot fit the base product.

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
- [x] `log_basis > 8` in commitment/prover paths fails with `AkitaError`
      before reaching i8 decomposition assertions.
- [x] Understated `z_pre_max_abs` / `z_pre_centered_inf_norm` cannot underplan
      capacity or reach an unchecked centered LUT access. Fused quotient code
      computes the actual centered infinity norm once, uses the larger of the
      provided hint and actual bound for capacity/LUT selection, and then uses
      unchecked LUT conversion only when that one-time bound proves it safe.
- [x] Public sparse signed-ring construction and commit agree on the allowed
      coefficient range: sparse coefficients are signed units only.
- [x] New adversarial tests include Q128/fp128 cases, Q32 capacity sanity, and
      Q64 dispatch sanity coverage.
- [x] Public predecomposed digit APIs reject digits outside the balanced
      `log_basis` range, and Q128 many-block small-row digit coverage exercises
      the path that would otherwise use block-parallel accumulation with an
      unsafe full width.
- [x] Recursive witness commits with `num_digits_inner = 1` use the canonical
      block-major raw signed-i8 path, so ZK blinding/sign-unit streams are not
      incorrectly treated as balanced binary digit planes.
- [x] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
      and `cargo test` pass.
- [x] Profile comparison has no unexplained material regression. CI benchmark
      run `26612741059` on head `58187d0e` accepted the required chunking cost
      after the tight-LUT/validation follow-up, and proof sizes were unchanged.
      The final code keeps the safe-width one-shot path, allocation-free
      per-basis digit LUTs, and the raw signed-i8 recursive-witness
      specialization.

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
  reconstruction capacity tests.
- `crates/akita-prover/src/api/commitment.rs` tests for `log_basis > 8`
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
- Contract tests for invalid `log_basis`, centered-bound underreporting,
  balanced digit LUT lookup, public predecomposed digit range rejection, raw
  signed-i8 recursive-witness coefficients, sparse non-signed-unit
  coefficients, and digit block-parallel width mismatch.

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

PR #134 uses the CI benchmark matrix as the review artifact. Benchmark run
`26612741059` completed on head `58187d0e` and passed the repository threshold
policy with unchanged proof sizes. The largest accepted positive deltas versus
the merge-base baseline were:

- setup: about +9.85% (`dense_fp16_d32`);
- prove: about +8.45% (`onehot_fp128_d32`, `nv30`, `np4`);
- verify: about +10.64% (`onehot_fp64_d32`).

Commit time improved in every reported case. The accepted increases are
consistent with the work needed to avoid previously unsafe CRT accumulation and
single-run benchmark variance; the one-shot fast path remains in place whenever
the full effective width fits the CRT lift range.

Memory overhead should remain bounded by the existing accumulator shape plus
the final field result. Do not allocate per-column rings or materialize a full
chunk-result matrix when a row/block accumulator can be reused.

## Design

### Issue Inventory

| Area | Representative path | Issue | Implemented resolution |
|------|---------------------|-------|------------------------|
| Fused split-eq | `crates/akita-prover/src/kernels/linear/fused_quotients.rs` | D/B cyclic rows and A quotient rows reconstructed wide CRT accumulators once. | Keep the fused one-shot path only when every role is safe; otherwise chunk D and B cyclic rows by their digit bounds, chunk A quotient cyclic/negacyclic intermediates independently, reconstruct each chunk, and add native field rings. If one centered term is too large for CRT, use a field-native exact quotient path. |
| Generic quotient | `crates/akita-prover/src/kernels/linear/crt_matvec.rs` | A production full-field CRT quotient path would be unsafe for fp128/Q128 because even one full-field RHS term may not fit. | Remove this as a production path. The file now contains `cfg(test)` dense CRT helpers used only as fixtures. |
| i8/digit matvec | `i8_matvec.rs`, `digits.rs`, `single_cyclic.rs`, `block_parallel.rs` | Balanced i8 RHS is small but wide fp128 rows can exceed Q128 lift range. | Compute the safe width from `D`, field modulus, RHS bound, and CRT product; use `log_basis <= 8` to plan Akita-owned predecomposed digits with the balanced bound rather than the full `i8` bound; preserve one-shot accumulation when safe; otherwise chunk and add reconstructed native field results. |
| Recursive witness raw-i8 | `compute/cpu.rs`, `ntt_matvec.rs`, `digits.rs` | The `num_digits_inner = 1` recursive witness specialization is a direct signed-i8 coefficient stream; ZK blinding can include `+1`, which is outside the balanced binary digit range. | Route the block-major stream through the canonical raw-i8 path. It computes the actual signed coefficient bound once, converts rows directly with `from_i8_with_params`, and chunks by that bound instead of using the balanced digit LUT. |
| Block-parallel clamp | `digits.rs`, `block_parallel.rs` | Fast path could bypass the generic `inner_width = min(mat_width, data_width)` clamp. | Dispatch to block-parallel paths only when the full effective width is both present and safe; otherwise use the shared chunked generic path. |
| Centered LUT bound | `crt_ntt_repr.rs`, `fused_quotients.rs` | `z_pre_max_abs` sized a LUT that was later indexed unchecked by actual coefficients. | Fused quotient code computes the actual centered bound once, uses it for capacity/LUT selection, avoids giant LUTs when the bound is too large, and calls unchecked LUT conversion only after the bound proves every coefficient is covered. |
| Digit LUT contract | `crt_ntt_repr.rs` and predecomposed digit kernels | Full-`i8` LUT coverage solved a broader contract than Akita-owned digit kernels need, while using full `i8` as the capacity bound over-chunks fp128 paths. | `DigitMontLut` is a fixed-array, const-generic balanced-digit table sized to exactly `2^log_basis` entries. Public predecomposed digit APIs validate caller-owned rows once; lower kernels thread `log_basis` into LUT selection and capacity planning, then use allocation-free unchecked lookup in hot loops. |
| Commit log basis | `api/commitment.rs`, `protocol/ring_switch.rs`, `protocol/quadratic_equation.rs`, `kernels/linear/ntt_matvec.rs` | Commit validation accepted `1..=128`; i8 decomposition supports `1..=8`. | Centralize `MAX_I8_LOG_BASIS = 8` in `validation.rs` and reject invalid setup/input log bases before decomposition. |
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
- fused centered quotient capacity uses the actual coefficient bound rather
  than trusting caller hints;
- LUT-backed digit kernels use the `log_basis <= 8` balanced capacity and
  exact per-basis lookup contract, with public predecomposed digit range
  validation before unchecked hot-loop lookup;
- recursive-witness direct signed-i8 streams do not use the balanced digit
  LUT contract when `num_digits_inner = 1`;
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
local error hardening: i8 `log_basis` must be `1..=8`, and sparse ring
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
6. Tightened local input contracts: `log_basis`, centered bound
   underreporting, allocation-free per-basis digit LUT safety,
   log-basis-bounded predecomposed digit capacity, public predecomposed digit
   range validation, and sparse signed-ring coefficient semantics.
7. Routed direct `num_digits_inner = 1` recursive-witness streams through the
   canonical block-major raw signed-i8 kernel.
8. Ran targeted tests, full format/clippy/test, line-cap checking, and release
   profile/CI benchmark commands.
9. Audited and documented performance in the PR body.

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
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/backend/sparse_ring.rs`
- `crates/akita-prover/src/validation.rs`
- `specs/SPEC_REVIEW.md`
