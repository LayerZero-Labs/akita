# Spec: CRT NTT Accumulation Safety

| Field       | Value                    |
|-------------|--------------------------|
| Author(s)   | Quang Dao                |
| Created     | 2026-05-28               |
| Status      | proposed                 |
| PR          | TBD                      |

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
and tightening the public helper contracts that currently allow unchecked LUT
access or later panics.

## Intent

### Goal

Make every supported CRT/NTT linear kernel produce the same field result as an
independent schoolbook ring computation for all valid Akita schedules and
inputs, with minimal production-code surface and minimal unavoidable
performance overhead.

### Invariants

- Every Garner reconstruction from a CRT accumulator must be preceded by an
  operation-specific capacity argument. If `P` is the product of the auxiliary
  CRT primes, `A` bounds the centered setup coefficient magnitude, `B` bounds
  the RHS coefficient magnitude, `D` is the ring degree, and `W` is the number
  of accumulated columns, the conservative safety condition is:

  ```text
  W * D * A * B < P / 2
  ```

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
  digit LUT users must validate or otherwise prove digits are in `[-32, 31]`;
  centered-i32 LUT users must prove the supplied max-abs bound covers the
  actual coefficients; sparse signed-ring coefficients must match the commit
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
  reconstruction or a slow schoolbook fallback to hot valid paths.

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

- [ ] The implementation introduces a shared way to compute or enforce the
      safe CRT accumulation width for each linear-kernel operation.
- [ ] Q128 fused split-eq quotient tests fail on current `main` and pass after
      the fix by comparing against an independent schoolbook high-half oracle.
- [ ] Q128 i8/digit matvec tests cover widths above the old single-accumulator
      capacity and pass by chunking before reconstruction.
- [ ] Cyclic and negacyclic quotient tests cover the case where relying on
      final quotient cancellation would be wrong.
- [ ] The generic unreduced quotient path is either made exact for its actual
      caller bounds or is replaced/guarded where one full-field term cannot fit
      the CRT lift range.
- [ ] Block-parallel digit kernels apply the same effective width clamp and
      safety policy as the generic digit kernels.
- [ ] `log_basis > 6` in commitment/prover paths fails with `AkitaError`
      before reaching i8 decomposition assertions.
- [ ] Understated `z_pre_max_abs` / `z_pre_centered_inf_norm` cannot reach an
      unchecked centered LUT access.
- [ ] Public sparse signed-ring construction and commit agree on the allowed
      coefficient range; either values other than `-1` or `1` are rejected up
      front or the commit path handles them exactly.
- [ ] New adversarial tests include Q128/fp128 cases and at least one
      near-capacity Q32 or Q64 sanity case to protect the bound itself.
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
      and `cargo test` pass.
- [ ] A repeated profile comparison shows no unexplained material regression.
      Any median regression above 5% in the canonical profile, or above 10% in
      a directly affected kernel span, must be explained and either optimized
      or explicitly accepted in review.

### Testing Strategy

Existing checks that must remain green:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

New or updated tests should live near the affected code:

- `crates/akita-prover/src/kernels/linear/tests.rs` or a split sibling test
  module for fused split-eq, i8/digit matvec, block-parallel clamp parity, and
  generic quotient behavior.
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

The test suite should include at least:

- Fused split-eq A quotient with fp128/Q128, `D=32`, centered `z_pre`, and
  enough columns to exceed the old one-shot reconstruction bound.
- i8 single-row, pair/triple or block-parallel paths with fp128/Q128, `D=64`,
  and a width that requires chunking.
- A cyclic-only case for D/B rows where no quotient cancellation exists.
- A quotient case that demonstrates cyclic and negacyclic intermediates must be
  safe independently.
- A near-capacity Q32 or Q64 test that remains under the bound and verifies the
  capacity formula is not overly pessimistic by orders of magnitude.
- Contract tests for invalid `log_basis`, understated centered LUT bounds,
  out-of-range public digit planes if such planes remain public, sparse
  non-signed-unit coefficients, and digit block-parallel width mismatch.

### Performance

The performance target is "pay only for necessary correctness." The
implementation should preserve the existing single-accumulator path whenever
the full operation width is safe, and should choose the largest safe chunk size
otherwise.

Measure the canonical profile before and after the implementation on the same
machine, preferably with at least five runs and median comparison:

```bash
AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 cargo run --release --example profile
```

If the dense representative profile remains flaky or fails on base `main`,
record that fact and use targeted kernel/profile spans instead of treating one
failed profile command as a regression. Single-run timing deltas are not enough
evidence for either approval or rejection.

Memory overhead should remain bounded by the existing accumulator shape plus
the final field result. Do not allocate per-column rings or materialize a full
chunk-result matrix when a row/block accumulator can be reused.

## Design

### Issue Inventory

| Area | Representative path | Issue | Required fix |
|------|---------------------|-------|--------------|
| Fused split-eq | `crates/akita-prover/src/kernels/linear/fused_quotients.rs` | D/B cyclic rows and A quotient rows reconstruct wide CRT accumulators once. | Chunk per role using operation-specific RHS bounds; reconstruct each chunk and add native field results. |
| Generic quotient | `crates/akita-prover/src/kernels/linear/crt_matvec.rs` | Cyclic and negacyclic intermediates are reconstructed once, and full-field RHS may not fit even one Q128 CRT term. | Prove actual caller RHS bound and chunk, or route to exact field-native logic / checked rejection where CRT cannot be exact. |
| i8/digit matvec | `i8_matvec.rs`, `digits.rs`, `single_cyclic.rs`, `block_parallel.rs` | Balanced i8 RHS is small but wide fp128 rows can exceed Q128 lift range. | Compute max safe width from `D`, field modulus, digit bound, and CRT product; chunk only when needed. |
| Block-parallel clamp | `digits.rs`, `block_parallel.rs` | Fast path can bypass the generic `inner_width = min(mat_width, data_width)` clamp. | Pass the effective width into block-parallel kernels or otherwise enforce the same range before indexing. |
| Centered LUT bound | `crt_ntt_repr.rs`, `fused_quotients.rs` | `z_pre_max_abs` sizes a LUT that is later indexed unchecked by actual coefficients. | Validate/recompute the bound before using the LUT, or use checked construction that cannot index out of range. |
| Digit LUT contract | `crt_ntt_repr.rs` and public digit kernels | Safe callers can pass arbitrary `i8`, but LUT covers only `[-32, 31]`. | Validate public digit planes once, make unsafe preconditions explicit internally, or use a safe fallback for unchecked external input. |
| Commit log basis | `api/commitment.rs`, `ring/cyclotomic/decomposition.rs` | Commit validation accepts `1..=128`; i8 decomposition supports `1..=6`. | Reject unsupported log bases before decomposition. |
| Sparse signed-ring contract | `backend/sparse_ring.rs` | Constructor accepts any nonzero `i8`; commit assumes signed units and uses `unreachable!`. | Align constructor and commit semantics with a single checked coefficient contract. |

### Architecture

Add a small shared capacity helper near the linear CRT kernels. It should answer
two questions:

1. Is this operation safe to run as one CRT accumulation?
2. If not, what is the largest safe chunk width?

The helper should be conservative and cheap. It can use bit-length arithmetic or
checked integer arithmetic over the available modulus/CRT metadata; it does not
need dynamic big integers in the hot loop. It should be parameterized by the
RHS magnitude bound and by whether the operation reconstructs cyclic,
negacyclic, or quotient intermediates. Quotient operations must use the
intermediate bound, not the final high-half bound.

The linear kernels should then share one chunking pattern:

1. determine the effective input width once;
2. compute the max safe chunk width for the operation;
3. if the full width is safe, keep the current fast path shape;
4. otherwise, process contiguous column chunks, reconstruct each chunk to
   `CyclotomicRing<F, D>`, and accumulate those chunk results in native field
   rings.

The block-parallel kernels should not fork a separate safety policy. They
should receive the same effective width/chunk plan or call the same shared
helper.

For the generic full-field quotient helper, the implementation must first
identify the actual production callers and their RHS bounds. If the RHS can be
full-width field elements under a supported fp128 schedule, one-term Q128 CRT
is not enough; that path needs field-native exact arithmetic or a checked error
rather than CRT chunking.

Validation fixes should stay local:

- commitment parameter validation should reject unsupported i8 `log_basis`
  values;
- centered LUT construction/use should verify the advertised max-abs against
  actual coefficients before unchecked indexing;
- public digit-kernel entrypoints should validate digit range or avoid exposing
  unchecked LUT preconditions;
- sparse signed-ring construction and commit should agree on whether values are
  signed units only or arbitrary small signed coefficients.

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

This spec is the durable design note for the fix. The implementation PR should
reference it in the PR body and briefly summarize:

- the capacity formula used by the shared helper;
- which paths chunk and which paths remain one-shot;
- any path that uses native field fallback because one-term CRT is impossible;
- the measured performance impact and the commands used.

No user-facing README changes are required unless the implementation changes a
public API error condition in a way downstream users need to know.

## Execution

Recommended implementation order:

1. Add or update adversarial tests first, confirming at least the fused
   split-eq and i8/digit cases fail on current `main`.
2. Add the shared CRT accumulation capacity helper and unit-test its boundary
   decisions.
3. Apply chunking to fused split-eq D/B cyclic rows and A quotient rows.
4. Apply the same chunking policy to i8, digit, single, cyclic, and
   block-parallel kernels; fix the block-parallel effective-width clamp at the
   same time.
5. Resolve the generic unreduced quotient path after auditing the production
   caller bounds. Use chunking only if one-term CRT is safe; otherwise use an
   exact native-field path or checked rejection.
6. Tighten the local input contracts: `log_basis`, centered LUT bound, public
   digit range, and sparse signed-ring coefficient semantics.
7. Run targeted tests, then full format/clippy/test.
8. Run repeated profile measurements and include the median comparison in the
   PR notes.
9. Before review, audit the diff file by file. Every production change should
   map to a listed bug, a shared helper, or a measured performance/code-quality
   improvement.

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
- `crates/akita-prover/src/kernels/linear/crt_matvec.rs`
- `crates/akita-prover/src/kernels/linear/i8_matvec.rs`
- `crates/akita-prover/src/kernels/linear/digits.rs`
- `crates/akita-prover/src/kernels/linear/single_cyclic.rs`
- `crates/akita-prover/src/kernels/linear/block_parallel.rs`
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`
- `crates/akita-algebra/src/ring/partial_split_ntt.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/backend/sparse_ring.rs`
- `specs/SPEC_REVIEW.md`
