# Spec: CRT/NTT Range Chunking

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | @quangvdao                                 |
| Created     | 2026-05-26                                 |
| Status      | proposed                                   |
| PR          | #108                                       |

## Summary

Dense small-field commitments currently pay conservative CRT/NTT prime counts
for every matrix-vector product because each output row accumulates the full
matrix width in the CRT domain before reconstructing back into the base field.
That is safe, but it overpays for the common case where the right-hand side is a
small balanced digit vector. This spec proposes range chunking for CRT/NTT
matrix-vector kernels: accumulate only as many columns as the active CRT product
can reconstruct unambiguously, reconstruct that partial result into the target
field, then add the partials in the field. This lets Q32 use a proper i16 fast
profile and lets Q64 use fewer i32 primes without changing proof bytes,
transcripts, schedules, or public APIs.

## Intent

### Goal

Implement prover-side CRT/NTT range chunking for small-balanced-digit
matrix-vector kernels, and use it to replace the current conservative Q32/Q64
CRT profiles with smaller profiles that are safe for dense commitments.

The primary affected surfaces are:

- `akita-algebra::ntt::tables`: CRT prime tables and prime-count constants.
- `akita-algebra::ring::crt_ntt_repr`: CRT/NTT conversion, accumulation, and
  reconstruction helpers.
- `akita-prover::kernels::crt_ntt`: protocol-facing Q32/Q64/Q128 dispatch.
- `akita-prover::kernels::linear`: i8 and balanced-digit NTT matvec kernels.
- `akita-prover::backend::dense`: dense commitment paths that call the i8
  matvec kernels.
- `akita-prover::api::commitment`: the outer B commitment matvec.
- `akita-pcs::examples::profile`: performance validation for dense fp16/fp32/fp64.

The intended first-cut CRT profiles are:

| profile | current | proposed | purpose |
| --- | ---: | ---: | --- |
| Q32 | 6 x i16 | 4 x i16 | fp16/fp32, D <= 64 |
| Q64 | 5 x i32 | 3 x i32 | fp64, D <= 1024 |
| Q128 | 5 x i32 | unchanged | fp128, D <= 1024 |

For Q32, the implementation should not simply take the first four current
primes. It should choose four large i16 NTT primes below `2^14` with
`128 | (p - 1)`, so the same set supports D32 and D64. A suitable starting set
is:

```text
16001, 15361, 15233, 14593
```

Their product is about `2^55.60`, enough for the current dense fp32 D32 nv26
root width in one chunk and enough for larger widths through range chunking.

### Invariants

1. The mathematical output of every changed matvec must equal the existing
   full-width CRT accumulation reduced in the target field.
2. Range chunking must be transparent to proof bytes, Fiat-Shamir ordering,
   transcript labels, public claims, schedule selection, and verifier behavior.
3. The reconstruction bound must be enforced from first principles, not from
   benchmark-specific constants:

   ```text
   chunk_cols * D * floor(q / 2) * digit_abs < P_crt / 2
   ```

   where `q` is the target field modulus, `D` is the ring dimension,
   `digit_abs = 2^(log_basis - 1)`, and `P_crt` is the product of the active CRT
   primes.
4. The bound must be conservative for negacyclic ring multiplication. Each
   output coefficient of one column product is a signed sum of at most `D`
   products of a centered setup coefficient and a balanced digit.
5. A chunk size of zero is invalid. If a proposed profile cannot safely
   reconstruct even one column for a supported field and ring dimension, setup
   must fail loudly or the profile must not be selected.
6. Q128 must remain unchanged unless a separate proof shows that a smaller
   profile can safely reconstruct at least one q128 D32 product. Four 30-bit
   i32 primes are not enough for q128.
7. Existing verifier no-panic constraints remain unchanged. This is prover-side
   arithmetic and cache work; verifier-reachable validation and proof parsing do
   not gain unwraps, unchecked assumptions, or compatibility shims.
8. Setup serialization must remain canonical. Expanded setup artifacts store
   the shared field matrix, not serialized CRT/NTT caches, so changing CRT
   profiles must rebuild caches deterministically from the same field matrix.
9. There is no backward-compatibility layer. Internal constants and dispatch
   names may be cut over in place, and all call sites must use the new range
   chunking path.

### Non-Goals

1. No protocol, transcript, verifier, proof object, or serialized proof change.
2. No generated schedule-table change.
3. No public API for selecting legacy CRT prime counts.
4. No runtime compatibility shim for old Q32/Q64 constants.
5. No Q128 prime-count reduction in this spec.
6. No AVX/NEON rewrite. Existing scalar and NEON NTT kernels may be reused, but
   architecture-specific vectorization is a separate performance spec.
7. No security-parameter change. SIS modulus families and schedule derivation
   stay exactly as they are; this spec changes only prover-side CRT arithmetic
   used to compute the same field result.
8. No separate Q16 CRT family in the first implementation. A three-prime i16
   fp16-only profile may be evaluated later, but the first cut keeps fp16 on the
   same Q32 i16 machinery as fp32.

## Evaluation

### Acceptance Criteria

- [ ] Q32 uses a four-prime i16 profile with D32 and D64 NTT validity checked by
      tests.
- [ ] Q64 uses a three-prime i32 profile with D32 through D1024 NTT validity
      checked by tests.
- [ ] Q128 remains on the existing five-prime i32 profile.
- [ ] Every changed i8 matvec kernel range-chunks by a computed safe column
      width and reconstructs partial sums into the target field before adding
      them.
- [ ] The safe chunk-width helper has unit tests for fp16, fp32, fp64, Q32,
      Q64, and too-small profiles.
- [ ] Dense fp16/fp32/fp64 D32 commitment tests match the scalar ring-matvec
      reference over randomized small fixtures.
- [ ] Existing dense and one-hot end-to-end tests continue to pass.
- [ ] Profile benchmarks for dense fp16/fp32/fp64 D32 run successfully and show
      no proof-byte or verifier-result changes.
- [ ] Performance results are recorded for the current bench targets:
      `dense_fp16_d32:26:1`, `dense_fp32_d32:26:1`, and the re-enabled
      `dense_fp64_d32:26:1`.

### Testing Strategy

Required focused tests:

- `akita-algebra` table tests:
  - verify every Q32 prime is prime, below `2^14`, and satisfies
    `128 | (p - 1)`;
  - verify every Q64/Q128 prime satisfies `2048 | (p - 1)`;
  - verify Garner constants for the new prime counts.
- `akita-prover::kernels::crt_ntt` tests:
  - Q32 still dispatches for fp16/fp32 with D32 and D64;
  - Q64 still dispatches for fp64 and for small fields with D > 64;
  - Q128 still dispatches for supported q128 families.
- `akita-prover::kernels::linear` tests:
  - range-chunked `mat_vec_mul_ntt_single_i8` equals a scalar ring reference;
  - range-chunked dense digit kernels equal a scalar ring reference;
  - forced tiny chunk widths exercise multi-chunk accumulation deterministically;
  - too-small profiles that cannot fit one column are rejected.
- End-to-end tests:
  - existing `akita-pcs` dense and one-hot tests remain green;
  - fp16/fp32/fp64 dense D32 profiles commit/prove/verify with unchanged proof
    success.

Required local checks before implementation PR review:

```bash
cargo fmt -q
cargo test -q -p akita-algebra ntt
cargo test -q -p akita-prover crt_ntt
cargo test -q -p akita-prover linear
cargo test -q -p akita-pcs akita_e2e
cargo clippy --all --message-format=short -q -- -D warnings
```

### Performance

Performance must be measured against the profile example in release mode:

```bash
AKITA_MODE=dense_fp16_d32 AKITA_NUM_VARS=26 AKITA_NUM_POLYS=1 cargo run --release -p akita-pcs --example profile
AKITA_MODE=dense_fp32_d32 AKITA_NUM_VARS=26 AKITA_NUM_POLYS=1 cargo run --release -p akita-pcs --example profile
AKITA_MODE=dense_fp64_d32 AKITA_NUM_VARS=26 AKITA_NUM_POLYS=1 cargo run --release -p akita-pcs --example profile
```

When CI coverage includes D32 for all fp16/fp32/fp64 dense modes, the profile
bench comment should compare before/after setup, commit, prove, verify, and
proof bytes.

Expected direction:

- Q32 setup and dense commitment should improve because each cached NTT element
  stores four i16 limbs instead of six.
- Q64 setup and dense commitment should improve because each cached NTT element
  stores three i32 limbs instead of five.
- Very wide fp32 cases may add several reconstruction chunks. The implementation
  is acceptable only if the reduced per-column prime count beats or roughly
  ties the extra partial reconstruction overhead on the benchmark matrix.
- Proof bytes and verifier time should remain unchanged except for measurement
  noise.

Approximate current dense D32 root widths:

| config | root B width | proposed profile behavior |
| --- | ---: | --- |
| fp16 nv26 | 98,304 | Q32 four-prime i16, one chunk |
| fp32 nv26 | 114,688 | Q32 four-prime i16, one chunk with larger primes |
| fp64 nv26 | 135,168 | Q64 three-prime i32, one chunk |
| fp32 nv32 | 2,097,152 | Q32 four-prime i16, multiple chunks |

## Design

### Architecture

The current kernels already tile by L2 cache size, but those tiles are only
performance tiles. They are reduced back into one CRT accumulator and
reconstructed once:

```text
for cache_tile in columns:
    acc_crt += A_tile * digit_tile
return reconstruct(acc_crt)
```

Range chunking changes the accumulation boundary:

```text
out = 0 in R_q
for range_chunk in columns:
    acc_crt = 0
    for cache_tile inside range_chunk:
        acc_crt += A_tile * digit_tile
    out += reconstruct(acc_crt)
return out
```

Cache tiling and range chunking are independent. The range chunk is the
correctness boundary. The cache tile remains a performance detail inside each
range chunk.

The implementation should add a small helper near the CRT/NTT parameter set or
linear kernels that computes:

```text
safe_chunk_cols(params, field_modulus, D, log_basis) -> Result<usize, AkitaError>
```

For Q32 and Q64, exact `u128` arithmetic is enough for the proposed products.
For Q128, the implementation can keep the current unchunked five-prime path or
use a conservative bit-width helper, but it must not silently truncate the CRT
product.

The helper should use:

```text
digit_abs = 1 << (log_basis - 1)
coeff_abs = floor(q / 2)
per_col_bound = D * coeff_abs * digit_abs
safe_cols = floor((P_crt / 2 - 1) / per_col_bound)
```

The changed kernels should preserve their current row/block parallelism where
possible. For each row or block output, they should accumulate each safe range
chunk in CRT form, reconstruct to `CyclotomicRing<F, D>`, and add into the
field-domain output. Existing `CyclotomicRing` field addition performs the
final modular reduction in `F`.

Initial kernel coverage should include:

- `mat_vec_mul_ntt_single_i8`;
- `mat_vec_mul_ntt_dense_digits_i8`;
- `mat_vec_mul_ntt_digits_i8`;
- `mat_vec_mul_ntt_i8_dense`;
- the strided i8 variants used by recursive or batched paths when they share
  the same accumulation contract.

Any kernel left on the old full-width accumulation path must either keep a CRT
profile large enough for its worst-case width or be explicitly listed as out of
scope before implementation review.

### i16 Fast Path

The right Q32 fast path is four i16 primes plus range chunking.

Two i16 primes are not enough for a fp32-capable path. Their product is about
`2^28`, while one fp32 D32 ring-column product with `log_basis = 2` needs about
`D * q * 2 ~= 2^38` of reconstruction range. That fails before width is even
considered. For fp16, two i16 primes would fit only tiny chunks, producing too
many reconstruct-and-add rounds on real dense widths.

Three i16 primes are plausible for a future fp16-only profile, but not for
fp32. Adding a separate Q16 CRT family in this pass would complicate dispatch
and testing while leaving fp32 unsolved. This spec therefore keeps one Q32 i16
path for both fp16 and fp32.

A single i32 prime is also not enough for fp32 D32: a 30-bit NTT prime is below
the same one-column fp32 bound. Two i32 primes are an alternative for Q32, with
about the same memory per coefficient as four i16 primes and fewer CRT limbs.
The reason not to choose it as the default is that Akita already has i16 NTT
and i16 NEON coverage, and four i16 limbs preserve that lane density. The
implementation may benchmark a two-i32 Q32 spike, but it should not replace
the four-i16 design without updating this spec with evidence.

For Q64, three i32 primes are the minimum sensible target. Two i32 primes are
about `2^60`, while one fp64 D32 ring-column product with `log_basis = 3` needs
about `2^71`.

For Q128, five i32 primes remain required. Four i32 primes are about `2^120`,
which is below the one-column q128 D32 bound.

### Alternatives Considered

1. Keep six Q32 primes and five Q64 primes.

   This is safe and simple, but it bakes the largest matrix width into every
   multiply. It misses the fact that field-domain addition after reconstruction
   can safely compose smaller CRT partials.

2. Drop Q32 to the first four existing i16 primes without changing kernels.

   Rejected. The first four current Q32 primes have product about `2^54.74`,
   just below the current fp32 dense nv26 one-shot requirement. Also, without
   range chunking, wider matrices would remain unsafe.

3. Use two i16 primes for Q32.

   Rejected. It is not enough for even one fp32 D32 product and would force very
   small chunks for fp16.

4. Use a single i32 prime for Q32.

   Rejected. It is not enough for even one fp32 D32 product.

5. Use two i32 primes for Q32.

   Plausible as a measured spike. It may reduce loop count relative to four
   i16 primes, but it gives up the existing i16 kernel shape and may lose on
   SIMD lane density. The default design remains four i16 primes.

6. Add a separate Q16 CRT profile for fp16.

   Deferred. A three-prime i16 fp16-only profile might be useful, but it adds
   another dispatch and test matrix. This spec first fixes the shared small
   field path.

7. Reduce Q128 with range chunking.

   Rejected in this spec. Four i32 primes do not safely reconstruct one q128
   D32 product, so a Q128 reduction would need a different prime family or a
   different multiplication strategy.

## Documentation

This spec is the main documentation artifact. Implementation should also update
inline comments in:

- `crates/akita-algebra/src/ntt/tables.rs`, to describe the smaller profiles
  and the range-chunking dependency;
- `crates/akita-prover/src/kernels/linear.rs`, to distinguish range chunks from
  cache tiles;
- `AGENTS.md` or profiling docs only if the canonical profile modes or bench
  interpretation changes.

## Execution

Suggested implementation slices:

1. Add a range-bound helper and focused tests using existing prime sets.
2. Teach `mat_vec_mul_ntt_single_i8` to range-chunk while preserving current
   output and row parallelism.
3. Extend the helper to dense digit, generic i8, and strided kernels.
4. Reduce Q64 to three i32 primes and validate fp64 dense profiles.
5. Replace Q32 with four large D64-valid i16 primes and validate fp16/fp32
   dense profiles.
6. Run focused tests, workspace clippy, and release profiles.
7. Record benchmark movement in the implementation PR description and update
   this spec if measurements force a different prime-count choice.

Risks to resolve early:

- Reconstruction inside many chunks may move cost from pointwise multiply into
  inverse NTT and Garner reconstruction. Benchmarks must decide whether four
  i16 beats two i32 for the largest fp32 dense target.
- Existing helper names use "tile" for cache locality. The implementation must
  avoid mixing cache tile width with range chunk width.
- Q128 products exceed `u128`; helper code must not accidentally reuse exact
  product arithmetic where it does not fit.
- Some kernels use cyclic NTT paths for quotient construction. If a cyclic path
  is range-chunked, the bound must account for the same coefficient magnitude
  and tests must compare against the existing scalar quotient reference.

## References

- `specs/SPEC_REVIEW.md`
- `specs/TEMPLATE.md`
- `specs/small-field-prover-opening-optimization.md`
- `crates/akita-algebra/src/ntt/tables.rs`
- `crates/akita-algebra/src/ring/crt_ntt_repr.rs`
- `crates/akita-prover/src/kernels/crt_ntt.rs`
- `crates/akita-prover/src/kernels/linear.rs`
- `crates/akita-prover/src/backend/dense.rs`
- `crates/akita-prover/src/api/commitment.rs`
