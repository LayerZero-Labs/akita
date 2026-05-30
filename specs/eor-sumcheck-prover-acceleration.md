# Spec: EOR Sumcheck Prover Acceleration (small-field one-hot)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Taghi Badakhshan               |
| Created     | 2026-05-30                     |
| Status      | implemented                    |
| PR          | `taghi/perf/eor-sc` ([#136](https://github.com/LayerZero-Labs/akita/pull/136))     |

## Summary

This PR accelerates the Akita prover for small-field, one-hot polynomial
commitments without changing proofs. The work is concentrated in the
extension-opening-reduction (EOR) sumcheck and its surrounding root/recursive
prep, which the profile identified as the dominant prover cost for small fields:
because a small base field (e.g. fp32) carries too little soundness on its own,
the protocol runs the EOR over a degree-`d` extension (`d = 4` for fp32, `8` for
fp16, `2` for fp64, `1` for fp128), so the extension-heavy phases dominate prove
time and scale with `d`. Every change is **byte-identical** (proof bytes,
transcript, schedules, and setup artifacts are unchanged) and **prover-only**
(the verifier is untouched). The net result is a large prover speedup that grows
with field-extension degree: ~26% at nv32 for `onehot_fp32_d32` from the latest
three commits alone, and ~2.5× at nv32 for `onehot_fp16_d32` once the same EOR
fast paths are ported to the degree-8 field.

## Intent

### Goal

Make the EOR sumcheck and its root/recursive prep faster for small-field one-hot
modes in `akita-sumcheck`, `akita-prover`, and `akita-field`, without changing
proof format, transcript bytes, schedules, setup artifacts, the verifier, or any
public PCS API.

### Invariants

- **Byte-identical proofs.** For every benchmarked mode/`num_vars`,
  `proof_size_bytes` and the full proof are unchanged versus the pre-PR baseline,
  and the proof verifies. Protected by
  `sparse_tensor_factor_matches_dense_factor_rounds` and its per-prime
  instantiations (`..._fp32_*`, `..._fp8_fp16_*`) in
  `crates/akita-sumcheck/tests/extension_opening_reduction.rs`, the EOR
  round-by-round equality tests, and the proof-size gate in the profile harness.
- **Delayed-reduction exactness.** Where a field sets
  `HasUnreducedOps::DELAYED_PRODUCT_SUM_IS_EXACT = true`, the wide accumulator
  must equal per-term `Mul` reduction exactly (no `u128`/`u64` wrap or
  underflow). Protected by the `*_product_accum_matches_direct_mul` and
  `*_accum_summation_large_operands` tests in
  `crates/akita-field/src/fields/ext/tests.rs`.
- **Specialized fold == generic fold.** Each `HasOptimizedFold` matrix
  (`FoldMatrixFp32/Fp16/Fp64`) must reproduce `even + r·(odd − even)` exactly.
  Protected by `*_fold_matrix_matches_generic` / `*_optimized_fold_matches_generic`
  tests.
- **Field specialization is opt-in.** Only the intended challenge field of each
  mode is specialized (`RingSubfieldFp4<Fp32>`, `RingSubfieldFp8<Fp16>`,
  `Fp2<Fp64>`); all other base/extension types keep the existing generic paths
  via default-impl macros.
- **Verifier no-panic contract unchanged.** No new verifier-reachable
  `panic!`/unchecked indexing/shape assumptions; the EOR verifier and all
  verifier paths are byte-for-byte unchanged.
- **Parallel reductions are order-independent.** All newly parallelized folds,
  maps, and dot products reduce with associative/commutative field operations,
  so the result is identical to the serial form regardless of chunking.

### Non-Goals

- No protocol/soundness change. The extension degree, challenge sampling, and
  SIS layout are untouched; this is a constant-factor prover speedup, not an
  asymptotic or security change.
- No proof layout, schedule search policy, transcript binding, or serialization
  change.
- No verifier optimization.
- No SIMD-packed EOR factor evaluation (evaluated and rejected — see
  Alternatives).
- The one-hot "multiply-by-basis shift" is **not** landed (evaluated and
  rejected — see Alternatives).

## Evaluation

### Acceptance Criteria

- [x] `cargo fmt -q`
- [x] `cargo clippy --all --all-targets --message-format=short -q -- -D warnings`
- [x] `cargo test` (notably `-p akita-field`, `-p akita-sumcheck`, `-p akita-prover`)
- [x] `cargo doc -q --no-deps --all-features`
- [x] Byte-identical: `proof_size_bytes` matches the pre-PR baseline for
  `onehot_fp32_d32` (34432/36048 @ nv26/28), `onehot_fp16_d32` (29808/30608),
  and `onehot_fp64_d32` (40808/41688), each with `exit_code 0`.
- [x] Controlled, drift-canceled local A/B confirms the prover is faster than
  `origin/main` on every benchmarked mode (no regression on non-targeted modes).

### Testing Strategy

The EOR byte-identicality guards compare the optimized sparse/lazy path against
the dense reference round-by-round and on final terms, instantiated per base
prime and per field family (fp32 `RingSubfieldFp4`, fp16 `RingSubfieldFp8`, fp64
`Ext2`/`Fp2`). New unit tests cover each wide accumulator against direct `Mul`
(including large-operand and 4096-product summation cases) and each fold matrix
against the generic fold (including worst-case `p−1` coordinates). End-to-end
correctness is the profile harness proof-size gate plus `verify OK` on every
benchmarked case. Benchmarks are CI-faithful: release, `RUSTFLAGS=-C
target-cpu=native`, all cores, median/min over interleaved drift-canceled runs.

### Performance

Measured locally on Apple Silicon (NEON), release + native, drift-canceled
interleaved A/B; medians unless noted. CI raw deltas are unreliable because the
CI baseline runs in a separate job on a different runner (a uniform ~+28% on the
untouched `verify`/`setup` columns reveals the machine difference); normalize by
an unchanged phase or trust the same-machine A/B below.

Whole branch vs `origin/main` (prover, `prove_total_s`):

| mode | nv | Δ vs main |
|------|----|-----------|
| `onehot_fp32_d32` | 32 | ~−26% (1381-sample drift-canceled A/B) |
| `onehot_fp64_d32` | 32 | −22.5% |
| `dense_fp32_d32`  | 26 | −15% |
| `onehot_fp128_d32`| 32 | −2.9% |

fp16 EOR fast-path port vs pre-port HEAD (`onehot_fp16_d32`):

| nv | Δ | speedup |
|----|----|---------|
| 28 | −32.3% | 1.46× |
| 30 | −47.9% | 1.92× |
| 32 | −59.6% | 2.48× |

The speedup grows with `num_vars` and with extension degree, because the EOR's
share of prove time grows in both. `fp64_fold` is intentionally small (~−1% at
nv30/32): at degree 2 the generic Karatsuba fold is already cheap. `fp128`
(degree 1) gains little because its EOR is negligible.

## Design

### Architecture

All changes live in three crates and are reachable only from prover paths.

**`akita-sumcheck` — EOR sumcheck core** (`src/extension_opening_reduction/`):
- Algorithmic: drop the redundant `c1` term and recover it from the claim; drop
  the `w1 == 0` branch in sparse accumulation.
- `TensorEqualityFactor`: lazy prefix/suffix tensor-equality factor that avoids
  materializing the dense factor table and takes the delayed-reduction
  `eval_state_at_suffix_fast` path when the field's accumulator is exact;
  `factor_pair` fuses the `a0`/`a1` evaluations to share the suffix-table column
  load.
- Delayed-reduction accumulation in the round message: accumulate wide products
  in `E::ProductAccum` and reduce once per round.

**`akita-field` — field-gated fast paths** (`src/fields/`):
- `HasUnreducedOps`: wide `ProductAccum` types and `mul_to_product_accum` for
  `RingSubfieldFp4<Fp32>` (`RingSubfieldFp4Fp32ProductAccum`), `Fp2<Fp64>`
  (`Fp2Fp64ProductAccum`), and `RingSubfieldFp8<Fp16>`
  (`RingSubfieldFp8Fp16ProductAccum`), each with `DELAYED_PRODUCT_SUM_IS_EXACT =
  true`; non-target bases keep the identity accumulator.
- `HasOptimizedFold`: precomputed "multiply-by-`r`" fold matrices
  `FoldMatrixFp32` (4×4), `FoldMatrixFp16` (8×8), `FoldMatrixFp64` (2×2) so the
  EOR fold uses base-field limb arithmetic with a single delayed reduction
  instead of per-element extension multiplies; non-target types keep the generic
  fold.
- `Fp64::reduce_sum_of_two_products` carry-correct reduction, and the
  `Fp2<Fp64>` accumulator carry fix that makes its delayed sum exact.

**`akita-prover` — prep and kernels**:
- One-hot sparse root-extension column partials
  (`backend/onehot/`, `tensor_extension_column_partials_batch`): tensor-factor
  the high eq-weights and scatter by head (per-chunk add instead of a dense
  `O(2^high)` build + per-chunk multiply), parallel over outer blocks.
- Parallelize the recursive-EOR-prep serial loops
  (`protocol/flow/recursive.rs`, `extension_opening_reduction/mod.rs`):
  column-partials fold, input-claim dot product, and witness-evals maps,
  threshold-gated.
- `algebra/poly.rs` parallel `fold_evals_in_place`; raise
  `SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS` (4→7) in the block-parallel NTT kernel.

The EOR prover and sumcheck code remain generic over the field `E`; all
field-specificity is isolated to the two `akita-field` trait hooks, so adding a
new field's fast path is a self-contained `akita-field` change.

### Alternatives Considered

- **One-hot "multiply-by-basis" shift** (replace `e_head · r` with a structured
  basis multiply at the fold/accumulate). Rejected: the witness×factor multiply
  is the minority of EOR work (the general tensor-factor evaluation dominates),
  so even an inlined, byte-identical implementation was e2e-neutral to slightly
  negative, and the per-entry head detection scaled the wrong way. Not landed.
- **NEON SIMD-packing of the EOR factor evaluation** (`PackedRingSubfieldFp4`
  across lanes). Rejected: ~22% slower in the kernel — the sparse scattered
  gather dominates, the campaign prime takes the heavy per-product carry path,
  and NEON lacks lane-wise add-with-carry for the add-bound accumulation. The
  safe fallback (fuse `a0`/`a1` evals) was kept instead (`factor_pack`).
- **Karatsuba `RingSubfieldFp4` wide multiply** (16→~9 products). Rejected: the
  `Fp32` wide multiply is add/carry-bound (16 cheap `umull` + carry chains), not
  multiply-bound, so trading multiplies for more recombination adds would
  regress.
- **Specialized fold / delayed reduction for fp128 (degree 1)**. Skipped: the
  EOR is negligible at degree 1, so there is no payoff.

## Documentation

- This retrospective spec is the PR-specific spec artifact.
- Benchmarks are reproduced with the committed harness `scripts/profile_bench_report.py`
  and the canonical `cargo run --release --example profile` command (see References),
  run drift-canceled and interleaved.
- No README or profile-guide changes are required: the optimizations are
  transparent (same modes, same proofs); the canonical profile command is
  unchanged.

## Execution

Implemented as a sequence of byte-identical commits on `taghi/perf/eor-sc`
(see the branch git log for exact revisions):

- Skip `c1` in EOR accumulation and recover it from the claim.
- Remove the redundant `w1 == 0` branch in sparse EOR accumulation.
- Delayed-reduction accumulators for EOR.
- Precomputed dense fold (`FoldMatrixFp32`) + fused fold/accumulate.
- Parallel fold, wider block-parallel NTT threshold, lazy tensor factor
  (`TensorEqualityFactor`).
- Preserve carry in the `Fp2<Fp64>` delayed-reduction accumulator (enables
  fp64's delayed path).
- One-hot sparse root-extension column partials.
- Parallelize recursive EOR prep.
- Fuse `a0`/`a1` tensor-factor evals in EOR accumulate.
- fp16 EOR delayed-reduction + 8×8 specialized fold (`RingSubfieldFp8<Fp16>`).
- fp64 specialized 2×2 fold (`Fp2<Fp64>`).

Risk notes: the only correctness-sensitive surface is the wide accumulators
(`DELAYED_PRODUCT_SUM_IS_EXACT`) and the fold matrices; each is guarded by a
direct-vs-generic unit test plus the end-to-end EOR byte-identicality guard and
the profile proof-size gate. The `u64` lanes for the fp16 accumulator have ~2^28
accumulation headroom; widening to `u128` is the safe fallback at extreme
`num_vars` with very few threads.

## References

- Profile: `AKITA_MODE=onehot_fp32_d32 AKITA_NUM_VARS=32 RUSTFLAGS="-C target-cpu=native" cargo run --release --example profile -p akita-pcs`
- Benchmark harness: `scripts/profile_bench_report.py`.
- Related specs: `specs/small-field-prover-opening-optimization.md`,
  `specs/fp16-small-field-support.md`, `specs/extension-field-opening-batching.md`,
  `specs/fp31-field-optimization-retrospective.md`.
