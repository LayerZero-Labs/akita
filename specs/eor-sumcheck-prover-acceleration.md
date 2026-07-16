# Spec: EOR Sumcheck Prover Acceleration (small-field one-hot)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Taghi Badakhshan               |
| Created     | 2026-05-30                     |
| Status      | in progress                    |
| PR          | `taghi/perf/eor-sc` ([#136](https://github.com/LayerZero-Labs/akita/pull/136))     |

## Summary

This work accelerates the Akita prover for small-field, one-hot polynomial
commitments without changing proofs.
The work is concentrated in the extension-opening-reduction (EOR) sumcheck and
its surrounding root and recursive prep, which the profile identified as the
dominant prover cost for small fields.
Because a small base field (e.g. fp32) carries too little soundness on its own,
the protocol runs the EOR over a degree-`d` extension (`d = 4` for fp32, `8` for
fp16, `2` for fp64, `1` for fp128), so the extension-heavy phases dominate prove
time and scale with `d`.

The PR also performs a crate-boundary cutover: EOR is an Akita protocol gadget,
so its concrete prover state and instance implementation live in `akita-prover`,
while `akita-sumcheck` stays protocol-independent.

Every change is **byte-identical** (proof bytes, transcript bytes, schedules,
and setup artifacts are unchanged).
Verifier semantics are unchanged, but verifier code may be adjusted mechanically
to consume the moved implementation.
The net result is a large prover speedup that grows with field-extension degree:
about 26% at nv32 for `onehot_fp32_d32` from the EOR-focused commits, and about
2.5× at nv32 for `onehot_fp16_d32` once the same EOR fast paths are ported to the
degree-8 field.

## Intent

### Goal

Make the EOR sumcheck and its root/recursive prep faster for small-field one-hot
modes, and enforce a clear crate boundary where protocol-specific prover code
stays in `akita-prover` (not `akita-sumcheck`).

Maintain byte-identical proof and transcript bytes.
Do not change proof format, schedule search policy, transcript binding, setup
artifacts, soundness, or any public PCS API.

### Invariants

- **Byte-identical proofs.** For every benchmarked mode/`nuposition_index_bits`,
  `proof_size_bytes` and the full proof are unchanged versus the pre-PR baseline,
  and the proof verifies. Protected by
  `sparse_tensor_factor_matches_dense_factor_rounds` and its per-prime
  instantiations (`..._fp32_*`, `..._fp8_fp16_*`) in the EOR test suite (moved
  out of `akita-sumcheck` as part of the crate-boundary cutover), the EOR
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
- **Verifier no-panic contract preserved.** No new verifier-reachable
  `panic!`/unchecked indexing/shape assumptions.
  Verifier code may move or be mechanically rewritten, but it must preserve the
  same validation boundaries and reject malformed inputs with `AkitaError`.
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
- No SIMD-packed EOR factor evaluation (evaluated and rejected, see
  Alternatives).
- The one-hot "multiply-by-basis shift" is **not** landed (evaluated and
  rejected, see Alternatives).

### Requirements and Guardrails

- **Full cutover, no backward compatibility.**
  When moving code, update all call sites in one pass.
  Do not keep deprecated re-exports, shims, or parallel old and new APIs.
- **`akita-sumcheck` stays protocol-independent.**
  It may own generic sumcheck proof containers, traits, and transcript drivers.
  It must not own Akita-specific prover state that carries witness tables.
- **Prover-only performance paths must not leak verifier-side risk.**
  Any verifier-reachable helper must be total over malformed input, return
  `AkitaError`, and avoid unchecked indexing and overflow-prone arithmetic unless
  earlier validation proves the invariant.
- **No proof or transcript drift.**
  Keep proof bytes and transcript bytes identical for existing modes.
  Treat any drift as a blocker, not an acceptable behavior change.

## Evaluation

### Acceptance Criteria

- [x] `cargo fmt -q`
- [x] `cargo clippy --all --all-targets --message-format=short -q -- -D warnings`
- [x] `cargo test` (notably `-p akita-field`, `-p akita-prover`, `-p akita-sumcheck`, `-p akita-verifier`)
- [x] `cargo doc -q --no-deps --all-features`
- [x] Byte-identical: `proof_size_bytes` matches the pre-PR baseline for
  `onehot_fp32_d32` (34432/36048 @ nv26/28), `onehot_fp16_d32` (29808/30608),
  and `onehot_fp64_d32` (40808/41688), each with `exit_code 0`.
- [x] Controlled, drift-canceled local A/B confirms the prover is faster than
  `origin/main` on every benchmarked mode (no regression on non-targeted modes).
- [x] Crate-boundary cutover complete: EOR prover code lives in `akita-prover`, and `akita-sumcheck`
  contains only protocol-independent sumcheck machinery.

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

The speedup grows with `nuposition_index_bits` and with extension degree, because the EOR's
share of prove time grows in both. `fp64_fold` is intentionally small (~−1% at
nv30/32): at degree 2 the generic Karatsuba fold is already cheap. `fp128`
(degree 1) gains little because its EOR is negligible.

## Design

### Crate Boundary and Ownership

This PR enforces a clean ownership split:

- `akita-sumcheck` owns only protocol-independent sumcheck machinery.
  Proof containers, traits (`SumcheckInstanceProver`, `SumcheckInstanceVerifier`,
  eq-factored variants), and generic transcript drivers stay here.
- `akita-prover` owns Akita protocol gadgets that compute sumcheck round
  polynomials from witness data.
  EOR is one such gadget, so its concrete prover state and instance
  implementation live in `akita-prover`.
- `akita-verifier` owns verifier replay and final checks.
  Verifier semantics are unchanged, but the verification code may be rewritten
  to call generic sumcheck drivers plus EOR-specific final checks.
- `akita-types` may host pure EOR tensor and output helper functions if doing so
  avoids duplicating logic across prover and verifier crates.
  The concrete EOR prover instance and its witness-bearing mutable state must
  stay in `akita-prover`.

### Architecture

All performance-critical EOR prover logic lives under `akita-prover`, and field
gated fast paths live under `akita-field`.
`akita-sumcheck` remains a generic sumcheck crate, not a home for protocol
gadgets.

**`akita-prover` — EOR prover gadget and prep**:
- EOR prover state over witness and factor tables, plus sparse and batched EOR
  terms.
- Root and recursive EOR prep, including one-hot sparse root-extension column
  partials and any threshold-gated parallel folds.
- Call sites in `protocol/flow/{root_extension,recursive}.rs` and related paths.

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

**`akita-prover` — kernels**:
- One-hot sparse root-extension column partials
  (`backend/onehot/`, `tensor_extension_column_partials_batch`): tensor-factor
  the high eq-weights and scatter by head (per-chunk add instead of a dense
  `O(2^high)` build + per-chunk multiply), parallel over outer blocks.
- Parallelize the recursive-EOR-prep serial loops
  (`protocol/flow/suffix.rs`, `extension_opening_reduction/mod.rs`):
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
  across lanes). Rejected: ~22% slower in the kernel, the sparse scattered
  gather dominates, the campaign prime takes the heavy per-product carry path,
  and NEON lacks lane-wise add-with-carry for the add-bound accumulation. The
  safe fallback (fuse `a0`/`a1` evals) was kept instead (`factor_pack`).
- **Karatsuba `RingSubfieldFp4` wide multiply** (16→~9 products). Rejected: the
  `Fp32` wide multiply is add/carry-bound (16 cheap `umull` + carry chains), not
  multiply-bound, so trading multiplies for more recombination adds would
  regress.
- **Specialized fold / delayed reduction for fp128 (degree 1)**. Skipped: the
  EOR is negligible at degree 1, so there is no payoff.

## Implementation Order and Checklist

This PR is intentionally a full cutover, not a transitional migration.
The checklist is ordered to keep the proof and transcript invariants protected
throughout.

### Phase 0: Spec and worklog

- [x] Update this spec to reflect the intended final architecture and crate boundary.
- [x] Maintain a visible, untracked worklog at repo root: `WORKLOG-NEVER-COMMIT.md`.

### Phase 1: Move EOR out of `akita-sumcheck`

- [x] Move `crates/akita-sumcheck/src/extension_opening_reduction/` to
  `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- [x] Remove `extension_opening_reduction` from `akita-sumcheck/src/lib.rs` exports.
- [x] Update all imports in prover, verifier, tests, benches, and examples to point to
  the new module location.
- [x] Ensure `akita-sumcheck` compiles without any EOR-specific modules.

### Phase 2: Remove the EOR-specific sumcheck driver type

- [x] Delete `ExtensionOpeningReductionSumcheck` and use generic sumcheck drivers instead.
  - Prover: call `SumcheckInstanceProverExt::prove` (or ZK equivalent) on the EOR prover instance.
  - Verifier: replay rounds via `SumcheckProof::verify` after absorbing the input claim.
- [x] Keep transcript bytes identical by preserving the same absorb and challenge sampling order.

### Phase 3: Place verifier-shared EOR helpers

- [x] Identify EOR helper functions needed by both prover and verifier
  (for example transparent tensor-factor evaluation at a point).
- [x] Place those helpers in `akita-types` (or a lower crate) if that avoids duplication.
  Verifier-used helpers must remain defensive and satisfy the verifier no-panic contract.
  Prover table-materialization helpers may use the optional `parallel` feature.

### Phase 4: Move tests and benches to match ownership

- [x] Move EOR unit tests out of `akita-sumcheck` into `akita-prover` (or `akita-pcs` integration
  tests if they require multiple crates).
- [x] Update any benchmarks that import EOR types to use the new module location.

### Phase 5: Verification gate

- [x] `cargo fmt -q`
- [x] `cargo clippy --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier -- -D warnings`
- [x] `cargo clippy --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier --features zk -- -D warnings`
- [x] `cargo test -p akita-prover --test extension_opening_reduction`
- [x] `cargo doc -q --no-deps --all-features`
- [x] Byte-identical: proof bytes and transcript bytes unchanged for the benchmarked modes.

Risk notes: the only correctness-sensitive surface is the wide accumulators
(`DELAYED_PRODUCT_SUM_IS_EXACT`) and the fold matrices.
Each must be guarded by a direct-vs-generic unit test plus the end-to-end EOR
byte-identicality guard and the profile proof-size gate.
The `u64` lanes for the fp16 accumulator have about \(2^{28}\) accumulation headroom.
Widening to `u128` is the safe fallback at extreme `nuposition_index_bits` with very few threads.

## References

- Profile: `AKITA_MODE=onehot_fp32_d32 AKITA_NUM_VARS=32 RUSTFLAGS="-C target-cpu=native" cargo run --release --example profile -p akita-pcs`
- Benchmark harness: `scripts/profile_bench_report.py`.
- Related specs: `specs/small-field-prover-opening-optimization.md`,
  `specs/fp16-small-field-support.md`, `specs/extension-field-opening-batching.md`,
  `specs/fp31-field-optimization-retrospective.md`.
