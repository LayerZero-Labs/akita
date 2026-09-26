# Coverage inventory

Scope: the Akita workspace at the commit this file ships with, the pinned
`jolt-field`/`jolt-poly` revision it depends on, and the 16 shipped schedule
artifacts in `artifacts/schedules/`. "Existing testing" summarizes the
workspace's own tests; targets are in `src/targets/` and registered in
`campaign/targets.toml`.

Properties are named as in the campaign contract: **R** robustness (no
crash or hang on arbitrary input at a public entry point), **C**
completeness (valid supported statements prove and verify), **S** soundness
testing (false statements and invalid proofs are rejected), **M** resource
behavior (bounded requests stay within limits).

## Supported statements

A valid Akita opening is a catalog row: an exact final group
`(num_vars, num_polys)` plus an ordered list of precommitted group profiles.
Rows not in the loaded catalog are *unsupported*, not invalid, and the
targets require `UnsupportedSchedule` for them rather than success. The
end-to-end targets enumerate rows from the artifacts at startup, so the case
list tracks the catalogs without restating any sizing rule. At the default
per-process cost caps (total committed coefficients across groups) the
planned rows are:

| Family | Planned rows (final; precommitted) | Excluded by cost |
|---|---|---|
| fp128_dense | 14:1, 15:2, 16:1, 16:2, 17:4 (dense target: ≤2^18); 16:1 with 14:1 | 24–32 |
| fp128_dense_bounded | 14:1 | 24, 26 |
| fp128_dense_multi_chunk | 16:1 | – |
| fp128_onehot | 12:1, 14:1, 14:2, 15:1, 15:4, 16:1, 16:2, 18:1, 20:1; 16:1 with 14:1 (one-hot and bounded-dense profiles); 16:1 with 14:1 + dense 15:2 | 20:2 and larger |
| fp128_onehot_multi_chunk | 16:1 | 32, 34 |
| fp128_onehot_multi_chunk_w2r2 | 14:1; 14:1 with 14:1 | 32 |
| fp128_onehot_multi_chunk_w4r2 | – | 32:1 (only row) |
| fp128_dense_recursive | 20:1 | 22–28 |
| fp128_onehot_recursive, …_multi_chunk_w8r2 | – | all rows ≥ 2^32 |
| fp32_dense | 20:1 | 20:1 with 20:1, 26–30 |
| fp32_onehot | 14:1, 16:1, 16:2, 20:1 | 20:1 with 14:1, 28–34 |
| fp32_dense_recursive | 20:1 | 22–30 |
| fp64_dense | 14:1, 16:1, 20:1 | 20:1 with 16:1, 26–30 |
| fp64_onehot | – | all rows ≥ 2^28 |
| fp64_dense_recursive | – | 21:1 is 2^21 (raise the cap to include) |

Caps are `AKITA_FUZZ_MAX_CASE_COEFFS` (per process) with defaults 2^17
(reject, parallel), 2^18 (dense), 2^20 (one-hot, batch), 2^21 (recursive),
2^16 (prover boundary) and 2^20 (verifier boundary, which proves once per case). `akita-fuzz-dev cases 20` prints the exact list.
Production-sized rows (nv ≥ 22 and most multi-chunk and recursive rows) are
not fuzzed: one case would take minutes and gigabytes under ASan. They are
covered only by the workspace's ignored release tests.

## Inventory

| Subsystem | Production entry points | Configs / backends | Existing testing | Target (props) | Valid-input generator | Oracle | Boundaries exercised | Cost / exec (ASan) | Excluded, why |
|---|---|---|---|---|---|---|---|---|---|
| Base fields (jolt-field, pinned) | `Field`, `CanonicalEncoding` ops; `from_u128_*`, `from_bytes_le_*`, `mul_*` | Prime32Offset99, Prime64Offset59, Prime128OffsetA7F7 | Fixed and seeded identities; bigint oracle for the two fp128 primes only | `field_arith` (R, C) | Boundary-biased scalar tokens (0, ±1, ±2^k±1, q/2 neighbors, i8/i16/i32 edges), raw bytes | `num-bigint` | Canonical vs noncanonical bytes; reduction of every u128 | µs | Fields unused by Akita configs |
| Unreduced accumulators (dependency contracts Akita relies on) | `mul_unreduced`, `mul_base_unreduced`, `mul_u64_unreduced`, `reduce_*`, `Wide`, `scale_wide`, `WithCommitAccumulator` | same fields + FpExt4, Ext2 | Indirect only | `field_arith` (C) | Single terms for every field; up to 32-term sums; wide sums at exactly `MAX_COMMIT_ACCUMULATIONS` and one below; fp128 sums of 1–1024 products, including all-`(q−1)` factors | Per-term reduced arithmetic. Multi-term product sums are required only where exactness is claimed: `SUM_IS_EXACT` (FpExt4, Ext2) and the fp128 headroom `eval_ring_at_pows_fast` documents | Declared headroom | ≤ ms | `scale_wide` beyond ±2^15 (contract unspecified) |
| Extension fields (dependency) | `ExtField`, Frobenius, `Fold` | FpExt4<fp32>, Ext2<fp64> | Inversion smoke tests | `field_arith` (C) | Coordinatewise tokens | Field laws, base embedding, Frobenius order and multiplicativity | Zero inverse | µs | Restating the defining polynomials (deliberately avoided) |
| Ring arithmetic, CRT+NTT, SIMD | `CyclotomicCrtNtt::{from_ring, from_ring_pair, from_i8_*, from_centered_i32_*, to_ring(_cyclic), add_assign_pointwise_mul, mat_vec_i16}`, `mat_vec_i16_with_tail`, `DigitMontLut`, `CenteredMontLut`, ring `sigma`, shifts, sparse multiply | q32/q64/q128 prime sets, D = 64…2048 (1024 for q128); NEON/AVX2/IFMA dispatch vs scalar via `AKITA_SCALAR_NTT` lane variant | Fixed-vector SIMD-vs-scalar and CRT-vs-schoolbook unit tests | `ring_ntt@simd`, `ring_ntt@scalar` (C, R) | Full ring elements from pattern+edit tables; digits at ±bound; centered i32 up to 2^20 with LUT misses; mat-vec up to 3×6 | Schoolbook negacyclic `Mul`, harness cyclic schoolbook, transform equalities | Exactness only where `CrtCapacity::supports` admits the width/bound (the predicate production uses) | ms (D=2048 schoolbook dominates) | Products beyond CRT capacity (not produced by production); `mat_vec_i16` with ≥ 4 rows |
| Gadget decomposition | `BalancedDecomposePow2Params`, `balanced_decompose_coefficients_pow2_i8_into` (NEON/AVX2/u64/generic), `balanced_decompose_pow2_i16_into` | all three fields; bases from `opening_basis_range ∪ inner_basis_range` | Fixed boundary tests | `decompose` (C) | Coefficients across the full asymmetric representable range `[−(b/2)·Σb^l, (b/2−1)·Σb^l]`; exact-width full field; depths capped at the parameter limit | Σ d·b^l recomposition, digit range, i8 == i16 path | Centering threshold at exact width; lengths ≡ 0 mod 8 for SIMD bulk and tails | µs | `log_basis` < 3 (see FINDINGS F-2) |
| Multilinear, eq, bases, folding | `EqPolynomial::{evals, evals_serial, evals_parallel, evals_with_scaling, evals_mapped, evals_cached, evals_prefix, prefix_sum, mle}`, `SplitEqEvals`, `GruenSplitEq`, `lagrange/monomial/basis_weights(_prefix)`, `multilinear_eval`, `fold_evals_in_place` | base and extension opening fields | Seeded unit tests | `multilinear` (C) | Points with Boolean / non-Boolean coordinates, nv ≤ 12 | Per-index product weights (`oracle::index_weight`), direct folds | Empty and over-capacity prefixes, odd fold lengths | ≤ 10 ms | nv > 12 (covered by end-to-end cases) |
| Sumcheck drivers | `prove_sumcheck_native`, `verify_sumcheck_native`, `NativeSumcheckShape`, compression | fp128, FpExt4, Ext2 | Six unit tests, golden vector | `sumcheck_roundtrip` (C, S), `sumcheck_rounds` (R, original) | Product of 1–4 multilinear tables, ≤ 10 rounds | Harness-independent product evaluation at the Fiat-Shamir point | False claim, flipped byte, truncation (all must be `InvalidProof`) | ms | Akita relation kernels (reached only through end-to-end targets) |
| Transcript | native prover/verifier states, public bytes/fields, send/receive field/extension/bytes, field and extension challenges, EOF | blake2b backend | Known-answer and EOF unit tests | `transcript_roundtrip` (C, S), `transcript_labels` (R, original) | Up to 24 operations, arbitrary sessions/instances/sites | Replayed values and challenges equal; session separates the first challenge. The byte-change check only detects non-injective decoding | Empty proofs; bounded labels | µs | keccak backend (not the default feature) |
| Public deserialization | `AkitaDeserialize` for `CommittedGroup` (3 fields), `AkitaVerifierSetup`, `AkitaExpandedSetup`, setup descriptor/seed, instance descriptor, selection, field vectors | – | Round-trip unit tests | `public_deserialize` (R) | Honest encodings as seeds | Accepted ⇒ canonical re-encoding and decode(encode) identity | Length prefixes under `-malloc_limit_mb` | ms | Setups larger than `max_len` (only malformed ones are fuzzed) |
| Schedule artifacts | `TrustedScheduleCatalog::from_artifact_bytes`, `to_artifact_bytes` | all 16 families | 15 decoder rejection tests | `schedule_artifact` (R) | Shipped artifacts as seeds | No panic. ("Admitted ⇒ canonical" is also asserted, but admission already requires canonical bytes, so this is robustness-only in practice) | Family and policy binding | 10–100 ms | – |
| Setup | `setup_prover`, `setup_verifier`, `setup_verifier_for_schedule` | per family | Unit tests | end-to-end targets (C) | Sized from the planned rows | Narrowed verifier setup accepts the honest proof | Prover setup capacity exceeded → `InvalidInput` (`prover_boundary`) | per family, once per process | Random setup capacities (large ones are supported but costly) |
| Dense commitments | `CpuBackend::import_source`/`commit` with `DensePoly` (i8 fast path and general path) | fp128, fp32, fp64 Dense | e2e at fixed sizes | `pcs_dense` (C) | Pattern + ≤ 63 edits per table: zero, constant, alternating, ramp, random, small i8/i16, bound edges, single nonzero, sparse, powers of two, and an all-i8 table broken by exactly one ±127/128/129 neighbor (the fast-path decision boundary) | Independent Lagrange/monomial evaluation; honest verify | i8 boundary (±127/128), q/2 neighbors | 50 ms–2 s | – |
| Bounded dense | same with `fp128::DenseBounded` | – | u64 fixture tests | `pcs_dense` (C), `prover_boundary` (S) | Values inside production's own `accepted_bounds` for the group's profile, `[−2^64, 2^64 − 1]`, with bias toward both (asymmetric) endpoints | as above | One coefficient just past either endpoint → `InvalidInput` at commit | as above | – |
| One-hot | `OneHotPoly` sources | fp128/fp32 OneHot, W2R2; one-hot sources under balanced-digit schedules | e2e at fixed sizes | `pcs_onehot`, `pcs_dense` (C), `prover_boundary` (S) | Absent chunks (`None`) vs selected position 0, last position, random, sparse | Weight sum over hot indices | Dense source under a one-hot schedule → rejected | 20 ms–1 s | Chunk sizes other than 256 (no shipped config) |
| Openings, both bases | `SelectedProverOpeningData`, `batched_prove`, `batched_verify` | all planned rows, Lagrange and monomial | e2e, soundness tests | all `pcs_*` | Points of Boolean/non-Boolean/mixed coordinates; arbitrary sessions | Honest verify must succeed; the selected row must be the planned row | – | – | – |
| Batching, precommitted, heterogeneous | scheduler with/without precommitted groups, explicit profiles, `import_commitment` | rows with precommitted groups (own, same-class explicit, cross-class imported) | Fixed e2e | `pcs_batch` (C), `pcs_reject` (S) | Independent or prefix-shared points per group | as above; committed profile equals the row's profile | Handle order/count mismatches (`prover_boundary`) | 0.2–2 s | Cross-class pairs other than fp128 one-hot ← dense/bounded (none shipped; planner excludes with a reason) |
| Multi-chunk | same, W8R2/W2R2 | fp128 dense 16:1, one-hot 16:1, W2R2 14:1 | ignored e2e | `pcs_dense`, `pcs_onehot`, `pcs_batch` | as above | as above | – | ~1 s | W4R2 and 32/34-variable rows (cost) |
| Recursive setup offloading | recursive catalogs, setup-prefix slots, Stage 3 | fp128/fp32 dense recursive 20:1 | ignored e2e | `pcs_recursive` (C) | as above | as above | – | seconds | All one-hot recursive rows and fp64 (≥ 2^21 coefficients) |
| Binding (and some soundness) | `batched_verify` | every direct planned row | 5 fixed offsets, claim +1, session | `pcs_reject` (S, mostly binding) | Honest baseline first, then one of: evaluation + nonzero delta (false), point change (false or still-true, counted separately), session change, proof byte XOR, truncation, append, opposite basis, commitment to a different polynomial whose evaluation differs (false), another row's selection | Rejection with `InvalidProof` (statement-shape errors also allowed for selection changes). See the binding caveat below | – | ≈ one proof | – |
| Verifier no-panic contract | `batched_verify`, `GroupBatchStatement::new` | every planned row ≤ 2^20, including recursive setup offloading (setup-prefix commitments, Stage 3) | mixed-dimension rejection tests | `verifier_boundary` (R, S) | Honest proofs as seeds; arbitrary bytes, spliced patches, truncation; malformed statement shapes; random row digests | Only the byte-identical honest proof may verify; malformed bytes must be `InvalidProof`; malformed statements may fail with any class except `InvalidSetup` | EOF, noncanonical atoms | ms | – |
| Prover public boundary | `DensePoly::from_field_evals`, `OneHotPoly::new`, `import_source`, `commit`, selection, `batched_prove` | as above | Some unit tests | `prover_boundary` (R, S) | False claims, wrong point dimension, missing/extra/reordered handles, out-of-bound bounded values, dense under one-hot, uncataloged shapes within setup capacity, capacity overflow, bad constructors | Must fail before a proof exists, with the error classes each boundary documents (point-dimension and handle checks also accept `InvalidProof`, FINDINGS F-3) | – | ≈ one proof | – |
| Parallelism and determinism | Rayon paths in commit/prove/verify; shared `CpuBackend` | direct rows ≤ 2^17 | two-thread ownership test | `pcs_parallel` (C, M) | 1–8 thread scoped pools; two concurrent proofs | Byte-identical proofs | – | 2–3 proofs | Sequential feature graph: build with `prepare --sequential` |
| Resource behavior | all | all | – | every target (M) | – | libFuzzer timeout, RSS and malloc limits; cgroup `MemoryMax` when available | – | – | Hard per-input CPU limits |

## What the checks establish, and what they do not

- A passing valid-input target shows that the sampled statements of the
  planned rows prove and verify; it does not show that every admissible
  table, point, or row does. Arithmetic bugs that depend on specific values
  without new control flow are hit only as likely as the generators make
  them; boundary-biased tokens and value profiling (primitive targets) raise
  that likelihood but give no guarantee.
- **`pcs_reject` mostly tests binding, not algebraic soundness.** The
  verifier absorbs commitments, points, evaluations, basis, and session into
  the Fiat-Shamir transcript before any algebraic check, so changing the
  statement (false evaluation, point, session, basis, commitment) changes the
  challenges and the proof fails at its first check. A verifier whose
  algebraic checks were missing or wrong would still reject those mutations.
  Only proof-byte changes (`pcs_reject` ProofByte, `verifier_boundary`
  splices) and the sumcheck false-claim/tamper cases reach the algebraic
  checks, and none of them is a *cheating prover* that builds a consistent
  transcript for a false statement. No target does that; it would need a
  malicious prover implementation, which is out of scope. Rejection of the
  constructed false statements therefore shows binding and some algebraic
  checking, not soundness in general.
- A proof-byte change rejected with `InvalidProof` is evidence of canonical
  encoding and binding, not of knowledge soundness.
- Extension-field multiplication is checked by field laws only; a wrong but
  consistent defining polynomial would pass them, and the end-to-end targets
  would not notice either because prover and verifier share it.
- Timeouts show only that no sampled input exceeded the per-input limit.
- The recursive and multi-chunk families, fp64 one-hot, and every row above
  the caps run only at their smallest shapes or not at all (table above).
- Kernels whose selection depends on the CPU are exercised only for the
  machine the campaign runs on (for example AVX-512 IFMA paths only on CPUs
  that have them). `ring_ntt@scalar` forces the scalar NTT on any CPU.
- The workspace default transcript backend (blake2b) is built; keccak is not.
