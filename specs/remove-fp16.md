# Spec: Remove `fp16` and Q16 infrastructure

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao, Cursor assistant (GPT-5.2 draft, Claude Opus 4.8 revisions) |
| Created   | 2026-06-04 |
| Status    | implemented |
| PR        | https://github.com/LayerZero-Labs/akita/pull/149 |

## Summary

Remove all `fp16` code from the Akita Rust codebase, including the `Fp16` prime field implementation, its SIMD packing backends, and the Q16-specific protocol, SIS, and CRT/NTT tables and dispatch (the Q16 16-bit modulus family).
This is a full cutover, not a deprecation.
The generic `i16` NTT prime width is retained as reusable infrastructure (see Non-Goals); only the Q16 family that currently consumes it is removed.
Specs may continue to mention 16-bit fields as historical context, but the executable codebase must not.

The immediate motivation is PR 146, which corrected major security underestimates in weak-binding norm pricing and challenge sizing, and made Q16 schedules infeasible for meaningful instances.

## Intent

### Goal

Fully remove:

- The `akita-field` `Fp16` field family (including `Prime16Offset99`) and its fp16-specific optimizations (SIMD packing, wide accumulators, and extension-field specializations).
- The Q16 SIS family (`SisModulusFamily::Q16`) and all Q16 floor table rows and wire-format tags.
- Q16-specific CRT/NTT tables, dispatch, and tests (the Q16 prime tables and Garner constants, and the Q16 enum variants). The generic `i16` NTT prime width is **retained**, not removed (see the `akita-algebra` notes and Non-Goals below).

The result is that Akita only supports the shipped and security-viable modulus families (Q32/Q64/Q128) in code, and any proof artifacts that attempt to use Q16 are rejected at the verifier boundary with a structured error, not a panic.

### Invariants

- **No backward compatibility shims.**
  All call sites are updated in one pass.
  There is no `fp16` module kept around for old code paths.

- **Verifier no-panic contract is preserved.**
  Any removed wire-format tags or malformed inputs must be rejected with `AkitaError` or `SerializationError`, not by panicking.

- **Instance descriptor decoding is strict.**
  Removing `SisModulusFamily::Q16` means the descriptor tag previously used for Q16 must be rejected explicitly.
  Q16 must not be treated as an unknown-but-ignored case.

- **No behavior changes for supported families.**
  Removing fp16 and Q16 must not change arithmetic, transcript binding, or schedule behavior for Q32/Q64/Q128.

- **Surviving wire tags are preserved.**
  The `SisModulusFamily` tag values for the kept families stay unchanged: `Q32 = 0`, `Q64 = 1`, `Q128 = 2` in `crates/akita-types/src/descriptor_bytes.rs` (`sis_family_tag`) and the matching decoder in `crates/akita-types/src/instance_descriptor.rs`.
  Only the Q16 arms are deleted, so transcript instance-descriptor bytes for supported families do not move.

### Non-Goals

- This spec does not introduce a replacement “small field” family.
- This spec does not require deleting historical `specs/*` documents that mention fp16.
- This spec does not attempt to preserve the ability to verify old Q16 proofs.
  The repo makes no compatibility guarantees, and this is an intentional break.
- This spec does not remove the generic `RingSubfieldFp8` extension type.
  Only the `RingSubfieldFp8<Fp16>` specialization and its fp16-only optimized paths are removed.
  `RingSubfieldFp8<Fp32>`, `RingSubfieldFp8<Fp64>`, and `RingSubfieldFp8<Fp128>` remain as generic extension infrastructure.
- This spec does not remove the generic `i16` NTT prime width.
  `PrimeWidth`'s `i16` arm and the i16 butterfly/Montgomery/SIMD kernels stay as reusable infrastructure: a future schedule could route Q32/Q64 through several small `i16` NTT primes instead of `i32` primes, and choosing `i32` primes today does not foreclose that.
  Only the Q16-specific concrete tables and dispatch are removed.

## Evaluation

### Acceptance Criteria

- [x] `Fp16` is not present in the Rust codebase outside `specs/`.
- [x] `Prime16Offset99` is not present in the Rust codebase outside `specs/`.
- [x] `Fp16Packing`, `PackedFp16Neon`, `PackedFp16Avx2`, and `PackedFp16Avx512` are not present in the Rust codebase outside `specs/`.
- [x] `FoldMatrixFp16` and `RingSubfieldFp8Fp16ProductAccum` are not present in the Rust codebase outside `specs/`.
- [x] `SisModulusFamily::Q16` does not exist, and there are no Q16 SIS floor rows shipped in code.
- [x] No shipped or profile preset resolves to `SisModulusFamily::Q16`; the only families selected are Q32/Q64/Q128.
- [x] The Q16-specific NTT tables (`Q16_PRIMES`, `Q16_NUM_PRIMES`, `Q16_MODULUS`, `q16_garner`) and the `ProtocolCrtNttParams::Q16` / `NttSlotCache::Q16` dispatch variants are removed, while the generic `i16` `PrimeWidth` implementation and its NTT/SIMD kernels remain and still build under `-D warnings`.
- [x] The instance descriptor decoding rejects the removed Q16 tag (historically tag `3`) with `SerializationError::InvalidData(...)`, while tags `0`/`1`/`2` still decode to Q32/Q64/Q128 unchanged.
- [x] `cargo fmt -q` is clean.
- [x] `cargo clippy --all --message-format=short -q -- -D warnings` is clean.
- [x] `cargo test` passes.

### Testing Strategy

- Run the standard workspace commands:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

- Audit that fp16 does not appear in code (excluding `specs/`):

```bash
rg -n --glob '!specs/**' 'Fp16|Prime16Offset99|fp16\b|Fp16Packing|FoldMatrixFp16|RingSubfieldFp8Fp16|SisModulusFamily::Q16|\bQ16_PRIMES\b|\bQ16_NUM_PRIMES\b|\bQ16_MODULUS\b|\bq16_garner\b'
```

- Do not audit for bare `i16`.
  The generic `i16` NTT prime width is intentionally retained, and both `akita-algebra` and `akita-prover` keep unrelated `i16` uses; the audit targets only Q16-named symbols.

- Add a targeted unit test for strict descriptor decoding.
  The test should feed the historical Q16 family tag (`3`) into the descriptor decoding path and assert it returns `SerializationError::InvalidData(...)`.

### Performance

No performance regressions are expected for supported families because the removed code paths are not used in shipped or profile presets after PR 146.
This change removes dead code and does not change hot loops for Q32/Q64/Q128.

## Design

### Architecture

This is a full deletion across multiple crates, with the main ownership boundaries below.

- `akita-field`
  - Removes `Fp16` and all fp16-only optimizations and packings.
  - Removes `Prime16Offset99` from the pseudo-Mersenne registry.
  - Removes any fp16 specialization of extension fields (notably `RingSubfieldFp8<Fp16>` paths).
  - Keeps the generic `RingSubfieldFp8` extension type and its supported non-fp16 base-field instantiations.

- `akita-algebra`
  - Removes the Q16-specific CRT/NTT tables and Garner constants (`Q16_PRIMES`, `Q16_NUM_PRIMES`, `Q16_MODULUS`, `q16_garner` in `crates/akita-algebra/src/ntt/tables.rs`) and the Q16 prime-table tests.
  - Keeps the generic `i16` NTT prime width.
    `PrimeWidth`'s `i16` implementation (`crates/akita-algebra/src/ntt/prime.rs`), the i16 butterfly/Montgomery/NEON/AVX kernels (`ntt/neon.rs`, `ntt/butterfly.rs`, `ntt/avx/*`), and the `size_of::<W>() == size_of::<i16>()` dispatch in `ring/crt_ntt_repr/ops.rs` stay in place.
    They lose their only shipped consumer once Q16 is gone, but they remain referenced by the generic NTT dispatch and exercised by their synthetic-prime unit tests (e.g. `ntt/avx/tests.rs` builds primes via `NttPrime::compute(15361_i16)`, not `Q16_PRIMES`), so they continue to build cleanly under `-D warnings`.

- `akita-prover`
  - Removes the Q16 CRT/NTT protocol parameter variant (`ProtocolCrtNttParams::Q16`) and the `NttSlotCache::Q16` cache variant in `crates/akita-prover/src/kernels/crt_ntt.rs`, plus any Q16 kernel dispatch.
  - Removes Q16-specific tests and capacity profiles.

- `akita-types`
  - Removes `SisModulusFamily::Q16` and every match arm over it, including the `ceil_supported_collision` `(Q16, d)` rows in `crates/akita-types/src/sis/ajtai_key.rs`.
  - Removes the Q16 rows from the generated SIS floor table `crates/akita-types/src/sis/generated_sis_table.rs`.
    That table is generated by `sage -python scripts/gen_sis_table.py --family <q16|q32|q64|q128>` (lattice-estimator).
    The Q16 block is deleted by hand: the per-family invocations are independent, so the estimator is not re-run and the Q32/Q64/Q128 rows must stay byte-identical.
    The file header's generation comment drops `q16` from the family loop.
  - Updates descriptor encoding (`sis_family_tag`) and decoding so the removed Q16 tag `3` is rejected, while `Q32 = 0`, `Q64 = 1`, `Q128 = 2` stay unchanged.

- `akita-config`
  - Removes the `SisModulusFamily::Q16` arm from the small-field capacity test (`crates/akita-config/src/proof_optimized/tests.rs`).
    No preset selects Q16, so this arm is unreachable today and must be deleted once the enum variant is gone.

- `akita-pcs`
  - Removes the fp16/Q16 references in benches (`crates/akita-pcs/benches/field_arith/*`) and integration tests (`crates/akita-pcs/tests/algebra/ntt_crt.rs`).

### Alternatives Considered

- **Keep Q16 but delete only `Fp16`.**
  Rejected.
  Q16 exists primarily to support `q <= 2^16` schedules, which are infeasible under the corrected security accounting in PR 146.
  Keeping Q16 would retain dead and misleading protocol branches, including wire-format handling and table rows, with no supported producer.

- **Leave Q16 tag decoding as “unknown”.**
  Rejected.
  The verifier boundary must be strict and explicit.
  Q16 artifacts must be rejected deterministically with an error.

- **Also remove the generic `i16` NTT prime width.**
  Rejected.
  The `i16` width is generic infrastructure parametrized by an arbitrary 16-bit NTT prime, not Q16-specific.
  A future schedule could route Q32/Q64 through several small `i16` primes instead of `i32` primes, so the width is kept even though no shipped family consumes it today.
  It stays compiled and tested via the generic dispatch and the synthetic-prime SIMD unit tests, so retaining it does not leave warned-on dead code.

## Documentation

- Add this spec.
- No other documentation updates are required by this spec.
  Non-spec docs and helper scripts that advertise fp16/Q16 modes (`docs/compute-backend-baselines.md`, `docs/crt-ntt-capacity-profile.md`, `scripts/gen_crt_capacity_profile.py`) are updated in the implementation PR as mechanical cleanup, but they are not required for spec approval.

## Execution

Suggested implementation order, to keep the branch easy to bisect:

1. Delete fp16 consumers (tests, benches, examples) that reference `Fp16` or `Prime16Offset99`.
2. Delete `akita-field` fp16 implementation and exports, including SIMD packing and extension-field specializations.
3. Delete the Q16-specific CRT/NTT tables and dispatch in `akita-algebra` and `akita-prover`, keeping the generic `i16` prime width.
4. Remove `SisModulusFamily::Q16` plus all descriptor-tag handling and SIS table rows in `akita-types`, and the dependent `akita-config` test arm.
5. Run the acceptance commands and `rg` audit.

Risks to watch:

- Verifier-reachable tag decoding must return `SerializationError`, not panic.
- Removing `SisModulusFamily::Q16` is a wire-format break, so tests must assert explicit rejection, not silent acceptance.
- The retained `i16` NTT kernels lose their only shipped consumer (Q16).
  Confirm they stay referenced by the generic dispatch and their synthetic-prime unit tests so `-D warnings` does not flag them as dead.
  If removing `Q16_PRIMES` orphans an i16 test helper (e.g. `assert_i16_prime_profile`), keep a synthetic-prime test rather than deleting the helper.

## References

- PR 146 (security correction and A-role reprice): `https://github.com/LayerZero-Labs/akita/pull/146`
- `specs/weak-binding-norm-fix.md` (context and derivation history)
- `crates/akita-types/src/sis/norm_bound.rs` (committed-fold collision pricing implementation)
