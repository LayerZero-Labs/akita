# Spec: Remove `fp16` and Q16 infrastructure

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao, Cursor assistant (model: GPT-5.2) |
| Created   | 2026-06-04 |
| Status    | proposed |
| PR        | (fill on merge) |

## Summary

Remove all `fp16` code from the Akita Rust codebase, including the `Fp16` prime field implementation, its SIMD packing backends, and all protocol, SIS, and CRT/NTT infrastructure that exists to support the Q16 (16-bit modulus) family.
This is a full cutover, not a deprecation.
Specs may continue to mention 16-bit fields as historical context, but the executable codebase must not.

The immediate motivation is PR 146, which corrected major security underestimates in weak-binding norm pricing and challenge sizing, and made Q16 schedules infeasible for meaningful instances.

## Intent

### Goal

Fully remove:

- The `akita-field` `Fp16` field family (including `Prime16Offset99`) and its fp16-specific optimizations (SIMD packing, wide accumulators, and extension-field specializations).
- The Q16 SIS family (`SisModulusFamily::Q16`) and all Q16 floor table rows and wire-format tags.
- Q16 CRT/NTT tables, dispatch, and tests.

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

### Non-Goals

- This spec does not introduce a replacement “small field” family.
- This spec does not require deleting historical `specs/*` documents that mention fp16.
- This spec does not attempt to preserve the ability to verify old Q16 proofs.
  The repo makes no compatibility guarantees, and this is an intentional break.

## Evaluation

### Acceptance Criteria

- [ ] `Fp16` is not present in the Rust codebase outside `specs/`.
- [ ] `Prime16Offset99` is not present in the Rust codebase outside `specs/`.
- [ ] `Fp16Packing`, `PackedFp16Neon`, `PackedFp16Avx2`, and `PackedFp16Avx512` are not present in the Rust codebase outside `specs/`.
- [ ] `FoldMatrixFp16` and `RingSubfieldFp8Fp16ProductAccum` are not present in the Rust codebase outside `specs/`.
- [ ] `SisModulusFamily::Q16` does not exist, and there are no Q16 SIS floor rows shipped in code.
- [ ] The instance descriptor decoding rejects the removed Q16 tag (historically tag `3`) with `SerializationError::InvalidData(...)`.
- [ ] `cargo fmt -q` is clean.
- [ ] `cargo clippy --all --message-format=short -q -- -D warnings` is clean.
- [ ] `cargo test` passes.

### Testing Strategy

- Run the standard workspace commands:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

- Audit that fp16 does not appear in code (excluding `specs/`):

```bash
rg -n --glob '!specs/**' 'Fp16|Prime16Offset99|fp16\\b|Fp16Packing|FoldMatrixFp16|RingSubfieldFp8Fp16|SisModulusFamily::Q16|\\bQ16_PRIMES\\b|\\bQ16_NUM_PRIMES\\b|\\bQ16_MODULUS\\b'
```

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

- `akita-algebra`
  - Removes Q16 CRT/NTT tables and any Q16 Garner constants used for i16 CRT profiles.

- `akita-prover`
  - Removes Q16 CRT/NTT protocol parameter variants and any Q16 kernel dispatch.
  - Removes Q16-specific tests and capacity profiles.

- `akita-types`
  - Removes `SisModulusFamily::Q16`.
  - Removes the Q16 rows from generated SIS floor tables.
  - Updates descriptor encoding and decoding so the removed Q16 tag is rejected.

### Alternatives Considered

- **Keep Q16 but delete only `Fp16`.**
  Rejected.
  Q16 exists primarily to support `q <= 2^16` schedules, which are infeasible under the corrected security accounting in PR 146.
  Keeping Q16 would retain dead and misleading protocol branches, including wire-format handling and table rows, with no supported producer.

- **Leave Q16 tag decoding as “unknown”.**
  Rejected.
  The verifier boundary must be strict and explicit.
  Q16 artifacts must be rejected deterministically with an error.

## Documentation

- Add this spec.
- No other documentation updates are required by this spec.
  If there are non-spec docs or benchmark artifacts that advertise fp16 modes, they should be updated in the implementation PR as mechanical cleanup, but they are not required for spec approval.

## Execution

Suggested implementation order, to keep the branch easy to bisect:

1. Delete fp16 consumers (tests, benches, examples) that reference `Fp16` or `Prime16Offset99`.
2. Delete `akita-field` fp16 implementation and exports, including SIMD packing and extension-field specializations.
3. Delete Q16 CRT/NTT tables and dispatch in `akita-algebra` and `akita-prover`.
4. Remove `SisModulusFamily::Q16` plus all descriptor-tag handling and SIS table rows in `akita-types`.
5. Run the acceptance commands and `rg` audit.

Risks to watch:

- Verifier-reachable tag decoding must return `SerializationError`, not panic.
- Removing `SisModulusFamily::Q16` is a wire-format break, so tests must assert explicit rejection, not silent acceptance.

## References

- PR 146 (security correction and A-role reprice): `https://github.com/LayerZero-Labs/akita/pull/146`
- `specs/weak-binding-norm-fix.md` (context and derivation history)
- `crates/akita-types/src/sis/norm_bound.rs` (committed-fold collision pricing implementation)
