# Spec: Production fp16 Small-Field Support

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-17 |
| Status | implemented |
| PR | #86 (`quang/akita-fp16`) |

## Summary

Add first-class fp16 support to Akita, matching the level of integration currently available for fp32 and fp64:

- concrete prime field for `q = 65437 = 2^16 - 99`, the largest 16-bit prime congruent to `5 mod 8`;
- degree-8 challenge/opening extension field in Akita's canonical cyclotomic ring-subfield basis, so the Fiat-Shamir/proof scalar width stays near 128 bits;
- correct SIS floors for the actual fp16 modulus;
- generated schedule tables for fp16 presets;
- config presets, profile modes, setup/prover/verifier coverage, and proof-size accounting;
- performance work sufficient for fp16 to be a credible production path, not only a planner experiment.

The initial proof-size target is the D32 schedule family, because generated schedule/profile runs show D32 is best for the canonical workloads once the module-rank search is allowed to go high enough.

This spec deliberately excludes smaller base-field support such as fp8/fp5/fp4. In particular, there is no 8-bit base-field `K = 8` path.

## Intent

### Goal

Make fp16 a complete Akita field family:

- `Field = Prime16Offset99`;
- `ExtensionField = RingSubfieldFp8<Field>` or an exactly equivalent same-width degree-8 cyclotomic ring-subfield extension with the canonical basis and multiplication law specified below;
- production schedule generation using the actual fp16 SIS floor;
- end-to-end `commit`, `prove`, and `verify` support for dense and one-hot workloads;
- generated-schedule proof-size accounting that matches serialized proofs;
- profile ergonomics comparable to the existing fp32/fp64 modes.

The result should let us answer, with implementation-backed numbers, whether fp16 is worth pursuing beyond planner estimates.

### Invariants

- The base field prime is fixed to `65437 = 2^16 - 99`.
- The extension degree is `K = 8`, giving 8 fp16 limbs per challenge-field element.
- The degree-8 extension is the canonical `K = 8` cyclotomic ring subfield used by `RingSubfieldEncoding`, not an arbitrary tower or power-basis `Fp8`.
- Serialized fp16 base-field elements must be exactly 2 canonical little-endian bytes. This is a hard requirement for the expected proof-size win.
- Serialized fp16 degree-8 extension elements must be exactly 16 bytes in canonical ring-subfield limb order.
- Serialization, transcript absorption, `RingSubfieldEncoding`, tensor embedding, and proof-size accounting must all use the same canonical limb order.
- fp16 introduces no new transcript labels or transcript ordering. Prover and verifier must use the existing extension-limb append/sample scheme in increasing limb order `0..7`.
- SIS security must be computed for the actual `q = 65437` modulus, not approximated by the fp32 family.
- Q16 SIS floor generation must record the external estimator command, estimator version or commit, modulus, security target, model, and sweep bounds so checked-in rows can be independently regenerated or spot-checked.
- The prover and verifier must use the same proof protocol shape as the current small-field extension-opening path.
- Planner proof-size estimates must be checked against real `AkitaSerialize::serialized_size()` on generated proofs.
- Disk-persisted setup/cache artifacts must be keyed so fp16 cannot reuse fp32/fp64 artifacts and stale incompatible caches are rejected or regenerated.
- Existing fp32 and fp64 behavior must remain covered by tests and profiles.
- The implementation should not entrench the current planner confusion between root claims and opening points. If this spec does not complete the incidence-generalization cutover, it must clearly isolate any temporary alignment with the current runtime layout.

### Non-Goals

- No 8-bit, 5-bit, or 4-bit base-field production support.
- No 8-bit base-field `K = 8` experiment.
- No arbitrary dynamic prime configuration.
- No new PCS protocol.
- No backwards-compatibility shim for older serialized fp16 experiments.
- No mixed-ring-dimension schedule execution in this phase. Mixed-D planning can be handled in a later spec once the single-family fp16 path is real.
- No true tower `F < E < L` work unless the fp16 same-width `E = L` route exposes a correctness blocker.
- No transcript domain-separation rework. The existing small-field path initializes transcripts with a caller-supplied domain label and does not absorb a setup digest. fp16 inherits that posture unchanged. Cross-family soundness isolation by setup digest is a separate follow-up that should not expand under this spec.

## Evaluation

### Acceptance Criteria

1. Field support
   - `Prime16Offset99` exists as a real 16-bit field type, not merely a low-modulus `Fp32` alias with 4-byte serialization.
   - `Prime16Offset99::MODULUS == 65437`.
   - `Prime16Offset99::BITS == 16`.
   - `AkitaSerialize::serialized_size()` for one base-field element is 2 bytes.
   - Base-field serialization is exactly two little-endian bytes for the canonical representative `0 <= x < 65437`.
   - Validated deserialization rejects all 2-byte encodings in `[65437, 65535]`.
   - Arithmetic, inversion, batch inversion, sampling, serialization, validation, and canonical conversion tests pass.

2. Extension support
   - A degree-8 extension type exists for fp16, provisionally `RingSubfieldFp8<Prime16Offset99>`.
   - The extension uses the canonical Akita ring-subfield basis `[1, e1, ..., e7]` and multiplication law from the design section.
   - It implements the same field/extension traits used by fp32 `RingSubfieldFp4` and fp64 `Ext2`.
   - `RingSubfieldEncoding` supports degree 8 for the concrete fp16 extension.
   - `RingSubfieldEncoding::to_ring_subfield_coords()` returns the canonical `[c0, ..., c7]` coordinate order.
   - Base-limb embedding and recovery round-trip for `K = 8`.
   - Serialized extension elements are 16 bytes.
   - Extension serialization is exactly eight consecutive canonical fp16 limbs in `[c0, ..., c7]` order, with no alternate compressed form.
   - K=8 multiplication is compatible with `embed_subfield`: `embed_subfield::<F, D, 8>((x * y).coeffs) == embed_subfield::<F, D, 8>(x.coeffs) * embed_subfield::<F, D, 8>(y.coeffs)` for supported `D`.
   - Root tensor projection and extension-opening reduction work for D32 and D64 at minimum.

3. SIS floors
   - A generated SIS family exists for fp16, preferably named `Q16` or `Q16Offset99`.
   - Floors are generated by the lattice estimator for `q = 65437`.
   - The checked-in floor table or generator documentation records the exact estimator command, estimator version or git commit, estimator model, security target, modulus, and sweep bounds.
   - At least one checked-in or documented spot-check lets reviewers reproduce selected Q16 rows independently of schedule generation.
   - The sweep covers at least:
     - `D = 32`, ranks through 20;
     - `D = 64, 128, 256, 512`, ranks through at least 12, with a documented reason if any bound is raised or lowered.
   - The in-tree floor table at `crates/akita-types/src/generated/sis_floor.rs` is reshaped from the current `[u64; MAX_RANK]` row to a per-cell `&'static [u64]` so each `(family, D, collision_inf)` cell carries only the rank cap it actually needs. The global `MAX_RANK` constant is removed.
   - Existing Q32/Q64/Q128 floor values are preserved as prefixes after the reshape. Q32/D32 is then explicitly extended through rank 20 to support the fp32 D32 default schedule; Q64/Q128 rank caps remain unchanged unless a future measured schedule requires more rows.
   - The planner rejects configs without a valid fp16 SIS floor instead of silently falling back to fp32/fp64 floors.

4. Planner and generated schedules
   - `gen_schedule_tables` can emit fp16 full and one-hot schedule tables.
   - Generated fp16 schedule modules are wired into `akita-types/src/generated`.
   - `akita-config` exposes fp16 presets comparable to fp32/fp64.
   - Singleton and batched schedule keys are generated for dense and one-hot shapes.
   - fp16 schedule tables are generated against the current `AkitaScheduleLookupKey`
     shape. Regeneration after any future root-profile cutover in
     `specs/planner-incidence-generalization.md` is the same `gen_schedule_tables`
     invocation and is treated as expected churn, not a blocker for fp16.
   - The fp16 generator MUST NOT introduce any code path that derives `num_w_vectors` from `num_claims` or from the extension degree `K`. Singleton root keys continue to set `num_w_vectors = 1`.
   - D32 one-hot and dense profiles assert that real serialized proof bytes match the generated schedule plan exactly. Any divergence is documented as an intentional estimator/table change.

5. Prover/verifier integration
   - Existing generic prover and verifier paths accept the fp16 config without protocol-specific forks.
   - Dense and one-hot proofs verify for small debug instances.
   - Release-profile examples run for canonical planner workloads.
   - The root extension-opening reduction emits `K` partials for a singleton root opening, not `K * num_w_vectors` caused by misinterpreting `w`.
   - Transcript append/sample tests show fp16 uses the existing `append_ext_field` and `sample_ext_challenge` limb-label scheme in canonical limb order `0..7`, with no new labels or prover/verifier ordering changes.
   - With `disk-persistence` enabled, setup-cache tests show fp16 cache filenames/keys are distinct from fp32/fp64 and stale incompatible cache files are rejected or regenerated.

6. Proof-size accountability
   - Planner-reported proof size equals actual `AkitaSerialize::serialized_size()` byte-for-byte for representative fp16 profiles. The planner accounting includes every byte the serializer emits, including per-`Vec` length prefixes; there is no documented per-byte tolerance.
   - The planner-side root EOR partial count in `crates/akita-types/src/layout/proof_size.rs::extension_opening_reduction_proof_bytes` and the prover-side `prepare_root_extension_opening_reduction` derive `partials` from a single shared helper, e.g. `root_extension_opening_partials(claim_ext_degree, num_points)`. This is the structural form of acceptance criterion 5: the equality cannot drift because both sides read the same function.
   - The proof-size report breaks down root, per-level, tail, and witness components, matching the style of `profile.rs`.
   - The profile output names the field family, extension degree, ring dimension, module ranks, and SIS family used by every level.
   - The same planner-side accounting (helper, length-prefix discipline, byte equality) is verified for the existing fp32 D32 onehot nv32 and fp32 D32 dense singleton nv26 profiles before any fp16 floor lands, so the structural fix is checked against a known-good baseline.

7. Performance
   - fp16 profiles are not accidentally dominated by avoidable widening, heap churn, or generic extension arithmetic.
   - At minimum, D32 one-hot nv32 and dense singleton nv27 complete in release mode on the same machine used for fp32/fp64 comparisons.
   - Hot spots in base-field arithmetic, extension multiplication, root EOR, and witness serialization are profiled before declaring the path production-ready.

### Current Proof-Size Targets

These are implementation-backed profile results from generated fp16 schedule tables. The profile path serializes the proof, asserts `proof.size()` equals the actual uncompressed serialization length, and asserts `proof.size() == plan.exact_proof_bytes` whenever a generated plan is present. The removed planner dry-run binary is not a production proof-size oracle.

Reproducer:

```bash
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 AKITA_PROFILE_ANSI=0 \
AKITA_PROFILE_LOG=warn AKITA_MODE=onehot_fp16_d32 AKITA_NUM_VARS=32 \
  cargo run --release --example profile

AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 AKITA_PROFILE_ANSI=0 \
AKITA_PROFILE_LOG=warn AKITA_MODE=full_fp16_d32 AKITA_NUM_VARS=20 \
  cargo run --release --example profile
```

Measured proof sizes with tight 2-byte fp16 serialization:

| Workload | fp16 D32 proof bytes | Notes |
| --- | ---: | --- |
| onehot nv32 | 34,168 B | 6 folded levels, 13,848 B packed final witness |
| dense singleton nv20 | 28,664 B | 5 folded levels, 13,688 B packed final witness |

The generated schedule and profile accounting are the source of truth. If a generated schedule changes, the profile-reported byte totals and this table must be updated together.

The key caveat is serialization. If fp16 is implemented as `Fp32<65437>` with the current fixed-width `Fp32` serializer, the base-field elements still serialize to 4 bytes and these proof-size savings do not materialize.

### Testing Strategy

- Unit tests for `Prime16Offset99` arithmetic:
  - addition/subtraction wraparound near modulus;
  - multiplication and squaring near modulus;
  - inverse and batch inverse;
  - random sampling;
  - canonical byte serialization and validation rejects out-of-range encodings.
- Unit tests for `RingSubfieldFp8`:
  - canonical basis conversion for `[1, e1, ..., e7]`;
  - Chebyshev multiplication table spot checks, including `e1^2 = 2 + e2`, `e4^2 = 2`, and wraparound signs such as `e7^2 = 2 - e2`;
  - multiplication, squaring, inverse;
  - Frobenius behavior if required by existing generic bounds;
  - serialization size and round-trip;
  - embedding/recovery through `RingSubfieldEncoding`.
- Field-reduction tests:
  - `validate_ring_subfield_role` accepts fp16/D32/K8 and fp16/D64/K8;
  - `embed_subfield::<_, D, 8>` is multiplicative for the typed `RingSubfieldFp8` representation across representative supported ring dimensions;
  - typed `psi` trace inner-product identity holds for `K = 8`;
  - tensor projection round-trips for D32;
  - extension-opening reduction singleton emits 8 root partials.
- Transcript tests:
  - config-level fp16 claim-field appends match direct `append_ext_field` appends over 8 limbs;
  - config-level fp16 challenge sampling matches direct `sample_ext_challenge` sampling over 8 limb labels;
  - prover/verifier transcript ordering remains unchanged relative to the existing small-field extension-opening path.
- Planner tests:
  - fp16 SIS floor lookup succeeds for generated dimensions and ranks;
  - lookup failure is explicit for missing dimensions/ranks;
  - Q16 lookups do not fall back to Q32, Q64, or Q128;
  - generated schedule tables contain expected singleton and batched keys;
  - proof-size accounting uses 2 bytes per base limb and 16 bytes per challenge element.
- Setup/cache tests, under `disk-persistence`:
  - fp16 cache names include enough modulus/config/schedule/D/rank information to avoid fp32/fp64 reuse;
  - incompatible cached setup artifacts are rejected or regenerated.
- End-to-end tests:
  - dense small-nv prove/verify;
  - one-hot small-nv prove/verify;
  - batched same-point and distinct-point prove/verify where supported by existing incidence shape.
- Regression tests:
  - existing fp32 and fp64 generated schedules still compile and verify;
  - current fp32/fp64 proof-size snapshots do not move except where intentionally changed by shared planner fixes.

Required repo checks:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

Recommended focused checks while developing:

```bash
cargo test -p akita-field fp16
cargo test -p akita-types field_reduction
cargo test -p akita-planner fp16
cargo check -p akita-config --bin gen_schedule_tables
AKITA_MODE=onehot_fp16_d32 AKITA_NUM_VARS=32 cargo run --release --example profile
AKITA_MODE=full_fp16_d32 AKITA_NUM_VARS=27 cargo run --release --example profile
```

Exact mode names are subject to the profile-mode naming chosen during implementation.

### Performance

The implementation should avoid treating fp16 as a cosmetic alias over fp32.

Performance requirements:

- Base-field storage should be compact enough to reduce memory bandwidth and serialization size.
- Scalar multiplication should use a `u32` product path with cheap pseudo-Mersenne reduction for `2^16 - 99`.
- SIMD kernels may use `u16` lanes for storage/add/sub and widen to `u32` lanes for products. On NEON this likely means `vmull_u16`/`vmull_high_u16`; on AVX2 this likely means `mullo/mulhi` plus unpacking to `u32`; on AVX-512BW this likely means wider masked `u16`/`u32` lanes.
- Accumulation should use the existing wide/unreduced abstractions where they buy real throughput, but product-sum bounds must be documented before fusing many fp16 products into one reduction.
- Extension multiplication for `RingSubfieldFp8` must start with the canonical ring-subfield multiplication law, then receive targeted optimization based on profile data.
- The first optimized `RingSubfieldFp8` candidates should compare direct Chebyshev-structure multiplication against an internal tower/Karatsuba backend that converts to/from the canonical `[1, e1, ..., e7]` representation.
- Batched SIMD extension kernels may use a structure-of-arrays or transposed limb-major layout internally, but public values, serialization, transcript absorption, and `RingSubfieldEncoding` remain canonical `[c0, ..., c7]`.
- Serialization should bulk-pack fp16 limbs without per-element allocation.
- Generated schedules should prefer D32 where the planner says it is best, but keep D64+ presets available for comparison and regressions.

Optimization phases:

1. Scalar-correct fp16 base field and degree-8 extension.
2. End-to-end proof generation and verification.
3. Profile dense nv27 and one-hot nv32.
4. Optimize the observed hot paths.
5. Re-run fp16/fp32 comparison with real serialized proofs.

## Design

### Architecture

#### 1. Field Layer

Add a dedicated fp16 field implementation rather than aliasing `Fp32<P>`.

Candidate shape:

```rust
#[repr(transparent)]
pub struct Fp16<const P: u32>(u16);
pub type Prime16Offset99 = Fp16<65437>;
```

Use a `u32` const modulus even though the stored value is `u16`. This keeps compile-time arithmetic and trait implementations aligned with the existing `Fp32` style while preserving compact storage.

The stored representation invariant is canonical storage only: the inner `u16` is always in `[0, 65437)`.

Required trait surface should mirror the parts of `Fp32` consumed by the rest of Akita:

- core field traits;
- `CanonicalField`;
- pseudo-Mersenne metadata;
- serialization and validation;
- random generation;
- packed/wide helper traits used by polynomial and matrix code;
- extension-field base trait bounds.

Also update the pseudo-Mersenne registry with:

- bit width: 16;
- offset: 99;
- modulus: 65437;
- congruence: `5 mod 8`.

The registry update is only the metadata piece. It does not replace the need for a true 2-byte field representation.

#### 2. Serialization Contract

Base-field serialization is part of the protocol contract, not an optimization detail.

Canonical fp16 byte layout:

```text
Prime16Offset99(x) -> little_endian_u16(x)
where 0 <= x < 65437
```

Validated deserialization must reject every two-byte value `x >= 65437`. Unvalidated deserialization may reduce only if that matches the existing field deserialization convention, but validated proof/setup decoding must reject non-canonical encodings.

Canonical degree-8 extension byte layout:

```text
RingSubfieldFp8([c0, c1, c2, c3, c4, c5, c6, c7])
  -> ser_fp16(c0) || ser_fp16(c1) || ... || ser_fp16(c7)
```

There is no length prefix or alternate compressed form at the extension-element level. Any internal SIMD, tower, or transposed representation must convert to this canonical limb order before serialization, transcript absorption, proof-size accounting, or `RingSubfieldEncoding`.

#### 3. Degree-8 Extension Layer

Add `RingSubfieldFp8<F>` or an equivalent degree-8 ring-subfield extension.

This should follow the fp32 `RingSubfieldFp4` pattern, but with `K = 8`. It is not an arbitrary degree-8 extension field. It is the `K = 8` Akita cyclotomic ring subfield whose coordinates are consumed by `psi_embed`, `embed_subfield`, and `RingSubfieldEncoding`.

Canonical basis:

```text
[1, e1, e2, e3, e4, e5, e6, e7]
```

For any valid ring dimension `D` and `K = 8`:

```text
step = D / 16
e_j = X^(j * step) + X^(-j * step) in F[X] / (X^D + 1)
```

A coordinate vector `[c0, ..., c7]` denotes:

```text
c0 + c1*e1 + c2*e2 + c3*e3 + c4*e4 + c5*e5 + c6*e6 + c7*e7
```

Multiplication must use the Chebyshev/cyclotomic ring-subfield law:

```text
1 * e_j = e_j
e_i * e_j = phi(i + j) + phi(|i - j|) for i, j > 0

phi(0)     = 2
phi(r)     = e_r        for 1 <= r <= 7
phi(8)     = 0
phi(8 + r) = -e_(8 - r) for 1 <= r <= 7
```

Important spot checks:

```text
e1^2 = 2 + e2
e2^2 = 2 + e4
e4^2 = 2
e7^2 = 2 - e2
e5 * e7 = e2 - e4
```

These identities are independent of the concrete production `D` as long as `SubfieldParams<D, 8>` validates. They are the K=8 analogue of the existing `RingSubfieldFp4` Chebyshev table; a naive power-basis or unrelated tower-basis `Fp8` is not acceptable unless it is hidden behind an exact canonical-basis conversion and all public/protocol boundaries still expose the canonical ring-subfield coordinates.

The public value representation should be:

```rust
#[repr(transparent)]
pub struct RingSubfieldFp8<F: FieldCore> {
    pub coeffs: [F; 8],
}
```

Implementation checklist:

- representation as 8 base-field limbs;
- `Field` and extension trait impls;
- base-limb conversion;
- multiplication/squaring/inversion;
- serialization and validation;
- `RingSubfieldEncoding` impl with `EXT_DEGREE = 8`;
- support in any dispatchers that currently match `EXT_DEGREE` values.

Current code already has several K=8 dispatch arms in `field_reduction.rs`, but the concrete field type is missing. The implementation should verify every generic bound needed by the prover, verifier, sumcheck, challenges, and profile example, instead of only compiling the field crate.

Internal multiplication backends may optimize the canonical law. Plausible backends include:

- direct Chebyshev structure-constant multiplication in `[1, e1, ..., e7]`;
- a private tower/Karatsuba backend using the nested elements `u = e4`, `v = e2`, `w = e1`, where `u^2 = 2`, `v^2 = 2 + u`, and `w^2 = 2 + v`;
- batched SIMD backends that transpose many extension elements into limb-major lanes.

All optimized backends must round-trip to the canonical `[c0, ..., c7]` representation and pass the `embed_subfield` multiplicativity tests.

#### 4. Transcript Contract

fp16 does not introduce new transcript labels or a new prover/verifier transcript order.

Claim-field and challenge-field elements use the existing extension-field transcript helpers over the base fp16 transcript field:

```text
append_ext_field(label, [c0, ..., c7])
sample_ext_challenge(label) -> [c0, ..., c7]
```

For `K = 8`, the helpers must append or sample limbs in increasing canonical limb order `0..7` using the existing derived limb-label scheme. The config-level `append_claim_field` and `sample_challenge_field` helpers must agree byte-for-byte/challenge-for-challenge with direct calls to the transcript helpers.

#### 5. SIS Floor Generation

Extend the SIS-generation workflow for `q = 65437`.

Inputs:

- modulus family: `Q16` or `Q16Offset99`;
- ring dimensions: `32, 64, 128, 256, 512`;
- module-rank sweep:
  - D32 through rank 20;
  - D64+ through rank 12 at minimum;
- same concrete security target currently used by fp32/fp64 production floors.

Outputs:

- generated Rust table in the new per-cell `&'static [u64]` shape (see below);
- checked-in or documented generator provenance: exact command, estimator version or commit, estimator model, modulus, security target, and sweep bounds;
- a small set of reproducible row spot-checks so reviewers can verify that the checked-in Q16 table matches the external estimator;
- optional CSV/debug artifacts under `target/` only;
- no checked-in ad hoc dry-run output.

##### Table shape change

The current `crates/akita-types/src/generated/sis_floor.rs` stores each row as `[u64; MAX_RANK]` with a global `MAX_RANK = 4`. fp16 forces this constant past every existing call site because small primes need higher ranks (the spec requires D32 sweeps through rank 20). Bumping `MAX_RANK` to 20 globally would pad every Q32/Q64/Q128 row with placeholder values and bake a misleading rank cap into the table.

Replace `[u64; MAX_RANK]` with `&'static [u64]`. Each `(family, D, collision_inf)` cell carries exactly the rank cap it actually needs as the slice length. The global `MAX_RANK` constant goes away, and `sis_max_widths` returns `Option<&'static [u64]>` (or a thin accessor over it).

Migration discipline:

- Preserve existing Q32/Q64/Q128 floor values as slice prefixes. Extend Q32/D32
  through rank 20 for the fp32 D32 default schedule, and gate the change behind
  snapshot tests that assert the existing prefix values are bit-identical.
- The reshape ships as its own commit, ahead of the Q16 floor. The fp16 floor is added only after the snapshot test is green on the existing families.
- Callers that iterate `0..MAX_RANK` (today: `crates/akita-config/src/proof_optimized.rs` and `crates/akita-planner/src/sis_security.rs`) are updated to iterate `slice.len()`.

The planner should make the modulus family explicit in configs and schedule generation. A missing fp16 SIS floor must be a hard planning error.

#### 6. Planner and Schedule Tables

Extend `crates/akita-config/src/bin/gen_schedule_tables.rs` with fp16 families.

Proposed generated families:

- `Fp16D32Full`
- `Fp16D32OneHot`
- `Fp16D64Full`
- `Fp16D64OneHot`

Wire D32 as the primary production target and D64 as the only comparison schedule. Do not generate D128-or-larger schedules for any prime family unless a future measured proof-size result reverses the current ordering.

For singleton dense workloads, use `ScheduleKey::singleton(num_vars)`. Do not set `num_w_vectors = K` at the root. The root EOR partial count for one dense opening is `K`.

For one-hot nv32, use the same workload conventions as the fp32/fp64 profile comparisons.

The generated table modules should be wired in `akita-types/src/generated/mod.rs` for normal and `zk` configurations.

#### 7. Config Presets

Add `pub mod fp16` in `crates/akita-config/src/proof_optimized.rs`.

The module should expose the same style of types as fp32/fp64:

```rust
pub type Field = Prime16Offset99;
pub type ExtensionField = RingSubfieldFp8<Field>;
```

Presets should include full and one-hot variants for D32 and D64. Larger-D preset structs may remain planner-backed experiments, but they must not be wired to generated schedule tables.

Each preset must select:

- ring dimension;
- terminal threshold;
- matrix/evaluation/window parameters;
- SIS family `Q16`;
- generated schedule table for D32/D64 only;
- small-field challenge extension degree 8;
- profile display metadata.

#### 8. Prover and Verifier

The expected design is no protocol fork.

The existing small-field opening pipeline should handle fp16 once:

- the base field implements the required traits;
- the degree-8 extension implements the required traits;
- `RingSubfieldEncoding` works for K=8;
- generated config and setup parameters are available.

Areas to inspect carefully:

- root extension-opening reduction;
- recursive level witness construction;
- ring-switch witnesses;
- transcript append and challenge sampling;
- setup cache typing and persistence keys;
- proof serialization and validation;
- verifier replay of tensor projections.

Setup-cache typing and persistence keys are acceptance-level behavior for fp16. Disk cache names/keys must distinguish the field modulus, config family, schedule key, ring dimension, and rank envelope so fp16 cannot reuse fp32/fp64 artifacts. Incompatible cached artifacts must be rejected or regenerated.

Root extension-opening reduction. The runtime must compute the partial count via the shared `root_extension_opening_partials(claim_ext_degree, num_points)` helper introduced in Phase 0. The planner side already calls the same helper through `extension_opening_reduction_proof_bytes`. The "one helper" rule is what makes the byte-equality invariant in acceptance criterion 6 structural rather than aspirational.

Any type-specific special case should be justified by performance data or trait limitations.

#### 9. Computation Representation And SIMD

Canonical protocol representation and computational representation are deliberately separate.

`Prime16Offset99` should store canonical `u16` values and serialize as `u16`, but arithmetic kernels may widen:

- scalar multiplication uses a `u32` product and pseudo-Mersenne reduction for `2^16 - 99`;
- SIMD add/sub/storage can use `u16` lanes;
- SIMD multiplication should widen to `u32` lanes before product reduction;
- fused product-sum kernels must document accumulator bounds and reduction frequency.

Architecture-specific paths should be introduced only behind the existing packed/wide abstraction style. The likely mapping is:

- NEON: `u16` lanes for storage/add/sub and `vmull_u16`/`vmull_high_u16` for products;
- AVX2: `u16` lanes with `mullo/mulhi` and unpacking to `u32` lanes for reduction;
- AVX-512BW: wider `u16`/`u32` lanes and masks for comparison/subtraction.

`RingSubfieldFp8` may use internal tower or SIMD layouts for hot kernels, but those layouts are not protocol-visible. The implementation should benchmark at least:

- direct canonical Chebyshev multiplication;
- private tower/Karatsuba multiplication converted to/from canonical coordinates;
- batched limb-major SIMD multiplication if extension multiplication is a profile hot spot.

#### 10. Profile and Documentation

Add profile modes mirroring fp32/fp64.

Suggested modes:

- `onehot_fp16_d32`;
- `full_fp16_d32`;
- `onehot_fp16_d64`;
- `full_fp16_d64`;
- `all_fp16` if useful for sweeps.

Update relevant docs:

- `AGENTS.md` or command docs only if the canonical profile instructions change;
- `README` or crate docs if fp16 becomes public API;
- the spec status as implementation progresses.

### Alternatives

#### Alternative A: Alias `Prime16Offset99 = Fp32<65437>`

This is useful for quick arithmetic experiments, but it is not sufficient for production fp16 support.

Problem: current `Fp32` serialization is fixed at 4 bytes. The proof-size reduction assumes 2-byte base limbs. Using `Fp32<65437>` would likely give close to fp16 arithmetic semantics but fp32 proof bytes.

Decision: reject for production. It can be used only as an intermediate spike if clearly removed before the spec is completed.

#### Alternative B: Make `Fp32<P>` Serialize Width-Aware

This would make `Fp32<65437>` serialize to 2 bytes based on `P::BITS`.

Pros:

- less new field code;
- existing packed/wide paths might compile sooner.

Cons:

- changes serialization semantics for every low-modulus `Fp32<P>`;
- risks surprising proof compatibility and validation behavior;
- still stores values in 32-bit lanes, so memory-bandwidth gains are limited.

Decision: possible but not preferred. If chosen, the spec must explicitly accept the serialization behavior change and add broad regression tests.

#### Alternative C: Dedicated `Fp16`

Pros:

- honest 2-byte serialization;
- compact storage;
- clearer public API;
- easier to reason about proof-size accounting;
- does not perturb existing `Fp32` semantics.

Cons:

- more trait implementation work;
- packed/wide helpers need new coverage;
- extension code may expose missing generic bounds.

Decision: preferred.

#### Alternative D: Stop at Planner-Only fp16

This would keep planner-only estimates without implementation.

Decision: reject. The user goal is to decide whether fp16 is worth implementation effort, and the only way to answer safely is to make proof-size estimates line up with real serialized proofs and measure performance.

#### Alternative E: Public Generic Tower-Basis `Fp8`

This would implement fp16 challenge values as an ordinary tower or power-basis degree-8 extension and then adapt it to the ring-subfield boundary later.

Decision: reject for the public/protocol type. Akita's extension-opening reduction consumes the cyclotomic ring-subfield coordinates used by `RingSubfieldEncoding`, `psi_embed`, and `embed_subfield`. A private tower/Karatsuba backend is acceptable only if it is an internal optimization with exact conversion to and from canonical `[1, e1, ..., e7]` coordinates at every protocol boundary.

## Documentation

Update documentation in three places:

1. Spec status
   - Mark this spec as proposed, accepted, in progress, and implemented as work advances.
   - Record any deviations from the D32-first plan.

2. User-facing profile docs
   - Add fp16 profile modes and example commands.
   - Document that fp16 uses `q = 65437` and extension degree 8.

3. Internal generated-table docs
   - Document how to regenerate fp16 SIS floors and schedule tables.
   - Include the exact estimator command, estimator version or commit, estimator model, modulus, security target, and rank sweep bounds.
   - Include reproducible spot-check rows for the Q16 SIS table.
   - State where generated artifacts should and should not be committed.

All proof-size numbers committed to this spec come from generated schedule tables checked by the release profile path. No private drafting note or dry-run helper is part of the spec's reproducibility surface.

## Execution

### Phase 0: Preflight, Baseline, and Shared Infrastructure

Phase 0 is intentionally infrastructure-heavy: it productizes the shared pieces every later phase depends on, so the fp16 field implementation does not also drag in plumbing churn.

- Confirm current branch/worktree state.
- Remove local planner dry-run scaffolding once generated schedule tables and profile byte-accounting tests exist. Generated schedules and release profiles are the source of truth for §"Current Proof-Size Targets".
- Reshape `crates/akita-types/src/generated/sis_floor.rs` from `[u64; MAX_RANK]` to per-cell `&'static [u64]` (see §"SIS Floor Generation / Table shape change"). Preserve existing Q32/Q64/Q128 values as prefixes, extend Q32/D32 through rank 20 for the fp32 D32 baseline, and gate the change behind snapshot tests.
- Add the shared `root_extension_opening_partials(claim_ext_degree, num_points)` helper. Route the existing fp32/fp64 planner-side `extension_opening_reduction_proof_bytes` callers and the prover-side `prepare_root_extension_opening_reduction` callers through it. Verify the existing fp32 D32 onehot nv32 and dense singleton nv26 planner estimates match real `AkitaSerialize::serialized_size()` byte-for-byte. This is the baseline that fp16 must continue to satisfy.
- Tighten per-`Vec` length-prefix accounting in `proof_size.rs` so the planner estimate equals the real serialized size without a tolerance term. Any required adjustment must move both fp32 and fp64 numbers in the same way; document the diff in this spec.
- Capture current fp32/fp64 profile outputs for:
  - one-hot nv32 D32;
  - dense singleton nv26 D32;
  - any D64+ comparison still used by docs.
- Record exact current proof-size estimates and serialized proof sizes.

Exit criteria:

- No planner dry-run binary is required for production proof-size accounting.
- `sis_floor.rs` is reshaped to `&'static [u64]`; the Q32/Q64/Q128 floor prefix snapshot test is green.
- The shared root EOR partial helper exists and is used by both the planner and the prover paths.
- Real fp32 D32 serialized proof bytes equal planner estimate bytes.
- Existing tests pass before fp16 work begins, or failures are documented as pre-existing.

### Phase 1: fp16 Base Field

- Add `Fp16` implementation.
- Add `Prime16Offset99`.
- Add pseudo-Mersenne metadata.
- Implement serialization as 2 canonical little-endian bytes.
- Implement validation rejecting encodings `>= 65437`.
- Implement field arithmetic and required helper traits.
- Add field unit tests.

Exit criteria:

- `cargo test -p akita-field fp16` passes.
- `Prime16Offset99::serialized_size()` is 2.

### Phase 2: Degree-8 Extension Field

- Add `RingSubfieldFp8`.
- Define the canonical `[1, e1, ..., e7]` basis and implement the Chebyshev/cyclotomic multiplication law.
- Implement field operations and extension traits.
- Implement `RingSubfieldEncoding`.
- Implement canonical 16-byte serialization as eight fp16 limbs in `[c0, ..., c7]` order.
- Add K=8 tests for multiplication-table spot checks, embedding, projection, trace identity, and serialization.
- Ensure transcript/challenge code can append and sample the extension field using existing 8-limb transcript helper ordering.

Exit criteria:

- `cargo test -p akita-field ring_subfield_fp8` or equivalent focused tests pass.
- `cargo test -p akita-types field_reduction` passes with fp16/K8 coverage.
- Transcript helper tests pass for fp16 claim/challenge field appends and sampling.

### Phase 3: SIS Floors

- Extend the SIS-floor generation script/tooling for `q = 65437`.
- Record the estimator command, estimator version or commit, estimator model, modulus, security target, and sweep bounds.
- Run estimator sweeps:
  - D32 ranks 1 through 20;
  - D64 ranks 1 through at least 12.
- Generate Rust floor tables.
- Add reproducible spot-checks for selected generated Q16 rows.
- Add fp16 floor lookup tests.

Exit criteria:

- Planner can query fp16 SIS floors for all intended generated families.
- Missing floor coverage is a clear error.
- Q16 floors can be regenerated or spot-checked from the documented estimator invocation.

### Phase 4: Schedule Generation

- Add fp16 family definitions to `gen_schedule_tables`.
- Generate singleton and batched schedule tables.
- Wire generated modules into `akita-types`.
- Add fp16 presets in `akita-config`.
- Add schedule tests for one-hot nv32 and dense singleton nv26/nv27.

Exit criteria:

- `cargo check -p akita-config --bin gen_schedule_tables` passes.
- `cargo test -p akita-planner fp16` passes.
- Generated fp16 table estimates match profile-reported serialized proof sizes unless estimator/table improvements intentionally change them.

### Phase 5: End-to-End Proving

- Add profile modes for fp16 D32.
- Verify setup-cache key separation under `disk-persistence`.
- Run small debug prove/verify tests.
- Run release profiles:
  - one-hot nv32;
  - dense singleton nv27;
  - dense singleton nv26 if comparing directly against fp32 nv26.
- Compare planner size vs actual serialized proof size.

Exit criteria:

- fp16 proofs verify.
- Actual serialized size exactly matches the planner/profile accounting, except for explicitly modeled framing bytes or intentionally documented estimator/table changes.
- Any discrepancy is explained and either fixed or recorded in this spec.
- fp16 setup caches cannot reuse fp32/fp64 cache artifacts.

### Phase 6: Performance Pass

- Profile fp16 D32 workloads.
- Identify hot spots in:
  - base-field arithmetic;
  - degree-8 extension multiplication/inversion;
  - root EOR tensor projection;
  - recursive witness construction;
  - serialization.
- If extension multiplication is hot, compare direct Chebyshev multiplication against private tower/Karatsuba and, where practical, batched SIMD layouts.
- If base arithmetic or serialization is hot, evaluate architecture-specific SIMD kernels behind the existing packed/wide abstractions.
- Optimize only measured bottlenecks.
- Compare wall time and proof size against fp32 D32.

Exit criteria:

- We have real proof-size and runtime tables for fp16 vs fp32.
- Any architecture-specific SIMD path is covered by scalar equivalence tests and has a portable fallback.
- The team can decide whether the approximately 12-13% proof-size reduction justifies the implementation and maintenance cost.

### Phase 7: Cleanup and Docs

- Confirm generated schedule tables plus release profiles are the only proof-size oracle the spec depends on; remove any remaining local-only dry-run scaffolding.
- Update user-facing profile docs.
- Mark this spec implemented or document remaining follow-up specs (notably the cross-family transcript-isolation follow-up called out in Non-Goals).
- Run full workspace checks.

Exit criteria:

- `cargo fmt -q` passes.
- `cargo clippy --all --message-format=short -q -- -D warnings` passes.
- `cargo test` passes.
- The final implementation has no planner-only artifacts masquerading as production support.

## Risks

- The degree-8 extension may compile through field tests but expose missing trait bounds in prover/verifier generics.
- A generic tower or power-basis `Fp8` implementation can accidentally pass field-level tests while failing the `RingSubfieldEncoding` contract. The canonical basis, multiplication table, and `embed_subfield` multiplicativity tests are required to catch this.
- `Fp16` may require more packed/wide helper work than expected if current polynomial backends assume 32-bit scalar storage.
- SIMD or tower/Karatsuba backends can diverge from the canonical representation if conversion boundaries are not tested. Every optimized backend must have scalar/canonical equivalence tests.
- Fused fp16 product accumulation can overflow if accumulator bounds are guessed. Any fused kernel must document product-sum bounds and reduction frequency.
- D32 rank requirements are higher than earlier sweeps assumed; schedule generation must keep the rank search high enough. The `MAX_RANK = 4` storage shape in today's `sis_floor.rs` cannot express the required ranks; Phase 0 reshapes the table to `&'static [u64]` ahead of any Q16 floor.
- Reshaping the SIS floor table touches every existing field family. The fp32/fp64/fp128 floor snapshot test gates this risk: the reshape commit must not move any existing floor value.
- The Q16 SIS floors depend on an external estimator/toolchain. The generated table must preserve estimator provenance and reproducible spot-checks so tool drift is visible.
- The planner/prover root EOR partial counts can silently drift apart if computed in two places. The shared `root_extension_opening_partials` helper introduced in Phase 0 is the structural defense; this risk only re-appears if a later change bypasses that helper.
- The observed proof-size win is modest, around 12-13%, so runtime regressions can erase the practical value.
- If actual serialized proofs do not match planner estimates, proof-size accounting must be fixed before making any product decision.
- Setup cache keys may accidentally collide across small-field families if they omit the modulus, config family, schedule key, ring dimension, or rank envelope.
- Incidence-generalization work may overlap with fp16 schedule-key semantics. The fp16 implementation should avoid adding new dependence on the old `w = num_claims` interpretation.

## References

- `specs/TEMPLATE.md`
- `specs/extension-field-opening-batching.md`
- `specs/planner-incidence-generalization.md`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-config/src/bin/gen_schedule_tables.rs`
- `crates/akita-types/src/layout/proof_size.rs`
- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-field/src/fields/pseudo_mersenne.rs`
- `crates/akita-field/src/fields/fp32.rs`
- `crates/akita-field/src/fields/ext.rs`
- `crates/akita-setup/src/lib.rs`
- `crates/akita-transcript/src/lib.rs`
