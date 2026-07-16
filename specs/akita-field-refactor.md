# Spec: `akita-field` Refactor Proposal

| Field     | Value                        |
| --------- | ---------------------------- |
| Author(s) | Taghi Badakhshan             |
| Status    | superseded by `jolt-field-unification.md` |
| Branch    | `taghi/refactor/akita-field` |

## Summary

This PR makes `akita-field` a cleaner standalone field crate. The change has four
connected parts:

- Akita owns its native field trait hierarchy directly.
- Jolt interop is isolated behind the opt-in `jolt-compat` feature.
- Public field APIs use meaningful root names and role modules instead of the
  old implementation umbrella `akita_field::fields::*`.
- The implementation tree is reorganized into role-named modules:
  `prime`, `ext`, `unreduced`, `packed`, and `fft`.

The intended result is a crate whose public API describes algebraic concepts,
not file paths or temporary integration seams.

## Goals

- Make `akita-field` the source of truth for Akita's field traits:
  `AdditiveGroup`, `RingCore`, `FieldCore`, `Invertible`, `FromPrimitiveInt`,
  `RandomSampling`, byte/canonicalization traits, accumulator traits, and related
  capability traits.
- Preserve downstream imports of common field vocabulary from the crate root:
  `akita_field::{FieldCore, Fp32, Fp64, Fp128, Prime31Offset19, FpExt2, ...}`.
- Keep specialized concepts in semantic public modules:
  `akita_field::packed`, `akita_field::unreduced`, and `akita_field::fft`.
- Remove `akita_field::fields::*` as public API.
- Move Jolt-specific trait impls into one feature-gated seam:
  `akita_field::compat::jolt`.
- Keep `jolt-field` out of the normal dependency graph unless
  `jolt-compat` is explicitly enabled.
- Use consistent names for extension fields: `FpExt2`, `FpExt4`, and `FpExt8`
  denote extension degree; `Fp32`, `Fp64`, and `Fp128` remain prime field bit
  widths.
- Keep the refactor behavior-preserving: no arithmetic, serialization, proof
  layout, transcript, or verifier behavior changes.

## Non-Goals

- Do not remove Jolt interop; make it optional and isolated.
- Do not mirror Jolt-only abstractions such as the Jolt `Field` umbrella,
  `OptimizedMul`, `Limbs`, `signed`, or `MontgomeryConstants`.
- Do not define custom `Zero` / `One`; continue using `num_traits`.
- Do not rename prime fields (`Fp32`, `Fp64`, `Fp128`) or named prime types.
- Do not expose implementation details such as pseudo-Mersenne per-prime
  `*_MODULUS` / `*_OFFSET` constants as public API.
- Do not preserve the old `fields` umbrella for compatibility. This repository
  does not guarantee backward compatibility, and the old path is not meaningful
  API.

## Public API

The public API is:

```rust
use akita_field::{
    CanonicalField, FieldCore, Fp128, Fp32, Fp64, FpExt2, Prime31Offset19,
    Prime64Offset59, Prime128Offset275, RandomSampling, RingSubfieldFpExt4,
    TowerBasisFpExt4,
};

use akita_field::fft::{primitive_nth_root, rs_extend_fft, SmoothDomain};
use akita_field::packed::{HasPacking, PackedField, PackedFpExt2};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
```

The root carries common field vocabulary. The `packed`, `unreduced`, and `fft`
modules carry specialized concepts that benefit from namespacing.

The old API shape is intentionally removed:

```rust
use akita_field::fields::...; // not public API
```

Prime and extension implementation modules are private. Their public types are
re-exported at the crate root.

## Trait Ownership And Jolt Interop

`akita-field` defines the native trait hierarchy in `traits.rs` and re-exports
those traits from the crate root. Downstream Akita crates continue to write
`F: akita_field::FieldCore`; the trait identity is now Akita-owned rather than
re-exported from Jolt.

Jolt interop lives only in `compat::jolt`, behind the `jolt-compat` feature. That
module implements `jolt_field` traits for concrete Akita field types by
delegating to the native implementations. The compat layer is a bridge, not the
owner of core algebraic behavior.

The normal library build does not require `jolt-field`. The feature is available
for external Jolt-facing code that needs it:

```toml
akita-field = { path = "...", features = ["jolt-compat"] }
```

## Naming

Extension fields use `Ext` in the name:

| Concept | Public name |
| --- | --- |
| Quadratic extension | `FpExt2`, `FpExt2Config` |
| Quartic power basis | `PowerBasisFpExt4`, `PowerBasisFpExt4Config`, `PowerBasisFpExt4MulBackend` |
| Quartic tower basis | `TowerBasisFpExt4`, `TowerBasisFpExt4Config` |
| Quartic ring subfield | `RingSubfieldFpExt4`, `RingSubfieldFpExt4MulBackend` |
| Octic ring subfield | `RingSubfieldFpExt8`, `RingSubfieldFpExt8MulBackend` |
| Packed extension wrappers | `PackedFpExt2`, `PackedPowerBasisFpExt4`, `PackedTowerBasisFpExt4`, `PackedRingSubfieldFpExt4`, `PackedRingSubfieldFpExt8` |

This avoids overloading `Fp{N}` across two meanings. `Fp32`, `Fp64`, and
`Fp128` remain bit-width prime field names.

The old "wide" accumulator concept is exposed as `unreduced`. `wide` describes
representation; `unreduced` describes the semantic contract: values are
accumulated before reduction back into a canonical field element.

## Implementation Layout

The final crate layout is role-named:

```text
crates/akita-field/src/
  lib.rs
  traits.rs              # Akita-owned field trait hierarchy
  compat/
    mod.rs
    jolt.rs              # optional Jolt adapter
  prime/
    mod.rs
    fp32.rs
    fp64.rs
    fp128/
      mod.rs
      add_sub.rs
      core.rs
      mul.rs
      primes.rs
      reduce.rs
      tests.rs
      traits.rs
      wide.rs
    native_algebra.rs
    native_capability.rs
    pseudo_mersenne.rs
    util.rs
  ext/
    mod.rs
    fp_ext2.rs
    lift.rs
    native_algebra.rs
    power_fp_ext4.rs
    ring_subfield_fp_ext4.rs
    ring_subfield_fp_ext8.rs
    tests.rs
    tower_fp_ext4.rs
  unreduced/
    mod.rs
    accum.rs
    native_algebra.rs
    tests.rs
  packed/
    mod.rs
    ext/
      mod.rs
      tests.rs
    avx2/
      mod.rs
      fp32.rs
      fp64.rs
      fp128.rs
    avx512/
      mod.rs
      fp32.rs
      fp64.rs
      fp128.rs
    neon/
      mod.rs
      fp32.rs
      fp64.rs
      fp128.rs
  fft.rs
  error.rs
  parallel.rs
```

Every multi-file module uses a directory with `mod.rs`. Standalone modules remain
single `.rs` files.

## Dependency Boundaries

The intended production dependency direction is:

```text
traits
  ↑
prime
  ↑
unreduced
  ↑
ext
  ↑
packed

fft    → traits
compat → traits, prime, ext, unreduced
```

Guardrails:

- `traits.rs` must stay a leaf and must not import concrete field modules.
- `prime` must not depend on `ext`, `unreduced`, or `packed`.
- `unreduced` may depend on `prime`, but production code should stay free of
  `ext` imports.
- `ext` may depend on `prime` and `unreduced`.
- `packed` may depend on `prime` and `ext`; packed extension kernels are part of
  the packed surface.
- `compat::jolt` is the only code module that may name `jolt_field`.

## Test Layout

White-box unit tests stay colocated with their modules:

- Directory modules with substantial tests use sibling `tests.rs` files, such as
  `prime/fp128/tests.rs`, `ext/tests.rs`, `unreduced/tests.rs`, and
  `packed/ext/tests.rs`.
- Small single-file modules may keep inline `#[cfg(test)] mod tests`.
- Cross-crate behavior remains in integration tests outside `akita-field`.

## Expected Behavior

The refactor should not change:

- field arithmetic,
- reduction behavior,
- canonical byte encodings,
- serialization,
- transcript event streams,
- commitments,
- proof bytes,
- verifier behavior,
- packed backend selection,
- FFT semantics,
- benchmark-relevant algorithms.

The PR changes ownership, names, visibility, and source layout.

## Verification

The PR should be accepted only if these remain green:

```bash
cargo fmt --check
cargo clippy -p akita-field --all-targets --message-format=short -- -D warnings
cargo clippy -p akita-field --no-default-features --lib --message-format=short -- -D warnings
cargo clippy -p akita-field --features jolt-compat --all-targets --message-format=short -- -D warnings
RUSTFLAGS="-C target-feature=+avx2" cargo clippy -p akita-field --lib --target x86_64-apple-darwin --message-format=short -- -D warnings
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo clippy -p akita-field --lib --target x86_64-apple-darwin --message-format=short -- -D warnings
scripts/check-rust-file-lines.sh --no-baseline
cargo build --workspace --all-targets --message-format=short
cargo test --workspace --message-format=short
```

Additional hygiene checks:

```bash
rg 'crate::fields|akita_field::fields' crates/akita-field/src
rg 'jolt_field|jolt-field' crates/
```

The first should have no matches in `akita-field/src`. The second should be
limited to `crates/akita-field/src/compat/**`, `crates/akita-field/Cargo.toml`,
and documentation/comments that explicitly discuss the compat seam.
