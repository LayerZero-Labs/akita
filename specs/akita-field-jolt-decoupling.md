# Spec: akita-field Owns Its Trait Hierarchy — Jolt Interop Behind a Feature

| Field     | Value                                              |
| --------- | -------------------------------------------------- |
| Author(s) | Taghi Badakhshan                                   |
| Created   | 2026-06-04                                         |
| Status    | proposed                                           |
| PR        |                                                    |
| Base      | `main`                                             |
| Branch    | `taghi/refactor/akita-field` — folded into the structural-refactor PR (see §Execution, §Alternatives #6) |

## Summary

`akita-field` is currently two crates wearing one hat: a field crate **and** a
compatibility shim over Jolt's field traits. Today `akita_field::FieldCore`,
`RingCore`, `AdditiveGroup`, `RandomSampling`, and ~15 other core trait names are
**`pub use jolt_field::*` re-exports** ([`crates/akita-field/src/arithmetic.rs:5-11`](../crates/akita-field/src/arithmetic.rs)).
That means Jolt *owns the trait identity* of Akita's entire algebra: every
`F: FieldCore` bound in `akita-algebra`, `akita-prover`, `akita-verifier`, etc.
resolves to a Jolt-defined trait, and the concrete impls live in a Jolt-named
module ([`crates/akita-field/src/jolt_traits.rs`](../crates/akita-field/src/jolt_traits.rs)).

This coupling was a **deliberate first integration slice** (PR #65,
[`specs/akita-crate-followup-jolt-integration.md`](akita-crate-followup-jolt-integration.md)):
adopt Jolt-shaped traits to start the Jolt integration without a full field
cutover, while "track[ing] the broader Jolt field-trait refactor direction." That
broader refactor has since landed on the Jolt side — `jolt-field` is now a *slim,
non-BN254-specific* hierarchy (`AdditiveGroup → RingCore → FieldCore`, plus
orthogonal capability traits). With the slim hierarchy stable and `jolt-field`
**already fully encapsulated inside `akita-field`** (it is named by exactly 6
source files, all under `crates/akita-field/src/` — `arithmetic.rs`,
`jolt_traits.rs`, and `fields/{fp32,fp64,fp128,ext}.rs`; nothing else in the
workspace — including the in-repo Jolt recursion profile — names `jolt_field`),
the natural next step is
to **invert the dependency**: let `akita-field` *own* the trait definitions it
needs, and demote Jolt interop to an optional, feature-gated adapter.

This is a **pure structural / trait-ownership refactor**: no field arithmetic, no
serialized bytes, no transcript stream, no proof layout, and no performance
characteristic changes. The downstream surface (`use akita_field::FieldCore`, …)
is preserved name-for-name; only the *definition site* of those names moves from
`jolt_field` into `akita-field`.

## Intent

### Goal

Make `akita-field` the single source of truth for the slim algebraic trait
hierarchy Akita actually uses, and isolate every reference to `jolt_field` inside
one feature-gated `compat::jolt` module so the crate builds, tests, and ships
without Jolt unless Jolt interop is explicitly requested.

Key abstractions introduced or moved:

- **Native trait hierarchy in `akita-field::traits`** (the native-traits module,
  renamed `arithmetic` → `traits`). Akita-owned definitions of the slim hierarchy
  and capability traits it currently re-exports from `jolt_field`:
  `AdditiveGroup`, `RingCore`, `FieldCore`, `Invertible`, `FromPrimitiveInt`,
  `MulPow2`, `MulPrimitiveInt`, `CanonicalBytes`, `ReducingBytes`, `FixedBytes<N>`,
  `FixedByteSize`, `CanonicalBitLength`, `CanonicalU64`, `RandomSampling`,
  `TranscriptChallenge`, `WithAccumulator`, `AdditiveAccumulator`,
  `RingAccumulator`, `NaiveAccumulator`. These keep their exact current names and
  method signatures (so downstream bounds are unaffected).
- **`akita-field::compat::jolt` (feature `jolt-compat`).** The *only* module that
  names `jolt_field`. Renamed and re-scoped from today's `jolt_traits.rs`. It
  implements each `jolt_field::*` trait for each concrete Akita field type by
  delegating to the native impls. This is what lets Akita field elements plug into
  Jolt-side code (`jolt-core` verifier, `jolt-transcript`).
- **`jolt-field` becomes an `optional` dependency** behind the `jolt-compat`
  feature; `jolt-transcript` (dev-dependency) is likewise gated to the
  compat-layer tests.

Already native and explicitly retained as Akita-owned (no change):
`CanonicalField`, `HalvingField`, `BalancedDigitLookup`, `PseudoMersenneField`,
`SmoothFftField` ([`arithmetic.rs:13-105`](../crates/akita-field/src/arithmetic.rs)).
PR #65 Invariant 6 already established these as Akita-specific; this spec keeps
that boundary.

### Invariants

The implementation must preserve the following. Where a check is mechanically
enforced, the protecting test/command is named.

1. **Behavioral identity.** Serialized proof bytes, transcript event streams,
   commitments, byte encodings, RNG-derived samples, and all observable
   prover/verifier output are identical before and after this change for any
   fixed `(setup, polynomial, opening point, transcript)`. The guarantee is
   structural — no arithmetic, constant, byte-layout, or sampling logic is
   rewritten, only relocated and relabeled. Protected by the full
   `cargo test --workspace` suite (default features) and the `akita-field` unit
   tests.

2. **Downstream surface is name-for-name unchanged.** Every external import of
   the form `use akita_field::{FieldCore, RingCore, RandomSampling, …}` continues
   to compile and resolve to a trait with the identical method set. No consumer
   crate (`akita-algebra`, `akita-prover`, `akita-verifier`, `akita-types`,
   `akita-config`, `akita-challenges`, `akita-sumcheck`, `akita-scheme`,
   `akita-r1cs`, `akita-setup`, `akita-pcs`, and the recursion `glue`) changes a
   single trait-import line as a result of this change. Protected by
   `cargo check --workspace --all-targets` and `scripts/check-crate-deps.sh`.

3. **Single Jolt seam.** After the change, `rg "jolt_field|jolt-field"` over
   `crates/` matches **only** files inside `crates/akita-field/src/compat/` and
   `crates/akita-field/Cargo.toml`. No `jolt_field` reference survives in
   `traits.rs` (renamed from `arithmetic.rs`), any field-implementation module
   under `fields/**` (`fields/{fp32,fp64,fp128,ext}.rs` are the four that name it
   today), or `lib.rs`. Protected by a CI grep guard
   (new `scripts` check or an existing dependency-hygiene step) plus
   `cargo check -p akita-field --no-default-features` (compiles with Jolt absent).

4. **Jolt interop still works when requested.** Under `--features jolt-compat`,
   every concrete Akita field type (`Fp32<P>`, `Fp64<P>`, `Fp128<P>`, `FpExt2`,
   the `PowerBasis`/`TowerBasis`/`RingSubfield` extension families, and the wide
   accumulator types) implements the same set of `jolt_field` traits it does
   today. Protected by the relocated compat tests
   (`prime_fields_satisfy_jolt_byte_capabilities`,
   `jolt_digest_transcripts_accept_akita_fields`) now gated behind
   `feature = "jolt-compat"`.

5. **Verifier no-panic contract preserved** (per `AGENTS.md`). Field paths are
   verifier-reachable. This change adds no new `panic!`/`unwrap`/`expect`/unchecked
   indexing on any verifier-reachable path; native trait impls are the byte-for-byte
   relocation of existing impls. The compat layer is not verifier-reachable in any
   in-workspace configuration (the verifier consumes native `akita_field` traits).

6. **`One`/`Zero` foundation is shared, not forked.** Akita's native
   `AdditiveGroup`/`RingCore` keep `num_traits::{Zero, One}` as supertrait bounds
   (exactly as `jolt_field` does). Akita does not define its own `Zero`/`One`.
   This keeps the compat bridge for those bounds *empty* and avoids conversion
   churn. (`num-traits` stays a direct dependency.)

7. **Orphan rule respected.** The compat layer impls foreign (`jolt_field`) traits
   only for **local** Akita types (per concrete type, as the existing
   `impl_prime_jolt_traits!` macro does); there is no blanket
   `impl<T: akita::FieldCore> jolt::FieldCore for T` (impossible under the orphan
   rule, and not attempted).

8. **File-size cap holds.** Every `.rs` file stays under the 1500-line cap
   (`scripts/check-rust-file-lines.sh`). Splitting `traits.rs` into a
   `traits/` submodule, if done, is to aid review, not because of the cap.

### Non-Goals

- **Not removing Jolt interop.** The point is to make it *optional and isolated*,
  not to delete it. `profile/akita-recursion` and any future Jolt host/guest
  integration keep a supported path via `--features jolt-compat`.
- **Not mirroring Jolt's `Field` umbrella, `OptimizedMul`, `MaybeAllocative`,
  `MontgomeryConstants`, `Limbs`, or `signed`.** Akita does not surface these
  today and does not need them natively. Note Jolt's `Field` umbrella requires
  `FixedBytes<32>`, while Akita primes are `FixedBytes<4|8|16>` — Akita fields
  intentionally do **not** satisfy `jolt_field::Field` (this is precisely the
  "too BN254-specific" path PR #65 declined). The compat layer targets the slim
  *capability* traits, not the umbrella.
- **Not redefining `One`/`Zero`** (see Invariant 6).
- **Not changing field arithmetic, constants, memory layout, packing, FFT, or any
  algorithm.** This is trait relocation + delegation only.
- **Not touching consumer crates' call sites.** If any consumer edit is required
  beyond a `Cargo.toml` feature wiring, the native trait surface was defined
  wrong; fix the surface, not the call site.
- **Not the `fields/` → role-named module split.** The companion reorg that breaks
  the `fields/` umbrella into `prime/`/`unreduced/`/`ext/`/`packed/`/`fft` is
  **deferred**: this change keeps the `fields/` umbrella and only inverts trait
  ownership (`arithmetic`→`traits`, `jolt_traits`→`compat`, native traits). The
  split can be a later PR (see §Module layout). Note this change *does* ride in the
  `taghi/refactor/akita-field` PR (§Alternatives #6), so that PR is no longer a
  strictly behavior-free refactor — an accepted trade-off.
- **Not introducing a Jolt rev bump or a `[workspace.dependencies]` anchor for
  `jolt-field`.** Pin handling is orthogonal; can be a follow-up.

## Evaluation

### Acceptance Criteria

- [x] **Module renamed.** The native-traits module is renamed `arithmetic` →
      `traits` (`arithmetic.rs` → `traits.rs`, or a `traits/` dir); `lib.rs`
      declares `pub mod traits;` and the crate-root re-exports are unchanged. The
      new top-level modules are `traits` / `compat`; the existing `fields/`
      umbrella is **retained as-is** (the `fields/` → role-named split is deferred;
      see §Module layout).
- [x] **Native traits defined.** `akita-field` defines the slim hierarchy and
      capability traits listed in §Goal natively (same names, same method
      signatures, same default-method bodies as the current `jolt_field`
      versions). The renamed `traits` module no longer contains
      `pub use jolt_field::…`.
- [x] **Concrete impls rehomed.** The prime field modules (`fp32`/`fp64`/`fp128`)
      and the `ext` families implement the **native** `FromPrimitiveInt` /
      `Invertible` / `RandomSampling` (today resolved to `jolt_field` — directly in
      the four module roots, or via the crate-root re-export in the `ext`
      submodules). The `jolt_traits.rs` macros are **split**: their `std`/
      `num_traits` supertrait impls (`Zero`/`One`/`Display`/`Hash`/`Sum`/`Product`)
      and byte/bit logic land on the native traits/modules, while only the
      `jolt_field::*` impls go to `compat/jolt.rs` (see §Design).
- [x] **Compat module.** `crates/akita-field/src/compat/jolt.rs` exists, is the
      only `jolt_field`-naming module, and is gated by `#[cfg(feature = "jolt-compat")]`.
      It provides the `jolt_field::*` impls for every concrete Akita field type by
      delegating to native impls.
- [x] **Cargo wiring.** `jolt-field` is `optional = true`; `jolt-compat` feature
      gates it. `jolt-compat` was in `default` through Phases 0–3 (zero downstream
      churn) and is **removed from `default` in Phase 4** (`default = []`); no
      workspace crate needs it back. `jolt-transcript` dev-dependency is used only
      under `jolt-compat` tests.
- [x] **Builds without Jolt.** `cargo check -p akita-field --no-default-features`
      and `cargo check -p akita-field --no-default-features --features parallel`
      both succeed with no `jolt-field` in the **normal** dependency graph
      (`cargo tree -p akita-field --no-default-features -e normal` shows no
      `jolt-field`). The dev-only `jolt-transcript` keeps Jolt in the *test* graph
      (dev-deps can't be `optional`); that is expected — see the Cargo.toml note.
- [x] **Builds and tests with Jolt.** `cargo test -p akita-field` (default, i.e.
      `jolt-compat` on) passes, including the relocated compat tests.
- [x] **Seam guard.** `rg "jolt_field|jolt-field" crates/` matches only
      `crates/akita-field/src/compat/**` and `crates/akita-field/Cargo.toml`.
      A companion guard for `jolt_transcript|jolt-transcript` matches only the
      compat tests and `crates/akita-field/Cargo.toml`.
- [x] **Workspace unaffected.** `cargo check --workspace --all-targets` and
      `cargo test --workspace` pass with **no edits to any consumer crate's
      trait imports**.
- [x] **Recursion glue.** `profile/akita-recursion` (pinned Rust 1.95 + RISC-V)
      builds against the decoupled `akita_field` — it consumes the native traits
      (`FieldCore`, `RandomSampling`, …) directly; no `jolt-compat` is needed there.
      **Phase 4 jolt-compat-OFF rebuild done:** with `akita-field` `default = []`,
      `cargo build --release` (host-side `artifact` + `host`, plus `glue`/`guest`)
      compiles, and the sub-workspace lock drops `jolt-field` entirely — jolt-core
      uses arkworks, not `jolt-field`, so nothing else retains it and the recursion
      tree is now fully `jolt-field`-free on every edge. Runtime confirmation: the
      `artifact` run (nv=20 D=32 OneHot) reports `host-side verify OK` and
      `decoded-blob verify OK`.

  *Out-of-scope finding (NOT addressed here).* A full guest end-to-end run
  (`nv=20` D=32 OneHot trace-only) additionally trips a verifier-reachable
  `getrandom` panic that is **independent of this trait decoupling**: the
  schedule-search DP memo (`akita-planner::schedule_params::ScheduleMemo`) is a
  default-`RandomState` `HashMap`, and `RandomState` seeding aborts inside the
  Jolt zkVM guest. It reproduces regardless of the jolt-field split, so it is left
  as a separate `akita-planner` follow-up (candidate fix: a deterministic /
  getrandom-free memo) and is not part of this work.
- [x] **Lints / format / cap.** `cargo fmt --all --check`,
      `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
      `cargo clippy --workspace --all-targets --no-default-features -- -D warnings`,
      and `scripts/check-rust-file-lines.sh --no-baseline` all clean. **Post-Phase-4**
      (`default = []`) the plain `cargo clippy --workspace --all-targets -- -D warnings`
      is itself jolt-free on normal edges, and `cargo tree -e normal` shows no
      `jolt-field` anywhere in the workspace (it remains only on `akita-field`'s dev
      edge via `jolt-transcript`, which cannot be `optional`).

### Testing Strategy

No new behavior, so validation is "the existing suite keeps passing" plus a small
set of structural gates:

**Existing tests that must keep passing (unchanged assertions):**

- `cargo test --workspace` (default features) — the whole protocol suite exercises
  the native traits through every `F: FieldCore` bound.
- `akita-field` unit tests: prime/extension arithmetic, packed parity
  (`packed_ext.rs`), FFT (`omega_has_declared_order`), serialization order.
- fp128 / fp32 / fp64 E2E in `crates/akita-pcs/tests/akita_e2e.rs`.
- Cross-architecture compile of the packed backends:
  `cargo check -p akita-field --lib --target x86_64-apple-darwin` (AVX2/AVX-512)
  in addition to the native aarch64/NEON build.

**Tests relocated (not rewritten), now `#[cfg(all(test, feature = "jolt-compat"))]`:**

- `prime_fields_satisfy_jolt_byte_capabilities` and
  `jolt_digest_transcripts_accept_akita_fields`
  (currently `jolt_traits.rs::tests`) move into `compat/jolt.rs` under the feature
  gate, asserting the Jolt-trait impls still hold.

**New structural gates:**

- Feature-matrix builds in CI for `akita-field`: `--no-default-features`,
  `--no-default-features --features parallel`, default (`jolt-compat`),
  `--all-features`.
- Workspace compile matrix: default, `--no-default-features`, `--all-features`.
  Post-Phase-4 the default workspace build is itself jolt-free on normal edges
  (`default = []`), so `cargo tree -e normal` showing no `jolt-field` is now a
  whole-workspace "no Jolt on normal edges" assertion (it lingers only on the
  `akita-field` dev edge via `jolt-transcript`, which cannot be `optional`).
- The seam grep guard (Invariant 3).

**Feature combinations to run `cargo test`/`cargo check` under:**
default; `--no-default-features`; `--features jolt-compat`;
`--all-features`.

### Test layout

Tests stay **co-located in-crate unit tests** (`#[cfg(test)] mod tests`), never
`tests/` integration tests: the existing suites reach into crate internals
(`pub(crate)` consts like `pseudo_mersenne::PRIME31_OFFSET19_MODULUS`, private
limb fields, `crate::fields::wide::*`) that an integration test cannot see. This
holds with or without the deferred split.

**In scope for this change:** only the Jolt-coupled test pair moves (last
paragraph below). Every other field/extension/packed/fft test stays exactly where
it is today, because the `fields/` umbrella is retained. The relocation table
below is **deferred** — it documents where tests go *when* the `fields/` split
eventually happens, not now.

Placement follows the convention already in the crate, with the 1500-line cap as
tiebreaker (the checker counts **all** physical lines, tests included):

- **Directory module ⇒ separate `tests.rs` submodule** (`#[cfg(test)] mod tests;`),
  as `ext/tests.rs`, `fp128/tests.rs`, `packed_ext/tests.rs` do today. Since the
  split turns flat files into directories, their tests become `tests.rs` siblings.
- **Single-file module ⇒ inline** `#[cfg(test)] mod tests { … }`, unless inlining
  pushes the file near the cap — then externalize.
- **Shared oracles/helpers** go in a `test_support` submodule (the `fft.rs`
  pattern: one `naive_dft` oracle reused by the per-prime test modules).

Tests travel with their module. Relocations under the target tree (deferred —
applies only when the `fields/` split lands):

| Tests today | After split |
| --- | --- |
| `wide/mod.rs` inline (~250 ln) | `unreduced/tests.rs` |
| `lift.rs` inline | fold into `ext/tests.rs` (or `ext/lift_tests.rs`) |
| `packed.rs` inline + `packed_ext/tests.rs` | under `packed/` (`tests.rs`, `ext_tests.rs`) |
| `ext/tests.rs`, `fp128/tests.rs` | relocate as-is (already separate) |
| `fp32.rs` / `fp64.rs` inline (small) | keep inline (well under cap) |
| `fft.rs` `test_support` + test mods | keep inline (or `fft/tests.rs` if `fft` becomes a dir) |

The only non-relocation test change is the Jolt-coupled pair noted above
(`prime_fields_satisfy_jolt_byte_capabilities`,
`jolt_digest_transcripts_accept_akita_fields`): they move into `compat/jolt.rs`
and gain `#[cfg(all(test, feature = "jolt-compat"))]` (they use the
`jolt-transcript` dev-dep). No other test module acquires a `feature` condition.

### Performance

No effect, by construction. Marker traits compile to nothing; the native
capability impls are the relocated bodies of today's impls; the compat-layer
delegations are `#[inline]` one-liners that the optimizer collapses (and they are
not on any in-workspace hot path — the workspace consumes native traits directly).
The accumulator hot path (`RingAccumulator::fmadd`, `NaiveAccumulator`) is
preserved exactly (see §Design — accumulators). No benchmark is gated on this
change; "no regression" is structural. Sanity smoke:
`AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=20 cargo run --release --example profile`.

## Design

### Architecture

Today (coupling): `arithmetic.rs` re-exports Jolt traits; `jolt_traits.rs` impls
them; `fields/*.rs` import Jolt traits directly.

```text
                 ┌─────────────── jolt_field (git) ───────────────┐
                 │ AdditiveGroup RingCore FieldCore Invertible …   │
                 └───────────────▲───────────────▲────────────────┘
   arithmetic.rs ── pub use ─────┘               │ impl for Akita types
   fields/fp64.rs ── use jolt_field::{…} ────────┤
   jolt_traits.rs ── impl jf::* for Fp* / ext ───┘
        ▲
        │ pub use (names)
   rest of workspace:  F: FieldCore  ==  jolt_field::FieldCore
```

Target (ownership inverted, Jolt optional):

```text
   crates/akita-field/src/
     traits[.rs|/]            NATIVE trait defs (no jolt_field; was arithmetic.rs)
        AdditiveGroup, RingCore, FieldCore, Invertible,
        FromPrimitiveInt, MulPow2, MulPrimitiveInt,
        CanonicalBytes, ReducingBytes, FixedBytes<N>, FixedByteSize,
        CanonicalBitLength, CanonicalU64, RandomSampling,
        TranscriptChallenge, WithAccumulator, AdditiveAccumulator,
        RingAccumulator, NaiveAccumulator
        + (already native) CanonicalField, HalvingField,
          BalancedDigitLookup, PseudoMersenneField, SmoothFftField
     prime/ unreduced/ ext/   impl NATIVE traits  (no jolt_field)
     packed/ fft                 (former fields/ split — see §Module layout)
     compat/
       mod.rs                 #[cfg(feature = "jolt-compat")] pub mod jolt;
       jolt.rs                the ONLY module naming jolt_field:
                              impl jolt_field::* for each Akita type
                              by delegating to native impls
     lib.rs                   re-exports native traits (surface unchanged)

        ▲ pub use (same names)
   rest of workspace:  F: FieldCore  ==  akita_field::FieldCore   (native)

   profile/akita-recursion, jolt-core, jolt-transcript
        └── opt in via  akita-field/jolt-compat  when talking to Jolt
```

**Trait ownership split.**

| Trait(s) | Today | Target | Bridge cost in `compat::jolt` |
| --- | --- | --- | --- |
| `AdditiveGroup`, `RingCore`, `FieldCore` | `pub use jolt` | native (marker) | empty marker impl per type |
| `Invertible` | `pub use` + `fields/*` impl | native | delegate `inverse` |
| `FromPrimitiveInt`, `MulPrimitiveInt`, `MulPow2` | `pub use` + impl | native | delegate `from_u64/i64/u128/i128` (+ markers) |
| `CanonicalBytes`, `ReducingBytes`, `FixedBytes<N>`, `FixedByteSize`, `CanonicalBitLength`, `CanonicalU64` | `pub use` + macro impl | native | delegate byte/bit methods |
| `RandomSampling`, `TranscriptChallenge` | `pub use` + impl | native | delegate `random` / `from_challenge_bytes` |
| `WithAccumulator`, `AdditiveAccumulator`, `RingAccumulator`, `NaiveAccumulator` | `pub use` | native (deepest) | see accumulators below |
| `CanonicalField`, `HalvingField`, `BalancedDigitLookup`, `PseudoMersenneField`, `SmoothFftField` | native | native (unchanged) | n/a (Akita-only) |
| `One`, `Zero` † | `pub use num_traits` | keep `num_traits` | n/a (shared) |
| `Field` umbrella, `OptimizedMul`, `Limbs`, `signed`, `MontgomeryConstants` | not surfaced | stay Jolt-only | referenced only inside compat, if at all |

† The per-type `Zero`/`One`/`Display`/`Hash`/`Sum`/`Product` **impls** (today in
`jolt_traits.rs`) are `std`/`num_traits` impls, **not** Jolt impls — they relocate
to the native field modules, never to `compat/` (see "splitting `jolt_traits.rs`"
below).

**The bridge pattern.** Because native traits keep identical bounds and method
names, the compat impls are mechanical. Markers are empty; methodful traits
delegate:

```rust
// crates/akita-field/src/compat/jolt.rs   (#[cfg(feature = "jolt-compat")])
use jolt_field as jf;
use crate::{Fp64, FromPrimitiveInt, Invertible, /* native traits */};

// marker chain (native impl already satisfies the supertrait bounds)
impl<const P: u64> jf::AdditiveGroup for Fp64<P> {}
impl<const P: u64> jf::RingCore for Fp64<P> {}
impl<const P: u64> jf::FieldCore for Fp64<P> {}

// methodful: forward to the native impl
impl<const P: u64> jf::FromPrimitiveInt for Fp64<P> {
    #[inline] fn from_u64(v: u64) -> Self { <Self as FromPrimitiveInt>::from_u64(v) }
    #[inline] fn from_i64(v: i64) -> Self { <Self as FromPrimitiveInt>::from_i64(v) }
    #[inline] fn from_u128(v: u128) -> Self { <Self as FromPrimitiveInt>::from_u128(v) }
    #[inline] fn from_i128(v: i128) -> Self { <Self as FromPrimitiveInt>::from_i128(v) }
}
impl<const P: u64> jf::Invertible for Fp64<P> {
    #[inline] fn inverse(&self) -> Option<Self> { <Self as Invertible>::inverse(self) }
}
// … CanonicalBytes / ReducingBytes / FixedBytes / RandomSampling / TranscriptChallenge similarly
```

This is the same *shape* as today's `impl_prime_jolt_traits!` macro, but that
macro is **split** rather than retained wholesale (next paragraph): only the
`jolt_field` markers/delegations live in `compat/jolt.rs`, while the method *logic*
(`to_bytes_le`, `from_le_bytes_mod_order`, …) and the `std`/`num_traits` supertrait
impls move to the native side.

**Splitting `jolt_traits.rs`: ~half of it is not Jolt.** Today `jolt_traits.rs`
bundles two unrelated kinds of impl in the same macros (`impl_prime_jolt_traits!`,
`impl_wide_additive!`) and inline blocks:

- **`std`/`num_traits` impls** — `Zero`, `One`, `Display`, `Hash`, `Sum`/`Sum<&>`,
  `Product`/`Product<&>` for every prime and extension type;
  `Zero`/`Add<&Self>`/`Sub<&Self>` for every wide accumulator and for `AccumPair`;
  plus the free fn `reduce_le_bytes_mod_order`. **None name `jolt_field`.** They are
  the **supertrait obligations of the native hierarchy** — native `AdditiveGroup`
  requires `Zero + Add<&> + Sub<&>`; native `RingCore` requires
  `One + Display + Hash + Sum + Sum<&> + Product + Product<&>` (mirroring
  `jolt_field`, Invariant 6). They therefore **move to the native field modules**
  (`prime/`, `ext/`, `unreduced/`) and stay **non-gated**. Parking them behind
  `jolt-compat` would make `impl RingCore for Fp64` fail its own supertraits under
  `--no-default-features` — breaking the final "builds without Jolt" gates.
- **`jolt_field::*` impls** — the markers (`jf::AdditiveGroup`/`RingCore`/
  `FieldCore`/`MulPow2`/`MulPrimitiveInt`/…) and the methodful capability impls
  (`jf::CanonicalBytes`/`ReducingBytes`/`FixedBytes`/`TranscriptChallenge`/
  `WithAccumulator`/…). Only these go to `compat/jolt.rs` (gated). The *logic* in
  the methodful ones (`to_bytes_le`, `from_le_bytes_mod_order`, `num_bits`,
  `to_canonical_u64_checked`, `reduce_le_bytes_mod_order`) moves to the native
  capability impls; compat forwards to them.

So each macro is **split in two**: a native macro beside the concrete types
(emitting the `std`/`num_traits` impls + byte/bit logic on the native traits) and a
thin compat macro in `compat/jolt.rs` (emitting `jf::*` markers + delegations).
The `std` traits that are already native need no relocation — the prime structs
derive `Debug, Clone, Copy, PartialEq, Eq, Default`
([`fp64.rs:25`](../crates/akita-field/src/fields/fp64.rs)) and the extension types
already provide their own `Default` (e.g. `FpExt2`'s hand-written impl) — so only
the hand-written `Zero`/`One`/`Display`/`Hash`/`Sum`/`Product` (+ wide
`Add<&>`/`Sub<&>`) currently inside `jolt_traits.rs` cross over. `AccumPair`'s native `Zero`/`Add<&>`/`Sub<&>`
re-bind their `A: jf::AdditiveGroup` bound to the native `AdditiveGroup`.

Two layout facts make this cheaper than it looks: only the four module-root files
(`fields/{fp32,fp64,fp128,ext}.rs`) name `jolt_field` directly; the deeper
extension impls already resolve `FromPrimitiveInt`/`Invertible`/`RandomSampling`
through crate-root names (`use crate::{…}`), so they follow the native traits the
moment the re-export flips — no per-impl edits there. Compat markers for the
generic extension types must carry the **same** where-bounds as the native impls
(e.g. `F: FieldCore + Valid + …`).

**Accumulators (deepest tendril).** `WithAccumulator::Accumulator` points at a
concrete accumulator type. Today the prime fields use
`type Accumulator = jf::NaiveAccumulator<Self>`
([`jolt_traits.rs:90-92`](../crates/akita-field/src/jolt_traits.rs)), and the wide
accumulator *types* (`Fp64ProductAccum`, `Fp128ProductAccum`, …) are already Akita
types that impl `jf::AdditiveGroup`. To own this cleanly, `akita-field` defines
native `AdditiveAccumulator` / `RingAccumulator` / `NaiveAccumulator<R>` (the
`NaiveAccumulator` body is ~30 lines, copied verbatim from `jolt-field`'s
`accumulator.rs`), the wide types impl the native accumulator traits, and
the **native** `WithAccumulator::Accumulator = akita_field::NaiveAccumulator<Self>`.

The Jolt compat trait should preserve Jolt's associated-type identity unless there
is a concrete reason not to: `impl jf::WithAccumulator for Fp*` can keep
`type Accumulator = jf::NaiveAccumulator<Self>` once the compat module also
provides `jf::RingCore + jf::FromPrimitiveInt` for `Self`. That is stronger
interop than routing Jolt callers through Akita's native `NaiveAccumulator`, and
avoids unnecessary foreign-trait impls for the native accumulator. The Akita wide
accumulator types still need their Jolt-side additive marker impls in compat,
because they are local types that Jolt-facing code may name through existing
associated types. This slice is the highest-effort and is sequenced last (it can
ship as its own phase; until then accumulators may remain re-exported from Jolt
under `jolt-compat`).

**`Cargo.toml`.**

```toml
[dependencies]
jolt-field = { git = "…", rev = "…", default-features = false, optional = true }
num-traits = "0.2"
rand_core = { version = "0.6", features = ["getrandom"] }
rayon = { version = "1.10", optional = true }
thiserror = "2.0"
akita-serialization = { version = "0.1.0", path = "../akita-serialization" }

[dev-dependencies]
jolt-transcript = { git = "…", rev = "…", default-features = false }  # compat tests only
rand = "0.8"

[features]
default = ["jolt-compat"]          # keep ON first → zero downstream churn
jolt-compat = ["dep:jolt-field"]
parallel = ["dep:rayon"]
```

Keeping `jolt-compat` in `default` means step-1 lands invisibly. A later,
separate decision flips `default = []` and makes `profile/akita-recursion`
(and only it) request the feature — at which point the broad workspace stops
compiling Jolt entirely.

`jolt-transcript` stays an unconditional `dev-dependency` — Cargo does not allow
`optional` dev-dependencies — so it (and transitively `jolt-field`) remains in the
**test** graph even with `--no-default-features`. The single compat test that uses
it is `#[cfg(all(test, feature = "jolt-compat"))]`, so it is skipped without the
feature. The "Jolt absent from the graph" guarantee (Invariant 3 / acceptance) is
thus about **normal** edges and the shipped library, not dev/test builds.

### Module layout

This change introduces two crate-root modules — `traits/` (the `arithmetic`
rename) and `compat/` (the Jolt seam) — and **retains today's `fields/` umbrella
unchanged**. The trait inversion does not require restructuring `fields/`.

The `fields/` → role-named split is **deferred to a separate future effort**; the
target tree, dependency DAG, and guardrails below are recorded as the eventual
destination, not as work in this change. When that split happens, the §Test layout
relocation table applies and the seam guard (Invariant 3) is re-pointed at the new
module names. The **application plan** for that effort (a stabilize-then-split
sequencing, with a prerequisite workspace-wide public-surface decouple as Phase A)
lives in [`akita-field-fields-split.md`](akita-field-fields-split.md); this section
remains the design of record it references.

Deferred target tree:

```text
akita-field/src/
  traits/        # = arithmetic.rs renamed; native trait hierarchy only (leaf)
  prime/         # fp32, fp64, fp128/, pseudo_mersenne, util
  unreduced/     # = wide/ renamed; limb accumulators + Has{UnreducedOps,Wide,OptimizedFold}, ReduceTo, ScaleI32
  ext/           # fp_ext2, power/tower/ring_subfield, lift (kept whole)
  packed/        # packed core + packed_ext + avx2/ avx512/ neon/
  fft.rs         # (or fft/ if it grows a dir)
  compat/        # jolt adapter (feature-gated; the only jolt_field seam)
  lib.rs         # re-export hub (public surface unchanged)
```

Verified production dependency DAG (acyclic — confirmed against the current tree):

```text
traits        leaf: num_traits, akita_serialization, jolt-only-via-compat
  ↑
prime         → traits
  ↑
unreduced     → traits, prime
  ↑
ext           → traits, prime, unreduced
  ↑
packed        → traits, prime, ext
fft           → traits
compat        → traits, prime, ext, unreduced
```

Confirmations from the current code: prime fields import none of
`ext`/`unreduced`/`packed`; `unreduced` imports only `prime` in non-test code (the
ext-named accumulators are plain limb wrappers whose `HasUnreducedOps` impls live
in `ext/`); `packed.rs` imports extension config/schedule traits, so `packed → ext`
is real and intended (packed extension kernels, not just SIMD primes); `fft.rs`
needs only trait-level bounds (`SmoothFftField`, `Invertible`).

**`lift.rs` stays whole, relocated to `ext/lift.rs` — do not split it.** It defines
`ExtField`/`LiftBase`/`MulBase`/`FrobeniusExtField`/`MulBaseUnreduced`, and the
concrete impls for the extension types live in the `ext` submodules
(`fp_ext2.rs`, `ring_subfield_*`), so `lift` and those submodules are mutually
referential — fine inside one module, a cross-module tangle if the trait defs are
hoisted into `traits/`. `MulBaseUnreduced: ExtField + HasUnreducedOps` would also
drag `HasUnreducedOps` up. Keeping `lift` in `ext/` is consistent with the spec's
scope (`traits/` = exactly today's `arithmetic.rs`; the extension traits are
`fields`-level and stay with the extension types).

**`util`** (`mul64_wide`, `is_pow2_u64`, `log2_pow2_u64`) is used by `prime`
(`fp64`, `fp128`) and `packed` (`packed_neon/fp128`). Put it in `prime/util`;
`packed → prime::util` is acyclic (`packed → prime` already holds). No crate-level
util module is needed.

Four guardrails — each would reintroduce an awkward or cyclic dependency:

1. **Keep `traits/` a leaf.** Do not move `ExtField` or any concrete-type impl
   into it; it must import nothing from `prime`/`ext`/`unreduced`/`packed`.
2. **Leave `Has*` capability traits in their feature modules**
   (`HasUnreducedOps`/`HasWide`/`HasOptimizedFold` in `unreduced/`;
   `HasPacking`/`PackedField`/`PackedValue` in `packed/`). Hoisting them into
   `traits/` cascades (e.g. via `MulBaseUnreduced`).
3. **Keep `unreduced/` free of production `ext` imports** (it is today). The
   ext-named accumulator types stay plain limb wrappers; their impls stay in `ext/`.
4. **Accept `packed → ext`.** Do not try to make `packed/` extension-free; the
   packed extension kernels are intentional.

### Alternatives Considered

1. **Quarantine only (rename `jolt_traits.rs` → `compat/jolt.rs`, make
   `jolt-field` optional, keep the `pub use jolt_field::*` in `arithmetic.rs`
   behind the feature).** Lowest effort and a strict improvement, but it does
   **not** achieve the stated goal: with `arithmetic.rs` still re-exporting Jolt
   traits, `akita_field::FieldCore` is *still* `jolt_field::FieldCore`, so the
   crate cannot build without Jolt and Jolt still owns the trait identity. Good as
   **Phase 0**, insufficient as the endpoint. Adopted as the first phase, not the
   whole spec.

2. **Status quo (do nothing).** Defensible — the coupling is already
   graph-encapsulated and the slim hierarchy is stable. Rejected because it blocks
   ever building/evolving the field layer independently of a Jolt rev, and leaves
   a foreign crate owning Akita's most fundamental abstraction.

3. **Define Akita-native `Zero`/`One` too (full independence from `num_traits`).**
   Rejected (Invariant 6): `num_traits` is tiny, stable, and shared by Jolt, so
   keeping it makes the compat bridge for those bounds empty. Forking them buys
   nothing and adds conversion noise.

4. **Mirror Jolt's `Field` umbrella + `OptimizedMul` natively.** Rejected
   (Non-Goal): Akita uses the slim traits individually and never the umbrella; the
   umbrella's `FixedBytes<32>` bound is BN254-shaped and Akita primes don't (and
   shouldn't) satisfy it.

5. **Blanket bridge `impl<T: akita::FieldCore> jolt::FieldCore for T`.**
   Impossible under the orphan rule (foreign trait, generic type). Per-type impls
   in compat are the only sound option and already exist in macro form.

6. **Land separately from `taghi/refactor/akita-field` (fresh branch off `main`).**
   Considered — it would keep the refactor PR a strict *behavior-free* rename/move
   ([`akita-field-refactor.md`](akita-field-refactor.md) Non-Goals: "No new public
   API, trait, or capability"). **Not chosen:** the decision is to fold this work
   into the refactor PR so the `akita-field` clean-ups land as one reviewable unit.
   The trade-off is explicit and accepted — that PR is no longer a pure refactor,
   and its "no new trait/capability" claim is superseded here (the native-trait +
   `jolt-compat` surface is still name-for-name identical downstream, so consumer
   call sites are unaffected). `akita-field-refactor.md` should get a one-line note
   that the trait-ownership change rides along.

## Documentation

- **This spec** is the design record.
- **`AGENTS.md`** crate description for `akita-field` updated: it owns the field
  trait hierarchy; Jolt interop is the optional `jolt-compat` feature
  (`compat::jolt`). Today's wording ("Implementations of Jolt's slim field
  hierarchy") is reversed.
- **`crates/akita-field`** module docs: a short note on `traits` (native traits,
  renamed from `arithmetic`) vs `compat::jolt` (feature-gated adapter) and the
  "single Jolt seam" rule.
- **`docs/crate-graph.md`** (referenced by PR #65): note that `jolt-field` is now
  an optional edge of `akita-field`, not a mandatory one.
- **`specs/akita-field-refactor.md`** gets a one-line note that this
  trait-ownership change rides along in the same PR, so that PR's "pure refactor /
  no new trait or capability" framing is superseded by this spec.
- No consumer-facing API docs change (surface is name-preserved).

## Execution

Sequenced so the workspace compiles and tests green at the end of every phase.
Lands on `taghi/refactor/akita-field`, folded into the structural-refactor PR
(§Alternatives #6). The refactor's packed-split + `FpExt` rename are already
committed; these phases stack on top of that tree and target today's `fields/*.rs`
paths (the `fields/` umbrella is retained — §Module layout).

**Status (current):** Phases 0–4 are **done and verified** — `akita-field` owns
its entire trait surface natively, `default = []` (jolt-compat is opt-in), and no
workspace crate needs the compat seam. The `profile/akita-recursion` sub-workspace
was rebuilt with compat off: it compiles and its dependency tree is now fully
`jolt-field`-free (the package is gone from its lock). The only remaining deferred
item is the `fields/` → role-named split (§Module layout), explicitly out of scope
here. One unrelated, pre-existing follow-up surfaced during the recursion audit (a
`getrandom`/`HashMap` panic in the planner schedule DP inside the Jolt guest) — it
is independent of this decoupling and tracked separately.

1. **Phase 0 — Quarantine + make optional. ✅ DONE.** `git mv jolt_traits.rs compat/jolt.rs`;
   add `compat/mod.rs` with `#[cfg(feature = "jolt-compat")] pub mod jolt;`; add
   the `jolt-compat` feature (in `default`) and mark `jolt-field` `optional`. Gate
   the moved tests with the feature. Temporarily keep `arithmetic.rs`'s
   `pub use jolt_field::*` *behind* `#[cfg(feature = "jolt-compat")]` (the file is
   still named `arithmetic.rs` until Phase 1). Outcome:
   "Jolt interop behind a feature"; default-feature consumers are unchanged. This
   phase does **not** claim `--no-default-features` support yet — without native
   trait definitions, no-default consumers would lose the re-exported trait names.
2. **Phase 1 — Native markers + module rename. ✅ DONE.** `git mv arithmetic.rs
   traits.rs` (update `lib.rs`: `pub mod arithmetic;` → `pub mod traits;`). Define
   native `AdditiveGroup`/`RingCore`/`FieldCore`/`Invertible` in `traits` (shapes
   mirror `jolt_field` exactly, incl. `RingCore::square` and `Invertible::inv_or_zero`
   defaults); switch `traits`/`lib.rs` to export the native ones; move the marker
   impls for all field types onto native traits; **relocate the `std`/`num_traits`
   supertrait impls (`Zero`/`One`/`Display`/`Hash`/`Sum`/`Product`; wide
   `Add<&>`/`Sub<&>`) out of `compat/jolt.rs`** so native `RingCore`/`AdditiveGroup`
   resolve without `jolt-compat`; reduce `compat/jolt.rs` to `jf::` forwarding.
   Outcome: the core marker hierarchy is Akita-owned under the default build.
   Do not run the package-local no-Jolt acceptance gate here yet: existing native
   Akita traits (`CanonicalField`, `BalancedDigitLookup`, etc.) still mention
   capability traits such as `FromPrimitiveInt`, and the field modules still carry
   direct/indirect Jolt capability impls until Phase 2.

   *As-built (deviations from the sketch above):*
   - The relocated supertrait boilerplate + empty core-algebra markers live in one
     consolidated, Jolt-free module, `fields/native_algebra.rs` (a prime macro,
     per-type ext impls, a wide-accumulator macro, and `AccumPair`). When the
     `fields/` split lands these redistribute to `prime`/`ext`/`unreduced`; until
     then a single interim home is far lower-risk than scattering into 8 files.
   - The non-trivial `RingCore::square` / `Invertible::inverse` impls stay
     **co-located** in the ext modules (`fp_ext2.rs`, `tower_fp_ext4.rs`, …): they
     auto-retarget to the native traits via the `ext.rs` import flip (`use crate::{…
     Invertible, RingCore}` instead of `use jolt_field::{…}`), and `FpExt2::square`
     relies on the private `mul_nr` helper so it cannot move out of `fp_ext2.rs`
     anyway.
   - `compat/jolt.rs` forwards `jf::RingCore::square` and `jf::Invertible::inverse`
     to the native impls (`<Self as crate::RingCore>::square(self)` etc.) so Jolt
     callers observe identical (custom-`square`) behavior, not the slim `self*self`
     default.
   - The four-module Jolt seam is now `fp32.rs` / `fp64.rs` / `fp128.rs` (drop
     `Invertible` from their `use jolt_field`) and `ext.rs` (drop `Invertible`,
     `RingCore`); `compat/jolt.rs` remains the only module naming `jolt_field`
     wholesale.
   - Verified green: `cargo fmt --check`, `cargo clippy --all --all-targets -D
     warnings`, and the full test suite (`akita-field` 155; rest of workspace incl.
     `akita-pcs` end-to-end prove/verify — all `0 failed`).
3. **Phase 2 — Native capability traits. ✅ DONE.** Defined native
   `FromPrimitiveInt`, `MulPow2`, `MulPrimitiveInt`, `CanonicalBytes`,
   `ReducingBytes`, `FixedBytes`, `FixedByteSize`, `CanonicalBitLength`,
   `CanonicalU64`, `RandomSampling`, `TranscriptChallenge` in `traits.rs`; flipped
   the `fields/*.rs` impls to them; reduced `compat/jolt.rs` to delegations. Outcome
   met: the single seam (Invariant 3) now holds for everything except accumulators —
   `--no-default-features` fails *only* on the four accumulator re-exports.

   *As-built (deviations from the sketch above):*
   - `FromPrimitiveInt`/`RandomSampling` already carried per-type logic in the prime
     (`fp32/64/128.rs`) and ext (`ext/*.rs`) modules, so they were retargeted by
     flipping `use jolt_field::{…}` → `use crate::{…}` (no body moves). After the
     flip **none** of `fp32.rs` / `fp64.rs` / `fp128.rs` / `ext.rs` name `jolt_field`.
   - The *derived* prime capabilities (byte/transcript surface + `MulPow2`/
     `MulPrimitiveInt` markers + the `reduce_le_bytes_mod_order` Horner helper) moved
     out of the old compat macro into a new Jolt-free `fields/native_capability.rs`.
   - `compat/jolt.rs` became pure forwarding. Two design points: (a) extension
     forwarding uses a `where Self: crate::Trait` macro (`forward_ext_jolt_traits!`)
     so it tracks the native `*MulBackend` bounds without restating them; (b) the
     seam references natives via **fully-qualified `crate::` paths only** (never
     `use`d) so the `#[cfg(test)]` module's `use super::*` cannot pull a native trait
     into scope and collide with the `jolt_field` trait of the same name on a
     concrete field type (the prime `from_u64`/`from_canonical_u128`/`zero` calls
     resolve to inherent methods, so they are unaffected either way).
   - The gated `pub use jolt_field::{…}` in `traits.rs` now lists **only** the four
     accumulators (`AdditiveAccumulator`/`NaiveAccumulator`/`RingAccumulator`/
     `WithAccumulator`); the Phase-0 capability re-export is gone.
   - Verified green: `cargo fmt --check`, `cargo clippy --all --all-targets -D
     warnings`, and the full workspace test suite (`akita-field` 155; everything else
     incl. `akita-pcs` end-to-end prove/verify — all `0 failed`). Downstream
     `akita-algebra`'s `RandomSampling for CyclotomicRing` retargeted automatically
     via the `akita_field` re-export with no edit.
4. **Phase 3 — Native accumulators. ✅ DONE.** Defined native
   `AdditiveAccumulator` / `RingAccumulator` / `WithAccumulator` + the
   `NaiveAccumulator<R>` struct (with its `Default`/`AdditiveAccumulator`/
   `RingAccumulator` impls) in `traits.rs`, mirroring `jolt_field`; pointed the
   prime fields' native `WithAccumulator::Accumulator` at the native
   `NaiveAccumulator`; kept compat's `jf::WithAccumulator::Accumulator =
   jf::NaiveAccumulator<Self>` unchanged. Outcome met: `cargo build -p akita-field
   --no-default-features [--all-targets]` builds the **entire** crate, and
   `cargo tree -p akita-field --no-default-features -e normal` shows **no** `jolt-*`.

   *As-built (deviations from the sketch above):*
   - **"impl native accumulator traits for the wide types" did not apply.** The wide
     product accumulators (`Fp32ProductAccum`, `Fp64x4i32`, …) are *P-agnostic*: a
     single value reduces via a const-generic `reduce::<P>() -> Fp{32,64,128}<P>`,
     so they have no fixed `AdditiveAccumulator::Element` and cannot impl the
     jolt-shaped accumulator traits. Their accumulator role is the additive algebra
     (`AdditiveGroup`), already native since Phase 1. The fixed-`Element` accumulator
     is `NaiveAccumulator<R>`, exactly as in `jolt_field`.
   - **No compat churn for accumulators.** Since compat keeps `jf::NaiveAccumulator`
     (which `jolt_field` already endows with its accumulator-trait impls), the seam's
     prime `jf::WithAccumulator` impl is untouched; no `jf::AdditiveAccumulator` /
     `jf::RingAccumulator` forwarding is needed. The two accumulator worlds (native vs
     Jolt) never meet in one bound, so the associated-type-identity risk is moot.
   - `traits.rs` no longer contains any `jolt_field` *code* (the gated accumulator
     re-export is gone); the only residual mentions are doc-comment references. The
     single code seam is now `compat/jolt.rs` (+ the gated `mod jolt;` in
     `compat/mod.rs`). `lib.rs`'s re-export list is unchanged — the four accumulator
     names now resolve to the native definitions.
   - Verified green: `cargo fmt --check`; `cargo clippy -p akita-field --all-targets`
     and `… --no-default-features --all-targets` (both `-D warnings`); workspace
     `cargo clippy --all --all-targets -D warnings`; `akita-field` tests **155**
     (default) / **153** (`--no-default-features`, the 2 `jolt-compat` tests gate
     out); full workspace `cargo test` — all `0 failed`.
5. **Phase 4 — Flip the default. ✅ DONE.** Set `default = []` (dropped
   `jolt-compat` from default). Outcome met: the broad workspace no longer resolves
   `jolt-field` on normal edges and the recursion sub-workspace still compiles
   against native `akita_field::{FieldCore, RandomSampling}`.

   *As-built (deviations from the sketch above):*
   - **No consumer needed `akita-field/jolt-compat`.** The sketch said "make
     `profile/akita-recursion` request `akita-field/jolt-compat`", but a repo-wide
     audit found `jolt_field` is named in **only** `crates/akita-field/src/compat/**`
     (+ doc comments) and nowhere else — no workspace crate and nothing in
     `profile/akita-recursion` (`glue`/`artifact`/`host`/`guest`) uses it. The flip
     required **zero** consumer edits; nothing requests the feature back.
   - **Main `Cargo.lock`: no churn.** `jolt-field` stays in the main lock via
     `akita-field`'s `jolt-transcript` dev-dep, and the lock does not record feature
     selections, so flipping `default` changed nothing there.
   - **Recursion sub-workspace lock synced (this *is* the Phase-4 work).** With
     compat off and no dev-dep to retain it, `akita-field` drops its `jolt-field`
     edge and the `jolt-field` package is removed from the recursion lock entirely
     (jolt-core uses arkworks, not `jolt-field`). The same resync records the
     `akita-types → num-traits` edge the **main** lock already carries
     (`akita-types/Cargo.toml` declares `num-traits`) — committed delta, not spurious
     churn; it had been reverted during Phases 0–3 precisely because it "rides along
     with Phase 4".
   - Verified green: `cargo clippy --workspace --all-targets -- -D warnings` (exit 0);
     `akita-field` tests **156** (default, compat off) / **158** (`--features
     jolt-compat`); `cargo tree -e normal` shows no `jolt-field`; recursion
     `cargo build --release` compiles and `artifact` (nv=20) reports `host-side
     verify OK` + `decoded-blob verify OK`.

**`fields/` split — deferred (not part of this work).** The `fields/` →
`prime`/`unreduced`/`ext`/`packed`/`fft` split (§Module layout) is **out of scope**
here: Phases 0–4 run against today's `fields/*.rs` paths and keep the `fields/`
umbrella. When the split is eventually done (separate PR), the §Test layout
relocation table and §Module layout guardrails apply, and the seam guard
(Invariant 3) is re-pointed at the new module names. Its sequenced application plan
is [`akita-field-fields-split.md`](akita-field-fields-split.md).

**Risks to resolve first:**

- **Accumulator associated-type identity. ✅ Resolved in Phase 3.** Native
  `WithAccumulator` uses the native `NaiveAccumulator`; Jolt compat keeps
  `jf::NaiveAccumulator`. The two never appear in the same bound (compat needs no
  native accumulators, and nothing else in the workspace uses the accumulator
  traits at all), so no trait-bound conflict can arise.
- **Supertrait obligations are non-Jolt. ✅ Resolved in Phase 1.** The native
  `RingCore`/`AdditiveGroup` markers only compile if
  `Zero`/`One`/`Display`/`Hash`/`Sum`/`Product` (and wide `Add<&>`/`Sub<&>`) exist
  natively. These were split out of the Jolt macros in `jolt_traits.rs` into
  `fields/native_algebra.rs` (Jolt-free); none remain in `compat/`.
- **Dev-dep keeps Jolt in the test graph.** `jolt-transcript` is a plain
  `dev-dependency` and Cargo forbids `optional` dev-deps, so `jolt-field` stays in
  the *test* graph even under `--no-default-features`. Scope the "no Jolt"
  assertion to normal edges (`cargo tree … -e normal`) / the shipped lib; the
  compat test that needs it is `#[cfg(all(test, feature = "jolt-compat"))]` and
  simply does not run otherwise.
- **Hidden `jolt_field` method reliance. ✅ Resolved in Phase 2.** The only
  `jolt_field` imports under `fields/**` were `FromPrimitiveInt` / `Invertible` /
  `RandomSampling` (all mirrored), now flipped to `crate::`; no module under
  `fields/**` names `jolt_field` anymore.
- **`FixedBytes<N>` const-generic forwarding. ✅ Resolved in Phase 2.** Native
  `FixedByteSize::NUM_BYTES` and `FixedBytes<N>` carry `4`/`8`/`16` per prime type
  (`native_capability.rs`); the compat seam forwards `jf::FixedByteSize::NUM_BYTES`
  to the native const and impls `jf::FixedBytes<4|8|16>` over the matching `N`, so
  the round-trip is exact (covered by `prime_fields_satisfy_jolt_byte_capabilities`).

## References

- [`crates/akita-field/src/traits.rs`](../crates/akita-field/src/traits.rs)
  — native trait hierarchy (post-Phase-3: owns the **entire** surface — core
  `AdditiveGroup`/`RingCore`/`FieldCore`/`Invertible`, the capability traits
  `FromPrimitiveInt`/`MulPow2`/`MulPrimitiveInt`/`CanonicalBytes`/`ReducingBytes`/
  `FixedBytes`/`FixedByteSize`/`CanonicalBitLength`/`CanonicalU64`/`RandomSampling`/
  `TranscriptChallenge`, and the accumulators `AdditiveAccumulator`/`RingAccumulator`/
  `WithAccumulator` + the `NaiveAccumulator<R>` struct). No `jolt_field` code remains
  (doc-comment mentions only). Was `arithmetic.rs`.
- [`crates/akita-field/src/fields/native_algebra.rs`](../crates/akita-field/src/fields/native_algebra.rs)
  — consolidated native supertrait impls + core-algebra markers (Phase 1).
- [`crates/akita-field/src/fields/native_capability.rs`](../crates/akita-field/src/fields/native_capability.rs)
  — native prime derived-capability impls: byte/transcript surface, `MulPow2`/
  `MulPrimitiveInt` markers, the `reduce_le_bytes_mod_order` helper (Phase 2), and
  the prime `WithAccumulator → NaiveAccumulator` association (Phase 3).
- [`crates/akita-field/src/compat/jolt.rs`](../crates/akita-field/src/compat/jolt.rs)
  — the single Jolt seam: `jf::` forwarding to native (markers + method delegation +
  `where Self:` ext forwarder), referencing natives via fully-qualified `crate::`
  paths only. Was `jolt_traits.rs`.
- [`specs/akita-crate-followup-jolt-integration.md`](akita-crate-followup-jolt-integration.md)
  (PR #65) — the deliberate adoption of Jolt traits this spec evolves; Invariants
  5–6 (avoid BN254 path, keep Akita-specific helpers separate) are upheld here.
- [`specs/general-field-support.md`](general-field-support.md) (PR #60) — the
  general-field direction; `ExtField`/`LiftBase` are Akita-native already and are
  unaffected.
- [`specs/akita-field-refactor.md`](akita-field-refactor.md) — the *structural*
  `akita-field` refactor on `taghi/refactor/akita-field`; this spec is the
  *trait-ownership* counterpart and is deliberately separate.
- `jolt-field` slim hierarchy (rev `2509bdc`): crate root doc and
  `field_core.rs` / `ring_core.rs` / `additive_group.rs` / `field.rs` /
  `invertible.rs` / `accumulator.rs` in
  `~/.cargo/git/checkouts/jolt-…/crates/jolt-field/src/` — the trait shapes the
  native definitions mirror.
- `AGENTS.md` — "Verifier No-Panic Contract" (Invariant 5) and the
  "no backward-compatibility guarantees" rule (justifies a clean cutover of the
  trait definition site).
