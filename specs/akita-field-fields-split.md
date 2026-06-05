# akita-field `fields/` role-named split

## Status

**Phase A DONE ✅. Phase B DONE ✅ (step 6 deferred by design).** Prerequisite
met: the trait-ownership decoupling
([`akita-field-jolt-decoupling.md`](akita-field-jolt-decoupling.md),
Phases 0–4) is **done** (`akita-field` owns its trait surface natively;
`jolt-compat` is opt-in).

Phase A landed the curated public surface: the `packed`, `unreduced`, and `fft`
facade modules exist at the `akita_field` crate root, every downstream crate
imports through them (or the root vocabulary), and `pub mod fields` is now
`pub(crate)`. The deep `akita_field::fields::*` umbrella is no longer public API.

Phase B reorganized the internal `fields/` tree into role-named modules
(`prime/`, `unreduced/`, `ext/`, `packed/`, `fft`) — a pure move with zero
behavior change and zero downstream impact (the public facades repointed their
*source* paths only; every exported name is unchanged). The optional crate-root
hoist (step 6) stays deferred: `fields/` remains the internal wrapper.

This spec owns the **application plan** for the split. The **design** (target
module tree, dependency DAG, the four guardrails, and the test-relocation table)
already lives in
[`akita-field-jolt-decoupling.md` §Module layout / §Test layout](akita-field-jolt-decoupling.md)
and is **referenced, not duplicated**, here.

### As-built (Phase A)

- **Facades created:** `akita_field::packed`, `akita_field::unreduced`,
  `akita_field::fft` (inline `pub mod` re-export blocks in `lib.rs` sourcing from
  `crate::fields::{packed, packed_ext, wide, fft}`). **No** `prime`/`ext` facades
  were added — the common field/prime/ext vocabulary stays at the root per the
  migration table, so those two facades from the original sketch were unnecessary.
  Facade names are stable; Phase B repoints their *source* paths only.
- **Root trimmed:** the packed family (`PackedField`, `PackedValue`, `HasPacking`,
  `NoPacking`, `Fp{32,64,128}Packing`) and the wide/unreduced family
  (`HasWide`, `HasUnreducedOps`, `HasOptimizedFold`, `ReduceTo`, `AccumPair`,
  `FoldMatrix*`, `*ProductAccum`, `*x*i32`, `Fp128MulU64Accum`) were **removed**
  from the `akita_field` root — they live **only** in their facade now (no
  dual-homing). Field/prime/ext/lift vocabulary stays rooted.
- **Refused consts demoted:** every `pseudo_mersenne` per-prime `*_MODULUS` /
  `*_OFFSET` const is now `pub(crate)`; the only consumers (`akita-pcs`
  field-arith benches) were rewritten to `<Prime* as HasPacking>::Packing`.
- **`fields` sealed → `pub(crate)`**, and all of its submodules
  (`pub mod` → `pub(crate) mod`), so the deep paths are gone. The fft test-only
  scanners `field_pow_u128` / `find_primitive_nth_root` were gated `#[cfg(test)]
  pub(crate)` (they were never production API; dead in non-test builds once the
  deep path closed). NEON/AVX2/AVX-512 backend `FP*_WIDTH` consts and `pub use`
  globs demoted to `pub(crate)`.
- **Umbrella mirrored:** `akita-pcs` re-exports the three facade modules
  (`pub use akita_field::{fft, packed, unreduced};`) and dropped the flattened
  packed symbols from its root, so it faithfully mirrors `akita-field`.
- **Internal seam fixes (not in the original plan):** several `akita-field`
  internal modules reached the wide-accumulator structs through the now-removed
  `fields`-level re-export (`native_algebra.rs`, `compat/jolt.rs`) or the crate
  root (`lift.rs`); all were repointed at `crate::fields::wide::*` directly.
- **Grep caveat learned:** nested-group imports like
  `akita_field::{fields::Prime128Offset275, …}` (one site in `akita-types`) and
  root-path users like `akita_field::{…, PackedField}` are **not** caught by an
  `akita_field::fields::` grep; the compiler (`cargo build --keep-going`) was the
  authoritative migration driver.
- **Verified:** `cargo clippy --workspace --all-targets -- -D warnings` green;
  `akita-field` × {default, `jolt-compat`} clippy `-D warnings` green; cross-arch
  `x86_64-apple-darwin` × {`+avx2`, `+avx512f,+avx512dq`} clippy `-D warnings`
  green; native aarch64/NEON green; `rg "fields::"` shows no matches outside
  `crates/akita-field/`.

### As-built (Phase B)

Executed in the DAG order below, green at every step.

- **`wide/` → `unreduced/`** (step 1): `git mv` of the dir; internal
  `super::wide` / `crate::fields::wide` refs repointed; the `unreduced` facade
  source in `lib.rs` now reads `crate::fields::unreduced::{…}`.
- **`prime/`** (step 2): `fp32.rs`, `fp64.rs`, `fp128.rs` + `fp128/`,
  `pseudo_mersenne.rs`, `util.rs` moved under `prime/` with a new `prime/mod.rs`
  re-export hub. Internal users updated to `super::prime::{…}` /
  `crate::fields::prime::{util, pseudo_mersenne}::…`. *Gotcha:* the lib-only
  build surfaced the real break (`ext.rs`'s `use super::{fp128, fp32, fp64}`);
  `--all-targets` had masked it behind 90+ downstream trait-bound errors. Always
  diagnose with `--lib` first.
- **`lift.rs` → `ext/lift.rs`** (step 3): kept whole; `lift` re-exported via
  `ext::lift`.
- **`packed/`** (step 4): packed core → `packed/mod.rs`; `packed_ext` →
  `packed/ext`; `packed_avx2`/`packed_avx512`/`packed_neon` →
  `packed/{avx2,avx512,neon}` with their `cfg(target_arch/target_feature)` gates
  preserved verbatim; the `packed` facade ext source now reads
  `crate::fields::packed::ext::{…}`.
- **Redistribute interim consolidations** (step 5): the decoupling's
  `native_algebra.rs` was split — prime macro + invocations → `prime/native_algebra.rs`,
  per-type ext impls → `ext/native_algebra.rs`, wide-accumulator macro + `AccumPair`
  → `unreduced/native_algebra.rs`; `native_capability.rs` (prime derived-capability
  impls + `reduce_le_bytes_mod_order`) → `prime/native_capability.rs`.
- **Step 6 deferred** (default): `fields/` stays the internal wrapper; no
  crate-root hoist.
- **Docs/guard** (step 7): no seam-guard test hardcodes the old module names
  (`rg` over `crates/` for `packed_avx*`, `fields::wide`, `::wide::` is clean);
  stale `src/algebra/fields/fft.rs` doc paths in `traits.rs` and
  `prime/fp128/primes.rs` corrected to `src/fields/fft.rs`. `AGENTS.md`'s
  "wide/packed helpers" line is conceptual (not a path) and left unchanged.
- **Verified:** `cargo fmt --check` clean; `akita-field` clippy `-D warnings` ×
  {default `--all-targets`, `--no-default-features --lib`, `jolt-compat
  --all-targets`} green; cross-arch `x86_64-apple-darwin` × {`+avx2`,
  `+avx512f,+avx512dq`} `--lib` clippy `-D warnings` green; native aarch64/NEON
  green; `scripts/check-rust-file-lines.sh --no-baseline` passes (largest file
  `fft.rs` at 1088, pre-existing; all moved/new files well under the 1500 cap);
  `cargo build --workspace --all-targets` + `cargo test --workspace` green.

## Goal / non-goals

- **Goal.** Reorganize `crates/akita-field/src/fields/` from its mixed flat-file
  layout into role-named modules — `prime/`, `unreduced/` (= `wide/` renamed),
  `ext/`, `packed/`, `fft` — per the decoupling spec's deferred target tree.
  Enforces the layering DAG, improves navigation, and relieves the 1500-line
  file cap (which already forced `fp128` into a directory).
- **Pure move, zero behavior change** — except one deliberate public-surface
  tightening (Phase A3, called out explicitly).
- **Non-goal.** Any algebra/serialization/performance change; any new
  trait/capability; the `traits/` and `compat/` crate-root modules (already
  landed in the decoupling) are untouched.

## Key finding that drives the approach

`fields::` submodule paths are **de-facto public API** — downstream crates reach
*into* the module tree, not just the crate root:

- `akita_field::fields::wide::{HasWide, ReduceTo, HasOptimizedFold}` and
  `akita_field::fields::HasUnreducedOps` — used across `akita-prover`,
  `akita-sumcheck`, `akita-setup`, `akita-algebra`, `akita-pcs`.
- `akita_field::fields::packed_ext::*`, `akita_field::fields::fft::*`,
  `akita_field::fields::pseudo_mersenne::*`, `akita_field::fields::fp32::Fp32`
  — in `akita-pcs` benches/examples and `akita-prover` tests.

A naive "dissolve `fields/` + rename `wide → unreduced`" therefore breaks imports
in ~5 downstream crates. The repo's "no backward-compatibility" rule *permits*
that, but it needlessly couples a public-API break to an internal move.

**The unlock:** `lib.rs` already re-exports *most* of the commonly used field
vocabulary at the `akita_field` root (`ReduceTo`, `HasWide`, `HasUnreducedOps`,
`HasOptimizedFold`, `Fp32/64/128`, all `Prime*`, ext types). Only specialized
surfaces (`packed_ext`, `fft`) and some `pseudo_mersenne::*` glob members need a
curation decision. So the public surface can be **decoupled from the module tree
cheaply**, after which the split is an `akita-field`-internal operation.

## Intended public API (the target surface)

Design the surface first; Phase A then *curates* toward it (it does not merely
preserve whatever is imported today).

**Principle — meaningful public namespaces, not implementation paths.** The
public surface is a small ergonomic root plus curated, role-named modules:
`akita_field::{...}` for the common vocabulary (field/prime/extension types and
core traits) and the specialized facades `akita_field::{unreduced, packed, fft}`
for the wide-accumulator, packing, and FFT surfaces. *(As built, prime and ext
stayed root vocabulary, so no `prime`/`ext` facade modules were needed.)* These
modules are **facades**: their names are semantic API categories, while the
actual implementation layout can remain under `fields/` (or later move to
crate-root directories) without changing call sites. The old
`akita_field::fields::*` umbrella is not meaningful API and should disappear.

This is cleaner than a flat root: algorithms like FFTs and packed-extension
kernels are real public concepts, but they are specialized enough to deserve a
namespace. The crate root stays readable and still carries the common field
language (`FieldCore`, `Fp128`, named primes, extension types, core traits).

**Canonical call shape.** Common protocol code imports the usual vocabulary from
the root:

```rust
use akita_field::{
    FieldCore, CanonicalField, RandomSampling,
    Fp32, Fp64, Fp128,
    Prime31Offset19, Prime64Offset59, Prime128Offset275,
    FpExt2, TowerBasisFpExt4, RingSubfieldFpExt4,
};
```

Specialized code imports from the semantic facade that owns the concept:

```rust
use akita_field::fft::{primitive_nth_root, rs_extend_fft, SmoothDomain};
use akita_field::packed::{HasPacking, PackedField, PackedFpExt2};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
```

The public module name explains the abstraction. It does not expose the physical
file path or force consumers through a catch-all `fields` module.

**Idioms — express things through abstractions, not parameters.**

- *Packed form of a field* → `<F as HasPacking>::Packing` (e.g.
  `<Prime31Offset19 as HasPacking>::Packing`), **not** `Fp32Packing<{ RAW_MODULUS }>`.
- *A specific prime* → the named type (`Prime31Offset19`), never its modulus const.

```rust
// leaks the const-generic parameter (today):
type P31 = Fp32Packing<{ PRIME31_OFFSET19_MODULUS }>;
// pure public abstraction (target):
type P31 = <Prime31Offset19 as HasPacking>::Packing;
```

**Curated out (deliberately not public):**

- `pseudo_mersenne` per-prime `*_MODULUS` / `*_OFFSET` consts — implementation
  parameters behind the named `Prime*` types; reach them via the named type +
  `HasPacking`.
- `fields` and `util` paths — `pub(crate)`. Public paths are the semantic
  facades (`prime`, `ext`, `unreduced`, `packed`, `fft`), not the implementation
  wrapper.
- fft internal helpers not part of the contract (candidates: `field_pow_u128`,
  `find_primitive_nth_root`) — stay `pub(crate)` unless a real consumer needs them.

**Namespace decision — curated public modules.** Use `akita_field::fft::{...}`,
`akita_field::packed::{...}`, and `akita_field::unreduced::{...}` for specialized
surfaces. Keep root re-exports for the common field vocabulary and high-use
convenience types. Avoid both extremes: no catch-all `fields::*` public module,
and no ever-growing flat root containing every helper function.

## Approach: stabilize-then-split

Two phases, so the workspace-wide churn and the internal move never mix. **Phase
A is independently valuable and landable on its own.**

### Phase A — decouple the public surface (workspace-wide, mechanical, low-risk) — DONE ✅

- **A1. Curate the public facades** ✅ in `crates/akita-field/src/lib.rs` toward
  §Intended public API — promote real abstractions, refuse leaked parameters:
  - Keep root re-exports for common field vocabulary (field traits, `Fp*`,
    named primes, core extension types, high-use capability traits).
  - **Promote** `packed_ext` types under the `packed` facade:
    `PackedFpExt2`, `PackedTowerBasisFpExt4`,
    `PackedPowerBasisFpExt4`, `PackedRingSubfieldFpExt4`,
    `PackedRingSubfieldFpExt8` (packed counterparts of already-public ext types).
  - **Promote** the fft contract under the `fft` facade: `SmoothDomain`, `field_pow`,
    `primitive_nth_root`, `rs_extend_fft`. Keep `field_pow_u128` /
    `find_primitive_nth_root` `pub(crate)` unless a real consumer needs them.
  - **Refuse** `pseudo_mersenne` per-prime `*_MODULUS` / `*_OFFSET` consts: do
    **not** export. They are const-generic parameters behind the named `Prime*`
    types — rewrite consumers to `<Prime* as HasPacking>::Packing` (the named
    `Prime*` types and `PRIME_OFFSET_{MAX,IMPLEMENTED_MAX_BITS,SPECS}` stay
    rooted as before).
- **A2. Migrate downstream imports** ✅ to the curated surface — mostly path
  moves from the catch-all umbrella to a semantic facade
  (`akita_field::fields::wide::HasWide` → `akita_field::unreduced::HasWide`,
  `akita_field::fields::fft::SmoothDomain` → `akita_field::fft::SmoothDomain`),
  but a few are **rewrites** to the meaningful idiom (the `pseudo_mersenne`
  modulus consts → `<Prime* as HasPacking>::Packing`; table below). Per-crate,
  staying green.
  Authoritative list at implementation time: `rg "akita_field::fields::"` across
  the workspace (the table below is representative, gathered from that grep).
- **A3. Tighten the surface.** ✅ Demote `pub mod fields` → `pub(crate) mod fields`
  in `lib.rs` (and submodules `pub` → `pub(crate)` where they were public only
  for cross-crate reach). The public field surface is then **exactly** the root
  convenience re-exports plus the curated facade modules.
  - *This is the one behavior-affecting step* (a deliberate API tightening, not a
    pure move) — flag it in the commit/PR description.
  - **Gate:** `cargo build --workspace --all-targets` + `cargo test --workspace`
    green; `rg "akita_field::fields"` returns no **code** matches outside
    `crates/akita-field/` (doc/spec mentions excepted).

### Phase B — internal split (akita-field-only, zero downstream impact) — DONE ✅ (step 6 deferred)

DAG order (leaves first), green at every step:

1. ✅ **`wide/` → `unreduced/`** — rename the dir + module; update internal
   `crate::fields::wide` refs and the root re-export source in `lib.rs`/`fields/mod.rs`.
2. ✅ **`prime/`** ← `fp32.rs`, `fp64.rs`, `fp128.rs` + `fp128/`,
   `pseudo_mersenne.rs`, `util.rs` (`util` → `prime/util`; `packed → prime::util`
   stays acyclic).
3. ✅ **`lift.rs` → `ext/lift.rs`** — kept whole (mutually referential with the
   ext types; see decoupling spec §Module layout).
4. ✅ **`packed/`** ← packed core + `packed_ext` + `packed_avx2`/`packed_avx512`/
   `packed_neon` (preserve the `cfg(target_arch/target_feature)` gates verbatim).
5. ✅ **Redistribute** the decoupling's interim consolidations:
   `native_algebra.rs` (prime macro → `prime/`; per-type ext impls → `ext/`;
   wide-accumulator macro + `AccumPair` → `unreduced/`) and `native_capability.rs`
   (prime derived-capability impls + `reduce_le_bytes_mod_order` helper → `prime/`).
6. ⏸️ *(Optional, cosmetic — deferred decision.)* Dissolve the internal `fields/`
   wrapper, hoisting implementation directories to `src/` beside `traits/` +
   `compat/`. The public facade modules stay unchanged either way. **Default:
   keep `fields/` as the internal wrapper** (lower churn); hoist only if the flat
   crate-root implementation tree from the decoupling spec is explicitly wanted.
7. ✅ **Re-point the seam guard** (Invariant 3 module names) and update `AGENTS.md`
   + module docs. *(As built: no seam-guard test hardcodes the old names, so
   nothing to re-point; corrected two stale `src/algebra/fields/fft.rs` doc
   paths; `AGENTS.md`'s conceptual "wide/packed helpers" line left unchanged.)*

### Verification gates (run after each Phase-B step, and finally)

- `cargo build`/`clippy`/`test -p akita-field` × {default, `--no-default-features`,
  `--features jolt-compat`} (clippy `-D warnings`).
- Cross-arch packed: `cargo check -p akita-field --lib --target x86_64-apple-darwin`
  (AVX2/AVX-512) plus the native aarch64/NEON build.
- `scripts/check-rust-file-lines.sh --no-baseline` (1500-line cap).
- Final: `cargo test --workspace`; seam grep guard.

## Phase-A2 downstream migration table

Disposition legend: **rooted / facade** = the symbol already has a meaningful
public home or gets one through a facade; **promote** = add a curated facade
re-export (A1), then migrate; **refuse + rewrite** = keep the symbol internal,
rewrite the caller to the meaningful idiom (e.g. `HasPacking`).

| Deep path (today) | Target public API | Status | Downstream sites |
| --- | --- | --- | --- |
| `fields::wide::{HasWide, ReduceTo, HasOptimizedFold}` | `akita_field::unreduced::{…}` (root convenience optional for high-use traits) | rooted / facade | `akita-algebra` ring/cyclotomic; `akita-prover` compute, protocol/flow, backend/{onehot/mod, field_reduction, sparse_ring, multilinear_polynomial, onehot/column_sweep, onehot/inner_ajtai}, protocol/sumcheck/{akita_stage1, akita_stage2}, protocol/extension_opening_reduction/mod, tests/extension_opening_reduction; `akita-pcs` examples/profile/{modes, workload}, benches/onehot_root_projection_commit |
| `fields::HasUnreducedOps` | `akita_field::unreduced::HasUnreducedOps` (root convenience optional) | rooted / facade | `akita-sumcheck` compact_fold; `akita-prover` protocol/sumcheck/two_round_prefix/{stage1, stage2, common}, …/{akita_stage1, akita_stage2}, protocol/flow, protocol/extension_opening_reduction/mod, tests/extension_opening_reduction |
| `fields::wide::HasWide` | `akita_field::unreduced::HasWide` (root convenience optional) | rooted / facade | `akita-setup` src/lib |
| `fields::packed_ext::{PackedFpExt2, PackedPowerBasisFpExt4, PackedTowerBasisFpExt4}` | `akita_field::packed::{…}` | **promote** | `akita-pcs` benches/field_arith/{ext2, ext4} |
| `fields::fft::{field_pow, primitive_nth_root, rs_extend_fft, SmoothDomain}` | `akita_field::fft::{…}` | **promote** | `akita-pcs` benches/fft_smooth |
| `fields::pseudo_mersenne::*` (per-prime `*_MODULUS` consts) | `<Prime* as HasPacking>::Packing` | **refuse + rewrite** | `akita-pcs` benches/field_arith/{base, kernel, cases} |
| `fields::fp32::Fp32` | `akita_field::Fp32` (or `akita_field::prime::Fp32` when namespacing matters) | rooted / facade | `akita-pcs` benches/field_arith/cases |
| `fields::{Prime24Offset3, Prime32Offset99, TowerBasisFpExt4, TwoNr, UnitNr, RingSubfieldFpExt4, …}` | `akita_field::{…}` | rooted | `akita-prover` backend/dense, protocol/extension_opening_reduction/sparse/tests, backend/onehot/tests |

## Decisions

- **Approach:** stabilize-then-split (Phase A → Phase B). *Agreed.*
- **Curated, meaningful exports:** the public surface is intentional, not a
  snapshot of deep-path usage — **promote** real abstractions (`packed_ext`
  types, the fft contract) and **refuse** leaked const-generic parameters
  (`pseudo_mersenne` `*_MODULUS` consts → `<Prime* as HasPacking>::Packing`).
  *Agreed.*
- **Namespacing:** use curated public modules plus root conveniences for common
  field vocabulary. Do not use a catch-all public `fields` module, and do not
  flatten every specialized helper into the root. *Agreed; as built the
  specialized facades are `unreduced`, `packed`, `fft`, while prime/ext/field
  vocabulary stayed at the root (no `prime`/`ext` facade was needed).*
- **End-state:** keep `fields/` as the internal implementation wrapper by
  default; dissolving implementation directories to the crate root is
  deferred/optional (cosmetic after Phase A because public facades are stable).
- **Phase A3 public tightening:** intended and accepted (deep `fields::` paths
  should not be public API).

## References

- [`akita-field-jolt-decoupling.md`](akita-field-jolt-decoupling.md) — §Module
  layout (target tree, verified acyclic DAG, four guardrails, `lift`/`util`
  placement) and §Test layout (co-located-tests rule, relocation table). The
  design of record for the target structure.
- `AGENTS.md` — "no backward-compatibility guarantees" (justifies the Phase A3
  surface tightening); the verifier no-panic contract is not implicated (pure
  move, no verifier-reachable logic changes).
