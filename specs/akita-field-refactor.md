# Spec: akita-field Structural Refactor — Packed-Backend Modularization + `FpExt{N}` Extension Rename

| Field     | Value                                                       |
| --------- | ----------------------------------------------------------- |
| Author(s) | Taghi Badakhshan                                            |
| Created   | 2026-06-01                                                  |
| Status    | implemented (Part 1 committed `26e9745e`; Part 2 in worktree) |
| PR        | `taghi/refactor/akita-field`                                |
| Base      | `main`                                                      |

## Summary

This PR is a two-part **pure refactor** of `akita-field` with no
behavioral, proof-byte, transcript, layout, or performance change. It
bundles two mechanical clean-ups that accumulated as the SIMD and
small-field optimization passes layered onto the crate:

1. **Packed-backend modularization** (committed as `26e9745e`). The
   SIMD optimization passes had grown `packed_avx2.rs`, `packed_avx512.rs`,
   and `packed_neon.rs` past the 1500-line file cap enforced by
   [`scripts/check-rust-file-lines.sh`](../scripts/check-rust-file-lines.sh).
   Each monolith bundled four independent per-width backends
   (`Fp16`/`Fp32`/`Fp64`/`Fp128`). This change extracts the
   `Fp16`/`Fp64`/`Fp128` backends into `fp16`/`fp64`/`fp128` submodules —
   mirroring the `fp32` split that already existed — leaving each root
   with only shared intrinsics, module declarations, and re-exports.

2. **`FpExt{N}` extension-field rename** (this change). The scalar prime
   fields are named by **bit width** (`Fp16`, `Fp32`, `Fp64`, `Fp128`),
   while the extension fields are named by **extension degree** (`Fp2`,
   and the degree-4/8 families `PowerBasisFp4`, `TowerBasisFp4`,
   `RingSubfieldFp4`, `RingSubfieldFp8`). Two unrelated quantities share
   one `Fp{N}` surface form, which is a persistent source of confusion.
   This change renames every extension-degree marker `Fp{2,4,8}` →
   `FpExt{2,4,8}` (and the snake-case `fp{2,4,8}` → `fp_ext{2,4,8}`)
   across types, traits, configs, multiplication backends, packed
   wrappers, free functions, test names, module names, and source-file
   names, so that "Ext" unambiguously denotes an extension and `Fp{bits}`
   is reserved for prime fields.

Neither part changes a single line of arithmetic, a constant, a memory
layout, or a public algorithm. Part 1 is a byte-identical code move;
Part 2 is a 1:1 identifier substitution.

## Intent

### Goal

Make the `akita-field` source tree structurally uniform and unambiguously
named without altering any runtime behavior: every per-architecture
packed backend is split along the same per-width axis, and every
extension-field identifier carries an explicit `Ext` marker that cannot
be mistaken for a prime-field bit width.

The full rename map (Part 2):

| Concept                       | Before                                                  | After                                                            |
| ----------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| Quadratic extension           | `Fp2`, `Fp2Config`                                      | `FpExt2`, `FpExt2Config`                                         |
| Quartic, power basis          | `PowerBasisFp4{,Config,MulBackend}`                     | `PowerBasisFpExt4{,Config,MulBackend}`                          |
| Quartic, tower basis          | `TowerBasisFp4{,Config}`                                | `TowerBasisFpExt4{,Config}`                                     |
| Quartic, ring subfield        | `RingSubfieldFp4{,MulBackend}`                          | `RingSubfieldFpExt4{,MulBackend}`                               |
| Octic, ring subfield          | `RingSubfieldFp8{,MulBackend}`                          | `RingSubfieldFpExt8{,MulBackend}`                               |
| Packed wrappers / aliases     | `PackedFp2`, `Packed{Power,Tower}BasisFp4`, `PackedRingSubfieldFp{4,8}`, `PF*Fp2`, `F*RingSubfieldFp4`, … | `…FpExt2` / `…FpExt4` / `…FpExt8` |
| Free fns / methods / tests    | `fp2_mul`, `power_basis_fp4_mul`, `tower_basis_fp4_mul`, `ring_subfield_fp{4,8}_{mul,square,inverse}`, … | `fp_ext2_mul`, `power_basis_fp_ext4_mul`, `ring_subfield_fp_ext{4,8}_…`, … |
| Modules / source files        | `ext/{fp2,power_fp4,tower_fp4,ring_subfield_fp4,ring_subfield_fp8}.rs` | `ext/{fp_ext2,power_fp_ext4,tower_fp_ext4,ring_subfield_fp_ext4,ring_subfield_fp_ext8}.rs` |

Explicitly **unchanged** (see Non-Goals): the `Ext2` type alias
(`pub type Ext2<F> = FpExt2<F, TwoNr>`), the prime fields
`Fp16`/`Fp32`/`Fp64`/`Fp128`, and the packed submodule filenames
`packed_*/fp{16,32,64,128}.rs` (those are bit-width fields, correctly
named).

### Invariants

1. **Behavioral identity (both parts).** Serialized proof bytes,
   transcript event streams, commitments, and all observable
   prover/verifier output are identical before and after this PR for any
   fixed `(setup, polynomial, opening point, transcript)`. There is no
   algorithmic, constant, or layout change to verify against — the
   guarantee is structural. Protected by the full `cargo test --workspace`
   suite (default features) and the `akita-field` unit tests (157 cases,
   incl. the extension parity and packed-parity tests).

2. **Part 1 is a byte-identical code move.** Every relocated backend body
   in `packed_avx2/{fp16,fp64,fp128}.rs`, `packed_avx512/{…}.rs`,
   `packed_neon/{…}.rs` is character-for-character the prior in-file
   version. The single content delta is in `packed_neon/fp128.rs`, where
   `use super::util::{…}` becomes `use crate::fields::util::{…}` because
   the module moved one level deeper.

3. **Part 2 is a collision-free 1:1 substitution.** The substring rename
   provably cannot touch a prime field: no bit-width field name
   (`Fp16`/`Fp32`/`Fp64`/`Fp128`, `fp16`/`fp32`/`fp64`/`fp128`) contains
   the substring `Fp2`/`Fp4`/`Fp8` (or `fp2`/`fp4`/`fp8`), and no
   extension marker is ever adjacent to another digit
   (`rg "[Ff]p[248][0-9]"` is empty across the crate). No marker is built
   by token-pasting (`paste!`/`concat_idents!`), so every occurrence is a
   literal token captured by the rename. Post-rename, `rg "[Ff]p[248]"`
   over all `.rs` is empty.

4. **All four codegen backends compile** — scalar `NoPacking`, NEON
   (aarch64), AVX2, and AVX-512 (x86_64). The per-width submodule split
   does not alter the `cfg(target_feature = …)` gating in
   [`packed.rs`](../crates/akita-field/src/fields/packed.rs); the
   re-exports (`pub use fpN::*;`) preserve the exact public surface of
   each backend root.

5. **File-size cap holds.** Every `.rs` file is under the 1500-line cap
   (`AKITA_RUST_FILE_LINE_CAP`, default 1500) in
   [`scripts/check-rust-file-lines.sh`](../scripts/check-rust-file-lines.sh).
   The largest residual files are the `packed_*/fp32.rs` submodules
   (859–891 lines), well within budget.

6. **Verifier no-panic contract** (per `AGENTS.md`) is preserved trivially
   — no verifier-reachable logic, validation, or arithmetic is touched;
   only identifiers and module paths change.

### Non-Goals

- **No behavioral or performance change.** This is a rename plus a code
  move. Identifier renaming is invisible to the compiler's monomorphized
  output, and relocating a body across module boundaries does not change
  generated code. No benchmark is gated on this PR.
- **Not renaming the prime fields.** `Fp16`/`Fp32`/`Fp64`/`Fp128` keep
  bit-width naming; that is the convention the rename *protects*. The
  packed submodule filenames `packed_*/fp{16,32,64,128}.rs` likewise stay.
- **Not renaming the `Ext2` alias.** `Ext2` carries no bit-width-confusable
  marker; only its right-hand side updates to `FpExt2<F, TwoNr>`.
- **Not splitting the `packed_*/fp32.rs` submodules further.** At
  859–891 lines they are the largest residual files but remain under the
  cap; splitting them is unnecessary churn.
- **No new public API, trait, or capability.** The crate's exported names
  are renamed, not added to or removed.
- **Historical specs are deliberately not rewritten.** Every existing
  `specs/*.md` is an immutable per-PR record (see §Alternatives). The
  old → new identifier map lives in *this* spec instead, so a future
  reader who greps an old spec for `RingSubfieldFp8` finds the bridge here
  rather than a silently revised record.

## Evaluation

### Acceptance Criteria

- [x] **Format:** `cargo fmt --all` clean.
- [x] **Native build (aarch64 / NEON):**
      `cargo check --workspace --all-targets --all-features` succeeds.
- [x] **Cross build (x86_64 / AVX2 + AVX-512):**
      `cargo check -p akita-field --lib --target x86_64-apple-darwin`
      succeeds (exercises the arch-gated `packed_avx2`/`packed_avx512`
      submodules).
- [x] **Lints:**
      `cargo clippy --workspace --all-targets --all-features -- -D warnings`
      clean.
- [x] **Tests:** `cargo test --workspace` green with **default features**
      (so the `#[cfg(all(test, not(feature = "zk")))]` extension tests in
      `ext/tests.rs` actually run). `akita-field` reports 157/157.
- [x] **Line cap:** `scripts/check-rust-file-lines.sh --no-baseline`
      passes; all packed files < 1500 lines.
- [x] **Rename completeness:** `rg "[Ff]p[248]"` over all `.rs` returns no
      matches; the 5 `ext/` files are recorded by git as renames
      (`R100`, history preserved).

### Testing Strategy

No new tests are introduced — a pure refactor is validated by the
*existing* suite continuing to pass unchanged:

- **Extension correctness** is covered by the renamed tests in
  [`ext/tests.rs`](../crates/akita-field/src/fields/ext/tests.rs):
  `fp_ext2_*` (Karatsuba, inverse, Frobenius/conjugation, norm),
  `ring_subfield_fp_ext{4,8}_*` (multiplication tables, embedding
  multiplicativity, serialization order), `power_basis_fp_ext4_*`, and
  `tower_basis_fp_ext4_*`. These run under default features only.
- **Packed-extension parity** is covered by
  [`packed_ext.rs`](../crates/akita-field/src/fields/packed_ext.rs) and
  its `tests` submodule, comparing `PackedRingSubfieldFpExt{4,8}` and the
  packed quartic wrappers against their scalar references on all active
  backends.
- **Cross-architecture compilation** is required because the AVX2/AVX-512
  submodules are inactive on the aarch64 dev host; the x86_64 `--lib`
  cross-check above is the gate.

Local validation:

```bash
cargo fmt --all
cargo check --workspace --all-targets --all-features                  # aarch64 / NEON
cargo check -p akita-field --lib --target x86_64-apple-darwin         # AVX2 / AVX-512
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace                                                # default features
```

### Performance

No effect, by construction. Identifier renaming does not survive into
codegen, and moving a function body between modules does not change the
monomorphized machine code. There is nothing to benchmark; "no
regression" is guaranteed structurally rather than measured.

## Design

### Architecture

**Part 1 — packed-backend layout.** Before this PR each backend root was
a single file mixing four per-width SIMD implementations. After:

```
fields/
  packed_avx2.rs        (62 lines: shared movehdup/mul64_64 helpers + mod decls + re-exports)
  packed_avx2/
    fp16.rs   (328)   fp32.rs (891, pre-existing)   fp64.rs (267)   fp128.rs (231)
  packed_avx512.rs      (61 lines: shared helpers + mod decls + re-exports)
  packed_avx512/
    fp16.rs   (311)   fp32.rs (876)                 fp64.rs (246)   fp128.rs (206)
  packed_neon.rs        (41 lines: shared to_vec/from_vec/mask_to_bit + mod decls + re-exports)
  packed_neon/
    fp16.rs   (311)   fp32.rs (859)                 fp64.rs (224)   fp128.rs (314)
```

The split axis is **per-width field**, identical across all three
backends, so the three architectures are now structurally parallel and
each root reads as a table of contents. Backend selection is unchanged:
the mutually-exclusive `cfg(target_feature = …)` gates in
[`packed.rs`](../crates/akita-field/src/fields/packed.rs) still pick
exactly one root, and each root's `pub use fpN::*;` reproduces its former
flat surface.

**Part 2 — rename mechanics and surface.** The rename touches **50 `.rs`
files across 9 crates** (`akita-field`, `akita-pcs`, `akita-types`,
`akita-prover`, `akita-config`, `akita-verifier`, `akita-transcript`,
`akita-r1cs`, `akita-scheme`) plus the 5 `ext/` source-file renames. The
re-export hub in
[`fields/ext.rs`](../crates/akita-field/src/fields/ext.rs) and the
crate-level re-exports in `lib.rs` are updated so downstream crates see
only the new names. Because the prime fields are bit-width-named and the
extensions are now `Ext`-marked, the two naming axes no longer collide at
any call site.

### Alternatives Considered

- **Rename scope.** Three options were weighed: (a) type/trait names only,
  (b) types + snake-case functions/tests, (c) types + functions + module
  and source-file names. Option (c) was chosen for full internal
  consistency — a half-renamed crate (type `FpExt2` living in module `fp2`
  with helper `fp2_mul`) would reintroduce the very ambiguity this PR
  removes.
- **Raising the line cap instead of splitting (Part 1).** Bumping
  `AKITA_RUST_FILE_LINE_CAP` or extending the ratchet baseline was
  rejected: the monoliths genuinely interleaved four independent backends,
  and the `fp32` submodule split already established the per-width
  precedent. Splitting improves locality and reviewability, not just the
  line count.
- **Rewriting historical specs to the new names.** Considered and
  **rejected.** Every `specs/*.md` follows
  [`TEMPLATE.md`](TEMPLATE.md) with an `Author/Status/PR` header and a
  status lifecycle that already encodes change via `superseded by …`,
  `retrospective`, and `historical scaffolding spec`. That is the ADR
  convention — records are immutable; you supersede, you do not back-edit.
  Editing PR #86's spec to say it introduced `RingSubfieldFpExt8` would
  falsify a true statement about history (the name did not exist then).
  The spec edits that were briefly applied were reverted in full
  (`git checkout HEAD -- specs/`); this spec is the forward-looking
  record and the old → new bridge instead.
- **Branch placement.** The rename was applied on the current
  `taghi/refactor/akita-field` branch (stacked on the
  packed-modularization commit) rather than a fresh branch off `main`,
  keeping the two related `akita-field` clean-ups in one reviewable unit.

## Documentation

This spec is the only documentation artifact, and intentionally so: the
historical per-PR specs are left untouched (see §Alternatives), and the
old → new identifier map in §Intent serves as the discoverability bridge
for anyone tracing a pre-rename name. No README, crate-doc, or example
changes are required — the public surface is renamed, not restructured.

## Execution

The work was applied mechanically and verified, not hand-edited
file-by-file:

- **Part 1** extracted each `Fp16`/`Fp64`/`Fp128` backend body verbatim
  into its submodule, slimmed each root to shared intrinsics + `mod`/`pub
  use`, fixed the one relative-import in `packed_neon/fp128.rs`, and
  `git mv`-ed nothing (new submodule files, root files rewritten).
  Committed as `26e9745e`.
- **Part 2** ran a case-sensitive substring substitution
  (`Fp2→FpExt2`, `Fp4→FpExt4`, `Fp8→FpExt8`, `fp2→fp_ext2`,
  `fp4→fp_ext4`, `fp8→fp_ext8`) over the exact set of marker-bearing
  `.rs` files (`rg -l "[Ff]p[248]" -g "*.rs"`), then `git mv`-ed the five
  `ext/` source files to their `fp_ext*` names so the module declarations
  (already rewritten by the substitution) resolve. The substitution order
  is interference-free: no replacement output contains a source marker.
- **Collision-safety pre-checks** (all confirmed before editing): the
  bit-width fields share no substring with the markers; no marker is
  digit-adjacent; no marker is token-pasted. These are what make the blunt
  substitution sound, and the compiler + full test suite are the backstop.

Risks were limited to (a) accidentally renaming a bit-width field —
ruled out by the substring/adjacency analysis — and (b) missing a
macro-generated name — ruled out by the absence of `paste!`/`concat_idents!`
in the marker neighborhood.

## References

- Packed-modularization commit `26e9745e` — Part 1 of this PR.
- [`specs/avx-simd-port.md`](avx-simd-port.md) — the AVX backend work that
  grew the packed roots past the cap.
- [`specs/simd-ring-subfield-fp8.md`](simd-ring-subfield-fp8.md) —
  introduced `RingSubfieldFp8` (now `RingSubfieldFpExt8`), the largest
  extension family renamed here.
- [`specs/rust-file-line-cap.md`](rust-file-line-cap.md) and
  [`scripts/check-rust-file-lines.sh`](../scripts/check-rust-file-lines.sh)
  — the 1500-line policy Part 1 satisfies.
- [`crates/akita-field/src/fields/ext.rs`](../crates/akita-field/src/fields/ext.rs)
  — the extension re-export hub updated by Part 2.
- [`crates/akita-field/src/fields/packed.rs`](../crates/akita-field/src/fields/packed.rs)
  — the unchanged `cfg(target_feature)` backend selector.
