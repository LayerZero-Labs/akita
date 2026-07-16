# Spec: Unify the Akita and Jolt Field Stacks in `jolt-field`

| Field         | Value          |
|---------------|----------------|
| Author(s)     | Alberto Centelles |
| Created       | 2026-07-15     |
| Status        | active         |
| PR            |                |
| Supersedes    | [`specs/akita-field-refactor.md`](akita-field-refactor.md) |
| Superseded-by |                |
| Book-chapter  |                |

## Summary

Akita and Jolt currently carry overlapping field traits, concrete field types,
and compatibility implementations. This duplicates security-sensitive
arithmetic, increases the audit surface, and creates a reverse dependency from
Jolt's Akita integration back into Akita's field crate.

This spec makes the Jolt repository's `jolt-field` package the single owner of
the complete field stack supported by Akita and Jolt. Akita will depend on that
package directly and `akita-field` will be deleted. The move includes the
optimized Solinas field implementations, their extensions, SIMD and unreduced
backends, FFT helpers, and their tests and benchmarks; it is not a thin trait
crate or an Akita compatibility wrapper.

`jolt-akita` remains a substantive PCS integration crate. It may translate
transcripts, configuration, setup, batching, serialization errors, and Jolt's
PCS-facing abstractions, but it must not define or wrap field identities.

## Intent

### Goal

Make the Jolt repository's `jolt-field` package the canonical, independently
versioned implementation of every field family used by Jolt and Akita, with
both repositories importing the same Rust types and traits directly.

### Decisions

- The package is named `jolt-field` and lives under `crates/jolt-field` in the
  Jolt repository.
- Akita depends on `jolt-field` directly. The dependency is intentional: field
  arithmetic is a shared primitive owned by Jolt, not a dependency on the Jolt
  prover or protocol.
- The complete supported Akita field stack moves. Keeping a second
  `akita-field` implementation or a reduced Akita-only backend would defeat the
  audit objective.
- There is no `crates/jolt-akita` field facade and no Akita re-export crate.
  Downstream imports are changed to `jolt_field::...` as part of one coordinated
  breaking cutover.
- `jolt-akita` remains only for PCS integration that cannot naturally live in
  either primitive crate.
- Akita's serialization format and optimized serialization implementation are
  preserved byte for byte. Serialization API redesign and renaming are outside
  this spec.
- Binary-field/GF(2) support is a follow-up backend. The shared hierarchy must
  not make it impossible, but this migration does not claim to implement it.

### Invariants

- There is exactly one production definition and implementation of each field
  trait, field type, extension type, packed backend, accumulator, and FFT helper
  in scope. Compatibility modules must not duplicate arithmetic or forward one
  field API to another.
- Scalar arithmetic remains identical for every supported prime and extension
  field, including canonical representatives and reduction behavior.
- Packed NEON, AVX2, and AVX-512 arithmetic remains lane-for-lane equal to the
  scalar implementation. Existing architecture-specific safety arguments,
  boundary tests, and documented `unsafe` preconditions move with the code.
- Canonical field bytes, Akita proof bytes, setup bytes, transcript absorption,
  transcript challenge outputs, and validation behavior remain identical.
- Proof size, setup size, proof object layout, and prover/verifier schedules do
  not change as a consequence of the package move.
- Akita's optimized serialization paths remain optimized. The migration must
  not replace them with a generic serde or Arkworks encoding.
- Verifier-reachable shared code obeys Akita's no-panic contract: malformed
  input is rejected with `AkitaError` or `SerializationError`, not a panic,
  unchecked index, or unbounded allocation.
- A `jolt-field` release used by Akita and Jolt is immutable during its audit
  window. Both repositories resolve exactly the same package identity and
  version.
- Moving code across repositories preserves audit provenance. The landing PR
  records the originating Akita commit and a source-to-destination file map.
- The broad Jolt `Field` capability is a programming convenience, not a claim
  that all backends share odd-prime-field operations. Future binary fields must
  not be forced to implement halving, signed integer embedding, or Solinas-only
  capabilities.

### Non-Goals

- Redesigning or renaming `AkitaSerialize`, proof encodings, or vector framing.
- Changing field moduli, representations, extension bases, FFT domains, or
  transcript challenge derivation.
- Adding a wrapper crate, re-export facade, compatibility type alias, or newtype
  solely to preserve old `akita_field` import paths.
- Moving Akita PCS policy, setup generation, transcript scheduling, proof
  batching, or verifier logic into `jolt-field`.
- Removing `jolt-akita`; it remains the Jolt-to-Akita PCS integration boundary.
- Implementing a binary/GF(2) backend in the initial cutover.
- Preserving backward compatibility. Both repositories may make the breaking
  import and trait-bound changes required for a single source of truth.

## Evaluation

### Acceptance Criteria

- [x] `jolt-field` directly owns the shared trait hierarchy and the BN254 and
  Solinas backend families, with no production dependency on an Akita crate.
- [x] The full Akita field stack listed in the migration inventory below has
  moved to Jolt, including tests, fuzz targets, benchmarks, architecture code,
  safety comments, and arithmetic derivations.
- [x] Every Akita crate imports shared field types and traits from `jolt_field`;
  `akita-field` is removed from the workspace and filesystem.
- [x] Akita's `compat/jolt.rs`, Jolt's reverse `src/akita.rs` adapter, and their
  bootstrap features/dependencies are deleted after the coordinated cutover.
- [x] No field wrapper, pass-through implementation, or re-export facade remains
  in Akita or `jolt-akita`.
- [x] `AkitaError` remains Akita-owned and is moved out of the deleted field
  crate without introducing a dependency cycle.
- [x] `jolt-field` does not depend on `akita-serialization`. Existing
  `AkitaSerialize`, `AkitaDeserialize`, and `Valid` behavior for the moved types
  is owned by `akita-serialization` and remains byte-identical.
- [ ] Golden fixtures demonstrate identical canonical field, extension-field,
  proof, setup, and transcript bytes before and after the cutover.
- [ ] Akita's full formatter, clippy, test, and documentation guardrails pass;
  Jolt's host and host+zk formatter, clippy, and test matrices pass.
- [x] `jolt-field` passes no-default, BN254-only, Solinas-only, and combined
  feature builds. Enabling one backend does not silently select another.
- [ ] NEON, AVX2, and AVX-512 scalar/packed parity tests pass on representative
  hardware or their established CI runners.
- [x] A package-identity check demonstrates that an integrated Jolt+Akita build
  contains one resolved `jolt-field` package ID and no `akita-field` package.
- [ ] Existing field microbenchmarks show no unexplained regression greater
  than 2% in median hot-operation latency or throughput on the same compiler and
  hardware.
- [x] Jolt's landing PR records the originating Akita revision and an auditable
  mapping from each moved source/test/benchmark file to its new location.

The implementation remains `active` until the stacked repositories replace
their sibling path dependencies with the released or immutable Git package,
the architecture-specific CI runners report NEON/AVX2/AVX-512 parity, and the
same-machine microbenchmark comparison and complete host/zk CI matrices pass.

### Testing Strategy

The move must be validated at three levels.

#### `jolt-field` feature matrix

At minimum, CI runs:

```bash
cargo check -p jolt-field --no-default-features
cargo test -p jolt-field --no-default-features --features bn254
cargo test -p jolt-field --no-default-features --features solinas
cargo test -p jolt-field --no-default-features --features bn254,solinas
cargo test -p jolt-field --no-default-features --features solinas,parallel
```

The exact Jolt-wide feature aggregation may differ, but `jolt-field` features
remain additive and individually testable. Architecture-specific builds repeat
the Solinas tests with the repository's established NEON, x86-64-v3/AVX2, and
AVX-512 target flags.

#### Akita repository

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```

In addition to existing unit and end-to-end tests, add migration fixtures that
are generated on the pre-cutover revision and consumed on the post-cutover
revision. They cover:

- canonical scalar encodings for boundary and random values of every named
  production prime;
- extension encodings in every basis used by Akita;
- validated and unvalidated decoding, including malformed encodings;
- transcript absorption and challenge bytes for fixed schedules;
- complete proof and setup serialization for fixed deterministic fixtures.

The fixtures are checked in or generated from a pinned pre-cutover helper so a
single post-cutover implementation cannot accidentally bless its own output.

#### Jolt repository and integrated graph

Run Jolt's repository-required formatter, clippy, and nextest suites for both
`host` and `host,zk`. The integration job additionally runs `cargo tree -d` (or
an equivalent metadata check) and rejects duplicate `jolt-field` package IDs or
any surviving `akita-field` dependency.

### Performance

This is an ownership migration, so the expected arithmetic change is zero.
Capture pre-cutover baselines using the existing Akita and Jolt field arithmetic
benchmarks, then rerun them using the same Rust toolchain, target features,
machine, governor, and Criterion settings.

A median change within +/-2% is treated as noise. A larger regression in a hot
scalar or packed operation blocks the cutover unless it is reproduced,
explained, and explicitly approved. Improvements may land, but arithmetic
optimizations should normally be separate commits so reviewers can distinguish
the code move from semantic or performance changes. Proof size and setup size
must be exactly unchanged.

## Design

### Target Architecture

After the final cutover, dependency direction is:

```text
                       registry `jolt-field`
                         /              \
                        v                v
                  Jolt crates       Akita primitive/protocol crates
                        \                /
                         v              v
                            `jolt-akita`
                         (PCS integration only)

Akita serialization crates ---> `jolt-field`
Akita protocol crates -------> Akita error owner
```

There is no edge from `jolt-field` back to Akita. `jolt-akita` can depend on
both projects because integration is its purpose, but neither project reaches
shared field identities through it.

### Migration Inventory

The following is one ownership unit and moves from `akita-field` into
`jolt-field`:

| Area | Required contents |
|------|-------------------|
| Trait hierarchy | additive group, ring and field cores, inversion, primitive conversion, canonical byte capabilities, challenge decoding, random sampling, halving, lifting/base multiplication, accumulator, packing, and related capability traits |
| Prime fields | generic `Fp32`, `Fp64`, and `Fp128` pseudo-Mersenne/Solinas families and all named supported primes |
| Extensions | `FpExt2`, `FpExt4`, `FpExt8`, their basis/configuration types, Frobenius/lifting operations, and specialized multiplication backends |
| Unreduced arithmetic | wide and unreduced accumulator representations, reduction traits, optimized folds, and dot-product paths |
| Packed arithmetic | generic packed interfaces, packed extensions, and NEON, AVX2, and AVX-512 implementations for every supported width |
| Domains | FFT roots, smooth-domain construction, Reed-Solomon extension helpers, and parallel helpers used by them |
| Assurance | property/unit tests, packed-vs-scalar parity and boundary tests, fuzz targets, benches, safety comments, and reduction derivations |

Existing Jolt BN254 support stays in the same package. The merge is allowed to
organize code by capability and backend, but it must not create an `akita`
module containing a second trait family. Solinas types are first-class
`jolt-field` exports.

### Trait Reconciliation

Before moving implementations, the Jolt PR must include a reviewable contract
diff between the current trait families. Similar names are not sufficient proof
of identical semantics. Known differences include Jolt's `RingCore::pow2`,
transcript scalar-challenge construction, and broad signed-accumulator bounds.

For each difference, the implementation must choose one canonical contract and
record whether it is:

1. a generally valid algebraic capability on the shared trait;
2. an optional capability trait implemented only by suitable backends; or
3. protocol-specific behavior that remains outside `jolt-field`.

Default methods that affect transcript decoding, canonical bytes, inversion,
or reduction require equivalence tests before adoption. Capability traits are
preferred over forcing all fields to implement odd-prime or
representation-specific operations. This is particularly important for the
future binary backend.

### Features and Backend Selection

The intended feature surface is:

| Feature | Meaning |
|---------|---------|
| `bn254` | Arkworks/BN254 backend and its dependencies |
| `solinas` | complete Fp32/Fp64/Fp128, extension, packed, unreduced, and FFT stack moved from Akita |
| `parallel` | Rayon-backed parallel helpers where supported |
| `allocative` | allocation instrumentation implementations |

The library default should be empty unless Jolt-wide compatibility requires a
staged change. In all cases, backend features are additive: `bn254,solinas`
must compile together and neither feature changes the identity or semantics of
types exported by the other. SIMD selection continues to use target-feature
configuration rather than Cargo features.

Do not add an empty `binary` feature during this migration. The future GF(2)
backend should add a real capability and tests when implemented.

### Serialization Ownership

Serialization is deliberately preserved, not redesigned. However,
`jolt-field` cannot remain independently publishable if its core traits or
implementations depend on an Akita crate. The cutover therefore separates
algebraic field capability from Akita proof-format capability:

- `jolt-field::CanonicalField` and arithmetic implementations do not require
  `AkitaSerialize`, `AkitaDeserialize`, or Akita's `Valid` trait.
- `akita-serialization`, which owns those traits, implements them for the
  Jolt-owned prime and extension types using their canonical coefficient/byte
  APIs.
- `jolt-field` exposes only the minimal constructors and coefficient accessors
  required to encode and validate the types without representation-dependent
  copying. Those APIs are field APIs, not Akita wrappers.
- Akita code that actually crosses a serialization boundary adds explicit
  serialization bounds. Arithmetic-only generic code remains serialization
  agnostic.

This preserves the optimized Akita implementation and wire format while
removing the reverse dependency. It does not authorize replacing current
formats, changing validation modes, or renaming serialization traits.

### Akita Error Ownership

`AkitaError` is a PCS/protocol error, not a field primitive. It must not move to
`jolt-field`. Because low-level Akita crates such as challenges and witness code
already return it, placing it in a higher protocol/type crate would create
cycles. The cutover creates a small `akita-error` crate that owns the existing
enum and its current semantics. This is an ownership boundary, not a facade:
the enum is defined there once, and consumers import it directly.

If dependency-graph analysis during implementation finds an existing lower
Akita crate that can own the enum without a cycle or semantic mismatch, that
placement may be used instead, but the acceptance conditions remain: one Akita
definition, no field-crate ownership, no re-export facade, and no dependency
from `jolt-field` to it.

### `jolt-akita` Boundary

`jolt-akita` remains responsible for substantive integration such as:

- mapping Jolt polynomial/opening order to Akita claims;
- transcript and sponge scheduling at the PCS boundary;
- batching and proof/setup configuration;
- invoking Akita proving and verification APIs;
- translating Akita serialization and protocol errors into Jolt-facing errors.

It must not own field traits, field types, arithmetic implementations, aliases,
or pass-through field helpers. Once both sides use the same `jolt-field` types,
field compatibility code in `jolt-akita` is deleted rather than simplified into
another forwarding layer.

### Package Identity and Release

`jolt-field` is published as a standalone registry package. During development,
Jolt may use a workspace path plus an exact version, while Akita pins that exact
registry version. A temporary root `[patch]` or Git revision is allowed only as
bootstrap machinery for stacked cross-repository PRs; it is removed from the
final audited graph.

The package version is frozen for the audit window. Any arithmetic or
serialization-affecting change after the freeze requires a new version and an
explicit audit delta. CI inspects Cargo metadata and fails if the integrated
graph resolves more than one `jolt-field` package identity.

### Cross-Repository Landing Sequence

The migration cannot be atomic across two repositories, so it lands in three
reviewable phases.

1. **Jolt foundation.** Add the Solinas stack and reconciled traits to
   `jolt-field`, with provenance, tests, feature matrix, package-identity gate,
   and a publishable dependency graph. A temporary reverse Akita adapter or
   pinned bootstrap dependency may remain only to keep the current
   `jolt-akita` revision buildable.
2. **Akita cutover.** Pin the new `jolt-field`, move serialization impls and
   `AkitaError` to their proper Akita owners, change every consumer to direct
   `jolt_field` imports, pass golden compatibility tests, remove `akita-field`,
   and update the crate graph and book.
3. **Jolt cleanup.** Pin the migrated Akita revision, delete Jolt's reverse
   Akita adapter and all bootstrap dependencies/features, then run the full
   integrated Jolt matrix and freeze the audit version.

Temporary compatibility code is named in the phase-one PR description and has
a deletion issue/PR linked before phase one merges. It must not survive phase
three or be published as part of the final package surface.

### Alternatives Considered

#### Keep `akita-field` canonical and make Jolt depend on it

This makes primitive ownership appear Akita-specific and leaves Jolt's core
field package split across repositories. It also complicates Jolt's existing
BN254 backend ownership. The agreed direction is one multi-backend
`jolt-field` package in Jolt.

#### Introduce a neutral third repository or `crates/jolt-akita`

A neutral package would avoid the name-direction concern, but adds release,
versioning, review, and audit indirection. A `jolt-akita` field facade would be
more problematic: both projects would reach primitive identities through an
integration layer. Neither is justified while `jolt-field` can directly own
the optimized backends.

#### Keep traits in `jolt-field` and implementations in Akita

This preserves two audit locations and requires adapters or cross-repository
implementation dependencies. Moving only thin traits does not achieve the audit
surface reduction.

#### Preserve `akita-field` as a re-export crate

This offers import compatibility at the cost of a permanent extra layer and
duplicate package surface. Akita explicitly makes no backward-compatibility
guarantees, so consumers should import the canonical crate directly.

#### Move Akita serialization into `jolt-field`

This would reverse the unwanted dependency rather than remove it and would mix
PCS wire-format policy into a field primitive. Keeping the optimized
implementations with the Akita-owned serialization traits preserves behavior
without coupling Jolt to Akita.

## Documentation

The implementation PRs must update:

- `docs/crate-graph.md` for the deleted `akita-field`, new direct
  `jolt-field` edges, and Akita error/serialization ownership;
- `book/src/how/architecture.md` for crate ownership and the role of
  `jolt-akita`;
- `book/src/foundations/rings-and-fields.md` for the canonical field package,
  supported backends, and feature model;
- relevant crate READMEs and feature-flag documentation;
- the Jolt repository's crate graph and field documentation;
- this spec's status, PR links, and acceptance checkboxes as each phase lands.

After the durable design is folded into the two owning Akita Book pages, set
`Book-chapter` and archive this spec according to `specs/PRUNING.md`.

## Execution

Recommended implementation checklist:

1. Freeze the pre-cutover field API, feature, package, serialization, transcript,
   and benchmark baselines.
2. Produce the trait semantic-diff table and resolve every mismatch before
   mechanically moving concrete types.
3. Move code with history-preserving commits where practical. Keep mechanical
   relocation separate from trait reconciliation and optimizations.
4. Make `jolt-field` independent of all Akita crates and pass its standalone
   feature/architecture matrix.
5. Relocate Akita serialization implementations and `AkitaError`; add golden
   compatibility fixtures before deleting their old definitions.
6. Cut every Akita consumer directly to `jolt_field`, then delete
   `akita-field` and all compatibility modules in one change set.
7. Run both repository matrices, compare benchmarks and bytes, inspect Cargo
   metadata for duplicate package IDs, and remove bootstrap configuration.
8. Freeze and publish the audited `jolt-field` version, then update the spec and
   durable documentation.

The highest-risk areas are trait-default semantic drift, orphan-rule constraints
around Akita serialization, accidentally changing extension coefficient order,
and compiling only the host's SIMD backend. These are addressed before the
crate deletion, not deferred to cleanup.

## References

- [`specs/akita-field-refactor.md`](akita-field-refactor.md) — current Akita
  field organization and trait inventory.
- [`specs/cross-repo-field-microbench.md`](cross-repo-field-microbench.md) —
  existing arithmetic benchmark matrix and SIMD measurement guidance.
- [`specs/akita-crate-followup-jolt-integration.md`](akita-crate-followup-jolt-integration.md)
  — current `jolt-akita` integration boundary.
- [`docs/verifier-contract.md`](../docs/verifier-contract.md) — Akita's
  verifier no-panic requirements.
- [`docs/crate-graph.md`](../docs/crate-graph.md) — current Akita dependency
  graph.
- [`book/src/foundations/rings-and-fields.md`](../book/src/foundations/rings-and-fields.md)
  — current field foundations narrative.
- [Jolt field hierarchy unification spec](https://github.com/a16z/jolt/blob/main/specs/unify-field-hierarchy.md)
  — Jolt-side prior design work.
