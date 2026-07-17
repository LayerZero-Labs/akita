# Spec: Unify the Akita and Jolt Field Stacks in `jolt-field`

| Field         | Value          |
|---------------|----------------|
| Author(s)     | Alberto Centelles |
| Created       | 2026-07-15     |
| Status        | active         |
| PR            | [Jolt PR A #1684](https://github.com/a16z/jolt/pull/1684); Akita PR B TBD; Jolt PR C TBD |
| Supersedes    | [`specs/akita-field-refactor.md`](archive/2026-Q3/akita-field-refactor.md) |
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
PCS-facing abstractions, but it must not define or wrap field identities. Its
public `AkitaField` name is retained only as a transparent alias for the
concrete field fixed by `AkitaScheme`; it creates no second type identity.

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
  Akita imports are changed to `jolt_field::...` as part of one coordinated
  breaking cutover. `jolt_akita::AkitaField` remains available as the
  transparent semantic name for the field fixed by `AkitaScheme`, not as a
  wrapper or compatibility identity.
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
- The `jolt-field` source used by Akita is immutable during its audit window:
  either a released package or a full upstream Jolt Git revision. Its contents
  match the audited Jolt workspace member. Every resolved integration graph
  contains exactly one `jolt-field` Cargo package identity.
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
- Removing the existing `jolt_akita::AkitaField` semantic alias. Retaining that
  transparent alias does not preserve an `akita_field` path or create a second
  field identity.
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
- [ ] Akita's `compat/jolt.rs`, Jolt's reverse `src/akita.rs` adapter, and their
  bootstrap features/dependencies are deleted after the coordinated cutover.
- [x] No field wrapper, pass-through implementation, or re-export facade remains
  in Akita or `jolt-akita`. The public `jolt_akita::AkitaField` name is a
  transparent alias for `jolt_field::Prime128OffsetA7F7`, not a facade.
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
- [ ] A package-identity check demonstrates that an integrated Jolt+Akita build
  contains one resolved `jolt-field` package ID and no `akita-field` package.
- [ ] PR C derives the headerless commitment decode coefficient count from the
  validated statement and canonical Akita schedule, without adding
  `backend_num_coeffs` to `AkitaCommitment` or changing the commitment bytes or
  transcript.
- [ ] Existing field microbenchmarks show no unexplained regression greater
  than 2% in median hot-operation latency or throughput on the same compiler and
  hardware.
- [x] Jolt's landing PR records the originating Akita revision and an auditable
  mapping from each moved source/test/benchmark file to its new location.

The implementation remains `active` until Jolt PR A lands, Akita replaces its
temporary PR-head fork pin with the released package or immutable upstream Git
revision, PR C removes the remaining bootstrap adapter and resolves the final
integrated graph to one `jolt-field` identity, the architecture-specific CI
runners report NEON/AVX2/AVX-512 parity, and the same-machine microbenchmark
comparison and complete host/zk CI matrices pass.

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
                  immutable `jolt-field` source
                    /                   \
                   v                     v
       Jolt workspace member      Akita primitive/protocol crates
                   \                     /
                    v                   v
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

The library default remains `bn254` during this cutover to preserve Jolt's
existing feature behavior. Backend features are additive: `bn254,solinas` must
compile together and neither feature changes the identity or semantics of types
exported by the other. SIMD selection continues to use target-feature
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

It must not own field traits, field types, arithmetic implementations, or
pass-through field helpers. It retains `jolt_akita::AkitaField` as the public
semantic name for the field fixed by `AkitaScheme`; that name is transparently
identical to `jolt_field::Prime128OffsetA7F7` and carries no implementation.
Once both sides use the same `jolt-field` types, field compatibility code in
`jolt-akita` is deleted rather than simplified into another forwarding layer.

Akita's optimized commitment encoding is headerless and therefore requires a
coefficient count as deserialization context. PR C derives that count from the
validated Jolt statement and the same canonical Akita commitment schedule used
by the verifier, with checked arithmetic. `AkitaCommitment` must not serialize
an attacker-supplied `backend_num_coeffs` field or absorb a new coefficient
count into the Jolt transcript. This preserves the pre-cutover wrapper bytes
and Fiat-Shamir schedule while bounding decoder allocation from trusted shape.

### Package Identity and Release

Akita consumes `jolt-field` from either a standalone registry release or an
immutable full upstream Jolt Git revision. During stacked review it temporarily
pins commit `09b2f7b6ddd9427c756b781c39530a6c005e332d` on the contributor fork,
the head used by Jolt PR A #1684. That fork pin is replaced with the upstream
merge revision or a released package identity once PR A lands.

Jolt itself uses the local `crates/jolt-field` workspace member. When Jolt later
consumes the migrated Akita revision, PR C must use a root Cargo source patch or
an equivalent resolution mechanism to map Akita's immutable `jolt-field`
dependency to that workspace member. The mechanism is final integration wiring,
not a field facade, and the resolved Jolt graph must contain one package
identity. A graph containing both the local workspace member and a Git or
registry copy is invalid.

The package release or Git revision is frozen for the audit window. Any
arithmetic or serialization-affecting change after the freeze requires a new
immutable identity and an explicit audit delta. CI inspects Cargo metadata and
fails if the integrated graph resolves more than one `jolt-field` package
identity.

### Cross-Repository Landing Sequence

The migration cannot be atomic across two repositories. It uses three code PRs
with an immutable field-identity checkpoint between PR A and PR B.

1. **Jolt foundation.** Jolt PR A #1684 adds the Solinas stack and reconciled
   traits to `jolt-field`, with provenance, tests, feature matrix, and an
   intermediate package-identity gate. It retains the reverse Akita adapter and
   unpublished optional bootstrap dependencies so the existing `jolt-akita`
   revision remains buildable; registry packaging is therefore deferred.
2. **Freeze the field identity.** Land PR A, then select its full upstream Git
   revision or publish the standalone package. Akita must not cut over to a
   mutable branch, sibling path, or contributor-fork identity intended only for
   review.
3. **Akita cutover.** Pin the frozen `jolt-field`, move serialization impls and
   `AkitaError` to their proper Akita owners, change every consumer to direct
   `jolt_field` imports, pass golden compatibility tests, remove `akita-field`,
   and update the crate graph and book.
4. **Jolt cleanup.** Pin the migrated Akita revision, map its `jolt-field`
   source back to Jolt's workspace member, derive commitment decode context
   without changing wrapper bytes or transcript scheduling, delete Jolt's
   reverse Akita adapter and all bootstrap dependencies/features, and run the
   full integrated Jolt matrix.

Temporary compatibility code is named in the phase-one PR description and has
a deletion issue/PR linked before phase one merges. It must not survive the
Jolt cleanup phase or be published as part of the final package surface.

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
4. Make the moved `jolt-field` implementation independent of Akita crates and
   pass its standalone feature/architecture matrix while retaining only the
   documented PR A bootstrap edge.
5. Land PR A and freeze an immutable upstream Git revision or registry release
   before changing Akita's production dependency.
6. Relocate Akita serialization implementations and `AkitaError`; add golden
   compatibility fixtures before deleting their old definitions.
7. Cut every Akita consumer directly to `jolt_field`, then delete
   `akita-field` and all compatibility modules in one change set.
8. In PR C, pin migrated Akita, unify Cargo package identity, derive bounded
   commitment decode context without changing Jolt bytes or transcript state,
   remove bootstrap configuration, run both repository matrices, compare
   benchmarks and bytes, and update the spec and durable documentation.

The highest-risk areas are trait-default semantic drift, orphan-rule constraints
around Akita serialization, accidentally changing extension coefficient order,
and compiling only the host's SIMD backend. These are addressed before the
crate deletion, not deferred to cleanup.

## References

- [`specs/akita-field-refactor.md`](archive/2026-Q3/akita-field-refactor.md) — former Akita
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
