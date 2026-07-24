# Spec: Planner-Free Runtime Schedule Boundary

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-23 |
| Status        | proposed |
| PR            | |
| Supersedes    | Runtime-fallback portions of [`archive/2026-Q2/planner-refactor.md`](archive/2026-Q2/planner-refactor.md) and [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md) |
| Superseded-by | |
| Book-chapter  | book/src/how/configuration.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita currently treats generated schedules as a cache over planner search.
Runtime code may invoke the dynamic planner when a catalog row is absent, and
multi-group verification may invoke planner logic while constructing the
lookup key itself. This gives the verifier authority to discover protocol
parameters at runtime and makes `akita-planner` part of the verifier's
dependency and attack surface.

This specification replaces that cache contract with a finite support
contract:

1. Runtime code constructs a semantic schedule request.
2. `akita-schedule` performs strict catalog lookup.
3. `akita-schedule` expands the compact row.
4. `akita-schedule` validates the expanded schedule.
5. Prover and verifier consume the resulting `ValidatedSchedule`.

A missing row is unsupported input and is rejected. No runtime path searches
for a replacement schedule.

Only two schedule-specific crates remain:

- `akita-schedule`: runtime schemas, catalogs, expansion, and validation;
- `akita-planner`: offline search and catalog emission.

Planner modularity and the two distinct precommitment mechanisms are specified
separately in [`modular-planner-and-precommit-roles.md`](modular-planner-and-precommit-roles.md).

## Status and baseline

This design was written against `origin/main` commit
`af482ab7c2d8d3edcdc901f0c3950378f954787a`, informed by the PR 323 mirror at
`75d7a6ddc246a97c9da86b03c0ac0ac5356d14a1`.

At the baseline:

- `akita-verifier` depends on `akita-config`;
- `akita-config` depends unconditionally on `akita-planner`;
- generated schedule types and validation live partly in `akita-planner`;
- `CommitmentConfig::runtime_schedule` can fall back to planner DP;
- grouped schedule-key construction invokes planner logic;
- the generated catalog is treated as an optimization rather than the
  definition of runtime support.

The implementation is complete only when the dependency rule holds under
default features, no default features, and all features. Removing a direct
call while retaining a transitive planner dependency is not completion.

## Terminology

- A **schedule request** is a semantic description of the proof to be opened.
  It contains public group layouts and runtime capability choices, not
  planner-derived ranks, bounds, splits, or decomposition choices.
- A **root-precommit recipe** is the canonical commitment layout for a user
  polynomial group that is committed before its eventual grouped root opening.
- A **compact schedule** records the independent decisions needed to
  reconstruct one expanded schedule.
- An **expanded schedule** is the typed root, recursive, and terminal schedule
  consumed by protocol code.
- A **catalog** is a finite mapping from supported semantic requests to compact
  schedules and, where applicable, root-precommit recipes.
- A **runtime policy** contains protocol, security, and capability rules
  required to validate schedules. It does not contain search objectives or
  planner heuristics.
- A **validated schedule** is an expanded schedule that has passed canonical
  validation for a request, runtime policy, and setup capability.
- A **setup-prefix slot** is a schedule-owned commitment to a prefix of public
  setup data, consumed at one specified recursive transition.

## Intent

### Goal

Create one planner-independent runtime boundary that provides strict catalog
lookup, deterministic expansion, and complete schedule validation to both
prover and verifier.

### Invariants

#### Dependency boundary

1. `akita-verifier` **MUST NOT** depend directly or transitively on
   `akita-planner` under any supported feature graph.
2. `akita-config` and `akita-schedule` **MUST NOT** depend on `akita-planner`
   under any supported feature graph.
3. Runtime verifier entry points **MUST NOT** search, enumerate, optimize, or
   invoke planner DP.
4. Runtime prover entry points **MUST** use the same finite catalog contract as
   the verifier. Production proving must not select an uncataloged schedule.
5. Feature flags **MUST NOT** restore planner fallback in verifier-reachable
   code.
6. Dependency checks **MUST** cover default, no-default-feature, and all-feature
   graphs so Cargo feature unification cannot silently reintroduce the planner.

The intended dependency direction is:

```text
akita-types
    ↑
akita-schedule
    ↑              ↑
akita-config       akita-planner
    ↑
akita-prover / akita-verifier
```

`akita-planner` may depend on `akita-schedule` to reuse the canonical compact
schema, expansion, and validation. The reverse dependency is forbidden.

#### Resolution boundary

1. Runtime resolution **MUST** be a total, non-searching operation over an
   enabled catalog.
2. A missing catalog or missing row **MUST** return a typed error and **MUST**
   reject at the public verifier boundary.
3. Identity mismatch, malformed generated data, insecure parameters, and setup
   incompatibility **MUST NOT** be collapsed into "not found."
4. No resolution error **MAY** trigger a planner call.
5. Prover and verifier **MUST** resolve the same request to byte-equivalent
   expanded schedules.
6. Runtime resolution **MUST NOT** depend on environment variables, benchmark
   overrides, filesystem discovery, or nondeterministic iteration.

#### Validation boundary

1. Compact rows and expanded schedules are untrusted until validated.
2. One canonical validation implementation **MUST** enforce every security and
   structural property on which verification relies.
3. Validation **MUST** use the same SIS, norm, matrix-envelope, decomposition,
   and setup-capability primitives that verifier equations use.
4. Validation **MUST** reject arithmetic overflow, impossible dimensions,
   inconsistent group descriptors, invalid setup-prefix slots, and unsupported
   capability combinations before allocation or indexing.
5. Verifier-reachable validation **MUST** obey the repository no-panic
   contract.
6. Protocol entry points **SHOULD** accept `ValidatedSchedule`, not an
   unvalidated expanded schedule.

#### Catalog contract

1. Catalog identity **MUST** bind every runtime policy value that can change
   schedule meaning or validation.
2. Catalog identity **MUST NOT** bind offline-only heuristics that cannot
   change emitted rows.
3. Generated rows **MUST** be deterministic and stable under regeneration from
   the same source revision and generator inputs.
4. Duplicate request keys **MUST** be rejected during generation or catalog
   initialization.
5. Every enabled row **MUST** expand and validate.
6. Catalog generation **MUST** fail if a configured required request has no
   valid schedule.
7. A family may intentionally omit unsupported requests, but runtime behavior
   for those requests is rejection.

### Non-goals

This specification does not:

- redesign planner search, objectives, or frontier pruning;
- define the root-precommit optimization objective;
- define setup-prefix placement policy;
- permit prover-supplied schedules;
- require compatibility with old schedule or proof artifacts;
- add a third `akita-schedules`, `akita-schedule-gen`, or `xtask` crate;
- guarantee that every shape formerly accepted by runtime DP remains
  supported.

## Runtime design

### Crate ownership

`akita-schedule` owns:

- the semantic request and catalog-key types;
- the root-precommit recipe schema used at runtime;
- compact schedule rows;
- deterministic expansion;
- runtime policy and catalog identity;
- complete validation;
- generated catalog modules;
- strict lookup and resolution errors.

`akita-planner` owns:

- candidate enumeration and search;
- objectives and frontier pruning;
- diagnostic rejection reasons;
- root-precommit recipe generation;
- setup-prefix placement search;
- compact-row emission;
- the catalog generation binary.

`akita-types` remains the owner of protocol-neutral arithmetic, security, proof,
setup, and schedule data primitives. Schedule-specific validation policy must
not leak into `akita-types` merely to avoid a crate dependency.

### Public request

The request key **MUST** be semantic. In particular, it **MUST NOT** require the
caller to know a planner-selected split or SIS rank before lookup.

A conceptual request is:

```rust
pub struct ScheduleRequest {
    pub groups: Vec<PolynomialGroupLayout>,
    pub witness_chunks: u32,
    pub recursion_mode: RecursionMode,
}
```

The exact representation may use bounded collections or family-specific
wrappers. The following rules are normative:

- group ordering is explicit and transcript-consistent;
- dimensions and polynomial counts are checked before lookup;
- root-precommitted groups are identified by semantic layout and their
  commitment-bound descriptors;
- the final group is identified explicitly;
- setup-prefix parameters, matrix ranks, and fold bases are catalog outputs,
  not request inputs.

### Catalog structure

One runtime catalog may contain two logical sections:

```text
ScheduleCatalog
├── root_precommit_recipes
└── complete_schedules
```

This is not a requirement for two files or two crates.

A complete schedule row that consumes a root-precommitted group **MUST**
reference the canonical recipe expected for that group and catalog family.
Resolution **MUST** compare the descriptor bound into the commitment with the
referenced recipe. A mismatch rejects before proof verification.

All complete rows that claim the same root-precommit compatibility domain
**MUST** reference the same recipe. Catalog consistency tests **MUST** enforce
this invariant.

Setup-prefix commitments are different. Their slots and exact commitment
parameters belong to the compact complete schedule because they are created for
and consumed by a particular recursive transition. They **MUST NOT** be looked
up as reusable root-precommit recipes.

### Compact and expanded schedules

The compact representation **SHOULD** record decisions rather than derived
values. Expansion computes all deterministic consequences through canonical
arithmetic and security primitives.

Expansion **MUST**:

1. validate compact discriminants and bounded integer conversions;
2. derive root, recursive, and terminal parameters;
3. derive setup-prefix slot parameters named by the row;
4. check commitment-bound root-precommit descriptors;
5. produce an expanded schedule;
6. pass that schedule to canonical validation.

Expanded schedules **MUST NOT** be trusted merely because they came from
generated Rust.

### Strict resolution API

A conceptual API is:

```rust
pub fn resolve_schedule(
    catalog: &ScheduleCatalog,
    request: &ScheduleRequest,
    policy: &RuntimeSchedulePolicy,
    setup: &SetupCapability,
) -> Result<ValidatedSchedule, ScheduleResolutionError>;
```

The error type **SHOULD** distinguish:

- missing catalog family;
- unsupported request;
- catalog identity mismatch;
- malformed compact row;
- root-precommit descriptor mismatch;
- setup-prefix slot mismatch;
- insecure or structurally invalid expanded schedule;
- insufficient setup capability.

Public verifier APIs may map these to the existing verifier error surface, but
internal string matching is forbidden.

### Validation pipeline

Validation proceeds in this order:

1. request and catalog identity;
2. bounded decoding and dimension arithmetic;
3. root-precommit recipe consistency;
4. fold topology and cross-level shape consistency;
5. basis, decomposition, and digit-layout consistency;
6. SIS ranks, bounds, and security;
7. setup envelope and setup-prefix slot consistency;
8. transcript-visible ordering and terminal constraints;
9. resource bounds required by verifier-safe allocation.

The validator **MUST** validate values as used, not only family-wide maxima. A
family identity check does not replace per-row validation.

### Configuration surface

Configuration selects an enabled catalog family and supplies runtime capability
values. It **MUST NOT** assemble `PlannerPolicy` or expose planner objectives.

`CommitmentConfig` may provide a method that returns or identifies a catalog,
but schedule resolution itself belongs to `akita-schedule`. Configuration
adapters **SHOULD** delegate only semantic runtime policy and capability data.

### Setup sizing

Setup generation and verifier setup checks **MUST** derive their requirements
from validated catalog rows and setup-prefix slots. They **MUST NOT** rerun the
planner to reconstruct an envelope.

If one setup artifact supports a family of rows, the family envelope **MUST** be
computed offline, emitted with the catalog, and checked against the actual
validated rows.

## Catalog generation

Catalog generation is a developer operation owned by `akita-planner`. The
repository **SHOULD** expose it as a feature-gated planner binary:

```bash
cargo run --release -p akita-planner \
  --features catalog-gen \
  --bin gen_schedule_tables -- \
  crates/akita-schedule/src/generated
```

The implementation need not introduce an `xtask` framework. The generator
**MUST**:

1. enumerate the configured semantic request domain;
2. generate canonical root-precommit recipes where required;
3. search for complete schedules;
4. expand and validate every emitted row through `akita-schedule`;
5. check catalog-wide recipe and identity invariants;
6. write deterministic source;
7. fail on required missing requests or drift in check mode.

The runtime crate does not gain a planner dependency when the generation
feature is enabled. Generation features belong to `akita-planner`.

## Evaluation

### Acceptance criteria

#### Dependency and runtime behavior

- [ ] `akita-verifier`, `akita-config`, and `akita-schedule` have no direct or
  transitive `akita-planner` dependency under every supported feature graph.
- [ ] Repository dependency checks enforce the exclusion.
- [ ] No verifier-reachable symbol calls planner search or enumeration.
- [ ] Prover and verifier both reject a missing catalog row.
- [ ] Catalog identity and validation failures never fall back.
- [ ] Grouped request construction is semantic and planner-free.

#### Catalog and validation

- [ ] Every shipped compact row expands and validates.
- [ ] Every shipped row matches its prior expanded schedule unless a semantic
  change is explicitly reviewed.
- [ ] Root-precommit descriptor mismatches reject before proof verification.
- [ ] Setup-prefix slot mismatches reject.
- [ ] Duplicate keys and inconsistent root-precommit recipe references fail
  catalog checks.
- [ ] Validation obeys the no-panic contract for malformed input.
- [ ] Catalog generation is deterministic.

#### Operational behavior

- [ ] The planner-owned generation command regenerates all enabled families.
- [ ] Generated-table drift checks use the same command and schemas.
- [ ] Runtime binaries do not include planner-only search code.
- [ ] Documentation identifies supported schedule families and rejection
  behavior.

### Test strategy

Tests **MUST** cover:

- catalog hit parity between prover and verifier;
- missing family and missing row rejection;
- malformed compact integers and arithmetic overflow;
- identity mismatch;
- insecure SIS parameters;
- setup capability underflow;
- inconsistent fold topology;
- root-precommit recipe mismatch;
- inconsistent recipe references across complete rows;
- missing and malformed setup-prefix slots;
- default, no-default-feature, and all-feature dependency graphs;
- deterministic regeneration and clean drift checks;
- fuzzed or property-generated malformed rows without panics.

For every currently shipped request, a migration fixture **SHOULD** compare:

- expanded schedule;
- setup requirements;
- transcript-visible schedule events;
- proof serialization shape;
- verifier acceptance.

## Execution

The work is intentionally staged:

1. Freeze representative expanded schedules and failure behavior.
2. Establish `akita-schedule` with compact schemas, expansion, and validation.
3. Move generated catalogs into `akita-schedule`.
4. Replace operational grouped keys with semantic requests.
5. make catalog lookup strict for prover and verifier.
6. Remove every runtime dependency on `akita-planner`.
7. Move generation ownership into a planner binary.
8. Regenerate catalogs and run dependency, parity, and verifier-safety checks.

Planner restructuring may proceed after step 6, using
[`modular-planner-and-precommit-roles.md`](modular-planner-and-precommit-roles.md),
without reopening the runtime dependency boundary.

## Risks

### Validation accidentally depends on planner materialization

Current validation shares code with planner candidate construction and costing.
The migration must identify protocol derivation primitives and move only those
to the runtime boundary. Search-space enumeration and objective computation
remain planner-only.

### Semantic keys omit a schedule-relevant fact

A key that is too small can alias distinct protocol requests. Request-key
design therefore requires parity fixtures and catalog duplicate checks. Adding
planner-derived values to the key is not an acceptable shortcut.

### Generated churn hides semantic changes

Crate moves and regeneration can create large diffs. Handwritten schema and
validation changes should be reviewed separately from generated rows, and
expanded-schedule parity should be reported independently of textual drift.

### Feature unification restores planner reachability

Optional dependencies are insufficient if another feature enables them in a
verifier build. CI must inspect resolved graphs for each supported feature
configuration.

### Setup coverage changes silently

Strict catalogs convert runtime planner success into explicit support policy.
The generated request domain and setup envelope must therefore be reviewed as
protocol support, not merely test data.

## Future work

A future protocol may let the prover transmit part or all of a schedule while
the verifier begins with an incomplete request. That design would require:

- a bounded schedule wire format;
- transcript binding;
- complete validation independent of catalog trust;
- explicit setup-capability negotiation;
- denial-of-service bounds.

Nothing in this specification permits that path today. Until separately
specified, a catalog miss rejects.

## References

- [`modular-planner-and-precommit-roles.md`](modular-planner-and-precommit-roles.md)
- [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- [`distributed-setup-offloading.md`](distributed-setup-offloading.md)
- [`digit-innermost-layout.md`](digit-innermost-layout.md)
- [`../docs/verifier-contract.md`](../docs/verifier-contract.md)
- [`../docs/crate-graph.md`](../docs/crate-graph.md)
