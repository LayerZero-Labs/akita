# Spec: External Schedule Catalog Ownership

| Field         | Value                               |
|---------------|-------------------------------------|
| Author(s)     | Quang Dao                           |
| Created       | 2026-09-03                          |
| Status        | active                              |
| PR            | #428                                |
| Supersedes    | Compiled schedule-catalog ownership |
| Superseded-by |                                     |
| Book-chapter  |                                     |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

Akita schedule rows are protocol parameters selected offline, but the current
delivery mechanism compiles every enabled row into Rust libraries and recursion
programs. That makes executable size grow with the supported workload surface
and gives applications no ordinary state slot in which to own their schedules.
PR #428 moves complete expanded rows into versioned external artifacts, validates
one full catalog at the application or preprocessing boundary, and makes one
`AkitaCommitmentScheme` instance use that catalog for setup, commitment, proving,
and verification.

This is the native full-catalog layer. It lands before any Merkle authority or
authenticated recursion-subset protocol. That staging gives Jolt a stable API
without making its variable trace, advice, bytecode, or program-image surface
part of PR #428. The planner remains offline. `OpeningScheduleSelection` remains
a 32-byte public statement input, and the canonical `AkitaBatchedProof` body
remains headerless and contains no artifact or schedule header.

## Scope and staging

The design has three ordered stages:

1. **Akita external catalog ownership — PR #428.** Store full family artifacts
   outside code, validate them once, size setup from their exact rows, and remove
   compiled table paths.
2. **Jolt full-catalog integration — downstream.** Store the required full
   catalog or catalogs in Jolt verifier preprocessing, restore them without
   planning or global state, and exercise every reachable profile class.
3. **Authenticated recursion subsets — later design.** Bind an approved authority
   to preprocessing and authenticate only the distinct rows used by a recursive
   batch. This stage may use a Merkle multiproof, but its wire format is not part
   of PR #428.

Each stage must be usable and testable before the next stage starts. In
particular, stage 1 MUST NOT add a temporary process-global registry to ease the
stage 2 migration, and stage 3 MUST NOT force changes to Akita's canonical proof
body.

## Intent

### Goal

Replace compiled schedule tables with canonical external artifacts and one
explicitly supplied, instance-owned `TrustedScheduleCatalog` that is the sole
schedule source for exact setup sizing, commitment, proving, and verification.

### Terminology

- A **row** is one expanded `CommittedGroupBatchProfile` and `FoldSchedule` pair.
- A **family artifact** is a versioned external collection of rows for one
  `CommitmentConfig::schedule_family_name()` and planner policy.
- A **trusted catalog boundary** is application setup or verifier preprocessing
  that obtains approved artifact bytes from ordinary storage. Proof decoding is
  not a trusted catalog boundary.
- A **working catalog** is the validated, indexed `TrustedScheduleCatalog` used
  by one scheme instance.
- A **catalog digest** is the deterministic identity of one validated full
  catalog. In PR #428 it is a cache, persistence, and equality identity; it is
  not a membership-proof root.
- A **schedule selection** is the 32-byte `OpeningScheduleSelection` carried by
  the public verification statement or an integration's outer envelope. It is
  not part of the headerless Akita proof body.

### Invariants

- **No compiled schedule data.** Production crates, default features, examples,
  benches, and recursion programs MUST NOT compile generated schedule rows,
  catalog literals, or `include_bytes!` copies of schedule artifacts.
- **External means caller-owned storage.** Akita APIs accept artifact bytes or an
  already validated catalog. They MUST NOT assume a filesystem, database, object
  store, package layout, or global registry. The caller chooses the normal
  storage slot and supplies the data explicitly.
- **Offline planning only.** Setup creation, persisted-setup restoration, proof
  decoding, commitment, proving, verification, and guest execution MUST NOT run
  schedule search. Family artifacts contain complete expanded rows.
- **One runtime owner.** One validated working catalog is injected into an
  `AkitaCommitmentScheme` and is used by setup, commitment, proving, and
  verification. No `TypeId` registry, process-global row registry, ambient
  lookup, or deserialization side effect may change schedule resolution.
- **One semantic audit.** External artifacts and test-only row construction use
  the same `ResolvedScheduleRow` validation path. Prover and verifier do not
  maintain separate row validators.
- **Exact catalog sizing.** Setup matrix capacity and recursive setup-prefix
  slots are derived by scanning the exact validated rows admitted by the supplied
  catalog and the caller's `max_num_vars` and batch bound. Static config tables,
  planner search, and a separately reconstructed key range MUST NOT affect the
  result.
- **Exact lookup.** Honest commitment and proving resolve exact profile keys from
  the working catalog. Verification resolves only the explicit
  `OpeningScheduleSelection` row digest from the same catalog.
- **Headerless proof preservation.** `AkitaBatchedProof` serialization MUST NOT
  gain a catalog, row, selection, version, length, Serde, or bincode header.
  Downstream envelopes MAY serialize the public statement and proof together,
  but that framing remains outside the canonical Akita proof body.
- **Family and policy binding.** Catalog construction validates artifact version,
  protocol epoch, family name, planner-policy digest, row identities, and every
  verifier-consumed row field before returning a working catalog.
- **Canonical artifact identity.** The same validated family and rows produce the
  same canonical artifact bytes and catalog digest. Duplicate prover keys or row
  digests are rejected.
- **Bounded decoding.** Artifact byte length, family-name length, and row count are
  bounded before they can drive unbounded work or allocation. Malformed data
  returns `AkitaError` or `SerializationError` and never panics.
- **Owned setup registries remain allowed.** A setup-owned
  `SetupPrefixProverRegistry` or `SetupPrefixVerifierRegistry` is explicit
  artifact state and is not a forbidden process-global schedule registry.
- **No accidental scope expansion.** This cut removes schedule tables. Fixed SIS
  security tables, operator-norm certificates, and other independently specified
  numeric constants remain governed by their own specifications.

### Jolt workload requirements

Jolt is the target downstream integration. One verifier preprocessing may need
many rows because these inputs can change an exact Akita opening profile:

- padded trace length and final-group arity;
- K16, K256, or another PCS family;
- trusted and untrusted advice presence, capacity, and physical arity;
- bytecode chunk count and chunk arities;
- program-image shape;
- ordered precommitted-group profiles; and
- recursive setup-prefix topology.

RAM, memory-layout, and program parameters that do not change an Akita row still
belong in Jolt preprocessing identity. They do not need duplicate schedule rows.
Values that collapse to the same exact `CommittedGroupBatchProfile` share a row.

The first Jolt integration MUST own the required full catalog or catalogs in
verifier preprocessing or in explicitly referenced preprocessing storage. It
MUST reconstruct the working catalog from artifact bytes without planner calls
or process-global installation. A single preprocessing MAY own multiple family
catalogs, but every Akita scheme instance resolves through exactly one
family-bound working catalog.

Jolt's outer Serde or bincode framing is downstream. Akita exposes canonical
artifact bytes, a working catalog, the statement-level row selection, and the
headerless proof body; it does not prescribe how Jolt packages those pieces.

The downstream profile enumerator MUST derive the finite set of actually
reachable exact lookup profiles. It MUST NOT blindly materialize a Cartesian
product of independent bounds, and it MUST NOT serialize a planner recipe for
later execution during preprocessing restoration or in a guest.

### Non-goals

- Merkle roots, membership proofs, or a recursion-bundle wire format in PR #428.
- Modifying the Jolt repository in PR #428.
- Defining Jolt's outer Serde or bincode representation.
- Adding schedule or artifact fields to `AkitaBatchedProof`.
- Running schedule search during setup restoration, proof generation, proof
  decoding, or verification.
- A mutable network service, `TypeId` registry, or process-global catalog.
- Preserving old generated modules, schedule feature names, setup encodings, or
  artifact bytes. This repository makes no backward-compatibility guarantee.
- Choosing product policy for which Jolt trace, advice, bytecode, or program
  bounds are enabled.

## Evaluation

### Acceptance criteria: PR #428

- [x] `akita-schedules` accepts and emits canonical versioned JSON family
  artifacts containing expanded rows, protocol epoch, family identity, planner-
  policy identity, and produces a deterministic full-catalog digest.
- [x] Artifact decoding rejects empty or larger-than-64-MiB inputs, rows larger
  than 1 MiB, more than 16,384 rows, invalid family names, noncanonical JSON,
  unsupported versions, wrong epochs, wrong families, wrong policies, duplicate
  lookup keys, duplicate row digests, and invalid expanded rows.
- [x] `TrustedScheduleCatalog` is the sole production row index and provides
  canonical row iteration plus exact resolution by selection, lookup key, and
  committed profiles.
- [x] `AkitaCommitmentScheme::new(TrustedScheduleCatalog)` and
  `AkitaCommitmentScheme::from_schedule_artifact(&[u8])` are the production
  construction paths. Scheme clones share the validated catalog through owned
  `Arc` state.
- [x] `AkitaCommitmentScheme::schedules()` exposes a borrow of the exact catalog
  used by all scheme operations; no operation consults config-owned rows.
- [x] `SetupRequirements::from_catalog::<Cfg>(&catalog, max_num_vars,
  max_num_batched_polys)` scans only eligible exact catalog rows and accounts for
  both full schedules and independent precommit matrix requirements.
- [x] Prover setup construction and persisted-setup restoration receive the
  catalog explicitly, derive recursive prefix slots from it, and use its digest
  in catalog-dependent cache identity. Neither path invokes the planner.
- [x] Commitment, proving, and verification primitives receive the same borrowed
  catalog directly. Proof decoding never installs catalog state.
- [x] `CommitmentConfig::schedule_catalog()`, config-owned row-resolution
  methods, `AkitaCommitmentScheme::from_embedded_schedule_catalog`, and all
  generated-table conversion APIs are removed.
- [x] The `schedules-default`, `all-schedules`, and per-family schedule features
  are removed from production crate feature graphs. Selecting a config identifies
  its expected family and policy but does not link row data.
- [x] `crates/akita-schedules/src/generated/` contains no generated row data.
  Reusable audit logic remains only if it is part of the canonical artifact path.
- [x] The planner emitter writes tracked external artifacts under
  `artifacts/schedules/`. Drift CI regenerates into a temporary directory and
  byte-compares the artifacts without compiling them.
- [x] Tests, examples, benches, and profiles load artifact bytes at runtime or
  construct narrowly scoped test catalogs. None relies on an embedded fallback.
- [x] The Akita recursion profile contains no compiled schedule table. If it
  temporarily receives a full catalog through benchmark input, documentation
  clearly states that this is not an authenticated production recursion format.
- [x] `OpeningScheduleSelection` remains exactly 32 bytes in statement or outer-
  envelope serialization, and `AkitaBatchedProof` remains headerless with no
  schedule selection or artifact bytes.
- [x] Existing configurations produce the same expanded rows and proof/setup
  geometry as the pre-removal generated tables. Any catalog-digest change is
  explained by the external artifact identity.
- [x] A linkage check demonstrates that the default Akita profile binary does
  not contain removed row arrays or external artifact payloads.
- [ ] A follow-up linkage check covers an actual Jolt recursion guest or host
  binary after that integration owns preprocessing artifacts.
- [x] The CI feature matrix, Book text, crate graph, generation instructions,
  recursion profile documentation, and PR description match the external full-
  catalog design.

### Acceptance criteria: Jolt full-catalog follow-up

- [ ] Jolt verifier preprocessing owns the required catalog artifacts or explicit
  storage references and reconstructs each working catalog once.
- [ ] Setup restoration, proof decoding, proving, verification, and guest code
  perform no planner provisioning and install no process-global schedule state.
- [ ] Setup capacity is computed from the explicitly supplied catalog that the
  same Akita scheme instance later uses.
- [ ] Preprocessing covers reachable profiles across trace length, PCS family,
  advice, bytecode, program image, and setup-prefix topology without a blind
  Cartesian expansion.
- [ ] Jolt's outer serialization preserves Akita's headerless canonical proof
  bytes and treats `OpeningScheduleSelection` as statement/envelope data.
- [ ] Variable-axis fixtures cover boundary trace lengths, every advice-presence
  combination, bytecode chunk boundaries, program-image variation, and recursive
  setup-prefix use.
- [ ] Preprocessing size, restored-catalog size, setup size, binary size, and
  verification cycles are recorded against the compiled-table baseline.

### Acceptance criteria: authenticated recursion follow-up

- [ ] An approved authority is bound to verifier preprocessing and to the outer
  statement or transcript that makes that preprocessing authoritative.
- [ ] A recursive batch carries only distinct used rows plus a bounded shared
  membership proof; repeated inner proofs reference one validated local
  dictionary.
- [ ] Semantically valid but unauthorized rows are rejected.
- [ ] Multiple preprocessings keep authority namespaces separate.
- [ ] The authenticated-subset format does not alter `AkitaBatchedProof` bytes.
- [ ] Full-catalog and authenticated-subset cycle and input-size measurements are
  reported before the subset format becomes the production recursion path.

### Testing strategy

Before deleting generated modules, generation MUST freeze canonical family
artifacts, expanded-row digest lists, representative setup capacities, recursive
prefix-slot identities, and proof geometry for every supported family. These
fixtures are the parity oracle for the storage cutover.

Unit tests in `akita-schedules` cover canonical JSON, row audit, catalog identity,
lookup, and all decoder bounds. Config tests cover family and policy mismatch.
Setup tests prove that only eligible catalog rows contribute to matrix capacity
and that grouped precommit matrices and recursive prefix slots are not omitted.
PCS end-to-end tests receive explicit catalogs and retain current dense, one-hot,
small-field, bounded-source, multi-chunk, grouped, and recursive coverage.

Serialization tests separately protect the 32-byte statement selection and the
headerless proof body. Downstream envelope round trips do not count as proof-wire
tests because they exercise a different framing layer.

All changed documentation runs `scripts/check-doc-guardrails.sh`. Final
implementation validation uses the current commands in `.github/workflows/ci.yml`
after that workflow is updated to remove schedule-table features and drift jobs.

### Performance

- Full catalog decoding and semantic audit occur once per scheme or preprocessing
  construction, not once per proof.
- Proof bytes do not grow. `OpeningScheduleSelection` remains outside the proof
  body and remains one 32-byte digest.
- Setup size is the exact maximum required by eligible rows in the supplied
  catalog, not the maximum of compiled families or a planner-derived approximation.
- PR #428 records the current default profile binary size. Before/after default
  and recursion measurements remain follow-up evidence and are not claimed by
  this storage cutover.
- A first Jolt full-catalog integration MAY increase preprocessing or guest input
  size. It records that cost explicitly rather than hiding it in the executable.
- The authenticated-subset follow-up is responsible for reducing recursion input
  from all catalog rows to the distinct rows used by one batch.

## Design

### Stage 1 architecture

```text
offline planner
    |
    v
external family artifact
    |
    | application/preprocessing loads approved bytes
    v
TrustedScheduleCatalog
    |
    v
AkitaCommitmentScheme<Cfg> (owns Arc<TrustedScheduleCatalog>)
    |
    +--> exact setup sizing and setup-prefix materialization
    +--> commitment/profile lookup
    +--> proving/profile lookup
    +--> verification/statement selection lookup
```

Artifact storage and approval are application concerns. Akita validates format,
family, policy, and row semantics, but PR #428 does not prove that bytes came from
an operator-approved source. The caller MUST establish that provenance before
constructing the scheme.

### External artifact contract

The host-facing artifact is deterministically formatted canonical JSON for
auditability, readable diffs, and operational tooling. Structural catalog and
row boundaries use line breaks while nested row payloads stay compact. It
contains fully expanded rows; loading never runs planner search. Each serialized
row contains only `schedule`. Its validated root groups determine the final and
ordered precommitted profiles; the artifact does not serialize a second copy.
The runtime retains these derived profiles for borrowed lookup and uses them in
the unchanged semantic row-digest encoding.
`TrustedScheduleCatalog::from_artifact_bytes` checks the configured bounds and
metadata. It borrows raw row slices and enforces the 1-MiB per-row limit before
typed nested collections are decoded. It then reconstructs every
`ResolvedScheduleRow`, sorts rows by digest, builds the separate prover-key
index, rejects duplicates, and recomputes the catalog digest.

`TrustedScheduleCatalog::to_artifact_bytes` emits the same canonical
representation. The repository tracks release and test artifacts under
`artifacts/schedules/`, but libraries MUST NOT use `include_bytes!` or an
equivalent mechanism to link those files into executables.

### Native API contract

The stable native shape is:

```rust
impl TrustedScheduleCatalog {
    fn from_artifact_bytes(...) -> Result<Self, AkitaError>;
    fn to_artifact_bytes(&self) -> Result<Vec<u8>, AkitaError>;
    fn try_new(...) -> Result<Self, AkitaError>;
    fn rows(&self) -> impl ExactSizeIterator<Item = &ResolvedScheduleRow>;
    fn resolve_selection(...) -> Result<&ResolvedScheduleRow, AkitaError>;
    fn resolve_key(...) -> Result<&ResolvedScheduleRow, AkitaError>;
    fn resolve_profiles(...) -> Result<&ResolvedScheduleRow, AkitaError>;
    fn catalog_digest(&self) -> [u8; 32];
}

impl<Cfg: CommitmentConfig> AkitaCommitmentScheme<Cfg> {
    fn new(catalog: TrustedScheduleCatalog) -> Result<Self, AkitaError>;
    fn from_schedule_artifact(bytes: &[u8]) -> Result<Self, AkitaError>;
    fn schedules(&self) -> &TrustedScheduleCatalog;
}
```

The config-bound construction path validates `schedule_family_name()` and
`policy_of::<Cfg>()` once. `AkitaCommitmentScheme::new` takes ownership and stores
the catalog in `Arc`; cloning a scheme does not clone or revalidate row bodies.
Applications that persist preprocessing serialize canonical artifact bytes or an
explicit storage reference, then reconstruct the working catalog once during
restoration.

Low-level setup, commitment, prover, and verifier functions accept
`&TrustedScheduleCatalog` directly when they need row resolution. They do not
reconstruct it from `Cfg`, use associated static tables, or fetch it from ambient
state.

### Exact setup capacity

`SetupRequirements::from_catalog::<Cfg>` validates the catalog/config binding and
the requested bounds, then scans the supplied catalog once. A row contributes its
full schedule matrix capacity only when its exact lookup key fits both requested
bounds. Each precommitted profile can independently contribute its commit-only
inner/outer matrix requirement when that profile fits the bounds, even if the
larger grouped row does not.

The same scan collects recursive prefix slots for eligible rows.
`SetupRequirements` carries the matrix capacity and ordered prefix IDs through
cache loading, coverage checks, repair, and fresh setup generation. These paths
reuse the requirements instead of rescanning the catalog. Setup generation
materializes exactly those required slots. Persisted setup restoration
receives the catalog explicitly, validates the stored registry against the exact
required identifiers, and keys catalog-dependent persistence by
`catalog_digest()`.

This is a completeness rule, not only an optimization. Setup sizing MUST NOT use
a second schedule resolver or planner-derived approximation because that could
produce a setup that the verifier's admitted catalog exceeds.

### Proof and integration wire boundary

Akita's proof body is schedule-shaped and headerless. The row digest is public
statement context used to resolve the shape before decoding or verifying the
proof. It is not a field of `AkitaBatchedProof`.

An integration may serialize this tuple in any outer format:

```text
(verifier preprocessing, public statement, OpeningScheduleSelection,
 canonical AkitaBatchedProof bytes)
```

Serde, bincode, postcard, or a Jolt-specific blob may frame that tuple. Such
framing is not Akita proof serialization and MUST NOT be pulled into the core
proof type merely to simplify one integration.

### Deferred authenticated recursion layer

Supplying a complete full catalog as untrusted guest input does not establish
operator approval. It is safe only when the exact catalog identity is already
bound by trusted preprocessing or the outer statement, or when used as an
explicitly non-production benchmark fixture.

A later specification will define the authenticated recursion layer. It should
bind a program- or preprocessing-specific authority, qualify rows by family,
carry only distinct used rows, verify one bounded shared membership proof, audit
each row once, and build an immutable guest-local resolver. A Merkle multiproof is
the leading design, but PR #428 deliberately does not freeze its domains, padding
rule, or wire encoding.

### Alternatives considered

**Compile all families.** Lookup is simple, but binary size scales with every
supported workload and integrations must predict their future row surface at
compile time. This is the design being removed.

**Run the planner during restoration.** This makes preprocessing restoration
slow and policy-dependent, duplicates offline selection logic, and cannot work
cleanly in constrained guests. Artifacts contain expanded rows instead.

**Install rows in a process-global registry.** This hides authority, makes results
depend on process history, complicates concurrent preprocessings, and creates
deserialization side effects. Scheme and preprocessing ownership remain explicit.

**Design the Merkle wire before the native API.** This couples #428 to unresolved
Jolt batching and outer-statement choices. The full-catalog contract is useful on
its own and is also the source from which a later authenticated subset is built,
so it lands first.

**Carry a full catalog in every recursive input permanently.** This removes
compiled data but repeats bytes and semantic audit. It is acceptable as a clearly
marked bring-up baseline, not the final production recursion format.

**Add schedule metadata to the Akita proof.** This would break the headerless
canonical body and conflate a statement/integration concern with proof messages.
The public row selection remains outside the proof.

## Documentation

PR #428 updates `book/src/how/configuration.md`,
`book/src/how/architecture.md`, `book/src/how/verification.md`,
`docs/crate-graph.md`, `docs/ci-test-timing.md`, and
`profile/akita-recursion/README.md`. These pages describe only the shipped full-
catalog layer as current behavior. They may name authenticated subsets as a
deferred design but MUST NOT present a Merkle format or downstream Jolt migration
as implemented.

After implementation and Book folding, set `Book-chapter` to
`book/src/how/configuration.md`, mark this spec implemented, and archive it under
the applicable quarter so the live-spec set returns to its steady-state cap.

## Execution

The order is normative because deleting generated rows too early erases the
parity oracle, while integrating Jolt before native ownership stabilizes would
encode another temporary registry or restoration path.

1. **Land the storage and ownership contract.** Review this spec, especially the
   trusted catalog boundary, exact setup-sizing rule, headerless proof boundary,
   and separation from authenticated recursion.
2. **Capture parity fixtures.** From the current generated modules, emit every
   family artifact, expanded-row digest list, representative setup capacities,
   recursive prefix-slot identifiers, proof geometry, and binary-size baseline.
3. **Harden the full artifact type.** Complete bounded canonical JSON decoding,
   family/policy validation, duplicate rejection, row audit, digest computation,
   exact indexes, and round-trip tests in `akita-schedules`.
4. **Stabilize native ownership and sizing.** Finalize
   `TrustedScheduleCatalog`, `AkitaCommitmentScheme::new`,
   `from_schedule_artifact`, `schedules`, explicit low-level catalog borrows,
   `SetupRequirements::from_catalog`, and catalog-driven prefix-slot materialization.
5. **Migrate restoration and runtime callers.** Make setup restoration, tests,
   examples, benches, profiles, commitment, proving, and verification use the
   explicitly supplied catalog. Remove planner and ambient-registry behavior from
   every restoration and execution path.
6. **Move generation into normal storage.** Change the planner emitter and drift
   scripts to write `artifacts/schedules/`, regenerate, and prove parity with the
   fixtures from step 2.
7. **Remove compiled schedule ownership.** Delete generated row modules,
   config-owned row APIs, embedded constructors, schedule feature flags, stale CI
   feature matrices, and fallback call sites. Add source, metadata, and linkage
   guards against their return.
8. **Validate the full-catalog cutover.** Run cheap preflight, artifact drift,
   feature-graph Clippy, PCS end-to-end and negative tests, persisted-setup tests,
   recursion smoke/profile, documentation guardrails, and binary-size reports.
   Update the PR description with the final native API and compatibility break.
9. **Integrate Jolt full catalogs downstream.** Add preprocessing-owned artifact
   storage/restoration, enumerate variable profile classes, use exact catalog
   setup sizing, preserve outer framing and headerless proof bytes, and remove
   lazy planner or global-registry behavior. Do not edit Jolt from this Akita task.
10. **Design authenticated recursion subsets separately.** After full-catalog
    measurements, specify preprocessing authority binding, row qualification,
    shared membership proofs, guest-local dictionaries, bounds, tamper tests, and
    cycle/input-size targets.

The implementation MUST pause for focused review after steps 4 and 7. Step 4 is
the API that Jolt will consume. Step 7 is the irreversible removal of the old
parity source and feature surface.

## References

- [PR #428](https://github.com/LayerZero-Labs/akita/pull/428)
- [`setup-offloading-planner.md`](setup-offloading-planner.md) for planner and
  recursive setup selection policy
- [`archive/2026-Q3/runtime-schedule-boundary.md`](archive/2026-Q3/runtime-schedule-boundary.md)
  for the earlier rule that proof bytes do not supply schedule policy
- [`book/src/how/configuration.md`](../book/src/how/configuration.md)
- [`book/src/how/verification.md`](../book/src/how/verification.md)
- [`profile/akita-recursion/README.md`](../profile/akita-recursion/README.md)
