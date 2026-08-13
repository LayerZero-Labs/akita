# Review of `parameter-struct-consolidation.md`

## Verdict

Do not approve the specification as written.

The consolidation goal is good. The code has genuine duplicated fields,
validators, encoders, and expansion paths. However, the proposed target model
introduces invalid states, conflicts with the repository's typed-topology
design, and contains a recursively sized type that cannot compile.

The safer direction is to consolidate shared components while retaining
distinct semantic boundary types for frozen profiles, executable levels,
terminal parameters, and schedule stages.

## Blocking findings

### 1. `CommitParams` is recursively sized and cannot compile

The proposed `CommitKind::Level` contains
`Option<SetupPrefixSlotId>` ([proposal, line 278](parameter-struct-consolidation.md#L278)).
The existing ownership chain is:

1. `SetupPrefixSlotId` owns `PrecommittedLevelParams`
   ([`setup_prefix.rs`](../crates/akita-types/src/proof/setup_prefix.rs#L36)).
2. `PrecommittedLevelParams` owns `CommittedGroupProfile`
   ([`precommitted.rs`](../crates/akita-types/src/layout/params/precommitted.rs#L20)).
3. The proposal replaces `CommittedGroupProfile` with `CommitParams`.

The result is this by-value cycle:

```text
CommitParams
  -> CommitKind::Level
    -> SetupPrefixSlotId
      -> PrecommittedLevelParams
        -> CommitParams
```

There is no indirection in this path. Rust will reject the definition as an
infinitely sized recursive type.

Putting `SetupPrefixSlotId` behind `Box` could break the cycle, but that would
add allocation, complicate static generation, and materially change the
proposal. Keeping `CommittedGroupProfile` separate also breaks the cycle, but
that abandons the central three-to-one unification.

### 2. The setup-prefix source of truth is reversed

The proposal deletes `RecursiveFoldParams.incoming_setup_prefix` and reads
`witness.setup_prefix` instead
([proposal, line 391](parameter-struct-consolidation.md#L391)).

That contradicts both the current code and the typed-topology design:

- `CommittedGroupParams::setup_prefix` is explicitly documented as a derived
  mirror. `RecursiveFoldParams::incoming_setup_prefix` is authoritative
  ([`params.rs`](../crates/akita-types/src/layout/params.rs#L128)).
- The topology specification says that the successor edge is the canonical
  statement that the predecessor offloaded setup
  ([`typed-schedule-topology-cutover.md`](typed-schedule-topology-cutover.md#L100)).
- `FoldSchedule::validate_structure` currently enforces agreement between the
  authority and mirror
  ([`schedule.rs`](../crates/akita-types/src/schedule.rs#L361)).

The correct consolidation is the opposite: remove
`CommittedGroupParams::setup_prefix` and retain the successor-owned schedule
edge. Planner or sizing code that needs the edge should receive it as context
instead of storing a second copy in group parameters.

### 3. The `kind + Option` model does not make invalid states impossible

The proposed representation stores the same state in two forms:

```rust
kind: CommitKind,
outer: Option<RoleParams<Outer>>,
open: Option<RoleParams<Open>>,
```

The claim that "with `Option` there is nothing to compare" is incorrect. The
proposed `validate()` must compare `kind` against both options
([proposal, line 342](parameter-struct-consolidation.md#L342)). That is a
parallel-tag invariant of exactly the kind that the document seeks to remove.

The example also makes `kind` public. A caller can construct a valid `Level`
and then assign `CommitKind::Terminal`, leaving both optional roles present.
Making every field private would close that public mutation path, but the
internal representation would still admit invalid states.

A true sum type would place role data inside each enum variant:

```text
Frozen  { blocks, inner, outer, ... }
Level   { blocks, inner, outer, open, ... }
Terminal { blocks, inner }
```

Common accessors could expose `blocks` and `inner`. In this repository,
retaining distinct public wrappers around shared components is likely simpler
and preserves stronger semantic boundaries.

### 4. A unified `FoldStep` weakens the typed schedule topology

The proposed `FoldStep` contains a stage enum and an optional output length
([proposal, line 396](parameter-struct-consolidation.md#L396)). It permits all
of these invalid states:

- A terminal stage in `FoldSchedule.root`.
- A root stage inside `recursive_folds`.
- A recursive stage with no output length.
- A terminal stage with an output length.

This conflicts directly with the architectural requirement that root,
recursive, and terminal roles use different Rust types
([`typed-schedule-topology-cutover.md`](typed-schedule-topology-cutover.md#L80)).

The three existing wrappers are not accidental duplication. They encode
different topology and arity. Their small amount of repeated syntax provides
useful compile-time guarantees. Keep them unless the entire schedule is
intentionally redesigned as a validated sequence, which would be a different
and much larger proposal.

### 5. Step 6 reintroduces the duplication that the proposal claims to remove

`CommitKind::Level` contains `precommitted_groups`
([proposal, line 283](parameter-struct-consolidation.md#L283)). Later,
`FoldStage::Root` also contains `precommitted_groups`
([proposal, line 399](parameter-struct-consolidation.md#L399)).

The root final group and root stage would therefore continue to own parallel
copies of the same parameters. Their equality audit cannot disappear.

The target schema is also internally stale:

- Step 5 refers to `final_group.geometry.open`, although the proposed
  `CommitParams` has `blocks` and `open`, not `geometry`.
- Step 6 continues to use `CommittedGroupParams` and
  `TerminalCommittedGroupParams` after Step 4 deletes or aliases them.
- Step 6 retains root precommitted groups even though the claimed result says
  the corresponding mirrors are gone.

The specification needs one complete final schema, separate from temporary
migration states, so its invariants can be checked as a whole.

### 6. `LevelParamsLike` is not eliminated by the proposed unification

`LevelParamsLike` is implemented for:

- `CommittedGroupParams`
- `PrecommittedLevelParams`

It is not a bridge between `CommittedGroupParams`, `CommittedGroupProfile`, and
`TerminalCommittedGroupParams`. Unifying those three types does not unify the
two implementations that caused the trait to exist.

This matters especially for the D role. A precommitted group carries an
opening basis and opening digit depth, but it does not own the D matrix; the
consuming fold owns the shared matrix. Therefore the assertion that basis,
digit depth, and matrix "always move together" is false for a type explicitly
included in the proposal's inventory.

Deleting `LevelParamsLike` requires either:

- A shared embedded component that full and precommitted group parameters both
  own.
- A consumer-by-consumer refactor to accept the smaller capabilities each
  operation actually needs.

The proposed profile/level/terminal unification does not provide either one.

### 7. Terminal projection validation is lost

The existing `TerminalCommittedGroupParams::try_from_expanded_group` does more
than copy fields. It validates the terminal matrix's certified response
capacity and the planner's minimum-retention heuristic.

The proposed constructor:

```rust
CommitParams::terminal(blocks, inner) -> Self
```

does not receive the sparse challenge configuration or other inputs needed to
reproduce those checks. Nevertheless, the proposal says that the existing
projection functions will disappear.

The specification must identify the new canonical location for every check
performed by `try_from_expanded_group`. Merely adding a structural
`CommitParams::validate()` is insufficient.

The same issue applies at consumption sites. A function intended only for
terminal parameters could accept any `CommitParams` and read its unconditional
`inner` fields without checking `kind`. This changes a compile-time guarantee
into an unchecked semantic convention. Combined with the unified `FoldStep`,
that is a significant verifier-boundary regression.

## Wire format and descriptor review

### One atomic `RoleParams` encoder cannot preserve current byte order

The current descriptor formats arrange equivalent fields differently:

- Level parameters encode all three bases, then all three matrices, then block
  geometry, then all digit depths
  ([`params.rs`](../crates/akita-types/src/layout/params.rs#L513)).
- Frozen profiles encode block geometry, followed by each role's basis, depth,
  and matrix
  ([`profiles.rs`](../crates/akita-types/src/schedule/profiles.rs#L65)).
- Terminal parameters use another order: A basis, A matrix, block geometry,
  then A digit depth.

A single atomic `RoleParams::append_descriptor_bytes` cannot be called by all
three larger encoders while preserving each existing order.

The specification must choose one explicit policy:

1. Preserve existing bytes by sharing smaller field-level encoding helpers.
2. Intentionally change formats and bump every affected version, protocol
   epoch, identity digest, and generated catalog.

Storage layout and canonical encoding order should be designed separately.
Changing one should not silently change the other.

### Public wire encoding is missing from the migration plan

`CommittedGroupProfile` is embedded in public committed-group and setup-prefix
wire data. It has manual serialization and deserialization that checks version,
role, matrix identity, and geometry
([`commitment.rs`](../crates/akita-types/src/proof/commitment.rs#L237)).

The proposal discusses canonical descriptor bytes but does not provide a plan
for these actual wire encoders. If the public parameter type becomes
`CommitParams`, every wire boundary must prove that it constructed the
`Frozen` case and reject `Level` and `Terminal` cases. That validation cannot
be optional or deferred to an unrelated caller.

The repository permits breaking changes, but a deliberate wire break still
needs an explicit version and regeneration plan.

### One known-schedule test is insufficient

The suggested byte-for-byte check for one known schedule is too narrow. The
migration should cover:

- Every checked-in generated catalog.
- Single-group and multi-group roots.
- Setup-prefix schedules.
- Chunked and non-chunked schedules.
- Recursive and terminal descriptors.
- Frozen committed-group wire fixtures.
- Setup-prefix wire fixtures.
- Malformed role tags, kinds, lengths, and matrices.

If bytes are meant to remain stable, compare all shipped table entries before
and after each relevant step. If bytes intentionally change, test the version
or epoch rejection behavior as well.

## Generated tables, trait derivations, and performance

Several cost claims in Section 9 need correction.

### Static construction is not solved by `Vec::new()` being const

Generated schedule structs embed `CommittedGroupProfile` and derive `Copy`
transitively
([`generated/mod.rs`](../crates/akita-schedules/src/generated/mod.rs#L47)).
Generated Rust code currently constructs profiles with direct struct literals
from another crate
([`emit/mod.rs`](../crates/akita-planner/src/emit/mod.rs#L319)).

The proposal makes optional role fields private and shows non-const
constructors. The generated-table crate could therefore neither write the
struct literal nor call the constructor in a static initializer. Whether
`Vec::new()` is const does not solve that problem.

### `Hash` does not work without further changes

`Vec<T>` implements `Hash` only when `T: Hash`.
`PrecommittedLevelParams` currently implements equality but not `Hash`.
Therefore `CommitKind::Level` and the proposed unified `CommitParams` cannot
simply derive `Hash` without additional work. The statement that no change is
needed is incorrect.

### The `Copy` impact is wider than stated

Losing `Copy` affects generated precommitted groups and every generated parent
that derives `Copy`, not only a few profile call sites. It also changes emitted
literal handling and collection code. The proposal needs a complete derive and
const-construction audit.

### The size estimate is too optimistic

The largest `CommitKind` variant contains a `SetupPrefixSlotId`, which owns
substantial precommitted commitment parameters. If the recursive type were
fixed by boxing, the trade would become heap allocation and pointer chasing.
If it were flattened, a frozen profile would likely grow by hundreds of bytes,
not merely a few tens.

Measure at least:

- `size_of` for every old and proposed type.
- Generated binary or static-table size.
- Schedule-key clone and hash cost.
- Commit lookup latency.
- Allocation count if indirection is introduced.

## Inventory and factual issues

The inventory is useful, but several statements should be corrected before it
drives implementation:

- The generated-mirror list contains twelve named types, although the document
  calls them eleven.
- Static data can contain a `Vec`; the real generated/runtime distinction is
  compact representation, checked expansion, policy re-audit, const-friendly
  emission, and table size.
- `CommitmentGeometry` currently converts from `CommittedGroupParams`; it does
  not provide the claimed common conversion from `CommittedGroupProfile`.
  `CommitInnerPlan` has the two constructors instead.
- The three principal owned types do not differ only by role count. A frozen
  profile is versioned public data, an ordinary level is executable schedule
  state, and the terminal type is the output of a security-sensitive
  projection.
- Generated table mirrors are not inherently duplicate authorities. They are
  compact planner inputs that are expanded and re-audited into runtime
  parameters. That trust-boundary purpose should be preserved even if their
  internal code is consolidated.

The inventory should add columns for:

- Canonical owner versus derived mirror.
- Public wire, transcript/descriptor, generated input, or runtime-only.
- Validation boundary.
- Const and `Copy` requirements.
- Topology role.
- Whether conversion is a projection, expansion, or simple field copy.

These distinctions are more important than shared field names.

## Parts of the proposal worth retaining

### Generic matrix implementation

An internal `CommitMatrixParams<R>` is a reasonable way to remove the repeated
macro implementation. Recommended constraints:

- Seal `MatrixRole` so downstream crates cannot invent marker roles.
- Retain public role-specific aliases or wrappers for protocol readability.
- Validate that the stored `SisTableKey.role` agrees with `R::ROLE` at every
  checked construction and deserialization boundary.
- Preserve role-specific function signatures where accepting the wrong role
  would be a protocol error.

### `BlockGeometry`

Extracting exact `N`, `M`, and live block count into one validated value object
is sound. It should also own or expose the checked domain computations that
currently validate these fields together.

### Shared role components

Grouping a basis, digit depth, and matrix is useful where all three share the
same owner. Do not claim that this applies universally, especially to the
consumer-owned D matrix used for precommitted groups.

### Removing actual mirrors

Delete mirrors only after documenting the correct authority. Likely candidates
include duplicated D matrices and fold challenge configuration. The
setup-prefix mirror must be deleted from group parameters, not from the
successor edge.

### Descriptor helper extraction

Small encoding primitives can remove mechanical repetition without coupling
storage layout to byte order. This is safer than requiring every containing
type to encode an entire `RoleParams` atomically.

### Generated expansion consolidation

`expand.rs` deserves consolidation, but generated structures should remain
compact checked inputs. Share expansion primitives while preserving the
generated/runtime validation boundary.

## Recommended design direction

Prefer layered composition over universal unification:

1. Introduce a validated `BlockGeometry`.
2. Introduce a generic internal matrix implementation and role-specific public
   names.
3. Introduce role components where basis, depth, and matrix genuinely share
   ownership.
4. Keep `CommittedGroupProfile`, executable level parameters, and terminal
   parameters as distinct semantic wrappers.
5. Keep typed root, recursive, and terminal schedule steps.
6. Remove `CommittedGroupParams::setup_prefix`; retain
   `RecursiveFoldParams::incoming_setup_prefix` as the topology authority.
7. Choose exactly one owner for root precommitted groups.
8. Refactor `LevelParamsLike` according to its actual full/precommitted
   consumers, independently of profile/terminal consolidation.
9. Separate canonical encoding policy from in-memory composition.
10. Preserve every validation performed by frozen-profile admission, runtime
    level expansion, terminal projection, and schedule topology validation.

A plausible structure is to share an ordinary A/B commitment core while
retaining semantic wrappers:

```text
BlockGeometry
CommitMatrixParams<Role>
RoleParams<Role>
StandaloneCommitCore { blocks, inner, outer }

CommittedGroupProfile {
    version,
    group,
    commitment: StandaloneCommitCore,
}

CommittedGroupParams {
    commitment: StandaloneCommitCore,
    open,
    level-only execution fields,
}

TerminalCommittedGroupParams {
    blocks,
    inner,
}
```

This reduces field declarations and validator duplication without letting a
wire profile masquerade as executable schedule state or a normal level
masquerade as terminal parameters.

## Required specification revisions before implementation

Before implementation starts, the revised document should include:

1. A complete final target schema that is free of old-name migration aliases.
2. An authority table for every duplicated field.
3. A proof that the target types are finite-sized and statically constructible.
4. A validation-path table showing where every current security and geometry
   check moves.
5. A wire and canonical-descriptor compatibility policy.
6. A generated-table regeneration and catalog-identity policy.
7. A compile-time invalid-state analysis for commitment kinds and fold stages.
8. A complete `Copy`, `Hash`, `Eq`, const-construction, size, and allocation
   impact assessment.
9. A consumer map for removing `LevelParamsLike`.
10. Golden tests covering all shipped catalogs and public serialized forms.

## Final assessment

The document correctly identifies excess duplication, but it currently treats
distinct trust boundaries and topology roles as if they were only repeated
field shapes. Those distinctions are intentional and security-relevant.

Accept the component-level consolidation ideas. Reject the unified
`CommitParams` and unified `FoldStep` designs in their current form. Revise the
plan around shared validated components, correct ownership, typed topology,
and explicit wire and validation boundaries.
