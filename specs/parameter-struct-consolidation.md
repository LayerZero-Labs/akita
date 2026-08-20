# Spec: Parameter Struct Consolidation

| Field | Value |
|---|---|
| Author(s) | Omid Bodaghi |
| Created | 2026-08-12 |
| Revised | 2026-08-19 (against `main` @ `8e552d2ac`; steps 0-4 landed) |
| Status | active |
| PR | |
| Supersedes | |
| Superseded-by | |
| Book-chapter | book/src/how/configuration.md |

## Summary

Akita spreads one set of commitment parameters across 24 inventory entries. Ten
runtime audits exist only to check that a value still equals a copy of itself,
two more mirror pairs have no audit at all, and one security-relevant quantity —
the certified terminal response cap — is computed by two functions that
**disagree**. The `LevelParamsLike` trait, 22 methods over 2 implementations,
exists to paper over the cause.

The cause is one asymmetry. A fold's final group is stored as flat fields on
`CommittedGroupParams`, next to the D matrix that every group in the fold shares.
A precommitted group is a `PrecommittedLevelParams`. A setup prefix is a
`ScheduledSetupPrefix`. All three describe the same thing — one group in one
fold's opening batch — but the final group cannot share their type, because it is
carrying the fold's shared state.

This spec moves the shared D matrix up to the fold and gives every group one
type. That single change removes the trait, the duplicate constructors, the
borrowed views, all ten mirror audits, both unaudited mirror pairs, and the
divergent security calculation. It replaces 24 entries with 10 types without
merging a single trust boundary.

## Intent

### Goal

Give every commitment group in a fold the same Rust type, and make the fold own
the state its groups share.

The target is five types plus five kept unchanged:

```rust
// Leaf components (new).
pub struct BlockGeometry { live_ring_elements_per_claim, positions_per_block, live_blocks }
pub struct GadgetDigits  { log_basis: u32, num_digits: usize }
pub struct LinfCommitMatrix<R: LinfMatrixRole> { /* replaces a 2-role macro */ }
pub struct RoleParams<M>  { digits: GadgetDigits, matrix: M }

/// Frozen at commit time. Public, versioned wire form. Renamed from
/// `CommittedGroupProfile`; same bytes, same `VERSION` of 4, same 288 bytes.
pub struct GroupCommitPhaseParams {
    pub version: u8,
    pub group: PolynomialGroupLayout,
    pub blocks: BlockGeometry,
    pub outer_slice_count: CommitmentSliceCount,
    pub inner: InnerRoleParams,
    pub outer: OuterRoleParams,
}

/// One group in one fold's opening batch: the final/new group, any precommitted
/// group, or the setup prefix. Replaces `PrecommittedLevelParams`, and absorbs
/// `ScheduledSetupPrefix`.
pub struct GroupOpenPhaseParams {
    pub profile: GroupCommitPhaseParams,   // commit-phase identity
    pub opening: GroupOpeningPlan,         // policy chosen by the consuming fold
    pub setup_natural_len: Option<usize>,  // `Some` iff this is the prefix
}

/// One fold level: root or recursive. Owns what its groups share.
pub struct FoldParams {
    pub payload_mode: CommitmentPayloadMode,
    pub source_encoding: CommittedSourceEncoding,
    pub groups: Vec<GroupOpenPhaseParams>, // ordered; prefix at 0, final group last
    pub open_matrix: OpenCommitMatrixParams, // the shared D matrix, stored once
    pub witness_chunk: ChunkedWitnessCfg,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

/// The last fold. No outer/open matrix, no groups, no chunking, no successor.
pub struct TerminalFoldParams {
    pub blocks: BlockGeometry,
    pub inner: InnerRoleParams,
    pub fold: GadgetDigits,
    pub fold_challenge_config: SparseChallengeConfig,
    pub response_shape: TerminalResponseShape,
    pub input_witness_len: usize,
}

pub struct FoldSchedule {
    pub root: FoldParams,
    pub recursive_folds: Vec<FoldParams>,
    pub terminal: TerminalFoldParams,
}
```

Kept unchanged: `InnerCommitMatrixParams` (it carries
`InnerCommitSecurityRoute`, so it cannot join the generic), `GroupOpeningPlan`,
`OpeningMethod`, `CommittedSourceEncoding`, `SetupPrefixSlotId`.

Deleted: `LevelParamsLike`, `ScheduledSetupPrefix`, `RootFinalGroupParams`,
`RootPrecommittedGroupParams`, `RootFoldParams`, `RecursiveFoldParams`, the three
`*FoldStep` wrappers, `TerminalCommittedGroupParams`, `WitnessPartition`, and all
four borrowed views (`CommitmentGeometry`, `FoldScheduleDescriptorStep`,
`TerminalFoldDescriptor`, `OpeningExecutionGroup`).

### Invariants

| Invariant | Protected by |
|---|---|
| `GroupCommitPhaseParams` wire bytes and `VERSION = 4` are unchanged | `CommittedGroup<F>` deserialize ([proof/commitment.rs:443-489](../crates/akita-types/src/proof/commitment.rs#L443-L489)); new golden wire fixtures |
| Catalog `key_digest` and entry sort order change **exactly once**, in step 6 | Per-family `CatalogIdentityExpectation`; asserted separately at step 5 and step 6 |
| No type loses `Copy` or const-constructibility | 30 struct literals and 60 `new_unchecked` calls sit in `static` position across 13 generated catalogs; `cargo build` is the gate |
| No new heap allocation | `FoldParams` allocates the one `Vec` that `CommittedGroupParams` already allocated. No `Box`, no `Arc` |
| Verifier never panics | `FoldParams::final_group()` returns `Result`, never indexes |
| Every validation that exists today has exactly one destination | Validation-path audit (see Testing Strategy); no check is dropped |
| Exactly one function computes the certified terminal response cap | Prerequisite in Execution step 0; after the merge no config argument exists to pass wrongly |
| Prover and verifier read the same group parameters | Both consume `FoldParams::groups`; the erasure point that let them diverge is deleted |
| A generated row's A digit depth still equals `ceil(log_commit_bound / log_basis_inner)` for its level | `audit_committed_params` ([audit.rs:216-230](../crates/akita-schedules/src/audit.rs#L216-L230)), kept unchanged |

### Non-Goals

- **Moving `source_encoding` onto the commit-phase type.** More honest, but costs
  a version bump from 4 to 5, new profile bytes, and a new `key_digest` for all
  13 catalogs. Deferred; see Alternatives.
- **Consolidating the three ordered-profile containers** —
  `AkitaScheduleLookupKey`, `CommittedGroupBatchProfile`,
  `PrecommittedGroupProfiles`. They touch the catalog lookup path this spec
  deliberately keeps byte-stable.
- **Removing the apparently dead `Hash` derives** on `AkitaScheduleLookupKey` and
  the commit-phase type. Independent cleanup.
- **Renaming the `profile` and `opening` fields** to match the new type names.
  Would touch every accessor path; separate decision.
- **Changing `FoldSchedule`'s shape.** It keeps three named fields with distinct
  types, so exactly one root and exactly one terminal stay guaranteed by the type.
- **Any change to proof structure, security argument, or planner objective.**

## Evaluation

### Acceptance Criteria

- [ ] `LevelParamsLike` does not exist. No `&dyn LevelParamsLike` remains.
- [ ] `FoldParams::groups` is the only place a fold's groups are stored, and
      `open_matrix`, `witness_chunk`, and `source_encoding` each have one owner.
- [ ] All ten mirror-equality audits are deleted, not preserved: `audit.rs:502`,
      `:503`, `:504`, `:519`, `:520`, `:521`, `:550`; `schedule.rs:440`;
      `expand.rs:95`, `:96`, `:116`.
- [ ] Both previously unaudited mirror pairs are gone: the
      `sparse_challenge_config` fields, and `witness_partition` vs
      `witness_chunk`.
- [ ] One function computes the certified terminal response cap, and it takes no
      `SparseChallengeConfig` argument.
- [ ] `GroupCommitPhaseParams` descriptor and wire fixtures are byte-identical to
      today's `CommittedGroupProfile` fixtures, per family.
- [ ] Catalog sort order is unchanged; `key_digest` changes once, at step 6.
- [ ] Generated mirror types go from 15 to 8; `expand.rs`'s three parallel
      expansion paths become one.
- [ ] `scripts/generate-schedule-tables.sh` produces no diff after each step.
- [ ] Measured `size_of` matches the Performance table within the stated bounds.
- [ ] Full CI matrix passes at every step, under each explicitly selected
      transcript backend.

### Testing Strategy

**Build the fixture harness in step 1, before any type change.** It must cover
all 13 generated catalogs at every level — commit-phase params, opening plan,
group, fold, schedule — plus `ScheduleRowDigest` per row, `key_digest` and
`CatalogIdentityExpectation` per family, and `CommittedGroup<F>` wire round-trips.

Coverage must include single-group roots, multi-group roots, setup-prefix
schedules, chunked and unchunked schedules, recursive folds, terminal folds, and
these four paths that are newest and most likely to break silently: B-sliced
groups (`outer_slice_count > 1`), subring-packing folds, terminal L2 routes, and
bounded committed dense sources (`fp128_dense_bounded`). The grouped
`fp128_onehot` row is the only one where two groups in one fold carry different A
digit depths; it must be in the set.

**Byte-stability assertions.** Steps 1–4 produce zero fixture diffs. Step 5
changes fixtures above the commit-phase type and must leave commit-phase fixtures
and catalog sort order unchanged — assert that explicitly, per family.

**Rejection tests.** Each must fail closed. Grouped by what they protect:

- *Wire and identity*: version other than 4 in both directions; unknown modulus,
  policy, or role tag; a matrix role that does not match its slot; a rank, width,
  or L-infinity bound the SIS table does not certify; a coefficient count above
  `MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS`; `protocol_epoch` of 1 after the bump.
- *Geometry*: `M` not a power of two; `B != ceil(N / M)`;
  `N · d_a != 2^num_vars`; an inadmissible `outer_slice_count`.
- *Security routes*: an L2 A route where the boundary requires `Linf` (root final
  group, precommitted group, terminal `Linf` cap); an L2 terminal route that also
  carries a scheduled `Linf` cap; a terminal matrix retaining less than half the
  unconstrained target.
- *Opening policy*: an opening basis below the frozen outer basis; a fold whose
  groups disagree about the opening-method family; a nonterminal level using the
  wrong family for its absolute level; a packing group whose
  `challenge_subring_dimension` does not validate the challenge family; a packing
  group carrying a `TensorSubfieldProjection` encoding.
- *Bounded sources*: an A digit depth that is not
  `ceil(log_commit_bound / log_basis_inner)` for its level, in both directions, on
  a bounded and a full-width row; a commitment whose source coefficients exceed
  the declared bound. State the second against `decompose_centering_threshold` and
  `checked_balanced_digit_representable_bounds` — a naive range check wrongly
  rejects valid full-width dense commitments.
- *New structural rules*: an empty `groups` list; a root fold naming an incoming
  setup prefix; more than one prefix group, or a prefix at an index other than 0;
  `setup_natural_len` of zero, above `n_prefix`, or not a multiple of `d_setup()`;
  a standalone or precommitted profile with a non-canonical source encoding.

**Existing suites.** `cargo test -p akita-config --features all-schedules --test
generated_tables` and the full matrix in `.github/workflows/ci.yml`, including the
table drift gate. The matrix selects transcript backends explicitly and runs pure
Blake and Keccak semantic suites; the step-5 byte break must be validated under
each.

### Performance

No runtime cost is expected. The change removes indirection rather than adding
it: `&dyn LevelParamsLike` dispatch becomes direct field reads, and the
`!Sync` obstacle those trait objects created disappears.

Allocation count is unchanged. Sizes, measured on `8e552d2ac` (debug,
aarch64-darwin):

| Type | Today | After | Note |
|---|---|---|---|
| `FoldSchedule` | 1264 | ~440 | main structural win |
| `CommittedGroupParams` → `FoldParams` | 816 | ~160 | profile and 2 of 3 matrices move to the groups |
| `RecursiveFoldParams` + `RecursiveFoldStep` | 1296 + 1312 | — | merged; 1312 is the largest parameter value in the tree, held in a `Vec` |
| `RootFoldParams` + `RootFoldStep` | 960 + 976 | — | merged |
| `PrecommittedLevelParams` → `GroupOpenPhaseParams` | 352 | ~360 | plus `Option<usize>`; `ScheduledSetupPrefix` (368) deleted |
| Terminal: 3 types | 176 + 240 + 256 | ~264 | merged |
| `GroupCommitPhaseParams` | 288 | 288 | regrouped and renamed only |
| `SetupPrefixSlotId` | 304 | 304 | kept, but derived rather than stored |

`FoldSchedule` currently carries about 1.3 KiB per recursive level in
`RecursiveFoldStep` alone. The "After" column is estimated; the measurement gate
in Testing Strategy makes it a check rather than a claim.

Proof size, security, and planner objectives are unaffected. Old proof bytes stop
verifying at the epoch bump (see Design), which is a compatibility break, not a
performance change.

## Design

### Architecture

Four rules produce the whole schema:

1. **One shape, one type.** If two types hold the same fields, they are one type.
2. **One owner per field.** A field is stored where it is decided, and read from
   there, never copied.
3. **Shared data lives on the sharer.** The D matrix is shared by every group in
   a fold, so the fold owns it.
4. **Distinct trust boundaries keep distinct types.** Public wire data,
   executable fold state, and terminal parameters stay separate Rust types. No
   enum tag replaces a type.

Rule 4 is why the target has 10 types and not 1. Rules 1–3 are why it has 10 and
not 24.

**Affected crates.** `akita-types` (the type definitions, validators, and
encoders), `akita-schedules` (audits, generated schema, expansion),
`akita-planner` (the emitter), `akita-prover` and `akita-verifier` (the ~150 call
sites that read group parameters), `akita-config` (schedule selection constants).

**Where the duplication goes.** Every field with more than one owner today gets
exactly one:

| Field | Owners today | Single owner after |
|---|---|---|
| Shared D matrix | 3 | `FoldParams::open_matrix` |
| Fold challenge family | 4 | `groups[i].opening.fold_challenge_config` |
| Witness chunking | 3 | `FoldParams::witness_chunk` |
| Precommitted groups | 3 | `FoldParams::groups` |
| Frozen group descriptor | 2 | `groups[i].profile` |
| Setup prefix | 2 stored | `groups[0].setup_natural_len`, with `SetupPrefixSlotId` derived |
| Setup-prefix opening plan | 1 stored + re-derived | derived only |
| Source encoding | 1 field + 1 hard-coded trait arm | `FoldParams::source_encoding` |
| Certified terminal response cap | 2 functions that disagree | 1 function |

**On the setup prefix.** Two fields claim authority today, and the docs and the
dataflow disagree about which wins. Storing the prefix once as `groups[0]`
satisfies both: it is owned by the fold that *consumes* it, and it is literally
group 0, which is what the ~31 call sites reading `precommitted_group_iter`
require. Removing either field alone would have forced one side to change.

The consuming-fold rule and the "`SetupPrefixSlotId` is the canonical runtime
identity" rule were stated in
[`typed-schedule-topology-cutover.md`](archive/2026-Q3/typed-schedule-topology-cutover.md#L100),
which #414 archived without folding either rule into
`book/src/how/configuration.md`. Neither is documented anywhere live, so this
spec treats the code as the authority — the two `BTreeMap`s keyed on
`SetupPrefixSlotId`, and the field doc comments — and closes the documentation
gap as part of the work (see Documentation).

One ordering conflict must be settled during step 5b:
`precommitted_group_iter` puts the prefix **first**, while
`validate_nonterminal_opening_execution` builds its list with the witness first
and the prefix **last**. Today neither is canonical, so both can be right.
`FoldParams::groups` makes one ordering canonical; the execution check must
iterate `groups` instead of rebuilding a list, and the "prefix at index 0" rule
becomes the single answer.

**Removing `LevelParamsLike`.** The trait exists for one method,
`CommittedGroupParams::group_params`, which says "the final group is the fold
itself; any other group is a `PrecommittedLevelParams`; the caller must not care
which." After the change `groups[i]` is a concrete `GroupOpenPhaseParams`, so
`group_params(i)` becomes `groups.get(i)` and the trait object, the
`opening_batch` argument, and the index remapping all disappear. Its 22 methods
become field reads; `position_index_bits` and `block_index_bits` get one
definition on `BlockGeometry` instead of three copies.

The cost is that sites reading the final group gain one hop —
`params.num_positions_per_block` becomes
`fold.final_group()?.profile.blocks.positions_per_block`. Bind the group once per
function. Sites already indexed by group get shorter.

### Byte policy

**Preserved byte-for-byte:** `GroupCommitPhaseParams` and its wire form,
`LinfCommitMatrix<R>`, `BlockGeometry` (the `N, M, B` triple is already
contiguous in all three encoders), `RoleParams<M>` (the profile's
`basis, depth, matrix` grouping already matches it), and `GroupOpeningPlan`.

This constraint is worth keeping: commit-phase bytes feed the catalog
`key_digest` and are the catalog sort key, so freezing them means entry ordering
does not shift and wire fixtures keep passing.

**Broken once, deliberately:** `GroupOpenPhaseParams`, `FoldParams`,
`TerminalFoldParams`, `FoldSchedule`. Their storage no longer matches the old
layout, so their bytes cannot.

| Constant | Location | Today | After |
|---|---|---|---|
| `AKITA_INSTANCE_DESCRIPTOR_VERSION` | `instance_descriptor/mod.rs:37` | `1` | `2` |
| `SCHEDULE_ROW_DOMAIN_V2` | `akita-types/src/schedule_selection.rs:16` | `…/v2` | `…/v3`, renamed `_V3` |
| `FoldSchedule` descriptor byte | `schedule/descriptor.rs:38`, `:72` | `1` | `2` |
| `GroupCommitPhaseParams::VERSION` | `schedule/profiles.rs:113` | `4` | `4`, unchanged |
| `SETUP_PREFIX_CONTENT_TAG` | `proof/setup_prefix.rs:25` | `b"SPF4"` | unchanged |

`protocol_epoch` is `AKITA_INSTANCE_DESCRIPTOR_VERSION` and every generated table
embeds it, so all 13 tables regenerate together and **old proof bytes stop
verifying**. Note that #403 moved canonical-schedule validation ahead of
allocation, so a descriptor-order mistake now surfaces as a pre-allocation
rejection.

One encoding rule replaces 21 hand-written encoders:

> **The canonical byte order is the declared field order, top to bottom.** A
> containing type encodes each field by calling that field's encoder in
> declaration order.

This couples storage layout to encoding order on purpose. Reordering a field then
changes the golden fixtures and the committed tables, and both fail loudly. The
alternative is two orders kept in step by hand, which is what produced 21
divergent encoders — three of which order the same geometry three different ways.

### What the type system gains and loses

Twelve parallel-tag invariants stop being representable: two fields disagreeing
about the same value, a group claiming a D matrix it does not own, and a terminal
cap computed from another fold's challenge config. Two constraints become
validated that are unenforced today: at most one prefix group per fold, and a
recursive fold with more than one precommitted group.

**One guarantee is traded.** Today `RootFoldParams` has no prefix field, so a root
cannot name a prefix at the type level. Since root and recursive share
`FoldParams`, `validate_structure` must now reject it. This is a real if small
loss, and it buys the deletion of ten mirror audits, two split-brain pairs, and
one divergent security calculation.

Root and recursive folds still cannot be confused: `FoldSchedule` names the three
positions, so no role is inferred from an array index. The properties that
separate them — the root payload must be compressed, the root consumes no prefix,
the root A route must be `Linf` — are validated constraints, and three of the four
already are today.

### Alternatives Considered

**One `CommitParams` type with a `CommitKind` enum** (a rejected earlier draft).
It was recursively sized and could not compile, and `kind + Option` does not make
invalid states impossible — it relocates them to runtime. This design has no kind
enum, no optional role fields, and no borrowed views. Every distinct trust
boundary keeps its own type.

**Keep separate `RootFoldParams` and `RecursiveFoldParams`.** After the shared D
matrix moves to the fold, the two hold identical fields. A second type that only
recomposes the first is what `AGENTS.md` forbids. The distinguishing properties
are validated constraints either way.

**Keep `LevelParamsLike` and unify only the leaf components.** This is the
conservative option, and it leaves the root cause in place: the final group still
cannot share a type with the other groups, so the trait, the duplicate
constructors, and the mirror audits all stay. The leaf consolidation is worth
doing regardless — it is steps 2–4 — but it does not pay for itself without step 5.

**Move `source_encoding` onto the commit-phase type.** Arguably more honest, and
it would put the "a standalone commitment must be canonical" rejection at the
boundary where it belongs. It costs a version bump to 5, new bytes, and a new
`key_digest` for 13 catalogs, breaking the byte-preservation property the rest of
this plan leans on. Recorded as a follow-up.

**A three-role generic over the A, B, and D matrices.** No longer possible. The A
role left the shared macro when it gained `InnerCommitSecurityRoute`; its
`sis_table_key()` and `coeff_linf_bound()` return `Option` and its `validate`
branches on the route. Forcing it into the generic would reintroduce exactly the
`Option`-shaped tag this design rejects. What is shared instead is the audit code,
which already is.

## Documentation

- `book/src/how/configuration.md` owns the durable content and must be updated in
  the step-5 PR: the group/fold parameter model and the shared-D ownership rule.
- **Close a pre-existing gap in the same chapter.** #414 archived
  `typed-schedule-topology-cutover.md`, which was the only written statement of
  two rules this spec depends on: the consuming fold owns the setup-prefix edge,
  and `SetupPrefixSlotId` is the canonical runtime identity of a committed prefix.
  Neither was folded into the book, so both are currently undocumented outside the
  code. Step 5b changes the first rule's representation, so it is the right place
  to write both down.
- `book/src/usage/commitment-api.md` if any public signature changes shape.
- **Registry entries, required for a live spec after #414.** This spec must appear
  in `book/src/foundations/spec-index.md` and in the `live_specs` array of
  `scripts/check-spec-references.sh`, which are kept in sync. Without the second,
  CI does not scan this spec at all. Both are added by this spec's own PR.
  Note that `specs/` root is at the PRUNING.md steady-state target of 15 specs
  with this one included, so archiving on completion is not optional.
- This spec's `Status` moves to `active` at step 1 and `implemented` at step 6,
  then folds into the book chapter and archives per
  [`specs/PRUNING.md`](PRUNING.md). Update the status and the two registry entries
  in the same PR that changes it.
- `docs/doc-blast-radius.json` may need a region entry for the renamed types.
- The companion draft-1 review is archived at
  [`parameter-struct-consolidation-review.md`](archive/2026-Q3/parameter-struct-consolidation-review.md).
  All seven of its blocking findings are resolved by this design; two were
  resolved by the codebase itself before this revision.

## Execution

Steps 1–4 change no bytes and no call sites. Step 5 is the cutover.

| Step | Work | Bytes | Tables |
|---|---|---|---|
| 0 | **Prerequisite, own PR.** Resolve the certified-response divergence below. | none | none |
| 1 | Golden fixture harness, all 13 catalogs, all four newest paths, each transcript backend. | none | none |
| 2 | `BlockGeometry` and `GadgetDigits`, with `validate` and the index-bit methods. | identical | none |
| 3 | `LinfCommitMatrix<R>` with a sealed `LinfMatrixRole`; delete the 2-role macro; keep the `OuterCommitMatrixParams` / `OpenCommitMatrixParams` aliases. `InnerCommitMatrixParams` untouched. | identical | none |
| 4 | `RoleParams<M>`; rename `CommittedGroupProfile` to `GroupCommitPhaseParams` and restructure to 6 nested fields. Update the emitter. | identical | regenerate; `key_digest` unchanged |
| 5a | `GroupOpenPhaseParams` from `PrecommittedLevelParams`: add `setup_natural_len` and `slot_id()`, absorb `ScheduledSetupPrefix`, derive `SetupPrefixSlotId`. | break | regenerate |
| 5b | `FoldParams` with the uniform `groups` list; D matrix and `source_encoding` to the fold; merge the six root/recursive types; delete `WitnessPartition`; delete the 8 schedule-side mirror audits; settle the prefix ordering; add the 3 new checks. Bump the byte-policy constants. | break | regenerate |
| 5c | Delete `LevelParamsLike`, `group_params` and its geometry twin, the 10 single-method wrappers, `CommitInnerPlan::from_profile`, and all four borrowed views. | none | none |
| 5d | `TerminalFoldParams` merging 3 types; move the 6 fallible admission steps into `admit`; drop the config argument from the response-cap functions. | break | regenerate |
| 6 | Generated schema 15 → 8; drop the stored prefix `GroupOpeningPlan` and its 3 audits; one `expand_group`; delete 5 `emit_*` helpers. | none beyond step 5 | regenerate; `key_digest` moves once |

Steps 5a–5d must land together. Splitting them across PRs would need temporary
mirror fields with temporary audits, which is the thing being deleted. Split them
into four commits for review and keep the tree compiling at each.

### Step 0: the prerequisite

Two functions compute the certified terminal response cap and disagree.
`TerminalCommittedGroupParams::certified_response_linf_cap`
([schedule.rs:211](../crates/akita-types/src/schedule.rs#L211)) applies an
`i16::MAX` clamp and takes its challenge config from the caller.
`CommittedGroupParams::terminal_response_linf_limit_for_params`
([params.rs:405](../crates/akita-types/src/layout/params.rs#L405)) omits the clamp
and reads the **receiver fold's** config even though a per-group `params` is the
argument. Both gate proof acceptance against the same `z_admission_linf_cap`, and
the two live call sites confirm the mismatch is reachable: a multi-group terminal
fold computes each group's security cap from the fold's config.

This changes acceptance behaviour, so it must not hide inside a mechanical
refactor. Decide three things: whether the `i16::MAX` clamp is a kernel
representation limit both paths need (the comment at `schedule.rs:233-234` says
the terminal NTT kernels consume signed `i16`, which suggests yes); whether the
per-group challenge config is the correct input (`GroupOpeningPlan` already
carries one per group); and whether the explicit A-role check adds anything over
the typed field. Then keep one function.

After the consolidation the divergence cannot return: the group carries its own
config, `TerminalFoldParams` carries its own, and the surviving function takes no
config argument.

### Risks

- **The prefix ordering conflict** is the one place two correct-looking
  constructions must be reconciled. Verify that
  `validate_level_opening_execution`'s use of `groups.first()` as the family
  reference still selects the intended group after the reorder — today it reads
  the witness, and after the reorder it reads the prefix.
- **The emitter has no type-level link to the schema.** It is string templating,
  and `emit_profile_matrix` formats enums with `{:?}`, so renaming a variant
  silently changes emitted code. The only guard is the table drift gate. Every
  step that changes a struct must change the matching `emit_*` in the same commit
  and commit the regenerated tables.
- **`key_digest` must move exactly once.** Dropping the stored prefix opening plan
  changes `entries_key_digest`; sequence it with the step-5 epoch bump so the
  digest moves in step 6 only, and assert both values separately.
- **Terminal admission has six fallible steps** including a minimum-retention
  heuristic. Move `TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM/DEN` and their doc
  comment verbatim; the heuristic is easy to lose in a merge.
- **A relocated rejection.** `GroupCommitPhaseParams::try_from_params` currently
  refuses to build a profile with a non-canonical source encoding, which blocks a
  recursive fold whose witness uses `TensorSubfieldProjection`. Move that
  rejection to `validate_frozen_precommit`, the precommit admission boundary. This
  is a relocation, not a relaxation, and needs its own rejection test.

## References

- [`specs/subring-coefficient-packing.md`](subring-coefficient-packing.md) — the
  opening-method split this design carries per group (#394). Live spec.
- Archived design records, for provenance only — per
  [`specs/PRUNING.md`](PRUNING.md) these are not a source for new behavior:
  [`typed-schedule-topology-cutover.md`](archive/2026-Q3/typed-schedule-topology-cutover.md)
  (prefix identity and `SetupPrefixSlotId`; not yet folded into the book, see
  Documentation) and
  [`commitment-slicing.md`](archive/2026-Q3/commitment-slicing.md)
  (`outer_slice_count` and the single B-width authority, #388; folded into
  `book/src/how/commitment.md`).
- Bounded committed dense sources (#407) — `log_commit_bound` as a declarable
  input, the per-level A-depth audit, and the `fp128_dense_bounded` catalog.
- Transcript and verifier trust boundaries (#403) — pre-allocation schedule
  validation and explicit transcript backend selection.
- An exhaustive prior revision of this spec, with a per-check validation-path
  table, per-method call-site counts, and verified `file:line` references for
  every claim, is preserved in git history at `584ab96cb`. Retrieve it with
  `git show 584ab96cb:specs/parameter-struct-consolidation.md` when implementing
  a specific step.
- Regeneration: `scripts/generate-schedule-tables.sh`, then
  `git diff --exit-code -- crates/akita-schedules/src/generated`.
