# Spec: Parameter Struct Consolidation

> This document uses ASD-STE100 (Simplified Technical English).
> Sentences are short. The voice is active. Code names keep their code spelling.

| Field         | Value                                  |
|---------------|----------------------------------------|
| Author(s)     | —                                      |
| Created       | 2026-08-12                             |
| Status        | draft                                  |
| PR            | —                                      |
| Supersedes    | Draft 1 of this file (rejected)        |
| Superseded-by |                                        |
| Book-chapter  | book/src/how/configuration.md          |

## Summary

Akita stores the same commitment parameters in 18 types and one trait. Each copy
has its own validator, its own byte encoder, and an audit that compares it with
the other copies. Six of these audits compare a value with a copy of itself.

This document replaces those 18 types with **8**, and the `LevelParamsLike`
trait with nothing. `FoldSchedule` keeps its shape and is not counted on either
side. The reduction comes from one structural change:

> **A fold owns one ordered list of groups. Every group has the same type.**

Today a fold stores its final group as a set of flat fields, its precommitted
groups as `Vec<PrecommittedLevelParams>`, and its setup prefix as a third thing
inside `SetupPrefixSlotId`. All three are the same shape. `LevelParamsLike`
exists only to hide that fact behind 18 methods. One list removes the trait, the
two `CommitInnerPlan` constructors, the `CommitmentGeometry` view, six mirror
audits, and the recursive type cycle that stopped draft 1.

Draft 1 proposed one `CommitParams` type with a `CommitKind` enum. Draft 1 was
rejected, and correctly so. This draft introduces **no kind enum, no `Option`
role fields, and no borrowed view types**. Every distinct trust boundary keeps
its own Rust type. The reduction comes from deleting duplication, not from
merging boundaries.

## 1. How this draft answers the review

Draft 1 had seven blocking findings. This draft resolves all seven. Most are
resolved by structure, not by argument.

| # | Blocking finding | Resolution in this draft |
|---|---|---|
| 1 | `CommitParams` is recursively sized and cannot compile | The cycle ran through `CommittedGroupParams::setup_prefix` → `SetupPrefixSlotId` → `PrecommittedLevelParams`. Section 5 deletes `SetupPrefixSlotId`: a setup prefix becomes an ordinary entry in the fold's group list. No `Box`, no allocation. Section 6 proves finite size. |
| 2 | The setup-prefix source of truth is reversed | Neither field wins. Both are deleted and replaced by one field on the consuming fold's group list (§5.4). This satisfies the spec rule that the successor edge is canonical **and** the 20 call sites that need the prefix to be group 0. |
| 3 | `kind + Option` does not make invalid states impossible | Agreed, and dropped. There is no kind enum and no optional role field in this draft. Section 8 gives the full invalid-state analysis. |
| 4 | A unified `FoldStep` weakens the typed topology | The `FoldStage` enum and `Option<output_witness_len>` are dropped. `FoldSchedule` keeps three named fields with distinct types, so all four invalid states the review listed stay unrepresentable (§8). |
| 5 | Step 6 reintroduces the duplication it claims to remove | Root precommitted groups had two owners. Now they have one: `FoldParams::groups`. The equality audits at `audit.rs:368`, `:383`, `:384`, `:385` are deleted, not preserved. |
| 6 | `LevelParamsLike` is not eliminated | Draft 1 unified the wrong three types. The trait bridges `CommittedGroupParams` and `PrecommittedLevelParams`. §5.3 makes those one type, which is what actually removes the trait. §12 gives the consumer map. |
| 7 | Terminal projection validation is lost | §9 names a new home for each of the 7 checks in `try_from_expanded_group`, including the minimum-retention heuristic. Terminal parameters keep a distinct Rust type, so no ordinary fold can reach a terminal-only function. |

The review also asked for corrections and for 10 new sections. Section 15 lists
the factual corrections. The 10 required sections are §5 (schema), §7
(authority), §6 (sizing), §9 (validation), §10 (bytes), §11 (tables), §8
(invalid states), §13 (derives and size), §12 (`LevelParamsLike`), §14 (tests).

## 2. What the code holds today

The whole problem is in the first four rows below. Each holds the same six
fields: block geometry, the A role, and the B role.

| Concept | Type today | Block geometry | A role | B role | D digits | D matrix |
|---|---|---|---|---|---|---|
| Frozen, committed, on the wire | `CommittedGroupProfile` | yes | yes | yes | no | no |
| A precommitted group in a batch | `PrecommittedLevelParams` | through `layout` | through `layout` | through `layout` | yes | no |
| A setup prefix | `SetupPrefixSlotId` | through `commitment_params` | same | same | same | no |
| A fold's final group | `CommittedGroupParams` (flat fields) | yes | yes | yes | yes | **yes** |
| The last fold | `TerminalCommittedGroupParams` | yes | yes | no | no | no |

Read the D-matrix column. Only one row owns a D matrix, and that matrix is
shared across every group in the fold. The D matrix is a property of the
**fold**, not of a group. The code already says this in four places, for example
[precommitted.rs:249-253](../crates/akita-types/src/layout/params/precommitted.rs#L249-L253):

> Group metadata owns its A/B dimensions. The D role is batch-shared, so the
> caller supplies the consuming level's opening dimension.

Because `CommittedGroupParams` stores the shared D matrix next to its own
group-local fields, it cannot have the same type as the other groups. That is
the root cause of `LevelParamsLike`, of `role_dims()` versus
`role_dims(shared_dim)`, of `CommitInnerPlan::from_level` versus `from_profile`,
and of `CommitmentGeometry`. Move the D matrix up to the fold and the asymmetry
disappears.

### 2.1 Inventory

Columns are the ones the review asked for.

| Type | Owner or mirror | Surface | Validation boundary | `Copy` / const needed | Topology role | Fate |
|---|---|---|---|---|---|---|
| `CommittedGroupProfile` | owner | public wire + descriptor + generated static | `validate`, `validate_root_geometry`, wire deserialize | **both** | any group | kept, restructured (§5.2) |
| `CommittedGroupParams` | owner, plus 1 mirror field (`setup_prefix`) | descriptor + runtime | `validate`, expansion re-audit | neither | root or recursive fold | split into `FoldParams` + `GroupParams` |
| `PrecommittedLevelParams` | owner | descriptor + runtime | `admit`, `validate` | neither | any non-final group | becomes `GroupParams` (§5.3) |
| `SetupPrefixSlotId` | owner | descriptor + runtime | `validate_structure` | neither | prefix group | deleted, merged into `GroupParams` |
| `TerminalCommittedGroupParams` | owner | descriptor + generated | `try_from_expanded_group` | neither | terminal | merged into `TerminalFoldParams` |
| `RootFinalGroupParams` | pass-through | runtime | none | neither | root | deleted |
| `RootPrecommittedGroupParams` | mirror (`descriptor` = `commitment.layout`) | runtime | `audit.rs:384` | neither | root | deleted |
| `RootFoldParams` | 3 mirrors | descriptor + runtime | `audit.rs:367`, `:368` | neither | root | merged into `FoldParams` |
| `RecursiveFoldParams` | 4 mirrors | descriptor + runtime | `audit.rs:412`, `schedule.rs:361` | neither | recursive | merged into `FoldParams` |
| `TerminalFoldParams` | owner | descriptor + runtime | `audit_terminal` | neither | terminal | kept as the merged terminal type |
| `RootFoldStep` / `RecursiveFoldStep` / `TerminalFoldStep` | wrappers | descriptor + runtime | `validate_structure` | neither | one each | merged into their params types |
| `WitnessPartition` | lossy mirror of `ChunkedWitnessCfg` | descriptor + runtime | none | neither | any fold | deleted |
| `InnerCommitMatrixParams` / `Outer` / `Open` | owners | wire + descriptor + generated static | `try_new` SIS re-audit | **both** | per role | one generic + 3 aliases (§5.1) |
| `LevelParamsLike` (trait, 18 methods) | symptom | runtime | none | n/a | any group | deleted (§12) |
| `CommitmentGeometry<'a>` | borrowed mirror | runtime | `validate_commitment_geometry` | n/a | commit path | deleted |

Out of scope, unchanged: `DecompositionParams`, `SparseChallengeConfig`,
`PlannerPolicy`, `SisSecurityPolicyId`, `SisModulusProfileId`, `SisTableKey`,
`CommitmentRingDims`, `AkitaInstanceDescriptor`, `AkitaSetupDescriptor`,
`PolynomialGroupLayout`, `ChunkedWitnessCfg`, `TerminalResponseShape`, and the
SIS estimator configs. These hold policy or computed values, not duplicated
commitment geometry. `CommitmentRingDims` stays because it is a derived triple,
not stored state.

## 3. The five faults, with evidence

**Fault 1 — six audits compare a value with a copy of itself.**

| Copy | Audit that keeps it honest |
|---|---|
| `RootFoldParams::open_commit_matrix` | [audit.rs:367](../crates/akita-schedules/src/audit.rs#L367) |
| `RootFoldParams::precommitted_groups.len()` | [audit.rs:368](../crates/akita-schedules/src/audit.rs#L368) |
| `RootPrecommittedGroupParams::descriptor` | [audit.rs:383](../crates/akita-schedules/src/audit.rs#L383), [:384](../crates/akita-schedules/src/audit.rs#L384) |
| `RootPrecommittedGroupParams::commitment` | [audit.rs:385](../crates/akita-schedules/src/audit.rs#L385) |
| `RecursiveFoldParams::open_commit_matrix` | [audit.rs:412](../crates/akita-schedules/src/audit.rs#L412) |
| `CommittedGroupParams::setup_prefix` | [schedule.rs:361](../crates/akita-types/src/schedule.rs#L361) |

Three more mirror pairs have **no audit at all**: both
`sparse_challenge_config` fields and both `witness_partition` fields. Those are
latent split-brain bugs today.

**Fault 2 — the role lives in the name, not in the data.** One macro at
[ajtai_key.rs:438](../crates/akita-types/src/sis/ajtai_key.rs#L438) emits about
170 lines three times, once per role.

**Fault 3 — 11 hand-written byte encoders.** Each one re-lists its fields in a
fixed order. Three of them order the same six fields three different ways
(§10.1). Nothing links an encoder to its struct, so a new field is silently
unbound.

**Fault 4 — a trait hides fault 5.** `LevelParamsLike` has 18 methods and 2
implementations. Two of its methods are dead weight:
`num_live_ring_elements_per_claim` has **zero** call sites in the repository and
`b_col_len` has one.

**Fault 5 — the fold's final group is not a group.** See §2. This is the root
cause. It also produces a real defect:
`terminal_response_linf_limit_for_params` receives a per-group `params` but
reads the **root's** `fold_challenge_config`, and it omits the `i16::MAX` clamp
that its sibling `certified_response_linf_cap` applies. Two functions compute
"the certified terminal response cap" and disagree. `AGENTS.md` forbids exactly
this. §16 makes the fix a prerequisite.

## 4. The principle

Four rules produce the whole schema.

1. **One shape, one type.** If two types hold the same fields, they are one type.
2. **One owner per field.** A field is stored where it is decided. It is read
   from there, never copied.
3. **Shared data lives on the sharer.** The D matrix is shared by all groups in
   a fold, so the fold owns it.
4. **Distinct trust boundaries keep distinct types.** Public wire data,
   executable fold state, and terminal parameters stay separate Rust types. No
   enum tag replaces a type.

Rule 4 is why this draft has 8 types and not 1. Rules 1 to 3 are why it has 8
and not 20.

## 5. The final schema

This is the complete target. It has no migration aliases and no old names.

### 5.1 Leaf components

```rust
/// Exact block geometry of one commitment group.
///
/// Field names match the generated mirror, so the runtime and the static tables
/// use one vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockGeometry {
    /// Live source ring elements per claim (`N`).
    pub live_ring_elements_per_claim: usize,
    /// Positions per block (`M`), a power of two.
    pub positions_per_block: usize,
    /// Live blocks (`B = ceil(N / M)`).
    pub live_blocks: usize,
}

impl BlockGeometry {
    pub const fn new(/* … */) -> Self;
    /// `M` is a power of two and `B == ceil(N / M)`.
    pub fn validate(&self) -> Result<(), AkitaError>;
    pub fn position_index_bits(&self) -> usize;
    pub fn block_index_bits(&self) -> usize;
}

/// One gadget decomposition: a basis and an exact depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GadgetDigits {
    pub log_basis: u32,
    pub num_digits: usize,
}

impl GadgetDigits {
    pub const fn new(log_basis: u32, num_digits: usize) -> Self;
    /// A `SignedDigitKernel` exists for `log_basis`, and `num_digits` is in
    /// `(0, compute_num_digits_field_width(field_bits, log_basis)]`.
    pub fn validate(&self, field_bits: usize) -> Result<(), AkitaError>;
}

/// Sealed role marker. Downstream crates cannot invent roles.
pub trait MatrixRole: sealed::Sealed { const ROLE: SisMatrixRole; }
pub struct Inner; pub struct Outer; pub struct Open;

/// One audited Ajtai matrix identity. Replaces the three macro copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CommitMatrix<R: MatrixRole> {
    output_rank: usize,
    input_width: usize,
    sis_table_key: SisTableKey,
    _role: PhantomData<R>,
}

/// Kept permanently. These names document the protocol role at every signature
/// and at all 48 generated static-table call sites.
pub type InnerCommitMatrixParams = CommitMatrix<Inner>;
pub type OuterCommitMatrixParams = CommitMatrix<Outer>;
pub type OpenCommitMatrixParams  = CommitMatrix<Open>;

/// A gadget decomposition and the matrix that consumes it.
///
/// Used only for roles whose matrix is group-owned: A and B. The D matrix is
/// fold-owned, so a group carries `GadgetDigits` alone for D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoleParams<R: MatrixRole> {
    pub digits: GadgetDigits,
    pub matrix: CommitMatrix<R>,
}
```

`CommitMatrix::new_unchecked` stays `const`. The static tables need it.
`CommitMatrix::try_new` checks that `sis_table_key.role == R::ROLE` and
re-audits the rank against the checked-in SIS table, exactly as the macro does
today.

### 5.2 The frozen group — public wire data

```rust
/// Group metadata frozen when a standalone commitment group is created.
///
/// This is the public, versioned wire form. It holds only what the commit step
/// fixes: the group shape, the block geometry, and the two group-owned roles.
/// A group's D digits and fold challenge family are chosen by the fold that
/// opens it, so they are not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommittedGroupProfile {
    pub version: u8,
    pub group: PolynomialGroupLayout,
    pub blocks: BlockGeometry,
    pub inner: RoleParams<Inner>,
    pub outer: RoleParams<Outer>,
}
```

Same name, same wire bytes, same `Copy`, same 240 bytes as today. Only the
field grouping changes: 10 flat fields become 5 nested ones.

### 5.3 A group inside one fold's opening batch

```rust
/// One commitment group taking part in one fold's opening batch.
///
/// Every group in a fold has this type: the final/new group, each precommitted
/// group, and the setup prefix. The fold owns the shared D matrix; a group
/// owns only its contribution of D digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupParams {
    /// Frozen commit-time identity of this group.
    pub profile: CommittedGroupProfile,
    /// This group's fresh `e_hat` digits for the fold's shared D matrix.
    pub open: GadgetDigits,
    /// Exact folded-witness digit depth for this group.
    pub num_digits_fold: usize,
    /// Sparse fold-challenge family certified for this group's native A ring.
    pub fold_challenge_config: SparseChallengeConfig,
    /// Active setup-weight support, in flat field coefficients.
    ///
    /// `Some` exactly when this group is the consuming fold's setup prefix.
    /// This is the sole record of that fact and the sole record of the length.
    pub setup_natural_len: Option<usize>,
}
```

`GroupParams` is `Copy`. Sites pass it by value.

`setup_natural_len` is not a tag beside a payload. It **is** the payload. A
group is a setup prefix when it has a setup-weight support length, and that
length is the field. There is no second field to compare it against, so there
is no audit.

### 5.4 One fold level

```rust
/// Parameters for one fold level: the root fold or one recursive fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldParams {
    pub payload_mode: CommitmentPayloadMode,
    /// Groups in canonical transcript order. Never empty.
    ///
    /// The last entry is the final/new group. Earlier entries are precommitted.
    /// A setup prefix, when present, is `groups[0]`.
    pub groups: Vec<GroupParams>,
    /// The shared D matrix over every group's `w_hat` segment.
    pub open_matrix: CommitMatrix<Open>,
    pub witness_chunk: ChunkedWitnessCfg,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

impl FoldParams {
    /// The final/new group. `Err` when `groups` is empty.
    pub fn final_group(&self) -> Result<&GroupParams, AkitaError>;
    /// The incoming setup prefix, when this fold consumes one.
    pub fn setup_prefix(&self) -> Option<&GroupParams>;
    /// Presence of the incoming prefix is the canonical statement that the
    /// predecessor offloaded its setup contribution.
    pub fn predecessor_setup_contribution_mode(&self) -> SetupContributionMode;
    /// This group's A/B dimensions with this fold's shared D dimension.
    pub fn role_dims(&self, group: &GroupParams) -> CommitmentRingDims;
    /// Structural and per-group validation. Takes the same admission policy that
    /// `PrecommittedLevelParams::admit` takes today, so this stays in
    /// `akita-types`. The `PlannerPolicy` audits stay in `akita-schedules`.
    pub fn validate(
        &self,
        policy: PrecommittedGroupAdmissionPolicy,
    ) -> Result<(), AkitaError>;
}
```

`final_group()` returns `Result`, never panics. `validate` rejects an empty
`groups` at every boundary, so the `Err` arm is unreachable in practice. The
verifier no-panic contract forbids relying on that, so the signature stays
fallible.

Four kinds of mirror field vanish here. `open_commit_matrix` is stored once.
`sparse_challenge_config` is read from the group that uses it, which also fixes
fault 5. `witness_partition` is read as `witness_chunk.num_chunks`.
`precommitted_groups` and the flat final-group fields are one list.

### 5.5 The terminal fold

```rust
/// The last fold. It binds only the source decomposition through the inner
/// matrix. It has no outer or open matrix, no groups, and no chunking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalFoldParams {
    pub blocks: BlockGeometry,
    pub inner: RoleParams<Inner>,
    pub fold_challenge_config: SparseChallengeConfig,
    pub response_shape: TerminalResponseShape,
    pub input_witness_len: usize,
}

impl TerminalFoldParams {
    /// Project an ordinary fold's final group into terminal parameters and
    /// certify the directly checked response bound.
    ///
    /// Returns the admitted L-infinity capacity. Carries all 7 checks that
    /// `TerminalCommittedGroupParams::try_from_expanded_group` performs today.
    ///
    /// `group` supplies the block geometry, the inner role, and the challenge
    /// family, so none of the three is a separate argument.
    pub fn admit(
        group: &GroupParams,
        response_shape: TerminalResponseShape,
        input_witness_len: usize,
    ) -> Result<(Self, u128), AkitaError>;

    /// The one certified-capacity function. §16 removes today's second copy.
    pub fn certified_response_linf_cap(
        &self,
    ) -> Result<u128, AkitaError>;
}
```

`TerminalFoldParams` merges three types: `TerminalCommittedGroupParams`,
today's `TerminalFoldParams`, and `TerminalFoldStep`. It has no
`output_witness_len` field, so a terminal fold cannot claim a committed
successor.

### 5.6 The schedule

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldSchedule {
    pub root: FoldParams,
    pub recursive_folds: Vec<FoldParams>,
    pub terminal: TerminalFoldParams,
}
```

Three named fields with distinct types, exactly as today. Exactly one root and
exactly one terminal stay guaranteed by the field types. There is no stage enum
and no `Option<output_witness_len>`.

Root and recursive share `FoldParams` because after §5.4 they hold identical
fields. Their only differences are validated constraints, and they are validated
constraints **today** as well (§8). `AGENTS.md` forbids a second type that only
recomposes the first.

## 6. The types are finite-sized and statically constructible

**Finite size.** The draft-1 cycle was
`CommitParams → CommitKind::Level → SetupPrefixSlotId → PrecommittedLevelParams
→ CommitParams`, with no indirection at any step. `rustc` rejects it with
`E0072`. This draft has no such edge:

```text
FoldParams → Vec<GroupParams>            (heap, and not the cycle anyway)
FoldParams → CommitMatrix<Open>          → SisTableKey → scalars
GroupParams → CommittedGroupProfile      → RoleParams<R> → CommitMatrix<R> → scalars
GroupParams → GadgetDigits               → scalars
GroupParams → Option<usize>              → scalars
TerminalFoldParams → BlockGeometry, RoleParams<Inner>, TerminalResponseShape
```

`CommittedGroupProfile` is a leaf: every field is a scalar or a `Copy` leaf
struct. No type reachable from it names `GroupParams` or `FoldParams`. The graph
is a DAG, so every type is finite-sized without `Box`. **No allocation is added
anywhere.**

**Static constructibility.** The generated tables embed 24
`CommittedGroupProfile` struct literals and 48 `new_unchecked` calls in
`static` position, for example
[fp128_onehot.rs:100](../crates/akita-schedules/src/generated/fp128_onehot.rs#L100).
These types therefore must stay `Copy`, `'static`, and const-constructible.
This draft keeps that:

| Type | `Copy` | const-constructible | Needed in a `static`? |
|---|---|---|---|
| `BlockGeometry` | yes | public fields + `const fn new` | yes |
| `GadgetDigits` | yes | public fields + `const fn new` | yes |
| `CommitMatrix<R>` | yes | `const fn new_unchecked` (kept) | yes |
| `RoleParams<R>` | yes | public fields | yes |
| `CommittedGroupProfile` | yes | public fields | yes |
| `GroupParams` | yes | not required | no — expanded at run time |
| `FoldParams` | no (`Vec`) | not required | no |
| `TerminalFoldParams` | no (`Vec` in response shape) | not required | no |

`PhantomData<R>` is a zero-sized `const` field, so `RoleParams<R>` literals
compile in `static` position. The three type aliases keep the spelling that the
emitter already writes, so §11 changes the emitter for nesting only, not for
matrix syntax.

## 7. Authority table

Every field that has two or more owners today gets exactly one owner.

| Field | Owners today | Single owner after | Deleted mirror(s) |
|---|---|---|---|
| Shared D matrix | `CommittedGroupParams::open_commit_matrix`, `RootFoldParams::open_commit_matrix`, `RecursiveFoldParams::open_commit_matrix` | `FoldParams::open_matrix` | 2 fields, audits `:367`, `:412` |
| Fold challenge family | `CommittedGroupParams::fold_challenge_config`, `PrecommittedLevelParams::fold_challenge_config`, `RootFoldParams::sparse_challenge_config`, `RecursiveFoldParams::sparse_challenge_config` | `GroupParams::fold_challenge_config`, per group | 2 fields, no audit existed |
| Witness chunking | `CommittedGroupParams::witness_chunk`, `RootFoldParams::witness_partition`, `RecursiveFoldParams::witness_partition` | `FoldParams::witness_chunk` | 2 fields + `WitnessPartition` type, no audit existed |
| Precommitted groups | `CommittedGroupParams::precommitted_groups`, `RootFoldParams::precommitted_groups` | `FoldParams::groups` | 1 field, audits `:368`, `:385` |
| Frozen group descriptor | `RootPrecommittedGroupParams::descriptor`, `…::commitment.layout` | `GroupParams::profile` | 1 field, audits `:383`, `:384` |
| Setup prefix | `CommittedGroupParams::setup_prefix`, `RecursiveFoldParams::incoming_setup_prefix` | `FoldParams::groups[0].setup_natural_len` | 2 fields + `SetupPrefixSlotId` type, check `schedule.rs:361` |
| Final group geometry and roles | `CommittedGroupParams` flat fields, and a copy in `CommittedGroupProfile::from_params` | `FoldParams::groups.last()` | `RootFinalGroupParams`, `from_params` copy |
| Block index bit widths | duplicated formula in both `LevelParamsLike` impls | `BlockGeometry` methods | 1 duplicate formula |
| Opening basis dominance check | `PrecommittedLevelParams::admit:110` and `validate:198` | `GroupParams::validate` | 1 duplicate check |
| Certified terminal response cap | `certified_response_linf_cap` and `terminal_response_linf_limit_for_params`, which **disagree** | one function (§16) | 1 divergent copy |

### 7.1 On the setup prefix, specifically

The review and draft 1 disagreed about which of the two setup-prefix fields is
authoritative. Both were right about their own evidence, and both remedies were
worse than deleting the question.

- The documentation and [typed-schedule-topology-cutover.md:100](typed-schedule-topology-cutover.md#L100)
  say the consuming fold's edge is canonical.
- The dataflow says otherwise: every writer populates
  `CommittedGroupParams::setup_prefix` first and clones it into the edge
  ([runtime.rs:535](../crates/akita-schedules/src/runtime.rs#L535)). About 20
  sites read the group-params field and about 8 read the edge.
- The group-params field is load-bearing beyond identity. It makes the prefix
  **precommitted group index 0**, through `precommitted_group_iter`
  ([params.rs:326-333](../crates/akita-types/src/layout/params.rs#L326-L333)).
  Shared-D width, relation ordering, and verifier payload assembly all depend on
  that ordering.

One field satisfies both constraints. `FoldParams::groups[0]` is owned by the
fold that **consumes** the prefix, which is the successor, so the spec rule
holds. It is also literally group 0, so the 20 dataflow readers keep working
without an extra argument. Removing either existing field alone would have
forced the other side's consumers to change; removing both and storing the
prefix once forces neither.

## 8. Compile-time invalid-state analysis

| Invalid state | Representable today | Representable after | Note |
|---|---|---|---|
| A wire profile used as executable fold state | no | no | distinct types, no enum |
| An ordinary fold passed to a terminal-only function | no | no | `TerminalFoldParams` is a distinct type |
| Terminal fold with a B or D matrix | no | no | fields absent |
| Terminal fold with groups, chunking, or a prefix | no | no | fields absent |
| Terminal fold with an `output_witness_len` | no | no | field absent |
| Root or recursive fold without an `output_witness_len` | no | no | field is not `Option` |
| Terminal fold in `FoldSchedule::root` | no | no | field types differ |
| Root fold inside `recursive_folds` | no | no | both are `FoldParams`; arity is still typed, see below |
| Two fields disagreeing about the same value | **yes**, 6 audits plus 3 unaudited pairs | **no** | the main gain |
| A group claiming a D matrix it does not own | **yes** | **no** | D matrix only on `FoldParams` |
| A recursive fold with more than one precommitted group | **yes, unvalidated** | yes, **validated** | gain |
| A root fold naming an incoming setup prefix | no (field absent) | yes, **validated** | **the one trade** |

Two entries need comment.

**"Root fold inside `recursive_folds`" is still prevented.** Root and recursive
share `FoldParams`, so a value could be placed in either field. What the schema
guarantees is what it guarantees today: exactly one root, exactly one terminal,
and an ordered list between them. No role is inferred from an array index,
because `FoldSchedule` names the three positions. The properties that separate a
root from a recursive fold — the root payload must be compressed, the root
consumes no prefix, a recursive fold has at most one prefix group — are
validated constraints. Two of the three are validated constraints today as well.

**The one trade.** Today `RootFoldParams` has no prefix field, so a root cannot
name a prefix at the type level. After the change, `FoldSchedule::validate_structure`
must reject it. This is a real, if small, loss. It buys the deletion of 6
mirror audits, 3 unaudited split-brain pairs, one recursive type cycle, and one
divergent security calculation. §14 requires a rejection test for it.

Net: one type-level guarantee traded, two gained, and nine parallel-tag
invariants deleted.

## 9. Validation-path table

Every check that exists today has a named destination. No check is dropped.

| Check today | Location today | Destination |
|---|---|---|
| version equality; matrix `validate`; power-of-two A/B dims with `d_b \| d_a`; digit kernel exists; depth within field width; nonzero outer basis and depth; `field_bits` agreement; exact A width; exact B width | `CommittedGroupProfile::validate` | `CommittedGroupProfile::validate(field_bits)`, unchanged, plus geometry below |
| `N · d_a == 2^num_vars`; `M` power of two; `B == ceil(N / M)` | `validate_root_geometry` | `M`/`B` go to `BlockGeometry::validate`; `N · d_a == 2^num_vars` stays on `CommittedGroupProfile::validate` |
| conjunction of the two above | `validate_frozen_precommit` | deleted; `validate` is the single entry point |
| digit kernel and depth-within-field-width | `CommittedGroupProfile::validate` and `expand.rs`, twice | `GadgetDigits::validate(field_bits)`, one place |
| opening basis dominates the frozen outer basis | `PrecommittedLevelParams::admit:110` **and** `validate:198` | `GroupParams::validate`, once |
| `natural_len != 0`; `natural_len <= n_prefix`; `d_setup() != 0`; `n_prefix % d_setup() == 0` | `FoldSchedule::validate_structure:366-377` | `GroupParams::validate`, which owns the field |
| setup-prefix mirror agreement | `validate_structure:361` | **deleted** — there is no mirror |
| root payload is compressed | `validate_structure:287` | unchanged |
| payload-phase cutover policy per recursive fold | `validate_structure:302-314` | unchanged; reads `setup_prefix()` |
| witness-length chaining; stage-2 successor capacity | `validate_structure:317-354`, `:378-411` | unchanged |
| terminal lengths nonzero | `validate_structure:412-418` | unchanged |
| `groups` is non-empty | n/a (flat fields) | **new** in `FoldParams::validate` |
| root names no setup prefix | implied by the type | **new** in `validate_structure` |
| at most one prefix group per fold, at index 0 | not enforced | **new** in `FoldParams::validate` |
| fold-coefficient count fits `usize`; cap config admissible; challenge norms derivable; witness norms bounded; unconstrained target computable; certified capacity nonzero; minimum-retention heuristic `>= 1/2` of target | `TerminalCommittedGroupParams::try_from_expanded_group` (7 checks) | `TerminalFoldParams::admit`, all 7. `TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM/DEN` and their doc comment move verbatim |
| A-role check on the inner matrix | `terminal_response_linf_limit_for_params:399` | `TerminalFoldParams::certified_response_linf_cap` |
| `i16::MAX` representation clamp | `certified_response_linf_cap:226` only | same single function, after §16 resolves the divergence |
| wire: version; tag well-formedness; role identity; SIS rank re-audit; geometry; untrusted-coefficient allocation cap; payload consistency | `CommittedGroup<F>` deserialize, 7 checks | unchanged. The wire form is byte-identical and constructs `CommittedGroupProfile` only |
| mirror equality audits | `audit.rs:367`, `:368`, `:383`, `:384`, `:385`, `:412` | **deleted** — single owners |
| shared-D width; shared-D basis; terminal audit | `audit.rs:151-160`, `:257`, `audit_terminal:262` | kept; read `FoldParams::groups` |
| generated expansion re-audits rank against the SIS table | `expand.rs`, 3 parallel paths | one `expand_group`, same checks (§11) |

The review asked whether a wire boundary must prove it built the frozen case.
It does not have to prove anything: `CommittedGroupProfile` is the only type the
wire encoder accepts, and no `FoldParams` or `TerminalFoldParams` can appear
there. There is no case to reject because there is no enum.

## 10. Byte policy

### 10.1 Why byte preservation is impossible above the profile

The three encoders order the same fields three different ways today.

| Position | `CommittedGroupParams` | `CommittedGroupProfile` | `TerminalCommittedGroupParams` |
|---|---|---|---|
| 1 | payload mode tag | version | A basis |
| 2 | A basis | num_vars | **A matrix** |
| 3 | B basis | num_polynomials | N |
| 4 | D basis | N | M |
| 5 | **A matrix** | M | B |
| 6 | **B matrix** | B | A depth |
| 7 | **D matrix** | A basis | — |
| 8 | N | A depth | — |
| 9 | M | **A matrix** | — |
| 10 | B | B basis | — |
| 11 | sparse challenge | B depth | — |
| 12 | A depth | **B matrix** | — |
| 13-18 | B, D, fold depths; chunk; groups; prefix | — | — |

`CommittedGroupProfile` is already role-atomic: basis, depth, matrix, twice.
`TerminalCommittedGroupParams` splits the atom. `CommittedGroupParams` groups by
kind, not by role, so its roles interleave across the whole record. No single
role encoder can serve all three.

### 10.2 The policy: preserve the profile, break everything above it, once

**Preserved, byte for byte:**

- `CommitMatrix<R>` bytes. The encoder is already shared by all three roles.
- `BlockGeometry` bytes. The `N, M, B` triple is contiguous in all three
  encoders today, so an atomic geometry encoder is byte-neutral everywhere.
- `RoleParams<R>` bytes as `basis, depth, matrix`.
- **`CommittedGroupProfile` bytes and wire form.** Its declared field order in
  §5.2 reproduces today's order exactly. `version` stays `2`.

This is worth the constraint. Profile bytes are the only parameter bytes that
reach the catalog `key_digest`
([catalog_identity.rs:617](../crates/akita-schedules/src/catalog_identity.rs#L617)),
and they are the catalog sort key
([generated/mod.rs:301](../crates/akita-schedules/src/generated/mod.rs#L301)).
Keeping them fixed means entry ordering does not shift, committed `key_digest`
values stay meaningful, and `CommittedGroup<F>` wire fixtures keep passing.

**Changed once, deliberately:** `GroupParams`, `FoldParams`,
`TerminalFoldParams`, `FoldSchedule`. Their storage no longer matches the old
layout, so their bytes cannot. Required with the break:

| Constant | Today | After |
|---|---|---|
| `AKITA_INSTANCE_DESCRIPTOR_VERSION` | `1` | `2` |
| `SCHEDULE_ROW_DOMAIN_V2` | `b"akita/schedule-row/v2"` | `…/v3` |
| `FoldSchedule` leading descriptor byte | `1` | `2` |
| `CommittedGroupProfile::VERSION` | `2` | `2`, unchanged |
| `SETUP_PREFIX_CONTENT_TAG` | `b"SPF1"` | unchanged, now inside the group encoding |

`protocol_epoch` is `AKITA_INSTANCE_DESCRIPTOR_VERSION`, and every generated
table embeds it, so all 12 tables regenerate. Old proof bytes stop verifying.
`AGENTS.md` allows this; §14 requires the rejection tests.

### 10.3 One encoding rule replaces 11 encoders

> **The canonical byte order is the declared field order, top to bottom.**
> A containing type encodes each field by calling that field's encoder in
> declaration order.

This makes every encoder mechanical and reviewable, and it makes an unbound new
field impossible: adding a field to a struct adds it to the digest.

The review preferred to design storage layout and encoding order separately, so
that changing one cannot silently change the other. This draft couples them on
purpose, and removes the "silently": reordering a field changes the golden
fixtures of §14 and the committed tables, and both fail loudly in CI. The
alternative is two orders kept in step by hand, which is what produced 11
divergent encoders. The coupling is stated here so a reviewer knows that field
order is protocol-visible.

The resulting orders:

```text
GadgetDigits        := log_basis:u32, num_digits:u64
BlockGeometry       := N:u64, M:u64, B:u64
CommitMatrix<R>     := unchanged (8 items)
RoleParams<R>       := GadgetDigits, CommitMatrix<R>
CommittedGroupProfile := version:u8, num_vars:u64, num_polynomials:u64,
                         BlockGeometry, RoleParams<Inner>, RoleParams<Outer>
GroupParams         := CommittedGroupProfile, GadgetDigits(open),
                       num_digits_fold:u64, sparse_challenge,
                       has_prefix:u8, [b"SPF1", setup_natural_len:u64]
FoldParams          := payload_mode:u8, groups.len():u64, groups…,
                       CommitMatrix<Open>, ChunkedWitnessCfg,
                       input_witness_len:u64, output_witness_len:u64
TerminalFoldParams  := BlockGeometry, RoleParams<Inner>, sparse_challenge,
                       TerminalResponseShape, input_witness_len:u64
FoldSchedule        := 2u8, FoldParams(root), recursive.len():u64,
                       recursive…, TerminalFoldParams
```

`ChunkedWitnessCfg` is now encoded unconditionally. The conditional at
[params.rs:533](../crates/akita-types/src/layout/params.rs#L533) exists to keep
single-chunk descriptors byte-identical to a historical layout. That invariant
retires with the break, and removing the branch removes a way to collide two
different configurations.

## 11. Generated tables

### 11.1 Schema: 15 mirrors become 8

`GeneratedInnerCommitMatrix`, `GeneratedOuterCommitMatrix` and
`GeneratedOpenCommitMatrix` hold the same two or three fields. Merge them.
`GeneratedRootFinalGroup`, `GeneratedRootPrecommittedGroup` and
`GeneratedSetupPrefixInput` are the same three group cases. Merge them.
`GeneratedRootFold` and `GeneratedRecursiveFold` mirror types that are now one
type. Merge them. `GeneratedWitnessPartition` mirrors a lossy mirror. Delete it.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    /// Balanced slice count. `1` for the A role.
    pub slice_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedGroup {
    pub layout: akita_types::PolynomialGroupLayout,
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedMatrix,
    pub outer_commit_matrix: GeneratedMatrix,
    pub num_digits_fold: u32,
    /// Persisted frozen identity, for groups that are catalog lookup keys.
    /// `None` when the expansion recomputes it.
    pub frozen_profile: Option<akita_types::CommittedGroupProfile>,
    /// `Some` when this group is the consuming fold's setup prefix.
    pub setup_natural_len: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFold {
    pub payload_mode: akita_types::CommitmentPayloadMode,
    pub groups: &'static [GeneratedGroup],
    pub open_commit_matrix: GeneratedMatrix,
    pub witness_chunk: akita_types::ChunkedWitnessCfg,
}
```

`GeneratedBlockGeometry`, `GeneratedTerminalFold`,
`GeneratedFoldScheduleEntry`, `GeneratedScheduleCatalogIdentity` and
`GeneratedScheduleTable` stay. All 8 keep `Copy` and const construction.

`frozen_profile: Option<…>` is not a redundant tag. Precommitted groups persist
their exact frozen identity because it is the sorted lookup key and it feeds
`key_digest`. Final groups recompute theirs. The `Option` records which, and
there is nothing to compare it against.

The generated/runtime trust boundary is preserved. These types stay compact
planner inputs. Expansion still re-derives every rank with `secure_rank`
against the checked-in SIS table and still re-audits policy. They are not a
second authority.

### 11.2 `expand.rs`: three copies become one

`expand.rs` is 951 lines. About 640 of its 727 non-test lines are three
near-parallel copies of one 11-step algorithm:
`expand_to_precommitted_group` (164 lines),
`expand_to_level_params_with_setup` (271 lines), and
`expand_to_multi_group_root_level_params_with_setup` (205 lines). All three run
the same steps in the same order and differ only in the struct they assemble.

With one `GeneratedGroup` and one `GroupParams`, write one `expand_group`. The
three real differences become parameters:

- the opening ring dimension, which a prefix derives from its own outer
  dimension because its consumer's D matrix opens it
  ([expand.rs:83-89](../crates/akita-schedules/src/generated/expand.rs#L83-L89));
- the D bucket basis, `log_basis_open` for a scalar fold and
  `shared_d_digit_log_basis(...)` for a multi-group fold;
- `live_ring_elements_per_claim`, derived from the witness length or from
  `B · M`.

The terminal path (87 lines) stays separate. It has no B or D matrices and it
adds its own response-contract check.

### 11.3 Regeneration and identity

The emitter in [emit/mod.rs](../crates/akita-planner/src/emit/mod.rs) is string
templating with no type-level link to the schema. The only thing preventing
drift is one CI step:

```bash
scripts/generate-schedule-tables.sh
git diff --exit-code -- crates/akita-schedules/src/generated
```

Every step that changes a struct must change `emit_*` in the same commit and
commit the regenerated tables. `emit_profile_matrix` formats enums with `{:?}`,
so it depends on `Debug` output being valid Rust; renaming a variant silently
changes emitted code. Keeping the three matrix aliases (§5.1) means that
function needs no change at all.

Identity effects, precisely:

- `key_digest` reads only `CommittedGroupProfile` bytes, which §10.2 preserves.
  Regeneration changes literal shape, not the digest.
- `protocol_epoch` moves from `1` to `2` at the byte break, which invalidates
  all 12 committed tables at once. Regenerate them in that same step.
- Catalog sort order is unchanged, because it is keyed on profile bytes.

## 12. Removing `LevelParamsLike`

### 12.1 Why it exists, and why the fold list removes it

`LevelParamsLike` bridges `CommittedGroupParams` and `PrecommittedLevelParams`.
Its purpose is one method:

```rust
pub fn group_params<'a>(&'a self, opening_batch: &OpeningClaimsLayout, group_index: usize)
    -> Result<&'a dyn LevelParamsLike, AkitaError>
```

This says "the final group is the fold itself; any other group is a
`PrecommittedLevelParams`; the caller must not care which." That is the
erasure point, and about 30 call sites are downstream of it. Only 8 signatures
name the trait directly.

After §5.4, `groups[i]` is a concrete `GroupParams`. The trait object, the
`opening_batch` argument, and the index remapping all disappear.
`group_params(i)` becomes `groups.get(i)`.

### 12.2 Method destinations

| Trait method | Non-test call sites | Destination |
|---|---|---|
| `num_live_blocks` | ~28 | `group.profile.blocks.live_blocks` |
| `num_positions_per_block` | ~22 | `group.profile.blocks.positions_per_block` |
| `num_digits_open` | ~14 | `group.open.num_digits` |
| `num_digits_inner` | ~13 | `group.profile.inner.digits.num_digits` |
| `a_rows_len` | ~13 | `group.profile.inner.matrix.output_rank()` |
| `num_digits_outer` | ~10 | `group.profile.outer.digits.num_digits` |
| `log_basis_open` | ~10 | `group.open.log_basis` |
| `num_digits_fold` | ~8 | `group.num_digits_fold` |
| `log_basis_inner` / `log_basis_outer` | ~6 each | `group.profile.{inner,outer}.digits.log_basis` |
| `b_rows_len` | 6 | `group.profile.outer.matrix.output_rank()` |
| `inner_commit_matrix_params` | 5 | `&group.profile.inner.matrix` |
| `a_col_len` | 5 | `group.profile.inner.matrix.input_width()` |
| `position_index_bits` / `block_index_bits` | 4 each | `BlockGeometry` methods — one definition instead of two copies |
| `fold_challenge_config` | 3 | `group.fold_challenge_config` |
| `b_col_len` | 1 | `group.profile.outer.matrix.input_width()` |
| `num_live_ring_elements_per_claim` | **0** | delete the accessor; it has no caller anywhere. The field stays in `BlockGeometry`, where geometry validation reads it |

Also deleted with it:

- `CommittedGroupParams::group_params`, `validate_opening_batch`,
  `precommitted_group_count`, `precommitted_group_params`,
  `precommitted_group_iter`, `group_role_dims`. All become list operations.
- The 10 wrapper functions in
  [setup_contribution/plan/types.rs:140-238](../crates/akita-types/src/setup_contribution/plan/types.rs#L140-L238).
  Each re-exposes exactly one trait method; two pairs are byte-identical bodies.
- `CommitmentGeometry<'a>`: private, 9 fields, one constructor, two
  construction sites. Both consumers fabricate a fake `opening` dimension that
  `FoldParams::role_dims` supplies truthfully.
- `CommitInnerPlan::from_profile`. `from_level` and `from_profile` have
  identical bodies over different receivers. One `from_profile(&CommittedGroupProfile)`
  serves both, because all four of its fields live in the profile.
  `CommitInnerPlan` itself stays: it is a kernel shape plan on a public
  extension boundary, with out-of-tree implementations in
  [commitment_contract.rs](../crates/akita-pcs/tests/commitment_contract.rs).

### 12.3 The cost

Sites that read the final group get one more hop:
`params.num_positions_per_block` becomes
`fold.final_group()?.profile.blocks.positions_per_block`. This is the main cost
of the change and it touches roughly 100 lines. Bind the group once per
function and read fields from it. Many sites are already group-indexed and get
shorter, not longer.

## 13. Derives, size, and allocation

**`Copy`.** Every type that a `static` table needs stays `Copy`
(§6). `GroupParams` gains `Copy`, which today's `PrecommittedLevelParams` lacks
only because it derives `Clone` alone. `FoldParams` and `TerminalFoldParams` are
not `Copy` because they own a `Vec`, exactly as `CommittedGroupParams` and
today's `TerminalFoldParams` already are. **No type loses `Copy`.** The review
correctly warned that draft 1 would lose it on generated parents; this draft
does not, because the wire profile stays a `Copy` leaf.

**`Hash` and `Eq`.** `AkitaScheduleLookupKey` keeps
`Vec<CommittedGroupProfile>`, and `CommittedGroupProfile` remains all-scalar, so
its derived `Hash` still compiles. `GroupParams` cannot derive `Hash`, because
`SparseChallengeConfig` has none — and it does not need to, since it is not in
the lookup key. Draft 1's problem was that it put a `Vec<PrecommittedLevelParams>`
inside the hashed type. Separately: the `Hash` derives on
`AkitaScheduleLookupKey` and `CommittedGroupProfile` appear to be **dead**. No
map or set is keyed on them, and lookup uses `partition_point` ordering, not
hashing. Removing them is a small independent cleanup, not part of this plan.

`GroupParams` derives `PartialEq`. Today `PrecommittedLevelParams` hand-writes
it over all five fields, which a derive reproduces exactly. A hand-written
`PartialEq` that must be edited whenever a field is added is the same class of
fault as a hand-written byte encoder, so derive it.

**Size.** Measured today: `SisTableKey` 64, `CommitMatrix<R>` 80,
`CommittedGroupProfile` 240, `GeneratedRootPrecommittedGroup` 304,
`GeneratedFoldScheduleEntry` 224. Expected after:

| Type | Today | Expected | Why |
|---|---|---|---|
| `CommittedGroupProfile` | 240 | 240 | same fields, regrouped only |
| `GroupParams` | 288 (`PrecommittedLevelParams`) | ~296 | plus `Option<usize>`; `SetupPrefixSlotId` (296) is deleted |
| `FoldParams` | 400+ (`CommittedGroupParams`) plus a wrapper | ~144 | two of three matrices move to the groups; wrappers are gone |
| `GeneratedGroup` | 304 / 72 / 64, three types | ~320 | one type; adds `Option<u64>` |

No `Box` and no `Arc` appear anywhere, so the allocation count is unchanged.
`FoldParams` allocates one `Vec` where `CommittedGroupParams` already allocated
one. §14 makes the measured table a gate, since these are estimates.

**One allocation worth noting.** `precommitted_group_sort_key` builds a fresh
`Vec<u8>` per profile per comparison inside a comparator
([generated/mod.rs:301](../crates/akita-schedules/src/generated/mod.rs#L301)).
Tables hold at most 28 entries, so it is not hot today. This plan does not
change it. It is recorded here so it is not mistaken for new cost.

## 14. Tests

A single known-schedule byte check is not enough. Build the harness in step 1,
before any type changes, and re-run it after every step.

**Golden byte fixtures**, committed to the repository:

- Descriptor bytes for every entry of all 12 generated catalogs, at every level:
  profile, group, fold, schedule.
- `ScheduleRowDigest` for every catalog row.
- `key_digest` and the full `CatalogIdentityExpectation` per family.
- Coverage of single-group roots, multi-group roots, setup-prefix schedules,
  chunked and non-chunked schedules, recursive folds, and terminal folds.
- `CommittedGroup<F>` wire fixtures, serialized and deserialized, for each of
  the above.

**Byte-stability assertions.** Steps 1 to 4 must produce **zero** fixture diffs.
Step 5 changes the fixtures above the profile and must leave the profile
fixtures, `key_digest` values, and catalog sort order **unchanged**. Assert that
explicitly, per family.

**Rejection tests.** Every one must fail closed:

- profile version other than 2, in both directions;
- unknown modulus, policy, or role tag;
- a matrix role that does not match its slot;
- a rank, width, or L-infinity bound that the SIS table does not certify;
- geometry violations: `M` not a power of two, `B != ceil(N / M)`,
  `N · d_a != 2^num_vars`;
- an opening basis below the frozen outer basis;
- `setup_natural_len` of zero, above `n_prefix`, or not a multiple of
  `d_setup()`;
- **a root fold that names an incoming setup prefix** (the §8 trade);
- more than one prefix group in a fold, or a prefix group at an index other
  than 0;
- an empty `groups` list;
- a coefficient count above `MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS`;
- a terminal matrix that retains less than half of the unconstrained target;
- an instance descriptor with `protocol_epoch` of 1 after the bump.

**Measurement gate.** A committed test printing `size_of` for every type in §5
and every type in §11, before and after. Compare against §13. Also record
generated-table byte size and the `partition_point` lookup cost.

**Existing suites.** `cargo test -p akita-config --features all-schedules
--test generated_tables` and the full CI matrix in `.github/workflows/ci.yml`
must pass at every step, including the table drift gate.

## 15. Corrections to draft 1 and to the review

Draft 1:

- It listed 11 generated mirrors and named 12. The real count is **15**.
- It said `CommitmentGeometry` gives one shape to `CommittedGroupParams` and to
  a profile. It converts from `CommittedGroupParams` only.
  `CommitInnerPlan` has the two constructors.
- It said its three principal types differ only in role count. They differ in
  trust boundary: public versioned wire data, executable fold state, and the
  output of a security-sensitive projection. That is why this draft keeps them
  as distinct types.
- It cited `audit.rs:263` in its list of case-specific signatures. That line is
  the parameter list of `audit_terminal`, not a mirror audit. The mirror audits
  are at `:367`, `:368`, `:383`, `:384`, `:385` and `:412`.
- It claimed `Vec::new()` being `const` made static construction safe. It does
  not. Generated tables write struct literals with public fields, and cannot
  call a non-const constructor or set a private field. §6 keeps that working.
- It claimed `Hash` needs no change. `PrecommittedLevelParams` has no `Hash`
  impl, so draft 1's `Level` variant could not have derived one.
- It said `CommittedGroupParams::setup_prefix` should read from the group and
  that the schedule edge should go. The field's own doc comment says the
  opposite. §7.1 removes both.

The review:

- It counted the generated mirrors as twelve. There are 15.
- It said static data cannot hold a `Vec`. It can. The real constraints are
  const construction with public fields, `Copy`, and compactness.
- It is right that `Vec<T>: Hash` needs `T: Hash`, and right that the `Copy` and
  const-construction impact is wider than draft 1 stated. §6 and §13 address
  both with measured facts.
- Its recommendation to keep `CommittedGroupProfile`, executable level
  parameters, and terminal parameters as distinct semantic wrappers is adopted.
  Its `StandaloneCommitCore` sketch is adopted in the form of
  `CommittedGroupProfile` (§5.2).
- Its recommendation to keep typed root, recursive and terminal schedule steps
  is adopted at the schedule level (§5.6) and declined for the step wrapper
  types, which after §5.4 hold identical fields. §8 accounts for the exact
  guarantee that trades.

## 16. Prerequisite: resolve the certified-response divergence

Two functions compute the certified terminal response cap and disagree:

| | `certified_response_linf_cap` | `terminal_response_linf_limit_for_params` |
|---|---|---|
| Location | `schedule.rs:202` | `params.rs:389` |
| `i16::MAX` clamp | applied | **not applied** |
| A-role check | absent | present |
| Challenge config used | the group's | **the root's**, though a per-group value is available |

Both results gate proof acceptance, and both are compared against the same
`z_admission_linf_cap`. This is the split-brain that `AGENTS.md` forbids.

**Do this first, in its own PR.** It changes acceptance behavior, so it must not
hide inside a mechanical refactor. Decide whether the `i16::MAX` clamp is a
kernel representation limit that both paths need, and whether the per-group
challenge config is the correct input. Then keep one function. After the
consolidation, `GroupParams` carries the per-group config and the terminal
projection takes a `&GroupParams`, so the divergence becomes impossible to
re-introduce.

## 17. Order of work

Steps 1 to 4 change no bytes and no call sites. Step 5 is the cutover.

| Step | Work | Bytes | Tables |
|---|---|---|---|
| 0 | §16 prerequisite: one certified-response function | none | none |
| 1 | Golden fixture harness (§14) | none | none |
| 2 | `BlockGeometry` and `GadgetDigits`, with `validate` and the index-bit methods. Use the atomic geometry encoder where the triple is already contiguous; keep field-level digit encoding in the two encoders that interleave. | **identical** | none |
| 3 | `CommitMatrix<R>` with a sealed `MatrixRole`; delete the 3-role macro; keep the three aliases. | **identical** | none |
| 4 | `RoleParams<R>`; restructure `CommittedGroupProfile` to `version, group, blocks, inner, outer`. Update the emitter for nesting. | **identical**, verified by step 1 | regenerate; `key_digest` unchanged |
| 5a | `GroupParams` from `PrecommittedLevelParams`; absorb `SetupPrefixSlotId`, including its `Ord`, `Hash`, and serialize impls. | break | regenerate |
| 5b | `FoldParams` with the uniform `groups` list; D matrix to the fold; merge `RootFoldParams`, `RecursiveFoldParams`, `RootFinalGroupParams`, `RootPrecommittedGroupParams`, `RootFoldStep`, `RecursiveFoldStep`; delete `WitnessPartition`; delete the 6 mirror audits and the `validate_structure` mirror check; add the 3 new checks. Bump the §10.2 constants here. | break | regenerate |
| 5c | Delete `LevelParamsLike`, `group_params`, the 10 wrappers, `CommitmentGeometry`, `CommitInnerPlan::from_profile`. | none | none |
| 5d | `TerminalFoldParams` merging 3 types; move the 7 checks into `admit`. | break | regenerate |
| 6 | Generated schema 15 → 8; one `expand_group`; update the emitter. | none beyond step 5 | regenerate |

Steps 5a to 5d land together. Splitting them across PRs would need temporary
mirror fields with temporary audits, which is the thing being deleted. Split
them into four commits for review, and keep the tree compiling at each commit.

## 18. Result

| Item | Today | After |
|---|---|---|
| Parameter types in scope | 18 | **8** |
| Traits over parameter types | 1 (`LevelParamsLike`, 18 methods) | **0** |
| Borrowed view types | 1 (`CommitmentGeometry`, 9 fields) | 0 |
| Matrix implementations | 3 macro copies, ~510 lines | 1 generic + 3 aliases |
| Types holding block geometry | 6 | 1 |
| Types holding the shared D matrix | 3 | 1 |
| Types holding the fold challenge family | 4 | 1, per group |
| Types holding the setup prefix | 3 | 1 field |
| Mirror equality audits | 6 | **0** |
| Mirror pairs with no audit | 3 | 0 |
| Byte encoders | 11 hand-written | 1 rule, mechanical |
| Group-parameter accessors on the fold | 6 plus a trait object | list operations |
| Generated mirror types | 15 | **8** |
| `expand.rs` duplicated paths | 3 copies, ~640 lines | 1 |
| Recursive type cycles | n/a (draft 1 had one) | 0, no `Box` |
| Kind enums or `Option` role tags | n/a (draft 1 had 3) | **0** |
| Types that lose `Copy` | n/a | **0** |
| Divergent security calculations | 1 | 0 (§16) |
