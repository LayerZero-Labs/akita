# Spec: Parameter Struct Consolidation

> This document uses ASD-STE100 (Simplified Technical English).
> Sentences are short. The voice is active. Code names keep their code spelling.

| Field         | Value                                                      |
|---------------|------------------------------------------------------------|
| Author(s)     | —                                                          |
| Created       | 2026-08-12                                                 |
| Revised       | 2026-08-19 (draft 4: rebased onto `main` after #394 landed) |
| Status        | draft                                                      |
| PR            | —                                                          |
| Supersedes    | Draft 1 (rejected), draft 2 (`main` @ `74c17ba4f`), draft 3 (`f37d07089`) |
| Superseded-by |                                                            |
| Book-chapter  | book/src/how/configuration.md                              |
| Base commit   | `8e552d2ac` (`main`)                                       |

## Summary

Akita stores the same commitment parameters in **24** types and one trait. Each
copy has its own validator, its own byte encoder, and an audit that compares it
with the other copies. **Ten** of these audits compare a value with a copy of
itself.

This document replaces those 24 types with **10**, and the `LevelParamsLike`
trait with nothing. `FoldSchedule` keeps its shape and is not counted on either
side. The reduction comes from one structural change:

> **A fold owns one ordered list of groups. Every group has the same type.**

Today a fold stores its final group as a set of flat fields, its precommitted
groups as `Vec<PrecommittedLevelParams>`, and its setup prefix as a third thing
inside `ScheduledSetupPrefix`. All three are the same shape. `LevelParamsLike`
exists only to hide that fact behind 22 methods. One list removes the trait, the
two `CommitInnerPlan` constructors, the `CommitmentGeometry` view, ten mirror
audits, and the recursive type cycle that stopped draft 1.

Draft 1 proposed one `CommitParams` type with a `CommitKind` enum. Draft 1 was
rejected, and correctly so. This draft introduces **no kind enum, no `Option`
role fields, and no borrowed view types**. Every distinct trust boundary keeps
its own Rust type. The reduction comes from deleting duplication, not from
merging boundaries.

## 0. What changed in the code since drafts 2 and 3

Draft 2 was written against `main` at `74c17ba4f`. Draft 3 was written against
`f37d07089`, a tip of `quang/subring-coefficient-carrier-spec`. The base is now
`main` at `8e552d2ac`, which contains the same work plus two more merged PRs.
§0.1 to §0.6 describe what the packing line changed and still hold. §0.7 to §0.11
describe what changed after draft 3 and are new. **Read this whole section before
§5.** Every `file:line` reference in the rest of the document is restated against
`8e552d2ac`.

Selective L2 folds (#369), tiered commitments as dyadic B slicing (#388), and the
subring-coefficient-packing line are the changes §0.1 to §0.6 cover. Six of them
bear directly on this plan.

**0.1 The profile/policy split already landed, and it validates the thesis.**
`PrecommittedLevelParams` is no longer five flat fields. It is now
[precommitted.rs:165](../crates/akita-types/src/layout/params/precommitted.rs#L165):

```rust
pub struct PrecommittedLevelParams {
    pub layout: CommittedGroupProfile,   // frozen commitment identity
    pub opening: GroupOpeningPlan,       // policy owned by the consuming fold
}
```

`GroupOpeningPlan` ([precommitted.rs:112](../crates/akita-types/src/layout/params/precommitted.rs#L112))
holds `opening_method`, `fold_challenge_config`, `log_basis_open`,
`num_digits_open`, and `num_digits_fold`. That is exactly draft 2's `GroupParams`
minus `setup_natural_len`. Commit `f8c06ad1d` ("separate opening policy from
commitment identity") did the hard half of §5.3 already. §5.3 shrinks to a
rename plus one field, and the "shared data lives on the sharer" rule (§4) is now
an established pattern in the tree rather than a proposal.

**0.2 The A role diverged from B and D. `CommitMatrix<R>` over three roles is dead.**
`InnerCommitMatrixParams` is now hand-written
([ajtai_key.rs:430](../crates/akita-types/src/sis/ajtai_key.rs#L430)) and carries
`InnerCommitSecurityRoute` ([physical_l2.rs:155](../crates/akita-types/src/sis/physical_l2.rs#L155)),
which is `Linf(SisTableKey)` or `L2 { table_key: SisL2TableKey, response_l2_sq_cap, norm_proof_shape }`.
`sis_table_key()` and `coeff_linf_bound()` now return `Option`. The 3-role macro
at [ajtai_key.rs:750](../crates/akita-types/src/sis/ajtai_key.rs#L750) is a
2-role macro. §5.1 is rewritten: two roles merge, the A role keeps its own type.

**0.3 B slicing added a fifth geometry field to commitment identity.**
`outer_slice_count: CommitmentSliceCount` is on both `CommittedGroupProfile` and
`CommittedGroupParams`, and `CommitmentSliceGeometry`
([commitment_slicing.rs:178](../crates/akita-types/src/commitment_slicing.rs#L178))
is the single B-width authority. Both are already single-owner, so they need no
consolidation. They do change the profile's field list, its bytes, and its size.

**0.4 The D segment width is now opening-method dependent.**
`opening_d_segment_width` ([precommitted.rs:79](../crates/akita-types/src/layout/params/precommitted.rs#L79))
branches on `OpeningMethod`, and packing routes through
`SubringCoefficientPackingGeometry`. The D **matrix** is still fold-owned and the
D **digits** are still group-owned, so §2's root-cause analysis is unchanged. The
new fact is that a group's D contribution now needs its `opening_method` to be
sized, which is one more reason the opening plan belongs with the group.

**0.5 The setup-prefix type already split, and the draft-1 cycle is gone.**
`SetupPrefixSlotId` ([setup_prefix.rs:37](../crates/akita-types/src/proof/setup_prefix.rs#L37))
is now `{ natural_len, commitment_profile: CommittedGroupProfile }` — a registry
key with hand-written `Ord`, `Hash`, `Valid`, and serialize impls, used by
`SetupPrefixProverRegistry` and `SetupPrefixVerifierRegistry`. The schedule-side
type is `ScheduledSetupPrefix` ([setup_prefix.rs:46](../crates/akita-types/src/proof/setup_prefix.rs#L46)):
`{ natural_len, commitment_params: PrecommittedLevelParams }`.

Two consequences. First, **draft 2 named the wrong type for deletion.**
`SetupPrefixSlotId` must stay: [typed-schedule-topology-cutover.md:105](typed-schedule-topology-cutover.md#L105)
names it the canonical runtime identity of a committed prefix, and it keys two
`BTreeMap`s. `ScheduledSetupPrefix` is the type that absorbs into `GroupParams`,
and `SetupPrefixSlotId` is then **derived** from a group by
`GroupParams::slot_id()`. Second, the draft-1 `E0072` cycle is already broken by
this split, so §6's cycle proof is now history rather than a live constraint.
§6 keeps only the part that still binds: static constructibility.

**0.6 New validation and new borrowed views arrived.**
`FoldSchedule::validate_nonterminal_opening_execution`
([schedule.rs:518](../crates/akita-types/src/schedule.rs#L518)) is a second
structural boundary that admits opening methods per level. It works through
`OpeningExecutionGroup<'a>` ([schedule.rs:586](../crates/akita-types/src/schedule.rs#L586)),
and two more borrowed views appeared for descriptor encoding:
`FoldScheduleDescriptorStep<'a>` ([schedule.rs:327](../crates/akita-types/src/schedule.rs#L327))
and `TerminalFoldDescriptor<'a>` ([schedule.rs:336](../crates/akita-types/src/schedule.rs#L336)).
So the borrowed-view count went from 1 to 4 while this spec sat unmerged. §12.4
deals with all four.

**0.7 The packing line landed as one squash commit, so draft 3's base is gone.**
Draft 3's base `f37d07089` was a branch tip carrying 53 commits of the packing
line. `main` absorbed all of it as `8e552d2ac` ("feat(protocol)!: add subring
coefficient packing (#394)"), and the squashed content is **not** identical to the
branch tip: 251 files differ between `f37d07089` and `main`. Nothing in this
document may be checked against `f37d07089` any more. Every claim below is stated
against `main`.

**0.8 Descriptor encoding moved out of `schedule.rs` into its own module.**
`crates/akita-types/src/schedule/descriptor.rs` is new and holds 228 lines: both
`FoldSchedule::append_descriptor_bytes` variants, `append_root_fold_descriptor_bytes`,
`append_recursive_fold_descriptor_bytes`, the `TerminalFoldStep` and
`TerminalFoldDescriptor` encoders, and the `WitnessPartitionDescriptor` helper
pair. `schedule.rs` lost 224 lines and now holds the type declarations and the
validators only.

This is a location change, not a behaviour change: the byte order is identical.
It affects three things in this plan. The `FoldSchedule` leading descriptor byte
that §10.2 bumps is now written at
[descriptor.rs:38](../crates/akita-types/src/schedule/descriptor.rs#L38) and
[:72](../crates/akita-types/src/schedule/descriptor.rs#L72), not in `schedule.rs`.
The `TerminalFoldDescriptor` tag byte `3` is at
[descriptor.rs:196](../crates/akita-types/src/schedule/descriptor.rs#L196). And
step 5b's edits to the fold encoders are confined to one new file, which makes the
byte break easier to review than draft 3 assumed. The struct **declarations** for
the two descriptor views stay in `schedule.rs` at `:327` and `:336`, so §12.4's
deletion targets are unchanged.

**0.9 Bounded committed dense sources (#407) pinned the A digit depth per level.**
`DecompositionParams::log_commit_bound` is now a declarable schedule-generation
input rather than a value that presets only ever set to `1` or the field width.
The bound enters the protocol at exactly one place: the A-role digit depth
`ceil(log_commit_bound / log_basis_inner)`.

Two consequences for this plan.

First, `audit_committed_params` gained a check that the stored `num_digits_inner`
equals the depth the declared committed-source bound demands at this level's
selected A basis, with the root distinguished from recursive levels
([audit.rs:216-230](../crates/akita-schedules/src/audit.rs#L216-L230)). §9 carries
it as a new row. It is **not** a mirror audit in the sense of §3 fault 1, and it
must not be deleted: `num_digits_inner` has one owner today, and the check pins
that owner against a policy-derived formula. It is also **level-aware**, so
`GadgetDigits::validate(field_bits)` (§5.1) cannot subsume it — see the note in
§5.1.

Second, there is a **thirteenth** generated catalog family, `fp128_dense_bounded`,
behind the `fp128-dense-bounded` feature
([generated/mod.rs:348](../crates/akita-schedules/src/generated/mod.rs#L348)).
Draft 3 said "12 generated tables" throughout. The number is 13 everywhere,
including §6's static-constructibility argument, §11.3's regeneration step, and
§14's fixture coverage.

**0.10 Transcript and verifier hardening (#403) constrains the byte break.**
#403 validates canonical schedules and proof shapes **before allocation**,
enforces exactly one transcript backend with v2 domain separation, and rejects
trailing top-level bytes. None of it changes a parameter type. It changes the
cost of getting §10.2 wrong: a descriptor-order mistake now surfaces as a
pre-allocation shape rejection rather than a late mismatch, which is better, and
§14's rejection tests must be run under each transcript backend the CI matrix
selects explicitly, not only the default one.

**0.11 Two new items on the params surface, both out of scope.**
`OpeningFamily<Trace, Packing>`
([precommitted.rs:32](../crates/akita-types/src/layout/params/precommitted.rs#L32))
is a two-variant generic carrier that preserves the `OpeningMethod` distinction
while each protocol stage supplies its own payloads. It holds no commitment
geometry and has no mirror, so it is not consolidated; §2.1 lists it as out of
scope so the inventory stays honest.
`OpeningMethod::physical_coefficient_width`
([precommitted.rs:53](../crates/akita-types/src/layout/params/precommitted.rs#L53))
factored the width computation out of `opening_d_segment_width`. It is the sizing
authority §0.4 describes, now reachable directly.

Separately, `SetupProjectionGroupGeometry` **lost** its `d_active_cols` field; the
value is computed at the use site instead. The `d_active_cols` composite on
`plan/types.rs` is untouched, so §12.3's "the two composites stay" still holds.

## 1. How this draft answers the original review

Draft 1 had seven blocking findings. This draft resolves all seven. Most are
resolved by structure, not by argument.

| # | Blocking finding | Resolution in this draft |
|---|---|---|
| 1 | `CommitParams` is recursively sized and cannot compile | The cycle ran through `CommittedGroupParams::setup_prefix` → the prefix type → `PrecommittedLevelParams` → `CommittedGroupParams`. The tree already broke it (§0.5): `SetupPrefixSlotId` now holds a `CommittedGroupProfile`, which is a leaf. This draft keeps the graph a DAG. No `Box`, no allocation. §6 proves finite size. |
| 2 | The setup-prefix source of truth is reversed | Neither field wins. Both are deleted and replaced by one field on the consuming fold's group list (§5.4). This satisfies the spec rule that the successor edge is canonical **and** the ~31 call sites that need the prefix to be group 0. |
| 3 | `kind + Option` does not make invalid states impossible | Agreed, and dropped. There is no kind enum and no optional role field in this draft. Section 8 gives the full invalid-state analysis. |
| 4 | A unified `FoldStep` weakens the typed topology | The `FoldStage` enum and `Option<output_witness_len>` are dropped. `FoldSchedule` keeps three named fields with distinct types, so all four invalid states the review listed stay unrepresentable (§8). |
| 5 | Step 6 reintroduces the duplication it claims to remove | Root precommitted groups had two owners; they now have three (§3, fault 1). After this change they have one: `FoldParams::groups`. The equality audits at `audit.rs:502`, `:504`, `:519`, `:520`, `:521` are deleted, not preserved. |
| 6 | `LevelParamsLike` is not eliminated | Draft 1 unified the wrong three types. The trait bridges `CommittedGroupParams` and `PrecommittedLevelParams`. §5.3 makes those one type, which is what actually removes the trait. §12 gives the consumer map. |
| 7 | Terminal projection validation is lost | §9 names a new home for every check in `try_from_expanded_group`, including the minimum-retention heuristic and the new L2-route rejection. Terminal parameters keep a distinct Rust type, so no ordinary fold can reach a terminal-only function. |

## 2. What the code holds today

The whole problem is in the first four rows below. Each holds the same block
geometry, the A role, and the B role.

| Concept | Type today | Block geometry | Slice count | A role | B role | D digits | D matrix |
|---|---|---|---|---|---|---|---|
| Frozen, committed, on the wire | `CommittedGroupProfile` | yes | yes | yes | yes | no | no |
| A precommitted group in a batch | `PrecommittedLevelParams` | through `layout` | through `layout` | through `layout` | through `layout` | through `opening` | no |
| A setup prefix | `ScheduledSetupPrefix` | through `commitment_params` | same | same | same | same | no |
| A fold's final group | `CommittedGroupParams` (flat fields) | yes | yes | yes | yes | yes | **yes** |
| The last fold | `TerminalCommittedGroupParams` | yes | no | yes | no | no | no |

Read the D-matrix column. Only one row owns a D matrix, and that matrix is
shared across every group in the fold. The D matrix is a property of the
**fold**, not of a group. The code already says this at
[precommitted.rs:396-397](../crates/akita-types/src/layout/params/precommitted.rs#L396-L397):

> Group metadata owns its A/B dimensions. The D role is batch-shared, so the
> caller supplies the consuming level's opening dimension.

Because `CommittedGroupParams` stores the shared D matrix next to its own
group-local fields, it cannot have the same type as the other groups. That is
the root cause of `LevelParamsLike`, of `role_dims()` versus
`role_dims(shared_dim)`, of `CommitInnerPlan::from_level` versus `from_profile`,
and of `CommitmentGeometry`. Move the D matrix up to the fold and the asymmetry
disappears.

Row 2 shows how far the tree already came. A precommitted group reaches all its
frozen data through one `layout` field and all its consumer-owned policy through
one `opening` field. §5.3 gives the final group the same two fields.

### 2.1 Inventory

The table has 24 rows, which is where the headline "24 types" comes from. Two
rows name more than one type — the three fold-step wrappers, and the two
L-infinity matrix aliases — and two rows are not consolidated types at all: the
`LevelParamsLike` trait, and `CommittedGroupBatchProfile`, which is out of scope.
So the row count and the type count are not the same number. **The count this
document commits to is the row count**: 24 inventory entries become the 10 named
in §5, with `FoldSchedule` excluded on both sides. Draft 3 wrote "24 types and one
trait", which double-counts the trait row; the arithmetic is otherwise the same.

| Type | Location | Owner or mirror | Surface | Validation boundary | `Copy` / const needed | Fate |
|---|---|---|---|---|---|---|
| `CommittedGroupProfile` | `schedule/profiles.rs:84` | owner | public wire + descriptor + generated static | `validate`, `validate_root_geometry`, `validate_frozen_precommit`, wire deserialize | **both** | kept, restructured (§5.2) |
| `CommittedGroupParams` | `layout/params.rs:87` | owner, plus 1 mirror field (`setup_prefix`) | descriptor + runtime | `validate_commitment_request`, expansion re-audit | neither | split into `FoldParams` + `GroupParams` |
| `PrecommittedLevelParams` | `layout/params/precommitted.rs:165` | owner | descriptor + runtime | `admit`, `validate` | neither | becomes `GroupParams` (§5.3) |
| `GroupOpeningPlan` | `layout/params/precommitted.rs:112` | owner | descriptor + runtime + generated static | through `PrecommittedLevelParams::validate` | **both** | kept verbatim as `GroupParams::opening` |
| `OpeningMethod` | `layout/params/precommitted.rs:17` | owner | descriptor + runtime + generated static | `validate_level_opening_execution` | **both** | kept, unchanged |
| `CommittedSourceEncoding` | `schedule/profiles.rs:17` | owner, plus 1 hard-coded copy | descriptor + runtime | `validate`, `validate_level_opening_execution` | **both** | kept; moves to `FoldParams` (§5.4, §7.2) |
| `ScheduledSetupPrefix` | `proof/setup_prefix.rs:46` | owner | descriptor + runtime | `validate_structure` | neither | deleted, merged into `GroupParams` |
| `SetupPrefixSlotId` | `proof/setup_prefix.rs:37` | owner | registry key + wire | `Valid::check` | neither | **kept**; derived by `GroupParams::slot_id()` |
| `TerminalCommittedGroupParams` | `schedule.rs:108` | owner | descriptor + generated | `try_from_expanded_group` | neither | merged into `TerminalFoldParams` |
| `RootFinalGroupParams` | `schedule.rs:61` | pass-through (1 field) | runtime | none | neither | deleted |
| `RootPrecommittedGroupParams` | `schedule.rs:66` | mirror (`descriptor` = `commitment.layout`) | runtime | `audit.rs:519`, `:520` | neither | deleted |
| `RootFoldParams` | `schedule.rs:72` | 4 mirrors | descriptor + runtime | `audit.rs:488`, `:489`, `:490` | neither | merged into `FoldParams` |
| `RecursiveFoldParams` | `schedule.rs:81` | 4 mirrors | descriptor + runtime | `audit.rs:550`, `schedule.rs:440` | neither | merged into `FoldParams` |
| `TerminalFoldParams` | `schedule.rs:291` | owner | descriptor + runtime | `audit_terminal` | neither | kept as the merged terminal type |
| `RootFoldStep` / `RecursiveFoldStep` / `TerminalFoldStep` | `schedule.rs:298`, `:305`, `:312` | wrappers | descriptor + runtime | `validate_structure` | neither | merged into their params types |
| `WitnessPartition` | `schedule.rs:46` | lossy mirror of `ChunkedWitnessCfg` | descriptor + runtime | none | neither | deleted |
| `InnerCommitMatrixParams` | `sis/ajtai_key.rs:430` | owner | wire + descriptor + generated static | `try_new`, `validate` SIS re-audit | **both** | kept hand-written (§0.2, §5.1) |
| `OuterCommitMatrixParams` / `OpenCommitMatrixParams` | `sis/ajtai_key.rs:923`, `:928` | owners, 2 macro copies | wire + descriptor + generated static | `try_new`, `validate` SIS re-audit | **both** | one generic + 2 aliases (§5.1) |
| `LevelParamsLike` (trait, 22 methods) | `layout/params/precommitted.rs:431` | symptom | runtime | none | n/a | deleted (§12) |
| `CommitmentGeometry<'a>` | `akita-prover/src/api/commitment.rs:125` | borrowed mirror (10 fields) | runtime | `validate_commitment_geometry` | n/a | deleted (§12.4) |
| `FoldScheduleDescriptorStep<'a>` | `schedule.rs:327` | borrowed view | descriptor | none | n/a | deleted (§12.4) |
| `TerminalFoldDescriptor<'a>` | `schedule.rs:336` | borrowed view | descriptor | none | n/a | deleted (§12.4) |
| `OpeningExecutionGroup<'a>` | `schedule.rs:586` | borrowed view | runtime | none | n/a | deleted (§12.4) |
| `CommittedGroupBatchProfile` | `schedule/profiles.rs:530` | owner, overlaps `AkitaScheduleLookupKey` | runtime | none | neither | out of scope; see §15.4 |

Out of scope, unchanged: `DecompositionParams`, `SparseChallengeConfig`,
`PlannerPolicy`, `SisSecurityPolicyId`, `SisModulusProfileId`, `SisTableKey`,
`SisL2TableKey`, `PhysicalL2NormProofShape`, `InnerCommitSecurityRoute`,
`CommitmentRingDims`, `CommitmentSliceCount`, `CommitmentSliceGeometry`,
`SubringCoefficientPackingGeometry`, `AkitaInstanceDescriptor`,
`AkitaSetupDescriptor`, `PolynomialGroupLayout`, `ChunkedWitnessCfg`,
`TerminalResponseShape`, `AkitaScheduleLookupKey`, `PrecommittedGroupProfiles`,
`OpeningFamily<Trace, Packing>` (§0.11), and the SIS estimator configs. These
hold policy, computed values, or a stage-generic payload carrier — not duplicated
commitment geometry.

## 3. The five faults, with evidence

**Fault 1 — ten audits compare a value with a copy of itself.** Draft 2 counted
six. The new code added four more.

| Copy | Audit that keeps it honest |
|---|---|
| `RootFoldParams::precommitted_groups.len()` vs the lookup key | [audit.rs:502](../crates/akita-schedules/src/audit.rs#L502) |
| `RootFoldParams::open_commit_matrix` | [audit.rs:503](../crates/akita-schedules/src/audit.rs#L503) |
| `CommittedGroupParams::precommitted_groups.len()` | [audit.rs:504](../crates/akita-schedules/src/audit.rs#L504) |
| `RootPrecommittedGroupParams::descriptor` | [audit.rs:519](../crates/akita-schedules/src/audit.rs#L519), [:520](../crates/akita-schedules/src/audit.rs#L520) |
| `RootPrecommittedGroupParams::commitment` | [audit.rs:521](../crates/akita-schedules/src/audit.rs#L521) |
| `RecursiveFoldParams::open_commit_matrix` | [audit.rs:550](../crates/akita-schedules/src/audit.rs#L550) |
| `CommittedGroupParams::setup_prefix` | [schedule.rs:440](../crates/akita-types/src/schedule.rs#L440) |
| `GeneratedSetupPrefixInput::opening.log_basis_open` | [expand.rs:96](../crates/akita-schedules/src/generated/expand.rs#L96) |
| `GeneratedSetupPrefixInput::opening.fold_challenge_config` | [expand.rs:95](../crates/akita-schedules/src/generated/expand.rs#L95) |
| `GeneratedSetupPrefixInput::opening` (whole plan) | [expand.rs:116](../crates/akita-schedules/src/generated/expand.rs#L116) |

The last three are new and instructive. `GeneratedSetupPrefixInput` stores a
whole `GroupOpeningPlan`, and `expand_to_precommitted_group` re-derives that
plan from the consuming fold and then rejects the row when the two disagree.
The stored plan carries no information. It is a mirror with three audits, added
after draft 2 was written, and §11.1 deletes it.

Two more mirror pairs have **no audit at all**: both `sparse_challenge_config`
fields on `RootFoldParams` / `RecursiveFoldParams` versus the group-owned
config, and both `witness_partition` fields versus `witness_chunk`. Those are
latent split-brain bugs today.

**Fault 2 — the role lives in the name, not in the data.** The macro at
[ajtai_key.rs:750](../crates/akita-types/src/sis/ajtai_key.rs#L750) emits about
170 lines twice, once for B and once for D. The A role escaped the macro when it
gained `InnerCommitSecurityRoute`, so the duplication is smaller than draft 2
claimed and it is now permanent unless a generic covers exactly two roles.

**Fault 3 — 21 hand-written byte encoders on the parameter surface** (25 in
`akita-types` and `akita-schedules` overall). Each re-lists its fields in a fixed
order. Three of them order the same geometry three different ways (§10.1).
Nothing links an encoder to its struct, so a new field is silently unbound.

**Fault 4 — a trait hides fault 5.** `LevelParamsLike` has 22 methods and 2
implementations. One of its methods is a lie by construction: the
`PrecommittedLevelParams` impl of `source_encoding` returns a hard-coded
`CanonicalCoefficientTable` ([precommitted.rs:546-547](../crates/akita-types/src/layout/params/precommitted.rs#L546-L547))
rather than reading stored state, because a precommitted group has nowhere to
store it. §7.2 fixes the underlying ownership question.

Draft 2 also claimed `num_live_ring_elements_per_claim` and `b_col_len` were
dead accessors. **That is no longer true.** The packing work gave
`num_live_ring_elements_per_claim` three non-test call sites — all three of them:
[coefficient_packing_relation.rs:275](../crates/akita-types/src/proof/coefficient_packing_relation.rs#L275),
[ring_relation.rs:166](../crates/akita-prover/src/protocol/ring_relation.rs#L166),
and [coefficient_packing.rs:37](../crates/akita-verifier/src/protocol/core/fold/coefficient_packing.rs#L37).
`b_col_len` has two, at
[setup_prefix.rs:1071](../crates/akita-types/src/proof/setup_prefix.rs#L1071) and
[prepare.rs:502](../crates/akita-types/src/setup_contribution/plan/prepare.rs#L502).
Neither accessor may be dropped; both become field reads. Draft 3 said "four" for
the first accessor while §12.2 said "~3"; three is correct.

**Fault 5 — the fold's final group is not a group.** See §2. This is the root
cause. It still produces a real defect:
`terminal_response_linf_limit_for_params` receives a per-group `params` but
reads the **receiver fold's** `fold_challenge_config`, and it omits the
`i16::MAX` clamp that its sibling `certified_response_linf_cap` applies. Two
functions compute "the certified terminal response cap" and disagree.
`AGENTS.md` forbids exactly this. §16 makes the fix a prerequisite.

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

Rule 4 is why this draft has 10 types and not 1. Rules 1 to 3 are why it has 10
and not 24.

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
    pub fn block_index_domain_size(&self) -> Result<usize, AkitaError>;
}

/// One gadget decomposition: a basis and an exact depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GadgetDigits {
    pub log_basis: u32,
    pub num_digits: usize,
}

impl GadgetDigits {
    /// A `SignedDigitKernel` exists for `log_basis`, and `num_digits` is in
    /// `(0, compute_num_digits_field_width(field_bits, log_basis)]`.
    ///
    /// This is a **bound**, not an exact depth. Since #407 the A role's exact
    /// depth is `ceil(log_commit_bound / log_basis_inner)`, which is level-aware
    /// and strictly below the field-width bound for a bounded source. The exact
    /// check stays where it is today, in `audit_committed_params` (§0.9, §9).
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError>;
    pub const fn new(log_basis: u32, num_digits: usize) -> Self;
}

/// Sealed role marker for the two table-keyed roles. Downstream crates cannot
/// invent roles.
pub trait LinfMatrixRole: sealed::Sealed { const ROLE: SisMatrixRole; }
pub struct Outer; pub struct Open;

/// One audited L-infinity Ajtai matrix identity. Replaces the two macro copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinfCommitMatrix<R: LinfMatrixRole> {
    output_rank: usize,
    input_width: usize,
    sis_table_key: SisTableKey,
    _role: PhantomData<R>,
}

/// Kept permanently. These names document the protocol role at every signature
/// and at all 60 generated static-table call sites.
pub type OuterCommitMatrixParams = LinfCommitMatrix<Outer>;
pub type OpenCommitMatrixParams  = LinfCommitMatrix<Open>;

/// A gadget decomposition and the matrix that consumes it.
///
/// Generic over the matrix type, not over the role, because the A role's matrix
/// is `InnerCommitMatrixParams` and carries a security route (§0.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoleParams<M> {
    pub digits: GadgetDigits,
    pub matrix: M,
}

pub type InnerRoleParams = RoleParams<InnerCommitMatrixParams>;
pub type OuterRoleParams = RoleParams<OuterCommitMatrixParams>;
pub type OpenRoleParams  = RoleParams<OpenCommitMatrixParams>;
```

`InnerCommitMatrixParams` stays exactly as it is today. It is not a
`LinfCommitMatrix<Inner>`: its `sis_table_key()` and `coeff_linf_bound()` return
`Option`, it owns `InnerCommitSecurityRoute`, and its `validate` and
`append_descriptor_bytes` branch on the route. Forcing it into the generic would
reintroduce exactly the `Option`-shaped tag that finding 3 rejected. What is
shared instead is the **audit code**, which already is:
`audit_commit_matrix_fields` and `min_rank_commit_matrix_fields`
([ajtai_key.rs:398](../crates/akita-types/src/sis/ajtai_key.rs#L398)) serve all
three roles today.

`LinfCommitMatrix::new_unchecked` stays `const`. The static tables need it.
`try_new` checks that `sis_table_key.role == R::ROLE` and re-audits the rank
against the checked-in SIS table, exactly as the macro does today.

### 5.2 The frozen group — public wire data

```rust
/// Group metadata frozen when a standalone commitment group is created.
///
/// This is the public, versioned wire form. It holds only what the commit step
/// fixes: the group shape, the block geometry, the B slicing, and the two
/// group-owned roles. A group's D digits, opening method, and fold challenge
/// family are chosen by the fold that opens it, so they are not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommittedGroupProfile {
    pub version: u8,
    pub group: PolynomialGroupLayout,
    pub blocks: BlockGeometry,
    pub outer_slice_count: CommitmentSliceCount,
    pub inner: InnerRoleParams,
    pub outer: OuterRoleParams,
}
```

Same name, same wire bytes, same `Copy`, same 288 bytes as today. Only the field
grouping changes: 12 flat fields become 6 nested ones. `VERSION` stays `4`.

### 5.3 A group inside one fold's opening batch

```rust
/// One commitment group taking part in one fold's opening batch.
///
/// Every group in a fold has this type: the final/new group, each precommitted
/// group, and the setup prefix. The fold owns the shared D matrix; a group
/// owns only its contribution of D digits, through `opening`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupParams {
    /// Frozen commit-time identity of this group.
    pub profile: CommittedGroupProfile,
    /// Opening policy chosen by the fold that consumes this group. Unchanged
    /// from today's `GroupOpeningPlan`: opening method, fold challenge family,
    /// `log_basis_open`, `num_digits_open`, `num_digits_fold`.
    pub opening: GroupOpeningPlan,
    /// Active setup-weight support, in flat field coefficients.
    ///
    /// `Some` exactly when this group is the consuming fold's setup prefix.
    /// This is the sole record of that fact and the sole record of the length.
    pub setup_natural_len: Option<usize>,
}

impl GroupParams {
    /// Registry identity for a prefix group. `None` for an ordinary group.
    ///
    /// `SetupPrefixSlotId` stays the runtime registry key (§0.5); it is now
    /// derived here instead of stored twice.
    pub fn slot_id(&self) -> Option<SetupPrefixSlotId>;
    /// Admission path. Today's `PrecommittedLevelParams::admit`, plus the
    /// prefix-geometry checks moved out of `validate_structure` (§9).
    pub fn admit(/* … */) -> Result<Self, AkitaError>;
    pub fn validate(&self) -> Result<(), AkitaError>;
}
```

`GroupParams` is `Copy`: `CommittedGroupProfile`, `GroupOpeningPlan`, and
`Option<usize>` are all `Copy` today. Sites pass it by value.

`setup_natural_len` is not a tag beside a payload. It **is** the payload. A
group is a setup prefix when it has a setup-weight support length, and that
length is the field. There is no second field to compare it against, so there
is no audit.

This is the smallest of the five type changes: `profile` renames `layout`,
`opening` is untouched, and one `Option<usize>` arrives. `ScheduledSetupPrefix`
disappears into it.

### 5.4 One fold level

```rust
/// Parameters for one fold level: the root fold or one recursive fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldParams {
    pub payload_mode: CommitmentPayloadMode,
    /// Physical source encoding this fold's own new witness is committed under.
    /// Precommitted and prefix groups are canonical by admission (§7.2).
    pub source_encoding: CommittedSourceEncoding,
    /// Groups in canonical transcript order. Never empty.
    ///
    /// The last entry is the final/new group. Earlier entries are precommitted.
    /// A setup prefix, when present, is `groups[0]`.
    pub groups: Vec<GroupParams>,
    /// The shared D matrix over every group's `w_hat` segment.
    pub open_matrix: OpenCommitMatrixParams,
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
    /// A group's A/B dimensions with this fold's shared D dimension.
    pub fn role_dims(&self, group: &GroupParams) -> CommitmentRingDims;
    /// Source encoding of `groups[index]`: this fold's own for the final group,
    /// `CanonicalCoefficientTable` for every earlier group.
    pub fn source_encoding_of(&self, index: usize) -> CommittedSourceEncoding;
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

Five kinds of mirror field vanish here. `open_commit_matrix` is stored once.
`sparse_challenge_config` is read from the group that uses it, which also fixes
fault 5. `witness_partition` is read as `witness_chunk.num_chunks`.
`precommitted_groups` and the flat final-group fields are one list.
`source_encoding` has one owner instead of one owner plus one hard-coded trait
arm.

### 5.5 The terminal fold

```rust
/// The last fold. It binds only the source decomposition through the inner
/// matrix. It has no outer or open matrix, no groups, and no chunking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalFoldParams {
    pub blocks: BlockGeometry,
    pub inner: InnerRoleParams,
    /// Response basis and depth this terminal fold was planned against.
    pub fold: GadgetDigits,
    pub fold_challenge_config: SparseChallengeConfig,
    pub response_shape: TerminalResponseShape,
    pub input_witness_len: usize,
}

impl TerminalFoldParams {
    /// Project an ordinary fold's final group into terminal parameters and
    /// certify the directly checked response bound.
    ///
    /// Returns the admitted L-infinity capacity, or `None` for an L2 route.
    /// Carries all 6 fallible steps that
    /// `TerminalCommittedGroupParams::try_from_expanded_group` performs today.
    ///
    /// `group` supplies the block geometry, the inner role, and the challenge
    /// family, so none of the three is a separate argument.
    pub fn admit(
        group: &GroupParams,
        response_shape: TerminalResponseShape,
        input_witness_len: usize,
    ) -> Result<(Self, Option<u128>), AkitaError>;

    /// The one certified-capacity function. §16 removes today's second copy.
    /// Reads its own `fold_challenge_config`; it takes no config argument.
    pub fn certified_response_linf_cap(&self) -> Result<u128, AkitaError>;

    /// Route-aware wire check. Unchanged behaviour, one fewer argument.
    pub fn validate_terminal_linf_cap(
        &self,
        scheduled_cap: Option<u128>,
    ) -> Result<(), AkitaError>;

    pub fn response_l2_sq_cap(&self) -> Option<u128>;
}
```

`TerminalFoldParams` merges three types: `TerminalCommittedGroupParams`,
today's `TerminalFoldParams`, and `TerminalFoldStep`. It has no
`output_witness_len` field, so a terminal fold cannot claim a committed
successor.

The merge is what removes the §16 divergence permanently. Today
`certified_response_linf_cap` takes a `SparseChallengeConfig` argument even
though `TerminalFoldParams::sparse_challenge_config` sits one field away from
the `witness` it is called on. That argument is the vector by which a caller can
supply the wrong config. After the merge there is no argument to get wrong.

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
`CommitParams → CommitKind::Level → the prefix type → PrecommittedLevelParams
→ CommitParams`, with no indirection at any step. `rustc` rejects it with
`E0072`. The tree already broke that edge (§0.5). This draft keeps the graph a
DAG:

```text
FoldParams  → Vec<GroupParams>              (heap, and not the cycle anyway)
FoldParams  → OpenCommitMatrixParams        → SisTableKey → scalars
GroupParams → CommittedGroupProfile         → RoleParams<M> → matrix → scalars
GroupParams → GroupOpeningPlan              → SparseChallengeConfig, scalars
GroupParams → Option<usize>                 → scalars
TerminalFoldParams → BlockGeometry, InnerRoleParams, GadgetDigits,
                     TerminalResponseShape
```

`CommittedGroupProfile` is a leaf: every field is a scalar or a `Copy` leaf
struct. No type reachable from it names `GroupParams` or `FoldParams`.
`SetupPrefixSlotId` already reaches only `CommittedGroupProfile`, so deriving it
from a `GroupParams` adds no edge. Every type is finite-sized without `Box`.
**No allocation is added anywhere.**

**Static constructibility.** The 13 generated tables embed **30**
`CommittedGroupProfile` struct literals and **60** `new_unchecked` calls in
`static` position, for example
[fp128_onehot.rs:93](../crates/akita-schedules/src/generated/fp128_onehot.rs#L93).
These types therefore must stay `Copy`, `'static`, and const-constructible.
This draft keeps that:

| Type | `Copy` | const-constructible | Needed in a `static`? |
|---|---|---|---|
| `BlockGeometry` | yes | public fields + `const fn new` | yes |
| `GadgetDigits` | yes | public fields + `const fn new` | yes |
| `InnerCommitMatrixParams` | yes | `const fn new_unchecked` (kept) | yes |
| `LinfCommitMatrix<R>` | yes | `const fn new_unchecked` (kept) | yes |
| `RoleParams<M>` | yes | public fields | yes |
| `CommittedGroupProfile` | yes | public fields | yes |
| `GroupOpeningPlan` | yes | public fields (already in a `static`) | yes |
| `OpeningMethod` | yes | enum literal (already in a `static`) | yes |
| `CommittedSourceEncoding` | yes | enum literal | no |
| `GroupParams` | yes | public fields | no — expanded at run time |
| `FoldParams` | no (`Vec`) | not required | no |
| `TerminalFoldParams` | no (`Vec` in response shape) | not required | no |

`PhantomData<R>` is a zero-sized `const` field, so `RoleParams<M>` literals
compile in `static` position. `GeneratedSetupPrefixInput` already embeds a
`GroupOpeningPlan` literal in a `static`, so that boundary is proven, not
assumed. The two matrix aliases keep the spelling that the emitter already
writes, so §11 changes the emitter for nesting only, not for matrix syntax.

Note that the A route only ever appears as `Linf` in a generated table: the
`new_unchecked` const constructor builds an `InnerCommitSecurityRoute::Linf`
([ajtai_key.rs:620-641](../crates/akita-types/src/sis/ajtai_key.rs#L620-L641)),
and an L2 route is installed at expansion time from
`GeneratedRecursiveFold::response_l2_sq_cap`. §11 does not change that.

## 7. Authority table

Every field that has two or more owners today gets exactly one owner.

| Field | Owners today | Single owner after | Deleted mirror(s) |
|---|---|---|---|
| Shared D matrix | `CommittedGroupParams::open_commit_matrix`, `RootFoldParams::open_commit_matrix`, `RecursiveFoldParams::open_commit_matrix` | `FoldParams::open_matrix` | 2 fields, audits `:503`, `:550` |
| Fold challenge family | `CommittedGroupParams::fold_challenge_config`, `GroupOpeningPlan::fold_challenge_config`, `RootFoldParams::sparse_challenge_config`, `RecursiveFoldParams::sparse_challenge_config` | `GroupParams::opening.fold_challenge_config`, per group | 2 fields, no audit existed |
| Witness chunking | `CommittedGroupParams::witness_chunk`, `RootFoldParams::witness_partition`, `RecursiveFoldParams::witness_partition` | `FoldParams::witness_chunk` | 2 fields + `WitnessPartition` type, no audit existed |
| Precommitted groups | `CommittedGroupParams::precommitted_groups`, `RootFoldParams::precommitted_groups`, the lookup key's `precommitteds` | `FoldParams::groups` | 1 field, audits `:502`, `:504` |
| Frozen group descriptor | `RootPrecommittedGroupParams::descriptor`, `…::commitment.layout` | `GroupParams::profile` | 1 field, audits `:519`, `:520`, `:521` |
| Setup prefix | `CommittedGroupParams::setup_prefix`, `RecursiveFoldParams::incoming_setup_prefix` | `FoldParams::groups[0].setup_natural_len` | 2 fields + `ScheduledSetupPrefix` type, check `schedule.rs:440` |
| Setup-prefix opening plan | `GeneratedSetupPrefixInput::opening`, re-derived by expansion | derived only | 1 field, audits `expand.rs:95`, `:96`, `:116` |
| Final group geometry and roles | `CommittedGroupParams` flat fields, and a copy in `CommittedGroupProfile::from_params_fields` | `FoldParams::groups.last()` | `RootFinalGroupParams`, the `from_params_fields` copy |
| Source encoding | `CommittedGroupParams::source_encoding`, hard-coded in the `PrecommittedLevelParams` trait arm | `FoldParams::source_encoding` (§7.2) | 1 hard-coded arm |
| Block index bit widths | duplicated formula in `CommittedGroupParams` and both `LevelParamsLike` impls | `BlockGeometry` methods | 2 duplicate formulas |
| Opening basis dominance check | `PrecommittedLevelParams::admit:236` **and** `validate:342` | `GroupParams::validate` | 1 duplicate check |
| Certified terminal response cap | `certified_response_linf_cap` and `terminal_response_linf_limit_for_params`, which **disagree** | one function (§16) | 1 divergent copy |

### 7.1 On the setup prefix, specifically

The review and draft 1 disagreed about which of the two setup-prefix fields is
authoritative. Both were right about their own evidence, and both remedies were
worse than deleting the question.

- The documentation and [typed-schedule-topology-cutover.md:100](typed-schedule-topology-cutover.md#L100)
  say the consuming fold's edge is canonical.
- The dataflow says otherwise: every writer populates
  `CommittedGroupParams::setup_prefix` first and clones it into the edge
  ([runtime.rs:689](../crates/akita-schedules/src/runtime.rs#L689)). About 31
  non-test sites read the group-params field and about 34 read the edge.
- The group-params field is load-bearing beyond identity. It makes the prefix
  **precommitted group index 0**, through `precommitted_group_iter`
  ([params.rs:337-343](../crates/akita-types/src/layout/params.rs#L337-L343)).
  Shared-D width, relation ordering, and verifier payload assembly all depend on
  that ordering.

One field satisfies both constraints. `FoldParams::groups[0]` is owned by the
fold that **consumes** the prefix, which is the successor, so the spec rule
holds. It is also literally group 0, so the dataflow readers keep working
without an extra argument. Removing either existing field alone would have
forced the other side's consumers to change; removing both and storing the
prefix once forces neither.

`SetupPrefixSlotId` is unaffected and stays. It is a registry key, not a schedule
field, and [typed-schedule-topology-cutover.md:105](typed-schedule-topology-cutover.md#L105)
requires it. It becomes derived rather than stored, which removes the third copy
of the prefix's frozen profile.

**One ordering discrepancy to resolve during step 5b.**
`precommitted_group_iter` puts the prefix **first**, but
`validate_nonterminal_opening_execution` builds its group list with the witness
first and pushes the prefix **last**
([schedule.rs:570-575](../crates/akita-types/src/schedule.rs#L570-L575)). Those
are different orderings of the same set. Today two independent constructions can
hold different opinions without either being wrong, because neither is named as
canonical. `FoldParams::groups` makes one ordering canonical, so the execution
check must be rewritten to iterate `groups` and the "prefix at index 0" rule
becomes the single answer. Verify that
`validate_level_opening_execution`'s use of `groups.first()` as the family
reference still selects the intended group after the reorder; today it reads the
witness, and after the reorder it reads the prefix.

### 7.2 On `source_encoding`, specifically

`CommittedSourceEncoding` is documented as commitment identity, not opening
policy ([profiles.rs:11-15](../crates/akita-types/src/schedule/profiles.rs#L11-L15)),
yet it is not stored on `CommittedGroupProfile`, which is the commitment-identity
type. It is stored on `CommittedGroupParams`, and the
`PrecommittedLevelParams` arm of `LevelParamsLike::source_encoding` returns a
hard-coded constant.

That hard-coding is correct today, for a non-obvious reason:
`CommittedGroupProfile::try_from_params` **rejects** any non-canonical encoding
([profiles.rs:125-129](../crates/akita-types/src/schedule/profiles.rs#L125-L129)),
so a group with a tensor projection can never become a standalone or
precommitted profile. The constant is a consequence of a rejection several
modules away.

Two remedies exist. This plan takes the cheaper one.

- **Taken: keep the encoding off the profile.** `FoldParams::source_encoding`
  describes the fold's own new witness. `source_encoding_of(index)` returns
  `CanonicalCoefficientTable` for every non-final group. The hard-coded trait arm
  disappears and profile bytes do not move.
- **Not taken: move `source_encoding` onto `CommittedGroupProfile`.** This is
  arguably more honest, and it would let the rejection live at the
  standalone-commit boundary where it belongs. It costs a profile version bump
  from 4 to 5, new profile bytes, a new `key_digest` for all 13 catalogs, and it
  breaks the §10.2 byte-preservation property that the rest of this plan leans
  on. Record it as a follow-up, not as part of this change.

One prerequisite either way: under §5.4 the fold's final group is a
`GroupParams` with a `profile`. A recursive fold whose witness uses
`TensorSubfieldProjection` therefore needs a `CommittedGroupProfile` that
`try_from_params` currently refuses to build. Move that rejection out of the
constructor and into `validate_frozen_precommit` — the precommit admission
boundary — where the rule "a standalone commitment must be canonical" actually
belongs. This is a relocation, not a relaxation: §9 lists it, and §14 requires
its rejection test.

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
| Two fields disagreeing about the same value | **yes**, 10 audits plus 2 unaudited pairs | **no** | the main gain |
| A group claiming a D matrix it does not own | **yes** | **no** | D matrix only on `FoldParams` |
| A terminal cap computed from another fold's challenge config | **yes** | **no** | no config argument survives (§5.5) |
| A recursive fold with more than one precommitted group | **yes, unvalidated** | yes, **validated** | gain |
| Groups in one fold disagreeing about the opening-method family | yes, validated at `validate_level_opening_execution` | yes, validated in `FoldParams::validate` | unchanged guarantee, one home |
| A root fold naming an incoming setup prefix | no (field absent) | yes, **validated** | **the one trade** |

Two entries need comment.

**"Root fold inside `recursive_folds`" is still prevented.** Root and recursive
share `FoldParams`, so a value could be placed in either field. What the schema
guarantees is what it guarantees today: exactly one root, exactly one terminal,
and an ordered list between them. No role is inferred from an array index,
because `FoldSchedule` names the three positions. The properties that separate a
root from a recursive fold — the root payload must be compressed, the root
consumes no prefix, a recursive fold has at most one prefix group, the root A
route must be `Linf` ([audit.rs:488-496](../crates/akita-schedules/src/audit.rs#L488-L496)) —
are validated constraints. Three of the four are validated constraints today as
well.

**The one trade.** Today `RootFoldParams` has no prefix field, so a root cannot
name a prefix at the type level. After the change,
`FoldSchedule::validate_structure` must reject it. This is a real, if small,
loss. It buys the deletion of 10 mirror audits, 2 unaudited split-brain pairs,
one recursive type cycle, and one divergent security calculation. §14 requires a
rejection test for it.

Net: one type-level guarantee traded, two gained, and twelve parallel-tag
invariants deleted.

## 9. Validation-path table

Every check that exists today has a named destination. No check is dropped.

| Check today | Location today | Destination |
|---|---|---|
| version equality; both matrix `validate`s; power-of-two A/B dims with `d_b \| d_a`; digit kernel exists; depth within field width; nonzero outer basis and depth; slice-count admissibility; `field_bits` agreement; exact A width; exact B width via `CommitmentSliceGeometry` | `CommittedGroupProfile::validate` (`profiles.rs:189`) | `CommittedGroupProfile::validate(field_bits)`, unchanged, plus geometry below |
| `N · d_a == 2^num_vars`; `M` power of two; `B == ceil(N / M)` | `validate_root_geometry` (`profiles.rs:262`) | `M`/`B` go to `BlockGeometry::validate`; `N · d_a == 2^num_vars` stays on `CommittedGroupProfile::validate` |
| conjunction of the two above | `validate_frozen_precommit` (`profiles.rs:301`) | **kept.** Draft 2 said "deleted". That was wrong: it is the entry point for the `CommittedGroup<F>` wire deserialize (`proof/commitment.rs:207` and `:459` — two sites, not one) and for `PrecommittedLevelParams::admit` |
| standalone commitments require a canonical source encoding | `CommittedGroupProfile::try_from_params:125` | `validate_frozen_precommit`, so an in-fold final-group profile can carry a tensor projection (§7.2) |
| digit kernel and depth-within-field-width | `CommittedGroupProfile::validate` and `expand.rs`, twice | `GadgetDigits::validate(field_bits)`, one place |
| **exact** A digit depth equals `ceil(log_commit_bound / log_basis_inner)`, with the root distinguished from recursive levels | `audit_committed_params` (`audit.rs:216-230`), added by #407 | **kept, unchanged**, in `akita-schedules`; reads `groups.last()`. Not a mirror audit: `num_digits_inner` already has one owner. `GadgetDigits::validate` gives only the field-width bound and cannot replace this (§0.9, §5.1) |
| opening basis dominates the frozen outer basis | `PrecommittedLevelParams::admit:236` **and** `validate:342` | `GroupParams::validate`, once |
| A and B bounds cover the certified opening basis; frozen B bound covers its own basis; outer digit depth matches its frozen basis; modulus-profile agreement; A is not an L2 route | `PrecommittedLevelParams::admit:205-280` | `GroupParams::admit`, unchanged |
| fold-challenge family validates for the A ring or the packing subring | `PrecommittedLevelParams::validate:322-333` | `GroupParams::validate`, unchanged |
| `natural_len != 0`; `natural_len <= n_prefix`; `d_setup() != 0`; `n_prefix % d_setup() == 0`; prefix slice-count admissibility | `FoldSchedule::validate_structure:445-466` | `GroupParams::validate`, which owns the field |
| setup-prefix mirror agreement | `validate_structure:440` | **deleted** — there is no mirror |
| root payload is compressed | `validate_structure:372` | unchanged |
| root A route must be `Linf` | `audit.rs:488` | unchanged; reads `groups.last()` |
| payload-phase cutover policy per recursive fold | `validate_structure:375-401` | unchanged; reads `setup_prefix()` |
| witness-length chaining; stage-2 successor capacity | `validate_structure:396-499` | unchanged |
| terminal lengths and response length nonzero | `validate_structure:502-506` | unchanged |
| opening-method family per absolute level; one family per fold; source-encoding/method compatibility | `validate_level_opening_execution:591` | the per-fold parts move to `FoldParams::validate`; the per-level parts stay on `FoldSchedule`. Iterate `groups`, not a rebuilt list (§7.1) |
| `groups` is non-empty | n/a (flat fields) | **new** in `FoldParams::validate` |
| root names no setup prefix | implied by the type | **new** in `validate_structure` |
| at most one prefix group per fold, at index 0 | not enforced | **new** in `FoldParams::validate` |
| fold-coefficient count fits `usize`; cap config admissible; witness cap computable; certified capacity nonzero and not L2; minimum-retention heuristic `>= 1/2` of target | `TerminalCommittedGroupParams::try_from_expanded_group` (6 fallible steps) | `TerminalFoldParams::admit`, all 6. `TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM/DEN` and their doc comment move verbatim |
| terminal L2 route carries no independent Linf cap; Linf route carries a nonzero cap within its certified capacity | `validate_terminal_linf_cap:240` | `TerminalFoldParams::validate_terminal_linf_cap`, one fewer argument |
| A-role check on the inner matrix; L2 route rejected | `terminal_response_linf_limit_for_params:405` | `TerminalFoldParams::certified_response_linf_cap` |
| `i16::MAX` representation clamp | `certified_response_linf_cap:235` only | same single function, after §16 resolves the divergence |
| wire: version; tag well-formedness; role identity; SIS rank re-audit; geometry; slice-count source coefficients; untrusted-coefficient allocation cap; payload consistency | `CommittedGroup<F>` deserialize (`proof/commitment.rs:443-489`) | unchanged. The wire form is byte-identical and constructs `CommittedGroupProfile` only |
| mirror equality audits | `audit.rs:502`, `:503`, `:504`, `:519`, `:520`, `:521`, `:550`; `schedule.rs:440`; `expand.rs:95`, `:96`, `:116` | **deleted** — single owners |
| shared-D width; shared-D basis; terminal audit | `audit.rs:206`, `:256`, `audit_terminal` | kept; read `FoldParams::groups` |
| generated expansion re-audits rank against the SIS table | `expand.rs`, 3 parallel paths | one `expand_group`, same checks (§11) |

The original review asked whether a wire boundary must prove it built the frozen
case. It does not have to prove anything: `CommittedGroupProfile` is the only
type the wire encoder accepts, and no `FoldParams` or `TerminalFoldParams` can
appear there. There is no case to reject because there is no enum.

## 10. Byte policy

### 10.1 Why byte preservation is impossible above the profile

The three encoders order the same fields three different ways today. Recomputed
against `8e552d2ac`.

| Position | `CommittedGroupParams` (`params.rs:614`) | `CommittedGroupProfile` (`profiles.rs:172`) | `TerminalCommittedGroupParams` (`schedule.rs:278`) |
|---|---|---|---|
| 1 | payload mode tag | version | A basis |
| 2 | source encoding | num_vars | fold basis |
| 3 | opening method | num_polynomials | fold digit count |
| 4 | A basis | N | **A matrix** |
| 5 | B basis | M | N |
| 6 | D basis | B | M |
| 7 | **A matrix** | slice count | B |
| 8 | **B matrix** | A basis | A depth |
| 9 | **D matrix** | A depth | — |
| 10 | N | **A matrix** | — |
| 11 | M | B basis | — |
| 12 | B | B depth | — |
| 13 | slice count | **B matrix** | — |
| 14 | sparse challenge | — | — |
| 15-21 | A, B, D, fold depths; chunk (conditional); groups (conditional); prefix flag | — | — |

`CommittedGroupProfile` is already role-atomic: basis, depth, matrix, twice.
`TerminalCommittedGroupParams` splits the atom and interleaves the fold
decomposition into it. `CommittedGroupParams` groups by kind, not by role, so its
roles interleave across the whole record. No single role encoder can serve all
three.

Two facts survive from draft 2 and still hold. The `N, M, B` triple is
contiguous in all three encoders, so an atomic `BlockGeometry` encoder is
byte-neutral everywhere. The profile's `(basis, depth, matrix)` grouping matches
`RoleParams` exactly, so a `RoleParams` encoder is byte-neutral there.

### 10.2 The policy: preserve the profile, break everything above it, once

**Preserved, byte for byte:**

- `LinfCommitMatrix<R>` bytes. The macro's encoder is already identical for B and
  D. `InnerCommitMatrixParams::append_descriptor_bytes` is untouched.
- `BlockGeometry` bytes. The `N, M, B` triple is contiguous in all three
  encoders today, so an atomic geometry encoder is byte-neutral everywhere.
- `RoleParams<M>` bytes as `basis, depth, matrix`.
- `GroupOpeningPlan` bytes. Its encoder (`precommitted.rs:151`) already exists
  and is unchanged.
- **`CommittedGroupProfile` bytes and wire form.** Its declared field order in
  §5.2 reproduces today's order exactly. `version` stays `4`.

This is worth the constraint. Profile bytes reach the catalog `key_digest`
through `entries_key_digest`
([catalog_identity.rs:562](../crates/akita-schedules/src/catalog_identity.rs#L562),
[:659](../crates/akita-schedules/src/catalog_identity.rs#L659)), and they are the
catalog sort key
([generated/mod.rs:305](../crates/akita-schedules/src/generated/mod.rs#L305)).
Keeping them fixed means entry ordering does not shift, committed `key_digest`
values stay meaningful, and `CommittedGroup<F>` wire fixtures keep passing.

Note that `key_digest` is **not** the `key_digest(keys: &[PolynomialGroupLayout])`
function at [catalog_identity.rs:546](../crates/akita-schedules/src/catalog_identity.rs#L546),
which hashes lookup keys only. The `GeneratedScheduleCatalogIdentity::key_digest`
field is set from `entries_key_digest` at
[catalog_identity.rs:230](../crates/akita-schedules/src/catalog_identity.rs#L230).
Draft 2 conflated the two.

**Changed once, deliberately:** `GroupParams`, `FoldParams`,
`TerminalFoldParams`, `FoldSchedule`. Their storage no longer matches the old
layout, so their bytes cannot. Required with the break:

| Constant | Location | Today | After |
|---|---|---|---|
| `AKITA_INSTANCE_DESCRIPTOR_VERSION` | `instance_descriptor/mod.rs:37` | `1` | `2` |
| `SCHEDULE_ROW_DOMAIN_V2` | `akita-types/src/schedule_selection.rs:16` | `b"akita/schedule-row/v2"` | `…/v3`, renamed to `_V3` |
| `FoldSchedule` leading descriptor byte | `schedule/descriptor.rs:38`, `:72` | `1` | `2` |
| `TerminalFoldDescriptor` tag byte | `schedule/descriptor.rs:196` | `3` | `3`, unchanged |
| `CommittedGroupProfile::VERSION` | `schedule/profiles.rs:113` | `4` | `4`, unchanged |
| `SETUP_PREFIX_CONTENT_TAG` | `proof/setup_prefix.rs:25` | `b"SPF4"` | unchanged, still inside the slot-id encoding |

`SCHEDULE_ROW_DOMAIN_V2` lives in **`akita-types`**, not `akita-config`. Both
crates have a `schedule_selection.rs`; draft 3 named the file without the crate
and the wrong one is the obvious guess.

The two `FoldSchedule` descriptor-byte sites moved out of `schedule.rs` into the
new `schedule/descriptor.rs` module (§0.8). The byte order did not change.

`protocol_epoch` is `AKITA_INSTANCE_DESCRIPTOR_VERSION`, and every generated
table embeds it, so all 13 tables regenerate. Old proof bytes stop verifying.
`AGENTS.md` allows this; §14 requires the rejection tests. Under #403 (§0.10) a
descriptor-order error now fails as a pre-allocation canonical-schedule rejection,
so run the §14 rejection tests against every transcript backend the CI matrix
selects explicitly.

### 10.3 One encoding rule replaces 21 encoders

> **The canonical byte order is the declared field order, top to bottom.**
> A containing type encodes each field by calling that field's encoder in
> declaration order.

This makes every encoder mechanical and reviewable, and it makes an unbound new
field impossible: adding a field to a struct adds it to the digest.

The original review preferred to design storage layout and encoding order
separately, so that changing one cannot silently change the other. This draft
couples them on purpose, and removes the "silently": reordering a field changes
the golden fixtures of §14 and the committed tables, and both fail loudly in CI.
The alternative is two orders kept in step by hand, which is what produced 21
divergent encoders. The coupling is stated here so a reviewer knows that field
order is protocol-visible.

The resulting orders:

```text
GadgetDigits          := log_basis:u32, num_digits:u64
BlockGeometry         := N:u64, M:u64, B:u64
InnerCommitMatrixParams := unchanged (route-tagged, 5 or 8 items)
LinfCommitMatrix<R>   := unchanged (8 items)
RoleParams<M>         := GadgetDigits, matrix
CommittedGroupProfile := version:u8, num_vars:u64, num_polynomials:u64,
                         BlockGeometry, CommitmentSliceCount,
                         RoleParams<Inner>, RoleParams<Outer>
GroupOpeningPlan      := unchanged (opening_method, log_basis_open,
                         sparse_challenge, num_digits_open, num_digits_fold)
GroupParams           := CommittedGroupProfile, GroupOpeningPlan,
                         has_prefix:u8, [b"SPF4", setup_natural_len:u64]
FoldParams            := payload_mode:u8, source_encoding, groups.len():u64,
                         groups…, LinfCommitMatrix<Open>, ChunkedWitnessCfg,
                         input_witness_len:u64, output_witness_len:u64
TerminalFoldParams    := BlockGeometry, RoleParams<Inner>, GadgetDigits(fold),
                         sparse_challenge, TerminalResponseShape,
                         input_witness_len:u64
FoldSchedule          := 2u8, FoldParams(root), recursive.len():u64,
                         recursive…, TerminalFoldParams
```

`GroupOpeningPlan`'s own order is kept as written rather than normalized,
because it already encodes atomically and reordering it would change generated
`key_digest` values for no gain.

`ChunkedWitnessCfg` is now encoded unconditionally. The conditional at
[params.rs:641-643](../crates/akita-types/src/layout/params.rs#L641-L643) exists
to keep single-chunk descriptors byte-identical to a historical layout. That
invariant retires with the break, and removing the branch removes a way to
collide two different configurations. The same applies to the conditional
`precommitted_groups` block at
[params.rs:645-650](../crates/akita-types/src/layout/params.rs#L645-L650): an
empty list must encode as a zero length, not as nothing.

## 11. Generated tables

### 11.1 Schema: 15 mirrors become 8

`GeneratedInnerCommitMatrix`, `GeneratedOuterCommitMatrix` and
`GeneratedOpenCommitMatrix` are now **byte-identical** two-field structs
(`ring_dimension`, `log_basis`); the slice count moved to
`GeneratedCommittedGroup` with B slicing. Merge them.
`GeneratedRootFinalGroup`, `GeneratedRootPrecommittedGroup`,
`GeneratedSetupPrefixInput` and `GeneratedCommittedGroup` are four spellings of
one group case. Merge them. `GeneratedRootFold` and `GeneratedRecursiveFold`
mirror types that are now one type. Merge them. `GeneratedWitnessPartition`
mirrors a lossy mirror. Delete it.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedGroup {
    pub layout: akita_types::PolynomialGroupLayout,
    pub geometry: GeneratedBlockGeometry,
    pub outer_slice_count: u32,
    pub inner_commit_matrix: GeneratedMatrix,
    pub outer_commit_matrix: GeneratedMatrix,
    pub num_digits_inner: u32,
    pub num_digits_fold: u32,
    pub opening_method: akita_types::OpeningMethod,
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
    pub response_l2_sq_cap: Option<u128>,
}
```

`GeneratedBlockGeometry`, `GeneratedTerminalFold`,
`GeneratedFoldScheduleEntry`, `GeneratedScheduleCatalogIdentity` and
`GeneratedScheduleTable` stay. All 8 keep `Copy` and const construction.

Two notes on what is deliberately **not** stored.

`GeneratedSetupPrefixInput::opening` is dropped, not merged.
`expand_to_precommitted_group` already re-derives the whole plan from the
consuming fold and rejects the row on disagreement
([expand.rs:95-121](../crates/akita-schedules/src/generated/expand.rs#L95-L121)),
so the stored plan carries no information and costs three audits. `GeneratedGroup`
keeps only `opening_method` and `num_digits_fold`, which is what the root and
recursive rows store today; `log_basis_open`, `num_digits_open` and
`fold_challenge_config` stay derived per fold. **Confirm before step 6** that no
prefix in any of the 13 committed tables pins a `log_basis_open` that the
consuming fold's shared basis does not already supply; the audit at
`expand.rs:96` proves this for the current tables, but it must be re-proven after
regeneration.

`frozen_profile: Option<…>` is not a redundant tag. Precommitted groups persist
their exact frozen identity because it is the sorted lookup key and it feeds
`key_digest`. Final groups recompute theirs. The `Option` records which, and
there is nothing to compare it against.

The generated/runtime trust boundary is preserved. These types stay compact
planner inputs. Expansion still re-derives every rank with `secure_rank`
against the checked-in SIS table and still re-audits policy. They are not a
second authority.

### 11.2 `expand.rs`: three copies become one

`expand.rs` is 1027 lines, 935 of them non-test (`mod tests` starts at `:936`).
The three function lengths below are exact, and they sum to 652. Roughly 650 of those are
three near-parallel copies of one algorithm:
`GeneratedSetupPrefixInput::expand_to_precommitted_group`
([expand.rs:69](../crates/akita-schedules/src/generated/expand.rs#L69), 57 lines),
`GeneratedCommittedGroup::expand_to_level_params_with_setup`
([:159](../crates/akita-schedules/src/generated/expand.rs#L159), ~354 lines), and
`expand_to_multi_group_root_level_params_with_setup`
([:513](../crates/akita-schedules/src/generated/expand.rs#L513), ~241 lines). All
three run the same steps in the same order and differ only in the struct they
assemble.

With one `GeneratedGroup` and one `GroupParams`, write one `expand_group`. The
real differences become parameters:

- the opening ring dimension, which a prefix derives from its own outer
  dimension because its consumer's D matrix opens it;
- the D bucket basis, `log_basis_open` for a scalar fold and
  `shared_d_digit_log_basis(...)` for a multi-group fold;
- `live_ring_elements_per_claim`, derived from the witness length or from
  `B · M`;
- the D segment width, which now branches on `OpeningMethod` through
  `opening_d_segment_width` (§0.4). This is a call, not a fourth copy.

The terminal path (`expand.rs:755`, 159 lines) stays separate. It has no B or D
matrices, it installs the L2 route when `response_l2_sq_cap` is present, and it
adds its own response-contract check.

### 11.3 Regeneration and identity

The emitter in [emit/mod.rs](../crates/akita-planner/src/emit/mod.rs) is 993
lines of string templating with no type-level link to the schema. The only thing
preventing drift is one CI step:

```bash
scripts/generate-schedule-tables.sh
git diff --exit-code -- crates/akita-schedules/src/generated
```

Every step that changes a struct must change the matching `emit_*` in the same
commit and commit the regenerated tables. `emit_profile_matrix`
([emit/mod.rs:440](../crates/akita-planner/src/emit/mod.rs#L440)) formats enums
with `{:?}`, so it depends on `Debug` output being valid Rust; renaming a variant
silently changes emitted code. Keeping the two matrix aliases (§5.1) means that
function needs no change at all.

Merging four group emitters into one deletes `emit_committed_group`
([:465](../crates/akita-planner/src/emit/mod.rs#L465)),
`emit_setup_prefix` ([:513](../crates/akita-planner/src/emit/mod.rs#L513)),
`emit_group_opening_plan` ([:525](../crates/akita-planner/src/emit/mod.rs#L525)),
`emit_open_matrix` ([:477](../crates/akita-planner/src/emit/mod.rs#L477)) and
`emit_partition` ([:484](../crates/akita-planner/src/emit/mod.rs#L484)).

Identity effects, precisely:

- `key_digest` reads `CommittedGroupProfile` bytes for precommitted descriptors
  and setup prefixes, which §10.2 preserves. Dropping the stored prefix
  `opening` plan **does** change `entries_key_digest`, because it hashes
  `prefix.opening.canonical_descriptor_bytes()`
  ([catalog_identity.rs:600](../crates/akita-schedules/src/catalog_identity.rs#L600)).
  Sequence that with the step-5 epoch bump so the digest moves exactly once.
- `protocol_epoch` moves from `1` to `2` at the byte break, which invalidates
  all 13 committed tables at once. Regenerate them in that same step.
- Catalog sort order is unchanged, because it is keyed on profile bytes.

## 12. Removing `LevelParamsLike`

### 12.1 Why it exists, and why the fold list removes it

`LevelParamsLike` bridges `CommittedGroupParams` and `PrecommittedLevelParams`.
Its purpose is one method
([params.rs:908](../crates/akita-types/src/layout/params.rs#L908)):

```rust
pub fn group_params<'a>(&'a self, opening_batch: &OpeningClaimsLayout, group_index: usize)
    -> Result<&'a dyn LevelParamsLike, AkitaError>
```

This says "the final group is the fold itself; any other group is a
`PrecommittedLevelParams`; the caller must not care which." That is the
erasure point. About 14 non-test signatures name the trait directly, **four** of
them as `&dyn LevelParamsLike` behind a struct field:
[tail_segments.rs:69](../crates/akita-types/src/proof/tail_segments.rs#L69),
[fold_grind.rs:134](../crates/akita-prover/src/protocol/fold_grind.rs#L134)
(`FoldGrindGroup`, whose sibling `group` field is now generic over `G`),
[coeffs.rs:23](../crates/akita-prover/src/protocol/ring_switch/coeffs.rs#L23),
and [schedule.rs:587](../crates/akita-types/src/schedule.rs#L587), which **is**
`OpeningExecutionGroup`'s field. Draft 3 said five and listed `schedule.rs:743`
and `OpeningExecutionGroup` as separate entries; they are the same field.

After §5.4, `groups[i]` is a concrete `GroupParams`. The trait object, the
`opening_batch` argument, and the index remapping all disappear.
`group_params(i)` becomes `groups.get(i)`. One `&dyn` field becomes a
`&GroupParams`, which also removes the `!Sync` obstacle noted at
[relation_weights.rs:697](../crates/akita-prover/src/protocol/ring_switch/relation_weights.rs#L424).

### 12.2 Method destinations

Call-site counts are approximate: they are name-based and include inherent
methods of the same name on other types. Use them for effort sizing, not as a
contract.

Draft 3 gave these counts without saying how they were produced, so nobody could
tell drift from a different counting rule. They are reproducible, on `8e552d2ac`,
with:

```bash
grep -rno "\.<name>(" crates/ --include='*.rs' \
  | grep -v "/generated/\|/tests/\|/test/\|tests\.rs\|_tests\.rs" | wc -l
```

That is, method-call syntax only, excluding the generated tables and every test
module. Keep this rule when the numbers are next refreshed.

| Trait method | Call sites | Destination |
|---|---|---|
| `num_live_blocks` | 129 | `group.profile.blocks.live_blocks` |
| `num_positions_per_block` | 49 | `group.profile.blocks.positions_per_block` |
| `num_digits_fold` | 35 | `group.opening.num_digits_fold` |
| `num_digits_inner` | 26 | `group.profile.inner.digits.num_digits` |
| `num_digits_open` | 22 | `group.opening.num_digits_open` |
| `a_rows_len` | 20 | `group.profile.inner.matrix.output_rank()` |
| `log_basis_open` | 20 | `group.opening.log_basis_open` |
| `inner_commit_matrix_params` | 18 | `&group.profile.inner.matrix` |
| `opening_method` | 17 | `group.opening.opening_method` |
| `position_index_bits` / `block_index_bits` | 16 / 15 | `BlockGeometry` methods — one definition instead of three copies |
| `num_digits_outer` | 13 | `group.profile.outer.digits.num_digits` |
| `log_basis_inner` | 13 | `group.profile.inner.digits.log_basis` |
| `fold_challenge_config` | 9 | `group.opening.fold_challenge_config` |
| `log_basis_outer` | 8 | `group.profile.outer.digits.log_basis` |
| `logical_b_rows_len` | 7 | `group.profile.outer_slice_count.logical_output_rows(...)` |
| `outer_slice_count` | 6 | `group.profile.outer_slice_count` |
| `a_col_len` | 5 | `group.profile.inner.matrix.input_width()` |
| `b_rows_len` | 5 | `group.profile.outer.matrix.output_rank()` |
| `num_live_ring_elements_per_claim` | 3 (**not zero**, see §3) | `group.profile.blocks.live_ring_elements_per_claim` |
| `source_encoding` | 3 | `fold.source_encoding_of(index)` (§7.2) |
| `b_col_len` | 2 | `group.profile.outer.matrix.input_width()` |

Eleven of the 21 rows are unchanged from draft 3. Five grew: `num_live_blocks`
119→129, `num_digits_fold` 28→35, `num_digits_inner` 22→26,
`num_positions_per_block` 46→49, and `log_basis_open` 19→20. Five fell by one or
two: `fold_challenge_config` 11→9, `num_digits_open` 23→22,
`inner_commit_matrix_params` 19→18, `opening_method` 18→17, and
`block_index_bits` 16→15.

The net is upward, and it is concentrated in the four largest rows — the block
geometry and the digit counts. That is the same direction the rest of this
document reports: the duplication this plan removes is still accumulating, and
the accessors that §5 turns into plain field reads are the ones growing fastest.
Adding the generated tables and tests back roughly doubles the two largest rows,
which is why the rule above excludes them.

### 12.3 Also deleted with it

- `CommittedGroupParams::group_params`, `group_params_geometry`,
  `validate_opening_batch`, `validate_opening_batch_geometry`,
  `precommitted_group_count`, `precommitted_group_params`,
  `precommitted_group_iter`, `has_precommitted_groups`, `group_role_dims`,
  `group_role_dims_geometry`. All become list operations.
- The 10 single-method wrapper functions in
  [setup_contribution/plan/types.rs:140-238](../crates/akita-types/src/setup_contribution/plan/types.rs#L140-L238).
  Each re-exposes exactly one trait method, and `n_a_for`/`n_a` are
  byte-identical bodies. The two composites `t_vector_width` and `d_active_cols`
  stay: they compute something.
- `CommitInnerPlan::from_profile`
  ([operation_plans.rs:60](../crates/akita-prover/src/compute/operation_plans.rs#L60)).
  `from_level` and `from_profile` have identical bodies over different
  receivers. One `from_profile(&CommittedGroupProfile)` serves both, because all
  four of its fields live in the profile. `CommitInnerPlan` itself stays: it is a
  kernel shape plan on a public extension boundary, with out-of-tree
  implementations in
  [commitment_contract.rs](../crates/akita-pcs/tests/commitment_contract.rs).

### 12.4 The four borrowed views

Draft 2 listed one borrowed view. There are now four, and each has a different
disposition.

| Type | Fields | Construction sites | Disposition |
|---|---|---|---|
| `CommitmentGeometry<'a>` (`akita-prover/src/api/commitment.rs:125`) | 10, of which 9 are a strict subset of `CommittedGroupProfile` | **2** (`commitment.rs:236` and `:635`, both via the `From<&CommittedGroupParams>` impl at `:138`) | **Delete.** Draft 2 said its consumers fabricate a fake `opening` dimension. That is no longer true — it has no opening field at all. Its 9 payload fields are exactly a `CommittedGroupProfile` minus `version`, `group`, and `N`. Replace it with `&CommittedGroupProfile` plus the `context: &'static str` label as a separate argument. Draft 3 said one site at `:545`; there are two, and both go through the `From` impl, so deleting the impl catches both. |
| `FoldScheduleDescriptorStep<'a>` (`schedule.rs:327`) | 4 | planner candidate encoding, consumed in `schedule/descriptor.rs` (§0.8) | **Delete.** It exists so the planner can encode a candidate without building a `FoldSchedule`. Its `params`, `payload_mode`, and two lengths are all fields of `FoldParams` after §5.4, so the planner encodes a `&FoldParams` directly. |
| `TerminalFoldDescriptor<'a>` (`schedule.rs:336`) | 4 | terminal descriptor encoding, now in `schedule/descriptor.rs` (§0.8) | **Delete.** Its four borrowed fields are exactly `TerminalFoldParams` after the §5.5 merge. |
| `OpeningExecutionGroup<'a>` (`schedule.rs:586`) | 2 | 3 | **Delete.** It pairs a `&dyn LevelParamsLike` with an expected source encoding. After §5.4 the loop iterates `&GroupParams` and asks the fold for the encoding (§7.2). |

### 12.5 The cost

Sites that read the final group get one more hop:
`params.num_positions_per_block` becomes
`fold.final_group()?.profile.blocks.positions_per_block`. This is the main cost
of the change and it touches roughly 150 lines. Bind the group once per
function and read fields from it. Many sites are already group-indexed and get
shorter, not longer.

## 13. Derives, size, and allocation

**`Copy`.** Every type that a `static` table needs stays `Copy`
(§6). `GroupParams` gains `Copy`, which today's `PrecommittedLevelParams` lacks
only because it derives `Clone` alone; both of its fields are `Copy` already.
`FoldParams` and `TerminalFoldParams` are not `Copy` because they own a `Vec`,
exactly as `CommittedGroupParams` and today's `TerminalFoldParams` already are.
**No type loses `Copy`.** The original review correctly warned that draft 1
would lose it on generated parents; this draft does not, because the wire
profile stays a `Copy` leaf.

**`Hash` and `Eq`.** `AkitaScheduleLookupKey` keeps
`Vec<CommittedGroupProfile>`, and `CommittedGroupProfile` remains all-scalar, so
its derived `Hash` still compiles. `GroupParams` cannot derive `Hash`, because
`SparseChallengeConfig` has none — and it does not need to, since it is not in
the lookup key. `SetupPrefixSlotId` keeps its hand-written `Hash` and `Ord` over
`CommittedGroupProfile` descriptor bytes, unchanged; only its construction moves
(§5.3). Draft 1's problem was that it put a `Vec<PrecommittedLevelParams>`
inside the hashed type.

Separately: the `Hash` derives on `AkitaScheduleLookupKey` and
`CommittedGroupProfile` appear to be **dead**. No map or set is keyed on them,
and lookup uses `partition_point` ordering, not hashing. Removing them is a
small independent cleanup, not part of this plan.

`GroupParams` derives `PartialEq`. Today `PrecommittedLevelParams` derives it too,
so this is a no-op.

**Size.** Re-measured on `8e552d2ac`, debug profile, aarch64-darwin, with
`core::mem::size_of`. Every value in the "Today" column is confirmed: it is
unchanged from draft 3's measurement on `f37d07089`, which is expected, because
every struct declaration in this table is byte-identical across the two bases.

| Type | Today | Expected after | Why |
|---|---|---|---|
| `SisTableKey` | 64 | 64 | untouched |
| `InnerCommitMatrixParams` | 128 | 128 | untouched (holds `InnerCommitSecurityRoute`) |
| `OuterCommitMatrixParams` / `OpenCommitMatrixParams` | 80 / 80 | 80 | generic adds a ZST |
| `CommittedGroupProfile` | 288 | 288 | same fields, regrouped only |
| `GroupOpeningPlan` | 56 | 56 | untouched |
| `PrecommittedLevelParams` → `GroupParams` | 352 | ~360 | plus `Option<usize>`; `ScheduledSetupPrefix` (368) is deleted |
| `CommittedGroupParams` → `FoldParams` | 816 | ~160 | the profile and two of three matrices move to the groups |
| `RootFoldParams` + `RootFoldStep` | 960 + 976 | — | merged into `FoldParams` |
| `RecursiveFoldParams` + `RecursiveFoldStep` | 1296 + 1312 | — | merged into `FoldParams` |
| `RootFinalGroupParams` | 816 | — | deleted (one-field wrapper) |
| `RootPrecommittedGroupParams` | 640 | — | deleted |
| `TerminalCommittedGroupParams` + `TerminalFoldParams` + `TerminalFoldStep` | 176 + 240 + 256 | ~264 | three merge into one |
| `FoldSchedule` | 1264 | ~440 | wrappers gone, one `FoldParams` inline |
| `GeneratedRootPrecommittedGroup` / `GeneratedRootFinalGroup` / `GeneratedSetupPrefixInput` / `GeneratedCommittedGroup` | 320 / 88 / 352 / 48 | ~344 | one `GeneratedGroup`; adds `Option<u64>`, drops the stored `GroupOpeningPlan` |
| `GeneratedRootFold` / `GeneratedRecursiveFold` | 120 / 480 | ~80 | one `GeneratedFold`; the group array moves behind a slice |
| `GeneratedFoldScheduleEntry` | 288 | ~200 | one fold type |
| `GeneratedInnerCommitMatrix` / `Outer` / `Open` | 8 / 8 / 8 | 8 | one `GeneratedMatrix` |
| `SetupPrefixSlotId` | 304 | 304 | kept; derived, not stored (§0.5) |
| `GeneratedBlockGeometry` | 24 | 24 | kept (§11.1) |
| `GeneratedTerminalFold` | 144 | 144 | kept (§11.1) |
| `GeneratedWitnessPartition` | 8 | — | deleted (§11.1) |

The last four rows are new in draft 4; draft 3 measured the other 25 but omitted
these, and `SetupPrefixSlotId` in particular is worth stating because §5.3 makes
it derived rather than stored, which removes 304 bytes per prefix from the
schedule without changing the type.

No `Box` and no `Arc` appear anywhere, so the allocation count is unchanged.
`FoldParams` allocates one `Vec` where `CommittedGroupParams` already allocated
one. §14 keeps the measured table a gate, because the "after" column is still
estimated — only the "Today" column is measured.

`RecursiveFoldStep` at 1312 bytes is the largest parameter value in the tree and
it is stored in a `Vec`, so `FoldSchedule` currently carries about 1.3 KiB per
recursive level. The estimate above is the main non-hygiene payoff of the change.

**One allocation worth noting.** `precommitted_group_sort_key` builds a fresh
`Vec<u8>` per profile per comparison inside a comparator
([generated/mod.rs:305](../crates/akita-schedules/src/generated/mod.rs#L305)).
`SetupPrefixSlotId::cmp` and `::hash` do the same
([setup_prefix.rs:132](../crates/akita-types/src/proof/setup_prefix.rs#L132),
[:148](../crates/akita-types/src/proof/setup_prefix.rs#L148)). Tables hold at
most a few dozen entries and the registries at most `MAX_SETUP_PREFIX_SLOTS`, so
neither is hot today. This plan does not change either. They are recorded here so
they are not mistaken for new cost.

## 14. Tests

A single known-schedule byte check is not enough. Build the harness in step 1,
before any type changes, and re-run it after every step.

**Golden byte fixtures**, committed to the repository:

- Descriptor bytes for every entry of all 13 generated catalogs, at every level:
  profile, opening plan, group, fold, schedule.
- `ScheduleRowDigest` for every catalog row.
- `key_digest` and the full `CatalogIdentityExpectation` per family.
- Coverage of single-group roots, multi-group roots, setup-prefix schedules,
  chunked and non-chunked schedules, recursive folds, terminal folds, **B-sliced
  groups (`outer_slice_count > 1`)**, **subring-packing folds**, **terminal
  L2 routes**, and **bounded committed dense sources (`fp128_dense_bounded`,
  feature `fp128-dense-bounded`)**. The last four did not exist when draft 2 was
  written and are the paths most likely to break silently. The bounded family also
  covers the grouped `fp128_onehot` row that pairs a bounded dense precommit with
  a one-hot final group, which is the only row where two groups in one fold carry
  different A digit depths.
- `CommittedGroup<F>` wire fixtures, serialized and deserialized, for each of
  the above.

**Byte-stability assertions.** Steps 1 to 4 must produce **zero** fixture diffs.
Step 5 changes the fixtures above the profile and must leave the profile
fixtures and catalog sort order **unchanged**. Assert that explicitly, per
family. `key_digest` moves exactly once, in step 6, when the stored prefix
opening plan is dropped (§11.3); assert the step-5 value and the step-6 value
separately.

**Rejection tests.** Every one must fail closed:

- profile version other than 4, in both directions;
- unknown modulus, policy, or role tag;
- a matrix role that does not match its slot;
- a rank, width, or L-infinity bound that the SIS table does not certify;
- an L2 A route where the boundary requires `Linf`: a root final group, a
  precommitted group, and a terminal `Linf` cap;
- an L2 terminal route that also carries a scheduled `Linf` cap;
- geometry violations: `M` not a power of two, `B != ceil(N / M)`,
  `N · d_a != 2^num_vars`;
- an inadmissible `outer_slice_count` for its block count and payload mode;
- an opening basis below the frozen outer basis;
- a packing group whose `challenge_subring_dimension` does not validate the
  fold challenge family;
- a packing group carrying a `TensorSubfieldProjection` source encoding;
- a fold whose groups disagree about the opening-method family;
- a nonterminal level that uses the wrong family for its absolute level;
- **a standalone or precommitted profile with a non-canonical source encoding**
  (the check relocated in §7.2);
- `setup_natural_len` of zero, above `n_prefix`, or not a multiple of
  `d_setup()`;
- **a root fold that names an incoming setup prefix** (the §8 trade);
- more than one prefix group in a fold, or a prefix group at an index other
  than 0;
- an empty `groups` list;
- a coefficient count above `MAX_UNTRUSTED_COMMITMENT_COEFFICIENTS`;
- a terminal matrix that retains less than half of the unconstrained target;
- an instance descriptor with `protocol_epoch` of 1 after the bump;
- **an A digit depth that is not `ceil(log_commit_bound / log_basis_inner)` for
  its level**, in both directions, on a bounded and on a full-width row (§0.9).
  This one is easy to lose: it lives in `akita-schedules`, not in the type-level
  `validate` this plan consolidates;
- **a commitment whose source coefficients exceed the declared
  `log_commit_bound`**, which `commit` rejects since #407. State the test against
  `decompose_centering_threshold` and
  `checked_balanced_digit_representable_bounds`, not against a naive range check —
  the saturating reach is a conservative lower bound and the sign rule is not
  centering at `q/2`, so a naive check rejects valid full-width dense commitments.

**Measurement gate.** A committed test printing `size_of` for every type in §5
and every type in §11, before and after. Compare against §13. Also record
generated-table byte size and the `partition_point` lookup cost. §13's "Today"
column was measured on `f37d07089`; every struct declaration in it is
byte-identical on `8e552d2ac`, but re-measure on `main` before trusting the
"after" estimates.

**Existing suites.** `cargo test -p akita-config --features all-schedules
--test generated_tables` and the full CI matrix in `.github/workflows/ci.yml`
must pass at every step, including the table drift gate. Under #403 (§0.10) the
matrix now selects transcript backends explicitly and runs pure Blake and Keccak
semantic suites; the byte break in step 5 must be validated under each, not only
under the default backend.

## 15. Corrections

### 15.1 To draft 3, from `main`

Draft 3 was accurate against `f37d07089`. These claims are now false. Every one
is already fixed in place above; they are listed here so a reviewer who read
draft 3 knows what moved.

**Counts.**

| Draft 3 said | Correct on `8e552d2ac` |
|---|---|
| 12 generated tables / catalogs | **13** — `fp128_dense_bounded` was added by #407 |
| 29 `CommittedGroupProfile` struct literals | **30** |
| 58 `new_unchecked` calls in `static` position | **60** |
| `num_live_ring_elements_per_claim` has "four" non-test call sites (§3) | **3**, which is what §12.2 already said |
| 9 single-method wrappers in `plan/types.rs:140-238` | **10** |
| `CommitmentGeometry` has 1 construction site | **2**, both through the `From` impl |
| `validate_frozen_precommit` is the wire entry point at `proof/commitment.rs:459` | two sites there, `:207` and `:459` |
| five `&dyn LevelParamsLike` struct fields | **4**; `schedule.rs:743` and `OpeningExecutionGroup` were the same field, counted twice |
| `emit/mod.rs` is 1114 lines | **993** |
| `expand.rs` has "about 914" non-test lines | **935**; the three function lengths were exact and still are, summing to 652 |
| §12.2's 21 call-site counts | 11 unchanged, 5 grew, 5 fell — see §12.2, which now records the counting rule so drift is distinguishable from a different method |
| "24 types and one trait" | 24 inventory **rows**, which is the number the plan commits to — see §2.1 |

**Locations.** `emit/mod.rs` was rewritten, so all six `emit_*` references moved
by about +90 lines. `audit.rs` shifted uniformly **+14** after line 216, where
#407 inserted the A-depth audit. `precommitted.rs` shifted **+21** after line 100,
where `OpeningFamily` and `physical_coefficient_width` were inserted, except
`opening_d_segment_width`, which moved +30. All descriptor **encoders** left
`schedule.rs` for `schedule/descriptor.rs` (§0.8), which pulled
`OpeningExecutionGroup` from `:742` to `:586` and
`validate_level_opening_execution` from `:747` to `:591`; the descriptor **view
struct declarations** stayed at `:327` and `:336`.

**Substance.** `SCHEDULE_ROW_DOMAIN_V2` is in `akita-types`, not `akita-config`.
The `FoldSchedule` descriptor byte is no longer in `schedule.rs`. §9 was missing
the #407 A-depth audit entirely, and §5.1 implied `GadgetDigits::validate` could
own the A depth, which it cannot — that check is level-aware.

**Unchanged and re-verified:** all 25 sizes in §13, **re-measured** on
`8e552d2ac` rather than carried over, plus 4 new rows; the `LevelParamsLike`
method count (22); `expand.rs` at 1027 lines with its four expansion entry points
at `:69`, `:159`, `:513`, `:755` and all four function lengths exact (57, 354,
241, 159); the 15 generated mirror types; the 21 parameter-surface byte encoders;
all of §10.1's three encoder locations; `CommittedGroupProfile` at
`profiles.rs:84` with 12 flat fields; `CommittedGroupParams` at `params.rs:87`;
the two-field byte-identical generated matrix structs (§11.1); the
`typed-schedule-topology-cutover.md` references at `:100` and `:105`; and the
whole of `catalog_identity.rs`.

### 15.2 To draft 2, from the packing line

Draft 2 was accurate against `74c17ba4f`. Six of its claims are now false.

- **"18 types and one trait", "six mirror audits", "3 macro copies", "11 byte
  encoders", "18 trait methods".** The real numbers on `8e552d2ac` are 24 types,
  10 mirror audits, 2 macro copies, 21 parameter-surface byte encoders, and 22
  trait methods.
- **§5.1's `CommitMatrix<R>` over three roles.** The A role left the macro when
  it gained `InnerCommitSecurityRoute` (§0.2). Two roles merge, not three.
- **"delete `SetupPrefixSlotId`".** Wrong type. `ScheduledSetupPrefix` is the
  schedule-side type that merges away; `SetupPrefixSlotId` is a registry key the
  topology spec requires (§0.5).
- **"`num_live_ring_elements_per_claim` has zero call sites; delete the
  accessor."** The packing work gave it four (§3). Likewise `b_col_len`.
- **"`CommitmentGeometry` consumers fabricate a fake `opening` dimension."** It
  has no opening field, one construction site, and 10 fields (§12.4).
- **"`key_digest` reads only `CommittedGroupProfile` bytes."** It reads generated
  row content, including profile bytes and the stored prefix opening plan
  (§10.2, §11.3). Draft 2 also cited the wrong function and line.
- **"`CommittedGroupProfile::VERSION` is 2", "240 bytes", "10 flat fields", "24
  profile literals", "48 `new_unchecked` calls", "`SETUP_PREFIX_CONTENT_TAG` is
  `b"SPF1"`".** Now 4, 288, 12, 30, 60, and `b"SPF4"`.
- **"`validate_frozen_precommit` is deleted."** It is the wire deserialize's
  entry point and must stay (§9).

Every `file:line` reference in draft 2 moved. All of them are restated here
against `8e552d2ac`.

### 15.3 To draft 1 and the original review

Draft 1:

- It listed 11 generated mirrors and named 12. The real count is **15**.
- It said `CommitmentGeometry` gives one shape to `CommittedGroupParams` and to
  a profile. It converts from `CommittedGroupParams` only.
  `CommitInnerPlan` has the two constructors.
- It said its three principal types differ only in role count. They differ in
  trust boundary: public versioned wire data, executable fold state, and the
  output of a security-sensitive projection. That is why this draft keeps them
  as distinct types.
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

### 15.4 Deliberately out of scope

Three overlaps surfaced during this review and are **not** part of this plan.
Each deserves its own change.

- **Three containers over ordered profiles.** `AkitaScheduleLookupKey`
  (`profiles.rs:314`), `CommittedGroupBatchProfile` (`profiles.rs:530`), and
  `PrecommittedGroupProfiles` (`profiles.rs:328`) all hold ordered
  `CommittedGroupProfile` values with slightly different non-emptiness and
  final-group conventions. Consolidating them touches the catalog lookup path,
  which this plan deliberately leaves byte-stable.
- **`source_encoding` on the profile.** See §7.2. It is the more honest design
  and it costs a profile version bump.
- **The dead `Hash` derives.** See §13.

## 16. Prerequisite: resolve the certified-response divergence

Two functions compute the certified terminal response cap and disagree. Both
still exist on `8e552d2ac`, and the argument list of one of them changed in a way
that makes the divergence easier to trigger, not harder.

| | `certified_response_linf_cap` | `terminal_response_linf_limit_for_params` |
|---|---|---|
| Location | [schedule.rs:211](../crates/akita-types/src/schedule.rs#L211) | [params.rs:405](../crates/akita-types/src/layout/params.rs#L405) |
| Receiver | `TerminalCommittedGroupParams` | `CommittedGroupParams` — a *fold*, not a group |
| `i16::MAX` clamp | applied (`:235`) | **not applied** |
| A-role check | implicit: the field is typed `InnerCommitMatrixParams` | explicit `table_key.role` check |
| L2 route | rejected explicitly (`:215-222`) | rejected indirectly, via `sis_table_key()` returning `None` |
| Challenge config used | **passed in by the caller** | `self.fold_challenge_config` — the *receiver fold's*, although a per-group `params` is the argument |
| Acceptance path | `validate_terminal_linf_cap` (`schedule.rs:240`) | `z_cap > security_cap` rejection at [tail_segments.rs:754](../crates/akita-types/src/proof/tail_segments.rs#L754) and [:928](../crates/akita-types/src/proof/tail_segments.rs#L928) |

Both results gate proof acceptance, and both are compared against the same
`z_admission_linf_cap`. This is the split-brain that `AGENTS.md` forbids.

The two call sites confirm the mismatch is live: `lp` is the fold's
`CommittedGroupParams` and `params` / `group.params` is the per-group view, so a
multi-group terminal fold computes each group's security cap from the fold's
challenge config.

**Do this first, in its own PR.** It changes acceptance behavior, so it must not
hide inside a mechanical refactor. Decide three things:

1. whether the `i16::MAX` clamp is a kernel representation limit that both
   paths need — the comment at `schedule.rs:233-234` says the terminal NTT
   kernels consume signed `i16`, which suggests yes;
2. whether the per-group challenge config is the correct input — `GroupOpeningPlan`
   already carries one per group, so the data is available;
3. whether the explicit A-role check adds anything over the typed field.

Then keep one function. After the consolidation, `GroupParams` carries the
per-group config, `TerminalFoldParams` carries its own, and
`certified_response_linf_cap` takes no config argument (§5.5), so the divergence
becomes impossible to re-introduce.

## 17. Order of work

Steps 1 to 4 change no bytes and no call sites. Step 5 is the cutover.

| Step | Work | Bytes | Tables |
|---|---|---|---|
| 0 | §16 prerequisite: one certified-response function | none | none |
| 1 | Golden fixture harness (§14), including the B-slicing, packing, L2, and bounded-dense cases, across all 13 catalogs and each explicitly selected transcript backend | none | none |
| 2 | `BlockGeometry` and `GadgetDigits`, with `validate` and the index-bit methods. Use the atomic geometry encoder where the triple is already contiguous; keep field-level digit encoding in the two encoders that interleave. | **identical** | none |
| 3 | `LinfCommitMatrix<R>` with a sealed `LinfMatrixRole`; delete the 2-role macro; keep the two aliases. `InnerCommitMatrixParams` untouched. | **identical** | none |
| 4 | `RoleParams<M>`; restructure `CommittedGroupProfile` to `version, group, blocks, outer_slice_count, inner, outer`. Update the emitter for nesting. | **identical**, verified by step 1 | regenerate; `key_digest` unchanged |
| 5a | `GroupParams` from `PrecommittedLevelParams`: rename `layout` to `profile`, add `setup_natural_len`, add `slot_id()`. Absorb `ScheduledSetupPrefix`. Derive `SetupPrefixSlotId` instead of storing it. | break | regenerate |
| 5b | `FoldParams` with the uniform `groups` list; D matrix and `source_encoding` to the fold; merge `RootFoldParams`, `RecursiveFoldParams`, `RootFinalGroupParams`, `RootPrecommittedGroupParams`, `RootFoldStep`, `RecursiveFoldStep`; delete `WitnessPartition`; delete the 8 schedule-side mirror audits and the `validate_structure` mirror check; fold the per-fold parts of `validate_level_opening_execution` into `FoldParams::validate` and settle the prefix ordering (§7.1); add the 3 new checks. Rewrite the two fold encoders in `schedule/descriptor.rs` (§0.8) — they are all in one file now. Keep the #407 A-depth audit (§9). Bump the §10.2 constants here. | break | regenerate |
| 5c | Delete `LevelParamsLike`, `group_params` and its geometry twin, the 10 wrappers, `CommitInnerPlan::from_profile`, the `From<&CommittedGroupParams> for CommitmentGeometry` impl, and all four borrowed views (§12.4). | none | none |
| 5d | `TerminalFoldParams` merging 3 types; move the 6 fallible steps into `admit`; drop the config argument from `certified_response_linf_cap` and `validate_terminal_linf_cap`. | break | regenerate |
| 6 | Generated schema 15 → 8; drop the stored prefix `GroupOpeningPlan` and its 3 audits; one `expand_group`; update the emitter and delete 5 `emit_*` helpers. | none beyond step 5 | regenerate; `key_digest` moves once |

Steps 5a to 5d land together. Splitting them across PRs would need temporary
mirror fields with temporary audits, which is the thing being deleted. Split
them into four commits for review, and keep the tree compiling at each commit.

## 18. Result

| Item | Today (`8e552d2ac`) | After |
|---|---|---|
| Parameter types in scope | 24 | **10** |
| Traits over parameter types | 1 (`LevelParamsLike`, 22 methods) | **0** |
| Borrowed view types | 4 | **0** |
| L-infinity matrix implementations | 2 macro copies, ~340 lines | 1 generic + 2 aliases |
| Types holding block geometry | 5 | 1 |
| Types holding the shared D matrix | 3 | 1 |
| Types holding the fold challenge family | 4 | 1, per group |
| Types holding the setup prefix | 3 stored | 1 field + 1 derived key |
| Owners of `source_encoding` | 1 field + 1 hard-coded trait arm | 1 field |
| Mirror equality audits | 10 | **0** |
| Mirror pairs with no audit | 2 | 0 |
| Byte encoders on the parameter surface | 21 hand-written | 1 rule, mechanical |
| Group-parameter accessors on the fold | 10 plus a trait object | list operations |
| Generated mirror types | 15 | **8** |
| `expand.rs` duplicated paths | 3 copies, ~650 lines | 1 |
| `emit_*` helpers for group and matrix shapes | 8 | 3 |
| Recursive type cycles | 0 (already broken upstream) | 0, no `Box` |
| Kind enums or `Option` role tags | 0 | **0** |
| Types that lose `Copy` | n/a | **0** |
| `FoldSchedule` size (debug, 1 recursive level) | 1264 B | ~440 B |
| Divergent security calculations | 1 | 0 (§16) |
