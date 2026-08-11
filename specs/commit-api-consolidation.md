# Spec: Commit API consolidation

| Field         | Value                   |
|---------------|-------------------------|
| Author(s)     |                         |
| Created       | 2026-08-10              |
| Status        | implemented             |
| PR            |                         |
| Supersedes    |                         |
| Superseded-by |                         |
| Book-chapter  | usage/commitment-api.md |

## Summary

Before this cutover Akita exposed four scheme-level commitment methods:
`commit`, `batched_commit`, `commit_group`, and
`commit_final_group`. They all commit one polynomial group, but their names
also encode validation history and the group's future position in an opening
batch.

The implemented API has one public `commit` method and one `GroupContext`
input. Its prior-group axis distinguishes no prior groups from exact ordered
prior profiles; its parameter-source axis selects either the generated S/G
catalog or caller-supplied explicit parameters. The precommitted/multi-group
case therefore remains protocol-visible without owning a second P parameter
source or a separate public commitment function.

The role-preserving design discussion in the main body records the first
cutover and its baseline constraints. Appendix A is authoritative for the
subsequent S-only parameter-authority implementation. Appendix B is
authoritative for the final `GroupContext` and single-function public API.

This is a source-API breaking refactor. No compatibility wrappers, aliases, or
deprecated entry points are required. The freedom to break the Rust API does
not permit a protocol-output change: current generated parameter payloads and
all deterministic commitment/proof outputs are frozen by this spec.

Implementation began only after this spec was explicitly approved and coding
was requested.

## Hard constraints

The implementation is accepted only if all three constraints hold together.

1. **Protocol and efficiency parity.** For every currently valid end-to-end
   opening lifecycle, the same configuration, setup, inputs, and transcript
   produce byte-identical commitments and proofs with identical proof sizes.
   Generated schedule/profile payloads and setup envelopes do not change.
   Commit, prove, and verify performance must not regress. For every retained
   homogeneous role input, the corresponding old and new commit calls also
   have the same accept/reject result; no role gains a stricter setup-capacity
   check. A mixed-arity bundle accepted by a commit-only validator but rejected
   by the opening lifecycle is the sole declared accepted-input removal and is
   specified separately below.
2. **Single-source implementation.** Homogeneous-group validation, normalized
   commitment geometry/footprint validation, tensor projection, commitment
   arithmetic, and result assembly each have one implementation. The role
   branch selects the S/P/G parameter authority and the preserved admission
   scope (commit-only or full schedule); it does not duplicate the shared
   validators or arithmetic.
3. **Explicit architecture.** The public type names the group's protocol
   position, each generated domain has one stated owner, and row/profile
   resolution crosses a typed configuration boundary.

Backward compatibility of Rust symbols and call signatures is explicitly not
a constraint.

## Review resolution

The claims in `commit-api-consolidation-review.md`,
`commit-api-consolidation-review-2.md`, and
`commit-api-consolidation-review-3.md` were checked against the current code,
generated tables, and commit/prove flow. Review 2 identifies the decisive flaw
in the original proposal: a raw optional-priors argument cannot distinguish a
sole group from a non-final prior group without changing one lifecycle's
parameters. Review 3 validates that correction and identifies capacity,
cache-isolation, and migration-scope details tightened below.

The post-review rebase baseline is `origin/main` at `276143195`. That baseline
contains one relevant refinement after review 3: `RecursiveCommitmentConfig`
resolves empty-prior scalar keys in its recursive catalog instead of
delegating them to the wrapped configuration. Such an S row can carry a
recursive fold with `incoming_setup_prefix`, and setup-capacity enumeration
now includes that scalar recursive key. This does not change the S/P/G API
conclusion, but it makes active-configuration row preservation and the
separation between commit-only admission and full setup provisioning explicit
requirements below.

The verified resolutions are:

| Review claim | Resolution |
| --- | --- |
| A single canonical profile per configuration/layout conflicts with unchanged proofs and working grouped flows. | Accepted. S, P, and G remain intentionally distinct role-specific domains. The `S ⊆ P` and S/P-equality program is deleted. |
| Replanning each differing `S ∩ P` scalar row onto P changes existing generated outputs. | Accepted. No scalar, prior-profile, or grouped-row payload is replanned or regenerated. |
| Every `S ∩ P` profile differs. | Rejected as universal. For example, fp32 OneHot `(16, 2)` has equal S and P A/B fields. The fp128 overlaps cited by the review do differ. One differing overlap is sufficient to block context-free unification. |
| Different root geometry guarantees a different serialized proof size for every overlap. | Not established. Changing the exact profile and row identity already violates the requirement and can change commitment/proof values; exact length changes would require replanning and measurement. |
| The fp128 OneHot `S \ P` list in review 2 is exhaustive. | Qualified. It omits scalar layouts `(32, 1)`, `(32, 4)`, `(40, 1)`, `(44, 1)`, and `(50, 1)`. The final implementation must derive domain reports from generated artifacts rather than prose lists. |
| The P resolver is an uncached full scan. | Accepted. It validates identity and expands every profile on each lookup. Under the rejected P-for-all design this would regress the dominant path. Under the role-based design, sole traffic remains on S and P is neither enlarged nor moved onto that path. |
| P lookup must be cached for the consolidation to be viable. | Rejected as a blocker. The new `Prior` path has the same traffic and registry size as current `commit_group`. A separate lazy index is an optional measured optimization and must never make `Sole` materialize P. |
| `get_params_for_batched_commitment` and `committed_group_profile` were missing from the migration inventory. | Accepted. The former is deleted in favor of a row-preserving selector; the latter is renamed as the explicit prior-profile boundary. No aliases remain. |
| `commit_with_params` lacks a reason to survive. | Accepted. Its retained capability is catalog-independent arithmetic for custom backend/source conformance tests and deliberately hand-selected microbenchmark parameters. |
| Mixed-arity padding needs an explicit scope decision. | Accepted. One committed group is homogeneous. Different groups may have different arities. Padding-only helpers and re-exports with no other owner are removed. |
| “Zero catalog regeneration” is possible. | Qualified. No S/P/G payload is replanned or changed. Binding existing P bytes into catalog identity requires metadata-only generated identity updates; those identity constants are not row digests or transcript inputs. |
| `OpeningScheduleSelection` should remain outside the commit result. | Accepted. It identifies the complete ordered opening batch and is derived at batch assembly from exact committed profiles. |
| `Sole` must not be charged for a full opening schedule. | Accepted as a commit-call parity requirement. Current `Sole` and `Prior` require only their commit-only materialization after group-local seed checks; only `Final` performs the full schedule-fit check. The review's “A/B” shorthand omitted the B-output compression-chain prefix, and its proposed `Prior` row omitted current seed checks. |
| The previous spec's aggregate Final seed check preserves current behavior. | Rejected. Today Final applies seed limits only to its current bundle, then validates the exact key and full schedule footprint. Adding aggregate or prior-group seed checks during commit would create a new rejection. |
| Live P validation can share the entries-validation cache if P is in its key. | Rejected. Any recomputation from the live P slice on the shared S/G path is a regression. P uses a separate lazy validation domain; the entries-validation and materialized-row caches never traverse P. |
| The ordered batch-profile extractor and private `SelectedProverOpeningData` are both prerequisites for the commit cutover. | Qualified. Ordered extraction and selection before profile stripping are required for the cutover; private prover-input encapsulation is a separate parity-gated architecture slice, still required before this spec is marked implemented. To preserve allocation parity, extraction consumes the same owned `PriorGroupProfiles` value that Final commitment borrowed rather than recollecting profiles. |
| The resolved Sole-row profile needs a provenance equality gate. | Accepted. The requested input-derived layout, row profile, and old locally assembled profile must agree exactly, including bytes. |
| Sole currently resolves the schedule twice in every configuration. | Rejected as universal. The second selector call occurs for `EXT_DEGREE > 1`; degree-one configurations already short-circuit after one. The target performs one in both cases. |
| Recursive scalar S selection delegates to the wrapped configuration. | Rejected for the rebased baseline. The active recursive configuration now resolves empty-prior scalar keys in its own catalog, and those schedules may use carried setup prefixes. The target must preserve that exact row and must not infer delegation from an empty prior list. |

Review 2 suggests that an empty `Final` could be normalized to `Prior`. That
normalization is incorrect: `Prior` selects P, while an empty-prior final would
describe a sole batch and select S. This spec rejects empty `Final` explicitly
instead of silently changing the requested role.

## Current issue

The four scheme methods divide one commitment operation along two different
axes:

| Method | Current protocol role | Current parameter source | Current input rule | Return |
| --- | --- | --- | --- | --- |
| `commit` | sole group | scalar S row | nonempty; equal `num_vars` | committed group + hint |
| `batched_commit` | sole group | scalar S row | nonempty; layout uses maximum `num_vars` | committed group + hint |
| `commit_group` | non-final prior group | standalone P profile | nonempty; equal `num_vars` | committed group + hint |
| `commit_final_group` | final group after known priors | grouped G row | nonempty; layout uses maximum `num_vars` | committed group + hint + selection |

The split causes concrete problems:

- `commit` already accepts multiple polynomials, so “batched” does not name a
  distinct commitment operation.
- `commit_group` hides the important fact that it selects parameters for a
  non-final group in a future multi-group opening.
- `commit_final_group` encodes protocol position in a method name and consumes
  a `Vec` even though selection only reads the prior profiles.
- validation, tensor projection, parameter checks, arithmetic, and result
  assembly are repeated across entry points;
- `commit_with_validated_params` and
  `commit_with_validated_profile` duplicate the same kernel;
- `commit_with_params` and `batched_commit_with_params` differ only in their
  public validators; and
- the scheme-local `CommitmentWithHint` alias shadows a prover alias with a
  different tuple meaning.

The parameter-source distinction itself is not duplication. It is protocol
state and must remain visible in the consolidated API.

## Verified parameter domains

For each source configuration and generated policy, define:

- **S — sole-group rows.** Empty-prior generated schedule rows selected by the
  active configuration when the commitment is the only group in its opening
  batch. This includes adapter-local rows and any rows reached through an
  adapter's intentional delegation. Empty priors do not imply delegation to a
  wrapped configuration or imply that the selected schedule has no recursive
  folds.
- **P — prior-group profiles.** Generated standalone A/B profiles used for a
  non-final group that will be supplied as an exact prior profile to a later
  final selection.
- **G — grouped-final rows.** Generated schedule rows keyed by a final layout
  and the exact ordered profiles of all prior groups.

S, P, and G are different artifacts with different optimization contexts.
There is no required subset or equality relation between S and P.

For fp128 OneHot `(14, 1)`, the current values are:

| Role/source | N | M | B | `d_A` | `d_B` |
| --- | ---: | ---: | ---: | ---: | ---: |
| sole S row | 64 | 32 | 2 | 256 | 128 |
| prior P profile | 256 | 128 | 2 | 64 | 64 |
| G prior descriptor | 256 | 128 | 2 | 64 | 64 |

The P profile matches the descriptor embedded for that prior position in G,
not the S row. Standalone planning deliberately uses a suffix dimension and a
different objective from full schedule planning.

Coverage also differs. fp128 Dense `(15, 2)` exists in P without a scalar S
row and is used as a Dense prior in a OneHot grouped row. Conversely, several
high-arity layouts exist in S without a P profile. This is supported behavior,
not a catalog defect.

`CommittedGroupProfile` freezes the A/B commitment identity: layout, block
geometry, bases/digit counts, and A/B matrix identities. It does not own
`num_digits_fold`, opening/D geometry, witness partitions, or the consuming
row's recursive plan. Those remain row/configuration-owned.

A profile produced as a grouped final may later be supplied as a prior only if
an exact G row accepts its bytes. Cross-configuration prior descriptors are
also valid; the heterogeneous Dense-to-OneHot flow is the current example.

## Target API

### Public types and signature

The exact ordered prior metadata has one transient ownership type:

```rust
#[derive(Debug, Default)]
pub struct PriorGroupProfiles {
    profiles: Vec<CommittedGroupProfile>,
}

impl PriorGroupProfiles {
    pub fn from_profiles(
        profiles: Vec<CommittedGroupProfile>,
    ) -> Self;

    pub fn from_ordered_groups<'a, F, I>(groups: I) -> Self
    where
        F: FieldCore + 'a,
        I: IntoIterator<Item = &'a CommittedGroup<F>>,
        I::IntoIter: ExactSizeIterator;

    pub fn as_slice(&self) -> &[CommittedGroupProfile];
}
```

Its field is private and ordered. `from_profiles` consumes an existing vector
without cloning it; `from_ordered_groups` is the canonical extraction path
when committed groups are available. These ownership constructors do not
revalidate profiles: Final's configuration selector owns structural
key/profile validation exactly once. An empty value is valid as the prefix of
a sole batch but is rejected when used with `GroupPosition::Final`.

The consolidated role type is:

```rust
/// Position of this commitment in the opening batch that will contain it.
pub enum GroupPosition<'a> {
    /// The only group in its opening batch. Selects S.
    Sole,
    /// A non-final group in a multi-group batch. Selects P.
    Prior,
    /// The final group after these exact ordered prior profiles. Selects G.
    Final {
        prior_group_profiles: &'a PriorGroupProfiles,
    },
}
```

The one scheme method is:

```rust
pub fn commit<P, B>(
    setup: &AkitaProverSetup<Cfg::Field>,
    polys: &[P],
    stack: &UniformProverStack<'_, Cfg::Field, B>,
    position: GroupPosition<'_>,
) -> Result<CommitOutput<Cfg::Field>, AkitaError>;
```

The named result is:

```rust
pub struct CommitOutput<F: FieldCore> {
    pub committed_group: CommittedGroup<F>,
    pub hint: AkitaCommitmentHint<F>,
}
```

`GroupPosition` and `CommitOutput` are prover API types;
`PriorGroupProfiles` is an `akita-types` ownership type. The PCS crate
re-exports them with `AkitaCommitmentScheme`. None is serialized or added to
the transcript.

### Why the position is explicit

The originally proposed `Option<&[CommittedGroupProfile]>` has only two useful
states: no supplied profiles and one or more supplied profiles. The protocol
has three states. Both `Sole` and `Prior` have no earlier profiles, but they
must select different parameters on differing S/P overlaps.

Using `None` for both would force S/P canonicalization and change current
outputs. Giving `None` and `Some(&[])` different meanings would hide protocol
state in an empty-container convention. Adding a second Boolean or enum beside
the optional slice would allow contradictory combinations.

`GroupPosition` is the minimal tagged union. Metadata remains optional in
substance: only `Final` carries it. The field name
`prior_group_profiles` replaces the vague current name `precommitteds` across
the live public protocol types. With no source-compatibility requirement, the
cutover also renames `AkitaScheduleLookupKey::precommitteds` and
`CommittedGroupBatchProfile::precommitteds` to `prior_group_profiles`. These
are source-only field renames: ordering, canonical bytes, row digests, and wire
formats do not change.

Generated/internal catalog names such as `GeneratedPrecommittedProfile` and
`GeneratedScheduleTable::precommitted_profiles` retain “precommitted” as the
name of the standalone P artifact, not a public commit-position API. Schedule
internals may likewise use “precommitted group” for an already materialized
root participant. The public caller-facing term is consistently “prior group.”

### Role semantics

| New position | Replaces | Required behavior |
| --- | --- | --- |
| `Sole` | `commit`; homogeneous uses of `batched_commit` | select the same empty-prior S row and current final-group parameters |
| `Prior` | `commit_group` | select the same exact P profile |
| `Final { prior_group_profiles }` | `commit_final_group` | select the same exact G row using the supplied profiles in order |

`Final` requires a nonempty profile sequence and rejects an empty sequence with
`AkitaError::InvalidInput` before schedule resolution. Callers use `Sole` for
a one-group batch. The ownership object is borrowed, not retained or mutated,
during commitment. Its vector is copied at most once when the existing owned
`AkitaScheduleLookupKey` representation is required, then the original
allocation is consumed by batch assembly rather than rebuilt.

The caller must supply every prior profile in opening-claim and transcript
order. The API cannot detect a group that was never supplied; it rejects when
the exact sequence has no generated G row.

### One execution pipeline

Only parameter resolution branches by position:

```text
commit(setup, polys, stack, position)
    |
    +-- validate one homogeneous polynomial-group layout
    |
    +-- resolve exact role source
    |      Sole  -> one S ResolvedScheduleRow
    |      Prior -> one P CommittedGroupProfile
    |      Final -> one G ResolvedScheduleRow
    |
    +-- normalize to one validated internal A/B geometry
    |
    +-- decide and, if required, run one tensor projection
    |
    +-- run one commitment kernel
    |
    +-- assemble one CommitOutput from the exact selected profile
```

The shared pipeline applies the role's existing admission and matrix-footprint
contract before tensor or commitment work. It derives tensor projection and
commitment geometry from the same selected source, so a transform cannot use
one row while the kernel uses another.

#### `Sole`

1. Validate the homogeneous group, apply the current group-local setup-seed
   bounds, and derive its `OpeningClaimsLayout`.
2. Select the exact empty-prior S row once through the active
   configuration-owned row-preserving boundary. In particular, a recursive
   adapter must retain its scalar recursive-catalog row rather than delegating
   to the wrapped configuration.
3. Apply the current commit-parameter structural validation, including its D
   metadata checks, but charge only the exact root commit-only footprint: A,
   B, and B-output compression setup. Do not call the full schedule-fit
   admission or charge D/recursive/setup-prefix/terminal storage, even when
   the selected recursive S schedule carries an incoming setup prefix.
4. Use the row's root-final A/B parameters for both tensor selection and the
   commitment kernel.
5. Require the requested layout to equal
   `row.profiles().final_group.group`, and require the complete row profile to
   equal `CommittedGroupProfile::from_params(requested_layout,
   &row.schedule().root.params.final_group.commitment)`. Return that exact
   profile with the resulting commitment and hint.

This replaces the current selection through
`get_params_for_batched_commitment` and the conditional second selection in
`root_transform_ring_dim`. The old path performs two selector calls when
`EXT_DEGREE > 1`, even if projection is ultimately disabled, but already
performs only one when `EXT_DEGREE == 1`. The target performs one in both
cases. Removing the explicit degree-one early return is byte-safe because
`root_tensor_projection_enabled` itself requires an extension width greater
than one.

#### `Prior`

1. Validate the homogeneous group, apply the current group-local setup-seed
   bounds, and derive its `PolynomialGroupLayout`.
2. Resolve the same exact P descriptor that current `commit_group` resolves.
3. Validate the frozen profile and its exact commit-only setup footprint: A,
   B, and B-output compression. Do not perform full-schedule admission or
   charge D/recursive/terminal storage.
4. Use that profile's A-role dimension for tensor selection and the same
   profile for the commitment kernel.
5. Return that exact profile with the resulting commitment and hint.

No S row is required. P-only layouts such as fp128 Dense `(15, 2)` remain
valid prior commitments.

#### `Final`

1. Reject an empty prior-profile ownership object; validate the homogeneous
   current group with the same current-group-local setup-seed bounds as today.
2. Construct one `AkitaScheduleLookupKey` from the current layout and an
   order-preserving copy of `prior_group_profiles.as_slice()`.
3. Select one exact G `ResolvedScheduleRow` and retain it. The configuration
   selector owns the key validation, including every prior profile, exactly
   once and guarantees that the audited row matches the requested final
   layout and exact order. It does not add an aggregate-polynomial or
   prior-group setup-seed check that current `commit_final_group` lacks.
4. Run the same full `ensure_prover_schedule_fits_setup` admission as today,
   then apply the current root commit-parameter/A/B checks.
5. Use `row.schedule().root.params.final_group.commitment` for tensor selection
   and commitment.
6. Return `row.profiles().final_group` with the resulting commitment and hint.

Request-key/profile equality and the schedule/profile audit live once in the
configuration selector that constructs `ResolvedScheduleRow`; the prover does
not duplicate those checks after receiving the validated handle.

The current group's tensor decision uses its own arity and A-role dimension,
not a prior group's larger arity.

### Per-role capacity contract

The public API cutover preserves the following current commit-time admission
sequence for every retained homogeneous input. “Commit-only footprint” is one
canonical sizing primitive over the same normalized commitment geometry used
by the kernel. It returns the maximum of the exact A matrix, B matrix, and
`CompressionChainPlan::max_setup_field_elements` for the B output. It excludes
opening/D, recursive-fold, setup-prefix, and terminal matrices. Admission,
setup tests, and capacity differentials call this primitive rather than
reimplementing its arithmetic.

| Position | Setup-seed and metadata checks | Materialized setup charged |
| --- | --- | --- |
| `Sole` | current group-local seed bounds; selected root-parameter structural validation, including D metadata | root commit-only footprint; no full schedule |
| `Prior` | current group-local seed bounds; frozen P-profile validation | profile commit-only footprint; no full schedule |
| `Final` | current final-group-local seed bounds; exact ordered key/profile validation | full schedule/root runtime footprint, followed by root A/B validation |

Full schedule admission remains part of proving for every complete batch. It
must not be pulled earlier into `Sole` or `Prior`: a setup that materializes
the exact commit-only footprint can validly create either commitment even
when it cannot prove a full schedule. Conversely, `Final` retains its existing
full schedule check. The refactor adds no aggregate comparison of prior-group
arity or total batch polynomial count with the commit setup's seed; those
checks remain at their current opening boundary.

Commit-time admission and setup generation are separate concerns. Under a
recursive configuration, an empty-prior S row may contain a recursive fold
whose `incoming_setup_prefix` must be provisioned by the current setup-capacity
path. `recursive_group_batch_candidates_for_capacity` must therefore continue
to include supported scalar recursive keys, and
`setup_prefix_slot_ids_for_capacity` must continue to return their exact
carried-prefix slots. The consolidated `Sole` commit does not consume or
charge those slots, but the refactor must not shrink the setup envelope or
change the later full-schedule admission.

## Batch-owned schedule selection

`OpeningScheduleSelection` identifies a complete ordered opening batch, so it
is not part of `CommitOutput`.

### Required cutover slice

`akita-types` adds one exact iterator-based constructor:

```rust
impl CommittedGroupBatchProfile {
    pub fn from_ordered_groups<'a, F, I>(
        groups: I,
        prior_group_profiles: PriorGroupProfiles,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + 'a,
        I: IntoIterator<Item = &'a CommittedGroup<F>>,
        I::IntoIter: ExactSizeIterator;
}
```

The iterator is nonempty and in transcript order. The constructor requires the
owned prefix length to equal `groups.len() - 1`, checks every prefix profile
against the corresponding group without reordering, and uses the last group's
profile as `final_group`. It then moves the prefix vector directly into the
batch profile. A one-element iterator therefore requires an empty prefix and
produces a sole batch without allocating a prior vector. The iterator form
lets `OpeningClaims` map directly over its commitments without allocating an
intermediate `Vec<&CommittedGroup<_>>`.

For a grouped lifecycle, callers build one `PriorGroupProfiles` value from the
ordered prior committed groups, borrow it for `GroupPosition::Final`, and
consume the same value during batch assembly after `commit` returns. This
matches the current lifecycle's one caller-owned prior vector plus the one
internal key clone; it does not collect the profiles again from all claims.

At the batch-assembly point where complete self-describing claims still
exist, every caller must:

1. build the exact batch profile with `from_ordered_groups`, consuming the
   same prior-profile ownership object borrowed by Final commitment;
2. call `CommitmentConfig::select_schedule_for_profiles` exactly once; and
3. only then construct profile-stripped `ProverOpeningData`.

This ordering is required for the commit-API cutover. Until the separately
gated encapsulation below lands, callers may pair the derived selection with
the existing tuple alias, but no caller may recover or reconstruct profiles
after stripping them. Selection failure is an assembly error where the
complete public claims still exist.

For a grouped batch, the assembly-time selection must equal the selection
currently returned by `commit_final_group`, including the exact `row_digest`.
For a sole batch, it must equal the row selected by the current prove-input
assembly. Selection resolution does not absorb transcript data; proving
receives the same selection and expanded row as before.

Batch assembly supports:

- S profiles from a sole commitment;
- P-only prior profiles;
- heterogeneous prior profiles produced under another configuration; and
- context-dependent profiles returned by a previous G final row when another
  exact G row accepts them.

### Separately gated prover-input encapsulation

Changing `SelectedProverOpeningData` from its public tuple alias is not a
semantic prerequisite for replacing the commit methods. It is an isolated
post-parity architecture slice with its own source migration and differential
gate. The slice may land separately from the public commit cutover, but it is
required before this spec is marked implemented because it prevents honest
prover callers from manually pairing a selection with profile-stripped data.
Verifier security does not depend on this convenience boundary: verification
continues to reconstruct exact profiles and compare them with the selected
row.

The final type is a struct with private selection/opening-data fields. Its
named prover-owned constructor (proposed as
`SelectedProverOpeningData::from_committed_claims::<Cfg>`) atomically performs
the required assembly sequence above. Its inputs are the owned
`PriorGroupProfiles`, self-describing committed `OpeningClaims`, group hints,
and ordered polynomial/prepared-group sources needed by
`ProverOpeningData::new`; claims alone are insufficient. It derives selection
before consuming/stripping the claims, then stores the row identity beside the
resulting opening data. It exposes a read-only `selection()` accessor so the
same identity can be placed in `GroupBatchStatement`; it does not expose a
public tuple conversion, `into_parts`, or constructor from an already
profile-stripped value.

The argument contract is therefore equivalent to:

```rust
pub fn from_committed_claims<'a, Cfg, P>(
    prior_group_profiles: PriorGroupProfiles,
    opening_claims: OpeningClaims<
        'a,
        Cfg::ExtField,
        CommittedGroup<Cfg::Field>,
    >,
    hints: Vec<AkitaCommitmentHint<Cfg::Field>>,
    polynomial_groups: Vec<&'a [&'a P]>,
) -> Result<
    SelectedProverOpeningData<
        'a,
        Cfg::ExtField,
        PreparedProverGroup<'a, P>,
        Cfg::Field,
    >,
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    P: RootPolyMeta<Cfg::Field>;
```

The implementation may generalize the final source argument to already
prepared groups, but it may not omit any of the four semantic inputs or accept
profile-stripped claims.

The low-level prover consumes `SelectedProverOpeningData` directly and owns
any decomposition of its private fields; `akita-pcs` only forwards the value.
`GroupBatchStatement` uses the accessor's exact selection with the same
ordered claims. Verification continues to accept that explicit public
selection from the statement/wire and resolves it through the verifier
boundary.

The existing heterogeneous grouped fixture differentially asserts that the
old final-returned selection, the new prover-derived selection, and the
statement selection have the same row digest, transcript events, and proof
bytes. Separate extractor tests cover empty, singleton, and ordered
multi-group inputs. A structural source audit proves that neither constructor
collects a temporary vector of group references and that batch assembly moves
the owned prefix. Whole-lifecycle allocation differentials prove parity. If an
absolute allocation assertion is added, it runs under an isolated counting
allocator: prefix construction allocates at most its one owned vector, while
the batch-profile extractor itself performs no allocation for that vector.

## Polynomial-group arity

One `CommittedGroupProfile` contains one `PolynomialGroupLayout`. The public
and opening-lifecycle rule is therefore:

- `polys` is nonempty;
- every polynomial within this group has the same `num_vars`;
- that common arity and `polys.len()` define the group layout; and
- the group-local count fits the commit setup seed, while complete-batch
  counts are validated at the existing opening boundary.

Different groups in the same multi-group opening may have different arities.
Only within-group mixed arity is removed.

The old maximum-arity validators do not form a coherent end-to-end contract:
the prepared opening group and some commitment kernels require same-shape
sources. There are no current in-repository mixed-arity commitment callers.
With no compatibility requirement, the consolidated API rejects such input
rather than preserving an implicit padding path.

As part of the cutover, re-audit and remove, when they have no other live
owner:

- `prepare_batched_commit_inputs`;
- `padded_scalar_batch_num_vars`;
- `validate_scalar_point_matches_poly_arity`;
- their public re-exports; and
- padding-only tests.

The byte-identity guarantee covers every retained valid homogeneous input.
Mixed-arity calls are an intentional source/behavior removal, not a new proof
encoding.

## Architecture

### Ownership after cutover

| Area | Responsibility |
| --- | --- |
| `crates/akita-pcs/src/scheme/mod.rs` | one user-facing `commit` method; direct ring-policy validation; re-export the role/result types |
| `crates/akita-prover/src/api/commitment.rs` | one role-aware orchestrator, one group validator, one normalized geometry view with canonical commit-only footprint sizing, one tensor branch, and one commitment kernel |
| `crates/akita-prover` | `GroupPosition`, `CommitOutput`, the private selected-prover-input boundary and atomic constructor, and the catalog-independent explicit-params arithmetic boundary; no unused commitment trait |
| `crates/akita-config` | one row-preserving honest-prover selector, one P/profile resolver, exact-profile batch selection, explicit-selection verification, and setup admission |
| `crates/akita-schedules` | unchanged S/P/G payloads, strict row/profile resolution, materialized row cache, and identity-bound P registry |
| `crates/akita-planner` | deterministic generation/drift checks and generator-only G provenance; no commitment-parameter replanning for this refactor |
| `crates/akita-types` | existing profile/key/row-selection identities, `PriorGroupProfiles`, and one ownership-reusing committed-claims-to-batch-profile extractor |

### Configuration selection boundaries

The current schedule-only accessors cause the commit path to discard a
resolved row and resolve it again. The clean target has one honest-prover row
selection primitive, proposed as:

```rust
fn select_schedule_for_key(
    key: &AkitaScheduleLookupKey,
) -> Result<ResolvedScheduleRow, AkitaError>;
```

A configuration-owned opening-layout method may assemble its scalar key and
call that primitive:

```rust
fn select_schedule_for_opening(
    layout: &OpeningClaimsLayout,
) -> Result<ResolvedScheduleRow, AkitaError>;
```

This method owns the configuration-specific layout-to-key mapping; it contains
no second selection implementation. Existing schedule-only consumers read or
consume `row.schedule()` from this boundary. The schedule-only
`runtime_schedule` / `get_params_for_prove` surfaces are migrated rather than
retained as pass-through aliases when they represent the same selection
concept.

The row selector is the canonical overridable `CommitmentConfig` boundary. A
default implementation may use the generated catalog, but callers never
bypass a configuration override by invoking generated resolution directly.
Every current custom or synthetic configuration override migrates to this
row-returning contract, including off-catalog test configurations. Recursive
adapters preserve their adapter-local empty-prior S selection and companion G
selection; any other adapter that intentionally delegates must preserve that
delegation. Tests pin the exact rows returned by generated, recursive, and
synthetic overrides.

`CommitmentConfig::get_params_for_batched_commitment` is deleted. It is a thin
schedule-to-root clone and would recreate the duplicate-resolution problem.
Its tests, benches, examples, setup code, verifier test code, and profiling
artifact callers migrate to the selected row's root-final parameters.

The P boundary is named for its role, for example:

```rust
fn resolve_prior_group_profile(
    layout: &PolynomialGroupLayout,
) -> Result<CommittedGroupProfile, AkitaError>;
```

It replaces `akita_config::committed_group_profile`; the old generic name is
not retained as an alias. Configuration adapters select or delegate the same
S and G rows they do now and use their existing P registry. Recursive adapter
tests must specifically prove that scalar S resolution remains adapter-local;
tests for delegating adapters must prove that delegation returns the same exact
rows and profiles.

`select_schedule_for_profiles` remains the honest-prover exact-batch boundary,
and `resolve_schedule_selection` remains the verifier's explicit-row-identity
boundary. They are distinct concepts and are not wrappers for commit.

### Internal consolidation

`commit_with_validated_params` and
`commit_with_validated_profile` are replaced by one function over a borrowed
internal geometry view. The view contains all execution inputs, including:

- A/B dimensions and matrices;
- live ring/block geometry;
- inner/outer bases and digit counts; and
- the compression modulus profile.

The extraction is mechanical. It must preserve arithmetic iteration order,
backend calls, compression-chain construction, allocations that affect
performance, and hint assembly. The old functions do not remain as forwarding
wrappers.

`commit_with_params` remains as the one catalog-independent lower arithmetic
boundary. It is needed for:

- custom polynomial-source/backend conformance tests that compare against the
  dense oracle; and
- isolated tensor-projection/commit microbenchmarks that deliberately use
  caller-selected or non-cataloged parameters.

It validates caller-supplied concrete parameters and calls the same normalized
kernel. It performs no configuration-aware lookup and is not a second PCS
commit API. `batched_commit_with_params` is deleted, and its duplicate contract
test is folded into the retained primitive's test.

Touched-file cleanup also removes:

- the scheme-local shadowing `CommitmentWithHint` alias;
- obsolete `CommittedGroupWithHint` and
  `FinalCommittedGroupWithHint` public result aliases;
- `validate_policy_ring_dim` and
  `validate_verifier_policy_ring_dim`, which ignore their setup argument;
- `should_transform_group_commitment`, which only forwards to
  `root_tensor_projection_enabled`;
- `root_transform_ring_dim`, whose extension-field path repeats the Sole
  selector call and whose degree-one early return is already implied by
  `root_tensor_projection_enabled`;
- `should_transform_final_group_commitment`, whose conditional
  `runtime_schedule` call duplicates the selected Final row; and
- `CommitmentProver` and its re-exports, after one final implementor audit.

No removed item survives as a deprecated shim or pass-through alias.

### Separately gated catalog hardening

Catalog identity and G-provenance hardening are not needed to distinguish the
three commit roles or to replace the public methods. They form an isolated
post-parity slice with their own generated diff and performance gate. The
slice may land separately from the API cutover, but it is required before this
spec is marked implemented because P is a configured parameter authority. A
hardening failure is never repaired by changing an S/P/G payload.

P is already a generated authority for prior commitments, but its count and
descriptor bytes are not currently included in
`GeneratedScheduleCatalogIdentity`. Add an ordered P count and digest over a
deterministic encoding of every compact `GeneratedPrecommittedProfile` field
to the generated identity. The constant fields participate in
`identity_digest`, so caches may use the embedded identity to namespace a
catalog without reading the live P slice. Together with the already-bound
policy, those fields determine the expanded canonical descriptor exactly.
Drift tests compare compact expansion with the canonical descriptor bytes so
the two representations cannot split.

This is a wiring guard, not a protocol change. Metadata-only re-emission may
change embedded catalog identity constants. It must not change:

- any `GeneratedFoldScheduleEntry` payload;
- any `GeneratedPrecommittedProfile` payload;
- an `OpeningScheduleSelection` or schedule row digest;
- setup matrix parameters or capacity;
- a commitment, transcript, or proof byte; or
- a public serialization format.

P validation rejects every duplicate layout, even when the repeated descriptor
bytes are equal, and rejects identity/profile-array mismatches. It does not
impose S/P equality.

Live-array validation is lazy and lives in a separate P-registry validation
domain invoked only by `resolve_prior_group_profile`. The existing shared
entries-validation cache and materialized-row cache retain entries-owned
control flow and never traverse, hash, expand, or index
`precommitted_profiles`. Their key shape remains entries pointer/length,
embedded identity digest, and policy digest; the embedded digest value may
include the new constant P metadata.

The dedicated P-validation cache is keyed by the live static P-array pointer
and length, embedded identity digest, and policy digest. On a cache miss, the
P boundary checks the live count, computes the ordered compact-record digest,
rejects every duplicate layout, and compares the result with the embedded P
identity. A successful cache hit for the same immutable `&'static` slice does
not recompute that digest. If the registry ever stops being static/immutable,
the P-only boundary must instead rehash it on every hit; that must still never
affect S/G resolution.

The runtime entry/policy/hook identity expectation is therefore separate from
live P-registry validation. Adding P fields to the current whole-identity
mirror without this split would force S/G validation to derive them from the
live slice and is forbidden. Merely embedding the P digest is also
insufficient at the P boundary, because a differently wired live array must
not reuse an earlier successful validation.

Cold P validation must be fused with the resolver's existing complete registry
scan, or be structured equivalently, so it does not add a second traversal of
compact records or any additional profile expansion. Repeated P lookup
performs no more registry traversal or expansion than today unless a
separately benchmarked lazy index improves it. During an operation-scoped
measurement taken after setup/fixture construction, cold and warm `Sole` and
G-only resolution perform zero compact P-record visits, live P digests,
profile expansions, or P-index materializations. Setup sizing and an
explicitly requested full-catalog audit may inspect P and are outside that
counter window.

Every prior descriptor embedded in G must remain reproducible by a
generator-owned recipe: either its producing P configuration/layout or an
exact earlier G producer context. This provenance remains generator/test-only;
it is not added to `CommittedGroupProfile`, lookup keys, serialized values, or
the transcript. The provenance audit is independent of the API cutover and
must not rewrite a descriptor to make it pass.

The current P resolver may otherwise remain at its present cost because its
domain and traffic do not grow. If profile lookup is indexed in the hardening
slice, the index is a separate lazy P materialization keyed by the same live
array/catalog/policy identity. A `Prior`-only call must not materialize all S/G
rows. Cold and repeated P lookup benchmarks gate that optional optimization.

## Protocol and performance invariants

### Exact-output invariants

- `Sole` selects the same S row and exact root-final parameters as current
  homogeneous `commit`/`batched_commit`.
- `Prior` selects the same P descriptor as current `commit_group`.
- `Final` selects the same G row and exact final parameters as current
  `commit_final_group` for the same ordered prior profiles.
- S, P, and G payload bytes are unchanged. There is no `S ⊆ P` requirement
  and no S/P canonicalization.
- Tensor projection is enabled under the same condition and runs at the same
  A-role dimension as the corresponding old path.
- The commitment kernel receives identical polynomial values, geometry,
  matrices, digit parameters, and iteration order.
- `CommittedGroupProfile`, serialized commitment, and
  `AkitaCommitmentHint` values are identical role-for-role.
- The Sole profile obtained from the selected row is identical to the old
  profile assembled from the input-derived layout and selected root params.
- Batch assembly selects the same row digest and expanded schedule.
- Transcript event order/content, serialized proof bytes, and proof length are
  identical.
- Setup seed requirements, exact matrix footprints, and aggregate setup
  capacity envelopes are identical.
- Recursive scalar `Sole` selection preserves the active recursive-catalog row,
  including any `incoming_setup_prefix`, and setup-capacity enumeration keeps
  provisioning that row's exact carried-prefix slots.
- Commit-time acceptance is identical per role for retained homogeneous
  inputs: no full-schedule charge is added to `Sole`/`Prior`, and no aggregate
  prior-group seed check is added to `Final`.
- Wire formats and `CommittedGroupProfile` fields do not change.
- Malformed profiles, unsupported exact sequences, and empty `Final` input
  reject with `AkitaError` or `SerializationError`, never panic.

Any failure of an exact-output invariant blocks the refactor. It is not a
measurement to accept after review.

### Work invariants

The consolidated path performs no more protocol work than the corresponding
current role:

| Position | Resolution work after cutover | Arithmetic work |
| --- | --- | --- |
| `Sole` | one S row selection; matches the old degree-one count and removes the old extension-path second selector call | at most one tensor projection and exactly one commitment kernel |
| `Prior` | one P resolution with no larger registry | at most one tensor projection and exactly one commitment kernel |
| `Final` | one G key selection during commit and one exact-profile selection at batch assembly | at most one tensor projection and exactly one commitment kernel |

The Final exact-profile selection is the existing
`select_schedule_for_profiles` work moved out of `commit_final_group`, not an
additional resolution. Sole proving flows already perform their batch
selection outside commitment. Removing the transform helpers' repeat lookups
must reduce redundant work; it does not create a performance allowance for
any other stage.

Catalog hardening adds zero live P-record visits, digests, expansions, or
index materializations to both cold and warm `Sole` and G-only resolution,
measured after setup/fixture construction. Cold P validation does not add a
second full registry traversal or extra profile expansions to the current P
lookup. These are operation-scoped structural counters, not wall-clock-only
expectations.

The orchestrator adds no allocation proportional to polynomial length. The
grouped lifecycle creates one caller-owned `PriorGroupProfiles` vector, as the
current caller does, and commitment makes at most the same one
order-preserving clone needed for its owned schedule key. Batch assembly moves
the caller-owned vector into `CommittedGroupBatchProfile`; it does not collect
a second copy from claims. The existing exact-profile selector may make the
same owned-key clone as today. Differential allocation counters must show that
the complete commit-and-assembly lifecycle performs no more metadata copies,
allocations, or schedule/profile clones than the corresponding current
lifecycle.

Benchmarks compare old and new role-for-role under the same build, machine,
features, setup, and inputs. Record commit latency, prove latency, verify
latency, peak/total allocation where available, and resolver/transform/kernel
operation counts. Exact structural counters are the primary regression guard;
wall-clock results must remain within the repository's normal benchmark noise
or improve. A material regression blocks the cutover.

## Evaluation

### Acceptance criteria

- [ ] `AkitaCommitmentScheme` exposes one public method named `commit` with
      `setup`, `polys`, `stack`, and `GroupPosition` in that order.
- [ ] `GroupPosition::{Sole, Prior, Final { prior_group_profiles }}` maps
      exactly to S, P, and G respectively.
- [ ] `PriorGroupProfiles` privately owns one exact ordered vector. Final
      commitment borrows it and batch assembly later consumes the same
      allocation; construction from an existing vector does not clone or
      reallocate it.
- [ ] `Final` rejects an empty `PriorGroupProfiles` value and preserves the
      exact order of a nonempty borrowed value.
- [ ] `batched_commit`, `commit_group`, and `commit_final_group` are removed
      without aliases or wrappers.
- [ ] `CommitOutput` contains a named committed group and hint, and no
      group-local schedule selection.
- [ ] S/P/G generated payloads and all planner-selected parameters remain
      unchanged; generated diffs are limited to approved identity metadata.
- [ ] Current and new role paths produce identical profile bytes, serialized
      commitments, hints, selections, transcript logs, serialized proofs, and
      proof lengths for deterministic fixtures.
- [ ] Setup matrix footprints and aggregate capacity envelopes are identical
      before and after.
- [ ] For every retained homogeneous input, old and new role paths have the
      same commit-time accept/reject result. In particular, the exact
      commit-only capacity on an otherwise seed-valid setup remains sufficient
      for `Sole` and `Prior`, while `Final` retains its current full-schedule
      admission; no aggregate prior-group seed check is added to commit.
- [ ] Batch assembly calls
      `CommittedGroupBatchProfile::from_ordered_groups` before profiles are
      discarded: the last group is final and every preceding group is prior,
      with no reordering and no intermediate group-reference vector. It
      consumes the same owned prior vector borrowed by Final rather than
      collecting another copy.
- [ ] In the separately gated encapsulation slice,
      `SelectedProverOpeningData` has private fields and one atomic
      committed-claims constructor that derives the batch profile, selects the
      row, and only then strips profiles into opening data; no public manual
      `(selection, opening_data)` pairing, tuple conversion, or `into_parts`
      remains. A read-only `selection()` accessor supplies the statement
      identity, and only low-level prover code may decompose the value.
- [ ] P-only layouts remain valid `Prior` commitments, including fp128 Dense
      `(15, 2)`.
- [ ] S-only layouts remain valid `Sole` commitments without gaining P
      descriptors.
- [ ] Every current G row remains selectable with its exact prior profiles;
      the heterogeneous Dense-prior/OneHot-final round trip remains supported.
- [ ] Recursive configurations preserve current active-configuration scalar S
      selection, P lookup, G selection, row identities, and proof bytes.
      Empty-prior scalar keys remain recursive-catalog selections where they
      are today; they are not redirected to the wrapped configuration.
- [ ] Recursive setup-capacity enumeration retains supported empty-prior
      scalar keys and their exact `incoming_setup_prefix` slots. `Sole`
      commitment does not charge those slots, while setup provisioning and
      later full-schedule admission remain byte- and capacity-identical.
- [ ] Nonempty `Final` input rejects malformed, altered, unsupported, or
      incorrectly ordered distinct profiles before commitment kernels run.
- [ ] One row-preserving configuration boundary serves S and G selection; the
      selected row is reused for setup, tensor, parameter, and profile checks.
- [ ] For every `Sole` fixture, the requested group layout equals the selected
      row's final profile layout, and that complete profile is byte-identical
      to the old profile locally assembled from the requested layout and root
      parameters.
- [ ] Generated, recursive, and every custom or synthetic configuration
      override migrate to the row-returning boundary and return the same exact
      rows as before; no caller bypasses an override by resolving the
      generated catalog directly.
- [ ] `get_params_for_batched_commitment` is removed and all callers migrate
      to the row-preserving source of truth.
- [ ] `committed_group_profile` is replaced by a clearly named prior-profile
      boundary with no old-name alias.
- [ ] Live public lookup/batch-profile fields use `prior_group_profiles`; the
      source-only rename from `precommitteds` changes no canonical bytes,
      ordering, digest, or wire format.
- [ ] Polynomials inside one group must have equal `num_vars`; different
      groups may have different arities.
- [ ] Padding-only helpers, re-exports, and tests with no live owner are
      removed.
- [ ] One normalized geometry function replaces both validated kernels and
      preserves arithmetic order and outputs.
- [ ] One canonical commit-only setup-footprint primitive prices the exact A,
      B, and B-output compression matrices used by commitment; Sole/Prior
      admission and capacity tests call it, and it never charges D,
      recursive, setup-prefix, or terminal matrices.
- [ ] `commit_with_params` remains only as the documented
      catalog-independent testing/benchmark primitive;
      `batched_commit_with_params` is removed.
- [ ] The unused `CommitmentProver` trait and re-exports are removed after a
      final zero-implementor audit.
- [ ] Shadowing result aliases, ignored-setup wrappers, trivial transform
      wrappers, and redundant schedule resolution are removed.
- [ ] In the separately gated hardening slice, generated catalog identity
      binds the ordered P count and a deterministic digest of the compact
      `GeneratedPrecommittedProfile` records; expansion drift tests prove that
      those records still produce the exact canonical descriptor bytes.
- [ ] P validation rejects every duplicate layout and its live-array cache
      is separate from both entries-owned caches. Its identity binds the
      actual static P array, length, embedded identity, and policy; its cold
      validation compares the recomputed live digest without adding P work to
      `Sole` or G selection.
- [ ] Operation-scoped counters reset after setup/fixture construction show
      zero live P-record visits, digests, expansions, and index
      materializations for cold and warm `Sole` and G-only resolution. Cold P
      validation adds no second full registry traversal or extra expansion.
- [ ] The consolidated role paths perform no more transforms, commitment
      kernels, polynomial-sized allocations, or schedule/profile resolutions
      than specified in the work-invariant table.
- [ ] Test-local counters at the canonical role-selector leaves, reset after
      setup/fixture construction, pin: degree-one `Sole` at one before/after;
      extension-field `Sole` at two/one; `Prior` at one P resolution
      before/after; and `Final` at one G-key plus one exact-profile selection
      after. A tensor-enabled Final fixture pins the old path at two G-key plus
      one exact-profile selection and the target at one plus one. Tensor
      decisions and outputs remain unchanged.
- [ ] No live Rust caller or live documentation uses a removed method, except
      this migration record and explicitly historical snapshots.
- [ ] Documentation guardrails and every affected CI feature graph pass.

### Characterization and differential testing

Before deleting any old method, capture deterministic baselines from the
current code:

1. The exact S and P profile bytes for every physical `S ∩ P` overlap,
   including the matching fp32 OneHot `(16, 2)` counterexample and every
   differing fp128 overlap.
2. Complete S, P, and G domains for every generated family, derived by tooling
   rather than hand-maintained lists.
3. Canonical bytes/digests of every generated S row, P descriptor, and G row.
4. Role fixtures for sole scalar, equal-arity multi-polynomial sole, prior,
   final, heterogeneous, recursive, and setup-offloaded flows. This includes
   a recursive scalar S row with empty priors and a carried incoming setup
   prefix.
5. For each tractable role fixture: selected parameters, tensor decision and
   dimension, profile bytes, serialized commitment, hint, selection row
   digest, transcript event log, serialized proof bytes, and proof length.
6. For every Sole fixture: the input-derived layout, resolved-row final
   profile layout, profile reconstructed from that layout and the row's root
   parameters, and the old returned profile, compared through
   `canonical_descriptor_bytes()` as well as structural equality.
7. Setup matrix envelopes, physical setup-field requirements, and old/new
   accept/reject outcomes across bounded capacities for each role.

The old public `commit` already occupies the target name, so Rust cannot host
both public signatures during migration. Build the role-aware orchestrator
temporarily under a distinct private or crate-private name such as
`commit_for_position`. Prover unit differentials invoke the old role functions
and this temporary function on the same deterministic setup and input. At the
scheme boundary, capture the old public paths as compact committed golden
fixtures or hashes. Once both comparisons are green, atomically replace the
public `commit` signature, remove the other public methods, and rename the
internal orchestrator to its canonical name. The golden comparisons retain
the commitment/proof guard after the old entry points are deleted.

Negative tests cover:

- empty `polys`;
- mixed arity within one group;
- empty `Final` profiles;
- malformed profiles;
- distinct-profile reordering, omission, duplication, and alteration;
- unsupported exact G sequences;
- P-only use as `Sole`; and
- setup seed or matrix-footprint insufficiency.

Capacity differentials compute `commit_required` with the canonical
commit-only sizing primitive and `full_required` with the same production
schedule-sizing primitive used by `ensure_prover_schedule_fits_setup`. Pinned
Sole and Prior fixtures generated at `commit_required` succeed before and
after. A pinned G fixture first asserts `commit_required < full_required`, then
proves that a setup generated at `commit_required` is rejected by Final before
and after. Each old/new pair uses identical setup bytes and inputs; Prior is
not assigned a fictitious standalone full schedule.

Reordering tests use distinct descriptors; swapping equal descriptors is not
observable.

Performance tests instrument row/profile resolution counts, tensor-projection
count, commitment-kernel count, compact P-record visits/digests/expansions,
and relevant allocations. Counters are test-local, reset after setup and
fixture construction, and placed at the canonical selector leaves rather than
both a layout assembler and its delegated selector. Pin degree-one Sole at
one selector call before/after and extension-field Sole at two/one; current
concrete candidates are fp128 Dense and fp32 OneHot. Pin Prior at one P
resolution before/after. For Final, count G-key selection separately from
exact-profile selection: the target performs one of each. The existing fp32
extension grouped fixture is a candidate that must record two old G-key calls
plus one old exact-profile call, versus one plus one after cutover.

All direct runtime P traversal goes through one test-instrumented boundary
reporting records visited, digest finalizations, profiles expanded, and index
materializations. A cold test uses a fresh registry/cache identity. With `N`
records it visits at most `N` records total, finalizes one live digest, and
performs no more expansions than the old baseline; warm resolution performs
no live digest and no more resolver/index work than the old baseline. Cold and
warm Sole/G-only fixtures report zero in every P counter, while Prior does not
materialize S/G rows. If an optional P index is implemented, its warm counts
may improve but remain under the same upper bounds.

## Alternatives considered

### Keep only optional prior profiles

Rejected. An absent/empty list cannot distinguish `Sole` from `Prior`, whose
current S and P profiles differ on supported layouts.

### Give `None` and `Some(&[])` different roles

Rejected. This encodes protocol state in an empty-container convention and is
easy to misuse. `GroupPosition` makes the same state space explicit.

### Add a no-prior role beside `Option<&[...]>`

Rejected. Two arguments would encode one three-way fact and permit invalid
combinations. A sum type is smaller and exhaustive.

### Canonicalize S and P

Rejected. Choosing P changes sole commitments/proofs on differing overlaps;
choosing S changes prior descriptors embedded in G. The user requires current
values from both lifecycles.

### Infer the role from layout coverage

Rejected. Layouts in `S ∩ P` are ambiguous, and some have different role
profiles. Future use cannot be inferred from polynomial values.

### Return `Option<OpeningScheduleSelection>`

Rejected. The selection identifies a complete batch and belongs at batch
assembly. A conditional commit result would expose catalog coverage without
helping the caller.

### Pass prior `CommittedGroup` values

Rejected. Parameter selection needs exact profiles only. Commitment values are
bound by opening claims and the transcript later.

### Preserve within-group maximum-arity padding

Rejected. There is no coherent live opening contract or current caller for
it, and the prepared group requires homogeneous arity. Heterogeneous arity
across distinct groups remains supported.

### Keep compatibility wrappers

Rejected. The repository gives no backward-compatibility guarantee, and thin
wrappers would violate the single-source-of-truth policy.

## Detailed execution plan

1. **Freeze the baseline.** Inventory all four methods, aliases, re-exports,
   direct callers, configuration accessors, padding helpers, tests, benches,
   examples, profile artifacts, and documentation. Capture the exact-output
   and performance fixtures before changing behavior.
2. **Audit S, P, and G from generated artifacts.** Emit domains,
   intersections, differences, exact profile mismatches, G prior descriptors,
   adapter-local selection/delegation behavior, recursive scalar setup-prefix
   requirements, and producer provenance. This is an audit only; do not plan
   new rows or profiles.
3. **Unify the private arithmetic kernel without changing callers.** Extend
   the internal geometry view, route both params/profile paths through one
   function, add the canonical commit-only setup-footprint primitive from the
   same geometry, and run the old-path differential suite. Remove the two
   duplicate validated functions only after equality is established.
4. **Make schedule selection row-preserving.** Establish one canonical
   configuration-owned key-to-`ResolvedScheduleRow` boundary and one
   layout-to-key assembler. Preserve each adapter's current behavior exactly:
   in particular, recursive empty-prior scalar keys remain adapter-local
   recursive-catalog selections and are not delegated to the wrapped
   configuration. Migrate schedule-only consumers to the row and delete
   `get_params_for_batched_commitment` rather than wrapping it.
5. **Name the prior-profile boundary.** Replace
   `committed_group_profile` with `resolve_prior_group_profile` (or the final
   agreed role-exact name), migrate callers, and keep the current P payload and
   lookup semantics. Do not add S-derived P entries.
6. **Build the temporary role-aware orchestrator.** Introduce
   `PriorGroupProfiles`, `GroupPosition`, `CommitOutput`, and a distinctly
   named private or crate-private implementation such as
   `commit_for_position` over the three selectors and shared geometry/kernel
   pipeline. Keep the public surface unchanged while prover-unit differentials
   compare each old role function with this internal implementation, including
   exact role-specific capacity admission and Sole profile provenance.
7. **Move required selection to batch assembly.** Add the iterator-based
   `CommittedGroupBatchProfile::from_ordered_groups`, derive the selection
   exactly once before `ProverOpeningData` strips profiles, and compare it with
   the old final-returned or sole-assembly selection. Do not allocate an
   intermediate group-reference vector or recollect prior profiles: borrow one
   `PriorGroupProfiles` value for Final and consume it into assembly. Keep the
   existing selected-data tuple during this slice so commit-output parity can
   be isolated.
8. **Cut over arity validation.** Use one homogeneous group validator in all
   roles. Confirm distinct-group heterogeneity remains covered; remove
   unowned padding helpers, re-exports, and tests.
9. **Perform one atomic public-API cutover.** Replace the existing public
   `commit` signature, remove `batched_commit`, `commit_group`, and
   `commit_final_group`, and rename the internal orchestrator to the canonical
   implementation name in the same change. In that same buildable change,
   migrate every scheme/integration test, setup/verifier helper, example,
   bench, profile workload, recursion artifact, and live public-field use to
   explicit positions, reusable `PriorGroupProfiles`, and
   `prior_group_profiles` naming. Compare the replacement scheme path with the
   captured old-surface golden fixtures.
10. **Remove wrapper slop.** Delete `batched_commit_with_params`, obsolete
    aliases, ignored-setup wrappers, transform wrappers/lookups, and the
    zero-implementor `CommitmentProver` trait. Retain and document only
    `commit_with_params` as the catalog-independent arithmetic/test boundary.
11. **Audit the removed surface.** Use repository-wide symbol and field-name
    searches to prove that no live caller remains on a removed method, result
    alias, `precommitteds` public field, or old batch-assembly pattern. This is
    an audit after the atomic migration, not a later caller-fix step.
12. **Apply separately gated prover-input encapsulation.** After commit and
    batch-selection parity is green, replace the public selected-data tuple
    with its private-field type and atomic committed-claims constructor. Make
    low-level prover code own decomposition, expose only the read-only
    selection accessor, and differentially pin the heterogeneous prover and
    statement selection to the old row digest and proof bytes.
13. **Apply the separately gated catalog hardening.** After role-for-role
    parity is green, bind the ordered P count and deterministic compact-record
    digest into generated identity. Add a P-only validation cache bound to the
    immutable live registry and policy; do not add any live P operation to the
    entries-validation or materialized-row caches. Add duplicate/tamper tests
    and generator-owned G provenance drift tests. Review the generated diff
    independently: only identity metadata may change, never an S/P/G payload
    or schedule row.
14. **Update live documentation.** Reconcile at least:

    - `book/src/usage/commitment-api.md`;
    - `book/src/usage/quickstart.md`;
    - `book/src/how/commitment.md`;
    - `book/src/how/architecture.md`;
    - `specs/multi-group-batching.md`;
    - `specs/distributed-setup-offloading.md`;
    - `docs/compute-backends.md`; and
    - every additional live reference found by `rg`.

    Historical snapshots remain historical or are archived according to their
    headers; they are not silently rewritten as current documentation.
15. **Validate the cutover.** Run focused differential/e2e tests, generated
    drift checks, the cheap repository preflight, documentation guardrails,
    path-specific workflows, all CI Clippy feature graphs, and the current
    CI-fidelity test command from `.github/workflows/ci.yml`. Review generated
    diffs to prove that only identity metadata changed.

Each pre-cutover step must leave the old-versus-new differential suite green;
after the atomic cutover, the captured golden suite provides the same gate. If
an exact-output or work invariant fails, stop and revise the implementation;
do not replan catalogs or accept new proof values under this spec.

## Non-goals

- Replanning, canonicalizing, or expanding S, P, or G parameter domains.
- Changing commitment construction, opening relations, transcript order, or
  serialization.
- Making every P profile openable as a sole group.
- Making every returned profile reusable in every later position.
- Moving fold depth or opening/D parameters into `CommittedGroupProfile`.
- Inferring a group's future role from layout, mutable state, or prior
  commitment payloads.
- Combining commitment and proving.
- Retaining old Rust APIs for compatibility.
- Using catalog identity metadata as a new public protocol identity.

## Decision log

This is the interaction point for design review. The spec remains `proposed`
until the role API and execution plan are approved. There is no catalog
measurement waiver: byte identity is a hard gate.

| Date | Question or decision | State |
| --- | --- | --- |
| 2026-08-10 | Replace the optional-priors-only shape with `GroupPosition::{Sole, Prior, Final}` because current S/P roles are byte-distinct. | proposed |
| 2026-08-10 | Name final-role metadata `prior_group_profiles` and borrow it. | proposed |
| 2026-08-10 | Preserve all current S/P/G payloads, commitments, selections, transcripts, proof bytes/sizes, and setup envelopes exactly. | required |
| 2026-08-10 | Permit source-API breaks and remove all compatibility wrappers. | required |
| 2026-08-10 | Keep `OpeningScheduleSelection` batch-owned and omit it from `CommitOutput`. | proposed |
| 2026-08-10 | Reject within-group mixed arity while preserving different arities across groups. | proposed |
| 2026-08-10 | Delete `get_params_for_batched_commitment`; rename the P resolver by its prior-group role. | proposed |
| 2026-08-10 | Retain `commit_with_params` only for catalog-independent backend tests and microbenchmarks. | proposed |
| 2026-08-10 | Bind unchanged P bytes into internal catalog identity; do not change protocol rows or descriptors. | proposed |
| 2026-08-10 | Treat a separate lazy P index as optional and benchmark-gated, not as a prerequisite for role-based consolidation. | proposed |
| 2026-08-10 | Preserve each role's current commit-time capacity checks: commit-only A/B plus B-output compression for Sole/Prior, full schedule for Final, and no new aggregate prior-group seed check. | required |
| 2026-08-10 | Isolate live P validation from entries/materialized-row caches and require zero P work on Sole/G paths. | required |
| 2026-08-10 | Make ordered batch-profile extraction part of the commit cutover, but land private selected-prover-data encapsulation as a separate parity-gated slice. | proposed |
| 2026-08-10 | Own prior metadata once in `PriorGroupProfiles`: Final borrows it and batch assembly consumes it, preserving the current vector allocation/copy count. | required |
| 2026-08-10 | Rename public `precommitteds` fields to `prior_group_profiles`; retain generated “precommitted” names only for the internal P artifact vocabulary. | proposed |
| 2026-08-10 | Preserve rebased-main recursive scalar behavior: empty-prior S keys resolve in the active recursive catalog, carried setup-prefix slots remain provisioned, and commit-time Sole admission remains commit-only. | required |

## Documentation

The durable public behavior belongs in
`book/src/usage/commitment-api.md`. While the design is in flight, this spec
owns the API and byte/performance acceptance criteria, and
`specs/multi-group-batching.md` owns the underlying multi-group protocol. Once
implemented and folded into the book, archive this spec according to
`specs/PRUNING.md`.

## References

- `commit-api-consolidation-review.md`
- `commit-api-consolidation-review-2.md`
- `commit-api-consolidation-review-3.md`
- `crates/akita-pcs/src/scheme/mod.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/api/scheme.rs`
- `crates/akita-prover/src/protocol/core/root_group.rs`
- `crates/akita-prover/src/types/opening_data.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/precommitted_commitment.rs`
- `crates/akita-config/src/recursive_commitment.rs`
- `crates/akita-config/src/setup_prefix_slots.rs`
- `crates/akita-planner/src/generated_families.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-schedules/src/catalog_identity.rs`
- `crates/akita-schedules/src/generated/mod.rs`
- `crates/akita-schedules/src/resolve.rs`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-types/src/schedule_selection.rs`
- `crates/akita-types/src/opening_claims.rs`
- `crates/akita-types/src/compression/chain.rs`
- `crates/akita-types/src/proof/setup_envelope.rs`
- `specs/mixed-ring-dimension-per-level.md`
- `specs/multi-group-batching.md`
- `book/src/usage/commitment-api.md`

## Appendix A: Proposed S-only commitment authority

### Status and relationship to this specification

This appendix records the implemented follow-on simplification: P is no longer
a runtime commitment-parameter authority. Every ordinary commitment uses its
scalar S profile, whether the group is opened alone or retained for use before
a later final group.

The precommitted/multi-group protocol case is not removed. Earlier commitments,
their ordered profiles, G-row selection, setup-prefix handling, and grouped
verification remain. Only the second parameter source is removed: an earlier
group is now committed as `GroupPosition::Independent` with S and that exact S
profile is supplied to `GroupPosition::Final`.

The desired end state is:

```text
one ordinary commitment authority C(layout) = existing S(layout)

ordinary/independent group  -> commit with S
grouped final               -> select G by exact ordered S profiles
```

The old caller-visible sole/prior positions are collapsed into
`GroupPosition::Independent`. `Final { prior_group_profiles }` remains distinct
because it still needs the exact ordered prefix to select the complete grouped
row.

### Implementation and measurement result

The generated P registry, resolver, validation cache, emitter output, and
catalog count/digest fields have been removed. Every former P profile consumed
by a shipped G row or lifecycle was mapped to an existing S row or gained the
required scalar S key (notably fp128 Dense `(15,2)` and fp128 OneHotMultiChunk
`(16,1)`). Registry-only P shapes that had no grouped consumer are not carried
forward as a second standalone domain. No grouped/precommitted lifecycle was
removed. Grouped rows are regenerated from exact ordered S profiles. The
generated diff contains no deletion of a scalar row with
`precommitted_groups: &[]`; existing scalar S row payloads are byte-for-byte
unchanged. Permanent catalog coverage also checks that every grouped prior
descriptor has an exact generated scalar S producer.

The temporary measurement prints were removed after recording these compressed
proof sizes. Baselines use P-keyed G rows; candidates use S-keyed G rows. Every
honest commit/prove/verify path below succeeded.

| Case | Ordered layouts (prior(s) → final) | P/G bytes | S/G bytes | Delta | Delta % |
| --- | --- | ---: | ---: | ---: | ---: |
| fp32 extension-field | `(14,1) → (20,1)` | 72,599 | 72,748 | +149 | +0.21% |
| heterogeneous OneHot/Dense | `(14,1), (15,2) → (16,1)` | 77,812 | 78,932 | +1,120 | +1.44% |
| OneHot prior smaller, final batch | `(14,1) → (20,2)` | 77,829 | 77,842 | +13 | +0.02% |
| OneHot prior larger | `(20,1) → (14,1)` | 75,283 | 77,840 | +2,557 | +3.40% |
| OneHot two-polynomial prior | `(14,2) → (20,1)` | 77,816 | 77,804 | -12 | -0.02% |
| multi-chunk W2R2 | `(14,1) → (14,1)` | 75,192 | 75,265 | +73 | +0.10% |
| recursive setup offload | `2 × (16,1) → (32,2)` | 87,051 | 87,057 | +6 | +0.01% |
| distributed W8R2 setup offload | `2 × (16,1) → (32,2)` | unavailable¹ | 88,527 | — | — |

¹ The baseline distributed fixture did not compile because it still passed a
`ResolvedScheduleRow` where `&FoldSchedule` was required. The API-stale fixture
was repaired during this cutover; its S/G honest lifecycle now runs and
verifies. This is not represented as a zero delta.

Across comparable cases the largest increase is 2,557 bytes (3.40%); six of
seven changes are at most 1.44%, and one is a 12-byte improvement. The global
instance-descriptor version remains unchanged because bumping it would alter
the scalar S transcripts that this appendix requires to preserve. Changed G
rows already receive new row/key digests and old P-profile selections are not
accepted by the S-only catalogs.

### Non-negotiable S preservation

Existing S behavior is the fixed side of this experiment. For every currently
generated S key, the migration must preserve exactly:

- the compact generated S row payload;
- expanded root, recursive, and terminal parameters;
- the root-final `CommittedGroupProfile` and its canonical bytes;
- tensor-projection choice and role dimension;
- setup matrix capacity and setup-prefix requirements;
- commitment and hint bytes for deterministic fixtures;
- `OpeningScheduleSelection` and row digest;
- transcript events, serialized proof bytes, and proof length; and
- commit/prove/verify acceptance behavior.

Existing S rows are inputs to the migration, not planner candidates. Generation
must pin and reuse their current root commitment profiles rather than replan or
reselect them. A generated S-row payload diff is a hard failure even if the new
row has equal cost.

Temporary differential tests should snapshot every current S row and the
tractable deterministic S lifecycles before changing P/G generation. These
tests compare the pre-migration and candidate branches structurally and by
canonical bytes. After the measurements are accepted and the migration is
complete, remove the temporary branch-comparison tests, printing code, test
counters, and any test-only production hooks. Retain the repository's ordinary
functional, drift, and protocol regression tests in their final S-only form.

### Complete current-P inventory

Before modifying the planner, derive the complete ordered P domain from every
generated family. Do not use a prose-maintained list. Partition it into:

```text
P ∩ S  layouts with an existing S row
P \ S  layouts currently supported only as prior commitments
```

For `P ∩ S`, the candidate prior profile is the existing S row's exact
root-final profile. For `P \ S` consumed by a shipped grouped row or lifecycle,
"always use S" requires extending the S authority: generate a scalar S-style
row/profile under the existing S objective without changing any existing S
row. P-registry-only shapes with no grouped consumer are removed with that
role-specific registry; this repository provides no backward-compatibility
guarantee for unused standalone parameter lookups.

The inventory must also map every current G prior descriptor to its P producer
and replacement S profile. This includes cross-configuration and recursive
cases, especially the existing fp128 Dense `(15, 2)` prior used by a OneHot
grouped row.

### Candidate G generation

Current G rows are keyed by exact P descriptor bytes, so merely changing the
commit call from P to S will make those rows unselectable. The experiment must
generate a separate candidate G catalog whose ordered prior descriptors are
the replacement S profiles.

For each current supported grouped request:

1. preserve the final group layout and prior-group order;
2. substitute each exact P descriptor with its canonical S replacement;
3. re-run grouped planning under the same policy and security bounds;
4. require the candidate row to pass all structure, SIS, setup, and verifier
   audits; and
5. record unsupported candidates as migration blockers rather than silently
   dropping them.

The baseline P/G catalogs and candidate S/G catalogs must coexist only in the
measurement harness. Production catalog replacement happens atomically after
approval; runtime selection must never guess between old and new descriptor
domains.

### Required end-to-end measurements

Run every existing lifecycle that currently commits at least one
`GroupPosition::Prior` twice with identical fields, polynomials, points, setup
seeds, transcript labels, backends, and feature graph:

```text
baseline:  prior groups use P; final selection uses current G
candidate: prior groups use S; final selection uses candidate G
```

At minimum this includes:

- same-family OneHot prior/final flows;
- same-family Dense prior/final flows;
- the P-only fp128 Dense `(15, 2)` case;
- heterogeneous Dense-prior/OneHot-final flows;
- fp32 extension-field grouped flows;
- recursive setup-offloading grouped flows;
- distributed/multi-chunk grouped flows; and
- every additional P-using test, example, benchmark, or profile workload found
  by a repository-wide caller inventory.

Both variants must commit, prove, and verify successfully. For each case, print
one stable report row containing:

| Field | Meaning |
| --- | --- |
| case | Stable fixture name |
| configuration | Exact `Cfg`/generated family |
| group layouts | Ordered prior layouts followed by the final layout |
| baseline profile source | P profile identities used by the current flow |
| candidate profile source | Replacement S profile identities |
| baseline proof bytes | Serialized current P/G proof length |
| candidate proof bytes | Serialized candidate S/G proof length |
| delta bytes | `candidate - baseline` |
| delta percent | `(candidate - baseline) / baseline * 100` |
| baseline setup fields | Current full setup envelope |
| candidate setup fields | Candidate full setup envelope |
| status | commit/prove/verify result for both variants |

Also print component-level proof-size estimates or serialized segment lengths
when available so a total-size change can be attributed to the root,
recursive, or terminal portion. Report negative deltas as improvements rather
than hiding them behind an absolute value.

The measurement output must be deterministic and checked into a reviewable
report or pasted into the change record before the temporary printing harness
is removed. "Not noticeable" is not an implicit numerical waiver: every
nonzero proof-size or setup-envelope delta must be shown, and the acceptable
absolute/percentage budget must be explicitly approved after reviewing the
table. No candidate is accepted solely because it passes verification.

### Temporary-test cleanup gate

The measurement implementation may use temporary differential tests, parallel
baseline/candidate catalogs, counters, and diagnostic printing. Before the
final production commit:

- preserve the measurement report;
- remove all temporary baseline/candidate comparison tests;
- remove diagnostic `println!`/`dbg!` output and test-only counters/hooks;
- remove temporary dual-catalog wiring and feature flags;
- regenerate the production catalogs from the accepted S-only authority; and
- run the normal repository and CI suites against only the final design.

Tests that express permanent behavior are not temporary: ordinary groups use
S, grouped rows accept exact ordered S profiles, all formerly supported P
lifecycles still run, and existing S outputs remain frozen.

### Completed production cutover

The production catalog now uses S for every ordinary commitment, includes
former `P \ S` layouts as scalar rows, and keys replanned G rows by the exact S
profiles. The current caller surface is `GroupContext`; the earlier
`GroupPosition::{Independent, Final}` surface recorded in this appendix was
subsequently removed by Appendix B. The P registry, resolver, validation cache,
identity fields, emitter output, and obsolete registry tests are removed. Book
and active multi-group documentation describe the S-only authority.

The global instance-descriptor epoch was deliberately not bumped: it is shared
by scalar S transcripts, and changing it would violate the fixed-S requirement.
Changed grouped rows are already separated by their exact row/key digests.

No compatibility shim retains P selection. Commitments made with old P
profiles are not assumed to be openable under the S-only G catalog; any need to
support them requires an explicit versioned verifier/migration design.

## Appendix B: Implemented `GroupContext` and parameter-source consolidation

### Status and public types

The public cutover was completed on 2026-08-11. `GroupPosition` is replaced by
two independent axes carried by a private-field context:

```rust
pub struct GroupContext<'a> {
    prior_groups: PriorGroupContext<'a>,
    parameter_source: GroupParameterSource<'a>,
}

pub enum PriorGroupContext<'a> {
    NoPriorGroups,
    WithPriorGroups(&'a PriorGroupProfiles),
}

pub enum GroupParameterSource<'a> {
    Scheduler,
    Explicit(&'a CommittedGroupParams),
}
```

The four supported combinations are exposed through named constructors:

```text
scheduler_without_prior_groups()
scheduler_with_prior_groups(&PriorGroupProfiles)
explicit_without_prior_groups(&CommittedGroupParams)
explicit_with_prior_groups(&PriorGroupProfiles, &CommittedGroupParams)
```

Both constructors with prior groups reject an empty owner. Borrowing preserves
the caller's ability to move that same `PriorGroupProfiles` allocation into
`SelectedProverOpeningData` after the final commitment.

### Semantic matrix

| Prior context | Parameter source | Behavior |
| --- | --- | --- |
| `NoPriorGroups` | `Scheduler` | Select and validate the exact scalar S row. |
| `WithPriorGroups` | `Scheduler` | Select the exact G row keyed by ordered prior profiles and perform full-schedule setup admission. |
| `NoPriorGroups` | `Explicit(params)` | Require validated scalar root params without catalog lookup. |
| `WithPriorGroups` | `Explicit(params)` | Require grouped root params whose embedded prior descriptors exactly match the supplied profiles in count and order. |

`Scheduler` means selection from an already-generated catalog; commitment
never runs the offline planner. Explicit grouped params reject setup-prefix
metadata, missing profiles, reordered profiles, and any descriptor mismatch.

### One commitment operation

The prover and PCS facade expose only:

```rust
pub fn commit<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
    context: GroupContext<'_>,
) -> Result<CommitOutput<Cfg::Field>, AkitaError>;
```

`commit_with_params`, public `CommitmentWithHint`, and `GroupPosition` are
removed without aliases or forwarding wrappers. Match arms normalize to one
owned-or-borrowed parameter value plus one profile. Projection, commitment
arithmetic, and `CommitOutput` assembly then occur once inline in `commit`.

Tensor projection has one authority only:
`root_tensor_projection_enabled(field, extension, A ring dimension,
source num_vars)`. There is no secondary Boolean or parameter-source override.
Consequently explicit parameters bypass catalog selection, but do not bypass
canonical tensor projection. Already-projected sources cannot use explicit
mode as a public projection bypass because the projected representation retains
the original `num_vars`.

### Validation and outcome

Permanent tests establish:

- scheduled S and G behavior remains unchanged;
- explicit scalar output matches the built-in backend oracle;
- explicit grouped output matches the scheduled grouped commitment for the
  same generated parameters;
- empty, missing, and reordered prior profiles reject;
- malformed explicit params reject before commitment arithmetic; and
- the tensor kernel runs exactly when the canonical geometry predicate enables
  it.

The PCS facade, examples, benches, tests, recursion artifact, and book use
`GroupContext`. The generated schedule catalogs are unchanged by this API-only
refactor.
