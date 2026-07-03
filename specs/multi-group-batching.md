# Spec: Multi-commitment groups and conservative-rank root batching


| Field        | Value                         |
| ------------ | ----------------------------- |
| Author(s)    |                               |
| Created      | 2026-06-17                    |
| Status       | proposed                      |
| PR           |                               |
| Book-chapter | book/src/how/configuration.md |


## Summary

Akita currently supports batching several polynomials inside one commitment
object. This spec defines the first production model for batching several
commitment groups in one root proof. The first supported shape is deliberately
narrow:

- every opened polynomial is a one-hot polynomial;
- the final group defines the shared padded opening arity, and every
  precommitted group has `num_vars <= final_group.num_vars / 2`;
- all groups are opened at one shared point;
- group sizes `K_g` may differ;
- multi-group structure exists only at the root fold;
- recursive suffix folds remain singleton after the root produces one recursive
witness commitment;
- tiered multi-group commitments remain out of scope.

> **Common-z refinement.** The root model below gives every commitment group its
> own folded `z` witness. A general refinement is tracked in
> [`heterogeneous-basis-multi-group-common-z.md`](heterogeneous-basis-multi-group-common-z.md):
> it embeds local opening digits into one logical folded witness, shares
> coordinates only when their exponent and full embedded `A` profile match, and
> automatically splits incompatible profiles into separate namespaces.

The original root model gives every commitment group its own folded `z` witness:

```text
group g has z_g, and after decomposition it contributes z_hat_g
```

All root relations are checked per group except the `D` role. The root emits one
`v` for the shared `D` relation, and `D` is sized large enough to cover every
group's `w_hat_g` witness segment.

This spec also introduces a **conservative-rank** configuration family for
standalone group commitments. A precommitted group uses a B rank that is
conservative for the maximum allowed root decomposition basis. Later, when a
final group is committed and all group shapes are known, the proof planner can
use the precommitted groups without breaking the SIS security of their B
relations.

## Motivation

The one-commitment batch shape is efficient when all polynomials are known and
committed together. It is not enough for workflows that naturally produce several
commitments over time and later want one opening proof at the same point.

Multi-group batching is therefore a workflow feature, not the preferred
proof-size path when the caller can choose the commitment shape up front. If all
polynomials are available before committing and the caller does not need separate
commitment identities, the caller should use the existing one-commitment
`batched_commit` path.

The root proof binds several algebraic objects:

```text
t_hat: commitment-side decomposed inner images
w_hat: opening-side folded witnesses
z_hat: folded response witnesses
u:     public commitment rows
v:     public D commitment rows
```

## Goals

- Support root proofs for several one-hot commitment groups opened at one shared
point.
- Allow unequal group sizes `K_g`.
- Give each group its own `z_hat_g` at the root.
- Keep recursive suffix folds singleton: the grouped root produces one next
witness commitment and the existing suffix machinery takes over.
- Add conservative-rank presets, parallel to `proof_optimized`, for standalone
precommitted groups.
- Add final-group planning metadata so the final commit can build a grouped root
proof with already committed groups.
- Freeze enough per-group root layout metadata at standalone commit time for the
final planner and verifier to reconstruct the same `t_hat_g` shape later.
- Bind group partition, frozen per-group layouts, conservative B ranks, and the
final grouped root schedule in the Fiat-Shamir instance descriptor.
- Preserve the verifier no-panic contract. Malformed group metadata, schedules,
commitments, witness shapes, and descriptors must return `AkitaError`.

## Non-Goals

- Multipoint openings. The batch still has one shared opening point.
- Dense polynomial support for the initial multi-group root batching rollout.
- Precommitted groups whose `num_vars` exceed `final_group.num_vars / 2`.
- Tiered multi-group commitments. Existing tiered paths may keep rejecting
`G > 1`.
- Recursive setup-contribution support for `G > 1` in the first rollout, unless
explicitly implemented in the same phase.
- Backward compatibility with old proof bytes or descriptor bytes.

## Usage Guidance

Use the existing one-commitment same-point batch when all polynomials are known
before committing:

```text
batched_commit([p_0, ..., p_{N-1}]) -> one commitment object -> scalar root proof
```

Use multi-group batching only when the workflow needs separate commitment
objects that are produced over time and later opened together:

```text
commit_group(group_0), ..., commit_group(group_{G-2}), commit_final(group_{G-1})
```

For `G = 1`, grouped APIs must eventually normalize to the existing scalar
schedule and proof bytes. They must not introduce a parallel planner path or
extra descriptor fields beyond the versioned grouped encoding. Until that
normalization is implemented, public grouped schedule/proof APIs must reject or
avoid exposing `G = 1` grouped requests.

## Terminology

`G`
: Number of commitment groups.

`K_g`
: Number of polynomials committed in group `g`.

`l_g`
: `log_basis` used by group `g` for gadget decomposition.

`l_max`
: Maximum allowed root `log_basis` from the configuration's basis range.

`n_b'_g`
: Conservative B rank used when committing group `g`.

`z_g`
: Folded group response before decomposition.

`z_hat_g`
: Gadget decomposition of `z_g`.

`w_hat_g`
: Decomposed group opening-side witness bound by the root `D` relation.

## Current State

The codebase has partial building blocks but not end-to-end multi-commitment
groups.

Already useful:

- `OpeningClaims` / `OpeningClaimsLayout` have been cleaned up around the new
  commitment-group design: one shared point plus an ordered list of polynomial
  group records.
- `PolynomialGroupClaims` carries the point-coordinate selection for that group
  and dense claimed evaluations for the group's committed polynomials.
- `OpeningClaimsLayout::from_group_sizes` already builds the shape-only grouped
  batch for group sizes such as `[1, 3]`, and descriptor bytes bind the group
  partition and point-variable selections.
- `batched_commit` and its input preparation can commit several group slices
with one shared scalar root layout.
- `LevelParams` has row-offset helpers such as `m_row_count_for`,
`b_inner_start`, and `a_start` that accept a commitment count.
- `generate_y` can represent several commitment row blocks when supplied the
right row slices.
- `repeated_b_commitment_rows` contains the right style of per-group B-row
computation and padding for unequal `K_g`.
- Suffix code already treats recursive levels as singleton commitments.

Still missing or scalar today:

- `PolynomialGroupLayout` is scalar and only describes one group.
- Generated table entries inline `PolynomialGroupLayout` for the final group.
- Planner root witness sizing treats the root as one group.
- `AkitaScheduleLookupKey` and `PrecommittedGroupParams` exist, but grouped
  root proof support is still incomplete.
- Conservative-rank presets and conservative B rank selection do not exist.
- Public `commit_group` and `commit_final` APIs do not exist.
- `compute_relation_quotient`, verifier ring-switch replay, terminal witness
layout, and setup-contribution input construction still contain single-group
assumptions.
- Instance descriptors bind the prove schedule and call shape, but not the
grouped final-commit shape, commitment group layouts, or conservative B ranks.

This spec describes the target state and the staged path to get there.

## Opening Claims Shape

Do not introduce a separate root-only incidence type. `OpeningClaimsLayout` is the
canonical normalized shape for one shared opening point. The old flattened
slot/routing vocabulary has been removed in favor of explicit commitment
groups:

```rust
pub struct PointVariableSelection {
    indices: Vec<usize>,
}

pub struct PolynomialGroupClaims<'a, F, C> {
    point_vars: PointVariableSelection,
    evaluations: Vec<F>,
    commitment: C,
    // ...
}

pub struct OpeningClaims<'a, F, C> {
    point: OpeningPoints<'a, F>,
    groups: Vec<PolynomialGroupClaims<'a, F, C>>,
}
```

Required constructors and accessors:

```rust
impl OpeningClaimsLayout {
    pub fn new(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError>;

    pub fn from_group_sizes(
        num_vars: usize,
        num_polys_per_commitment_group: &[usize],
    ) -> Result<Self, AkitaError>;
}

impl<'a, F: Clone, C> OpeningClaims<'a, F, C> {
    pub fn from_groups(
        point: impl Into<OpeningPoints<'a, F>>,
        groups: Vec<PolynomialGroupClaims<'a, F, C>>,
    ) -> Result<Self, AkitaError>;
    pub fn layout(&self) -> OpeningClaimsLayout;
    pub fn num_vars(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn num_groups(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;
}
```

The first supported grouped shape requires:

```text
all claims use one shared opening point
all groups are nonempty
group sizes K_g may differ
each group's point_vars is ordered, duplicate-free, and indexes into the shared point
```

`[K, K]` and `[2K]` must remain distinct at the schedule, descriptor, and
transcript layers. Descriptor bytes must also distinguish groups that use
different `PointVariableSelection` orders.

## Protocol Shape

For each group `g`, the root proves:

```text
1. B_g * t_hat_g = u_g
2. output_g
3. c_g^T * w_hat_g = a_g^T * z_hat_g
4. c_g^T * t_hat_g = A_g * z_hat_g
```

The root also proves one global D relation:

```text
5. D * concat(w_hat_0, ..., w_hat_{G-1}) = v
```

`concat` is the exact concatenation of the group witness segments. The group
segments may have different sizes. The shared `D` key must be selected with a
column width large enough for the full concatenated witness:

```text
width_D >= sum_g width(w_hat_g)
```

The grouped root design uses this concatenated D relation. It does not use per-group `v_g`
commitments. A future per-group D design must show that the extra D blocks and
extra `v_g` payloads are cheaper for a concrete workload, and it must give a
separate binding argument for the group-wise D outputs.

The root relation is a direct sum of group subrelations plus one shared D block.
The suffix sees only the resulting recursive witness commitment and remains
unchanged:

```text
grouped root -> one committed recursive witness -> singleton suffix folds
```

## Root Counts

The grouped root uses these counts:

```text
num_commitment_groups = G
num_t_vectors_total   = sum_g K_g
num_w_vectors_root    = G
num_z_vectors_root    = G
num_public_rows       = G
```

`num_w_vectors_root = G` means each group contributes its own `w_hat_g` segment.
`num_z_vectors_root = G` means each group contributes its own `z_hat_g` segment.

Recursive suffix levels continue to use:

```text
num_commitment_groups = 1
num_t_vectors         = 1
num_w_vectors         = 1
num_z_vectors         = 1
```

The root planner and proof-size formulas must keep these two worlds separate.

## Schedule Keys

Keep `PolynomialGroupLayout` as the per-group entry shape:

```rust
pub struct PolynomialGroupLayout {
    pub num_vars: usize,
    pub num_polynomials: usize,
}
```

For scheduler adoption, each group entry records that group's committed arity:

```text
num_vars      = committed arity for this group
num_t_vectors = K_g
num_w_vectors = 1
num_z_vectors = 1
```

The final group sets the shared padded opening arity used by the grouped
root. Precommitted group entries may have smaller arity; their
`PointVariableSelection` chooses the coordinates they use from the shared point.
Precommitted entries with arity greater than `final_group.num_vars / 2` must
reject. This is the integer-division bound enforced by
`AkitaScheduleLookupKey::validate`.

Use the group-batch key for final commit/prove planning:

```rust
pub struct AkitaScheduleLookupKey {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: Vec<PrecommittedGroupParams>,
}

pub struct PrecommittedGroupParams {
    pub group: PolynomialGroupLayout,
    pub m_vars: usize,
    pub r_vars: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub conservative_n_b: usize,
}
```

`PrecommittedGroupParams` records the root layout that was used to create the
group commitment. The final planner must use the same `t_hat_g` shape for that
group. In the `commit_final`/opening phase, every precommitted group must verify against the frozen
`conservative_n_b`. The final group may use a non-conservative B rank because it
is committed after the full grouped root shape is known.

The key means:

```text
Given the precommitted groups and their commitment group layouts,
what is the root configuration for committing/proving the new group together
with them?
```

The vector is a deterministic representation of the group list supplied by the
caller. Group indices are derived from this vector for transcript and descriptor
binding.

Derived aggregate counts:

```text
G                   = 1 + precommitteds.len()
num_t_vectors_total = new.num_polynomials + sum(precommitted.group.num_polynomials)
num_w_vectors_root  = G
num_z_vectors_root  = G
num_public_rows     = G
```

## Generated Schedule Keys

Generated schedule entries must inline the same shape as `AkitaScheduleLookupKey`:

```rust
pub struct GeneratedScheduleTableEntry {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: &'static [PrecommittedGroupParams],
    pub steps: &'static [GeneratedStep],
}
```

Generated lookup must compare the full group-batch key, including the
precommitted group keys, commitment group layouts, and conservative B ranks. It
must never collapse a multi-group key into a single scalar key such as
`[sum_g K_g]`.
A table miss is safe because runtime DP can derive a schedule. A false table hit
is not safe.

Initial scheduler adoption may use DP fallback for grouped keys. Once the shape stabilizes, table
generation should add a small grid:

```text
G in {1, 2, 4}
K_g in {1, 2, 4} including unequal group sizes
num_vars in supported family ranges
```

## Conservative-Rank Configuration

Conservative-rank standalone group commits use the existing one-hot
`proof_optimized` presets. The widening is selected by
`CommitmentConfig::get_params_for_group_commit`; dense backends and tiered
multi-group roots should return explicit `AkitaError`.

### Standalone Conservative Commit

For a group committed before the final grouped proof is known:

1. Pick the group log basis as the minimum allowed root `log_basis` in the
   code/config basis range:
   ```text
   l_g = l_min = min_basis(Cfg)
   ```
2. Use the existing proof-optimized schedule planner with the basis search range
   pinned to `log_basis = l_g`, for example by resolving the standalone group
   key under:
   ```text
   basis_range = (l_g, l_g)
   ```
   The planner keeps its normal proof-size and weak-binding-aware objective; it
   does not switch to a separate "minimize `t_hat_g`" objective.
3. Require a one-hot root layout in the initial standalone `commit_group` API.
4. Freeze the fields that determine the committed `t_hat_g` shape:
   ```text
   key, m_vars, r_vars, log_basis = l_g, n_a, b_width
   ```
5. Pick the highest allowed root basis:
   ```text
   l_max = max_basis(Cfg)
   ```

6. Ask the SIS estimator for the B rank required for the frozen `b_width` at the
   B-role norm induced by `l_max`. Call it `n_b'`.
7. Commit this group using:
   ```text
   m_g, r_g, n_a_g, b_width_g, n_b'_g, log_basis = l_g, num_digits_open(l_g), ...
   ```

8. Store the frozen fields and `n_b'_g` in `PrecommittedGroupParams`.

This protects the B relation for the precommitted group against any later
root-group choice whose `log_basis <= l_max`, but only for the B role and only
for the frozen `t_hat_g` shape. The final grouped root must not change the
precommitted group's `m`, `r`, `n_a`, `log_basis`, or B width. The A and D roles
must still be sized and bound according to their own actual layouts in the final
grouped root plan.

### Last Group

The last group can use a non-conservative B rank because all group dimensions are
known:

```text
commit_final(new_group, precommitteds)
```

`commit_final` must:

- build the full `AkitaScheduleLookupKey`;
- use the precommitted groups' `PrecommittedGroupParams` values and conservative
  `n_b'` values;
- commit the last group with the final grouped plan;
- materialize the grouped root schedule and singleton suffix schedule;
- return a batch plan containing the grouped key, precommitted metadata, last
group metadata, and proof schedule identity.

## Config Surface

Extend `CommitmentConfig` with grouped hooks:

```rust
fn get_params_for_group_commit(
    key: &PolynomialGroupLayout,
) -> Result<LevelParams, AkitaError>;

fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>;

fn get_params_for_grouped_batched_commitment(
    key: &AkitaScheduleLookupKey,
) -> Result<LevelParams, AkitaError>;
```

`get_params_for_group_commit` is the standalone conservative commit planner.
`runtime_schedule` is the unified scalar and grouped root planner.
`get_params_for_grouped_batched_commitment` reads the main group's root commit params from that schedule.

`GroupedRootSchedule` is a root-only plan:

```rust
pub struct GroupedRootSchedule {
    pub group_params: Vec<LevelParams>,
    pub d_key: AjtaiKeyParams,
    pub root_witness_layout: GroupedRootWitnessLayout,
    pub suffix: Schedule,
}
```

## Commit and Prove APIs

Add explicit group handles:

```rust
pub struct CommittedGroupScheduleMeta {
    pub layout: PrecommittedGroupParams,
}

pub struct CommittedGroupHandle<C, H> {
    pub schedule: CommittedGroupScheduleMeta,
    pub commitment: C,
    pub hint: H,
}
```

There is no `params_digest` in this phase. The verifier and final planner
recompute the relevant params from `PrecommittedGroupParams`, setup, and config
policy. If a later design allows non-deterministic or externally supplied group
params, it can add canonical params bytes or a digest then.

Add commit APIs:

```rust
fn commit_group<P, B>(
    setup: &Self::ProverSetup,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    polys: &[P],
) -> Result<CommittedGroupHandle<Self::Commitment, Self::CommitHint>, AkitaError>
where
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>;

fn commit_final<P, B>(
    setup: &Self::ProverSetup,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    polys: &[P],
    precommitteds: &[CommittedGroupScheduleMeta],
) -> Result<GroupBatchCommitFinal<Self::Commitment, Self::CommitHint>, AkitaError>
where
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>;
```

`commit(polys)` remains the singleton convenience helper:

```text
commit(polys) == commit_final(polys, [])
```

`batched_commit(polys)` keeps its current meaning: one commitment object bundling
many polynomials. Multi-commitment calls should use `commit_group` /
`commit_final` or a helper that calls `commit_group` for precommitted groups and
`commit_final` for the last group.

Update claim types to support commitment groups:

```rust
pub type VerifierClaims<'a, F, C> =
    (OpeningPoints<'a, F>, Vec<CommittedOpenings<'a, F, C>>);

pub type ProverClaims<'a, F, P, C, H> =
    (OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>);
```

The vector index is the commitment group index. Existing singleton helpers may
wrap one `CommittedOpenings` / `CommittedPolynomials` entry.

## Root Witness Layout

Introduce an explicit root witness layout:

```rust
pub struct GroupedRootWitnessLayout {
    pub groups: Vec<GroupRootWitnessSegment>,
    pub total_d_w_rings: usize,
    pub z_hat_segments: usize,
}

pub struct GroupRootWitnessSegment {
    pub group_index: usize,
    pub t_hat_rings: usize,
    pub w_hat_rings: usize,
    pub z_hat_rings: usize,
}
```

The grouped opening layout uses:

```text
z_hat_segments = G
total_d_w_rings = sum_g w_hat_rings_g
```

The `D` key width and `D * concat(w_hat_g) = v` relation use
`total_d_w_rings`. Prover and verifier must derive the same layout from the
grouped root schedule and group metadata.

## Root M-Row Layout

For the initial non-tiered grouped opening, the M rows are:

```text
consistency rows
D rows
for each group g:
    COMMIT rows for B_g * t_hat_g = u_g
    output rows for group g
    A rows for c_g^T * t_hat_g = A_g * z_hat_g
```

There is no shared A block in the grouped root model. Each group relation has
its own A role, even if two groups happen to use identical dimensions or the same
setup prefix widths.

Row offsets should be represented by a grouped layout object rather than by
pretending one `LevelParams` describes all groups. In any bridge implementation
that temporarily uses one shared `LevelParams` value for all groups, row sizing
must still account for `G` group blocks and `G` public rows. Hardcoding `G = 1`
in a verifier-reachable path is invalid.

## Relation Quotient Requirements

The grouped root quotient must:

- validate `commitments.len() == G`;
- validate `hints.len() == G` on the prover side;
- validate each commitment's row length against that group's params;
- produce `G` COMMIT row blocks;
- produce one `z_hat_g` segment per group;
- compute the D rows over `concat(w_hat_0, ..., w_hat_{G-1})`;
- compute per-group A/B quotient rows with the correct per-group `m`, `r`,
`log_basis`, digit depths, and ranks;
- produce a single root `w` for the suffix.

The existing extra-`z` loop in `compute_relation_quotient` is not sufficient by
itself. Extra `z` segments must be connected to group-scoped equations, not only
added into one aggregate A quotient.

## Setup Contribution

The initial grouped opening should reject:

```text
G > 1 && SetupContributionMode::Recursive
```

For Direct mode, setup contribution and verifier ring-switch replay must receive
the true group shape:

```text
num_segments = G
num_polys_per_segment = [K_0, K_1, ..., K_{G-1}]
```

Verifier code must never silently construct `num_segments = 1` for a grouped
proof.

## Instance Descriptor and Transcript

Add a descriptor section:

```rust
pub struct CommitSection {
    pub group_batch_key: AkitaScheduleLookupKey,
    pub grouped_root_schedule_digest: DescriptorDigest,
    pub singleton_suffix_schedule_digest: DescriptorDigest,
}
```

The instance descriptor must bind:

- group count;
- `num_polys_per_commitment_group`;
- claim-to-group and claim-to-poly routing;
- the full `AkitaScheduleLookupKey`;
- each precommitted group's `PrecommittedGroupParams`;
- each precommitted group's conservative B rank;
- the grouped root schedule digest;
- the singleton suffix schedule digest;
- setup seed and policy fields, including the basis range that defines `l_g` and
  `l_max`;
- one-hot chunk size and decomposition policy.

The transcript absorption order remains:

```text
instance descriptor
opening batch / group shape
commitments in group vector order
shared opening point
claim values
root messages
suffix messages
```

Changing the group partition or commitment vector order must change the
descriptor and transcript.

`AKITA_INSTANCE_DESCRIPTOR_VERSION` stays at `1` until the codebase is frozen for
audit. Pre-audit wire-format extensions (for example grouped `CallSection` fields)
land without bumping this constant. After audit freeze, incompatible layout
changes must increment it. The grouped `CommitSection` is a new top-level
descriptor field serialized after `CallSection` when Phase 2 lands.

### Canonical Encoding

Grouped metadata uses the same canonical byte helpers as the existing instance
descriptor digests in `crates/akita-types/src/descriptor_bytes.rs`:

```text
usize   -> u64 little-endian
u32     -> u32 little-endian
u128    -> u128 little-endian
usize[] -> usize length prefix, then each element in order
digest  -> 32 raw bytes (Blake2b-256 output)
```

`PrecommittedGroupParams` encodes in this fixed order:

```text
group.num_vars
group.num_polynomials
m_vars
r_vars
log_basis
n_a
conservative_n_b
```

`AkitaScheduleLookupKey` encodes in this fixed order:

```text
precommitteds.len()
for g in 0..precommitteds.len():
    PrecommittedGroupParams(precommitteds[g])
final_group PolynomialGroupLayout
```

`CommitSection` encodes in this fixed order:

```text
AkitaScheduleLookupKey
grouped_root_schedule_digest[32]
singleton_suffix_schedule_digest[32]
```

The grouped opening-batch digest in `CallSection` remains separate and uses the
existing `digest_opening_batch` encoding over:

```text
num_vars
num_claims
num_commitment_groups
for each group:
    group.num_claims
    group.point_vars.indices[]
```

Descriptor digest domain labels:

```text
opening_batch_digest      = Blake2b-256("akita/opening_batch" || canonical bytes)
effective_schedule_digest = Blake2b-256(schedule.append_descriptor_bytes(...))
commit_section_digest     = Blake2b-256("akita/commit_section" || CommitSection bytes)
```

The `commit_final`/opening phase does not add new proof-body fields beyond the
extended descriptor and the existing batched proof containers. Grouped proof
metadata lives in the descriptor and transcript, not in prover-supplied side
channels.

Malformed grouped descriptor bytes must reject before any verifier-reachable
schedule lookup, matrix prefix selection, or ring-switch replay. Exact rejection
cases:

```text
descriptor version < AKITA_INSTANCE_DESCRIPTOR_VERSION for a grouped proof -> SerializationError
precommitteds.len() == 0 && grouped proof claims G>1 -> AkitaError::InvalidProof
precommitteds.len() + 1 != G                         -> AkitaError::InvalidProof
any PrecommittedGroupParams field overflow or zero where forbidden -> AkitaError::InvalidProof
group vector order differs from commitment vector order -> AkitaError::InvalidProof
grouped_root_schedule_digest mismatch after recompute  -> AkitaError::InvalidProof
scalar [4] descriptor bytes presented as grouped [1,3] -> reject at descriptor parse
```

### Verifier Boundary

`PrecommittedGroupParams` values are not a separate prover-supplied side channel
in the `commit_final`/opening phase. They are serialized inside `CommitSection` in the instance
descriptor.
The verifier must follow this order:

1. Parse and validate the instance descriptor bytes at the current schema version.
2. Reject malformed `CommitSection` bytes before Fiat-Shamir replay continues.
3. Reconstruct `AkitaScheduleLookupKey` from `CommitSection`.
4. Recompute each `PrecommittedGroupParams` deterministically from setup, config
   policy, and the public opening batch.
5. Reject if any recomputed layout differs from the descriptor-bound layout.
6. Reject if any precommitted group's commitment row count differs from its
  frozen `conservative_n_b`.
7. Resolve the grouped root schedule from the bound key and compare the grouped
  root schedule digest.
8. Validate commitment row counts and opening-batch routing.
9. Only then run ring-switch replay and suffix verification.

The verifier must never trust handle metadata, hints, or proof-local structs for
commitment group layout fields when the descriptor already binds them.

## Setup Capacity

Setup capacity must be group-aware:

```text
max_num_vars
max_total_polys
max_commitment_groups
max_polys_per_group
```

The initial grouped setup may derive `max_commitment_groups` from `max_total_polys`, but setup
envelope scans must include partitions, not only total polynomial counts:

```text
[4] and [1, 3] are distinct setup shapes
```

Conservative-rank setup sizing must include:

- the largest conservative B footprint over supported precommit shapes;
- the grouped root D footprint for `concat(w_hat_g)`;
- the A footprints for every allowed one-hot group layout;
- ZK blinding columns if the `zk` feature is enabled;
- descriptor/cache version bumps so a single-group setup is not reused for a
grouped proof shape that needs more matrix.

## Efficiency Rules

The grouped root is expected to cost more than a scalar same-point batch when all
polynomials are known up front. The grouped root pays for:

- one `z_hat_g` segment per group;
- one public output row per group;
- one COMMIT block per group;
- one A block per group in the initial grouped relation;
- repeated B traversal and padding for unequal `K_g`;
- conservative B ranks for precommitted groups.

The main offset is the concatenated D relation. It can size D against the number
of group `w_hat_g` segments rather than the total number of polynomials, but this
does not by itself make grouped roots the preferred path for known-upfront
batches.

The planner should therefore expose three modes:

- `G = 1` normalizes to the scalar schedule once singleton grouped APIs are
exposed. In the current phase, grouped-root planning is only supported for
`G >= 2`.
- All groups known before any commit use the existing scalar same-point batch,
unless the caller explicitly needs separate commitment objects.
- Staggered workflows use `commit_group` and `commit_final`, and pay the
conservative-rank cost for precommitted groups.

## Validation Rules

At standalone `commit_group` time:

- the group must be one-hot;
- the group must be nonempty;
- `log_basis` must be `min_basis(Cfg)`;
- the `PrecommittedGroupParams` must be derived by the proof-optimized planner
  with `basis_range = (min_basis(Cfg), min_basis(Cfg))`;
- the `PrecommittedGroupParams` must determine the same `t_hat_g` shape used by
  the commit witness;
- conservative `n_b'` must pass `AjtaiKeyParams::try_new` for
  `(b_width_g, norm_B(l_max))`;
- the selected params must fit setup capacity.

At `commit_final` time:

- `precommitteds` must be well-formed group metadata;
- the full `AkitaScheduleLookupKey` must be derivable from
  `precommitteds + new`;
- each precommitted group must have
  `group.num_vars <= final_group.num_vars / 2`;
- each precommitted group must keep its `PrecommittedGroupParams` `m`, `r`,
  `log_basis`, `n_a`, and B width;
- each precommitted group must use the frozen conservative B row count in the
  grouped root relation;
- the final grouped root schedule must fit setup capacity;
- the last group commitment must match the final group's params.

At prove time:

- `OpeningClaimsLayout` must be internally consistent.
- `commitments.len() == G`.
- `hints.len() == G`.
- `sum_g K_g == num_claims`.
- Each commitment row count must match its group params.
- Tiered multi-group proofs must reject until implemented.
- Recursive setup contribution must reject for `G > 1` until implemented.

At verify time:

- The verifier must reconstruct the `AkitaScheduleLookupKey` from
  public claims and descriptor data.
- The verifier must recompute grouped root params from the key.
- The verifier must reject if a precommitted group's final root layout differs
  from its `PrecommittedGroupParams`.
- The verifier must reject if a precommitted group's B row count differs from
  its frozen conservative `n_b'`.
- The verifier must recompute root `w_len` from the grouped witness layout.
- The verifier must validate group commitment row counts before ring-switch
  replay.
- All malformed sizes must return `AkitaError`, not panic.

## Unsupported Shape Rejects


| Shape                                           | Rejection point                            | Error                                    |
| ----------------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| Tiered preset + `G > 1`                         | Root relation quotient / prove entry       | `AkitaError::InvalidSetup`               |
| Dense polynomial + `G > 1`                      | `commit_group` / `commit_final`            | `AkitaError::InvalidInput`               |
| Precommitted `num_vars > final_group.num_vars / 2` | `commit_final` / grouped key validation | `AkitaError::InvalidInput`               |
| Recursive setup contribution + `G > 1`          | Prove / verify entry                       | `AkitaError::InvalidSetup`               |
| Scalar table lookup collapsing `[1,3]` to `[4]` | Generated schedule lookup                  | table miss or `AkitaError::InvalidSetup` |
| Grouped proof with descriptor version 1         | Descriptor parse                           | `SerializationError`                     |
| `log_basis != min_basis(Cfg)` at precommit      | `commit_group`                             | `AkitaError::InvalidSetup`               |


## Rollout Plan

### Phase 0: Opening Claims Cleanup and Guards

- Clean up the old flattened commitment-group routing from `OpeningClaimsLayout`.
- Make `OpeningClaimsLayout` follow the new group design:
  - one shared `point`;
  - ordered `groups`;
  - each `CommitmentGroup` has `PointVariableSelection` plus dense `claims`.
- Keep `OpeningClaimsLayout::new` as the scalar same-bundle constructor.
- Add `OpeningClaimsLayout::from_group_sizes` for shape-only grouped batches.
- Bind group partition and point-variable selections in the instance descriptor.
- Add explicit rejects for unsupported proof paths while the grouped proof is not
implemented:
  - tiered + `G > 1`;
  - recursive setup contribution + `G > 1`;
  - dense polynomial multi-group root batching;
  - scalar table lookup that would collapse a grouped key into `[sum_g K_g]`.
- Update docs that say multi-commitment same-point folded recursion is "not yet"
to point here.

Phase 0 done means: the old slot/routing vocabulary is gone from
`OpeningClaimsLayout`, grouped batch shape and descriptor binding exist, scalar paths
still work, and unsupported grouped proof paths fail explicitly.

### Phase 1: Scheduler and commit_group

- Add `AkitaScheduleLookupKey`.
- Add generated/DP schedule resolution for grouped keys without collapsing
`[K_0, K_1, ...]` into `[sum_g K_g]`.
- Thread grouped root counts through planner and proof-size formulas:
  - `G`;
  - `num_t_vectors_total = sum_g K_g`;
  - `num_w_vectors_root = G`;
  - `num_z_vectors_root = G`;
  - `num_public_rows = G`.
- Add `PrecommittedGroupParams`.
- Add `CommittedGroupScheduleMeta` and `CommittedGroupHandle`.
- Add standalone `commit_group`.
- Implement conservative B rank selection for standalone groups.
- Keep `commit_final`, grouped opening, and folded grouped root prove guarded
until Phase 2.

Phase 1 done means: standalone `commit_group` returns commitment metadata with a
frozen `PrecommittedGroupParams`, grouped scheduler lookup/DP fallback is available
for final planning, conservative B rank selection works, and grouped schedule
negative tests pass. No grouped opening proof is required yet.

### Phase 2: commit_final and Opening

- Add `commit_final(new_group, precommitteds)`.
- Bind `CommitSection`, precommitted group metadata, final group metadata, and
the final grouped root schedule in descriptor bytes at the current schema version.
- Reconstruct and validate the full `AkitaScheduleLookupKey` from
`commit_final` metadata and public opening claims.
- Commit the final group with the final grouped plan.
- Implement grouped root witness layout with `concat(w_hat_g)`.
- Route prover root B rows through per-group B computation for `G > 1`.
- Generalize root verifier row counts and terminal witness shapes.
- Support unequal `K_g`.
- Support `SetupContributionMode::Direct`.
- Add folded non-tiered two-group one-hot same-point E2E.

Phase 2 done means: `commit_final` can combine precommitted one-hot groups with a
new final group, produce a folded grouped same-point opening proof, and verify it
for unequal `K_g`, with recursive suffix folds remaining singleton.

### Phase 3: Tables, Performance, and Broader Shapes

- Add generated table grids for common grouped one-hot shapes, including unequal
`K_g`.
- Add profile modes for grouped roots.
- Add fused B kernels if repeated per-group B traversal is too slow.
- Consider per-group D commitments only in a separate spec update with a concrete
cost model and security argument.
- Consider tiered multi-group support in a separate spec update.

Phase 3 done means: generated grouped tables exist for the supported grouped
grid, profile modes report grouped vs scalar costs, and fused B kernels land if
B-row time is a bottleneck.

## Test Matrix

### Unit Tests

- `OpeningClaimsLayout::from_group_sizes(nv, &[1, 3])` derives two groups.
- `[1, 3]` and `[4]` produce different keys and descriptor bytes.
- Generated group-batch schedule lookup compares precommitted group keys,
  commitment group layout fields, and `conservative_n_b`.
- Descriptor bytes change when a precommitted group's `PrecommittedGroupParams`
  `m`, `r`, `log_basis`, `n_a`, B width, or conservative `n_b'` changes.
- Grouped root witness layout reports `z_hat_segments = G`.
- Grouped root witness layout reports `total_d_w_rings = sum_g w_hat_rings_g`.
- Root terminal witness shape uses `G` segments at level 0 and singleton segments
  below level 0.
- Conservative B rank uses the frozen precommit geometry and norm from
  `l_max`.
- `log_basis != min_basis(Cfg)` rejects.

### Commit Tests

- `commit_group(group0)` returns metadata for a standalone conservative commit.
- `commit_group(group1)` does not require `group0` metadata.
- `commit_final(group_last, precommitteds)` returns a grouped batch plan.
- `commit_final` rejects if a precommitted group metadata record does not
  recompute to its `PrecommittedGroupParams`.
- Undersized conservative `n_b` rejects.
- Exceeding setup group capacity rejects.
- Dense polys reject for `G > 1`.

### Prove / Verify Tests

- Root-direct two-group one-hot same-point round trip.
- Folded non-tiered two-group one-hot same-point round trip.
- Folded non-tiered unequal group sizes, for example `[1, 3]`.
- Suffix remains singleton after a grouped root.
- Serialize/deserialize grouped proofs round trip with the current descriptor schema.
- Non-canonical `CommitSection` length prefix rejects.
- Reordered precommitted group vector rejects.
- Duplicate or missing precommitted metadata rejects.
- Unknown descriptor version rejects for grouped proofs.
- Scalar `[4]` descriptor bytes do not verify as grouped `[1, 3]`.
- Swapping group commitments rejects.
- Tampering group `1` opening rejects.
- Tampering group `1` hint or `t_hat` segment rejects.
- Truncating one group's commitment rows rejects.
- Changing a precommitted group's `PrecommittedGroupParams` rejects.
- Changing a precommitted group's conservative `n_b'` rejects.
- Descriptor `[1, 3]` with proof `[4]` rejects.
- Tiered two-group proof rejects with a clear error.
- Recursive setup-contribution two-group proof rejects until supported.

### Planner and Table Tests

- Grouped table lookup misses rather than collapsing to a scalar `[sum_g K_g]`
key.
- Generated grouped entries, once emitted, match DP.
- Root `w_len` from planner matches prover-built grouped witness length.
- Singleton `G = 1` grouped APIs reject or remain unexposed until scalar
normalization lands; once exposed, their schedules remain byte-identical to the
scalar path.
- Setup envelope scans include unequal partitions such as `[1, 3]`.

### Performance Tests

Compare grouped `G = 2, K = [1, 3]` against scalar `[4]` under the active
one-hot preset. Collect proof bytes, B-row time, DP fallback time, and
descriptor size.

Baseline scalar command:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=28 cargo run --release --example profile
```

Grouped command (after profile mode exists):

```bash
AKITA_MODE=onehot_fp128_d64_grouped AKITA_NUM_VARS=28 \
  AKITA_GROUP_SIZES=1,3 cargo run --release --example profile \
  --no-default-features --features parallel,profile-ci
```

Pass criteria for the spec gate:

```text
grouped proof bytes >= scalar [4] proof bytes for the same num_vars
grouped B-row time is reported separately from scalar baseline
grouped descriptor size > scalar descriptor size when G > 1
DP fallback time for grouped keys is reported even when tables miss
```

Track B-row time from repeated per-group B traversal.
Track proof-size delta from extra `z_hat` segments and wider `r` tails.
Track DP fallback time and table hit rate.

## Security Notes

Conservative B rank is necessary but not sufficient by itself. The security
argument also depends on:

- descriptor binding of the full group-batch key;
- verifier recomputation of grouped root params from that key;
- group partition binding;
- frozen precommit layout binding for every precommitted group;
- equality between the final precommitted-group B row count and the frozen
conservative `n_b'`;
- setup capacity covering the selected matrix prefixes;
- D rank and width sized for the full concatenated root witness;
- rejecting unimplemented tiered and recursive setup-contribution combinations.

No verifier path may trust prover-supplied group metadata without recomputing it
from public claims, setup, and config policy.

## Alternatives Considered

### Scalar re-bundling when all polynomials are known

The caller can use the existing same-point `batched_commit` path and commit all
polynomials in one commitment object. This is the preferred path when the
polynomials are available at the same time and the caller does not need separate
commitment identities.

This path avoids conservative B ranks and avoids per-group `z_hat_g`, COMMIT,
and A blocks. It does not support workflows where earlier commitments must keep
their original identity.

### Known-final-schedule precommit

A caller can avoid conservative rank when it knows the final grouped root layout
before committing the earlier groups. In that mode, each group commits with the
actual final root layout rather than the `l_max` B norm.

This is cheaper than conservative precommit, but it is only safe when the caller
binds the final grouped root key before the first group commit.

### Shared A for identical group layouts

Groups with identical dimensions could share one A block and use a fresh
group-batching challenge to combine their A equations. This could reduce the
`G * n_a` A-row cost in symmetric workloads.

This is not part of the initial grouped opening. It needs a separate soundness
argument because the current design isolates each group with its own A relation.

### Per-group D outputs

The design could replace `D * concat(w_hat_g) = v` with one `D_g * w_hat_g = v_g` per group. This can make each D relation narrower, but it also sends more D
outputs and adds more relation rows.

The initial grouped opening keeps one concatenated D relation. A per-group D design should only be
adopted if a later cost model shows a win for a supported workload.

### Two-phase prove and link

The prover could prove each group separately and then prove a linking statement
that combines the group openings. This keeps each group close to the singleton
path, but it adds extra transcript and proof material. It also moves complexity
into a new linking proof.

This spec does not choose that shape because the grouped root keeps one opening
proof and one recursive suffix.

## Resolved Design Decisions

The following choices close the open questions from the first draft:

1. **Public API shape:** expose `commit_final` directly. Do not add a separate
  `finalize_group_batch` sugar layer in the initial API rollout.
2. **Setup capacity:** derive `max_commitment_groups` from `max_total_polys` in
  the initial grouped setup. Add an explicit setup API field only if envelope
  scans show a need.
3. **Recursive setup contribution:** explicitly delay generalization until
  a later root-layout generalization. The `commit_final`/opening phase keeps the
  existing reject.
4. **Planner key exposure:** keep `AkitaScheduleLookupKey` as an
  internal planner input behind `commit_group` / `commit_final`. Public callers
   use `CommittedGroupScheduleMeta` and grouped claim vectors.

## Open Questions

None for the initial grouped implementation. Revisit only if tiered multi-group or
known-final-schedule precommit become active follow-up specs.
