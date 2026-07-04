# Spec: Multi-commitment groups and conservative-rank root batching


| Field     | Value                              |
| --------- | ---------------------------------- |
| Author(s) |                                    |
| Created   | 2026-06-17                         |
| Status    | scheduler/final commit implemented |
| PR        |                                    |
| Book      | configuration chapter              |


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
- multi-group structure exists only at the root level;
- folded grouped roots hand off to singleton recursive suffix folds after the
  root produces one recursive witness commitment;
- tiered multi-group commitments remain out of scope.

The root model gives every commitment group its own folded `z` witness:

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
- Derive final-group planning state from public opening shape, setup, and config
policy so the final commit can build a grouped root proof with already committed
groups.
- Bind group partition through `OpeningClaimsLayout` / `opening_batch_digest` and
bind the final grouped root schedule through the existing effective schedule
digest.
- Preserve the verifier no-panic contract. Malformed group shapes, schedules,
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
conservative batched_commit(group_0), ..., conservative batched_commit(group_{G-2})
commit_final_group(group_{G-1}, [key_0, ..., key_{G-2}])
```

For `G = 1`, schedule lookup already normalizes to the scalar path through
`AkitaScheduleLookupKey::single`. The current public grouped commitment API is
final-commit-only; grouped opening proofs remain unimplemented. When grouped
proofs land, they must preserve scalar normalization rather than adding a
parallel singleton proof path.

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

The codebase has implemented the grouped scheduler and the commitment-side
final-group path. It still does not implement end-to-end multi-group opening
proofs.

Implemented now:

- `OpeningClaims` / `OpeningClaimsLayout` record one shared opening point plus
  ordered polynomial groups. `PolynomialGroupClaims` carries the point-variable
  selection, claimed evaluations, and commitment for one group.
- `PolynomialGroupLayout`, `PrecommittedGroupParams`, and
  `AkitaScheduleLookupKey` exist in `akita-types`.
- `CommitmentConfig::runtime_schedule` resolves the unified
  `AkitaScheduleLookupKey`. A scalar key is represented as
  `AkitaScheduleLookupKey::single(final_group)` and delegates to the scalar
  scheduler. A grouped key goes through generated group-batch lookup first, then
  DP fallback.
- `find_group_batch_schedule` builds grouped root-direct or folded-root
  schedules for one-hot, non-tiered configs. The grouped root `LevelParams` holds
  the final group in the normal root fields and all precommitted groups in
  `precommitted_groups`.
- Grouped root planning sizes the shared `D` key over one `w_hat_g` segment per
  commitment group, not one segment per polynomial.
- Generated tables use one entry shape for scalar and grouped rows; selected
  one-hot families emit rows whose `precommitteds` list is nonempty.
- `ConservativeCommitmentConfig<Cfg>` derives the standalone conservative
  precommit layout at `min_basis`, widens the B rank for `max_basis`, and can be
  used with the ordinary `batched_commit` API for independent precommitted
  groups.
- `commit_final_group` is exposed through `akita-prover` and the public
  `CommitmentProver` / PCS scheme surface. It validates the final group,
  reconstructs precommitted `PrecommittedGroupParams` values from
  `PolynomialGroupLayout`s under `ConservativeCommitmentConfig<Cfg>`,
  resolves grouped params through `Cfg::get_params_for_grouped_batched_commitment`,
  applies the same tensor root-projection decision as the grouped final schedule,
  and emits the final commitment plus hint.
- Setup-envelope sizing includes conservative commitments for eligible
  proof-optimized one-hot configs.

Still future / guarded:

- Grouped opening proofs are not implemented.
- `batched_prove` rejects `G > 1` before schedule lookup. Tiered,
  recursive-setup-contribution, and dense grouped roots return specific errors;
  otherwise the prover returns the generic grouped-root unsupported error.
- `batched_verify` rejects `G > 1` before schedule lookup, with tiered and
  recursive setup-contribution returning specific setup errors and generic
  grouped claims returning `AkitaError::InvalidProof`.
- Grouped root relation quotient, verifier ring-switch replay, and setup
  contribution remain future work. Folded grouped roots are planned only when the
  root can hand off to a singleton recursive suffix; immediately terminal
  grouped root folds remain guarded.
- There is no separate descriptor `CommitSection`, and the design should not add
  one for this flow. The current descriptor binds call shape through
  `CallSection.opening_batch_digest` and binds the materialized grouped schedule
  through `PlanSection.effective_schedule_digest`; that schedule digest includes
  grouped-root `LevelParams.precommitted_groups` when a grouped schedule is
  materialized. `commit_final_group` itself returns only the final commitment and
  hint.

This spec records the implemented scheduler/final-commit behavior and the
remaining grouped opening work.

## Opening Claims Shape

Do not introduce a separate root-only incidence type. `OpeningClaimsLayout` is the
canonical normalized shape for one shared opening point. The old flattened
slot/routing vocabulary has been removed in favor of explicit commitment-group
shape records:

```rust
pub struct PointVariableSelection {
    indices: Vec<usize>,
}

pub struct PolynomialGroupLayout {
    num_vars: usize,
    num_polynomials: usize,
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

Implemented constructors and accessors:

```rust
impl OpeningClaimsLayout {
    pub fn new(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError>;

    pub fn from_group_sizes(
        num_vars: usize,
        num_polys_per_commitment_group: &[usize],
    ) -> Result<Self, AkitaError>;
    pub fn from_groups(groups: Vec<PolynomialGroupLayout>) -> Result<Self, AkitaError>;
    pub fn check(&self) -> Result<(), AkitaError>;

    pub fn max_num_vars(&self) -> usize;
    pub fn groups(&self) -> &[PolynomialGroupLayout];
    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;
    pub fn opening_batch_digest(&self) -> DescriptorDigest;
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
custom non-prefix point-variable routing is still rejected by current descriptors
```

`[K, K]` and `[2K]` must remain distinct at the schedule, descriptor, and
transcript layers. Current descriptor bytes bind the normalized prefix
point-variable selections derived from each group's active arity.

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
num_w_vectors_root    = sum_g W_g
num_z_vectors_root    = G
num_z_segments        = G
```

`W_g` is the number of opening-side `w_hat` vectors contributed by group `g`.
The current first rollout has `W_g = 1` for every group, so
`num_w_vectors_root = G` in the implemented scheduler. `num_z_segments = G`
means each group contributes one folded response segment `z_hat_g`. Public
openings bind through the fused trace term in stage-2 sumcheck, not through
dedicated M-matrix rows (see
[`specs/commitment-compression-cutover.md`](commitment-compression-cutover.md)).

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

It describes one commitment group's opening geometry:

```text
num_vars        = committed/opened arity for this group
num_polynomials = K_g
```

The final group sets the shared padded opening arity used by the grouped
root. Precommitted group entries may have smaller arity; their
`PointVariableSelection` chooses the coordinates they use from the shared point.
Precommitted entries with arity greater than `final_group.num_vars / 2` must
reject. This is the integer-division bound enforced by
`AkitaScheduleLookupKey::validate`.

`AkitaScheduleLookupKey` is the common key shape for scalar and grouped
planning:

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
group. In the `commit_final_group`/opening phase, every precommitted group must
verify against the frozen `conservative_n_b`. The final group may use a
non-conservative B rank because it is committed after the full grouped root
shape is known.

The public final-commit API accepts precommitted `PolynomialGroupLayout`s and
recomputes `PrecommittedGroupParams` internally under
`ConservativeCommitmentConfig<Cfg>`.

The key means:

```text
Given the precommitted groups and their frozen group params,
what is the root configuration for committing/proving the new group together
with them?
```

The vector is a deterministic representation of the group list supplied by the
caller. Group indices are derived from this vector for transcript and descriptor
binding.

Scalar same-point schedules use:

```rust
AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, K))
```

That form has an empty `precommitteds` vector and delegates byte-for-byte to the
scalar scheduler.

Derived aggregate counts:

```text
G                   = 1 + precommitteds.len()
num_t_vectors_total = final_group.num_polynomials + sum(precommitted.group.num_polynomials)
num_w_vectors_root  = sum_g W_g
num_z_vectors_root  = G
num_z_segments      = G
```

For the current supported grouped root, each group contributes one `w_hat_g`, so
`sum_g W_g = G`. `num_z_segments` is the witness `z_folded` segment count, not
an M-row count. Public openings bind through the fused trace term, not through
dedicated M-matrix rows.

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
precommitted group params and conservative B ranks. It
must never collapse a multi-group key into a single scalar key such as
`[sum_g K_g]`.
A table miss is safe because runtime DP can derive a schedule. A false table hit
is not safe.

The runtime lookup order is:

```text
validate AkitaScheduleLookupKey
if precommitteds is empty:
    resolve scalar schedule
else if a generated entry with matching precommitteds exists:
    expand and validate that grouped row
else:
    run grouped DP fallback
```

Current generated group-batch tables are emitted only for eligible one-hot,
non-tiered families that opt in through `GeneratedFamily.emit_group_batch`.
The emitter enumerates main keys from the normal generated family key grid
(`num_polynomials in {1, 4}` today), sets each precommitted group's `num_vars`
to `main.num_vars / 2`, and emits one- or two-precommit patterns:

```text
precommitted group counts: 1 or 2
first precommitted K:      1
second precommitted K:     max(main.K / 2, 1)
```

The generated row is kept only if conservative precommit params and grouped DP
both succeed. Table misses remain safe because runtime DP can derive a schedule.
A false table hit is not safe.

Follow-up table generation can broaden the grid:

```text
G in {1, 2, 4}
K_g in {1, 2, 4} including more unequal group sizes
num_vars in supported family ranges
```

## Conservative-Rank Configuration

Conservative-rank standalone group commits use the existing one-hot
`proof_optimized` presets through `ConservativeCommitmentConfig<Cfg>`. The
adapter overrides ordinary scalar schedule/commit layout selection so
precommitted groups can be produced with the existing `batched_commit` API while
using a B rank conservative for the parent config's maximum root basis. Dense
backends and tiered conservative/grouped roots return explicit `AkitaError`.

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
3. Require a one-hot root layout in the conservative precommit path.
4. Freeze the fields that determine the committed `t_hat_g` shape:
   ```text
   key, m_vars, r_vars, log_basis = l_g, n_a
   ```
5. Pick the highest allowed root basis:
   ```text
   l_max = max_basis(Cfg)
   ```

6. Ask the SIS estimator for the B rank required for the derived B width
   `params.b_key.col_len()` at the B-role norm induced by `l_max`. Call it
   `n_b'`.
7. Commit this group using:
   ```text
   m_g, r_g, n_a_g, derived_B_width_g, n_b'_g, log_basis = l_g, num_digits_open(l_g), ...
   ```

8. Store the frozen fields and `n_b'_g` in `PrecommittedGroupParams`.

This protects the B relation for the precommitted group against any later
root-group choice whose `log_basis <= l_max`, but only for the B role and only
for the frozen `t_hat_g` shape. The final grouped root must not change the
precommitted group's `m`, `r`, `n_a`, `log_basis`, or derived B width. The A and
D roles must still be sized and bound according to their own actual layouts in
the final grouped root plan.

### Last Group

The last group can use a non-conservative B rank because all group dimensions are
known:

```text
commit_final_group(new_group, precommitted_layouts)
```

`commit_final_group` currently:

- derives the final group key from `new_group` through the ordinary
  `batched_commit` input rules;
- recomputes each precommitted group's conservative `PrecommittedGroupParams`
  from its `PolynomialGroupLayout` under `ConservativeCommitmentConfig<Cfg>`;
- builds the full `AkitaScheduleLookupKey`;
- resolves the grouped final-root commit params through
  `Cfg::get_params_for_grouped_batched_commitment`;
- commits the last group with the final grouped plan and returns only the final
  commitment plus hint.

## Config Surface

`CommitmentConfig` now has a unified grouped/scalar schedule entry point plus a
root-params accessor for final grouped commitments:

```rust
fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>;

fn get_params_for_grouped_batched_commitment(
    key: &AkitaScheduleLookupKey,
) -> Result<LevelParams, AkitaError>;
```

`runtime_schedule` delegates scalar keys to `resolve_schedule` and grouped keys
to `resolve_group_batch_schedule`; both paths validate catalog identity on table
hits and fall back to DP on misses. `get_params_for_grouped_batched_commitment`
reads the main/final group's root commit params from the first schedule step
(`Fold.params` or root-direct `Direct.params`).

Standalone conservative precommit scheduling is not a public trait hook. It is
implemented by `ConservativeCommitmentConfig<Cfg>`, which overrides
`get_params_for_prove` / `get_params_for_batched_commitment` and uses
crate-private helpers to plan at `min_basis` and widen B for `max_basis`.

There is no separate `GroupedRootSchedule` type in the implementation. A grouped
root is represented by the first schedule step's `LevelParams`:

```rust
pub struct GroupRootParams {
    pub layout: PrecommittedGroupParams,
    pub a_key: AjtaiKeyParams,
    pub b_key: AjtaiKeyParams,
    pub num_blocks: usize,
    pub block_len: usize,
    pub num_digits_commit: usize,
    pub num_digits_open: usize,
    pub num_digits_fold_one: usize,
}

pub struct LevelParams {
    // normal root fields describe the final/new group
    pub a_key: AjtaiKeyParams,
    pub b_key: AjtaiKeyParams,
    pub d_key: AjtaiKeyParams,
    pub num_blocks: usize,
    pub block_len: usize,
    pub m_vars: usize,
    pub r_vars: usize,
    pub log_basis: u32,
    // ...
    pub precommitted_groups: Vec<GroupRootParams>,
}
```

When `precommitted_groups` is nonempty, the level is a grouped root. The normal
`a_key`, `b_key`, and block fields describe the final group; each
`GroupRootParams` describes one precommitted group; and `d_key` is the shared D
matrix over all group `w_hat_g` segments.

## Commit and Prove APIs

The implemented commitment API is key-based. The design does not need public
`CommittedGroupHandle`, `CommittedGroupScheduleMeta`, or `params_digest` types in
this or later phases. Precommitted groups are ordinary conservative commitments;
their layout is deterministically reconstructed from `PolynomialGroupLayout`,
setup, and config policy. Opening claims already have transcript and descriptor
plumbing through `OpeningClaims::append_to_transcript` and
`OpeningClaimsLayout::opening_batch_digest`.

Implemented final-group API:

```rust
fn commit_final_group<P, B>(
    setup: &Self::ProverSetup,
    polys: &[P],
    stack: &UniformProverStack<'_, F, B, D>,
    precommitteds: Vec<PolynomialGroupLayout>,
) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
where
    P: RootCommitPoly<F, D>,
    B: RootCommitBackend<F, P, Self::ExtField, D>;
```

Precommitted groups use:

```rust
type PrecommitCfg = ConservativeCommitmentConfig<Cfg>;
PrecommitScheme::batched_commit(setup, group_polys, stack)
```

The caller keeps the matching `PolynomialGroupLayout` for each
precommitted group. The final-group path then:

- validates the final group with `prepare_batched_commit_inputs`: nonempty,
  padded to the maximum arity in that final bundle, and setup capacity respected;
- rejects an empty `precommitteds` list;
- reconstructs precommitted layouts by resolving each key under
  `ConservativeCommitmentConfig<Cfg>::get_params_for_batched_commitment`;
- builds an `AkitaScheduleLookupKey` from those layouts and the final group key;
- validates `precommitted.group.num_vars <= final_group.num_vars / 2`;
- resolves grouped params through `Cfg::get_params_for_grouped_batched_commitment`;
- validates setup footprint and one-hot chunk size for the final group;
- applies tensor root projection when the grouped final schedule starts with a
  fold and the field tower supports root tensor projection;
- commits the final group with the grouped params and returns the final
  commitment plus hint.

`batched_commit(polys)` keeps its current meaning: one commitment object bundling
many polynomials under a scalar schedule. The conservative adapter changes the
schedule policy for this same API; it does not add a separate precommit method.

```text
ConservativeCommitmentConfig<Cfg>::batched_commit(group)
    == one ordinary commitment object with conservative B rank
```

`commit_final_group` is a commitment-only endpoint. Phase 2 grouped opening work
should consume the existing grouped opening-batch vectors, commitments, and hints
in group order; it should not introduce a public handle or params digest side
channel.

Claim inputs already support group order through `OpeningClaims` and
`ProverOpeningData`:

```rust
fn batched_verify<T: Transcript<F>>(
    proof: &Self::BatchedProof,
    setup: &Self::VerifierSetup,
    transcript: &mut T,
    claims: OpeningClaims<'_, Self::ExtField, &Self::Commitment>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>;

fn batched_prove<'a, T, P, B>(
    setup: &Self::ProverSetup,
    claims: ProverOpeningData<'a, Self::ExtField, P, F, D>,
    stacks: &'a impl LevelProveStacks<'a, F, D, Commit = B, Opening = B, Tensor = B, RingSwitch = B>,
    transcript: &mut T,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<Self::BatchedProof, AkitaError>;
```

The group vector index is the commitment group index. Existing singleton helpers
wrap one group entry.

## Root Witness Layout

The current implementation has scheduler-side grouped witness sizing, not yet a
serialized/prover-side `GroupedRootWitnessLayout` object. A grouped schedule can
be root-direct (`Step::Direct` with `params: Some(grouped_root_params)`) or
fold-rooted (`Step::Fold` followed by a singleton recursive suffix).

For each group, the planner prices:

```text
e_hat_g = num_blocks_g * num_digits_open_g
t_hat_g = K_g * num_blocks_g * n_a_g * num_digits_open_g
z_hat_g = block_len_g * num_digits_commit_g * num_digits_fold_g
```

For a folded grouped root, the grouped root's next recursive witness ring count
is:

```text
sum_g (e_hat_g + t_hat_g + z_hat_g)
+ r_tail(grouped root M rows)
```

`grouped_root_next_w_len` returns this count multiplied by `ring_dimension`,
matching the schedule's field-element witness lengths.

For a grouped root-direct schedule, the direct witness length is the sum of raw
group witness lengths:

```text
sum_g K_g * 2^{num_vars_g}
```

The implemented D width uses one `w_hat_g` segment per group:

```text
d_width = decomposed_w_ring_count(main.num_digits_open, main.num_blocks, 1)
        + sum_precommitted group.d_segment_width()
```

For Phase 2 proof construction, introduce an explicit root witness layout:

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

The proof layout must use `z_hat_segments = G` and
`total_d_w_rings = sum_g w_hat_rings_g`. Prover and verifier must derive the
same layout from the grouped root schedule and opening batch.

## Root M-Row Layout

For scheduler sizing of the initial non-tiered grouped opening, the grouped-root
M-row count is:

```text
1 consistency row
final/main group A rows
final/main group B rows
for each precommitted group:
    A rows for that group
    B rows for that group
optional D rows, present for MRowLayout::WithDBlock
```

This matches the scalar post-PR1 layout (`consistency | A | B | D`) extended
per group: each group relation has its own A and B blocks, and the shared D block
trails all group blocks. Public openings bind through the fused trace term, not
through M rows.

There is no shared A block in the grouped root model. Each group relation has
its own A role, even if two groups happen to use identical dimensions or the same
setup prefix widths.

Exact prover/verifier row offsets are still Phase 2 work. The current scheduler
stores the params needed to derive them in the root `LevelParams`: the normal
fields describe the final group, and `precommitted_groups` describes the earlier
groups. Scalar row-offset helpers reject grouped roots today. Any future proof or
verifier path that consumes grouped params must account for `G` group blocks and
one shared D block. Hardcoding `G = 1` in a verifier-reachable grouped path is
invalid.

## Phase 2 Relation Quotient Requirements

The grouped root quotient must:

- validate `commitments.len() == G`;
- validate `hints.len() == G` on the prover side;
- validate each commitment's row length against that group's params;
- bind public openings through the fused trace term for the whole grouped batch;
- produce `G` B (commitment) row blocks;
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

### Phase 1 Descriptor Binding

The current descriptor has `AlgebraSection`, `SetupSection`, `PlanSection`, and
`CallSection`. There is no top-level `CommitSection`, and this grouped flow
should not add one.

`CallSection` binds the public grouped opening shape:

```text
num_polys
num_commitment_groups
num_polys_per_commitment_group
point_variable_selections
basis_mode
opening_point_arity
opening_batch_digest
```

`OpeningClaimsLayout::opening_batch_digest` uses:

```text
num_vars
num_polynomials
num_commitment_groups
for each group:
    group.num_polynomials
    prefix point-variable indices as a length-prefixed usize vector
```

`PlanSection.effective_schedule_digest` binds
`schedule.append_descriptor_bytes(...)`. When a grouped schedule is materialized,
the root `LevelParams` descriptor includes `precommitted_groups`; each
`GroupRootParams` descriptor includes its frozen `PrecommittedGroupParams`, A key,
B key, block geometry, and digit counts.

`PrecommittedGroupParams` descriptor bytes currently encode:

```text
group.num_vars
group.num_polynomials
m_vars
r_vars
log_basis
n_a
conservative_n_b
```

Changing the group partition or normalized point-variable arity changes the
opening-batch digest. Changing the generated grouped schedule row or
precommitted group params changes the effective schedule digest. Both digests
are already part of the instance descriptor absorbed by the transcript.

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

`AKITA_INSTANCE_DESCRIPTOR_VERSION` stays at `1` until the codebase is frozen for
audit. Pre-audit wire-format extensions land without bumping this constant.
After audit freeze, incompatible layout changes must increment it.

### No Separate Commit Section

Grouped opening should not add a `CommitSection` or a separate params digest. The
existing descriptor already has the two bindings needed for this shape:

- `CallSection.opening_batch_digest` binds the public grouped opening shape,
  including group count, per-group polynomial counts, and point-variable
  selections.
- `PlanSection.effective_schedule_digest` binds the materialized schedule. For a
  grouped root, the schedule descriptor includes the root `LevelParams`, its
  `precommitted_groups`, each `PrecommittedGroupParams`, conservative B ranks, A/B
  keys, block geometry, and digit counts.

Setup seed and policy fields, including the basis range that defines `l_g` and
`l_max`, remain bound through the existing `SetupSection` / `AlgebraSection`
descriptor fields. One-hot chunk size and decomposition policy remain bound
through the existing level/schedule descriptor bytes.

### Canonical Encoding

Grouped opening and schedule encodings use the same canonical byte helpers as
the existing instance descriptor digests in
`crates/akita-types/src/descriptor_bytes.rs`:

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

Grouped schedule precommitted params encode in transcript order through the
materialized root `LevelParams`:

```text
precommitteds.len()
for g in 0..precommitteds.len():
    PrecommittedGroupParams(precommitteds[g])
```

The grouped opening-batch digest in `CallSection` remains separate and uses the
existing `opening_batch_digest` encoding over:

```text
num_vars
num_polynomials
num_commitment_groups
for each group:
    group.num_polynomials
    prefix point-variable indices as a length-prefixed usize vector
```

Descriptor digest domain labels:

```text
effective_schedule_digest = Blake2b-256(schedule.append_descriptor_bytes(...))
```

The grouped opening phase should not add new proof-body fields beyond the
existing descriptor and batched proof containers. Grouped proof shape lives in
the opening batch and effective schedule, not in prover-supplied side channels.

Malformed Phase 2 grouped descriptor bytes must reject before any
verifier-reachable matrix prefix selection or ring-switch replay. Exact
rejection cases:

```text
schedule.precommitted_groups.len() + 1 != opening_batch.G -> AkitaError::InvalidProof
any PrecommittedGroupParams field overflow or zero where forbidden -> AkitaError::InvalidProof
unknown descriptor version                           -> SerializationError
group vector order differs from commitment vector order -> AkitaError::InvalidProof
effective_schedule_digest mismatch after recompute      -> AkitaError::InvalidProof
scalar [4] opening-batch digest presented as grouped [1,3] -> AkitaError::InvalidProof
```

### Phase 2 Verifier Boundary

`PrecommittedGroupParams` values are not a separate prover-supplied side channel in
the Phase 2 grouped opening phase. They are derived from the public
`OpeningClaimsLayout`, setup, and config policy, then bound indirectly through
the effective schedule digest.
The verifier must follow this order:

1. Parse and validate the instance descriptor bytes at the current schema version.
2. Validate that `CallSection.opening_batch_digest` matches the public
   `OpeningClaimsLayout`.
3. Reconstruct `AkitaScheduleLookupKey` from the public opening batch, setup, and
   config policy.
4. Resolve the grouped root schedule from that key and compare
   `PlanSection.effective_schedule_digest`.
5. Reject if any schedule-derived precommitted layout differs from the layout
   recomputed from config policy.
6. Reject if any precommitted group's commitment row count differs from its
  frozen `conservative_n_b`.
7. Validate commitment row counts and opening-batch routing.
8. Only then run ring-switch replay and suffix verification.

The verifier must never trust handles, hints, proof-local structs, or a separate
digest for commitment group layout fields; the opening batch and effective
schedule are the sources of truth.

## Setup Capacity

Current setup capacity remains expressed as:

```text
max_num_vars
max_num_batched_polys
max_setup_len
```

For eligible one-hot, non-tiered proof-optimized configs, setup-envelope sizing
inflates `max_setup_len` with conservative standalone commitment footprints.
Grouped root setup capacity is still represented through the selected schedule's
effective `LevelParams`, not a separate public `max_commitment_groups` field.

Phase 2 setup envelope scans must include partitions, not only total polynomial
counts:

```text
[4] and [1, 3] are distinct setup shapes
```

Conservative-rank setup sizing must include:

- the largest conservative B footprint over supported precommit shapes;
- the grouped root D footprint for `concat(w_hat_g)`;
- the A footprints for every allowed one-hot group layout;
- descriptor/cache version bumps or schedule-digest changes so a single-group
setup is not reused for a grouped proof shape that needs more matrix.

## Efficiency Rules

The grouped root is expected to cost more than a scalar same-point batch when all
polynomials are known up front. The grouped root pays for:

- one `z_hat_g` segment per group;
- one B block per group;
- one A block per group in the initial grouped relation;
- repeated B traversal and padding for unequal `K_g`;
- conservative B ranks for precommitted groups.

The main offset is the concatenated D relation. It can size D against the number
of group `w_hat_g` segments rather than the total number of polynomials, but this
does not by itself make grouped roots the preferred path for known-upfront
batches.

The planner exposes three modes:

- `G = 1` is `AkitaScheduleLookupKey::single(...)` and delegates to the scalar
  schedule.
- All groups known before any commit use the existing scalar same-point batch,
unless the caller explicitly needs separate commitment objects.
- Staggered workflows currently use `ConservativeCommitmentConfig<Cfg>` plus
  ordinary `batched_commit` for precommitted groups, then `commit_final_group`
  for the final group. They pay the conservative-rank cost for precommitted
  groups.

## Validation Rules

At conservative precommit time:

- the group must be one-hot;
- the group must be nonempty;
- `log_basis` must be `min_basis(Cfg)`;
- the `PrecommittedGroupParams` must be derived by the proof-optimized planner
  with `basis_range = (min_basis(Cfg), min_basis(Cfg))`;
- the `PrecommittedGroupParams` must determine the same `t_hat_g` shape used by
  the commit witness;
- conservative `n_b'` must pass `AjtaiKeyParams::try_new` for
  `(derived_B_width_g, norm_B(l_max))`;
- the selected params must fit setup capacity.

At Phase 1 grouped schedule lookup time:

- `AkitaScheduleLookupKey::single(final_group)` must delegate to the scalar
  scheduler.
- `precommitteds` must be well-formed derived group params.
- The full grouped key must not be collapsed into a scalar total-polynomial key.
- Each precommitted group must have
  `group.num_vars <= final_group.num_vars / 2`.
- Dense and tiered grouped roots must return `AkitaError::InvalidSetup`.
- Grouped folded roots must hand off to a singleton recursive suffix; grouped
  terminal root folds remain rejected until the terminal witness layout is
  implemented.

At current `commit_final_group` time:

- `precommitteds` must be a nonempty list of well-formed
  `PolynomialGroupLayout`s;
- each precommitted layout must be recomputable under
  `ConservativeCommitmentConfig<Cfg>`;
- the full `AkitaScheduleLookupKey` must be derivable from those recomputed
  layouts plus the final group;
- each precommitted group must have
  `group.num_vars <= final_group.num_vars / 2`;
- each precommitted group must keep its `PrecommittedGroupParams` `m`, `r`,
  `log_basis`, `n_a`, and B width;
- each precommitted group must use the frozen conservative B row count in the
  grouped root relation;
- the final grouped root schedule must fit setup capacity;
- the last group commitment must match the final group's params.

At current prove time:

- `ProverOpeningData` / `OpeningClaimsLayout` must be internally consistent.
- `G > 1` must reject before schedule lookup.
- Tiered multi-group proofs must reject with `AkitaError::InvalidSetup`.
- Recursive setup contribution must reject with `AkitaError::InvalidSetup`.
- Dense multi-group proofs must reject with `AkitaError::InvalidInput`.
- Generic one-hot multi-group proofs must reject with
  `AkitaError::InvalidInput(GROUPED_ROOT_UNSUPPORTED)`.

At Phase 2 prove time:

- `OpeningClaimsLayout` must be internally consistent.
- `commitments.len() == G`.
- `hints.len() == G`.
- `sum_g K_g == num_claims`.
- Each commitment row count must match its group params.

At current verify time:

- `OpeningClaims` must be internally consistent.
- `G > 1` must reject before schedule lookup.
- Tiered multi-group proofs must reject with `AkitaError::InvalidSetup`.
- Recursive setup contribution must reject with `AkitaError::InvalidSetup`.
- Generic multi-group proofs must reject with `AkitaError::InvalidProof`.

At Phase 2 verify time:

- The verifier must reconstruct the `AkitaScheduleLookupKey` from the public
  opening batch, setup, and config policy.
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


| Shape                                           | Rejection point                         | Error                                    |
| ----------------------------------------------- | --------------------------------------- | ---------------------------------------- |
| Tiered preset + grouped schedule key            | `runtime_schedule` / grouped DP         | `AkitaError::InvalidSetup`               |
| Tiered preset + `G > 1` proof                   | Prove / verify entry                    | `AkitaError::InvalidSetup`               |
| Dense config + grouped schedule key             | `runtime_schedule` / grouped DP         | `AkitaError::InvalidSetup`               |
| Dense polynomial at conservative precommit      | `ConservativeCommitmentConfig` commit params / one-hot validators | `AkitaError::InvalidSetup` / `InvalidInput` |
| Dense polynomial + `G > 1` proof                | Prove entry                             | `AkitaError::InvalidInput`               |
| Precommitted `num_vars > final_group.num_vars / 2` | grouped key validation               | `AkitaError::InvalidInput`               |
| Recursive setup contribution + `G > 1`          | Prove / verify entry                    | `AkitaError::InvalidSetup`               |
| Scalar table lookup collapsing `[1,3]` to `[4]` | scalar key construction / grouped lookup | table miss or `AkitaError::InvalidSetup` |
| Generic grouped proof before Phase 2            | Prove / verify entry                    | `AkitaError::InvalidInput` / `InvalidProof` |
| `log_basis != min_basis(Cfg)` at precommit      | conservative layout validation / grouped root params | `AkitaError::InvalidSetup`               |


## Rollout Plan

### Phase 0: Opening Claims Cleanup and Guards

- Clean up the old flattened commitment-group routing from `OpeningClaimsLayout`.
- Make `OpeningClaimsLayout` follow the new group design:
  - one shared `point`;
  - ordered `groups`;
  - each `PolynomialGroupClaims` has `PointVariableSelection` plus dense
    evaluations and a commitment.
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

Phase 0 is implemented. The old slot/routing vocabulary is gone from
`OpeningClaimsLayout`, grouped batch shape and descriptor binding exist, scalar
paths still work, and unsupported grouped proof paths fail explicitly.

### Phase 1: Scheduler and Conservative Precommit

- Implemented `AkitaScheduleLookupKey`.
- Implemented generated/DP schedule resolution for grouped keys without collapsing
`[K_0, K_1, ...]` into `[sum_g K_g]`.
- Threaded grouped root counts through planner and proof-size formulas:
  - `G`;
  - `num_t_vectors_total = sum_g K_g`;
  - `num_w_vectors_root = sum_g W_g`;
  - `num_z_vectors_root = G`;
  - `num_z_segments = G`.
- Implemented `PrecommittedGroupParams`.
- Implemented conservative B rank selection for standalone groups through
  `ConservativeCommitmentConfig<Cfg>`.
- Added generated grouped table entries for selected one-hot families.
- Kept grouped opening and grouped root prove guarded until Phase 2.

Phase 1 is implemented. Conservative precommit layouts are reproducible from
`PolynomialGroupLayout`, grouped scheduler lookup/DP fallback is available
for final planning, conservative B rank selection works, and grouped schedule
negative tests pass.

### Phase 1.5: Final Group Commitment

- Implemented `commit_final_group(new_group, precommitted_layouts)`.
- Recompute precommitted layouts under `ConservativeCommitmentConfig<Cfg>`.
- Commit the final group with the final grouped plan.
- Return the final commitment plus hint only.

Phase 1.5 is implemented. It validates the final-group commitment layout and
final commitment row count, but it does not produce a grouped opening proof.

### Phase 2: Grouped Opening

- Reuse `OpeningClaimsLayout` / `CallSection.opening_batch_digest` for group
  partition and point-variable selection binding.
- Reconstruct and validate the full `AkitaScheduleLookupKey` from the public
  opening batch, setup, and config policy.
- Bind the final grouped root schedule through
  `PlanSection.effective_schedule_digest`.
- Implement grouped root witness layout with `concat(w_hat_g)`.
- Route prover root B rows through per-group B computation for `G > 1`.
- Generalize root verifier row counts and terminal witness shapes.
- Support unequal `K_g`.
- Support `SetupContributionMode::Direct`.
- Add folded non-tiered two-group one-hot same-point E2E.

Phase 2 done means: a final-group commitment can be combined with precommitted
one-hot groups to produce and verify a grouped same-point opening proof for
unequal `K_g`, with recursive suffix folds remaining singleton.

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
- `[1, 3]` and `[4]` produce different opening-batch digests.
- Generated group-batch schedule lookup compares precommitted group params,
  frozen group params, and `conservative_n_b`.
- Descriptor bytes change when a precommitted group's `PrecommittedGroupParams`
  `m`, `r`, `log_basis`, `n_a`, or conservative `n_b'` changes.
- Scheduler grouped root sizing accounts for one `z_hat_g` segment per group.
- Scheduler grouped D width reports `total_d_w_rings = sum_g w_hat_rings_g`.
- Grouped terminal root folds reject until the terminal witness layout exists.
- Conservative B rank uses the derived B width and norm from `l_max`.
- `log_basis != min_basis(Cfg)` rejects.

### Commit Tests

- `ConservativeCommitmentConfig<Cfg>::batched_commit(group0)` commits with a
  reproducible frozen layout.
- Independent conservative precommitted groups do not depend on each other.
- Undersized conservative `n_b` rejects.
- Exceeding setup capacity rejects.
- Dense policies reject in the conservative/grouped precommit path.
- `commit_final_group(group_last, precommitted_layouts)` commits the final group
  with grouped params.
- `commit_final_group` rejects if a precommitted key cannot recompute to a valid
  conservative `PrecommittedGroupParams`.
- Phase 2: grouped opening finalization uses the existing opening-batch and
  effective-schedule descriptor plumbing; it does not add a handle side channel
  or a params digest.

### Prove / Verify Tests

- Current prove rejects generic one-hot `G > 1` with the grouped-root unsupported error.
- Current verify rejects generic `G > 1` with `AkitaError::InvalidProof`.
- Current prove/verify reject tiered `G > 1` with `AkitaError::InvalidSetup`.
- Current prove/verify reject recursive setup contribution `G > 1` with
  `AkitaError::InvalidSetup`.
- Phase 2: root-direct two-group one-hot same-point round trip.
- Phase 2: folded non-tiered two-group one-hot same-point round trip.
- Phase 2: folded non-tiered unequal group sizes, for example `[1, 3]`.
- Phase 2: suffix remains singleton after a grouped root.
- Phase 2: serialize/deserialize grouped proofs round trip with the current descriptor schema.
- Phase 2: opening-batch digest mismatch rejects.
- Phase 2: reordered precommitted group vector rejects.
- Phase 2: duplicate or missing precommitted group layout derivation rejects.
- Phase 2: unknown descriptor version rejects for grouped proofs.
- Phase 2: scalar `[4]` descriptor bytes do not verify as grouped `[1, 3]`.
- Phase 2: swapping group commitments rejects.
- Phase 2: tampering group `1` opening rejects.
- Phase 2: tampering group `1` hint or `t_hat` segment rejects.
- Phase 2: truncating one group's commitment rows rejects.
- Phase 2: changing a precommitted group's `PrecommittedGroupParams` rejects.
- Phase 2: changing a precommitted group's conservative `n_b'` rejects.
- Phase 2: descriptor `[1, 3]` with proof `[4]` rejects.

### Planner and Table Tests

- Grouped table lookup misses rather than collapsing to a scalar `[sum_g K_g]`
key.
- Generated grouped entries, once emitted, match DP.
- Root `w_len` from planner matches the Phase 2 prover-built grouped witness length.
- Singleton `G = 1` grouped schedule lookup delegates to the scalar path.
- Phase 2 setup envelope scans include unequal partitions such as `[1, 3]`.

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

- effective-schedule binding of the materialized grouped root;
- verifier recomputation of grouped root params from that key;
- group partition binding;
- frozen precommit layout binding for every precommitted group;
- equality between the final precommitted-group B row count and the frozen
conservative `n_b'`;
- setup capacity covering the selected matrix prefixes;
- D rank and width sized for the full concatenated root witness;
- rejecting unimplemented tiered and recursive setup-contribution combinations.

No verifier path may trust prover-supplied group layout data without recomputing
it from the public opening batch, setup, and config policy.

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

1. **Public API shape:** use `ConservativeCommitmentConfig<Cfg>` with ordinary
  `batched_commit` for precommitted groups, and expose `commit_final_group` for
  the final-group commitment. Do not add a separate `commit_group` handle layer
  in the initial API rollout.
2. **Setup capacity:** keep the public setup surface on `max_num_vars`,
  `max_num_batched_polys`, and `max_setup_len` for Phase 1. Add an explicit
  group-count setup API field only if Phase 2 envelope scans show a need.
3. **Recursive setup contribution:** explicitly delay generalization until
  a later root-layout generalization. The grouped opening phase keeps the
  existing reject.
4. **Planner key exposure:** keep `AkitaScheduleLookupKey` as the canonical
  config/planner key. Public final-commit callers pass
  `PolynomialGroupLayout`s for precommitted groups; opening callers should use
  grouped `OpeningClaims` vectors and the effective schedule digest rather than
  proof-local side channels.

## Open Questions

None for the initial grouped implementation. Revisit only if tiered multi-group or
known-final-schedule precommit become active follow-up specs.