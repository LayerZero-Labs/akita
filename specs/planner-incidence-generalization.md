# Spec: Planner Incidence Generalization

> **Superseded (schedule keys):** portions of this spec that describe schedule lookup
> keys, shipped-table selection, or preset↔table binding are superseded by
> [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md). This file remains
> for historical witness-layout / incidence notes until archived.

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     |                                |
| Created       | 2026-05-27                     |
| Status        | active                         |
| PR            |                                |
| Book-chapter  | book/src/how/configuration.md  |

## Goal

Clean up and generalize the Akita planner so root schedule selection is driven by
the actual opening incidence structure, not by legacy aggregate
`{ num_claims, num_commitment_groups, num_points }` shapes.

The planner's job should be narrow:

1. Receive a protocol-size profile derived from validated opening incidence.
2. Brute-force root dimensions such as `m`, `r`, `log_basis`, and matrix
   widths. Digit depths are computed during candidate/layout evaluation, but
   should not be treated as independent persisted schedule choices.
3. Choose the schedule that minimizes proof size.

The planner should not know about commitments, claimed evaluations, transcript
objects, field values, or prover hints. Those belong to the proof/protocol layer.

## Existing Structures

The relevant existing structure is `ClaimIncidenceSummary` in
`crates/akita-types/src/proof/incidence.rs`.

It already records the canonical public routing:

- `num_vars`: number of variables in every opening point.
- `num_points`: number of distinct opening points.
- `num_groups`: number of distinct committed groups.
- `num_claims`: number of individual `(point, group, poly)` openings.
- `claim_to_point[claim_idx]`: opening point used by each flattened claim.
- `claim_to_group[claim_idx]`: committed group used by each flattened claim.
- `claim_poly_indices[claim_idx]`: polynomial index within the committed group.
- `group_poly_counts[group_idx]`: number of polynomials in each group.
- `point_claim_counts[point_idx]`: number of claims at each point.
- `point_group_counts[point_idx]`: number of distinct groups touched by each point.

After the opening-claims cutover, the main schedule-facing projection is
`AkitaScheduleLookupKey` in `crates/akita-types/src/schedule.rs`:

```rust
pub struct AkitaScheduleLookupKey {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: Vec<PrecommittedGroupParams>,
}
```

This key intentionally no longer carries setup capacity. In particular,
`max_num_vars` is not a scheduler/planner key dimension after preprocessing.
Setup capacity still exists in `AkitaSetupSeed` and setup sizing policy, but
runtime schedule selection is keyed only by actual root group geometry and any
frozen precommit metadata.

The current key is still an interim projection. It records only:

```text
num_vars
num_t_vectors
num_w_vectors
num_z_vectors
```

That is enough to remove the old setup-capacity bucket, and it is enough to
distinguish the current `z`-sharing cases. It still does not explicitly carry
`num_commitment_groups` or `num_public_y_rows`, so the remaining incidence
generalization work is to make the full root profile authoritative instead of
encoding only the counts needed by today's materializer.

## Desired Conceptual Model

The planner should reason in terms of the number of protocol vectors that
determine root witness/proof size:

- `t` vectors: determined by the committed polynomials and their sizes.
- `w` vectors: determined by the opening points.
- `z` vectors: determined by commitment-group point-set incidence.
- public `y` rows: always one row per distinct opening point.

This is intentionally a protocol-level model, not just a refactor of the current
formula. If current prover/verifier code counts these objects differently, update
the protocol implementation so it matches this model.

### `t` Vectors

`t` is commitment-side data. It is determined by the number of committed
polynomials and their sizes.

Current code assumes all polynomials in a batched commit have the same
`num_vars`, and in practice all groups in the same batch use one root layout.
Under that current assumption, the number of `t` vectors is the total number of
polynomials across all groups:

```text
num_t_vectors = sum(group_poly_counts)
```

If mixed polynomial sizes become supported later, the planner input should not be
a single count. It should carry per-size buckets, for example:

```text
[(num_vars, count), ...]
```

For the current cleanup, preserving the same-size invariant is acceptable.

### `w` Vectors

`w` is opening-point-side data. It should be determined by the number of distinct
opening points:

```text
num_w_vectors = num_points
```

This is a conceptual protocol requirement. Do not derive the planner's `w` count
from `num_claims`.

### `z` Vectors

`z` requires the most care.

Normally, each commitment group contributes one `z`. However, if multiple
commitment groups are opened at exactly the same set of opening points, they
should share one `z` vector. In other words:

```text
num_z_vectors = number of distinct point-sets among commitment groups
```

Derive this from `ClaimIncidenceSummary` by building, for every group, the set of
points where that group appears:

```text
group 0 -> {point 0, point 1}
group 1 -> {point 0, point 1}
group 2 -> {point 2}

num_z_vectors = 2
```

This is exactly the information that aggregate claim/group/point counts cannot
represent.

Important validation rule: every group is already required to be used by
`ClaimIncidenceSummary::validate`, so every group point-set should be nonempty.

### Public `y` Rows

There must always be one public `y` row per opening point:

```text
num_public_y_rows = num_points
```

This is not optional. The batching protocol supports one public output row per
distinct opening point, and the planner/proof-size code should preserve that
invariant.

## Proposed New Type

Add a schedule/planner-facing profile derived from incidence. Naming is flexible,
but the type should live in `akita-types` because config, planner, generated
tables, prover, and verifier all need to agree on it.

Example:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootPlannerProfile {
    pub num_vars: usize,
    pub num_t_vectors: usize,
    pub num_w_vectors: usize,
    pub num_z_vectors: usize,
    pub num_commitment_groups: usize,
    pub num_public_y_rows: usize,
}
```

The current implementation has already moved `num_vars` into the schedule key
and removed the separate scheduler `max_num_vars` dimension. The next step is to
extend the schedule key/profile so it also carries the remaining root-size inputs
explicitly: `num_commitment_groups` and `num_public_y_rows`.

The key point is that the planner input must encode `t/w/z/y/group` counts
directly instead of reusing aggregate claim/group/point shapes.

Add a conversion method on `ClaimIncidenceSummary`:

```rust
impl ClaimIncidenceSummary {
    pub fn root_planner_profile(&self) -> Result<RootPlannerProfile, AkitaError> {
        // num_t_vectors = sum(self.group_poly_counts)
        // num_w_vectors = self.num_points
        // num_public_y_rows = self.num_points
        // num_commitment_groups = self.num_groups
        // num_z_vectors = distinct group point-sets
    }
}
```

Use `BTreeSet<usize>` or sorted `Vec<usize>` for canonical point-sets so the
result is deterministic.

Current superseded schedule-key status:

- Scalar same-point paths use `AkitaScheduleLookupKey::from_layout` on an
  `OpeningClaimsLayout`; that projection rejects multi-group layouts instead of
  collapsing them.
- Grouped-root planning uses `AkitaScheduleLookupKey` with `final_group` plus
  `PrecommittedGroupParams` for earlier groups, as specified in
  [`multi-group-batching.md`](multi-group-batching.md).
- The older incidence-derived schedule-key plan in this file should not be
  continued directly for production paths.

For the production same-point multi-commitment rollout, do not continue this
older aggregate-incidence plan directly. Follow
[`multi-group-batching.md`](multi-group-batching.md), which keeps scalar
same-bundle schedules separate from multi-group root keys and requires explicit
rejects until the grouped descriptor and schedule shape land.

## Root Witness Size Formula

Replace the legacy aggregate shape formula with a profile-based formula.

Current rough shape:

```text
w_hat = K * 2^r * delta_open
t_hat = K * 2^r * n_A * delta_open
z_pre = P * 2^m * delta_commit * delta_fold
r     = (n_D + n_B * G + P + 1 + n_A) * delta_R
```

Desired shape:

```text
w_hat = num_w_vectors * 2^r * delta_open
t_hat = num_t_vectors * 2^r * n_A * delta_open
z_pre = num_z_vectors * 2^m * delta_commit * delta_fold
r     = (n_D
       + n_B * num_commitment_groups
       + num_public_y_rows
       + 1
       + n_A) * delta_R
```

With `zk`, keep B-blinding counted per commitment group unless the commitment
hiding protocol is also changed:

```text
blinding = num_commitment_groups * blinding_cols(...)
```

## Current Code To Replace Or Adapt

The main current shape carriers are:

- `AkitaScheduleLookupKey` and `GeneratedScheduleTableEntry` key fields
  (`final_group`, `precommitteds`).
- `w_ring_element_count_with_counts_for_layout`.
- `root_w_ring_element_count` in `crates/akita-planner/src/schedule_params.rs`.
- `find_optimal_schedule`.
- `gen_schedule_tables.rs`, which emits generated schedule entries keyed by the
  exact root profile counts.
- Config-policy call sites in `crates/akita-config/src/lib.rs`,
  `crates/akita-config/src/transcript_binding.rs`, and
  `crates/akita-config/src/proof_optimized.rs` (`policy_of`, `CommitmentConfig`).

The long-term direction should be:

1. Derive `RootPlannerProfile` from `ClaimIncidenceSummary`.
2. Use `RootPlannerProfile` as the planner input.
3. Keep setup capacity out of scheduler/planner keys.
4. Update generated schedule keys to include the profile counts needed for exact
   lookup.
5. Keep `ClaimIncidenceSummary` as the canonical protocol routing object.

`AkitaScheduleLookupKey` can either evolve into a fuller root profile or become a
thin wrapper around it. It should not regain setup-capacity fields.

## Generated Schedule Entries

Generated schedule tables should persist only the planner decisions that are not
cheaply derivable at runtime.

Generated rows inline the runtime lookup-key fields:

```rust
pub struct GeneratedScheduleTableEntry {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: &'static [PrecommittedGroupParams],
    pub steps: &'static [GeneratedStep],
}
```

Do not reintroduce `max_num_vars` here. Generated schedules are keyed by the
actual root problem shape, not by setup capacity. A setup that supports
`max_num_vars = N` must instead size its matrix envelope over all generated or
planner-supported runtime shapes with `num_vars <= N`.

Each generated fold step stores the chosen layout/search parameters:

```rust
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub position_index_bits: u32,
    pub block_index_bits: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
}
```

The terminal direct step is only a marker:

```rust
pub struct GeneratedDirectStep;
```

Do not store cached materialization results in generated entries. In particular,
avoid reintroducing:

- `current_w_len`, `next_w_len`, or direct `witness_shape`.
- `challenge_l1_mass`; derive it from the runtime sparse-challenge policy for
  the selected `ring_d`.
- `delta_open`, `delta_fold`, or `delta_commit`.
- `w_ring`, `level_bytes`, `direct_bytes`, or `total_bytes`.
- direct terminal `entry_d` or `entry_nb`.

Runtime materialization in `schedule_plan_from_generated_entry` derives these
from the key, previous folds, `LevelParams`, decomposition policy, direct-level
config policy, and proof-size helpers. This keeps generated artifacts focused on
the planner's actual choices while preserving exact proof-size accounting.

## Setup Capacity And Envelope Sizing

`max_num_vars` remains a setup-capacity concept:

- `AkitaSetupSeed::max_num_vars` bounds accepted commitment/proof inputs.
- `ClaimIncidence::validate` and batched input validation reject claims whose
  actual `num_vars` exceeds setup capacity.
- Setup matrix sizing must conservatively cover every actual runtime shape the
  setup may serve.

Because schedule lookup no longer includes `max_num_vars`, setup preprocessing
must not size only the all-up shape. It must take the maximum over:

```text
1 <= num_vars' <= setup.max_num_vars
1 <= num_polys' <= setup.max_num_batched_polys
1 <= num_commitment_groups' <= num_polys'
1 <= num_points' <= min(num_polys', setup.max_num_points)
```

This is the purpose of `proof_optimized_max_setup_matrix_size`: a smaller
runtime `num_vars` or differently multi-group batch may select a schedule with larger
row or stride requirements than the setup's maximum-arity case.

## Protocol Changes Needed

This is not only a planner cleanup if the current prover/verifier witness layout
still counts `w`, `t`, or `z` according to old aggregate formulas.

Audit and update these areas:

- Root folded prover construction in `crates/akita-prover/src/protocol/flow.rs`.
- Batched quadratic-equation construction in
  `crates/akita-prover/src/protocol/quadratic_equation.rs`.
- Ring-switch prover/verifier row preparation in
  `crates/akita-prover/src/protocol/ring_switch.rs` and
  `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Root verifier replay in `crates/akita-verifier/src/protocol/levels.rs`.
- Direct/root commitment checks in `crates/akita-verifier/src/protocol/batched.rs`
  and `crates/akita-verifier/src/proof/direct.rs`.

The implementation should make the runtime recursive witness length exactly
match the new profile-based planner formula. Any place that currently computes
`w_len`, `w_ring`, `m_row_count`, `z_pre` length, or proof size from
`num_claims`, `num_groups`, and `num_points` should be checked.

## API And Compatibility Notes

The current public prover/verifier input is point-local:

```text
Vec<(opening_point, Vec<group-at-that-point>)>
```

The current normalizers preserve each caller-provided group occurrence. That
means the same commitment repeated under multiple points is treated as multiple
groups unless a more canonical incidence API is introduced.

To fully benefit from the new model, add or expose an API that can express one
commitment group opened at multiple points as one group with multiple incident
claim edges. `ClaimIncidence` already supports this internally.

Recommended direction:

- Keep existing ergonomic APIs as adapters.
- Add lower-level prove/verify entry points that accept `ClaimIncidence` or a
  structure that can canonicalize into it.
- Make both adapters converge to the same `ClaimIncidenceSummary`.

## Tests To Add

Add unit tests for profile derivation from incidence:

- Singleton: one point, one group, one polynomial.
- Same-point batch: one point, multiple groups and multiple polynomials.
- One group opened at multiple points.
- Multiple groups opened at the same exact set of points should share one `z`.
- Multiple groups opened at different point sets should produce multiple `z`
  vectors.

Add planner tests:

- `find_optimal_schedule` accepts the schedule/profile key without any
  `max_num_vars` dimension.
- Root witness size changes when `num_z_vectors` changes while
  `num_points`/`num_groups` stay fixed.
- Public proof size still scales with `num_public_y_rows == num_points`.
- Generated non-`zk` schedules do not increase `exact_proof_bytes` versus
  upstream entries after normalizing away removed cached materialization fields.
- `zk` schedule/proof-size comparisons are tracked separately, because current
  branch ZK accounting may differ from upstream even when fold choices match.

Add protocol/e2e tests:

- One committed polynomial opened at multiple points.
- Two commitment groups opened at the same points, confirming the proof follows
  the shared-`z` layout.
- A mixed incidence where groups have different point sets.
- Serialization/deserialization of proofs produced by the new layout.

Run:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
cargo test batched
cargo test multipoint
```

## Implementation Order

1. Remove scheduler/planner `max_num_vars` from lookup keys, generated keys, and
   schedule inputs. Done.
2. Ensure setup preprocessing sizes the matrix envelope over all actual
   `num_vars <= setup.max_num_vars`. Done.
3. Add `RootPlannerProfile` and derive it from `ClaimIncidenceSummary`.
4. Update planner sizing functions to use the full profile, including
   `num_commitment_groups` and `num_public_y_rows`.
5. Update runtime witness-size helpers in `akita-types`.
6. Update schedule lookup/generated key types with any remaining profile fields.
7. Update config policy to pass profiles rather than partial schedule keys.
8. Update prover/verifier protocol code so actual root witness layout matches
   the profile formula.
9. Add incidence-level and e2e tests.

