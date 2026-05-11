# Planner Incidence Generalization

## Goal

Clean up and generalize the Akita planner so root schedule selection is driven by
the actual opening incidence structure, not by the current aggregate
`WitnessShape { num_claims, num_commitment_groups, num_points }`.

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

`WitnessShape` currently lives in `crates/akita-types/src/schedule.rs` and is too
coarse. It records only:

```text
K = num_claims
G = num_commitment_groups
P = num_points
```

That loses routing information. In particular, it cannot distinguish shapes where
multiple commitment groups are opened at exactly the same set of points from
shapes where those groups touch different point sets.

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

This is exactly the information that `WitnessShape` cannot represent.

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

The `num_vars` field can also remain outside the profile if that better matches
existing schedule APIs. The key point is that the planner input must encode
`t/w/z/y/group` counts directly instead of reusing `WitnessShape`.

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

## Root Witness Size Formula

Replace the current `WitnessShape` formula with a profile-based formula.

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

- `WitnessShape` in `crates/akita-types/src/schedule.rs`.
- `AkitaRootBatchSummary` in `crates/akita-types/src/schedule.rs`.
- `AkitaScheduleLookupKey` and `GeneratedScheduleKey`.
- `w_ring_element_count_with_counts`.
- `root_w_ring_element_count` in `crates/akita-planner/src/schedule_params.rs`.
- `find_optimal_schedule` and `find_optimal_schedule_with_max`.
- `gen_schedule_tables.rs`, which emits generated schedule entries keyed by the
  exact root profile counts.
- Config-policy call sites in `crates/akita-config/src/lib.rs`,
  `crates/akita-config/src/schedule_policy.rs`, and
  `crates/akita-config/src/proof_optimized.rs`.

The long-term direction should be:

1. Derive `RootPlannerProfile` from `ClaimIncidenceSummary`.
2. Use `RootPlannerProfile` as the planner input.
3. Remove `WitnessShape` from planner APIs.
4. Update generated schedule keys to include the profile counts needed for exact
   lookup.
5. Keep `ClaimIncidenceSummary` as the canonical protocol routing object.

`AkitaRootBatchSummary` can either be removed or reduced to a compatibility shim.
If it remains, it should not be the authoritative planner input because it cannot
represent `num_z_vectors`.

## Generated Schedule Entries

Generated schedule tables should persist only the planner decisions that are not
cheaply derivable at runtime.

The generated key is profile-shaped:

```rust
pub struct GeneratedScheduleKey {
    pub max_num_vars: usize,
    pub num_vars: usize,
    pub num_t_vectors: usize,
    pub num_w_vectors: usize,
    pub num_z_vectors: usize,
}
```

Each generated fold step stores the chosen layout/search parameters:

```rust
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
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

- `find_optimal_schedule` accepts the new profile and no longer requires
  `WitnessShape`.
- Root witness size changes when `num_z_vectors` changes while
  `num_points`/`num_groups` stay fixed.
- Public proof size still scales with `num_public_y_rows == num_points`.

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

1. Add `RootPlannerProfile` and derive it from `ClaimIncidenceSummary`.
2. Update planner sizing functions to use the profile.
3. Update runtime witness-size helpers in `akita-types`.
4. Update schedule lookup/generated key types.
5. Update config policy to pass profiles rather than `WitnessShape`.
6. Update prover/verifier protocol code so actual root witness layout matches
   the profile formula.
7. Add incidence-level and e2e tests.
8. Remove or deprecate `WitnessShape` after all call sites move.

