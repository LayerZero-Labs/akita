# Spec: Setup-Offloading Planner

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     |                                            |
| Created       | 2026-07-10                                 |
| Status        | proposed                                   |
| PR            |                                            |
| Supersedes    |                                            |
| Superseded-by |                                            |
| Book-chapter  | book/src/roadmap/verifier-offloading.md    |

## Summary

Recursive setup contribution currently runs Stage 3 and can select a committed
setup prefix, but the planner neither decides which folds should offload nor
guarantees that the next fold can prove the resulting prefix opening. Recursive
suffix planning also assumes one commitment group, while complete offloading
requires the successor to prove two openings together: its newly committed
folded witness and the setup-prefix commitment selected by the preceding fold.

This design adds `RecursiveCommitmentConfig<Cfg>`, parallel to
`ConservativeCommitmentConfig<Cfg>`. The ordinary `Cfg` always resolves a
direct-only schedule. Selecting the recursion adapter activates setup
offloading only for the planner's genuine multi-group path. Scalar/singular
keys continue through the ordinary direct planner and ordinary generated
catalog, even under the recursion adapter. For a multi-group key, fold levels 0
and 1 use `SetupContributionMode::Recursive` when that fold's setup prefix is
larger than `2^10`; every later fold is direct. The recursive multi-group
planner rejects candidate edges whose successor cannot commit a required
prefix, then uses the existing planner comparison to select the smallest
remaining proof. Recursive successors use the existing multi-group
representation with the setup prefix as a precommitted group and the folded
witness as the final group. Recursive multi-group generated schedules are
stored separately from ordinary schedules. The design reuses `SetupPrefixSlotId`,
`SetupPrefixSlot`, `SetupPrefixVerifierSlot`, `OpeningClaims`, and the existing
grouped `LevelParams` machinery rather than adding parallel requirement,
geometry, or carried-claim models.

## Intent

### Goal

Provide an explicit recursion config that activates offloading only for
multi-group batches, makes that planner path enforce recursive setup on
threshold-qualified fold levels 0 and 1, guarantees every recursive edge has a
compatible preprocessed setup-prefix commitment, and leaves singular planning
direct-only.

### Invariants

- **Config selection activates recursion.** Ordinary `Cfg` planning is
  direct-only. Only the multi-group path under
  `RecursiveCommitmentConfig<Cfg>` may emit recursive levels.
- **Singular planning never offloads.** `AkitaScheduleLookupKey` values with no
  precommitted groups use the existing scalar planner and direct catalog. Every
  level is `Direct`, including under `RecursiveCommitmentConfig<Cfg>`.
- **The recursion window is fixed.** Only fold levels 0 and 1 may be recursive.
  Fold levels 2 and above are always direct.
- **Threshold-qualified recursion is mandatory.** In a recursion config, a
  fold at level 0 or 1 whose successor is also nonterminal and whose
  `N_prefix > 2^10` must be recursive. If its successor cannot commit the
  prefix, that candidate edge is infeasible; the planner may not turn that edge
  direct as a fallback.
- **Per-level mode is authoritative after config selection.** Every
  `LevelParams` records
  `SetupContributionMode`. Prover, verifier, generated-table replay, setup
  preprocessing, descriptor hashing, and proof-size accounting use that field.
- **Recursive means an actual carried setup opening.** A recursive fold runs
  Stage 3, exposes `S_i(rho_setup)`, and passes the matching prefix slot into the
  successor's opening batch. It may not silently revert to a local setup scan.
- **Mode and successor shape are equivalent.** When fold `i` has a successor,
  it is `Recursive` if and only if fold `i + 1` contains exactly one setup-prefix
  precommitted group beside its witness group. It is `Direct` if and only if fold
  `i + 1` contains only its witness group. Generated replay, proving, and
  verification enforce both directions.
- **Direct means no outgoing setup group.** A direct fold may consume an
  incoming setup group, but it creates no setup claim for its successor.
- **Terminal folds are scalar and direct.** A terminal fold has no successor
  commitment, so it cannot offload its setup claim or consume an incoming setup
  group. It consumes exactly one witness group.
- **Grouped steps are nonterminal folds.** A schedule `Direct` step and the last
  `Fold` consume exactly one group. Any fold that consumes a setup-prefix group
  must itself have another `Fold` as its successor. This is the canonical shape
  defined by `specs/multi-group-batching.md`.
- **One setup-prefix identity.** `SetupPrefixSlotId` remains the canonical
  identity. `natural_len` and `n_prefix` identify the prefix domain;
  `level_params_digest` identifies the exact commitment params, including
  `log_basis`, `position_index_bits`, `block_index_bits`, group params, and per-level mode.
- **One total-prefix calculation.** `active_setup_field_len` is the canonical
  challenge-free calculation of active setup coefficients. Planner,
  preprocessing, prover, and verifier do not maintain separate formulas.
- **Shared D remains shared.** Multi-group folds use one D relation over the
  concatenation of all groups' opening segments. This design does not introduce
  per-group D commitments. `LevelParams::d_key`/generated `n_d` are shared by
  the final witness group and every precommitted setup-prefix group.
- **Existing group model is canonical.** The setup prefix is represented by the
  existing precommitted-group fields in `LevelParams`; the next witness is the
  final group. The setup-prefix group has its own A/B matrices and block
  geometry. It does not borrow the successor witness group's A/B column
  capacities. `OpeningClaimsLayout::root_group_order` determines proof order.
- **Local minimization remains bounded.** Recursive suffix candidate generation
  continues to retain one locally smallest next-witness candidate per basis.
  Prefix compatibility filters candidates but does not create an exponential
  frontier.
- **Generated and fallback schedules agree.** A generated row stores the same
  per-level mode chosen by dynamic planning, and the canonical row walker
  recomputes every incoming-prefix transition and grouped witness length.
- **Generated catalogs do not alias.** Direct and recursive schedules are
  emitted into separate generated tables. The recursion adapter never reads the
  ordinary config's direct table.
- **Preprocessing is complete for planned schedules.** Every recursive edge in
  every setup-supported selected schedule has an exact `SetupPrefixSlot`.
  Setup construction never truncates `natural_len`.
- **No verifier panics.** Bad group counts, wrong slot identity, unsupported
  mode, missing required slots, malformed prefix lengths, and arithmetic
  overflow return `AkitaError` or serialization errors.

### Non-Goals

- A new setup-prefix metadata or planner-requirement type.
- A generic carried-opening enum or wrapper around folded-witness claims.
- Per-group D matrices or D commitments.
- Distributed or multi-chunk setup offloading.
- Composition of recursive and conservative config adapters in the first
  rollout.
- Setup offloading for singular/scalar schedule keys.
- Setup offloading at ring dimensions other than the supported uniform D64
  shape.
- Globally enumerating every suffix `(log_basis, m, r)` combination.
- Backward compatibility for old generated rows, descriptors, setup artifacts,
  or proof bytes.
- Full-ladder setup artifact policy. This design materializes the exact slots
  needed by the selected supported schedules.

## Eligibility and Fold Transitions

### Per-Fold Eligibility

Add:

```rust
pub const SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN: usize = 1 << 10;
```

A fold is required to be marked `Recursive` exactly when:

```text
recursive config is selected
the root schedule key is genuinely multi-group (precommitteds is nonempty)
fold level is 0 or 1
the successor is a nonterminal Fold, itself followed by another Fold
all active role dimensions equal SETUP_OFFLOAD_D_SETUP = 64
the level does not use distributed/multi-chunk witness layout
N_prefix > SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN
```

The strict comparison is intentional: a prefix of exactly `2^10`
coefficients remains direct. Fold levels 2 and above remain direct regardless
of prefix size.

Successor fit is a candidate-feasibility condition, not another mode gate. When
the rules above require recursion, retain a proposed edge only if the successor
is nonterminal and can carry `N_prefix` as an independently derived setup-prefix
precommitted group.
The planner tries its normal alternative parameters and suffixes, then chooses
the smallest feasible proof with the existing comparator.

The planner does not use artifact registry contents to decide mode. Registry
contents are setup-instance state and could differ between prover and verifier.
It decides from public geometry, then setup construction must materialize every
required slot.

## Recursion Config Adapter

Add a new config adapter:

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveCommitmentConfig<Cfg>(PhantomData<Cfg>);
```

`RecursiveCommitmentConfig<Cfg>` implements `CommitmentConfig` by delegating
field, ring, decomposition, challenge, SIS, basis, one-hot, and setup-capacity
properties to `Cfg`. It differs in schedule planning:

```text
ordinary Cfg:
  planner recursion flag = false
  schedule catalog = Cfg::schedule_catalog()
  every emitted LevelParams mode = Direct

RecursiveCommitmentConfig<Cfg>:
  scalar key:
    delegate to Cfg::runtime_schedule(key)
    use Cfg::schedule_catalog()
    every level is Direct
  genuine multi-group key:
    planner recursion flag = true
    use Cfg::recursive_multi_group_schedule_catalog()
    levels 0 and 1 recurse exactly when N_prefix > 2^10
    levels 2 and above are Direct
```

Add a default-disabled config hook:

```rust
fn recursive_setup_planning() -> bool {
    false
}
```

The adapter overrides it to `true`, but uses that policy only after determining
that the key is genuinely multi-group. `policy_of::<Self>()` copies the value
into `PlannerPolicy` for `find_group_batch_schedule`. Scalar keys delegate
directly to `Cfg::runtime_schedule` and never invoke the recursion-enabled
policy.

Add a second optional catalog hook:

```rust
fn recursive_multi_group_schedule_catalog()
    -> Option<akita_planner::GeneratedScheduleTable>
{
    None
}
```

The base config's `schedule_catalog()` remains the direct table.
`RecursiveCommitmentConfig<Cfg>` uses
`Cfg::recursive_multi_group_schedule_catalog()` only for a key with nonempty
`precommitteds`.

Its `runtime_schedule` routing is:

```text
if key.precommitteds.is_empty():
    return Cfg::runtime_schedule(key)

validate recursive D64/non-chunked policy
resolve_group_batch_schedule(
    key,
    recursion-enabled policy_of<RecursiveCommitmentConfig<Cfg>>,
    recursive_multi_group_schedule_catalog,
)
```

The adapter should reject unsupported base configs before planning a
multi-group key:

```text
Cfg::D != SETUP_OFFLOAD_D_SETUP
Cfg::chunked_witness_cfg().uses_multi_chunk()
```

Levels 0 and 1 apply the prefix threshold. When the threshold requires
recursion, successor fit filters candidate edges before the existing proof-size
comparison.

The public scheme/config choice therefore determines the planner family:

```rust
type DirectScheme = AkitaPcs<Cfg>;
type RecursiveScheme = AkitaPcs<RecursiveCommitmentConfig<Cfg>>;
```

Exact scheme aliases may differ, but callers must select recursion through the
config type rather than through a prove/verify mode argument.

### State Transition

These transitions are reachable only from a genuinely multi-group root
schedule. A scalar root never creates `S_i` and remains on the one-group direct
path for its entire schedule.

Let fold `i` enter with either:

```text
[W_i]
```

or:

```text
[S_{i-1}, W_i] in OpeningClaims storage order
```

where `S_{i-1}` is precommitted and `W_i` is the final/new group. Existing
`root_group_order()` processes the final group first, so protocol order is:

```text
[W_i, S_{i-1}]
```

If fold `i` is recursive, Stage 3 produces a setup-prefix opening and fold
`i + 1` receives:

```text
[S_i, W_{i+1}] in storage order
[W_{i+1}, S_i] in proof order
```

Fold `i + 1` must be nonterminal. The planner must not create this transition
when `i + 1` would be the last fold.

If fold `i` is direct, fold `i + 1` receives only `[W_{i+1}]`, even when fold
`i` itself consumed two groups. This allows an initial recursive prefix of the
schedule followed by direct suffix levels.

## Existing Types and Required Changes

### `LevelParams`

Add to `akita_types::LevelParams`:

```rust
pub setup_contribution_mode: SetupContributionMode,
```

The field participates in:

- `LevelParams` validation and descriptor bytes;
- `digest_level_params`;
- generated schedule expansion;
- effective schedule digest;
- setup-prefix `level_params_digest`;
- proof-shape and schedule drift tests.

`ExecutionSchedule` does not duplicate it; runtime reads
`exec.params.setup_contribution_mode`.

Remove the call-wide `SetupContributionMode` prove/verify argument. The selected
config chooses direct versus recursion-aware planning, and the resulting
schedule chooses each fold's mode. Keeping a second call-time selector would
allow the proof request to disagree with committed/generated params.

### Generated Rows

Add `setup_contribution_mode` to `GeneratedFoldStep`. Generated rows explicitly
store the planner's decision instead of re-running policy after generation.
Generated row bytes, emitted tables, and schedule catalog identity consequently
change.

Ordinary generated rows contain only `Direct`. Recursive generated rows may
mark only levels 0 and 1 recursive; every row at level 2 or above is direct.

Generated rows must also store the compact setup-prefix group params for folds
that **consume** an incoming setup prefix:

```rust
pub struct GeneratedSetupPrefixGroup {
    pub position_index_bits: u32,
    pub block_index_bits: u32,
    pub n_a: u32,
    pub n_b: u32,
}

pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub position_index_bits: u32,
    pub block_index_bits: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
    pub setup_prefix_group: Option<GeneratedSetupPrefixGroup>,
    pub setup_contribution_mode: SetupContributionMode,
}
```

`position_index_bits/block_index_bits/n_a/n_b` on `GeneratedFoldStep` describe the final folded-witness
group. `setup_prefix_group` describes the offloaded setup-prefix precommitted
group. `log_basis` is shared across all groups in that fold and is stored only on
`GeneratedFoldStep`. `n_d` is also shared and stored only on `GeneratedFoldStep`.
Do not add per-group `log_basis` or per-group `n_d` to the generated row.

`setup_prefix_group` is `Some(...)` exactly when the fold consumes an incoming
setup prefix. It may appear on a fold whose own `setup_contribution_mode` is
`Direct`, because a direct fold can prove an incoming setup-prefix opening and
then stop forwarding setup.

### Setup-Prefix Slots

Keep these existing types:

```text
SetupPrefixSlotId
SetupPrefixSlot
SetupPrefixVerifierSlot
SetupPrefixProverRegistry
SetupPrefixVerifierRegistry
```

No persistent metadata is missing for planning:

```text
natural_len             exact active coefficient count
n_prefix                committed power-of-two domain
level_params_digest     exact proposed commitment params
commitment              verifier-visible prefix commitment
hint                    prover-only commitment witness material
```

Transcript-derived opening points and evaluations must not be stored in a
reusable slot.

### Recursive State

Do not add `CarriedOpeningClaim` or `CarriedOpeningKind`. Preserve the existing
folded-witness fields in `SuffixProverState` and `SuffixVerifierState`, and add
only optional setup-specific state:

```text
prover:
  selected SetupPrefixSlot reference
  setup opening point
  setup opening value

verifier:
  selected SetupPrefixVerifierSlot reference
  setup opening point
  setup opening value
```

Concrete ownership may use an ID plus registry lookup instead of a long-lived
reference if required by Rust lifetimes. It must still use the existing slot
types, not a duplicate claim model.

When setup state is present, construct the successor batch with existing
`OpeningClaims::from_groups`, `PolynomialGroupClaims`, and `ProverOpeningData`
APIs.

### Stage-3 Proof

Add one challenge-field element to `SetupSumcheckProof`:

```rust
pub setup_prefix_opening: F,
```

The offloaded verifier cannot derive `S_i(rho_setup)` by scanning the setup
matrix. Stage 3 verifies this supplied value in its terminal relation, binds it
to the transcript, and carries it with the selected verifier slot into the
successor fold.

## Canonical Setup-Prefix Size

### Total Active Coefficients

Generalize the existing:

```rust
pub fn active_setup_field_len(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    d_setup: usize,
) -> Result<usize, AkitaError>;
```

to grouped `LevelParams`. It continues to return one `usize`: the number of
active setup coefficients. Per-role quantities are implementation locals, not
new public structures.

For each group `g`, use the existing `LevelParamsLike` view and let:

```text
K_g       = group polynomial count
B_g       = live_block_count_g
L_g       = positions_per_block_g
delta_c_g = num_digits_commit_g
delta_o_g = num_digits_open_g
n_a_g     = A rows
n_b_g     = B rows
```

Compute:

```text
A_g = n_a_g * L_g * delta_c_g
B_g = n_b_g * K_g * n_a_g * B_g * delta_o_g
D_width_g = K_g * B_g * delta_o_g

D_shared = n_d * sum_g(D_width_g)
N_active^R = max(max_g(A_g), max_g(B_g), D_shared)
natural_len = N_active^R * D_setup
```

Equivalently, the requested form is:

```text
max over groups(max(prefix_A, prefix_B, prefix_D))
```

with every group's `prefix_D` equal to the same full shared-D footprint.

All operations are checked. The implementation should extract one internal
checked arithmetic routine and call it from:

- `active_setup_field_len`;
- `setup_required_for_inputs`;
- the footprint calculation in `SetupContributionPlan::prepare`.

Only `active_setup_field_len` is the public prefix-size result.

### Padding and Successor Fit

Keep:

```rust
padded_setup_prefix_len(natural_len)
```

as the only public padding function.

For planner successor fit, derive an independent setup-prefix precommitted
group. Do **not** test the prefix against the successor witness group's A/B
columns. The successor has two groups:

```text
final group:       folded witness, described by GeneratedFoldStep.{m,r,n_a,n_b}
precommitted group setup prefix, described by GeneratedSetupPrefixGroup.{m,r,n_a,n_b}
```

The setup-prefix group shares:

```text
ring_d      = SETUP_OFFLOAD_D_SETUP
log_basis   = successor fold log_basis
delta_open  = successor fold delta_open
delta_commit= successor fold delta_commit
fold shape  = successor fold challenge shape
```

It owns:

```text
live_block_count_prefix = 2^r_prefix
positions_per_block_prefix  = 2^m_prefix
n_a_prefix
n_b_prefix
A_prefix key
B_prefix key
```

For `ring_slots = n_prefix / D_setup`, search deterministic power-of-two block
splits satisfying:

```text
live_block_count_prefix * positions_per_block_prefix = ring_slots
```

For each split:

```text
A_width_prefix = positions_per_block_prefix * delta_commit
B_width_prefix = live_block_count_prefix * n_a_prefix * delta_open
```

derive SIS-secure `n_a_prefix` and `n_b_prefix` exactly as a singleton
precommitted group would. Select one deterministic local minimum for the prefix
group, for example the smallest grouped witness segment footprint under the same
local-minimum heuristic used elsewhere. This selected prefix group is then
inserted into `candidate.precommitted_groups`.

After the prefix group is inserted, derive one shared D key over:

```text
D_width_total = D_width_final_witness + D_width_setup_prefix
```

and store its rank in the successor `LevelParams::d_key` / generated `n_d`.
There is no per-group D key and no generated per-group `n_d`.

`setup_prefix_level_params` may still be used by setup-slot commitment code to
construct the concrete commitment params for a prefix artifact, but planner
successor fit and generated replay must not use it as a witness-group capacity
test. Reusing the successor witness group's A/B columns for the setup prefix is
incorrect.

## Planner Algorithm

### Additional DP Input

The current memo key is:

```text
(level, current_witness_len, current_witness_len_terminal, current_lb)
```

Extend it with one raw integer:

```text
incoming_n_prefix_or_zero
```

and pass `incoming_n_prefix: Option<usize>` to
`derive_candidate_level_params`.

This value is necessary because equal-length main witnesses may arrive with
different setup-prefix domains and therefore admit different current params.
`natural_len` does not affect candidate fit and remains only in the eventual
slot ID.

### Locally Minimized Candidate Derivation

Retain the current algorithm: for each `log_basis`,
`derive_candidate_level_params` scans `block_index_bits` and keeps only the candidate
with the smallest outgoing witness.

The scalar `find_schedule` path never computes or forwards an outgoing setup
prefix, every candidate mode is `Direct`, and its current DP behavior is
preserved. It does not accept `incoming_n_prefix`.

Only `find_group_batch_schedule` with a genuinely multi-group key and
`policy.recursive_setup_planning == true` uses the edge logic below. Its suffix
context retains that root-path fact while planning later folds; setup
offloading does not become available merely because a scalar suffix happens to
have two commitments.

For each existing `block_index_bits` candidate:

1. Derive main-group block geometry, A key, B key, digit depths, norms, and
   chunk metadata as today.
2. Assemble provisional main-group `LevelParams`.
3. When `incoming_n_prefix` is present, derive an independent setup-prefix
   precommitted group:
   - `group = PolynomialGroupLayout::singleton(log2(incoming_n_prefix))`;
   - `live_block_count_prefix * positions_per_block_prefix = incoming_n_prefix / D_setup`;
   - `log_basis`, digit depths, fold shape, and ring dimension are shared with
     the current fold candidate;
   - `n_a_prefix`, `n_b_prefix`, `A_prefix`, and `B_prefix` are derived for the
     prefix group itself.
4. Skip the candidate when no deterministic prefix-group split has audited A/B
   ranks.
5. Store the derived setup-prefix group in `candidate.precommitted_groups`.
6. Compute the main and setup groups' opening-segment widths.
7. Derive one SIS-secure D key over their concatenation and store it on
   `candidate.d_key`.
8. Compute the grouped intermediate witness length. Compute a terminal witness
   length only after confirming that the candidate has one group.
9. Keep only the smallest outgoing witness for this basis.

This work stays inside `derive_candidate_level_params`; no
`PrimaryLevelCandidate`, `FinalizedLevelCandidate`, or finalization helper is
introduced.

### Terminal Branch

For a fold-then-direct branch:

- require `incoming_n_prefix = None`;
- require the current opening layout to contain exactly one witness group;
- set `setup_contribution_mode = Direct`;
- use the scalar terminal row layout;
- create no outgoing setup prefix;
- derive the terminal witness shape from the scalar opening layout.

If an incoming setup prefix exists, this terminal candidate is infeasible. The
planner may choose a longer fold suffix, but it may not drop the prefix, merge it
into the witness group, or reinterpret the last fold through a grouped terminal
codec. A root-direct schedule remains valid only for a scalar root.

### Fold-Again Branch

For a fold-then-fold branch:

1. Derive `natural_len` from the current candidate's actual groups.
2. Compute `n_prefix = padded_setup_prefix_len(natural_len)`.
3. Validate the recursion config's D64 and non-chunked invariants.
4. Set:

   ```text
   must_recurse =
       level <= 1
       && n_prefix > 2^10
       && child has another Fold successor
   ```

5. If `must_recurse`, mark the current candidate `Recursive` and recursively
   plan the child with `incoming_n_prefix = Some(n_prefix)`. If no compatible
   child first fold exists, discard this fold-again candidate.
6. Otherwise mark the current candidate `Direct` and plan the child with
   `incoming_n_prefix = None`.
7. Price the resulting current level and suffix, then compare the
   existing direct/fold suffix alternatives.

Keep `SuffixResult.best_direct` and `best_fold_per_lb`; do not add a candidate
frontier. The setup rule only filters candidate edges; the existing comparator
still selects the smallest feasible proof. Search remains bounded by the
existing recursion cap.

## Generalizing Existing Grouped Layout Methods

Do not add free `group_*` sizing functions. Generalize the existing methods on
`LevelParams`:

```text
validate_root_opening_batch -> validate_opening_batch
root_group_params           -> group_params
root_group_commitment_rows  -> group_commitment_rows
root_commitment_row_range   -> commitment_row_range
root_a_row_range            -> a_row_range
root_next_w_len             -> next_w_len
root_segment_rings          -> segment_rings (private)
```

Private arithmetic should accept `&(impl LevelParamsLike + ?Sized)` where that
eliminates duplicated main/precommitted cases.

`m_row_count_for` remains the only M-row count. Its grouped branch already
counts:

```text
consistency row
final group's A rows (the A * Z relation)
final group's B rows
each precommitted group's A rows (its A * Z relation)
each precommitted group's B rows
shared D rows when WithDBlock
```

The spec does not introduce another row formula. Generalized intermediate
witness layout code calls:

```text
m_row_count_for(opening_batch.num_groups(), layout)
segment_rings for each group
```

Intermediate witness and tail functions accept the actual
`OpeningClaimsLayout`. Terminal witness and tail functions remain scalar and
must reject an opening layout with more than one group. No grouped terminal
shape helper is introduced.

## Generated Replay

### Separate Catalogs

Generate direct and recursive artifacts independently:

```text
Cfg planner policy
  -> ordinary generated module/table, including scalar keys

RecursiveCommitmentConfig<Cfg> multi-group planner policy
  -> recursive generated module/table containing only genuine multi-group keys
```

Use distinct generated module names and table constructors, for example:

```text
fp128_d64_onehot
fp128_d64_onehot_recursive
```

The exact suffix follows the existing generator naming policy. Recursive
multi-group rows must never be appended to or looked up in the ordinary table.
The recursion adapter continues to use the ordinary table for scalar keys.

Extend generated-family metadata so an eligible family can opt into a recursive
multi-group companion table. The generator runs the ordinary key grid with the
ordinary policy, then runs only the supported multi-group key grid with the
recursion adapter policy. It must not emit duplicate scalar rows into the
recursive companion. Drift guards independently regenerate and compare both
catalogs.

The recursive multi-group catalog identity binds the recursion-planning policy
bit. Supplying a direct catalog to the recursion adapter's multi-group resolver,
or a recursive catalog to an ordinary multi-group resolver, must fail identity
validation even if a row key happens to match. Scalar delegation through
`RecursiveCommitmentConfig<Cfg>` intentionally uses the ordinary catalog.

### Canonical Replay

The canonical generated walker tracks:

```rust
let mut incoming_n_prefix: Option<usize>;
```

For each fold it:

1. Expands the main params and stored mode.
2. If an incoming prefix exists, reconstructs the existing precommitted group
   from the generated `setup_prefix_group` fields and the shared fold
   `log_basis`. It must derive the prefix group's own A/B keys, not clone the
   final witness group's A/B keys.
3. Recomputes and validates shared D rank, M rows, next witness length, and
   proof bytes.
4. For a non-terminal fold, recomputes `natural_len`, `n_prefix`, and:

   ```text
   expected_mode =
     if multi_group_path
        && recursive policy
        && level <= 1
        && n_prefix > 2^10
        && successor is a nonterminal Fold
     then Recursive
     else Direct
   ```

5. Rejects when the stored mode differs from `expected_mode`.
6. For `Recursive`, validates that the generated successor is nonterminal, has
   a compatible `setup_prefix_group`, and forwards the prefix.
7. For `Direct`, forwards no prefix.
8. Rejects a terminal fold marked recursive or carrying an incoming prefix.

`schedule_from_entry`, proof-byte estimation, and public generated-row
validation already share this walker; no second replay implementation is
introduced.

On a recursive multi-group table miss, `RecursiveCommitmentConfig<Cfg>` runs
`find_group_batch_schedule` fallback with recursion planning enabled. A scalar
miss delegates to `Cfg::runtime_schedule` and the direct `find_schedule`
fallback. Table-hit and table-miss behavior therefore remain path-consistent.

## Setup Preprocessing

Current setup-prefix population resolves one synthetic schedule and clamps the
natural length. Replace that behavior by reusing the generated-key/setup-envelope
scan already owned by `akita-config`.

Refactor the existing scan so setup-envelope sizing and prefix population visit
the same deterministic genuine multi-group schedule keys selected under
`RecursiveCommitmentConfig<Cfg>`. Scalar keys and ordinary `Cfg` setup do not
populate offloading slots. Do not introduce a second `SetupScheduleCase`
representation.

For every selected schedule and every edge whose current mode is `Recursive`:

1. Derive the current opening layout.
2. Compute `natural_len` with `active_setup_field_len`.
3. Compute `n_prefix` with `padded_setup_prefix_len`.
4. Use the finalized successor's generated/precommitted setup-prefix group
   params, not the successor witness group params, as the prefix commitment
   params.
5. Build the existing `SetupPrefixSlotId`.
6. Commit and insert the existing `SetupPrefixSlot`.
7. Deduplicate by slot ID.

Scanning every supported selected schedule prepares each reachable prefix for
every successor parameter set the planner can emit. Distinct `log_basis`, `m`,
or `r` values produce distinct parameter digests and do not alias.

Delete the `.min(available_field_len)` truncation. Both natural and rounded
prefix lengths must fit setup capacity; otherwise setup construction returns
`AkitaError`. Setup envelope sizing includes the rounded prefix capacity:

```text
n_prefix / setup_generation_ring_dimension
```

## Prover and Verifier Flow

### Recursive Fold

1. Require a recursion-config schedule derived from a genuinely multi-group
   root key and resolve `exec.params.setup_contribution_mode`.
2. Run stages 1 and 2.
3. Derive the exact prefix slot selected by current geometry and successor
   params.
4. Require the slot to exist and match `natural_len`, `n_prefix`, and params
   digest.
5. Run Stage 3 and emit both `W_{i+1}(rho_w)` and
   `S_i(rho_setup)`.
6. Store the existing witness state plus optional setup slot, point, and value.
7. Construct the successor's two-group opening batch through existing opening
   APIs.

### Direct Fold

1. Evaluate setup directly as today. Under an ordinary config, every fold takes
   this path.
2. If an incoming setup group exists, require another successor fold and prove
   that incoming opening as part of the current grouped fold.
3. Emit only the next witness state.
4. Construct a one-group successor batch.

### Verifier Rejection Rules

Reject:

- a recursive fold with no successor fold;
- a recursive fold whose successor is terminal;
- a recursive fold from the scalar/singular planner path;
- a recursive fold outside uniform D64;
- a recursive chunked/distributed fold;
- a recursive fold with `n_prefix <= 2^10`;
- a recursive fold whose successor has no compatible setup-prefix group;
- a missing required prefix slot;
- a slot whose ID, lengths, commitment params, or commitment rows differ;
- an incoming-prefix presence that disagrees with the predecessor mode;
- any `Direct` step or terminal fold with more than one group;
- a generated row mode that differs from dynamically derived eligibility;
- malformed group order, row count, point projection, or setup opening.

### Rejection Ownership

The same invariant is enforced at each boundary for a different reason:

1. The planner discards grouped direct and grouped terminal candidates. If no
   supported candidate remains, planning returns `AkitaError::InvalidSetup`.
2. Canonical schedule validation rejects stale generated rows and manually
   constructed schedules whose mode, successor group, or terminal shape does not
   match this policy.
3. Setup preprocessing must materialize every exact slot required by the selected
   schedules. A missing or mismatched slot is `AkitaError::InvalidSetup`; it is
   never repaired by truncation or direct evaluation.
4. The prover repeats schedule and slot validation before transcript mutation.
   It returns `AkitaError` rather than constructing a different proof shape.
5. The verifier reconstructs the expected schedule from public inputs. A received
   grouped direct proof, grouped terminal proof, missing recursive payload, extra
   prefix group, or wrong prefix identity is `AkitaError::InvalidProof`.

These checks are intentionally redundant at trust boundaries. The planner owns
selection, while schedule validation owns the canonical structural rule.

## Proof-Size Accounting

Keep existing proof-size APIs and generalize only arguments that currently
hard-code one suffix group.

For `Recursive`, include:

```text
existing direct-mode level bytes
+ Stage-3 setup claim
+ Stage-3 carried witness opening
+ Stage-3 sumcheck rounds
+ setup-prefix opening value
```

The Stage-3 round count remains:

```text
max(setup-domain rounds, witness-domain rounds)
```

For `Direct`, preserve current bytes. Prefix commitments live in setup metadata
and are not per-proof bytes.

## Evaluation

### Acceptance Criteria

- [ ] Ordinary `Cfg` schedules are direct-only.
- [ ] `RecursiveCommitmentConfig<Cfg>` activates recursion-aware DP only for
      genuine multi-group keys.
- [ ] Scalar keys under `RecursiveCommitmentConfig<Cfg>` delegate to the
      ordinary scalar planner/catalog and contain only direct levels.
- [ ] `LevelParams` and generated rows carry a per-fold setup mode.
- [ ] In a recursion config, only levels 0 and 1 are recursive when their
      prefixes exceed `2^10` and their successor is nonterminal; levels 2 and
      above are always direct.
- [ ] A required-recursive edge with incompatible successor params is discarded
      instead of downgraded to direct.
- [ ] After applying those constraints, the existing planner comparator selects
      the smallest feasible proof.
- [ ] Recursive successors use two existing opening groups; direct successors
      use one.
- [ ] Every fold that consumes an incoming setup prefix is nonterminal; direct
      steps and terminal folds consume exactly one group.
- [ ] Generated recursive rows store `setup_prefix_group` for every fold that
      consumes an incoming setup prefix.
- [ ] `setup_prefix_group.{position_index_bits,block_index_bits,n_a,n_b}` describe the prefix group's
      own A/B matrices and never duplicate or capacity-check against the final
      witness group.
- [ ] `active_setup_field_len` retains scalar arithmetic parity and agrees with
      runtime setup use for grouped-root and witness-plus-prefix suffix layouts;
      scalar parity does not enable scalar offloading.
- [ ] Every selected recursive edge has an exact preprocessed slot.
- [ ] The recursive verifier no longer scans setup to obtain the terminal
      prefix opening.
- [ ] Generated table replay and DP fallback produce identical modes, params,
      witness lengths, and proof-byte totals.
- [ ] Direct and recursive generated catalogs are separate and reject
      cross-catalog identity mismatches.
- [ ] Terminal, unsupported, malformed, or missing-slot cases reject without
      panic.

### Testing Strategy

`akita-types`:

- scalar prefix-size parity;
- grouped `[1,3]` size with per-group A/B maxima and concatenated shared D;
- two-group `[witness, setup-prefix]` size;
- existing `m_row_count_for` includes each A/`A*Z`, B, and shared D block;
- descriptor and slot digest changes when per-level mode changes;
- threshold boundary at `2^10`.

`akita-planner`:

- scalar `find_schedule` never forwards an incoming prefix and emits only
  direct modes;
- multi-group recursion policy activates edge-aware mode selection;
- incoming prefix participates in memo identity;
- incompatible local candidates are filtered before minimum selection;
- local minimization and the existing proof comparator remain deterministic and
  bounded;
- exact level-0/level-1 threshold rule;
- required-recursive successor-fit rejection rather than direct fallback;
- independent prefix-group A/B derivation for incoming prefixes;
- terminal mode is always direct;
- terminal candidates with an incoming prefix are infeasible;
- grouped direct roots and grouped terminal folds reject;
- level 2 and later are always direct;
- generated-row and DP parity;
- direct/recursive catalog identity mismatch rejection.

`akita-config`:

- recursion adapter delegates algebra/security policy to the base config;
- recursion adapter delegates scalar keys to the base config's ordinary
  catalog/runtime planner;
- recursion adapter selects the recursive companion catalog only for genuine
  multi-group keys;
- ordinary config selects only the direct catalog;
- unsupported D or multi-chunk recursion adapters reject multi-group keys while
  scalar keys still delegate directly;
- scalar/direct and multi-group/recursive table misses invoke the matching
  planner path and policy bit.
- recursive generated catalog materializes table hits with nonempty
  `precommitted_groups` whenever `setup_prefix_group` is present.

`akita-setup`:

- all recursive edges across the shared key scan produce slots;
- different basis/split digests do not alias;
- duplicate slot IDs deduplicate;
- natural lengths are never truncated;
- rounded prefix capacity is included in the setup envelope.

Prover/verifier end to end:

- scalar root under both ordinary and recursion-adapter configs remains
  one-group/direct;
- grouped root to two-group suffix;
- zero, one, and two recursive levels according to the two threshold checks;
- level 2 and later remain direct even when their prefix exceeds the threshold;
- a two-group direct fold returning to a one-group successor;
- rejection when that two-group fold would be terminal;
- tampering with setup commitment, opening, point, slot ID, or group order;
- missing planned slot rejection.

### Performance

Track:

- dynamic planner and table-expansion time;
- generated row count and table bytes;
- setup-prefix preprocessing time and artifact bytes;
- proof bytes per fold by mode;
- verifier cycles saved by eliminating the setup scan;
- transition level where `n_prefix` falls to `2^10`.

The local-minimum suffix heuristic must remain within the existing recursion
bound. The planner must not enumerate a full split frontier.

## Execution

1. Add `RecursiveCommitmentConfig<Cfg>`, scalar-versus-multi-group routing,
   recursion policy plumbing, and the separate recursive multi-group catalog
   hook.
2. Add per-level mode to `LevelParams` and generated rows.
3. Generalize total setup-prefix sizing using existing group views.
4. Generalize existing grouped `LevelParams` methods for nonterminal folds while
   keeping terminal-layout methods scalar.
5. Extend local suffix candidate derivation with incoming `n_prefix`.
6. Implement edge-aware mode selection in the existing DP.
7. Update generated replay; emit `setup_prefix_group`; regenerate separate
   direct and recursive tables.
8. Reuse the existing setup-envelope scan for complete slot materialization.
9. Add the Stage-3 setup opening and optional setup state.
10. Generalize suffix proving/verifying and remove recursive-group guards.
11. Add tests, profiling, and durable documentation.

## Alternatives Considered

### New setup requirement and footprint structs

Rejected. `SetupPrefixSlotId`, slots, and `active_setup_field_len` already own
the durable identity and total size. Exposing per-role footprint objects would
duplicate internal arithmetic without serving the protocol.

### Call-Wide Setup Mode

Rejected. A call-time mode does not select the matching generated catalog and
cannot bind which individual folds recurse. Config selection chooses the planner
family; per-level mode in `LevelParams` binds the exact transition point.

### Scalar-Path Offloading

Rejected. The scalar planner remains the stable direct-only path. Setup
offloading relies on the multi-group machinery to carry the setup-prefix
commitment beside the folded witness, so recursion-aware planning is entered
only from a genuinely multi-group root key.

### Mixing Recursive Rows into Ordinary Tables

Rejected. The same lookup key could then resolve to different schedules
depending on an out-of-band mode, and direct-only users would pay table and
planning complexity for recursion. Separate catalogs keep config identity,
generated lookup, and DP fallback aligned.

### Exhaustive suffix candidate frontier

Rejected. Keeping all feasible splits grows quickly with recursion depth. The
existing locally minimized candidate heuristic remains the bounded planning
model; setup compatibility is an additional local filter.

### Generic carried-opening object

Rejected. Folded-witness state has no natural or padded prefix length.
Setup-prefix metadata already lives in `SetupPrefixSlot`; the existing opening
batch APIs can combine it with the witness claim.

## Documentation

When implementation lands, fold durable behavior into:

- `book/src/roadmap/verifier-offloading.md`;
- `book/src/how/configuration.md`;
- `book/src/how/proving/sumcheck-stages.md`;
- `book/src/how/recursion.md`;
- `book/src/how/verifying/matrix_evaluation.md`.

Update the statuses of related specs when their deferred work is completed.

## References

- `STACK.md`
- `specs/setup-layout-repack.md`
- `specs/setup-prefix-ladder.md`
- `specs/batched-stage3-setup-opening.md`
- `specs/setup-product-sumcheck.md`
- `specs/multi-group-batching.md`
- `specs/planner-incidence-generalization.md`
- `crates/akita-types/src/proof/setup_prefix.rs`
- `crates/akita-types/src/layout/params.rs`
- `crates/akita-types/src/opening_claims.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-planner/src/generated/walk.rs`
- `crates/akita-config/src/conservative_commitment.rs`
- `crates/akita-config/src/generated_families.rs`
- `crates/akita-setup/src/recursion.rs`
