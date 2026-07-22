# Spec: Setup-Offloading Planner

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     | Amirhossein Khajehpour, Quang Dao          |
| Created       | 2026-07-10                                 |
| Status        | active                                     |
| PR            | #301; revised by #318                      |
| Supersedes    | Fixed two-level rollout in this document   |
| Superseded-by |                                            |
| Book-chapter  | book/src/roadmap/verifier-offloading.md    |

## Revision authority

The current target is the planner-selected policy in this revision. It
supersedes the original rollout rule that forced setup offloading at fold
levels 0 and 1 above a fixed prefix threshold. That original rule is preserved
under [Legacy fixed-window rollout (archival)](#legacy-fixed-window-rollout-archival)
for review history only. It is not a current schedule invariant, generated-row
validation rule, or verifier acceptance condition.

This revision is intentionally narrower than the future multi-objective
planner. It specifies the remediation that can land with PR #318: exact
recursive proof accounting, explicit direct/offloaded alternatives, a minimum
recursive-witness contraction, and a verifier-first schedule comparator. It
does not add mixed ring dimensions, independent role bases, commitment slicing,
or a full Pareto frontier.

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
catalog, even under the recursion adapter.

For each supported nonterminal edge on the genuine multi-group path, the
planner considers two transitions:

```text
Direct:    successor receives [W]
Offloaded: successor receives [S_prefix, W]
```

An offloaded transition is feasible only when the successor can commit the
exact prefix, the complete successor witness contracts the entering balanced
witness by at least threefold, and the resulting suffix strictly reduces the
first remaining direct setup scan. The planner may select zero, one, or several
offloaded levels. No fold index, contiguity rule, or prefix-size threshold
decides the count.

The selected schedule minimizes the first remaining direct setup footprint and
uses exact estimated proof bytes, including Stage 3, as its tie-breaker.
Recursive successors use the existing multi-group representation with the setup
prefix as a precommitted group and the folded witness as the final group.
Recursive multi-group generated schedules are stored separately from ordinary
schedules. The design reuses `SetupPrefixSlotId`, `SetupPrefixSlot`,
`SetupPrefixVerifierSlot`, `OpeningClaims`, and the existing grouped commitment
machinery rather than adding parallel requirement, geometry, or carried-claim
models.

## Intent

### Goal

Provide an explicit recursion config that activates offloading only for
multi-group batches, makes offload depth a planner decision, guarantees every
selected recursive edge has a compatible preprocessed setup-prefix commitment,
and leaves singular planning direct-only.

### Invariants

- **Config selection activates recursion.** Ordinary `Cfg` planning is
  direct-only. Only the multi-group path under
  `RecursiveCommitmentConfig<Cfg>` may emit recursive levels.
- **Singular planning never offloads.** `AkitaScheduleLookupKey` values with no
  precommitted groups use the existing scalar planner and direct catalog. Every
  level is `Direct`, including under `RecursiveCommitmentConfig<Cfg>`.
- **The planner chooses offload depth.** Every supported nonterminal edge has a
  direct alternative and may also have an offloaded alternative. The planner
  may select any number of feasible offloaded edges within the ordinary
  recursion-depth bound.
- **Offloading is never mandatory by level or prefix size.** A large prefix
  makes offloading potentially valuable, but does not determine the transition.
  If an offloaded successor is incompatible or fails the viability rules, the
  direct alternative remains available.
- **The successor edge is authoritative.** A recursive fold's
  `incoming_setup_prefix` identifies the setup prefix produced by its
  predecessor. Prover, verifier, generated-table replay, setup preprocessing,
  descriptor hashing, and proof-size accounting derive the predecessor's
  offload action from that successor-owned edge.
- **Offloaded edges must contract the balanced witness.** Let `W_in` be the
  ordinary balanced-digit witness entering the successor, excluding the raw
  full-field setup prefix, and let `W_out` be the complete balanced-digit witness
  emitted after folding both groups. A selected offloaded edge satisfies
  `bits(W_in) / bits(W_out) >= 3`.
- **Offloaded suffixes must reduce direct verifier setup work.** Relative to
  evaluating the producer setup directly, the first later direct setup scan in
  the selected suffix is strictly smaller in natural field coefficients.
- **Proof accounting is complete.** Candidate proof bytes include the direct
  fold payload, extension-opening reduction, terminal payload, and every Stage
  3 setup-product payload induced by offloaded edges.
- **Recursive means an actual carried setup opening.** A recursive fold runs
  Stage 3, exposes `S_i(rho_setup)`, and passes the matching prefix slot into the
  successor's opening batch. It may not silently revert to a local setup scan.
- **The successor shape is the mode.** Fold `i` offloads if and only if recursive
  fold `i + 1` has `incoming_setup_prefix = Some(...)` and contains the matching
  setup-prefix group beside its witness group. There is no independent
  producer-side mode bit.
- **Direct means no outgoing setup group.** A direct fold may consume an
  incoming setup group, but it creates no setup claim for its successor.
- **Terminal folds are scalar and direct.** A terminal fold has no successor
  commitment, so it cannot offload its setup claim or consume an incoming setup
  group. It consumes exactly one witness group.
- **Grouped steps are nonterminal folds.** The last fold and structural terminal
  consume exactly one group. Any fold that consumes a setup-prefix group must
  itself have another fold as its successor. This is the canonical shape
  defined by `specs/multi-group-batching.md`.
- **One setup-prefix identity.** `SetupPrefixSlotId` remains the canonical
  identity. `natural_len` and `n_prefix` identify the prefix domain;
  `level_params_digest` identifies the exact commitment params, including
  `log_basis`, `position_index_bits`, `block_index_bits`, group params, and the
  successor-owned incoming-prefix edge.
- **One total-prefix calculation.** `active_setup_field_len` is the canonical
  challenge-free calculation of active setup coefficients. Planner,
  preprocessing, prover, and verifier do not maintain separate formulas.
- **The opening matrix remains shared.** Multi-group folds use one opening
  relation over the
  concatenation of all groups' opening segments. This design does not introduce
  per-group opening commitments. The recursive fold's
  `open_commit_matrix` is shared by the final witness group and every
  precommitted setup-prefix group.
- **Existing group model is canonical.** The setup prefix is represented by the
  successor's existing precommitted-group fields; the next witness is the final
  group. The setup-prefix group has its own inner/outer matrices and block
  geometry. It does not borrow the successor witness group's matrix-column
  capacities. `OpeningClaimsLayout::root_group_order` determines proof order.
- **Local minimization remains bounded in PR #318.** Recursive suffix candidate
  generation continues to retain one locally smallest next-witness candidate
  per basis. Direct/offloaded alternatives and proof-only/setup-first suffixes
  are retained, but this remediation does not create the future full Pareto
  frontier.
- **Generated and fallback schedules agree.** A generated row stores the exact
  incoming-prefix topology chosen by dynamic planning, and the canonical row
  walker recomputes every prefix transition and grouped witness length.
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
- Distributed or multi-chunk setup offloading. (No longer a non-goal: the
  `W8R2` composition of recursive setup offloading with the multi-chunk witness
  layout shipped in [`specs/distributed-setup-offloading.md`](distributed-setup-offloading.md).)
- Composition of recursive and conservative config adapters in the first
  rollout.
- Setup offloading for singular/scalar schedule keys.
- Setup offloading at ring dimensions other than the supported uniform D64
  shape.
- Globally enumerating every suffix `(log_basis, m, r)` combination.
- The future Pareto planner over proof bytes, verifier work, outgoing witness
  bits, prover work, setup storage, preprocessing, and communication.
- Backward compatibility for old generated rows, descriptors, setup artifacts,
  or proof bytes.
- Full-ladder setup artifact policy. This design materializes the exact slots
  needed by the selected supported schedules.

## Eligibility and Fold Transitions

### Per-Fold Eligibility

An offloaded candidate exists when:

```text
recursive config is selected
the root schedule key is genuinely multi-group (precommitteds is nonempty)
the producer has a nonterminal recursive successor
the successor can commit the exact padded setup prefix
the active role dimensions and witness partition are supported
the successor can consume the prefix and still emit a supported witness
```

The fold index and prefix length do not select the mode. For every supported
edge, the planner retains the ordinary direct successor and may retain an
offloaded successor. It discards the offloaded alternative unless:

```text
balanced_witness_bits_entering_successor
+ padded_setup_prefix_field_elements * field_bits
    >= 3 * complete_witness_bits_leaving_successor

first_later_direct_setup_field_len
    < producer_direct_setup_field_len
```

The contraction numerator includes both sources consumed by the successor:
the balanced-digit recursive witness and the padded full-field setup prefix.
Omitting the prefix biases the planner toward artificially inflating the
producer witness solely to pass the heuristic. The denominator includes every
balanced-digit output produced from both successor groups, including relation
or commitment suffixes represented in the current witness format.

Successor fit and contraction are candidate-feasibility conditions. They are
not verifier security assumptions. Security continues to follow from the exact
prefix commitment, descriptor binding, Stage 3 verification, and the SIS
parameters of the selected commitment matrices.

Among feasible complete schedules, the PR #318 policy compares:

```text
(
    first_direct_setup_field_len,
    exact_estimated_proof_bytes,
)
```

where `exact_estimated_proof_bytes` includes every Stage 3 payload. The future
Pareto planner may replace this policy, but generated catalogs must bind whichever
selection policy produced them.

The recursive search also rejects candidates whose exact setup-matrix envelope
exceeds `MAX_SETUP_MATRIX_FIELD_ELEMENTS`. This is a supported-runtime ceiling,
not a claim that offloading has no storage cost relative to the independently
optimized direct schedule. Comparing the direct and offloaded envelope–proof
frontiers is explicitly deferred to the multi-objective planner.

The generated catalog binds:

```text
cost model      = ExactPayloadAndSetupEnvelope
direct policy   = MinEstimatedProofPayload
recursive policy = MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
```

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
  every recursive fold has incoming_setup_prefix = None

RecursiveCommitmentConfig<Cfg>:
  scalar key:
    delegate to Cfg::runtime_schedule(key)
    use Cfg::schedule_catalog()
    every recursive fold has incoming_setup_prefix = None
  genuine multi-group key:
    planner recursion flag = true
    use Cfg::recursive_multi_group_schedule_catalog()
    enumerate direct and feasible offloaded transitions at every nonterminal edge
    let the selected suffix determine the number of offloaded levels
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

The adapter rejects unsupported base configurations before planning a
multi-group key. Under the current implementation, recursive offloading still
requires the configured setup-offload ring dimension. Distributed support is
capability-specific; the shipped W8R2 family is governed by
[`distributed-setup-offloading.md`](distributed-setup-offloading.md).

For example, an unsupported ring dimension is rejected by:

```text
Cfg::D != SETUP_OFFLOAD_D_SETUP
```

No adapter field specifies an offload count. The planner derives that count by
choosing direct or offloaded transitions in the schedule search.

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

If fold `i` has an offloaded successor, Stage 3 produces a setup-prefix opening
and fold `i + 1` receives:

```text
[S_i, W_{i+1}] in storage order
[W_{i+1}, S_i] in proof order
```

Fold `i + 1` must be nonterminal. The planner must not create this transition
when `i + 1` would be the last fold.

If fold `i` has a direct successor, fold `i + 1` receives only `[W_{i+1}]`, even
when fold `i` itself consumed two groups. A later edge remains structurally free
to offload again. Under the setup-first comparator such a transition is normally
dominated after the first direct setup scan, but this is a selection consequence,
not a schedule-validation rule.

## Typed ownership and required changes

### Successor-owned setup-prefix edge

The current typed topology is authoritative:

```rust
pub struct RecursiveFoldParams {
    pub witness: CommittedGroupParams,
    pub open_commit_matrix: OpenCommitMatrixParams,
    pub incoming_setup_prefix: Option<SetupPrefixSlotId>,
    pub witness_partition: WitnessPartition,
}
```

`incoming_setup_prefix` determines whether the predecessor offloads. Runtime
code may temporarily mirror this identity inside `witness.setup_prefix` for
layout compatibility, but canonical validation must require equality and the
mirror must eventually be derived or removed. No call-wide or producer-side
`SetupContributionMode` may choose a different proof shape.

### Generated Rows

Generated rows store the selected successor topology rather than a duplicated
producer-side mode. A recursive fold consumes an offloaded prefix exactly when
`incoming_setup_prefix` is present:

```rust
pub struct GeneratedSetupPrefixInput {
    pub natural_len: u64,
    pub d_setup: u32,
    pub commitment: GeneratedCommittedGroup,
}

pub struct GeneratedRecursiveFold {
    pub witness: GeneratedCommittedGroup,
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub incoming_setup_prefix: Option<GeneratedSetupPrefixInput>,
    pub witness_partition: GeneratedWitnessPartition,
}
```

The generated row records whichever offload count the planner selected. Replay
must not derive that count from the fold index, a prefix-size threshold, or the
artifact registry. It expands the exact stored successor edge and validates its
prefix length, commitment parameters, shared opening matrix, witness size, and
descriptor binding.

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

An offloaded edge uses the existing fused proof:

```rust
pub struct SetupSumcheckProof<E> {
    pub claim: E,
    pub setup_prefix_eval: E,
    pub next_w_eval: E,
    pub sumcheck: SumcheckProof<E>,
}
```

The offloaded verifier does not derive `setup_prefix_eval` by scanning the setup
matrix. Stage 3 verifies it in the fused setup-product and carried-witness
relation, binds it to the transcript, and carries it with the selected verifier
slot into the successor fold. The planner's exact proof estimate includes all
three field claims and the complete degree-two sumcheck.

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
B_g       = num_live_blocks_g
L_g       = num_positions_per_block_g
delta_c_g = num_digits_inner_g
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
num_live_blocks_prefix = 2^r_prefix
num_positions_per_block_prefix  = 2^m_prefix
n_a_prefix
n_b_prefix
A_prefix key
B_prefix key
```

For `ring_slots = n_prefix / D_setup`, search deterministic power-of-two block
splits satisfying:

```text
num_live_blocks_prefix * num_positions_per_block_prefix = ring_slots
```

For each split:

```text
A_width_prefix = num_positions_per_block_prefix * delta_commit
B_width_prefix = num_live_blocks_prefix * n_a_prefix * delta_open
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
(level, current_witness_len, current_lb, incoming_setup_prefix_or_zero)
```

Pass `incoming_setup_prefix: Option<usize>` to
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
prefix, and its current DP behavior is preserved. It does not accept an
incoming setup prefix.

Only `find_group_batch_schedule` with a genuinely multi-group key and
`policy.recursive_setup_planning == true` uses the edge logic below. Its suffix
context retains that root-path fact while planning later folds; setup
offloading does not become available merely because a scalar suffix happens to
have two commitments.

For each existing `block_index_bits` candidate:

1. Derive main-group block geometry, A key, B key, digit depths, norms, and
   chunk metadata as today.
2. Assemble provisional main-group `CommittedGroupParams`.
3. When `incoming_setup_prefix` is present, derive an independent setup-prefix
   precommitted group:
   - `group = PolynomialGroupLayout::singleton(log2(n_prefix))`;
   - `num_live_blocks_prefix * num_positions_per_block_prefix = n_prefix / D_setup`;
   - `log_basis`, digit depths, fold shape, and ring dimension are shared with
     the current fold candidate;
   - `n_a_prefix`, `n_b_prefix`, `A_prefix`, and `B_prefix` are derived for the
     prefix group itself.
4. Skip the candidate when no deterministic prefix-group split has audited A/B
   ranks.
5. Store the derived setup-prefix group in `candidate.precommitted_groups`.
6. Compute the main and setup groups' opening-segment widths.
7. Derive one SIS-secure opening matrix over their concatenation and store it
   on the recursive fold.
8. Compute the grouped intermediate witness length. Compute a terminal witness
   length only after confirming that the candidate has one group.
9. Keep only the smallest outgoing witness for this basis.

This work stays inside `derive_candidate_level_params`; no
`PrimaryLevelCandidate`, `FinalizedLevelCandidate`, or finalization helper is
introduced.

### Terminal Branch

For a fold-then-direct branch:

- require `incoming_setup_prefix = None`;
- require the current opening layout to contain exactly one witness group;
- use the scalar terminal row layout;
- create no outgoing setup prefix;
- derive the terminal witness shape from the scalar opening layout.

If an incoming setup prefix exists, this terminal candidate is infeasible. The
planner may choose a longer fold suffix, but it may not drop the prefix, merge it
into the witness group, or reinterpret the last fold through a grouped terminal
codec. The folded-only protocol has no root-direct fallback; an infeasible
scalar root is rejected as `UnsupportedSchedule` as well.

### Fold-Again Branch

For a fold-then-fold branch:

1. Derive `natural_len` from the current candidate's actual groups.
2. Compute `n_prefix = padded_setup_prefix_len(natural_len)`.
3. Validate the recursion config's supported ring-dimension and witness-partition
   capabilities.
4. Plan the direct child with `incoming_setup_prefix = None`.
5. When the child is nonterminal, independently plan the offloaded child with
   `incoming_setup_prefix = Some(natural_len)`.
6. Discard the offloaded alternative if prefix derivation, successor fit, the
   threefold contraction rule, or strict direct-setup reduction fails.
7. Add the current direct payload, extension-opening reduction, applicable
   Stage 3 payload, and child suffix payload.
8. Retain setup-first and proof-only choices per successor basis.

The search remains bounded by the existing recursion cap and local
one-layout-per-basis minimization. PR #318 does not retain the future full
candidate frontier.

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

The canonical generated walker expands the successor-owned edge directly. It
tracks:

```rust
let mut incoming_setup_prefix: Option<GeneratedSetupPrefixInput>;
```

For each fold it:

1. Expands the root, recursive folds, and terminal step from the generated row.
2. If a recursive fold has `incoming_setup_prefix`, reconstructs that prefix
   group's own inner and outer commitment matrices. It must not clone the
   ordinary witness group's matrix parameters.
3. Recomputes the predecessor's `natural_len` and padded prefix length and
   validates them against the stored input.
4. Recomputes and validates the shared opening-matrix rank, relation rows,
   complete next-witness length, Stage 3 bytes, and total proof bytes.
5. Validates that the generated incoming prefix is compatible with the
   predecessor setup envelope, successor group geometry, commitment params,
   witness partition, and supported ring dimensions.
6. Forwards the exact stored prefix edge to the next recursive fold. Absence of
   `incoming_setup_prefix` means the predecessor evaluates setup directly.
7. Rejects a terminal step carrying an incoming setup prefix.

Replay does not re-run the selection policy and does not derive an expected
offload count from fold indices or prefix lengths. The generated row is the
selected topology; replay proves that this topology is internally consistent
and recomputes the policy metrics used for audit output.

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

For every selected schedule and every recursive successor whose
`incoming_setup_prefix` is present:

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

### Fold With an Offloaded Successor

1. Require a recursion-config schedule derived from a genuinely multi-group
   root key and resolve the successor's `incoming_setup_prefix`.
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

- an incoming setup prefix on the terminal step or without a predecessor fold;
- an incoming setup prefix on the scalar/singular planner path;
- an incoming setup prefix outside the capabilities bound by the selected
  catalog family, including unsupported ring-dimension or witness-partition
  combinations;
- an incoming setup prefix whose natural or padded length differs from the
  predecessor's active setup envelope;
- an incoming setup prefix whose commitment params or group geometry are
  incompatible with the successor;
- a missing required prefix slot;
- a slot whose ID, lengths, commitment params, or commitment rows differ;
- duplicated prefix authorities that disagree with the successor-owned edge;
- malformed group order, row count, point projection, or setup opening.

The verifier does not re-evaluate the planner's threefold contraction heuristic
or compare alternative schedules. Those are deterministic selection rules bound
by catalog identity. The verifier enforces only the selected schedule's exact
topology, commitment security, transcript binding, and setup-opening equations.

### Rejection Ownership

The same invariant is enforced at each boundary for a different reason:

1. The planner discards grouped direct and grouped terminal candidates. If no
   supported candidate remains, planning returns `AkitaError::InvalidSetup`.
2. Canonical schedule validation rejects stale generated rows and manually
   constructed schedules whose successor prefix, group geometry, or terminal
   shape is inconsistent.
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

For every offloaded edge, include:

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

The DP comparator and `FoldScheduleEstimate` use the same complete accounting.
Stage 3 is not appended only after schedule selection: its setup claim, carried
witness opening, sumcheck messages, and setup-prefix opening value are part of
the candidate score that decides whether and how long to offload.

## Evaluation

### Acceptance Criteria

- [ ] Ordinary `Cfg` schedules are direct-only.
- [ ] `RecursiveCommitmentConfig<Cfg>` activates recursion-aware DP only for
      genuine multi-group keys.
- [ ] Scalar keys under `RecursiveCommitmentConfig<Cfg>` delegate to the
      ordinary scalar planner/catalog and contain only direct levels.
- [ ] Every supported nonterminal edge considers a direct successor and may
      consider an offloaded successor; no fixed fold count or prefix threshold
      selects the mode.
- [ ] The planner may select zero, one, or several offloaded edges,
      bounded only by ordinary recursion depth and capability constraints; it
      does not impose contiguity as a structural rule.
- [ ] Every selected offloaded edge contracts the entering balanced witness by
      at least threefold after counting both the recursive witness and padded
      full-field prefix inputs, and strictly reduces the first remaining direct
      setup scan.
- [ ] The selected schedule lexicographically minimizes first direct setup
      footprint and exact estimated proof bytes within the supported setup
      envelope.
- [ ] The materialized estimate reports the exact setup envelope and selected
      offload-edge count, and recomputation agrees with the cached DP value.
- [ ] Exact proof accounting includes every Stage 3 payload before candidate
      comparison.
- [ ] Recursive successors use two existing opening groups; direct successors
      use one.
- [ ] Every fold that consumes an incoming setup prefix is nonterminal, and the
      successor-owned `incoming_setup_prefix` is the sole topology authority.
- [ ] Generated recursive rows store the exact setup-prefix commitment params
      for every fold that consumes an incoming prefix.
- [ ] Setup-prefix commitment params describe the prefix group's own inner and
      outer matrices and never clone the ordinary witness group's matrices.
- [ ] `active_setup_field_len` retains scalar arithmetic parity and agrees with
      runtime setup use for grouped-root and witness-plus-prefix suffix layouts;
      scalar parity does not enable scalar offloading.
- [ ] Every selected recursive edge has an exact preprocessed slot.
- [ ] The recursive verifier no longer scans setup to obtain the terminal
      prefix opening.
- [ ] Generated table replay and DP fallback produce identical topology,
      params, witness lengths, Stage 3 bytes, and proof-byte totals.
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
- descriptor and slot digest changes when successor prefix topology changes;
- malformed or terminal incoming-prefix edges reject without panic.

`akita-planner`:

- scalar `find_schedule` never forwards an incoming prefix and emits only
  direct transitions;
- multi-group recursion policy enumerates direct and offloaded alternatives;
- incoming prefix participates in memo identity;
- incompatible local candidates are filtered before minimum selection;
- threefold contraction boundary and strict direct-setup-reduction boundary;
- exact Stage 3 accounting can change the selected suffix;
- local minimization and the setup-first/proof-only comparator remain
  deterministic and bounded;
- incompatible offloaded successor rejection preserves the direct alternative;
- independent prefix-group A/B derivation for incoming prefixes;
- terminal candidates with an incoming prefix are infeasible;
- schedules with more than two feasible offloaded edges are representable and
  replay exactly;
- generated-row and DP parity;
- direct/recursive catalog identity mismatch rejection.

`akita-config`:

- recursion adapter delegates algebra/security policy to the base config;
- recursion adapter delegates scalar keys to the base config's ordinary
  catalog/runtime planner;
- recursion adapter selects the recursive companion catalog only for genuine
  multi-group keys;
- ordinary config selects only the direct catalog;
- unsupported capability combinations reject multi-group offloaded candidates
  while scalar keys still delegate directly;
- scalar/direct and multi-group/recursive table misses invoke the matching
  planner path and policy bit.
- recursive generated catalog materializes table hits with nonempty
  `precommitted_groups` whenever `incoming_setup_prefix` is present.

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
- zero, one, two, and more offloaded levels when chosen by the planner;
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
- Stage 3 bytes per offloaded edge;
- balanced-witness contraction for every selected offloaded edge;
- first remaining direct setup footprint;
- number of selected offloaded edges;
- verifier cycles saved by eliminating the setup scan;
- exact selected proof bytes against the direct-only schedule.

The local-minimum suffix heuristic must remain within the existing recursion
bound. The planner must not enumerate a full split frontier.

## Execution

1. Remove the fixed level window and prefix-threshold mode rule from the
   recursive multi-group DP.
2. Enumerate direct and offloaded successors at every supported nonterminal
   edge.
3. Price the exact Stage 3 payload before comparing suffixes.
4. Enforce threefold balanced-witness contraction and strict reduction of the
   first remaining direct setup scan.
5. Store the exact successor-owned setup-prefix topology in generated rows and
   replay it without re-running selection.
6. Reuse the existing setup-envelope scan for complete slot materialization.
7. Regenerate recursive catalogs and add topology, accounting, and malformed
   schedule tests.
8. Add profiling and audit output for the selected offload count and every
   comparator component.

## Legacy fixed-window rollout (archival)

This section preserves the original PR #301 rollout decision for historical
review. It is not normative after the PR #318 revision above.

The first implementation used a deliberately rigid rule:

```text
eligible fold levels = 0 and 1
mandatory offload when padded prefix > 2^10
fold levels >= 2 are always direct
```

If a threshold-qualified edge could not construct a compatible successor, the
candidate was discarded rather than downgraded to direct. Generated rows stored
a producer-side `SetupContributionMode`, replay recomputed the same threshold
rule, and the existing proof-only comparator selected the smallest surviving
schedule. Distributed recursion was rejected wholesale and recursive setup
required uniform D64.

That policy was valuable as a bounded integration path: it established Stage 3,
prefix slots, carried setup openings, and generated recursive catalogs without
requiring a broader scheduler. It is superseded because the fixed window can
offload an unproductive edge, cannot choose a useful later edge, and omits the
setup footprint and Stage 3 payload from the actual planning tradeoff.

## Alternatives Considered

### New setup requirement and footprint structs

Rejected. `SetupPrefixSlotId`, slots, and `active_setup_field_len` already own
the durable identity and total size. Exposing per-role footprint objects would
duplicate internal arithmetic without serving the protocol.

### Call-Wide Setup Mode

Rejected. A call-time mode does not select the matching generated catalog and
cannot bind which individual folds offload. Config selection chooses the planner
family; each recursive successor's `incoming_setup_prefix` binds the exact
transition point.

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
