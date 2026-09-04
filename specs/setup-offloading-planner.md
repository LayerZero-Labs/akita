# Spec: Setup-Offloading Planner

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     | Amirhossein Khajehpour, Quang Dao          |
| Created       | 2026-07-10                                 |
| Status        | implemented                                |
| PR            | #301; revised by #318                      |
| Supersedes    | Fixed two-level rollout in this document   |
| Superseded-by | Flat setup/capacity portions superseded by `flat-public-matrix-and-exact-ntt-cache.md`; recursive selection remains the current policy |
| Book-chapter  | book/src/how/setup-offloading.md           |

> **Commit-API update (2026-08-10).** The public commitment flow is one
> `AkitaCommitmentScheme::commit` entry point taking a `GroupContext`:
> `scheduler_without_precommitted_groups()` for each independent prior commitment and
> `scheduler_with_precommitted_groups` for the grouped final commitment. A recursive
> companion catalog ships no row without precommitted groups at a precommit layout, so
> the caller precommits under the base `Cfg` and proves the grouped root under
> `RecursiveCommitmentConfig<Cfg>`. The recursive planning and setup-offloading
> contracts in this document are unchanged.

## Revision authority

The public-matrix derivation, setup-capacity unit, and NTT-cache contracts in
this document are historical where they conflict with
[`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md).
This document remains authoritative for recursive offloading feasibility,
contraction, and the mode-specific schedule-selection objectives.

The current target is the planner-selected policy in this revision. It
supersedes the original rollout rule that forced setup offloading at fold
levels 0 and 1 above a fixed prefix threshold. That original rule is preserved
under [Legacy fixed-window rollout (archival)](#legacy-fixed-window-rollout-archival)
for review history only. It is not a current schedule invariant, generated-row
validation rule, or verifier acceptance condition.

This revision is intentionally narrower than the future multi-objective
planner. It records the remediation shipped by PR #318: exact
recursive proof accounting, explicit direct/offloaded alternatives, a minimum
recursive-witness contraction, and a verifier-first schedule comparator. It
does not add mixed ring dimensions, independent role bases, commitment slicing,
or a full Pareto frontier. The planner policy and generated schedule contract
are shipped. The Book setup offloading chapter explains that behavior for
readers.

## Summary

Recursive setup contribution runs Stage 3 and can select a committed setup
prefix. The planner decides which folds should offload and guarantees that a
selected successor can prove the resulting prefix opening. Recursive suffix
planning uses the two-group representation when the successor carries both its
newly committed folded witness and the setup-prefix commitment selected by the
preceding fold.

This design adds `RecursiveCommitmentConfig<Cfg>`. Precommitted groups use the
generated `GroupCommitPhaseParams` and the independent commit flow specified in
[`archive/2026-Q3/multi-group-batching.md`](archive/2026-Q3/multi-group-batching.md); the earlier conservative
config adapter has been removed. The ordinary `Cfg` resolves a direct-only
schedule. Selecting the recursion adapter activates setup-aware planning for
supported scalar and genuine multi-group roots. The current compiled recursive
companions are the fp128 one-hot table and the fp128 one-hot W8R2 table. Other
base configurations do not expose a recursive catalog and are rejected. Each
supported family replays rows from its own generated catalog.

For each supported nonterminal edge under the recursion adapter, the planner
considers two transitions:

```text
Direct:    successor receives [W]
Offloaded: successor receives [S_prefix, W]
```

An offloaded transition is feasible only when the successor can commit the
exact prefix, the complete successor witness contracts the entering balanced
witness by at least threefold, and the resulting suffix strictly reduces the
power-of-two capacity of the first remaining direct setup scan. The planner may
select zero, one, or several offloaded levels. No fold index, contiguity rule,
or prefix-size threshold decides the count.

The selected recursive schedule first minimizes the power-of-two capacity
covering its total physical setup envelope. Within that capacity, it minimizes
the padded capacity of the first remaining direct setup footprint and then
exact estimated proof bytes, including Stage 3. Equal candidates use root
output-witness length and the canonical schedule descriptor.
Recursive successors use the existing multi-group representation with the setup
prefix as a precommitted group and the folded witness as the final group.
Recursive multi-group generated schedules are stored separately from ordinary
schedules. The design reuses `SetupPrefixSlotId`, `SetupPrefixSlot`,
`SetupPrefixVerifierSlot`, `OpeningClaims`, and the existing grouped commitment
machinery rather than adding parallel requirement, geometry, or carried-claim
models.

## Intent

### Goal

Provide an explicit recursion config that activates offloading for supported
scalar and multi-group roots, makes offload depth a planner decision within an
identity-bound search domain, and
guarantees every selected recursive edge has a compatible preprocessed
setup-prefix commitment.

### Invariants

- **Config selection activates recursion.** Ordinary `Cfg` planning is
  direct-only. Supported scalar and grouped paths under
  `RecursiveCommitmentConfig<Cfg>` may emit setup-offloaded levels. The
  adapter does not provide a recursive catalog for every base configuration.
- **Scalar roots use the same carried-opening machinery.** A scalar root may
  produce a setup-prefix opening; its successor receives
  `[S_prefix, W]` through the existing two-group recursive representation.
- **The planner chooses offload depth within the catalog-bound domain.** Every
  supported nonterminal edge has a direct alternative. An edge admitted by
  `RecursiveSetupSearchPolicy` may also have an offloaded alternative.
  `Exhaustive` admits every feasible producer level;
  `RootAndFirstChildV1` admits producer levels zero and one.
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
  defined by `specs/archive/2026-Q3/multi-group-batching.md`.
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
  per basis. Direct and offloaded alternatives retain both first-direct and
  payload projections.
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
  layout shipped in [`specs/archive/2026-Q3/distributed-setup-offloading.md`](archive/2026-Q3/distributed-setup-offloading.md).)
- Composition of recursive and conservative config adapters in the first
  rollout.
- Recursive setup offloading for arbitrary base configurations. The current
  adapter wires only the fp128 one-hot and fp128 one-hot W8R2 companion
  catalogs; unsupported configurations return no recursive catalog.
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
the root schedule key is a supported scalar or genuine multi-group key
the producer has a nonterminal recursive successor
the recursive setup search policy admits the producer level
the successor can commit the exact padded setup prefix
the active role dimensions and witness partition are supported
the successor can consume the prefix and still emit a supported witness
```

The prefix length does not select the mode. The planner retains the ordinary
direct successor at every supported edge. At producer levels admitted by the
recursive setup search policy, it may also retain an offloaded successor. It
discards the offloaded alternative unless:

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

Among feasible complete schedules, adaptive direct planning compares:

```text
(
    first_direct_padded_setup_capacity,
    exact_estimated_proof_bytes,
    total_setup_field_elements,
    root_output_witness_len,
    canonical_descriptor,
)
```

### Why adaptive direct planning starts with first-direct capacity

This section records the motivation for the current versioned objective. It is
not a protocol invariant, a security claim, or a claim that setup size alone
predicts verifier time. Measurements or a better verifier cost model may
justify a different objective in a later policy version.

The setup quantities in the objective have different scopes:

| Quantity | Definition |
| --- | --- |
| First-direct natural setup length | Exact active A, B, or D coefficient prefix scanned at the first edge whose setup is not offloaded |
| First-direct padded setup capacity | First-direct natural length rounded up to the smallest power of two |
| Total setup field elements | Largest physical setup-matrix or setup-slot footprint used anywhere in the schedule; this is a reusable maximum, not a sum |
| Padded total-setup capacity | Total setup field elements rounded up to the smallest power of two |

For an adaptive direct schedule, the first direct edge is the root. Its padded
capacity is a deliberately rough, verifier-focused proxy: direct setup scanning
is expected to be a major verifier cost, so the planner first prefers the
smallest power-of-two bucket for that scan. It does not minimize the natural
length exactly. If the smallest feasible natural length lies in a bucket of
capacity `C`, every other length in that same bucket is at most `C` and less
than twice the smallest length. Proof bytes can therefore select a moderately
larger scan within the winning bucket instead of suffering an arbitrary proof
regression to save a small number of setup fields.

The objective intentionally does not begin with the maximum setup footprint
over the complete direct schedule. Adaptive search gives the first two fold
levels a wider parameter domain than the uniform suffix. On small rows, a
second or later fold can consequently have a larger setup matrix than the root.
Making that suffix maximum the leading direct objective can reward a shallow
schedule that avoids the later matrix by stopping early and returning a larger
terminal witness. First-direct capacity isolates the scan the direct policy is
trying to improve from this small-row artifact. Total setup remains a later
comparison coordinate and a resource-admission quantity; the planner does not
otherwise ignore it.

The same definition also composes with setup offloading. Each offloaded edge
moves the first remaining direct scan farther into the schedule, so the metric
continues to describe the direct verifier work that remains instead of naming
a fixed fold index. Terminal A-matrix work and other later verifier costs are
not fully represented by this proxy. That limitation is part of the rationale
for keeping the objective versioned and revisable.

Recursive setup planning compares a power-of-two setup-envelope capacity
instead of the exact first coordinate:

```text
(
    next_power_of_two(total_setup_field_elements),
    first_direct_padded_setup_capacity,
    exact_estimated_proof_bytes,
    first_direct_output_witness_len,
    canonical_descriptor,
)
```

### Why recursive planning starts with padded total-setup capacity

This rationale is an empirical design hypothesis, not a protocol or security
requirement. Recursive setup offloading can reduce the first remaining direct
scan by moving a producer's active setup into a committed prefix. That move does
not eliminate setup work: the prefix has its own commitment matrices and setup
slot, and its successor adds a Stage 3 proof. Optimizing only the first-direct
scan could therefore relocate cost into a larger setup object instead of
reducing the schedule's overall setup requirement.

The recursive objective uses each coordinate for a separate purpose:

1. **Padded total-setup capacity** bounds the largest setup matrix or setup slot
   needed anywhere in the schedule. The power-of-two bucket matches the setup
   prefix's index-domain and provisioning granularity. Exact footprints within
   one bucket are deliberately treated as equivalent.
2. **First-direct padded setup capacity** then minimizes the direct verifier
   scan that remains after the selected offloaded prefix. Unlike a fixed fold
   index, this coordinate follows the point where direct setup work resumes.
3. **Exact estimated proof bytes** account for the complete wire cost, including
   every Stage 3 payload introduced by offloading. This coordinate prevents a
   setup-equivalent schedule from buying a smaller intermediate witness with a
   larger proof.
4. **First-direct output-witness length** breaks remaining proof-byte ties in
   favor of less downstream work after direct setup resumes. It is primarily a
   coarse prover-work proxy, but a smaller witness can also help later verifier
   work and sometimes downstream proof geometry. These effects are correlated,
   not a calibrated runtime or byte-cost model, so this heuristic remains after
   the setup and complete-proof coordinates. That placement also limits its
   ability to favor a larger gadget basis or another parameter choice merely
   because that choice emits fewer field elements.
5. **Canonical descriptor** gives the final deterministic representative.

There is no exact-total-setup coordinate after the padded envelope. Adding one
would partially undo the intended slack by distinguishing schedules that use
the same provisioned setup capacity. A setup budget may still reject an exact
footprint before objective comparison.

The history behind this split is also empirical. The setup-envelope pruning
experiment first applied an exact envelope objective to adaptive direct rows
and a padded envelope objective to recursive rows. Exact envelope-first direct
selection produced shallow, proof-heavy schedules, so commit `a9f8bd814`
restored direct V2 while retaining padded recursive V3. The padded recursive
coordinate also made the schedule-wide maximum an early monotone bound; local
release measurements during that experiment reduced representative recursive
searches by roughly 6–16 times. A later output-before-proof experiment in
commit `9698637df` reduced selected first-direct witnesses but exposed proof-size
regressions. The current order consequently treats first-direct output as a
proof-byte tie-break. These observations explain the policy; they do not show
that it is globally optimal, and future end-to-end measurements may justify a
new policy version.

Generated catalogs bind the versioned selection policy that produced them.

The recursive search applies `PlannerPolicy::setup_field_budget` when a host
sets it to `Some(limit)`. The shipped policy uses `None` because the
deterministic public stream has no protocol length ceiling. An explicit host
budget remains a candidate-feasibility input, so the generated catalog identity
binds `None` or the exact `Some(limit)` value alongside the selection policy.
The contraction threshold is likewise explicit as
`PlannerPolicy::min_offloaded_witness_contraction`, with a shipped value of
three.

A host setup budget is an admission and resource policy. It is not a setup
decoder allocation guard and it is not proof-system semantics. Comparing the
direct and offloaded capacity and proof frontiers remains deferred to the
multi-objective planner.

The generated catalog binds:

```text
cost model      = ExactPayloadAndSetupEnvelope
uniform direct policy = MinEstimatedProofPayloadV2
adaptive direct policy = MinFirstDirectSetupThenPayloadV2
recursive policy = MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
optional setup field budget = policy.setup_field_budget
minimum offload contraction = policy.min_offloaded_witness_contraction
```

The selection objective is an explicit catalog-identity input derived from the
schedule mode. Uniform direct planning selects `MinEstimatedProofPayloadV2`.
Adaptive direct planning retains `MinFirstDirectSetupThenPayloadV2`.
Recursive setup planning selects
`MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3`. The scalar boundary
disables recursive setup search but retains the adaptive objective when its
dimension domain remains adaptive.

The planner does not use artifact registry contents to decide mode. Registry
contents are setup-instance state and could differ between prover and verifier.
It decides from public geometry, then setup construction must materialize every
required slot.

## Recursion Config Adapter

The shipped config adapter is:

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
  planner recursion flag = true
  use the recursive companion schedule catalog
  accept selected scalar and genuine multi-group keys
  enumerate direct transitions at every nonterminal edge
  enumerate feasible offloaded transitions admitted by RecursiveSetupSearchPolicy
  let the selected suffix determine the number of offloaded levels
```

The base trait hook is default-disabled:

```rust
fn recursive_setup_planning() -> bool {
    false
}
```

The adapter overrides it to `true`. `policy_of::<Self>()` copies the value into
`PlannerPolicy` for `find_schedule`, including scalar keys selected for the
recursive companion catalog. Shipped recursive catalogs bind
`RecursiveSetupSearchPolicy::RootAndFirstChildV1`; audit callers may select
`Exhaustive` explicitly.

The base config's `schedule_catalog()` remains the direct table.
`RecursiveCommitmentConfig<Cfg>` overrides the same `schedule_catalog()` hook
and selects the matching compiled recursive companion by base-config type and
feature. There is no second `recursive_multi_group_schedule_catalog()` hook.
The recursive table contains the selected scalar and grouped keys for the
supported companion family.

Its `resolve_catalog_row_for_key` routing is:

```text
validate the SIS modulus profile
validate the config and recursive policy
resolve_generated_catalog_row_for_key(
    key,
    recursion-enabled policy_of<RecursiveCommitmentConfig<Cfg>>,
    RecursiveCommitmentConfig<Cfg>::schedule_catalog(),
)
```

Catalog resolution rejects unsupported base configurations before accepting a
recursive row. Recursive offloading uses the exact setup-prefix A and B
dimensions chosen for the consuming fold. Distributed support is
capability-specific. The shipped W8R2 family is governed by
[`archive/2026-Q3/distributed-setup-offloading.md`](archive/2026-Q3/distributed-setup-offloading.md).

For example, an unsupported setup-prefix dimension is rejected by:

```text
d_setup is not admitted by the consuming fold's A-role dispatch
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

These transitions are reachable from scalar or genuinely multi-group roots
selected under the recursion adapter. A scalar root begins with `[W_i]` and may
create `S_i`; its successor then uses the same two-group path as a grouped
root.

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
to offload again. The setup-envelope-first comparator retains that transition
when it can improve the complete schedule; this is a selection consequence,
not a schedule-validation rule.

## Typed ownership and required changes

### Successor-owned setup-prefix edge

The current typed topology is authoritative. A fold has one runtime wrapper,
and its `CommittedGroupParams` owns every group and the shared opening matrix:

```rust
pub struct FoldParams {
    pub params: CommittedGroupParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}
```

The prefix is a `GroupOpenPhaseParams` entry with `setup_natural_len` set. It is
the first entry in `params.groups`, before ordinary precommitted groups and the
fold's own group. `CommittedGroupParams::setup_prefix` reads that entry. No
second prefix identity or producer side `SetupContributionMode` may choose a
different proof shape.

### Generated Rows

Generated rows store the selected successor topology rather than a duplicated
producer side mode. A recursive fold consumes an offloaded prefix exactly when
`setup_prefix` is present:

```rust
pub struct GeneratedRecursiveFold {
    pub core: GeneratedFoldCore,
    pub setup_prefix: Option<GeneratedFrozenGroup>,
    pub payload_mode: CommitmentPayloadMode,
    pub response_l2_sq_cap: Option<u128>,
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
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    d_setup: usize,
) -> Result<usize, AkitaError>;
```

to grouped `CommittedGroupParams`. It continues to return one `usize`: the number of
active setup coefficients. Per-role quantities are implementation locals, not
new public structures.

For each group `g`, read the concrete `GroupOpenPhaseParams` in the fold's
canonical group slice and let:

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
final group:       folded witness, described by GeneratedFoldCore.{m,r,n_a,n_b}
precommitted group setup prefix, described by GeneratedSetupPrefix.{m,r,n_a,n_b}
```

The setup-prefix group shares:

```text
ring_d      = SETUP_OFFLOAD_D_SETUP
log_basis   = successor fold log_basis
delta_open  = successor fold delta_open
delta_commit= successor fold delta_commit
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

and store its rank in the successor `CommittedGroupParams::d_key` / generated `n_d`.
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

Pass the prefix cache and natural length together as
`RecursiveSetupPrefix::Search` to `derive_fold_candidates`.

This value is necessary because equal-length main witnesses may arrive with
different setup-prefix domains and therefore admit different current params.
`natural_len` does not affect candidate fit and remains only in the eventual
slot ID. Candidate fit always uses the complete `n_prefix` source: no planner,
generated row, or runtime validator may substitute
`ceil(natural_len / D_setup)` for `n_prefix / D_setup`.

### Locally Minimized Candidate Derivation

Retain the current algorithm: for each `log_basis`, `derive_fold_candidates`
scans `block_index_bits`. `FoldCandidatePolicy::Best` keeps the best
contracting candidate, while `Frontier` retains every contracting split needed
by setup-offloading search.

Any `find_schedule` request with `policy.recursive_setup_planning == true`
uses the edge logic below at producer levels admitted by
`policy.recursive_setup_search_policy`. This includes a scalar application
root: its first fold may produce an outgoing setup-prefix claim, and the
successor then opens the setup prefix and folded witness as two groups.
Ordinary scalar families retain direct-only planning.

For each existing `block_index_bits` candidate:

1. Derive main-group block geometry, A key, B key, digit depths, norms, and
   chunk metadata as today.
2. Assemble provisional main-group `CommittedGroupParams`.
3. When `incoming_setup_prefix` is present, derive an independent setup-prefix
   precommitted group:
   - `group = PolynomialGroupLayout::singleton(log2(n_prefix))`;
   - `num_live_blocks_prefix * num_positions_per_block_prefix = n_prefix / D_setup`;
   - `log_basis`, digit depths, and ring dimension are shared with
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

This work stays inside `derive_fold_candidates`; no
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
8. Retain first-direct and payload projections per successor basis.

The search remains bounded by the existing recursion cap and local
one-layout-per-basis minimization. PR #318 does not retain the future full
candidate frontier.

## Generalizing Existing Grouped Layout Methods

Do not add free `group_*` sizing functions. Generalize the existing methods on
`CommittedGroupParams`:

```text
validate_root_opening_batch -> validate_opening_batch
root_group_params           -> group_params
root_group_commitment_rows  -> group_commitment_rows
root_commitment_row_range   -> commitment_row_range
root_a_row_range            -> a_row_range
root_next_w_len             -> next_w_len
root_segment_rings          -> segment_rings (private)
```

Private arithmetic should accept `&GroupOpenPhaseParams` where group-specific
geometry is needed. Fold-wide values such as the shared D matrix remain on
`CommittedGroupParams`.

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

RecursiveCommitmentConfig<Cfg> setup-aware planner policy
  -> recursive generated module/table containing selected scalar and
     multi-group keys
```

Use distinct generated module names and table constructors, for example:

```text
fp128_d64_onehot
fp128_d64_onehot_recursive
```

The exact suffix follows the existing generator naming policy. Recursive rows
must never be appended to or looked up in the ordinary table. The recursion
adapter resolves both scalar and grouped recursive keys from its companion
catalog.

Extend generated-family metadata so an eligible family can opt into a recursive
companion table. The generator runs the ordinary key grid with the ordinary
policy, then runs the selected scalar and grouped recursive key grid with the
recursion adapter policy. Drift guards independently regenerate and compare
both catalogs.

The recursive catalog identity binds the recursion-planning policy bit.
Supplying a direct catalog to the recursion adapter's resolver, or a recursive
catalog to an ordinary resolver, must fail identity validation even if a row
key happens to match.

### Canonical Replay

The canonical generated walker expands the successor-owned edge directly. It
tracks:

```rust
let mut incoming_setup_prefix: Option<GeneratedFrozenGroup>;
```

For each fold it:

1. Expands the root, recursive folds, and terminal step from the generated row.
2. If a recursive fold has `setup_prefix`, reconstructs that prefix
   group's own inner and outer commitment matrices. It must not clone the
   ordinary witness group's matrix parameters.
3. Recomputes the predecessor's `natural_len` and full-prefix length and
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

At runtime, a recursive table miss is unsupported and
`RecursiveCommitmentConfig<Cfg>::resolve_catalog_row_for_key` returns an error.
The planner is used offline by catalog generation and drift checks, not as a
runtime fallback. This keeps table-backed resolution strict and prevents a
prover and verifier from silently selecting different schedules.

## Setup Preprocessing

Setup-prefix population reuses the generated-key and setup-envelope scan owned
by `akita-config`. It visits the same deterministic scalar and grouped schedule
keys selected under `RecursiveCommitmentConfig<Cfg>`. Ordinary `Cfg` setup does
not populate offloading slots. There is no second `SetupScheduleCase`
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

Natural and rounded prefix lengths are checked against setup capacity. Setup
construction returns `AkitaError` when either does not fit, and does not
truncate the required prefix. Setup envelope sizing includes the rounded prefix
capacity:

```text
n_prefix / setup_generation_ring_dimension
```

## Prover and Verifier Flow

### Fold With an Offloaded Successor

1. Require a supported recursion-config schedule and resolve the successor's
   `incoming_setup_prefix`. The application root may be scalar or genuinely
   multi-group; a scalar root becomes a two-group successor when it offloads.
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
- an incoming setup prefix outside the capabilities bound by the selected
  catalog family, including unsupported ring-dimension or witness-partition
  combinations;
- an incoming setup prefix whose natural support or full-prefix length differs from the
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

- [x] Ordinary `Cfg` schedules are direct-only.
- [x] `RecursiveCommitmentConfig<Cfg>` activates recursion-aware DP for
      selected scalar and genuine multi-group keys.
- [x] Scalar recursive rows use the companion catalog and provision every
      carried setup-prefix opening required by the selected schedule.
- [x] Every supported nonterminal edge considers a direct successor. An edge
      admitted by `RecursiveSetupSearchPolicy` may also consider an offloaded
      successor; no prefix threshold selects the mode.
- [x] The planner may select zero, one, or several offloaded edges,
      bounded by the identity-bound search domain, ordinary recursion depth,
      and capability constraints. It does not impose contiguity as a
      structural rule.
- [x] Every selected offloaded edge contracts the entering balanced witness by
      at least threefold after counting both the recursive witness and padded
      full-field prefix inputs, and strictly reduces the padded capacity of the
      first remaining direct setup scan.
- [x] The selected recursive schedule lexicographically minimizes padded total
      setup-envelope capacity, first-direct padded setup capacity, first-direct
      output-witness length, exact estimated proof bytes, and the canonical
      descriptor.
- [x] The materialized estimate reports the exact setup envelope and selected
      offload-edge count, and recomputation agrees with the cached DP value.
- [x] Exact proof accounting includes every Stage 3 payload before candidate
      comparison.
- [x] Recursive successors use two existing opening groups; direct successors
      use one.
- [x] Every fold that consumes an incoming setup prefix is nonterminal, and the
      successor-owned `incoming_setup_prefix` is the sole topology authority.
- [x] Generated recursive rows store the exact setup-prefix commitment params
      for every fold that consumes an incoming prefix.
- [x] Setup-prefix commitment params describe the prefix group's own inner and
      outer matrices and never clone the ordinary witness group's matrices.
- [x] `active_setup_field_len` retains scalar arithmetic parity and agrees with
      runtime setup use for grouped-root and witness-plus-prefix suffix layouts;
      scalar parity does not enable scalar offloading.
- [x] Every selected recursive edge has an exact preprocessed slot.
- [x] The recursive verifier no longer scans setup to obtain the terminal
      prefix opening.
- [x] Generated table replay and offline planner regeneration produce identical
      topology, params, witness lengths, Stage 3 bytes, and proof-byte totals.
- [x] Direct and recursive generated catalogs are separate and reject
      cross-catalog identity mismatches.
- [x] Terminal, unsupported, malformed, or missing-slot cases reject without
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

- ordinary scalar `find_schedule` emits only direct transitions;
- recursive scalar and multi-group policies enumerate direct and offloaded
  alternatives;
- incoming prefix participates in memo identity;
- incompatible local candidates are filtered before minimum selection;
- threefold contraction boundary and strict direct-setup-reduction boundary;
- exact Stage 3 accounting can change the selected suffix;
- local minimization and the first-direct/payload projections remain
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
- recursion adapter selects scalar and grouped rows from the recursive
  companion catalog;
- ordinary config selects only the direct catalog;
- unsupported capability combinations reject offloaded candidates;
- offline direct and recursive catalog generation uses the matching planner
  path and policy bit; runtime catalog misses reject;
- recursive generated rows carry at least one `incoming_setup_prefix`,
  regardless of whether the application root is scalar or grouped.

`akita-setup`:

- all recursive edges across the shared key scan produce slots;
- different basis/split digests do not alias;
- duplicate slot IDs deduplicate;
- natural lengths are never truncated;
- rounded prefix capacity is included in the setup envelope.
- schedule-scoped verifier capacity skips every offloaded producer, retains
  the first remaining direct producer and terminal matrix, and rejects a
  prefix shorter than either requirement;
- all verifier setup-prefix registry entries survive matrix-prefix narrowing.

Prover/verifier end to end:

- scalar root under ordinary config remains one-group/direct;
- scalar root under recursion-adapter config may hand off to a two-group
  setup-prefix-plus-witness suffix;
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
- first remaining direct padded setup capacity and natural length;
- number of selected offloaded edges;
- verifier cycles saved by eliminating the setup scan;
- exact selected proof bytes against the direct-only schedule.

The catalog identity binds both `RecursiveSplitSearchPolicy` and
`RecursiveSetupSearchPolicy`. Production catalogs use
`BoundedBalancedExtremesV1` for recursive splits: states through twelve reduced
variables are exhaustive, while larger states search both extremes and a
radius-two window around the balance estimate. Shipped recursive setup catalogs
use `RootAndFirstChildV1`, which considers offloaded edges produced at levels
zero and one while retaining direct traversal at every level. `Exhaustive`
remains available for small-domain oracles and audited workloads. Results under
either bounded policy are selected under that named domain and are not
described as globally optimal.

## Execution

1. Remove the fixed level window and prefix-threshold mode rule from the
   recursive multi-group DP.
2. Enumerate direct successors at every supported nonterminal edge and
   offloaded successors at levels admitted by `RecursiveSetupSearchPolicy`.
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
mandatory offload when full prefix > 2^10
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

Accepted under `RecursiveCommitmentConfig<Cfg>`. The scalar root itself remains
one application group; when it offloads setup, the successor uses the existing
multi-group machinery to carry the setup-prefix commitment beside the folded
witness. Ordinary `Cfg` remains the stable direct-only path.

### Mixing Recursive Rows into Ordinary Tables

Rejected. The same lookup key could then resolve to different schedules
depending on an out-of-band mode, and direct-only users would pay table and
planning complexity for recursion. Separate catalogs keep config identity,
generated lookup, and DP fallback aligned.

### Exhaustive suffix candidate frontier

Available as the `Exhaustive` catalog-bound policy for oracle coverage and
explicit workloads. Production uses `BoundedBalancedExtremesV1` because the
full frontier grows quickly with recursion depth. The policy tag prevents the
two search domains from sharing a catalog identity.

### Generic carried-opening object

Rejected. Folded-witness state has no natural support or full-prefix length.
Setup-prefix metadata already lives in `SetupPrefixSlot`; the existing opening
batch APIs can combine it with the witness claim.

## Documentation

The implementation is shipped. Durable behavior is folded into:

- `book/src/how/setup-offloading.md`;
- `book/src/how/configuration.md`;
- `book/src/how/proving/sumcheck-stages.md`;
- `book/src/how/recursion.md`;
- `book/src/how/verifying/matrix_evaluation.md`.

Update the statuses of related specs when their deferred work is completed.

## References

- `STACK.md`
- `specs/archive/2026-Q3/setup-layout-repack.md`
- `specs/archive/2026-Q3/setup-prefix-ladder.md`
- `specs/archive/2026-Q3/group-local-opening-points.md`
- `book/src/how/proving/sumcheck-stages.md`
- `specs/archive/2026-Q3/multi-group-batching.md`
- `specs/heterogeneous-group-source-contracts.md`
- `crates/akita-types/src/proof/setup_prefix.rs`
- `crates/akita-types/src/layout/params.rs`
- `crates/akita-types/src/opening_claims.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-planner/src/generated/walk.rs`
- `crates/akita-config/src/conservative_commitment.rs`
- `crates/akita-config/src/generated_families.rs`
- `crates/akita-setup/src/recursive_prefixes.rs`
