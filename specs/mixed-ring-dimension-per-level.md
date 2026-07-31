# Mixed Ring Dimensions Across Fold Levels and Commitment Matrices

| Field | Value |
|---|---|
| Status | Bounded direct scalar mixed-D search and one fp128 one-hot generated/runtime catalog implemented; broader topology pending |
| Review snapshot | 2026-07-31, PR #334 reviewed through `a0b436dc5` plus the fixes recorded below |
| Benchmark snapshot | 2026-07-28, release build of `25a1e94a6` |
| Recursive benchmark snapshot | 2026-07-28, working tree based on `af770e1296` |
| Primary workload | fp128 one-hot, `nv = 36`, `np = 1` |
| Primary profile mode | `onehot_fp128_mixed_dim` |
| Related spec | `specs/runtime-ring-cutover.md` |
| Projected digit layout | `specs/role-native-projected-digit-layout.md` |
| Current synthetic implementation | `crates/akita-pcs/src/test_support.rs` |
| Planner implementation | `crates/akita-planner/src/schedule_params/` |

## Purpose of this document

This is the current-state and handoff document for mixed ring dimensions in
Akita. It is written for an engineer or AI agent that needs to answer these
questions without reconstructing the branch history:

1. What dimension is owned by each matrix and each fold?
2. Which mixed schedules can the current code construct?
3. Which parts are production protocol support versus test-only scheduling?
4. Which correctness and performance problems have already been fixed?
5. Which benchmark numbers are current, historical, or invalidated?
6. How does the current offline planner select schedules?
7. How should the planner search and select per-level, per-matrix dimensions?
8. What should be tested or implemented next?

The older version of this spec grew as a chronological experiment log. That
made stale measurements and superseded names look authoritative. This rewrite
uses the current code as the source of truth and keeps historical data only
where its provenance is explicit.

## Executive state

Akita now supports both forms of mixed ring dimension needed by this
experiment:

- **Across levels:** a large-ring leading band can hand off to a smaller-ring
  recursive suffix.
- **Within a level:** the A, B, and D commitment matrices can use distinct
  dimensions `d_a/d_b/d_d`, subject to A-to-role projection divisibility.

The protocol, setup-contribution, quotient, direct verifier, recursive
verifier, and multi-group paths all understand this geometry. The offline
planner now has an opt-in direct scalar search over explicit per-matrix
dimension tuples. One generated fp128 one-hot family now records and replays a
planner-selected mixed schedule at runtime. Multi-group root search and
recursive setup offload remain outside the mixed search. The older transition
profiles in this document are synthetic schedules built in `akita-pcs` test
support and are labeled as such.

The proposed next step is to make dimension choice part of the offline planner.
The requested selection order is:

```text
1. smallest physical setup matrix, measured in base-field elements;
2. smallest exact modeled proof payload;
3. deterministic canonical tie-break only.
```

This policy is implemented by the offline direct scalar entry point and bound
to the generated mixed catalog as
`MinSetupMatrixFieldElementsThenProofPayload`. Prover and verifier remain
catalog-only and never run the planner.

The currently preferred measured design remains:

```text
L0: 128/128/128
L1: 128/64/64
L2+: 64/64/64
```

This is profile E below. It retains the D128 root commitment saving, keeps a
short D64 tail, and avoids a mixed root. The complete benchmark matrix was
rerun after the scheduler rewrite; E remains the best balanced point among the
measured D128-root profiles.

The D512 profile remains exploratory:

```text
L0: 512/128/128
L1: 128/64/64
L2+: 64/64/64
```

Its matrices and suffix boundaries are now reconstructed correctly, but its
root geometry is still promoted from a D256 planner result. Its current
measurement is valid for that corrected synthetic geometry, not evidence that
the production planner should select it.

## Terminology and invariants

### Matrix names

`RingRole` is the historical API name for a commitment matrix's fixed job. A
matrix never changes jobs:

| Matrix | API role | Dimension | Job |
|---|---|---:|---|
| A | `RingRole::Inner` | `d_a` | Carries the relation/fold witness |
| B | `RingRole::Outer` | `d_b` | Commits the next witness |
| D | `RingRole::Opening` | `d_d` | Commits opening digits |

Use these terms in new prose:

- **per-matrix ring dimensions** for a tuple such as `128/32/64`;
- **ring-dimension transition** for a change between levels;
- `role_dims` only when referring to the existing Rust API.

Do not call a transition a “role switch.” The roles are fixed; only their
dimensions change.

### A is the projection source

The canonical role-dimension validator enforces:

```text
d_b divides d_a
d_d divides d_a
d_b <= d_a
d_d <= d_a
```

B and D are independent. Neither is ordered relative to the other, so both
`128/32/64` and `128/64/32` are valid shapes when field dispatch and SIS tables
admit them.

For a matrix dimension smaller than A, the physical matrix width expands:

```text
B physical width = native B width * (d_a / d_b)
D physical width = native D width * (d_a / d_d)
```

This is an exact subcolumn expansion, not padding and not a ring embedding.

The coefficient-level order of this expansion is defined by
[`role-native-projected-digit-layout.md`](role-native-projected-digit-layout.md).
B and D **MUST** split an A value into native role subrings before gadget
decomposition. Their physical columns use
`[semantic value][role subcolumn][role digit]`.

### Across-level setup rule

The setup is generated at `CommitmentConfig::D`. Every scheduled matrix
dimension must be supported by field dispatch and divide the generation
dimension. `validate_schedule_ring_dims` is the schedule boundary check.

### One compact outgoing witness per multi-group level

In a multi-group level:

- every group owns native A and B matrix dimensions;
- the consuming level owns one shared D dimension;
- every Z/E/T segment is stored at its group's exact native coefficient width;
- quotient rows are stored at their exact native row dimensions; and
- only the complete live coefficient vector is zero-extended for the successor
  commitment and Boolean domain.

There is no batch ring dimension derived from `max_g d_a,g`. Physical units are
chunk-major with authenticated group order inside each chunk, as specified by
[`role-native-projected-digit-layout.md`](role-native-projected-digit-layout.md).

## Planner integration proposal

### Decision summary

The planner should search `d_a`, `d_b`, and `d_d` together with the existing
digit-basis and block-split choices. It must derive every width and SIS rank
from the selected dimensions, plan the complete continuation from the exact
outgoing witness, and retain enough alternatives to optimize the non-additive
setup objective correctly.

The approved search policy limits dimension choice to L0 and L1. Dimensions
are component-wise non-increasing, and L2 and later are uniform D64. Rank-one
dimension pruning is not part of the authoritative search: B and D widths
depend on upstream ranks, so a geometry-only bucket is not an equivalence
class. The correctness baseline is exhaustive L0/L1 enumeration with Pareto
frontier retention and descriptor-byte tie-breaking.

The proposed direct-schedule score is:

```text
score(schedule) = (
    max_setup_matrix_field_elements(schedule),
    exact_estimated_proof_payload_bytes(schedule),
    canonical_schedule_descriptor_bytes(schedule),
)
```

The first two components are semantic objectives. Descriptor bytes are used
only to make an exact tie deterministic; they are not a performance model.

This design deliberately does not add empirical timing weights. Generated
catalogs must be reproducible on every machine, so commit/prove/verify wall
times cannot be planner inputs.

### Current planner ownership and runtime boundary

The relevant dependency and execution flow on current `main` is:

```text
CommitmentConfig
  └─ policy_of::<Cfg>() -> PlannerPolicy
       ├─ offline: akita-planner searches and emits compact catalog rows
       └─ runtime: akita-schedules validates and expands a selected row

runtime prove/verify
  └─ CommitmentConfig::runtime_schedule
       └─ resolve_group_batch_schedule
            ├─ validate catalog identity
            ├─ binary-search exact AkitaScheduleLookupKey
            └─ expand row; never invoke the planner
```

The crate boundary is intentional:

- `akita-planner` owns offline search and table emission;
- `akita-schedules` owns planner-free runtime row validation and expansion;
- `akita-config` projects a concrete preset into a plain-value policy and
  attaches an optional generated catalog;
- `akita-types` owns security, proof-size, relation-geometry, and setup-envelope
  primitives used by both search and replay.

This proposal preserves that boundary. Mixed-D search is offline-only. A
missing catalog or row remains `UnsupportedSchedule`; it must not trigger
verifier-reachable dynamic programming.

### Current `PlannerPolicy`

`policy_of::<Cfg>()` currently projects these material policy inputs:

| Policy input | Current meaning |
|---|---|
| `ring_dimension` | One scalar `Cfg::D`, used as A, B, D, terminal, setup-generation, and suffix dimension |
| `decomposition`, `basis_range` | Digit policy; root basis is pinned to the configured minimum, later bases are searched and non-decreasing |
| SIS profile, policy, table digest | Exact role-aware minimum-rank lookup identity |
| ring challenge closure | Sparse A-role fold challenge selected by dimension |
| claim/challenge extension degrees | Extension-opening proof pricing |
| one-hot chunk size | Root one-hot collision and fold bounds |
| witness chunk policy | Single- or multi-chunk layout by level |
| recursive setup flag and limits | Whether setup-prefix offload edges may be searched |
| cost/selection IDs | Catalog-bound meaning of estimates and final comparison |

`PlannerCostModelId::ExactPayloadAndSetupEnvelope` records exact modeled proof
payload, setup envelope, Stage-3 bytes, and recursive offload metadata.

The shipped selection policies are:

| Policy | Current comparison |
|---|---|
| `MinEstimatedProofPayload` | Direct schedules: exact proof payload only |
| `MinFirstDirectSetupThenPayloadWithinSupportedEnvelope` | Recursive setup schedules: first later direct setup scan, then exact proof payload, subject to an envelope cap |

The ordinary direct policy computes a setup envelope but does not use it for
selection. The fp128 `best_dense_schedule` and `best_onehot_schedule` helpers
compare separately generated uniform-D64 and uniform-D128 catalogs by proof
bytes, then use smaller uniform D as a tie-break. They are family selectors
outside the schedule DP; they cannot produce a mixed schedule.

### Current search algorithm

For one fixed scalar D, the scalar planner does the following:

1. At the root, enumerate the configured root `log_basis` and valid
   `block_index_bits`.
2. Derive A/B/D widths, coefficient bounds, and minimum secure ranks.
3. Derive the exact outgoing witness field length.
4. Enter the memoized suffix search at that exact boundary.
5. At each recursive `(level, witness_len, current_basis, incoming_prefix)`
   state:
   - enumerate non-decreasing bases;
   - select one block split per basis with `layout_candidate_score`;
   - compare direct termination with another fold;
   - optionally compare direct and setup-offloaded child edges.
6. Materialize the selected typed `FoldSchedule`.
7. Recompute proof bytes and setup envelope and reject any disagreement with
   the cached estimates.

`layout_candidate_score` combines next-witness physical width, tensor challenge
work, chunk work, and chunk imbalance. It is a local recursive-split heuristic.
The source explicitly notes that selecting the smallest next witness is not the
same as globally minimizing current proof plus suffix cost. Therefore the
current planner should not be described as exhaustive over all recursive
block splits, even though the suffix termination decision is dynamic
programming.

The suffix memo currently retains two maps per first-fold basis:

- best by first-direct-setup then payload;
- best by payload.

That is sufficient for the two current scalar-D policies. It is not sufficient
for a setup-envelope-first mixed-D objective.

### Current mixed-D gap

The production planner assumes uniform D in all candidate derivation:

- reduced variables and ring-element counts use `policy.ring_dimension`;
- A, B, and D SIS keys use that same dimension;
- B and D widths omit A-to-role physical subcolumn ratios;
- the suffix context resolves one ring challenge and reuses it at every level;
- memo state does not distinguish dimension-dependent candidates;
- terminal expansion requires the policy's scalar D;
- setup generation and candidate dimension are one field;
- generated-row expansion rejects an A dimension different from policy D and
  reconstructs final-group B/D matrices at A's dimension.

The compact generated schema is already close to the target: it stores the
inner, outer, opening, and terminal ring dimensions independently. The table
emitter also copies dimensions from a completed schedule. The blocking work is
search, runtime expansion, policy identity, and setup accounting—not a second
schedule or proof format.

The synthetic builders demonstrate protocol feasibility but are not suitable
planner architecture. They plan a uniform schedule, mutate or rebuild selected
matrices, recompute the boundary, and invoke `plan_optimal_suffix` again. A
native planner must produce the final matrices directly; it must not generate
a uniform candidate and retarget it after selection.

### Proposed ring-dimension policy

Replace the overloaded scalar planner dimension with a plain-value search
policy conceptually equivalent to:

```rust
PlannerRingDimensionPolicy {
    setup_generation_dimension,
    a_candidates,
    b_candidates,
    d_candidates,
}
```

The exact Rust shape is an implementation detail, but the semantics are
normative:

- `setup_generation_dimension` is the setup envelope's generation ring
  dimension. It must be a supported power of two and a multiple of every
  admitted A/B/D dimension.
- `a_candidates` are dimensions with production fold-challenge support and an
  audited A-role SIS cell.
- `b_candidates` and `d_candidates` are independently audited role domains.
- Candidate lists are sorted, unique, non-empty, and catalog-identity-bound.
- The planner enumerates the Cartesian product and keeps only tuples satisfying
  the canonical A-to-role divisibility validation.
- B and D remain independent; the planner must not impose `d_b == d_d` or an
  ordering between them.
- Terminal candidates use the A domain because a terminal has only the inner
  commitment matrix.

For fp128 one-hot, the intended eventual role domains are:

```text
A: 64, 128, 256, 512
B: 32, 64, 128, 256
D: 32, 64, 128, 256
```

Actual admission is still exact-cell driven: a tuple is infeasible when its
computed coefficient bucket or width lacks a minimum secure rank.

Today D512 A coverage uses the additive
`SisTableDigest::Q128_INNER_D512`, while all existing cells use
`SisTableDigest::CURRENT`. A single `PlannerPolicy::sis_table_digest` cannot
honestly describe a schedule whose A key uses the additive digest and whose
B/D keys use the current digest. Before D512 becomes a native candidate, fold
the audited D512 A cell into one canonical generated SIS table and issue one
new whole-table digest. Do not add dimension-specific digest switching inside
the planner.

### Canonical setup objective

The selection metric must be physical base-field elements, not ring elements
at a level-local A dimension.

For one matrix:

```text
matrix_field_elements = rows * physical_columns * matrix_ring_dimension
```

For one schedule:

```text
max_setup_matrix_field_elements =
    max(
        every root/recursive/terminal A matrix,
        every root/recursive B matrix,
        every root/recursive D matrix,
        every frozen precommit A/B matrix,
        every materialized setup-prefix matrix and padded prefix
    )
```

The current `SetupMatrixEnvelope::max_setup_len` is expressed in ring elements
at the active level's A dimension. Taking a raw maximum across levels with
different A dimensions is not a general cross-D comparison. The planner cost
model should instead retain `max_setup_matrix_field_elements`. Setup
construction converts once:

```text
max_setup_len_at_generation_D =
    ceil(max_setup_matrix_field_elements / setup_generation_dimension)
```

This makes the planner score, setup-capacity check, generated-row replay, and
actual flat matrix allocation use one physical quantity. No parallel setup
formula should be introduced; extend the canonical
`setup_matrix_envelope_for_schedule`/accumulation primitives and use them
everywhere.

The policy's setup ceiling must use the same field-element unit. Any existing
field named `max_setup_envelope_field_elements` must be checked for unit
consistency rather than compared with a level-local ring-element count.

### Per-level candidate derivation

For every explicit tuple `(d_a, d_b, d_d)`, basis, and block split, derive the
candidate in this order:

1. Validate the tuple, input-witness alignment, setup-generation divisibility,
   challenge support at `d_a`, and level/chunk constraints.
2. Derive root or recursive block geometry using `d_a`.
3. Derive A's native decomposed width, A collision bucket, and minimum secure
   rank at `d_a`.
4. Derive B's native A-source width, then project it to physical B columns:

   ```text
   physical_B_width = native_B_width * (d_a / d_b)
   ```

5. Derive B's collision bucket and minimum secure rank at `d_b`.
6. Derive D's physical width from every native A-source segment. For a
   scalar level:

   ```text
   physical_D_width = native_D_width * (d_a / d_d)
   ```

   For a multi-group level, project each segment from its owning group before
   summing:

   ```text
   physical_D_width =
       sum_g(native_D_width[g] * (d_a[g] / d_d))
   ```

7. Derive D's collision bucket and minimum secure rank at `d_d`.
8. Build the final role-typed matrix parameters directly.
9. Derive fold bounds and exact proof components from those final parameters.
10. Derive the exact compact outgoing witness in coefficients:

    ```text
    live = sum_chunk_group(Z_coeffs + E_coeffs + T_coeffs)
           + sum_relation_rows(quotient_depth * native_row_dim)
    committed = successor_A_dim
                * next_power_of_two(ceil(live / successor_A_dim))
    ```

11. Compute the candidate's physical setup field-element footprint.

Missing challenge, norm bucket, SIS rank, dispatch support, alignment, or
checked arithmetic makes that candidate infeasible. It does not abort the
whole search unless the policy itself is malformed.

These formulas already exist in canonical primitives or in the exact
test-support reconstruction path. Move orchestration into the planner by
extending the existing canonical candidate functions. Do not retain
`retarget_commitment_matrices` as a production post-planning pass or add
`*_for_dims` forwarding wrappers.

### Exact dynamic programming and Pareto frontier

The setup objective composes by `max`, while proof bytes compose by addition:

```text
combined.setup = max(level.setup, child.setup)
combined.proof = level.proof + child.proof
```

This breaks a one-best-child DP. For example:

```text
child X = (setup 10, proof 100)
child Y = (setup 20, proof  50)
parent setup = 30
```

X is lexicographically better in isolation, but both complete schedules have
setup 30 under the parent, so Y is globally better. Discarding Y at the child
state is incorrect.

Each suffix state must therefore retain the nondominated frontier of:

```text
(max_setup_matrix_field_elements, exact_proof_payload_bytes)
```

Within one exact parent-visible first-step partition, candidate X can safely
prune Y only when:

```text
X.setup <= Y.setup
X.proof < Y.proof
```

A setup-only improvement is not sufficient: a larger parent can mask both
setup footprints, leaving an exact cost tie whose canonical descriptor still
has to be compared. This proof-strict rule is intentionally narrower than an
ordinary two-objective Pareto frontier.

Parent proof pricing depends on the exact first child fold (outgoing
commitment, terminal binding, and optional Stage 3). Safe pruning must be
partitioned by the child edge-visible first-step descriptor. Conceptually, a
suffix result is:

```text
first-step descriptor -> nondominated cost frontier
```

The descriptor may be represented by the existing full first
`CommittedGroupParams` or its canonical descriptor bytes. It must not be a new
partial geometry model that can drift from `level_proof_bytes`.

The approved mixed search is deliberately bounded:

1. L0 and L1 enumerate every feasible basis, admitted dimension tuple, and
   block split.
2. A child tuple is admitted only when each of `d_a`, `d_b`, and `d_d` is no
   larger than the corresponding parent dimension.
3. From L2 onward, dimensions are fixed to `64/64/64` and candidate split
   derivation reuses the existing uniform-D64 planner path.
4. A mixed domain must contain `64/64/64`, and every admitted component must
   be at least 64 so the transition back to D64 cannot increase a dimension.
5. Enumerate direct-terminal and direct-child edges, price them with the
   existing exact proof-size functions, combine physical setup cost by `max`,
   and retain the required frontier per first-step descriptor. Exact-cost ties
   survive until the root descriptor comparator chooses a canonical winner.
6. Do not terminate before L2: the terminal and every fold from L2 onward use
   D64.
7. At the root, choose the global minimum by the requested score.

`derive_candidate_level_params_all_splits` is required only at the two mixed
levels. Once the schedule returns to D64, `derive_candidate_level_params`
restores the existing uniform planner's exact split policy instead of carrying
the mixed-D exhaustive-split expansion through the complete suffix.

The L1 mixed-D memo state includes the complete parent A/B/D tuple. L2 and
later states canonicalize that ceiling to `64/64/64`, allowing suffix memo
reuse across different roots without weakening the monotonic transition.

The suffix context must resolve the A-role ring challenge per candidate
`d_a`; it cannot cache one policy-wide challenge. If setup-prefix dimensions
become dynamic, the memo key must carry the complete incoming prefix identity,
including `d_setup`, not only its natural length.

### Selection policy and the current nv=36 data

Add a new catalog-bound selection policy with semantics equivalent to:

```text
MinSetupMatrixFieldElementsThenProofPayload
```

Do not change the meaning of `MinEstimatedProofPayload` in place. Existing
uniform catalogs may continue to use the old ID until intentionally
regenerated; adaptive mixed-D catalogs use the new ID.

The current benchmark makes the requested policy's result concrete:

| Profile | Physical setup field elements | Proof bytes | Mean verify |
|---|---:|---:|---:|
| A | 135,266,304 | 93,400 | 36.196 ms |
| A′ | **67,633,152** | **94,428** | 27.712 ms |
| B | **67,633,152** | 97,824 | **21.945 ms** |
| E | **67,633,152** | 95,768 | **21.869 ms** |
| F | 90,177,536 | 98,229 | 27.153 ms |
| C | 135,266,304 | 108,171 | 35.367 ms |
| D | **67,633,152** | 108,183 | 25.931 ms |

Under an unrestricted choice among only these seven predefined schedules,
`(setup field elements, proof bytes)` selects A′: it ties B/E/D on setup and
has the smallest proof among them. That historical comparison is not a correctness reference
for the approved planner domain. The planner additionally enforces
component-wise descent and returns to D64 at L2. It also searches block splits
that do not necessarily reproduce any predefined profile.

This also proves that setup footprint alone is not a complete verifier-time
model: B and E verify about 21% faster than A′ despite the same setup
envelope. If the actual product goal is lowest verifier latency, define and
validate a separate deterministic verifier-work model, then use a new cost and
selection policy such as:

```text
(setup field elements, modeled verifier work, proof bytes)
```

Do not infer timing weights from one machine or quietly add fold count as a
hidden objective. The first implementation should follow the requested
two-component policy unless this decision is changed before coding.

### Recursive setup and multi-chunk policy

Recursive setup catalogs currently optimize:

```text
(first later direct setup scan, proof payload)
```

and the production planner/catalog restricts setup-prefix commitments to D64.
The synthetic experiment below proves that the protocol can consume a D128
prefix source with a D64 outer commitment, but it does not change that planner
policy. Recursive setup selection is a distinct semantic policy, not a special
case of setup-envelope-first selection.

Initial planner-native mixed-D work should:

- implement direct scalar schedules first;
- keep recursive setup and recursive multi-chunk families on their current
  singleton-D64 domain and selection policy;
- reject enabling mixed-D candidates when
  `recursive_setup_planning == true`.

A later cut may extend recursive setup after choosing an explicit objective.
It must either retain the current first-direct-scan priority in a
multi-objective frontier or introduce a new catalog-bound comparator. It must
also decide whether planner candidates remain fixed at
`SETUP_OFFLOAD_D_SETUP = 64` or gain an admitted prefix-dimension domain.

Ordinary direct multi-chunk schedules can follow the scalar mixed-D design once
candidate widths, chunk alignment, and local split frontiers are validated.

### Multi-group roots

Multi-group support is a second implementation cut, not a different model.
Frozen precommitted descriptors continue to own their exact A/B dimensions.
For each candidate final-group tuple:

- final A/B use the selected `d_a/d_b`;
- the root owns one shared selected `d_d`;
- shared D must divide every group's A projection source;
- each precommitted D segment is projected from that group's native A into the
  shared `d_d`;
- the outgoing witness uses exact group-native coefficient segments and native
  quotient rows;
- physical setup cost includes every frozen and final A/B matrix plus shared D;
- key identity continues to include exact frozen precommit descriptors.

The planner must replace current uses of `policy.ring_dimension` in
`d_segment_width` with the selected shared D. Outgoing sizing must call the
compact `WitnessLayout` and successor-domain geometry rather than recomputing
ring slots. Authenticated order fixes bytes; changing stable group identifiers
without changing that order must not change the length.

### Generated catalog and replay integration

Generated rows already record:

- root and recursive A dimensions;
- root and recursive B dimensions;
- root and recursive D dimensions;
- terminal A dimension;
- precommitted A/B descriptors.

Keep that schema shape. Extend the one canonical generated-entry walker so
runtime expansion:

1. validates selected dimensions against the policy's admitted domains;
2. derives input ring counts from the stored A dimension;
3. derives A width/rank at stored `d_a`;
4. projects B width and derives its rank at stored `d_b`;
5. projects all D segments and derives its rank at stored `d_d`;
6. derives exact live and committed outgoing coefficient lengths from the
   canonical compact layout;
7. recomputes physical setup field elements and exact proof bytes;
8. rejects any mismatch with the generated topology or policy.

Remove the current checks that require every stored A/terminal dimension to
equal one scalar policy D. Do not add a second mixed-D expansion path.

Catalog identity now binds:

- setup generation dimension;
- ordered A/B/D candidate domains;
- new cost-model and selection-policy IDs;
- whole SIS table digest;
- challenge-hook digest over the entire admitted A domain, including candidates
  that happened not to win a row;
- existing decomposition, extension, chunk, recursive, key, and topology
  inputs;
- dimensions actually used by emitted entries.

Changing the search domain invalidates the catalog even if the winning rows
remain byte-identical. Validation also checks that every emitted root and
recursive A/B/D tuple belongs to the bound domain, and challenge-hook coverage
is recomputed over every admitted A dimension.

### Setup/config cutover

`CommitmentConfig::D` and `PlannerPolicy::ring_dimension` currently conflate:

1. setup generation dimension;
2. uniform planner candidate dimension;
3. root/backend policy dimension.

Planner-native mixed D requires separating these concepts. The config should
expose one canonical setup-generation dimension and one canonical
ring-dimension candidate policy. Scheme setup validation dispatches on the
former; commitment and fold operations dispatch on dimensions read from the
selected schedule.

Because this repository makes no backward-compatibility guarantee, prefer a
direct rename/cutover over compatibility accessors. Do not keep a scalar-D
planner wrapper that constructs a singleton candidate list; uniform presets
can express their policy with singleton A/B/D domains.

Setup capacity still scans every supported catalog row under the requested
`(max_num_vars, max_num_polys)` capacity and takes the largest physical
field-element envelope. It then converts that envelope to generation-ring
elements once. Cache identity must include the setup-generation dimension,
catalog/policy digest, and effective schedule digest.

### Determinism and tie-breaking

The planner must be independent of hash-map iteration and thread scheduling:

- canonicalize candidate domains by sorting and deduplicating explicit
  `(d_a, d_b, d_d)` tuples (`RingDimensionSearchDomain::new`); equivalent
  reordered or duplicated input shares one value identity;
- reject empty domains and tuples that fail A-to-role divisibility validation or that are
  not divisors of the setup-generation dimension;
- enumerate bases, dimensions, and splits in a documented order;
- store frontiers in ordered collections or sort before selection/emission;
- compare semantic cost components first;
- on an exact semantic tie, compare canonical schedule descriptor bytes
  lexicographically;
- never use wall time, pointer identity, randomized hashes, or “first worker
  finished” order.

The emitted schedule descriptor and runtime-expanded descriptor must be
byte-identical.

### Non-goals

- No planner invocation in prover or verifier runtime.
- No backward-compatible aliases or thin scalar-D forwarding APIs.
- No learned or host-specific commit/prove/verify timing model.
- No arbitrary ring dimension outside explicit challenge, dispatch, and SIS
  coverage.
- No post-selection matrix retargeting.
- No change to proof serialization solely to implement planner search.
- No mixed-D recursive setup offload in the first implementation cut.

### Implementation sequence

#### Cut 0: metric and identity foundation

1. Make physical setup field elements the canonical planner/setup accounting
   unit.
2. Add the new cost/selection identity without changing existing policy IDs.
3. Separate setup-generation dimension from candidate dimensions.
4. Bind candidate domains and challenge coverage into catalog identity. ✅
5. Merge D512 A coverage into one canonical SIS table digest before admitting
   D512.

#### Cut 1: direct scalar search

1. Extend existing root/recursive candidate construction to accept explicit
   `CommitmentRingDims`.
2. Derive role-local physical widths, norms, and ranks directly.
3. Enumerate all admitted tuples and block splits at L0 and L1.
4. Enforce component-wise non-increasing transitions and a uniform-D64 L2+
   suffix.
5. Retain the unpruned L0/L1 frontier; do not apply rank-one dimension caps
   until an equivalence key is proved against the unpruned reference traversal.
6. Retain edge-safe setup/proof frontiers across the mixed boundary, then
   reuse the existing uniform-D64 split search.
7. Select by setup field elements, then proof bytes, then descriptor bytes.
8. Keep recursive setup families on singleton D64.

#### Cut 2: catalog replay and shipped adaptive family (implemented for fp128 one-hot)

1. Make the canonical generated-entry walker replay per-matrix/per-level
   dimensions.
2. Add DP-to-generated-to-runtime exact parity tests.
3. Add one adaptive fp128 one-hot family rather than another set of fixed-D
   selector wrappers.
4. Regenerate tables and update setup capacity/cache identity.

#### Cut 3: broader topology

1. Enable direct multi-chunk search.
2. Enable multi-group final-root dimension search with frozen precommits.
3. Design and implement mixed-D recursive setup offload under a separately
   approved objective.
4. Remove synthetic profile adapters only after planner-selected schedules
   reproduce their coverage and benchmarks.

### Implementation checkpoint and planner example

The current branch implements the first offline direct scalar cut while
preserving all generated catalogs:

- `RingDimensionSearchDomain` admits canonical explicit
  `(d_a, d_b, d_d)` tuples and binds the setup-generation dimension used to
  validate them;
- the one canonical `find_schedule` entry point requires an explicit domain;
  a uniform caller passes a singleton domain, while a mixed caller selects by
  physical setup field elements and then exact modeled proof bytes;
- root and recursive candidates derive role-local widths, SIS keys, and
  matrices directly;
- L0 and L1 exhaustively enumerate splits over admissible, component-wise
  descending tuples;
- dimensions are uniform D64 from L2 through the terminal;
- rank-one dimension pruning is disabled in the authoritative mixed search;
  a test-only unpruned traversal checks the production frontier and canonical
  selection while deliberately sharing canonical candidate construction and
  pricing primitives;
- hand-calculated regressions independently pin exact field-element setup
  rounding, candidate-local EOR pricing, unsupported SIS-cell skipping, and
  complete-schedule descriptor ties;
- the dedicated mixed-search memo includes the parent dimension ceiling, while
  L2+ states canonicalize to D64 and reuse the fixed planner's split policy;
- mixed-boundary states retain the required setup/proof alternatives per exact
  parent-visible first fold;
- recursive setup policies are rejected by the mixed entry point and continue
  to use the existing grouped planner.

`crates/akita-planner/examples/mixed_dimension_search.rs` exercises both the
implemented and preserved paths. With setup generation D256, `nv=18`, and:

```text
64/64/64
128/64/64
128/128/128
256/128/128
```

the constrained mixed search now selects:

```text
L0:       128/64/64, ranks 2/1/1, input 262,144, output 225,152
L1:       128/64/64, ranks 2/1/1, input 225,152, output 138,752
L2:       64/64/64,  ranks 4/1/1, input 138,752, output 105,984
terminal: D64, rank 4, input 105,984
```

Its selected score is 88,064 physical setup field elements and 77,320 modeled
proof bytes.

Release-process measurements after the bounded-search change were:

| `nv` | Observed wall time | L0 | L1 | L2+ | Physical setup fields | Proof bytes |
|---:|---:|---|---|---|---:|---:|
| 18 | 0.16 s | `128/64/64` | `128/64/64` | D64 | 88,064 | 77,320 |
| 24 | 0.22 s | `256/128/128` | `128/64/64` | D64 | 524,288 | 90,976 |
| 36 | 0.46 s | `256/128/128` | `64/64/64` | D64 | 67,108,864 | 99,368 |

The `nv=24` and `nv=36` rows supersede the earlier rank-one-pruned
checkpoint. Exhaustive L0/L1 enumeration admits the D256 root and selects it
because setup fields are the primary objective, even though the `nv=36` proof
is larger than the former pruned result.

These are planner smoke-test wall times, not a controlled benchmark; the
process and filesystem caches were warm after the first run. The material
result is that `nv=24` and `nv=36` now complete normally. Before the bounded
policy, `nv=24` exceeded one minute and `nv=36` was stopped after five minutes.

The speedup comes from policy, not approximate pruning: mixed dimensions and
exhaustive split enumeration stop after L1, monotonicity removes upward
transitions, and the complete D64 suffix reuses the existing fixed planner
split derivation. Rank-one dimension caps are intentionally absent from the
authoritative search until an equivalence key is proved and checked against the
unpruned traversal.

For the PR recursive multi-group shape, the new entry point returns the
expected unsupported-policy error because mixed recursive setup is a later
cut. The preserved grouped D64 planner still produced a valid nine-level
schedule with one setup-offload edge, a 524,288-ring-element D64 setup
envelope, and a 102,732-byte modeled proof. This confirms behavior
preservation, not planner-native recursive mixed-D support.

### Acceptance criteria

The planner integration is complete only when all of these hold:

1. Existing `find_schedule` reproduces uniform schedules, estimates, and
   generated/runtime descriptor bytes; the opt-in mixed entry point requires
   uniform D64 in its domain.
2. A small-domain unpruned traversal agrees with the constrained L0/L1 search
   and selected score; independently calculated tests cover the concrete
   formulas that traversal shares with production.
3. L0 and L1 can select different A/B/D dimensions; every later fold and the
   terminal use D64.
4. B and D can be selected independently and their widths include exact
   A-to-role projection ratios.
5. Every selected role key is covered by the canonical SIS table at its exact
   width and coefficient bucket.
6. Planner, generated-row replay, setup allocation, and
   `ensure_schedule_fits_setup` agree on physical setup field elements.
7. DP output and generated-row expansion produce identical schedule descriptor
   bytes, proof-byte estimates, level counts, witness transitions, and setup
   cost.
8. Catalog validation rejects changes to candidate domains, selection policy,
   setup generation D, SIS digest, or challenge hooks.
9. Runtime row misses reject without planner fallback or panic.
10. Existing uniform direct, recursive, multi-chunk, and multi-group benchmark
    paths retain their fast verifier kernels.
11. Mixed-D E2Es cover honest verification, wrong openings, proof/commitment
    tampering, malformed dimensions, unsupported SIS cells, and setup
    under-capacity.
12. The `nv=36` constrained search completes, obeys all transition/rank caps,
    and the complete A/A′/B/E/F/C/D benchmark matrix is rerun from one build.

### Required planner tests

| Test | Required assertion |
|---|---|
| Candidate-domain validation | Sorted unique powers of two; setup D divisible by every role candidate; uniform D64 present; no component below D64 |
| Role-width unit tests | B/D widths equal native width times exact A-source/role ratio |
| SIS admission tests | Unsupported role/dimension/bucket/width is candidate infeasibility; malformed policy is an error |
| Unpruned reference traversal | Constrained L0/L1 frontier and selected schedule match the same canonical candidate set without production pruning |
| Independent formula regressions | Hand-calculated setup rounding, EOR feasibility, SIS-cell skipping, and complete-schedule descriptor ties match the implementation |
| Parent-envelope counterexample | The DP retains the lower-proof child after a larger parent setup masks child setup differences |
| Transition tests | A/B/D are component-wise non-increasing; L2+ is exactly D64 |
| Deterministic tie-break | Exact-cost ties resolve by full canonical schedule descriptor bytes |
| Terminal tests | Mixed search does not terminate before the D64 suffix boundary |
| Generated parity | Planner schedule equals emitted/expanded schedule and estimate |
| Identity drift | Every candidate-domain or objective change invalidates the old table |
| Setup parity | Planned field-element envelope equals allocated setup capacity and runtime fit checks |
| Determinism | Repeated and parallel table generation emits byte-identical rows |
| Benchmark policy | The constrained `nv=36` search completes and reports `256/128/128 → D64` with 67,108,864 physical setup fields |

### Resolved search policy

The approved first planner cut uses the deterministic
`(physical setup fields, proof bytes, descriptor bytes)` comparator subject to
two catalog-bound constraints:

1. only L0 and L1 search mixed dimensions;
2. dimensions never increase and L2+ is uniform D64.

Rank-one pruning is intentionally absent from this cut. Reintroduce it only
after proving an equivalence key and checking it against the unpruned
traversal.

Measured verifier latency remains a possible later objective. It requires a
versioned deterministic work model; host timings are not planner inputs.

## Supported synthetic profiles

The labels below are local to this spec.

| Label | Schedule |
|---|---|
| A | uniform D64 |
| A′ | root D128, then uniform D64 (`switch = 1`) |
| B | L0–L1 uniform D128, then uniform D64 (`switch = 2`) |
| E | L0 `128/128/128`, L1 `128/64/64`, then uniform D64 |
| F | L0 `512/128/128`, L1 `128/64/64`, then uniform D64 |
| C | root `128/64/64`, then a freshly planned uniform-D128 suffix |
| D | root `128/128/64`, then a freshly planned uniform-D128 suffix |

C and D are per-matrix-root experiments. They are not D64-tail variants and
must not be described as alternatives that share E's suffix policy.

## Current deterministic schedule snapshot

This section was regenerated from the schedule builders at `25a1e94a6` for
`nv = 36`, `np = 1`. It contains geometry, not wall-clock measurements.

`setup length` is `SetupMatrixEnvelope::max_setup_len` in generation-ring
elements. Multiplying by the generation dimension and 16-byte fp128 field
elements gives the setup-vector byte footprint.

| Profile | Levels including terminal | Root `d_a/d_b/d_d` | Root ranks `n_a/n_b/n_d` | Root output field elements | Setup length | Terminal input / D |
|---|---:|---|---|---:|---:|---|
| A | 9 | `64/64/64` | `6/2/1` | 146,041,728 | 2,113,536 @ D64 | 91,904 / 64 |
| A′ | 9 | `128/128/128` | `3/1/1` | 157,319,424 | 528,384 @ D128 | 91,904 / 64 |
| B | 7 | `128/128/128` | `3/1/1` | 157,319,424 | 528,384 @ D128 | 127,488 / 64 |
| E | 7 | `128/128/128` | `3/1/1` | 157,319,424 | 528,384 @ D128 | 127,488 / 64 |
| F | 7 | `512/128/128` | `1/1/1` | 247,552,000 | 176,128 @ D512 | 127,488 / 64 |
| C | 6 | `128/64/64` | `3/2/1` | 157,324,928 | 1,056,768 @ D128 | 155,648 / 128 |
| D | 6 | `128/128/64` | `3/1/1` | 157,319,424 | 528,384 @ D128 | 155,648 / 128 |

Important transition rows:

| Profile | L1 dimensions | L1 ranks | L1 output |
|---|---|---|---:|
| B | `128/128/128` | `3/1/1` | 5,644,288 |
| E | `128/64/64` | `3/1/1` | 5,644,288 |
| F | `128/64/64` | `3/2/1` | 6,323,968 |
| C | `128/128/128` | `3/1/1` | 5,644,288 |
| D | `128/128/128` | `3/1/1` | 5,644,288 |

Exact root matrix input widths:

| Profile | A width | B width | D width |
|---|---:|---:|---:|
| A | 262,144 | 1,056,768 | 176,128 |
| A′ / B / E | 131,072 | 528,384 | 176,128 |
| F | 32,768 | 704,512 | 704,512 |
| C | 131,072 | 1,056,768 | 352,256 |
| D | 131,072 | 528,384 | 352,256 |

The F widths are the key correction from the latest scheduler work. A D512
A source with D128 B/D matrices has four physical subcolumns per native column.
Scaling matrices inherited from the D256 seed geometry undercounted these
widths; deriving them from the final D512 A source produces `704,512` for both B
and D.

## Schedule construction

All builders below live in `crates/akita-pcs/src/test_support.rs`. They are
cached by configuration type, input arity, and dimension parameters to avoid
rerunning offline dynamic programming during profiles.

### Uniform leading band: `mixed_d_per_level_schedule`

```rust
mixed_d_per_level_schedule::<Envelope, Suffix>(
    num_vars,
    num_polynomials,
    switch_at_fold,
)
```

Construction:

1. Plan the envelope schedule.
2. Keep the root and exactly the recursive folds before
   `switch_at_fold`.
3. Read the kept prefix's exact output witness length and opening basis.
4. Call `akita_planner::plan_optimal_suffix` under the suffix policy.
5. Convert the planned suffix into `FoldSchedule` wire types and validate the
   result.

This fixes the original “repriced suffix” defect. Repricing an existing D128
tail at D64 did not preserve shrinkage and could terminate with a very large
cleartext witness. The suffix planner, not the synthetic builder, remains the
single authority for fold geometry.

Adapter:

```rust
MixedDConfig<Envelope, Suffix, SWITCH_AT_FOLD>
```

### Per-matrix root: `per_matrix_ring_dims_root_schedule`

```rust
per_matrix_ring_dims_root_schedule::<Env>(
    num_vars,
    num_polynomials,
    b_ring_dim,
    d_ring_dim,
)
```

Construction:

1. Plan a uniform `Env` root.
2. Rebuild B and D from the final A-source projection geometry.
3. Recompute the root's exact outgoing witness length.
4. Replan the **complete** uniform-`Env` suffix from that boundary.
5. Derive the setup envelope from the completed mixed schedule.

Adapter:

```rust
PerMatrixRingDimsRootConfig<Env, B_RING_DIM, D_RING_DIM>
```

The suffix is uniform `Env::D`. With `Env = D128OneHot`, this produces C and D,
whose terminal remains D128.

### Multi-band transition: `ring_dimension_transition_schedule`

```rust
ring_dimension_transition_schedule::<Root, Mid, Suffix>(
    num_vars,
    num_polynomials,
    root_dims,
    middle_dims,
)
```

This is the canonical builder for E, F, and the active transition E2E:

1. Plan the root under `Root`.
2. If `Root::D == 512`, temporarily plan D256 root geometry and promote A to
   D512 using the audited D512 A-role SIS table.
3. Rebuild root B/D from the **final** A geometry.
4. Recompute the root output.
5. Plan a `Mid` suffix from that exact root output and retain its first fold.
6. Rebuild that fold's B/D matrices from its final A geometry.
7. Recompute the middle-fold output.
8. Plan the complete `Suffix` continuation from that exact output.
9. Validate the finished schedule and exact setup envelope.

The builder accepts only singleton batches. Its cache key contains
`Root`/`Mid`/`Suffix` type IDs plus root and middle B/D dimensions. The A
dimensions are encoded by the `Root` and `Mid` types.

Adapters:

```rust
RingDimensionTransitionConfig<
    Env,
    Suffix,
    MID_BD_RING_DIM,
    ROOT_D_RING_DIM,
>

ThreeBandRingDimensionTransitionConfig<
    Root,
    Mid,
    Suffix,
    ROOT_BD_RING_DIM,
    L1_BD_RING_DIM,
>
```

### Exact matrix retargeting

`retarget_commitment_matrices` is the shared B/D reconstruction path. It:

1. validates `CommitmentRingDims`;
2. derives native B and D column counts from the final A parameters;
3. multiplies by the exact A-to-role subcolumn ratio;
4. derives the collision norm from the audited SIS policy at the target
   dimension;
5. reconstructs each matrix with `try_new_with_min_rank`.

Do not reintroduce independent “retarget B,” “retarget D,” or width-scaling
helpers. The final A dimension is the projection source of truth.

## Protocol support

The schedule builders are test support, but the geometry they exercise is
handled by the regular prover and verifier.

### Canonical relation address geometry

`RelationAddressGeometry` owns:

- the exact live and committed coefficient lengths;
- the common relation-witness coefficient block;
- relation-lane capacity and variable count;
- the split between low coefficient variables and lane/column variables.

The same geometry is used by range checking, relation evaluation,
setup-contribution planning, Stage 3, and verifier replay. Do not reconstruct
these counts locally.

### Digit range check

The original transition cliff set the range-check low-variable count to zero
whenever current dimensions differed from the outgoing witness dimension. That
forced the full root domain through the expensive materialized column phase.

The current prover and verifier bind the actual shared low coefficient block:

```text
digit_range_equality_low_variable_count =
    relation_address_geometry.common_relation_witness_variable_count()
```

Uniform schedules retain their historical split. Transition and per-matrix
paths keep the compact low-round phase.

### Setup contribution

`SetupContributionPlan` is the canonical source for direct, recursive, and
offloaded setup contributions.

The plan:

- validates group ownership and row ranges;
- builds one bounded equality window;
- derives exact D/B/A physical subcolumns;
- records checked outgoing-aware contribution spans;
- materializes direct E/T/Z equality slices when required;
- evaluates structured contributions from the same spans;
- materializes or evaluates the Stage-3 setup-index weight polynomial.

For mixed dimensions, relation lanes use:

```text
common base       = common relation-witness coefficient count
role lanes        = role D / common base
role subcolumns   = A-source D / role D
lane alpha weight = alpha^(common base * lane)
```

The A matrix lane-sums into one physical A column. Smaller B/D matrices spread
their lanes across physical subcolumns.

The subcolumn axis precedes the digit axis. Projection powers
`alpha^(role subcolumn * role dimension)` are part of the projected equality
tensor. Using the notation from the role native projected digit layout spec,
`q_R = 1` omits that tensor factor and every multiplication by one. All live
segments are exact coefficient intervals, so there is no padded subcolumn
factor to skip.

### Direct verifier

The direct verifier uses one canonical projected digit plan. The plan may use
local compile-time specialization when `q_R = 1`. It **MUST NOT**
select a second relation implementation at a broad uniform-versus-mixed
dispatch boundary.

For `q_R = 1`, the plan uses the existing contiguous equality and setup
kernels. It does not allocate projection powers, evaluate a
projection power MLE, or multiply by one. Other shapes use the same plan with
explicit compact subcolumn tensor axes.

### Recursive and setup-offloaded verifier

`SetupContributionPlan::evaluate_setup_index_weight_mle` evaluates uniform,
mixed-D, multigroup, and multichunk geometry. The plan selects a contiguous
one-lane inner kernel or a projected multi-lane inner kernel while preserving
the same compact pair-equality recurrence and canonical D/B/A formula.

The performance restorations at `2205555a6` and `2f0c35b66` are intentional:

- keep one-lane deferred structured E/T/Z evaluation on its contiguous kernel;
- keep uniform Stage-3 setup-index evaluation succinct;
- reuse one alpha-power table across uniform root groups;
- skip alpha-power generation for a one-lane projection;
- materialize uniform A weights into a preallocated slice;
- dispatch the uniform group evaluator at compile-time D.

Do not replace shape-specific inner kernels without identical-geometry
benchmark evidence; they remain internal implementations of the shared plan.

### Multi-group ownership

`CommittedGroupParams::group_role_dims` resolves group-native A/B dimensions
and the level-shared D dimension. `RelationRhsLayout::row_ring_dims` is the
canonical row-dimension map.

Each group owns:

- its consistency row in native A;
- A quotient rows in native A;
- B quotient rows in native B;
- its challenge and opening arithmetic in native dimensions.

The level owns the trailing shared D rows. A shared consistency row at one
group's A dimension is invalid because coefficient partitioning between
`F[X]/(X^128 + 1)` and `F[X]/(X^64 + 1)` is not a ring homomorphism in odd
characteristic.

T and Z are serialized directly in exact group-native coefficient ranges after
native group relations are formed. Only the complete live witness receives one
successor-domain zero suffix.

## Review of the 2026-07-28 pulled commits

The pull was a fast-forward from `2f0c35b66` to `25a1e94a6` and contained
exactly two commits.

### `3f5efc333` — replan mixed matrix-ring schedules

This is a behavioral test-support fix, not only a test rename.

Changes:

- Replaced mutation of already-planned suffixes with staged calls to
  `plan_optimal_suffix`.
- Replaced separate B/D retarget helpers with
  `retarget_commitment_matrices`.
- Made width derivation use the final A projection source.
- Recomputed outgoing witness lengths after each retargeted level.
- Replanned C/D's complete D128 suffix.
- Replanned E/F's middle and D64 continuations at their exact boundaries.
- Derived every adapter's setup envelope from its actual synthetic schedule.
- Added schedule tests for:
  - exact physical B/D widths;
  - independently reproduced suffix plans;
  - D256 tableless planning;
  - D512 A-role SIS-table selection;
  - exact configured-versus-required setup envelopes.
- Replaced separate transition schedule tests with one consolidated schedule
  suite.

Review result:

- The implementation follows the single-source-of-truth rule: planner geometry
  comes from the planner, and exact matrix widths come from one reconstruction
  function.
- The D512 promotion is now internally consistent, but still heuristic because
  the planner does not choose the D512 root geometry natively.
- Existing E/F/C/D benchmark provenance became ambiguous. This spec now marks
  it explicitly.

### `25a1e94a6` — name ring dimensions precisely

This is a breaking terminology/API cleanup with no intended arithmetic change.
Backward compatibility is not provided.

| Old | Current |
|---|---|
| `compressed_role_root_schedule` | `per_matrix_ring_dims_root_schedule` |
| `CompressedRoleRootConfig` | `PerMatrixRingDimsRootConfig` |
| `role_switch_schedule` / `three_band_role_switch_schedule` | `ring_dimension_transition_schedule` |
| `RoleSwitchConfig` | `RingDimensionTransitionConfig` |
| `ThreeBandRoleSwitchConfig` | `ThreeBandRingDimensionTransitionConfig` |
| `compressed_role_e2e.rs` | `per_matrix_ring_dims_root_e2e.rs` |
| `role_switch_e2e.rs` | `ring_dimension_transition_e2e.rs` |
| `three_band_schedule.rs` | folded into `ring_dimension_transition_schedule.rs` |

Profile environment renames:

| Old | Current |
|---|---|
| `AKITA_MIXED_ROLE` | `AKITA_PER_MATRIX_RING_DIMS_ROOT` |
| `AKITA_MIXED_OUTER_D` | `AKITA_ROOT_B_RING_DIM` |
| `AKITA_MIXED_OPEN_D` | `AKITA_ROOT_D_RING_DIM` |
| `AKITA_MIXED_ROLESWITCH` | `AKITA_RING_DIMENSION_TRANSITION` |
| `AKITA_MIXED_ROLESWITCH_ROOT_D` | `AKITA_TRANSITION_ROOT_D_RING_DIM` |
| `AKITA_MIXED_THREEBAND` | `AKITA_THREE_BAND_RING_DIMENSION_TRANSITION` |
| `AKITA_MIXED_THREEBAND_ROOT_D` | `AKITA_THREE_BAND_ROOT_A_RING_DIM` |

`RingRole` and `role_dims` remain API names. User-facing descriptions now say
“A/B/D matrices,” “per-matrix ring dimensions,” and “ring-dimension
transition.”

## Test coverage

### Active acceptance tests

| Test | What it proves |
|---|---|
| `mixed_d_per_level_e2e.rs` | Cross-level D128→D64 prove/verify, replay, malformed input, tamper rejection |
| `per_matrix_ring_dims_root_e2e.rs` | Active `128/32/64` root through public PCS API; wrong opening and commitment tamper rejection |
| `ring_dimension_transition_e2e.rs` | Direct L0 `128/128/64`, L1 `128/64/64`, then D64; public PCS API and tamper rejection |
| `ring_dimension_transition_schedule.rs` | Exact widths, independently planned suffixes, D256/D512 schedule invariants, setup envelope equality, dynamic D128 recursive prefix, and W8R2 partition preservation |
| `recursive_ring_dimension_transition_e2e.rs` | CI-sized recursive mixed-D (`256/128/128 → 128/64/64 → 64`) plain and W8R2 prove/verify at `nv=24`, serialize round-trip, transcript agreement, wrong-opening rejection, and missing D64 outer-NTT rejection. The cases run serially to cap peak memory. Stage 3 tampering remains in the shared recursive-profile E2E coverage. |
| profile modes `onehot_fp128_mixed_d_multi_group_recursive*` | Benchmark-only `nv=32` recursive prove/verify for the plain and W8R2 mixed-D workloads; excluded from active `profile-ci` |

The disabled legacy fixture `mixed_role_e2e.rs` has been deleted. Active
per-matrix coverage is `per_matrix_ring_dims_root_e2e.rs`.

### Lower-level coverage

- planner tests compare the mixed Pareto frontier with an unpruned traversal,
  show that a lower D256 A rank can reduce B width even after both B matrices
  reach rank one, preserve a lower-proof child when a larger parent setup masks
  child setup differences, and require descriptor-identical concurrent
  generation.
- `akita-types` setup-contribution span tests compare dense, materialized,
  direct, deferred, single-chunk, multi-chunk, and mixed-role projections.
- verifier ring-switch tests check prepared relation geometry and deferred
  mixed setup claims.
- multi-group parameter tests check group-local dimensions, row offsets,
  compact-length independence from stable group identifiers, and descending
  group A dimensions.
- recursive and distributed setup-offload E2Es cover the deferred verifier
  path under their production feature guards.
- setup-prefix selection tests reject insufficient natural and padded capacity,
  while the recursive mixed-D E2E checks the exact dynamic D128 slot identity
  against canonical Stage 3 sizing and prepared setup capacity.

## Benchmark status

### Results (nv = 36; after exact staged replanning; timings are 2-run means)

All columns below were measured on 2026-07-28 from one release build of
`25a1e94a6`. The host was an Apple M4 Max MacBook Pro (16 cores, 64 GB) running
macOS 26.5.2. The profile used the default feature set, direct setup
contributions, 16 Rayon prove threads, 16 Rayon verify threads, tracing off,
one discarded warmup, and two sequential measured runs per column. With two
samples, the harness median is also their arithmetic mean.

| Metric | A: `D = 64` all | A′: `D = 128` root only (`switch = 1`) | B: `D = 128` L0–L1 (`switch = 2`) | E: `128/128/128 → 128/64/64 → 64` | F: `512/128/128 → 128/64/64 → 64` | C: root `128/64/64` | D: root `128/128/64` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Commit | 22.61 s | 13.58 s | 13.75 s | 13.52 s | **10.17 s** | 13.79 s | 13.42 s |
| Prove | 2.921 s | 3.308 s | 3.076 s | **3.047 s** | 4.656 s | 3.189 s | 3.159 s |
| Verify | 0.0362 s | 0.0277 s | 0.0219 s | **0.0219 s** | 0.0272 s | 0.0354 s | 0.0259 s |
| Proof bytes | **93,400** | 94,428 | 97,824 | 95,768 | 98,229 | 108,171 | 108,183 |
| Fold / terminal bytes | 41,012 / 52,388 | 42,036 / 52,392 | 34,956 / 62,868 | 32,908 / 62,860 | 35,372 / 62,857 | 35,672 / 72,499 | 35,672 / 72,511 |
| Setup vector / prover NTT cache | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | 1.44 GB / 3.61 GB | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB |
| Verifier NTT cache | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB |
| Peak RSS | 11.40 GiB | 10.45 GiB | 10.32 GiB | 10.34 GiB | 20.48 GiB | 16.51 GiB | 10.47 GiB |
| Root ranks `n_a/n_b/n_d` | `6/2/1` | `3/1/1` | `3/1/1` | `3/1/1` | `1/1/1` | `3/2/1` | `3/1/1` |
| Fold levels, including terminal | 9 | 9 | 7 | 7 | 7 | 6 | 6 |

The two retained timing samples were:

| Profile | Commit samples | Prove samples | Verify samples |
|---|---|---|---|
| A | 22.6260 s, 22.6016 s | 2.9033 s, 2.9388 s | 36.607 ms, 35.785 ms |
| A′ | 13.5415 s, 13.6132 s | 3.3199 s, 3.2969 s | 27.486 ms, 27.939 ms |
| B | 13.7369 s, 13.7621 s | 3.0878 s, 3.0644 s | 22.000 ms, 21.891 ms |
| E | 13.6481 s, 13.3853 s | 3.0292 s, 3.0643 s | 21.972 ms, 21.766 ms |
| F | 10.1663 s, 10.1765 s | 4.7063 s, 4.6053 s | 27.240 ms, 27.067 ms |
| C | 13.7317 s, 13.8389 s | 3.1873 s, 3.1916 s | 35.359 ms, 35.374 ms |
| D | 13.4459 s, 13.3866 s | 3.1818 s, 3.1368 s | 25.704 ms, 26.157 ms |

#### Follow-up reproduction after planner-native mixed-D cut

The full seven-profile matrix was rerun from `8d7598fff` on 2026-07-28 after
adding the opt-in per-matrix planner search. The build, host, direct setup
mode, thread counts, warmup count, and retained-run count matched the protocol
above.

Every deterministic result reproduced exactly:

- proof bytes and fold/terminal decomposition;
- setup vector, prover NTT cache, and verifier NTT cache;
- root ranks and fold count;
- successful prove/verify for both retained runs of every profile.

The follow-up measurements were:

| Metric | A: `D = 64` all | A′: `D = 128` root only | B: `D = 128` L0–L1 | E: `128/128/128 → 128/64/64 → 64` | F: `512/128/128 → 128/64/64 → 64` | C: root `128/64/64` | D: root `128/128/64` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Commit | 23.410 s | 13.885 s | 13.790 s | 15.996 s | **11.241 s** | 14.782 s | 14.617 s |
| Prove | 2.972 s | 3.335 s | **3.091 s** | 3.125 s | 4.894 s | 3.232 s | 3.241 s |
| Verify | 0.0360 s | 0.0271 s | **0.0212 s** | 0.0223 s | 0.0281 s | 0.0368 s | 0.0273 s |
| Proof bytes | **93,400** | 94,428 | 97,824 | 95,768 | 98,229 | 108,171 | 108,183 |
| Fold / terminal bytes | 41,012 / 52,388 | 42,036 / 52,392 | 34,956 / 62,868 | 32,908 / 62,860 | 35,372 / 62,857 | 35,672 / 72,499 | 35,672 / 72,511 |
| Setup vector / prover NTT cache | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | 1.08 GB / 2.71 GB | 1.44 GB / 3.61 GB | 2.16 GB / 5.41 GB | 1.08 GB / 2.71 GB |
| Verifier NTT cache | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB | 1.44 MB |
| Peak RSS | 11.40 GiB | 10.46 GiB | 10.32 GiB | 10.32 GiB | 20.54 GiB | 16.51 GiB | 10.35 GiB |
| Root ranks `n_a/n_b/n_d` | `6/2/1` | `3/1/1` | `3/1/1` | `3/1/1` | `1/1/1` | `3/2/1` | `3/1/1` |
| Fold levels, including terminal | 9 | 9 | 7 | 7 | 7 | 6 | 6 |

The retained samples were:

| Profile | Commit samples | Prove samples | Verify samples |
|---|---|---|---|
| A | 23.5282 s, 23.2922 s | 3.0298 s, 2.9149 s | 36.189 ms, 35.838 ms |
| A′ | 13.7643 s, 14.0052 s | 3.3601 s, 3.3094 s | 27.301 ms, 26.858 ms |
| B | 13.7956 s, 13.7841 s | 3.0777 s, 3.1044 s | 21.368 ms, 21.040 ms |
| E | 15.1230 s, 16.8683 s | 3.1357 s, 3.1144 s | 22.806 ms, 21.712 ms |
| F | 11.3390 s, 11.1421 s | 4.8520 s, 4.9355 s | 27.898 ms, 28.230 ms |
| C | 14.7149 s, 14.8497 s | 3.2271 s, 3.2359 s | 36.907 ms, 36.597 ms |
| D | 14.5899 s, 14.6444 s | 3.2066 s, 3.2750 s | 26.843 ms, 27.731 ms |

Relative to the earlier idle-machine table, prove changed by `+0.5%` to
`+5.1%` and verify by `-3.4%` to `+5.2%`. Commit was more sensitive to host
state: A/A′/B changed by only `+0.3%` to `+3.5%`, while the later E/F/C/D
runs were `+7.2%` to `+18.3%`. E's two commit samples alone differed by
11.5%, while its proof geometry and prove/verify times remained stable.
An independent E repeat after the matrix measured 14.282 s commit, 3.092 s
prove, and 22.649 ms verify, reducing its commit delta from `+18.3%` to
`+5.7%` without any structural change. Therefore the follow-up confirms exact
functional and sizing reproduction but does not supersede the earlier
idle-machine timing table.

Proof size, fold/tail decomposition, level count, setup footprint, and verifier
cache footprint matched across both retained runs of every profile. The
reported phase rows use the benchmark's `commit`, `prove`, and `verify OK`
measurements; setup and one-time backend preparation are not folded into those
three rows.

The corrected F result supersedes the invalid historical D512 measurement.
Its current root widths are `32,768/704,512/704,512`, its root output is
`247,552,000`, and its L1 B rank is 2. Relative to E, F saves 3.35 seconds
(24.8%) in commit, but adds 1.61 seconds (52.8%) in prove, 5.28 ms (24.2%) in
verify, 2,461 proof bytes, and approximately 10.1 GiB of peak RSS. E therefore
remains the balanced profile; F is only attractive when root commitment time
dominates those costs.

### Recursive mixed-D results (`nv = 32`, two multi-group workloads)

This experiment applies the requested recursive transition to the two PR #331
CI benchmark workloads:

| Workload | D64 control mode | Mixed-D mode |
|---|---|---|
| Recursive multi-group | `onehot_fp128_d64_multi_group_recursive` | `onehot_fp128_mixed_d_multi_group_recursive` |
| Recursive multi-group W8R2 | `onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2` | `onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2` |

Both workloads open two precommitted 16-variable singleton groups plus two
32-variable final-group polynomials. The profile argument is therefore
`nv=32`, `np=4`. Both mixed runs use:

```text
L0 final and precommitted groups: A/B/D = 256/128/128
L1:                              A/B/D = 128/64/64
L2 and later levels:             A/B/D = 64/64/64
setup prefix produced by L0:     source A = 128, commitment B = 64
```

W8R2 additionally preserves the eight-way witness partition at L1; its
configured chunk policy is eight chunks over the first two activated levels.
The plain mixed schedule has eight proof levels including the terminal level,
versus nine for its D64 control. Both W8R2 schedules have ten.

#### Measurement protocol

The measurements were taken on 2026-07-28 on the same Apple M4 Max MacBook Pro
(16 cores, 64 GB, macOS 26.5.2), from the working tree based on
`af770e1296`. All builds used Cargo `--release`, 16 Rayon prove threads, 16
Rayon verify threads, tracing disabled, one discarded warmup, and two
sequential retained runs.

The D64 controls were built with the exact CI profiler feature graph:

```text
--no-default-features --features parallel,profile-ci
```

The experimental mixed modes intentionally are not linked into
`profile-ci`, because they use the offline planner through `akita-pcs`
test-support. They were built from the same source with:

```text
--no-default-features --features parallel
```

This preserves the production rule that the CI/runtime profiler does not link
the offline planner. It does mean the comparison uses two feature graphs, not
one binary. The measured prover and verifier protocol implementations are the
same; only schedule availability and experimental test-support linkage differ.

#### Two-run means

Positive deltas mean the mixed profile is larger or slower. Memory units are
binary MiB/GiB.

| Metric | Plain D64 | Plain mixed | Mixed delta | W8R2 D64 | W8R2 mixed | Mixed delta |
|---|---:|---:|---:|---:|---:|---:|
| Setup expand | 1.4549 s | 0.1727 s | -88.1% | 3.3308 s | 0.1786 s | -94.6% |
| Backend prepare | 0.1717 s | 2.8820 s | +1578.2% | 0.4814 s | 2.9268 s | +507.9% |
| Setup total | 1.6266 s | 3.0547 s | +87.8% | 3.8122 s | 3.1054 s | -18.5% |
| Commit | 1.5994 s | 2.0683 s | +29.3% | 1.6094 s | 2.0631 s | +28.2% |
| Prove | 2.1767 s | 3.4511 s | +58.5% | 6.4145 s | 6.7657 s | +5.5% |
| Verify | 14.462 ms | 33.892 ms | +134.3% | 20.538 ms | 41.801 ms | +103.5% |
| Proof bytes | 97,826 | 100,039 | +2.3% | 107,757 | 100,353 | -6.9% |
| Fold / terminal bytes | 45,428 / 52,398 | 37,152 / 62,887 | -18.2% / +20.0% | 55,368 / 52,389 | 47,944 / 52,409 | -13.4% / +0.04% |
| Setup vector | 1,720 MiB | 1,024 MiB | -40.5% | 4,128 MiB | 1,032 MiB | -75.0% |
| Prover NTT cache | 4,300 MiB | 7,680 MiB | +78.6% | 10,320 MiB | 7,740 MiB | -25.0% |
| Verifier NTT cache | 1.375 MiB | 1.375 MiB | 0.0% | 1.375 MiB | 1.375 MiB | 0.0% |
| Peak RSS | 8.073 GiB | 11.949 GiB | +48.0% | 18.299 GiB | 12.161 GiB | -33.5% |
| Levels including terminal | 9 | 8 | -1 | 10 | 10 | 0 |

The retained timing samples were:

| Profile | Setup samples | Commit samples | Prove samples | Verify samples |
|---|---|---|---|---|
| Plain D64 | 1.6692 s, 1.5840 s | 1.5373 s, 1.6616 s | 2.1699 s, 2.1835 s | 15.193 ms, 13.731 ms |
| Plain mixed | 2.9664 s, 3.1430 s | 2.0696 s, 2.0670 s | 3.4364 s, 3.4657 s | 34.206 ms, 33.577 ms |
| W8R2 D64 | 3.7772 s, 3.8473 s | 1.5801 s, 1.6388 s | 6.3856 s, 6.4433 s | 20.389 ms, 20.686 ms |
| W8R2 mixed | 3.0623 s, 3.1485 s | 2.0711 s, 2.0551 s | 6.7031 s, 6.8283 s | 41.498 ms, 42.103 ms |

#### Interpretation

The requested mixed profile is not a recursive performance win in its current
form:

- verifier time is approximately 2.0–2.3 times the all-D64 control even though
  the verifier NTT cache size is unchanged;
- commit is about 28–29% slower in both workloads;
- the plain prover is 58.5% slower, while W8R2 amortizes most of that penalty
  and is 5.5% slower;
- the larger generation dimension sharply reduces the expanded setup vector;
  for W8R2 it also reduces NTT cache and peak RSS, but the plain case needs
  D256, D128, and D64 prepared slots and therefore uses more cache and RSS;
- W8R2 proof size improves by 7,404 bytes, while the plain proof grows by
  2,213 bytes.

The verifier slowdown is therefore a real schedule/geometry outcome, not an
NTT-cache-footprint increase on the verifier. Before planner integration, the
next profiling step should attribute the extra verifier time by fold and
stage, especially the mixed L0 setup-product path and the D128 setup-prefix
edge. The planner proposal should not treat this single requested tuple as an
assumed recursive optimum.

#### Correctness findings exposed by this experiment

Uniform D64 had hidden two setup-prefix assumptions. The recursive mixed-D
profile did not verify until both canonical paths were corrected:

1. `active_setup_field_len` omitted the additional B and D subcolumns induced
   by `d_a / d_b` and `d_a / d_d`. For a `256/128/128` root it planned exactly
   half of the required flat setup prefix.
2. `commit_setup_prefix` used the source A dimension for its B commitment.
   A D128 prefix with a D64 B matrix therefore serialized each B row at twice
   the expected width. Prefix extraction/inner commitment and outer commitment
   now dispatch independently, as the ordinary commitment path already does.

The profile setup path prepares exact NTT slots for both prefix dimensions.
Prover and verifier still reject a missing slot, mismatched dimension,
undersized prefix, or wrong commitment geometry; no validation was weakened.

### Verifier regression closure

Before `2f0c35b66`, PR CI reported the plain recursive verifier at approximately
28.6 ms versus a 25.7 ms main baseline. A matched local x86_64 comparison after
the uniform-path fixes measured:

```text
branch median: 24.11 ms
main median:   23.85 ms
delta:         +1.1%
```

This is local evidence, not a replacement for the next CI report. The two
2026-07-28 pulled commits change synthetic schedule construction, tests,
profile controls, and terminology; they do not modify the production verifier
kernels fixed at `2f0c35b66`.

## Reproduction

Use the default profile feature set. The mixed experimental modes are compiled
out by `profile-ci` and by the dedicated D64-only profile feature.

```bash
# A: uniform D64
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile

# A′ or B: D128 leading band, then D64
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64_root_d128 \
AKITA_MIXED_ROOT_D=128 \
AKITA_MIXED_SWITCH=1 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile

# Change AKITA_MIXED_SWITCH to 2 for B.

# E: uniform D128 root, 128/64/64 at L1, then D64
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64_root_d128 \
AKITA_RING_DIMENSION_TRANSITION=1 \
AKITA_TRANSITION_ROOT_D_RING_DIM=128 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile

# F: temporary D512 A-only root
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64_root_d128 \
AKITA_THREE_BAND_RING_DIMENSION_TRANSITION=1 \
AKITA_THREE_BAND_ROOT_A_RING_DIM=512 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile

# C: root 128/64/64, then uniform D128
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64_root_d128 \
AKITA_PER_MATRIX_RING_DIMS_ROOT=1 \
AKITA_ROOT_B_RING_DIM=64 \
AKITA_ROOT_D_RING_DIM=64 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile

# D: root 128/128/64, then uniform D128
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64_root_d128 \
AKITA_PER_MATRIX_RING_DIMS_ROOT=1 \
AKITA_ROOT_B_RING_DIM=128 \
AKITA_ROOT_D_RING_DIM=64 \
AKITA_PROFILE_TRACE=0 \
AKITA_PROFILE_LOG=info \
cargo run --release -p akita-pcs --example profile
```

Reproduce the recursive controls with the CI profiler build:

```bash
cargo build --release -p akita-pcs --example profile \
  --no-default-features --features parallel,profile-ci

python3 scripts/profile_bench_report.py run \
  --binary ./target/release/examples/profile \
  --output-dir /tmp/akita-recursive-d64 \
  --runs 2 --warmups 1 \
  --case onehot_fp128_d64_multi_group_recursive:32:4:recursive

python3 scripts/profile_bench_report.py run \
  --binary ./target/release/examples/profile \
  --output-dir /tmp/akita-recursive-w8r2-d64 \
  --runs 2 --warmups 1 \
  --case onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2:32:4:recursive
```

Then build and run the test-support mixed profiles:

```bash
cargo build --release -p akita-pcs --example profile \
  --no-default-features --features parallel

python3 scripts/profile_bench_report.py run \
  --binary ./target/release/examples/profile \
  --output-dir /tmp/akita-recursive-mixed \
  --runs 2 --warmups 1 \
  --case onehot_fp128_mixed_d_multi_group_recursive:32:4:recursive

python3 scripts/profile_bench_report.py run \
  --binary ./target/release/examples/profile \
  --output-dir /tmp/akita-recursive-w8r2-mixed \
  --runs 2 --warmups 1 \
  --case onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2:32:4:recursive
```

Extract the comparable phase metrics:

```bash
rg '\] (setup|commit|prove|verify OK|proof: total)' <log>
```

For a publishable table:

1. build once from the exact commit under test;
2. run every profile on the same idle machine;
3. run at least one warmup and two measured passes;
4. record raw samples, not only means;
5. record proof bytes, fold/tail split, levels, setup vector, prover NTT cache,
   and verifier NTT cache;
6. rerun A and B as controls even if their builders did not change;
7. do not reuse the invalidated pre-replanning F result.

## Recommended validation

Focused schedule and E2E checks:

```bash
cargo test -p akita-pcs --test ring_dimension_transition_schedule
cargo test -p akita-pcs --test per_matrix_ring_dims_root_e2e
cargo test -p akita-pcs --test ring_dimension_transition_e2e
cargo test -p akita-pcs --test mixed_d_per_level_e2e
cargo test -p akita-pcs --test recursive_ring_dimension_transition_e2e
cargo test -p akita-planner --lib --features catalog-gen
```

Affected lower-level suites:

```bash
cargo test -p akita-types --lib
cargo test -p akita-verifier --lib
cargo test -p akita-prover --lib
```

Documentation changes must pass:

```bash
./scripts/check-doc-guardrails.sh
```

Use the exact CI Clippy commands from `AGENTS.md` before handing the branch
back.

## Implementation map

| Concern | Canonical location |
|---|---|
| Planner public entry points and root search | `crates/akita-planner/src/schedule_params.rs` |
| Planner candidate construction | `crates/akita-planner/src/schedule_params/candidate.rs` |
| Planner suffix dynamic programming | `crates/akita-planner/src/schedule_params/suffix_dp.rs` |
| Multi-group root planning | `crates/akita-planner/src/planner.rs` |
| Generated catalog emission | `crates/akita-planner/src/emit/mod.rs` |
| Runtime policy and selection IDs | `crates/akita-schedules/src/runtime.rs` |
| Generated-row expansion and replay | `crates/akita-schedules/src/generated/` |
| Catalog identity | `crates/akita-schedules/src/catalog_identity.rs` |
| Config-to-policy projection | `crates/akita-config/src/lib.rs` |
| Proof-byte and setup-envelope accounting | `crates/akita-types/src/proof/` |
| Mixed schedule builders and adapters | `crates/akita-pcs/src/test_support.rs` |
| Profile selection and environment variables | `crates/akita-pcs/examples/profile/modes.rs` |
| A/B/D dimension validation | `crates/akita-types/src/layout/ring_dims.rs` |
| Relation-address geometry | `crates/akita-types/src/proof/relation_address.rs` |
| Relation RHS and row dimensions | `crates/akita-types/src/proof/relation.rs` |
| Setup-contribution plan | `crates/akita-types/src/setup_contribution/plan/` |
| Recursive setup-index evaluator | `crates/akita-types/src/setup_contribution/plan/setup_index_weight.rs` |
| Prover ring-switch layout | `crates/akita-prover/src/protocol/ring_switch/` |
| Verifier relation evaluation | `crates/akita-verifier/src/protocol/ring_switch.rs` and `ring_switch/` |
| Prover Stage 3 | `crates/akita-prover/src/protocol/sumcheck/akita_stage3/` |
| Verifier Stage 3 | `crates/akita-verifier/src/stages/stage3.rs` |
| Active acceptance tests | `crates/akita-pcs/tests/{mixed_d_per_level_e2e,per_matrix_ring_dims_root_e2e,ring_dimension_transition_e2e,ring_dimension_transition_schedule}.rs` |

## Known limits and next work

### P0: broaden cataloged mixed planning

The direct scalar fp128 one-hot family completes the bounded Cut 2 path. The
next planner work is multi-group and recursive-setup admission under separately
specified objectives. Remove the D512 D256-promotion heuristic only after the
native planner reproduces or improves its geometry.

### P1: production heterogeneous-group admission

`PlannerPolicy` still exposes one scalar `ring_dimension`. The protocol can
consume group-local dimensions, but production planning and shipped catalogs
cannot emit a heterogeneous-group root. Add explicit final/precommitted
`CommitmentRingDims` to the planner boundary and generate an end-to-end catalog
row.

### P1: dynamic setup-prefix dimension

Setup-prefix offload still uses the D64 registry contract for catalog
recursive families. The synthetic recursive mixed-D profile materializes a
dynamic D128 prefix outside that registry and is covered by
`recursive_ring_dimension_transition_e2e.rs`. A production mixed batch whose
common relation dimension is below 64 remains rejected until setup generation,
registry lookup, planner admission, and verifier dispatch select `d_setup`
consistently.

### P2: expand the sweep

Using the current complete matrix as the baseline:

- compare `switch ∈ {1, 2, 3}`;
- compare D128 and D256 leading bands;
- sweep larger `nv`;
- add `np > 1` only where the synthetic builder admits it;
- profile direct, recursive, and multi-chunk verification separately.

## Agent handoff checklist

Before changing this area:

1. Read this spec and `specs/runtime-ring-cutover.md`.
2. Confirm the branch and diff base; do not rely on the untracked historical
   handoff file.
3. Identify whether the change affects:
   - schedule construction only;
   - proof/transcript geometry;
   - direct verification;
   - recursive/setup-offloaded verification;
   - multi-group ownership.
4. Use `RelationAddressGeometry`, `group_role_dims`, and
   `SetupContributionPlan`; do not create parallel geometry helpers.
5. Preserve the one-lane inner kernels inside the canonical projected plan
   unless identical-geometry benchmark evidence justifies a change.
6. Run the focused schedule/E2E tests before expensive repository-wide gates.
7. If benchmark geometry changes, invalidate old numbers explicitly before
   adding new ones.
