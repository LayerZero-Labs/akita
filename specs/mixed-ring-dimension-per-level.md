# Mixed Ring Dimensions Across Fold Levels and Commitment Matrices

| Field | Value |
|---|---|
| Status | Adaptive direct, multi-chunk, grouped recursive-setup planning, catalog generation, and runtime replay implemented |
| Review snapshot | 2026-07-28, planner reviewed on `main` at `af770e129` |
| Benchmark snapshot | 2026-07-28, release build of `25a1e94a6` |
| Recursive benchmark snapshot | 2026-07-28, working tree based on `af770e1296` |
| Primary workload | fp128 one-hot, `nv = 36`, `np = 1` |
| Primary profile mode | `onehot_fp128` |
| Related spec | `specs/runtime-ring-cutover.md` |
| Projected digit layout | `specs/role-native-projected-digit-layout.md` |
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

> **Scoped revision (stacked follow-up to PR #334):** Mixed schedule and planner
> findings remain in force, but setup no longer has a generation/carrier
> dimension. Flat public matrix derivation, exact field capacity, setup-prefix
> padding, and per-domain NTT cache sizing are specified by
> [`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md).

Akita's protocol and direct scalar planner support both forms of mixed ring
dimension needed by this experiment:

- **Across levels:** a large-ring leading band can hand off to a smaller-ring
  recursive suffix.
- **Within a level:** the A, B, and D commitment matrices can use distinct
  dimensions `d_a/d_b/d_d`, subject to A-to-role projection divisibility.

The protocol, setup-contribution, quotient, and verifier paths consume this
geometry. For the default direct scalar fp128 one-hot and dense families,
including direct multi-chunk variants, the offline planner searches the
admitted A dimensions and block splits. It currently derives B and D by minimum
secure rank before complete schedule comparison. Full Cartesian B/D
optimization for these direct families is deferred.

Grouped recursive-setup requests use a distinct adaptive path. Their suffix DP
searches explicit per-matrix dimension tuples and jointly chooses role
dimensions, block geometry, commitment payload mode, and whether each supported
edge evaluates setup directly or offloads it through a carried setup-prefix
opening. Direct grouped roots without recursive setup planning preserve their
committed profiles and use the uniform D64 suffix policy. The fp128 direct and
recursive families emit and replay generated catalogs; prover and verifier do
not invoke the planner at runtime.

For recursive schedules, adaptive search is deliberately scoped to grouped
requests with precommitted inputs: those inputs provide setup contributions
that can actually be offloaded. A scalar request under
`RecursiveCommitmentConfig<OneHot>` is rejected by the scalar mixed-search
entry point. This is a request-shape restriction, not a lack of adaptive
recursive support.

The production recursive catalogs currently cover the profiling key with a
32-variable, two-polynomial final group and two 16-variable singleton
precommitted groups. The selected schedules are:

```text
single chunk:
  root  256/128/128
  L1    256/128/128  (consumes setup prefix committed at 256/128)
  L2+    64/64/64

W8R2:
  root  256/128/64   (8 witness chunks)
  L1    256/128/64   (8 chunks; consumes prefix at 256/128)
  L2     64/64/64    (single chunk; consumes prefix at 64/64)
  L3+    64/64/64
```

The frozen precommit descriptors remain part of the root lookup key. The
single-chunk precommits use A/B `256/64`; W8R2 precommits use `64/64`. At every
offloaded edge the setup-prefix commitment inherits the consuming fold's exact
A/B dimensions; it does not use a global fixed prefix dimension.

### Adaptive recursion implementation

Supporting recursion required closing six planner/runtime gaps; registering an
adaptive base config on the old recursive planner would still have produced a
uniform-D64 suffix.

1. **Do not take the grouped adaptive fallback.** Direct grouped requests still
   preserve the established uniform-suffix fallback because their committed
   input descriptors are frozen. A grouped request with
   `recursive_setup_planning = true` now enters the setup-aware suffix DP, where
   the root and recursive folds can search exact role tuples.
2. **Enumerate exact tuples under a per-role ceiling.** For each searched level,
   the DP enumerates the Cartesian product of the configured A/B/D domains,
   rejects non-divisor projections, and enforces
   `next.d_role <= current.d_role` independently for A, B, and D. The selected
   tuple becomes the child state's ceiling. At `num_search_levels`, enumeration
   stops and the configured uniform suffix dimension is used.
3. **Price each tuple with its own geometry.** Root group expansion receives a
   fixed exact A/B/D tuple, including the shared opening D used by frozen
   precommits. Recursive candidates derive their fold challenge, extension
   opening reduction bytes, block splits, matrices, ranks, and setup footprint
   at that tuple. During adaptive levels all feasible block splits are retained
   for comparison; after the search window the ordinary local split selection
   is sufficient.
4. **Make setup-prefix dimensions edge-local.** A prefix is produced for a
   successor but committed as an input of that successor. Prefix derivation
   therefore receives the consuming candidate's exact A and B dimensions.
   Prefix slot metadata, natural/padded length, SIS rows, and challenge config
   all agree with those dimensions. This removes the old assumption that every
   recursive prefix uses `Cfg::D`.
5. **Preserve enough information in the suffix frontier.** Two successor
   schedules with the same ordinary outer-commitment payload can have different
   Stage-3 setup-product payloads when their D/prefix geometry differs. The
   parent-visible frontier key now contains both byte counts. Pruning only on
   outer payload would discard a candidate that can be better after an
   offloaded parent edge.
6. **Ship distinct catalog identities and runtime routes.** The generated
   families `fp128_onehot_recursive` and
   `fp128_onehot_recursive_multi_chunk_w8r2` bind the adaptive domains,
   recursive selection policy, exact lookup keys, and selected schedules.
   `RecursiveCommitmentConfig<OneHot>` and
   `RecursiveCommitmentConfig<OneHotMultiChunk>` route to those tables. The
   profiler and E2Es use these configs, so runtime proving never depends on the
   offline search implementation.

The recursive objective remains
`MinFirstDirectSetupThenPayload`: it first minimizes the setup footprint at the
first direct edge after any offloaded prefix, then exact proof payload and the
remaining deterministic tie-breaks. Dimension is not optimized in isolation;
larger A/B/D candidates survive only when their effect on ranks, witness
contraction, proof bytes, and the setup envelope wins under that objective.

The implemented direct-scalar policy is:

```text
1. smallest physical setup matrix, measured in base-field elements;
2. smallest exact modeled proof payload;
3. deterministic canonical tie-break only.
```

This policy is catalog-bound for the default direct scalar fp128 one-hot and
dense families. Their generated identities use
`MinSetupMatrixFieldElementsThenProofPayload`. A caller that supplies an
adaptive dimension policy with a proof-payload selection policy is rejected
rather than silently changing objectives. Prover and verifier remain
catalog-only and never run the planner.

The current implementation applies this objective after choosing B and D by
minimum secure rank. It is therefore not a global optimum over the full A/B/D
Cartesian product. The complete Cartesian search and matching oracle are P0
follow-up work below.

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

The public setup matrix is a dimension-free stream of base-field elements.
Setup capacity is the largest physical matrix requirement in the schedule.
There is no setup generation dimension and no global divisibility rule.

Each scheduled A, B, and D dimension must have support for its protocol role
and its SIS security table. A dimensions must also have production fold
challenge support. The schedule separately enforces A-to-role projection and
per-level transition rules.

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

## Planner integration and scope

### Decision summary

The planner should search `d_a`, `d_b`, and `d_d` together with the existing
digit-basis and block-split choices. It must derive every width and SIS rank
from the selected dimensions, plan the complete continuation from the exact
outgoing witness, and retain enough alternatives to optimize the non-additive
setup objective correctly.

The target search policy limits dimension choice to L0 and L1. Dimensions are
component-wise non-increasing, and L2 and later are uniform D64. The target
does not use rank-one dimension pruning because B and D widths depend on
upstream ranks. A geometry-only bucket is not an equivalence class. The target
correctness baseline is exhaustive L0/L1 tuple enumeration with Pareto frontier
retention and descriptor-byte tie-breaking. The current implementation reaches
this baseline for A choices and block splits, but not for B and D choices.

The direct-schedule score is:

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
| `uniform_ring_dimension` | Uniform-only A/B/D candidate; ignored by adaptive search |
| `ring_dimension_schedule_mode` | Uniform candidate or catalog-bound adaptive A/B/D domains plus the uniform suffix |
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
| `MinSetupMatrixFieldElementsThenProofPayload` | Adaptive direct schedules: physical setup fields, then exact proof payload, then descriptor bytes |
| `MinFirstDirectSetupThenPayload` | Recursive setup schedules: first later direct setup scan, then exact proof payload, subject to an optional host budget |

The ordinary uniform direct policy computes a setup envelope but does not use
it for selection. The canonical fp128 one-hot and dense presets each resolve
one adaptive generated family; there is no runtime cross-family selector.

### Current search algorithm

The shared suffix planner now does the following:

1. At the root, enumerate the configured `log_basis`, valid block splits, and
   every exact A/B/D tuple admitted at level 0.
2. Derive tuple-local widths, coefficient bounds, secure ranks, extension
   opening bytes, and the exact outgoing witness length. Frozen precommit D
   segments are projected once to the selected shared D and added after the
   main group's A-to-D projection.
3. Enter the memoized suffix search with the exact witness boundary and the
   selected tuple as the componentwise dimension ceiling.
4. At each recursive state, enumerate non-decreasing bases and every exact
   tuple below that ceiling. Adaptive searched levels retain every feasible
   block split; uniform suffix levels use `layout_candidate_score` to select the
   local split per basis.
5. Compare direct termination with another fold and, for recursive-setup
   policies, compare direct and setup-offloaded child edges.
6. Materialize the selected typed `FoldSchedule`; recompute proof bytes and
   setup envelope and reject any disagreement with the cached estimates.

The suffix memo distinguishes the dimension ceiling and retains both
first-direct-setup and payload objectives. Its parent-visible frontier key
contains the successor's ordinary outer payload and Stage-3 payload, so an
adaptive D/prefix choice cannot be incorrectly pruned merely because its B
payload ties another successor.

### Historical pre-cutover mixed-D gap

The following list records the implementation gaps that motivated this cut.
They are historical: the adaptive scalar, multi-chunk, and grouped recursive
paths now address them. Direct grouped requests still intentionally use the
uniform-suffix fallback rather than searching a heterogeneous root.

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

The removed synthetic builders demonstrated protocol feasibility but were not
suitable planner architecture. They planned a uniform schedule, mutated or
rebuilt selected matrices, and recomputed boundaries outside the native
planner. A native planner must produce the final matrices directly; it must not
generate a uniform candidate and retarget it after selection.

### Target ring-dimension policy

The planner uses a plain-value search policy conceptually equivalent to:

```rust
PlannerRingDimensionPolicy {
    a_candidates,
    b_candidates,
    d_candidates,
    uniform_suffix_dimension,
}
```

The exact Rust shape is an implementation detail, but the semantics are
normative:

- The public setup matrix has no ring dimension. Candidate admission does not
  use a setup carrier or require divisibility against one.
- `a_candidates` are dimensions with production fold-challenge support and an
  audited A-role SIS cell.
- `b_candidates` and `d_candidates` are independently audited role domains.
- `uniform_suffix_dimension` is the dimension used after the adaptive search
  prefix. It must be admitted for A, B, and D.
- Candidate lists are sorted, unique, non-empty, and catalog-identity-bound.
- The planner enumerates the Cartesian product and keeps only tuples satisfying
  the canonical A-to-role divisibility validation.
- B and D remain independent; the planner must not impose `d_b == d_d` or an
  ordering between them.
- Terminal candidates use the A domain because a terminal has only the inner
  commitment matrix.

The current bounded fp128 implementation is a partial implementation of this
target. It enumerates A but derives B and D locally by minimum secure rank. The
P0 follow-up must enumerate the full role-valid Cartesian product and apply the
catalog objective only after each complete schedule has been priced.

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

The flat setup cutover now uses `max_setup_matrix_field_elements` directly.
There is no current `SetupMatrixEnvelope::max_setup_len` conversion and no
setup-generation dimension. The planner score, setup-capacity check,
generated-row replay, and flat matrix allocation all consume the same physical
field-element quantity. The old ring-element conversion is retained only in
the historical snapshot below.

An optional host setup budget uses the same field-element unit through
`setup_field_budget`. The shipped policy is uncapped. The historical
`max_setup_envelope_field_elements` and `max_num_setup_field_elements` names
are removed.

### Per-level candidate derivation

For every explicit tuple `(d_a, d_b, d_d)`, basis, and block split, derive the
candidate in this order:

1. Validate role projection within the tuple, input-witness alignment,
   role-specific dispatch/SIS coverage, challenge support at `d_a`, and
   level/chunk constraints.
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

The catalog-bound selection policy has semantics:

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

The planner must replace current uses of `policy.uniform_ring_dimension` in
`d_segment_width` and carrier sizing with the selected shared D and the
canonical maximum group carrier. Group order must not influence the result.
Outgoing sizing must call the compact `WitnessLayout` and successor-domain
geometry rather than recomputing ring slots. Authenticated order fixes bytes;
changing stable group identifiers without changing that order must not change
the length.

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

Catalog identity for the implemented direct scalar family binds:

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

### Historical setup/config cutover (superseded)

The original mixed-D proposal treated `CommitmentConfig::D` and a scalar
`PlannerPolicy::ring_dimension` as a setup-generation dimension, a uniform
planner candidate, and a backend policy at once. That model was removed by
the flat setup cutover in
[`flat-public-matrix-and-exact-ntt-cache.md`](flat-public-matrix-and-exact-ntt-cache.md).

Current code keeps `CommitmentConfig::D` only as the uniform candidate for
presets that choose one, exposes `PlannerPolicy::uniform_ring_dimension`, and
measures setup capacity directly in base-field elements. The public matrix has
no generation dimension, and cache/catalog identity does not include one.
The historical design below is retained only to explain why the separation
was required; it is not an active implementation contract.

### Determinism and tie-breaking

The planner must be independent of hash-map iteration and thread scheduling:

- require each catalog-bound A/B/D candidate domain to be strictly sorted,
  duplicate-free, and non-empty;
- reject role dimensions without protocol-dispatch, challenge, or SIS-table
  coverage; candidate pairing separately enforces A-source-to-role projection
  divisibility;
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
3. Remove setup-generation dimension from candidate admission and setup
   accounting.
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

#### Cut 2: catalog replay and shipped adaptive families (implemented for fp128 one-hot and dense)

1. Make the canonical generated-entry walker replay per-matrix/per-level
   dimensions.
2. Add DP-to-generated-to-runtime exact parity tests.
3. Add adaptive fp128 one-hot and dense families rather than another set of
   fixed-D selector wrappers.
4. Regenerate tables and update setup capacity/cache identity.

#### Historical Cut 3 plan: broader topology

1. Enable direct multi-chunk search.
2. Enable multi-group final-root dimension search with frozen precommits.
3. Design and implement mixed-D recursive setup offload under a separately
   approved objective.
4. Remove synthetic profile adapters only after planner-selected schedules
   reproduce their coverage and benchmarks.

### Historical direct-scalar checkpoint and planner example

This checkpoint described the first offline direct scalar cut. It is retained
as rollout history and is superseded by the adaptive recursion implementation
above:

- `RingDimensionScheduleMode::AdaptiveDimension` is the catalog-bound source
  of independently audited A/B/D domains and the D64 suffix;
- the one canonical `find_schedule` entry point dispatches by schedule mode:
  uniform mode preserves the proof-payload objective, while adaptive mode
  selects by physical setup field elements and then exact modeled proof bytes;
- root and recursive candidates derive role-local widths, SIS keys, and
  matrices directly after choosing A and deriving B and D by minimum secure
  rank;
- L0 and L1 exhaustively enumerate block splits and admissible,
  component-wise descending A choices;
- dimensions are uniform D64 from L2 through the terminal;
- a test-only unpruned traversal checks the production frontier and canonical
  selection over an A-varying domain with B and D fixed at D64. It deliberately
  shares canonical candidate construction and pricing primitives. It does not
  establish global optimality over the full B/D domain;
- hand-calculated regressions independently pin exact field-element setup
  rounding, candidate-local EOR pricing, unsupported SIS-cell skipping, and
  complete-schedule descriptor ties;
- the dedicated mixed-search memo includes the parent dimension ceiling, while
  L2+ states canonicalize to D64 and reuse the fixed planner's split policy;
- mixed-boundary states retain the required setup/proof alternatives per exact
  parent-visible first fold;
- scalar recursive requests without precommitted inputs are rejected; grouped
  recursive requests use the adaptive setup-aware suffix planner.

`crates/akita-planner/examples/mixed_dimension_search.rs` exercises both the
implemented and preserved paths. With `nv=18` and the following candidate
tuples:

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

The `nv=24` and `nv=36` rows supersede the earlier A-pruned checkpoint.
Exhaustive L0/L1 A enumeration admits the D256 root and selects it because
setup fields are the primary objective, even though the `nv=36` proof is
larger than the former pruned result.

These are planner smoke-test wall times, not a controlled benchmark; the
process and filesystem caches were warm after the first run. The material
result is that `nv=24` and `nv=36` now complete normally. Before the bounded
policy, `nv=24` exceeded one minute and `nv=36` was stopped after five minutes.

The speedup comes from stopping mixed dimensions and exhaustive block-split
enumeration after L1. Monotonicity removes upward transitions, and the complete
D64 suffix reuses the existing fixed planner split derivation. The current
local B/D rank choice also reduces the search space, but it is not equivalent
to the target Cartesian objective. The P0 follow-up must remove it.

For the PR recursive multi-group shape, the new entry point returns the
expected unsupported-policy error because mixed recursive setup is a later
cut. The preserved grouped D64 planner still produced a valid nine-level
schedule with one setup-offload edge, a 524,288-ring-element D64 setup
envelope, and a 102,732-byte modeled proof. This confirms behavior
preservation, not planner-native recursive mixed-D support.

### Acceptance criteria

The list below is the complete heterogeneous-planner target. This PR implements
the catalog and runtime path and the bounded scalar fp128 protocol path. It does
not yet satisfy the full planner optimality requirements in items 2 and 4. The
checked unpruned traversal varies A while holding B and D at D64, and production
chooses B and D by minimum secure rank before the complete schedule comparison.
Full Cartesian B/D search is P0 follow-up work. Mixed multi-group replay,
direct multi-chunk mixed search, and fp32/fp64 mixed catalogs also remain
deferred. This PR does not describe these items as shipped capabilities.

1. Existing `find_schedule` reproduces uniform schedules, estimates, and
   generated/runtime descriptor bytes. An adaptive policy must admit D64 for
   every role and uses uniform D64 from L2 onward.
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
   `ensure_prover_schedule_fits_setup` agree on physical setup field elements.
7. DP output and generated-row expansion produce identical schedule descriptor
   bytes, proof-byte estimates, level counts, witness transitions, and setup
   cost.
8. Catalog validation rejects changes to candidate domains, selection policy,
   SIS digest, or challenge hooks.
9. Runtime row misses reject without planner fallback or panic.
10. Existing uniform direct, recursive, multi-chunk, and multi-group benchmark
    paths retain their fast verifier kernels.
11. The scalar fp128 mixed-D E2Es cover honest verification, wrong openings,
    proof/commitment tampering, malformed dimensions, unsupported SIS cells,
    and setup under-capacity. Mixed multi-group and mixed multi-chunk E2Es are
    a later acceptance gate after their planner/catalog paths are implemented.
12. The `nv=36` constrained search completes, obeys all transition/rank caps,
    and the complete A/A′/B/E/F/C/D benchmark matrix is rerun from one build.

### Required planner tests

| Test | Required assertion |
|---|---|
| Candidate-domain validation | Sorted unique role domains; uniform D64 suffix present; each advertised dimension has role-specific dispatch, challenge (A), and SIS-table coverage |
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

These tests describe the implemented direct scalar cut. A future mixed
multi-group or mixed multi-chunk cut must add corresponding frozen-precommit,
chunk-partition, and generated-replay rows before claiming those capabilities.

### Target search policy

The approved first planner cut uses the deterministic
`(physical setup fields, proof bytes, descriptor bytes)` comparator subject to
two catalog-bound constraints:

1. only L0 and L1 search mixed dimensions;
2. dimensions never increase and L2+ is uniform D64.

A rank-one pruning is absent from this cut. B and D still use a local minimum
rank choice, which is not equivalent to the complete schedule objective. The
P0 Cartesian follow-up must remove that choice. Any later pruning requires an
equivalence proof checked against a full-domain traversal.

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

## Historical deterministic schedule snapshot

This section was regenerated from the pre-flat schedule builders at
`25a1e94a6` for `nv = 36`, `np = 1`. It contains historical geometry, not
wall-clock measurements or a current setup-allocation contract.

The historical `setup length` values below are
`SetupMatrixEnvelope::max_setup_len` in generation-ring elements. The `@ D`
annotations and conversion to bytes are obsolete under the flat setup model;
current capacity is reported directly as base-field elements by the flat setup
spec.

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

The synthetic PCS mixed-D builders described below were test-only artifacts.
They have been removed from `akita-pcs`; production callers must exercise the
planner and PCS through normal schedule resolution rather than depending on
crate-local fixtures.

### Removed synthetic PCS fixtures

The former test-local PCS builders covered uniform leading bands, per-matrix
root overrides, and multi-band transitions. Those fixtures were intentionally
removed so no crate depends on planner test artifacts. Mixed-D behavior should
now be exercised either in planner tests or through production schedule
resolution.

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

- Replaced mutation of already-planned suffixes with staged suffix replanning.
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

The former `akita-pcs` synthetic mixed-D integration tests have been deleted.
Verifier malformed-proof coverage remains in `akita-verifier`; planner tests
continue to cover mixed-D search and validation at the planning boundary.

### Lower-level coverage

- planner tests compare the mixed Pareto frontier with an A-varying unpruned
  traversal whose B and D dimensions remain D64. They also show that a lower
  D256 A rank can reduce B width after both B matrices reach rank one, preserve
  a lower-proof child when a larger parent setup masks child setup differences,
  and require descriptor-identical concurrent generation.
- `akita-types` setup-contribution span tests compare dense, materialized,
  direct, deferred, single-chunk, multi-chunk, and mixed-role projections.
- verifier ring-switch tests check prepared relation geometry and deferred
  mixed setup claims.
- multi-group parameter tests check group-local dimensions, row offsets,
  compact-length independence from stable group identifiers, and descending
  group A dimensions.
- recursive and distributed setup-offload E2Es cover adaptive generated-table
  replay under their production feature guards.
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

### Archived Recursive Mixed-D Results (`nv = 32`, two multi-group workloads)

This experiment applies the requested recursive transition to the two PR #331
CI benchmark workloads:

| Workload | D64 control mode | Mixed-D mode |
|---|---|---|
| Recursive multi-group | `onehot_fp128_d64_multi_group_recursive` | `onehot_fp128_mixed_d_multi_group_recursive` |
| Recursive multi-group W8R2 | `onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2` | `onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2` |

Both workloads opened two precommitted 16-variable singleton groups plus two
32-variable final-group polynomials. The profile argument was therefore
`nv=32`, `np=4`. Both mixed runs used:

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

At measurement time, the experimental mixed modes were intentionally not linked
into `profile-ci`, because they used offline-planner schedule fixtures through
the former `akita-pcs` test-support path. They were built from the same source
with:

```text
--no-default-features --features parallel
```

These PCS profile modes have since been removed because recursive mixed-D
verification is not a production-supported verifier path. Keep recursive
mixed-D experiments in tests or planner-only tooling until that verifier path
is promoted.

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
- the mixed schedule sharply reduces the expanded setup vector; for W8R2 it
  also reduces NTT cache and peak RSS, but the plain case needs D256, D128, and
  D64 prepared slots and therefore uses more cache and RSS;
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

Use the default profile feature set for production-supported profile modes.
Synthetic mixed-D PCS profile modes that depended on `akita-pcs` test fixtures
have been removed from the profile example.

```bash
AKITA_NUM_VARS=36 \
AKITA_MODE=onehot_fp128_d64 \
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
  --output-dir /tmp/akita-recursive-adaptive \
  --runs 2 --warmups 1 \
  --case onehot_fp128_multi_group_recursive:32:4:recursive

python3 scripts/profile_bench_report.py run \
  --binary ./target/release/examples/profile \
  --output-dir /tmp/akita-recursive-w8r2-adaptive \
  --runs 2 --warmups 1 \
  --case onehot_fp128_multi_group_recursive_multi_chunk_w8r2:32:4:recursive
```

The archived recursive mixed-D runs are no longer reproducible through
`akita-pcs --example profile`. Extract comparable phase metrics from archived
logs with:

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
| Planner public entry point and multi-group root search | `crates/akita-planner/src/planner.rs` |
| Scalar and mixed candidate construction | `crates/akita-planner/src/schedule_params/candidate/` |
| Planner suffix dynamic programming | `crates/akita-planner/src/schedule_params/suffix_dp.rs` |
| Multi-group root planning | `crates/akita-planner/src/planner.rs` |
| Generated catalog emission | `crates/akita-planner/src/emit/mod.rs` |
| Runtime policy and selection IDs | `crates/akita-schedules/src/runtime.rs` |
| Generated-row expansion and replay | `crates/akita-schedules/src/generated/` |
| Catalog identity | `crates/akita-schedules/src/catalog_identity.rs` |
| Config-to-policy projection | `crates/akita-config/src/lib.rs` |
| Proof-byte and setup-envelope accounting | `crates/akita-types/src/proof_size.rs`, `crates/akita-types/src/proof/setup_envelope.rs` |
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

## Known limits and next work

### P0: broaden cataloged mixed planning

The direct scalar fp128 one-hot and dense families complete the catalog and
runtime part of the bounded Cut 2 path. Their planner does not yet produce the
global optimum over the full A/B/D Cartesian product. For example, at fp128
one-hot `nv=14`, the current search selects `256/64/64` with 73,764 proof bytes.
An explicit tuple traversal selects `256/128/128` with the same 65,536 setup
field elements and 73,652 proof bytes.

The first follow-up must enumerate every role-valid B/D choice at L0 and L1,
remove the local rank-based dimension choice, and compare the selected score
and descriptor against a full-domain traversal. Later planner work includes
multi-group and recursive setup admission under separately specified
objectives. Remove the D512 D256-promotion heuristic only after the native
planner reproduces or improves its geometry.

### P1: production heterogeneous-group admission

`PlannerPolicy` still exposes one scalar `uniform_ring_dimension`. The protocol can
consume group-local dimensions, but production planning and shipped catalogs
cannot emit a heterogeneous-group root. Add explicit final/precommitted
`CommitmentRingDims` to the planner boundary and generate an end-to-end catalog
row.

### P1: dynamic setup-prefix dimension

Setup-prefix offload still uses the D64 registry contract for catalog
recursive families. A production mixed batch whose common relation dimension is
below 64 remains rejected until setup-prefix materialization, registry lookup,
planner admission, and verifier dispatch select `d_setup` consistently.

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
