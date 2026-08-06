# Declarative Ring-Dimension Schedule Policy


| Field                  | Value                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Status                 | Implemented                                                                                |
| Revised                | 2026-08-05                                                                                  |
| Scope                  | Offline ring-dimension search, generated-catalog identity, and runtime replay               |
| Related spec           | `[mixed-ring-dimension-per-level.md](mixed-ring-dimension-per-level.md)`                    |
| Planner implementation | `[crates/akita-planner/src/schedule_params/](../crates/akita-planner/src/schedule_params/)` |
| Configuration bridge   | `[crates/akita-config/src/lib.rs](../crates/akita-config/src/lib.rs)`                       |


## Summary

Akita represents ring-dimension behavior with one declarative enum:

```text
RingDimensionScheduleMode =
    UniformDimension
    | AdaptiveDimension
```

`UniformDimension` uses one uniform ring dimension for A, B, and D at every
fold and at the terminal.

`AdaptiveDimension` performs a bounded brute-force search over potential A-ring
dimensions at a configured number of leading fold levels. A dimensions must
be non-increasing between adjacent searched levels. Each complete A-dimension
path first receives its best preprocessing/proof geometry. The planner then
selects the path lexicographically by `(minimum secure A rank, A dimension)` at
each searched level: a strict rank reduction justifies increasing dimension,
rank one stops further growth, and an equal rank keeps the smaller dimension.
If no dimension reaches rank one, the smallest rank found remains valid. From
the end of the configured search window onward, every role uses one fixed
uniform suffix dimension.

B and D dimensions are not branching planner choices. Once an A dimension,
block geometry, basis, and role-native width are known, the planner scans the
ordered supported dimensions for that role from smallest to largest. It keeps
trying larger dimensions while deriving the exact secure rank, stops early if
rank one is reached, and otherwise selects the dimension with the smallest rank
found. A B- or D-role rank greater than one is valid when the allowed role
dimensions cannot reduce it further. Equal-rank choices prefer the smaller
dimension.

The planner evaluates all valid A-dimension paths within the bounded search
window, derives B/D and all matrix ranks canonically, and plans the complete
uniform suffix. It first selects the best complete geometry for each exact A
path by physical preprocessing fields, modeled proof bytes, and descriptor
bytes. It then compares those path representatives by the per-level A
rank/dimension rule above.

With two searched levels, potential A dimensions including D64, D128, and D256,
and a uniform-D64 suffix, the search includes:

```text
256/128/128 -> 64/64/64 -> uniform D64 suffix

256/128/128 -> 128/64/64 -> uniform D64 suffix

256/128/128 -> 256/64/64 -> uniform D64 suffix
```

The first path is the D256 direct transition, the second is the D256
three-band transition, and the third retains D256 for A at L1. None needs to
be passed as an explicit topology.

The policy is implemented in the runtime policy, offline planner, catalog
identity, generated-table emitter, and the fp128 one-hot adaptive preset.

## Implementation status

The implementation has the following shape:

- `CommitmentConfig::RING_DIMENSION_SCHEDULE_MODE` is the single configuration
  input. The former tuple-valued config field and separate public search-domain
  object are removed.
- `find_schedule` dispatches from the enum. Uniform policies reuse the existing
  uniform suffix DP. Adaptive policies run the bounded leading-level frontier
  and then rejoin the uniform suffix derivation.
- Root and recursive candidate construction share one
  `RingDimensionCandidate` implementation. It fixes A for a search branch and
  derives B/D from the canonical projected width and SIS table.
- Complete schedules are grouped by their exact A-dimension path. Setup/proof
  scoring chooses one representative per path before representatives are
  compared by per-level A rank and dimension.
- Generated catalog identity stores the complete enum value and all three
  ordered adaptive capability lists. Challenge-hook identity covers every A
  candidate, including non-winners.
- `OneHot` uses two searched levels, a D64 uniform suffix, A
  candidates `[64, 128, 256]`, and B/D candidates `[64, 128]`.
- Adaptive direct multi-chunk and recursive-setup planning remain rejected as
  specified. Scalar dispatch through a recursive adapter first applies its
  direct-only policy and is therefore supported.

The adaptive orchestration remains in `mixed_search.rs`, but it is no longer a
second public configuration/search API and contains no preset-name dispatch.
Its matrix construction, SIS pricing, setup accounting, proof pricing, and
materialization all call the same canonical primitives as uniform planning.
An unpruned adaptive traversal remains test-only and checks that frontier
pruning selects the same descriptor and estimate.

## Decisions

This revision makes the following decisions:

1. Use `RingDimensionScheduleMode` with exactly two modes:
  `UniformDimension` and `AdaptiveDimension`.
2. Brute-force A dimensions only.
3. Restrict A brute force to an explicit number of leading levels.
4. Require A to be smaller than or equal to the preceding A dimension between
  adjacent searched levels.
5. Use a configured uniform suffix dimension after the search window.
6. Do not branch over B or D dimensions.
7. Derive B and D independently by scanning supported dimensions from smallest
  to largest and minimizing exact secure rank.
8. Stop a B/D scan early at rank one; if no rank-one dimension exists, retain
  the dimension with the lowest rank found, preferring the smaller dimension
   on a rank tie.
9. For each exact A-dimension path, select its complete schedule by physical
  preprocessing matrix fields, then proof bytes, then descriptor bytes.
10. Select among those path representatives by the lexicographic per-level
  `(A rank, A dimension)` vector. Rank one is an early stop; if it is
  unavailable, the smallest rank wins.
11. Keep all planner work offline and runtime schedule selection catalog-only.

## Goals

1. Find the smallest secure A rank over a small, explicit dimension domain
   without sacrificing the best geometry available to each dimension path.
2. Avoid the much larger Cartesian search over A, B, D, and cross-level role
  tuples.
3. Keep dimension search bounded by configuration rather than planner constants.
4. Preserve canonical security sizing and verifier-visible schedule data.
5. Admit both D256 direct and D256 three-band paths naturally.
6. Keep planner logic independent of the concrete `OneHot` preset name.
7. Preserve deterministic generation across machines and executions.

## Non-goals

- The planner does not brute-force B dimensions.
- The planner does not brute-force D dimensions.
- The planner does not search adaptive dimensions after the configured level
window.
- The planner does not reject an A path merely because no allowed dimension
  reaches rank one.
- The planner does not optimize measured commit, prove, or verify wall time.
- Runtime proving and verification do not invoke the planner.
- Uniform D512 B/D commitments are not made valid by this policy.
- Mixed recursive-setup and mixed multi-chunk planning are not automatically
enabled.

## Terminology and dimensions

### Setup-generation dimension

The setup-generation dimension is the ring dimension used to derive and store
the shared preprocessing matrix. It is currently represented by
`CommitmentConfig::D`.

It is separate from the role dimensions used by a particular fold. Every
role-local ring dimension must divide the setup-generation dimension.

For example, a D256 setup may support D256, D128, and D64 role views, but it
cannot support a D512 role.

### Commitment-role dimensions

Every non-terminal fold has three dimensions:

```text
d_a: inner A matrix, fold arithmetic, and A-native source ring
d_b: outer B matrix
d_d: opening D matrix
```

They are represented as:

```text
CommitmentRingDims {
    inner: d_a,
    outer: d_b,
    opening: d_d,
}
```

The A-native source is projected into B and D. Each selected tuple must satisfy:

```text
d_b <= d_a
d_d <= d_a
d_a % d_b == 0
d_a % d_d == 0
```

B and D are selected independently. Neither role is ordered relative to the
other.

### Searched level

A searched level is a leading fold level at which the planner enumerates
potential A dimensions.

If `num_search_levels = 2`, the search applies only to:

```text
L0 and L1
```

L2 and every later fold use the uniform suffix dimension.

The terminal always uses the uniform suffix dimension.

### Uniform suffix

The uniform suffix begins at `num_search_levels`:

```text
first_uniform_level = num_search_levels
```

For example:

```text
num_search_levels = 2
uniform_suffix_dimension = 64

L0: search A
L1: search the same or a smaller A
L2+: A/B/D = 64/64/64
terminal: D64
```

The suffix may equal the A dimension selected at the final searched level. This
is how the search represents a direct early transition such as:

```text
L0 A256
L1 A64
L2+ uniform D64
```

Non-increasing descent applies between adjacent searched levels: the next A may
be smaller or equal, but never larger. Once the uniform suffix begins, repeated
D64 levels are expected and obey the same monotonic rule.

### Preprocessing matrix size

The primary geometry objective inside each exact A-dimension path is the
largest physical base-field footprint of any setup matrix required by the
complete schedule:

```text
setup_matrix_field_elements_for_schedule(schedule)
```

This is the canonical quantity comparable across mixed role dimensions and
different setup-generation dimensions.

It is not:

- the sum of all A/B/D matrices at all levels;
- the number of setup-generation ring elements without multiplying by their
physical field width;
- setup-seed serialization size; or
- observed resident memory.

Akita stores one shared preprocessing vector. Role matrices are prefix views,
so the schedule requires an envelope equal to its largest physical matrix or
setup-prefix footprint.

## Policy model

The following is a semantic model, not final Rust syntax:

```text
enum RingDimensionScheduleMode {
    UniformDimension {
        ring_dimension: usize,
    },

    AdaptiveDimension {
        num_search_levels: usize,
        uniform_suffix_dimension: usize,
        potential_a_dimensions: &'static [usize],
        potential_b_dimensions: &'static [usize],
        potential_d_dimensions: &'static [usize],
    },
}
```

The setup-generation dimension remains a separate config/policy value. The
mode is carried alongside the existing decomposition policy, SIS identity,
challenge hooks, basis range, chunk policy, recursive-setup policy, setup cap,
and selection policy.

### UniformDimension

`UniformDimension` defines one topology:

```text
L0 through terminal:
    d_a = d_b = d_d = ring_dimension
```

Examples:

```text
UniformDimension { ring_dimension: 64 }
UniformDimension { ring_dimension: 128 }
```

This mode performs no dimension search. It should preserve existing uniform
schedule descriptors and estimates when all other policy inputs are unchanged.

### AdaptiveDimension

`AdaptiveDimension` defines a bounded A search and deterministic B/D derivation.

An initial fp128 one-hot policy may use:

```text
AdaptiveDimension {
    num_search_levels: 2,
    uniform_suffix_dimension: 64,
    potential_a_dimensions: [64, 128, 256, 512],
    potential_b_dimensions: [64, 128],
    potential_d_dimensions: [64, 128],
}
```

D512 remains subject to the additional setup-generation and honest-fold
requirements described later. A D256 setup must filter or reject D512 rather
than attempting to search it.

The lists are role capability domains, not complete A/B/D tuple candidates.
A produces search branches. B and D lists are scanned deterministically after
the A-dependent physical widths are known.

## Bounded A-dimension brute force

### Search domain

At every searched level, the planner considers each supported A dimension that:

1. is a nonzero power of two;
2. divides the setup-generation dimension;
3. has ring-challenge and fold-arithmetic support;
4. has the required A-role SIS coverage for the derived width and bound;
5. is at least the uniform suffix dimension; and
6. is no larger than the A dimension at the preceding searched level.

At the root there is no preceding A dimension. The setup-generation dimension
is a capability ceiling, not a required root choice.

For searched levels `i > 0`:

```text
d_a[i] <= d_a[i - 1]
```

At the boundary to the fixed suffix:

```text
uniform_suffix_dimension <= d_a[last searched level]
```

Equality at the boundary is allowed because the final searched level may have
already transitioned to the suffix dimension.

### Exact bounded depth

The planner must not continue A-dimension search after:

```text
level == num_search_levels - 1
```

At:

```text
level == num_search_levels
```

the only allowed tuple is:

```text
uniform_suffix_dimension / uniform_suffix_dimension / uniform_suffix_dimension
```

This bound is part of catalog identity. Changing it changes the search policy
even if the winning rows happen to remain the same.

### Search size

Because A must be non-increasing, the A search is smaller than an unrestricted
per-level product while retaining schedules that keep the same A dimension for
multiple leading folds.

For four ordered A dimensions and two searched levels, the complete
non-increasing A paths are:

```text
512 -> 512
512 -> 256
512 -> 128
512 -> 64
256 -> 256
256 -> 128
256 -> 64
128 -> 128
128 -> 64
64 -> 64
```

Protocol support, setup divisibility, contraction, SIS availability, and setup
capacity may remove additional paths.

The planner does not construct B/D branches for each of these A paths.

### Block geometry and basis

Brute-forcing A dimensions does not mean matrix parameters are copied from a
dimension-only table. For every A path, the planner still evaluates the
canonical basis and block-layout choices allowed by the configured planner
policy.

The dependency remains:

```text
A dimension
    + current input witness
    + basis
    + block split
    + challenge shape
    -> A width, bound, and minimum secure rank
    -> B/D native widths
    -> deterministic B/D dimensions
    -> exact outgoing witness
    -> complete continuation
```

A path is not comparable until the complete schedule has been constructed and
priced.

## Deterministic B and D dimension selection

### No B/D search branches

For one fixed A-level candidate, B and D are derived independently. Their
dimension alternatives do not create multiple child schedules.

The planner scans each role's candidate list in ascending dimension order. It
tracks the supported dimension producing the smallest exact secure rank and
stops early only when rank one is reached:

```text
choose_role_dimension(role, native_width, d_a, parent_role_ceiling):
    best = None

    for d in potential_role_dimensions ascending:
        skip unless d <= d_a
        skip unless d divides d_a and setup generation
        skip unless d <= parent_role_ceiling, when a parent exists
        derive projected physical width at d
        derive role-local coefficient bound at d
        derive minimum secure SIS rank at d

        if best is None or (rank, d) < (best.rank, best.dimension):
            best = d and the derived matrix parameters

        if rank == 1:
            break

    return best, or candidate infeasible if no admissible matrix exists
```

This scan is a deterministic derivation rule, not planner branching: exactly
one B matrix and one D matrix survive for an A/geometry candidate. A secure
rank greater than one does not make the candidate infeasible.

### Why selection occurs after A derivation

B's native width depends on A's output rank. Both B and D physical widths
depend on projection from the selected A ring:

```text
physical_B_width = native_B_width * (d_a / d_b)
physical_D_width = native_D_width * (d_a / d_d)
```

Therefore B/D dimensions cannot be selected from A dimension alone. The
planner must first derive the exact A matrix and level geometry.

### B and D are independent

The planner applies the minimum-rank scan independently:

```text
d_b = supported B dimension minimizing (secure rank, dimension)
d_d = supported D dimension minimizing (secure rank, dimension)
```

Rank one is an early-stop condition because no larger dimension can improve the
rank below one. It is not a validity requirement. The policy does not require
`d_b == d_d`.

### Cross-level role ceilings

A is required to be non-increasing inside the search window. To preserve the
same monotonic transition contract, B and D must not increase:

```text
d_b[level] <= d_b[level - 1]
d_d[level] <= d_d[level - 1]
```

The preceding selected role dimension is therefore a ceiling during the
deterministic scan. If admissible dimensions exist but all produce rank greater
than one, the scan keeps the smallest rank among them. The A/geometry candidate
is infeasible only when no admissible, SIS-supported role matrix exists.

At the uniform suffix boundary:

```text
suffix_dimension <= final d_a
suffix_dimension <= final d_b
suffix_dimension <= final d_d
```

If these inequalities do not hold, the candidate cannot transition to the
configured suffix.

### A rank minimization and non-rank-one fallback

A dimensions are globally searched, but rank comparison must not select an
inferior block geometry merely to lower rank. For each exact A-dimension path,
the planner first chooses that path's best complete schedule by setup fields,
proof bytes, and descriptor bytes. It then compares the resulting path
representatives by the per-level vector:

```text
[(n_a[0], d_a[0]), (n_a[1], d_a[1]), ...]
```

The vector is compared lexicographically. At each searched level, a larger
dimension is useful only when it strictly lowers the minimum secure A rank. An
equal rank keeps the smaller dimension, and rank one prevents any larger
dimension from winning. If every allowed dimension has rank greater than one,
the smallest rank found remains valid.

This fallback is necessary for workloads such as `nv=40`, where current D256
coverage requires A rank two. Recursive and terminal A matrices also commonly
have ranks greater than one. B and D may likewise remain above rank one after
their deterministic scans exhaust the allowed dimensions.

### Known tradeoff

Choosing B/D by minimum secure rank, with smaller dimension as the tie-break,
is a policy simplification. It is not a proof that equal-rank dimensions are
globally equivalent in preprocessing size, proof bytes, or verifier work.

This proposal accepts that limitation to avoid B/D brute force. The exact rule
must be catalog-bound so that generation and runtime replay agree. If later
measurements show that the simplification excludes important schedules, a new
selection policy may admit B/D branching explicitly.

## Required D256 paths

With:

```text
num_search_levels = 2
uniform_suffix_dimension = 64
potential_a_dimensions containing 64, 128, and 256
```

the A brute-force search must include:

```text
A path 1: 256 -> 64
A path 2: 256 -> 128
A path 3: 256 -> 256
```

For current fp128 one-hot widths, deterministic minimum-rank B/D selection is
expected to materialize the corresponding role topologies:

```text
path 1:
    L0: 256/128/128
    L1: 64/64/64
    L2+: 64/64/64

path 2:
    L0: 256/128/128
    L1: 128/64/64
    L2+: 64/64/64

path 3:
    L0: 256/128/128
    L1: 256/64/64
    L2+: 64/64/64
```

The first is the D256 direct path. The second is the required D256 three-band
path:

```text
256/128/128 -> 128/64/64 -> 64/64/64
```

The dimensions above are expected outcomes, not hard-coded tuples. B/D must be
re-derived from each candidate's exact widths and minimum-rank scan. A regression
test should pin the topology for workloads where the current SIS table produces
these selections.

The L1 matrices of the three-band path must be constructed from the exact
witness produced by its D256 root. They must not be copied from an independently
planned D128-root schedule.

For the current `nv=36` representatives, the three L1 A ranks are:

```text
D64  -> rank 5
D128 -> rank 3
D256 -> rank 2
```

No allowed dimension reaches rank one, so path 3 is selected with rank two.

## Selection objective

### Per-path geometry objective

For every valid complete A path and geometry, compute:

```text
physical_preprocessing_fields =
    setup_matrix_field_elements_for_schedule(schedule)
```

For each exact A-dimension path, select the schedule with the smallest value.

The planner must not compare only `SetupMatrixEnvelope::max_setup_len`, because
that value counts setup-generation ring elements and is not directly comparable
across generation dimensions.

### A-path selection and deterministic tie-breaking

The complete selection is:

```text
representative(path) = min_by(
    physical_preprocessing_matrix_field_elements,
    exact_modeled_proof_payload_bytes,
    canonical_schedule_descriptor_bytes,
)

winner = min_by(
    per_level_(A_rank, A_dimension)_vector,
    representative_setup_fields,
    representative_proof_bytes,
    representative_descriptor_bytes,
)
```

Optimizing each dimension path before comparing ranks is important. Otherwise,
the planner could choose an unnecessarily expensive block geometry solely
because that geometry happens to lower A rank. Proof bytes select among
schedules with equal preprocessing capacity inside a path. Descriptor bytes
make an exact cost tie deterministic. A-dimension enumeration order, hash-map
order, and thread scheduling must not affect the result.

### Consequence for the D256-root paths

Current measurements in the related mixed-dimension spec show that the D256
direct and three-band schedules can have the same preprocessing footprint:

```text
D256 direct:     45,088,768 fp128 field elements
D256 three-band: 45,088,768 fp128 field elements
```

Under the revised rank-first A-path policy, the path whose representative has
the smaller L1 A rank wins. At `nv=36`, the retained-D256 L1 has rank two and
therefore wins over the three-band rank-three and direct rank-five paths. Setup
and proof bytes decide only after the per-level rank/dimension vectors tie.

The measured verifier-time advantage of one topology is not part of this
deterministic objective. Adding verifier work requires a separate versioned
cost model.

### Setup capacity

Any candidate exceeding the configured physical setup cap is infeasible.
Capacity is a feasibility constraint; it does not replace comparison among the
remaining schedules.

## Complete adaptive search algorithm

For one workload under `AdaptiveDimension`:

1. Validate the search window, suffix dimension, and role capability lists.
2. Start at L0 with every supported A dimension.
3. For every root A dimension, basis, and feasible block split:
  1. Derive A width, coefficient bound, and minimum secure rank.
  2. Derive B's native width and select B by the deterministic minimum-rank
    scan.
  3. Derive D's native width and select D by the deterministic minimum-rank
    scan.
  4. Derive the exact outgoing witness and reject non-contraction.
4. At each later searched level:
  1. enumerate only A dimensions no larger than the preceding A;
  2. derive A from the exact incoming witness;
  3. derive B/D using the deterministic minimum-rank rule and parent ceilings;
  4. derive the exact outgoing witness; and
  5. retain the required setup/proof continuation frontier.
5. Stop A enumeration after exactly `num_search_levels` leading levels.
6. Transition to uniform `suffix/suffix/suffix`.
7. Use the canonical uniform suffix planner from the exact boundary witness.
8. Reject candidates that cannot transition, contract, terminate, satisfy SIS
  coverage, or fit setup capacity.
9. Materialize each complete schedule and recompute exact setup/proof estimates.
10. Group complete schedules by their exact A-dimension path and select the
   best geometry in each group by setup fields, proof bytes, and descriptor
   bytes.
11. Compare the path representatives by their lexicographic per-level
   `(A rank, A dimension)` vectors. A strict rank reduction wins; an equal rank
   keeps the smaller dimension; rank one prevents further dimension growth;
   and the smallest available rank remains valid when rank one is unavailable.

An unsupported SIS cell for one B/D dimension causes the role scan to try the
next allowed dimension. An A/geometry candidate becomes infeasible when the
scan finds no admissible SIS-supported matrix for either role. Malformed config,
inconsistent catalog identity, invalid dimension arithmetic, or an invalid
capability list is a policy error. If all candidates are infeasible, the
workload is unsupported.

## Frontier and memoization requirements

### Non-additive setup objective

Setup and proof costs compose differently:

```text
combined.setup = max(level.setup, child.setup)
combined.proof = level.proof + child.proof
```

The planner cannot retain only one locally best suffix. A larger parent may
mask two child setup footprints, after which the child with fewer proof bytes
wins.

The planner must retain the nondominated setup/proof alternatives needed by the
complete-schedule objective, partitioned by exact first-child parameters where
parent proof pricing depends on them.

### Adaptive memo state

The memo state must distinguish at least:

```text
level
input witness length
current basis
previous A dimension
previous selected B dimension
previous selected D dimension
remaining searched levels or absolute boundary
```

Once the search enters the uniform suffix, dimension state canonicalizes to the
configured suffix dimension and can reuse the ordinary uniform suffix state.

### B/D derivation and memoization

B/D choices need not be stored as alternative branches, but their selected
dimensions are parent-visible transition ceilings and part of the exact fold
descriptor. Memo keys or state partitioning must not merge suffixes reached
through incompatible B/D selections.

## Validation requirements

### Mode validation

- `UniformDimension.ring_dimension` is a nonzero supported power of two and,
  in the current uniform engine, equals the setup-generation dimension.
- `AdaptiveDimension.num_search_levels` is positive.
- The uniform suffix dimension is a nonzero supported power of two.
- Each potential dimension list is nonempty, sorted, and unique.
- Every potential dimension divides the setup-generation dimension or is
rejected as incompatible with that config.
- The suffix dimension appears in all three role capability domains, and no
  listed role dimension is smaller than the suffix.

### A-domain validation

- Every A dimension has challenge and fold support.
- Every A dimension is at least the suffix dimension.
- The domain contains at least one A dimension at or above the suffix
dimension.
- Search never increases A inside the configured search window; repeating the
preceding A dimension is allowed.

### B/D-domain validation

- Every B/D dimension has role-local SIS and execution support.
- D512 must not be inferred for B or D merely because it appears in the A list.
- B and D domains may differ.
- The deterministic ascending minimum-rank rule and its rank-one early stop are
bound into policy identity.

### Transition validation

Inside the search window:

```text
child.d_a <= parent.d_a
child.d_b <= parent.d_b
child.d_d <= parent.d_d
```

At the suffix boundary:

```text
suffix <= parent.d_a
suffix <= parent.d_b
suffix <= parent.d_d
```

After the boundary, every fold and terminal use the uniform suffix dimension.

## D512 policy

D512 may appear in `potential_a_dimensions`, but only when the surrounding
policy supports it.

### Setup generation

A D512 role requires a setup-generation dimension divisible by 512. A current
D256 mixed setup cannot admit D512.

### Role support

Current fp128 support treats D512 as an A-role experiment. It does not certify
uniform D512 B/D matrices. Therefore an initial adaptive policy may use:

```text
potential_a_dimensions = [64, 128, 256, 512]
potential_b_dimensions = [64, 128]
potential_d_dimensions = [64, 128]
```

It must not derive `512/512/512`.

### Honest-fold sizing

Current D256 and D512 one-hot presets use different root honest-fold norm
inputs. Combining them in one A search requires either:

- a dimension-dependent honest-fold sizing hook; or
- one conservative policy proven valid across the complete A domain.

Until setup generation and honest-fold sizing are resolved, D512 must be
filtered from configs that cannot support it rather than producing a partially
valid search.

## Current `mixed_search.rs` boundary

### Why it exists

`find_schedule` now dispatches directly from the catalog-bound enum:

```text
UniformDimension  -> find_schedule_inner
AdaptiveDimension -> mixed_search::find_schedule
```

The uniform suffix DP assumes one dimension throughout its state. The adaptive
orchestration module adds:

- per-level dimension state;
- component-wise transition ceilings;
- candidate-local A challenges;
- the forced uniform suffix boundary;
- prevention of early terminal selection;
- setup/proof frontier retention; and
- per-path geometry selection followed by rank-first A-path selection.

The module is search orchestration, not a separate source of matrix security or
protocol geometry. It calls canonical candidate, SIS, proof-size, witness, and
setup primitives.

### Consolidation under the enum

The public design uses one topology-aware planner entry point:

```text
find_schedule(policy)
    -> UniformDimension: one fixed uniform path
    -> AdaptiveDimension: bounded non-increasing A search with derived B/D
```

Root and recursive matrix construction, termination, materialization, and exact
estimate revalidation are shared. The adaptive module owns only the additional
A-path/frontier state and the transition to the existing uniform suffix.

The preprocessing-first objective applies inside every exact A path. Comparing
A ranks before selecting a path representative would be incorrect because it
could distort block geometry solely to obtain a lower rank. Once every path has
its canonical representative, per-level A rank/dimension comparison selects
among paths.

Moving the remaining adaptive frontier into the uniform suffix engine is an
optional internal consolidation. `mixed_search.rs` may be removed only when:

1. one planner handles both enum modes;
2. adaptive memo state carries the required A and derived B/D ceilings;
3. the common frontier implements per-path preprocessing-first semantics;
4. uniform regressions remain unchanged where policy is unchanged;
5. adaptive selections agree with a bounded unpruned reference traversal; and
6. no planner branch refers to `OneHot` by name.

Simply moving the current module's contents into another file is not
consolidation.

## Configuration ownership

### CommitmentConfig surface

`CommitmentConfig` should project the enum mode and its static dimension domains
into the plain-value planner/runtime policy.

Configuration remains the single source of truth for:

```text
setup-generation dimension
ring-dimension schedule mode
number of searched levels
uniform suffix dimension
A/B/D capability domains
A/B/D minimum-rank derivation rules and rank-one early stops
selection objective
decomposition and basis range
SIS identity
challenge and honest-fold policy
chunk policy
recursive-setup policy
generated catalog
```

Environment variables must not alter catalog-bound policy during proving or
verification.

### Avoiding OneHot special-casing

The planner must dispatch on `RingDimensionScheduleMode`, not on the type name
`OneHot`.

Whether the marker type remains temporarily as a concrete catalog/setup policy
identity is a separate cleanup decision. The requirement is one canonical
implementation, not necessarily immediate deletion of every named zero-sized
policy type.

### Adapters

Recursive-setup and multi-chunk adapters must explicitly preserve, narrow, or
reject `AdaptiveDimension`. They must not inherit an adaptive search domain
accidentally when their planner path supports only uniform D64.

## Generated schedule table storage

The offline emitter writes generated Rust modules under:

```text
crates/akita-schedules/src/generated/
```

The bootstrap command is:

```bash
scripts/generate-schedule-tables.sh
```

For the current native fp128 mixed family, the generated files are:

```text
fp128_onehot.rs
fp128_onehot_precommitted.rs
```

Large generated `fp*.rs` family files are intentionally ignored by Git and
recreated locally and in CI. The checked-in `generated/mod.rs` supplies
feature-gated wiring and constructs the `GeneratedScheduleTable`.

Generated rows are compact Rust static data. Runtime expansion reconstructs
widths, SIS keys, and secure ranks through canonical primitives.

The current `fp128_onehot` generated family contains only:

```text
nv = 32
num_polynomials = 1
```

The complete adaptive search policy—not only the winning dimension path—must
be bound into catalog identity. Generated runtime rows need contain only the
selected schedule.

## Current benchmark and config mapping

The repository currently exercises mixed dimensions through distinct paths.


| Experiment                           | Effective config                                                                     | Schedule source                     |
| ------------------------------------ | ------------------------------------------------------------------------------------ | ----------------------------------- |
| Stock `onehot_fp128`, nv32 | `OneHot`                                                                     | Generated runtime catalog           |
| Planner diagnostic, nv18–nv40        | Policy derived from `OneHot`                                                 | Direct offline `find_schedule` call |
| D256 direct timing profile           | `MixedDConfig<D256OneHot, D64OneHot, 1>`                                             | Synthetic test-support builder      |
| Historical D256 three-band profile   | D256 root, uniform-D128 middle policy, D64 suffix                                    | Synthetic test-support builder      |


### Native generated adaptive benchmark

The stock mode:

```text
AKITA_MODE=onehot_fp128
```

uses `fp128::OneHot`, fixes nv32, calls
`Cfg::runtime_schedule`, and gives that exact generated row to the PCS prover
and verifier.

### Direct offline diagnostics

The `mixed_dimension_search` example derives policy from
`OneHot` but calls the offline planner directly. The nv18, nv24,
nv32, nv36, and diagnostic over-cap nv40 results discussed in this design
review came from this path and are not current runtime catalog rows.

### Synthetic D256 profiles

The measured D256 direct and three-band schedules currently use test-support
config adapters. They do not use the native mixed generated table.

Under this proposal, a two-level bounded A search includes these A paths:

```text
256 -> 64
256 -> 128
256 -> 256
```

After deterministic B/D selection and uniform-D64 suffix planning, the
rank/dimension rule may select any of these paths. For current `nv=36`, the
representatives have L1 A ranks 5, 3, and 2 respectively, so `256 -> 256`
wins. Once generated-catalog coverage and equivalent protocol tests exist, the
synthetic adapters may be removed.

## Catalog and runtime boundary

Adaptive search remains offline:

```text
CommitmentConfig
    -> RingDimensionScheduleMode
    -> bounded A search and deterministic B/D derivation
    -> selected generated row
```

Runtime proving and verification perform only:

```text
validate catalog identity
look up the exact workload key
expand the frozen selected row
reject a missing row
```

Runtime must not rerun A search or select B/D based on locally observed ranks.

Catalog identity must bind:

- enum mode and stable variant identity;
- setup-generation dimension;
- number of searched levels;
- uniform suffix dimension;
- ordered A/B/D dimension capability domains;
- non-increasing A and B/D monotonic ceiling rules;
- deterministic B/D minimum-rank selection, smaller-dimension tie-break, and
rank-one early-stop semantics;
- deterministic A rank/dimension path selection, non-rank-one fallback, and
  rank-one early-stop semantics;
- preprocessing/proof/descriptor representative selection inside each exact A
  path;
- SIS modulus profile, security policy, and table digest;
- challenge coverage over the complete A domain;
- honest-fold sizing identity;
- decomposition and basis range;
- chunk policy; and
- recursive-setup policy.

Changing any of these invalidates the catalog even when current winning rows do
not change.

## Implementation sequence

This section describes a possible future sequence; it does not authorize code
changes.

### Cut 1: policy and identity

1. Add `UniformDimension` and `AdaptiveDimension`.
2. Add search-level, suffix, and role capability fields.
3. Define and bind the deterministic B/D minimum-rank rule and rank-one early
  stop.
4. Define and bind A path representative selection followed by per-level
  rank/dimension selection and non-rank-one fallback.
5. Extend catalog identity with every new semantic input.
6. Express existing uniform configs through `UniformDimension`.

### Cut 2: bounded adaptive engine

1. Enumerate non-increasing A paths only inside the configured window.
2. Derive B/D without branching.
3. Preserve exact setup/proof frontiers inside each A path.
4. Hand the exact boundary witness to the uniform suffix planner.
5. Select each path representative by preprocessing fields, proof bytes, and
  descriptor bytes, then select the A path by per-level rank and dimension.

### Cut 3: D256 generated coverage

1. Add workloads that exercise A paths `256 -> 64`, `256 -> 128`, and
  `256 -> 256`.
2. Pin derived D256 role topologies where current SIS data supports them.
3. Compare generated/runtime replay with current synthetic protocol coverage.
4. Remove synthetic adapters only after coverage is equivalent.

### Cut 4: planner consolidation

1. Generalize the common schedule state over the enum mode.
2. Remove config-name dispatch.
3. Delete `mixed_search.rs` when its state/frontier behavior is fully owned by
  the common engine.

### Cut 5: optional D512 admission

1. Resolve D512 setup generation.
2. Resolve dimension-dependent honest-fold sizing.
3. Validate D512 A-role search with D128/D64 B/D derivation.
4. Regenerate catalogs under the new identity.

## Required tests


| Area                      | Required property                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Enum validation           | Uniform and adaptive modes reject malformed policy data                                                         |
| Uniform preservation      | Equivalent `UniformDimension` policies preserve existing descriptors and estimates                              |
| Search boundary           | A is enumerated only at levels `< num_search_levels`                                                            |
| Non-increasing A          | Smaller and equal A transitions are accepted; increasing A transitions inside the search window are rejected    |
| Suffix boundary           | Level `num_search_levels` and later, including terminal, are uniform at the configured suffix dimension         |
| A enumeration             | Every supported non-increasing A path in the bounded domain is considered                                       |
| Per-path representative   | Each exact A path first selects its best complete setup/proof/descriptor geometry                                |
| A rank scan               | Path representatives minimize A rank at each searched level; rank one stops dimension growth                    |
| A rank tie                | Equal A ranks prefer the smaller A dimension                                                                     |
| A non-rank-one fallback   | If no allowed A dimension reaches rank one, the smallest secure rank remains valid                               |
| No B/D branching          | Each A/geometry candidate produces at most one derived B and one derived D choice                               |
| B rank scan               | B dimensions are tried in ascending order, the lowest exact secure rank wins, and rank one stops the scan early |
| D rank scan               | D dimensions are tried in ascending order, the lowest exact secure rank wins, and rank one stops the scan early |
| B/D rank tie              | Equal secure ranks select the smaller role dimension                                                            |
| B/D non-rank-one fallback | If no dimension reaches rank one, the lowest-rank supported B/D matrix remains valid                            |
| B/D infeasibility         | An A/geometry candidate is infeasible only when a role has no admissible SIS-supported dimension                |
| Role monotonicity         | Derived B/D dimensions do not increase across searched levels                                                   |
| A rank                    | A candidates with rank greater than one remain eligible only when no allowed dimension lowers that rank         |
| D256 direct               | The two-level search can derive `256/128/128 -> 64/64/64 -> D64`                                                |
| D256 three-band           | The two-level search can derive `256/128/128 -> 128/64/64 -> D64` from exact boundary witnesses                 |
| Objective                 | A rank/dimension paths win first; setup, proof, and descriptor select the representative within each path       |
| Setup tie                 | Equal preprocessing sizes within one A path compare proof bytes, then descriptor bytes                          |
| Frontier                  | Parent-masked child setup differences retain the correct lower-proof alternative                                |
| Setup cap                 | Over-capacity candidates do not exclude valid siblings                                                          |
| D512 setup                | D512 A candidates reject under D256 setup generation                                                            |
| D512 roles                | D512 is never inferred for unsupported B/D roles                                                                |
| Catalog drift             | Changing domains, search depth, suffix, B/D rule, or objective invalidates catalogs                             |
| Runtime boundary          | Missing rows reject without planner fallback                                                                    |
| Adapter policy            | Recursive and multi-chunk adapters explicitly preserve, narrow, or reject adaptive mode                         |


## Acceptance criteria

The first implementation is complete when:

1. `RingDimensionScheduleMode` exposes `UniformDimension` and
  `AdaptiveDimension`.
2. `AdaptiveDimension` carries a bounded leading search depth, uniform suffix,
  and separate A/B/D capability domains.
3. Only A dimensions produce search branches.
4. A is smaller than or equal to its preceding dimension between adjacent
  searched levels.
5. A search stops at the configured boundary.
6. B and D are independently derived by ascending dimension scans that minimize
  exact secure rank, stop early at rank one, and prefer smaller dimensions on
   rank ties.
7. A uses the smallest secure rank available at each searched level, prefers
  the smaller dimension on a rank tie, and remains valid above rank one when
  the domain cannot reduce it further.
8. Each exact A-dimension path first selects its best complete schedule by
  preprocessing fields, proof bytes, and descriptor bytes; path
  representatives are then selected by their per-level A rank/dimension vector.
9. A two-level search includes both D256 paths:
  ```text
   256/128/128 -> 64/64/64 -> D64
   256/128/128 -> 128/64/64 -> D64
  ```
10. Planner logic contains no branch keyed on `OneHot`.
11. Runtime remains generated-catalog-only.

## Remaining implementation questions

The main policy direction is resolved. These implementation-level questions
remain:

1. Should invalid dimensions be rejected when constructing the policy, or
  filtered as unsupported capabilities before catalog identity is computed?
2. Should `num_search_levels` be capped by a small protocol constant to bound
  adversarial custom policies in offline tools?
3. Should basis and block-split enumeration remain exhaustive at every searched
  A level or use the current canonical suffix split heuristic at later levels?
4. Which config type initially owns the adaptive policy after planner
  special-casing is removed?
5. Which adapters narrow adaptive mode to uniform D64?
6. What honest-fold API is required before D512 enters the active A domain?
