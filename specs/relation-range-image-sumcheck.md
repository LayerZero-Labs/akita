# Relation, evaluation-trace, and range-image sum-check

| Field | Value |
|---|---|
| Author(s) | Quang Dao (protocol and implementation direction); Codex (design synthesis) |
| Created | 2026-07-20 |
| Status | implementation in progress; verifier characterization slice |
| Branch | `quang/relation-range-image-rewrite` |
| Base | `main` at `e131faf48938b975ca63b12b59ac6d86894048e0` (includes PR #312) |
| Integration dependencies | PR #309 at `b0c2d4683539b0c2a465b996f48adfc465a20198`; PR #310 at `4cb4113b02a58889230f3dbaa81deb56895bb4ca` as cross-feature evidence |
| Related | [`digit-range-pipeline-refactor.md`](digit-range-pipeline-refactor.md), [`digit-innermost-layout.md`](digit-innermost-layout.md), [`runtime-ring-cutover.md`](runtime-ring-cutover.md), [`packed-sumcheck.md`](packed-sumcheck.md) |

## Summary

This document specifies the intended reimplementation of the direct/non-offloaded fused
sum-check over the digit witness. The current PR head has landed checked geometry,
relation-weight factorization, prepared evaluation-trace support, and direct mixed-role
verifier differential coverage. It has **not** yet replaced the existing Stage 2 prover
state machine, removed its x/y and `TraceTable` representations, or demonstrated genuine
mixed-role proving end to end.

The sum-check has three semantic terms:

1. the factorized linear relation;
2. the mandatory evaluation trace that binds opening claims to the committed witness; and
3. range-image consistency with the independent Stage 1 evaluation.

The goals and acceptance criteria below describe the final PR head, not every additive
implementation slice. At the current verifier-correction slice, direct mixed-role
evaluation has W=1/W=2 differential parity, but genuine mixed-role statement construction
and end-to-end proving remain part of the prover cutover below. Recursive mixed-role setup
offload is deliberately rejected until the later Stage 3 boundary change.

All three share one transcript lifecycle and checked protocol geometry. The prover owns
one round-result reducer and the storage it needs to produce messages. The verifier owns
separate succinct evaluators built from the same checked parameters; it never consumes a
prover table, event stream, folded state, or other witness-sized representation. The
three terms keep distinct named arithmetic; this is not a generic expression engine.

## Motivation

The current implementation is correct but organized around accidental representations:

- public x/y geometry and layout-specific prefix/dense paths;
- `ring_bits == 0` as a mixed-dimension sentinel;
- duplicated serial and parallel arithmetic;
- a uniform-dimension column factorization beside a dense mixed-dimension fallback;
- `FieldSparse` and `RingDense` evaluation-trace tables that encode the same tensor in
  different physical forms;
- full trace-table folding, hot sparse searches, and source/destination remapping;
- special root, recursive, multigroup, and EOR trace branches; and
- wrappers around shared two-round code rather than one equation-owning prover.

These branches obstruct the main performance levers: binding the alpha coordinates
first, delaying witness materialization, applying lane factors after local reduction,
retaining evaluation-trace tensor structure, and specializing compact kernels per digit
basis.

## Goals

- One checked `RelationRangeImagePlan` and one `RelationRangeImageProver`.
- One LSB-first flat Boolean address order for homogeneous and mixed dimensions.
- Distinguish the relation-only alpha-reset boundary
  `common_relation_coeff_count = min(d_a, d_b, d_d)` from the current joint
  relation/witness address boundary
  `common_relation_witness_coeff_count = min(common_relation_coeff_count, d_out)`.
- Alpha/common-coordinate rounds before relation-lane rounds.
- Mandatory evaluation trace represented by per-claim tensor factors, never a flat table.
- One fused relation/evaluation-trace/range-image round reducer.
- Measured one/two/three-round compact strategies for every LB2-LB6 basis.
- Full multi-group, multi-chunk, and mixed-dimension cross-product support.
- One checked common-parameter authority for transcript challenges, layouts, role
  dimensions, row ranges, group/chunk ownership, opening points, and setup attribution.
- Separate minimal prover and verifier representations derived from those parameters:
  prover events/factors for round production, and compact structured verifier evaluators
  for final-point replay.
- Preserve or improve PR #312 verifier performance in every primary benchmark cell;
  a candidate may regress by at most 5% under the pinned comparison protocol below.
- Atomic deletion of old x/y, dense, sparse, and forwarding paths.
- No proof-byte, transcript, degree, challenge, or final-claim change.

## Non-goals

- Change the Stage 1 range-product proof implemented by PR #312.
- Move the relation into Stage 1 or change recursive setup-offload proof shape.
- Delete or move the current numeric setup-contribution stage. This PR adapts only the
  reusable mixed-dimension/group/chunk geometry it consumes.
- Introduce compressed commitments or the fused negative-binary range check.
- Add mixed per-address range polynomials. Current supported folds certify the complete
  digit witness under one checked range basis.
- Add a production dense fallback, exact-fringe provider, runtime kernel selector,
  compatibility wrapper, or generic sum-of-products abstraction.

The later recursive-offload PR gets its own spec when implemented. Its stable seam is
recorded near the end of this document; this PR does not carry a speculative proof-wire
plan for it.

## Shared inputs, separate consumers

“Single source of truth” applies to protocol facts, not runtime representations. The
dependency graph is deliberately

```text
checked common protocol parameters
  -> prover-owned compact/foldable state

checked common protocol parameters
  -> verifier-owned succinct evaluation state
```

Common parameters include the transcript-derived challenges, `LevelParams`,
`OpeningClaimsLayout`, `WitnessLayout`, authenticated group/claim order, role dimensions,
row ranges, gadget bases, opening points, and setup-contribution attribution. Neither side
recomputes those facts through an alternate layout or sizing formula.

The two sides have different cost models and therefore must not share a representation
merely to share code. Prover event streams, dense/lane weights, trace materialization, and
folded factor state may scale with the witness because the prover already touches it.
Verifier state must remain succinct and retain the established structured contractions.
Cross-side agreement is enforced through the checked inputs and differential direct-
equation tests, not by making the verifier replay prover-scale data.

A verifier evaluator is the production implementation, not a “compact fast path” beside
a generic slow implementation. There is one verifier algorithm per concept, with no
runtime representation switch or wrapper around a prover consumer.

## Protocol boundary

PR #312 Stage 1 supplies

```text
range_image_evaluation
  = MLE_z(digit_witness(z) * (digit_witness(z) + 1), range_check_point).
```

After sampling `range_binding_challenge`, this PR proves the unchanged standard
degree-three claim

```text
relation_claim
  + evaluation_trace_claim
  + range_binding_challenge * range_image_evaluation

= sum_z digit_witness(z) * [
      common_alpha_factor(coefficient(z))
        * relation_lane_weights(lane(z))
    + evaluation_trace_weight(z)
    + range_binding_challenge
        * Eq(range_check_point, z)
        * (digit_witness(z) + 1)
  ].
```

The output remains `next_witness_evaluation` at the complete flat Stage 2 point. The
relation and evaluation-trace terms are linear in the digit witness. The range term is
quadratic in it and linear in equality, so the standard degree remains three.

The direct pipeline remains:

```text
Stage 1: equality-factored range-product tree
  -> range_image_evaluation

Stage 2: relation + evaluation trace + range-image consistency
  -> next_witness_evaluation

Current setup-contribution stage: unchanged proof position and wire
```

## Naming contract

| Mathematical object | Production name |
|---|---|
| balanced digit table | `digit_witness` |
| pointwise `w(w+1)` table | `range_image` |
| largest role-common alpha-reset block | `common_relation_coeff_count` |
| current joint relation/witness low-address block | `common_relation_witness_coeff_count` |
| alpha powers over the joint low-address block | `common_alpha_factor` |
| high-lane linear-relation factor | `relation_lane_weights` |
| prover opening-trace factors | `EvaluationTraceWeights` |
| prepared trace contraction/fold state | `EvaluationTraceProverState` |
| verifier opening-trace claims | `PreparedEvaluationTrace` |
| verifier relation-matrix state | `RelationMatrixEvaluator` |
| complete checked direct plan | `RelationRangeImagePlan` |
| equation-owning prover | `RelationRangeImageProver` |
| proof object | `RelationRangeImageProof` |

`W` is acceptable only in short equations. Ambiguous `S` names are forbidden. Production
code does not use `TraceWeightState`, `ExactFringeWeight`, x/y-prefix names, or numeric
stage names for new internal objects.

## One flat address order

All rounds bind the raw physical field-coefficient address LSB first. Two counts must not
be conflated:

```text
common_relation_coeff_count = min(d_a, d_b, d_d)
common_relation_witness_coeff_count
  = min(common_relation_coeff_count, outgoing_witness_ring_dimension)

physical_address
  = common_relation_witness_coeff_count * relation_lane
    + coeff_within_common_block
0 <= coeff_within_common_block < common_relation_witness_coeff_count
```

The first `log2(common_relation_witness_coeff_count)` challenges bind the joint low
coefficient block. The rest address physical relation lanes and padded witness capacity.
x/y is not a public protocol abstraction.

The outgoing dimension appears only because the current prover groups the flat witness
into outgoing ring elements before Stage 2. Keeping the low block no wider than one such
element preserves the outgoing opening domain, point order, and current Stage 3 handoff;
a wider relation-only block could fold across outgoing column boundaries. The outgoing
dimension is not used by the relation equation, role-local alpha resets, or matrix
semantics. A verifier relation evaluator that does not consume that storage representation
should derive its algebra from `common_relation_coeff_count`; if it splits the final point
at the smaller joint count, that split is address geometry, not an additional relation
contract.

This Stage 2 storage split does not reinterpret the already established Stage 1 tau0
point. The protocol's digit-range equality point retains its incoming geometry: the
historical uniform current/outgoing case uses its column-then-ring permutation, while
mixed-role or cross-level ring transitions retain the historical flat tau0 order. A
descriptive `digit_range_equality_low_variable_count` records that independent boundary;
it is not inferred from the joint relation/witness variable count.

`FlatBooleanDomain` checks the live prefix, padded power-of-two domain, point width, and
LSB-first interpretation. `WitnessLayout` checks semantic group/chunk units inside that
prefix. Callers do not pass independent `live_x_cols`, `col_bits`, and `ring_bits` values
that can disagree with those authorities.

## Range-basis authority

The direct sum-check receives one checked `digit_witness_range_log_basis` for the complete
flat witness. For schedules integrated from PR #309, role-specific inner/outer/open bases
still describe how semantic rows were decomposed; they do not silently select different
Stage 1 range polynomials inside one proof. The current shared authority is the level's
open/fold range basis.

Every security/norm/sizing calculation that relies on the Stage 1 digit bound must price
the same global witness bound that the verifier enforces. If a future schedule wants to
claim tighter role-specific range bounds, it needs a separately specified mixed-range
proof and cannot obtain them merely from decomposition metadata. This PR fails checked
integration rather than creating a split-brain range/security contract.

## Relation factorization

For each role dimension, write

```text
d_role = common_relation_witness_coeff_count * role_lane_count

alpha^(common_relation_witness_coeff_count * high_exponent
        + coeff_within_common_block)
  = alpha^coeff_within_common_block
      * (alpha^common_relation_witness_coeff_count)^high_exponent.
```

The complete non-trace relation weight is

```text
RelationWeight(
  common_relation_witness_coeff_count * relation_lane
    + coeff_within_common_block
)
  = CommonAlphaFactor(coeff_within_common_block)
      * RelationLaneWeights(relation_lane).
```

Role resets, quotient denominators, row challenges, claim coefficients, group weights,
setup amplitudes, and overlaps are additive contributions to high-lane weights. They do
not break the common low factor. With outgoing dimension at least 32, mixed
`128/64/32` therefore uses a common factor of length 32, not 128 and not a full-domain
table. If the outgoing dimension is smaller, the current factorization may use the
smaller joint block even though the relation-only boundary remains 32.

The prover's relation compiler uses one closed emitter of checked relation events:

```rust
struct RelationWeightEvent<E> {
    physical_coefficients: Range<usize>,
    alpha_exponent_start: usize,
    scalar: E,
    contribution: RelationWeightContribution,
}
```

The canonical compiler returns both factors together as
`RelationWeightFactorization { common_alpha_factor, relation_lane_weights }`. The
ring-switch handoff carries that object into Stage 2, which consumes both vectors; it
does not discard and regenerate the alpha factor from the challenge. The current
consumer still uses the existing x/y-shaped Stage 2 state machine; the later cutover
changes that storage/state representation, not the factorization contract.

Every production event has physical and exponent starts aligned to
`common_relation_witness_coeff_count` and preserves the low common coordinate. Role-local
resets begin new aligned events. Overlaps use `+=`
exactly once. Unsupported unaligned events return `AkitaError`; there is no production
fringe or dense fallback. A dense exact-live vector exists only as a differential oracle.

`contribution` distinguishes protocol-constraint arithmetic from setup-matrix arithmetic,
so direct setup evaluation and a deferred complete setup claim cannot be mixed. All emitted
exponents are consecutive; an exponent-pattern enum would add no information. The current
dense test oracle and production factorized consumer preserve Stage 2 behavior while the fused
prover is cut over. They consume these events and contain no second relation formula.

The emitter explicitly covers E/D consistency, T/B, every per-chunk Z response, the
shared quotient-R suffix, setup contributions, and row-family resets. It is the prover
authority for lane compilation and dense prover-side differential oracles. Source events
are dropped after coalescing; no prover provider retains dense and factorized copies.

The verifier does **not** build or consume these events. Its relation evaluator is derived
directly from the checked common inputs and retains the PR #312 structured algorithm:
prepared claim/block challenge evaluations; bounded equality windows and affine-interval
E/T contractions; reuse of setup-plan E/T/Z equality slices; a compact quotient-tail
contraction; and one coefficient-side alpha evaluation for uniform dimensions. Direct
setup and deferred setup claims remain explicit mutually exclusive inputs. Refactoring
may improve ownership, validation, names, and module boundaries, but may not replace
these contractions with leaf event replay.

## Binding order and relation arithmetic

The joint common alpha coordinates bind first. Relation lane weights are constant over each
low-coordinate block and are not folded during these rounds. For

```text
w(T) = w_0 + T * delta_w,
a(T) = a_0 + T * delta_a,
```

the local relation polynomial is `w(T) * a(T)`. Accumulate its coefficients over a
lane-aligned block, then multiply the totals by `relation_lane_weights[lane]` once. Do
not multiply every endpoint separately.

After the final joint common-coordinate challenge, retain

```text
common_alpha_evaluation = MLE(CommonAlphaFactor, common_point).
```

Only then bind and fold lane weights. Remaining rounds accumulate

```text
common_alpha_evaluation
  * digit_witness(T)
  * relation_lane_weight(T).
```

Serial and Rayon schedules call the same block arithmetic. Compact rounds use bounded
signed/unreduced accumulators; field rounds use delayed reduction only under the field's
proved contract.

## Mandatory evaluation trace

Every production nonterminal fold has opening claims, so evaluation trace is mandatory.
Root, recursive, multigroup, and extension-opening-reduction paths differ only in their
claim list and coefficients. The terminal path has no Stage 2 after #311.

For claim `ell`, source coordinates `(block, digit, nu)` have weight

```text
evaluation_trace_weight_ell(block,digit,nu)
  = normalized_claim_coefficient_ell
      * block_opening_weight_ell(block)
      * opening_digit_gadget_ell(digit)
      * inner_trace_ell(nu).
```

The normalized coefficient already includes the public row coefficient and any EOR
factor. Prover and verifier normalize it once while preparing the claim list; the Stage 2
builder does not branch on root, recursive, grouped, or EOR mode. The EOR factor is one
when no reduction applies. `inner_trace` is the source ring-coordinate trace from the tail
opening point; for scalar openings it is the ordinary equality row. For extension openings,
extension-linearity keeps the block/gadget scalar outside and changes only the inner factor.

The complete trace is a sum of rank-one claim factors:

```text
evaluation_trace_weight(block,digit,nu)
  = sum_ell column_factor_ell(block,digit) * inner_trace_ell(nu).
```

The prover's normalized trace input is a nonempty `EvaluationTraceWeights` containing one
`EvaluationTraceTerm` per opening claim. There are no prover-side `Field`, `Ring`, `Root`,
`Recursive`, `Absent`, sparse-table, or dense-table semantic variants.

The landed term representation owns exactly:

- one normalized claim coefficient;
- the block-opening point and basis;
- the group's exact live-block count and source ring dimension;
- opening-digit gadget weights and one precomputed inner-trace row; and
- one `EvaluationTraceSegment` per witness chunk, recording the flat physical coefficient
  start, group-global block start, and exact live-block count.

The verifier receives its own minimal prepared group descriptors from the same checked
claim coefficients, opening points, group/chunk ownership, basis, and source dimensions.
Each descriptor retains compact chunk geometry shared by every claim in the group, not
one copy of the prover physical segments per claim. It contracts the rank-one block,
digit, and inner-coordinate factors in closed form. It must not scan or materialize the
prover physical trace segments. The current Stage 2 prover compiles
`EvaluationTraceWeights` into the prover-owned `PreparedProverEvaluationTrace`: one
scaled factor per exact live opening block/digit, one shared source-inner vector per
claim, and common-coordinate column ranges. `into_stage2_fold_table` then adapts that
support to the existing foldable `TraceTable` policy: scalar same-dimension claims use
sparse
columns, while extension or mixed-dimension claims use one exact-live flat dense table.
The prepared support survives the fused prover cutover; the bridge and `TraceTable`
disappear in Step 6. Deleted historical trace implementations may remain only under
`cfg(test)` as differential oracles.

The implementation differential-tests the current coordinate algebra against the direct
E-linear formula. Where the packing map admits the expected adjoint, each inner vector is
computed once in `O(D_source * extension_degree)` rather than by repeatedly opening a
ring row.

## Multi-group and multi-chunk contract

The complication matrix is:

| Axis | Range-image term | Relation term | Evaluation trace | Required special support |
|---|---|---|---|---|
| multiple groups | none after flat layout | group-specific rows, gadgets, claim offsets, and exponent resets | one prepared point and claim slice per group | compile authenticated root-group order into runs |
| multiple chunks | none after flat layout | E/T split by block ownership, Z repeated per chunk, R shared | one physical E segment per claim/unit; global block weights do not reset | retain uneven global block ranges and unit ownership |
| mixed role dimensions | none | common factor uses the joint relation/witness count; role subcolumns map into common lanes | each claim retains its own source ring split | remove the existing mixed-D/multichunk guard only after flat-oracle parity |
| EOR | none | no change | one reduction factor scales the authenticated claim coefficients | normalize a missing reduction factor to one; never make trace optional |

Only the range-image column is genuinely indifferent to these axes. Relation and trace
arithmetic generalize cleanly once geometry is prepared, but that preparation is a
load-bearing part of the implementation rather than incidental indexing.

### Canonical physical layout

The implementation must support the full cross-product, not just singleton/single-chunk
fixtures. The canonical layout is supplied by `WitnessLayout`:

```text
for group in opening_batch.root_group_order():
  for unit in witness_layout.units_for_group(group):
    [z(group, chunk) | e(group, chunk) | t(group, chunk)]
[shared quotient r suffix]
```

Each `WitnessUnitLayout` owns a group id, chunk id, global block range, and exact z/e/t
physical ranges. The block partition may be uneven; chunks before the remainder boundary
own one extra block. Kernels iterate prepared units and runs. They do not assume equal
chunk widths, `groups * chunks` rectangularity, or reconstruct offsets.

### What generalizes without special arithmetic

- Stage 1's result and the Stage 2 range-image term see one flat digit witness. Group and
  chunk boundaries do not change `w(w+1)` or equality arithmetic.
- The common-alpha scan is unchanged because every unit and role-subcolumn boundary is
  aligned to a physical ring coefficient boundary and therefore to
  `common_relation_witness_coeff_count`.
- One transcript challenge folds every physical address. Chunks do not receive separate
  sum-check challenges.
- The shared quotient-R suffix is one ordinary relation segment; it is not replicated by
  the number of groups or chunks.
- Padding remains one zero suffix after the live physical layout, not padding between
  units.

### Group-specific inputs that require preparation

Each group may have a different claim count, number of live blocks, positions per block,
opening/commit/witness digit depths, semantic gadget bases, prepared opening point, and
relation-row range. Preparation resolves:

- the authenticated root-group order;
- global claim offsets and per-claim batching coefficients;
- group-specific block-opening weights;
- role-specific relation events and exponent resets; and
- exact unit-local z/e/t runs.

No hot loop calls `group_params`, searches claim offsets, or repeatedly scans
`unit_for_block`. `RelationRangeImagePlan` compiles those checks once into coalesced runs
while retaining `WitnessLayout` as the semantic source of truth.

### Chunk-specific semantics

Chunks partition a group's **global** block axis. They do not create claims. For a unit
with global block start `B`, local block `j`, claim `c`, and opening digit `d`, the E
column is conceptually

```text
unit.e_start
  + depth_open * (c * unit.num_blocks + j)
  + d,

global_block = B + j.
```

The evaluation trace uses `block_opening_weight(global_block)`. It must not reset the
block index to zero at each chunk. E and T supports are split across units; Z is a separate
fold response replicated once per chunk; R remains global. Relation emission and setup
attribution must preserve those distinctions.

### Evaluation trace over the cross-product

Each `EvaluationTraceTerm` owns its normalized group/claim coefficient, source ring
dimension, digit gadget and inner trace vectors, and a list of physical E segments—one per
relevant witness unit. A segment records its flat physical coefficient start and global
block start/count; digit depth is term-owned and claim placement was already resolved into
the physical start. This expresses multigroup and multichunk support without
`TraceTermBatch`, `dense_evals`, or a remapped table.

The prover measures three exact strategies:

1. structured trace values inside the main pair/quad/octet witness scan;
2. an opening-only scan over prepared E segments; and
3. contraction-first scanning per claim, unit, and ring-coordinate chunk.

The selected side scan is still part of the one round reducer. It is permitted only when
it visits strictly smaller support and wins complete-stage measurement. It never creates
a second prover or witness table.

The verifier uses the prepared claim/group/chunk descriptors to contract block-opening,
digit, and inner-coordinate factors without visiting the physical segment contents. An
arbitrary physical E interval is not assumed to be a Boolean subcube, but carry-aware
equality arithmetic is performed over compact factor boundaries rather than one term per
physical coefficient. Complexity is succinct in claims, groups, chunks, point width, and
short gadget/factor dimensions; it is not proportional to the padded or live witness
domain.

### Mixed ring dimensions with chunks

The previous verifier explicitly rejected multi-chunk layouts when B/D role dimensions
differed from the A dimension. The direct-setup path removes that guard: the relation
emitter, trace segments, setup attribution, and verifier evaluator use the same checked
role/lane mapping and must agree with dense flat oracles across the mixed-dimension/chunk
cross-product.

Recursive setup offload remains a separate cutover boundary. The current Stage 3 builder
drops `log2(d_a)` Stage 2 coordinates before constructing the setup product, whereas a
mixed relation binds only `log2(common_relation_witness_coeff_count)` common alpha
coordinates before its remaining role
lanes. Reusing the uniform split would silently omit the A/B lane coordinates. Until the
Stage 3 setup boundary is generalized to consume the checked common-coordinate count,
mixed dimensions plus a deferred setup claim are rejected explicitly. The later setup
boundary slice must remove that rejection and add direct/deferred parity across mixed
dimensions, groups, and chunks; it must not reinterpret or pad the old `x/y` split.

The address rule is unchanged by chunks:

```text
common_relation_coeff_count = min(d_a, d_b, d_d)
common_relation_witness_coeff_count
  = min(common_relation_coeff_count, outgoing_witness_ring_dimension)
```

The relation algebra alone admits `common_relation_coeff_count`, but the current Stage 2
prover uses the intersection with the outgoing dimension so its factorized witness state
is also aligned to the outgoing witness representation. Role subcolumns map into
`common_relation_witness_coeff_count`-sized lanes inside each unit. Chunking changes which
unit owns a block, not alpha exponents or source ring-coordinate order. Evaluation-trace terms retain their own
`D_source` split

```text
source_address = D_source * opening_column + ring_coordinate
```

against the same flat challenge vector. No source-to-destination trace remap is allowed.

### Existing PR evidence and what it does not prove

At the recorded heads:

- PR #309 includes the end-to-end
  `multi_group_multi_chunk_fold_round_trips` fixture and carries explicit
  inner/outer/open basis metadata through witness/relation/trace construction.
- PR #310 proves a generated multi-group recursive setup-offload schedule with a
  multi-chunk witness layout end to end.

These are required integration fixtures and useful address-layout oracles. They do not
authorize copying current dense trace tables, the mixed-D rejection, or old stage-shaped
dispatch into the new implementation. The final semantic shape in this spec wins.

## One fused prover lifecycle

`RelationRangeImageProver` owns one checked plan and one phase state:

```rust
enum RelationRangeImageState<E> {
    CompactPrefix(CompactPrefixState<E>),
    FoldedSuffix(FoldedSuffixState<E>),
}
```

The phase is matched once per round, outside hot scans. Compact state owns exactly one
signed witness, common-alpha state, relation lanes/runs, range equality/recovery state,
prepared trace factors/segments, and the selected basis cache. Folded state owns exactly
one field witness plus the minimum surviving factor states.

Every compact reducer returns three named subtotals:

```text
FusedRoundCoefficients {
  relation,
  evaluation_trace,
  range_image,
}
```

The final round polynomial is assembled once. This result is not a generic expression
graph. Basis-specific loops may enumerate digits differently but call canonical relation,
trace, and range arithmetic.

There is one joint compact-to-field witness transition. If the selected prefix defers
`r` rounds, it materializes exactly `N / 2^r` witness values after the `r`th challenge and
drops superseded compact state. Individual trace/range factors may remain analytical
longer; they do not create independent witness materialization depths.

```text
compact common-coordinate prefix          rounds 0 .. materialization_round
field common-coordinate suffix             rounds materialization_round .. common_variable_count
field relation-lane suffix                 rounds common_variable_count .. num_variables
```

## Initial-round policy and optimization candidates

All viable candidates are implemented against the same dense oracle and measured with
the feature-pruned profile build. The best complete-stage strategy remains per basis;
losing production code and selectors are deleted.

| Basis | Mandatory candidates |
|---:|---|
| LB2 / 4 | pair; factorized two-round quad; three-round octet with 256 range-image patterns and direct signed relation/trace arithmetic |
| LB3 / 8 | pair; factorized two-round quad; three-round octet with folded-quad/Taylor range arithmetic and direct signed relation/trace arithmetic |
| LB4 / 16 | pair; factorized two-round quad; 4,096-key challenge-dependent range quartet plus direct relation/trace |
| LB5 / 32 | pair; factorized two-round rescan with split equality, delayed reduction, and compact range aggregation |
| LB6 / 64 | pair; factorized two-round rescan with split equality, delayed reduction, cache blocking, and prefetched pair indices |

Range-image classes cannot index the signed relation term: `w` and `-w-1` collide under
the range image but contribute different linear values. LB2's 256-entry octet table is
therefore range-only; relation and trace stay signed/direct in the same traversal.

Every candidate also tests, where applicable:

- applying equality, relation lanes, and trace column factors once per local block;
- compact rescan versus retained field prefix state;
- contract-gated delayed reduction;
- fused post-prefix materialization plus immediate next-round accumulation, retaining all
  existing split-equality and aggregation optimizations;
- cache blocking and pair-index prefetch inside the canonical iterator; and
- the LB5 challenge-dependent quartet table with its roughly 5 MiB footprint fully
  charged to complete-stage performance.

Ordinary transcript causality is preserved. Two- or three-round deferral sends and
receives ordinary messages/challenges before materialization; it is not a batched
Fiat-Shamir message.

Multi-group/multi-chunk layouts do not choose independent kernels. The complete witness
has one checked range basis and one joint prefix boundary. Group/unit segment boundaries
are ring-aligned and therefore pair/quad/octet aligned for supported dimensions.

## Later rounds

One canonical field-round implementation handles remaining common-coordinate and lane
rounds. It folds relation lane weights only after common alpha becomes scalar. It folds
each evaluation-trace inner vector once per source-coordinate challenge, not once per
opening column, and folds surviving column factors afterward.

After source ring-coordinate bits collapse, benchmark retaining structured segment
factors against materializing one exact column vector. Never materialize the full
column-by-coordinate tensor.

A fold-and-next-round fusion is retained only when it removes a physical read and improves
the complete stage. The negative Stage 1 experiment is evidence, not a prohibition,
because Stage 2 has different product terms and cache pressure.

## Verifier

The verifier replays the unchanged degree-three sum-check. Its final expected value is

```text
range_binding_challenge
  * Eq(range_check_point, next_witness_point)
  * next_witness_evaluation
  * (next_witness_evaluation + 1)

+ next_witness_evaluation
  * common_alpha_evaluation(common_point)
  * relation_lane_weight_evaluation(lane_point)

+ next_witness_evaluation
  * evaluation_trace_weight_evaluation(next_witness_point).
```

The canonical trace evaluator computes

```text
sum_claims claim_coefficient
  * extension_opening_reduction_scale
  * opening_column_weight(point)
  * inner_trace_evaluation(point).
```

It accepts one minimal prepared claim list for root, recursive, multigroup, multichunk,
and EOR cases. It does not allocate dense tables, build prover trace terms/segments,
remap points, or dispatch by historical representation. Typed point views validate
common, lane, and per-trace source splits.

The relation evaluator is especially performance-sensitive and preserves the compact
PR #312 implementation strategy. There is one verifier-owned relation formula. Uniform
dimensions may specialize prepared lane factors and matrix contractions locally, but
they do not select a second top-level evaluator. Dispatch may depend on semantic role
dimensions, never merely on whether the outgoing witness happens to use the same ring
dimension.

The canonical point preparation is `PreparedRelationPoint`. It owns the checked
`coeff_count`, the common coefficient evaluation, one bounded equality window over the
remaining lane/column point, role-specific lane-alpha factors, and checked role-subcolumn
addresses. Its characterization matrix covers:

- uniform roles with the same outgoing dimension;
- uniform roles followed by a smaller outgoing dimension;
- mixed `128/64/64` roles; and
- mixed `128/64/32` roles.

Until the contribution-level differential tests and performance baseline are complete,
this point representation is test-only scaffolding. Making an unused production state or
routing the existing uniform evaluator through it would create a third implementation or
prematurely perturb the PR #312 hot path.

The current head does not yet meet that contract: `RelationMatrixEvaluator::eval_flat_at_point`
owns the uniform formula while `mixed_relation::evaluate_mixed_relation_at_point` owns a
second E/T/Z/setup/R formula and is also selected for uniform current roles followed by a
different outgoing dimension. The verifier-stabilization slice must consolidate those
formula owners before further prover work. Dense relation weights remain test oracles,
not a production verifier fallback.

Malformed dimensions, group/chunk layouts, claim offsets, point lengths, degrees, round
counts, and proof-derived allocations return `AkitaError`. Verifier-reachable code adds no
panic, assertion, unwrap, unchecked indexing, or unbounded allocation.

## Current setup-contribution boundary

This PR does not move the current setup-contribution proof. It may refactor reusable
geometry so the existing prover/verifier:

- consume the same typed mixed-dimension relation point;
- use `WitnessLayout` unit ownership for multigroup/multichunk E, T, and replicated Z;
- use one semantic setup-weight builder and checked role-address helper;
- remove uniform-D assumptions only where dense cross-product oracles prove parity; and
- define `setup_contribution_evaluation` as the complete scalar, not a fragment that a
  later stage multiplies by common alpha again.

No new numeric-stage wrapper or architecture lands; the later offload PR will delete that
stage.

## Checkpoint and gated implementation order

The current head has portions of the old Steps 1–5, but their ownership and support
claims are not yet stable. Proceed in these reviewable slices:

1. Correct names and documentation without changing proof behavior. In particular,
   separate `common_relation_coeff_count` from
   `common_relation_witness_coeff_count`, and record actual rather than intended support.
2. Stabilize the verifier before touching the prover state machine: one semantic relation
   evaluator, one succinct evaluation-trace evaluator, local uniform specializations,
   malformed-input coverage, and PR #312 performance parity.
3. Move prover-only relation events, trace factors, and fold storage out of `akita-types`.
   Keep only checked protocol geometry and facts consumed by both sides in the common crate.
4. Freeze dense-oracle, transcript, proof-byte, and end-to-end baselines across uniform,
   outgoing-transition, mixed-role, group, and chunk fixtures. A parity test for a manually
   constructed mixed evaluator is not evidence of mixed proving support.
5. Only after Steps 1–4 pass review, implement and measure the prover consumers—structured
   main scan, opening-support side scan, and contraction-first—inside the
   compact-prefix/folded-suffix state-machine
   cutover. Keep the complete-stage winner for each actual geometry class and delete the
   losing candidate code and every selection switch before landing.
6. Implement every per-basis candidate, select on complete-stage/end-to-end measurements,
   and delete losers.
7. Adapt the current setup boundary only as required by mixed/group/chunk geometry.
8. Delete x/y, dense, sparse, two-round wrapper, old constructors, and duplicate
   serial/parallel implementations atomically.

Additive oracle/test scaffolding may precede cutover. An unused production prover,
compatibility wrapper, runtime switch, or second semantic implementation may not land.

## Module ownership and intended diff

```text
sumcheck/relation_range_image/
  mod.rs                         one proof lifecycle
  relation_weights.rs            common-alpha compilation and lane folds
  evaluation_trace.rs            prover contractions and folded factor state
  compact_prefix.rs              selected fused basis kernels
  folded_rounds.rs               canonical later rounds
  tests.rs
```

Checked common protocol geometry and address helpers live in the shared types layer.
Prover relation events, trace terms, storage, and round production are prover-facing.
Verifier relation/trace preparation and final-point evaluators are verifier-owned and
retain only succinct state. A simple verifier evaluator may be a function rather than a
state type; no type is introduced solely to mirror prover structure.

For the verifier relation matrix, PR #312 is also the source-level baseline. Its prepared
tensor/flat challenge evaluations, equality-window and affine-interval contractions,
setup-plan reuse, and quotient-tail contraction remain the optimized primitives. The
stabilized evaluator assembles the relation formula once and specializes those primitives
for uniform dimensions; it does not retain the current uniform/mixed pair of complete
formula implementations. Verifier changes relative to PR #312 require a specific semantic
justification—mixed dimensions, compact trace ownership, no-panic hardening, removal of
dead state, or a strictly local simplification. Equivalent rewrites and naming-only churn
are rejected even when their measured runtime is similar.

The final setup kernel is generic at `coeff_count`-ring granularity, not through dynamic
per-coefficient dispatch. Point preparation bulk-combines equality and lane-alpha weights;
the hot scan remains const-generic and contiguous. For the uniform case, all lane and role
ratios are one, preparation borrows the existing E/T/Z slices, and the current
`SetupContributionPlan` identity scan remains the executable specialization. A strategy
or geometry match occurs once outside hot loops; no trait object, enum match, closure call,
checked lane reconstruction, or role lookup occurs per coefficient or setup entry.

| Surface | Responsibility |
|---|---|
| `akita-types` | checked flat/group/chunk/mixed geometry and common protocol parameters |
| ring-switch finalization | prepare prover lane weights and trace factors from checked inputs |
| `akita-prover::sumcheck` | one fused prover and selected kernels |
| current setup contribution | reusable typed geometry only |
| verifier | minimal prepared relation and trace state; compact final-point checks |
| PCS tests/benches/profile | dense cross-product oracles, epochs, kernel selection |
| book/spec | direct Stage 2 architecture and supported layout matrix |

Proof containers, stage placement, planner topology, serialized round counts, compressed
commitments, and terminal proof logic are outside this diff.

## Tests

Round-by-round dense differential tests compare every coefficient after every challenge:

- factorized relation events versus full flat weights;
- structured trace factors/contractions versus exact flat trace weights;
- fused compact messages versus direct dense summation;
- folded state after each challenge;
- final prover and compact verifier evaluations against the same direct equation; and
- current setup contribution versus a direct flat dot product.

The landed evaluation-trace differential matrix crosses Lagrange/monomial bases, multiple
claim terms, split chunk segments, source and destination ring dimensions 2/4/8, nontrivial
output scaling, scalar sparse storage, mixed dense storage, and direct final-point MLE
evaluation. End-to-end fixtures additionally cover Ext4/EOR and the existing
multi-group/multi-chunk proof.

The required layout matrix includes:

- singleton/single-chunk;
- multigroup/single-chunk;
- singleton/multichunk;
- multigroup/multichunk;
- uniform `64/64/64` and `128/128/128`;
- mixed `128/64/64` and `128/64/32` crossed with all four group/chunk shapes;
- one and multiple claims per group;
- unequal group block counts and uneven chunk partitions;
- scalar and extension-ring inner traces;
- absent/present EOR **scaling** (trace itself remains mandatory);
- full, three-quarter, odd, and short live prefixes;
- LB2-LB6, serial and parallel, fp128 primary plus fp64/Ext2 and fp32/Ext4 smoke;
- alpha zero, one, and random; additive relation overlaps and role resets; and
- invalid group order, duplicate/missing unit, chunk count, block ownership, role
  dimensions, source split, claim offsets, point width, degree, and serialized length.

Retain and extend PR #309's `multi_group_multi_chunk_fold_round_trips` and PR #310's
distributed recursive fixture rather than cloning competing setup/prove helpers.

## Tracing and performance

Coarse spans cover plan construction, prover relation compilation, trace preparation,
compact prefix, witness materialization, folded rounds, verifier relation constraints,
verifier setup scan, quotient tail, verifier trace contraction, and folds. Spans record
claim/group/chunk and ring-dimension metadata. No span or event enters pair, class,
coefficient, lane, claim-segment, or Rayon-item loops.

Measure feature-pruned profile builds for every basis and layout cell above at one and
production thread counts. Report relation construction, trace preparation, compact
prefix, materialization, later rounds/folds, complete Stage 2, complete prover, allocations,
peak field elements, verifier time, proof bytes, and transcript events.

Before the production verifier cutover, add a dedicated relation-evaluator Criterion
benchmark. Run the identical benchmark-only harness on pinned baseline
`147720907cef7a2db50a864a3032a0ffcbdc8203` and the candidate with the same Rust toolchain,
target CPU, feature set, setup data, inputs, and fixed Rayon thread count. Measure one
thread and a representative parallel configuration separately. For each primary uniform,
lane-factored, and mixed cell, accept only when Criterion's 95% change-interval upper bound
is at most +5%; a lower bound above +5% rejects the candidate, and an interval crossing the
threshold is inconclusive and must be rerun with a longer measurement.

The minimum comparison triplet holds A dimension, rows, claims, groups, flat witness
length, setup coefficient footprint, and transcript inputs fixed:

| Cell | Role dimensions | Outgoing dimension | Purpose |
|---|---:|---:|---|
| U | `128/128/128` | 128 | current uniform PR #312 floor |
| L | `128/128/128` | 32 | isolates unavoidable lane/address work |
| M | `128/64/32` | 32 | isolates role heterogeneity at the same lane geometry |

Candidate/baseline time is capped at 1.05 in every cell. In addition, M/L is capped at
1.05 so mixed roles do not demonstrably cost more than the equivalent homogeneous
lane-factored case. L/U is reported separately and is not mislabeled as role-mixing cost.
Complete verification remains a secondary <=5% gate because whole-verifier timing can
hide a relation-kernel regression.

Kernel winners are selected by complete Stage 2 and end-to-end prover results. PR #312 is
the verifier performance floor: no primary verifier cell may regress by more than 5%, and
verifier work may not scale with prover relation-event count or physical trace support.
CI benchmarks enforce the contract but do not replace local feature-pruned release/profile
comparison before a slice lands. Production carries no
measured/unmeasured duplicate, ad hoc timer, or strategy knob.

## Risks and stop conditions

| Risk | Required prevention |
|---|---|
| group/chunk layout is recomputed in several crates | `WitnessLayout` is the semantic authority; compiled runs are derived and compared once |
| block equality resets at a chunk boundary | trace segments retain global block starts and uneven ranges |
| per-chunk Z or shared R is counted incorrectly | closed relation events distinguish replicated Z units from one R suffix |
| mixed-D multi-chunk silently uses the old guard/fallback | direct setup removes the guard only after dense-oracle parity; recursive setup remains an explicit rejection until Stage 3 consumes the checked joint relation/witness split rather than assuming `log2(d_a)` low coordinates |
| role-specific decomposition bases imply unsupported range claims | one explicit global range-basis authority is shared with security/sizing |
| trace is treated as optional | a nonterminal prover plan requires nonempty `EvaluationTraceWeights`, while the verifier requires nonempty `PreparedEvaluationTrace`; missing EOR scales become ones |
| nominal fusion still reads the full witness twice | relation and range share one block reducer; trace side scan must have strictly smaller support and measured benefit |
| terms choose separate witness deferral depths | one joint per-basis materialization boundary; analytical term shortcuts share it |
| generic abstraction hides degree/batching factors | three named subtotals and equation-owning prover; no expression engine |
| microkernel win loses end to end | include construction, allocation, parallelism, and complete stage in selection |
| “single source” forces prover storage into verifier replay | share checked protocol inputs, then derive separate minimal consumers; PR #312 verifier performance is a blocking floor |
| a generic fallback is relabeled a verifier fast path | one compact production verifier implementation per concept; no slow prover-event/table path remains reachable for supported verifier shapes |
| verifier restoration becomes a large equivalent rewrite | preserve PR #312's source structure and require a semantic justification for every changed hot-path region |

## Future recursive-offload seam

The later protocol-changing PR will have its own spec. The stable handoff from this PR is:

- evaluation trace is part of the linear relation and moves with it into the final Stage 1
  leaf;
- relation and the final range leaf share at least the first compact witness traversal;
- offloaded Stage 2 contains setup contribution plus independent
  range-image/witness reduction, not evaluation trace; and
- direct/non-offloaded folds retain the architecture in this document.

No future proof field, transcript challenge, stage enum, or inactive branch is added here.

## Acceptance criteria

- One checked common-parameter authority feeds one prover implementation and one compact
  verifier implementation; their runtime representations are deliberately separate and
  minimal for their cost models.
- Homogeneous direct proof bytes, transcript order, challenges, degree, final point, and
  final evaluation match the incoming epoch.
- Common alpha coordinates bind first and relation lane state is at most
  `N / common_relation_witness_coeff_count`.
- No `ring_bits == 0` sentinel, dense mixed relation table, exact fringe, full trace table,
  trace remap, hot unit search, or x/y architecture remains.
- Multigroup, multichunk, and their mixed-D cross-product prove and verify for every
  scheduled supported shape.
- Global block ownership, per-chunk Z, per-unit E/T, and shared R agree round by round with
  dense oracles.
- Evaluation trace is mandatory and exact for scalar/extension, root/recursive,
  multigroup/multichunk, and EOR-scaled claims.
- Every LB2-LB6 candidate is measured; one complete-stage winner remains per basis.
- No primary Stage 2/prover benchmark cell regresses beyond measurement noise; targeted
  prover cells show material wins. No primary verifier cell exceeds 1.05x its pinned PR
  #312 baseline under the Criterion confidence-interval gate, mixed `128/64/32` does not
  exceed 1.05x lane-equivalent homogeneous `128/128/128`, and verifier relation/trace work
  never scales with prover event count or physical trace-table size.
- Numeric setup code touched by the PR contains only reusable final geometry.
- Documentation guardrails and all repository-required format, lint, and test commands
  pass at the final head.
