# Relation, evaluation-trace, and range-image sum-check

| Field | Value |
|---|---|
| Author(s) | Quang Dao (protocol and implementation direction); Codex (design synthesis) |
| Created | 2026-07-20 |
| Status | active |
| Branch | `quang/relation-range-image-rewrite` |
| Base | PR #312 at `8a4ec9b140e23514b8e53e61b885774f8d8397d7` |
| Integration dependencies | PR #309 at `b0c2d4683539b0c2a465b996f48adfc465a20198`; PR #310 at `4cb4113b02a58889230f3dbaa81deb56895bb4ca` as cross-feature evidence |
| Related | [`digit-range-pipeline-refactor.md`](digit-range-pipeline-refactor.md), [`digit-innermost-layout.md`](digit-innermost-layout.md), [`runtime-ring-cutover.md`](runtime-ring-cutover.md), [`packed-sumcheck.md`](packed-sumcheck.md) |

## Summary

This PR reimplements the current direct/non-offloaded fused sum-check over the digit
witness. It keeps the proof statement and wire unchanged while replacing the prover's
x/y/layout branch matrix with one flat-address state machine and optimized LB2-LB6
kernels. It also makes mixed ring dimensions, multiple commitment groups, and
multi-chunk witness layouts first-class supported combinations.

The sum-check has three semantic terms:

1. the factorized linear relation;
2. the mandatory evaluation trace that binds opening claims to the committed witness; and
3. range-image consistency with the independent Stage 1 evaluation.

All three share one transcript lifecycle, one checked witness domain, and one round-result
reducer. They keep distinct named arithmetic; this is not a generic expression engine.

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
- Exact common-alpha factorization with `g = min(d_a, d_b, d_d)`.
- Alpha/common-coordinate rounds before relation-lane rounds.
- Mandatory evaluation trace represented by per-claim tensor factors, never a flat table.
- One fused relation/evaluation-trace/range-image round reducer.
- Measured one/two/three-round compact strategies for every LB2-LB6 basis.
- Full multi-group, multi-chunk, and mixed-dimension cross-product support.
- One semantic relation-event authority and one semantic evaluation-trace authority used
  by prover preparation, verifier replay, setup attribution, and dense test oracles.
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
| common `[1, alpha, ..., alpha^(g-1)]` factor | `common_alpha_factor` |
| high-lane linear-relation factor | `relation_lane_weights` |
| mandatory opening-trace factors | `EvaluationTraceWeights` |
| prepared trace contraction/fold state | `EvaluationTraceProverState` |
| complete checked direct plan | `RelationRangeImagePlan` |
| equation-owning prover | `RelationRangeImageProver` |
| proof object | `RelationRangeImageProof` |

`W` is acceptable only in short equations. Ambiguous `S` names are forbidden. Production
code does not use `TraceWeightState`, `ExactFringeWeight`, x/y-prefix names, or numeric
stage names for new internal objects.

## One flat address order

All rounds bind the raw physical field-coefficient address LSB first. For nested role
dimensions

```text
g = min(d_a, d_b, d_d),
k = log2(g),
z = g * lane + coefficient,  0 <= coefficient < g.
```

The first `k` challenges are common coefficient coordinates. The rest address physical
relation lanes and padded witness capacity. x/y is not a public protocol abstraction.

`WitnessDomain` checks the live prefix, padded power-of-two domain, point width, and
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

For each role dimension `d_role = g * q`, write

```text
alpha^(g * high_exponent + coefficient)
  = alpha^coefficient * (alpha^g)^high_exponent.
```

The complete non-trace relation weight is

```text
RelationWeight(g * lane + coefficient)
  = CommonAlphaFactor(coefficient) * RelationLaneWeights(lane).
```

Role resets, quotient denominators, row challenges, claim coefficients, group weights,
setup amplitudes, and overlaps are additive contributions to high-lane weights. They do
not break the common low factor. Mixed `128/64/32` therefore uses a common factor of
length 32, not 128 and not a full-domain table.

One closed semantic emitter produces checked relation events:

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
does not discard and regenerate the alpha factor from the challenge. Until the atomic
mixed-dimension prover cutover, only the uniform path consumes this representation and
the pre-existing mixed path remains a dense transitional fallback.

Every production event has `g`-aligned physical and exponent starts and preserves the
low common coordinate. Role-local resets begin new aligned events. Overlaps use `+=`
exactly once. Unsupported unaligned events return `AkitaError`; there is no production
fringe or dense fallback. A dense exact-live vector exists only as a differential oracle.

`contribution` distinguishes protocol-constraint arithmetic from setup-matrix arithmetic,
so direct setup evaluation and a deferred complete setup claim cannot be mixed. All emitted
exponents are consecutive; an exponent-pattern enum would add no information. The temporary
dense and uniform-column consumers keep the current Stage 2 prover behavior while the fused
prover is cut over. They consume these events and contain no second relation formula.

The emitter explicitly covers E/D consistency, T/B, every per-chunk Z response, the
shared quotient-R suffix, setup contributions, and row-family resets. It is the single
source for production lane compilation, verifier evaluation, setup attribution, and the
dense oracle. Source events are dropped after coalescing; no provider retains dense and
factorized copies.

## Binding order and relation arithmetic

The `k` common alpha coordinates bind first. Relation lane weights are constant over each
low-coordinate block and are not folded during these rounds. For

```text
w(T) = w_0 + T * delta_w,
a(T) = a_0 + T * delta_a,
```

the local relation polynomial is `w(T) * a(T)`. Accumulate its coefficients over a
lane-aligned block, then multiply the totals by `relation_lane_weights[lane]` once. Do
not multiply every endpoint separately.

After challenge `k-1`, retain

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

The normalized semantic input is a nonempty `EvaluationTraceWeights` containing one
`EvaluationTraceTerm` per opening claim. There are no `Field`, `Ring`, `Root`,
`Recursive`, `Absent`, sparse-table, or dense-table semantic variants.

The landed term representation owns exactly:

- one normalized claim coefficient;
- the block-opening point and basis;
- the group's exact live-block count and source ring dimension;
- opening-digit gadget weights and one precomputed inner-trace row; and
- one `EvaluationTraceSegment` per witness chunk, recording the flat physical coefficient
  start, group-global block start, and exact live-block count.

The verifier evaluates these segments directly at the final flat point with the existing
carry-aware affine-interval equality primitive. The current Stage 2 prover temporarily
materializes this same semantic term list into its pre-existing foldable `TraceTable`
storage. Scalar same-dimension claims retain sparse columns; extension or mixed-dimension
claims write one exact-live flat dense table. This is transitional storage, not a second
semantic representation: no source/destination remap or alternate trace formula remains in
production. The table disappears with the fused prover cutover in Step 6. The deleted trace
implementations are compiled only under `cfg(test)` as differential oracles.

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
| mixed role dimensions | none | common factor uses global `g`; role subcolumns map into common lanes | each claim retains its own source ring split | remove the existing mixed-D/multichunk guard only after flat-oracle parity |
| EOR | none | no change | per-claim scale multiplies the same trace term | normalize missing scales to one; never make trace optional |

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
  aligned to a physical ring coefficient boundary and therefore to `g`.
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

The verifier evaluates each segment directly at the flat point with the canonical
carry-aware offset/interval equality primitive. An arbitrary physical E interval is not
assumed to be a Boolean subcube. Complexity is proportional to claims and prepared
segments, not the padded witness domain.

### Mixed ring dimensions with chunks

The current verifier explicitly rejects multi-chunk layouts when B/D role dimensions
differ from the A dimension. This PR removes that guard only after the relation emitter,
trace segments, setup attribution, and verifier evaluator agree with dense flat oracles.

The final rule is unchanged by chunks:

```text
g = min(d_a, d_b, d_d).
```

Role subcolumns map into `g`-sized common lanes inside each unit. Chunking changes which
unit owns a block, not alpha exponents or source ring-coordinate order. Evaluation-trace
terms retain their own `D_source` split

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
compact common-coordinate prefix          rounds 0 .. r
field common-coordinate suffix             rounds r .. k
field relation-lane suffix                 rounds k .. num_variables
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

It accepts one claim list for root, recursive, multigroup, multichunk, and EOR cases. It
does not allocate dense tables, remap points, or dispatch by historical representation.
Typed point views validate common, lane, and per-trace source splits.

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

## Implementation order

1. Add/extend dense test oracles and freeze homogeneous/multigroup/multichunk/mixed-D
   protocol epochs at the actual integration base.
2. Make `WitnessDomain`, `WitnessLayout`, global claim order, and range-basis authority the
   checked inputs to one `RelationRangeImagePlan`.
3. Land the semantic relation emitter and compare compiled lane weights/final evaluation
   against dense weights for every layout cross-product. **Landed:** the checked event
   authority, direct/deferred setup split, canonical verifier point evaluator, checked
   common-alpha compiler, uniform Stage-2 handoff, and mixed-dimension reconstruction tests
   now share one source. The dense mixed consumer remains only until Step 6 cuts over the
   fused prover.
4. Land `EvaluationTraceWeights` and its verifier evaluator; replace root/multigroup/EOR
   variants and dense/sparse semantic tables in one cutover. **Landed:** prover and verifier
   build the same per-claim terms from `RelationRangeImagePlan`; fully scaled coefficients
   are normalized at preparation; chunk segments retain group-global block indices; the
   verifier evaluates the flat point directly; mixed source/destination dimensions use flat
   physical addresses with no remap; and all former production trace variants/builders are
   test-only differential oracles. The existing prover `TraceTable` remains solely as the
   temporary fold storage consumed by the old Stage 2 state machine until Step 6.
5. Implement and measure structured, opening-only, and contraction-first trace
   preparation.
6. Cut the prover to one compact-prefix/folded-suffix state machine, initially with the
   simplest correct fused pair kernel.
7. Implement every per-basis candidate, select on complete-stage/end-to-end measurements,
   and delete losers.
8. Adapt the current setup boundary only as required by mixed/group/chunk geometry.
9. Delete x/y, dense, sparse, two-round wrapper, old constructors, and duplicate
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

Checked relation events, `EvaluationTraceTerm` geometry, witness/group/chunk address
helpers, and final point evaluators live in the shared types layer used by prover and
verifier. The prover module owns storage and round production only.

| Surface | Responsibility |
|---|---|
| `akita-types` | checked flat/group/chunk/mixed geometry, semantic events/trace terms, final evaluators |
| ring-switch finalization | prepare one plan, lane weights, and trace factors |
| `akita-prover::sumcheck` | one fused prover and selected kernels |
| current setup contribution | reusable typed geometry only |
| verifier | canonical factorized relation and trace final checks |
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
- final prover and canonical verifier evaluations; and
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

Coarse spans cover plan construction, relation compilation, trace factor/contraction
preparation, compact prefix, witness materialization, folded rounds, and folds. The landed
trace cutover records one `stage2_evaluation_trace_preparation` span on each side, including
claim/group/chunk and ring-dimension metadata; verifier final evaluation remains under
`stage2_trace_oracle`. No span or event enters pair, class, coefficient, lane, claim-segment,
or Rayon-item loops.

Measure feature-pruned profile builds for every basis and layout cell above at one and
production thread counts. Report relation construction, trace preparation, compact
prefix, materialization, later rounds/folds, complete Stage 2, complete prover, allocations,
peak field elements, verifier time, proof bytes, and transcript events.

Kernel winners are selected by complete Stage 2 and end-to-end prover results. CI
benchmarks catch later regressions. Production carries no measured/unmeasured duplicate,
ad hoc timer, or strategy knob.

## Risks and stop conditions

| Risk | Required prevention |
|---|---|
| group/chunk layout is recomputed in several crates | `WitnessLayout` is the semantic authority; compiled runs are derived and compared once |
| block equality resets at a chunk boundary | trace segments retain global block starts and uneven ranges |
| per-chunk Z or shared R is counted incorrectly | closed relation events distinguish replicated Z units from one R suffix |
| mixed-D multi-chunk silently uses the old guard/fallback | remove the guard only after every cross-product dense oracle passes |
| role-specific decomposition bases imply unsupported range claims | one explicit global range-basis authority is shared with security/sizing |
| trace is treated as optional | nonterminal plan requires nonempty `EvaluationTraceWeights`; missing EOR scales become ones |
| nominal fusion still reads the full witness twice | relation and range share one block reducer; trace side scan must have strictly smaller support and measured benefit |
| terms choose separate witness deferral depths | one joint per-basis materialization boundary; analytical term shortcuts share it |
| generic abstraction hides degree/batching factors | three named subtotals and equation-owning prover; no expression engine |
| microkernel win loses end to end | include construction, allocation, parallelism, and complete stage in selection |

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

- One checked plan, one relation emitter, one evaluation-trace semantic form, one prover,
  and one verifier evaluator remain.
- Homogeneous direct proof bytes, transcript order, challenges, degree, final point, and
  final evaluation match the incoming epoch.
- Common alpha coordinates bind first and relation lane state is at most `N/g`.
- No `ring_bits == 0` sentinel, dense mixed relation table, exact fringe, full trace table,
  trace remap, hot unit search, or x/y architecture remains.
- Multigroup, multichunk, and their mixed-D cross-product prove and verify for every
  scheduled supported shape.
- Global block ownership, per-chunk Z, per-unit E/T, and shared R agree round by round with
  dense oracles.
- Evaluation trace is mandatory and exact for scalar/extension, root/recursive,
  multigroup/multichunk, and EOR-scaled claims.
- Every LB2-LB6 candidate is measured; one complete-stage winner remains per basis.
- No primary Stage 2/prover/verifier benchmark cell regresses beyond measurement noise;
  targeted cells show material wins.
- Numeric setup code touched by the PR contains only reusable final geometry.
- Documentation guardrails and all repository-required format, lint, and test commands
  pass at the final head.
