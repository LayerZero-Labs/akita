# Digit-range and relation sum-check pipeline

| Field | Value |
|---|---|
| Author(s) | Quang Dao (protocol and implementation direction); Codex (design synthesis) |
| Created | 2026-07-18 |
| Revised | 2026-07-20; scope reduced to a three-PR implementation stack and Stage 2 redesigned around common-dimension alpha-first binding |
| Status | active |
| PR | [#312](https://github.com/LayerZero-Labs/akita/pull/312) implements the specification and complete Stage 1 cutover |
| PR #312 branch | `quang/plan-digit-range-pipeline` |
| PR #312 base | PR #311, `quang/terminal-direct-ring-relations` at `fad006e2280e880fa16f1cd13b5ea2df599364d0` |
| PR #312 head at this revision | `f5520cd8ca6fe314b5f14faef4300807e6c43ac8` before the documentation/CI-fix commit |
| Related | [`digit-innermost-layout.md`](digit-innermost-layout.md), [`runtime-ring-cutover.md`](runtime-ring-cutover.md), [`packed-sumcheck.md`](packed-sumcheck.md), [`akita-sumcheck-unification.md`](akita-sumcheck-unification.md) |

## Decision summary

This document is the central design record for three PRs. The boundaries are semantic,
not chronological implementation packets:

| PR | Delivers | Deliberately does not deliver |
|---|---|---|
| **#312: digit-range cutover** | This specification; one Stage 1 range prover; streaming range-product storage; selected LB2-LB6 initial-round kernels; descriptive range-image names; proof/transcript parity | Stage 2 rewrite, mixed-dimension relation execution, relation stage movement, setup-offload protocol changes |
| **Stacked PR: relation/range-image prover** | Reimplement the current fused Stage 2; preserve its proof statement and wire; bind the common alpha coordinates first; retain and extend compact initial-round deferral; support mixed ring dimensions; adapt current setup-contribution code only where mixed dimensions require it | Move the relation to Stage 1, change proof size, remove the setup-contribution stage, or add compressed commitments |
| **Stacked PR: two-stage offloading cutover** | In recursive setup-offload mode, move the relation into the final Stage 1 range subcheck; move setup contribution and witness/range-image reduction into Stage 2; remove numeric Stage 3; update proof shape, transcript, sizing, planner, prover, and verifier atomically | Change the direct non-offloaded placement; implement compressed commitments or the fused negative-binary range check |

This replaces the previous many-packet plan. There is no longer a separate PR for each
Stage 1 kernel experiment, provider type, proof container, or cleanup pass. #312 owns the
entire Stage 1 implementation. The next PR owns the entire behavior-preserving Stage 2
rewrite and mixed-dimension execution. The final PR owns the protocol-changing offload
placement.

The discipline remains additive before cutover and atomic at cutover:

- test oracles, benchmarks, and directly used arithmetic primitives may land before a
  replacement becomes canonical;
- an unused production prover, compatibility wrapper, runtime feature switch, or second
  semantic implementation may not land;
- a cutover PR changes all callers and deletes the superseded implementation in the same
  diff;
- an intentional protocol change updates its versioned proof/transcript oracle in the
  same diff. The existing oracles are not immutable across declared protocol epochs.

## The protocol after the three PRs

Direct setup evaluation retains the prover-efficient placement:

```text
Stage 1
  optimized equality-factored range-product tree
  -> range_image_eval at range_check_point

Stage 2
  one standard relation + range-image-consistency sum-check
  -> next_witness_eval
```

Recursive setup offloading uses the paper-motivated placement:

```text
Stage 1
  same optimized equality-factored range-product prefix
  final range leaf + complete linear relation in one standard sum-check
  -> range_image_eval and digit_witness_eval at range_relation_point

Stage 2
  setup contribution + range-image/witness reduction
  -> setup_prefix_eval and next_witness_eval
```

The terminal fold remains the quotient-free, sum-check-free path from #311. It has no
Stage 1 range proof, relation sum-check, Stage 2 reduction, or outgoing witness binding.
Nothing in this series reintroduces those objects at the terminal.

## Vocabulary and naming contract

Code names must describe the mathematical object or lifecycle phase. Numeric stage names
may remain only where a public proof epoch still requires them during the second PR; the
final protocol cutover removes them.

| Mathematical shorthand | Production name | Meaning |
|---|---|---|
| `W` | `digit_witness` | balanced signed-digit multilinear table |
| `W(W+1)` | `range_image` | pointwise image of a Boolean-vertex digit |
| `Q_b` | `range_image_polynomial` | vanishing polynomial over valid range-image values |
| `A_g` | `common_alpha_factor` | `[1, alpha, ..., alpha^(g-1)]` over the common coefficient dimension |
| `M` | `relation_lane_weights` | high-lane relation weights after the common alpha factor is removed |
| `T` | `trace_weight` | additive trace polynomial, not forced into alpha factorization |
| `S`, `s_table`, `s_claim` | forbidden | ambiguous legacy names for the range image or setup data |

Use `digit_witness` in fields and APIs. `W` is acceptable only in displayed equations or
a very short local derivation. Likewise, equations may use `M`, but production fields and
functions use `relation_lane_weights`, `folded_relation_lane_weights`, or another equally
descriptive name.

The following target names are normative:

| Responsibility | Name |
|---|---|
| checked flat Boolean witness address space | `FlatBooleanDomain` |
| Stage 1 topology, roots, degrees, and child order | `DigitRangePlan` |
| compact signed digits and range-image class access | `CompactDigitSource` |
| direct Stage 1 output point | `RangeCheckPoint` |
| recursive joint Stage 1 output point | `RangeRelationPoint` |
| current direct fused proof after the Stage 2 rewrite | `RelationRangeImageProof` |
| current direct fused prover after the Stage 2 rewrite | `RelationRangeImageProver` |
| common-dimension relation factorization | `CommonAlphaRelationWeights` |
| independently additive trace representation | `TraceWeightState` |
| setup coefficient address space | `SetupCoefficientDomain` |
| complete setup contribution at one relation point | `setup_contribution_eval` |

Do not add aliases from old names, `_for_level` forwarding helpers, `Engine` facades,
generic expression graphs, or one-line wrappers around the canonical functions. A module
boundary is justified only by a distinct invariant, state representation, or substantial
kernel.

## Global invariants

### One Boolean address order

All witness sum-checks bind the raw physical field-coefficient address LSB first. A
homogeneous ring can be viewed internally as coefficient bits followed by column bits,
but x/y is not a public protocol abstraction. Mixed dimensions use the same flat address
order.

For the role dimensions `d_a`, `d_b`, and `d_d`, all supported tuples are nested powers
of two. Define

```text
g = gcd(d_a, d_b, d_d) = min(d_a, d_b, d_d)
k = log2(g)
z = g * lane + coefficient,  0 <= coefficient < g.
```

The first `k` bound variables are therefore the common coefficient coordinates. The
remaining variables address relation lanes and padded witness capacity. `g` is derived
from authenticated role dimensions; it is not a separate proof field or a sentinel.

### Range-image evaluations are independent MLE claims

At Boolean addresses,

```text
range_image[z] = digit_witness[z] * (digit_witness[z] + 1).
```

After folding, in general,

```text
range_image_eval != digit_witness_eval * (digit_witness_eval + 1).
```

Any prover state that has crossed a challenge must therefore fold the range-image and
digit-witness tables independently. Recomputing one from the other after a challenge is
incorrect.

### Honest-prover digit ownership

The ring-switch decomposition is the single authority for the balanced-digit invariant.
Stage 1 and Stage 2 are honest-prover internals and must not rescan the witness merely to
validate digits a second time. Checked constructors validate sizes, domains, basis, and
layout. Hot compact access uses the documented producer invariant and debug assertions.

### Exact prefixes and padding

The wire witness is zero outside its live prefix. Derived tables need not have zero
defaults. A quartic range leaf, a randomized leaf batch, or an intermediate product can
be nonzero at `range_image = 0`.

Every truncated derived table therefore carries:

```text
explicit rows + one exact default row + exact omitted equality mass.
```

No kernel may skip the padded suffix merely because the original digit witness is zero.
The canonical exact-prefix and equality-suffix functions are shared mechanics; the
semantic caller supplies the correct default contribution.

### Proof shape is not a kernel selector

Basis-specific tables, initial-round deferral depth, cache layout, serial/parallel mode,
and delayed reduction are local prover decisions. They never appear in schedules, proof
metadata, serialization, or verifier dispatch. Each supported basis has exactly one
selected production kernel at a given revision.

## PR #312: complete Stage 1 digit-range cutover

### Scope lock

#312 is complete when it contains:

- this central specification;
- one `DigitRangePlan` authority for LB2 through LB6;
- one `DigitRangeProver` and one verifier path;
- the selected high-basis streaming product kernels;
- the selected low-basis three-round deferral kernels;
- proof/transcript epoch tests, malformed-shape tests, benchmarks, allocation measurement,
  and tracing spans;
- deletion of the eager range forest, padded field-valued range-image table, duplicate
  Stage 1 prover, and layout-named Stage 1 modules.

#312 must not redesign the current Stage 2. The tiny Stage 2 edits already in its diff are
name/API fallout from the Stage 1 `range_image_evaluation` cutover and protocol-oracle
maintenance, not a Stage 2 optimization claim.

### Range polynomial and tree topology

Balanced basis `b = 2^LB` uses digits in `[-b/2, b/2 - 1]`. The class

```text
range_image_class(w) = w       when w >= 0
                     = -w - 1  when w < 0
```

satisfies

```text
range_image(w)
  = range_image_class(w) * (range_image_class(w) + 1),
0 <= range_image_class(w) < b/2.
```

The selected topology is:

| LB | Basis | Product substages | Final leaf |
|---:|---:|---|---|
| 2 | 4 | none | quadratic |
| 3 | 8 | none | quartic |
| 4 | 16 | binary root | two quartic leaves |
| 5 | 32 | arity-4 root | four quartic leaves |
| 6 | 64 | binary root, then arity-4 layer | eight quartic leaves |

This topology is fixed by `DigitRangePlan`. Prover, verifier, shape validation, child
ordering, degree enforcement, serialization sizing, and tests call that same authority.
No consumer reconstructs it from `trailing_zeros`, basis thresholds, or received vector
lengths.

Binary-only trees are rejected. They conserve the total per-round coefficient count while
adding substages, child claims, transcript challenges, scans, and state transitions. They
are neither simpler at the protocol boundary nor smaller on the wire.

### Canonical Stage 1 lifecycle

Every product substage follows one state machine:

1. Build the small class-indexed node rows required by the current substage.
2. Build ordered class-pair round coefficients after interstage batching weights are
   known.
3. Scan compact pair indices to produce the first ordinary round message.
4. Keep the witness compact for the selected number of initial challenges.
5. Materialize only the current substage's folded 1/2/4/8-lane field state.
6. Prove later rounds with direct fixed-degree arithmetic and in-place folds.
7. Read child claims in plan order, absorb them, and free the substage state.
8. Rescan the original compact source for the next product layer or range leaf.

The prover never materializes all range leaves or product-tree levels. Address-major
fixed-lane state has logical shape

```text
folded_address -> [lane_0, ..., lane_(LANES-1)].
```

Small explicit dispatch chooses lane width and quadratic versus quartic arithmetic. There
is no module or trait family per log basis.

### Selected high-basis kernels

| Topology | Initial compact strategy | First field state |
|---|---|---:|
| LB4 two-lane product | two-round challenge-dependent 4,096-key quartet coefficients | two lanes at `N/4` |
| LB4 scalar leaf | two-round challenge-dependent 4,096-key quartet coefficients | one lane at `N/4` |
| LB5 four-lane product | two-round factorized folded-pair rescan | four lanes at `N/4` |
| LB5 scalar leaf | optimized one-round ordered-pair scan | one lane at `N/2` |
| LB6 two-lane product | two-round factorized folded-pair rescan | two lanes at `N/4` |
| LB6 eight-lane product | two-round factorized folded-pair rescan | eight lanes at `N/4` |
| LB6 scalar leaf | optimized one-round ordered-pair scan | one lane at `N/2` |

The common one-round machinery includes:

- compact `u16` ordered class-pair indices;
- split-equality block accumulation, applying the outer equality weight once per block;
- contract-gated delayed product reduction;
- challenge-dependent folded-pair materialization;
- direct quadratic and quartic affine-product formulas;
- exact nonzero suffix accounting.

Two-round deferral preserves the ordinary transcript sequence. It sends round zero,
receives `r_0`, sends round one, receives `r_1`, and only then materializes `N/4` state.
It is not a multi-round Fiat-Shamir message.

LB4 uses the affordable `8^4 = 4,096` challenge-dependent quartet key space. LB5 and LB6
do not use full four-class tables: `16^4 = 65,536` and `32^4 = 1,048,576` are poor cache
and construction tradeoffs. Their product layers rescan compact pairs through a
`classes^2 * lanes` folded-pair table instead.

### Selected low-basis kernels

LB2 and LB3 use the same Stage 1 architecture but keep the compact source through three
ordinary challenges whenever at least three variables remain.

| Basis | Third-round compact representation | Challenge-dependent cache | First field state |
|---:|---|---|---:|
| LB2 / 4 | 256 range-image octet classes, direct quadratic coefficients | `256 x 3` field elements | one lane at `N/8` |
| LB3 / 8 | two 256-class folded quads per octet, direct quartic coefficients | 256 folded values and `256 x 4` Taylor rows | one lane at `N/8` |

The LB3 Taylor row is

```text
[Q(a), Q'(a), Q''(a)/2, Q'''(a)/6]
for Q(s) = s(s-2)(s-6)(s-12).
```

This is a challenge-time cache of range-polynomial arithmetic, not a second witness.

### Measurements and settled Stage 1 experiments

The selected high-basis path was measured on an Apple M4 with feature-pruned fp128
release builds at `2^18`, full and three-quarter live prefixes, and uniform, zero-heavy,
and alternating-endpoint digits.

- Ordered class pairs beat streaming node evaluation in all 18 LB4-LB6 cells, with a
  25.78% median improvement.
- The optimized one-round pipeline then beat the ordered-pair baseline in all 18 cells,
  with a 30.50% median improvement.
- The selected two-round policy improved all 18 point estimates over that optimized
  one-round baseline. The approximate geometric-mean improvements were 12.2% for LB4,
  18% for LB5 after excluding one noisy cell, and 19.9% for LB6.
- LB2 three-round deferral improved the measured full/uniform cell by about 39.8% and the
  three-quarter/uniform cell by about 30.8% before the later shared quadratic cleanup.
- LB3 three-round deferral improved one-thread full/uniform by 12.8% and
  three-quarter/uniform by 10.9%; the corresponding stable eight-thread improvements were
  9.3% and 6.1%.

These results select production owners. The following losing alternatives are deleted,
not hidden behind selectors:

- streaming node evaluation inside the address scan;
- full bivariate LB4 tables;
- full four-class LB5/LB6 lookup tables;
- two-round high-basis scalar-leaf deferral for LB5/LB6;
- caching LB2 octet identifiers or globally histogramming LB2 octets;
- a full `65,536 x 5` LB3 octet-pair coefficient table;
- adaptive low-diversity LB3 aggregation;
- packed LB3 quad identifiers, which helped one thread but regressed the production
  parallel path;
- fusing post-materialization state writes with the next round computation. The faithful
  candidate retained existing accumulation optimizations and still measured slightly
  slower or tied.

### Stage 1 module ownership

The final implementation is organized by invariant:

```text
digit_range/
  mod.rs                         topology choreography
  compact_digit_source.rs        compact digits and range-image classes
  range_class_tables.rs          class rows and ordered-pair coefficients
  class_indexed_state.rs         product/leaf state transition data
  class_indexed_product.rs       fixed-lane product subchecks
  class_indexed_range_leaf.rs    high-basis equality-factored leaf
  exact_prefix.rs                explicit prefix plus exact default
  round_accumulation.rs          bounded coefficient accumulation
  direct_range_leaf.rs           low-basis equality-factored leaf
  direct_range_leaf/
    initial_round_deferral.rs    selected compact prefix kernels
    live_prefix.rs               prefix-aware scans
    rounds.rs                    later field rounds
    sparse_low_variables.rs      small remaining-variable cases
```

Files are allowed to be substantial when they own a real kernel. Splitting a hot function
into forwarding helpers merely to satisfy a line target is forbidden. Conversely, no
file should mix topology, compact conversion, proof choreography, table construction,
and multiple unrelated kernels as the deleted Stage 1 modules did.

### #312 proof and verifier contract

#312 is compute-only with respect to Stage 1 protocol semantics:

- proof bytes and transcript events match the versioned post-#311 epoch;
- child claim order and degree are unchanged;
- `range_image_eval` and its point are unchanged;
- the verifier consumes `DigitRangePlan` and rejects malformed shape before allocation;
- no verifier-reachable panic, unchecked indexing, or unbounded received vector remains;
- terminal #311 bytes and events are unchanged.

The epoch fixtures are versioned. A later protocol-changing PR replaces the affected
expected digests with an explicit before/after delta; it does not pretend #312's fixtures
must remain immutable forever.

### #312 intended diff surface

The full merge-base diff may touch only these responsibilities:

| Surface | Allowed #312 work |
|---|---|
| `akita-prover::protocol::sumcheck` | Stage 1 range cutover and shared mechanics required directly by it |
| `akita-types::proof::stage1` and sizing | canonical range topology, descriptive fields, shape validation, byte accounting |
| `akita-verifier::stages::stage1` | replay through `DigitRangePlan`, descriptive outputs, malformed-shape rejection |
| Stage 2 call boundary | mechanical `range_image_evaluation` naming and Stage 1 output adaptation only |
| PCS tests/benches/profile report | epoch fixtures, differential tests, allocation measurement, basis benchmarks |
| transcript labels/book/specs | semantic range-image names and documentation |

Not allowed in #312: mixed-dimension provider construction, Stage 2 kernel selection,
relation point remapping, setup-offload proof changes, planner schedule changes, or a new
proof epoch.

## Stacked PR: reimplement the fused relation/range-image prover

### Purpose and protocol boundary

The second PR comprehensively replaces the current Stage 2 implementation while proving
the same statement, sending the same standard degree-3 messages, sampling the same number
of challenges, and returning the same `next_witness_eval`.

For the direct/non-offloaded path, the claim is

```text
relation_claim
  + range_binding_challenge * range_image_eval
  + trace_claim

= sum_z digit_witness(z) * [
      common_alpha_factor(coefficient(z))
        * relation_lane_weights(lane(z))
    + trace_weight(z)
    + range_binding_challenge
        * Eq(range_check_point, z)
        * (digit_witness(z) + 1)
  ].
```

Equivalently, the range term is

```text
range_binding_challenge
  * Eq(range_check_point,z)
  * digit_witness(z)
  * (digit_witness(z)+1).
```

The range, relation, and trace coefficient accumulators share one witness traversal. The
semantic terms remain explicit; there is no general sum-of-products framework.

### Current problems to delete

The current implementation is organized around storage/layout accidents:

- `y_prefix`, `x_prefix`, and dense variants;
- compact versus field copies of the same coefficient algebra;
- full versus Gruen-recovered range coefficients;
- serial and parallel copies;
- two-round prefix code shared through stage-specific wrappers;
- a `ring_bits == 0` sentinel that throws mixed dimensions into a dense full-domain
  relation table;
- trace folding interleaved into every layout branch;
- public constructor arguments `live_x_cols`, `col_bits`, and `ring_bits` that allow
  inconsistent geometry.

The rewrite deletes those axes. The only meaningful lifecycle split is:

```text
compact common-coefficient prefix
  -> challenge boundary
folded remaining-address suffix.
```

### Exact common-dimension factorization

For a role dimension `d_role = g * q`, every role-local alpha exponent can be written

```text
exponent = g * high_exponent + coefficient.
```

Therefore

```text
alpha^exponent
  = alpha^coefficient * (alpha^g)^high_exponent.
```

All non-trace linear relation contributions compile into

```text
RelationWeight(g * lane + coefficient)
  = CommonAlphaFactor(coefficient) * RelationLaneWeights(lane),

CommonAlphaFactor = [1, alpha, ..., alpha^(g-1)].
```

Role-specific exponent resets, quotient denominators, row challenges, group weights,
setup amplitudes, and overlaps are absorbed additively into `relation_lane_weights`.
They do not break the common low factor. In particular, for mixed `128/64/32`, the
common factor has length 32—not 128, and not a dense full-domain fallback.

The builder must prove:

- all role dimensions are nonzero nested powers of two;
- `g` divides each role dimension;
- physical segment starts and live lengths preserve the low `log2(g)` coefficient bits;
- every role-local exponent resets only at a multiple of `g`;
- overlapping contributions add to the same lane weight exactly once;
- no address outside the live witness receives a relation contribution.

Failure is an input/setup error. The production prover does not silently fall back to an
`N`-element dense relation table. A dense exact evaluator exists only as a test oracle.

### Binding order

The prover always binds the `k = log2(g)` common coefficient dimensions first. This is
both the canonical LSB-first physical order and the optimal order for the factorization.

During those rounds:

- adjacent witness values belong to one common-dimension lane;
- `relation_lane_weights[lane]` is constant across the entire low-coordinate block;
- `common_alpha_factor` folds from length `g` to one scalar;
- relation work uses compact signed digits;
- the range-image term uses the same compact digits and Stage 1 point equality;
- trace is accumulated as a separate additive provider in the same scan.

After the common dimensions are bound:

```text
common_alpha_eval = MLE(CommonAlphaFactor, common_point)
```

is one scalar. The remaining relation term is

```text
common_alpha_eval
  * folded_digit_witness(lane)
  * folded_relation_lane_weights(lane).
```

Only then does the prover bind the remaining address dimensions and fold
`relation_lane_weights`. There is no need for public x/y handling, x-prefix files,
y-prefix files, or a dense-mode dispatch.

This order is mandatory. Binding lane dimensions before the alpha dimensions destroys
the cheap constant-per-lane factor and is not an alternative production schedule.

### State ownership

`RelationRangeImageProver` owns one checked plan and one of two statically typed phases:

```rust
enum RelationRangeImageState<E> {
    CompactPrefix(CompactPrefixState<E>),
    FoldedSuffix(FoldedSuffixState<E>),
}
```

The enum is matched once per round outside the hot scan. It is not inspected per pair.

`CompactPrefixState` owns:

- the shared `Arc<[i8]>` digit witness;
- `CommonAlphaRelationWeights`;
- the range-check equality state;
- optional `TraceWeightState`;
- current claim-recovery data;
- any selected compact pair/quad/octet cache.

`FoldedSuffixState` owns:

- one folded field-valued digit-witness table;
- one folded relation-lane table;
- one folded trace state only when trace has not already collapsed into lane weights;
- the range-check equality state;
- current claim-recovery data.

There is exactly one transition. It materializes the folded digit witness at `N/2^r`
after the selected deferred prefix of `r` challenges, folds the alpha factor to one
scalar when the common prefix completes, and does not retain compact-derived field tables
that are no longer used.

### Relation arithmetic inside the low-coordinate scan

For one witness pair `w(T) = w_0 + T * delta_w` and alpha pair
`a(T) = a_0 + T * delta_a`, the non-trace relation polynomial before the lane weight is

```text
w(T) * a(T).
```

Accumulate its three coefficients for all pairs in one lane/block first, then multiply
the three lane totals by `relation_lane_weights[lane]` once. Do not form
`a_endpoint * relation_lane_weight` for every witness endpoint as the current code does.

Compact rounds use signed/unreduced accumulators with a proved bound. Field-valued later
rounds use the canonical delayed-product contract where available and canonical field
accumulation otherwise. The serial and parallel paths call the same block reducer; they
do not contain copies of the equation.

### Initial-round deferral is retained and generalized

The current two-round prefix is not discarded. The factorized relation term is especially
well suited to it because:

- the digit witness is a small balanced integer;
- the alpha factor is small and contiguous;
- the relation lane weight is constant over each low-coordinate block;
- a biquadratic relation prefix can be accumulated before multiplying by the lane weight;
- field witness materialization can be delayed until after two challenges.

The rewrite must implement and measure every viable strategy below, then keep the best
complete-Stage-2 implementation per basis. “May choose” is not sufficient: all candidates
are tried under the same harness; losing code is deleted.

| Basis | Mandatory candidates |
|---:|---|
| LB2 / 4 | optimized one-round pair scan; factorized two-round prefix; three-round range-image deferral using 256 range-image octet classes while computing the signed relation term directly |
| LB3 / 8 | optimized one-round pair scan; factorized two-round prefix; three-round range-image deferral using folded quad/Taylor techniques while computing the signed relation term directly |
| LB4 / 16 | optimized one-round pair scan; factorized two-round prefix; 4,096-key challenge-dependent range-image quartet table plus direct factorized relation prefix |
| LB5 / 32 | optimized one-round pair scan; factorized two-round rescan; compact range-image pair aggregation where it avoids field traffic |
| LB6 / 64 | optimized one-round pair scan; factorized two-round rescan; compact range-image pair aggregation where it avoids field traffic |

The range-image alphabet uses collision classes, while the relation term uses signed
digits. Never index the relation term by `RangeImageClass`: `w` and `-w-1` share a range
image but contribute different linear relation values.

For LB2, eight range-image values have only `2^8 = 256` class patterns, even though eight
signed digits have `4^8` patterns. Therefore the three-round candidate tables only the
range half. The relation half remains a direct compact symbolic accumulation sharing the
same octet traversal. The analogous separation is required whenever a range-image cache
would otherwise incorrectly erase digit sign.

The first-two-round implementation preserves ordinary transcript causality:

```text
send round_0; receive r_0;
send round_1; receive r_1;
materialize state at N/4.
```

A three-round implementation likewise sends and receives three ordinary messages and
challenges before materializing at `N/8`. No proof or verifier change is required.

The existing approach of computing eight compressed norm grid values and eight compressed
relation grid values is a baseline, not an architectural requirement. Compare it with a
coefficient-form prefix that exploits constant lane weights, split equality, compact
range classes, and claim recovery. Retain the bivariate prefix if it wins; do not preserve
its current wrapper/module structure merely because its algebra remains useful.

### Range-image half

The Stage 2 range term has inner quadratic

```text
digit_witness(T) * (digit_witness(T) + 1)
```

times the current equality factor, so the standard message degree is three. Use the same
exact `range_image` arithmetic and class tables as Stage 1 where doing so removes work,
but call the canonical functions directly. Do not create Stage-2 copies or forwarding
wrappers.

The Gruen recovery path may omit the recoverable linear inner coefficient. Full and
recovered forms must share one accumulator layout and one conversion to the standard
round polynomial. They are not separate scan functions.

The compact fold lookup is built from the known supported digit interval
`[-b/2, b/2-1]`; it must not find min/max by scanning the full witness. This is not digit
validation. It is using the already-validated basis contract to construct a bounded
lookup table directly.

### Trace handling

Trace is the only relation addend that is not required to share the common alpha factor.
It remains explicit:

```text
digit_witness(z) * trace_weight(z).
```

Use one closed state representation:

```rust
enum TraceWeightState<E> {
    Absent,
    SparseBlocks(/* active high lanes with exact low-coordinate rows */),
    DenseExactPrefix(/* only when the trace is genuinely dense */),
}
```

This enum describes real supported trace shapes, not alternate algorithms. Its semantic
builder is the single source of truth. It must:

- expose pair/quad values without a binary search per witness item;
- fold the low common coordinates under the same challenges as the witness;
- share the Stage 2 witness traversal and coefficient reducer;
- remain a separate additive coefficient accumulator;
- avoid a remap allocation when source and destination physical order already agree;
- appear exactly once in the final verifier relation.

Benchmark whether a sparse block is better folded directly or coalesced once. Keep only
the winning representation for each statically known trace shape; do not store both.

### Live prefixes and suffixes

The digit-witness, relation, and trace terms vanish on the padded suffix because the
digit witness is zero. The range-image term also vanishes pointwise at zero, but the
Gruen/equality-factored internal representation still owns current equality state and
claim recovery. Suffix handling must be derived from the semantic term, not copied from a
Stage 1 leaf with a nonzero default.

All prefix kernels use one checked pair/block iterator. It handles:

- an odd final live item paired with zero;
- blocks crossing a split-equality boundary;
- live high lanes followed by padded high lanes;
- the transition from compact common-coordinate rounds to field suffix rounds.

There are no separate prefix-x, prefix-y, and dense implementations.

### Later rounds

After the compact prefix, one canonical field round scans adjacent folded witness values.
It accumulates:

```text
range equality * witness * (witness + 1)
+ common_alpha_eval * witness * relation_lane_weight
+ witness * trace_weight.
```

When common-coordinate binding has finished, `common_alpha_eval` is scalar. Fold witness,
relation-lane weights, range equality, and trace state once per challenge. A fused
fold-and-next-round scan may be benchmarked only if it preserves the selected accumulation
strategy and actually removes a read; it is not presumed beneficial after the negative
Stage 1 result.

### Verifier path

The verifier continues replaying a standard degree-3 sum-check. Its expected final value
uses the same semantic relation-weight evaluator as the prover builder:

```text
range_binding_challenge
  * Eq(range_check_point, next_witness_point)
  * next_witness_eval * (next_witness_eval + 1)

+ next_witness_eval
  * common_alpha_eval(common_point)
  * relation_lane_weight_eval(lane_point)

+ next_witness_eval * trace_weight_eval(next_witness_point).
```

The verifier does not materialize either factor. A typed point view checks that the first
`log2(g)` challenges are the common-coordinate point and the remaining challenges are the
lane point. No caller slices a raw vector.

Malformed role dimensions, point lengths, layouts, proof degrees, round counts, and
allocation lengths return `AkitaError`. The rewrite adds no verifier-reachable `assert!`,
`panic!`, `unwrap`, unchecked indexing, or allocation based on unvalidated proof data.

### Mixed dimensions

The second PR integrates the semantic bases from #309 before building relation weights.
It consumes `log_basis_inner`, `log_basis_outer`, and `log_basis_open` according to their
existing ownership; it does not restore a largest-basis or uniform-basis shortcut.

Mixed dimensions are complete only when all of the following agree with a dense oracle:

- relation-weight construction;
- every Stage 2 round polynomial and fold;
- trace addressing;
- final verifier evaluation;
- local setup contribution evaluation;
- the current recursive setup-contribution proof boundary, if enabled for that schedule;
- multi-group and multi-chunk witness layouts.

The first required mixed tuple is `128/64/64`; `128/64/32` is the next correctness case.
The common alpha factor lengths are 64 and 32 respectively. Equal padded domain lengths
do not imply equal native coordinate meanings.

### Current setup-contribution stage in the second PR

The second PR does not move proof statements between stages. It may refactor the current
setup-contribution prover only to:

- consume the same typed mixed-dimension relation point;
- reuse the one semantic setup-weight builder;
- remove a `ring_bits == 0`/uniform-only assumption;
- define the complete `setup_contribution_eval` that the final offload PR will move;
- preserve current proof bytes, transcript order, and opening behavior.

Do not spend this PR building an elaborate numeric Stage 3 architecture that the next PR
will delete. Reusable domain, point, setup-weight, and fold mechanics land now; protocol
movement and proof-container replacement wait for the offload cutover.

### Target module structure for the second PR

The exact split may follow existing crate conventions, but ownership must be semantic:

```text
sumcheck/relation_range_image/
  mod.rs                         proof lifecycle and transcript-independent orchestration
  relation_weights.rs            common-alpha builder and folded lane state
  compact_prefix.rs              one/two/three-round compact kernels
  folded_rounds.rs               canonical later-round scan and folds
  trace_weights.rs               sparse/dense exact trace state
  tests.rs
```

Shared Stage 1 arithmetic is called at its canonical definition. The Stage 2 portion of
`two_round_prefix/`, `akita_stage2/{x_prefix,y_prefix,dense_terms,round2_prefix}`, and
duplicate serial/parallel branches are deleted. No compatibility module re-exports the
old paths.

### Intended diff surface for the second PR

| Surface | Responsibility |
|---|---|
| `akita-types` relation/setup geometry | checked role dimensions, common factorization plan, typed points, one semantic evaluator |
| ring-switch finalization | build `relation_lane_weights` instead of a uniform column table or mixed dense table |
| `akita-prover::sumcheck` | one `RelationRangeImageProver`, compact prefix kernels, folded suffix, trace state |
| current setup-contribution prover | only mixed-point/provider adaptation reusable by the next cutover |
| verifier | semantic factorized final evaluation and current setup-boundary replay |
| PCS tests/benches/profile | round-by-round dense oracle, mixed tuples, per-basis kernel selection, protocol epoch |
| book/spec | current Stage 2 implementation and mixed-dimension contract |

Proof containers, planner topology, setup-offload stage placement, and serialized round
counts remain unchanged.

### Tracing and measurement contract for the second PR

Trace phase owners, not hot loops:

```text
relation_range_image_prove
  build_common_alpha_relation_weights
  prepare_trace_weight_state
  compact_prefix {basis, deferred_rounds, strategy}
  materialize_folded_suffix
  folded_round {round}
  fold_round_state {round}
```

Do not emit events per pair, coefficient, class, lane, or Rayon item. Perfetto must remain
readable at production-scale domains.

Measure feature-pruned profile-CI builds for every LB2-LB6 basis, full and partial live
prefixes, trace absent/present, one and production thread counts, and uniform/mixed role
dimensions. Report separately:

- relation-weight construction;
- trace-state construction;
- compact prefix;
- field materialization;
- later scans and folds;
- complete Stage 2;
- complete prover;
- allocations and peak field elements;
- verifier time;
- proof bytes and transcript events.

The production winner is selected by complete Stage 2 and complete prover results, not a
microkernel alone. CI benchmark reporting catches later regressions; the implementation
does not add per-iteration measurement wrappers.

### Acceptance criteria for the second PR

- One relation/range-image prover and one semantic relation-weight builder remain.
- The proof, transcript, challenge order, final claim, and direct proof bytes match the
  incoming protocol epoch.
- `ring_bits == 0` is gone as a mixed-dimension sentinel.
- Public/current constructors do not accept independent `live_x_cols`, `col_bits`, and
  `ring_bits` geometry.
- Common alpha coordinates bind first and collapse to one scalar before lane binding.
- Relation lane weights occupy at most `N/g` field elements, excluding separately reported
  trace state and small prefix tables.
- The relation multiplier is applied once per lane/block in low-coordinate rounds, not
  once per witness endpoint.
- Every mandatory per-basis candidate is measured; one winner remains per basis.
- The two-round prefix remains when it wins, in the cleaned state machine rather than a
  wrapper around old Stage 2.
- LB2/LB3 three-round candidates are tested without confusing range-image classes with
  signed relation digits.
- Trace appears exactly once and shares witness scans without being falsely factorized.
- Dense and factorized oracles agree round by round for uniform and mixed dimensions.
- No primary Stage 2/prover/verifier cell regresses beyond measurement noise; targeted
  cells show a material win.
- Numeric-stage setup code touched by the PR contains only reusable mixed-dimension
  semantics, not a new throwaway abstraction.

## Stacked PR: two-stage recursive offloading cutover

### Scope

The third PR intentionally changes the recursive-offload proof protocol. It is one atomic
semantic cutover across plan, prover, verifier, proof types, wire sizing, transcript,
planner, setup routing, schedules, and tests.

Direct/non-offloaded folds retain the optimized Stage 1 plus fused
`RelationRangeImageProof` from the second PR. Only recursive setup-offload schedules use
the new placement.

### Recursive Stage 1 equation

All range-product layers before the final leaf remain the #312 equality-factored product
subchecks. Let `leaf_input_claim` and `leaf_anchor` be the claim and point from that
prefix, and let `LeafBatch` be the plan-derived quadratic or quartic range-image leaf.
After binding the linear relation claim, sample `range_relation_batch_challenge` and prove

```text
leaf_input_claim
  + range_relation_batch_challenge * linear_relation_claim

= sum_z [
    Eq(leaf_anchor,z) * LeafBatch(range_image(z))
  + range_relation_batch_challenge
      * digit_witness(z)
      * (CommonAlphaFactor(coefficient(z))
           * RelationLaneWeights(lane(z))
         + TraceWeight(z))
  ].
```

The final leaf is a standard sum-check because the two terms do not share one equality
factor. Its degree is three for LB2 and five for LB3-LB6.

### Mandatory fused first round

The first recursive final-leaf round walks each compact witness pair once. In that one
traversal it:

- loads the signed digit pair;
- derives the range-image pair;
- accumulates the anchored range leaf;
- accumulates the common-alpha relation term;
- accumulates trace if present.

After the first challenge, one compact traversal materializes two independent lanes:

```text
folded_range_image
folded_digit_witness.
```

The representations diverge after that transition, but they continue under the same
challenge sequence and combined claim. Two scans pretending to be a fused round, or one
fused first round followed by independent sum-checks, is not acceptable.

The factorized Stage 2 implementation from the previous PR is reused directly for the
relation half. Its common-coordinate prefix, compact batching techniques, relation lane
weights, and trace state are not reimplemented in a `relation_leaf` variant.

### Recursive Stage 1 output

Stage 1 returns at one `RangeRelationPoint`:

```text
range_image_eval
digit_witness_eval.
```

The verifier keeps a consuming deferred check whose final equation requires the complete
`setup_contribution_eval`. It cannot accept the Stage 1 leaf until Stage 2 binds that setup
claim. Consuming ownership prevents omission or double application.

### Recursive Stage 2 statements

Stage 2 owns two semantic obligations:

1. Prove the complete setup contribution against the selected committed setup prefix.
2. Reduce the independent `range_image_eval` and `digit_witness_eval` to the next witness
   opening by proving the pointwise range-image identity.

The witness reduction is

```text
range_image_binding_challenge * digit_witness_eval + range_image_eval

= sum_z Eq(range_relation_point,z) * [
     range_image_binding_challenge * digit_witness(z)
   + digit_witness(z) * (digit_witness(z)+1)
  ].
```

It ends at `next_witness_eval`. The setup product ends at `setup_prefix_eval` over its own
`SetupCoefficientDomain`. Witness and setup addresses are not two views of one domain.

### Separate and batched realizations

The initial cutover may land the simpler `Separate` realization first only if it is the
complete production shape in that PR:

```text
SetupContributionProof
then RangeImageConsistencyProof.
```

A batched realization may land in the same PR if fully implemented and selected, or in a
small immediately stacked capability PR. It must not land as an unused enum variant. For
native setup and witness round counts `lambda` and `mu`:

```text
separate round elements = 2 * (lambda + mu)
batched round elements  = 3 * max(lambda, mu).
```

The planner compares complete serialized proofs, not only these round terms. A batch over
unequal domains uses explicit checked lifts, one scale factor per native term, and typed
suffix projections. Equal padded lengths alone never authorize challenge reuse.

### Proof-size objective

Moving the relation to the standard final Stage 1 leaf adds one coefficient per witness
round relative to the equality-factored range-only leaf. It removes the old standard
degree-3 relation/range Stage 2 from the recursive path and combines setup plus witness
reduction in the new Stage 2.

Every scheduled recursive target must:

- be no larger than its matching pre-cutover recursive proof in complete serialized
  bytes;
- reduce measured verifier work by removing local setup scanning;
- preserve the #311 terminal proof exactly;
- preserve direct/non-offloaded proof bytes exactly.

Byte parity is acceptable when verifier work improves. A round-only estimate is not an
acceptance test; include scalars, envelopes, opening metadata, extension encoding, and the
outgoing witness binding.

### Numeric Stage 3 deletion

The cutover deletes `AkitaStage3Prover`, numeric Stage 3 proof fields, accessors, transcript
frames, shape branches, sizing formulas, verifier modules, and compatibility readers.
Setup contribution and witness reduction are semantic Stage 2 components. Do not leave a
`stage3` wrapper forwarding to them.

### Intended diff surface for the offloading PR

| Surface | Responsibility |
|---|---|
| fold-check plan/types | direct versus recursive topology and exact proof shape |
| Stage 1 prover/verifier | recursive joint final leaf only; earlier range tree unchanged |
| Stage 2 prover/verifier | setup contribution plus witness/range-image reduction |
| setup prefix routing | exact slot/domain and typed opening point |
| proof wire and sizing | new recursive epoch; unchanged direct and terminal epochs |
| planner/schedules | choose only eligible no-larger recursive shapes |
| tests/docs | new transcript oracle, proof-size parity, malformed proof, setup-slot failures |

No compressed-commitment file, negative-binary proof field, terminal relation path, or
unrelated commitment algorithm belongs in this diff.

### Acceptance criteria for the offloading PR

- Direct folds still use #312 Stage 1 plus the second PR's optimized fused Stage 2.
- Recursive folds use one final Stage 1 relation/range sum-check and one Stage 2 claim
  reduction.
- The recursive final leaf's first round and first materialization each traverse compact
  witness data once.
- `range_image_eval` and `digit_witness_eval` remain independent and share one point.
- The complete setup contribution closes the deferred Stage 1 verifier equation exactly
  once.
- Mixed role dimensions use the common alpha factorization already established by the
  second PR.
- Stage 3 is absent from production names, proof shape, serialization, sizing, planner,
  prover, and verifier.
- Each scheduled recursive proof is no larger in complete bytes and improves verifier
  time against its matching old proof.
- Direct and #311 terminal bytes/events are unchanged.
- Malformed proof, point, lift, route, and setup-slot data return errors without panic or
  attacker-controlled unbounded allocation.

## Dependency and conflict policy

### PR #311

#311 is the hard base for #312. Its current head is
`fad006e2280e880fa16f1cd13b5ea2df599364d0`. It removes terminal relation sum-checks, so
this series does not touch terminal proof payloads or invent empty fold-check placeholders.

### PR #309

#309 currently has head `b0c2d4683539b0c2a465b996f48adfc465a20198` and introduces
the semantic inner, outer, and opening digit-decomposition bases. It is not needed by the
#312 Stage 1 kernel because `DigitRangePlan` consumes the checked concrete range basis
already produced upstream.

It is required before the mixed-dimension Stage 2 PR because relation and setup builders
must consume the semantic role bases rather than infer one global basis. The second PR is
based on merged #312 plus merged/refreshed #309. If both are still open, use an explicit
integration base; do not copy #309 concepts into #312 or add compatibility adapters.

### Other open work

- Distributed setup-offload schedules must be integrated before the final offloading PR
  claims distributed coverage. Adapt their one canonical fixture; do not clone it.
- Compressed commitments remain a future consumer of the typed domain/point boundary.
  Do not edit their planners, wire, or commitment layout in this series.
- Packed sum-check work may use the final address-major scalar states after the Stage 2
  rewrite. It must not preserve the deleted x/y/prefix architecture.
- Divergent verifier kernels are prior art only. Port algebra after the semantic provider
  is canonical; never merge an old layout wholesale.

Before each stacked PR begins, refresh the exact open-PR heads and compare the full
merge-base diff. Conflict avoidance is defined by semantic ownership, not by hoping git
reports few textual conflicts.

## Test oracles

### Protocol epochs

`digit_range_protocol_epoch` and `fold_protocol_epoch` protect #312's declared
wire-preserving Stage 1 cutover. The second PR adds a complete Stage 2 epoch covering
proof bytes, round messages, challenge order, final point, final evaluation, and logging
events for each supported basis and trace shape.

The offloading PR intentionally creates a new recursive epoch and records:

- old and new complete recursive proof bytes;
- exact field/scalar additions and deletions;
- transcript-frame and challenge-order changes;
- direct and terminal digests that must remain unchanged.

### Dense mathematical oracles

Test-only dense implementations are permitted and required. They are not production
fallbacks. Round-by-round comparisons cover:

- Stage 1 class-indexed versus padded range tables;
- factorized relation weights versus full flat weights;
- compact prefix messages versus direct dense summation;
- trace sparse/dense states versus exact flat trace weights;
- mixed-dimension setup contribution versus direct flat dot product;
- recursive joint leaf versus a separately materialized standard sum-check;
- separate/batched Stage 2 reductions versus independent native proofs.

Compare coefficients, not only evaluations at `0` and `1`. Compare after every challenge
and fold, not only the final accepted proof.

### Required edge cases

- every valid digit and both out-of-range neighbors for LB2-LB6 arithmetic tests;
- all-zero, uniform, deterministic high-entropy, zero-heavy, and alternating endpoints;
- full, three-quarter, odd, and short positive live prefixes; reject zero length;
- alpha equal to zero, one, and random field values;
- trace absent, sparse, and dense;
- uniform `64/64/64`, uniform `128/128/128`, mixed `128/64/64`, and mixed
  `128/64/32`;
- singleton, multi-group, and multi-chunk layouts;
- serial and parallel execution;
- fp128 primary plus fp64/Ext2 and fp32/Ext4 smoke coverage;
- malformed proof counts, degrees, points, domains, role dimensions, setup slots, and
  serialized lengths.

## Performance and tracing policy

Optimization decisions use the repository's profile-CI feature set and dedicated
benchmarks. Criterion and CI benchmark output decide winners; Perfetto tracing explains
where time went. Neither replaces correctness or protocol-epoch tests.

Use coarse spans for construction, compact prefix, materialization, later rounds, and
folding. Never instrument pair, class, coefficient, lane, or Rayon-item loops. Record the
exact head SHA, base SHA, field, feature set, input shape, thread count, and machine for
every selection claim.

Candidate branches are disposable. Once a strategy wins:

- put the winner in the canonical state machine;
- delete the loser branch and production code;
- record the result here;
- do not add a runtime selector or schedule knob.

CI benchmark reporting is the ongoing regression detector. Production code must not carry
ad hoc timing wrappers, benchmark-only branches, or duplicated measured/unmeasured
kernels.

## Rejected architecture

The following are explicitly rejected:

- dual small-basis and large-basis Stage 1 provers;
- eager padded range-image tables or retained product forests;
- a binary-only range tree;
- one module or trait family per log basis;
- a second semantic digit-validation scan inside the honest prover;
- public x/y relation geometry;
- `ring_bits == 0` as a mixed-dimension mode;
- using `d_a` rather than `min(d_a,d_b,d_d)` for the common alpha factor;
- binding relation lanes before the common alpha coordinates;
- forcing trace into the common alpha factor;
- full `N`-element mixed relation weights in production;
- relation lookup tables indexed only by range-image class;
- a generic expression algebra, descriptor engine, or new protocol crate;
- wrapper functions that preserve old and new APIs simultaneously;
- proof/schedule fields selecting CPU kernels;
- an unused batched reduction variant;
- moving the relation to Stage 1 for direct non-offloaded folds;
- preserving numeric Stage 3 after the offload cutover;
- compressed commitments or negative-binary range checks in this series.

## Validation and merge gates

Each PR runs the current commands in `AGENTS.md`, focused protocol-epoch tests, and the
benchmarks for its owned surface. Documentation changes also run
`./scripts/check-doc-guardrails.sh`. A live process is not a completed validation result.

Before merge, inspect the complete diff from the actual PR base and verify:

- every touched production file belongs to the PR's intended surface;
- no old wrapper, decoder, or alternate engine survives the cutover;
- proof size formulas match actual serialization;
- prover and verifier logging transcripts agree;
- verifier-reachable malformed input is rejected without panic;
- benchmark claims name their exact source head;
- the spec header and stack ledger reflect the final merged state.

## Definition of done

The full series is done when:

- #312 is the complete, single-source Stage 1 range cutover for LB2-LB6;
- the relation/range-image prover has one compact-prefix/folded-suffix implementation;
- common alpha coordinates of length `min(d_a,d_b,d_d)` bind first;
- relation lane weights use at most `N/g` state and mixed dimensions no longer use a
  dense sentinel path;
- the best measured one/two/three-round prefix strategy is selected separately for every
  digit basis;
- trace shares witness traversal, remains independently additive, and appears once;
- direct setup retains the efficient Stage 1 range plus fused Stage 2 placement;
- recursive offload moves relation checking into the final Stage 1 leaf and setup plus
  witness reduction into Stage 2;
- numeric Stage 3 and all x/y/prefix wrapper architecture are deleted;
- direct and terminal epochs remain unchanged across the offload cutover;
- scheduled recursive proofs are no larger and reduce verifier work;
- compressed commitments and the fused negative-binary range check remain explicitly
  future work, with no dormant proof fields or code paths added here.
