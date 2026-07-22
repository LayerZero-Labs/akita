# Spec: Two-stage setup offload

| Field | Value |
|---|---|
| Author(s) | |
| Created | 2026-07-22 |
| Status | proposed |
| Branch | `quang/stage3-setup-contribution-refactor` |
| Base | PR #318 integration head `1275bc63` (including PR #317) |
| Supersedes | The recursive path in `setup-product-sumcheck.md` and `batched-stage3-setup-opening.md` after cutover |
| Superseded-by | |
| Book-chapter | |

## Summary

Recursive setup offload currently appends a third sum-check after the direct
relation/range-image sum-check. Stage 3 proves the setup product and carries the
Stage 2 witness opening to a new point. This placement is correct for uniform
ring dimensions, but it creates a second point-routing protocol and blocks
mixed ring dimensions because it reconstructs the setup projection from
`d_a` rather than the checked common relation geometry.

This spec replaces the recursive three-stage path with the two-stage protocol
described by the Akita paper:

1. Stage 1 runs the range tree and adds relation plus evaluation trace to the
   final range sum-check.
2. Stage 2 batches the setup product with a witness claim reduction. The claim
   reduction binds the carried range image to the witness and carries the
   Stage 1 witness opening to one fresh witness point.

Direct folds MAY retain the current Stage 1 range proof followed by the fused
relation/range-image Stage 2 proof. Recursive setup offload MUST use the
two-stage shape in this document. Stage 3 and its proof field are deleted when
the recursive cutover is complete.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14
(RFC 2119 and RFC 8174) when, and only when, they appear in all capitals.

## Motivation

The setup contribution depends on the random relation point. It cannot be
proved before relation evaluation fixes that point. The current implementation
therefore runs:

```text
Stage 1: range tree
Stage 2: relation + evaluation trace + range-image refold
Stage 3: setup product + carried Stage 2 witness opening
```

The paper observes that relation is linear in the witness and can ride the
final Stage 1 range leaf without increasing its degree. This fixes the relation
point one stage earlier. The setup product can then run beside the range-image
refold in Stage 2:

```text
Stage 1: range tree; final leaf also proves relation + evaluation trace
Stage 2: setup product + witness claim reduction
```

This placement has four advantages:

- recursive offload adds no third transcript stage;
- setup contribution consumes the checked relation point directly;
- the witness exits at one point rather than carrying unrelated Stage 1 and
  range-image claims; and
- mixed role dimensions no longer need a Stage 3-only `log2(d_a)` projection.

## Scope

This spec covers:

- the recursive setup-offload proof shape;
- prover and verifier transcript order;
- relation and evaluation-trace placement in Stage 1;
- setup-product and witness-claim-reduction placement in Stage 2;
- mixed nested role dimensions in the two sum-checks;
- setup-prefix and next-witness openings carried to the successor fold; and
- deletion of the old Stage 3 protocol after parity is established.

This spec does not cover:

- planner objective changes;
- new commitment matrices or setup-prefix layouts;
- non-nested or non-power-of-two role dimensions;
- range-image microkernel specialization; or
- zero-knowledge masking for recursive setup offload.

Range-image kernel specialization remains a separate performance PR. It MUST
not change the equations in this document.

## Typed topology authority

PR #317 makes setup offload a successor-owned edge property:

```text
RecursiveFoldParams.incoming_setup_prefix: Option<SetupPrefixSlotId>
```

This field MUST be the sole schedule authority for whether the predecessor uses
the recursive two-stage protocol. Runtime code MAY derive a local protocol
shape from this edge, but it MUST NOT store an independent producer-side setup
mode.

The mirrored `CommittedGroupParams.setup_prefix` field is transitional. Until
it is removed, schedule validation MUST require exact equality with
`incoming_setup_prefix`. Consumers MUST read the successor edge, not the
mirror, to select the protocol.

The terminal successor MUST NOT request an incoming setup prefix because no
later committed fold can consume the carried prefix opening.

## Prepared setup authority from PR #318

PR #318 makes `SetupContributionPlan` the canonical prepared setup-weight
authority in both direct and deferred setup modes. Stage 2 constructs the plan
while evaluating the relation, uses its prepared E/T/Z equality slices, and
caches the challenge-bound plan for the existing Stage 3 verifier.

The two-stage cutover MUST reuse this plan and its role-native setup geometry.
It MUST NOT add another setup-weight builder, equality-table representation, or
projection API solely for the relocated setup product. Once Stage 3 is deleted,
the plan passes directly from offloaded Stage 1 relation evaluation into the
new Stage 2 setup-product verifier instead of through the Stage 3 cache.

PR #318 also changes recursive schedule selection to prioritize the first
remaining direct setup footprint before proof bytes. That planner policy is
part of this spec's baseline but is not changed by the protocol cutover.

## Current boundary

The current code deliberately rejects recursive setup offload when predecessor
role dimensions differ from the successor inner ring dimension. That rejection
MUST remain until the complete prover, verifier, proof, transcript, and suffix
state cutover in this spec passes end to end.

The implementation MUST NOT remove the guard by replacing `log2(d_a)` with
`log2(coeff_count)` inside the old Stage 3. Such a change would still be wrong:
the A, B, and D setup weights have different native coefficient boundaries, so
one Stage 3 slice cannot represent every role projection.

## Prerequisite: optimize the current Stage 3 first

The two-stage relocation MUST NOT begin from an unnecessarily dense Stage 3.
The first implementation milestone is an optimization-only milestone. It keeps
the current protocol equations, proof fields, stage placement, and transcript
unchanged:

```text
Stage 1: current range tree
Stage 2: current relation + evaluation trace + range-image refold
Stage 3: setup product + linear Stage-2 witness-claim reduction
```

This milestone has two independent prover optimizations:

1. a two-pass prefix/suffix prover for the **linear Stage 3 witness claim**;
2. a rectangular prefix/suffix setup-product prover that preserves the setup
   source in the base field and exploits the exact alpha/setup-index
   factorization.

It also evaluates one independent optimization of the current Stage 2:

3. compile relation and evaluation-trace weights at the checked common
   coefficient boundary, while retaining the current nonlinear range-image
   sum-check and its two-round compact prefix.

The witness optimization MUST NOT change the Stage 2 range-image term. The
setup-product optimization MUST NOT be implemented by parameterizing the
witness reducer: the two terms have different transparent weights, source
representations, and profitable variable splits.

The Stage 2 optimization MUST NOT move relation or evaluation trace into Stage
1, move setup product into Stage 2, change the Stage 2 claim, or change the
transcript. Those changes belong to the later two-stage protocol cutover.

### Current setup-product algorithm

Let

```text
d0 = min(d_a, d_b, d_d)
L  = next_power_of_two(required common-base setup rows)
N  = L * d0.
```

For common setup-row index `i` and coefficient index `j`, the claim is

```text
sigma_setup = sum_i sum_j Setup[i,j] * H[i] * A[j],
A[j]         = alpha^j.
```

`build_setup_product_term` currently:

1. prepares one `SetupContributionPlan`;
2. materializes `H[0..L]` in the extension field;
3. materializes `A[0..d0]`;
4. copies and lifts all `N` base-field setup coefficients into an
   extension-field table; and
5. gives the three dense arrays to `FactoredProductTerm`.

The current algorithm is therefore algebraically factored but not
storage-factored. Its peak owned state is approximately

```text
N extension elements for Setup
+ L extension elements for H
+ d0 extension elements for A.
```

PR #318's `SetupIndexWeightEvaluator` evaluates `H~(rho_idx)` succinctly at
the terminal point. It does not currently provide round messages for the
prover and MUST NOT be described as a prover-side factorization.

### Exact setup-weight factorization

For role `R in {A,B,D}`, write `d_R = q_R d0`. A native role coefficient
index decomposes as `k = lane * d0 + j`, hence

```text
alpha^k = alpha^(lane*d0) * alpha^j.
```

The common coefficient factor is exactly

```text
A[j] = alpha^j, 0 <= j < d0.
```

The remaining role-lane scale `alpha^(lane*d0)` belongs in `H[i]`. With the
current packed layout, one group's index factor is the sum

```text
H_g[i] = H_D,g[i] + H_B,g[i] + H_A,g[i],
```

where, on each role's active projected footprint,

```text
H_D,g[i] = tau_D[row_D(i)] * alpha^(lane_D(i)*d0) * E_g[col_D(i)]
H_B,g[i] = tau_B[row_B(i)] * alpha^(lane_B(i)*d0) * T_g[col_B(i)]
H_A,g[i] = tau_A[row_A(i)] * alpha^(lane_A(i)*d0) * Z_g[col_A(i)].
```

Overlapping A/B/D and group footprints add their weights because the matrices
are prefix views of the same setup. Zero padding has weight zero. These
formulas are the single semantic definition; dense materialization, a
round-message factorization, and the verifier's terminal evaluator MUST be
differentially equal to them.

### Setup-product implementation candidates

The implementation MUST measure the following candidates against the same
complete Stage 3 and end-to-end profiles.

#### Candidate S0: current dense baseline

Lift `N` setup coefficients once and run the existing dense factored product.
This remains the differential and performance baseline until a replacement
passes all gates.

#### Candidate S1: delayed common-coefficient prefix

Keep setup coefficients in `F`. Order the `log2(d0)` common coefficient
variables before the setup-index variables. Delay `k` coefficient rounds by
rescanning the base setup source, then materialize only the table remaining
after those challenges:

```text
owned setup state = N / 2^k extension elements.
```

The first implementation SHOULD test `k = 2`, which is the direct analogue of
the existing compact witness prefix, and `k = log2(d0)`, which materializes one
alpha-evaluated setup value per common row. The latter costs one source scan per
coefficient round but reduces the subsequent dense table to `L` elements.

The implementation MUST report source passes, base-to-extension lifts,
extension multiplications, allocation bytes, and peak RSS. It MUST NOT select
`k` from an unpriced runtime heuristic.

#### Candidate S2: rectangular prefix/suffix setup-product prover

The existing rectangular factorization is sufficient for a two-pass
prefix/suffix prover. It does not require a balanced split, and it does not
require another factorization of `H`.

Use the setup-index variables as the prefix and the common coefficient
variables as the suffix. The first pass over the base-field setup source forms

```text
Q[i] = sum_j A[j] * Setup[i,j].
```

The prefix rounds prove

```text
sum_i H[i] * Q[i].
```

`Q` and `H` MUST remain separate multilinear factors. The prover MUST NOT fold
their pointwise product as one multilinear table.

After the sampled setup-index point `a` is known, the second pass over the
base-field setup source forms

```text
Setup_a[j] = sum_i eq(a,i) * Setup[i,j].
```

The suffix rounds prove

```text
H(a) * sum_j A[j] * Setup_a[j].
```

The target owned extension-field state is

```text
L + d0
```

rather than `L * d0`. This candidate remains useful when the rectangle is
unbalanced. A square-root split is not a requirement; the relevant condition
is `L + d0 << L * d0`.

Both setup passes MUST read `F` coefficients directly. The implementation MUST
use `MulBase<F>` or `MulBaseUnreduced<F>` and MUST NOT lift the setup prefix into
an `E` table. Delayed product accumulation MUST reduce only at an explicitly
bounded, backend-safe boundary.

#### Candidate S3: bounded-rank split inside the setup-index weight

Splitting the setup-index variables themselves is permitted only if the
`L`-element state in S2 remains a measured bottleneck and `H` is exposed as an
exact bounded-rank prefix/suffix decomposition

```text
H(u,v) = sum_{t < T} P_t(u) * Q_t(v).
```

For a balanced split of the setup-index variables, phase 1 forms

```text
R_t[u] = sum_v Setup_alpha[u,v] * Q_t(v),
```

and proves

```text
sum_u sum_t P_t(u) * R_t(u)
```

while retaining `P_t` and `R_t` as separate multilinear factors. It MUST NOT
fold their pointwise product as one multilinear table. After prefix challenges
`a`, phase 2 makes one more setup-source pass to form

```text
Setup_alpha,a[v] = sum_u eq(a,u) * Setup_alpha[u,v]
H_a[v]           = sum_t P_t(a) * Q_t(v),
```

then proves the suffix rounds as an ordinary degree-2 product sum-check.

The current A/B/D weights are tensor products on aligned role rectangles, but
physical offsets, partial spans, and the A-role fold-digit sum introduce carry
states. The decomposition MAY reuse the finite carry states already embodied
by `eval_compact_pair_eq`; it MUST state and enforce an explicit rank/work
bound. Calling the final-point evaluator once per round is not this
decomposition and is forbidden.

S3 is retained only if its `T * sqrt(L)` state and repeated setup scans improve
the complete prover over the best S0/S1/S2 candidate. No permanent runtime
selector or losing implementation remains after selection.

### Current linear witness claim

The current Stage 3 witness term is exactly

```text
C_w = W~(r_old) = sum_x eq(r_old, x) * W(x).
```

The source is a compact signed-digit array. The current implementation delays
two rounds and then materializes an extension-field table of size `N_w / 4`.
The replacement MUST use the two-pass square-root algorithm from Jolt's
`RegistersClaimReductionSumcheckProver`, adapted to Akita's checked LSB-first
opening layout and verifier error contract.

### Two-pass witness prefix/suffix algorithm

Let `n = log2(N_w)`, choose

```text
p = floor(n / 2),
s = n - p,
x = u + 2^p v,
r_old = (r_u, r_v),
```

with `u` occupying the first `p` LSB-first sum-check variables. The split MUST
be derived from the checked opening domain, including zero padding; it MUST NOT
be inferred from the live compact length.

#### Pass 1 and prefix rounds

Build

```text
Q[u] = sum_v small_mul(eq(r_v, v), W[u,v]).
```

This is one complete streaming pass over the compact witness. The prefix claim
is

```text
C_w = sum_u eq(r_u, u) * Q[u].
```

The prover stores `Q` and an implicit equality factor `(r_u, scale=1)`; it does
not need to materialize the equality table. It answers `p` degree-2 rounds by
folding `Q` and the equality factor low-to-high. Let the sampled prefix point
be `a`.

#### Pass 2 and suffix rounds

Make a second complete witness pass and build

```text
W_a[v] = sum_u small_mul(eq(a, u), W[u,v]).
```

After the prefix rounds, the claim is

```text
eq(r_u, a) * sum_v eq(r_v, v) * W_a[v].
```

The suffix state is therefore:

```text
table          = W_a[0..2^s]
equality point = r_v
equality scale = eq(r_u, a).
```

The prover answers the remaining `s` degree-2 rounds using the ordinary
equality-times-table kernel. At suffix point `b`, the verifier checks

```text
eq(r_old, (a,b)) * W~(a,b).
```

The output opening is the genuine witness opening `W~(a,b)`.

#### Witness complexity and invariants

The target costs are:

```text
compact witness passes: exactly 2
extension state:        O(2^p + 2^s) = O(sqrt(N_w))
sum-check degree:       2
proof bytes:            unchanged
transcript:             unchanged
```

Temporary equality tables used while building either state count toward peak
memory. `small_mul(weight, digit)` is mathematical notation for multiplying an
extension-field equality weight by one compact signed digit. It MUST NOT lift
or materialize the digit as an extension-field element, and it MUST NOT execute
a general extension-field multiplication.

The Stage 3 witness reducer accepts only the protocol's supported log bases
`2..=6`; the wider representational capacity of `i8` is not protocol support.
For accepted log basis `ell`, every digit MUST be validated against the exact
asymmetric balanced interval

```text
[-2^(ell-1), 2^(ell-1) - 1].
```

The generic contraction kernel MUST accumulate

```text
positive += weight.mul_u64_unreduced(abs(digit))  when digit > 0
negative += weight.mul_u64_unreduced(abs(digit))  when digit < 0
```

and call `reduce_signed_accum(positive, negative)` once per output. Before the
kernel starts, the implementation MUST check that

```text
contraction_len * observed_max_abs_digit <= 2^64 - 1,
```

which is the conservative addition headroom of the narrow accumulators used by
the native field backends.

Log basis 2 admits exactly `{-2, -1, 0, 1}`. The first, column-oriented source
pass uses a dedicated four-way kernel. For each equality weight it computes
`unit = weight.mul_u64_unreduced(1)` once and applies:

```text
-2 => negative += unit; negative += unit
-1 => negative += unit
 0 => no operation
 1 => positive += unit
```

The output tile is a physical kernel choice, not proof geometry. The pinned
eight-thread A/B benchmark selects four adjacent outputs independently for both
LB2 passes. This bounds the live state to four positive and four negative
accumulators while reusing one `unit` across four compact digits. Against the
generic 32-output kernel, LB2 tile 4 reduced the column contraction by about
51% on dense digits and 30% on sparse digits, and reduced the row contraction
by about 16% and 12%, respectively. Log bases 3 through 6 retain the generic
32-output kernel; this slice does not add an unpriced density-based selector.

Both source-pass Perfetto spans MUST record the selected kernel, log basis,
certified digit bound, and observed digit bound. Implementations MUST preserve
the canonical padded zeros and checked physical-to-opening address map.

This algorithm is specific to the linear Stage 3 claim. It MUST NOT replace or
reinterpret the nonlinear range-image refold in Stage 2.

## Optimization-only Stage 2 relation and evaluation trace

This milestone optimizes the current fused Stage 2 without changing its
statement. Stage 2 continues to prove the existing batched identity containing:

```text
range image + relation + evaluation trace.
```

The nonlinear range-image term retains its current compact signed-digit source,
two-round initial batch, transcript order, and folded suffix. Only the linear
relation and evaluation-trace contractions change representation.

### Common coefficient split

Let

```text
C = common relation witness coefficient count
K = padded relation-lane capacity.
```

The flat witness is indexed as `W[lane, coefficient]` with
`0 <= coefficient < C`. The split is the same checked split used by
`RelationRangeImagePlan`; implementations MUST NOT derive another coefficient
boundary.

The current production configurations usually have `C = 64`. The algorithm is
generic over every checked power-of-two `C`.

### Relation prefix/suffix factorization

The relation weight already factors exactly as

```text
RelationWeight(lane, coefficient)
    = M[lane] * A[coefficient],
A[coefficient]
    = alpha^coefficient.
```

During the initial compact-witness scan, form

```text
Q_relation[coefficient]
    = sum_lane W[lane, coefficient] * M[lane].
```

The coefficient rounds prove

```text
sum_coefficient Q_relation[coefficient] * A[coefficient].
```

`Q_relation` and `A` MUST remain separate multilinear factors. Folding their
pointwise product would prove the multilinear extension of the Boolean
pointwise product, not the required product of their multilinear extensions.

After the coefficient challenges `a`, the relation lane weight is

```text
A(a) * M[lane].
```

### Exact evaluation-trace rank at the common split

For trace group `g`, let its source ring dimension be

```text
D_g = q_g * C.
```

Checked trace preparation MUST require `D_g` to be divisible by `C`, and every
physical trace support interval MUST be aligned to `C`. An incompatible trace
MUST be rejected rather than sent through a dense fallback.

Split the group's inner trace into `q_g` common-coordinate factors:

```text
I_g,h[c] = inner_trace_g[h * C + c],
0 <= h < q_g.
```

Compile claim coefficients, chunk placement, block weights, opening-digit
weights, and overlapping support into sparse lane factors `F_g,h[lane]`. The
complete trace weight is exactly

```text
TraceWeight(lane, c)
    = sum_g sum_h F_g,h[lane] * I_g,h[c].
```

The algebraic trace rank is therefore bounded by

```text
T <= sum_g D_g / C.
```

Claims, chunks, blocks, and opening digits enlarge or overlap the sparse lane
factors. They MUST NOT be counted as new coefficient-rank terms when they share
the same group inner trace. In the uniform `D_g = C = 64` case, each group
contributes at most one rank-one term.

The factorization MUST be compiled from the semantic group trace parameters.
It MUST NOT infer rank from the current expanded prepared-lane-term count,
which can duplicate one inner trace across several claims.

During the initial compact-witness scan, form

```text
Q_trace[g,h,c]
    = sum_lane W[lane,c] * F_g,h[lane].
```

The coefficient rounds prove

```text
sum_g sum_h sum_c Q_trace[g,h,c] * I_g,h[c].
```

The target trace prefix state has

```text
C * T <= sum_g D_g
```

extension-field elements.

At coefficient point `a`, evaluate each short coefficient factor once and form

```text
TraceWeight_a(lane)
    = sum_g sum_h I_g,h(a) * F_g,h[lane].
```

### Reuse the range-image folded suffix

The prefix/suffix linear terms MUST NOT trigger a second compact-witness scan.
The nonlinear range-image prover already folds the same witness through every
coefficient challenge. At the coefficient/lane boundary, its retained table is
exactly

```text
W_a[lane] = sum_c eq(a,c) * W[lane,c].
```

The lane-round linear weight is therefore

```text
A(a) * M[lane] + TraceWeight_a(lane).
```

The lane rounds SHOULD fuse this linear weight with the existing range-image
scan over `W_a`. No second witness table or second lane scan is required.

### One specialized initial scan

The candidate implementation SHOULD read each compact witness digit once and
perform three logically separate accumulations:

```text
initial compact i8 scan
    range image: build the current two-round nonlinear batch
    relation:    build Q_relation
    trace:       build Q_trace
```

The implementation MAY use one physical loop, but the three arithmetic kernels
MUST retain separate types and independently testable oracles. A generic
expression engine is out of scope.

### Source-specialized arithmetic

The setup source and witness source require different arithmetic.

For setup coefficients in `F` and transparent weights in `E`, implementations
MUST use extension-times-base operations:

```text
MulBase<F>
MulBaseUnreduced<F>
```

They MUST NOT materialize setup coefficients in `E` solely to call a generic
extension multiplication.

For compact witness digits, implementations MUST preserve the `i8` source and
use extension-times-small-signed-integer accumulation. Positive and negative
products MUST use separate delayed accumulators and reduce at a checked safe
boundary. Implementations MUST NOT lift every digit into `E`.

For log basis 2, the exact balanced digit set is

```text
{-2, -1, 0, 1}.
```

The implementation SHOULD measure a dedicated four-way kernel against the
generic small-scalar kernel. The `-2` case MUST use a canonical exact
small-scalar operation or reduced field addition; it MUST NOT synthesize an
unreduced multiple through an unaudited accumulator recurrence.

For log bases 3 through 6, the implementation MUST measure at least:

1. direct `mul_u64_unreduced(abs(digit))` accumulation; and
2. reduced per-weight multiple tables when the same weight is reused across a
   complete coefficient row.

The retained implementation MAY specialize by certified or observed digit
range only when the selection is transcript-independent, fully priced, and
covered by the same dense differential oracle. Losing kernels and temporary
runtime selectors MUST be removed after selection.

### Required measurement

The current fused two-round implementation remains the A baseline. The
factorized-prefix implementation is the B candidate. Both MUST run the same
complete Stage 2 and end-to-end profiles.

Perfetto output MUST separately attribute at least:

```text
stage2_initial_compact_scan
stage2_range_image_prefix_build
stage2_relation_prefix_build
stage2_trace_prefix_build
stage2_range_image_coefficient_round
stage2_linear_coefficient_round
stage2_lane_round
```

Every profile MUST record `log_basis`, observed digit bounds, `C`, live lane
count, trace rank, nonzero trace-lane terms, source passes, allocation bytes,
and peak RSS where available.

The B candidate is retained only if complete Stage 2 or end-to-end proving
improves without exceeding the existing verifier, proof-size, transcript, or
memory guardrails. The optimization milestone MUST NOT move any sum-check to a
different stage.

## Shared geometry

Let:

```text
coeff_count = min(d_a, d_b, d_d, outgoing_witness_ring_dimension)
```

as checked by `RelationRangeImagePlan`. Let the flat digit witness use the
LSB-first point:

```text
r = (r_coeff, r_lane)
```

where `r_coeff` has `log2(coeff_count)` coordinates. This common split is a
storage and contraction boundary for the witness. It is not the setup
coefficient boundary for every role.

The setup contribution MUST evaluate each role at its own native boundary:

```text
role R uses log2(d_R) low coefficient coordinates
```

The checked setup projection MUST therefore accept the complete relation point
and derive each A/B/D role view from the flat address mapping. It MUST NOT drop
a fixed prefix based only on `d_a` or `coeff_count`.

Nested dimensions MAY reuse prefixes of one alpha-power ladder. The semantic
projection remains role-specific even when allocations are shared.

## Stage 1: range tree plus relation

All non-final range product levels remain unchanged. They continue to use the
equality-factored proofs selected by `DigitRangePlan`.

For a recursive offload edge, the final range leaf MUST be an ordinary
sum-check because the relation term does not share the range term's equality
factor. Let:

- `V_leaf` be the final range-tree input claim;
- `tau_leaf` be its equality anchor;
- `L(S(x))` be the final range-leaf polynomial;
- `V_rel` be the row-batched relation claim, including evaluation trace;
- `m_tau(r)` be the row-batched relation weight; and
- `zeta` be a fresh batching challenge sampled after both input claims are
  transcript-bound.

Stage 1 proves:

```text
V_leaf + zeta * V_rel
  = sum_x [
      eq(tau_leaf, x) * L(S(x))
      + zeta * W(x) * m_tau(x)
    ].
```

For bases 4 and 8, this is the only range-tree stage. For larger bases, only
the final leaf changes shape.

The final Stage 1 point is `r1`. Stage 1 outputs three claims:

```text
range_image_claim = S(r1)
witness_claim     = W(r1)
deferred_setup_claim = sigma_setup(r1)
```

`deferred_setup_claim` is the setup-dependent summand removed from the direct
relation evaluator. The prover binds this claimed scalar after Stage 1. The
verifier uses it with the local relation and evaluation-trace summands to check
the Stage 1 terminal equation, then accepts it only after Stage 2 proves the
setup product. The scalar is deferred, not trusted.

Evaluation trace MUST move with relation. It MUST NOT remain in offloaded Stage
2 or be represented by a second trace protocol.

### Stage 1 proof shape

The proof format MUST distinguish:

- equality-factored range stages; and
- the ordinary offloaded final leaf.

The implementation SHOULD use one explicit typed stage-proof enum. It MUST NOT
encode the distinction through sentinel degrees, empty coefficients, or a
schedule-external mode byte.

Direct folds MAY continue to serialize equality-factored Stage 1 leaves.

## Stage 2: setup product plus witness claim reduction

Stage 2 receives `r1`, `S(r1)`, `W(r1)`, and the deferred setup claim. It runs
two terms over one padded Boolean cube.

### Witness claim reduction

After sampling a fresh `theta`, the witness term proves:

```text
theta * W(r1) + S(r1)
  = sum_x eq(r1, x) * [theta * W(x) + W(x) * (W(x) + 1)].
```

This term simultaneously:

- binds the multilinear range-image table to the committed digit witness; and
- carries the independent Stage 1 witness opening to the same fresh point.

It outputs one opening `W(r_star)`.

### Setup product

The setup term proves:

```text
sigma_setup(r1)
  = sum_j SetupPrefix(j) * setup_weight(r1, j).
```

The setup weight factors into the setup-index component and the role-native
alpha component. Its evaluator MUST use the checked full relation point and
the typed setup-prefix slot selected by the successor edge.

The setup term outputs one opening `SetupPrefix(rho_setup)`.

### Batched Stage 2 geometry

The witness reduction and setup product MAY have different native round
counts. The existing padded-cube lifting rule remains valid:

```text
Lift_n_to_N(f) = 2^(-(N - n)) * f
```

where `N` is the maximum native round count. A fresh batching challenge MUST
combine the two input claims before the first Stage 2 round.

The verifier's final relation MUST check both projected terms and MUST return:

```text
next_witness_point = r_star
setup_prefix_point = rho_setup
```

Both points MUST be projections of the same Stage 2 challenge vector. The
existing suffix opening batch MAY reuse `BatchedStage3Geometry` only after that
type is renamed around its protocol-independent padded-product meaning. No
Stage 3 name or proof field may remain after cutover.

## Direct folds

Direct folds do not carry a setup-prefix opening. They MAY retain the current
architecture:

```text
Stage 1: range tree
Stage 2: relation + evaluation trace + range-image refold
```

The direct relation evaluator includes the setup contribution by scanning the
configured setup envelope.

The direct and offloaded paths MUST share:

- `RelationRangeImagePlan` geometry;
- semantic relation events;
- evaluation-trace inputs;
- witness layout and group/chunk order; and
- verifier setup-weight primitives.

They MUST NOT share mutable prover state or introduce a generic expression
engine solely to hide their different transcript placement.

## Transcript order

The recursive offload transcript MUST use this order:

1. Bind the outgoing witness commitment or terminal state.
2. Bind ring-switch claims and sample the existing relation row challenge.
3. Run all non-final Stage 1 range-tree levels.
4. Bind the final range input claim and row-batched relation claim.
5. Sample the Stage 1 relation batching challenge.
6. Run the ordinary final Stage 1 leaf.
7. Bind `S(r1)` and `W(r1)`.
8. Bind the deferred setup claim.
9. Sample the witness-claim-reduction challenge and the Stage 2 term-batching
   challenge.
10. Run the batched Stage 2 sum-check.
11. Bind `W(r_star)` and `SetupPrefix(rho_setup)`.
12. Pass the shared projected opening state to the successor fold.

Prover and verifier MUST use the same labels and order. The implementation
MUST add logging-transcript parity before deleting Stage 3.

## Proof and state changes

The recursive proof shape changes intentionally:

- Stage 1 gains an ordinary final-leaf variant and carries `W(r1)`.
- Stage 2 carries the setup claim, setup-prefix evaluation, next-witness
  evaluation, and one batched sum-check.
- `FoldLevelProof.stage3_sumcheck_proof` is deleted.
- Stage 3 proof shapes and deserialization branches are deleted.
- suffix state receives both projected Stage 2 openings directly.

No compatibility wrapper or pass-through alias is permitted. This repository
does not promise backward compatibility.

## Mixed ring dimensions

After this cutover, recursive setup offload SHOULD support the same nested
power-of-two role dimensions as direct Stage 2.

The acceptance geometry includes:

```text
(d_a, d_b, d_d) = (128, 64, 32)
```

with an independently selected successor inner dimension. Correctness depends
on role-native setup projection, not equality between predecessor roles and the
successor ring.

The implementation MUST retain the current rejection until all of these pass:

- setup-weight materialization versus succinct evaluation for mixed roles;
- direct versus offloaded relation-claim parity;
- prover/verifier final-relation parity;
- mixed recursive setup E2E; and
- malformed point, slot, and proof-shape rejection.

Mixed-role multigroup and multichunk schedules are not automatically claimed.
They become supported only after planner-authenticated schedules and E2E tests
exist for those combinations.

## Implementation sequence

### Slice 0A: optimize the current linear witness carry

- Replace only the current dense/delayed-prefix Stage 3 witness carry with the
  two-pass prefix/suffix state machine above.
- Add round-by-round differential tests against the existing dense witness
  term, including uneven variable splits, padding, and malformed layouts.
- Require exactly two source-pass spans in the profiling trace.
- Preserve proof bytes, transcript labels, Stage 2, and the setup-product term.

### Slice 0B: optimize the current setup product

- Introduce a base-field setup source view; do not lift the complete setup
  prefix eagerly.
- Retain S0 as the dense differential and performance baseline.
- Implement and measure S1 with `k=2` and `k=log2(d0)`.
- Implement S2 directly from the existing `H[i] * A[j]` rectangular
  factorization, with exactly two base-field setup-source passes.
- Add dense-versus-factorized round-polynomial and terminal-point parity for
  uniform and mixed `(128,64,32)` roles.
- Implement S3 only if S2's `L`-element state remains a measured bottleneck
  and the internal setup-index rank and work bound are explicit in code and
  tests.
- Keep one measured winner and delete losing kernels and selectors.

### Slice 0C: optimize current Stage 2 linear terms

- Compile the exact relation factorization at the checked common coefficient
  count `C`.
- Compile evaluation trace into at most `sum_g D_g / C` coefficient-rank
  terms, merging claims, chunks, blocks, and digits into sparse lane factors.
- Extend the existing initial compact-witness scan to build the range-image
  two-round batch, `Q_relation`, and `Q_trace`.
- Keep the nonlinear range-image state machine unchanged.
- Reuse its coefficient-folded witness at the lane boundary; do not make a
  second compact-witness pass.
- Differentially compare every round polynomial and terminal evaluation with
  the current fused Stage 2.
- Run the required Perfetto A/B matrix across log bases 2 through 6, uniform and
  mixed role dimensions, and representative trace ranks.
- Preserve current proof bytes, transcript labels, claims, and stage placement.

### Slice 1: checked two-stage equations and proof topology

- Add dense-oracle tests for the offloaded Stage 1 and Stage 2 equations.
- Record the exact proof variants, claims, points, and transcript order needed
  by the atomic recursive cutover.
- Keep the current runtime rejection and Stage 3 implementation unchanged.

### Slice 2: offloaded Stage 1

- Add the typed ordinary final-leaf proof variant as part of the working
  recursive path; do not merge a dormant proof variant.
- Prepare relation factorization and evaluation trace before Stage 1.
- Fuse them into the final range leaf.
- Verify the ordinary final leaf and close local relation terms.
- Preserve all earlier range-tree stages.

### Slice 3: offloaded Stage 2

- Reuse the current nonlinear range-image/witness machinery for claim
  reduction without changing its equation.
- Reuse the selected setup-product prover from Slice 0B and the structured
  setup-weight evaluator.
- Batch both terms over one padded cube.
- Return the two projected openings directly to suffix state.

### Slice 4: mixed setup projection

- Replace fixed `d_a` slicing with checked role-native projection.
- Add mixed direct/offloaded parity and E2E coverage.
- Remove the schedule rejection only after these tests pass.

### Slice 5: deletion and performance

- Delete Stage 3 types, modules, labels, proof fields, and sizing branches.
- Delete obsolete producer-side setup mode reconstruction.
- Regenerate proof-size and schedule-shape expectations.
- Run the complete CI feature matrix and path-specific setup-offload tests.
- Benchmark direct and offloaded proving, verification, proof bytes, and peak
  memory against pinned baselines.

Range-image specialization begins only after this protocol cutover is stable.

## Acceptance criteria

- The recursive setup-offload path runs exactly two sum-check stages.
- Relation and evaluation trace occur in the final Stage 1 range leaf.
- Offloaded Stage 2 contains only witness claim reduction and setup product.
- The recursive proof has no Stage 3 field or transcript frame.
- The successor edge is the sole setup-offload topology authority.
- Direct folds retain exact relation, trace, and range-image semantics.
- The setup contribution matches the direct flat matrix scan.
- The witness claim reduction matches a dense materialized oracle.
- Prover and verifier derive identical Stage 1 and Stage 2 points.
- The carried setup-prefix and witness openings share one padded-cube challenge.
- Uniform recursive proofs pass transcript and proof-shape tests.
- Mixed `(128, 64, 32)` recursive setup offload passes end to end before its
  schedule guard is removed.
- Malformed slots, points, proof variants, and missing carried claims return
  `AkitaError` on verifier-reachable paths.
- No unsupported mixed-shape claim is introduced by test-only schedule
  mutation.
- Range-image specialization remains absent from this PR.

## Security considerations

Moving relation changes when its point is sampled, not the relation being
proved. The Stage 1 batching challenge MUST be sampled after both input claims
are bound. Evaluation trace MUST use the same row challenge and relation point
as the other relation rows.

The range-image table and witness are independent multilinear tables away from
the Boolean cube. Stage 2 MUST prove the claim-reduction identity; checking
`S(r1) = W(r1)(W(r1)+1)` directly is unsound.

The setup prefix is transcript-independent, while alpha evaluation is
transcript-dependent. The commitment remains over flat setup coefficients; all
role-native alpha and address projections stay on the weight side.

Removing the mixed/offload guard before role-native projection is verified can
silently omit A/B lane coordinates. The guard is a security boundary, not a
temporary usability limitation.

## References

- `specs/relation-range-image-sumcheck.md`
- `specs/typed-schedule-topology-cutover.md`
- `specs/setup-product-sumcheck.md`
- `specs/batched-stage3-setup-opening.md`
- `specs/setup-layout-repack.md`
- `book/src/how/proving/sumcheck-stages.md`
- Jolt legacy reference implementation:
  `crates/jolt-prover-legacy/src/zkvm/claim_reductions/registers.rs`
  in the sibling Jolt repository; only its linear two-pass witness reduction is
  a reference for this spec.
- Akita paper, “Verifier offloading,” especially “Protocol placement: the
  two-stage form”
