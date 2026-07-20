# Digit-range and relation sum-check pipeline

| Field | Value |
|---|---|
| Author(s) | Quang Dao (protocol and implementation direction); Codex (design synthesis) |
| Created | 2026-07-18 |
| Revised | 2026-07-20; three-PR stack, alpha-first mixed-dimension design, and preservation audit of the prior detailed handoff |
| Status | active |
| PR | [#312](https://github.com/LayerZero-Labs/akita/pull/312) implements the specification and complete Stage 1 cutover |
| PR #312 branch | `quang/plan-digit-range-pipeline` |
| PR #312 base | PR #311, `quang/terminal-direct-ring-relations` at `fad006e2280e880fa16f1cd13b5ea2df599364d0` |
| PR #312 pre-audit head | `c2d9b84d9b89e78dd2e85833b2cd3c95f5b9fe37`; this preservation revision is the next commit |
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

### Preservation audit of the previous revision

The 2026-07-20 rewrite shortened the document by removing the old packet/branch workboard.
That change is acceptable only if it removes delivery bureaucracy rather than technical
memory. The table below is the disposition ledger for every major section of the prior
4,297-line revision. A future rewrite must update this ledger instead of silently deleting
content.

| Previous section | Disposition in this revision |
|---|---|
| Problem statement, terminology, and naming | Preserved in `Decision summary`, `Vocabulary and naming contract`, and `Current problems to delete` |
| Open-PR and conflict audit | Condensed into `Dependency and conflict policy`; exact heads remain recorded, while stale historical conflict counts are intentionally not normative |
| Single-source/no-wrapper rules | Preserved as global and per-PR requirements |
| Current Stage 1/2 implementation audit | Stage 1 is described as the deleted architecture and final #312 implementation; Stage 2's cartesian-product branches are enumerated in its stacked-PR section |
| `DigitRangePlan` and range topology | Preserved and implemented in #312 |
| `FoldCheckPlan` and topology authority | Restored below as a final-offload-PR contract; it is intentionally not claimed as #312 implementation |
| Flat witness domain and point mapping | Preserved and expanded below; public x/y geometry remains rejected |
| Exact-prefix/nonzero-tail derivation | Restored below with the explicit split-equality suffix formula |
| Stage 1 compact/class-indexed prover design | Preserved in the #312 lifecycle, selected-kernel tables, module map, and evidence ledger |
| Stage 1 verifier design | Restored below as an explicit verifier and malformed-shape contract |
| Direct Stage 2 equation and cleanup | Preserved and expanded into the complete second-PR handoff |
| Recursive joint Stage 1 leaf | Preserved and expanded in the final offload PR handoff |
| Separate/batched recursive Stage 2 | Restored below with lift equations, transcript order, proof types, proof-size accounting, and selection rules |
| Mixed-dimension common-base factorization | Preserved and expanded with semantic events, alpha-zero behavior, overlaps, typed points, and setup addressing |
| Planner eligibility, setup slots, and cache cap | Restored below; kernel choice remains outside planner metadata |
| Upstream digit emission, fold-grind, and verifier-kernel ideas | Preserved in `Deferred optimization backlog`; intentionally outside the three committed PRs |
| Fourteen-entry implementation packet stack | Superseded by the three semantic PRs. Packet identifiers and half-cutover branches are intentionally deleted |
| Benchmark program and tracing | Preserved per owned PR and expanded with the #312 evidence sources |
| Correctness/security matrix | Preserved in `Test oracles`, per-PR acceptance criteria, and verifier no-panic rules |
| Risk register | Restored near the end of this document and rewritten around the three PR boundaries |
| Rejected designs and definition of done | Preserved and updated for the alpha-first Stage 2 design |

The old revision remains recoverable in git at
`f5520cd8:specs/digit-range-pipeline-refactor.md`. It is historical evidence, not a second
implementation authority. Any item classified as “restored” above must be independently
understandable in this document without consulting that revision or this conversation.

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
| checked live witness layout plus point/address map | `WitnessDomain` |
| checked point in the physical witness domain | `WitnessDomainPoint` |
| Stage 1 topology, roots, degrees, and child order | `DigitRangePlan` |
| complete direct/recursive non-terminal proof shape | `FoldCheckPlan` |
| compact signed digits and range-image class access | `CompactDigitSource` |
| direct Stage 1 output point | `RangeCheckPoint` |
| recursive joint Stage 1 output point | `RangeRelationPoint` |
| current direct fused proof after the Stage 2 rewrite | `RelationRangeImageProof` |
| current direct fused prover after the Stage 2 rewrite | `RelationRangeImageProver` |
| common-dimension relation factorization | `CommonAlphaRelationWeights` |
| independently additive trace representation | `TraceWeightState` |
| setup coefficient address space | `SetupCoefficientDomain` |
| fresh setup-prefix opening point | `SetupOpeningPoint` |
| fresh next-witness opening point | `NextWitnessPoint` |
| setup-product statement and checked route | `SetupContributionPlan` |
| complete setup contribution at one relation point | `setup_contribution_eval` |
| recursive setup proof | `SetupContributionProof` |
| recursive witness/range-image reduction | `RangeImageConsistencyProof` |
| consuming unresolved recursive Stage 1 equation | `DeferredRangeRelationCheck` |

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

The checked domain authority has the conceptual shape:

```rust
pub struct FlatBooleanDomain {
    live_len: usize,
    domain_len: NonZeroPowerOfTwo,
    num_vars: usize,
    variable_order: LsbFirst,
}

pub struct WitnessAddressPlan {
    flat: FlatBooleanDomain,
    layout: WitnessLayout,
    transcript_point_map: TranscriptPointMap,
}
```

The exact production names may reuse existing checked layout types rather than adding
`WitnessAddressPlan`. The requirements are not optional:

- `domain_len = 2^num_vars` is checked without overflow, and
  `0 < live_len <= domain_len`;
- canonical current producers choose the minimal valid power-of-two domain; a future
  authenticated schedule may intentionally choose a wider capacity, and that choice then
  fixes the proof rounds and public zero suffix;
- the protocol address is the raw physical coefficient index;
- `live_len..domain_len` is the public zero suffix;
- the physical segment layout is sorted, nonoverlapping, in range, and mapped through one
  canonical address helper;
- the transcript map explicitly associates challenge slots with raw address bits and
  reproduces the current homogeneous order;
- row-family dimensions and algebra live in relation plans, not in the Boolean-domain
  type;
- no caller manually slices, concatenates, or reverses a raw challenge vector.

`RangeCheckPoint`, `RangeRelationPoint`, `NextWitnessPoint`, and `SetupOpeningPoint` are
distinct checked types even when their underlying field vectors happen to have equal
length. A private common-dimension view may split a witness point into its first `k`
coefficient coordinates and remaining lane coordinates. This view is not a second point,
protocol domain, or serialized object.

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

An exact-prefix table has semantics equivalent to:

```rust
pub struct ExactPrefixTable<T> {
    domain_len: NonZeroPowerOfTwo,
    explicit: Vec<T>,
    default: T,
}
```

Adjacent affine folding handles an odd final explicit item against `default`, materializes
`ceil(explicit.len()/2)` rows, halves `domain_len`, preserves the same default, and rejects
invalid sizes. It is not a protocol-wide `TailPolicy` abstraction.

For a split-equality scan whose first omitted pair index is `P` and whose remaining
tables are indexed as `j = j_high * num_first + j_low`, the omitted mass is

```text
h0 = P / num_first
l0 = P % num_first

mass_from(P) =
    e_second[h0] * sum(e_first[l0..])
  + sum(e_second[h0+1..]) * sum(e_first[..]).
```

When `P == num_first * e_second.len()`, define `mass_from(P) = 0` before computing
`h0`; the displayed indexed formula applies only while `P` is inside the table.

The helper consumes only `GruenSplitEq::remaining_eq_tables()`. It does not include the
current scalar or current linear equality factor, which belong to the caller's
eq-factored round. The caller multiplies the omitted mass by the constant local round
polynomial from the derived default exactly once.

Property tests compare this formula and every exact-prefix fold with fully padded tables
for all short positive live lengths, odd/even boundaries, random nonzero defaults, and
every remaining bind. This is load-bearing for Stage 1 product nodes and the future
recursive joint leaf.

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
- proof/transcript epoch tests, malformed-shape tests, a durable basis benchmark, and
  tracing spans;
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

For the record, if quartic leaves are retained and only internal nodes become binary, the
exact comparison is:

| LB | Selected substages | Binary/quartic substages | Selected child claims | Binary child claims | Wire delta |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 0 | 0 | 0 |
| 3 | 1 | 1 | 0 | 0 | 0 |
| 4 | 2 | 2 | 2 | 2 | 0 |
| 5 | 2 | 3 | 4 | 6 | `2 * extension_element_bytes` |
| 6 | 3 | 4 | 10 | 14 | `4 * extension_element_bytes` |

Both topologies still send `2*(LB-1)` range coefficients per witness round. A fully
binary tree with quadratic leaves raises child claims to `2^(LB-1)-2`, adds substages,
and doubles the selected high-basis one-round state peaks, while conserving the complete
recursive round-message count at `2*LB-1`. This is why a potentially faster binary
microkernel does not justify a binary proof topology.

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

Delayed accumulation has a correctness contract, not merely a fast-field dispatch. The
inner products reduce at every split-equality inner-block boundary. Delayed product sums
are used only when the field declares `DELAYED_PRODUCT_SUM_IS_EXACT` for that bounded
operation; every other field uses canonical accumulation with the identical block
factorization. Safety is never inferred from the domain size, a release build, or the
absence of observed overflow.

Pre-challenge digits and range-image classes are compact integers, but post-challenge
lanes are field elements. The magnitude ledger explains why large-basis parent arithmetic
must not be forced into fixed-width integers:

| Quantity | Basis 16 | Basis 32 | Basis 64 |
|---|---:|---:|---:|
| maximum range image | 56 | 240 | 992 |
| maximum quartic-leaf endpoint | about 23 bits | about 32 bits | about 40 bits |
| maximum one-round root coefficient | about 44 bits | about 123 bits | about 303 bits |
| maximum two-round root coefficient | about 46 bits | about 127 bits | about 305 bits |

An LB6 four-leaf node itself reaches roughly 158 bits. Therefore no LB5/LB6 parent or
pair LUT uses `i64`, `u64`, `i128`, or `u128`; no generic multi-limb integer accumulator
lands without a separate proved bound and measured win. LB4 narrow coefficients remain a
disposable experiment only, and randomized leaf batching remains field-valued.

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

Measurement provenance is part of the decision: node evaluation
`91d05acaf241f562c3d44fb796f71c68f1d61936` was compared with ordered pairs
`999678ce21af0a758816ccce32c6430e4fe3b999`; the optimized one-round successor was
`b9e44a5095160889f2156d592511a027589cb2fd`; the selected two-round candidate was
`7b94cb8b9299d8bdaec12ea473fec16a5645b964`. Low-basis checkpoints were `c2736370`
before LB3 three-round work,
`fae8d871` as the measured LB3 candidate (`77e7c870` integrated), and `0cdcdf40` as the
measured LB2 candidate (`b74220cf` integrated). These are historical experiment commits,
not alternate production branches.

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

The verifier contract is more specific than “consume `DigitRangePlan`”:

- validate the supported basis, number of substages, child count and order, degree, and
  round count from the plan before replay or allocation;
- borrow point and child-claim slices and use fixed small arrays for plan-bounded batching;
- absorb extension-field child claims through the same semantic transcript choreography
  as the prover;
- evaluate the plan's field-lifted quadratic or quartic leaf polynomial with the canonical
  univariate evaluator;
- reject every malformed count, degree, point width/order, serialized length, or temporary
  allocation request with `AkitaError`;
- never reconstruct topology from basis thresholds or a received vector length;
- never reorder a raw challenge vector. The checked `DigitRangeEqualityPoint` owns the
  LSB-first physical-address interpretation.

Proof-supplied lengths are capped before allocation. Verifier-reachable code contains no
new `panic!`, `assert!`, `unwrap`, unchecked indexing, or attacker-sized allocation. The
terminal path does not construct an empty digit-range plan or proof: #311's
`TerminalLevelProof` remains a separate plan, wire, and verifier authority.

### #312 evidence ledger

The executable protocol epoch originated at the literal #311 commit
`bc959ef34572aee143ba0114094b0b4212b4e111`. Restacking #312 on later #311 heads does not
alter that provenance: a wire-preserving #311 change must continue to satisfy the same
fixtures, while an intentional protocol change replaces them atomically and records its
new source commit.

`digit_range_protocol_epoch` serializes each equality-factored proof followed by its child
claims and `range_image_evaluation`. It also records the complete ordered
`LoggingTranscript`, replays through the verifier, and compares the final point and
evaluation. Its current expectations are:

| Basis | Bytes | Events | Proof digest | Event digest | Output-point digest | Range-image evaluation |
|---:|---:|---:|---|---|---|---|
| 4 | 144 | 9 | `abd1266b50d20cfe7b9a3ddf83e9b544` | `4491fc78622c42a3b932e6636f0fc667` | `7d9fafb86c4b931cffd679891031586b` | `a3597f88c2a199b05fe5c285c9366d15` |
| 8 | 272 | 9 | `9cda2cf6600a1cc46240993b5cf15b92` | `41f764d51da50603e671b3b9c4a57a8e` | `a1a125181986353fb9d84f93764b21d0` | `0092bf8c55d7731e323a60f3f9b603c7` |
| 16 | 432 | 21 | `3e0cc8bac0d1349a16c8e7d58fb0f3ee` | `250eb155de29e174e4e385bb21533a3a` | `a740f51168e899129c346b13f027e4ea` | `119a3282dc19d6bd68b08929aa82b8f4` |
| 32 | 592 | 23 | `fd0d48a7b9b1cf9e772df377a0c67849` | `f171889729c52a1829dc8949e1898c20` | `d7178d608a5edb274e90e2b583104b78` | `7335696cb66eb923716ac7c085344918` |
| 64 | 816 | 39 | `efb26d21239c4bff5d221216ab092e79` | `c4f7688506a87b26db7ce8e547c47e8e` | `97aa7de0a6605ff6eeeea0ec6afbe514` | `b62ebc70f19c7bcd71fa9c466d5c86b9` |

`fold_protocol_epoch` extends that check through the generated direct and recursive fold
schedules. Per-level Stage 1 payloads are checked separately so a terminal or envelope
change cannot conceal a range-proof delta.

| Fixture | Range bases | Complete proof bytes / digest | Events / digest | Terminal bytes / digest |
|---|---|---|---|---|
| direct-to-terminal, nv12 | `8` | 57,250 / `3a155ec04047e9942f2eb1685e778e50` | 164 / `57046ae9d1a2a2b0a63e1ecd34bc6dea` | 54,286 / `5a26d324461406760daa77a6e3009858` |
| recursive-nonterminal, nv20 | `64,64,64` | 74,231 / `7caa4641e201f1be5a6437f5fa3e7535` | 677 / `6fa3d54d166f79a4c4fe7054c5d4ed84` | 57,707 / `dd68f68783534944dad6c7a213866d45` |

During kernel selection, a one-off serial counting-allocator harness compared the
post-#311 baseline with #312 on the same full-prefix scenarios. That development harness
is deliberately not retained: ongoing performance coverage belongs to the checked-in
Criterion basis benchmark, profile CI, and coarse tracing spans. The captured allocation
results remain useful historical evidence:

| Basis | Baseline allocations | #312 allocations | Baseline bytes | #312 bytes |
|---:|---:|---:|---:|---:|
| 4 | 171 | 170 | 1,149,107 | 1,148,819 |
| 8 | 215 | 214 | 1,161,629 | 1,161,341 |
| 16 | 323 | 319 | 12,646,816 | 12,646,384 |
| 32 | 361 | 355 | 21,039,360 | 21,038,800 |
| 64 | 552 | 542 | 46,237,568 | 46,236,744 |

The high-basis three-quarter cells had the same allocation counts as their corresponding
full padded domains. The one-off callbacks did not report a synthetic peak-live value:
deallocation of an object created before the measured interval could not be attributed
soundly. Peak evidence instead used explicit retained-state formulas and fresh-process
maximum RSS. Future allocation investigations should use disposable measurement tooling
or an established repository-wide profiler rather than adding a second Stage 1 benchmark
driver.

The canonical Stage 1 tracing vocabulary is:

```text
digit_range_prepare_compact_source
digit_range_prove
digit_range_direct_leaf
digit_range_product_substage {stage_index, arity}
digit_range_polynomial_leaf
digit_range_build_node_table
digit_range_build_pair_coefficients
digit_range_product_initial_round
digit_range_product_materialized_round
digit_range_leaf_initial_round
digit_range_leaf_materialized_round
digit_range_prepare_deferred_second_round
digit_range_build_second_round_quartet_table
digit_range_build_folded_pair_table
digit_range_fold_lanes
digit_range_fold_range_image
digit_range_direct_leaf_round {round, phase}
digit_range_direct_leaf_fold
```

These spans identify phase owners. They do not enter address, pair, class, lane,
coefficient, or Rayon-item loops. A future rename updates profiling documentation and
queries in the same diff; it must not leave duplicate old/new spans around one owner.

### #312 intended diff surface

The full merge-base diff may touch only these responsibilities:

| Surface | Allowed #312 work |
|---|---|
| `akita-prover::protocol::sumcheck` | Stage 1 range cutover and shared mechanics required directly by it |
| `akita-types::proof::stage1` and sizing | canonical range topology, descriptive fields, shape validation, byte accounting |
| `akita-verifier::stages::stage1` | replay through `DigitRangePlan`, descriptive outputs, malformed-shape rejection |
| Stage 2 call boundary | mechanical `range_image_evaluation` naming and Stage 1 output adaptation only |
| PCS tests/benches/profile report | epoch fixtures, differential tests, durable basis benchmarks, and report field naming |
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
    + exact_fringe_weight(z)
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

The aligned factorable portion of the non-trace linear relation compiles into

```text
RelationWeight(g * lane + coefficient)
  = CommonAlphaFactor(coefficient) * RelationLaneWeights(lane),

CommonAlphaFactor = [1, alpha, ..., alpha^(g-1)].
```

The complete non-trace weight is

```text
non_trace_relation_weight(z)
  = CommonAlphaFactor(coefficient(z)) * RelationLaneWeights(lane(z))
    + ExactFringeWeight(z).
```

`ExactFringeWeight` is absent for fully aligned layouts. When present, it is a checked,
bounded sparse provider for only the nonfactorable boundary addresses; it is never an
`N`-element fallback.

Role-specific exponent resets, quotient denominators, row challenges, group weights,
setup amplitudes, and overlaps are absorbed additively into `relation_lane_weights`.
They do not break the common low factor. In particular, for mixed `128/64/32`, the
common factor has length 32—not 128, and not a dense full-domain fallback.

The builder must prove:

- all role dimensions are nonzero nested powers of two;
- `g` divides each role dimension;
- every factorized interior preserves the low `log2(g)` physical coefficient bits and has
  a `g`-aligned exponent phase;
- any unaligned boundary is isolated into an explicitly bounded exact sparse fringe;
- every role-local exponent reset is either `g`-aligned in the factorized interior or
  represented exactly in that fringe;
- overlapping contributions add to the same lane weight exactly once;
- no address outside the live witness receives a relation contribution.

Failure to cover the polynomial with the checked factorized interior plus bounded fringe
is an input/setup error. The production prover does not silently fall back to an
`N`-element dense relation table. A dense exact evaluator exists only as a test oracle.

The shared semantic compiler emits checked physical contributions before choosing a
storage form:

```rust
struct RelationWeightEvent<E> {
    physical_start: WitnessCoefficientIndex,
    length: usize,
    alpha_exponent_start: usize,
    scalar: E,
    exponent_pattern: RelationExponentPattern,
}
```

For the common linear case, `physical_start = g * p0`,
`alpha_exponent_start = g * e0`, and local offset `t = g * h + coefficient` give

```text
scalar * alpha^(alpha_exponent_start + t)
  = alpha^coefficient
      * [scalar * (alpha^g)^(e0+h)].

relation_lane_weights[p0+h]
  += scalar * (alpha^g)^(e0+h).
```

The compiler starts powers at the multiplicative identity and uses repeated
multiplication; it never divides by `alpha`. `alpha = 0` is valid and is covered by the
dense differential oracle. Periodic
role resets are emitted as checked additive high-lane patterns. If a source interval has
an aligned factorized interior and a genuine partial fringe, the compiler may split it
into factorized and exact sparse events; it may not expand the entire polynomial into a
dense fallback. Each physical contribution has one owner, and overlapping E/T/Z/R or
setup contributions use `+=` exactly once.

After compilation, retain exactly one high-lane representation: coalesced dense lane
weights, checked spans, or sparse runs. Source events are dropped after coalescing, and no
provider stores both factorized and dense copies of the same contribution. Final
evaluation computes `common_factor_eval * lane_weight_eval + fringe_eval` exactly once.

One semantic emitter is the source for all of the following:

- the test-only dense exact-live vector;
- the production common-factor/lane-weight compiler;
- direct verifier evaluation at a typed point;
- setup-contribution attribution and replay tests.

There must not remain separate public `compute_relation_matrix_col_evals` and
`compute_relation_weight_evals` authorities. Representations may differ, but the emitted
polynomial and row-family rules do not. The semantic emitter and checked verifier
evaluator belong in a shared types/geometry layer; the prover module owns only prepared
storage, compact scanning, and folding.

The emitter must explicitly cover every current family rather than relying on a generic
callback:

- A/E consistency and setup-D contributions may overlap one physical interval and add;
- T/B is analogous;
- Z and quotient-R exponents begin at zero within their native row families;
- quotient denominators, gadget, tau, row, setup-ring, group, and challenge weights are
  scalar amplitudes;
- each native role reset is represented in its checked high-lane pattern.

Adding a row family requires extending this closed semantic match and its dense oracle.
An open-ended expression graph or public provider callback is not the replacement.

### Binding order

The prover always binds the `k = log2(g)` common coefficient dimensions first. This is
both the canonical LSB-first physical order and the optimal order for the factorization.

During those rounds:

- adjacent witness values belong to one common-dimension lane;
- `relation_lane_weights[lane]` is constant across the entire low-coordinate block;
- `common_alpha_factor` folds from length `g` to one scalar;
- relation work uses compact signed digits;
- the range-image term uses the same compact digits and Stage 1 point equality;
- an exact fringe provider, when present, is accumulated additively in the same scan;
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
- optional `ExactFringeWeightState`;
- current claim-recovery data;
- any selected compact pair/quad/octet cache.

`FoldedSuffixState` owns:

- one folded field-valued digit-witness table;
- the partially folded common-alpha vector and number of common coordinates remaining;
- one folded relation-lane table;
- one folded trace state only when trace has not already collapsed into lane weights;
- one folded exact-fringe state only when the checked layout contains a fringe;
- the range-check equality state;
- current claim-recovery data.

There is exactly one compact-to-field transition. It materializes the folded digit witness
at `N/2^r` after the selected deferred prefix of `r` challenges and does not retain
compact-derived field tables that are no longer used. Since normally `r < k = log2(g)`,
materialization does **not** imply that the common alpha factor is already scalar. The
remaining lifecycle is:

```text
compact common-coordinate prefix          rounds 0 .. r
field-valued common-coordinate suffix      rounds r .. k
field-valued lane-coordinate suffix        rounds k .. num_vars.
```

During `r..k`, fold the digit witness, range equality, trace state, and partially folded
common-alpha vector. Do not fold `relation_lane_weights`: it is constant on those
coordinates. After challenge `k-1`, the alpha factor becomes `common_alpha_eval`; only
then do lane-coordinate rounds begin folding relation lane weights. This is one state
machine with an explicit phase counter, not three drivers or stage-named wrappers.

### Relation arithmetic inside the low-coordinate scan

For one witness pair `w(T) = w_0 + T * delta_w` and alpha pair
`a(T) = a_0 + T * delta_a`, the non-trace relation polynomial before the lane weight is

```text
w(T) * a(T).
```

Accumulate its three coefficients for all pairs in one lane-aligned subreduction first,
then multiply the three lane totals by `relation_lane_weights[lane]` once. A parallel or
split-equality block that crosses a lane boundary is split or carries distinct per-lane
totals; it never applies one lane multiplier to a mixed block. Do not form
`a_endpoint * relation_lane_weight` for every witness endpoint as the current code does.

Compact rounds use signed/unreduced accumulators with a proved bound. Field-valued later
rounds use the canonical delayed-product contract where available and canonical field
accumulation otherwise. The serial and parallel paths call the same block reducer; they
do not contain copies of the equation.

Fringe and trace coefficients are separate additive accumulators in the same witness
scan. They are not multiplied by `relation_lane_weights`, and the common-factor block
optimization does not claim to accelerate them.

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

Trace and a checked sparse fringe are the only relation addends not required to share the
common alpha factor. Trace remains semantically distinct and explicit:

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

After the compact prefix, one canonical field-round implementation scans adjacent folded
witness values. Its phase is explicit:

- in remaining common-coordinate rounds it uses the partially folded alpha factor and
  leaves relation lane weights unbound;
- in lane-coordinate rounds it uses scalar `common_alpha_eval` and folds the relation
  lane weights under every challenge.

The factorized relation formula differs across those phases:

```text
remaining common-coordinate round:
  digit_witness(T)
    * current_common_alpha(T)
    * constant_relation_lane_weight

lane-coordinate round:
  common_alpha_eval
    * digit_witness(T)
    * relation_lane_weight(T).
```

Range-image, exact-fringe, and trace terms are added in both phases. The complete semantic
sum in the lane-coordinate phase is:

```text
range equality * witness * (witness + 1)
+ common_alpha_eval * witness * relation_lane_weight
+ witness * exact_fringe_weight
+ witness * trace_weight.
```

Fold witness, range equality, exact-fringe state, and trace state once per challenge;
fold relation-lane weights only in lane-coordinate rounds. A fused
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

+ next_witness_eval * exact_fringe_weight_eval(next_witness_point)

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

The common-factor view of a typed witness point is private and checked:

```text
RelationWeight(point)
  = MLE(CommonAlphaFactor, point.common_coordinates)
      * MLE(RelationLaneWeights, point.lane_coordinates)
    + exact_nonfactorable_weight(point).
```

`exact_nonfactorable_weight` contains trace and any compiler-proved sparse fringe; it is
not a second dense relation table. The prover, verifier, and setup builder call the same
checked view rather than each slicing `point[..log2(g)]` independently.

Mixed-dimension planner rollout is not authorized merely because the prover accepts a
tuple. Before a schedule emits one, relation construction, every Stage 2 round, trace,
direct setup replay, the current recursive setup boundary, multi-group/chunk addressing,
proof sizing, and prepared-cache accounting must agree. The schedule preview compares a
homogeneous baseline, unrestricted mixed candidates, and a cache-capped mixed set:

```text
mixed_prepared_cache_bytes <= min(
    baseline_prepared_cache_bytes * 5 / 4,
    baseline_prepared_cache_bytes + 256 MiB
).
```

This is a schedule resource bound, not a serialized CPU-kernel selector. A mixed tuple
that fails it remains unscheduled even if its isolated sum-check is faster.

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

The reusable scalar is defined completely—not as a common-factor fragment:

```text
setup_contribution_eval
  = sum_j SetupCoefficient(j)
      * SetupRelationWeight(
          j;
          relation_point,
          tau_relation,
          alpha
        ).
```

Here `tau_relation` denotes the checked relation-row challenge owned by the semantic
relation plan, not a caller-sliced witness point.

For `j = g*J + coefficient` and `relation_point = (r_common, r_lane)`, an internally
factorized setup span has

```text
SetupRelationWeight(g*J + coefficient; relation_point)
  = MLE(CommonAlphaFactor, r_common)
      * CommonAlphaFactor[coefficient]
      * SetupLaneWeight(J; r_lane).
```

The first factor is already part of the full scalar. Stage 1 must not multiply the
received `setup_contribution_eval` by it again. At a fresh setup opening point
`(rho_common, rho_lane)`, the setup-product final weight contains the distinct factor
`CommonAlphaFactor(rho_common)` exactly once.

`SetupCoefficientDomain` is a separate flat Boolean domain with an exact live prefix,
padded zero suffix, and LSB-first binds. It is not a projection of `WitnessDomain`.
Overlapping A/B/D setup views add at one physical setup-coefficient address.

One checked role-address helper is shared by relation-event emission, direct setup replay,
setup-weight construction, and recursive proof preparation. For native role dimension
`D`, common dimension `g`, and `u in 0..D/g`, its semantic map is

```text
physical_common_lane
  = witness_column * (d_a/g)
  + native_role_subcolumn * (D/g)
  + u.
```

The helper receives a typed native setup column, checks divisibility, subcolumn bounds,
overflow, and final domain membership, and returns a typed index. Neither prover nor
verifier reconstructs this formula from matrix-column order. D/B columns retain their
A-witness role-subcolumn mapping; A uses native role subcolumn zero; Z contributes at
every required A sublane.

Do not name the setup coefficient table, setup weight, or setup claim `S`. The production
vocabulary is `setup_coefficients`, `setup_relation_weight`, and
`setup_contribution_eval`.

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
- Existing homogeneous schedules preserve the incoming proof bytes, transcript order,
  challenges, and final claim. A newly enabled mixed schedule binds an explicit mixed
  descriptor and establishes its own versioned fixture; it cannot “match” a previously
  nonexistent proof.
- `ring_bits == 0` is gone as a mixed-dimension sentinel.
- Public/current constructors do not accept independent `live_x_cols`, `col_bits`, and
  `ring_bits` geometry.
- Common alpha coordinates bind first and collapse to one scalar before lane binding.
- Relation lane weights occupy at most `N/g` field elements plus explicitly bounded exact
  sparse fringes. If all scheduled layouts prove full common-base alignment, the stronger
  `N/g` bound is enforced and the fringe representation is absent.
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

### One complete fold-check plan

The atomic cutover introduces one checked `FoldCheckPlan` as the complete non-terminal
proof-shape authority:

```rust
struct FoldCheckPlan {
    digit_range: DigitRangePlan,
    witness_domain: WitnessDomain,
    topology: FoldCheckTopology,
}

enum FoldCheckTopology {
    DirectSetup(DirectFoldCheckPlan),
    RecursiveSetupOffload(RecursiveSetupOffloadPlan),
}

enum RecursiveClaimReductionPlan {
    Separate(SeparateReductionShape),
    Batched(SetupWitnessBatchGeometry),
}
```

The reduction enum shown here is the final post-Batched form. If the initial cutover lands
only Separate, `RecursiveSetupOffloadPlan` owns `SeparateReductionShape` directly; it does
not add a one-variant enum in anticipation.

The exact Rust fields should reuse existing checked schedule and domain types. The
authority contract is fixed: this one plan determines topology, proof fields, subproof
degrees and round counts, child/scalar order, headerless serialization, transcript frame
order, size calculation, allocation caps, setup slot and opening route, and whether a
recursive reduction is separate or batched. Prover, verifier, serializer, deserializer,
security sizing, and planner query it directly. They do not infer shape from received
lengths, serialize a variant tag, or maintain parallel formulas.

Constructing a `FoldCheckPlan` binds the protocol version, flat coordinate order, role
dimensions, challenge/lift convention, recursive reduction shape, exact setup slot and
commitment parameters, and outgoing opening route. It is impossible for terminal levels:
#311's terminal plan and proof remain separate and sum-check-free.

### Recursive Stage 1 equation

All range-product layers before the final leaf remain the #312 equality-factored product
subchecks. Let `leaf_input_claim` and `leaf_anchor` be the claim and point from that
prefix, and let `LeafBatch` be the plan-derived quadratic or quartic range-image leaf.
Define

```text
linear_relation_trace_claim = linear_relation_claim + trace_claim,
```

where `linear_relation_claim` includes every non-trace relation contribution (including
the exact fringe and setup terms) and `trace_claim` is the one carried trace-opening
claim. After binding both, sample `range_relation_batch_challenge` and prove

```text
leaf_input_claim
  + range_relation_batch_challenge * linear_relation_trace_claim

= sum_z [
    Eq(leaf_anchor,z) * LeafBatch(range_image(z))
  + range_relation_batch_challenge
      * digit_witness(z)
      * (CommonAlphaFactor(coefficient(z))
           * RelationLaneWeights(lane(z))
         + ExactFringeWeight(z)
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
- accumulates the exact fringe if present;
- accumulates trace if present.

After the first challenge, one compact traversal materializes two independent lanes:

```text
folded_range_image
folded_digit_witness.
```

The representations diverge after that transition, but they continue under the same
challenge sequence and combined claim. Two scans pretending to be a fused round, or one
fused first round followed by independent sum-checks, is not acceptable.

The joint leaf reuses the canonical `CommonAlphaRelationWeights`,
`ExactFringeWeightState`, `TraceWeightState`, compact signed-relation coefficient
primitives, and compact-to-field transition mechanics from the previous PR.
`RangeRelationLeafProver` nevertheless owns its one joint round
polynomial and one transcript lifecycle; it does not wrap or invoke
`RelationRangeImageProver`, whose equation and degree are different.

### Recursive Stage 1 output

Stage 1 returns at one `RangeRelationPoint`:

```text
range_image_eval
digit_witness_eval.
```

The verifier keeps a consuming deferred check whose final equation requires the complete
`setup_contribution_eval`. It cannot accept the Stage 1 leaf until Stage 2 binds that setup
claim. Consuming ownership prevents omission or double application.

Write `non_setup_relation_weight_eval` for the factorized relation, exact fringe, and
trace contribution already available without committed setup. The consuming check closes
exactly this equation:

```text
final_joint_leaf_claim
  = Eq(leaf_anchor, range_relation_point)
      * LeafBatch(range_image_eval)
    + range_relation_batch_challenge
      * digit_witness_eval
      * (non_setup_relation_weight_eval + setup_contribution_eval).
```

`setup_contribution_eval` is the full scalar defined by the second PR. It is neither
omitted nor multiplied by `common_alpha_eval` again. The verifier cannot continue to the
claim-reduction frame until the consuming `DeferredRangeRelationCheck::close(...)`
succeeds, and no accessor exposes an already-consumed check.

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
Neither fresh Stage 2 point is a slice, suffix, or reinterpretation of the Stage 1
`RangeRelationPoint`; only a checked Batched geometry may project its own fresh padded
point to its two native suffixes.

### Separate and batched realizations

`Separate` runs two native proofs in a fixed order:

1. `SetupContributionProof`, standard degree 2 over `SetupCoefficientDomain`, ending at
   `setup_prefix_eval` and an independent `SetupOpeningPoint`;
2. after that frame closes, sample `range_image_binding_challenge` and run
   `RangeImageConsistencyProof`, equality-factored inner degree 2 over `WitnessDomain`,
   ending at `next_witness_eval` and an independent `NextWitnessPoint`.

There is no setup/range batch challenge in `Separate`. Its opening router must carry both
typed points; neither is coerced into a suffix of the other. For native setup and witness
round counts `lambda` and `mu`, it sends `2*(lambda+mu)` round elements.

`Batched` combines the standard degree-2 setup product with the standard degree-3 form of
the range-image consistency reduction. It is an adaptation of the old unequal-domain
batch geometry, not an invitation to preserve the numeric Stage 3 API. Let

```text
n_setup   = lambda
n_witness = mu
n         = max(lambda, mu)
delta_i   = n - n_i
lift_scale_i = 2^(-delta_i).
```

Term `i` is lifted over `delta_i` independent leading coordinates. In an inactive leading
round it emits the constant polynomial `current_claim / 2` and leaves its native table
untouched. In an active round it emits
`lift_scale_i * native_round_polynomial`, applying the persistent scale exactly once.
The final native MLE evaluation is likewise multiplied by `lift_scale_i` exactly once;
the table and initial native claim are not pre-scaled and then scaled again.

After binding both native claims and both domain descriptors, sample
`setup_range_binding_batch_challenge` and prove

```text
setup_contribution_eval
  + setup_range_binding_batch_challenge * range_image_consistency_claim

= sum_u [
    Lift_setup(SetupCoefficient * SetupRelationWeight)(u)
  + setup_range_binding_batch_challenge
      * Lift_witness(RangeImageConsistency)(u)
  ].
```

`SetupWitnessBatchGeometry` owns native round counts, inactive-prefix deltas, lift scales,
and checked suffix projections from the one fresh padded point to `SetupOpeningPoint` and
`NextWitnessPoint`. Equal padded lengths never authorize challenge reuse by themselves.

The final batched equality is

```text
final_stage2_claim
  = lift_scale_setup
      * setup_prefix_eval
      * setup_relation_weight_eval(
          setup_opening_point,
          range_relation_point
        )
    + setup_range_binding_batch_challenge
      * lift_scale_witness
      * Eq(range_relation_point, next_witness_point)
      * [
          range_image_binding_challenge * next_witness_eval
        + next_witness_eval * (next_witness_eval + 1)
        ].
```

The initial cutover may land `Separate` alone if it is the complete selected production
shape. `Batched` may land in the same PR or an immediately stacked capability PR only
after its complete-size and performance gate passes. No unused enum variant, dormant
proof field, serialization branch, or forwarding accessor is allowed.

### Setup slot, opening route, and fail-closed eligibility

Each offloaded level selects the least committed setup-prefix slot that covers its active
footprint and discharges that opening at the immediately following fold. At most one setup
opening is outstanding at a recursive boundary. There is no shared largest-prefix opening
across levels: later active footprints shrink, and forcing the largest object through
every level destroys the intended setup/witness cost balance.

Eligibility derives `g` from authenticated role dimensions and then validates the exact
slot identity, `d_setup`, live coverage, padded domain, commitment parameters, envelope,
and next-level opening route. `slot.d_setup == g` is necessary but not sufficient. The
current production setup-offload slot is D64, so the first mixed recursive case can cover
`128/64/64`; `128/64/32` requires a separately generated, committed, and audited D32
slot. APIs are parameterized for that future slot, but the D64 artifact never stands in
for it.

If any check fails, the planner selects direct mode and records why. Once a recursive
shape is authenticated, prover and verifier return `InvalidSetup` for a missing or
mismatched slot or route; they never dynamically downgrade, switch `Separate`/`Batched`,
or reinterpret proof lengths.

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

As a round-message cross-check, let `R_b` be all earlier equality-factored range-tree
messages and `M = max(lambda,mu)`:

```text
legacy recursive = R_b + 3*mu + 2*M
target Separate  = R_b + 1*mu + 2*lambda + 2*mu
                 = R_b + 3*mu + 2*lambda
target Batched   = R_b + 1*mu + 3*M.
```

The `1*mu` term is the standard joint leaf's one-coefficient increase. These equations
are diagnostics, not the selector. The planner computes actual complete bytes for legacy,
Separate, and Batched; discards targets larger than legacy; and among valid targets chooses
Batched only when it is strictly smaller than Separate. Equal complete size defaults to
Separate. Direct-versus-recursive scoring additionally includes verifier setup-scan
savings and the next-level cost of carrying the setup-prefix opening.

### Semantic proof containers and headerless wire order

The target ownership is equivalent to the following sketch; exact generic wrappers reuse
repository types rather than introducing aliases:

```rust
struct RangeProductLayerProof<E> {
    sumcheck: EqFactoredSumcheckProof<E>,
    child_claims: Vec<E>,
}

struct RangeOnlyLeafProof<E> {
    sumcheck: EqFactoredSumcheckProof<E>,
    range_image_eval: E,
}

struct DigitRangeProof<E> {
    product_layers: Vec<RangeProductLayerProof<E>>,
    final_leaf: RangeOnlyLeafProof<E>,
}

struct RangeRelationLeafProof<E> {
    sumcheck: SumcheckProof<E>,
    range_image_eval: E,
    digit_witness_eval: E,
}

struct DigitRangeRelationProof<E> {
    product_layers: Vec<RangeProductLayerProof<E>>,
    final_leaf: RangeRelationLeafProof<E>,
}

struct DirectRelationRangeImageProof<E> {
    sumcheck: SumcheckProof<E>,
    next_witness_eval: E,
}

struct SetupContributionProof<E> {
    sumcheck: SumcheckProof<E>,
    setup_prefix_eval: E,
}

struct RangeImageConsistencyProof<E> {
    sumcheck: EqFactoredSumcheckProof<E>,
    next_witness_eval: E,
}

struct BatchedSetupAndRangeImageProof<E> {
    sumcheck: SumcheckProof<E>,
    setup_prefix_eval: E,
    next_witness_eval: E,
}

enum RecursiveClaimReductionProof<E> {
    Separate {
        setup: SetupContributionProof<E>,
        range_image: RangeImageConsistencyProof<E>,
    },
    Batched(BatchedSetupAndRangeImageProof<E>),
}

enum FoldCheckProof<E> {
    DirectSetup {
        digit_range: DigitRangeProof<E>,
        relation_and_range_image: DirectRelationRangeImageProof<E>,
    },
    RecursiveSetupOffload {
        digit_range_relation: DigitRangeRelationProof<E>,
        setup_contribution_eval: E,
        claim_reduction: RecursiveClaimReductionProof<E>,
    },
}

struct FoldLevelProof<F, E> {
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    scaled_fold_witness: RingVec<F>,
    fold_grind_nonce: u32,
    next_witness_binding: NextWitnessBinding<F>,
    fold_check: FoldCheckProof<E>,
}
```

`FoldCheckProof` has exactly the schedule-selected direct or recursive semantic payload.
If only Separate lands initially, store its two proof fields directly; do not create a
one-variant `RecursiveClaimReductionProof`. Add a two-variant enum only when Batched is
implemented, selected, serialized, and verified. `setup_contribution_eval` occurs once in
the recursive common prefix because it closes Stage 1 before either Stage 2 suffix.

Headerless serialization follows transcript-use order exactly:

```text
Intermediate envelope:
  optional extension-opening reduction
  scaled fold witness
  fold-grind nonce
  next-witness binding (outer commitment or payload-free terminal-inner state)

Direct fold check:
  product layers: sumcheck, child claims
  range-only leaf: sumcheck, range_image_eval
  relation/range-image sumcheck, next_witness_eval

Recursive common prefix:
  product layers: sumcheck, child claims
  joint leaf sumcheck, range_image_eval, digit_witness_eval
  setup_contribution_eval

Separate suffix:
  setup sumcheck, setup_prefix_eval
  range-image consistency sumcheck, next_witness_eval

Batched suffix:
  batched sumcheck, setup_prefix_eval, next_witness_eval.
```

Proof structs, plan descriptors, serialization, deserialization, sizing, and transcript
read order mirror this sequence. Reject an extra or missing field before allocation. Do
not rely on Rust enum representation and do not serialize a topology tag already fixed by
the authenticated schedule. #311's terminal proof wire remains outside this sequence.

### Normative transcript order

Before messages, bind the protocol version, selected topology and reduction shape, native
round counts, LSB-first coordinate map, unequal-domain lift convention, role dimensions,
exact setup slot/commitment identity, and outgoing witness binding.

Direct mode retains the incoming order: complete and absorb the range proof, sample the
direct range-binding challenge, run the standard relation/range-image proof, absorb
`next_witness_eval`, and check the final relation with locally evaluated setup. It gains
no challenge or scalar in this cutover.

Recursive common order is:

1. complete every equality-factored product layer and absorb its child claims;
2. bind the final-leaf input claim, anchor, linear relation claim, and provider descriptor;
3. sample `range_relation_batch_challenge` and run the joint leaf;
4. absorb `range_image_eval`, then `digit_witness_eval`;
5. absorb the complete `setup_contribution_eval` and consume the deferred Stage 1 check.

Then execute exactly one Stage 2 frame:

- Separate: run setup under setup-specific labels, absorb `setup_prefix_eval`, close that
  frame, sample `range_image_binding_challenge`, run consistency under range-specific
  labels, then absorb `next_witness_eval`.
- Batched: sample `range_image_binding_challenge` to define the native consistency claim;
  after both native claims and descriptors are fixed, sample
  `setup_range_binding_batch_challenge`, run the combined proof, then absorb
  `setup_prefix_eval` followed by `next_witness_eval`.

Every claim, round, point, and final value has a semantic frame label. Range-tree
interlayer challenges, `range_relation_batch_challenge`,
`range_image_binding_challenge`, and `setup_range_binding_batch_challenge` are distinct
and never reused.

### Degree and soundness ledger

`FoldCheckPlan` enforces these values; the verifier does not infer degree from a received
coefficient vector:

| Subproof | Form | Enforced degree |
|---|---|---:|
| earlier range-product layer | equality-factored | inner 2 or 4 from `DigitRangePlan` |
| direct range-only leaf | equality-factored | inner 2 for LB2; inner 4 for LB3-LB6 |
| direct relation/range-image | standard | 3 |
| recursive range/relation leaf | standard | 3 for LB2; 5 for LB3-LB6 |
| Separate range-image consistency | equality-factored | inner 2 |
| Separate setup contribution | standard | 2 |
| Batched setup/range-image reduction | standard | 3 |

Every sum-check soundness/security budget and proof-size formula is rederived from this
ledger for both topologies and both reduction shapes. Neither paper shorthand nor the
deleted numeric Stage 3 degree remains a second authority.

### Offloading measurement matrix

Compare legacy recursive, target Separate, and target Batched on identical generated
levels with setup rounds shorter than, equal to, and longer than witness rounds. Include
LB2-LB6, trace absent/present, uniform and supported mixed role tuples, single/multiple
groups and chunks, flat/tensor challenge structures where scheduled, and both one-thread
and production-thread proving. Record:

- joint-leaf first round, first materialization, later rounds, and complete Stage 1;
- setup proof, consistency proof, unequal-domain inactive rounds, and complete Stage 2;
- relation/setup provider construction and retained field elements;
- complete prover and verifier time;
- actual proof bytes, formula bytes, transcript events, and carried opening costs;
- local setup-scan work removed from the verifier.

The acceptance decision is per complete scheduled level and end-to-end proof. A balanced
common-round win does not authorize a larger unequal-domain proof, and a smaller proof
does not authorize a verifier regression or a setup route that the next level cannot
discharge.

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
- every short positive prefix against a fully padded oracle, including poisoned omitted
  storage and randomized nonzero derived defaults;
- alpha equal to zero, one, and random field values;
- trace absent, sparse, and dense;
- uniform `64/64/64`, uniform `128/128/128`, mixed `128/64/64`, and mixed
  `128/64/32`;
- singleton, multi-group, and multi-chunk layouts;
- serial and parallel execution;
- fp128 primary plus fp64/Ext2 and fp32/Ext4 smoke coverage;
- aligned spans, additive overlaps, partial fringes, and role-local exponent resets;
- unequal setup/witness round counts in both directions and at equality;
- transcript separation for every interlayer, range/relation, consistency, and batch
  challenge;
- actual serialized size versus the plan formula for every direct/Separate/Batched shape;
- planner rejection for cache-cap, missing route, mismatched slot, and oversized recursive
  candidates;
- malformed proof counts, degrees, points, domains, lifts, suffix projections, role
  dimensions, setup slots, routes, and serialized lengths.

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

## Deferred optimization and feature backlog

These items remain valuable but are outside the three committed PRs. They are recorded so
the shorter stack does not silently discard them:

- emit balanced digits directly into final destination-oriented compact storage, avoiding
  a source-shaped temporary and transpose where the upstream ring-switch layout permits;
- reuse a bounded fold-grind workspace instead of rebuilding scratch ownership at every
  level;
- simplify ring-relation construction after semantic inner/outer/open bases and the mixed
  relation emitter are stable;
- port structured verifier prefix/carry-bucket kernels only after the one semantic
  relation evaluator lands, retaining the dense evaluator as oracle;
- integrate packed sum-checks over final address-major scalar states without restoring
  public x/y geometry;
- add compressed commitments only with an independently specified commitment-domain and
  opening-route cutover.

A future independently committed oracle cannot reuse challenges merely because its padded
round count matches another oracle. Its coordinate map must be injective, in range,
order-explicit, and distinguish active from inactive coordinates. A fixed-coordinate
embedding owns the equality selector for the actual fixed bits; a repeated-table lift owns
its `2^-inactive` scale exactly once. A contiguous interval is not assumed to be a Boolean
subcube.

The future fused negative-binary check belongs in whichever Stage 2 range-image
consistency composer is active. Its weight is the MLE of the pointwise Boolean table
`Eq(range_image_anchor,z) * binary_support_indicator(z)`, not the product of those two MLE
evaluations away from the cube. This series adds no dormant proof field, challenge,
coefficient slot, or inactive branch for it.

## Risk register

| Risk | Failure mode | Required prevention or stop condition |
|---|---|---|
| Nonzero derived padding is treated as zero | Stage 1 or the recursive leaf proves a different polynomial on partial domains | Exact-prefix defaults and split-equality suffix mass agree with dense padded oracles after every bind |
| Delayed reduction exceeds its contract | Release-only proof corruption | Reduce at the documented block boundary; use delayed products only under `DELAYED_PRODUCT_SUM_IS_EXACT`; canonical fallback otherwise |
| A compute cleanup changes the wire | Existing schedules silently drift | Compare bytes, transcript events, rounds, challenges, claims, and points against the current protocol epoch |
| The recursive cutover is partial | Prover, verifier, sizing, or transcript still interprets numeric Stage 3 | Land plan, types, wire, prover, verifier, sizing, schedule, and deletion in one atomic PR |
| Common-coordinate and lane phases are conflated | Lane weights are folded too early or alpha is treated as scalar too soon | Explicit `0..r`, `r..k`, `k..num_vars` lifecycle with round-by-round dense comparison |
| Mixed point slicing or setup addressing is wrong | Direct and recursively certified setup scalars disagree | Typed points and one checked role-address helper; compare full dense/factorized/direct/recursive scalars |
| Setup scalar is factored twice | Deferred Stage 1 equation applies `common_alpha_eval` again | Define and test `setup_contribution_eval` as the complete flat contribution |
| Unequal-domain lift is scaled twice | Batched proof accepts the wrong claim | Geometry owns `2^-delta`; inactive `/2` rounds and active persistent scaling are differentially tested |
| Generic abstraction hides algebra | A degree or batching factor is omitted or duplicated | Closed semantic events and equation-owning prover types; no expression engine or driver wrapper |
| Microbenchmark win regresses the prover | Cache construction, allocation, or parallel overhead erases a kernel gain | Select on complete stage and prover measurements; delete the loser and its knob |
| Future-feature scope creep | Dormant fields create ambiguous proof states | Document the seam only; land no inactive compressed/binary wire or production branch |

The old document's rigid file/function line thresholds are intentionally superseded. A
line count is not an invariant and splitting a hot kernel into forwarding helpers would
violate the single-source rule. Each PR instead performs an explicit ownership audit:
large files must own one coherent state machine or kernel, serial and parallel scheduling
must call the same arithmetic body, unreachable basis modes are deleted, and no old/new
wrapper, duplicated formula, or second representation survives cutover.

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
- relation lane weights use at most `N/g` state plus explicitly bounded sparse fringes
  (exactly `N/g` for fully aligned scheduled layouts), and mixed dimensions no longer use
  a dense sentinel path;
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
