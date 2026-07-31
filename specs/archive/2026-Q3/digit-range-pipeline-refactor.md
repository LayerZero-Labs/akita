# Stage 1 digit-range prover cutover

| Field | Value |
|---|---|
| Author(s) | Quang Dao (protocol and implementation direction); Codex (design synthesis) |
| Created | 2026-07-18 |
| Revised | 2026-07-20; scoped to the implementation and durable contract of PR #312 |
| Status | active |
| PR | [#312](https://github.com/LayerZero-Labs/akita/pull/312) |
| Base | PR #311, `quang/terminal-direct-ring-relations` at `fad006e2280e880fa16f1cd13b5ea2df599364d0` |
| Implemented head | `11163598a6a66b5376306fa9b97d64c29515446a` |
| Related | [`digit-innermost-layout.md`](digit-innermost-layout.md), [`packed-sumcheck.md`](packed-sumcheck.md), [`akita-sumcheck-unification.md`](akita-sumcheck-unification.md) |

## Summary

PR #312 replaces the old Stage 1 digit-range implementation with one streaming,
class-indexed prover for log bases 2 through 6. It preserves the existing proof statement,
transcript, challenge order, child-claim order, degrees, final point, and
`range_image_evaluation`, while substantially reducing prover work and retained state.

This document specifies only that cutover. Stage 2 relation proving, mixed-ring relation
weights, evaluation-trace representation, setup offloading, and movement of relations
between stages belong to their own implementation PRs and specs. Keeping those designs
out of this record prevents PR #312's completed contract from becoming a speculative
workboard for later protocol changes.

## Motivation and context

Stage 1 proves that every balanced digit lies in the range selected by the level's one
scheduled range basis. The old implementation reached the right statement through several
layout-specific provers, eager range-image and product-tree materialization, duplicated
serial/parallel arithmetic, and x/y-prefix branches. That made the prover difficult to
maintain and consumed avoidable memory and field operations, especially at log bases 4,
5, and 6.

The cutover relies on three facts:

1. A balanced digit can be mapped to a much smaller unsigned range-image class.
2. Every range-tree node depends only on a small tuple of those classes until transcript
   challenges force field-valued state.
3. Product layers can be streamed one at a time from the original compact digit source;
   the prover never needs the complete forest resident at once.

## Goals

- One checked topology authority for LB2-LB6.
- One Stage 1 prover lifecycle and one verifier replay path.
- Compact integer digits and class indices before the challenge boundary.
- Streaming product substages with bounded fixed-lane field state afterward.
- Basis-specific initial-round kernels where measurement establishes a material win.
- Exact live-prefix and derived-padding behavior.
- Proof/transcript parity with the post-#311 protocol epoch.
- Durable differential tests, malformed-proof tests, tracing spans, and basis benchmarks.
- Atomic deletion of the eager forest and layout-named Stage 1 implementations.

## Non-goals

- Redesign or optimize the fused Stage 2 relation/range-image sum-check.
- Introduce mixed-ring relation weights or change relation binding order.
- Support independently selected range polynomials inside one Stage 1 witness. The
  implemented prover receives one checked basis for the complete flat digit table.
- Move the relation into Stage 1 or change recursive setup-offload proof shape.
- Add compressed commitments or the fused negative-binary range check.
- Add runtime, schedule, or proof fields selecting CPU kernels.
- Preserve old internal APIs through wrappers or aliases.

The small Stage 2 changes in #312 are mechanical `range_image_evaluation` naming and
Stage 1 output adaptation. They are not a Stage 2 architecture claim.

## Naming and ownership

Production code uses mathematical names:

| Object | Production name |
|---|---|
| balanced signed digit table | `digit_witness` |
| `digit_witness * (digit_witness + 1)` | `range_image` |
| vanishing polynomial over valid range-image values | `range_image_polynomial` |
| checked topology and degree/child order | `DigitRangePlan` |
| compact digits and range-image classes | `CompactDigitSource` |
| final evaluation and checked point | `range_image_evaluation`, `DigitRangeEqualityPoint` |

`W` is acceptable only in short equations. Ambiguous `S`, `s_table`, and `s_claim` names
are forbidden. There is one canonical function per concept; no `_for_level` forwarding
helper, compatibility alias, or stage-named facade survives the cutover.

## Protocol statement

For basis `b = 2^LB`, valid balanced digits satisfy

```text
-b/2 <= digit_witness < b/2.
```

Define

```text
range_image_class(w) = w       when w >= 0
                     = -w - 1  when w < 0,

range_image(w)
  = range_image_class(w) * (range_image_class(w) + 1).
```

Then `0 <= range_image_class(w) < b/2`. Stage 1 proves the product-tree
decomposition of the corresponding range-image vanishing polynomial and returns the
independent MLE claim

```text
range_image_evaluation
  = MLE_z(digit_witness(z) * (digit_witness(z) + 1), range_check_point).
```

The MLE is not replaced by applying `w -> w(w+1)` after evaluating `digit_witness`.
That pointwise identity holds on Boolean vertices, not at a random multilinear point.

The honest prover does not rescan digits to validate their range. Upstream balanced
decomposition owns honest construction; Stage 1 proves the range statement to the
verifier. Lookup sizes are derived from the checked basis contract rather than observed
witness minima or maxima.

## Canonical topology

`DigitRangePlan` fixes every substage, arity, child order, degree, and final leaf:

| LB | Basis | Product substages | Final leaf |
|---:|---:|---|---|
| 2 | 4 | none | quadratic |
| 3 | 8 | none | quartic |
| 4 | 16 | binary root | two quartic leaves |
| 5 | 32 | arity-4 root | four quartic leaves |
| 6 | 64 | binary root, then arity-4 layer | eight quartic leaves |

Prover, verifier, shape validation, serialization sizing, and tests query this plan
directly. No consumer reconstructs topology from basis thresholds, `trailing_zeros`, or
received vector lengths.

Binary-only trees were evaluated and rejected. For LB5/LB6 they add substages, child
claims, challenges, scans, and transitions without reducing the complete round-message
coefficient count. A locally simple binary kernel is not a simpler or smaller protocol.

## Canonical prover lifecycle

Each product substage performs:

1. Build only the small class-indexed node rows needed by this substage.
2. Build ordered class-pair coefficients after interstage batching weights are known.
3. Scan compact pair/quad/octet indices to produce ordinary round messages.
4. Keep the witness compact for the selected initial challenges.
5. Materialize only the current substage's folded fixed-lane field state.
6. Prove remaining rounds with direct fixed-degree arithmetic and in-place folds.
7. Absorb child claims in `DigitRangePlan` order and release the substage state.
8. Rescan the original compact source for the next layer or leaf.

The address-major materialized state is

```text
folded_address -> [lane_0, ..., lane_(LANES-1)].
```

There is no retained range forest and no module/trait family per basis. Serial and Rayon
schedules invoke the same arithmetic reducers and differ only in partitioning.

## Selected high-basis kernels

| Topology | Initial strategy | First field state |
|---|---|---:|
| LB4 two-lane product | two-round 4,096-key challenge-dependent quartet coefficients | two lanes at `N/4` |
| LB4 scalar leaf | two-round 4,096-key challenge-dependent quartet coefficients | one lane at `N/4` |
| LB5 four-lane product | two-round factorized folded-pair rescan | four lanes at `N/4` |
| LB5 scalar leaf | optimized one-round ordered-pair scan | one lane at `N/2` |
| LB6 two-lane product | two-round factorized folded-pair rescan | two lanes at `N/4` |
| LB6 eight-lane product | two-round factorized folded-pair rescan | eight lanes at `N/4` |
| LB6 scalar leaf | optimized one-round ordered-pair scan | one lane at `N/2` |

Shared mechanics include compact `u16` ordered class-pair indices, split-equality block
accumulation, challenge-dependent folded-pair materialization, direct quadratic/quartic
affine-product formulas, and exact nonzero suffix accounting.

Delayed product accumulation is used only when the field declares
`DELAYED_PRODUCT_SUM_IS_EXACT` for the bounded operation. Reduction occurs at the
documented split-equality block boundary; other fields use canonical accumulation with
the same factorization. Safety is never inferred from release mode or observed inputs.

LB4's `8^4 = 4,096` quartet key space is cache-appropriate. LB5 and LB6 avoid full
four-class tables (`16^4` and `32^4`) and rescan compact pairs through a
`classes^2 * lanes` folded-pair table. Parent arithmetic remains field-valued; LB5/LB6
coefficients exceed safe fixed-width integer bounds.

Two-round deferral preserves ordinary Fiat-Shamir causality:

```text
send round_0; receive r_0;
send round_1; receive r_1;
materialize N/4 state.
```

## Selected low-basis kernels

LB2 and LB3 keep the compact source through three ordinary challenges when at least three
variables remain:

| Basis | Compact representation | Challenge-dependent cache | First field state |
|---:|---|---|---:|
| LB2 / 4 | 256 range-image octet classes, direct quadratic coefficients | `256 x 3` field elements | one lane at `N/8` |
| LB3 / 8 | two folded 256-class quads per octet, direct quartic coefficients | 256 folded values and `256 x 4` Taylor rows | one lane at `N/8` |

The LB3 Taylor row is

```text
[Q(a), Q'(a), Q''(a)/2, Q'''(a)/6]
for Q(s) = s(s-2)(s-6)(s-12).
```

This is a challenge-time arithmetic cache, not a second witness. Three-round deferral
still sends and receives three ordinary transcript rounds before materializing `N/8`.

## Exact live prefixes and padding

The flat Boolean domain is checked once. The live digit prefix is explicit; omitted
storage is interpreted through the exact semantic default required by the current
substage. Prefix kernels handle an odd live endpoint, split-equality boundaries, and a
partially occupied last block without reading omitted storage.

Derived suffix mass is computed algebraically from equality factors. It is not assumed
zero merely because the compact digit source ends. Differential tests poison omitted
storage and compare every challenge/fold against a fully padded oracle.

## Implementation ownership

```text
digit_range/
  mod.rs                         topology choreography
  compact_digit_source.rs        compact digits and range-image classes
  range_class_tables.rs          class rows and ordered-pair coefficients
  class_indexed_state.rs         product/leaf transition state
  class_indexed_product.rs       fixed-lane product subchecks
  class_indexed_range_leaf.rs    high-basis equality-factored leaf
  exact_prefix.rs                explicit prefix and exact default
  round_accumulation.rs          bounded coefficient accumulation
  direct_range_leaf.rs           low-basis equality-factored leaf
  direct_range_leaf/
    initial_round_deferral.rs    selected compact prefix kernels
    live_prefix.rs               prefix-aware scans
    rounds.rs                    later field rounds
    sparse_low_variables.rs      small remaining-variable cases
```

Large kernel files are acceptable when they own one coherent invariant. Splitting a hot
function into forwarding helpers to satisfy a line target is not. Conversely, topology,
compact conversion, table construction, and unrelated kernels must not be recombined into
one stage-shaped file.

## Verifier and wire contract

PR #312 is compute-only with respect to the Stage 1 protocol:

- proof bytes, transcript events, challenges, child order, and degrees match the
  post-#311 epoch;
- final point and `range_image_evaluation` are unchanged;
- terminal #311 remains a separate sum-check-free proof and verifier path;
- the verifier validates basis, topology, child count/order, degree, round count, point
  width, and serialized lengths from `DigitRangePlan` before replay or allocation;
- fixed small arrays or borrowed slices replace proof-sized temporary copies; and
- malformed input returns `AkitaError` without verifier-reachable panic, assertion,
  unwrap, unchecked indexing, or attacker-sized allocation.

Protocol epoch fixtures are versioned evidence, not immutable forever. A later declared
protocol change replaces its affected expected bytes/events atomically and records the
new epoch.

## Measurement decisions

All candidates were measured with feature-pruned release/profile builds and the durable
digit-range benchmark. The selected high-basis sequence progressed from streaming node
evaluation (`91d05aca`) to ordered pairs (`999678ce`), optimized ordered pairs
(`b9e44a50`), and two-round deferral (`7b94cb8b`). Integrated low-basis candidates were
LB2 `b74220cf` and LB3 `77e7c870`.

Material outcomes:

- ordered pairs beat streaming node evaluation in all 18 LB4-LB6 cells, with a 25.78%
  median improvement;
- the optimized one-round pipeline beat that baseline in all 18 cells, with a 30.50%
  median improvement;
- the selected two-round policy improved all 18 high-basis point estimates;
- LB2 three-round deferral improved the measured full/uniform and
  three-quarter/uniform cells by about 39.8% and 30.8%; and
- LB3 three-round deferral improved the corresponding one-thread cells by 12.8% and
  10.9%, with stable production-thread improvements.

Deleted losing implementations include streaming node evaluation, full LB5/LB6 quartet
tables, high-basis scalar-leaf two-round deferral where it lost, oversized LB3 tables,
adaptive low-diversity paths, packed LB3 identifiers that regressed parallel proving, and
post-materialization fusion that did not remove enough work.

Tracing spans cover preparation, product/leaf ownership, table construction,
materialization, later rounds, and folds. They never enter address, pair, class, lane,
coefficient, or Rayon-item loops. Criterion/profile CI measures regressions; production
code carries no ad hoc measurement wrapper.

## Tests and evidence

Required durable coverage includes:

- round-by-round compact/class-indexed versus dense padded oracles for LB2-LB6;
- every valid digit and both adjacent invalid values in arithmetic tests;
- full, three-quarter, odd, and short positive live prefixes;
- zero, uniform, high-entropy, zero-heavy, and alternating endpoints;
- serial and parallel schedules and supported field/extension combinations;
- malformed topology, child count/order, degree, point width, and serialized length;
- exact serialized size versus sizing formulas;
- `digit_range_protocol_epoch` for proof bytes, transcript events, point, and final
  evaluation; and
- `fold_protocol_epoch` to ensure direct/recursive envelopes and #311 terminals did not
  conceal a Stage 1 delta.

The one-off allocation harness used during development was deliberately removed. The
checked-in benchmark, profile CI, coarse tracing, and protocol oracles are the durable
artifacts.

## Intended diff surface

| Surface | Allowed responsibility |
|---|---|
| `akita-prover::protocol::sumcheck` | Stage 1 range cutover and directly shared arithmetic |
| `akita-types::proof::stage1` and sizing | topology, descriptive proof fields, validation, byte accounting |
| `akita-verifier::stages::stage1` | checked plan replay and malformed-shape rejection |
| Stage 2 call boundary | mechanical range-image naming/output adaptation only |
| PCS tests/benches/profile | epochs, differential tests, durable basis benchmark, report names |
| transcript labels/book/spec | semantic range-image naming and Stage 1 documentation |

Not in the intended surface: relation provider construction, Stage 2 kernel selection,
trace representation, mixed-dimension relation execution, setup-offload placement,
planner topology, compressed commitments, or a new proof epoch.

## Acceptance criteria

- One `DigitRangePlan`, one `DigitRangeProver`, and one verifier replay path remain.
- The selected LB2-LB6 kernels are the only production implementations.
- No eager range forest, padded range-image table, x/y Stage 1 module, compatibility
  wrapper, or second semantic range formula survives.
- Exact-prefix behavior agrees after every challenge and fold with the padded oracle.
- Proof bytes, transcript events, degrees, child order, final point, and
  `range_image_evaluation` match the post-#311 epoch.
- Durable tests and benchmarks cover every basis and malformed verifier shape.
- Documentation guardrails and the repository's required formatting, lint, and test
  commands pass at the final head.
