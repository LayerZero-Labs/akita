# Spec: RelationWeightPolynomial Sumcheck Refactor

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao (design), Codex draft |
| Created     | 2026-07-06                     |
| Status      | proposed                       |
| Branch      | `quang/relation-weight-polynomial-spec` |
| Related     | [`specs/y-ring-trace-internalization.md`](y-ring-trace-internalization.md) (superseded trace mechanism; see below), [`specs/runtime-ring-cutover.md`](runtime-ring-cutover.md), [`specs/setup-product-sumcheck.md`](setup-product-sumcheck.md), [`specs/sumcheck-kernel-cutover.md`](sumcheck-kernel-cutover.md) (prover pair-scan architecture) |

## Summary

Refactor the Stage-2 relation sumcheck around one explicit
`RelationWeightPolynomial`.

Today Stage 2 proves the right algebraic statement, but its implementation is
still shaped as a product of a ring-coordinate factor and an x-column factor,
plus a separate trace addend:

```text
w(x, y) * alpha_weight(y) * m_tau1(x)
  + w(x, y) * TraceWeight(x, y)
```

The target protocol shape is cleaner:

```text
<RelationWeightPolynomial, w> = V_alpha
```

where `RelationWeightPolynomial` is the single multilinear polynomial over the
next-witness Boolean domain whose evaluations are the full field-level,
row-batched relation weight. It already includes every row of the
ring-switched relation matrix: the evaluation/trace row, fold rows, setup rows,
quotient rows, and later compression rows.

The fused Stage-2 sumcheck proves exactly:

```text
V_alpha + gamma * s_claim
 =
sum_x [
    w(x) * RelationWeightPolynomial(x)
  + gamma * eq(stage1_point, x) * w(x) * (w(x) + 1)
]
```

This spec does not implement compressed commitments and does not enable
non-uniform `d_a/d_b/d_d` schedules. It prepares the code for both by removing
the current Stage-2 split and by making the ring-switch quotient layout
row-family typed.

This is a full cutover spec. The implementation must update the canonical
Stage-2 and ring-switch APIs in one pass, delete the old split relation API, and
rename misleading internal concepts while touching the area. Diff size is not a
reason to leave compatibility wrappers, aliases, duplicate helpers, or stale
terminology behind.

### Proof-language cutover (intentional)

This refactor **changes the accepted proof language**. It does not preserve
byte-for-byte compatibility with proofs produced under today's
`gamma^2`-batched trace summand or separate `trace_opening_claim`.

What changes:

- Stage-2 `input_claim` and `expected_output_claim` shape (two summands only).
- Fiat-Shamir transcript: no separate trace-batching scalar; no terminal
  `trace_gamma` squeeze; `EvaluationTrace` covered by the same `tau1` row batch
  as every other relation row.
- Stage-2 sumcheck round polynomials and on-wire proof bytes.

What is preserved for uniform schedules (`d_a = d_b = d_d`):

- Opening binding semantics via the `EvaluationTrace` relation row (same
  soundness obligation, different encoding).
- Quotient **witness tail** bytes (`r_hat` segment layout and order).
- Stage-1 range-check protocol and unrelated transcript labels/sampling order.

Old proofs do not verify under the new code; new proofs do not verify under the
old code. Update `transcript_hardening` fixtures and any pinned proof vectors as
part of the cutover.

## Motivation

The current Stage-2 code is hard to audit because the relation is split across
three implementation concepts:

- `m_evals_x`, the tau-weighted row combination after evaluating setup/ring rows
  at `alpha`;
- `alpha_evals_y`, the ring-coordinate power polynomial;
- `TraceTable`, a separate trace/evaluation addend folded beside the relation.

This matches an older implementation narrative, not the normative Section 4
shape. The trace row is an ordinary relation row, not a sibling
of the relation. It is batched by the same `tau1` row-equality weights as
the ring rows, even though it is field-level and has no quotient block and no
`alpha` weight. The verifier reasons about one row-batched relation
polynomial and one claimed value.

The split also keeps mixed ring dimensions awkward. The code already has partial
per-role support for `d_a`, `d_b`, and `d_d`, but the quotient witness and
`expected_output_claim` path still behave as if one ring-coordinate axis and one
quotient denominator control the whole relation.

The refactor makes the code say the same thing as the protocol:

1. Build one `RelationWeightPolynomial`.
2. Prove its inner product with the next witness.
3. Use one final verifier evaluation of that polynomial.

The refactor also makes later compression and mixed-dimension work feel
like adding row families, not changing the Stage-2 protocol again.

## Current State

### Stage-2 Prover

`AkitaStage2Prover` stores:

```rust
alpha_compact: Vec<E>,
m_compact: Vec<E>,
trace_table: Option<TraceTable<E>>,
```

and every relation accumulation computes either:

```rust
p = alpha_compact[y] * m_compact[x]
```

or `p + trace_table[x, y]`.

The trace term is algebraically fused into the same local witness scan, but it
is still represented as an optional side addend. The constructor also computes
the input claim as:

```text
gamma * s_claim + relation_claim + trace_opening_claim
```

instead of accepting one relation-weight claim.

### Stage-2 Verifier

`AkitaStage2Verifier::expected_output_claim` separately evaluates:

```text
alpha_val = alpha_evals_y(r_y)
row_val   = prepared_row_eval(r_x)
trace_val = TraceWeight(r_x, r_y)
```

and returns (intermediate levels):

```text
w_eval * alpha_val * row_val
  + trace_coeff * w_eval * trace_val
  + batching_coeff * eq(stage1_point, r) * w_eval * (w_eval + 1)
```

This is the main verifier-side place where the target protocol shape is
obscured. The trace piece is a separate summand with `trace_coeff = gamma^2`
today; the cutover folds it into `RelationWeightPolynomial` as the
`EvaluationTrace` relation row (no `trace_coeff`, no `gamma^2`).

### Ring-Switch Prover

`compute_m_evals_x` is already partway to mixed role dimensions: it computes
separate alpha-power vectors for `d_a`, `d_b`, and `d_d`, and it views the
shared setup matrix with per-role dimensions.

The remaining problem is that it returns only the x-column factor. Stage 2 then
reintroduces one global `alpha_evals_y` factor. This prevents the relation
weight from naturally representing row families whose ring-coordinate dimension
will differ.

### Quotient Tail

The quotient witness is still one homogeneous tail:

```text
r_tail_len = rows * r_decomp_levels
weight(row, level) = -eq_tau1[row] * (alpha^D + 1) * gadget[level]
```

That is correct for uniform `D`, but not for the intended mixed layout. The
mixed version needs row-family denominators:

```text
-(alpha^{d_A} + 1) for fold-evaluation / fold-consistency / A rows
-(alpha^{d_B} + 1) for B rows
-(alpha^{d_D} + 1) for D rows
```

and later analogous dimensions for compression-chain families.

## Target Model

Let the next witness be the multilinear polynomial

```text
w : {0,1}^{mu'} -> E
```

where the Boolean address covers the full flat witness coefficient stream. The
address is still decoded as `(x, y)` internally while current schedules use
a uniform ring dimension, but that is an implementation detail.

The relation construction produces one multilinear polynomial:

```text
RelationWeightPolynomial : {0,1}^{mu'} -> E
```

defined by:

```text
RelationWeightPolynomial(a)
  = sum_i eq(tau1, i) * M_alpha(i, a)
```

where `M_alpha` is the full field-level ring-switched relation matrix after
evaluating all row-family ring entries at the transcript challenge `alpha`.

The relation target is:

```text
V_alpha = sum_i eq(tau1, i) * h_i
```

including the evaluation/trace row target. The Stage-2 input claim is:

```text
V_alpha + gamma * s_claim
```

At the final sumcheck point `r`, the verifier checks
`expected_output_claim(r)` (`SumcheckInstanceVerifier`, `check_sumcheck_output_claim`):

```text
expected_output_claim(r)
  = w(r) * RelationWeightPolynomial(r)
  + gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
```

Do not call this object a "stage-2 oracle". That label collides with
`Stage2WitnessOracle` (how `w(r)` is obtained) and with per-summand local names
in the current code. Use **expected final evaluation** or cite
`expected_output_claim(r)` directly.

Terminal Stage 2 sets `gamma = 0`, so the range-binding term vanishes and only
the relation term remains.

### Logical Row Order

Treat the relation matrix as having this logical row-family order:

```text
EvaluationTrace | FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency
```

The current uncompressed implementation has no compression-chain families, so
this is the whole relation. The important convention is that
`EvaluationTrace` is a row family of the same `tau1`-batched relation, not an
extra Stage-2 term. It is field-level, contributes only to the `e_hat` columns,
has right-hand side equal to the scalar opening claim, and consumes no quotient
witness.

The canonical logical row order starts with `EvaluationTrace`. The quotient
witness order is separate because `EvaluationTrace` has no quotient block. For
uniform schedules, the quotient-bearing families keep the existing row-major
byte order:

```text
FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency
```

The implementation must not expose the old "ring rows plus separate trace"
shape. Any bridge from old physical row indices belongs inside the relation
row-layout constructor and must not leak into the Stage-2 prover or verifier
APIs.

### Protocol change: `EvaluationTrace` row (supersedes `gamma^2` trace batching)

[`y-ring-trace-internalization.md`](y-ring-trace-internalization.md) still
describes removing on-wire `y_ring` via a fused stage-2 addend
`gamma^2 * w * TraceWeight` with input contribution `trace_coeff * opening`.
**This spec supersedes that mechanism.**

The opening still binds through committed `e_hat`, but as a **field-level
relation row** (`EvaluationTrace`), not a third gamma-batched summand:

- Row equation: `omega_Tr^T e_hat = v` with scalar RHS `v` equal to the opening
  claim.
- Row weight: `eq(tau1, EvaluationTrace)` alongside every other relation row.
- `V_alpha` includes the trace row target; there is no separate
  `trace_opening_claim` in `input_claim`.
- Stage 2 has only two summands: relation (`w * RelationWeightPolynomial`) and
  range binding (`gamma * eq * w * (w+1)`).

This is an intentional protocol change. Transcript tests that expect a separate
trace batching scalar or terminal `trace_gamma` squeeze must be updated.

## Naming

Use `RelationWeightPolynomial`.

Avoid `RelationWeightTable` in public APIs and docs. Materialized evaluation
vectors may use local names such as `relation_weight_evals`, but the abstraction
is a polynomial. This matters because the verifier never materializes the full
object; it evaluates the same polynomial at one point.

Canonical API names:

- `RelationWeightPolynomial<E>`: prover-side materialized evaluations and fold
  state.
- `PreparedRelationWeightPolynomial<E>`: verifier-side succinct evaluator.
- `relation_weight_claim`: the target `V_alpha`.
- `relation_weight_eval_at_point`: verifier evaluation of
  `RelationWeightPolynomial(r)`.

Use semantic row-family names in new APIs:

- `EvaluationTrace`: the field-level row
  `omega_Tr^T e_hat = v`. This replaces "trace side term" language.
- `FoldEvaluation`: the row linking `z_hat` and `e_hat`; this is the current
  single "consistency" row.
- `FoldConsistency`: the A-role rows linking `z_hat` and `t_hat`.
- `OuterConsistency`: the B-role rows linking `t_hat` to the outer commitment
  path.
- `OpeningConsistency`: the D-role rows linking `e_hat` to the opening
  commitment path.

Avoid using bare `A`, `B`, and `D` as relation-family names outside local
matrix-role variables. They are useful role-matrix labels, but they do not say
which witness segments the row family connects. When a code boundary exposes a
family's role dimension, attach the role metadata to the semantic family:
`FoldConsistency { role: A, ring_dim: d_a }`, never `ARow`.

### Compression Naming

Compressed commitments extend the semantic row-family model, not punch
new A/B/D-shaped holes through it.

The current uncompressed families are:

```text
EvaluationTrace | FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency
```

When commitment compression lands, the extra rows are additional outer/opening
consistency layers:

```text
OuterConsistency(layer)
OpeningConsistency(layer)
```

The layer is explicit metadata:

```rust
pub enum ConsistencyLayer {
    Base,
    Compression { index: usize },
}
```

where the base B/D rows are layer 0 and compression rows are later layers. The
important point is that `OuterConsistency` and `OpeningConsistency` describe the
obligation across the whole commitment path. B and D are just the first role
matrices currently used to enforce those obligations.

Do not introduce row-family names like `BCompression`, `DCompression`,
`BInner`, or `DInner` unless the name is local to a matrix view. Public row
layout and quotient layout names stay semantic.

## Non-Goals

- Do not implement compressed commitments in this PR.
- Do not enable non-uniform `d_a/d_b/d_d` schedules in this PR.
- Do not leave the old trace-batching transcript shape in place. The canonical
  transcript has one row-batching challenge `tau1` covering `EvaluationTrace`
  and every other relation row, and one relation/range batching scalar `gamma`
  when the range term is present. It has no separate trace-batching scalar.
- Do not change the Stage-1 range-check protocol.
- Do not remove verifier offloading or setup-product sumcheck paths. The
  refactor preserves their current direct/recursive behavior.
- Do not add backward-compatibility wrappers for the old split API.
- Do not drop the current Stage-2 round-batching prover path. Reimplement it
  against `RelationWeightPolynomial`.

## Design

### 1. Prover-Side RelationWeightPolynomial

Replace the Stage-2 prover's split relation fields:

```rust
alpha_compact: Vec<E>,
m_compact: Vec<E>,
trace_table: Option<TraceTable<E>>,
```

with:

```rust
relation_weight: RelationWeightPolynomial<E>
```

`RelationWeightPolynomial<E>` lives in `akita-types` as the shared protocol
concept. The prover representation stores a materialized `Vec<E>` of
evaluations in the same flat order as `w_evals_compact`:

```text
relation_weight[a] where a = x * y_len + y
```

That is exactly what the sumcheck prover needs. The existing code already folds
`alpha_compact`, `m_compact`, and `trace_table` alongside the witness; this
becomes one fold of `relation_weight`.

The relation round contribution becomes uniformly:

```rust
accumulate_relation_coeffs(rel, w0, dw, weight0, weight1)
```

or the compact signed version:

```rust
accumulate_relation_coeffs_signed(rel, w0, dw, weight0, weight1)
```

No Stage-2 round kernel may know whether a relation weight came from an
`alpha` power, a setup row, a quotient row, or a trace row.

### 1a. Stage-2 Round Batching

The current prover has a first-two-round "prefix" fast path. The underlying
idea is round batching in the small-value setting: precompute an evaluation grid
for a contiguous window of rounds, then recover the ordinary sum-check messages
as verifier challenges arrive.

Keep the optimization, but rename it. "Prefix two round" is not the protocol
concept and collides with setup-prefix terminology. Use round-batching names:

| Current name pattern | Replacement name pattern |
|----------------------|--------------------------|
| `two_round_prefix` module | `round_batching` module |
| `Stage2TwoRoundPrefix` | `Stage2InitialRoundBatch` |
| `Stage2BivariateSkipProof` | `Stage2RoundBatchGrid` |
| `Stage2CompressedGrid` | `OmittedCornerEvaluationGrid` |
| `PrefixPoint` | `RoundBatchPoint` |
| `can_use_stage2_two_round_prefix` | `can_use_stage2_initial_round_batch` |
| `ensure_two_round_prefix` | `ensure_initial_round_batch` |
| `using_two_round_prefix` | `using_initial_round_batch` |
| `build_stage2_bivariate_skip_proof_from_compact` | `build_stage2_initial_round_batch_grid` |
| `fold_compact_to_round2` | `fold_compact_through_initial_batch` |

The shared round-batching module currently serves Stage 1 and Stage 2. Rename
the shared module and the Stage-1 names in the same cutover so the codebase does
not contain two vocabularies for the same optimization:

| Current name pattern | Replacement name pattern |
|----------------------|--------------------------|
| `Stage1TwoRoundPrefix` | `Stage1InitialRoundBatch` |
| `build_stage1_bivariate_skip_proof_from_s_compact` | `build_stage1_initial_round_batch_grid` |
| `stage1_prefix_points` | `stage1_round_batch_points` |
| `STAGE1_PREFIX_EVAL_COUNT` | `STAGE1_ROUND_BATCH_EVAL_COUNT` |

This optimization is non-negotiable for the Stage-2 cutover. The new prover
must implement the same initial two-round small-value path against the collapsed
relation-weight shape. The code is simpler than the current code because the
relation contribution is just:

```text
w(x) * RelationWeightPolynomial(x)
```

rather than:

```text
w(x, y) * alpha(y) * m(x) + w(x, y) * TraceWeight(x, y)
```

The round-batched grid stores and reconstructs the relation
inner-product contribution and the range-binding contribution as two ordinary
summands over the same witness grid. It must not reintroduce
`alpha_evals_y`, `m_evals_x`, or trace-side tables.

### 2. Constructor Contract

Change `AkitaStage2Prover::new` to accept:

```rust
relation_weight_evals: Vec<E>,
relation_weight_claim: E,
```

and remove:

```rust
alpha_evals_y
m_evals_x
trace_table
trace_opening_claim
```

from the Stage-2 constructor.

The constructor validates:

- `relation_weight_evals.len() == w_evals_compact.len()` after both vectors are
  represented over the same padded hypercube;
- `stage1_point.len() == col_bits + ring_bits`;
- `live_x_cols` is nonzero and fits the x-domain;
- any padding entries in `relation_weight_evals` are zero.

The input claim becomes:

```text
input_claim = batching_coeff * s_claim + relation_weight_claim
```

### 3. Ring-Switch Finalization

Rename and reshape prover ring-switch finalization around relation weights.

Current:

```rust
RingSwitchOutput {
    m_evals_x,
    alpha_evals_y,
    ...
}
```

Target:

```rust
RingSwitchOutput {
    relation_weight_evals,
    relation_weight_claim,
    ...
}
```

`relation_weight_claim` is computed from the canonical `RelationRowLayout`:

```text
relation_weight_claim =
  sum_row eq(tau1, row) * row_target(row)
```

`EvaluationTrace` contributes the scalar opening target through this same row
target mechanism. There is no separate `trace_opening_claim` passed to Stage 2.

Build `relation_weight_evals` by walking the witness segments and relation row
families:

```text
for family in relation_row_layout:
  add eq(tau1, family.rows) * family_weight_to_witness_segment
```

The builder must not return or store `m_evals_x` or `alpha_evals_y` as Stage-2
handoff objects. Existing arithmetic routines are reused only as private
implementation details after being renamed and reshaped; the canonical output is the flat
relation-weight polynomial.

### 4. Verifier-Side PreparedRelationWeightPolynomial

The verifier exposes one operation:

```rust
prepared_relation_weight.eval_at_point(challenges) -> Result<E, AkitaError>
```

Internally, this evaluator is a row-family evaluator:

```text
sum_row eq(tau1, row) * M_alpha(row, r)
```

The implementation can reuse the existing setup scan and closed-form trace
arithmetic, but only behind this row-family interface. The Stage-2 verifier sees
only:

```rust
let relation_weight_eval =
    self.prepared_relation_weight.eval_at_point(challenges)?;
let relation_term = w_eval * relation_weight_eval;
// full expected_output_claim adds the range-binding term when gamma != 0
```

`AkitaStage2Verifier` no longer stores `alpha_evals_y`, `prepared_row_eval`,
and `trace` as separate Stage-2 concepts. It stores
one prepared relation-weight polynomial.

### 5. Trace Row Integration

Trace construction moves from "Stage-2 side addend" to the
`EvaluationTrace` row contribution.

Delete `stage2_trace_coeff` as a Stage-2 batching concept. The replacement is
the `EvaluationTrace` row weight:

```text
evaluation_trace_row_weight = eq(tau1, EvaluationTrace)
```

multiplied by any public opening-claim batching scalar that is part of the row
itself. The name `trace_coeff` must not appear in Stage-2 APIs.

The resulting Stage-2 batching powers read:

```text
range-binding term: gamma
relation term: already includes every row, including trace
```

not:

```text
relation: gamma^0
range: gamma^1
trace: gamma^2
```

Terminal Stage 2 has no range term, so `gamma` is structurally zero and is not a
trace-related scalar. If existing transcript tests observe a separate trace or
terminal batching scalar, update them to the canonical row-batched statement in
this cutover.

### 6. Relation Row And Quotient Layout

Introduce explicit relation row and quotient layouts before enabling mixed
dimensions. These live in `akita-types` next to the existing layout and relation
instance types.

Canonical shape:

```rust
pub struct RelationRowLayout {
    pub families: Vec<RelationRowFamilyLayout>,
}

pub struct RelationRowFamilyLayout {
    pub kind: RelationRowFamily,
    pub row_start: usize,
    pub row_count: usize,
    pub ring_dim: Option<usize>,
    pub quotient: Option<RelationQuotientSlice>,
}

pub struct RelationQuotientSlice {
    pub witness_offset: usize,
    pub row_count: usize,
    pub ring_dim: usize,
    pub digit_depth: usize,
    pub log_basis: u32,
}

pub struct RelationQuotientLayout {
    pub slices: Vec<RelationQuotientSlice>,
}

pub enum ConsistencyLayer {
    Base,
    Compression { index: usize },
}

pub enum RelationRowFamily {
    EvaluationTrace,
    FoldEvaluation,
    FoldConsistency,
    OuterConsistency { layer: ConsistencyLayer },
    OpeningConsistency { layer: ConsistencyLayer },
}
```

`RelationRowLayout` is the single source of truth. `RelationQuotientLayout` is
derived from the quotient-bearing row families and is exposed through
`WitnessLayout`. `RingRelationInstance` returns the same row layout so prover
and verifier cannot disagree about row order, row weights, quotient offsets, or
dimensions.

`EvaluationTrace` has:

```rust
ring_dim = None
quotient = None
```

The quotient-bearing families have:

```rust
ring_dim = Some(d_family)
quotient = Some(...)
```

For the current uncompressed implementation, the logical row families are:

- `EvaluationTrace`: evaluation/opening row, field-level, no quotient;
- `FoldEvaluation`: current consistency row linking `z_hat` and `e_hat`, ring
  dimension `d_a`;
- `FoldConsistency`: A-role rows linking `z_hat` and `t_hat`, ring dimension
  `d_a`;
- `OuterConsistency { layer: Base }`: B-role rows linking `t_hat` to the outer
  commitment path, ring dimension `d_b`;
- `OpeningConsistency { layer: Base }`: D-role rows linking `e_hat` to the
  opening path, ring dimension `d_d` when present.

When `d_a = d_b = d_d`, the emitted witness must be byte-identical to the
current `r` tail. Tests must assert this.

### 6a. Ring-Switch Quotient Pipeline Naming

The quotient pipeline currently contains several names derived from an older
"split-eq quotient" framing. Those names must be removed during this cutover
because they are inaccurate:

- the fused NTT kernel does not split an equality polynomial;
- it computes role-family products used by the ring-switch relation;
- the outputs are a mixture of cyclic products and quotient products, not a
  single quotient object;
- the row families are no longer canonically A/B/D-shaped once compression and
  mixed dimensions are introduced.

Rename the quotient/product pipeline around relation families:

| Current name | Replacement name |
|--------------|------------------|
| `fused_split_eq_quotients.rs` | `fused_relation_family_products.rs` |
| `fused_split_eq_quotients` | `fused_relation_family_products` |
| `fused_split_eq_quotients_prover_bounds` | `fused_relation_family_products_with_bounds` |
| `fused_split_eq_quotients_with_digit_bound` | `fused_relation_family_products_with_digit_bounds` |
| `fused_split_eq_quotients_with_params` | `fused_relation_family_products_with_params` |
| `fused_split_eq_quotients_one_shot` | `fused_relation_family_products_one_shot` |
| `RingSwitchRelationRowsPlan` | `RelationFamilyProductsPlan` |
| `RingSwitchRelationRows` | `RelationFamilyProducts` |
| `RingSwitchRelationView` | `RelationFamilyProductSource` |
| `RingSwitchQuotientRowsPlan` | `FoldConsistencyQuotientPlan` for the current A-role-only helper |
| `RingSwitchQuotientView` | `FoldConsistencyQuotientSource` for the current A-role-only helper |

Rename output fields semantically:

| Current field | Replacement field |
|---------------|-------------------|
| `d_cyclic` | `opening_cyclic_products` |
| `b_cyclic` | `outer_cyclic_products` |
| `a_quotients` | `fold_consistency_quotients` |

These field names are deliberately still specific about the current product
kind. Later mixed-dimension work can replace them with a vector of
`RelationFamilyProduct` records keyed by `RelationRowFamily`. Until then, the
names make the current semantics clear without pretending that A/B/D are
the protocol-level row names.

Keep `SplitEqEvals` only for code that truly factors an equality polynomial
into inner and outer tables. `GruenSplitEq` is a general sum-check round state,
not a ring-switch quotient concept; the relation-weight implementation does not
rename it unless the shared round-state API is already being cut over. Do not
use `split_eq` in the ring-switch quotient/product pipeline.

### 7. Quotient Weight Evaluation

Replace homogeneous quotient-tail weighting:

```text
-(alpha^D + 1) * eq_tau1[row] * gadget[level]
```

with family-specific weighting:

```text
-(alpha^{family.ring_dim} + 1)
  * eq_tau1[row]
  * gadget_family[level]
```

Each `RelationQuotientSlice` stores its own `digit_depth` and `log_basis`. The
current schedule populates identical values for every quotient slice. Mixed
dimensions and future compression rows then vary per-family dimensions or
decomposition parameters without changing the layout API.

### 8. Witness Layout

The logical next witness remains:

```text
z_hat || e_hat || t_hat || r_hat
```

for this PR.

The `r_hat` segment becomes a concatenation of quotient-family slices:

```text
r_hat =
  r_hat_fold_evaluation
  || r_hat_fold_consistency
  || r_hat_outer_consistency
  || r_hat_opening_consistency
```

under the hood. In uniform mode this concatenation has the same order and bytes
as the current row-major `r` list:

```text
row 0, row 1, ..., row n-1
```

The public `WitnessLayout` exposes the quotient tail offset and length via the
derived `RelationQuotientLayout`, not `rows * levels`.

### 9. Setup Evaluator Boundary

The verifier-side setup scan is currently intentionally fused:

```text
D * e_hat + B * t_hat + A * z_hat
```

That performance optimization is fine, but it is an implementation of
`PreparedRelationWeightPolynomial::eval_at_point`, not part of Stage-2's public
shape.

Concretely:

- keep the fused setup scan for uniform schedules;
- make its inputs carry per-role dimensions explicitly;
- make the output contribute to one `relation_weight_eval`;
- do not expose separate setup/ring/trace terms to `AkitaStage2Verifier`.

### 10. Documentation Updates

After the implementation lands, update:

- the module docs in `akita_stage2/mod.rs`;
- `docs/verifier-contract.md` if new verifier-reachable validation boundaries
  are added;
- the Akita Book proving/verifying pages that describe the Stage-2 relation
  shape;
- any spec that still says the trace is a `gamma^2` side term instead of a
  relation row.

## Implementation Plan

Full cutover in sequenced PRs. Each slice should compile, test green, and
delete the API it replaces before the next slice depends on it. No compatibility
wrappers.

### Slice 0 — Types foundation (`akita-types` only)

Land layout and polynomial shells without changing prover/verifier behavior.

1. Add `layout/relation_rows.rs`: `RelationRowFamily`, `ConsistencyLayer`,
   `RelationRowFamilyLayout`, `RelationRowLayout`, `RelationQuotientSlice`,
   `RelationQuotientLayout`, `from_level_params` builder, validation.
2. Add `relation_weight/` module: `RelationWeightPolynomial<E>` (materialized
   evals, fold helpers, padding-zero validation).
3. Add `RingRelationInstance::relation_row_layout()` (additive; `segment_layout`
   unchanged for now).
4. Unit tests: uniform quotient length equals current `m_row_count_for *
   r_decomp_levels`; `EvaluationTrace` has `ring_dim = None`, `quotient = None`;
   bridge formula on fixtures:

   ```text
   relation_weight[a] == alpha_evals_y[y] * m_evals_x[x] + trace[x,y]
   ```

**Tests:** `cargo test -p akita-types`

### Slice 1 — Prover Stage-2 mechanical cutover

Depends on Slice 0.

1. `AkitaStage2Prover` owns `relation_weight: RelationWeightPolynomial<E>`;
   constructor takes `(relation_weight_evals, relation_weight_claim)` only.
2. `input_claim = batching_coeff * s_claim + relation_weight_claim`.
3. Round kernels use `accumulate_relation_coeffs(_signed)` with paired relation
   weights only; delete `alpha_compact`, `m_compact`, `trace_table` folds.
4. Update `akita_stage2/tests*.rs` to build `relation_weight_evals` via bridge
   formula (synthetic until Slice 3).

**Tests:** `cargo test -p akita-prover akita_stage2`

### Slice 2 — Round-batching rename + relation-weight grid

Depends on Slice 1.

1. Rename `two_round_prefix` → `round_batching` (Stage 1 + Stage 2 names per
   §1a tables).
2. `build_stage2_initial_round_batch_grid(w, relation_weight_evals, …)`; no
   `(alpha_quad, m_value, trace_quad)`.

**Tests:** `cargo test -p akita-prover round_batching` (after rename)

### Slice 3 — Prover ring-switch builder + fold wiring

Depends on Slices 0–1. **Protocol/transcript change lands here.**

1. Segment-wise `build_relation_weight_evals` in `ring_switch/evals.rs`; fold
   `EvaluationTrace` into relation weight (reuse `trace_weight/` as private
   builder).
2. `relation_weight_claim = sum_row eq(tau1, row) * row_target(row)` including
   opening scalar on `EvaluationTrace` row.
3. `RingSwitchOutput` → `{ relation_weight_evals, relation_weight_claim, … }`;
   delete public `m_evals_x` / `alpha_evals_y`.
4. `fold.rs`: remove `stage2_trace_coeff`, `trace_opening_claim`,
   `build_*_stage2_trace_table` from Stage-2 path; pass unified claim to
   `prove_stage2`.
5. Remove terminal `trace_gamma` squeeze from verifier fold path in the same PR
   (prover/verifier transcript must stay matched).

**Tests:** `cargo test -p akita-pcs --test ring_switch`;
`--test transcript_hardening`; `--test akita_e2e`

### Slice 4 — Verifier prepared evaluator + Stage-2 cutover

Depends on Slice 0; wire with Slice 3 for e2e.

1. `PreparedRelationWeightPolynomial::eval_at_point` wrapping deferred row eval +
   closed-form `EvaluationTrace` row eval (no `trace_coeff`).
2. `AkitaStage2Verifier` stores one prepared evaluator + `relation_weight_claim`.
3. `expected_output_claim(r) = w(r) * RelationWeightPolynomial(r) + gamma * eq *
   w * (w+1)`; `input_claim = relation_weight_claim + gamma * s_claim`.
4. `RingSwitchVerifyOutput` packages `prepared_relation_weight`.
5. Stage 3 keeps `SetupContributionPlanInputs` accessor on the prepared object.

**Tests:** `cargo test -p akita-verifier`; `ring_switch.rs` relation-claim tests;
`recursive_setup_e2e` if enabled

### Slice 5 — Quotient family layout + kernel rename

Depends on Slice 0; can follow Slice 3/4 once relation-weight path is stable.

1. Wire `RelationQuotientLayout` through `segment_layout` / `WitnessLayout`.
2. Family-specific quotient weights; uniform mode byte-identical `r` tail.
3. Rename `fused_split_eq_quotients` → `fused_relation_family_products`.
4. Delete all split Stage-2 fields, `stage2_trace_coeff`, stale module docs.

**Tests:** quotient byte identity; `cargo test` full workspace

### Slice 6 — Docs and book

Update `akita_stage2/mod.rs`, `docs/verifier-contract.md`, book sumcheck pages;
mark `y-ring-trace-internalization.md` gamma^2 sections superseded.

**Tests:** `./scripts/check-doc-guardrails.sh`

### Dependency summary

```text
Slice 0 (types)
  ├─→ Slice 1 (stage2 prover mechanical)
  │     └─→ Slice 2 (round batching)
  │           └─→ Slice 3 (ring-switch + fold + transcript)
  ├─→ Slice 4 (verifier prepared eval) ← wire with Slice 3
  └─→ Slice 5 (quotient layout + cleanup) ← after 3/4
        └─→ Slice 6 (docs)
```

**Blast radius:** ~40–50 Rust files across `akita-types`, `akita-prover`,
`akita-verifier`, `akita-pcs/tests`. Stage 3 setup-product sumcheck is coupled
via `SetupContributionPlanInputs`, not the relation-weight summand shape.

## Invariants

- Stage 2 has exactly one relation-weight claim.
- Trace/evaluation is part of the relation weight.
- No Stage-2 round kernel multiplies `alpha(y) * m(x)`.
- No Stage-2 round kernel branches on trace presence.
- No `trace_coeff`, `trace_opening_claim`, or `gamma^2` trace batching in Stage-2
  APIs or transcript.
- `expected_output_claim(r)` equals:

  ```text
  w(r) * RelationWeightPolynomial(r)
    + gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
  ```

  with `EvaluationTrace` inside `RelationWeightPolynomial`, not a third summand.
- Uniform schedules keep byte-identical **quotient witness tail** layout; opening
  binding is equivalent under the `EvaluationTrace` row model (see Proof-language
  cutover above). Proof/transcript bytes are **not** preserved.
- Mixed role dimensions remain rejected at the schedule validation boundary
  until all quotient-family computation is implemented. The row and quotient
  layout APIs already encode the per-family dimensions that removal will use.
- Verifier-reachable malformed shapes return `AkitaError`; no new panics,
  unwraps, unchecked indexing, or unbounded allocation.

## Acceptance Criteria

- [ ] `AkitaStage2Prover` no longer stores `alpha_compact`, `m_compact`, or
      `trace_table`.
- [ ] `AkitaStage2Verifier` no longer computes `alpha_val * row_val` and trace
      as separate Stage-2 concepts.
- [ ] The Stage-2 initial round-batching path exists under round-batching names
      and is implemented against `RelationWeightPolynomial`.
- [ ] `RelationWeightPolynomial` appears in prover and verifier Stage-2 APIs.
- [ ] Trace/evaluation-row contribution is built into relation weight and its
      claim.
- [ ] Quotient tail sizing and weighting are derived from
      `RelationQuotientLayout`.
- [ ] Ring-switch quotient/product kernels no longer expose `split_eq` names.
- [ ] Uniform `d_a = d_b = d_d` quotient **witness tail** bytes are unchanged.
- [ ] `transcript_hardening` fixtures updated for the new transcript shape (no
      trace-batching scalar).
- [ ] Existing direct setup scan and recursive setup-product modes both verify.
- [ ] `cargo fmt -q` passes.
- [ ] `cargo clippy --all --message-format=short -q -- -D warnings` passes.
- [ ] `cargo test` passes.
- [ ] `./scripts/check-doc-guardrails.sh` passes if docs/book files are edited.

## Testing Strategy

### Unit Tests

- Bridge test (pre-cutover equivalence only; delete after Slice 3):

  ```text
  relation_weight[x,y] == alpha_evals_y[y] * m_evals_x[x] + trace[x,y]
  ```

  After Slice 3, trace enters only through the `EvaluationTrace` row inside
  `RelationWeightPolynomial`, not as a separate table.

- Dense trace and sparse trace produce identical `RelationWeightPolynomial`
  evaluations.
- Initial round-batching builder matches the direct Stage-2 path with a
  materialized relation weight.
- `PreparedRelationWeightPolynomial::eval_at_point` equals multilinear
  evaluation of the materialized relation-weight vector on small fixtures.
- `RelationQuotientLayout` for uniform dimensions emits the current row-major
  quotient order.

### Integration Tests

- Existing `akita_stage2` tests.
- `crates/akita-pcs/tests/ring_switch.rs`, especially direct relation claim and
  chunked witness layout tests.
- Single-polynomial PCS e2e.
- Batched/root e2e.
- Extension-field EOR e2e.
- Recursive setup-contribution e2e if enabled in the current test set.

### Negative Tests

- Malformed relation-weight length is rejected.
- Nonzero relation-weight padding is rejected if padding is materialized.
- Missing trace row on a path with an opening claim is rejected.
- Invalid quotient-family layout with overlapping or out-of-order slices is
  rejected.
- Mixed dimensions remain rejected at the explicit not-yet-supported boundary,
  not by an incidental panic or decode failure.

## Performance Notes

The prover materializes `relation_weight_evals` of length equal to the
next-witness hypercube in this cutover. This is no worse asymptotically than the
current prover state, because the prover already scans/folds `w`, `alpha`, `m`,
and trace data. The implementation deletes the old split vectors in the same
cutover to keep memory neutral.

The verifier must not materialize the polynomial. It evaluates
`PreparedRelationWeightPolynomial` directly at the final sumcheck point, reusing
the existing structured row evaluator, fused setup scan, and closed-form trace
evaluator internally.

The segment-wise relation-weight builder emits contributions directly by witness
segment:

- `z_hat` segment: fold-consistency setup plus fold-evaluation contribution.
- `e_hat` segment: opening-consistency setup plus fold-evaluation plus
  evaluation-trace contribution.
- `t_hat` segment: outer-consistency setup plus fold-consistency contribution.
- `r_hat` segment: quotient-family contribution.

That segment builder is the landing point for mixed role dimensions.

## Rollout Risks

- **Transcript accidental change.** Intentional transcript deltas are limited to
  the trace-row cutover: no separate trace-batching scalar, no terminal
  `trace_gamma` squeeze. Preserve unrelated labels and sampling order outside
  that delta.
- **Trace scaling confusion.** The old `gamma^2` language is removed from
  Stage-2 interfaces. There is no trace-side scalar in terminal scheduling.
- **Quotient ordering drift.** Uniform-mode **quotient witness tail** bytes must
  stay unchanged until mixed dimensions deliberately change them. This does not
  apply to proof or transcript bytes.
- **Verifier offloading coupling.** Setup-product sumcheck code depends on the
  row-evaluation plan. The new prepared relation-weight evaluator must keep the
  setup contribution plan available for Stage 3.
- **Padding bugs.** A materialized relation-weight vector over a padded
  hypercube must zero every padded slot, or the sumcheck statement changes.

## Design Decisions

- `RelationWeightPolynomial`, `RelationRowLayout`, and
  `RelationQuotientLayout` live in `akita-types`.
- `EvaluationTrace` is a real row family in `RelationRowLayout`, weighted by
  `eq(tau1, EvaluationTrace)`.
- Stage 2 has no trace-side claim, trace-side coefficient, or trace-side
  transcript scalar.
- `WitnessLayout` exposes quotient-tail structure through the
  `RelationQuotientLayout` derived from `RelationRowLayout`.
- Every quotient slice carries explicit `ring_dim`, `digit_depth`, and
  `log_basis`, even while current schedules set the same decomposition
  parameters for every slice.
- Compressed commitments extend `OuterConsistency { layer }` and
  `OpeningConsistency { layer }`; they do not introduce B/D-named row families.

## References

- Stage-2 prover:
  `crates/akita-prover/src/protocol/sumcheck/akita_stage2/`.
- Ring-switch prover weight builder:
  `crates/akita-prover/src/protocol/ring_switch/evals.rs`.
- Ring-switch witness builder:
  `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`.
- Verifier Stage 2:
  `crates/akita-verifier/src/stages/stage2.rs`.
- Verifier ring-switch row evaluator:
  `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Trace weight code:
  `crates/akita-types/src/trace_weight/`.
