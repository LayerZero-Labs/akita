# Spec: Witness-shape-generic verifier row-MLE evaluation

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Omid                          |
| Created       | 2026-06-19                     |
| Status        | proposed                       |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | how/verifying/matrix_evaluation.md |

## Summary

During the relation-check sum-check, the verifier evaluates the
multilinear extension of the relation matrix `M`,
$\widetilde{M}(r_{\text{row}}, r_{\text{col}})$, at a random point — without
materializing `M` — by walking the structured row blocks (`e_hat`, `t_hat`,
`z_hat`, the SIS-matrix setup rows, the `r`-tail). This is the dominant verifier
cost. Today that walk assumes a single fixed **witness column layout**: each
component is one contiguous segment (a single `e_hat` over all blocks, a single
`t_hat`, a single `z_hat`, then `r`). A second layout now exists — produced by
the [distributed prover](../book/src/how/proving/distributed-prover.md) — in
which the witness is partitioned into contiguous **chunks**, each chunk holding
one slice of every component, with the shared `r`-tail at the end.

This spec generalizes the verifier row-MLE evaluation to be **layout-agnostic**:
it introduces a `WitnessType` enum describing the two column layouts, factors
both into one **chunk-list** common denominator, and refactors the evaluation so
a single code path serves both — the established layout becoming the
one-chunk special case. The implementation surface is
`RingSwitchDeferredRowEval::eval_at_point`
(`crates/akita-verifier/src/protocol/ring_switch.rs`), the structured/setup
evaluators in `crates/akita-verifier/src/protocol/slice_mle/`, and
`akita_types::SetupContributionPlan`
(`crates/akita-types/src/setup_contribution.rs`).

The cost target: the chunked layout must be **equal in its dominant cost** to the
established layout. The chunk-partitioned components (`e_hat`, `t_hat`) and the
SIS-matrix α-evaluation scan tile the same total work regardless of chunk count;
only the chunk-replicated component (`z_hat`) genuinely scales with the chunk
count, which is the intrinsic price of that layout, not an implementation
artifact.

> **A note on naming.** The chunked layout is the relation the *distributed
> prover* produces; the verifier itself is a single, ordinary verifier (nothing
> about verification is distributed). See
> [the distributed relation verifier](../book/src/how/verifying/distributed-relation-verifier.md)
> for the component-by-component theory and cost analysis.

## Intent

### Goal

Make the verifier row-MLE evaluation generic over a `WitnessType` by resolving
every shape to one `WitnessLayout` that the per-component evaluators fold over,
where the established single-segment layout is the one-chunk case and is
byte-identical to today.

Key abstractions and surfaces introduced or modified (all layout types live in
`akita-types`, alongside the existing `RingRelationSegmentLayout` and
`SetupContributionPlanInputs`, so both the verifier and the setup-contribution
planner consume the same definitions):

- **`WitnessType`** (new enum):

  ```rust
  pub enum WitnessType {
      /// Component-major layout: the witness is laid out one component at a
      /// time — a single contiguous `e_hat` spanning all `num_blocks` blocks,
      /// then `t_hat`, then `z_hat`, then the `r`-tail (the established
      /// verifier layout, including the `z_first` ordering knob). Every
      /// component is one fused segment; the block axis covers all blocks.
      ComponentMajor,

      /// Chunk-grouped layout: the witness is split into `num_groups`
      /// contiguous chunks, each chunk holding one slice of every component
      /// (`[ e_hat | t_hat | z_hat ]` per chunk), concatenated, with a single
      /// shared `r`-tail. Chunk-partitioned components (`e_hat`, `t_hat`) split
      /// their blocks evenly across chunks (`num_blocks / num_groups` each);
      /// the chunk-replicated component (`z_hat`) appears full-size in every
      /// chunk. Produced by the distributed prover, where each chunk is one
      /// proving group's local witness.
      ChunkGrouped(usize),
  }
  ```

- **`WitnessChunkOffset` / `WitnessLayout`** (new): the resolved, layout-agnostic
  description the evaluators actually consume.

  ```rust
  /// Where one chunk's per-component pieces start in the witness column space.
  pub struct WitnessChunkOffset {
      pub e_offset: usize,
      pub t_offset: usize,
      pub z_offset: usize,            // a full-`block_len` fold (replicated)
      pub global_block_base: usize,   // chunk_idx * blocks_per_chunk
  }

  /// A shape resolved to chunks. `chunks.len() == 1` for `ComponentMajor`,
  /// `== num_groups` for `ChunkGrouped`. `blocks_per_chunk` is the uniform
  /// per-chunk block window `B_w` (`== num_blocks` for `ComponentMajor`).
  pub struct WitnessLayout {
      pub blocks_per_chunk: usize,
      pub chunks: Vec<WitnessChunkOffset>,
      pub offset_u: usize,             // tiered `û_concat`; ComponentMajor only until chunked tiering is specified
      pub offset_r: usize,
  }
  ```

- **`WitnessType::resolve(lp, opening_shape, m_row_layout) -> Result<WitnessLayout,
  AkitaError>`** (new): the **only** place the two shapes diverge.
  `ComponentMajor` resolves to a one-element chunk list whose offsets come
  straight from today's
  [`RingRelationSegmentLayout`](../crates/akita-types/src/proof/ring_relation.rs)
  (preserving `z_first` and the blinding offsets) with `blocks_per_chunk =
  num_blocks` and `global_block_base = 0`. `ChunkGrouped(W)` computes the
  per-chunk offsets arithmetically (fixed `[e|t|z]` order). All shape validation
  (no-panic) happens here.
- **`RingSwitchDeferredRowEval`** carries the resolved `WitnessLayout` (replacing
  the single `witness_segment_layout` it stores today), and `eval_at_point`
  becomes a fold over `chunk_layout.chunks()`.
- **`PreparedChallengeEvals::summarize_chunk_block_carries`** (new, generalizes
  the existing `summarize_all_block_carries` in
  `crates/akita-verifier/src/protocol/ring_switch/tensor_challenges.rs`): the
  per-claim two-bucket `c_alpha` block summaries for **one chunk's** block window
  `[global_block_base, global_block_base + B_w)`, peeling `B_w` instead of `B`.
- **`SetupContributionPlanInputs`** / **`SetupContributionPlan::prepare`**: take
  the `WitnessLayout` so the precomputed column-weight vectors (`e_eq_slice`,
  `t_eq_slice_per_group`, `z_eq_slice`) are built against the resolved chunks.
  The hot α-evaluation loop `packed_slice_inner_sum` is **unchanged**.
- **`LevelParams`** gains a public `witness_shape: WitnessType` (default
  `ComponentMajor`) — the verifier's source of truth for the layout. It is a
  public schedule/layout parameter, not a prover-controlled proof byte.

### Invariants

- **`ComponentMajor` is byte-identical to today.** Evaluating under
  `ComponentMajor` returns exactly the same field element as the current
  `eval_at_point` for every fixture. Protected by a new regression test
  `component_major_matches_legacy_row_eval`.
- **`ChunkGrouped(1)` collapses to `ComponentMajor`.** Resolving
  `ChunkGrouped(1)` yields a one-chunk `WitnessLayout` whose evaluation equals
  `ComponentMajor` (modulo the `z_first` ordering knob, which `ChunkGrouped` does
  not use). Protected by `chunk_grouped_one_equals_component_major`. This is the
  structural proof that the chunk fold is a true generalization, not a parallel
  path.
- **Structured correctness for every power-of-two chunk count.** For every
  `W ∈ {1, 2, 4, …, num_blocks}` the chunked structured evaluation equals the
  brute-force materialized `M` over the resolved chunk layout (with `z_hat`
  replicated `W` times). Protected by `chunk_grouped_matches_materialized` (see
  Testing Strategy), run for both power-of-two and non-power-of-two `block_len`.
- **Dominant-cost equivalence (α-evaluations).** The shared SIS-matrix
  α-evaluation scan (`eval_ring_at_pows` inside `packed_slice_inner_sum`) runs
  exactly **once per shared-matrix entry**, with the same
  `r_max = max(n_d, n_b, n_a)` and the same `n_cols` regardless of chunk count.
  The chunk-replicated `A·z_hat` enters only through the additively combined,
  α-free `Z_comb` weight vector. The implementation must **not** re-scan or
  re-α-evaluate the setup matrix per chunk.
- **Chunk-partitioned cost is flat in chunk count.** The `e_hat`/`t_hat` block
  summaries cost `O(W · C · B_w) = O(C · B)` in total (the `W` chunk windows
  *tile* the same `B` blocks), and the shared `eq_low` table is built once at
  window `B_w`, not once per chunk.
- **Only chunk-replicated `z_hat` scales with chunk count.** The `z_hat`-tensor
  contribution and the `Z_comb` build are the only `O(W · …)` terms:
  `O(W · block_len + W · DF · DC)` (peelable `block_len`) /
  `O(W · DF · DC · block_len)` (dense `block_len`), and `O(W · A_cols)`
  (`Z_comb`).
- **No-panic boundary preserved.** All chunk counts, per-chunk offsets, the block
  window `B_w`, and the replicated `z_hat` capacity are validated in
  `WitnessType::resolve` (from public layout) before any proof data is read, per
  the [verifier no-panic contract](../book/src/how/verification.md). Malformed
  shape / chunk count / capacity is rejected with `AkitaError`, never a panic.
- **Transcript unchanged in shape, additive in rounds only.** Because `z_hat` is
  chunk-replicated, the next-level witness variable count rises by at most
  `log₂ W`, so stage-2 / setup-product sum-checks read up to `log₂ W` extra round
  polynomials. No new challenge *kinds*; challenge sampling order is unchanged.

### Non-Goals

- **No prover or planner changes.** Verifier-only. Producing the chunk-grouped
  witness, setting `witness_shape` in the schedule, and the distributed prover's
  partial-fold construction are out of scope. Test fixtures synthesize the
  resolved layout directly.
- **Non-power-of-two chunk count.** Only `W` a power of two dividing `num_blocks`
  (so the `e_hat`/`t_hat` chunk window `B_w` is a clean low-bit window) is in
  scope for the peeled fast path. A dense per-chunk fallback for non-power-of-two
  `B_w` is a follow-up (the exact analogue of the `z_hat`-tensor dense fallback).
  **Note:** this is the `B_w` (chunk block window) constraint and is *distinct*
  from `block_len` (the `z` in-block size): a **non-power-of-two `block_len` is
  fully in scope** and specified below (§4 Case 2 / §5), since it already arises
  at recursive levels and the dense fallback exists today.
- **ZK blinding interaction.** The `b_zk` / `d_zk` blinding segments
  (`crates/akita-verifier/src/protocol/slice_mle/zk_blinding.rs`) under chunked
  layout are out of scope; the spec targets the non-zk core evaluation. ZK
  chunking is a follow-up mirroring the same per-chunk offset treatment.
- **Tensor (factored) challenges under chunking.** The flat-`c_alpha` path is the
  primary target. Factored challenges require restricting the block-window to a
  chunk's contiguous range (which lands in the `left` factor of the left⊗right
  block factorization); a sub-task that may land separately (see
  Implementation Stages).

## Evaluation

### Acceptance Criteria

- [ ] `WitnessType`, `WitnessChunkOffset`, `WitnessLayout` exist in `akita-types`, and
  `WitnessType::resolve` yields `1` chunk for `ComponentMajor` and `W` for
  `ChunkGrouped(W)`.
- [ ] `RingSwitchDeferredRowEval::eval_at_point` evaluates `e_hat`, `t_hat`,
  `z_hat`, and the setup contribution as a fold over `chunk_layout.chunks()`,
  reusing the existing `EStructuredSlicesEvaluator` /
  `TStructuredSlicesEvaluator` / `ZStructuredPow2SlicesEvaluator` /
  `ZDenseSlicesEvaluator` unchanged in body.
- [ ] `SetupContributionPlan::prepare` builds `e_eq_slice`,
  `t_eq_slice_per_group`, and `Z_comb` (`z_eq_slice`) against the `WitnessLayout`;
  `packed_slice_inner_sum` is unchanged.
- [ ] `component_major_matches_legacy_row_eval` passes (byte-identical to legacy).
- [ ] `chunk_grouped_one_equals_component_major` passes.
- [ ] `chunk_grouped_matches_materialized` passes for
  `W ∈ {1, 2, 4, 8, …, num_blocks}` with `block_len` a power of two (Case 1).
- [ ] `chunk_grouped_matches_materialized_z_dense` passes for the same `W` with
  `block_len` **not** a power of two (Case 2, dense fallback), mirroring the
  existing single-segment `z_dense_matches_materialized_range_inner_product`
  (`block_len = 510`).
- [ ] A negative test rejects malformed shape (chunk count not a power of two,
  `W ∤ num_blocks`, or `W` replicated `z_hat` exceeds witness capacity) with
  `AkitaError`, no panic.
- [ ] `cargo fmt -q`, `cargo clippy --all -- -D warnings`, `cargo test` pass.

### Testing Strategy

New tests live next to the structured-slice tests
(`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`) and the
ring-switch tests (`crates/akita-verifier/src/protocol/ring_switch/tests.rs`),
reusing the existing `StructuredFixture`/`fixture()` shape
(`F = Prime128OffsetA7F7`, `D = 32`, `num_blocks = 8`, `block_len = 512`).

1. **`chunk_grouped_matches_materialized` (ground truth).** For each
   `W ∈ {1, 2, 4, 8}` (powers of two dividing `num_blocks = 8`):
   - Resolve the layout via `WitnessType::ChunkGrouped(W).resolve(...)`.
   - **Materialize** the full `M` row contribution as a flat vector over
     `[ e0|t0|z0 ]…[ e(W-1)|t(W-1)|z(W-1) ][ r ]`, using the per-cell formulas
     from the theory chapter (`e_hat`: `c_alpha[claim][blk_g]·g_open`; `t_hat`:
     with the `a_row` axis; `z_hat`: `g_commit·g_fold·a[blk]`, **the same in
     every chunk** — replicated; `c_alpha` read at the *global* block
     `blk_g = chunk·B_w + block_local` for `e_hat`/`t_hat`).
   - Form `eq(·, r_col)` densely over that layout, inner-product against the row
     weights, and compare to the chunked structured evaluation. This is the
     analogue of the existing
     `e_structured_matches_materialized_range_inner_product` /
     `z_structured_matches_materialized_range_inner_product` tests, extended over
     the chunk axis. The loop over `W` proves "any power-of-two chunk count."
2. **`chunk_grouped_matches_materialized_z_dense` (non-pow2 `block_len`).** Same
   structure as (1) with a fixture whose `block_len` is **not** a power of two
   (e.g. `block_len = 510`, as in the existing single-segment
   `z_dense_matches_materialized_range_inner_product`). Drives the §4 Case 2 /
   §5 dense `Z_comb` paths (`ZDenseSlicesEvaluator` summed over chunks, dense
   `z_eq_slice` summed over chunks). Loop `W ∈ {1, 2, 4, 8}`. The materialized
   `z_hat` is **replicated** (same `g_commit·g_fold·a[blk]` in every chunk, only
   the offset shifts).
3. **`component_major_matches_legacy_row_eval` (regression).** Evaluate under
   `ComponentMajor` and assert equality with the current single-segment result on
   the existing fixture (both `block_len` pow2 and non-pow2).
4. **`chunk_grouped_one_equals_component_major`.** Assert
   `ChunkGrouped(1).resolve(...)` evaluates to the same value as
   `ComponentMajor.resolve(...)` (with the `[e|t|z]` ordering), confirming the
   one-chunk collapse.
5. **Per-component equivalence.** Optionally split (1)/(2) into `e_hat`-only,
   `t_hat`-only, `z_hat`-only, and setup-only sub-tests for sharper failure
   localization.
6. **No-panic negatives.** `chunk_grouped_rejects_bad_shape`: assert `AkitaError`
   (not panic) for `ChunkGrouped(3)` (not pow2), `ChunkGrouped(16)`
   (`> num_blocks`), and a chunk count whose `W · z_len` exceeds the validated
   `w_len`.
7. **End-to-end.** Existing `crates/akita-pcs` / verifier integration tests must
   continue passing unchanged (they exercise `ComponentMajor`).

### Performance

- **Dominant term unchanged.** The setup-matrix α-evaluation scan
  (`O(r_max · n_cols · D)` ring ops) must be identical to `ComponentMajor` for
  any chunk count; verify via the profile harness
  (`AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`)
  comparing a `ComponentMajor` run with a synthetic `ChunkGrouped(W)` run on the
  `setup_contribution` span — the span time must be flat in `W`.
- **`e_hat`/`t_hat`/`r` flat in chunk count.** Block-summary build is `O(C·B)`
  total regardless of `W`; the only `W`-scaled additions are the cheap digit-axis
  outer combines (`O(W·C·do)`, `O(W·n_A·C·do)`) and the high tables (`O(W·…)`),
  far below the α-eval floor.
- **`z_hat` and `Z_comb` scale with chunk count** by design — acceptable because
  the `z_hat`-tensor is the cheap (α-free) part of the verifier.
- **Net expectation:** total verifier time `≈ ComponentMajor + O(W·block_len) +
  O(W·A_cols)`, negligible whenever `W ≪ r_max·D`.

## Detailed Review

### Architecture Review

The proposed architecture is directionally right: it normalizes the public witness
layout once into `WitnessLayout`, then keeps the verifier hot path expressed as a
fold over resolved chunks. That is the correct boundary because today's code has
three independent consumers of witness column geometry:

- `RingSwitchDeferredRowEval::eval_at_point`, which owns the structured
  `e_hat`/`t_hat`/`z_hat` and `r` contributions.
- `SetupContributionPlan::prepare`, which translates the same geometry into
  column-equality weights for the packed setup scan.
- `PreparedChallengeEvals::summarize_all_block_carries`, which binds the
  challenge vector's global block axis to the verifier's low-bit peeled window.

Factoring the layout into `akita-types` is also the right crate boundary. The
layout is not verifier-local: both the direct verifier setup scan and the
setup-product/bar-omega path need the same physical column mapping. Keeping one
definition avoids the most dangerous failure mode: the structured witness
contribution and the setup contribution silently evaluating different column
layouts.

The main architecture requirement is that `WitnessType::resolve` must become a
real verifier boundary, not just a convenience constructor. It should receive, or
be followed immediately by validation against, the validated witness column
capacity (`w_len / D`). Without that, the spec's no-panic invariant cannot be
fully enforced at the shape boundary; `eval_at_point` would still be able to
construct offsets that are algebraically valid but outside the committed witness
column domain. In implementation terms, either make `witness_len` part of
`WitnessTypeResolveInputs`, or add a non-optional
`WitnessLayout::validate_capacity(witness_len, r_tail_len)` call in
`prepare_ring_switch_row_eval` before the layout is stored.

`witness_shape` must also be transcript/descriptor-bound as a public layout
parameter. The proof should not choose it, but the verifier and prover must agree
on it through the same public schedule/layout descriptor that already binds the
level parameters. Adding `LevelParams::witness_shape` without including it in the
canonical descriptor/serialization path would create a configuration mismatch
risk: two executions could absorb the same commitments and proof bytes while
interpreting the relation matrix columns differently.

One current-code edge needs an explicit decision before implementation:
`RingRelationSegmentLayout` has `offset_u` and tiered-commitment replay uses it in
both `SetupContributionPlan::prepare` and `u_recompose_contribution`, but the
new `WitnessChunkOffset` only models `e_offset`, `t_offset`, and `z_offset`. Therefore
the first implementation must either:

- reject `ChunkGrouped(W)` when `lp.tier_split > 1` with `AkitaError` and state
  that chunked tiered commitments are outside this spec's first landing, or
- extend `WitnessChunkOffset`/`WitnessLayout` with `u_offset` (and the corresponding
  chunked `u` geometry) so tiered replay remains layout-generic.

The conservative path is to reject chunked+tiered at first, because the spec does
not yet define whether `û_concat` is partitioned with `t_hat`, replicated, or
kept as a shared segment. That rejection must be public-layout validation, not a
late panic or hidden assumption.

The `ComponentMajor` compatibility story is good but should be tested at two
levels. A direct result comparison protects behavior, while a resolved-layout
snapshot test protects the intended geometry (`offset_e`, `offset_t`, `offset_z`,
`offset_u`, `offset_r`). The latter is useful because a future refactor can
preserve one fixture's final scalar while still moving offsets in a way that
breaks setup contribution or ZK/tiered follow-ups.

### Efficiency Review

The strongest efficiency choice is to keep `packed_slice_inner_sum` unchanged and
feed it precombined weights. This preserves the dominant α-evaluation count:
every shared setup row entry is evaluated once, independent of `W`. Any
implementation that loops over chunks outside the packed setup scan is
architecturally simpler but unacceptable; it would multiply the verifier's
largest term by the chunk count.

The `e_hat`/`t_hat` plan is efficient because chunks tile the block axis. The
total challenge-summary work is `Σ_chunk O(C·B_w) = O(C·B)`, and the setup
weights still have one entry per original D/B setup column. To preserve that
property in code, the helper should map each *global* block index to exactly one
chunk:

```text
chunk_idx   = global_block / blocks_per_chunk
block_local = global_block % blocks_per_chunk
```

No setup column should be duplicated across chunks for `e_hat` or `t_hat`.

The high-equality tables need careful implementation. A single dense table from
the first chunk's high offset to the last chunk's high offset can accidentally
include large gaps introduced by the per-chunk `[e|t|z]` stride, especially
because every chunk carries a full replicated `z_hat`. Prefer per-chunk high
tables (or an absolute-index cache) for `e_hat` and `t_hat`:

```text
eq_hi_e[chunk][local_high_idx]
eq_hi_t[chunk][local_high_idx]
```

This keeps precompute cost proportional to the useful high-index domain
(`O(W·C·depth_open)` and `O(W·n_A·num_t_vectors·depth_open)`) rather than to the
span between sparse absolute offsets.

The `z_hat` cost scaling is real and acceptable, but it should be made explicit
in benchmarks. For power-of-two `block_len`, chunking adds one low-window carry
summary per chunk and one high-table combine per chunk. For non-power-of-two
`block_len`, the dense fallback can become allocation-heavy if it materializes
the same logical `z` segment once per chunk. The implementation should either
materialize the dense `z` segment once and evaluate it at `W` offsets, or stream
the offset-eq accumulation per offset without retaining `W` copies.

The performance acceptance test should measure two spans separately:

- `setup_contribution`: must be flat in `W` except for `Z_comb` precompute.
- `z_structured`: may grow with `W` and should match the expected `O(W·block_len)`
  or dense-fallback profile.

That split is important because total verifier time can still rise slightly with
chunk count; the invariant is not "no cost changes," but "the α-evaluation floor
does not multiply by `W`."

## Design

### Architecture

This section is the heart of the spec: how to evaluate two (and, later, more)
witness layouts through **one** computation path rather than branching the
verifier hot loop on layout.

#### The problem with the obvious approach

`eval_at_point` today is a straight-line sequence of contribution calls, each
hard-wired to a single layout assumption: `e_hat` lives at one `offset_e` and
spans all `num_blocks`; `t_hat` at one `offset_t`; `z_hat` at one `offset_z`;
the SIS scan maps columns to those single offsets; `r` tails the lot. Adding the
chunk-grouped layout by "branching on the shape" inside each of these — and
inside `SetupContributionPlan::prepare`, and inside the `c_alpha` summary builder
— would scatter the same `if chunked { … } else { … }` across every component,
duplicate the subtle peeled-block arithmetic, and make the no-panic surface (and
the dominant-cost guarantee) far harder to audit. We want the opposite: the
layout choice resolved **once**, at the edge, into a representation the hot path
treats uniformly.

#### The unifying observation: every layout is a list of chunks

The two layouts are not arbitrarily different. In both, the witness is a
concatenation of **chunks**, where a chunk is a contiguous run holding one slice
of each component, and the row MLE is **additive over chunks**:

```text
M̃(r) = Σ_chunk [ e_contribution(chunk) + t_contribution(chunk)
                 + z_contribution(chunk) ]
        + setup_contribution(all chunks)        # one matrix scan, weights summed
        + r_contribution()                       # single shared tail
```

- `ComponentMajor` is the **degenerate one-chunk** case: a single chunk whose
  `e_hat` spans all `num_blocks`, whose `z_hat` is the lone fold, and whose
  offsets are exactly today's `RingRelationSegmentLayout`.
- `ChunkGrouped(W)` is the **`W`-chunk** case: each chunk's `e_hat`/`t_hat`
  spans `B_w = num_blocks / W` blocks, and each chunk carries a full-`block_len`
  `z_hat`.

That additivity is exactly the verifier requirement stated up front: the
established single-segment layout is just the one-chunk case. The single-segment
verifier *is* the chunk fold with one iteration.

#### The common denominator: `WitnessLayout`

The design pivots on a single resolution step:

```text
WitnessType  ──resolve──▶  WitnessLayout { blocks_per_chunk, chunks[], offset_u, offset_r }
```

`WitnessLayout` is the *only* thing the evaluators see. They never branch on
`WitnessType`; they fold over `chunk_layout.chunks()`. All layout divergence —
`z_first`, blinding offsets, the chunk-offset formula — is confined to the two
arms of `resolve`, which is also where every no-panic check lives (chunk count is
a power of two, divides `num_blocks`, `B_w` is a clean window, `W·L + |r|` fits
the validated `w_len`). The hot path inherits the guarantee that all bounds were
checked once, at the edge.

This placement also fixes the crate boundary cleanly: `WitnessType` /
`WitnessChunkOffset` / `WitnessLayout` live in `akita-types` next to
`RingRelationSegmentLayout`, so the verifier's `ring_switch.rs` and the
`akita-types` `SetupContributionPlan` consume one definition — no duplicated
layout knowledge across the verifier/types boundary.

#### Tile vs. replicate falls out of the data, not control flow

The subtle part of the two layouts is that components behave differently under
chunking: `e_hat`/`t_hat` are **partitioned** (the union of chunks' pieces is the
whole component, each chunk covering a disjoint block sub-range), while `z_hat`
is **replicated** (each chunk carries a *full* fold). A clean architecture must
express this without a per-component "am I partitioned?" switch.

The chunk representation does this implicitly:

- A chunk's `e_hat`/`t_hat` pieces are addressed by `(global_block_base,
  blocks_per_chunk)`. Because the chunks' windows are disjoint and cover
  `[0, num_blocks)`, **summing** their contributions reconstructs the full
  component, and the total block scan is `Σ_chunk B_w = B` — flat in `W`. Tiling
  is just "disjoint windows that sum to the whole," which the fold gives for
  free.
- A chunk's `z_hat` piece is addressed by `z_offset` and always spans the full
  `block_len`. There are `#chunks` such pieces, so summing them yields `#chunks`
  copies. Replication is just "every chunk carries the full thing," again free
  from the fold.

No evaluator asks whether a component tiles or replicates; the geometry of the
chunk (`B_w`-window for `e`/`t`, full `block_len` for `z`) encodes it. The cost
asymmetry (`e`/`t` flat, `z` ×`W`) is therefore a *consequence* of the data
shape, not a special case in code.

#### Three layers

The consolidated design is three thin layers, with the case-split living only in
layer 1:

1. **Resolve (layout, shape-aware).** `WitnessType::resolve -> WitnessLayout`.
   Two arms; all validation. `ComponentMajor` reads `RingRelationSegmentLayout`
   (one chunk, `B_w = num_blocks`, `global_block_base = 0`). `ChunkGrouped(W)`
   computes per-chunk offsets from `L = |e^j| + |t^j| + |z^j|`:

   ```text
   |e^j| = depth_open · num_claims · B_w
   |t^j| = depth_open · n_A · num_t_vectors · B_w
   |z^j| = depth_fold · depth_commit · block_len        # full size, replicated
   chunk j:  e_offset = base + j·L
             t_offset = base + j·L + |e^j|
             z_offset = base + j·L + |e^j| + |t^j|
             global_block_base = j·B_w
   offset_r = base + W·L
   ```

2. **Per-chunk evaluate (shape-agnostic, reuses today's evaluators).** The
   existing `EStructuredSlicesEvaluator` / `TStructuredSlicesEvaluator` /
   `ZStructuredPow2SlicesEvaluator` / `ZDenseSlicesEvaluator` are instantiated
   once per chunk with the chunk's offsets; their bodies are unchanged. Only
   their *inputs* are chunk-derived (offset, `global_block_base`, the peel window
   `B_w`).

3. **Fold (orchestrator, shape-agnostic).** `eval_at_point` builds the
   chunk-independent precomputes once (the shared `eq_low` at window `B_w`, the
   shared `eq_low_z` at window `block_len`, gadgets, `alpha_pows`), then sums
   per-chunk contributions and adds the single setup + `r` contributions.

#### `e_hat` / `t_hat` per chunk (partitioned)

Per chunk, the only new helper is the chunk-windowed `c_alpha` summary:

```rust
// Per-chunk generalization of summarize_all_block_carries. Chunk `c` reads the
// global block range [c.global_block_base, c.global_block_base + B_w).
fn summarize_chunk_block_carries(
    &self, num_claims, global_block_base, blocks_per_chunk /* B_w */,
    num_blocks, x_low_challenges, eq_low /* len B_w */, offset_low /* e_offset mod B_w */,
) -> Result<Vec<[E; 2]>, AkitaError>;
// flat slice = c_alpha[claim*num_blocks + global_block_base .. + B_w]
```

`ComponentMajor` ⇒ `global_block_base = 0`, `B_w = num_blocks` ⇒ today's
`summarize_all_block_carries`. The `t_hat` evaluator reuses the chunk's `e_hat`
summaries (same in-window residue, since `|e^j|` is a multiple of `B_w`). `eq_low`
(window `B_w`) and `high_challenges` are built once and shared across chunks.

#### `z_hat` per chunk (replicated)

The in-block weight `a[blk]` is chunk-independent (the fold rows carry no
chunk-specific data — `a[blk]` and the gadget weights are global), so only the
offset differs. The verifier dispatches on `block_len.is_power_of_two()` exactly
as today; the dispatch depends on `block_len`, not on the chunk, so the chunk
loop sits **outside** the case split. Both cases combine additively:
`z_contribution = Σ_chunk Z_eval(chunk.z_offset)`.

- **Case 1 — `block_len` a power of two (root).** Peel the `block_len` window per
  chunk; build the two-bucket in-block summary with `z_lo = z_offset mod
  block_len` (the `eq_low_z` table is built once and shared — it depends only on
  `r_col`'s low bits and the window size, never on the offset), evaluate with
  `ZStructuredPow2SlicesEvaluator { offset_z: chunk.z_offset }`, sum. Overhead
  `O(W·block_len + W·DF·DC)`.
- **Case 2 — `block_len` not a power of two (recursive, dense).** No clean
  low-bit window to peel (`block_len = ceil(num_ring / num_blocks)` need not be a
  power of two at recursive levels). Fall back to materializing the structured
  `z` segment and running one generic offset-eq evaluation — per chunk, i.e.
  today's `ZDenseSlicesEvaluator` repeated `W` times, each at its own
  `chunk.z_offset`:

  ```rust
  z_contribution += ZDenseSlicesEvaluator {
      g1_commit, fold_gadget,
      consistency_weight: self.eq_tau1[0],
      a_evals_by_point,                 // shared across chunks (a[blk] is global)
      full_vec_randomness: x_challenges,
      offset_z: chunk.z_offset,         // the only per-chunk input
      block_len,
  }.evaluate()?;
  ```

  Since `a_evals_by_point` is shared, the materialized segment is identical in
  every chunk; an implementation may materialize the `O(DF·DC·block_len)` segment
  **once** and evaluate it at the `W` offsets. Overhead `O(W·DF·DC·block_len)`.
  Acceptable: the dense fallback only occurs at recursive levels where the
  witness (hence `block_len`) has shrunk; the dominant root level is Case 1.

For `W = 1` both cases reduce to today's single evaluator call at `offset_z`.

#### Setup contribution per chunk (dominant; α-evaluations once)

`SetupContributionPlan::prepare` receives the `WitnessLayout`. The three
precomputed column-weight vectors are built against the chunks; the hot loop
`packed_slice_inner_sum` (and its eq-weighted twin `bar_omega_segment_eval`) is
**unchanged** — this is what keeps the α-evaluation count layout-independent.

- **`W_col` / `e_eq_slice` (`D·e_hat`, partitioned).** `get_eq_indices_for_d`
  gains a chunk split: decode the SIS column to `(dig, blk_g, claim)` as today,
  then map the global block to its chunk:

  ```text
  chunk_idx    = blk_g / B_w
  block_local  = blk_g % B_w
  block_sum    = (chunks[chunk_idx].e_offset mod B_w) + block_local
  low_eq_idx   = block_sum & (B_w - 1)
  block_carry  = block_sum >> log₂(B_w)
  high_eq_idx  = (chunks[chunk_idx].e_offset >> log₂(B_w))
                 + (dig*num_claims + claim) + block_carry
  W_col[c]     = eq_low[low_eq_idx] * eq_high_e[high_eq_idx]
  ```

  `eq_low` is the shared `B_w`-window table; `eq_high_e` is the existing high
  table indexed at the per-chunk offset base (the "chunk axis" of size `W·C·do`
  is just `W` offset bases into one high table). `ComponentMajor` ⇒ one chunk,
  `chunk_idx = 0`, `block_local = blk_g` ⇒ today's `get_eq_indices_for_d`.
  **Footprint unchanged**: each SIS column maps to exactly one chunk's `e_hat`
  piece (partition), so `d_required`/`n_cols_e` are the same.
- **`T_col` / `t_eq_slice_per_group` (`B·t_hat`, partitioned).** Same chunk split
  in `get_eq_indices_for_b`, with the extra `a_row` axis and per-group sparsity;
  footprint unchanged.
- **`Z_comb` / `z_eq_slice` (`A·z_hat`, replicated).** For each `A` column `c`,
  sum the per-chunk `z_hat`-equality weight over all chunks:

  ```text
  Z_comb[c] = Σ_chunk z_weight(c, chunk.z_offset)
  ```

  Following the **same two `block_len` cases as §`z_hat` per chunk**, `prepare`
  dispatches once on `block_len.is_power_of_two()` and loops the chunk axis
  inside:
  - **`block_len` pow2:** reuse the per-chunk peeled in-block weights — the
    `s_per_dc_per_carry` table is rebuilt per chunk (its high offset
    `z_offset >> log₂(block_len)` differs), the shared `z_block_low_eq` window
    table is built once. Build cost `O(W·A_cols)`.
  - **`block_len` not pow2 (dense):** build the dense `z_eq_slice` per chunk via
    the existing one-shot peeled-equality cache (today's non-pow2 branch),
    summing into `Z_comb`. Build cost `O(W·A_cols)`.

  Either way the output length is `z_range = inner_width` (**unchanged**), so the
  downstream scan and its α-evaluation count are identical to `ComponentMajor`;
  only the precomputed `Z_comb` weights are summed over chunks. One chunk ⇒
  today's `z_eq_slice`.

Because `A`/`B`/`D` are the same seed-expanded matrix for every chunk, the scan
range (`r_max`, `n_cols`) and α-eval count are layout-independent. `Z_comb` is the
*only* place the chunk count enters the setup contribution, and it is α-free.

#### `r`-tail and tail segments

`compute_r_contribution` is **unchanged**: a single summed quotient `r` tails the
whole witness at `chunk_layout.offset_r`. Only the offset value changes (end of
the resolved layout); the evaluator body and cost are identical. The tiered
`u_recompose` term and (out of scope here) ZK blinding consume offsets from the
resolved layout the same way.

### Alternatives Considered

- **Branch on `WitnessType` inside each component evaluator.** Rejected: it
  duplicates the peeled-block arithmetic and the no-panic checks across every
  component and the setup planner, and it puts a layout `if` inside the hot loop.
  The chunk-list resolution keeps the case-split at one edge (`resolve`) and the
  hot path uniform.
- **Per-component "segment iterator" trait (no chunk object).** Each component
  exposes its own `Iterator<Item = (offset, window, block_base)>`, decoupling
  components from the chunk concept. Rejected as weaker: it hides the fact that
  `e`/`t`/`z` of one chunk share a `global_block_base` and offset region, makes
  the `t_hat`-reuses-`e_hat`-summaries sharing awkward to express, and offers no
  benefit since both layouts are genuinely chunk-structured.
- **General `ColumnAddressing` strategy object** mapping logical `(component,
  block, claim, dig)` to physical columns. The most general (could absorb
  `z_first`, blinding, arbitrary future layouts) but heavyweight: an indirection
  on the hot index path and far more surface than two layouts justify. The
  `WitnessLayout` is the minimal common denominator for the layouts that actually
  exist; revisit the strategy object only if a third, non-chunk-structured layout
  appears.
- **Collapse `ComponentMajor` into `ChunkGrouped(1)` (single variant).**
  Tempting, since `ChunkGrouped(1)` resolves to the same one-chunk evaluation.
  Rejected for two reasons: (a) `ComponentMajor` must preserve the established
  `z_first` ordering and blinding offsets sourced from `RingRelationSegmentLayout`,
  which `ChunkGrouped` does not model; (b) the named variant communicates intent
  at call sites and in the schedule. The equivalence is captured instead as the
  `chunk_grouped_one_equals_component_major` test invariant, which is the right
  place for it.
- **Re-scan the SIS matrix per chunk.** Rejected: it multiplies the dominant
  α-evaluation cost by the chunk count, violating the dominant-cost invariant.
  The `Z_comb` pre-combine keeps the scan single-pass.

## Documentation

- The chunked-layout theory (component-by-component cost, partitioned vs
  replicated, why the α-scan is unchanged) lives in the book at
  [`book/src/how/verifying/distributed-relation-verifier.md`](../book/src/how/verifying/distributed-relation-verifier.md);
  the single-segment row-eval is
  [`book/src/how/verifying/matrix_evaluation.md`](../book/src/how/verifying/matrix_evaluation.md);
  and the prover side is
  [`book/src/how/proving/distributed-prover.md`](../book/src/how/proving/distributed-prover.md).
  This spec is the implementation record; on land set `Status: implemented`, fill
  `PR:`, reference the new symbols (`WitnessType`, `WitnessLayout`,
  `summarize_chunk_block_carries`). When stable, fold the witness-shape
  abstraction into `matrix_evaluation.md` and archive per
  [`specs/PRUNING.md`](PRUNING.md).
- Update the verifier no-panic audit
  ([`docs/verifier-panic-audit.md`](../docs/verifier-panic-audit.md)) with the
  `WitnessType::resolve` boundary checks (shape / chunk-count / capacity rows).
- No `AGENTS.md` crate-graph change (no new crate; new types in `akita-types`).

## Implementation Stages

The implementation should land in small, reviewable stages. Each stage below has
a clear invariant, the code shape expected at the end of the stage, and the tests
that prove it before moving on.

### Stage 0 — Scope and Descriptor Boundary

Before changing the hot path, lock down two public-boundary decisions:

1. `witness_shape` is a public schedule/layout parameter and is included anywhere
   `LevelParams` participates in canonical descriptor bytes, schedule snapshots,
   or generated table identity.
2. `ChunkGrouped(W)` with `lp.tier_split > 1` is either rejected for the first
   landing or the spec is extended with explicit `û_concat` chunk geometry. The
   recommended first landing is rejection, because `u` is not defined in
   `WitnessChunkOffset`.

Expected code shape:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitnessType {
    ComponentMajor,
    ChunkGrouped(usize),
}

impl Default for WitnessType {
    fn default() -> Self {
        Self::ComponentMajor
    }
}

// Inside LevelParams.
pub witness_shape: WitnessType,
```

The early rejection, if tiered chunking is deferred, should happen before proof
data is used:

```rust
if matches!(lp.witness_shape, WitnessType::ChunkGrouped(_)) && lp.tier_split > 1 {
    return Err(AkitaError::InvalidSetup(
        "chunk-grouped verifier layout for tiered commitments is not specified".into(),
    ));
}
```

Tests:

- A descriptor/serialization snapshot changes when `witness_shape` changes.
- Existing schedules deserialize/build with `WitnessType::ComponentMajor`.
- `ChunkGrouped(W)` with `tier_split > 1` returns `AkitaError` if tiered chunking
  is deferred.

### Stage 1 — Layout Types and Capacity-Checked Resolution

Add the resolved layout types in `akita-types`, close to
`RingRelationSegmentLayout`. The important implementation detail is that
resolution validates both arithmetic and witness capacity; it should not leave
capacity checks to later indexing code.

Expected code shape:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkOffset {
    pub e_offset: usize,
    pub t_offset: usize,
    pub z_offset: usize,
    pub global_block_base: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkOffset>,
    pub offset_u: usize,
    pub offset_r: usize,
}

pub struct WitnessTypeResolveInputs {
    pub num_claims: usize,
    pub num_t_vectors: usize,
    pub m_row_layout: MRowLayout,
    pub witness_ring_len: usize,
}
```

`ComponentMajor` delegates to the existing layout and becomes a one-chunk
adapter:

```rust
let legacy = relation.segment_layout(lp)?;
WitnessLayout {
    blocks_per_chunk: lp.num_blocks,
    chunks: vec![WitnessChunkOffset {
        e_offset: legacy.offset_e,
        t_offset: legacy.offset_t,
        z_offset: legacy.offset_z,
        global_block_base: 0,
    }],
    offset_u: legacy.offset_u,
    offset_r: legacy.offset_r,
}
```

`ChunkGrouped(W)` computes `[e^j|t^j|z^j]` with checked arithmetic:

```rust
let blocks_per_chunk = lp.num_blocks.checked_div(w).ok_or(AkitaError::InvalidSetup(...))?;
let e_len_j = depth_open * inputs.num_claims * blocks_per_chunk;
let t_len_j = depth_open * lp.a_key.row_len() * inputs.num_t_vectors * blocks_per_chunk;
let z_len_j = depth_fold * depth_commit * lp.block_len;
let chunk_stride = e_len_j + t_len_j + z_len_j;

let chunks = (0..w)
    .map(|j| WitnessChunkOffset {
        e_offset: j * chunk_stride,
        t_offset: j * chunk_stride + e_len_j,
        z_offset: j * chunk_stride + e_len_j + t_len_j,
        global_block_base: j * blocks_per_chunk,
    })
    .collect::<Vec<_>>();
let offset_r = w * chunk_stride;
```

Validation checklist inside `resolve`:

- `W > 0`, `W.is_power_of_two()`, and `W <= num_blocks`.
- `num_blocks % W == 0`.
- `blocks_per_chunk.is_power_of_two()` for the peeled fast path.
- every offset/length uses checked arithmetic.
- `global_block_base + blocks_per_chunk <= num_blocks`.
- `offset_r + r_tail_len <= witness_ring_len` (or an equivalent validated
  capacity bound).

Tests:

- `component_major_resolves_to_legacy_segment_layout`.
- `chunk_grouped_one_resolves_to_single_chunk`.
- `chunk_grouped_offsets_are_contiguous_and_cover_blocks`.
- bad `W`, overflow-shaped inputs, and too-short witness capacity return
  `AkitaError`.

### Stage 2 — Prepare-Time Wiring

Change `RingSwitchDeferredRowEval` to store `WitnessLayout` instead of
`RingRelationSegmentLayout`. Do this before changing evaluation logic so the
compiler reveals every caller that still expects single offsets.

Expected code shape:

```rust
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    // ...
    pub(crate) chunk_layout: WitnessLayout,
}

pub(crate) fn chunk_layout(&self) -> &WitnessLayout {
    &self.chunk_layout
}
```

`prepare_ring_switch_row_eval` should resolve the layout after existing shape
checks and before returning the prepared evaluator:

```rust
let witness_ring_len = w_len / D;
let chunk_layout = lp.witness_shape.resolve(&WitnessTypeResolveInputs {
    num_claims,
    num_t_vectors,
    m_row_layout,
    witness_ring_len,
})?;
```

If `prepare_ring_switch_row_eval` does not currently receive `w_len`, thread it
through from `ring_switch_verifier_core`; the verifier already validates `w_len`
and computes `num_ring_elems`, so this is the right boundary.

Tests:

- Existing verifier tests still pass under `ComponentMajor`.
- A focused prepare test asserts `RingSwitchDeferredRowEval::chunk_layout()` is
  one chunk for default levels.

### Stage 3 — Chunk-Window Challenge Summaries

Generalize `summarize_all_block_carries` into a chunk-window helper. The flat
path is the first implementation target; the tensor path can preserve current
behavior for `ComponentMajor` and reject chunk windows until the follow-up lands.

Expected code shape:

```rust
pub(crate) fn summarize_chunk_block_carries<Base, const D: usize>(
    &self,
    num_claims: usize,
    x_low_challenges: &[F],
    eq_low: &[F],
    offset_low: usize,
    global_block_base: usize,
    blocks_per_chunk: usize,
    num_blocks: usize,
) -> Result<Vec<[F; 2]>, AkitaError>
where
    Base: FieldCore + FromPrimitiveInt,
    F: MulBase<Base>,
{
    match self {
        Self::Flat(c_alphas) => (0..num_claims)
            .map(|claim_idx| {
                let claim_start = claim_idx.checked_mul(num_blocks).ok_or_else(...)?;
                let start = claim_start.checked_add(global_block_base).ok_or_else(...)?;
                let end = start.checked_add(blocks_per_chunk).ok_or_else(...)?;
                let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                    expected: end,
                    actual: c_alphas.len(),
                })?;
                summarize_pow2_block_carries(eq_low, offset_low, values)
            })
            .collect(),
        Self::Tensor { .. } if blocks_per_chunk == num_blocks && global_block_base == 0 => {
            self.summarize_all_block_carries::<Base, D>(
                num_claims, x_low_challenges, eq_low, offset_low, num_blocks,
            )
        }
        Self::Tensor { .. } => Err(AkitaError::InvalidInput(
            "chunked tensor challenge summaries are not implemented".into(),
        )),
    }
}
```

Tests:

- For `global_block_base = 0`, `blocks_per_chunk = num_blocks`, the new helper
  matches `summarize_all_block_carries`.
- For `W ∈ {1,2,4,8}`, the flat helper matches a direct dense summary over the
  corresponding `c_alpha[claim][global_block_base..][..B_w]` window.
- Tensor `ComponentMajor` still passes existing tensor summary tests.
- Tensor `ChunkGrouped` returns `AkitaError` until the follow-up is implemented.

### Stage 4 — Structured `eval_at_point` Chunk Fold

Refactor `eval_at_point` so all layout-specific offsets come from
`chunk_layout.chunks`. The function still builds shared precomputes once, then
folds chunk contributions.

Expected code shape:

```rust
let layout = self.chunk_layout();
let block_bits = layout.blocks_per_chunk.trailing_zeros() as usize;
let eq_low = EqPolynomial::evals(&x_challenges[..block_bits])?;
let high_challenges = &x_challenges[block_bits..];

let mut e_structured_contribution = E::zero();
let mut t_structured_contribution = E::zero();
let mut z_structured_contribution = E::zero();

for chunk in &layout.chunks {
    let block_offset_low = chunk.e_offset & (layout.blocks_per_chunk - 1);
    let summaries = self.c_alphas.summarize_chunk_block_carries::<F, D>(
        self.num_claims,
        &x_challenges[..block_bits],
        &eq_low,
        block_offset_low,
        chunk.global_block_base,
        layout.blocks_per_chunk,
        self.num_blocks,
    )?;

    e_structured_contribution += EStructuredSlicesEvaluator {
        high_challenges,
        offset_high: chunk.e_offset >> block_bits,
        gadget_vector: &g1_open,
        challenge_block_summaries: &summaries,
        challenge_weight: self.eq_tau1[0],
    }.evaluate();

    t_structured_contribution += TStructuredSlicesEvaluator {
        high_challenges,
        offset_high: chunk.t_offset >> block_bits,
        gadget_vector: &g1_open,
        challenge_block_summaries: &summaries,
        a_row_weights: &self.eq_tau1[a_start..self.rows],
    }.evaluate();
}
```

The `z_hat` dispatch should branch once on `block_len.is_power_of_two()` and loop
inside the selected case:

```rust
if self.block_len.is_power_of_two() {
    for chunk in &layout.chunks {
        let z_offset_low = chunk.z_offset & (self.block_len - 1);
        let a_block_summary = vec![summarize_pow2_multiplier_block_carries(...)?];
        z_structured_contribution += ZStructuredPow2SlicesEvaluator {
            high_challenges: &x_challenges[z_offset_low_bits..],
            offset_high: chunk.z_offset >> z_offset_low_bits,
            // ...
        }.evaluate();
    }
} else {
    let a_evals_by_point = vec![/* build once */];
    for chunk in &layout.chunks {
        z_structured_contribution += ZDenseSlicesEvaluator {
            offset_z: chunk.z_offset,
            a_evals_by_point: &a_evals_by_point,
            // ...
        }.evaluate()?;
    }
}
```

`r_contribution` stays a single call at `layout.offset_r`. If tiered chunking is
deferred, `u_recompose_contribution` remains unchanged for `ComponentMajor` and
is unreachable for `ChunkGrouped`.

Tests:

- `component_major_matches_legacy_row_eval` for pow2 and non-pow2 `block_len`.
- `chunk_grouped_one_equals_component_major` under the non-tiered fixture.
- `chunk_grouped_matches_materialized` for structured-only `e/t/z/r`.
- per-component tests (`e_only`, `t_only`, `z_only`) if the combined test is hard
  to debug.

### Stage 5 — Setup Contribution Chunk Weights

Change `SetupContributionPlan::prepare` to receive `&WitnessLayout` instead of
four single offsets. The packed scan remains unchanged; only the precomputed
weight vectors become chunk-aware.

Expected signature:

```rust
pub fn prepare<F>(
    inputs: &SetupContributionPlanInputs<E>,
    full_vec_randomness: &[E],
    eq_low: Option<&[E]>,
    z_block_low_eq: Option<&[E]>,
    fold_gadget: &[F],
    chunk_layout: &WitnessLayout,
) -> Result<Self, AkitaError>
```

For `e_hat`, decode the existing logical setup column as before, then route its
global block into one chunk:

```rust
fn get_eq_indices_for_d_chunked(
    current_index: usize,
    layout: &WitnessLayout,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_e: usize,
    block_mask: usize,
    block_bits: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    let digit_idx = current_index % num_digits;
    let block_idx = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_e;
    let chunk_idx = block_idx / layout.blocks_per_chunk;
    let block_local = block_idx % layout.blocks_per_chunk;
    let chunk = layout.chunks.get(chunk_idx).ok_or(AkitaError::InvalidSetup(...))?;
    let block_sum = (chunk.e_offset & block_mask) + block_local;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_rel_idx = digit_idx * num_claims + claim_idx + block_carry;
    Ok((chunk_idx, low_eq_idx, high_rel_idx))
}
```

Use per-chunk high tables rather than one dense table across sparse absolute
offsets:

```rust
let eq_hi_e_tables: Vec<Vec<E>> = layout.chunks.iter()
    .map(|chunk| {
        let high_base = chunk.e_offset >> block_bits;
        (0..=e_hi_len)
            .map(|k| eq_eval_at_index(high_challenges, high_base + k))
            .collect()
    })
    .collect();
```

`t_hat` follows the same pattern, with the `a_row` and `flat_t_vector` axes
preserved.

For `Z_comb`, build one output vector and add every chunk's `z_hat` equality
weight into it:

```rust
let mut z_eq_slice = vec![E::zero(); z_range];
for chunk in &chunk_layout.chunks {
    let per_chunk = build_z_eq_slice_for_offset(
        inputs,
        full_vec_randomness,
        z_block_low_eq,
        fold_gadget,
        chunk.z_offset,
    )?;
    for (dst, src) in z_eq_slice.iter_mut().zip(per_chunk) {
        *dst += src;
    }
}
```

The direct setup evaluator keeps the same scan:

```rust
packed_slice_inner_sum::<F, E, D, HAS_D, HAS_B, HAS_A>(
    segment.lo..segment.hi,
    setup_flat,
    alpha_pows,
    // same prepared weights, now chunk-aware
)
```

Tests:

- `ComponentMajor` setup weights exactly match the old `prepare` output.
- `ChunkGrouped(1)` setup weights match `ComponentMajor`.
- `ChunkGrouped(W)` setup contribution matches a materialized row-MLE for
  `W ∈ {1,2,4,8}`.
- a test or review assertion confirms `packed_slice_inner_sum` is not called in a
  per-chunk loop.

### Stage 6 — End-to-End Verifier Fixtures

After structured and setup components pass independently, add end-to-end verifier
fixtures that synthesize the resolved chunked layout. These should not require a
distributed prover yet; the fixtures can materialize the relation row according
to `WitnessLayout` and compare the verifier's deferred row evaluation against it.

Expected test shape:

```rust
for w in [1, 2, 4, 8] {
    let shape = WitnessType::ChunkGrouped(w);
    let layout = shape.resolve(&inputs)?;
    let materialized = materialize_chunked_relation_row(&fixture, &layout)?;
    let expected = dense_eq_inner_product(&materialized, &fixture.full_vec_randomness);
    let got = prepared.eval_at_point::<_, D>(...)?;
    assert_eq!(got, expected);
}
```

Run this matrix:

- pow2 `block_len` (`512`) and dense fallback `block_len` (`510`).
- with and without the D block (`MRowLayout::WithDBlock` /
  `MRowLayout::WithoutDBlock`) where fixtures exist.
- `W = 1` plus every power-of-two divisor of `num_blocks`.
- negative malformed layouts: non-power-of-two `W`, `W ∤ num_blocks`, `W >
  num_blocks`, and too-small witness capacity.

### Stage 7 — Performance Gate

Add profiling/instrumentation before claiming the implementation satisfies the
cost target. The relevant measurements are span-level, not just whole-verifier
time.

Procedure:

1. Run the profile example under `ComponentMajor`.
2. Run the same shape with synthetic `ChunkGrouped(W)` for `W = 2, 4, 8`.
3. Compare `setup_contribution` span time and α-evaluation counts, if a counter
   is added in test/profile builds.
4. Separately record `z_structured` time; it is allowed to scale with `W`.

Expected result:

```text
setup_contribution(W=1) ≈ setup_contribution(W=2) ≈ setup_contribution(W=4)
z_structured(W)         grows roughly linearly with W
```

If `setup_contribution` grows linearly with `W`, stop and inspect whether the
packed setup scan was accidentally moved inside a chunk loop.

### Stage 8 — Production Enablement

Once verifier-only fixtures pass, connect the public layout selection to the
distributed prover/schedule side. This is the point where
`WitnessType::ChunkGrouped` becomes more than a synthetic verifier fixture.

Implementation checklist:

- schedule/config code sets `LevelParams::witness_shape` for distributed runs.
- generated table identity or runtime schedule descriptor includes
  `witness_shape`.
- prover output witness layout matches the exact `[e|t|z]... [r]` geometry used
  by `WitnessType::resolve`.
- verifier integration tests prove a distributed prover output with the chunked
  public layout and reject the same proof under `ComponentMajor`.

Follow-ups after the first full landing:

- tensor/factored `c_alpha` chunk windowing.
- non-power-of-two `B_w` dense per-chunk fallback.
- ZK blinding under chunking.
- chunked tiered commitments, if Stage 0 deferred them.

## References

- Single-segment row-eval theory:
  [`book/src/how/verifying/matrix_evaluation.md`](../book/src/how/verifying/matrix_evaluation.md)
- Distributed-relation verifier theory:
  [`book/src/how/verifying/distributed-relation-verifier.md`](../book/src/how/verifying/distributed-relation-verifier.md)
- Distributed prover:
  [`book/src/how/proving/distributed-prover.md`](../book/src/how/proving/distributed-prover.md)
- Code: `crates/akita-verifier/src/protocol/ring_switch.rs` (`eval_at_point`),
  `crates/akita-verifier/src/protocol/slice_mle/` (structured + setup
  evaluators), `crates/akita-types/src/setup_contribution.rs`
  (`SetupContributionPlan`), `crates/akita-types/src/proof/ring_relation.rs`
  (`RingRelationSegmentLayout`).
