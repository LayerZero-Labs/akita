# Spec: Witness-shape-generic verifier row-MLE evaluation


| Field         | Value                              |
| ------------- | ---------------------------------- |
| Author(s)     | Omid; Freya                               |
| Created       | 2026-06-19                         |
| Status        | superseded                         |
| PR            |                                    |
| Supersedes    |                                    |
| Superseded-by | `machine-major-distributed-prover.md` |
| Book-chapter  | how/verifying/matrix_evaluation.md |


> **Superseded layout.** The chunk-list evaluator in this document describes
> the first single-host multi-chunk relation. It does not define recursive
> machine ownership. The native hierarchical verifier target is specified in
> [`machine-major-distributed-prover.md`](machine-major-distributed-prover.md):
> machine-major globally, block-fast locally, with a local quotient segment in
> every machine witness.

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
it introduces a `ChunkedWitnessCfg` parameter in `LevelParams` describing the two
column layouts, factors both into one **chunk-list** common denominator, and
refactors the evaluation so a single code path serves both — the established
layout becoming the one-chunk special case. The implementation surface is
`RelationMatrixEvaluator::eval_at_point`
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

Make the verifier row-MLE evaluation generic over `num_chunks` by resolving
every shape to one `WitnessLayout` that the per-component evaluators fold over,
where the established single-segment layout is the one-chunk (`num_chunks = 1`)
case and is byte-identical to today.

Key abstractions and surfaces introduced or modified (all layout types live in
`crates/akita-types/src/witness/`, so both the verifier and the
setup-contribution planner consume the same definitions):

- **`ChunkedWitnessCfg` in `LevelParams`** (new struct, replaces `witness_shape: WitnessType`):
  ```rust
  /// Chunk-based witness layout parameters.
  /// `num_chunks = 1` is the single-chunk (standard) case.
  /// `num_activated_levels` controls for how many protocol levels the
  /// multi-chunk layout is active; ignored when `num_chunks = 1`.
  /// `num_chunks` must be a power of two.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub struct ChunkedWitnessCfg {
      pub num_chunks: usize,
      pub num_activated_levels: usize,
  }

  impl Default for ChunkedWitnessCfg {
      fn default() -> Self {
          Self { num_chunks: 1, num_activated_levels: 0 }
      }
  }
  ```
  (`Hash` is required: the planner embeds `ChunkedWitnessCfg` in the generated-table
  catalog identity.) `num_chunks = 1` is the single-chunk (previously `ComponentMajor`) case.
  `num_chunks = W` is the multi-chunk (previously `ChunkGrouped(W)`) case.
- **`WitnessChunkLengths` / `WitnessChunkLayout` / `WitnessLayout`** (new): the
  resolved, layout-agnostic description the evaluators consume. Lengths and
  offsets are kept in separate structs, mirroring the existing
  `RingRelationSegmentLengths` / `RingRelationSegmentLayout` split. Segments
  that are absent use `None` rather than `0` so call sites cannot silently
  treat an absent offset as a valid position.
  ```rust
  /// Per-chunk segment sizes (emission order `z ‖ e ‖ t ‖ r`).
  /// `r_len` is `None` for non-last chunks.
  pub struct WitnessChunkLengths {
      pub z_len: usize,            // replicated: same in every chunk
      pub e_len: usize,            // partitioned: total_e_len / num_chunks
      pub t_len: usize,            // partitioned: total_t_len / num_chunks
      pub r_len: Option<usize>,    // Some only in last chunk
  }

  /// Per-chunk segment offsets.
  /// `offset_r` mirrors `r_len`: `None` when absent.
  pub struct WitnessChunkLayout {
      pub offset_z: usize,
      pub offset_e: usize,
      pub offset_t: usize,
      pub offset_r: Option<usize>,
      pub global_block_base: usize,  // chunk_idx * blocks_per_chunk
  }

  /// Full witness column layout for num_chunks chunks.
  /// `chunks` and `chunk_lengths` are parallel Vecs of length `num_chunks`.
  pub struct WitnessLayout {
      pub blocks_per_chunk: usize,
      pub chunks: Vec<WitnessChunkLayout>,         // offsets; len == num_chunks
      pub chunk_lengths: Vec<WitnessChunkLengths>, // lengths; parallel to chunks
  }
  ```
  
- **`RingRelationInstance::segment_layout(lp) -> WitnessLayout`** (replaces
  returning `RingRelationSegmentLayout`): the **only** place the two shapes
  diverge. `lp.witness_chunk.num_chunks` drives how many chunks are produced.
  `num_chunks = 1` resolves to a one-element chunk list whose offsets come
  from the existing `RingRelationSegmentLayout` computation (blinding offsets
  preserved; z-first ordering unconditional) with `blocks_per_chunk = num_blocks`
  and `global_block_base = 0`. `num_chunks = W` computes per-chunk offsets
  arithmetically with fixed `[z|e|t]` order per chunk: for chunk `j`,
  `offset_z = j·stride`, `offset_e = offset_z + z_len_j`,
  `offset_t = offset_e + e_len_j`. Only the last chunk carries `r` (`Some`);
  all others carry `None`. All validation (no-panic) happens here.
  here. `RingRelationSegmentLayout` is deprecated and replaced by `WitnessLayout`.
- **`RelationMatrixEvaluator`** carries the resolved `WitnessLayout` (replacing
  `witness_segment_layout`), and `eval_at_point` becomes a fold over
  `chunk_layout.chunks()` zipped with `chunk_layout.chunk_lengths`.
- **`PreparedChallengeEvals::summarize_chunk_block_carries`** (new, generalizes
  `summarize_all_block_carries` in
  `crates/akita-verifier/src/protocol/ring_switch/tensor_challenges.rs`): the
  per-claim two-bucket `c_alpha` block summaries for one chunk's block window
  `[global_block_base, global_block_base + B_w)`, peeling `B_w` instead of `B`.
- **`SetupContributionPlanInputs`** / **`SetupContributionPlan::prepare`**: take
  the `WitnessLayout` so the precomputed column-weight vectors (`e_eq_slice`,
  `t_eq_slice_per_group`, `z_eq_slice`) are built against the resolved chunks.
  The hot α-evaluation loop `packed_slice_inner_sum` is **unchanged**.
- **`LevelParams`** gains a public `witness_chunk: ChunkedWitnessCfg` (default
  `ChunkedWitnessCfg::default()`) — the verifier's source of truth for the layout.
  It is a public schedule/layout parameter, not a prover-controlled proof byte.

### Invariants

- **`num_chunks = 1` is byte-identical to today.** Evaluating under
  `num_chunks = 1` returns exactly the same field element as the current
  `eval_at_point` for every fixture. Protected by a new regression test
  `single_chunk_matches_legacy_row_eval`.
- **`num_chunks = 1` collapses to the single-chunk layout.** `segment_layout`
  with `num_chunks = 1` yields a one-chunk `WitnessLayout` whose offsets and
  evaluation are identical to the old `ComponentMajor` / `RingRelationSegmentLayout`
  values: with `W=1`, `stride = z_len + e_len + t_len`, so `offset_z=0`,
  `offset_e=z_len`, `offset_t=z_len+e_len`. There is no ordering caveat;
  all shapes use the same z-first convention. Protected by
  `chunk_grouped_one_equals_single_chunk`. This is the structural proof that
  the chunk fold is a true generalization, not a parallel path.
- **Structured correctness for every power-of-two `num_chunks`.** For every
  `W ∈ {1, 2, 4, …, num_blocks}` the chunked structured evaluation equals the
  brute-force materialized `M` over the resolved chunk layout (with `z_hat`
  replicated `W` times). Protected by `chunk_grouped_matches_materialized` (see
  Testing Strategy), run for both power-of-two and non-power-of-two `block_len`.
- **Dominant-cost equivalence (α-evaluations).** The shared SIS-matrix
α-evaluation scan (`eval_ring_at_pows` inside `packed_slice_inner_sum`) runs
exactly **once per shared-matrix entry**, with the same
`r_max = max(n_d, n_b, n_a)` and the same `n_cols` regardless of chunk count.
The chunk-replicated `A·G_fold·z_hat` setup term enters only through the
additively combined, α-free `Z_comb` weight vector. The implementation must
**not** re-scan or re-α-evaluate the setup matrix per chunk.
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
  `segment_layout` (from public layout) before any proof data is read, per
  the [verifier no-panic contract](../book/src/how/verification.md). Malformed
  shape / chunk count / capacity is rejected with `AkitaError`, never a panic.
- **Transcript unchanged in shape, additive in rounds only.** Because `z_hat` is
  chunk-replicated, the next-level witness variable count rises by at most
  `log₂ W`, so stage-2 / setup-product sum-checks read up to `log₂ W` extra round
  polynomials. No new challenge *kinds*; challenge sampling order is unchanged.

### Non-Goals

- **No prover or planner changes (this spec).** Verifier-only. The planner side
  is **implemented** ([`specs/distributed-planner.md`](distributed-planner.md)):
  it stamps `LevelParams.witness_chunk` per fold level and prices the chunked
  schedule, so the verifier reads the per-level `num_chunks` the planner set and
  does not compute the cutover itself. The chunked witness/relation construction —
  a **single** prover that emits the same proof a future distributed prover will,
  by building the multi-`z` chunked relation — is specified separately in
  [`specs/distributed-prover.md`](distributed-prover.md). This spec's tests
  synthesize the resolved layout directly; once that prover lands, an end-to-end
  prove→verify roundtrip augments (does not replace) the synthesized fixtures —
  see Stage 6.
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

- [ ] `ChunkedWitnessCfg`, `WitnessChunkLengths`, `WitnessChunkLayout`, `WitnessLayout`
  exist in `crates/akita-types/src/witness/`, and
  `RingRelationInstance::segment_layout(lp)` yields `1` chunk for
  `num_chunks = 1` and `W` chunks for `num_chunks = W`.
- [ ] `RelationMatrixEvaluator::eval_at_point` evaluates `e_hat`, `t_hat`,
  `z_hat`, and the setup contribution as a fold over `chunk_layout.chunks()`,
  reusing the existing `EStructuredSlicesEvaluator` /
  `TStructuredSlicesEvaluator` / `ZStructuredPow2SlicesEvaluator` /
  `ZDenseSlicesEvaluator` unchanged in body; the `r` contribution is gated by
  `lens.r_len.is_some()`.
- [ ] `SetupContributionPlan::prepare` builds `e_eq_slice`,
  `t_eq_slice_per_group`, and `Z_comb` (`z_eq_slice`) against the `WitnessLayout`;
  `packed_slice_inner_sum` is unchanged.
- [ ] `single_chunk_matches_legacy_row_eval` passes (byte-identical to legacy).
- [ ] `chunk_grouped_one_equals_single_chunk` passes.
- [ ] `chunk_grouped_matches_materialized` passes for
  `W ∈ {1, 2, 4, 8, …, num_blocks}` with `block_len` a power of two (Case 1).
- [ ] `chunk_grouped_matches_materialized_z_dense` passes for the same `W` with
  `block_len` **not** a power of two (Case 2, dense fallback), mirroring the
  existing single-segment `z_dense_matches_materialized_range_inner_product`
  (`block_len = 510`).
- [ ] A negative test rejects malformed shape (`num_chunks` not a power of two,
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
   - Resolve the layout via `relation.segment_layout(&lp_with_num_chunks(W))`.
   - **Materialize** the full `M` row contribution as a flat vector over
     `[ z0|e0|t0 ]…[ z(W-1)|e(W-1)|t(W-1) ][ r ]` (z-first within each
     chunk, `r` only in last chunk as `Some`), using the per-cell formulas from
     the theory chapter (`e_hat`: `c_alpha[claim][blk_g]·g_open`; `t_hat`: with
     the `a_row` axis; opening-row `z_hat`: `g_commit·g_fold·a[blk]`,
     **the same in every chunk** — replicated; `c_alpha` read at the *global* block
     `blk_g = chunk·B_w + block_local` for `e_hat`/`t_hat`).
   - Form `eq(·, r_col)` densely over that layout, inner-product against the row
     weights, and compare to the chunked structured evaluation. The loop over
     `W` proves "any power-of-two chunk count."
2. **`chunk_grouped_matches_materialized_z_dense` (non-pow2 `block_len`).** Same
   structure as (1) with a fixture whose `block_len` is **not** a power of two
   (e.g. `block_len = 510`). Drives the §4 Case 2 / §5 dense `Z_comb` paths
   (`ZDenseSlicesEvaluator` summed over chunks, dense `z_eq_slice` summed over
   chunks). Loop `W ∈ {1, 2, 4, 8}`. The materialized opening-row `z_hat`
   contribution is **replicated**. It uses the same `g_commit·g_fold·a[blk]`
   in every chunk, and only the offset shifts.
3. **`single_chunk_matches_legacy_row_eval` (regression).** Evaluate under
   `num_chunks = 1` and assert equality with the current single-segment result on
   the existing fixture (both `block_len` pow2 and non-pow2).
4. **`chunk_grouped_one_equals_single_chunk`.** Assert `segment_layout` with
   `num_chunks = 1` evaluates to the same value as the legacy single-segment
   path (both use the same `[z|e|t]` ordering since PR #216), confirming the
   one-chunk collapse.
5. **Per-component equivalence.** Optionally split (1)/(2) into `e_hat`-only,
   `t_hat`-only, `z_hat`-only, and setup-only sub-tests for sharper failure
   localization.
6. **No-panic negatives.** `chunk_grouped_rejects_bad_shape`: assert `AkitaError`
   (not panic) for `num_chunks = 3` (not pow2), `num_chunks = 16`
   (`> num_blocks`), and a chunk count whose `W · z_len` exceeds the validated
   `w_len`.
7. **End-to-end.** Existing `crates/akita-pcs` / verifier integration tests must
   continue passing unchanged (they exercise `num_chunks = 1`).

### Performance

- **Dominant term unchanged.** The setup-matrix α-evaluation scan
  (`O(r_max · n_cols · D)` ring ops) must be identical to `num_chunks = 1` for
  any chunk count; verify via the profile harness
  (`AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`)
  comparing a `num_chunks = 1` run with a synthetic `num_chunks = W` run on the
  `setup_contribution` span — the span time must be flat in `W`.
- **`e_hat`/`t_hat`/`r` flat in chunk count.** Block-summary build is `O(C·B)`
  total regardless of `W`; the only `W`-scaled additions are the cheap digit-axis
  outer combines (`O(W·C·do)`, `O(W·n_A·C·do)`) and the high tables (`O(W·…)`),
  far below the α-eval floor.
- **`z_hat` and `Z_comb` scale with chunk count** by design — acceptable because
  the `z_hat`-tensor is the cheap (α-free) part of the verifier.
- **Net expectation:** total verifier time `≈ num_chunks=1 baseline + O(W·block_len) + O(W·A_cols)`, negligible whenever `W ≪ r_max·D`.

## Detailed Review

### Architecture Review

The proposed architecture is directionally right: it normalizes the public witness
layout once into `WitnessLayout`, then keeps the verifier hot path expressed as a
fold over resolved chunks. That is the correct boundary because today's code has
three independent consumers of witness column geometry:

- `RelationMatrixEvaluator::eval_at_point`, which owns the structured
`e_hat`/`t_hat`/`z_hat` and `r` contributions.
- `SetupContributionPlan::prepare`, which translates the same geometry into
column-equality weights for the packed setup scan.
- `PreparedChallengeEvals::summarize_all_block_carries`, which binds the
challenge vector's global block axis to the verifier's low-bit peeled window.

Factoring the layout into `akita-types` is also the right crate boundary. The
layout is not verifier-local: both the direct verifier setup scan and the
setup-product/bar-setup_index_weight path need the same physical column mapping. Keeping one
definition avoids the most dangerous failure mode: the structured witness
contribution and the setup contribution silently evaluating different column
layouts.

The main architecture requirement is that `segment_layout` must become a real
verifier boundary, not just a convenience constructor. It should receive, or be
followed immediately by validation against, the validated witness column capacity
(`w_len / D`). Without that, the spec's no-panic invariant cannot be fully
enforced at the shape boundary; `eval_at_point` would still be able to construct
offsets that are algebraically valid but outside the committed witness column
domain. In implementation terms, either make `witness_len` part of
`segment_layout`'s inputs, or add a non-optional
`WitnessLayout::validate_capacity(witness_len, r_tail_len)` call in
`prepare_relation_matrix_evaluator` before the layout is stored.

`witness_chunk` must also be transcript/descriptor-bound as a public layout
parameter. The proof should not choose it, but the verifier and prover must agree
on it through the same public schedule/layout descriptor that already binds the
level parameters. Adding `LevelParams::witness_chunk` without including it in the
canonical descriptor/serialization path would create a configuration mismatch
risk: two executions could absorb the same commitments and proof bytes while
interpreting the relation matrix columns differently.

One layout note for maintainers: tiered commitment and the `û_concat` witness
segment were removed in #257. The witness layout is always `z ‖ e ‖ t ‖ r`;
there is no `u` segment and no `tier_split` planner field.

The `num_chunks = 1` compatibility story is good but should be tested at two
levels. A direct result comparison protects behavior, while a resolved-layout
snapshot test protects the intended geometry (`offset_e`, `offset_t`, `offset_z`,
`offset_r`). The latter is useful because a future refactor can preserve one
fixture's final scalar while still moving offsets in a way that breaks setup
contribution or other follow-ups.

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
include large gaps introduced by the per-chunk `[z|e|t]` stride, especially
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
chunk-multi-group layout by "branching on the shape" inside each of these — and
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

- `num_chunks = 1` is the **degenerate one-chunk** case: a single chunk whose
  `e_hat` spans all `num_blocks`, whose `z_hat` is the lone fold, and whose
  offsets are exactly today's `RingRelationSegmentLayout`.
- `num_chunks = W` is the **W-chunk** case: each chunk's `e_hat`/`t_hat`
  spans `B_w = num_blocks / W` blocks, and each chunk carries a full-`block_len`
  `z_hat`.

That additivity is exactly the verifier requirement stated up front: the
established single-segment layout is just the one-chunk case. The single-segment
verifier *is* the chunk fold with one iteration.

#### The common denominator: `WitnessLayout`

The design pivots on a single resolution step:

```text
witness_chunk  ──segment_layout(lp)──▶  WitnessLayout { blocks_per_chunk, chunks[], chunk_lengths[] }
```

`WitnessLayout` is the *only* thing the evaluators see. They never branch on
`num_chunks`; they fold over `chunk_layout.chunks()` zipped with
`chunk_layout.chunk_lengths`. All layout divergence — blinding offsets, the
chunk-offset formula, `Option` presence of `u`/`r` — is confined to
`segment_layout`, which is also where every no-panic check lives (chunk count is
a power of two, divides `num_blocks`, `B_w` is a clean window, `W·L + |r|` fits
the validated `w_len`). The hot path inherits the guarantee that all bounds were
checked once, at the edge.

This placement also fixes the crate boundary cleanly: `ChunkedWitnessCfg` /
`WitnessChunkLayout` / `WitnessLayout` live in `crates/akita-types/src/witness/`,
so the verifier's `ring_switch.rs` and `SetupContributionPlan` consume one
definition — no duplicated layout knowledge across the verifier/types boundary.

#### Tile vs. replicate falls out of the data, not control flow

The subtle part of the two layouts is that components behave differently under
chunking: `e_hat`/`t_hat` are **partitioned** (the union of chunks' pieces is the
whole component, each chunk covering a disjoint block sub-range), while `z_hat`
is **replicated** (each chunk carries a *full* fold). A clean architecture must
express this without a per-component "am I partitioned?" switch.

The chunk representation does this implicitly:

- A chunk's `e_hat`/`t_hat` pieces are addressed by `(global_block_base, blocks_per_chunk)`. Because the chunks' windows are disjoint and cover
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

1. **Resolve (layout, shape-aware).** `segment_layout(lp) -> WitnessLayout`.
   All validation. `num_chunks = 1` reads `RingRelationSegmentLayout`
   (one chunk, `B_w = num_blocks`, `global_block_base = 0`). `num_chunks = W`
   computes per-chunk offsets with z-first ordering per chunk, from
   `L = |z^j| + |e^j| + |t^j|` (stride):
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

`num_chunks = 1` ⇒ `global_block_base = 0`, `B_w = num_blocks` ⇒ today's
`summarize_all_block_carries`. The `t_hat` evaluator reuses the chunk's `e_hat`
summaries (same in-window residue, since `|e^j|` is a multiple of `B_w`). `eq_low`
(window `B_w`) and `high_challenges` are built once and shared across chunks.

#### `z_hat` per chunk (replicated)

The in-block weight `a[blk]` is chunk-independent (the fold rows carry no
chunk-specific data — `a[blk]` and the gadget weights are global), so only the
offset differs. The verifier dispatches on `block_len.is_power_of_two()` exactly
as today; the dispatch depends on `block_len`, not on the chunk, so the chunk
loop sits **outside** the case split. Both cases combine additively:
`z_contribution = Σ_chunk Z_eval(chunk.offset_z)`.

> **New evaluator input: nonzero `offset_z`.** Single-chunk always places `z`
> first (`offset_z = 0`), so today's `ZStructuredPow2SlicesEvaluator` /
> `ZDenseSlicesEvaluator` are only ever called at `offset_z = 0`. Under chunking,
> chunk `j>0` has `offset_z = j·stride` (and `stride` is **not** a multiple of
> `block_len`, since it includes `e`/`t` lengths), so `z_lo = offset_z mod
> block_len ≠ 0` is exercised for the first time. The "body unchanged" claim
> therefore requires confirming the pow2 evaluator already handles a nonzero
> in-block shift and high-index base from its `offset_z` input; a dedicated
> `z_only` test at `W ∈ {2,4,8}` (Stage 4) must cover this rather than relying on
> the aggregate materialized test. (The dense evaluator already takes an explicit
> `offset_z`.)

- **Case 1 — `block_len` a power of two (root).** Peel the `block_len` window per
chunk; build the two-bucket in-block summary with `z_lo = z_offset mod block_len` (the `eq_low_z` table is built once and shared — it depends only on
`r_col`'s low bits and the window size, never on the offset), evaluate with
`ZStructuredPow2SlicesEvaluator { offset_z: chunk.offset_z }`, sum. Overhead
`O(W·block_len + W·DF·DC)`.
- **Case 2 — `block_len` not a power of two (recursive, dense).** No clean
low-bit window to peel (`block_len = ceil(num_ring / num_blocks)` need not be a
power of two at recursive levels). Fall back to materializing the structured
`z` segment and running one generic offset-eq evaluation — per chunk, i.e.
today's `ZDenseSlicesEvaluator` repeated `W` times, each at its own
`chunk.offset_z`:
  ```rust
  z_contribution += ZDenseSlicesEvaluator {
      g1_commit, fold_gadget,
      consistency_weight: self.eq_tau1[0],
      a_evals_by_point,                 // shared across chunks (a[blk] is global)
      full_vec_randomness: x_challenges,
      offset_z: chunk.offset_z,         // the only per-chunk input
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
precomputed column-weight vectors are built against the chunks; the hot loops
for direct setup scans and setup-index weight MLE evaluation share the same
packed segment partition. This is what keeps the α-evaluation count
layout-independent.

- `**W_col` / `e_eq_slice` (`D·e_hat`, partitioned).** `get_eq_indices_for_d`
gains a chunk split: decode the SIS column to `(dig, blk_g, claim)` as today,
then map the global block to its chunk:
  ```text
  chunk_idx    = blk_g / B_w
  block_local  = blk_g % B_w
  block_sum    = (chunks[chunk_idx].offset_e mod B_w) + block_local
  low_eq_idx   = block_sum & (B_w - 1)
  block_carry  = block_sum >> log₂(B_w)
  high_eq_idx  = (chunks[chunk_idx].offset_e >> log₂(B_w))
                 + (dig*num_claims + claim) + block_carry
  W_col[c]     = eq_low[low_eq_idx] * eq_high_e[high_eq_idx]
  ```
  `eq_low` is the shared `B_w`-window table; `eq_high_e` is the existing high
  table indexed at the per-chunk offset base (the "chunk axis" of size `W·C·do`
  is just `W` offset bases into one high table). `num_chunks = 1` ⇒ one chunk,
  `chunk_idx = 0`, `block_local = blk_g` ⇒ today's `get_eq_indices_for_d`.
  **Footprint unchanged**: each SIS column maps to exactly one chunk's `e_hat`
  piece (partition), so `d_required`/`n_cols_e` are the same.
- `**T_col` / `t_eq_slice_per_group` (`B·t_hat`, partitioned).** Same chunk split
in `get_eq_indices_for_b`, with the extra `a_row` axis and per-group sparsity;
footprint unchanged.
- `**Z_comb` / `z_eq_slice` (`A·G_fold·z_hat`, replicated).** For each `A`
column `c = (blk, dc)`, sum the per-chunk `G_fold` weighted `z_hat` equality
weight over all chunks:
  ```text
  Z_comb[blk, dc]
    = -Σ_chunk Σ_df G_fold[df]
       · eq_x(chunk.offset_z
              + blk
              + block_len · (df + depth_fold · dc))
  ```
  There is no `G_commit` factor in this setup weight. `G_commit` appears in the
  separate opening-row contribution for the `z_hat` segment.
  Following the **same two `block_len` cases as §`z_hat` per chunk**, `prepare`
  dispatches once on `block_len.is_power_of_two()` and loops the chunk axis
  inside:
  - `**block_len` pow2:** reuse the per-chunk peeled in-block weights — the
  `s_per_dc_per_carry` table is rebuilt per chunk (its high offset
  `z_offset >> log₂(block_len)` differs), the shared `z_block_low_eq` window
  table is built once. Build cost `O(W·A_cols)`.
  - `**block_len` not pow2 (dense):** build the dense `z_eq_slice` per chunk via
  the existing one-shot peeled-equality cache (today's non-pow2 branch),
  summing into `Z_comb`. Build cost `O(W·A_cols)`.
  Either way the output length is `z_range = inner_width` (**unchanged**), so the
  downstream scan and its α-evaluation count are identical to `num_chunks = 1`;
  only the precomputed `Z_comb` weights are summed over chunks. One chunk ⇒
  today's `z_eq_slice`.

Because `A`/`B`/`D` are the same seed-expanded matrix for every chunk, the scan
range (`r_max`, `n_cols`) and α-eval count are layout-independent. `Z_comb` is the
*only* place the chunk count enters the setup contribution, and it is α-free.

#### `r`-tail and tail segments

`compute_r_contribution` is **unchanged**: a single summed quotient `r` tails the
whole witness. Its offset comes from
`chunk_layout.chunks.last().unwrap().offset_r.expect("last chunk always has r")`.
The evaluator body and cost are identical.

### Alternatives Considered

- **Branch on `num_chunks` inside each component evaluator.** Rejected: it
  duplicates the peeled-block arithmetic and the no-panic checks across every
  component and the setup planner, and it puts a layout `if` inside the hot loop.
  The chunk-list resolution keeps the case-split at one edge (`segment_layout`) and
  the hot path uniform.
- **Per-component "segment iterator" trait (no chunk object).** Each component
  exposes its own `Iterator<Item = (offset, window, block_base)>`, decoupling
  components from the chunk concept. Rejected as weaker: it hides the fact that
  `e`/`t`/`z` of one chunk share a `global_block_base` and offset region, makes
  the `t_hat`-reuses-`e_hat`-summaries sharing awkward to express, and offers no
  benefit since both layouts are genuinely chunk-structured.
- **General `ColumnAddressing` strategy object** mapping logical `(component, block, claim, dig)` to physical columns. The most general (could absorb
  blinding, arbitrary future layouts) but heavyweight: an indirection
  on the hot index path and far more surface than two layouts justify. The
  `WitnessLayout` is the minimal common denominator for the layouts that actually
  exist; revisit the strategy object only if a third, non-chunk-structured layout
  appears.
- **Retain a `WitnessType` enum alongside `ChunkedWitnessCfg`.** Rejected: it
  introduces redundancy — `num_chunks = 1` and `num_chunks = W` are already
  unambiguous via the `ChunkedWitnessCfg` struct, and a separate enum would need its
  own default, serialization, and consistency checks. The `witness_chunk` field is
  the single source of truth; the equivalence between `num_chunks = 1` and the old
  single-chunk layout is captured as the `chunk_grouped_one_equals_single_chunk`
  test invariant.
- **Re-scan the SIS matrix per chunk.** Rejected: it multiplies the dominant
α-evaluation cost by the chunk count, violating the dominant-cost invariant.
The `Z_comb` pre-combine keeps the scan single-pass.

## Documentation

- The chunked-layout theory (component-by-component cost, partitioned vs
replicated, why the α-scan is unchanged) lives in the book at
`[book/src/how/verifying/distributed-relation-verifier.md](../book/src/how/verifying/distributed-relation-verifier.md)`;
the single-segment row-eval is
`[book/src/how/verifying/matrix_evaluation.md](../book/src/how/verifying/matrix_evaluation.md)`;
the prover theory is
`[book/src/how/proving/distributed-prover.md](../book/src/how/proving/distributed-prover.md)`;
the planner implementation is
[`specs/distributed-planner.md`](distributed-planner.md); and the prover
implementation is [`specs/distributed-prover.md`](distributed-prover.md).
This spec is the implementation record; on land set `Status: implemented`, fill
`PR:`, reference the new symbols (`ChunkedWitnessCfg`, `WitnessChunkLayout`,
`WitnessLayout`, `summarize_chunk_block_carries`). When stable, fold the
chunk abstraction into `matrix_evaluation.md` and archive per
`[specs/PRUNING.md](PRUNING.md)`.
- Update the verifier no-panic audit
(`[docs/verifier-panic-audit.md](../docs/verifier-panic-audit.md)`) with the
`segment_layout` boundary checks (shape / chunk-count / capacity rows).
- No `AGENTS.md` crate-graph change (no new crate; new types in `akita-types`).

## Implementation Stages

The implementation should land in small, reviewable stages. Each stage below has
a clear invariant, the code shape expected at the end of the stage, and the tests
that prove it before moving on.

### Stage 0 — Scope and Descriptor Boundary

Before changing the hot path, lock down the public-boundary decision:

`witness_chunk` is a public schedule/layout parameter and is included anywhere
`LevelParams` participates in canonical descriptor bytes, schedule snapshots,
or generated table identity.

Tiered commitment was removed in #257; there is no `û_concat` segment and no
`tier_split` guard for multi-chunk layouts.

Expected code shape:

```rust
// In crates/akita-types/src/witness/mod.rs
/// Chunk-based witness layout parameters.
/// `num_chunks = 1` is the single-chunk (standard) case.
/// `num_activated_levels` controls for how many protocol levels the
/// multi-chunk layout is active; ignored when `num_chunks = 1`.
/// `num_chunks` must be a power of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkedWitnessCfg {
    pub num_chunks: usize,
    pub num_activated_levels: usize,
}

impl Default for ChunkedWitnessCfg {
    fn default() -> Self {
        Self { num_chunks: 1, num_activated_levels: 0 }
    }
}

// Inside LevelParams.
pub witness_chunk: ChunkedWitnessCfg,  // default: ChunkedWitnessCfg::default()
```

Tests:

- A descriptor/serialization snapshot changes when `witness_chunk` changes.
- Existing schedules deserialize/build with `ChunkedWitnessCfg::default()`.

### Stage 1 — Layout Types and Capacity-Checked Resolution

Add the resolved layout types in `crates/akita-types/src/witness/`. The
important implementation detail is that resolution validates both arithmetic
and witness capacity; it should not leave capacity checks to later indexing code.

Expected code shape:

```rust
// crates/akita-types/src/witness/mod.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLengths {
    pub z_len: usize,              // replicated: same in every chunk
    pub e_len: usize,              // partitioned: total_e_len / num_chunks
    pub t_len: usize,              // partitioned: total_t_len / num_chunks
    pub r_len: Option<usize>,      // Some only in last chunk
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLayout {
    pub offset_z: usize,
    pub offset_e: usize,
    pub offset_t: usize,
    pub offset_r: Option<usize>,           // None if r absent
    pub global_block_base: usize,          // chunk_idx * blocks_per_chunk
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkLayout>,         // offsets; len == num_chunks
    pub chunk_lengths: Vec<WitnessChunkLengths>, // lengths; parallel to chunks
}

pub struct SegmentLayoutInputs {
    pub num_claims: usize,
    pub num_t_vectors: usize,
    pub relation_matrix_row_layout: RelationMatrixRowLayout,
    pub witness_ring_len: usize,
}
```

`num_chunks = 1` delegates to the existing layout and becomes a one-chunk
adapter:

```rust
let lens = ring_relation_segment_lengths(lp, inputs)?;  // includes r_len
let r_offset = lens.z_len + lens.e_len + lens.t_len;
let layout = WitnessChunkLayout {
    offset_z: 0,
    offset_e: lens.z_len,
    offset_t: lens.z_len + lens.e_len,
    offset_r: Some(r_offset),
    global_block_base: 0,
};
let lengths = WitnessChunkLengths {
    z_len: lens.z_len,
    e_len: lens.e_len,
    t_len: lens.t_len,
    r_len: Some(lens.r_len),
};
WitnessLayout {
    blocks_per_chunk: lp.num_blocks,
    chunks: vec![layout],
    chunk_lengths: vec![lengths],
}
```

`num_chunks = W` computes z-first `[z^j|e^j|t^j]` with checked arithmetic;
`r` only on the last chunk as `Some`:

```rust
let w = lp.witness_chunk.num_chunks;
let blocks_per_chunk = lp.num_blocks.checked_div(w).ok_or(AkitaError::InvalidSetup(...))?;
let z_len_j = depth_fold * depth_commit * lp.block_len;
let e_len_j = depth_open * inputs.num_claims * blocks_per_chunk;
let t_len_j = depth_open * lp.a_key.row_len() * inputs.num_t_vectors * blocks_per_chunk;
let chunk_stride = z_len_j + e_len_j + t_len_j;

let r_len_total = inputs.num_rows * r_decomp_levels::<F>(lp.log_basis);

let (chunks, chunk_lengths): (Vec<_>, Vec<_>) = (0..w)
    .map(|j| {
        let is_last = j == w - 1;
        let base = j * chunk_stride;
        let layout = WitnessChunkLayout {
            offset_z: base,
            offset_e: base + z_len_j,
            offset_t: base + z_len_j + e_len_j,
            offset_r: if is_last { Some(w * chunk_stride) } else { None },
            global_block_base: j * blocks_per_chunk,
        };
        let lengths = WitnessChunkLengths {
            z_len: z_len_j,
            e_len: e_len_j,
            t_len: t_len_j,
            r_len: if is_last { Some(r_len_total) } else { None },
        };
        (layout, lengths)
    })
    .unzip();
WitnessLayout { blocks_per_chunk, chunks, chunk_lengths }
```

`RingRelationSegmentLengths` gains `r_len: usize`; `ring_relation_segment_lengths`
gains a `num_rows: usize` parameter. In `segment_layout`, pass `self.y.len()` as
`num_rows`. `r_len = num_rows * r_decomp_levels::<F>(log_basis)`.

Validation checklist inside `segment_layout`:

- `W > 0`, `W.is_power_of_two()`, and `W <= num_blocks`.
- `num_blocks % W == 0`.
- `blocks_per_chunk.is_power_of_two()` for the peeled fast path.
- every offset/length uses checked arithmetic.
- `global_block_base + blocks_per_chunk <= num_blocks`.
- `chunks.last().offset_r + r_len_total <= witness_ring_len` (capacity bound).

Tests:

- `num_chunks_one_resolves_to_legacy_segment_layout`.
- `num_chunks_one_resolves_to_single_chunk`.
- `chunk_layout_offsets_are_contiguous_and_cover_blocks`.
- bad `W`, overflow-shaped inputs, and too-short witness capacity return
  `AkitaError`.

### Stage 2 — Prepare-Time Wiring

Change `RelationMatrixEvaluator` to store `WitnessLayout` instead of
`RingRelationSegmentLayout`. Do this before changing evaluation logic so the
compiler reveals every caller that still expects single offsets.

Expected code shape:

```rust
pub struct RelationMatrixEvaluator<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    // ...
    pub(crate) chunk_layout: WitnessLayout,
}

pub(crate) fn chunk_layout(&self) -> &WitnessLayout {
    &self.chunk_layout
}
```

`prepare_relation_matrix_evaluator` should resolve the layout after existing shape
checks and before returning the prepared evaluator:

```rust
let witness_ring_len = w_len / D;
let chunk_layout = relation.segment_layout(&lp, &SegmentLayoutInputs {
    num_claims,
    num_t_vectors,
    relation_matrix_row_layout,
    witness_ring_len,
})?;
```

If `prepare_relation_matrix_evaluator` does not currently receive `w_len`, thread it
through from `ring_switch_verifier_core`; the verifier already validates `w_len`
and computes `num_ring_elems`, so this is the right boundary.

Tests:

- Existing verifier tests still pass under `ChunkedWitnessCfg::default()`.
- A focused prepare test asserts `RelationMatrixEvaluator::chunk_layout()` is
  one chunk for default levels.

### Stage 3 — Chunk-Window Challenge Summaries

Generalize `summarize_all_block_carries` into a chunk-window helper. The flat
path is the first implementation target; the tensor path can preserve current
behavior for `num_chunks = 1` and reject chunk windows until the follow-up lands.

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
- Tensor `num_chunks = 1` still passes existing tensor summary tests.
- Tensor `num_chunks > 1` returns `AkitaError` until the follow-up is implemented.

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
    let block_offset_low = chunk.offset_e & (layout.blocks_per_chunk - 1);
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
        offset_high: chunk.offset_e >> block_bits,
        gadget_vector: &g1_open,
        challenge_block_summaries: &summaries,
        challenge_weight: self.eq_tau1[0],
    }.evaluate();

    t_structured_contribution += TStructuredSlicesEvaluator {
        high_challenges,
        offset_high: chunk.offset_t >> block_bits,
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
        let z_offset_low = chunk.offset_z & (self.block_len - 1);
        let a_block_summary = vec![summarize_pow2_multiplier_block_carries(...)?];
        z_structured_contribution += ZStructuredPow2SlicesEvaluator {
            high_challenges: &x_challenges[z_offset_low_bits..],
            offset_high: chunk.offset_z >> z_offset_low_bits,
            // ...
        }.evaluate();
    }
} else {
    let a_evals_by_point = vec![/* build once */];
    for chunk in &layout.chunks {
        z_structured_contribution += ZDenseSlicesEvaluator {
            offset_z: chunk.offset_z,
            a_evals_by_point: &a_evals_by_point,
            // ...
        }.evaluate()?;
    }
}
```

`r_contribution` is present only in the last chunk, indicated by
`lens.r_len.is_some()`. The `u`/`r` contributions use `zip` to pair each
chunk's offsets with its lengths, and `if let Some` to handle absent segments:

```rust
for (chunk, lens) in layout.chunks.iter().zip(&layout.chunk_lengths) {
    // always active
    acc += eval_z(chunk, lens.z_len);
    acc += eval_e(chunk, lens.e_len);
    acc += eval_t(chunk, lens.t_len);
    // present only in the last chunk
    if let Some(r_len) = lens.r_len {
        acc += eval_r(chunk, r_len);
    }
}
```

Tests:

- `single_chunk_matches_legacy_row_eval` for pow2 and non-pow2 `block_len`.
- `chunk_grouped_one_equals_single_chunk`.
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
    let block_sum = (chunk.offset_e & block_mask) + block_local;
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
        let high_base = chunk.offset_e >> block_bits;
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
        chunk.offset_z,
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

- `num_chunks = 1` setup weights exactly match the old `prepare` output.
- `num_chunks = 1` setup weights match the single-chunk layout.
- `num_chunks = W` setup contribution matches a materialized row-MLE for
  `W ∈ {1,2,4,8}`.
- a test or review assertion confirms `packed_slice_inner_sum` is not called in a
  per-chunk loop.

### Stage 6 — End-to-End Verifier Fixtures

After structured and setup components pass independently, add end-to-end verifier
fixtures that synthesize the resolved chunked layout. These do not require a
distributed prover; the fixtures materialize the relation row according to
`WitnessLayout` and compare the verifier's deferred row evaluation against it.

> Once the chunked-relation prover ([`specs/distributed-prover.md`](distributed-prover.md))
> lands, add a prove→verify roundtrip that proves with a multi-chunk preset and
> verifies with the same preset (`W ∈ {1,2,4,8}`). This is **additive**: the
> synthesized-layout fixtures remain the ground-truth (they pin the row-MLE value
> independent of the prover), while the roundtrip confirms the prover emits the
> exact `WitnessLayout` the verifier resolves. The prover spec owns that test;
> the shared `segment_layout` offset computation is the single source of truth
> both sides consume.

Expected test shape:

```rust
for w in [1, 2, 4, 8] {
    let lp_w = lp.with_witness_chunk(ChunkedWitnessCfg { num_chunks: w, num_activated_levels: 1 });
    let layout = relation.segment_layout(&lp_w, &inputs)?;
    let materialized = materialize_chunked_relation_row(&fixture, &layout)?;
    let expected = dense_eq_inner_product(&materialized, &fixture.full_vec_randomness);
    let got = prepared.eval_at_point::<_, D>(...)?;
    assert_eq!(got, expected);
}
```

Run this matrix:

- pow2 `block_len` (`512`) and dense fallback `block_len` (`510`).
- with and without the D block (`RelationMatrixRowLayout::WithDBlock` /
  `RelationMatrixRowLayout::WithoutDBlock`) where fixtures exist.
- `W = 1` plus every power-of-two divisor of `num_blocks`.
- negative malformed layouts: non-power-of-two `num_chunks`, `W ∤ num_blocks`,
  `W > num_blocks`, and too-small witness capacity.

### Stage 7 — Performance Gate

Add profiling/instrumentation before claiming the implementation satisfies the
cost target. The relevant measurements are span-level, not just whole-verifier
time.

Procedure:

1. Run the profile example under `num_chunks = 1`.
2. Run the same shape with `num_chunks = W` for `W = 2, 4, 8`.
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
distributed prover/schedule side. This is the point where `num_chunks > 1`
becomes more than a synthetic verifier fixture.

Implementation checklist:

- schedule/config code sets `LevelParams::witness_chunk` for distributed runs.
- generated table identity or runtime schedule descriptor includes `witness_chunk`.
- prover output witness layout matches the exact `[z|e|t]... [u][r]` geometry
  used by `segment_layout` (z-first within each chunk, matching PR #216; `u`/`r`
  only in last chunk as `Some`).
- verifier integration tests prove a distributed prover output with the chunked
  public layout and reject the same proof under `num_chunks = 1`.

Follow-ups after the first full landing:

- tensor/factored `c_alpha` chunk windowing.
- non-power-of-two `B_w` dense per-chunk fallback.
- ZK blinding under chunking.
- chunked ZK witness layout (rejected today).

## References

- Single-segment row-eval theory:
`[book/src/how/verifying/matrix_evaluation.md](../book/src/how/verifying/matrix_evaluation.md)`
- Distributed-relation verifier theory:
`[book/src/how/verifying/distributed-relation-verifier.md](../book/src/how/verifying/distributed-relation-verifier.md)`
- Distributed prover:
`[book/src/how/proving/distributed-prover.md](../book/src/how/proving/distributed-prover.md)`
- Code: `crates/akita-verifier/src/protocol/ring_switch.rs` (`eval_at_point`),
`crates/akita-verifier/src/protocol/slice_mle/` (structured + setup
evaluators), `crates/akita-types/src/setup_contribution.rs`
(`SetupContributionPlan`), `crates/akita-types/src/proof/ring_relation.rs`
(`RingRelationSegmentLayout`).
