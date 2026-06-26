# Spec: Prover for the chunked-witness relation

| Field         | Value                                                          |
|---------------|----------------------------------------------------------------|
| Author(s)     | Omid                                                           |
| Created       | 2026-06-26                                                    |
| Status        | proposed                                                      |
| PR            |                                                                |
| Supersedes    |                                                                |
| Superseded-by |                                                                |
| Book-chapter  | book/src/how/proving/distributed-prover.md                     |

## Summary

This spec teaches the existing **single** Akita prover (`crates/akita-prover`) to
prove a **modified per-level relation**. The original relation carries one folded
response $\mathbf z$ over a single contiguous witness
$[\,\mathbf z \mid \widehat{\mathbf e} \mid \widehat{\mathbf t}\,]\,\|\,
\widehat{\mathbf r}$. The modified relation splits the block index set into
$W$ contiguous windows and carries **one folded response per window**, laying the
next-level witness out as $W$ chunks
$[\,\mathbf z_i \mid \widehat{\mathbf e}_i \mid \widehat{\mathbf t}_i\,]$ (z-first
per chunk) followed by a single shared quotient tail $\widehat{\mathbf r}$. The
chunk count $W$ for each fold level is a public parameter the
[planner](distributed-planner.md) already stamps on every level
(`LevelParams.witness_chunk.num_chunks`) and prices.

> **Notation.** $W := \texttt{num\_chunks}$ denotes the **chunk count** (the book's
> machine count $\mathcal M$); the verifier spec uses the same $W$. We reserve
> $\mathbf M$ for the relation matrix and $M = 2^m$ (as in the book) for the
> **block inner dimension** appearing in the gadget $\mathbf G_{b,M}$ — these are
> not the chunk count. With $W = 1$ the modified relation is exactly the original.

**Motivation (the only place another prover is mentioned).** We support this
modified relation so that a verifier can be built and validated against real
proofs of it; that verifier is the one a future GPU prover — which splits the
witness across machines and therefore naturally folds each block window
separately — will reuse. The prover in *this* spec is an ordinary single prover:
it holds the entire witness and simply assembles the modified relation, so it
emits a proof identical to the one that future prover would produce. Nothing below
is about that other prover; this spec only describes how the single prover proves
the modified relation.

The change is confined to **relation/witness construction**. The commitment
matvec (`A·s`, `B·t̂`, `D·ê`), the ring-switch quotient lift, and the sum-check
provers (`AkitaStage{1,2,3}Prover`) are unchanged — they are generic over the
witness and run over the modified witness/relation as-is. With $W = 1$ the
modified relation is exactly the original relation, byte-for-byte.

## Background: the original relation vs. the modified relation

### The original per-level relation

At one fold level the prover proves, over $\mathbb Z_q[X]$, the extended relation
$\mathbf M_{\mathrm{ext}}\,\mathbf w' = \mathbf h$ with
$\mathbf w' = (\widehat{\mathbf e},\widehat{\mathbf t},\mathbf z \,\|\,
\widehat{\mathbf r})$, public output $\mathbf h = (\mathbf v,\mathbf u,0,\mathbf 0)$,
and

$$
\mathbf M =
\begin{bmatrix}
  \mathbf D & 0 & 0 \\
  0 & \mathbf B & 0 \\
  \mathbf c^{\top}\!\otimes \mathbf G_{b,1} & 0 & -\,\mathbf a^{\top}\mathbf G_{b,M} \\
  0 & \mathbf c^{\top}\!\otimes \mathbf G_{b,n_A} & -\,\mathbf A
\end{bmatrix},
$$

where $\mathbf z = \sum_{j=1}^{B} c_j\,\mathbf s_j$ is the single folded response,
$\widehat{\mathbf e},\widehat{\mathbf t}$ are the per-block opening / inner-commit
digits over all $B = \texttt{num\_blocks}$ blocks, and $\widehat{\mathbf r}$ is the
$(X^d+1)$ quotient. (See `crates/akita-types/src/proof/ring_relation.rs`.)

### The modified relation, parameterized by `num_chunks = W`

Partition the block index set $[B]$ into $W$ contiguous windows
$\mathcal I_i = [\,iB_{\mathsf{loc}},(i{+}1)B_{\mathsf{loc}})$ with
$B_{\mathsf{loc}} = B/W$ (require $W \mid B$ and $W$ a power of two, so each window
is a clean power-of-two block range). Window $i$ gets its **own** sub-witness
$\mathbf w_i = (\widehat{\mathbf e}_i,\widehat{\mathbf t}_i,\mathbf z_i)$ where:

- $\widehat{\mathbf e}_i,\widehat{\mathbf t}_i$ are the original $\widehat{\mathbf e},
  \widehat{\mathbf t}$ **restricted to the blocks in $\mathcal I_i$** (so the union
  over windows is the whole $\widehat{\mathbf e},\widehat{\mathbf t}$ —
  **partitioned**);
- $\mathbf z_i = \sum_{j\in\mathcal I_i} c_j\,\mathbf s_j$ is a folded response
  summing only over window $i$'s blocks, but living in the **full** ambient fold
  space (size `inner_width`, the same as the single global fold) — so the $W$
  responses are full-size copies, **replicated**.

The relation is the horizontal concatenation
$\mathbf M = [\,\mathbf M_0 \mid \dots \mid \mathbf M_{W-1}\,]$, where $\mathbf M_i$
is the original relation block restricted to window $i$ (its $\mathbf D_i,\mathbf B_i$
column slices, its challenge slice $\mathbf c^{(i)} = \{c_j : j\in\mathcal I_i\}$,
the shared $\mathbf A$ and $\mathbf a^{\top}\mathbf G$). The public output is
**unchanged**: because the matrix columns partition across windows,
$\mathbf v = \sum_i \mathbf D_i\widehat{\mathbf e}_i$ and
$\mathbf u = \sum_i \mathbf B_i\widehat{\mathbf t}_i$, and

$$
\mathbf M\,\mathbf w = \sum_{i=0}^{W-1}\mathbf M_i\,\mathbf w_i = \mathbf h,
\qquad \mathbf w = (\mathbf w_0,\dots,\mathbf w_{W-1}).
$$

The next-level witness is the concatenation of the per-window blocks, z-first
within each window, with the single shared quotient appended:

```text
[ z_0 | e_0 | t_0 ][ z_1 | e_1 | t_1 ] … [ z_{W-1} | e_{W-1} | t_{W-1} ] [ r̂ ]
```

This is the layout the [planner](distributed-planner.md) prices
(`w_ring_element_count_for_chunks`) and the
[verifier](distributed-verifier-row-eval.md) evaluates (`segment_layout` /
`eval_at_point`). The per-window segment lengths are:

- `z_len_i = num_digits_fold · num_digits_commit · block_len` (replicated, full),
- `e_len_i = num_digits_open · num_claims · blocks_per_chunk` (partitioned),
- `t_len_i = num_digits_open · n_a · num_t_vectors · blocks_per_chunk` (partitioned),
- per-window stride `L = z_len_i + e_len_i + t_len_i`,
- one shared `r̂` tail of `num_rows · r_decomp_levels(log_basis)` after window $W-1$,
  where `num_rows` is the **single-machine** relation row count (the windows stack
  horizontally — same rows, partitioned columns — and the partial quotients sum,
  $\widehat{\mathbf r} = \sum_i \widehat{\mathbf r}_i$, so the tail does **not**
  scale with $W$).

`W = 1` makes every window the whole block set, one `z`, and recovers the original
witness exactly.

### Why the relation grows, and where the cost lands

Replicating the fold response is the only growth: the next-level witness gains
$(W-1)\cdot \texttt{z\_len}_i$ extra columns versus the original, lifting its
variable count by at most $\log_2 W$. The partitioned $\widehat{\mathbf e},
\widehat{\mathbf t}$ do not grow (the windows tile the same blocks), the quotient
is unchanged (below), and the commit/sum-check machinery is identical. The planner
already accounts for this growth in the schedule's proof-byte total.

## Design: how the single prover proves the modified relation

The prover reads $W$ off the level it is proving and builds the modified relation
in three steps (fold → witness layout → relation MLE); everything else is reused.

### Per-level entry and the block partition

In `prove_fold` (`crates/akita-prover/src/protocol/core/fold.rs`), read the chunk
count the planner stamped and derive the windows:

```rust
let num_chunks = lp.witness_chunk.num_chunks;        // W; 1 on non-modified levels
let blocks_per_chunk = lp.num_blocks / num_chunks;   // B_loc, power of two
// window i owns global blocks [ i*B_loc, (i+1)*B_loc )
```

Validate at this boundary, before any witness math (no-panic contract):

| Rule | Error |
|------|-------|
| `num_chunks == 0` | `InvalidSetup` |
| `num_chunks > 1` and not a power of two | `InvalidSetup` |
| `num_chunks > 1` and `lp.num_blocks % num_chunks != 0` | `InvalidSetup` |
| `num_chunks > 1` and `lp.tier_split > 1` | `InvalidSetup` |
| `num_chunks > 1` under `feature = "zk"` | `InvalidSetup` |

(`tier_split > 1` and the `zk` blinding segments are not specified for the chunked
witness yet; reject rather than mis-shape. This matches the planner entry guard and
the verifier spec's Stage 0.)

### Step 1 — compute the $W$ folded responses

The original prover computes one fold $\mathbf z = \sum_j c_j\mathbf s_j$ in
`build_point_decompose_fold_witness` (`protocol/ring_relation.rs`), via the
block-parallel decompose-fold (`backend/poly_helpers/decompose_fold_partitioned.rs`)
that already accumulates a per-block contribution before reducing. For the modified
relation, group that accumulation into the $W$ windows and **emit one response per
window without the cross-window reduction**:

```text
for i in 0..W:
    z_i = Σ_{j ∈ I_i} c_j · s_j            // full inner_width vector
```

Each $\mathbf z_i$ is the same full `inner_width` size as the single fold (it is a
partial sum, not a $1/W$ slice) and is decomposed by `num_digits_fold` exactly as
today. The fold challenge $\mathbf c$ is the **same single transcript-sampled
vector**; window $i$ uses the slice $c_j, j\in\mathcal I_i$ indexed by the
**global** block (so the verifier reads $c_\alpha$ at global block
$iB_{\mathsf{loc}}+\text{block\_local}$). The fold-grind L∞ cap is unchanged: each
$\mathbf z_i$ is a sub-sum of the global fold under the same challenge, so it
respects the same per-fold cap the planner sized.

Output: `z_folded_rings_per_chunk: Vec<Vec<CyclotomicRing<F,D>>>` of length $W$
(the $W = 1$ case is a one-element vector = today's single `z_folded_rings`).

### Step 2 — assemble the chunked witness (`build_w_coeffs`)

`build_w_coeffs` (`protocol/ring_switch/coeffs.rs`) emits the per-window blocks
z-first, then the shared quotient:

```text
for i in 0..W:
    emit z_i        # full inner_width fold response for window i (emit_z_folded_block_inner)
    emit e_i        # e_hat blocks in I_i  (windowed slice of the block-major e_hat)
    emit t_i        # inner A digits for blocks in I_i
emit r̂              # decomposed shared quotient (after the last window)
```

- $\widehat{\mathbf e}_i,\widehat{\mathbf t}_i$ are **slices** of the existing
  block-major $\widehat{\mathbf e},\widehat{\mathbf t}$ over the contiguous block
  range $\mathcal I_i$ (the block axis is innermost within each digit/claim plane),
  so no value is recomputed — only the emit window changes.
- The within-window order is z-first, at offsets `offset_z = base`,
  `offset_e = base + z_len_i`, `offset_t = base + z_len_i + e_len_i`,
  `base = i·L`. These must equal the verifier's `WitnessChunkLayout`.
- The emitted flat length must equal the planner's `next_w_len` for this level
  (`w_ring_element_count_for_chunks(..., num_chunks)`); assert and reject on
  mismatch. (The shared `r̂` tail keeps its single-machine row count, so this is
  consistent with the value-identical quotient below.)
- `W = 1` reduces to today's single
  $\mathbf z \,\|\, \widehat{\mathbf e} \,\|\, \widehat{\mathbf t} \,\|\,
  \widehat{\mathbf r}$ emission.

> **Single source of offsets.** The per-window `(z_len_i, e_len_i, t_len_i)` and
> offsets should come from one shared `akita-types` helper (the same one the
> verifier's `segment_layout` uses), so the prover's emission and the verifier's
> reading cannot drift.

### Step 3 — evaluate the modified relation MLE (`compute_m_evals_x`)

The relation-check sum-check links the committed witness MLE
$\widetilde{\mathbf w}$ to the relation MLE $\widetilde{\mathbf M}$. Because the
relation matrix is now $\mathbf M = [\mathbf M_0\mid\dots\mid\mathbf M_{W-1}]$, the
prover-internal column evaluation `compute_m_evals_x`
(`protocol/ring_switch/evals.rs`) must evaluate the **chunked** column layout:

- the `e`/`t` rows read each column at its window's block range, with the
  consistency challenge $c_\alpha$ taken at the **global** block;
- the `z` rows are the same in every window (the fold rows carry no
  window-specific data — the opening weight $a[\text{blk}]$ and the gadget weights
  are global), only the column offset differs;
- the `D`/`B`/`A` setup rows map each setup column to its window's segment;
  $\mathbf A$ acts identically on every $\mathbf z_i$.

This is the column-MLE counterpart of the verifier's chunked row-MLE
(specified in [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)).
If the prover and verifier share the structured slice evaluators
(`crates/akita-verifier/src/protocol/slice_mle/`), the chunked support added there
is reused and this step is a thin fold over windows; otherwise the prover gets the
analogous per-window fold. The **sum-check prover bodies are unchanged** — they
consume `m_evals_x` and `w_evals_compact` exactly as before.

### Reused unchanged

- **Quotient.** `compute_relation_quotient` is **value-identical**. The modified
  relation's row values equal the original's: on the consistency row,
  $\sum_i\bigl[(\mathbf c^{(i)\top}\!\otimes G)\widehat{\mathbf e}_i - \mathbf a^{\top}G\,\mathbf z_i\bigr]
   = \sum_j c_j G\widehat{\mathbf e}_j - \mathbf a^{\top}G\sum_i\mathbf z_i
   = \sum_j c_j G\widehat{\mathbf e}_j - \mathbf a^{\top}G\,\mathbf z$
  (since $\sum_i\mathbf z_i = \mathbf z$), and likewise on the `A`-rows. So
  $\mathbf M\mathbf w = \mathbf h$ has the same lifted value as the original and
  the quotient $\widehat{\mathbf r} = (\mathbf M\mathbf w - \mathbf h)/(X^d+1)$ is
  identical; the prover computes it as today and emits it as one shared tail.
- **Commit.** `commit_next_w` commits the assembled witness using the next level's
  `LevelParams` (sized by the planner). The witness is longer; the matvec kernels
  are unchanged.
- **Sum-check.** `ring_switch_finalize` builds `w_evals_compact` from the chunked
  flat witness; `AkitaStage{1,2,3}Prover` run unchanged, emitting the
  $\le \log_2 W$ extra round polynomials the planner already priced.

### Levels with `W = 1`

Levels the planner did not modify carry `num_chunks = 1`; Steps 1–3 collapse to the
original single-`z` relation and emit byte-identical output. A level whose input
witness was chunked by the previous level but which is itself `W = 1` simply folds
the (larger) input as an ordinary flat witness — no special handling, because the
prover reads `num_chunks` per level and the chunked construction is inert for
`W = 1`.

### Terminal level

The terminal direct witness (`build_segment_typed_witness`, z-first
`SegmentTypedWitness`) is reached after the witness has shrunk. If a terminal
predecessor is itself a modified (`W > 1`) level, the planner prices the chunked
terminal **as an upper bound** via the chunked ring count
(`w_ring_element_count_for_chunks(.., WithoutDBlock, num_chunks)`) — its first
landing does not yet add a per-chunk `num_segments` to `tail_segment_layout`
(see [`distributed-planner.md`](distributed-planner.md) Step 5). The prover emits
the per-window terminal segments matching that count. Otherwise the terminal is the
ordinary single-segment witness.

### Prover flow

```text
            LevelParams.witness_chunk.num_chunks = W
                              │
        prove_fold(level)     ▼
                ┌──────────────────────────────────────────────┐
   step 1 ────► │ fold → W responses  z_i = Σ_{j∈I_i} c_j s_j   │  (full inner_width each)
                ├──────────────────────────────────────────────┤
   step 2 ────► │ build_w_coeffs:                               │
                │   [z_0|e_0|t_0]…[z_{W-1}|e_{W-1}|t_{W-1}] | r̂ │  (len == planner next_w_len)
                ├──────────────────────────────────────────────┤
   step 3 ────► │ compute_m_evals_x over the chunked columns    │
                └───────────────────────┬──────────────────────┘
                                        ▼
   UNCHANGED:  compute_relation_quotient  +  commit_next_w  +  AkitaStage{1,2,3}Prover
```

## Invariants

- **`W = 1` byte-identical.** With `witness_chunk = ChunkedWitnessCfg::default()`,
  Steps 1–3 reduce to today's path and produce a byte-identical
  `AkitaBatchedProof` for every existing preset/key.
- **Layout equals the verifier's.** The prover's per-window offsets and lengths
  equal `RingRelationInstance::segment_layout` for the same `LevelParams`
  (shared offset helper); the emitted flat length equals the planner's
  `next_w_len`.
- **Fold-response identity.** $\sum_i\mathbf z_i$ equals the single global fold,
  and each $\mathbf z_i$ is full `inner_width`. (The witness keeps the $W$ responses
  separate; the identity is a correctness check on Step 1.)
- **Global fold challenge.** $\mathbf c$ is the same single transcript-sampled
  vector; window $i$ uses its global-block-indexed slice.
- **Quotient unchanged.** $\widehat{\mathbf r}$ is identical to the original
  relation's quotient — one shared tail with the **single-machine** row count,
  computed once ($\widehat{\mathbf r} = \sum_i\widehat{\mathbf r}_i$, not scaled
  by $W$).
- **Commit / quotient / sum-check provers untouched.** No change to the matvec
  commit kernels, `compute_relation_quotient`, or the `AkitaStage{1,2,3}Prover`
  bodies; they consume the modified witness/relation.
- **No-panic / determinism.** Chunk count, block window, per-window offsets, and
  the replicated-`z` capacity are validated from public `LevelParams` before any
  witness math; malformed shapes reject with `AkitaError`, never panic. Same inputs
  → same proof bytes.
- **Tiered and ZK rejected.** `W > 1` with `tier_split > 1`, and `W > 1` under
  `zk`, reject with `AkitaError`.

## Implementation Stages

Small, reviewable stages; each has an invariant and tests.

### S0 — Read `witness_chunk`; `W = 1` inert

Read `lp.witness_chunk` in `prove_fold`; add the boundary validation; keep every
chunked branch a no-op for `W = 1`.

- **Invariant:** zero behavior change for existing presets.
- **Tests:** existing prover + `akita-pcs` tests pass; boundary rejections return
  `AkitaError`.

### S1 — Chunked witness assembly (`build_w_coeffs`)

Emit `[z_i|e_i|t_i]…|r̂` z-first per window. For this stage, derive the per-window
`z_i` by partitioning the already-computed global fold inputs, so the layout is
validated before the kernel change.

- **Invariant:** emitted length == planner `next_w_len`; per-window offsets ==
  verifier `segment_layout`.
- **Tests:** `witness_layout_matches_segment_layout` (`W ∈ {1,2,4,8}`); `W = 1`
  byte-identical.

### S2 — $W$ folded responses

Produce $W$ full-ambient responses in the decompose-fold path; wire
`z_folded_rings_per_chunk` into `build_w_coeffs`.

- **Invariant:** $\sum_i z_i$ == single global fold; each `z_i` full
  `inner_width`; L∞ cap satisfied per window.
- **Tests:** `fold_responses_sum_to_global_fold` (`W ∈ {2,4,8}`).

### S3 — Modified relation MLE (`compute_m_evals_x`)

Evaluate the chunked column layout (reusing the verifier's structured evaluators
where shared). Sum-check provers unchanged.

- **Invariant:** the prover's relation MLE matches the verifier's chunked row-MLE
  for the same `witness_chunk`.
- **Tests:** `relation_mle_matches_verifier_row_mle` (`W ∈ {1,2,4,8}`).

### S4 — Proof-size parity and cutover

Confirm commit + sum-check run over the modified witness and the produced proof
size equals the planner schedule; confirm `W = 1` levels resume the original
relation.

- **Invariant:** `total_bytes` == planner `Schedule.total_bytes` for the D64
  multi-chunk presets; `W = 1` levels are single-`z`.
- **Tests:** `proof_size_matches_planner_schedule` (`nv ∈ {32,43}`,
  `num_polys ∈ {1,4}`); `single_chunk_resumes_original_relation`.

### S5 — End-to-end prove → verify

With the chunked verifier landed, prove with a multi-chunk preset and verify with
the same preset for `W ∈ {1,2,4,8}`, `block_len` pow2 (root) and dense (recursive).

- **Invariant:** the modified-relation proof verifies; `W = 1` matches the legacy
  proof.
- **Tests:** `chunked_prove_verify_roundtrip` (gated on verifier landing);
  `single_chunk_roundtrip_is_legacy`.

## Evaluation

### Acceptance Criteria

- [ ] `W = 1` produces a byte-identical `AkitaBatchedProof` to today.
- [ ] Emitted witness layout equals `segment_layout` for `W ∈ {1,2,4,8}`.
- [ ] $\sum_i z_i$ equals the single global fold; each `z_i` full `inner_width`.
- [ ] The prover's relation MLE matches the verifier's chunked row-MLE.
- [ ] Produced proof size equals the planner `Schedule.total_bytes` for the D64
  multi-chunk presets (shared `r̂` tail keeps its single-machine row count).
- [ ] The modified-relation proof verifies under the chunked verifier for
  `W ∈ {2,4,8}` (pow2 and dense `block_len`); `W = 1` matches the legacy proof.
- [ ] No change to the matvec commit kernels, `compute_relation_quotient`, or the
  `AkitaStage{1,2,3}Prover` bodies (review assertion).
- [ ] `W > 1` with `tier_split > 1` and `W > 1` under `zk` reject with
  `AkitaError` (no panic).
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test` pass.

### Testing Strategy

1. **Layout cross-check** against `segment_layout` (`W ∈ {1,2,4,8}`).
2. **Fold-response unit:** `fold_responses_sum_to_global_fold`.
3. **Relation-MLE unit:** prover `compute_m_evals_x` vs the verifier-materialized
   chunked relation row.
4. **Proof-size parity** vs the planner schedule.
5. **End-to-end roundtrip** (gated on verifier landing).
6. **Determinism** and **no-panic negatives** (bad `num_chunks`,
   `num_chunks ∤ num_blocks`, tiered+chunked, zk+chunked).

### Performance

The modified relation's witness is larger by $(W-1)\cdot\texttt{z\_len}_i$ per
chunked level, so the prover does more work (more commit columns, $\le\log_2 W$
extra sum-check rounds) and the proof grows by the planner-priced ~4–6% (D64).
That growth is the intrinsic cost of the replicated fold response (the partitioned
$\widehat{\mathbf e},\widehat{\mathbf t}$ and the shared $\widehat{\mathbf r}$ tail
do not grow); the verifier's dominant cost is unchanged (verifier spec).

## References

- Relation theory: [`book/src/how/proving/distributed-prover.md`](../book/src/how/proving/distributed-prover.md)
  (the per-window relation block and the horizontal concatenation),
  [`book/src/how/verifying/distributed-relation-verifier.md`](../book/src/how/verifying/distributed-relation-verifier.md)
  (partitioned vs replicated components).
- Planner (prices the relation): [`specs/distributed-planner.md`](distributed-planner.md)
- Verifier (evaluates the relation): [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
- Prover surfaces: `crates/akita-prover` (`protocol/core/fold.rs`,
  `protocol/ring_switch/{coeffs,evals,finalize}.rs`, `protocol/ring_relation.rs`,
  `backend/poly_helpers/decompose_fold_partitioned.rs`)
- Witness/relation types: `crates/akita-types`
  (`proof/ring_relation.rs`, `witness.rs`, `layout/params.rs`)
