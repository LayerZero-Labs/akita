# Spec: Distributed-prover planner and D64 schedule tables

| Field         | Value                                                          |
|---------------|----------------------------------------------------------------|
| Author(s)     | Omid                                                           |
| Created       | 2026-06-25                                                     |
| Status        | proposed                                                       |
| PR            |                                                                |
| Supersedes    |                                                                |
| Superseded-by |                                                                |
| Book-chapter  | book/src/how/proving/distributed-prover.md                     |

## Summary

The [distributed prover](book/src/how/proving/distributed-prover.md) keeps a
**per-node folded response** $\mathbf z_j$ during the leading (expensive)
fold levels instead of all-reducing a single global $\mathbf z$. That changes
the **next-level witness shape**: witness columns are grouped into
$\texttt{num\_chunks}$ contiguous chunks, each holding a slice of $\widehat e$
and $\widehat t$ plus a **full-size** $\widehat z$, with a shared $r$-tail.

The planner today assumes the opposite everywhere: one $\widehat z$ segment
(single-chunk), and witness width computed by
[`w_ring_element_count_with_counts_for_layout_bits`](../crates/akita-types/src/schedule.rs)
with `num_public_rows = 1`. Multi-chunk witness layout therefore **mis-prices**
fold schedules: `next_w_len`, sum-check round counts, terminal tail sizing, and
optimal fold depth are all wrong once $\texttt{num\_chunks} > 1$.

This spec defines the **planner-only** changes needed to take a public
[`ChunkedWitnessCfg`](#1-chunkedwitnesscfg-akita-types) (chunk count and how many
leading fold **levels** stay in multi-chunk witness format before switching back
to single-chunk sizing), pass it to the planner through
[`PlannerPolicy`](../crates/akita-planner/src/lib.rs), search schedules under
the chunked witness model, and ship **new generated schedule tables for
`D = 64` only**, with a `_multi_chunk` filename suffix.

`ChunkedWitnessCfg` and the per-level `LevelParams.witness_chunk` field are
**owned by** [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
(it explicitly *replaced* an earlier `WitnessType`/`witness_shape` enum with this
struct). This spec consumes that type rather than introducing a parallel one, and
only adds planner-facing helpers/validation on it. Prover execution, verifier
row-MLE evaluation, and witness production are **out of scope here**; they depend
on the verifier spec and follow-on prover work, but the planner must populate
`witness_chunk` so those paths see consistent public layout metadata.

## Intent

### Goal

Extend the offline planner (`akita-planner`) and generated-table pipeline so that,
given a [`ChunkedWitnessCfg`](#1-chunkedwitnesscfg-akita-types) with `num_chunks`
and `num_activated_levels = R`:

1. **Leading fold levels** `0 .. R - 1` price witness growth and proof bytes under
   the **chunked** layout with $\texttt{num\_chunks}$ replicated $\widehat z$ segments.
2. **Level `R` and beyond** revert to today's **single-chunk** layout with a single
   $\widehat z$ (switch back to the single-node / CPU tail from the book).
3. **Schedule resolution** remains deterministic: table hit and DP miss produce
   the same [`Schedule`](../crates/akita-types/src/schedule.rs) for the same
   `(key, policy)` pair.
4. **`D = 64` presets** gain companion `_multi_chunk` generated tables whose
   **catalog identity** embeds `ChunkedWitnessCfg`; table **row keys** stay
   `(num_vars, num_polynomials)` like their non-chunked siblings.

**Configuration surface.** Presets declare multi-chunk witness parameters through
[`CommitmentConfig::chunked_witness_cfg()`](../crates/akita-config/src/lib.rs).
The existing [`policy_of`](../crates/akita-config/src/lib.rs) bridge copies that
struct into [`PlannerPolicy.witness_chunk`](../crates/akita-planner/src/lib.rs)
so the `Cfg`-free DP and [`resolve_schedule`](../crates/akita-planner/src/resolve.rs)
receive the same inputs. Presets that do not override the trait default use
`ChunkedWitnessCfg::default()` (`num_chunks = 1`, `num_activated_levels = 0`).

Every planner entry point already takes `&PlannerPolicy` alongside the lookup key
(`find_schedule`, `resolve_schedule`, table emission). After this spec lands,
**chunked witness pricing reads only `policy.witness_chunk`**, never extra
fields on `AkitaScheduleLookupKey`.

### Invariants

- **`ChunkedWitnessCfg::default()` is byte-identical to today.** With
  `num_chunks == 1` and `num_activated_levels == 0`, `find_schedule` /
  `resolve_schedule` / table expansion must reproduce the current schedule for
  every key the non-chunked tables cover, and every emitted `LevelParams` keeps
  `witness_chunk == ChunkedWitnessCfg::default()`. Protected by extending
  `generated_schedule_tables_match_find_schedule` with paired non-chunked vs
  default-config assertions on the **same** lookup keys.
- **Lookup key unchanged.** `AkitaScheduleLookupKey` remains
  `{ num_vars, num_polynomials }` only. No multi-chunk dimensions are added to
  the key or to [`GeneratedScheduleKey`](../crates/akita-planner/src/generated/mod.rs).
- **Policy is the layout selector.** `find_schedule(key, policy, …)` and
  `resolve_schedule(key, policy, …)` price chunked layout iff
  `policy.witness_chunk.uses_multi_chunk()`. Callers must pass the policy derived
  from the preset they intend to prove under; mismatched preset vs policy is out
  of scope (same as today for `tiered`, `basis_range`, etc.).
- **Block divisibility.** Multi-chunk root candidates require
  `num_blocks % num_chunks == 0` so each node owns an equal block window
  (`blocks_per_chunk = num_blocks / num_chunks`). Candidates violating this are
  skipped in the DP, not fixed up later.
- **Power-of-two `num_chunks`.** Initial scope: `num_chunks` is a power of two
  (matching the book's $2^N$ nodes and the verifier chunked fast path in
  [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)).
  Non-power-of-two chunk counts return `InvalidSetup` at plan time.
- **Single source of truth for witness width.** Chunked ring counts live in
  one new helper in `akita-types`; the planner DP, `schedule_from_entry`, and
  terminal tail sizing all call it. No duplicated closed forms in
  `schedule_params.rs`.
- **Single source of truth for the layout config.** There is exactly one struct
  describing chunked witness layout (`ChunkedWitnessCfg`, owned by the verifier
  spec). The planner reuses it; it does **not** introduce a second config type
  (e.g. a `MultiChunkWitnessCfg`) or a parallel `WitnessType`/`witness_shape`
  enum on `LevelParams`.
- **Catalog isolation.** Multi-chunk tables are separate modules / catalog
  identities from their non-chunked siblings; a `fp128_d64_onehot` policy must
  never alias a `fp128_d64_onehot_multi_chunk` table even when row keys match.
- **Verifier no-panic on planning path.** Invalid `(ChunkedWitnessCfg,
  num_blocks)` combinations reject with `AkitaError`; the DP does not panic on
  malformed public inputs.
- **Preset is source of truth.** `chunked_witness_cfg()` on each `Cfg` is
  the only place `(num_chunks, num_activated_levels)` constants are authored;
  `policy_of` and generated-table identity derive from it — no hand-written
  `PlannerPolicy` literals for multi-chunk fields.

### Non-Goals

- **Prover implementation** (partial commits, local $\mathbf M_j$, node
  orchestration). The planner only emits parameters the prover will later consume.
- **Verifier row-MLE refactor.** Assumed landed or in flight per
  `distributed-verifier-row-eval.md`; this spec only requires
  `LevelParams::witness_chunk` to be set consistently.
- **Multi-chunk tiered commitments.** `resolve_schedule` rejects
  `tiered && policy.witness_chunk.uses_multi_chunk()`; multi-chunk + tiered stays
  unsupported until [`specs/multi-group-batching.md`](multi-group-batching.md)-style
  design exists. (This is also why the chunked closed form below carries no
  $\widehat u$ term: $\widehat u$ is non-empty only under tiered commitment.)
- **Searching `num_activated_levels`.** The activated-level count is a **preset
  constant** chosen by the code author, not a DP search axis.
- **Non-`D = 64` generated tables.** Which ring dimensions get shipped
  `_multi_chunk` tables remains a **code-author decision** per preset family;
  this spec only adds D=64 companions. The planner does not auto-select ring
  dimension or trade proof size across families.
- **ZK schedule tables for multi-chunk.** Non-zk `_multi_chunk` tables first; zk
  is a follow-up mirroring the existing plain/zk split.

## Background: what changes in witness width

### Non-chunked (today)

[`w_ring_element_count_with_counts_for_layout_bits`](../crates/akita-types/src/schedule.rs)
prices one contiguous witness. The scalar same-point batch opens one claim per
polynomial, so `num_polynomials` from the lookup key drives both $\widehat e$ and
$\widehat t$ width at the root fold; recursive folds use `num_polynomials = 1`.
Public $\widehat z$ uses `num_public_rows = 1` (single opening point).

| Segment | Ring count (schematic) |
|---------|-------------------------|
| $\widehat e$ | `num_polynomials · num_blocks · num_digits_open` |
| $\widehat t$ | `num_polynomials · num_blocks · n_a · num_digits_open` |
| $\widehat z$ | `num_public_rows · inner_width · num_digits_fold` |
| $r$ | `m_row_count_for(num_commitments = 1, 0, layout) · r_decomp_levels` |

(`num_segments` in earlier drafts is the first `m_row_count_for` argument, named
`num_commitments` in [`params.rs`](../crates/akita-types/src/layout/params.rs);
today's single-chunk pricing passes `1`. The non-zk total also includes
`u_concat_ring_len_per_group()`, which is `0` for every non-tiered preset.)

At the root, `num_polynomials` comes from the lookup key; recursive levels use
`num_polynomials = 1` and `num_public_rows = 1`.

### Chunked (multi-chunk witness)

Per [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and
[the distributed prover book chapter](book/src/how/proving/distributed-prover.md),
a level with `witness_chunk.num_chunks = num_chunks > 1` concatenates
`num_chunks` chunks followed by a single shared tail

```text
[ z^0 | e^0 | t^0 ] ··· [ z^{num_chunks-1} | e^{num_chunks-1} | t^{num_chunks-1} ] | r
```

with (matching the verifier spec's per-chunk segment ordering and lengths):

- `blocks_per_chunk = num_blocks / num_chunks`
- $\widehat e^j$, $\widehat t^j$ each cover **only** chunk $j$'s block window
  (partitioned: their per-chunk lengths are the single-chunk totals divided by
  `num_chunks`; still scaled by root `num_polynomials` at level 0)
- $\widehat z^j$ is **full** `inner_width · num_digits_fold` (replicated per chunk,
  *not* divided by `num_chunks`)
- $\widehat u$ is empty (non-tiered) and so omitted
- $r$ is **shared** (one summed quotient $\mathbf r = \sum_j \mathbf r_j$, a single
  tail) and keeps the **single-machine shape**. The per-node relations stack
  *horizontally* ($\mathbf M = [\mathbf M_0 \mid \dots \mid \mathbf M_{num\_chunks-1}]$),
  so they **share the same row blocks** — concatenation adds columns, not rows —
  and the partial commitments $\mathbf u_j$ are summed into one $\mathbf u$ (one
  COMMIT block, not `num_chunks`). Its row count is therefore priced with
  `num_commitments = 1`, i.e. **unchanged from the single-chunk layout** (see the
  [distributed-prover book chapter](book/src/how/proving/distributed-prover.md):
  the summed quotient $\widehat{\mathbf r} = \mathbf G_{b,n}^{-1}(\mathbf r)$
  "recovers the same ring-switch witness shape as the single-machine protocol",
  $n = n_A + n_B + n_D + 1$; and the
  [verifier theory](book/src/how/verifying/distributed-relation-verifier.md):
  "the row axis is unchanged", `r_tail` delta `none`).

Closed form for total ring elements at an intermediate fold (non-zk, non-tiered):

```text
e_chunk = num_polynomials · blocks_per_chunk · δ_open
t_chunk = num_polynomials · blocks_per_chunk · n_a · δ_open
z_chunk = inner_width · δ_fold                         // full fold width, not / num_chunks
body    = e_chunk + t_chunk + z_chunk                  // u_concat = 0 for non-tiered
r_rows  = m_row_count_for(num_commitments = 1, 0, layout)  // summed quotient: single-machine shape, UNCHANGED
rings   = num_chunks · body + r_rows · r_decomp_levels
```

Note `num_chunks · e_chunk` and `num_chunks · t_chunk` equal the single-chunk
$\widehat e$ / $\widehat t$ totals exactly (the block window is merely
partitioned), so those segments do not grow.

**Growth vs today.** The **only** extra cost is
$(\texttt{num\_chunks} - 1) · z_chunk$ ring elements per multi-chunk level — the
witness-width cost of avoiding the cross-node $\widehat z$ all-reduce in the
distributed prover. The partitioned $\widehat e$ / $\widehat t$ tile the same
blocks and the shared $r$-tail keeps the single-machine row count
(`num_commitments = 1`), so neither grows. Pricing the $r$-tail with `num_chunks`
commitments would **over-count** it by $(\texttt{num\_chunks} - 1)\cdot n_B\cdot
\texttt{r\_decomp\_levels}$ rings and break the prover's
`emitted == next_w_len` and the verifier's single-machine `r_len` cross-checks.

### Cutover to single-chunk

Let `R = num_activated_levels`. In this codebase a fold step's `LevelParams`
describes the witness that step **commits/produces** — the witness whose size is
`next_w_len(L) = w_ring_element_count(params(L))`, which becomes the input of fold
`L + 1`. The distributed prover keeps per-node $\widehat z$ on the leading `R`
folds, so the witness committed at level `L` is chunked iff `L < R`. Define one
resolved chunk count per level:

```text
chunks_at_level(L) = num_chunks   if uses_multi_chunk() && L < R
                     1            otherwise
```

**Single source per level.** Both `params(L).witness_chunk` **and** the
`next_w_len(L)` width pricing use `chunks_at_level(L)` — they describe the same
witness, so a future verifier that recomputes the witness size from
`lp.witness_chunk` always agrees with the planner. There is no separate
input/output chunk count.

This gives a single, unambiguous cutover with no extra round:

- Levels `0 .. R - 1`: commit a **chunked** witness (`chunks_at_level = num_chunks`).
- Level `R` (the **cutover fold**): its input is the chunked witness committed by
  level `R - 1`, but it commits a **single-chunk** witness (`chunks_at_level(R) = 1`).
  The nodes coalesce to one logical prover here — modeled only as witness shrink in
  the planner; prover mechanics are out of scope.
- Level `R + 1` and beyond: single-chunk throughout.

Equivalently, exactly the leading `R` committed witnesses (levels `0 .. R - 1`)
carry replicated $\widehat z$; the divisibility constraint
(`num_blocks % num_chunks == 0`) applies to each such level's `num_blocks`.

If the optimal schedule has fewer than `R` folds, only the executed prefix uses
chunked pricing; the remaining configured activated levels are a no-op.

**Feasibility floor on chunked levels.** Because `num_chunks` and `num_blocks` are
both powers of two, `num_blocks % num_chunks == 0` is equivalent to
`r_vars(L) ≥ log₂(num_chunks)` at every chunked level `L < R`. The DP therefore
only considers `r`-splits with at least `log₂(num_chunks)` block bits on the
leading `R` folds; a cost-optimal split with fewer blocks is unavailable there.
If **no** candidate survives at a leading level (e.g. the witness has already
shrunk below `num_chunks` blocks), the DP finds no chunked schedule for that
`(key, policy)` and returns the usual "no schedule found" `AkitaError` rather than
silently falling back to single-chunk mid-prefix — keeping `chunks_at_level(L)`
and the stamped `witness_chunk` consistent. Presets pick `num_activated_levels`
so the leading folds always have `≥ num_chunks` blocks (the root is the largest).

## Design

**Terminology.** In prose this spec says **node** for a distributed prover
participant (matching the book's $P_j$). In code and identifiers we say
**chunk** for the same count: witness layout, config fields, and
`ChunkedWitnessCfg { num_chunks, .. }` all use `num_chunks` / `blocks_per_chunk`,
not `num_nodes`. A level is "chunked" when `lp.witness_chunk.num_chunks > 1`.

### New and modified types

#### 1. `ChunkedWitnessCfg` (`akita-types`)

`ChunkedWitnessCfg` is **defined by**
[`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and lives in
`crates/akita-types/src/witness/`, so both `akita-config` and `akita-planner` can
name it without a circular dependency and the verifier consumes the same type.
That spec defines:

```rust
/// Chunk-based witness layout parameters.
/// `num_chunks = 1` is the single-chunk (standard) case; `num_chunks` must be a
/// power of two. `num_activated_levels` is how many leading protocol levels the
/// multi-chunk layout is active; ignored when `num_chunks = 1`.
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

**This spec adds** the following planner-facing helpers/validation to the same
type (no new struct):

```rust
impl ChunkedWitnessCfg {
    /// Const equivalent of `Default::default()`, usable in const contexts
    /// (catalog identity literals).
    pub const fn default_non_chunked() -> Self {
        Self { num_chunks: 1, num_activated_levels: 0 }
    }

    /// True iff the planner should price chunked layout for the leading levels.
    pub const fn uses_multi_chunk(self) -> bool {
        self.num_chunks > 1 && self.num_activated_levels > 0
    }

    /// Layout-only validation (no dependency on planner internals).
    /// The depth bound against `MAX_RECURSION_DEPTH` is enforced separately at
    /// the planner entry — see below.
    pub fn validate(self) -> Result<(), AkitaError> { /* table below */ }

    /// Preset helper for the initial D64 multi-chunk tables (book example: 8 nodes).
    pub const fn d64_production() -> Self {
        Self { num_chunks: 8, num_activated_levels: 3 }
    }
}
```

**Validation** rules (`ChunkedWitnessCfg::validate`, except the last row):

| Rule | Error | Where |
|------|-------|-------|
| `num_chunks == 0` | `InvalidSetup` | `validate` |
| `num_chunks > 1` and not power of two | `InvalidSetup` | `validate` |
| `num_activated_levels > 0` and `num_chunks == 1` | `InvalidSetup` | `validate` |
| `num_chunks > 1` and `num_activated_levels == 0` | `InvalidSetup` (must specify level count) | `validate` |
| `num_activated_levels > MAX_RECURSION_DEPTH` | `InvalidSetup` | **planner entry** |

`MAX_RECURSION_DEPTH` is a private const in
[`akita-planner/src/schedule_params.rs`](../crates/akita-planner/src/schedule_params.rs),
not visible to `akita-types`. To respect the `akita-types ← akita-planner`
layering, `validate()` performs only the layout-only checks; the depth bound is
checked at the `find_schedule` / `resolve_schedule` entry (alongside the tiered
guard), where the const is in scope.

The verifier spec already includes `witness_chunk` in `LevelParams` descriptor
bytes (append-only; `ChunkedWitnessCfg::default()` reproduces today's bytes). No
additional descriptor-byte change is owned by this spec.

#### 2. `CommitmentConfig` hook (`akita-config/src/lib.rs`)

Add a trait method with a **default** that preserves today's behavior for every
existing preset:

```rust
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    // ... existing associated items ...

    /// Multi-chunk witness parameters for schedule planning and (future) prover
    /// orchestration. Default: single-chunk.
    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        ChunkedWitnessCfg::default()
    }
}
```

**Multi-chunk preset pattern** (e.g. `fp128::D64OneHotMultiChunk`):

```rust
impl CommitmentConfig for D64OneHotMultiChunk {
    // ... same Field / D / decomposition as D64OneHot ...

    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        ChunkedWitnessCfg::d64_production()
        // or Self::CHUNKED_WITNESS_CFG if stored as a const on the preset
    }
}
```

Non-chunked presets **do not override** the default. The macro-generated
`CommitmentConfig` impls in `proof_optimized.rs` need no change unless a preset
opts into multi-chunk witness layout.

#### 3. `policy_of` bridge (`akita-config/src/lib.rs`)

Extend the existing bridge — never hand-write multi-chunk literals on
`PlannerPolicy`:

```rust
pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    PlannerPolicy {
        ring_dimension: Cfg::D,
        // ... existing fields ...
        tiered: Cfg::TIERED_COMMITMENT,
        witness_chunk: Cfg::chunked_witness_cfg(),  // NEW
    }
}
```

Every path that already calls `policy_of::<Cfg>()` (`runtime_schedule`,
`find_schedule` regen hooks, generated-table emission, drift guards) picks up the
multi-chunk settings automatically.

**Entry guards** in `resolve_schedule` / `find_schedule`:

```rust
let mc = policy.witness_chunk;
mc.validate()?;                                   // layout-only rules
if policy.tiered && mc.uses_multi_chunk() {
    return Err(AkitaError::InvalidSetup(/* tiered + multi-chunk unsupported */));
}
if mc.num_activated_levels > MAX_RECURSION_DEPTH { // depth bound (planner-owned const)
    return Err(AkitaError::InvalidSetup(/* too many activated levels */));
}
```

There is **no** lookup-key coupling: `key.validate()` stays the existing
two-field check (`num_vars > 0`, `num_polynomials > 0`).

#### 4. `PlannerPolicy` (`akita-planner/src/lib.rs`)

Add one field (not two loose scalars):

```rust
pub struct PlannerPolicy {
    // ... existing fields ...
    /// Multi-chunk witness settings derived from CommitmentConfig.
    pub witness_chunk: ChunkedWitnessCfg,
}
```

The planner reads `policy.witness_chunk.num_chunks` and
`policy.witness_chunk.num_activated_levels` everywhere witness layout depends on
chunked vs single-chunk format. Convenience: import `ChunkedWitnessCfg` from
`akita-types` (re-export from `akita-planner` if helpful for emit tests).

**Defaults:** `PlannerPolicy` constructed in tests without `witness_chunk` uses
`ChunkedWitnessCfg::default()`.

#### 5. `LevelParams.witness_chunk` (`akita-types`)

This spec does **not** add a new layout type to `LevelParams`. The verifier spec
[`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) already adds
the field

```rust
// on LevelParams (owned by the verifier spec):
pub witness_chunk: ChunkedWitnessCfg,  // default: ChunkedWitnessCfg::default()
```

and explicitly *replaced* an earlier `witness_shape: WitnessType` enum with this
struct. The planner's only obligation is to **populate** `witness_chunk` on every
emitted `LevelParams`:

- A fold level at absolute level `L` sets
  `witness_chunk = ChunkedWitnessCfg { num_chunks: chunks_at_level(L), num_activated_levels: R }`
  when `chunks_at_level(L) > 1`, and `ChunkedWitnessCfg::default()` otherwise
  (so single-chunk and tail levels are byte-identical to today).
- `chunks_at_level(L)` describes the witness committed at level `L` (the one whose
  size is `next_w_len(L)`); the same count prices that level's `next_w_len`.

The catalog identity (Step 7) embeds the full `ChunkedWitnessCfg` so catalogs
cannot alias across multi-chunk vs non-chunked presets. No new descriptor-byte
field is owned by this spec — `witness_chunk` bytes come from the verifier spec.

#### 6. Witness width helper (`akita-types/src/schedule.rs` or `proof/witness_layout.rs`)

Introduce a layout-aware entry point aligned with the post-main scalar batch
model. It is parameterized by the resolved chunk count (`1` = single-chunk),
matching the verifier convention, rather than a separate enum:

```rust
pub fn w_ring_element_count_for_chunks(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    layout: MRowLayout,
    num_chunks: usize, // 1 = single-chunk (delegates to today's helper)
) -> Result<usize, AkitaError>
```

Behavior:

- `num_chunks == 1` → delegate to
  `w_ring_element_count_with_counts_for_layout_bits` with `num_public_rows = 1`
  (byte-identical to today).
- `num_chunks > 1` → implement the closed form in **Background**. The $r$-tail row
  count uses `num_commitments = 1` (the summed quotient keeps the single-machine
  shape — the horizontal $\mathbf M_j$ stacking adds columns, not rows — so the
  tail is byte-identical to the single-chunk delegate); only $\widehat z$ grows,
  by $(\texttt{num\_chunks}-1)\cdot z_chunk$. First validate
  `num_chunks.is_power_of_two()` and `lp.num_blocks % num_chunks == 0`, else
  `AkitaError::InvalidSetup`.

To keep the single-source-of-truth invariant, the `num_chunks > 1` branch must
mirror **every** segment of the delegate that is non-zero for the non-tiered
core ($\widehat e$, $\widehat t$, $\widehat z$, $r$); $\widehat u$ and the zk
blinding columns are zero in this phase (non-tiered, non-zk) and are added when
zk multi-chunk tables follow.

Unit tests in `akita-types` compare against the chunk offset arithmetic from
`distributed-verifier-row-eval.md` Stage 1 (`chunk_stride`, `offset_r`).

#### 7. Schedule lookup key — no multi-chunk fields

[`AkitaScheduleLookupKey`](../crates/akita-types/src/schedule.rs) and
[`GeneratedScheduleKey`](../crates/akita-planner/src/generated/mod.rs) stay:

```rust
pub struct AkitaScheduleLookupKey {
    pub num_vars: usize,
    pub num_polynomials: usize,
}
```

**Do not** add multi-chunk dimensions to the key. Table emission for multi-chunk
families enumerates the **same** `(num_vars, num_polynomials)` pairs as their
non-chunked siblings (via `AkitaScheduleLookupKey::new_from_opening_batch` /
existing family key lists). Multi-chunk vs non-chunked schedules differ because
the **policy** passed to `find_schedule` differs, and because each shipped table
module embeds a distinct catalog identity.

### Planner algorithm changes (step by step)

This is the core review section. Implement in roughly this order.

#### Step 1 — Config and policy plumbing

1. Add the planner helpers/validation on `ChunkedWitnessCfg` in `akita-types`
   (+ re-export). Do **not** define a second config struct.
2. Add `CommitmentConfig::chunked_witness_cfg()` with default
   `ChunkedWitnessCfg::default()`.
3. Extend `policy_of::<Cfg>()` to set `PlannerPolicy.witness_chunk`.
4. Validate `policy.witness_chunk` (layout rules + tiered guard + depth bound) at
   `find_schedule` / `resolve_schedule` entry.
5. Thread `PlannerPolicy` (with embedded config) through existing entry points:
   `find_schedule`, `resolve_schedule`, `schedule_from_entry`,
   `GeneratedFoldStep::expand_to_level_params`.

#### Step 2 — Witness width integration

1. Implement `w_ring_element_count_for_chunks`.
2. Add a single resolver `PlannerPolicy::chunks_at_level(fold_level) -> usize` —
   the chunk count of the witness committed at absolute level `fold_level`, with
   `let mc = self.witness_chunk; let R = mc.num_activated_levels`:
   - if `!mc.uses_multi_chunk()` → `1`
   - else if `fold_level < R` → `mc.num_chunks`
   - else → `1`
3. Replace direct calls to `w_ring_element_count_with_counts_for_layout_bits` in
   the planner with `w_ring_element_count_for_chunks`, passing
   `chunks_at_level(L)` when pricing the witness committed at level `L` (its
   `next_w_len`). The same count is stamped on `params(L).witness_chunk`, so the
   metadata and the priced size never diverge.

The cutover falls out automatically: level `R` commits a single-chunk witness
(`chunks_at_level(R) = 1`) consuming the chunked witness committed by level
`R - 1`.

#### Step 3 — Root DP enumeration (`find_schedule` / `schedule_params.rs`)

At the root-only loop over `(log_basis, r_vars)` (absolute level `L = 0`):

1. **Skip** candidates with `num_blocks % num_chunks != 0` when
   `mc.uses_multi_chunk()` (the root commits a chunked witness when `R >= 1`).
2. Compute `next_w_len` / `next_w_len_terminal` via
   `w_ring_element_count_for_chunks(..., key.num_polynomials, …,
   chunks_at_level(0))` — `num_chunks` when `R >= 1`, else `1` (single-chunk
   policies are unchanged).
3. Set the root fold step's `witness_chunk` from `chunks_at_level(0)` (i.e.
   `num_chunks` when `R >= 1`, else default). Root-direct `LevelParams` stays
   `ChunkedWitnessCfg::default()`.
4. **`level_proof_bytes`** already scales with `next_w_len` through
   [`sumcheck_rounds`](../crates/akita-types/src/layout/proof_size.rs) — no formula
   change once `next_w_len` is correct. Keep passing `num_claims =
   key.num_polynomials` at the root (see `resolve.rs`).

#### Step 4 — Suffix DP (`derive_optimal_suffix_schedule`)

The suffix memo key today is `(level, current_w_len, current_witness_len_terminal,
current_lb)`. Extend **`SuffixCtx`**:

```rust
struct SuffixCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_config: RingChallengeConfigFn<'a>,
    num_vars: usize,
    key: AkitaScheduleLookupKey,
    // `key` and `policy` are already present today; no new field is needed
    // because the absolute level is the existing `level` argument to
    // `derive_optimal_suffix_schedule`.
}
```

For each suffix fold at absolute level `L` (`L >= 1`; the root `L == 0` is handled
separately in Step 3), with `mc = policy.witness_chunk`:

1. `num_chunks = chunks_at_level(L)` — the chunk count of the witness committed at
   this level. Skip candidate `r`-splits whose `num_blocks % num_chunks != 0`.
2. Price `next_w_len` / `next_w_len_terminal` with
   `w_ring_element_count_for_chunks(..., 1, …, num_chunks)`. Use
   `num_polynomials = 1` for recursive suffix folds (same as today).
3. Set the fold step's `witness_chunk` from the same count:
   `ChunkedWitnessCfg { num_chunks, num_activated_levels: mc.num_activated_levels }`
   when `num_chunks > 1`, else `ChunkedWitnessCfg::default()`.
4. `derive_candidate_level_params` gains a `fold_level` argument so it can resolve
   `num_chunks`; SIS key geometry is otherwise unchanged.

Because `chunks_at_level` depends only on `(policy, L)` and the policy is fixed
per resolution, the existing memo key `(level, current_w_len,
current_witness_len_terminal, current_lb)` already distinguishes chunked vs
single-chunk states (different `current_w_len`); no memo-key change is required.

#### Step 5 — Terminal direct tail (`terminal_direct_suffix_cost`)

[`terminal_fold_num_polynomials`](../crates/akita-types/src/proof/direct_witness.rs)
returns `key.num_polynomials` at the root terminal predecessor and `1` at recursive
levels. Extend for multi-chunk layout:

- When the terminal predecessor fold is chunked (`chunks_at_level(L) > 1`),
  terminal tail layout must use **chunked** segment typing (future
  `SegmentTyped` generalization) **or** upper-bound with the chunked ring count
  converted to field elements.

**v1 planner approach (recommended):** add
`terminal_direct_witness_shape_chunked(...)` parallel to
`segment_typed_witness_shape`, producing `CleartextWitnessShape::SegmentTyped`
with per-chunk multiplicities encoded in `TailSegmentLayout` (may require adding
`num_chunks: usize` to the layout struct — coordinate with verifier spec Stage 8).

Until that lands, the planner **must still** price the correct byte count using
an upper-bound helper mirroring chunked ring count × `field_bytes` for the
non-`z` segments plus `segment_typed_z_payload_bytes` called with replicated
`z_coords = num_chunks · z_unit`.

#### Step 6 — Table expansion path (`resolve.rs` / `schedule_from_entry`)

Mirror the DP chunk-resolution when walking compact
[`GeneratedStep`](../crates/akita-planner/src/generated/mod.rs) entries:

1. Track absolute `fold_level` as today.
2. When expanding each fold's `LevelParams`, stamp `witness_chunk` from
   `chunks_at_level(fold_level)` exactly as in Steps 3–4 (the expander defaults it
   so a root-direct commit stays single-chunk; the walker overrides it for folds).
3. Recompute `next_w_len` with `w_ring_element_count_for_chunks` passing
   `chunks_at_level(fold_level)` instead of the singleton helper.
4. Validate at catalog load: multi-chunk tables embed
   `identity.witness_chunk == policy.witness_chunk`.

#### Step 7 — Catalog identity (`catalog_identity.rs`)

Extend [`GeneratedScheduleCatalogIdentity`](../crates/akita-planner/src/generated/mod.rs):

```rust
pub witness_chunk: ChunkedWitnessCfg,
```

Include in `identity_digest` / `validate_catalog_identity`. Regenerated
non-chunked tables set `ChunkedWitnessCfg::default()`.

#### Step 8 — Compact generated entries

**No change** to the 7-field [`GeneratedFoldStep`](../crates/akita-planner/src/generated/mod.rs)
tuple for v1: `witness_chunk` is derived deterministically from
`(policy.witness_chunk, fold_level)` via `chunks_at_level` at expansion time, not
stored per step.

#### Step 9 — Generated table emission (`akita-planner/src/emit`, `gen_schedule_tables`)

For each new multi-chunk family:

1. **`module_name`**: base name + `_multi_chunk` (e.g. `fp128_d64_onehot_multi_chunk`).
2. **`schedule_feature`**: e.g. `fp128-d64-onehot-multi-chunk` (new feature flags
   on `akita-schedules` / `akita-config`).
3. **`family_keys`**: **same enumeration** as the non-chunked sibling
   (`num_vars` / `num_polynomials` ranges unchanged). Example:

   ```rust
   let policy = policy_of::<D64OneHotMultiChunk>();
   // keys: AkitaScheduleLookupKey::new(num_vars, num_polys) — identical to D64OneHot
   ```

4. **`EmitSpec.policy`**: full `PlannerPolicy` including
   `witness_chunk: ChunkedWitnessCfg::d64_production()` (via `policy_of`).
5. Run the same DP regen hook: `find_schedule(key, &policy, …)`.

#### Step 10 — New `Cfg` types and `ALL_GENERATED_FAMILIES` rows (`akita-config`)

Add **D = 64 multi-chunk companions** for the existing non-zk D64 families:

| Base module | Multi-chunk module | Base `Cfg` (pattern) |
|-------------|-------------------|----------------------|
| `fp128_d64_onehot` | `fp128_d64_onehot_multi_chunk` | `fp128::D64OneHotMultiChunk` |
| `fp128_d64_full` | `fp128_d64_full_multi_chunk` | `fp128::D64FullMultiChunk` |

**Exclude** `fp128_d64_onehot_tiered_multi_chunk` until tiered + multi-$\widehat z$
is designed. The **tensor** verifier family (`fp128_d64_onehot_tensor`) does
**not** get a multi-chunk companion: the tensor-shaped root challenge is an
orthogonal verifier-cost optimization, kept separate from the distributed-prover
witness layout for now.

The companions delegate every layout parameter to their base `Cfg` via the
`impl_multi_chunk_companion!` helper and override only `chunked_witness_cfg()`
(→ `ChunkedWitnessCfg::d64_production()`) and `schedule_catalog()`.

Each multi-chunk `Cfg`:

- `D = 64`, same field / decomposition / one-hot settings as its base.
- `chunked_witness_cfg()` returns e.g.
  `ChunkedWitnessCfg { num_chunks: 8, num_activated_levels: 3 }` (initial
  production constants — document on the preset).
- `policy_of::<Cfg>()` picks up the config via the trait method (no override).
- `schedule_catalog()` points at the `_multi_chunk.rs` table.
- `runtime_schedule(key)` calls `resolve_schedule(key, &policy_of::<Self>(), …)`;
  no key mutation — multi-chunk behavior comes entirely from policy + catalog.

Wire modules in [`crates/akita-schedules/src/generated/mod.rs`](../crates/akita-schedules/src/generated/mod.rs)
behind the new feature flags.

#### Step 11 — Drift guards (`akita-config/tests/generated_tables.rs`)

1. Add multi-chunk rows to `ALL_GENERATED_FAMILIES` match arms
   (`family_catalog_is_linked`, `assert_family_catalog_enabled`, schedule
   comparison loop).
2. Extend `generated_schedule_tables_match_find_schedule` to cover multi-chunk
   families: for each key, compare table expansion under
   `policy_of::<MultiChunkCfg>()` against
   `find_schedule(key, &policy_of::<MultiChunkCfg>(), …)`.
3. Add regression: for a fixed key, `find_schedule(key, &policy_of::<D64OneHot>(), …)`
   equals `find_schedule(key, &policy_with_default_chunk, …)` where
   `policy_with_default_chunk` is `policy_of::<D64OneHot>()` with
   `witness_chunk: ChunkedWitnessCfg::default()` — confirms multi-chunk fields do
   not perturb non-chunked presets.

#### Step 12 — Regenerate tables

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
```

Commit new files:

- `crates/akita-schedules/src/generated/fp128_d64_onehot_multi_chunk.rs`
- `crates/akita-schedules/src/generated/fp128_d64_full_multi_chunk.rs`

Non-zk only in this spec phase.

### Architecture diagram

```text
   CommitmentConfig::chunked_witness_cfg()
              │
              ▼
   policy_of::<Cfg>()  ──►  PlannerPolicy.witness_chunk
              │                    │
              │                    │  num_chunks
              │                    │  num_activated_levels = R
              ▼                    ▼
     runtime_schedule      find_schedule / resolve_schedule
              │                    │
     AkitaScheduleLookupKey         │
     (num_vars, num_polynomials)   │  chunks_at_level(L) selects chunked vs single
              └──────────┬───────────┘
                         ▼
              ┌────────────────────────────────────────┐
              │  Root DP (level 0)                      │
              │  skip if num_blocks % num_chunks != 0   │
              │  commit chunks_at_level(0) witness      │
              ├────────────────────────────────────────┤
              │  Suffix DP (levels ≥ 1)                 │
              │  levels 0..R−1 commit chunked witness;  │
              │  level R commits single-chunk (cutover) │
              └──────────────┬─────────────────────────┘
                             ▼
              Schedule { Fold*, Direct }
              LevelParams.witness_chunk per level
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                           ▼
 fp128_d64_*_multi_chunk.rs                   DP fallback
 (table hit, identity.witness_chunk set)      (same key + policy)
```

### Alternatives considered

| Alternative | Why not (for v1) |
|-------------|------------------|
| Re-add `num_z_vectors` to the lookup key | Removed on main; activated levels are not opening-batch metadata; the policy struct is the right home. |
| Define a planner-only `MultiChunkWitnessCfg` distinct from the verifier's `ChunkedWitnessCfg` | The two would be field-identical (`num_chunks` + level count), duplicating the layout config across crates and violating single-source-of-truth. Reuse `ChunkedWitnessCfg`. |
| Add a `WitnessType`/`witness_shape` enum to `LevelParams` | The verifier spec (owner of `LevelParams` layout fields) explicitly rejected this enum and replaced it with `witness_chunk: ChunkedWitnessCfg`. The planner must populate that field, not add a competing one. |
| Encode multi-chunk layout only in table module name, share one `PlannerPolicy` | DP fallback and drift guards would not know which witness model to price without `policy.witness_chunk`. |
| Flat `num_chunks` / `num_activated_levels` on `PlannerPolicy` without struct | User-facing config should be one preset hook; the struct keeps planner, prover, and verifier aligned. |
| Store the per-level chunk count in the compact generated tuple | Extra bytes per step; derivable from `policy.witness_chunk` + fold level via `chunks_at_level`. |
| Search optimal `num_activated_levels` in DP | The activated-level count is a code-author preset constant, not a search axis. |
| Reuse non-chunked tables with runtime scaling | Violates catalog identity; hides witness-layout metadata from the verifier digest. |
| Skip generated tables; DP only | Breaks production preset latency; user explicitly requested D64 cached tables. |

## Evaluation

### Acceptance Criteria

- [ ] `ChunkedWitnessCfg::default()` / default trait method reproduces existing
  schedules bit-for-bit for all current `ALL_GENERATED_FAMILIES` keys when passed
  through `policy_of` (multi-chunk field ignored for pricing).
- [ ] `w_ring_element_count_for_chunks(num_chunks)` unit tests match manual chunk
  layout arithmetic for `num_chunks ∈ {1, 2, 4, 8}`, agree with the single-chunk
  delegate at `num_chunks == 1`, and reject invalid `(num_chunks, num_blocks)`
  pairs with `AkitaError`.
- [ ] `find_schedule` with `policy.witness_chunk =
  ChunkedWitnessCfg { num_chunks: 8, num_activated_levels: 3 }` produces
  `LevelParams.witness_chunk.num_chunks == 8` on fold levels `0..=2` and
  `== 1` from level `3` onward for a smoke `num_vars` key.
- [ ] Root DP skips `(log_basis, r_vars)` whose `num_blocks % 8 != 0` when
  `num_chunks = 8`.
- [ ] Two `_multi_chunk` D64 modules emitted (`onehot`, `full`);
  `validate_catalog_identity` passes with embedded `witness_chunk == d64_production()`.
- [ ] `generated_schedule_tables_match_find_schedule` passes with
  `--features all-schedules` including multi-chunk families (same keys as siblings,
  different policies).
- [ ] Multi-chunk preset rejects `tiered && witness_chunk.uses_multi_chunk()`.
- [ ] `CommitmentConfig::chunked_witness_cfg()` default unchanged for all
  non-chunked presets (no macro churn).
- [ ] `AkitaScheduleLookupKey` remains two-field; no legacy vector-count fields
  in planner multi-chunk paths.
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test` pass.

### Testing Strategy

1. **Unit (`akita-types`)** — witness width helper vs explicit chunk stride math;
   cutover level output width `<` purely chunked width for `num_chunks > 1`.
2. **Unit (`akita-planner/schedule_params`)** — small `num_vars` brute schedule
   with `policy.witness_chunk.num_chunks = 8` is deterministic; setting
   `witness_chunk: ChunkedWitnessCfg::default()` on the same policy matches the
   golden non-chunked schedule for the same key.
3. **Integration (`akita-config/tests/generated_tables.rs`)** — table hit vs DP
   for each `_multi_chunk` family and key cross-product under `policy_of`.
4. **Negative** — `num_chunks = 6` (not power of two) → `InvalidSetup`; `num_blocks
   = 5` with `num_chunks = 8` root candidate skipped (no panic, schedule still
   found if other `r_vars` valid).

### Performance

- **Proof size:** Multi-chunk schedules at the same `num_vars` carry more witness
  bytes when `num_chunks > 1`, driven by $(\texttt{num\_chunks} - 1)$ extra
  $\widehat z$ segments per multi-chunk level and longer sum-checks. The planner
  reports this in `Schedule.total_bytes`; it does not search for smaller proofs.
- **Table size:** Two new D64 modules with the **same row count** as their
  non-chunked siblings (one entry per `(num_vars, num_polynomials)`); schedules
  differ because emission runs DP with a multi-chunk `PlannerPolicy.witness_chunk`.
- **DP runtime:** Root loop skips more `r_vars` candidates due to divisibility;
  negligible vs existing exhaustive search.

Regenerate command (after implementation):

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
```

## Implementation stages (review order)

| Stage | Deliverable | Depends on |
|-------|-------------|------------|
| S0 | `ChunkedWitnessCfg` planner helpers + validation in `akita-types` | verifier spec's `ChunkedWitnessCfg` |
| S1 | `CommitmentConfig::chunked_witness_cfg()` + `policy_of` wiring | S0 |
| S2 | `LevelParams.witness_chunk` populated by the planner | verifier spec's `witness_chunk` field |
| S3 | `w_ring_element_count_for_chunks` + tests | — |
| S4 | `PlannerPolicy.witness_chunk` + planner entry validation | S0, S1 |
| S5 | Root DP wiring (`chunks_at_level`) | S3, S4 |
| S6 | Suffix DP wiring (`chunks_at_level`) | S3, S4 |
| S7 | `schedule_from_entry` expansion | S5, S6 |
| S8 | Terminal tail byte pricing | S3 |
| S9 | Catalog identity embeds `ChunkedWitnessCfg` | S0, S1 |
| S10 | Multi-chunk `Cfg` types + `_multi_chunk` tables | S1, S5–S9 |

S2 depends on the verifier spec having added the `witness_chunk` field to
`LevelParams`; the planner only populates it. Stages S0–S9 are
planner/config/`akita-types`-only otherwise and can land before the prover/verifier
*consume* `witness_chunk`. S10 ships the cached D64 tables.

## References

- Book: [The distributed prover](../book/src/how/proving/distributed-prover.md)
- Verifier layout spec: [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
- Planner crate map: [`crates/akita-planner/README.md`](../crates/akita-planner/README.md)
- Terminal tail sizing: [`crates/akita-types/src/proof/tail_segments.rs`](../crates/akita-types/src/proof/tail_segments.rs)
- Generated table pipeline: [`specs/schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- Prior art (terminal layout split): [`specs/terminal-fold-cutover.md`](terminal-fold-cutover.md)
