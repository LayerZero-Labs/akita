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
(`NonChunked`), and witness width computed by
[`w_ring_element_count_with_counts_for_layout_bits`](../crates/akita-types/src/schedule.rs)
with `num_public_rows = 1`. Multi-chunk witness layout therefore **mis-prices**
fold schedules: `next_w_len`, sum-check round counts, terminal tail sizing, and
optimal fold depth are all wrong once $\texttt{num\_chunks} > 1$.

This spec defines the **planner-only** changes needed to take a public
[`MultiChunkWitnessCfg`](#1-multichunkwitnesscfg-akita-types) (chunk count and
how many leading fold **rounds** stay in multi-chunk witness format before
switching back to non-chunked sizing), pass it to the planner through
[`PlannerPolicy`](../crates/akita-planner/src/lib.rs), search schedules under
the chunked witness model, and ship **new generated schedule tables for
`D = 64` only**, with a `_multi_chunk` filename suffix. Prover execution,
verifier row-MLE evaluation, and witness production are **out of scope here**;
they depend on
[`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and
follow-on prover work, but the planner must emit the public layout metadata those
paths will consume.

**Post-main architecture.** [`AkitaScheduleLookupKey`](../crates/akita-types/src/schedule.rs)
is now **two-field only**: `(num_vars, num_polynomials)`. The old
`num_t_vectors`, `num_w_vectors`, and `num_z_vectors` dimensions are gone.
Multi-chunk witness layout is **not** encoded in the lookup key; it is entirely
driven by [`MultiChunkWitnessCfg`](../crates/akita-types/src/lib.rs) embedded in
`PlannerPolicy` (from [`CommitmentConfig::multi_chunk_witness_cfg()`](../crates/akita-config/src/lib.rs)).
The same `(num_vars, num_polynomials)` key can therefore resolve to different
schedules under different presets or policies — non-chunked and multi-chunk table
families are kept in **separate catalogs** whose identity embeds the config.

## Intent

### Goal

Extend the offline planner (`akita-planner`) and generated-table pipeline so that,
given a [`MultiChunkWitnessCfg`](../crates/akita-types/src/lib.rs) with
`num_chunks` and `num_multi_chunk_rounds = R`:

1. **Leading fold rounds** `0 .. R - 1` price witness growth and proof bytes under
   the **chunked** layout with $\texttt{num\_chunks}$ replicated $\widehat z$ segments.
2. **Round `R` and beyond** revert to today's **non-chunked** layout with a
   single $\widehat z$ (switch back to the single-node / CPU tail from the book).
3. **Schedule resolution** remains deterministic: table hit and DP miss produce
   the same [`Schedule`](../crates/akita-types/src/schedule.rs) for the same
   `(key, policy)` pair.
4. **`D = 64` presets** gain companion `_multi_chunk` generated tables whose
   **catalog identity** embeds `MultiChunkWitnessCfg`; table **row keys** stay
   `(num_vars, num_polynomials)` like their non-chunked siblings.

**Configuration surface.** Presets declare multi-chunk witness parameters through
[`CommitmentConfig::multi_chunk_witness_cfg()`](../crates/akita-config/src/lib.rs).
The existing [`policy_of`](../crates/akita-config/src/lib.rs) bridge copies that
struct into [`PlannerPolicy.multi_chunk_witness`](../crates/akita-planner/src/lib.rs)
so the `Cfg`-free DP and [`resolve_schedule`](../crates/akita-planner/src/resolve.rs)
receive the same inputs. Presets that do not override the trait default use
(`num_chunks = 1`, `num_multi_chunk_rounds = 0`).

Every planner entry point already takes `&PlannerPolicy` alongside the lookup key
(`find_schedule`, `resolve_schedule`, table emission). After this spec lands,
**chunked witness pricing reads only `policy.multi_chunk_witness`**, never extra
fields on `AkitaScheduleLookupKey`.

### Invariants

- **`MultiChunkWitnessCfg::default_non_chunked()` is byte-identical to today.** With
  `num_chunks == 1` and `num_multi_chunk_rounds == 0`, `find_schedule` /
  `resolve_schedule` / table expansion must reproduce the current schedule for
  every key the non-chunked tables cover. Protected by extending
  `generated_schedule_tables_match_find_schedule` with paired non-chunked vs
  default-config assertions on the **same** lookup keys.
- **Lookup key unchanged.** `AkitaScheduleLookupKey` remains
  `{ num_vars, num_polynomials }` only. No multi-chunk dimensions are added to
  the key or to [`GeneratedScheduleKey`](../crates/akita-planner/src/generated/mod.rs).
- **Policy is the layout selector.** `find_schedule(key, policy, …)` and
  `resolve_schedule(key, policy, …)` price chunked layout iff
  `policy.multi_chunk_witness.uses_multi_chunk_witness()`. Callers must pass the
  policy derived from the preset they intend to prove under; mismatched preset vs
  policy is out of scope (same as today for `tiered`, `basis_range`, etc.).
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
- **Catalog isolation.** Multi-chunk tables are separate modules / catalog
  identities from their non-chunked siblings; a `fp128_d64_onehot` policy must
  never alias a `fp128_d64_onehot_multi_chunk` table even when row keys match.
- **Verifier no-panic on planning path.** Invalid `(MultiChunkWitnessCfg,
  num_blocks)` combinations reject with `AkitaError`; the DP does not panic on
  malformed public inputs.
- **Preset is source of truth.** `multi_chunk_witness_cfg()` on each `Cfg` is
  the only place `(num_chunks, num_multi_chunk_rounds)` constants are authored;
  `policy_of` and generated-table identity derive from it — no hand-written
  `PlannerPolicy` literals for multi-chunk fields.

### Non-Goals

- **Prover implementation** (partial commits, local $\mathbf M_j$, node
  orchestration). The planner only emits parameters the prover will later consume.
- **Verifier row-MLE refactor.** Assumed landed or in flight per
  `distributed-verifier-row-eval.md`; this spec only requires
  `LevelParams::witness_shape` (or equivalent) to be set consistently.
- **Multi-chunk tiered commitments.** `resolve_schedule` rejects
  `tiered && policy.multi_chunk_witness.uses_multi_chunk_witness()`; multi-chunk
  + tiered stays unsupported until
  [`specs/multi-group-batching.md`](multi-group-batching.md)-style design exists.
- **Searching `num_multi_chunk_rounds`.** The round count is a **preset constant**
  chosen by the code author, not a DP search axis.
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
| $r$ | `m_row_count_for(num_segments, 0, layout) · r_decomp_levels` |

At the root, `num_polynomials` comes from the lookup key; recursive levels use
`num_polynomials = 1` and `num_public_rows = 1`.

### Chunked (multi-chunk witness)

Per [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and
[the distributed prover book chapter](book/src/how/proving/distributed-prover.md),
layout `WitnessType::Chunked(num_chunks)` concatenates `num_chunks` chunks

```text
[ e^0 | t^0 | z^0 ] ··· [ e^{num_chunks-1} | t^{num_chunks-1} | z^{num_chunks-1} ] | r
```

with:

- `blocks_per_chunk = num_blocks / num_chunks`
- $\widehat e^j$, $\widehat t^j$ each cover **only** chunk $j$'s block window
  (still scaled by root `num_polynomials` at level 0)
- $\widehat z^j$ is **full** `inner_width · num_digits_fold` (replicated per chunk,
  *not* divided by `num_chunks`)
- $r$ is **shared** (one tail), with row count priced using
  `num_segments = num_chunks` in
  [`m_row_count_for`](../crates/akita-types/src/layout/params.rs) (virtually
  shared horizontal concatenation of $\mathbf M_j$)

Closed form for total ring elements at an intermediate fold (non-zk core):

```text
e_chunk = num_polynomials · blocks_per_chunk · δ_open
t_chunk = num_polynomials · blocks_per_chunk · n_a · δ_open
z_chunk = inner_width · δ_fold                         // full fold width, not / num_chunks
body    = e_chunk + t_chunk + z_chunk
rings   = num_chunks · body + r_rows · r_decomp_levels
```

**Growth vs today.** The dominant extra cost is
$(\texttt{num\_chunks} - 1) · z_chunk$ ring elements per multi-chunk level —
the witness-width cost of avoiding cross-node $\widehat z$ all-reduce in the
distributed prover.

### Cutover to non-chunked

After **`num_multi_chunk_rounds`** multi-chunk fold **rounds** (absolute fold
levels `0 .. R - 1`, 0-indexed), the planner switches to non-chunked sizing
(single $\widehat z$, `num_public_rows = 1`). **Round `R`** (when present) is the
**cutover fold**: its *input* witness is still chunked; its *output* witness
uses non-chunked width (nodes coalesce to one logical prover — modeled only
as witness shrink in the planner; prover mechanics are out of scope).

If the optimal schedule has fewer than `R` folds, only the executed prefix uses
chunked pricing; the remaining configured multi-chunk rounds are a no-op.

## Design

**Terminology.** In prose this spec says **node** for a distributed prover
participant (matching the book's $P_j$). In code and identifiers we say
**chunk** for the same count: witness layout, config fields, and
`WitnessType::Chunked(num_chunks)` all use `num_chunks` / `blocks_per_chunk`, not
`num_nodes`.

### New and modified types

#### 1. `MultiChunkWitnessCfg` (`akita-types`)

Public configuration for multi-chunk witness layout, shared by presets, planner,
and (future) prover runtime. Lives in `akita-types` so both `akita-config` and
`akita-planner` can name it without a circular dependency.

```rust
/// How many witness chunks and for how many leading fold rounds the schedule
/// prices Chunked layout before switching back to NonChunked sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiChunkWitnessCfg {
    /// Number of witness chunks / replicated ẑ segments while multi-chunk layout
    /// is active. `1` means non-chunked (default).
    pub num_chunks: usize,

    /// Count of leading fold **rounds** (absolute fold levels 0, 1, …, R−1)
    /// priced under Chunked layout. `0` disables multi-chunk planning even if
    /// `num_chunks > 1` (invalid combination — see validation).
    pub num_multi_chunk_rounds: usize,
}

impl MultiChunkWitnessCfg {
    /// Non-chunked default: no chunking, no extra ẑ replication in the planner.
    pub const fn default_non_chunked() -> Self {
        Self { num_chunks: 1, num_multi_chunk_rounds: 0 }
    }

    pub const fn uses_multi_chunk_witness(self) -> bool {
        self.num_chunks > 1 && self.num_multi_chunk_rounds > 0
    }

    /// Preset helper for the initial D64 multi-chunk tables (book example: 8 nodes).
    pub const fn d64_production() -> Self {
        Self { num_chunks: 8, num_multi_chunk_rounds: 3 }
    }
}
```

**Validation** (`MultiChunkWitnessCfg::validate` or inline in `find_schedule`):

| Rule | Error |
|------|-------|
| `num_chunks == 0` | `InvalidSetup` |
| `num_chunks > 1` and not power of two | `InvalidSetup` |
| `num_multi_chunk_rounds > 0` and `num_chunks == 1` | `InvalidSetup` |
| `num_chunks > 1` and `num_multi_chunk_rounds == 0` | `InvalidSetup` (must specify round count) |
| `num_multi_chunk_rounds > MAX_RECURSION_DEPTH` | `InvalidSetup` |

Include `MultiChunkWitnessCfg` in schedule / instance descriptor bytes when
multi-chunk layout is active (append-only; default config omits or sends
`(1, 0)` deterministically).

#### 2. `CommitmentConfig` hook (`akita-config/src/lib.rs`)

Add a trait method with a **default** that preserves today's behavior for every
existing preset:

```rust
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    // ... existing associated items ...

    /// Multi-chunk witness parameters for schedule planning and (future) prover
    /// orchestration. Default: non-chunked.
    fn multi_chunk_witness_cfg() -> MultiChunkWitnessCfg {
        MultiChunkWitnessCfg::default_non_chunked()
    }
}
```

**Multi-chunk preset pattern** (e.g. `fp128::D64OneHotMultiChunk`):

```rust
impl CommitmentConfig for D64OneHotMultiChunk {
    // ... same Field / D / decomposition as D64OneHot ...

    fn multi_chunk_witness_cfg() -> MultiChunkWitnessCfg {
        MultiChunkWitnessCfg::d64_production()
        // or Self::MULTI_CHUNK_WITNESS_CFG if stored as a const on the preset
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
        multi_chunk_witness: Cfg::multi_chunk_witness_cfg(),  // NEW
    }
}
```

Every path that already calls `policy_of::<Cfg>()` (`runtime_schedule`,
`find_schedule` regen hooks, generated-table emission, drift guards) picks up the
multi-chunk settings automatically.

**Entry guards** in `resolve_schedule` / `find_schedule`:

```rust
let mc = policy.multi_chunk_witness;
if policy.tiered && mc.uses_multi_chunk_witness() {
    return Err(AkitaError::InvalidSetup(/* tiered + multi-chunk unsupported */));
}
mc.validate()?;
```

There is **no** lookup-key coupling: `key.validate()` stays the existing
two-field check (`num_vars > 0`, `num_polynomials > 0`).

#### 4. `PlannerPolicy` (`akita-planner/src/lib.rs`)

Add one field (not two loose scalars):

```rust
pub struct PlannerPolicy {
    // ... existing fields ...
    /// Multi-chunk witness settings derived from CommitmentConfig.
    pub multi_chunk_witness: MultiChunkWitnessCfg,
}
```

The planner reads `policy.multi_chunk_witness.num_chunks` and
`policy.multi_chunk_witness.num_multi_chunk_rounds` everywhere witness layout
depends on chunked vs non-chunked format. Convenience: import
`MultiChunkWitnessCfg` from `akita-types` (re-export from `akita-planner` if
helpful for emit tests).

**Defaults:** `PlannerPolicy` constructed in tests without `multi_chunk_witness`
uses `MultiChunkWitnessCfg::default_non_chunked()`.

#### 5. `WitnessType` on `LevelParams` (`akita-types/src/layout/params.rs`)

Add the public layout selector anticipated in
[`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md):

```rust
pub enum WitnessType {
    NonChunked,
    Chunked(usize), // num_chunks
}
```

Each [`LevelParams`](../crates/akita-types/src/layout/params.rs) emitted by the
planner carries `witness_shape: WitnessType` (default `NonChunked` for backward
compatibility). Multi-chunk fold steps set
`Chunked(policy.multi_chunk_witness.num_chunks)`; tail steps set `NonChunked`.

Include `witness_shape` in:

- `LevelParams` descriptor / schedule digest bytes (append-only field with
  explicit enum tag so old digests remain reproducible when multi-chunk layout is
  inactive).
- `GeneratedScheduleCatalogIdentity` (embed full `MultiChunkWitnessCfg`) so
  catalogs cannot alias across multi-chunk vs non-chunked presets.

#### 6. Witness width helper (`akita-types/src/schedule.rs` or `proof/witness_layout.rs`)

Introduce a layout-aware entry point aligned with the post-main scalar batch model:

```rust
pub fn w_ring_element_count_for_witness_type(
    field_bits: u32,
    lp: &LevelParams,
    num_polynomials: usize,
    layout: MRowLayout,
    witness_type: WitnessType,
) -> Result<usize, AkitaError>
```

Behavior:

- `WitnessType::NonChunked` → delegate to
  `w_ring_element_count_with_counts_for_layout_bits` with `num_public_rows = 1`.
- `WitnessType::Chunked(num_chunks)` → implement the closed form in **Background**
  with `num_segments = num_chunks` for the $r$-tail row count; validate
  `num_chunks > 0`, `num_chunks.is_power_of_two()`,
  `lp.num_blocks % num_chunks == 0`.

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

1. Add `MultiChunkWitnessCfg` to `akita-types` (+ re-export).
2. Add `CommitmentConfig::multi_chunk_witness_cfg()` with default
   `default_non_chunked()`.
3. Extend `policy_of::<Cfg>()` to set `PlannerPolicy.multi_chunk_witness`.
4. Validate `policy.multi_chunk_witness` at `find_schedule` / `resolve_schedule`
   entry.
5. Thread `PlannerPolicy` (with embedded config) through existing entry points:
   `find_schedule`, `resolve_schedule`, `schedule_from_entry`,
   `GeneratedFoldStep::expand_to_level_params`.

#### Step 2 — Witness width integration

1. Implement `w_ring_element_count_for_witness_type`.
2. Add `witness_shape_for_level(policy, fold_level) -> WitnessType` using
   `let mc = policy.multi_chunk_witness; let R = mc.num_multi_chunk_rounds; let num_chunks = mc.num_chunks`:
   - if `!mc.uses_multi_chunk_witness()` → always `NonChunked`
   - else if `fold_level < R` → `Chunked(num_chunks)` (output of folds `0 .. R-1`)
   - else if `fold_level == R` → **cutover output** `NonChunked`
   - else → `NonChunked`
3. Replace direct calls to `w_ring_element_count_with_counts_for_layout_bits`
   in the planner with the layout-aware helper, passing the witness type for the
   **output** witness of the fold being priced.

#### Step 3 — Root DP enumeration (`find_schedule` / `schedule_params.rs`)

At the root-only loop over `(log_basis, r_vars)`:

1. **Skip** candidates with `num_blocks % num_chunks != 0` when
   `mc.uses_multi_chunk_witness()`.
2. Compute `next_w_len` / `next_w_len_terminal` via
   `w_ring_element_count_for_witness_type(..., key.num_polynomials, …,
   Chunked(num_chunks))` when `num_multi_chunk_rounds >= 1`; otherwise unchanged
   (non-chunked with `num_polynomials = key.num_polynomials`).
3. Set `candidate_params.witness_shape = Chunked(num_chunks)` on emitted root fold
   steps when multi-chunk layout is active; root-direct `LevelParams` stays
   `NonChunked`.
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
    // absolute fold level of the *next* fold this suffix will price
    // (equals `level` parameter to derive_optimal_suffix_schedule).
}
```

For each suffix fold at absolute level `L` (with
`R = policy.multi_chunk_witness.num_multi_chunk_rounds`,
`num_chunks = policy.multi_chunk_witness.num_chunks`):

1. Determine **input** witness type:
   - `L == 0` → never reached (root handled separately)
   - `L < R` → input `Chunked(num_chunks)`
   - `L == R` → input `Chunked(num_chunks)` (cutover fold consumes chunked witness)
   - `L > R` → input `NonChunked`
2. Determine **output** witness type for `next_w_len`:
   - `L < R - 1` → output `Chunked(num_chunks)`
   - `L == R - 1` → output `NonChunked` (last multi-chunk round hands off to
     non-chunked tail)
   - `L >= R` → output `NonChunked`
3. Use `num_polynomials = 1` for recursive suffix folds (same as today).
4. Set `candidate_params.witness_shape` from the **input** type at this level
   (what the relation MLE sees).
5. `derive_candidate_level_params` stays unchanged for SIS key geometry; only
   witness-length accounting changes.

#### Step 5 — Terminal direct tail (`terminal_direct_suffix_cost`)

[`terminal_fold_num_polynomials`](../crates/akita-types/src/proof/direct_witness.rs)
returns `key.num_polynomials` at the root terminal predecessor and `1` at recursive
levels. Extend for multi-chunk layout:

- When the terminal predecessor fold has `witness_shape = Chunked(num_chunks)`,
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

Mirror the DP witness-type rules when walking compact
[`GeneratedStep`](../crates/akita-planner/src/generated/mod.rs) entries:

1. Track absolute `fold_level` as today.
2. When expanding each fold's `LevelParams`, set `witness_shape` from Step 4.
3. Recompute `next_w_len` with `w_ring_element_count_for_witness_type` instead of
   the singleton helper, using `policy.multi_chunk_witness` to select witness type.
4. Validate at catalog load: multi-chunk tables embed
   `identity.multi_chunk_witness == policy.multi_chunk_witness`.

#### Step 7 — Catalog identity (`catalog_identity.rs`)

Extend [`GeneratedScheduleCatalogIdentity`](../crates/akita-planner/src/generated/mod.rs):

```rust
pub multi_chunk_witness: MultiChunkWitnessCfg,
```

Include in `identity_digest` / `validate_catalog_identity`. Regenerated
non-chunked tables set `MultiChunkWitnessCfg::default_non_chunked()`.

#### Step 8 — Compact generated entries

**No change** to the 7-field [`GeneratedFoldStep`](../crates/akita-planner/src/generated/mod.rs)
tuple for v1: `witness_shape` is derived deterministically from
`(policy.multi_chunk_witness, fold_level)` at expansion time, not stored per step.

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
   `multi_chunk_witness: MultiChunkWitnessCfg::d64_production()` (via `policy_of`).
5. Run the same DP regen hook: `find_schedule(key, &policy, …)`.

#### Step 10 — New `Cfg` types and `ALL_GENERATED_FAMILIES` rows (`akita-config`)

Add **D = 64 multi-chunk companions** for the existing non-zk D64 families:

| Base module | Multi-chunk module | Base `Cfg` (pattern) |
|-------------|-------------------|----------------------|
| `fp128_d64_onehot` | `fp128_d64_onehot_multi_chunk` | `fp128::D64OneHotMultiChunk` |
| `fp128_d64_full` | `fp128_d64_full_multi_chunk` | `fp128::D64FullMultiChunk` |
| `fp128_d64_onehot_tensor` | `fp128_d64_onehot_tensor_multi_chunk` | `tensor_verifier::fp128::D64OneHotTensorMultiChunk` |

**Exclude** `fp128_d64_onehot_tiered_multi_chunk` until tiered + multi-$\widehat z$
is designed.

Each multi-chunk `Cfg`:

- `D = 64`, same field / decomposition / one-hot settings as its base.
- `multi_chunk_witness_cfg()` returns e.g.
  `MultiChunkWitnessCfg { num_chunks: 8, num_multi_chunk_rounds: 3 }` (initial
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
   equals `find_schedule(key, &policy_with_default_multi_chunk, …)` where
   `policy_with_default_multi_chunk` is `policy_of::<D64OneHot>()` with
   `multi_chunk_witness: default_non_chunked()` — confirms multi-chunk fields do
   not perturb non-chunked presets.

#### Step 12 — Regenerate tables

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
```

Commit new files:

- `crates/akita-schedules/src/generated/fp128_d64_onehot_multi_chunk.rs`
- `crates/akita-schedules/src/generated/fp128_d64_full_multi_chunk.rs`
- `crates/akita-schedules/src/generated/fp128_d64_onehot_tensor_multi_chunk.rs`

Non-zk only in this spec phase.

### Architecture diagram

```text
   CommitmentConfig::multi_chunk_witness_cfg()
              │
              ▼
   policy_of::<Cfg>()  ──►  PlannerPolicy.multi_chunk_witness
              │                    │
              │                    │  num_chunks
              │                    │  num_multi_chunk_rounds = R
              ▼                    ▼
     runtime_schedule      find_schedule / resolve_schedule
              │                    │
     AkitaScheduleLookupKey         │
     (num_vars, num_polynomials)   │  policy selects Chunked vs NonChunked
              └──────────┬───────────┘
                         ▼
              ┌─────────────────────────────────────┐
              │  Root DP (level 0)                   │
              │  skip if num_blocks % num_chunks != 0│
              │  Chunked output when L < R           │
              ├─────────────────────────────────────┤
              │  Suffix DP (levels ≥ 1)              │
              │  rounds 0..R−1 chunked; round R cutover │
              └──────────────┬──────────────────────┘
                             ▼
              Schedule { Fold*, Direct }
              LevelParams.witness_shape per round
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                           ▼
 fp128_d64_*_multi_chunk.rs                   DP fallback
 (table hit, identity.multi_chunk_witness set) (same key + policy)
```

### Alternatives considered

| Alternative | Why not (for v1) |
|-------------|------------------|
| Re-add `num_z_vectors` to the lookup key | Removed on main; multi-chunk cutover rounds are not opening-batch metadata; policy struct is the right home. |
| Encode multi-chunk layout only in table module name, share one `PlannerPolicy` | DP fallback and drift guards would not know which witness model to price without `policy.multi_chunk_witness`. |
| Flat `num_chunks` / `num_multi_chunk_rounds` on `PlannerPolicy` without struct | User-facing config should be one preset hook; struct keeps planner and prover aligned. |
| Store `witness_shape` in compact generated tuple | Extra bytes per step; derivable from `policy.multi_chunk_witness` + fold level. |
| Search optimal `num_multi_chunk_rounds` in DP | Round count is a code-author preset constant, not a search axis. |
| Reuse non-chunked tables with runtime scaling | Violates catalog identity; hides witness-layout metadata from verifier digest. |
| Skip generated tables; DP only | Breaks production preset latency; user explicitly requested D64 cached tables. |

## Evaluation

### Acceptance Criteria

- [ ] `MultiChunkWitnessCfg::default_non_chunked()` / default trait method
  reproduces existing schedules bit-for-bit for all current
  `ALL_GENERATED_FAMILIES` keys when passed through `policy_of` (multi-chunk field
  ignored for pricing).
- [ ] `w_ring_element_count_for_witness_type(Chunked(num_chunks))` unit tests match
  manual chunk layout arithmetic for `num_chunks ∈ {1, 2, 4, 8}` and reject invalid
  `(num_chunks, num_blocks)` pairs with `AkitaError`.
- [ ] `find_schedule` with `policy.multi_chunk_witness =
  MultiChunkWitnessCfg { num_chunks: 8, num_multi_chunk_rounds: 3 }` produces
  `LevelParams.witness_shape = Chunked(8)` on fold rounds `0..2` and
  `NonChunked` from round `3` onward for a smoke `num_vars` key.
- [ ] Root DP skips `(log_basis, r_vars)` with `2^r_vars % 8 != 0` when
  `num_chunks = 8`.
- [ ] Three `_multi_chunk` D64 modules emitted; `validate_catalog_identity`
  passes with embedded `multi_chunk_witness == d64_production()`.
- [ ] `generated_schedule_tables_match_find_schedule` passes with
  `--features all-schedules` including multi-chunk families (same keys as siblings,
  different policies).
- [ ] Multi-chunk preset rejects `tiered && multi_chunk_witness.uses_multi_chunk_witness()`.
- [ ] `CommitmentConfig::multi_chunk_witness_cfg()` default unchanged for all
  non-chunked presets (no macro churn).
- [ ] `AkitaScheduleLookupKey` remains two-field; no legacy vector-count fields
  in planner multi-chunk paths.
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test` pass.

### Testing Strategy

1. **Unit (`akita-types`)** — witness width helper vs explicit chunk stride math;
   cutover level output width `<` purely chunked width for `num_chunks > 1`.
2. **Unit (`akita-planner/schedule_params`)** — small `num_vars` brute schedule
   with `policy.multi_chunk_witness.num_chunks = 8` is deterministic; setting
   `multi_chunk_witness: default_non_chunked()` on the same policy matches golden
   non-chunked schedule for the same key.
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
- **Table size:** Three new D64 modules with the **same row count** as their
  non-chunked siblings (one entry per `(num_vars, num_polynomials)`); schedules
  differ because emission runs DP with a multi-chunk `PlannerPolicy`.
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
| S0 | `MultiChunkWitnessCfg` in `akita-types` + validation | — |
| S1 | `CommitmentConfig::multi_chunk_witness_cfg()` + `policy_of` wiring | S0 |
| S2 | `WitnessType` + `LevelParams.witness_shape` + descriptor bytes | — |
| S3 | `w_ring_element_count_for_witness_type` + tests | S2 |
| S4 | `PlannerPolicy.multi_chunk_witness` + planner entry validation | S0, S1 |
| S5 | Root DP wiring | S3, S4 |
| S6 | Suffix DP wiring | S3, S4 |
| S7 | `schedule_from_entry` expansion | S5, S6 |
| S8 | Terminal tail byte pricing | S3 |
| S9 | Catalog identity embeds `MultiChunkWitnessCfg` | S0, S1 |
| S10 | Multi-chunk `Cfg` types + `_multi_chunk` tables | S1, S5–S9 |

Stages S0–S9 are planner/config/`akita-types`-only and can land before prover/verifier
consume `witness_shape`. S10 ships the cached D64 tables.

## References

- Book: [The distributed prover](../book/src/how/proving/distributed-prover.md)
- Verifier layout spec: [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
- Planner crate map: [`crates/akita-planner/README.md`](../crates/akita-planner/README.md)
- Terminal tail sizing: [`crates/akita-types/src/proof/tail_segments.rs`](../crates/akita-types/src/proof/tail_segments.rs)
- Generated table pipeline: [`specs/schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- Prior art (terminal layout split): [`specs/terminal-fold-cutover.md`](terminal-fold-cutover.md)
