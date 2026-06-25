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
$W = \texttt{num\_chunks}$ contiguous chunks, each holding a slice of $\widehat e$
and $\widehat t$ plus a **full-size** $\widehat z$, with a shared $r$-tail.

The planner today assumes the opposite everywhere: one $\widehat z$ segment
(`ComponentMajor`), and witness width computed by
[`w_ring_element_count_with_counts_for_layout_bits`](../crates/akita-types/src/schedule.rs)
with `num_public_rows = 1`. Distributed proving therefore **mis-prices** fold
schedules: `next_w_len`, sum-check round counts, terminal tail sizing, and
optimal fold depth are all wrong once $W > 1$.

This spec defines the **planner-only** changes needed to take a public
[`DistributedProverConfig`](#1-distributedproverconfig-akita-types) (chunk count
$W$ and how many leading fold **rounds** stay in distributed mode before switching
back to single-node sizing), pass it to the planner through
[`PlannerPolicy`](../crates/akita-planner/src/lib.rs), search schedules under
the chunked witness model, and ship **new generated schedule tables for
`D = 64` only**, with a `_distributed` filename suffix. Prover execution,
verifier row-MLE evaluation, and witness production are **out of scope here**;
they depend on
[`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and
follow-on prover work, but the planner must emit the public layout metadata those
paths will consume.

**Post-main architecture.** [`AkitaScheduleLookupKey`](../crates/akita-types/src/schedule.rs)
is now **two-field only**: `(num_vars, num_polynomials)`. The old
`num_t_vectors`, `num_w_vectors`, and `num_z_vectors` dimensions are gone.
Distributed mode is **not** encoded in the lookup key; it is entirely driven by
[`DistributedProverConfig`](../crates/akita-types/src/lib.rs) embedded in
`PlannerPolicy` (from [`CommitmentConfig::distributed_prover_config()`](../crates/akita-config/src/lib.rs)).
The same `(num_vars, num_polynomials)` key can therefore resolve to different
schedules under different presets or policies — non-distributed and distributed
tables are kept in **separate catalogs** whose identity embeds the config.

## Intent

### Goal

Extend the offline planner (`akita-planner`) and generated-table pipeline so that,
given a [`DistributedProverConfig`](../crates/akita-types/src/lib.rs) with
`num_chunks = W` and `distributed_rounds = R`:

1. **Leading fold rounds** `0 .. R - 1` price witness growth and proof bytes under
   the **chunk-grouped** layout with $W$ replicated $\widehat z$ segments.
2. **Round `R` and beyond** revert to today's **component-major** layout with a
   single $\widehat z$ (switch back to the single-node / CPU tail from the book).
3. **Schedule resolution** remains deterministic: table hit and DP miss produce
   the same [`Schedule`](../crates/akita-types/src/schedule.rs) for the same
   `(key, policy)` pair.
4. **`D = 64` presets** gain companion `_distributed` generated tables whose
   **catalog identity** embeds `DistributedProverConfig`; table **row keys** stay
   `(num_vars, num_polynomials)` like their non-distributed siblings.

**Configuration surface.** Presets declare distributed behavior through
[`CommitmentConfig::distributed_prover_config()`](../crates/akita-config/src/lib.rs).
The existing [`policy_of`](../crates/akita-config/src/lib.rs) bridge copies that
struct into [`PlannerPolicy.distributed`](../crates/akita-planner/src/lib.rs) so
the `Cfg`-free DP and [`resolve_schedule`](../crates/akita-planner/src/resolve.rs)
receive the same inputs. Non-distributed presets use the default
(`num_chunks = 1`, `distributed_rounds = 0`).

Every planner entry point already takes `&PlannerPolicy` alongside the lookup key
(`find_schedule`, `resolve_schedule`, table emission). After this spec lands,
**distributed witness pricing reads only `policy.distributed`**, never extra
fields on `AkitaScheduleLookupKey`.

### Invariants

- **`DistributedProverConfig::disabled()` is byte-identical to today.** With
  `num_chunks == 1` and `distributed_rounds == 0`, `find_schedule` /
  `resolve_schedule` / table expansion must reproduce the current schedule for
  every key the non-distributed tables cover. Protected by extending
  `generated_schedule_tables_match_find_schedule` with paired non-distributed vs
  disabled-config assertions on the **same** lookup keys.
- **Lookup key unchanged.** `AkitaScheduleLookupKey` remains
  `{ num_vars, num_polynomials }` only. No distributed dimensions are added to
  the key or to [`GeneratedScheduleKey`](../crates/akita-planner/src/generated/mod.rs).
- **Policy is the distributed selector.** `find_schedule(key, policy, …)` and
  `resolve_schedule(key, policy, …)` price chunk-grouped layout iff
  `policy.distributed.is_enabled()`. Callers must pass the policy derived from
  the preset they intend to prove under; mismatched preset vs policy is out of
  scope (same as today for `tiered`, `basis_range`, etc.).
- **Block divisibility.** Distributed root candidates require
  `num_blocks % num_chunks == 0` so each node owns an equal block window
  (`blocks_per_chunk = num_blocks / W`). Candidates violating this are skipped in
  the DP, not fixed up later.
- **Power-of-two `num_chunks`.** Initial scope: `num_chunks` is a power of two
  (matching the book's $2^N$ nodes and the verifier chunked fast path in
  [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)). Non-power-of-two
  $W$ returns `InvalidSetup` at plan time.
- **Single source of truth for witness width.** Chunk-grouped ring counts live in
  one new helper in `akita-types`; the planner DP, `schedule_from_entry`, and
  terminal tail sizing all call it. No duplicated closed forms in
  `schedule_params.rs`.
- **Catalog isolation.** Distributed tables are separate modules / catalog
  identities from their non-distributed siblings; a `fp128_d64_onehot` policy must
  never alias a `fp128_d64_onehot_distributed` table even when row keys match.
- **Verifier no-panic on planning path.** Invalid `(DistributedProverConfig,
  num_blocks)` combinations reject with `AkitaError`; the DP does not panic on
  malformed public inputs.
- **Preset is source of truth.** `distributed_prover_config()` on each `Cfg` is
  the only place distributed `(W, R)` constants are authored; `policy_of` and
  generated-table identity derive from it — no hand-written `PlannerPolicy`
  literals for distributed fields.

### Non-Goals

- **Prover implementation** (partial commits, local $\mathbf M_j$, node
  orchestration). The planner only emits parameters the prover will later consume.
- **Verifier row-MLE refactor.** Assumed landed or in flight per
  `distributed-verifier-row-eval.md`; this spec only requires
  `LevelParams::witness_shape` (or equivalent) to be set consistently.
- **Distributed tiered commitments.** `resolve_schedule` rejects
  `tiered && policy.distributed.is_enabled()`; distributed + tiered stays
  unsupported until [`specs/multi-group-batching.md`](multi-group-batching.md)-style
  design exists.
- **Optimizing `distributed_rounds`.** The round count is a **preset constant**
  for v1, not a DP search axis (see Design).
- **Non-`D = 64` generated tables.** Only `ring_dimension = 64` families get
  `_distributed` shipped tables in this phase.
- **ZK schedule tables for distributed.** Non-zk `_distributed` tables first; zk
  is a follow-up mirroring the existing plain/zk split.

## Background: what changes in witness width

### Component-major (today)

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

### Chunk-grouped (distributed)

Per [`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md) and
[the distributed prover book chapter](book/src/how/proving/distributed-prover.md),
layout `ChunkGrouped(W)` concatenates $W$ chunks

```text
[ e^0 | t^0 | z^0 ] ··· [ e^{W-1} | t^{W-1} | z^{W-1} ] | r
```

with:

- `blocks_per_chunk = num_blocks / W`
- $\widehat e^j$, $\widehat t^j$ each cover **only** chunk $j$'s block window
  (still scaled by root `num_polynomials` at level 0)
- $\widehat z^j$ is **full** `inner_width · num_digits_fold` (replicated per chunk,
  *not* divided by $W$)
- $r$ is **shared** (one tail), with row count priced using `num_segments = W`
  in [`m_row_count_for`](../crates/akita-types/src/layout/params.rs) (virtually
  shared horizontal concatenation of $\mathbf M_j$)

Closed form for total ring elements at an intermediate fold (non-zk core):

```text
e_chunk = num_polynomials · blocks_per_chunk · δ_open
t_chunk = num_polynomials · blocks_per_chunk · n_a · δ_open
z_chunk = inner_width · δ_fold                    // full fold width, not / W
body    = e_chunk + t_chunk + z_chunk
rings   = W · body + r_rows · r_decomp_levels
```

**Growth vs today.** The dominant extra cost is $(W - 1) · z_chunk$ ring elements
per distributed level — exactly the proof-size penalty the book accepts for
avoiding cross-node $\widehat z$ all-reduce.

### Cutover to component-major

After **`distributed_rounds`** distributed fold **rounds** (absolute fold levels
`0 .. R - 1`, 0-indexed), the planner switches to component-major sizing
(single $\widehat z$, `num_public_rows = 1`). **Round `R`** (when present) is the
**cutover fold**: its *input* witness is still chunk-grouped; its *output* witness
uses component-major width (nodes coalesce to one logical prover — modeled only
as witness shrink in the planner; prover mechanics are out of scope).

If the optimal schedule has fewer than `R` folds, only the executed prefix uses
chunk-grouped pricing; the remaining configured distributed rounds are a no-op.

## Design

**Terminology.** In prose this spec says **node** for a distributed prover
participant (matching the book's $P_j$). In code and identifiers we say
**chunk** for the same count: witness layout, config fields, and
`ChunkGrouped(W)` all use `num_chunks` / `blocks_per_chunk`, not `num_nodes`.

### New and modified types

#### 1. `DistributedProverConfig` (`akita-types`)

Public configuration for distributed proving, shared by presets, planner, and
(future) prover runtime. Lives in `akita-types` so both `akita-config` and
`akita-planner` can name it without a circular dependency.

```rust
/// How many witness chunks and for how many leading fold rounds the schedule
/// prices ChunkGrouped layout before switching back to ComponentMajor
/// (single-node) sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DistributedProverConfig {
    /// Number of witness chunks / replicated ẑ segments while distributed mode
    /// is active. `1` means single-node (default).
    pub num_chunks: usize,

    /// Count of leading fold **rounds** (absolute fold levels 0, 1, …, R−1)
    /// priced under ChunkGrouped layout. `0` disables distributed planning even
    /// if `num_chunks > 1` (invalid combination — see validation).
    pub distributed_rounds: usize,
}

impl DistributedProverConfig {
    /// Single-node default: no chunking, no extra ẑ replication in the planner.
    pub const fn disabled() -> Self {
        Self { num_chunks: 1, distributed_rounds: 0 }
    }

    pub const fn is_enabled(self) -> bool {
        self.num_chunks > 1 && self.distributed_rounds > 0
    }

    /// Preset helper for the initial D64 distributed tables (book example: 8 nodes).
    pub const fn production_d64() -> Self {
        Self { num_chunks: 8, distributed_rounds: 3 }
    }
}
```

**Validation** (`DistributedProverConfig::validate` or inline in `find_schedule`):

| Rule | Error |
|------|-------|
| `num_chunks == 0` | `InvalidSetup` |
| `num_chunks > 1` and not power of two | `InvalidSetup` |
| `distributed_rounds > 0` and `num_chunks == 1` | `InvalidSetup` |
| `num_chunks > 1` and `distributed_rounds == 0` | `InvalidSetup` (must specify how long to stay distributed) |
| `distributed_rounds > MAX_RECURSION_DEPTH` | `InvalidSetup` |

Include `DistributedProverConfig` in schedule / instance descriptor bytes when
distributed mode is enabled (append-only; disabled config omits or sends
`(1, 0)` deterministically).

#### 2. `CommitmentConfig` hook (`akita-config/src/lib.rs`)

Add a trait method with a **default** that preserves today's behavior for every
existing preset:

```rust
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    // ... existing associated items ...

    /// Distributed-prover parameters for schedule planning and (future) prover
    /// orchestration. Default: single-node, no distributed rounds.
    fn distributed_prover_config() -> DistributedProverConfig {
        DistributedProverConfig::disabled()
    }
}
```

**Distributed preset pattern** (e.g. `fp128::D64OneHotDistributed`):

```rust
impl CommitmentConfig for D64OneHotDistributed {
    // ... same Field / D / decomposition as D64OneHot ...

    fn distributed_prover_config() -> DistributedProverConfig {
        DistributedProverConfig::production_d64()
        // or Self::DISTRIBUTED_CONFIG if stored as a const on the preset
    }
}
```

Non-distributed presets **do not override** the default. The macro-generated
`CommitmentConfig` impls in `proof_optimized.rs` need no change unless a preset
opts into distributed mode.

#### 3. `policy_of` bridge (`akita-config/src/lib.rs`)

Extend the existing bridge — never hand-write distributed literals on
`PlannerPolicy`:

```rust
pub fn policy_of<Cfg: CommitmentConfig>() -> PlannerPolicy {
    PlannerPolicy {
        ring_dimension: Cfg::D,
        // ... existing fields ...
        tiered: Cfg::TIERED_COMMITMENT,
        distributed: Cfg::distributed_prover_config(),  // NEW
    }
}
```

Every path that already calls `policy_of::<Cfg>()` (`runtime_schedule`,
`find_schedule` regen hooks, generated-table emission, drift guards) picks up the
distributed settings automatically.

**Entry guards** in `resolve_schedule` / `find_schedule`:

```rust
let dist = policy.distributed;
if policy.tiered && dist.is_enabled() {
    return Err(AkitaError::InvalidSetup(/* tiered + distributed unsupported */));
}
dist.validate()?;
```

There is **no** lookup-key coupling: `key.validate()` stays the existing
two-field check (`num_vars > 0`, `num_polynomials > 0`).

#### 4. `PlannerPolicy` (`akita-planner/src/lib.rs`)

Add one field (not two loose scalars):

```rust
pub struct PlannerPolicy {
    // ... existing fields ...
    /// Distributed-prover settings derived from CommitmentConfig.
    pub distributed: DistributedProverConfig,
}
```

The planner reads `policy.distributed.num_chunks` and
`policy.distributed.distributed_rounds` everywhere witness layout depends on
distributed mode. Convenience: import `DistributedProverConfig` from `akita-types`
(re-export from `akita-planner` if helpful for emit tests).

**Defaults:** `PlannerPolicy` constructed in tests without `distributed` uses
`DistributedProverConfig::disabled()`.

#### 5. `WitnessType` on `LevelParams` (`akita-types/src/layout/params.rs`)

Add the public layout selector anticipated in
[`distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md):

```rust
pub enum WitnessType {
    ComponentMajor,
    ChunkGrouped(usize), // W = num_chunks
}
```

Each [`LevelParams`](../crates/akita-types/src/layout/params.rs) emitted by the
planner carries `witness_shape: WitnessType` (default `ComponentMajor` for backward
compatibility). Distributed fold steps set
`ChunkGrouped(policy.distributed.num_chunks)`; tail steps set `ComponentMajor`.

Include `witness_shape` in:

- `LevelParams` descriptor / schedule digest bytes (append-only field with
  explicit enum tag so old digests remain reproducible when distributed mode is
  disabled).
- `GeneratedScheduleCatalogIdentity` (embed full `DistributedProverConfig`) so
  catalogs cannot alias across distributed vs non-distributed presets.

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

- `WitnessType::ComponentMajor` → delegate to
  `w_ring_element_count_with_counts_for_layout_bits` with `num_public_rows = 1`.
- `WitnessType::ChunkGrouped(w)` → implement the closed form in **Background**
  with `num_segments = w` for the $r$-tail row count; validate `w > 0`,
  `w.is_power_of_two()`, `lp.num_blocks % w == 0`.

Unit tests in `akita-types` compare against the chunk offset arithmetic from
`distributed-verifier-row-eval.md` Stage 1 (`chunk_stride`, `offset_r`).

#### 7. Schedule lookup key — no distributed fields

[`AkitaScheduleLookupKey`](../crates/akita-types/src/schedule.rs) and
[`GeneratedScheduleKey`](../crates/akita-planner/src/generated/mod.rs) stay:

```rust
pub struct AkitaScheduleLookupKey {
    pub num_vars: usize,
    pub num_polynomials: usize,
}
```

**Do not** add distributed dimensions to the key. Table emission for distributed
families enumerates the **same** `(num_vars, num_polynomials)` pairs as their
non-distributed siblings (via `AkitaScheduleLookupKey::new_from_opening_batch` /
existing family key lists). Distributed vs non-distributed schedules differ because
the **policy** passed to `find_schedule` differs, and because each shipped table
module embeds a distinct catalog identity.

### Planner algorithm changes (step by step)

This is the core review section. Implement in roughly this order.

#### Step 1 — Config and policy plumbing

1. Add `DistributedProverConfig` to `akita-types` (+ re-export).
2. Add `CommitmentConfig::distributed_prover_config()` with default `disabled()`.
3. Extend `policy_of::<Cfg>()` to set `PlannerPolicy.distributed`.
4. Validate `policy.distributed` at `find_schedule` / `resolve_schedule` entry.
5. Thread `PlannerPolicy` (with embedded config) through existing entry points:
   `find_schedule`, `resolve_schedule`, `schedule_from_entry`,
   `GeneratedFoldStep::expand_to_level_params`.

#### Step 2 — Witness width integration

1. Implement `w_ring_element_count_for_witness_type`.
2. Add `witness_shape_for_level(policy, fold_level) -> WitnessType` using
   `let dist = policy.distributed; let R = dist.distributed_rounds; let W = dist.num_chunks`:
   - if `!dist.is_enabled()` → always `ComponentMajor`
   - else if `fold_level < R` → `ChunkGrouped(W)` (output of folds `0 .. R-1`)
   - else if `fold_level == R` → **cutover output** `ComponentMajor`
   - else → `ComponentMajor`
3. Replace direct calls to `w_ring_element_count_with_counts_for_layout_bits`
   in the planner with the layout-aware helper, passing the witness type for the
   **output** witness of the fold being priced.

#### Step 3 — Root DP enumeration (`find_schedule` / `schedule_params.rs`)

At the root-only loop over `(log_basis, r_vars)`:

1. **Skip** candidates with `num_blocks % W != 0` when `dist.is_enabled()`.
2. Compute `next_w_len` / `next_w_len_terminal` via
   `w_ring_element_count_for_witness_type(..., key.num_polynomials, …,
   ChunkGrouped(W))` when `distributed_rounds >= 1`; otherwise unchanged
   (component-major with `num_polynomials = key.num_polynomials`).
3. Set `candidate_params.witness_shape = ChunkGrouped(W)` on emitted root fold
   steps when distributed; root-direct `LevelParams` stays `ComponentMajor`.
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

For each suffix fold at absolute level `L` (with `R = policy.distributed.distributed_rounds`,
`W = policy.distributed.num_chunks`):

1. Determine **input** witness type:
   - `L == 0` → never reached (root handled separately)
   - `L < R` → input `ChunkGrouped(W)`
   - `L == R` → input `ChunkGrouped(W)` (cutover fold consumes chunked witness)
   - `L > R` → input `ComponentMajor`
2. Determine **output** witness type for `next_w_len`:
   - `L < R - 1` → output `ChunkGrouped(W)`
   - `L == R - 1` → output `ComponentMajor` (last distributed round hands off to
     single-node tail)
   - `L >= R` → output `ComponentMajor`
3. Use `num_polynomials = 1` for recursive suffix folds (same as today).
4. Set `candidate_params.witness_shape` from the **input** type at this level
   (what the relation MLE sees).
5. `derive_candidate_level_params` stays unchanged for SIS key geometry; only
   witness-length accounting changes.

#### Step 5 — Terminal direct tail (`terminal_direct_suffix_cost`)

[`terminal_fold_num_polynomials`](../crates/akita-types/src/proof/direct_witness.rs)
returns `key.num_polynomials` at the root terminal predecessor and `1` at recursive
levels. Extend for distributed:

- When the terminal predecessor fold has `witness_shape = ChunkGrouped(W)`,
  terminal tail layout must use **chunk-grouped** segment typing (future
  `SegmentTyped` generalization) **or** upper-bound with the chunked ring count
  converted to field elements.

**v1 planner approach (recommended):** add
`terminal_direct_witness_shape_chunk_grouped(...)` parallel to
`segment_typed_witness_shape`, producing `CleartextWitnessShape::SegmentTyped`
with per-chunk multiplicities encoded in `TailSegmentLayout` (may require adding
`num_chunks: usize` to the layout struct — coordinate with verifier spec Stage 8).

Until that lands, the planner **must still** price the correct byte count using
an upper-bound helper mirroring chunked ring count × `field_bytes` for the
non-`z` segments plus `segment_typed_z_payload_bytes` called with replicated
`z_coords = W · z_unit`.

#### Step 6 — Table expansion path (`resolve.rs` / `schedule_from_entry`)

Mirror the DP witness-type rules when walking compact
[`GeneratedStep`](../crates/akita-planner/src/generated/mod.rs) entries:

1. Track absolute `fold_level` as today.
2. When expanding each fold's `LevelParams`, set `witness_shape` from Step 4.
3. Recompute `next_w_len` with `w_ring_element_count_for_witness_type` instead of
   the singleton helper, using `policy.distributed` to select witness type.
4. Validate at catalog load: distributed tables embed
   `identity.distributed == policy.distributed`.

#### Step 7 — Catalog identity (`catalog_identity.rs`)

Extend [`GeneratedScheduleCatalogIdentity`](../crates/akita-planner/src/generated/mod.rs):

```rust
pub distributed: DistributedProverConfig,
```

Include in `identity_digest` / `validate_catalog_identity`. Regenerated
non-distributed tables set `DistributedProverConfig::disabled()`.

#### Step 8 — Compact generated entries

**No change** to the 7-field [`GeneratedFoldStep`](../crates/akita-planner/src/generated/mod.rs)
tuple for v1: `witness_shape` is derived deterministically from
`(policy.distributed, fold_level)` at expansion time, not stored per step.

#### Step 9 — Generated table emission (`akita-planner/src/emit`, `gen_schedule_tables`)

For each new distributed family:

1. **`module_name`**: base name + `_distributed` (e.g. `fp128_d64_onehot_distributed`).
2. **`schedule_feature`**: e.g. `fp128-d64-onehot-distributed` (new feature flags
   on `akita-schedules` / `akita-config`).
3. **`family_keys`**: **same enumeration** as the non-distributed sibling
   (`num_vars` / `num_polynomials` ranges unchanged). Example:

   ```rust
   let policy = policy_of::<D64OneHotDistributed>();
   // keys: AkitaScheduleLookupKey::new(num_vars, num_polys) — identical to D64OneHot
   ```

4. **`EmitSpec.policy`**: full `PlannerPolicy` including
   `distributed: DistributedProverConfig::production_d64()` (via `policy_of`).
5. Run the same DP regen hook: `find_schedule(key, &policy, …)`.

#### Step 10 — New `Cfg` types and `ALL_GENERATED_FAMILIES` rows (`akita-config`)

Add **D = 64 distributed companions** for the existing non-zk D64 families:

| Base module | Distributed module | Base `Cfg` (pattern) |
|-------------|-------------------|----------------------|
| `fp128_d64_onehot` | `fp128_d64_onehot_distributed` | `fp128::D64OneHotDistributed` |
| `fp128_d64_full` | `fp128_d64_full_distributed` | `fp128::D64FullDistributed` |
| `fp128_d64_onehot_tensor` | `fp128_d64_onehot_tensor_distributed` | `tensor_verifier::fp128::D64OneHotTensorDistributed` |

**Exclude** `fp128_d64_onehot_tiered_distributed` until tiered + multi-$\widehat z$
is designed.

Each distributed `Cfg`:

- `D = 64`, same field / decomposition / one-hot settings as its base.
- `distributed_prover_config()` returns e.g.
  `DistributedProverConfig { num_chunks: 8, distributed_rounds: 3 }` (initial
  production constants — document on the preset).
- `policy_of::<Cfg>()` picks up the config via the trait method (no override).
- `schedule_catalog()` points at the `_distributed.rs` table.
- `runtime_schedule(key)` calls `resolve_schedule(key, &policy_of::<Self>(), …)`;
  no key mutation — distributed behavior comes entirely from policy + catalog.

Wire modules in [`crates/akita-schedules/src/generated/mod.rs`](../crates/akita-schedules/src/generated/mod.rs)
behind the new feature flags.

#### Step 11 — Drift guards (`akita-config/tests/generated_tables.rs`)

1. Add distributed rows to `ALL_GENERATED_FAMILIES` match arms (`family_catalog_is_linked`,
   `assert_family_catalog_enabled`, schedule comparison loop).
2. Extend `generated_schedule_tables_match_find_schedule` to cover distributed
   families: for each key, compare table expansion under `policy_of::<DistributedCfg>()`
   against `find_schedule(key, &policy_of::<DistributedCfg>(), …)`.
3. Add regression: for a fixed key, `find_schedule(key, &policy_of::<D64OneHot>(), …)`
   equals `find_schedule(key, &policy_with_disabled_distributed, …)` where
   `policy_with_disabled_distributed` is `policy_of::<D64OneHot>()` with
   `distributed: disabled()` — confirms distributed fields do not perturb
   non-distributed presets.

#### Step 12 — Regenerate tables

```bash
cargo run --release -p akita-config --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
```

Commit new files:

- `crates/akita-schedules/src/generated/fp128_d64_onehot_distributed.rs`
- `crates/akita-schedules/src/generated/fp128_d64_full_distributed.rs`
- `crates/akita-schedules/src/generated/fp128_d64_onehot_tensor_distributed.rs`

Non-zk only in this spec phase.

### Architecture diagram

```text
   CommitmentConfig::distributed_prover_config()
              │
              ▼
   policy_of::<Cfg>()  ──►  PlannerPolicy.distributed
              │                    │
              │                    │  num_chunks = W
              │                    │  distributed_rounds = R
              ▼                    ▼
     runtime_schedule      find_schedule / resolve_schedule
              │                    │
     AkitaScheduleLookupKey         │
     (num_vars, num_polynomials)   │  policy selects ChunkGrouped vs ComponentMajor
              └──────────┬───────────┘
                         ▼
              ┌─────────────────────────────────────┐
              │  Root DP (level 0)                   │
              │  skip if num_blocks % W != 0         │
              │  ChunkGrouped output when L < R      │
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
 fp128_d64_*_distributed.rs                  DP fallback
 (table hit, identity.distributed set)        (same key + policy)
```

### Alternatives considered

| Alternative | Why not (for v1) |
|-------------|------------------|
| Re-add `num_z_vectors` to the lookup key | Removed on main; distributed cutover rounds are not opening-batch metadata; policy struct is the right home. |
| Encode distributed mode only in table module name, share one `PlannerPolicy` | DP fallback and drift guards would not know which witness model to price without `policy.distributed`. |
| Flat `num_chunks` / `distributed_rounds` on `PlannerPolicy` without struct | User-facing config should be one preset hook; struct keeps planner and prover aligned. |
| Store `witness_shape` in compact generated tuple | Extra bytes per step; derivable from `policy.distributed` + fold level. |
| Search optimal `distributed_rounds` in DP | Explodes search space; book treats cutover as deployment constant; revisit after profiling. |
| Reuse non-distributed tables with runtime scaling | Violates catalog identity; hides witness-layout metadata from verifier digest. |
| Skip generated tables; DP only | Breaks production preset latency; user explicitly requested D64 cached tables. |

## Evaluation

### Acceptance Criteria

- [ ] `DistributedProverConfig::disabled()` / default trait method reproduces
  existing schedules bit-for-bit for all current `ALL_GENERATED_FAMILIES` keys
  when passed through `policy_of` (distributed field ignored for pricing).
- [ ] `w_ring_element_count_for_witness_type(ChunkGrouped(W))` unit tests match
  manual chunk layout arithmetic for `W ∈ {1, 2, 4, 8}` and reject invalid
  `(W, num_blocks)` pairs with `AkitaError`.
- [ ] `find_schedule` with `policy.distributed =
  DistributedProverConfig { num_chunks: 8, distributed_rounds: 3 }` produces
  `LevelParams.witness_shape = ChunkGrouped(8)` on fold rounds `0..2` and
  `ComponentMajor` from round `3` onward for a smoke `num_vars` key.
- [ ] Root DP skips `(log_basis, r_vars)` with `2^r_vars % 8 != 0` when
  `num_chunks = 8`.
- [ ] Three `_distributed` D64 modules emitted; `validate_catalog_identity`
  passes with embedded `distributed == production_d64()`.
- [ ] `generated_schedule_tables_match_find_schedule` passes with
  `--features all-schedules` including distributed families (same keys as siblings,
  different policies).
- [ ] Distributed preset rejects `tiered && distributed.is_enabled()`.
- [ ] `CommitmentConfig::distributed_prover_config()` default unchanged for all
  non-distributed presets (no macro churn).
- [ ] `AkitaScheduleLookupKey` remains two-field; no `num_z_vectors` references
  in planner distributed paths.
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test` pass.

### Testing Strategy

1. **Unit (`akita-types`)** — witness width helper vs explicit chunk stride math;
   cutover level output width `<` purely chunked width for `W > 1`.
2. **Unit (`akita-planner/schedule_params`)** — small `num_vars` brute schedule
   with `policy.distributed.num_chunks = 8` is deterministic; setting
   `distributed: disabled()` on the same policy matches golden non-distributed
   schedule for the same key.
3. **Integration (`akita-config/tests/generated_tables.rs`)** — table hit vs DP
   for each `_distributed` family and key cross-product under `policy_of`.
4. **Negative** — `num_chunks = 6` (not power of two) → `InvalidSetup`; `num_blocks
   = 5` with `W = 8` root candidate skipped (no panic, schedule still found if
   other `r_vars` valid).

### Performance

- **Proof size:** Expect **increase** vs non-distributed schedules at the same
  `num_vars` when `num_chunks > 1`, driven by $(W-1)$ extra $\widehat z$ segments
  per distributed level and longer sum-checks. The planner must surface this in
  `Schedule.total_bytes` so presets can compare `fp128_d64_onehot` vs
  `fp128_d64_onehot_distributed` on identical lookup keys.
- **Table size:** Three new D64 modules with the **same row count** as their
  non-distributed siblings (one entry per `(num_vars, num_polynomials)`); schedules
  differ because emission runs DP with a distributed `PlannerPolicy`.
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
| S0 | `DistributedProverConfig` in `akita-types` + validation | — |
| S1 | `CommitmentConfig::distributed_prover_config()` + `policy_of` wiring | S0 |
| S2 | `WitnessType` + `LevelParams.witness_shape` + descriptor bytes | — |
| S3 | `w_ring_element_count_for_witness_type` + tests | S2 |
| S4 | `PlannerPolicy.distributed` + planner entry validation | S0, S1 |
| S5 | Root DP wiring | S3, S4 |
| S6 | Suffix DP wiring | S3, S4 |
| S7 | `schedule_from_entry` expansion | S5, S6 |
| S8 | Terminal tail byte pricing | S3 |
| S9 | Catalog identity embeds `DistributedProverConfig` | S0, S1 |
| S10 | Distributed `Cfg` types + `_distributed` tables | S1, S5–S9 |

Stages S0–S9 are planner/config/`akita-types`-only and can land before prover/verifier
consume `witness_shape`. S10 ships the cached D64 tables.

## References

- Book: [The distributed prover](../book/src/how/proving/distributed-prover.md)
- Verifier layout spec: [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
- Planner crate map: [`crates/akita-planner/README.md`](../crates/akita-planner/README.md)
- Terminal tail sizing: [`crates/akita-types/src/proof/tail_segments.rs`](../crates/akita-types/src/proof/tail_segments.rs)
- Generated table pipeline: [`specs/schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- Prior art (terminal layout split): [`specs/terminal-fold-cutover.md`](terminal-fold-cutover.md)
