# One-Hot Kernel Entry Unification

## Summary

The one-hot prover backend currently has two physical entry layouts:

- `SingleChunkEntry`: one stored ring element has exactly one hot coefficient.
- `MultiChunkEntry`: one stored ring element may have several hot coefficients.

Those layouts should stay separate. `SingleChunkEntry` is compact and allocation-free, while
`MultiChunkEntry` carries a coefficient list. Forcing the single-chunk case into the
multi-chunk representation, for example by using a `Vec<u16>` of length one, would add
allocation, pointer chasing, and larger records on a hot path.

The duplicated kernel logic should be unified instead. The proposed design is to introduce one
generic trait, implemented by both entry types, that exposes the common operation needed by the
fold, decompose-fold, inner Ajtai, and column-sweep kernels:

```rust
pub(super) trait OneHotEntry: Sync {
    fn pos_in_block(&self) -> usize;
    fn coeffs(&self) -> &[u16];

    #[inline(always)]
    fn commit_col(&self, num_digits: usize) -> usize {
        self.pos_in_block() * num_digits
    }
}
```

Kernels should be generic over `E: OneHotEntry`, not over trait objects. This preserves
monomorphized code for `SingleChunkEntry`, while letting every kernel use the same simple
`for &ci in entry.coeffs()` loop.

## Current Issue

The current implementation repeats the same algorithmic shape across single-chunk and multi-chunk
paths.

Examples:

- `fold_single_chunk_onehot_block` and `fold_multi_chunk_onehot_block` differ only in whether an
  entry exposes one coefficient or a coefficient slice.
- `fold_single_chunk_onehot_block_ring` and `fold_multi_chunk_onehot_block_ring` have the same
  difference for ring-valued scalars.
- `inner_ajtai_wide_single_chunk` and `inner_ajtai_wide_multi_chunk` both compute sparse Ajtai
  contributions from hot coefficients.
- `column_sweep_ajtai_single_chunk` and `column_sweep_ajtai_multi_chunk` both feed the same
  column-sweep core, but currently use separate wrapper logic.
- `single_chunk_onehot_accumulate` and `multi_chunk_onehot_accumulate` both accumulate rotated
  sparse challenges into coefficient rows.
- Tensor decompose-fold has the same single/multi split again with `i64` accumulators.

This duplication makes optimization work error-prone. Any change to tiling, accumulator safety,
parallel partitioning, tracing, or zero-skipping must be replicated and reviewed in multiple
places.

The important distinction is not the algorithm. It is only how an entry exposes its hot
coefficients:

```text
single chunk: pos_in_block + one coeff_idx
multi chunk:  pos_in_block + many coeff_idx values
```

That distinction is small enough to be hidden behind a generic entry trait.

## Non-Goals

This refactor should not merge the physical storage types.

In particular, the following are not recommended:

- Replacing `SingleChunkEntry { pos_in_block, coeff_idx }` with `MultiChunkEntry { pos_in_block,
  nonzero_coeffs: Vec<u16> }`.
- Using a heap-backed list for the single-chunk path.
- Introducing dynamic dispatch in inner kernels.
- Adding compatibility shims around the old helper functions after the generic helpers are in
  place.

The desired result is:

```text
same specialized storage
same enum-level dispatch
shared generic kernel bodies
```

## Proposed Structure

The entry abstraction should be centralized in
`crates/akita-prover/src/backend/onehot/entries.rs`.

That file should be the single owner of entry-level semantics:

- `OneHotIndex`
- `SingleChunkEntry`
- `MultiChunkEntry`
- the new `OneHotEntry` trait
- `OneHotEntry` implementations for both entry types
- small entry-level helper/default logic such as `commit_col` and, if useful,
  `shift_accumulation_count`

Other one-hot modules should not define their own entry traits, layout traits, coefficient-list
adapters, or `push_entries`-style entry abstraction closures. They should import `OneHotEntry`
from `entries.rs` and write generic kernels over `E: OneHotEntry`.

The kernel algorithms still belong in their current files:

- `fold.rs` owns fold kernels.
- `inner_ajtai.rs` owns per-block inner Ajtai kernels.
- `column_sweep.rs` owns column-sweep kernels.
- `accumulate.rs` owns sparse-challenge accumulation kernels.
- `decompose_fold.rs` owns decompose-fold orchestration.
- `ops.rs` and `compute.rs` keep enum dispatch into monomorphized generic helpers.

The structural goal is:

```text
entries.rs:      entry storage + entry semantics
kernel modules:  generic algorithms consuming OneHotEntry
ops/compute:     enum dispatch into concrete generic instantiations
```

As part of the refactor, remove entry-specific logic from other files when it becomes redundant:

- no local single/multi coefficient adapters in `fold.rs`
- no single/multi `push_entries` closures in `column_sweep.rs`
- no repeated `coeff_idx()` vs `nonzero_coeffs()` loops in kernel modules
- no separate per-module notion of "entry layout"

This keeps `entries.rs` as the single place a reader needs to understand what a one-hot entry
means, while preserving the existing module boundaries for actual computation.

Because this repo has no backward-compatibility constraint, the refactor should also remove the
current inherent accessors `pos_in_block`, `coeff_idx`, and `nonzero_coeffs` once all call sites
have migrated to the trait. Keeping same-named inherent methods beside trait methods makes
concrete and generic call sites resolve differently. Before deletion, grep the full workspace,
including tests and helper modules, because the entry types are re-exported from `mod.rs`.

`mod.rs` should re-export the trait for the sibling modules, for example with
`pub(super) use entries::OneHotEntry;`, so existing `use super::*;` imports carry the trait into
the kernel modules.

## Proposed Trait

The trait belongs in `crates/akita-prover/src/backend/onehot/entries.rs`, next to the concrete
entry definitions, constructors, and private field layout.

The trait name should be:

```rust
pub(super) trait OneHotEntry: Sync {
    fn pos_in_block(&self) -> usize;

    fn coeffs(&self) -> &[u16];

    #[inline(always)]
    fn commit_col(&self, num_digits: usize) -> usize {
        self.pos_in_block() * num_digits
    }
}
```

`OneHotEntry` is intentionally broader than an accessor trait. It describes the entry as seen by
the one-hot kernels. Today entry type and layout semantics are one-to-one:

- `SingleChunkEntry` means singleton coefficient iteration.
- `MultiChunkEntry` means list coefficient iteration.

Because of that one-to-one relationship, a single trait is simpler than splitting the design into
one entry trait plus one layout trait.

### `pos_in_block`

Returns the ring-element position inside the current block.

For `SingleChunkEntry`, this is the existing packed `pos_in_block: u32`.

For `MultiChunkEntry`, this is the existing packed `pos_in_block: u32`.

Current users:

- `fold.rs`: chooses `scalars[pos]`.
- `inner_ajtai.rs`: derives the A-matrix column.
- `column_sweep.rs`: derives the A-matrix column before bucketing by column.
- `accumulate.rs`: partitions entries by position and chooses the accumulator row.

Implementation sketch:

```rust
impl OneHotEntry for SingleChunkEntry {
    #[inline(always)]
    fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    // ...
}
```

`SingleChunkEntry` currently exposes `pos_in_block(self) -> usize`, while `MultiChunkEntry`
exposes `pos_in_block(&self) -> usize`. The trait should standardize on `&self`. Once all call
sites are migrated, remove the inherent `pos_in_block` accessors so concrete and generic code do
not silently resolve different methods with the same name.

### `coeffs`

Returns the hot coefficient indices inside this stored ring element.

This is the main unifying method. It should be a borrowed slice accessor, not a visitor closure.
`SingleChunkEntry` can return an allocation-free length-one slice using `std::slice::from_ref`,
while `MultiChunkEntry` can return its stored coefficient vector as a slice:

```rust
impl OneHotEntry for SingleChunkEntry {
    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        std::slice::from_ref(&self.coeff_idx)
    }
}

impl OneHotEntry for MultiChunkEntry {
    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}
```

This slice form is preferred over `for_each_coeff` because it:

- collapses coefficient iteration and coefficient counting into one method;
- removes closure-inlining concerns from the hot kernels;
- matches the existing multi-chunk loop shape, `for &ci in entry.coeffs()`;
- lets column sweep push `u16` coefficient indices directly without re-narrowing;
- remains allocation-free for the single-chunk case.

The tradeoff is that the single-chunk path becomes a loop over a length-one slice instead of a
statically straight-line visitor call. Benchmark the single-chunk fold and accumulate paths after
implementation. If that specific loop causes a measurable regression, revisit a visitor method or
a specialized singleton helper then, not before.

Current users:

- `fold.rs`: add a scalar into each hot coefficient slot.
- `fold.rs`: shift-accumulate a ring scalar for each hot coefficient.
- `inner_ajtai.rs`: shift-accumulate an A-column for each hot coefficient.
- `column_sweep.rs`: emit one `(col, local_block, coeff_idx)` tuple per hot coefficient.
- `accumulate.rs`: add each rotated sparse challenge into the destination row.
- Tensor accumulation: same as `accumulate.rs`, but using `i64`.
- `column_sweep.rs` and `inner_ajtai.rs`: count `entry.coeffs().len()` to enforce wide-accumulator
  safety.

Today single-chunk checks `entries.len() > MAX_WIDE_SHIFT_ACCUMULATIONS`, while multi-chunk sums
`nonzero_coeffs().len()`. After the trait, the helper can be shared in `entries.rs`:

```rust
#[inline]
fn shift_accumulation_count<E: OneHotEntry>(entries: &[E]) -> usize {
    entries.iter().map(|entry| entry.coeffs().len()).sum()
}
```

For `SingleChunkEntry`, this should optimize to the entry count. If that does not happen in
benchmarks, keep a tiny specialized helper for the singleton case, but start with the generic
form.

### `commit_col`

Returns the A-matrix column used for the inner Ajtai commitment:

```rust
self.pos_in_block() * num_digits
```

Both current layouts agree on this formula, so it should be a default method.

Current users:

- `inner_ajtai.rs`
- `column_sweep.rs`

This method exists mostly to make commitment kernels read in entry-level terms rather than
repeating the column formula everywhere.

## Decompose-Fold Accumulation Shape

The current code treats single and multi differently:

- Single-chunk accumulates `block_len` rows, then expands digit-zero rows when `num_digits > 1`.
- Multi-chunk accumulates `block_len * num_digits` rows, but writes only to `pos * num_digits`.

For one-hot entries, the nonzero digit column is still digit zero. The multi-chunk entry can carry
several coefficients inside the same ring element, but it does not imply nonzero higher gadget
digits. Therefore the generic decompose-fold path can accumulate both layouts into a compressed
`block_len` vector first, then expand once:

```text
compressed accumulator length: block_len
expanded witness length:      block_len * num_digits
```

This preserves the current single-chunk optimization and should improve the multi-chunk path by
avoiding direct work over the larger expanded width. This is not only a test-derived claim: the
current multi-chunk accumulator writes `local_pos = entry.pos_in_block() * num_digits - pos_start`
and never adds a nonzero digit offset, so rows for digits `1..num_digits` stay zero by
construction.

The generic compressed accumulator should partition by raw `pos_in_block()` rather than
`pos_in_block() * num_digits`. That removes a multiply from the `partition_point` comparator and
shrinks the accumulator allocation by a factor of `num_digits`.

Use the single-chunk thread-count shape for the unified accumulator:

```rust
let actual_threads = num_threads.min(block_len).max(1);
```

Do not copy the current multi-chunk expression based on `inner_width`; the unified compressed
accumulator is partitioning `block_len` positions.

The expansion helper can be shared:

```rust
fn expand_onehot_accum<const D: usize>(
    compressed: Vec<[i32; D]>,
    num_digits: usize,
) -> Vec<[i32; D]> {
    if num_digits == 1 {
        return compressed;
    }

    let mut expanded = Vec::with_capacity(compressed.len().saturating_mul(num_digits));
    for coeffs in compressed {
        expanded.push(coeffs);
        for _ in 1..num_digits {
            expanded.push([0i32; D]);
        }
    }
    expanded
}
```

Add a dedicated unit test for this change: for a fixed multi-chunk witness, compare the old
expanded-width accumulation result against the new compressed-then-expanded result before deleting
the old path.

## Expected Code Shape After Refactor

### Entry Definitions

`entries.rs` keeps both storage types.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleChunkEntry {
    pos_in_block: u32,
    coeff_idx: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiChunkEntry {
    pos_in_block: u32,
    nonzero_coeffs: Vec<u16>,
}
```

Add trait implementations:

```rust
impl OneHotEntry for SingleChunkEntry {
    #[inline(always)]
    fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        std::slice::from_ref(&self.coeff_idx)
    }
}

impl OneHotEntry for MultiChunkEntry {
    #[inline(always)]
    fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}
```

### Folding

`fold.rs` can replace four functions with two generic functions.

Field-scalar fold:

```rust
pub(super) fn fold_onehot_block<E, F, const D: usize>(
    entries: &[E],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    E: OneHotEntry,
    F: FieldCore,
{
    let mut coeffs_acc = [F::zero(); D];

    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            let s = scalars[pos];
            for &ci in entry.coeffs() {
                coeffs_acc[ci as usize] += s;
            }
        }
    }

    CyclotomicRing::from_coefficients(coeffs_acc)
}
```

Ring-scalar fold:

```rust
pub(super) fn fold_onehot_block_ring<E, F, const D: usize>(
    entries: &[E],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D>
where
    E: OneHotEntry,
    F: FieldCore,
{
    let mut acc = CyclotomicRing::<F, D>::zero();

    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            for &ci in entry.coeffs() {
                scalars[pos].shift_accumulate_into(&mut acc, ci as usize);
            }
        }
    }

    acc
}
```

`ops.rs` still matches on `OneHotBlocks`, but both branches call the same generic helper:

```rust
match blocks {
    OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_blocks)
        .map(|i| fold_onehot_block(flat.block(i), scalars, block_len))
        .collect(),
    OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_blocks)
        .map(|i| fold_onehot_block(flat.block(i), scalars, block_len))
        .collect(),
}
```

### Inner Ajtai

`inner_ajtai.rs` can use one generic implementation for normal blocks.

```rust
pub(crate) fn inner_ajtai_wide_onehot<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    entries: &[E],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in entries {
        let col = entry.commit_col(num_digits);
        for (a_row, t_w) in a_view.rows().zip(t_wide.iter_mut()) {
            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
            for &ci in entry.coeffs() {
                a_wide.shift_accumulate_into(t_w, ci as usize);
            }
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}
```

The safe tiled variant can also be generic. The current multi-chunk implementation can split a
coefficient slice in the middle of an entry. The unified version does not need that complexity:
flush at entry boundaries instead. For the currently supported ring dimensions, one entry's
coefficient count is bounded by `D`, far below `MAX_WIDE_SHIFT_ACCUMULATIONS = 1 << 15`. The
generic tiled kernel can therefore flush before processing an entry if
`shift_accumulations + entry.coeffs().len() > MAX_WIDE_SHIFT_ACCUMULATIONS`, then process the
whole entry.

The single-chunk fast path can still call the generic untiled implementation when the block is
known safe. It should monomorphize into the same shape as the current singleton loop.

### Column Sweep

`column_sweep_core` is already generic over an entry type, but it receives a layout-specific
`push_entries` closure. With `OneHotEntry`, it can push entries directly.

Current shape:

```rust
column_sweep_core::<SingleChunkEntry, F, D>(
    a_view,
    single_chunk_blocks,
    n_a,
    num_digits_commit,
    |block_entries, local_b, num_digits, sink| {
        for entry in block_entries {
            let col = entry.pos_in_block() * num_digits;
            sink.push((col, local_b, entry.coeff_idx() as u16));
        }
    },
)
```

Expected shape:

```rust
fn column_sweep_core<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    // ...
    for entry in block_entries {
        let col = entry.commit_col(num_digits_commit);
        for &ci in entry.coeffs() {
            sink.push((col, local_b, ci));
        }
    }
    // ...
}
```

Then the wrappers collapse into one generic wrapper:

```rust
pub(crate) fn column_sweep_ajtai_onehot<E, F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    blocks: &[&[E]],
    n_a: usize,
    active_a_cols: usize,
    num_digits_commit: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    E: OneHotEntry,
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    debug_assert!(
        active_a_cols <= a_view.num_cols(),
        "active A width exceeds setup envelope"
    );

    // 1. If any block would exceed MAX_WIDE_SHIFT_ACCUMULATIONS:
    //    use the generic tiled per-block kernel.
    // 2. Else if blocks_per_thread <= SWEEP_THRESHOLD:
    //    use the generic untiled per-block kernel.
    // 3. Else:
    //    use column_sweep_core::<E, F, D>(...).
}
```

The ordering is a correctness invariant. The untiled per-block kernel is only reachable after the
wide-accumulator overflow guard proves every block is safe. This matches the current single-chunk
ordering and also improves the no-overflow multi-chunk small-block case, which can avoid the
flushing overhead of the currently tiled multi-chunk inner kernel.

`active_a_cols` survives only as a debug invariant for the setup envelope check; the release-path
work is driven by `a_view` and `num_digits_commit`.

`compute.rs` still performs type dispatch:

```rust
match plan.blocks {
    OneHotCommitBlocks::SingleChunk(blocks) => column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
        &a_view,
        &blocks.block_slices()?,
        plan.n_a,
        active_a_cols,
        plan.num_digits_commit,
    ),
    OneHotCommitBlocks::MultiChunk(blocks) => column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
        &a_view,
        &blocks.block_slices()?,
        plan.n_a,
        active_a_cols,
        plan.num_digits_commit,
    ),
}
```

This match is not harmful duplication. It is the necessary place where the enum chooses the
concrete monomorphized function.

### Sparse Challenge Accumulation

`accumulate.rs` can replace single/multi accumulation with generic compressed accumulation over
`block_len`.

```rust
pub(super) fn onehot_accumulate<E, const D: usize>(
    blocks: &[&[E]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[i32; D]>
where
    E: OneHotEntry,
{
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    for &ci in entry.coeffs() {
                        let rot = &rotated[ci as usize];
                        for k in 0..D {
                            dst[k] += rot[k] as i32;
                        }
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}
```

Tensor accumulation is analogous:

```rust
pub(super) fn onehot_accumulate_tensor<E, const D: usize>(
    blocks: &[&[E]],
    tensor: &TensorChallengeSet,
    expected_blocks: usize,
    block_len: usize,
) -> Result<Vec<[i64; D]>, AkitaError>
where
    E: OneHotEntry,
{
    // same partitioning by pos_in_block
    // fill_rotated_tensor_challenge
    // for &ci in entry.coeffs() { ... }
}
```

### Decompose-Fold

`decompose_fold.rs` can replace the single and multi helpers with one generic helper:

```rust
fn decompose_fold_onehot<E>(
    blocks: &FlatBlocks<E>,
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
) -> DecomposeFoldWitness<F, D>
where
    E: OneHotEntry,
    F: CanonicalField,
{
    let num_blocks = challenges.len().min(blocks.num_blocks());
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let block_views: Vec<&[E]> = (0..blocks.num_blocks())
        .map(|i| blocks.block(i))
        .collect();

    let compressed = onehot_accumulate::<E, D>(
        &block_views,
        challenges,
        num_blocks,
        block_len,
    );

    let coeff_accum = expand_onehot_accum(compressed, num_digits);
    build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
}
```

Batched decompose-fold similarly builds a `Vec<&[E]>` and calls the same generic accumulator.

Top-level enum dispatch remains:

```rust
match blocks {
    OneHotBlocks::SingleChunk(blocks) => {
        self.decompose_fold_onehot::<SingleChunkEntry>(blocks, challenges, block_len, num_digits)
    }
    OneHotBlocks::MultiChunk(blocks) => {
        self.decompose_fold_onehot::<MultiChunkEntry>(blocks, challenges, block_len, num_digits)
    }
}
```

## What Stays Split

These pieces should stay split:

- `SingleChunkEntry`
- `MultiChunkEntry`
- `FlatBlocks<SingleChunkEntry>::from_indices`
- `FlatBlocks<MultiChunkEntry>::from_indices`
- `OneHotBlocks`
- `OneHotCommitBlocks`
- The small enum matches in `ops.rs` and `compute.rs`

The builders encode different layout materialization rules. The enums preserve the concrete entry
type so the kernels can be monomorphized. The refactor should remove duplicated algorithm bodies,
not erase useful type information.

## Performance Expectations

Single-chunk should not regress because:

- Storage remains `Copy`, compact, and allocation-free.
- Kernels remain generic and monomorphized.
- `SingleChunkEntry::coeffs` returns an allocation-free length-one slice with `std::slice::from_ref`.
- `entry.coeffs().len()` replaces separate coefficient-count logic.
- Decompose-fold continues to accumulate only `block_len` compressed rows before digit expansion.

Multi-chunk should improve because:

- Decompose-fold can avoid direct accumulation over `block_len * num_digits`.
- Partitioning by raw `pos_in_block()` removes the current multiply in the `partition_point`
  comparator.
- The same optimized tiling and accumulator safety logic will be shared with single-chunk.
- Future changes to column sweep or sparse challenge accumulation will apply once.

Risks:

- The length-one slice loop in the single-chunk path should be checked in benchmarks for the
  tightest fold and accumulate paths.
- The generic shift-accumulation count via `entry.coeffs().len()` should be checked in benchmarks
  for the singleton case.
- The compressed multi-chunk decompose-fold path must be validated against existing tests before
  deleting the expanded-width path.

A macro-based template could likely force nearly identical generated code, but it is worse for
readability, debugging, and future maintenance. Monomorphized generics should be the default
architecture unless profiling proves otherwise.

## Implementation Plan

1. Add `OneHotEntry` to `entries.rs`.
2. Implement it for `SingleChunkEntry` and `MultiChunkEntry`.
3. Re-export it from `mod.rs` for sibling modules, e.g. `pub(super) use entries::OneHotEntry;`.
4. Refactor `fold.rs` to generic fold helpers.
5. Refactor `inner_ajtai.rs` to a generic inner Ajtai helper and a generic safe/tiled helper.
6. Refactor `column_sweep.rs` to remove the `push_entries` closure and use the trait directly.
7. Refactor `accumulate.rs` to generic compressed digit-zero accumulation.
8. Refactor tensor accumulation using the same generic pattern.
9. Refactor `decompose_fold.rs` around generic helpers plus shared digit expansion.
10. Keep enum dispatch in `ops.rs` and `compute.rs`, but route each branch to generic helpers.
11. Update tests, test helpers, and `mod.rs` test-only re-exports when helper function names change.
12. Remove obsolete inherent accessors and single/multi duplicate helper names once all call sites
    are migrated.

## Test Plan

Run focused one-hot tests first:

```bash
cargo test -p akita-prover onehot
```

Then run broader prover and PCS tests:

```bash
cargo test -p akita-prover
cargo test -p akita-pcs
```

Add one focused unit test before deleting the old expanded-width multi-chunk decompose-fold path:
on a fixed multi-chunk witness, assert that compressed-then-expanded accumulation matches the old
expanded-width output.

For performance-sensitive validation, capture a baseline before the refactor and compare after on
both:

- a single-chunk mode where `K >= D`, for example `AKITA_MODE=onehot_fp128_d64`;
- a multi-chunk mode where `K < D`, chosen to exercise the multi-entry coefficient-list path.

The most important invariant is that single-chunk does not gain allocation or dynamic dispatch in
the hot kernels.
