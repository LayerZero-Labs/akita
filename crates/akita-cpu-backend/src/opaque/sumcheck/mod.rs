//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! Consumer-owned digit-range, relation/range-image, and stage implementations
//! live here; recursive witness storage remains in `opaque::recursive`.

pub(crate) mod digit_range;
mod physical_l2_norm;
mod prefix_lookup;
pub(crate) mod relation_range_image;
mod stage3;

// --- Shared helpers ------------------------------------------------------

use std::ops::AddAssign;

use jolt_field::Field;

/// Parallel task count: `per_thread` tasks for each Rayon worker, or one task
/// without `parallel`.
#[inline]
pub(crate) fn parallel_tasks(per_thread: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        per_thread * rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = per_thread;
        1
    }
}

/// Minimum live pair work per parallel Stage-2 task.
pub(crate) const MIN_PARALLEL_ROUND_PAIRS: usize = 2048;

/// Fold indexed work items, staying on the caller below two full task grains.
/// The indexed iterator supplies the item count and owns any disjoint mutable chunks.
#[cfg(feature = "parallel")]
pub(crate) fn par_fold_by_grain<I, T, Identity, Fold, Reduce>(
    items: I,
    pairs_per_item: usize,
    identity: Identity,
    fold: Fold,
    reduce: Reduce,
) -> T
where
    I: rayon::iter::IndexedParallelIterator,
    T: Send,
    Identity: Fn() -> T + Sync + Send,
    Fold: Fn(T, I::Item) -> T + Sync + Send,
    Reduce: Fn(T, T) -> T + Sync + Send,
{
    use rayon::iter::plumbing::{Producer, ProducerCallback};
    use rayon::iter::ParallelIterator;
    struct SerialFold<T, F>(T, F);
    impl<Item, T, F: Fn(T, Item) -> T> ProducerCallback<Item> for SerialFold<T, F> {
        type Output = T;
        fn callback<P: Producer<Item = Item>>(self, producer: P) -> T {
            producer.into_iter().fold(self.0, self.1)
        }
    }
    let per_task = MIN_PARALLEL_ROUND_PAIRS.div_ceil(pairs_per_item.max(1));
    if items.len() < 2 * per_task {
        items.with_producer(SerialFold(identity(), fold))
    } else {
        items
            .with_min_len(per_task)
            .fold(&identity, fold)
            .reduce(&identity, reduce)
    }
}

/// Serial counterpart of the indexed grain fold when parallelism is disabled.
#[cfg(not(feature = "parallel"))]
pub(crate) fn par_fold_by_grain<I, T, Identity, Fold, Reduce>(
    items: I,
    _pairs_per_item: usize,
    identity: Identity,
    fold: Fold,
    _reduce: Reduce,
) -> T
where
    I: ExactSizeIterator,
    Identity: Fn() -> T,
    Fold: Fn(T, I::Item) -> T,
    Reduce: Fn(T, T) -> T,
{
    items.fold(identity(), fold)
}

/// Add `right` into `left` element-wise.
#[inline]
pub(crate) fn add_assign_all<T: Copy + AddAssign>(left: &mut [T], right: &[T]) {
    for (left, &right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

/// Element-wise sum of per-task partial arrays.
pub(crate) fn sum_partials<T: Copy + AddAssign, const N: usize>(
    zero: T,
    partials: impl IntoIterator<Item = [T; N]>,
) -> [T; N] {
    partials.into_iter().fold([zero; N], |mut total, partial| {
        add_assign_all(&mut total, &partial);
        total
    })
}

/// Pairs per parallel work item when a sum-check round has a single outer row.
///
/// Stage 1 x rounds and Stage 2 lane rounds run after every coefficient
/// variable is bound, so their tables are one row and must split along pairs.
/// The tile holds whole split-eq inner blocks: a power-of-two block no larger
/// than the tile divides it, and a block of all live pairs gives one tile.
#[inline]
pub(crate) fn single_row_tile_pairs(block_size: usize) -> usize {
    const SINGLE_ROW_TILE_PAIRS: usize = 1 << 10;
    block_size.max(SINGLE_ROW_TILE_PAIRS)
}

/// Fold adjacent evaluations in a live-prefix row at a challenge `r`, treating
/// indices past the materialized prefix as implicit zero-padding.
#[inline]
pub(crate) fn fold_prefix_pair_with_zero_padding<E: Field>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}
