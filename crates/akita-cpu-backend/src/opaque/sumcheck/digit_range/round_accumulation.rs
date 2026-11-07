use super::exact_prefix::SplitEqualitySuffixMass;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::Unreduced;
use jolt_algebra::{Field, Zero};

/// Sum `N` products over the explicit pairs of a split-equality round.
///
/// This is the only traversal of the split-equality blocks: exact-prefix
/// reduction, canonical or delayed accumulation, outer weighting and the
/// parallel reduction all live here.
///
/// `pair_terms(pair_index, inner_weight)` returns two factor arrays whose
/// lane-wise products are the pair's terms, already scaled by its inner
/// equality weight. Each block of the inner table is summed first, then scaled
/// by its outer weight. Implicit pairs contribute nothing here, so callers use
/// this directly only for terms that vanish on a constant pair; see
/// [`accumulate_equality_weighted_values`] for terms that do not.
pub(super) fn accumulate_equality_weighted_pair_terms<E: Field + Unreduced, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    pair_terms: impl Fn(usize, E) -> ([E; N], [E; N]) + Sync,
) -> [E; N] {
    debug_assert!(explicit_pair_count <= first.len() * second.len());
    let explicit_block_count = explicit_pair_count.div_ceil(first.len());
    if E::SUM_IS_EXACT {
        cfg_fold_reduce!(
            0..explicit_block_count,
            || [E::Product::zero(); N],
            |mut outer, second_index| {
                let block_start = second_index * first.len();
                let block_end = explicit_pair_count.min(block_start + first.len());
                let mut inner = [E::Product::zero(); N];
                for pair_index in block_start..block_end {
                    let (factors, weighted) =
                        pair_terms(pair_index, first[pair_index - block_start]);
                    for ((destination, factor), weighted) in
                        inner.iter_mut().zip(factors).zip(weighted)
                    {
                        *destination += factor.mul_unreduced(weighted);
                    }
                }
                let second_weight = second[second_index];
                for (destination, inner) in outer.iter_mut().zip(inner) {
                    *destination += second_weight.mul_unreduced(E::reduce_product(inner));
                }
                outer
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            }
        )
        .map(E::reduce_product)
    } else {
        cfg_fold_reduce!(
            0..explicit_block_count,
            || [E::zero(); N],
            |mut outer, second_index| {
                let block_start = second_index * first.len();
                let block_end = explicit_pair_count.min(block_start + first.len());
                let mut inner = [E::zero(); N];
                for pair_index in block_start..block_end {
                    let (factors, weighted) =
                        pair_terms(pair_index, first[pair_index - block_start]);
                    for ((destination, factor), weighted) in
                        inner.iter_mut().zip(factors).zip(weighted)
                    {
                        *destination += factor * weighted;
                    }
                }
                let second_weight = second[second_index];
                for (destination, inner) in outer.iter_mut().zip(inner) {
                    *destination += second_weight * inner;
                }
                outer
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            }
        )
    }
}

/// Sum per-pair values over a split-equality round.
///
/// Explicit pairs contribute `values_at(pair_index)` weighted by their
/// equality weight; every implicit pair contributes `default_values`.
///
/// The explicit part is [`accumulate_equality_weighted_pair_terms`] with the
/// pair's values as the first factor and its inner equality weight as the
/// second. That function counts implicit pairs as zero, so the total equality
/// mass of the implicit suffix times `default_values` is added on top here.
pub(super) fn accumulate_equality_weighted_values<E: Field + Unreduced, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    values_at: impl Fn(usize) -> [E; N] + Sync,
    default_values: [E; N],
) -> [E; N] {
    let mut values = accumulate_equality_weighted_pair_terms(
        first,
        second,
        explicit_pair_count,
        |pair_index, weight| (values_at(pair_index), [weight; N]),
    );
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_pair_count))
        .expect("split equality and exact prefix were validated at construction");
    for (value, default) in values.iter_mut().zip(default_values) {
        *value += suffix_weight * default;
    }
    values
}

/// Explicit entries per class below which a block-local histogram would spend
/// more on scaling its classes than on multiplying each entry's weight.
const MIN_BLOCK_ENTRIES_PER_CLASS: usize = 8;

/// Class count above which every task keeping its own weight per class would
/// outgrow the cache; larger class counts are split into class ranges.
const MAX_REPLICATED_CLASS_COUNT: usize = 1 << 12;

/// Sum class-indexed values over a split-equality round.
///
/// Equivalent to [`accumulate_equality_weighted_values`] with
/// `values_at(index) = class_values(class_at(index))`, for values that depend
/// on an entry only through a class below `class_count`. The equality weight is
/// first totalled per class and each class's values are multiplied once, so the
/// traversal costs at most one multiplication per entry instead of `N`. When
/// blocks are long relative to `class_count` the per-entry work is additions
/// only: a block's inner weights are binned by class and the bins are scaled by
/// the block's outer weight. Above [`MAX_REPLICATED_CLASS_COUNT`] classes the
/// work is split by class range instead; see
/// [`accumulate_partitioned_class_values`].
pub(super) fn accumulate_equality_weighted_class_values<E: Field, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_count: usize,
    class_count: usize,
    class_at: impl Fn(usize) -> usize + Sync,
    class_values: impl Fn(usize) -> [E; N] + Sync,
    default_values: [E; N],
) -> [E; N] {
    debug_assert!(explicit_count <= first.len() * second.len());
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_count))
        .expect("split equality and exact prefix were validated at construction");
    let partitions = super::super::parallel_tasks(2);
    if class_count > MAX_REPLICATED_CLASS_COUNT && partitions > 1 {
        if let (Ok(explicit_count), Ok(_)) =
            (u32::try_from(explicit_count), u32::try_from(class_count))
        {
            let mut values = accumulate_partitioned_class_values(
                first,
                second,
                explicit_count,
                class_count,
                partitions,
                &class_at,
                &class_values,
            );
            for (value, default) in values.iter_mut().zip(default_values) {
                *value += suffix_weight * default;
            }
            return values;
        }
    }
    let block_count = explicit_count.div_ceil(first.len());
    let task_count = super::super::parallel_tasks(2).min(block_count).max(1);
    let blocks_per_task = block_count.div_ceil(task_count).max(1);
    let bin_blocks = first.len() >= class_count * MIN_BLOCK_ENTRIES_PER_CLASS;
    let class_weights = cfg_fold_reduce!(
        0..task_count,
        || vec![E::zero(); class_count],
        |mut weights: Vec<E>, task_index| {
            let blocks =
                task_index * blocks_per_task..block_count.min((task_index + 1) * blocks_per_task);
            let mut block_weights = vec![E::zero(); if bin_blocks { class_count } else { 0 }];
            for second_index in blocks {
                let block_start = second_index * first.len();
                let block_end = explicit_count.min(block_start + first.len());
                let second_weight = second[second_index];
                if bin_blocks {
                    block_weights.fill(E::zero());
                    for index in block_start..block_end {
                        block_weights[class_at(index)] += first[index - block_start];
                    }
                    for (weight, block_weight) in weights.iter_mut().zip(&block_weights) {
                        *weight += second_weight * *block_weight;
                    }
                } else {
                    for index in block_start..block_end {
                        weights[class_at(index)] += second_weight * first[index - block_start];
                    }
                }
            }
            weights
        },
        |mut left: Vec<E>, right: Vec<E>| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    );
    let mut values = default_values.map(|default| suffix_weight * default);
    for (class, weight) in class_weights.into_iter().enumerate() {
        if weight.is_zero() {
            continue;
        }
        for (value, class_value) in values.iter_mut().zip(class_values(class)) {
            *value += weight * class_value;
        }
    }
    values
}

/// Coarse class buckets that [`accumulate_partitioned_class_values`] counts
/// to balance its class ranges.
const PARTITION_BUCKETS: usize = 1 << 16;

/// Classes per task of the class-value pass in
/// [`accumulate_partitioned_class_values`].
const CLASS_VALUE_CHUNK: usize = 1 << 12;

/// Entries of task `task` when `explicit_count` entries split into
/// `partitions` contiguous runs of equal length, the last one shorter.
///
/// The endpoints are computed in `u64`: a run's rounded end can pass
/// `u32::MAX` even though every entry index fits `u32`.
fn task_entry_range(explicit_count: u32, partitions: usize, task: usize) -> core::ops::Range<u32> {
    let total = u64::from(explicit_count);
    let per_task = total.div_ceil(partitions.max(1) as u64).max(1);
    let start = total.min((task as u64).saturating_mul(per_task));
    let end = total.min(start.saturating_add(per_task));
    // Both ends are at most `explicit_count`.
    start as u32..end as u32
}

/// Explicit part of [`accumulate_equality_weighted_class_values`] for many
/// classes, split into `partitions` contiguous class ranges.
///
/// Classes can be far from uniform, so the ranges are balanced by entry
/// count: each task counts the entries of one stretch per coarse class bucket,
/// and the range boundaries fall where the running count crosses an equal
/// share. Each task then sorts its entries, as class and index, into the
/// ranges, and each range totals the equality weight of its entries in its
/// own slice of one weight table. Class counts and entry counts must fit
/// `u32`. Finally every nonzero class's values are multiplied once, in
/// equal chunks of classes.
fn accumulate_partitioned_class_values<E: Field, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_count: u32,
    class_count: usize,
    partitions: usize,
    class_at: &(impl Fn(usize) -> usize + Sync),
    class_values: &(impl Fn(usize) -> [E; N] + Sync),
) -> [E; N] {
    let partitions = partitions.max(1);
    let bucket_width = class_count.div_ceil(PARTITION_BUCKETS).max(1);
    let bucket_count = class_count.div_ceil(bucket_width);
    let task_entries = |task: usize| task_entry_range(explicit_count, partitions, task);
    let task_counts: Vec<Vec<u32>> = cfg_into_iter!(0..partitions)
        .map(|task| {
            let mut counts = vec![0u32; bucket_count];
            for index in task_entries(task) {
                counts[class_at(index as usize) / bucket_width] += 1;
            }
            counts
        })
        .collect();
    // `range_ends[p]` is one past the last bucket of class range `p`.
    let mut bucket_ranges = vec![0usize; bucket_count];
    let mut range_ends = vec![bucket_count; partitions];
    let mut running = 0u64;
    let mut range = 0usize;
    for (bucket, partition) in bucket_ranges.iter_mut().enumerate() {
        while range + 1 < partitions
            && running * partitions as u64 >= (range as u64 + 1) * u64::from(explicit_count)
        {
            range_ends[range] = bucket;
            range += 1;
        }
        *partition = range;
        running += task_counts
            .iter()
            .map(|counts| u64::from(counts[bucket]))
            .sum::<u64>();
    }
    let bucket_ranges = &bucket_ranges;
    // Each entry keeps its class, so the weight pass reads its entries in
    // order instead of revisiting the class source at scattered indices.
    let sorted: Vec<Vec<Vec<(u32, u32)>>> = cfg_into_iter!(0..partitions)
        .map(|task| {
            let mut ranges = vec![Vec::new(); partitions];
            for index in task_entries(task) {
                let class = class_at(index as usize);
                ranges[bucket_ranges[class / bucket_width]].push((class as u32, index));
            }
            ranges
        })
        .collect();
    let mut weights = vec![E::zero(); class_count];
    let mut slices = Vec::with_capacity(partitions);
    let mut rest = weights.as_mut_slice();
    let mut first_class = 0usize;
    for &end_bucket in &range_ends {
        let end_class = class_count.min(end_bucket * bucket_width).max(first_class);
        let (slice, tail) = rest.split_at_mut(end_class - first_class);
        slices.push((first_class, slice));
        rest = tail;
        first_class = end_class;
    }
    let sorted = &sorted;
    cfg_into_iter!(slices)
        .enumerate()
        .for_each(|(partition, (first_class, weights))| {
            for &(class, index) in sorted.iter().flat_map(|ranges| &ranges[partition]) {
                let index = index as usize;
                weights[class as usize - first_class] +=
                    second[index / first.len()] * first[index % first.len()];
            }
        });
    cfg_chunks!(weights, CLASS_VALUE_CHUNK)
        .enumerate()
        .map(|(chunk, weights)| {
            let mut values = [E::zero(); N];
            for (offset, &weight) in weights.iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                let class_values = class_values(chunk * CLASS_VALUE_CHUNK + offset);
                for (value, class_value) in values.iter_mut().zip(class_values) {
                    *value += weight * class_value;
                }
            }
            values
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold([E::zero(); N], |mut sum, values| {
            for (sum, value) in sum.iter_mut().zip(values) {
                *sum += value;
            }
            sum
        })
}

#[cfg(test)]
mod tests {
    use super::super::MAX_TREE_STAGE_Q_DEGREE;

    #[test]
    fn task_entry_ranges_cover_every_entry_near_u32_max() {
        for (explicit_count, partitions) in [
            (u32::MAX - 7, 16),
            (u32::MAX, 16),
            (u32::MAX, 7),
            (u32::MAX, 1),
            (5, 16),
            (0, 4),
        ] {
            let mut next = 0u32;
            for task in 0..partitions {
                let range = task_entry_range(explicit_count, partitions, task);
                assert_eq!(range.start, next, "count {explicit_count}, task {task}");
                assert!(range.start <= range.end);
                next = range.end;
            }
            assert_eq!(
                next, explicit_count,
                "count {explicit_count}, {partitions} tasks"
            );
        }
    }
    use super::*;
    use jolt_algebra::{FpExt4, Prime128Offset275, Prime32Offset99, Ring};

    fn check_blocked_accumulation<E: Field + Ring + Unreduced>() {
        // Inner tables below, at and above the block-binning threshold for
        // four classes.
        for first_len in [4, 16, 32, 64] {
            check_blocked_accumulation_with_inner_len::<E>(first_len);
        }
    }

    fn check_blocked_accumulation_with_inner_len<E: Field + Ring + Unreduced>(first_len: usize) {
        // A zero weight and a cancelling pair sit among generic weights.
        let mut first = (0..first_len)
            .map(|index| E::from_u64((index * index + 3 * index + 2) as u64))
            .collect::<Vec<_>>();
        first[1] = E::zero();
        first[3] = E::zero() - first[2];
        let second = [11, 13, 17, 19].map(E::from_u64);
        let rows = (0..first.len() * second.len())
            .map(|row| {
                std::array::from_fn(|coefficient| {
                    E::from_u64((row * 7 + coefficient * 3 + 1) as u64)
                })
            })
            .collect::<Vec<_>>();
        let default = [23, 29, 31, 37, 41].map(E::from_u64);

        for explicit_pair_count in 0..=rows.len() {
            let actual = accumulate_equality_weighted_values(
                &first,
                &second,
                explicit_pair_count,
                |pair_index| rows[pair_index],
                default,
            );
            let mut expected = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
            for pair_index in 0..rows.len() {
                let row = if pair_index < explicit_pair_count {
                    rows[pair_index]
                } else {
                    default
                };
                let weight = first[pair_index % first.len()] * second[pair_index / first.len()];
                for (expected, row) in expected.iter_mut().zip(row) {
                    *expected += weight * row;
                }
            }
            assert_eq!(actual, expected);

            // Many classes split into class ranges balanced by entry count,
            // including empty ranges and one class holding most entries.
            for (class_count, partitions, skewed) in [
                (rows.len(), 1, false),
                (rows.len(), 3, false),
                (rows.len(), 4, true),
                (5, 8, false),
                (5, 8, true),
            ] {
                let class_rows = &rows[..class_count];
                let class_at = |index: usize| {
                    if skewed && index % 4 != 3 {
                        class_count / 2
                    } else {
                        (index * 5 + 2) % class_count
                    }
                };
                let mut partitioned = accumulate_partitioned_class_values(
                    &first,
                    &second,
                    u32::try_from(explicit_pair_count).unwrap(),
                    class_count,
                    partitions,
                    &class_at,
                    &|class| class_rows[class],
                );
                let suffix_weight = SplitEqualitySuffixMass::new(&first, &second)
                    .and_then(|suffix| suffix.weight_from(explicit_pair_count))
                    .unwrap();
                for (value, default) in partitioned.iter_mut().zip(default) {
                    *value += suffix_weight * default;
                }
                assert_eq!(
                    partitioned,
                    accumulate_equality_weighted_values(
                        &first,
                        &second,
                        explicit_pair_count,
                        |index| class_rows[class_at(index)],
                        default,
                    )
                );
            }

            // Few classes bin each block; many classes weight each entry.
            for class_count in [1, 3, 4, rows.len()] {
                let class_rows = &rows[..class_count];
                let class_at = |index: usize| (index * 5 + 2) % class_count;
                assert_eq!(
                    accumulate_equality_weighted_class_values(
                        &first,
                        &second,
                        explicit_pair_count,
                        class_count,
                        class_at,
                        |class| class_rows[class],
                        default,
                    ),
                    accumulate_equality_weighted_values(
                        &first,
                        &second,
                        explicit_pair_count,
                        |index| class_rows[class_at(index)],
                        default,
                    )
                );
            }
        }
    }

    #[test]
    fn blocked_accumulation_matches_dense_for_canonical_fallback() {
        check_blocked_accumulation::<Prime128Offset275>();
    }

    #[test]
    fn blocked_accumulation_matches_dense_with_delayed_reduction() {
        type E = FpExt4<Prime32Offset99>;
        check_blocked_accumulation::<E>();
    }
}
