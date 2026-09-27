use super::exact_prefix::SplitEqualitySuffixMass;
use super::MAX_TREE_STAGE_Q_DEGREE;
use jolt_field::solinas::parallel::*;
use jolt_field::Unreduced;
use jolt_field::{Field, Zero};

#[cfg(feature = "parallel")]
// Above this prefix size, the separate fold no longer stays in its sequential
// region, so combining the fold and following round can amortize parallel work.
const FUSED_MIN_EXPLICIT_ROWS: usize = 1 << 13;

#[inline]
fn accumulate_canonical_blocks<E: Field>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: &(impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync),
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let explicit_block_count = explicit_pair_count.div_ceil(first.len());
    cfg_fold_reduce!(
        0..explicit_block_count,
        || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
        |mut outer, second_index| {
            let block_start = second_index * first.len();
            let block_end = explicit_pair_count.min(block_start + first.len());
            let mut inner = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
            for pair_index in block_start..block_end {
                let source = coefficients_at(pair_index);
                let first_weight = first[pair_index - block_start];
                for (destination, source) in inner.iter_mut().zip(source) {
                    *destination += first_weight * source;
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

#[inline(always)]
pub(super) fn add_scaled_round_coefficients<E: Field>(
    destination: &mut [E; MAX_TREE_STAGE_Q_DEGREE + 1],
    source: &[E; MAX_TREE_STAGE_Q_DEGREE + 1],
    scale: E,
) {
    for (destination, &source) in destination.iter_mut().zip(source.iter()) {
        *destination += scale * source;
    }
}

fn accumulate_delayed_blocks<E: Field + Unreduced>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: &(impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync),
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let explicit_block_count = explicit_pair_count.div_ceil(first.len());
    cfg_fold_reduce!(
        0..explicit_block_count,
        || [E::Product::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
        |mut outer, second_index| {
            let block_start = second_index * first.len();
            let block_end = explicit_pair_count.min(block_start + first.len());
            let mut inner = [E::Product::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
            for pair_index in block_start..block_end {
                let source = coefficients_at(pair_index);
                let first_weight = first[pair_index - block_start];
                for (destination, source) in inner.iter_mut().zip(source) {
                    *destination += first_weight.mul_unreduced(source);
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
}

/// Fold an exact-prefix table and accumulate the following equality-weighted
/// round from the folded pairs before they are read back from memory.
///
/// `fold` computes one table entry from an adjacent source pair. `coefficients`
/// computes the next round's polynomial coefficients from an adjacent pair of
/// folded entries. The explicit prefix, its implicit default suffix, and the
/// equality split all follow the same layout as
/// [`accumulate_equality_weighted_round`].
pub(super) fn fold_and_accumulate_equality_weighted_round<E, T>(
    table: &mut super::exact_prefix::ExactPrefixTable<T>,
    scratch: &mut Vec<T>,
    first: &[E],
    second: &[E],
    fold: impl Fn(T, T) -> T + Sync,
    coefficients: impl Fn(T, T) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync,
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1]
where
    E: Field + Unreduced,
    T: Copy + Send + Sync,
{
    let domain_len = table.domain_len();
    debug_assert!(domain_len >= 4);
    debug_assert!(first.len().is_power_of_two());
    debug_assert!(second.len().is_power_of_two());
    debug_assert_eq!(domain_len / 4, first.len() * second.len());

    #[cfg(feature = "parallel")]
    if table.explicit_len() > FUSED_MIN_EXPLICIT_ROWS {
        let (_, source, source_default) = table.parts();

        let output_len = source.len().div_ceil(2);
        let next_default = fold(source_default, source_default);
        if scratch.len() < output_len {
            scratch.resize(output_len, next_default);
        } else {
            scratch.truncate(output_len);
        }

        // Each task owns complete rows of the split equality table. This keeps
        // output writes disjoint and gives a small tail at least 1024 pair terms
        // per task without changing equality-block reduction boundaries.
        let rows_per_task = 1024usize.div_ceil(first.len());
        let pairs_per_task = rows_per_task * first.len();
        let output_chunk_len = 2 * pairs_per_task;

        let mut round_coefficients = if E::SUM_IS_EXACT {
            let partials = cfg_fold_reduce!(
                cfg_chunks_mut!(scratch, output_chunk_len).enumerate(),
                || [E::Product::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
                |mut outer, (task, output)| {
                    let pair_start = task * pairs_per_task;
                    let pair_end =
                        (pair_start + output.len().div_ceil(2)).min(output_len.div_ceil(2));
                    let mut pair_index = pair_start;
                    while pair_index < pair_end {
                        let second_index = pair_index / first.len();
                        let block_start = second_index * first.len();
                        let block_end = pair_end.min(block_start + first.len());
                        let mut inner = [E::Product::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                        for pair in pair_index..block_end {
                            let left_source = 4 * pair;
                            let left = fold(
                                source.get(left_source).copied().unwrap_or(source_default),
                                source
                                    .get(left_source + 1)
                                    .copied()
                                    .unwrap_or(source_default),
                            );
                            let local_left = 2 * (pair - pair_start);
                            output[local_left] = left;
                            let right = if local_left + 1 < output.len() {
                                let right_source = left_source + 2;
                                let right = fold(
                                    source.get(right_source).copied().unwrap_or(source_default),
                                    source
                                        .get(right_source + 1)
                                        .copied()
                                        .unwrap_or(source_default),
                                );
                                output[local_left + 1] = right;
                                right
                            } else {
                                next_default
                            };
                            let pair_coefficients = coefficients(left, right);
                            let first_weight = first[pair - block_start];
                            for (sum, coefficient) in inner.iter_mut().zip(pair_coefficients) {
                                *sum += first_weight.mul_unreduced(coefficient);
                            }
                        }

                        let second_weight = second[second_index];
                        for (sum, inner) in outer.iter_mut().zip(inner) {
                            *sum += second_weight.mul_unreduced(E::reduce_product(inner));
                        }
                        pair_index = block_end;
                    }
                    outer
                },
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                }
            );
            partials.map(E::reduce_product)
        } else {
            cfg_fold_reduce!(
                cfg_chunks_mut!(scratch, output_chunk_len).enumerate(),
                || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
                |mut outer, (task, output)| {
                    let pair_start = task * pairs_per_task;
                    let pair_end =
                        (pair_start + output.len().div_ceil(2)).min(output_len.div_ceil(2));
                    let mut pair_index = pair_start;
                    while pair_index < pair_end {
                        let second_index = pair_index / first.len();
                        let block_start = second_index * first.len();
                        let block_end = pair_end.min(block_start + first.len());
                        let mut inner = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
                        for pair in pair_index..block_end {
                            let left_source = 4 * pair;
                            let left = fold(
                                source.get(left_source).copied().unwrap_or(source_default),
                                source
                                    .get(left_source + 1)
                                    .copied()
                                    .unwrap_or(source_default),
                            );
                            let local_left = 2 * (pair - pair_start);
                            output[local_left] = left;
                            let right = if local_left + 1 < output.len() {
                                let right_source = left_source + 2;
                                let right = fold(
                                    source.get(right_source).copied().unwrap_or(source_default),
                                    source
                                        .get(right_source + 1)
                                        .copied()
                                        .unwrap_or(source_default),
                                );
                                output[local_left + 1] = right;
                                right
                            } else {
                                next_default
                            };
                            let pair_coefficients = coefficients(left, right);
                            let first_weight = first[pair - block_start];
                            for (sum, coefficient) in inner.iter_mut().zip(pair_coefficients) {
                                *sum += first_weight * coefficient;
                            }
                        }

                        let second_weight = second[second_index];
                        for (sum, inner) in outer.iter_mut().zip(inner) {
                            *sum += second_weight * inner;
                        }
                        pair_index = block_end;
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
        };

        let suffix_weight = SplitEqualitySuffixMass::new(first, second)
            .and_then(|suffix| suffix.weight_from(output_len.div_ceil(2)))
            .expect("split equality and exact prefix were validated at construction");
        add_scaled_round_coefficients(
            &mut round_coefficients,
            &coefficients(next_default, next_default),
            suffix_weight,
        );

        table.replace_after_fold(scratch, next_default);
        return round_coefficients;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = scratch;
    table
        .fold_in_place(fold)
        .expect("validated exact-prefix product state can fold");
    let explicit_pair_count = table.explicit_len().div_ceil(2);
    let default = table.default_value();
    accumulate_equality_weighted_round(
        first,
        second,
        explicit_pair_count,
        |pair_index| {
            coefficients(
                table.value_or_default(2 * pair_index),
                table.value_or_default(2 * pair_index + 1),
            )
        },
        coefficients(default, default),
    )
}

pub(super) fn accumulate_equality_weighted_round<E: Field + Unreduced>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync,
    default_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    debug_assert!(explicit_pair_count <= first.len() * second.len());
    let mut coefficients = if E::SUM_IS_EXACT {
        accumulate_delayed_blocks(first, second, explicit_pair_count, &coefficients_at)
    } else {
        accumulate_canonical_blocks(first, second, explicit_pair_count, &coefficients_at)
    };
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_pair_count))
        .expect("split equality and exact prefix were validated at construction");
    add_scaled_round_coefficients(&mut coefficients, &default_coefficients, suffix_weight);
    coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{FpExt4, Prime128Offset275, Prime32Offset99, Ring};

    fn check_blocked_accumulation<E: Field + Ring + Unreduced>() {
        let first = [2, 3, 5, 7].map(E::from_u64);
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
            let actual = accumulate_equality_weighted_round(
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
                add_scaled_round_coefficients(&mut expected, &row, weight);
            }
            assert_eq!(actual, expected);
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
