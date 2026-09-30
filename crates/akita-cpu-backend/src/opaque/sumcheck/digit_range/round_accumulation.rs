use super::exact_prefix::SplitEqualitySuffixMass;
use jolt_field::solinas::parallel::*;
use jolt_field::Unreduced;
use jolt_field::{Field, Zero};

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

#[cfg(test)]
mod tests {
    use super::super::MAX_TREE_STAGE_Q_DEGREE;
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
