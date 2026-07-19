use super::exact_prefix::SplitEqualitySuffixMass;
use super::MAX_TREE_STAGE_Q_DEGREE;
use akita_field::parallel::*;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, Zero};

#[inline]
fn accumulate_canonical_blocks<E: FieldCore>(
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
pub(super) fn add_scaled_round_coefficients<E: FieldCore>(
    destination: &mut [E; MAX_TREE_STAGE_Q_DEGREE + 1],
    source: &[E; MAX_TREE_STAGE_Q_DEGREE + 1],
    scale: E,
) {
    for (destination, &source) in destination.iter_mut().zip(source.iter()) {
        *destination += scale * source;
    }
}

fn accumulate_delayed_blocks<E: FieldCore + HasUnreducedOps>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: &(impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync),
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let explicit_block_count = explicit_pair_count.div_ceil(first.len());
    cfg_fold_reduce!(
        0..explicit_block_count,
        || [E::ProductAccum::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
        |mut outer, second_index| {
            let block_start = second_index * first.len();
            let block_end = explicit_pair_count.min(block_start + first.len());
            let mut inner = [E::ProductAccum::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
            for pair_index in block_start..block_end {
                let source = coefficients_at(pair_index);
                let first_weight = first[pair_index - block_start];
                for (destination, source) in inner.iter_mut().zip(source) {
                    *destination += first_weight.mul_to_product_accum(source);
                }
            }
            let second_weight = second[second_index];
            for (destination, inner) in outer.iter_mut().zip(inner) {
                *destination += second_weight.mul_to_product_accum(E::reduce_product_accum(inner));
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
    .map(E::reduce_product_accum)
}

pub(super) fn accumulate_equality_weighted_round<E: FieldCore + HasUnreducedOps>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync,
    default_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    debug_assert!(explicit_pair_count <= first.len() * second.len());
    let mut coefficients = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
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
    use akita_field::{FpExt4, FromPrimitiveInt, Prime128Offset275, Prime32Offset99};

    fn check_blocked_accumulation<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>() {
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
        assert!(!Prime128Offset275::DELAYED_PRODUCT_SUM_IS_EXACT);
        check_blocked_accumulation::<Prime128Offset275>();
    }

    #[test]
    fn blocked_accumulation_matches_dense_with_delayed_reduction() {
        type E = FpExt4<Prime32Offset99>;
        assert!(E::DELAYED_PRODUCT_SUM_IS_EXACT);
        check_blocked_accumulation::<E>();
    }
}
