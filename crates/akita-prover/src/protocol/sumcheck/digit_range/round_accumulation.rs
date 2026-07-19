use super::exact_prefix::SplitEqualitySuffixMass;
use super::MAX_TREE_STAGE_Q_DEGREE;
use akita_field::parallel::*;
use akita_field::FieldCore;

#[inline(always)]
pub(super) fn split_equality_weight<E: FieldCore>(
    first: &[E],
    second: &[E],
    pair_index: usize,
) -> E {
    let first_index = pair_index & (first.len() - 1);
    let second_index = pair_index >> first.len().trailing_zeros();
    first[first_index] * second[second_index]
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

pub(super) fn accumulate_precomputed_round<E: FieldCore>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    coefficients_at: impl Fn(usize) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] + Sync,
    default_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let mut coefficients = cfg_fold_reduce!(
        0..explicit_pair_count,
        || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
        |mut sum, pair_index| {
            add_scaled_round_coefficients(
                &mut sum,
                &coefficients_at(pair_index),
                split_equality_weight(first, second, pair_index),
            );
            sum
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right.iter()) {
                *left += *right;
            }
            left
        }
    );
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_pair_count))
        .expect("split equality and exact prefix were validated at construction");
    add_scaled_round_coefficients(&mut coefficients, &default_coefficients, suffix_weight);
    coefficients
}
