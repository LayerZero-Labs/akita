use super::MAX_TREE_STAGE_Q_DEGREE;
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
