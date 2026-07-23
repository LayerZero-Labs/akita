use akita_algebra::offset_eq::eq_eval_at_index;
use akita_field::parallel::*;
use akita_field::FieldCore;

#[cfg(test)]
pub(super) fn product_claim<E: FieldCore>(table: &[E], left_factor: &[E], right_factor: &[E]) -> E {
    let right_len = right_factor.len();
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row = &table[left_idx * right_len..(left_idx + 1) * right_len];
            for (&value, &right_weight) in row.iter().zip(right_factor) {
                acc += value * left_weight * right_weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[cfg(test)]
pub(super) fn accumulate_right_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_factor: &[E],
) -> (E, E, E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                let s0 = table[row_base + 2 * pair_idx];
                let s1 = table[row_base + 2 * pair_idx + 1];
                let f0 = left_weight * right_factor[2 * pair_idx];
                let f1 = left_weight * right_factor[2 * pair_idx + 1];
                let ds = s1 - s0;
                let df = f1 - f0;
                constant += s0 * f0;
                linear += s0 * df + ds * f0;
                quadratic += ds * df;
            }
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

pub(super) fn accumulate_left_round<E: FieldCore>(
    table: &[E],
    left_factor: &[E],
    right_weight: E,
) -> (E, E, E) {
    let half = left_factor.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = table[2 * pair_idx];
            let s1 = table[2 * pair_idx + 1];
            let f0 = left_factor[2 * pair_idx] * right_weight;
            let f1 = left_factor[2 * pair_idx + 1] * right_weight;
            let ds = s1 - s0;
            let df = f1 - f0;
            constant += s0 * f0;
            linear += s0 * df + ds * f0;
            quadratic += ds * df;
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

pub(super) fn accumulate_left_round_eq<E: FieldCore>(
    table: &[E],
    point: &[E],
    scale: E,
    right_weight: E,
) -> (E, E, E) {
    let half = table.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = table[2 * pair_idx];
            let s1 = table[2 * pair_idx + 1];
            let f0 = scale * eq_eval_at_index(point, 2 * pair_idx) * right_weight;
            let f1 = scale * eq_eval_at_index(point, 2 * pair_idx + 1) * right_weight;
            let ds = s1 - s0;
            let df = f1 - f0;
            constant += s0 * f0;
            linear += s0 * df + ds * f0;
            quadratic += ds * df;
            (constant, linear, quadratic)
        },
        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    )
}

pub(super) fn fold_pair<E: FieldCore>(left: E, right: E, challenge: E) -> E {
    left + challenge * (right - left)
}

pub(super) fn fold_factor_in_place<E: FieldCore>(factor: &mut Vec<E>, challenge: E) {
    let half = factor.len() / 2;
    *factor = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(factor[2 * idx], factor[2 * idx + 1], challenge))
        .collect();
}

#[cfg(test)]
pub(super) fn fold_right_round<E: FieldCore>(
    table: &mut Vec<E>,
    right_factor: &mut Vec<E>,
    challenge: E,
) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = table.len() / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for (pair_idx, slot) in row.iter_mut().enumerate() {
                *slot = fold_pair(
                    table[row_base + 2 * pair_idx],
                    table[row_base + 2 * pair_idx + 1],
                    challenge,
                );
            }
        });
    fold_factor_in_place(right_factor, challenge);
    *table = folded;
}

#[cfg(test)]
pub(super) fn fold_left_round<E: FieldCore>(
    table: &mut Vec<E>,
    left_factor: &mut Vec<E>,
    challenge: E,
) {
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], challenge))
        .collect();
    fold_factor_in_place(left_factor, challenge);
    *table = folded_table;
}

pub(super) fn fold_dense_left_round<E: FieldCore>(table: &mut Vec<E>, challenge: E) {
    let half = table.len() / 2;
    *table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], challenge))
        .collect();
}
