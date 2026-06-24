use akita_field::parallel::*;
use akita_field::{FieldCore, FromPrimitiveInt};

pub(super) fn product_claim<E: FieldCore>(table: &[E], left_factor: &[E], right_factor: &[E]) -> E {
    let right_len = right_factor.len();
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row = &table[left_idx * right_len..(left_idx + 1) * right_len];
            for (&value, &right_weight) in row.iter().zip(right_factor.iter()) {
                acc += value * left_weight * right_weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

pub(super) fn product_claim_compact<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    left_factor: &[E],
    right_factor: &[E],
) -> E {
    let right_len = right_factor.len();
    debug_assert_eq!(padded_len, left_factor.len() * right_len);
    cfg_fold_reduce!(
        0..left_factor.len(),
        E::zero,
        |mut acc, left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for (right_idx, &right_weight) in right_factor.iter().enumerate() {
                let Some(&digit) = digits.get(row_base + right_idx) else {
                    continue;
                };
                if digit != 0 {
                    acc += E::from_i64(i64::from(digit)) * left_weight * right_weight;
                }
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

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

pub(super) fn accumulate_right_round_compact<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    left_factor: &[E],
    right_factor: &[E],
) -> (E, E, E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    debug_assert_eq!(padded_len, left_factor.len() * right_len);
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                let s0 = compact_value_at::<E>(digits, row_base + 2 * pair_idx);
                let s1 = compact_value_at::<E>(digits, row_base + 2 * pair_idx + 1);
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

pub(super) fn accumulate_second_right_round_compact<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    left_factor: &[E],
    right_factor: &[E],
    first_challenge: E,
) -> (E, E, E) {
    let folded_right_len = right_factor.len();
    let original_right_len = folded_right_len * 2;
    let half = folded_right_len / 2;
    debug_assert_eq!(padded_len, left_factor.len() * original_right_len);
    cfg_fold_reduce!(
        0..left_factor.len(),
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), left_idx| {
            let left_weight = left_factor[left_idx];
            let row_base = left_idx * original_right_len;
            for pair_idx in 0..half {
                let digit_base = row_base + 4 * pair_idx;
                let s0 = fold_pair(
                    compact_value_at::<E>(digits, digit_base),
                    compact_value_at::<E>(digits, digit_base + 1),
                    first_challenge,
                );
                let s1 = fold_pair(
                    compact_value_at::<E>(digits, digit_base + 2),
                    compact_value_at::<E>(digits, digit_base + 3),
                    first_challenge,
                );
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

pub(super) fn accumulate_left_round_compact<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    left_factor: &[E],
    right_weight: E,
) -> (E, E, E) {
    debug_assert_eq!(padded_len, left_factor.len());
    let half = left_factor.len() / 2;
    cfg_fold_reduce!(
        0..half,
        || (E::zero(), E::zero(), E::zero()),
        |(mut constant, mut linear, mut quadratic), pair_idx| {
            let s0 = compact_value_at::<E>(digits, 2 * pair_idx);
            let s1 = compact_value_at::<E>(digits, 2 * pair_idx + 1);
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

pub(super) fn fold_pair<E: FieldCore>(left: E, right: E, r: E) -> E {
    left + r * (right - left)
}

pub(super) fn fold_right_round<E: FieldCore>(table: &mut Vec<E>, right_factor: &mut Vec<E>, r: E) {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = table.len() / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for pair_idx in 0..half {
                row[pair_idx] = fold_pair(
                    table[row_base + 2 * pair_idx],
                    table[row_base + 2 * pair_idx + 1],
                    r,
                );
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(right_factor[2 * idx], right_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    *table = folded;
}

pub(super) fn fold_compact_right_round<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    right_factor: &mut Vec<E>,
    r: E,
) -> Vec<E> {
    let right_len = right_factor.len();
    let half = right_len / 2;
    let left_len = padded_len / right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * right_len;
            for (pair_idx, slot) in row.iter_mut().enumerate() {
                *slot = fold_pair(
                    compact_value_at::<E>(digits, row_base + 2 * pair_idx),
                    compact_value_at::<E>(digits, row_base + 2 * pair_idx + 1),
                    r,
                );
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(right_factor[2 * idx], right_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    folded
}

pub(super) fn fold_compact_right_two_rounds<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    right_factor: &mut Vec<E>,
    first_challenge: E,
    second_challenge: E,
) -> Vec<E> {
    let folded_right_len = right_factor.len();
    let original_right_len = folded_right_len * 2;
    let half = folded_right_len / 2;
    let left_len = padded_len / original_right_len;
    let mut folded = vec![E::zero(); left_len * half];
    cfg_chunks_mut!(&mut folded, half)
        .enumerate()
        .for_each(|(left_idx, row)| {
            let row_base = left_idx * original_right_len;
            for (pair_idx, slot) in row.iter_mut().enumerate() {
                let digit_base = row_base + 4 * pair_idx;
                let w0 = fold_pair(
                    compact_value_at::<E>(digits, digit_base),
                    compact_value_at::<E>(digits, digit_base + 1),
                    first_challenge,
                );
                let w1 = fold_pair(
                    compact_value_at::<E>(digits, digit_base + 2),
                    compact_value_at::<E>(digits, digit_base + 3),
                    first_challenge,
                );
                *slot = fold_pair(w0, w1, second_challenge);
            }
        });
    let folded_right = cfg_into_iter!(0..half)
        .map(|idx| {
            fold_pair(
                right_factor[2 * idx],
                right_factor[2 * idx + 1],
                second_challenge,
            )
        })
        .collect::<Vec<_>>();
    *right_factor = folded_right;
    folded
}

pub(super) fn fold_left_round<E: FieldCore>(table: &mut Vec<E>, left_factor: &mut Vec<E>, r: E) {
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(table[2 * idx], table[2 * idx + 1], r))
        .collect::<Vec<_>>();
    let folded_left = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(left_factor[2 * idx], left_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *table = folded_table;
    *left_factor = folded_left;
}

pub(super) fn fold_compact_left_round<E: FieldCore + FromPrimitiveInt>(
    digits: &[i8],
    padded_len: usize,
    left_factor: &mut Vec<E>,
    r: E,
) -> Vec<E> {
    debug_assert_eq!(padded_len, left_factor.len());
    let half = left_factor.len() / 2;
    let folded_table = cfg_into_iter!(0..half)
        .map(|idx| {
            fold_pair(
                compact_value_at::<E>(digits, 2 * idx),
                compact_value_at::<E>(digits, 2 * idx + 1),
                r,
            )
        })
        .collect::<Vec<_>>();
    let folded_left = cfg_into_iter!(0..half)
        .map(|idx| fold_pair(left_factor[2 * idx], left_factor[2 * idx + 1], r))
        .collect::<Vec<_>>();
    *left_factor = folded_left;
    folded_table
}

pub(super) fn compact_value_at<E: FieldCore + FromPrimitiveInt>(digits: &[i8], idx: usize) -> E {
    digits
        .get(idx)
        .copied()
        .map_or_else(E::zero, |digit| E::from_i64(i64::from(digit)))
}
