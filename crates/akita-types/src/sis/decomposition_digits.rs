//! Gadget-decomposition digit counts.

use super::norm_bound::{fold_witness_beta, FoldChallengeNorms, FoldWitnessNorms};
use crate::DecompositionParams;

/// Maximum positive value representable by `num_digits` balanced base-`b`
/// digits, where `b = 2^log_basis`. Each balanced digit lies in
/// `[-b/2, b/2 - 1]`; the max positive value is the geometric series
/// `(b/2 - 1) · (b^n - 1) / (b - 1)`. When `b^n` overflows `u128` the result is
/// a conservative lower bound (safe: it can only add a digit, never drop one).
fn balanced_digit_max(log_basis: u32, num_digits: usize) -> u128 {
    let base: u128 = 1u128 << log_basis;
    let max_digit = base / 2 - 1;
    let base_minus_1 = base - 1;

    let mut base_pow = 1u128;
    for _ in 0..num_digits {
        base_pow = base_pow.saturating_mul(base);
    }

    max_digit.saturating_mul(base_pow.saturating_sub(1) / base_minus_1)
}

/// Minimum number of balanced base-`2^log_basis` digits needed to represent any
/// value in `[-V, V]` with `V < 2^log_bound`, using symmetric centering.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if `log_bound` exceeds 128.
pub fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    assert!(
        log_bound <= 128,
        "log_bound={log_bound} exceeds 128-bit field"
    );

    if log_bound == 0 {
        return 1;
    }

    let mut num_digits = (log_bound as usize).div_ceil(log_basis as usize);
    let total_bits = (num_digits as u32).saturating_mul(log_basis);
    if total_bits <= log_bound {
        let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
        if balanced_digit_max(log_basis, num_digits) < required_positive {
            num_digits += 1;
        }
    }
    num_digits.max(1)
}

/// Decomposition depth for full-field values using asymmetric centering:
/// `ceil(field_bits / log_basis)` with no +1 correction.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_num_digits_full_field(field_bits: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if field_bits == 0 {
        return 1;
    }
    (field_bits as usize).div_ceil(log_basis as usize).max(1)
}

/// Choose the correct digit-count function for an explicit field bit width.
/// Full-field bounds (`log_bound >= field_bits`) use asymmetric centering;
/// smaller bounds use symmetric centering.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if the effective symmetric
/// bound exceeds 128 bits.
pub fn num_digits_for_bound(log_bound: u32, field_bits: u32, log_basis: u32) -> usize {
    if log_bound >= field_bits {
        compute_num_digits_full_field(field_bits, log_basis)
    } else {
        compute_num_digits(log_bound, log_basis)
    }
}

/// `(δ_commit, δ_open)` for one decomposition (commit-bound digits, open-bound
/// digits). Renames the former `decomp_depths`.
pub fn decomp_depths(decomposition: DecompositionParams) -> (usize, usize) {
    let field_bits = decomposition.field_bits();
    let depth_commit = num_digits_for_bound(
        decomposition.log_commit_bound,
        field_bits,
        decomposition.log_basis,
    );
    let open_bound = decomposition
        .log_open_bound
        .unwrap_or(decomposition.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, field_bits, decomposition.log_basis);
    (depth_commit, depth_open)
}

/// `δ_commit`: digits per coefficient of the committed witness `s`, using the
/// level's gadget base `decomposition.log_basis`.
///
/// The root commits against its configured `log_commit_bound`; a recursive
/// level commits the balanced-digit witness, whose commit bound collapses to
/// `log_basis`.
pub fn num_digits_s_commit(decomposition: DecompositionParams, is_root: bool) -> usize {
    let field_bits = decomposition.field_bits();
    let bound = if is_root {
        decomposition.log_commit_bound
    } else {
        decomposition.log_basis
    };
    num_digits_for_bound(bound, field_bits, decomposition.log_basis)
}

/// `δ_open`: digits per coefficient of the opening witnesses `t̂` / `ŵ`,
/// which are opened at the field level (`log_open_bound`).
pub fn num_digits_open(decomposition: DecompositionParams) -> usize {
    let field_bits = decomposition.field_bits();
    let bound = decomposition
        .log_open_bound
        .unwrap_or(decomposition.log_commit_bound);
    num_digits_for_bound(bound, field_bits, decomposition.log_basis)
}

/// `δ_fold`: digits per coefficient of the folded witness `z = Σ c_i·s_i`.
///
/// Computes the folded-witness L∞ bound
/// `β = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`
/// (via [`fold_witness_beta`]) from the per-level fold challenge and witness
/// norms, then returns the balanced-digit count needed to represent `[-β, β]`.
pub fn num_digits_fold(
    r_vars: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
) -> usize {
    let beta = fold_witness_beta(r_vars, num_claims, challenge, witness);
    if beta == 0 {
        return 1;
    }
    if beta == u128::MAX {
        return compute_num_digits_full_field(field_bits, log_basis);
    }
    // `beta` bounds `|v|`, so `+beta` itself must be representable: add one bit
    // so the signed range's positive end covers `+beta`.
    let log_beta = (128 - beta.leading_zeros()).saturating_add(1);
    num_digits_for_bound(log_beta, field_bits, log_basis)
}

/// A-matrix committed width (ring columns): `block_len · δ_commit`.
#[inline]
pub fn decomposed_s_block_ring_count(block_len: usize, num_digits_commit: usize) -> Option<usize> {
    block_len.checked_mul(num_digits_commit)
}

/// B-matrix committed width (ring columns): `n_a · δ_open · num_blocks · t_vectors`.
#[inline]
pub fn decomposed_t_ring_count(
    n_a: usize,
    num_digits_open: usize,
    num_blocks: usize,
    t_vectors: usize,
) -> Option<usize> {
    n_a.checked_mul(num_digits_open)?
        .checked_mul(num_blocks)?
        .checked_mul(t_vectors)
}

/// D-matrix committed width (ring columns): `δ_open · num_blocks · t_vectors`.
#[inline]
pub fn decomposed_w_ring_count(
    num_digits_open: usize,
    num_blocks: usize,
    t_vectors: usize,
) -> Option<usize> {
    num_digits_open
        .checked_mul(num_blocks)?
        .checked_mul(t_vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_digit_max_cases() {
        assert_eq!(balanced_digit_max(2, 2), 5);
        assert_eq!(balanced_digit_max(3, 1), 3);
    }

    #[test]
    fn digits_basic() {
        assert_eq!(compute_num_digits(128, 2), 65);
        assert_eq!(compute_num_digits(128, 3), 43);
        assert_eq!(compute_num_digits(1, 2), 1);
        assert_eq!(compute_num_digits(0, 2), 1);
    }

    #[test]
    fn full_field_digits() {
        assert_eq!(compute_num_digits_full_field(128, 2), 64);
        assert_eq!(compute_num_digits_full_field(128, 3), 43);
        assert_eq!(compute_num_digits_full_field(128, 4), 32);
        assert_eq!(compute_num_digits_full_field(128, 8), 16);
    }

    #[test]
    fn num_digits_for_bound_selects_correctly() {
        assert_eq!(num_digits_for_bound(128, 128, 2), 64);
        assert_eq!(num_digits_for_bound(10, 128, 2), compute_num_digits(10, 2));
        assert_eq!(num_digits_for_bound(128, 128, 3), 43);
    }

    #[test]
    fn widths_are_checked() {
        assert_eq!(decomposed_s_block_ring_count(4, 3), Some(12));
        assert_eq!(decomposed_t_ring_count(2, 3, 4, 5), Some(120));
        assert_eq!(decomposed_w_ring_count(3, 4, 5), Some(60));
        assert_eq!(decomposed_s_block_ring_count(usize::MAX, 2), None);
    }

    #[test]
    fn num_digits_fold_derives_beta() {
        // Dense witness (||s||_inf = b/2, ||s||_1 = D·b/2) picks the
        // ||c||_1·||s||_inf side; one-hot (||s||_1 = 1) picks ||c||_inf and
        // needs strictly fewer digits.
        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        // dense: log_basis=3 ⇒ ||s||_inf = b/2 = 4, ||s||_1 = D·b/2 = 64·4.
        let dense = FoldWitnessNorms::new(3, 64, 1, false);
        // one-hot single-chunk: ||s||_inf = 1, ||s||_1 = 1.
        let onehot = FoldWitnessNorms::new(3, 64, 64, true);
        let dense_digits = num_digits_fold(8, 1, 128, 3, challenge, dense);
        let onehot_digits = num_digits_fold(8, 1, 128, 3, challenge, onehot);
        assert!(dense_digits > 0 && onehot_digits > 0);
        assert!(onehot_digits < dense_digits);
        // More claims never reduce the digit count.
        assert!(num_digits_fold(8, 4, 128, 3, challenge, dense) >= dense_digits);
    }
}
