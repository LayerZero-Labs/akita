//! Gadget-decomposition digit counts and the committed-matrix widths derived
//! from them.
//!
//! Three layers live here, lowest to highest:
//!
//! 1. **Core digit-count math** — how many balanced base-`2^log_basis` digits
//!    represent a bound. Two centering conventions exist:
//!    - `compute_num_digits` (crate-private): the *symmetric* signed range
//!      `[-2^(k-1), 2^(k-1) - 1]`, including the sign-bit correction. Reached
//!      only through the router below.
//!    - [`compute_num_digits_full_field`]: the *asymmetric* full-field residue,
//!      plain `ceil(field_bits / log_basis)` with no correction.
//!    - [`num_digits_for_bound`]: the router. Full-field bounds
//!      (`log_bound >= field_bits`) use the asymmetric count; smaller bounds use
//!      the symmetric one. This is the *only* symmetric entry point, so a caller
//!      cannot accidentally request the symmetric count of a full-field bound
//!      (the historical `compute_num_digits(128, _)` footgun).
//!
//! 2. **Per-role selectors** — map a [`DecompositionParams`] to the digit depth
//!    of a specific witness role, encoding which bound applies to each:
//!    - [`num_digits_s_commit`]: committed witness `s` (`log_commit_bound` at
//!      the root, `log_basis` at recursive levels).
//!    - [`num_digits_open`]: opening witnesses `t̂` / `ŵ` (`log_open_bound`).
//!    - [`num_digits_fold`]: folded witness `z` — the digit count for the
//!      norm-derived bound `β`, which is not a [`DecompositionParams`] field.
//!
//! 3. **Committed-matrix widths** — name the `checked_mul` products that turn a
//!    digit depth plus block geometry into a matrix's ring-column count:
//!    [`decomposed_s_block_ring_count`] (A), [`decomposed_t_ring_count`] (B),
//!    [`decomposed_w_ring_count`] (D). These are layout arithmetic, not digit
//!    math; they sit here so each width formula lives beside the depth it
//!    multiplies.

use akita_field::AkitaError;

use super::norm_bound::{
    fold_witness_beta, fold_witness_linf_cap, FoldChallengeNorms, FoldWitnessLinfCapConfig,
    FoldWitnessNorms,
};
use crate::DecompositionParams;

/// Maximum positive value representable by `num_digits` balanced base-`b`
/// digits, where `b = 2^log_basis`. Each balanced digit lies in
/// `[-b/2, b/2 - 1]`; the max positive value is the geometric series
/// `(b/2 - 1) · (b^n - 1) / (b - 1)`. When `b^n` overflows `u128` the result is
/// a conservative lower bound (safe: it can only add a digit, never drop one).
#[inline]
pub fn balanced_positive_digit_max(log_basis: u32, num_digits: usize) -> u128 {
    let base: u128 = 1u128 << log_basis;
    let max_digit = base / 2 - 1;
    let base_minus_1 = base - 1;

    let mut base_pow = 1u128;
    for _ in 0..num_digits {
        base_pow = base_pow.saturating_mul(base);
    }

    max_digit.saturating_mul(base_pow.saturating_sub(1) / base_minus_1)
}

/// Minimum number of balanced base-`2^log_basis` digits needed to represent a
/// `log_bound`-bit *signed* coefficient, using symmetric centering.
///
/// Following [`crate::DecompositionParams::log_commit_bound`], a bound of `k`
/// bits denotes the centered range `[-2^(k-1), 2^(k-1) - 1]`, i.e. one sign bit
/// plus `k-1` magnitude bits. The binding constraint is the positive end,
/// `2^(k-1) - 1`, since the balanced digit range `[-b/2, b/2 - 1]` reaches
/// further on the negative side. This is *not* `2^log_bound - 1`: the leading
/// bit is the sign, so callers that mean "magnitude up to `2^m`" must pass
/// `log_bound = m + 1` (this is exactly what [`num_digits_fold`] does).
///
/// The count is `ceil(log_bound / log_basis)`, plus one more digit when the
/// balanced-digit positive reach `balanced_digit_max` still falls short of
/// `2^(log_bound-1) - 1`. The extra digit is only ever needed when `log_basis`
/// divides `log_bound` exactly (otherwise `ceil(log_bound/log_basis)·log_basis
/// > log_bound`, so the reach already clears `2^(log_bound-1)`); the check is
/// run unconditionally because it is cheap and self-evidently correct. Both the
/// coverage and the minimality of the result are pinned by the
/// `compute_num_digits_covers_signed_range` unit test.
///
/// This symmetric count is for *small* bounds (`log_bound < field_bits`):
/// one-hot `log_commit_bound = 1`, recursive `log_basis`, and fold `log_beta`.
/// It is crate-private and reached only through [`num_digits_for_bound`], which
/// routes full-field bounds to the asymmetric [`compute_num_digits_full_field`]
/// instead — so no caller can ask for the symmetric count of a full-field bound.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or at least 128, or if `log_bound` exceeds 128.
pub(crate) fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    assert!(
        log_bound <= 128,
        "log_bound={log_bound} exceeds 128-bit field"
    );

    if log_bound == 0 {
        return 1;
    }

    let mut num_digits = (log_bound as usize).div_ceil(log_basis as usize);
    let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
    if balanced_positive_digit_max(log_basis, num_digits) < required_positive {
        num_digits += 1;
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
/// norms. Under [`crate::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind`], the signed range is
/// sized from `min(β_inf, t*)` with `t*²` from [`crate::sis::fold_witness_linf_tail_bound_sq`];
/// deterministic policies use `β_inf` alone.
///
/// # Errors
///
/// Returns `AkitaError::InvalidSetup` when [`fold_witness_beta`] rejects the
/// inputs (`r_vars >= 127` or `β` overflow), or when `β == 0` (a zero
/// challenge/witness norm or `num_claims == 0` — no well-formed level folds a
/// zero witness).
pub fn num_digits_fold(
    r_vars: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: FoldWitnessLinfCapConfig,
) -> Result<usize, AkitaError> {
    let beta = fold_witness_beta(r_vars, num_claims, challenge, witness)?;
    if beta == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_fold: folded-witness bound β = 0".to_string(),
        ));
    }
    let num_fold_blocks = (num_claims as u128)
        .checked_mul(1u128 << r_vars)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("num_digits_fold: num_fold_blocks overflows u128".to_string())
        })?;
    let witness_linf_sq = witness
        .infinity_norm()
        .saturating_mul(witness.infinity_norm());
    let cap = fold_witness_linf_cap(beta, num_fold_blocks, witness_linf_sq, &cap_config)?;
    if cap == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_fold: fold witness L∞ cap is zero".to_string(),
        ));
    }
    // `cap` bounds `|v|`, so `+cap` itself must be representable: add one bit
    // so the signed range's positive end covers `+cap`.
    let log_cap = (128 - cap.leading_zeros()).saturating_add(1);
    Ok(num_digits_for_bound(log_cap, field_bits, log_basis))
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
        assert_eq!(balanced_positive_digit_max(2, 2), 5);
        assert_eq!(balanced_positive_digit_max(3, 1), 3);
    }

    #[test]
    fn digits_basic() {
        // Production `compute_num_digits` inputs are small symmetric bounds:
        // one-hot `log_commit_bound = 1`, recursive `log_basis`, fold
        // `log_beta`. Full-field bounds go through `num_digits_for_bound` to
        // `compute_num_digits_full_field`, not here.
        assert_eq!(compute_num_digits(1, 2), 1);
        assert_eq!(compute_num_digits(0, 2), 1);
        // `log_basis` itself (the recursive commit bound): one base-`2^lb`
        // digit covers the balanced range `[-2^(lb-1), 2^(lb-1) - 1]` exactly.
        assert_eq!(compute_num_digits(2, 2), 1);
        assert_eq!(compute_num_digits(3, 3), 1);
    }

    /// The returned digit count must actually cover the signed range
    /// `[-2^(log_bound-1), 2^(log_bound-1) - 1]` its contract promises, for
    /// every production base and bound. This pins the invariant the previous
    /// conditional guard left unchecked whenever `log_basis ∤ log_bound`.
    #[test]
    fn compute_num_digits_covers_signed_range() {
        for log_basis in 2u32..=8 {
            for log_bound in 1u32..=120 {
                let n = compute_num_digits(log_bound, log_basis);
                let required_positive = (1u128 << (log_bound - 1)).saturating_sub(1);
                assert!(
                    balanced_positive_digit_max(log_basis, n) >= required_positive,
                    "log_bound={log_bound} log_basis={log_basis} n={n} \
                     reach={} < required={required_positive}",
                    balanced_positive_digit_max(log_basis, n),
                );
                // Minimality: one fewer digit must be insufficient (unless n==1).
                if n > 1 {
                    assert!(
                        balanced_positive_digit_max(log_basis, n - 1) < required_positive,
                        "non-minimal: log_bound={log_bound} log_basis={log_basis} n={n}",
                    );
                }
            }
        }
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
        let dense_digits = num_digits_fold(
            8,
            1,
            128,
            3,
            challenge,
            dense,
            FoldWitnessLinfCapConfig::worst_case_beta_only(),
        )
        .unwrap();
        let onehot_digits = num_digits_fold(
            8,
            1,
            128,
            3,
            challenge,
            onehot,
            FoldWitnessLinfCapConfig::worst_case_beta_only(),
        )
        .unwrap();
        assert!(dense_digits > 0 && onehot_digits > 0);
        assert!(onehot_digits < dense_digits);
        // More claims never reduce the digit count.
        assert!(
            num_digits_fold(
                8,
                4,
                128,
                3,
                challenge,
                dense,
                FoldWitnessLinfCapConfig::worst_case_beta_only()
            )
            .unwrap()
                >= dense_digits
        );
    }

    #[test]
    fn num_digits_fold_rejects_degenerate() {
        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::new(3, 64, 1, false);
        // r_vars >= 127 is rejected.
        assert!(num_digits_fold(
            127,
            1,
            128,
            3,
            challenge,
            witness,
            FoldWitnessLinfCapConfig::worst_case_beta_only()
        )
        .is_err());
        // num_claims == 0 ⇒ β = 0 is rejected.
        assert!(num_digits_fold(
            8,
            0,
            128,
            3,
            challenge,
            witness,
            FoldWitnessLinfCapConfig::worst_case_beta_only()
        )
        .is_err());
    }
}
