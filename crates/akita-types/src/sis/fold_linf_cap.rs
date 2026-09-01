//! Fold-l∞ tail-bound primitives for balanced signed-digit policy sizing.
//!
//! [`FoldWitnessLinfCapConfig`] supplies the proved tail-bound inputs used to
//! size digit depth from `min(β_inf, t*)`.
//! A-role MSIS pricing is separate: it uses
//! [`super::decomposition_digits::balanced_digit_abs_max`] at the
//! resulting `δ_fold` depth (see [`super::norm_bound::rounded_up_role_a_inf_norm`]).

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;

/// Per-challenge **grind** acceptance target `p_grind = NUM / DEN` used in the union-bound
/// sizing for `t*` (`specs/fold-linf-rejection.md`).
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM: u32 = 1;
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN: u32 = 8;

/// Rational ceiling for `ln(2)` used to bound natural logarithms without floats.
const LN2_CEIL_NUM: u128 = 71;
const LN2_CEIL_DEN: u128 = 100;

/// Conservative integer ceiling for `ln(x)` with `x >= 1`, via
/// `ln(x) <= ceil(log2 x) · ln(2)`.
#[inline]
fn ceil_natural_log(x: u128) -> u128 {
    if x <= 1 {
        return 0;
    }
    let ceil_log2 = 128u32.saturating_sub((x - 1).leading_zeros()) as u128;
    ceil_log2
        .saturating_mul(LN2_CEIL_NUM)
        .div_ceil(LN2_CEIL_DEN)
}

/// Direct union-bound ln for `ln(2·num_fold_coeffs / (1 - p_grind))`.
#[inline]
pub(crate) fn fold_witness_linf_grind_union_ln(
    num_fold_coeffs: u128,
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    if num_fold_coeffs == 0
        || grind_target_accept_num == 0
        || grind_target_accept_den == 0
        || grind_target_accept_num >= grind_target_accept_den
    {
        return Err(AkitaError::InvalidSetup(
            "fold grind sizing inputs must be positive with p_num < p_den".to_string(),
        ));
    }
    let miss = grind_target_accept_den - grind_target_accept_num;
    let numerator = 2u128
        .checked_mul(num_fold_coeffs)
        .and_then(|value| value.checked_mul(grind_target_accept_den))
        .ok_or_else(|| AkitaError::InvalidSetup("fold grind union bound overflows u128".into()))?;
    Ok(ceil_natural_log(numerator.div_ceil(miss)))
}

/// Squared `‖z‖_inf` tail bound `t*²` from the sub-Gaussian argument in
/// `specs/fold-linf-rejection.md`:
///
/// ```text
/// t*² = 2 · num_fold_blocks · challenge_l2_sq_max · witness_linf² · ln_term
/// ```
///
/// `ln_term` is a conservative integer for the grind union bound. The real square root is
/// taken only at digit-sizing boundaries. Digit sizing uses `min(β_inf, t*)`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when any argument is zero or the product
/// overflows `u128`.
fn rademacher_proxy_variance_from_ln(
    num_fold_blocks: u128,
    challenge_l2_sq_max: u128,
    witness_linf_sq: u128,
    ln_term: u128,
) -> Result<u128, AkitaError> {
    if num_fold_blocks == 0 || challenge_l2_sq_max == 0 || witness_linf_sq == 0 || ln_term == 0 {
        return Err(AkitaError::InvalidSetup(
            "rademacher_proxy_variance_from_ln: arguments must be positive".to_string(),
        ));
    }
    let two = 2u128;
    two.checked_mul(num_fold_blocks)
        .and_then(|v| v.checked_mul(challenge_l2_sq_max))
        .and_then(|v| v.checked_mul(witness_linf_sq))
        .and_then(|v| v.checked_mul(ln_term))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "rademacher_proxy_variance_from_ln: t*² overflows u128".to_string(),
            )
        })
}

pub fn rademacher_proxy_variance(
    num_live_blocks: usize,
    num_claims: usize,
    witness_linf_sq: u128,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<u128, AkitaError> {
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "rademacher_proxy_variance: num_live_blocks must be positive".to_string(),
        ));
    }
    let num_fold_blocks = (num_claims as u128)
        .checked_mul(num_live_blocks as u128)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "rademacher_proxy_variance: num_fold_blocks overflows u128".to_string(),
            )
        })?;
    rademacher_proxy_variance_from_ln(
        num_fold_blocks,
        cap_config.challenge_l2_sq_max,
        witness_linf_sq,
        cap_config.grind_union_ln,
    )
}

/// Level-static configuration for balanced signed-digit policy sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldWitnessLinfCapConfig {
    /// Family worst-case `max ‖c‖_2²` per logical block.
    challenge_l2_sq_max: u128,
    /// Precomputed union ln term.
    grind_union_ln: u128,
}

impl FoldWitnessLinfCapConfig {
    /// Tail-aware sizing inputs for a fold level from its sparse family and
    /// inner A-matrix width (`num_positions_per_block · δ_commit`).
    #[inline]
    pub fn for_fold_coeffs(
        fold_challenge_config: &SparseChallengeConfig,
        num_fold_coeffs: usize,
    ) -> Result<Self, AkitaError> {
        let num_fold_coeffs = num_fold_coeffs as u128;
        let grind_union_ln = fold_witness_linf_grind_union_ln(
            num_fold_coeffs,
            FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM as u128,
            FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN as u128,
        )?;
        Ok(Self {
            challenge_l2_sq_max: fold_challenge_config.challenge_l2_sq_max(),
            grind_union_ln,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::norm_bound::{FoldChallengeNorms, FoldWitnessNorms};

    #[test]
    fn rademacher_proxy_variance_from_ln_is_monotone_and_rejects_zero_inputs() {
        let base = rademacher_proxy_variance_from_ln(16, 71, 1, 24).unwrap();
        assert!(rademacher_proxy_variance_from_ln(32, 71, 1, 24).unwrap() >= base);
        assert!(rademacher_proxy_variance_from_ln(16, 71, 4, 24).unwrap() >= base);
        assert!(rademacher_proxy_variance_from_ln(0, 71, 1, 24).is_err());
    }

    #[test]
    fn fold_witness_linf_grind_union_ln_half_matches_ln_4n() {
        let term_16 = fold_witness_linf_grind_union_ln(1 << 16, 1, 2).unwrap();
        assert!((13..=15).contains(&term_16));
        let term_max = fold_witness_linf_grind_union_ln(1u128 << 32, 1, 2).unwrap();
        assert!((24..=26).contains(&term_max));
    }

    #[test]
    fn fold_witness_linf_grind_union_ln_eighth_at_2_16() {
        let grind_only = fold_witness_linf_grind_union_ln(1u128 << 16, 1, 8).unwrap();
        assert_eq!(grind_only, 13, "ceil_ln(2·2^16·8/7)");
    }

    #[test]
    fn fold_witness_linf_grind_union_ln_eighth_is_tighter_than_half() {
        let n = 100u128;
        let half = fold_witness_linf_grind_union_ln(n, 1, 2).unwrap();
        let eighth = fold_witness_linf_grind_union_ln(n, 1, 8).unwrap();
        assert!(eighth < half, "eighth={eighth} half={half}");
        let t_half = rademacher_proxy_variance_from_ln(1, 71, 1, half).unwrap();
        let t_eighth = rademacher_proxy_variance_from_ln(1, 71, 1, eighth).unwrap();
        assert!(t_eighth < t_half);
    }

    #[test]
    fn threshold_t_star_below_pessimistic_linf_envelope_at_production_shell() {
        use crate::layout::digit_math::isqrt_ceil;

        let challenge = FoldChallengeNorms {
            infinity_norm: 2,
            l1_norm: 51,
        };
        let witness = FoldWitnessNorms::sparse_binary(64, 64).unwrap();
        let tight_beta = 4u128
            * (challenge.infinity_norm * witness.l1_norm())
                .min(challenge.l1_norm * witness.infinity_norm());
        let pessimistic_linf_envelope = 16u128 * challenge.l1_norm * witness.infinity_norm();
        assert!(tight_beta < pessimistic_linf_envelope);
        let ln_term = fold_witness_linf_grind_union_ln(1u128 << 16, 1, 8).unwrap();
        let t_sq = rademacher_proxy_variance_from_ln(16, 71, 1, ln_term).unwrap();
        let t = isqrt_ceil(t_sq);
        assert!(
            t < pessimistic_linf_envelope,
            "t* = {t} pessimistic envelope = {pessimistic_linf_envelope}"
        );
        assert_eq!(t.min(tight_beta), tight_beta);
    }
}
