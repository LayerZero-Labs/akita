//! Fold-linf tail-bound and grind-union sizing for `num_digits_fold`.
//!
//! [`FoldWitnessLinfCapConfig`] selects whether digit depth uses worst-case
//! `β_inf` alone or `min(β_inf, t*)` under a proved tail certificate.
//! A-role MSIS pricing is separate: it uses
//! [`super::decomposition_digits::balanced_digit_abs_max`] at the
//! resulting `δ_fold` depth (see [`super::norm_bound::rounded_up_role_a_inf_norm`]).

use akita_challenges::{tensor_split, SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

/// Maximum Fiat-Shamir rerolls per committed fold level under tail-bound-with-grind policy.
pub const MAX_FOLD_GRIND_ATTEMPTS: u32 = 4096;

/// Per-challenge **grind** acceptance target `p_grind = NUM / DEN` used in the union-bound
/// sizing for `t*` (`specs/fold-linf-rejection.md`).
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM: u32 = 1;
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN: u32 = 8;

/// Minimum retained fraction of `t*` when snapping `δ_fold` downward after tail sizing.
/// Retain `1/2` of `t*` (at most 50% reduction vs the tail cap).
pub const FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM: u32 = 1;
pub const FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN: u32 = 2;

/// Whether [`crate::sis::num_digits_fold`] sizes `K` from the sub-Gaussian tail
/// `t*` (`min(β_inf, t*)`) or from the worst-case envelope `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoldWitnessLinfCapPolicy {
    /// Proved sub-Gaussian tail: production signed-sparse at `D = 64`, or a
    /// production pm1-only ladder entry at `D ≥ 128`; `cap = min(β_inf, t*)` and grind allowed.
    TailBoundWithGrind,
    /// Proved second-order tensor tail for tensor folds whose factors use the
    /// same certified sign-symmetric families as [`Self::TailBoundWithGrind`].
    TensorTailBoundWithGrind,
    /// No tail certificate yet: uncertified flat presets; `cap = β_inf` only
    /// and grind nonce must be zero.
    WorstCaseBetaOnly,
}

impl FoldWitnessLinfCapPolicy {
    #[inline]
    #[must_use]
    pub const fn allows_grind(self) -> bool {
        !matches!(self, Self::WorstCaseBetaOnly)
    }
}

/// Select the fold-linf threshold policy for a stage-1 sparse family at ring
/// degree `ring_dimension` with the given fold-challenge shape.
#[inline]
#[must_use]
pub fn fold_witness_linf_cap_policy(
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
) -> FoldWitnessLinfCapPolicy {
    let flat_certified = match (ring_dimension, fold_challenge_config.count_pm2) {
        (64, pm2) if pm2 > 0 => true,
        (d, 0) => {
            SparseChallengeConfig::production_for_ring_dim(d).as_ref()
                == Some(fold_challenge_config)
        }
        _ => false,
    };
    match (fold_shape, flat_certified) {
        (TensorChallengeShape::Flat, true) => FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        (TensorChallengeShape::Tensor, true) => FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
        _ => FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
    }
}

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
    let miss = grind_target_accept_den - grind_target_accept_num;
    let numerator = 2u128
        .saturating_mul(num_fold_coeffs)
        .saturating_mul(grind_target_accept_den);
    Ok(ceil_natural_log(numerator.div_ceil(miss)))
}

/// Conservative integer for `ln(2·num_fold_coeffs / (1 - p_grind))` with
/// `p_grind = grind_target_accept_num / grind_target_accept_den`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on zero denominators, zero numerators,
/// or `p_grind >= 1`.
pub fn fold_witness_linf_ln_term(
    num_fold_coeffs: u128,
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    if num_fold_coeffs == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_ln_term: num_fold_coeffs must be positive".to_string(),
        ));
    }
    if grind_target_accept_den == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_ln_term: probability denominators must be positive".to_string(),
        ));
    }
    if grind_target_accept_num == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_ln_term: probability numerators must be positive".to_string(),
        ));
    }
    if grind_target_accept_num >= grind_target_accept_den {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_ln_term: grind target accept probability must be < 1".to_string(),
        ));
    }
    fold_witness_linf_grind_union_ln(
        num_fold_coeffs,
        grind_target_accept_num,
        grind_target_accept_den,
    )
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
pub fn fold_witness_linf_tail_bound_sq(
    num_fold_blocks: u128,
    challenge_l2_sq_max: u128,
    witness_linf_sq: u128,
    ln_term: u128,
) -> Result<u128, AkitaError> {
    if num_fold_blocks == 0 || challenge_l2_sq_max == 0 || witness_linf_sq == 0 || ln_term == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_tail_bound_sq: arguments must be positive".to_string(),
        ));
    }
    let two = 2u128;
    two.checked_mul(num_fold_blocks)
        .and_then(|v| v.checked_mul(challenge_l2_sq_max))
        .and_then(|v| v.checked_mul(witness_linf_sq))
        .and_then(|v| v.checked_mul(ln_term))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_linf_tail_bound_sq: t*² overflows u128".to_string(),
            )
        })
}

/// Tensor folded-witness tail bound for a two-way tensor challenge.
///
/// ```text
/// t_tensor² =
///   4 · num_claims · left_len · right_len · witness_linf² · s2_factor²
///     · ln(4·N/(1-p_grind))
///     · min(
///         ln(4·N·num_claims·left_len·k_factor/(1-p_grind)),
///         ln(4·N·num_claims·right_len·k_factor/(1-p_grind))
///       )
/// ```
///
/// Here `N = num_fold_coeffs`, `s2_factor = max ||factor||_2²`, and
/// `k_factor` is the factor support bound.
#[allow(clippy::too_many_arguments)]
pub fn fold_witness_linf_tensor_tail_bound_sq(
    num_claims: u128,
    left_len: u128,
    right_len: u128,
    factor_l2_sq_max: u128,
    factor_nonzero_count_max: u128,
    witness_linf_sq: u128,
    num_fold_coeffs: u128,
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    if num_claims == 0
        || left_len == 0
        || right_len == 0
        || factor_l2_sq_max == 0
        || factor_nonzero_count_max == 0
        || witness_linf_sq == 0
        || num_fold_coeffs == 0
    {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_tensor_tail_bound_sq: arguments must be positive".to_string(),
        ));
    }
    let lambda_outer = fold_witness_linf_tensor_outer_ln(
        num_fold_coeffs,
        grind_target_accept_num,
        grind_target_accept_den,
    )?;
    let left_inner = fold_witness_linf_tensor_inner_ln(
        num_fold_coeffs,
        num_claims,
        left_len,
        factor_nonzero_count_max,
        grind_target_accept_num,
        grind_target_accept_den,
    )?;
    let right_inner = fold_witness_linf_tensor_inner_ln(
        num_fold_coeffs,
        num_claims,
        right_len,
        factor_nonzero_count_max,
        grind_target_accept_num,
        grind_target_accept_den,
    )?;
    let lambda_inner = left_inner.min(right_inner);
    4u128
        .checked_mul(num_claims)
        .and_then(|v| v.checked_mul(left_len))
        .and_then(|v| v.checked_mul(right_len))
        .and_then(|v| v.checked_mul(witness_linf_sq))
        .and_then(|v| v.checked_mul(factor_l2_sq_max))
        .and_then(|v| v.checked_mul(factor_l2_sq_max))
        .and_then(|v| v.checked_mul(lambda_outer))
        .and_then(|v| v.checked_mul(lambda_inner))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_linf_tensor_tail_bound_sq: t*² overflows u128".to_string(),
            )
        })
}

fn checked_grind_miss(
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    if grind_target_accept_den == 0
        || grind_target_accept_num == 0
        || grind_target_accept_num >= grind_target_accept_den
    {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_linf_tensor_tail_bound_sq: invalid grind target probability".to_string(),
        ));
    }
    Ok(grind_target_accept_den - grind_target_accept_num)
}

fn fold_witness_linf_tensor_outer_ln(
    num_fold_coeffs: u128,
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    let miss = checked_grind_miss(grind_target_accept_num, grind_target_accept_den)?;
    let numerator = 4u128
        .checked_mul(num_fold_coeffs)
        .and_then(|v| v.checked_mul(grind_target_accept_den))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_linf_tensor_outer_ln: ln numerator overflows u128".to_string(),
            )
        })?;
    Ok(ceil_natural_log(numerator.div_ceil(miss)))
}

fn fold_witness_linf_tensor_inner_ln(
    num_fold_coeffs: u128,
    num_claims: u128,
    tensor_len: u128,
    factor_nonzero_count_max: u128,
    grind_target_accept_num: u128,
    grind_target_accept_den: u128,
) -> Result<u128, AkitaError> {
    let miss = checked_grind_miss(grind_target_accept_num, grind_target_accept_den)?;
    let numerator = 4u128
        .checked_mul(num_fold_coeffs)
        .and_then(|v| v.checked_mul(num_claims))
        .and_then(|v| v.checked_mul(tensor_len))
        .and_then(|v| v.checked_mul(factor_nonzero_count_max))
        .and_then(|v| v.checked_mul(grind_target_accept_den))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_linf_tensor_inner_ln: ln numerator overflows u128".to_string(),
            )
        })?;
    Ok(ceil_natural_log(numerator.div_ceil(miss)))
}

pub fn fold_witness_linf_tail_bound_for_config_sq(
    r_vars: usize,
    num_claims: usize,
    witness_linf_sq: u128,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<u128, AkitaError> {
    let num_blocks = 1usize.checked_shl(r_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "fold_witness_linf_tail_bound_for_config_sq: r_vars too large".to_string(),
        )
    })?;
    let num_fold_blocks = (num_claims as u128)
        .checked_mul(num_blocks as u128)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_linf_tail_bound_for_config_sq: num_fold_blocks overflows u128"
                    .to_string(),
            )
        })?;
    match cap_config.policy {
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => Err(AkitaError::InvalidSetup(
            "fold_witness_linf_tail_bound_for_config_sq: deterministic policy has no tail bound"
                .to_string(),
        )),
        FoldWitnessLinfCapPolicy::TailBoundWithGrind => fold_witness_linf_tail_bound_sq(
            num_fold_blocks,
            cap_config.challenge_l2_sq_max,
            witness_linf_sq,
            cap_config.grind_union_ln,
        ),
        FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => {
            let (left_len, right_len) = tensor_split(num_blocks)?;
            fold_witness_linf_tensor_tail_bound_sq(
                num_claims as u128,
                left_len as u128,
                right_len as u128,
                cap_config.tensor_factor_l2_sq_max,
                cap_config.tensor_factor_nonzero_count_max,
                witness_linf_sq,
                cap_config.num_fold_coeffs,
                cap_config.grind_target_accept_num,
                cap_config.grind_target_accept_den,
            )
        }
    }
}

/// Level-static configuration for [`super::norm_bound::fold_witness_linf_digit_plan`] inside
/// [`crate::sis::num_digits_fold`].
///
/// When the policy is [`WorstCaseBetaOnly`](FoldWitnessLinfCapPolicy::WorstCaseBetaOnly),
/// tail-bound fields are ignored and sizing uses `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldWitnessLinfCapConfig {
    pub policy: FoldWitnessLinfCapPolicy,
    /// Family worst-case `max ‖c‖_2²` (per logical block); see
    /// [`TensorChallengeShape::effective_l2_sq_max`].
    pub challenge_l2_sq_max: u128,
    /// Tensor factor worst-case `max ‖c‖_2²`; only used by
    /// [`FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind`].
    pub tensor_factor_l2_sq_max: u128,
    /// Tensor factor support bound; only used by
    /// [`FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind`].
    pub tensor_factor_nonzero_count_max: u128,
    pub num_fold_coeffs: u128,
    /// Grind reroll target `p_grind` (`NUM / DEN`); copied from
    /// [`crate::FoldLinfProtocolBinding`] at level construction time.
    pub grind_target_accept_num: u128,
    pub grind_target_accept_den: u128,
    /// Precomputed flat union ln term, or tensor outer ln term.
    pub grind_union_ln: u128,
}

impl FoldWitnessLinfCapConfig {
    /// Worst-case `β_inf` sizing only (no tail certificate).
    #[inline]
    pub const fn worst_case_beta_only() -> Self {
        Self {
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            challenge_l2_sq_max: 0,
            tensor_factor_l2_sq_max: 0,
            tensor_factor_nonzero_count_max: 0,
            num_fold_coeffs: 0,
            grind_target_accept_num: 0,
            grind_target_accept_den: 1,
            grind_union_ln: 0,
        }
    }

    /// Tail-aware sizing inputs for a fold level from its sparse family, shape,
    /// ring degree, and inner A-matrix width (`block_len · δ_commit`).
    #[inline]
    pub fn for_fold_level(
        fold_challenge_config: &SparseChallengeConfig,
        fold_challenge_shape: TensorChallengeShape,
        ring_dimension: usize,
        inner_width: usize,
    ) -> Result<Self, AkitaError> {
        let (grind_target_accept_num, grind_target_accept_den) =
            crate::FoldLinfProtocolBinding::CURRENT.grind_target_accept_prob();
        let policy = fold_witness_linf_cap_policy(
            fold_challenge_config,
            fold_challenge_shape,
            ring_dimension,
        );
        Self::assemble(
            policy,
            fold_challenge_config,
            fold_challenge_shape,
            ring_dimension,
            inner_width,
            grind_target_accept_num,
            grind_target_accept_den,
        )
    }

    /// Build a tail-aware config for [`crate::layout::digit_math::optimal_m_r_split`] scoring.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn for_fold_level_scoring(
        policy: FoldWitnessLinfCapPolicy,
        fold_challenge_config: &SparseChallengeConfig,
        fold_challenge_shape: TensorChallengeShape,
        ring_dimension: usize,
        inner_width: usize,
        grind_target_accept_num: u128,
        grind_target_accept_den: u128,
    ) -> Result<Self, AkitaError> {
        Self::assemble(
            policy,
            fold_challenge_config,
            fold_challenge_shape,
            ring_dimension,
            inner_width,
            grind_target_accept_num,
            grind_target_accept_den,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        policy: FoldWitnessLinfCapPolicy,
        fold_challenge_config: &SparseChallengeConfig,
        fold_challenge_shape: TensorChallengeShape,
        ring_dimension: usize,
        inner_width: usize,
        grind_target_accept_num: u128,
        grind_target_accept_den: u128,
    ) -> Result<Self, AkitaError> {
        let num_fold_coeffs = (inner_width as u128).saturating_mul(ring_dimension as u128);
        let grind_union_ln = match policy {
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => 0,
            FoldWitnessLinfCapPolicy::TailBoundWithGrind => fold_witness_linf_grind_union_ln(
                num_fold_coeffs,
                grind_target_accept_num,
                grind_target_accept_den,
            )?,
            FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => {
                fold_witness_linf_tensor_outer_ln(
                    num_fold_coeffs,
                    grind_target_accept_num,
                    grind_target_accept_den,
                )?
            }
        };
        Ok(Self {
            policy,
            challenge_l2_sq_max: fold_challenge_shape.effective_l2_sq_max(fold_challenge_config),
            tensor_factor_l2_sq_max: match fold_challenge_shape {
                TensorChallengeShape::Flat => 0,
                TensorChallengeShape::Tensor => fold_challenge_config.challenge_l2_sq_max(),
            },
            tensor_factor_nonzero_count_max: match fold_challenge_shape {
                TensorChallengeShape::Flat => 0,
                TensorChallengeShape::Tensor => fold_challenge_config.nonzero_count_max() as u128,
            },
            num_fold_coeffs,
            grind_target_accept_num,
            grind_target_accept_den,
            grind_union_ln,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::norm_bound::{FoldChallengeNorms, FoldWitnessNorms};

    #[test]
    fn fold_witness_linf_tail_bound_sq_monotone_and_clamped_inputs() {
        let base = fold_witness_linf_tail_bound_sq(16, 71, 1, 24).unwrap();
        assert!(fold_witness_linf_tail_bound_sq(32, 71, 1, 24).unwrap() >= base);
        assert!(fold_witness_linf_tail_bound_sq(16, 71, 4, 24).unwrap() >= base);
        assert!(fold_witness_linf_tail_bound_sq(0, 71, 1, 24).is_err());
    }

    #[test]
    fn fold_witness_linf_cap_policy_certifies_production_flat_and_tensor_families() {
        let shell = SparseChallengeConfig {
            count_pm1: akita_challenges::D64_PRODUCTION_PM1_COUNT,
            count_pm2: akita_challenges::D64_PRODUCTION_PM2_COUNT,
        };
        assert_eq!(
            fold_witness_linf_cap_policy(&shell, TensorChallengeShape::Flat, 64),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        );
        assert_eq!(
            fold_witness_linf_cap_policy(&shell, TensorChallengeShape::Tensor, 64),
            FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
        );
        let uni = SparseChallengeConfig::pm1_only(31);
        assert_eq!(
            fold_witness_linf_cap_policy(&uni, TensorChallengeShape::Flat, 128),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        );
        assert_eq!(
            fold_witness_linf_cap_policy(&uni, TensorChallengeShape::Tensor, 128),
            FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
        );
        let ladder512 = SparseChallengeConfig::production_for_ring_dim(512).expect("ladder");
        assert_eq!(
            fold_witness_linf_cap_policy(&ladder512, TensorChallengeShape::Flat, 512),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        );
        let uncertified = SparseChallengeConfig::pm1_only(31);
        assert_eq!(
            fold_witness_linf_cap_policy(&uncertified, TensorChallengeShape::Flat, 64),
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
        );
        assert_eq!(
            fold_witness_linf_cap_policy(&uncertified, TensorChallengeShape::Tensor, 64),
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
        );
    }

    #[test]
    fn tensor_tail_bound_matches_hand_formula() {
        let t_sq = fold_witness_linf_tensor_tail_bound_sq(
            1,
            256,
            256,
            31,
            31,
            1,
            1 << 16,
            FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM as u128,
            FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN as u128,
        )
        .unwrap();
        assert_eq!(t_sq, 81_118_363_648);
    }

    #[test]
    fn fold_witness_linf_ln_term_rejects_zero_grind_target() {
        assert!(fold_witness_linf_ln_term(16, 0, 4).is_err());
    }

    #[test]
    fn fold_witness_linf_ln_term_grind_half_matches_ln_4n() {
        let term_16 = fold_witness_linf_ln_term(1 << 16, 1, 2).unwrap();
        assert!((13..=15).contains(&term_16));
        let term_max = fold_witness_linf_ln_term(1u128 << 32, 1, 2).unwrap();
        assert!((24..=26).contains(&term_max));
    }

    #[test]
    fn fold_witness_linf_ln_term_grind_eighth_matches_direct_union_ln_at_2_16() {
        let n = 1u128 << 16;
        let eighth = fold_witness_linf_ln_term(n, 1, 8).unwrap();
        let grind_only = fold_witness_linf_grind_union_ln(n, 1, 8).unwrap();
        assert_eq!(eighth, grind_only);
        assert_eq!(grind_only, 13, "ceil_ln(2·2^16·8/7)");
    }

    #[test]
    fn fold_witness_linf_ln_term_grind_eighth_is_tighter_than_half() {
        let n = 100u128;
        let half = fold_witness_linf_ln_term(n, 1, 2).unwrap();
        let eighth = fold_witness_linf_ln_term(n, 1, 8).unwrap();
        assert!(eighth < half, "eighth={eighth} half={half}");
        let t_half = fold_witness_linf_tail_bound_sq(1, 71, 1, half).unwrap();
        let t_eighth = fold_witness_linf_tail_bound_sq(1, 71, 1, eighth).unwrap();
        assert!(t_eighth < t_half);
    }

    #[test]
    fn threshold_t_star_below_pessimistic_linf_envelope_at_production_shell() {
        use crate::layout::digit_math::isqrt_ceil;

        let challenge = FoldChallengeNorms {
            infinity_norm: 2,
            l1_norm: 51,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let (_, tight_beta) = crate::sis::fold_witness_linf_digit_plan(
            4,
            1,
            128,
            3,
            challenge,
            witness,
            &FoldWitnessLinfCapConfig::worst_case_beta_only(),
        )
        .unwrap();
        let pessimistic_linf_envelope = 16u128 * challenge.l1_norm * witness.infinity_norm();
        assert!(tight_beta < pessimistic_linf_envelope);
        let ln_term = fold_witness_linf_ln_term(1u128 << 16, 1, 8).unwrap();
        let t_sq = fold_witness_linf_tail_bound_sq(16, 71, 1, ln_term).unwrap();
        let t = isqrt_ceil(t_sq);
        assert!(
            t < pessimistic_linf_envelope,
            "t* = {t} pessimistic envelope = {pessimistic_linf_envelope}"
        );
        assert_eq!(t.min(tight_beta), tight_beta);
    }
}
