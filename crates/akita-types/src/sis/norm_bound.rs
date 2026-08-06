//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! [`rounded_up_collision_inf_norm`] returns the audited SIS coefficient
//! `L∞` bucket ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{
    ceil_supported_linf_bound, SisMatrixRole, SisModulusProfileId, SisSecurityPolicyId,
    SisTableDigest,
};
use super::decomposition_digits::balanced_digit_abs_max;
#[cfg(test)]
use super::decomposition_digits::{balanced_digit_max, num_digits_for_bound};
use crate::layout::digit_math::isqrt_ceil;

pub use super::fold_linf_cap::{
    fold_witness_linf_cap_policy, rademacher_proxy_variance,
    rademacher_proxy_variance_flat_challenges, rademacher_proxy_variance_tensor_challenges,
    FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy, FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_DEN,
    FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_NUM, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM, MAX_FOLD_GRIND_ATTEMPTS,
};

/// Rounded-up SIS infinity norm when adding/subtracting two small digits. A
/// small digit is a digit that is between `-(basis/2)` and `basis/2 - 1`.
/// Therefore, the largest abs value of their subtraction is `basis - 1`.
pub fn rounded_up_collision_inf_norm(
    policy: SisSecurityPolicyId,
    sis_modulus_profile: SisModulusProfileId,
    role: SisMatrixRole,
    ring_dimension: usize,
    log_basis: u32,
) -> Option<u128> {
    let linf = 1u128.checked_shl(log_basis)?.checked_sub(1)?;
    ceil_supported_linf_bound(
        policy,
        SisTableDigest::CURRENT,
        sis_modulus_profile,
        role,
        ring_dimension as u32,
        linf,
    )
}

/// Weak-binding lemma physical coefficient-`L∞` norm bound:
/// `2 * challenge_l1_norm * z_inf_norm`.
///
/// Both inputs are physical ring coefficient vectors at this boundary. A
/// logical extension-field to physical-ring embedding factor therefore does
/// not belong in this formula.
pub fn weak_binding_inf_norm(challenge_l1_norm: u128, z_inf_norm: u128) -> Option<u128> {
    2u128
        .checked_mul(challenge_l1_norm)?
        .checked_mul(z_inf_norm)
}

/// Complete A-role collision price for two accepted folded responses.
///
/// Both the fold challenge and the raw response may differ between two valid
/// openings. Their exact symmetric difference intervals contribute one factor
/// of two each; [`weak_binding_inf_norm`] contributes the lemma's outer factor
/// of two.
pub fn role_a_collision_inf_norm_for_response_bound(
    challenge_l1_norm: u128,
    response_linf_bound: u128,
) -> Option<u128> {
    weak_binding_inf_norm(
        challenge_l1_norm.checked_mul(2)?,
        response_linf_bound.checked_mul(2)?,
    )
}

/// Complete physical squared-`L2` A-role collision bound for two accepted
/// folded responses.
///
/// If each response has squared norm at most `response_l2_sq_bound`, the same
/// extraction factors as [`role_a_collision_inf_norm_for_response_bound`]
/// bound the collision length by `8 * challenge_l1_norm * ||z||_2`. Squaring
/// that scale gives `64 * challenge_l1_norm^2 * response_l2_sq_bound`.
///
/// The response bound covers the complete physical coefficient vector across
/// every A-matrix input row. No embedding or matrix-width factor is applied.
#[must_use]
pub fn role_a_collision_l2_sq_for_response_bound(
    challenge_l1_norm: u128,
    response_l2_sq_bound: u128,
) -> Option<u128> {
    64u128
        .checked_mul(challenge_l1_norm.checked_mul(challenge_l1_norm)?)?
        .checked_mul(response_l2_sq_bound)
}

/// Largest raw folded-response `L∞` bound fitting an A-role collision bucket.
///
/// This is the exact integer inverse of
/// [`role_a_collision_inf_norm_for_response_bound`].
pub fn max_response_linf_for_role_a_collision(
    collision_linf_capacity: u128,
    challenge_l1_norm: u128,
) -> Option<u128> {
    let price_per_unit = role_a_collision_inf_norm_for_response_bound(challenge_l1_norm, 1)?;
    collision_linf_capacity.checked_div(price_per_unit)
}

/// A-role committed-level coefficient-`L∞` collision bucket for one exact
/// verifier-accepted fold digit depth.
///
/// Prices the folded witness sum `z = Σ c_i·s_i` in the L∞ MSIS table. Lemma 7
/// bounds the extracted kernel by challenge mass; stage-1 digit membership
/// accepts every balanced `δ_fold`-digit string, whose absolute coefficient
/// envelope is [`balanced_digit_abs_max`] at the selected `δ_fold` depth.
/// MSIS accounting prices the
/// weak-binding collision `2 · c_bar · z_bar · nu`, where the challenge slack
/// is `c_bar = 2 · challenge.l1_norm` and the digit envelope is
/// `z_bar = 2 · balanced_digit_abs_max`, then rounds up to the audited
/// bucket.
///
/// Returns `None` on overflow or when the collision exceeds every audited bucket
/// for `(sis_modulus_profile, ring_dimension)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rounded_up_role_a_inf_norm(
    policy: SisSecurityPolicyId,
    table_digest: SisTableDigest,
    sis_modulus_profile: SisModulusProfileId,
    d: usize,
    log_basis_response: u32,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    fold_decomposed_digits: usize,
) -> Option<u128> {
    let challenge = FoldChallengeNorms::new(fold_challenge_config, fold_shape);
    if log_basis_response == 0 || fold_decomposed_digits == 0 {
        return None;
    }
    let recomposed_inf_norm_bound =
        balanced_digit_abs_max(log_basis_response, fold_decomposed_digits);
    let collision_linf =
        role_a_collision_inf_norm_for_response_bound(challenge.l1_norm, recomposed_inf_norm_bound)?;
    ceil_supported_linf_bound(
        policy,
        table_digest,
        sis_modulus_profile,
        SisMatrixRole::Inner,
        d as u32,
        collision_linf,
    )
}

/// Effective fold-round challenge `(||c||_inf, ||c||_1)` for `beta_inf` sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldChallengeNorms {
    /// Effective challenge L∞ norm `||c||_inf`.
    pub infinity_norm: u128,
    /// Effective challenge L1 norm `||c||_1` (the paper's `ω`).
    pub l1_norm: u128,
}

impl FoldChallengeNorms {
    /// Build the `beta_inf` envelope norms for one fold level from config and shape.
    #[inline]
    #[must_use]
    pub fn new(
        fold_challenge_config: &SparseChallengeConfig,
        fold_shape: TensorChallengeShape,
    ) -> Self {
        Self {
            infinity_norm: fold_shape.effective_infinity_norm(fold_challenge_config) as u128,
            l1_norm: fold_shape.effective_l1_mass(fold_challenge_config) as u128,
        }
    }
}

/// Per-row committed-witness `(||s||_inf, ||s||_1)` for one fold level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldWitnessNorms {
    /// Witness L∞ norm `||s||_inf` (1 for one-hot, `b/2` for dense digits).
    infinity_norm: u128,
    /// Witness L1 norm `||s||_1 = nonzeros · ||s||_inf`.
    l1_norm: u128,
}

impl FoldWitnessNorms {
    /// Build an exact numeric honest-witness estimate for offline planning.
    #[inline]
    #[must_use]
    pub const fn new(infinity_norm: u128, l1_norm: u128) -> Self {
        Self {
            infinity_norm,
            l1_norm,
        }
    }

    /// Validate a nondegenerate numeric planning estimate.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.infinity_norm == 0 || self.l1_norm < self.infinity_norm {
            return Err(AkitaError::InvalidSetup(
                "fold witness norms require 0 < infinity_norm <= l1_norm".into(),
            ));
        }
        Ok(())
    }

    /// Witness L∞ norm `||s||_inf`.
    #[inline]
    #[must_use]
    pub fn infinity_norm(&self) -> u128 {
        self.infinity_norm
    }

    /// Witness L1 norm `||s||_1 = nonzeros · ||s||_inf`.
    #[inline]
    #[must_use]
    pub fn l1_norm(&self) -> u128 {
        self.l1_norm
    }

    /// Per-block committed witness `s` (`(||s||_inf, ||s||_1)`), used to derive
    /// the worst-case `‖z‖_inf` envelope `β_inf` on the fold sum `z = Σ c_i·s_i`.
    ///
    /// `||s||_inf = b/2 = 2^(log_basis-1)` and every ring coefficient may be
    /// nonzero.
    #[inline]
    #[must_use]
    pub fn bounded(log_basis: u32, ring_dimension: usize) -> Self {
        let infinity_norm = 1u128 << (log_basis.saturating_sub(1));
        Self {
            infinity_norm,
            l1_norm: infinity_norm.saturating_mul(ring_dimension as u128),
        }
    }

    /// Sparse-binary witness with at most one nonzero per logical chunk.
    ///
    /// `||s||_inf = 1` and `||s||_1 = ceil(D / K)`.
    #[inline]
    pub fn sparse_binary(ring_dimension: usize, chunk_size: usize) -> Result<Self, AkitaError> {
        if chunk_size == 0 || !chunk_size.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "sparse-binary fold witness chunk size must be a nonzero power of two".into(),
            ));
        }
        Ok(Self {
            infinity_norm: 1,
            l1_norm: (ring_dimension as u128).div_ceil(chunk_size as u128),
        })
    }
}

/// Canonical fold-l∞ digit sizing: pre-snap tail cap, optional digit snap-down,
/// and the grind cap aligned with the snapped `δ_fold`.
///
/// Returns `(decomposed_fold_digits, inf_norm_bound)`, where `inf_norm_bound` is
/// the honest-prover per-coefficient `‖z‖_inf` target after any snap-down.
///
/// # Errors
///
/// Propagates folded-witness bound / tail-bound setup errors.
#[cfg(test)]
pub(crate) fn fold_witness_digit_plan(
    num_live_blocks: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<(usize, u128), AkitaError> {
    let (mut inf_norm_bound, rademacher_inf_norm_bound) = fold_witness_unsnapped_linf_cap(
        num_live_blocks,
        num_claims,
        challenge,
        witness,
        cap_config,
    )?;
    let log_cap = (128 - inf_norm_bound.leading_zeros()).saturating_add(1);
    let mut fold_decomposed_digits = num_digits_for_bound(log_cap, field_bits, log_basis);

    // Optional digit snap-down: walk `δ_fold` downward while the symmetric
    // honest-prover digit envelope at `δ-1` still clears
    // `retain_num/retain_den · t*`.
    //
    // This pre-cutover regression oracle uses the historical field-specific
    // retain floor. Production policy ownership lives in honest_fold_policy.
    if let (
        FoldWitnessLinfCapPolicy::TailBoundWithGrind
        | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
        Some(rademacher_inf_norm_bound),
    ) = (cap_config.policy, rademacher_inf_norm_bound)
    {
        let (retain_num, retain_den): (u32, u32) = if field_bits == 32 { (3, 4) } else { (1, 2) };
        if retain_den > 0 && fold_decomposed_digits > 1 && rademacher_inf_norm_bound > 0 {
            let floor = (rademacher_inf_norm_bound.saturating_mul(u128::from(retain_num))
                / u128::from(retain_den))
            .max(1);
            while fold_decomposed_digits > 1 {
                let positive_lower = balanced_digit_max(log_basis, fold_decomposed_digits - 1);
                if positive_lower < floor {
                    break;
                }
                fold_decomposed_digits -= 1;
                inf_norm_bound = inf_norm_bound.min(positive_lower);
            }
        }
    }
    Ok((fold_decomposed_digits, inf_norm_bound))
}

/// Honest folded-response infinity-norm cap before any digit-boundary snap.
///
/// Terminal responses are encoded as raw centered integers, so their
/// completeness and codec sizing use this exact cap rather than a gadget
/// boundary selected for recursive witnesses.
pub fn fold_witness_unsnapped_linf_cap(
    num_live_blocks: usize,
    num_claims: usize,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<(u128, Option<u128>), AkitaError> {
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_digit_plan: num_live_blocks must be positive".to_string(),
        ));
    }
    // Worst-case negacyclic ring-product L∞ of
    // `c · s` is `min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`, so
    // `β_inf = num_claims · num_live_blocks · that min side`.
    let mut inf_norm_bound = challenge
        .infinity_norm
        .saturating_mul(witness.l1_norm)
        .min(challenge.l1_norm.saturating_mul(witness.infinity_norm))
        .checked_mul(num_claims as u128)
        .and_then(|t| t.checked_mul(num_live_blocks as u128))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_digit_plan: folded-witness bound β overflows u128".to_string(),
            )
        })?;
    if inf_norm_bound == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_digit_plan: folded-witness bound β = 0".to_string(),
        ));
    }
    let rademacher_inf_norm_bound;
    (inf_norm_bound, rademacher_inf_norm_bound) = match cap_config.policy {
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => (inf_norm_bound, None),
        FoldWitnessLinfCapPolicy::TailBoundWithGrind
        | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => {
            let witness_linf_sq = witness
                .infinity_norm()
                .saturating_mul(witness.infinity_norm());
            let rademacher_inf_norm_bound = isqrt_ceil(rademacher_proxy_variance(
                num_live_blocks,
                num_claims,
                witness_linf_sq,
                cap_config,
            )?);
            (
                inf_norm_bound.min(rademacher_inf_norm_bound),
                Some(rademacher_inf_norm_bound),
            )
        }
    };
    Ok((inf_norm_bound, rademacher_inf_norm_bound))
}

#[cfg(test)]
mod tests {
    use super::super::ajtai_key::DEFAULT_SIS_SECURITY_POLICY;
    use super::*;

    #[test]
    fn physical_response_bound_uses_the_complete_difference_interval() {
        let challenge_l1 = 41;
        let response_bound = 7;
        let price = role_a_collision_inf_norm_for_response_bound(challenge_l1, response_bound)
            .expect("collision price");
        assert_eq!(price, 8 * challenge_l1 * response_bound);
        let unit_price = price / response_bound;
        assert_eq!(
            max_response_linf_for_role_a_collision(price + unit_price - 1, challenge_l1,),
            Some(response_bound)
        );
    }

    #[test]
    fn l2_collision_scales_the_complete_physical_norm_once() {
        let challenge_l1 = 51u128;
        let response_l2_sq = 1u128 << 32;
        assert_eq!(
            role_a_collision_l2_sq_for_response_bound(challenge_l1, response_l2_sq),
            Some(64 * challenge_l1 * challenge_l1 * response_l2_sq),
        );
    }

    #[test]
    fn fold_witness_digit_plan_beta_picks_min_ring_product_side() {
        let beta = |c_inf, c_l1, s_inf, s_l1| {
            fold_witness_digit_plan(
                1,
                1,
                128,
                3,
                FoldChallengeNorms {
                    infinity_norm: c_inf,
                    l1_norm: c_l1,
                },
                FoldWitnessNorms {
                    infinity_norm: s_inf,
                    l1_norm: s_l1,
                },
                &FoldWitnessLinfCapConfig::worst_case_beta_only(),
            )
            .map(|(_, beta)| beta)
            .unwrap()
        };
        assert_eq!(beta(2, 8, 4, 10), 20);
        assert_eq!(beta(8, 2, 5, 1), 8);
    }

    #[test]
    fn fold_witness_digit_plan_prices_exact_live_blocks() {
        let challenge = FoldChallengeNorms {
            infinity_norm: 2,
            l1_norm: 8,
        };
        let witness = FoldWitnessNorms {
            infinity_norm: 4,
            l1_norm: 10,
        };
        let beta = |num_live_blocks| {
            fold_witness_digit_plan(
                num_live_blocks,
                1,
                128,
                3,
                challenge,
                witness,
                &FoldWitnessLinfCapConfig::worst_case_beta_only(),
            )
            .map(|(_, beta)| beta)
            .unwrap()
        };

        assert_eq!(beta(5), 100);
        assert_eq!(beta(8), 160);
    }

    #[test]
    fn witness_block_l1_norm_chunks() {
        // Dense: ||s||_1 = D · b/2 = 64 · 4.
        assert_eq!(FoldWitnessNorms::bounded(3, 64).l1_norm, 64 * 4);
        // one-hot single-chunk (K >= D): nonzeros = 1.
        assert_eq!(FoldWitnessNorms::sparse_binary(64, 64).unwrap().l1_norm, 1);
        // one-hot multi-chunk (K < D): nonzeros = ceil(D/K) = 8.
        assert_eq!(FoldWitnessNorms::sparse_binary(64, 8).unwrap().l1_norm, 8);
        assert!(FoldWitnessNorms::sparse_binary(64, 0).is_err());
        assert!(FoldWitnessNorms::sparse_binary(64, 3).is_err());
    }

    #[test]
    fn fold_witness_norm_levels() {
        // One-hot: ||s||_inf = 1. Dense: ||s||_inf = b/2 = 2^(lb-1), the same
        // at root and recursive (the committed witness is a balanced base-b
        // decomposition with digits in [-b/2, b/2-1] at every level).
        assert_eq!(
            FoldWitnessNorms::sparse_binary(64, 64)
                .unwrap()
                .infinity_norm,
            1
        );
        assert_eq!(FoldWitnessNorms::bounded(3, 64).infinity_norm, 4); // 2^2
        assert_eq!(FoldWitnessNorms::bounded(5, 64).infinity_norm, 16); // 2^4
    }

    #[test]
    fn rounded_up_role_a_inf_norm_matches_lemma7_envelope() {
        use crate::DecompositionParams;
        use akita_challenges::{
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_PM1_COUNT,
            D64_PRODUCTION_PM2_COUNT,
        };

        let fold_challenge_config = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        let fold_shape = TensorChallengeShape::Flat;
        // One-hot committed root (`log_commit_bound == 1`); `log_open_bound`
        // sets `field_bits = 128` for a realistic digit plan.
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 1,
            log_open_bound: Some(128),
        };
        let (d, num_live_blocks, num_claims, inner_width) = (64usize, 2usize, 1usize, 2u64);

        // Recompute the Lemma-7 envelope from the same primitives the function wires.
        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let witness = FoldWitnessNorms::sparse_binary(d, 64).unwrap();
        let cap_config = FoldWitnessLinfCapConfig::for_fold_coeffs(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize * d,
        )
        .unwrap();
        let (delta_fold, _) = fold_witness_digit_plan(
            num_live_blocks,
            num_claims,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        let z_bound = balanced_digit_abs_max(decomposition.log_basis, delta_fold);
        // Physical weak-binding collision `8 · ω · z`.
        let collision_linf = 8u128 * challenge.l1_norm * z_bound;
        let envelope = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::Inner,
            d as u32,
            collision_linf,
        )
        .unwrap();
        assert_eq!(
            rounded_up_role_a_inf_norm(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q32Offset99,
                d,
                decomposition.log_basis,
                &fold_challenge_config,
                fold_shape,
                delta_fold,
            )
            .unwrap(),
            envelope,
        );
        assert!(envelope >= collision_linf);
    }

    #[test]
    fn committed_fold_collision_prices_digit_envelope_not_honest_cap() {
        use crate::DecompositionParams;
        use akita_challenges::{
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_PM1_COUNT,
            D64_PRODUCTION_PM2_COUNT,
        };

        let fold_challenge_config = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        let fold_shape = TensorChallengeShape::Flat;
        // One-hot committed root (`log_commit_bound == 1`); `log_open_bound`
        // sets `field_bits = 128` so the tail-bound snap-down engages.
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 1,
            log_open_bound: Some(128),
        };
        let (d, num_live_blocks, num_claims, inner_width) = (64usize, 4usize, 1usize, 2u64);

        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let witness = FoldWitnessNorms::sparse_binary(d, 64).unwrap();
        let cap_config = FoldWitnessLinfCapConfig::for_fold_coeffs(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize * d,
        )
        .unwrap();
        let (delta_fold, honest_cap) = fold_witness_digit_plan(
            num_live_blocks,
            num_claims,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        let z_bound = balanced_digit_abs_max(decomposition.log_basis, delta_fold);
        assert!(
            z_bound >= honest_cap,
            "verifier envelope {z_bound} must cover honest cap {honest_cap}"
        );
        let digit_priced = rounded_up_role_a_inf_norm(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q64Offset59,
            d,
            decomposition.log_basis,
            &fold_challenge_config,
            fold_shape,
            delta_fold,
        )
        .unwrap();
        let cap_priced = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q64Offset59,
            SisMatrixRole::Inner,
            d as u32,
            8u128
                .checked_mul(challenge.l1_norm)
                .unwrap()
                .checked_mul(honest_cap)
                .unwrap(),
        )
        .unwrap();
        assert!(
            digit_priced > cap_priced,
            "digit-priced {digit_priced} must exceed honest-cap-priced {cap_priced}",
        );
    }

    #[test]
    fn fold_linf_digit_plan_applies_snap_for_tail_bound_levels() {
        use crate::DecompositionParams;
        use akita_challenges::{
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_PM1_COUNT,
            D64_PRODUCTION_PM2_COUNT,
        };

        let fold_challenge_config = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        let fold_shape = TensorChallengeShape::Flat;
        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let witness = FoldWitnessNorms::bounded(3, 64);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_coeffs(&fold_challenge_config, fold_shape, 64, 128)
                .unwrap();
        let (delta_fold, inf_norm_bound) = fold_witness_digit_plan(
            5,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        // Recompute the pre-snap cap independently: `t*` from the tail-bound
        // config and `β_inf` from the worst-case plan, so `pre_snap = min(β, t*)`.
        let witness_linf_sq = witness
            .infinity_norm()
            .saturating_mul(witness.infinity_norm());
        let t_star =
            isqrt_ceil(rademacher_proxy_variance(5, 1, witness_linf_sq, &cap_config).unwrap());
        let (_, beta) = fold_witness_digit_plan(
            5,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &FoldWitnessLinfCapConfig::worst_case_beta_only(),
        )
        .unwrap();
        let pre_snap_cap = beta.min(t_star);
        let delta_unsnapped = num_digits_for_bound(
            (128 - pre_snap_cap.leading_zeros()).saturating_add(1),
            decomposition.field_bits(),
            decomposition.log_basis,
        );
        if delta_fold < delta_unsnapped {
            assert!(inf_norm_bound <= pre_snap_cap);
            assert!(inf_norm_bound >= t_star / 2);
        }
    }

    #[test]
    fn committed_fold_collision_uses_num_digits_fold_verifier_bound() {
        use crate::DecompositionParams;
        use akita_challenges::{
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_PM1_COUNT,
            D64_PRODUCTION_PM2_COUNT,
        };

        let fold_challenge_config = SparseChallengeConfig {
            count_pm1: D64_PRODUCTION_PM1_COUNT,
            count_pm2: D64_PRODUCTION_PM2_COUNT,
        };
        let fold_shape = TensorChallengeShape::Flat;
        // Dense recursive witness path (`is_root = false` ⇒ `is_onehot = false`).
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let (d, num_live_blocks, num_claims, inner_width) = (64usize, 2usize, 1usize, 2u64);

        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let witness = FoldWitnessNorms::bounded(decomposition.log_basis, d);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_coeffs(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize * d,
        )
        .unwrap();
        let (delta_fold, _) = fold_witness_digit_plan(
            num_live_blocks,
            num_claims,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        let z_bound = balanced_digit_abs_max(decomposition.log_basis, delta_fold);
        let priced = rounded_up_role_a_inf_norm(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            d,
            decomposition.log_basis,
            &fold_challenge_config,
            fold_shape,
            delta_fold,
        )
        .unwrap();
        assert_eq!(
            priced,
            ceil_supported_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q32Offset99,
                SisMatrixRole::Inner,
                d as u32,
                8 * challenge.l1_norm * z_bound
            )
            .unwrap(),
        );
    }
}
