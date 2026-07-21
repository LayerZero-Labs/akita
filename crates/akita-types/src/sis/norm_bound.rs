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
use super::decomposition_digits::{
    balanced_digit_abs_max, balanced_digit_max, num_digits_for_bound,
};
use crate::layout::digit_math::isqrt_ceil;
use crate::{DecompositionParams, FoldLinfProtocolBinding};

pub use super::fold_linf_cap::{
    fold_witness_linf_cap_policy, rademacher_proxy_variance,
    rademacher_proxy_variance_flat_challenges, rademacher_proxy_variance_tensor_challenges,
    FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN,
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

/// Weak-binding lemma `L∞` norm bound:
/// `2 * challenge_l1_norm * ring_subfield_norm_bound * z_inf_norm`.
pub fn weak_binding_inf_norm(
    challenge_l1_norm: u128,
    ring_subfield_norm_bound: u32,
    z_inf_norm: u128,
) -> Option<u128> {
    2u128
        .checked_mul(challenge_l1_norm)?
        .checked_mul(u128::from(ring_subfield_norm_bound))?
        .checked_mul(z_inf_norm)
}

/// A-role committed-level coefficient-`L∞` collision bucket.
///
/// Prices the folded witness sum `z = Σ c_i·s_i` in the L∞ MSIS table. Lemma 7
/// bounds the extracted kernel by challenge mass; stage-1 digit membership
/// accepts every balanced `δ_fold`-digit string, whose absolute coefficient
/// envelope is [`balanced_digit_abs_max`] at the `δ_fold` depth
/// induced by [`fold_witness_digit_plan`]. MSIS accounting prices the
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
    sis_modulus_profile: SisModulusProfileId,
    d: usize,
    witness_decomposition: DecompositionParams,
    log_basis_response: u32,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    num_live_blocks: usize,
    num_claims: usize,
    inner_width: u64,
) -> Option<u128> {
    let challenge = FoldChallengeNorms::new(fold_challenge_config, fold_shape);
    let is_onehot = is_root && witness_decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
    let witness = FoldWitnessNorms::new(
        witness_decomposition.log_basis,
        d,
        onehot_chunk_size,
        is_onehot,
    );
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        fold_challenge_config,
        fold_shape,
        d,
        inner_width as usize,
    )
    .ok()?;
    let (fold_decomposed_digits, _) = fold_witness_digit_plan(
        num_live_blocks,
        num_claims,
        witness_decomposition.field_bits(),
        log_basis_response,
        challenge,
        witness,
        &cap_config,
    )
    .ok()?;
    let recomposed_inf_norm_bound =
        balanced_digit_abs_max(log_basis_response, fold_decomposed_digits);
    let collision_linf = weak_binding_inf_norm(
        2u128.checked_mul(challenge.l1_norm)?,
        ring_subfield_norm_bound,
        2u128.checked_mul(recomposed_inf_norm_bound)?,
    )?;
    ceil_supported_linf_bound(
        policy,
        SisTableDigest::CURRENT,
        sis_modulus_profile,
        SisMatrixRole::A,
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
    /// `||s||_inf` is `1` for one-hot or `b/2 = 2^(log_basis-1)` for dense
    /// balanced digits; `||s||_1 = nonzeros · ||s||_inf` with
    /// `nonzeros = ceil(D / K)`:
    ///
    /// - dense / full-field        : `K = 1`     ⇒ `nonzeros = D`
    /// - one-hot, chunk size `K ≥ D`: single-chunk ⇒ `nonzeros = 1`
    /// - one-hot, chunk size `K < D`: multi-chunk  ⇒ `nonzeros = D / K`
    #[inline]
    #[must_use]
    pub fn new(
        log_basis: u32,
        ring_dimension: usize,
        onehot_chunk_size: usize,
        is_onehot: bool,
    ) -> Self {
        let (infinity_norm, chunk) = if is_onehot {
            (1u128, onehot_chunk_size)
        } else {
            (1u128 << (log_basis.saturating_sub(1)), 1)
        };
        let nonzeros = (ring_dimension as u128).div_ceil((chunk.max(1)) as u128);
        Self {
            infinity_norm,
            l1_norm: infinity_norm.saturating_mul(nonzeros),
        }
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
pub fn fold_witness_digit_plan(
    num_live_blocks: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<(usize, u128), AkitaError> {
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
    let log_cap = (128 - inf_norm_bound.leading_zeros()).saturating_add(1);
    let mut fold_decomposed_digits = num_digits_for_bound(log_cap, field_bits, log_basis);

    // Optional digit snap-down: walk `δ_fold` downward while the symmetric
    // honest-prover digit envelope at `δ-1` still clears
    // `retain_num/retain_den · t*`.
    if let (
        FoldWitnessLinfCapPolicy::TailBoundWithGrind
        | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
        Some(rademacher_inf_norm_bound),
    ) = (cap_config.policy, rademacher_inf_norm_bound)
    {
        if FoldLinfProtocolBinding::CURRENT.snap_min_tstar_retain_den > 0
            && fold_decomposed_digits > 1
            && rademacher_inf_norm_bound > 0
        {
            // Integer retain floor `⌊retain_num/retain_den · t*⌋`, clamped to at least 1.
            let floor =
                (rademacher_inf_norm_bound.saturating_mul(u128::from(
                    FoldLinfProtocolBinding::CURRENT.snap_min_tstar_retain_num,
                )) / u128::from(FoldLinfProtocolBinding::CURRENT.snap_min_tstar_retain_den))
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

#[cfg(test)]
mod tests {
    use super::super::ajtai_key::DEFAULT_SIS_SECURITY_POLICY;
    use super::*;

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
        // dense (K=1): ||s||_1 = D · b/2 = 64 · 4.
        assert_eq!(FoldWitnessNorms::new(3, 64, 1, false).l1_norm, 64 * 4);
        // one-hot single-chunk (K >= D): nonzeros = 1.
        assert_eq!(FoldWitnessNorms::new(3, 64, 64, true).l1_norm, 1);
        // one-hot multi-chunk (K < D): nonzeros = ceil(D/K) = 8.
        assert_eq!(FoldWitnessNorms::new(3, 64, 8, true).l1_norm, 8);
    }

    #[test]
    fn fold_witness_norm_levels() {
        // One-hot: ||s||_inf = 1. Dense: ||s||_inf = b/2 = 2^(lb-1), the same
        // at root and recursive (the committed witness is a balanced base-b
        // decomposition with digits in [-b/2, b/2-1] at every level).
        assert_eq!(FoldWitnessNorms::new(3, 64, 64, true).infinity_norm, 1);
        assert_eq!(FoldWitnessNorms::new(3, 64, 1, false).infinity_norm, 4); // 2^2
                                                                             // No root/recursive split: dense is b/2 regardless of `is_onehot=false`.
        assert_eq!(FoldWitnessNorms::new(5, 64, 1, false).infinity_norm, 16); // 2^4
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
        let (d, is_root, onehot_chunk_size, num_live_blocks, num_claims, subfield, inner_width) =
            (64usize, true, 64usize, 2usize, 1usize, 1u32, 2u64);

        // Recompute the Lemma-7 envelope from the same primitives the function wires.
        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let is_onehot = is_root && decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
        let witness =
            FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize,
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
        // Weak-binding collision `8 · ω · z` for `subfield = 1`.
        let collision_linf = 8u128 * challenge.l1_norm * z_bound;
        let envelope = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::A,
            d as u32,
            collision_linf,
        )
        .unwrap();
        assert_eq!(
            rounded_up_role_a_inf_norm(
                DEFAULT_SIS_SECURITY_POLICY,
                SisModulusProfileId::Q32Offset99,
                d,
                decomposition,
                decomposition.log_basis,
                &fold_challenge_config,
                fold_shape,
                is_root,
                onehot_chunk_size,
                subfield,
                num_live_blocks,
                num_claims,
                inner_width,
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
        let (d, is_root, onehot_chunk_size, num_live_blocks, num_claims, subfield, inner_width) =
            (64usize, true, 64usize, 4usize, 1usize, 1u32, 2u64);

        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let is_onehot = is_root && decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
        let witness =
            FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize,
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
            SisModulusProfileId::Q64Offset59,
            d,
            decomposition,
            decomposition.log_basis,
            &fold_challenge_config,
            fold_shape,
            is_root,
            onehot_chunk_size,
            subfield,
            num_live_blocks,
            num_claims,
            inner_width,
        )
        .unwrap();
        let cap_priced = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q64Offset59,
            SisMatrixRole::A,
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
        let witness = FoldWitnessNorms::new(3, 64, 1, false);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(&fold_challenge_config, fold_shape, 64, 2)
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
        let (d, is_root, onehot_chunk_size, num_live_blocks, num_claims, subfield, inner_width) =
            (64usize, false, 1usize, 2usize, 1usize, 1u32, 2u64);

        let challenge = FoldChallengeNorms::new(&fold_challenge_config, fold_shape);
        let is_onehot = is_root && decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
        let witness =
            FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
        let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
            &fold_challenge_config,
            fold_shape,
            d,
            inner_width as usize,
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
            SisModulusProfileId::Q32Offset99,
            d,
            decomposition,
            decomposition.log_basis,
            &fold_challenge_config,
            fold_shape,
            is_root,
            onehot_chunk_size,
            subfield,
            num_live_blocks,
            num_claims,
            inner_width,
        )
        .unwrap();
        assert_eq!(
            priced,
            ceil_supported_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q32Offset99,
                SisMatrixRole::A,
                d as u32,
                8 * challenge.l1_norm * z_bound
            )
            .unwrap(),
        );
    }
}
