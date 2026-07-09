//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! [`rounded_up_collision_inf_norm`] returns the audited SIS coefficient
//! `L∞` bucket ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{ceil_supported_linf_bound, min_secure_rank, SisModulusFamily, SisTableKey};
use super::decomposition_digits::{
    balanced_digit_abs_max, fold_witness_representable_linf_bounds, num_digits_for_bound,
};
use crate::layout::digit_math::isqrt_ceil;
use crate::{DecompositionParams, FoldLinfProtocolBinding};

pub use super::fold_linf_cap::{
    fold_witness_linf_cap_policy, fold_witness_linf_tail_bound_for_config_sq,
    fold_witness_linf_tail_bound_sq,
    fold_witness_linf_tensor_tail_bound_sq, FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};

/// Rounded-up SIS infinity norm when adding/subtracting two small digits. A
/// small digit is a digit that is between `-(basis/2)` and `basis/2 - 1`.
/// Therefore, the largest abs value of their subtraction is `basis - 1`.
pub fn rounded_up_collision_inf_norm(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    ring_dimension: usize,
    log_basis: u32,
) -> Option<u128> {
    let linf = 1u128.checked_shl(log_basis)?.checked_sub(1)?;
    ceil_supported_linf_bound(min_security_bits, sis_family, ring_dimension as u32, linf)
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
/// is `c_bar = 2 · challenge_l1_mass` and the digit envelope is
/// `z_bar = 2 · balanced_digit_abs_max`, then rounds up to the audited
/// bucket.
///
/// Returns `None` on overflow or when the collision exceeds every audited bucket
/// for `(sis_family, ring_dimension)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rounded_up_role_a_inf_norm(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    ring_dimension: u32,
    challenge_l1_mass: u128,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
    decomposition: DecompositionParams,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Option<u128> {
    let (fold_decomposed_digits, _) = fold_witness_digit_plan(
        r_vars,
        num_claims,
        decomposition.field_bits(),
        decomposition.log_basis,
        challenge,
        witness,
        cap_config,
    )
    .ok()?;
    let recomposed_inf_norm_bound =
        balanced_digit_abs_max(decomposition.log_basis, fold_decomposed_digits);
    let collision_linf = weak_binding_inf_norm(
        2u128.checked_mul(challenge_l1_mass)?,
        ring_subfield_norm_bound,
        2u128.checked_mul(recomposed_inf_norm_bound)?,
    )?;
    ceil_supported_linf_bound(
        min_security_bits,
        sis_family,
        ring_dimension,
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

/// Build the `beta_inf` envelope norms for one fold level from config and shape.
#[inline]
#[must_use]
pub fn fold_challenge_norms(
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
) -> FoldChallengeNorms {
    FoldChallengeNorms {
        infinity_norm: fold_shape.effective_infinity_norm(fold_challenge_config) as u128,
        l1_norm: fold_shape.effective_l1_mass(fold_challenge_config) as u128,
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

/// Integer retain floor `⌊retain_num/retain_den · t*⌋`, clamped to at least 1.
#[must_use]
pub fn snap_min_tstar_retain_floor(t_star: u128, retain_num: u128, retain_den: u128) -> u128 {
    if t_star == 0 || retain_den == 0 {
        return 1;
    }
    (t_star.saturating_mul(retain_num) / retain_den).max(1)
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
    r_vars: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<(usize, u128), AkitaError> {
    if r_vars >= 127 {
        return Err(AkitaError::InvalidSetup(format!(
            "fold_witness_digit_plan: r_vars = {r_vars} >= 127"
        )));
    }
    // Pre-snap honest-prover L∞ cap. Worst-case negacyclic ring-product L∞ of
    // `c · s` is `min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`, so
    // `β_inf = num_claims · 2^r_vars · that min side`.
    let beta = challenge
        .infinity_norm
        .saturating_mul(witness.l1_norm)
        .min(challenge.l1_norm.saturating_mul(witness.infinity_norm))
        .checked_mul(num_claims as u128)
        .and_then(|t| t.checked_mul(1u128 << r_vars))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_digit_plan: folded-witness bound β overflows u128".to_string(),
            )
        })?;
    if beta == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_digit_plan: folded-witness bound β = 0".to_string(),
        ));
    }
    // Tail-bound policies cap at `min(β_inf, t*)` and expose `t*`; the worst-case
    // policy caps at `β_inf` alone.
    let (pre_snap_cap, t_star) = match cap_config.policy {
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => (beta, None),
        FoldWitnessLinfCapPolicy::TailBoundWithGrind
        | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => {
            let witness_linf_sq = witness
                .infinity_norm()
                .saturating_mul(witness.infinity_norm());
            let t_sq = fold_witness_linf_tail_bound_for_config_sq(
                r_vars,
                num_claims,
                witness_linf_sq,
                cap_config,
            )?;
            let t_star = isqrt_ceil(t_sq);
            (beta.min(t_star), Some(t_star))
        }
    };
    let log_cap = (128 - pre_snap_cap.leading_zeros()).saturating_add(1);
    let delta_base = num_digits_for_bound(log_cap, field_bits, log_basis);

    let binding = FoldLinfProtocolBinding::CURRENT;
    let snap_retain_num = u128::from(binding.snap_min_tstar_retain_num);
    let snap_retain_den = u128::from(binding.snap_min_tstar_retain_den);
    // Optional digit snap-down: walk `δ_fold` downward while the symmetric
    // honest-prover digit envelope at `δ-1` still clears
    // `retain_num/retain_den · t*`.
    let (decomposed_fold_digits, inf_norm_bound) = match (cap_config.policy, t_star) {
        (
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
            | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
            Some(t_star),
        ) if snap_retain_den > 0 && delta_base > 1 && t_star > 0 => {
            let floor = snap_min_tstar_retain_floor(t_star, snap_retain_num, snap_retain_den);
            let mut delta = delta_base;
            let mut grind_cap = pre_snap_cap;
            while delta > 1 {
                let (_, positive_lower) =
                    fold_witness_representable_linf_bounds(log_basis, delta - 1);
                if positive_lower < floor {
                    break;
                }
                delta -= 1;
                let (_, positive_at) = fold_witness_representable_linf_bounds(log_basis, delta);
                grind_cap = pre_snap_cap.min(positive_at);
            }
            (delta, grind_cap)
        }
        _ => (delta_base, pre_snap_cap),
    };
    Ok((decomposed_fold_digits, inf_norm_bound))
}

/// A-role collision bucket and audited secure rank at one geometry.
///
/// Prices with the effective challenge L1 mass `ω` from `fold_shape` and
/// [`SparseChallengeConfig::l1_norm`]. Returns `(collision_bucket, n_a)`.
#[allow(clippy::too_many_arguments)]
pub fn a_role_rank(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
) -> Option<(u128, usize)> {
    let challenge_l1_mass = fold_shape.effective_l1_mass(fold_challenge_config) as u128;
    if challenge_l1_mass == 0 {
        return None;
    }
    let is_onehot = is_root && decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
    let witness = FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge = fold_challenge_norms(fold_challenge_config, fold_shape);
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        fold_challenge_config,
        fold_shape,
        d,
        inner_width as usize,
    )
    .ok()?;
    let bucket = rounded_up_role_a_inf_norm(
        min_security_bits,
        sis_family,
        d as u32,
        challenge_l1_mass,
        challenge,
        witness,
        r_vars,
        num_claims,
        ring_subfield_norm_bound,
        decomposition,
        &cap_config,
    )?;
    let key = SisTableKey {
        min_security_bits,
        family: sis_family,
        ring_dimension: d as u32,
        coeff_linf_bound: bucket,
    };
    let rank = min_secure_rank(key, inner_width)?;
    Some((bucket, rank))
}

#[cfg(test)]
mod tests {
    use super::super::ajtai_key::DEFAULT_SIS_SECURITY_BITS;
    use super::*;

    #[test]
    fn fold_witness_digit_plan_beta_picks_min_ring_product_side() {
        let beta = |c_inf, c_l1, s_inf, s_l1| {
            fold_witness_digit_plan(
                0,
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

        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config = FoldWitnessLinfCapConfig::worst_case_beta_only();
        let (delta_fold, _) = fold_witness_digit_plan(
            2,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        let z_bound = balanced_digit_abs_max(decomposition.log_basis, delta_fold);
        let collision_linf = 8u128 * challenge.l1_norm * z_bound;
        let envelope = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q32,
            64,
            collision_linf,
        )
        .unwrap();
        assert_eq!(
            rounded_up_role_a_inf_norm(
                DEFAULT_SIS_SECURITY_BITS,
                SisModulusFamily::Q32,
                64,
                challenge.l1_norm,
                challenge,
                witness,
                2,
                1,
                1,
                decomposition,
                &cap_config,
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
        let challenge = fold_challenge_norms(&fold_challenge_config, fold_shape);
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(&fold_challenge_config, fold_shape, 64, 2)
                .unwrap();
        let (delta_fold, honest_cap) = fold_witness_digit_plan(
            4,
            1,
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
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q64,
            64,
            challenge.l1_norm,
            challenge,
            witness,
            4,
            1,
            1,
            decomposition,
            &cap_config,
        )
        .unwrap();
        let cap_priced = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q64,
            64,
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
    fn snap_min_tstar_retain_floor_uses_integer_division() {
        assert_eq!(snap_min_tstar_retain_floor(739, 1, 2), 369);
        assert_eq!(snap_min_tstar_retain_floor(739, 5_000, 10_000), 369);
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
        let challenge = fold_challenge_norms(&fold_challenge_config, fold_shape);
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
        let t_star = isqrt_ceil(
            fold_witness_linf_tail_bound_for_config_sq(5, 1, witness_linf_sq, &cap_config).unwrap(),
        );
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

        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config = FoldWitnessLinfCapConfig::worst_case_beta_only();
        let (delta_fold, _) = fold_witness_digit_plan(
            2,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
        )
        .unwrap();
        let z_bound = balanced_digit_abs_max(decomposition.log_basis, delta_fold);
        let priced = rounded_up_role_a_inf_norm(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q32,
            64,
            challenge.l1_norm,
            challenge,
            witness,
            2,
            1,
            1,
            decomposition,
            &cap_config,
        )
        .unwrap();
        assert_eq!(
            priced,
            ceil_supported_linf_bound(
                DEFAULT_SIS_SECURITY_BITS,
                SisModulusFamily::Q32,
                64,
                8 * challenge.l1_norm * z_bound
            )
            .unwrap(),
        );
    }
}
