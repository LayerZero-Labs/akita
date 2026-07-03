//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! `rounded_up_collision_linf_{s,t,w}` return the audited SIS coefficient
//! `L∞` bucket ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{ceil_supported_linf_bound, min_secure_rank, SisModulusFamily, SisTableKey};
use super::decomposition_digits::{
    fold_witness_verifier_linf_bound, num_digits_fold, num_digits_for_bound,
};
use crate::DecompositionParams;

pub use super::fold_linf_cap::{
    fold_witness_linf_cap_policy, fold_witness_linf_ln_term,
    fold_witness_linf_tail_bound_for_config_sq, fold_witness_linf_tail_bound_sq,
    fold_witness_linf_tensor_tail_bound_sq, FoldWitnessLinfCapConfig, FoldWitnessLinfCapPolicy,
    FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN, FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
    MAX_FOLD_GRIND_ATTEMPTS,
};

/// Worst-case `||lhs · rhs||_inf` of a negacyclic ring product, from the
/// per-operand L1/L∞ bounds:
///
/// ```text
/// ||lhs · rhs||_inf  <=  min( ||lhs||_inf · ||rhs||_1 ,  ||lhs||_1 · ||rhs||_inf ).
/// ```
///
/// Saturating arithmetic keeps this panic-free on the verifier-reachable path.
#[inline]
#[must_use]
pub fn ring_product_infinity_norm_bound(
    lhs_infinity_norm: u128,
    lhs_l1_norm: u128,
    rhs_infinity_norm: u128,
    rhs_l1_norm: u128,
) -> u128 {
    lhs_infinity_norm
        .saturating_mul(rhs_l1_norm)
        .min(lhs_l1_norm.saturating_mul(rhs_infinity_norm))
}

/// Smallest integer `s` with `s^2 >= v`.
#[inline]
#[must_use]
pub fn isqrt_ceil(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let mut lo = 1u128;
    let mut hi = v;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid.saturating_mul(mid) <= v {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo.saturating_mul(lo) < v {
        lo.saturating_add(1)
    } else {
        lo
    }
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
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
) -> FoldChallengeNorms {
    FoldChallengeNorms {
        infinity_norm: fold_shape.effective_infinity_norm(stage1_config) as u128,
        l1_norm: fold_shape.effective_l1_mass(stage1_config) as u128,
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

/// Result of sizing `δ_fold` and the grind cap together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldWitnessLinfDigitPlan {
    /// Balanced digit depth for folded witness `z`.
    pub delta_fold: usize,
    /// Honest-prover grind threshold after optional digit snap-down.
    pub grind_cap: u128,
    /// Pre-snap cap `min(β_inf, t*)` (or `β_inf` alone).
    pub pre_snap_cap: u128,
    /// Sub-Gaussian tail cap `t*` when tail-bound-with-grind is active.
    pub t_star: Option<u128>,
}

fn fold_witness_pre_snap_linf_cap(
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<(u128, Option<u128>), AkitaError> {
    let beta = fold_witness_beta(r_vars, num_claims, challenge, witness)?;
    if beta == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_honest_prover_linf_cap: folded-witness bound β = 0".to_string(),
        ));
    }
    let witness_linf_sq = witness
        .infinity_norm()
        .saturating_mul(witness.infinity_norm());
    match cap_config.policy {
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => Ok((beta, None)),
        FoldWitnessLinfCapPolicy::TailBoundWithGrind
        | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind => {
            let t_sq = fold_witness_linf_tail_bound_for_config_sq(
                r_vars,
                num_claims,
                witness_linf_sq,
                cap_config,
            )?;
            let t_star = isqrt_ceil(t_sq);
            Ok((beta.min(t_star), Some(t_star)))
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

/// Walk `δ_fold` downward while the verifier digit envelope at `δ-1` still clears
/// `retain_num/retain_den · t*`.
#[must_use]
pub fn snap_num_digits_fold_down(
    log_basis: u32,
    delta_base: usize,
    pre_snap_cap: u128,
    t_star: u128,
    retain_num: u128,
    retain_den: u128,
) -> (usize, u128) {
    if delta_base <= 1 || t_star == 0 || retain_den == 0 {
        return (delta_base, pre_snap_cap);
    }
    let floor = snap_min_tstar_retain_floor(t_star, retain_num, retain_den);
    let mut delta = delta_base;
    let mut grind_cap = pre_snap_cap;
    while delta > 1 {
        let z_lower = fold_witness_verifier_linf_bound(log_basis, delta - 1);
        if z_lower < floor {
            break;
        }
        delta -= 1;
        let z_at = fold_witness_verifier_linf_bound(log_basis, delta);
        grind_cap = pre_snap_cap.min(z_at);
    }
    (delta, grind_cap)
}

/// Canonical fold-l∞ digit sizing: pre-snap tail cap, optional digit snap-down,
/// and the grind cap aligned with the snapped `δ_fold`.
///
/// # Errors
///
/// Propagates [`fold_witness_beta`] / tail-bound setup errors.
pub fn fold_witness_linf_digit_plan(
    r_vars: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    cap_config: &FoldWitnessLinfCapConfig,
    snap_retain_num: u128,
    snap_retain_den: u128,
) -> Result<FoldWitnessLinfDigitPlan, AkitaError> {
    let (pre_snap_cap, t_star) =
        fold_witness_pre_snap_linf_cap(challenge, witness, r_vars, num_claims, cap_config)?;
    let log_cap = (128 - pre_snap_cap.leading_zeros()).saturating_add(1);
    let delta_base = num_digits_for_bound(log_cap, field_bits, log_basis);
    let (delta_fold, grind_cap) = match (cap_config.policy, t_star) {
        (
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
            | FoldWitnessLinfCapPolicy::TensorTailBoundWithGrind,
            Some(t),
        ) if snap_retain_den > 0 => {
            snap_num_digits_fold_down(
                log_basis,
                delta_base,
                pre_snap_cap,
                t,
                snap_retain_num,
                snap_retain_den,
            )
        }
        _ => (delta_base, pre_snap_cap),
    };
    Ok(FoldWitnessLinfDigitPlan {
        delta_fold,
        grind_cap,
        pre_snap_cap,
        t_star,
    })
}

/// Honest-prover coefficient-`L∞` target for the folded witness `z`.
///
/// Drives grind retries and sizes `δ_fold` via [`super::decomposition_digits::num_digits_fold`].
/// May be below `β_inf` when tail-bound-with-grind is enabled (`min(β_inf, t*)`).
/// Soundness prices A-role collision at [`fold_witness_verifier_linf_bound`] of the
/// resulting `δ_fold`, not at this cap directly.
///
/// # Errors
///
/// Propagates [`fold_witness_beta`] / tail-bound setup errors.
pub fn fold_witness_honest_prover_linf_cap(
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    field_bits: u32,
    log_basis: u32,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<u128, AkitaError> {
    Ok(fold_witness_linf_digit_plan(
        r_vars,
        num_claims,
        field_bits,
        log_basis,
        challenge,
        witness,
        cap_config,
        u128::from(FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM),
        u128::from(FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN),
    )?
    .grind_cap)
}

/// A-role committed-level coefficient-`L∞` collision bucket.
///
/// Prices the folded witness sum `z = Σ c_i·s_i` in the L∞ MSIS table. Lemma 7
/// bounds the extracted kernel by challenge mass; stage-1 digit membership
/// accepts every balanced `δ_fold`-digit string, whose absolute coefficient
/// envelope is [`fold_witness_verifier_linf_bound`] at the `δ_fold` depth
/// induced by [`fold_witness_honest_prover_linf_cap`]. MSIS accounting keeps
/// the natural coefficient-`L∞` collision:
///
/// ```text
/// collision_A_inf = 8 · challenge_l1_mass · fold_witness_verifier_linf_bound · nu,
///   challenge_l1_mass = ω (effective L1 mass per logical block),
///   nu     = ring_subfield_norm_bound.
/// ```
///
/// Returns `None` on overflow or when the collision exceeds every audited bucket
/// for `(sis_family, d)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn committed_fold_collision_linf_bound(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: u32,
    challenge_l1_mass: u128,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
    decomposition: DecompositionParams,
    cap_config: &FoldWitnessLinfCapConfig,
) -> Option<u128> {
    let log_basis = decomposition.log_basis;
    let delta_fold = num_digits_fold(
        r_vars,
        num_claims,
        decomposition.field_bits(),
        log_basis,
        challenge,
        witness,
        *cap_config,
    )
    .ok()?;
    let z_verifier_linf_bound = fold_witness_verifier_linf_bound(log_basis, delta_fold);
    let collision_linf = 8u128
        .checked_mul(challenge_l1_mass)?
        .checked_mul(z_verifier_linf_bound)?
        .checked_mul(u128::from(ring_subfield_norm_bound))?;
    ceil_supported_linf_bound(min_security_bits, sis_family, d, collision_linf)
}

/// A-role committed-fold collision bucket and audited secure rank at one geometry.
///
/// Prices with the effective challenge L1 mass `ω` from `fold_shape` and
/// [`SparseChallengeConfig::l1_norm`]. Returns `(collision_bucket, n_a)`.
#[allow(clippy::too_many_arguments)]
pub fn committed_fold_a_role_rank(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
    r_vars: usize,
    num_claims: usize,
    inner_width: u64,
) -> Option<(u128, usize)> {
    let challenge_l1_mass = fold_shape.effective_l1_mass(stage1_config) as u128;
    if challenge_l1_mass == 0 {
        return None;
    }
    let is_onehot = is_root && decomposition.log_commit_bound == 1 && onehot_chunk_size > 0;
    let witness = FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge = fold_challenge_norms(stage1_config, fold_shape);
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level(
        stage1_config,
        fold_shape,
        d,
        inner_width as usize,
    )
    .ok()?;
    let bucket = committed_fold_collision_linf_bound(
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

/// Next-level witness scoring cost for one fold geometry, matching
/// [`crate::layout::digit_math::optimal_m_r_split`]:
///
/// ```text
///   (1 + n_a) · δ_open · 2^r  +  δ_commit · δ_fold · m_eff
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fold_level_witness_scoring_cost(
    n_a: usize,
    r_vars: usize,
    num_claims: usize,
    inner_width: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
    fold_challenge: FoldChallengeNorms,
    fold_witness: FoldWitnessNorms,
) -> Option<u64> {
    let field_bits = decomposition.field_bits();
    let log_basis = decomposition.log_basis;
    let log_commit_bound = decomposition.log_commit_bound;
    let open_bound = log_commit_bound.max(field_bits);
    let delta_open = num_digits_for_bound(open_bound, field_bits, log_basis) as u64;
    let delta_commit = num_digits_for_bound(log_commit_bound, field_bits, log_basis) as u64;
    let block_len = inner_width.checked_div(delta_commit as usize)?;
    if block_len == 0 {
        return None;
    }
    let num_blocks = 1u64.checked_shl(r_vars as u32)?;
    let m_eff = block_len as u64;
    let cap_policy = fold_witness_linf_cap_policy(stage1_config, fold_shape, ring_dimension);
    let binding = crate::FoldLinfProtocolBinding::CURRENT;
    let (grind_target_accept_num, grind_target_accept_den) = binding.grind_target_accept_prob();
    let cap_config = FoldWitnessLinfCapConfig::for_fold_level_scoring(
        cap_policy,
        stage1_config,
        fold_shape,
        ring_dimension,
        inner_width,
        grind_target_accept_num,
        grind_target_accept_den,
    )
    .ok()?;
    let delta_fold = num_digits_fold(
        r_vars,
        num_claims,
        field_bits,
        log_basis,
        fold_challenge,
        fold_witness,
        cap_config,
    )
    .ok()? as u64;
    let per_block_cost = delta_open.saturating_add((n_a as u64).saturating_mul(delta_open));
    let opening_cost = per_block_cost.saturating_mul(num_blocks);
    let folding_cost = delta_commit
        .saturating_mul(delta_fold)
        .saturating_mul(m_eff);
    Some(opening_cost.saturating_add(folding_cost))
}

/// B-role (`t̂`) rounded-up SIS coefficient-`L∞` collision bucket.
///
/// The natural coefficient-`L∞` opening-digit collision is `2^lb − 1`.
pub fn rounded_up_collision_linf_t(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    let linf = 1u128.checked_shl(log_basis)?.checked_sub(1)?;
    ceil_supported_linf_bound(min_security_bits, sis_family, d as u32, linf)
}

/// D-role (`ŵ`) rounded-up SIS coefficient-`L∞` bucket. Identical bound to the B role.
pub fn rounded_up_collision_linf_w(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    rounded_up_collision_linf_t(min_security_bits, sis_family, d, log_basis)
}

/// Tiered-commitment second-tier (`F`) rounded-up SIS coefficient-`L∞` bucket. The
/// matrix `F` commits the balanced base-`2^log_basis` digits of `u_1 ‖ … ‖ u_f`,
/// so its collision is the same digit-range difference as the B/D roles.
pub fn rounded_up_collision_linf_tiered_commitment(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    rounded_up_collision_linf_t(min_security_bits, sis_family, d, log_basis)
}

/// Deterministic coefficient-`L∞` envelope on the folded witness sum
/// `z = Σ c_i·s_i` (written `β_inf` in specs):
///
/// ```text
/// β_inf = ||z||_inf bound
///       = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf).
/// ```
///
/// # Errors
///
/// Returns `AkitaError::InvalidSetup` when `r_vars >= 127` (a `2^r_vars` fold
/// arity no well-formed level reaches) or when the product overflows `u128`.
#[inline]
pub fn fold_witness_beta(
    r_vars: usize,
    num_claims: usize,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
) -> Result<u128, AkitaError> {
    if r_vars >= 127 {
        return Err(AkitaError::InvalidSetup(format!(
            "fold_witness_beta: r_vars = {r_vars} >= 127"
        )));
    }
    ring_product_infinity_norm_bound(
        challenge.infinity_norm,
        challenge.l1_norm,
        witness.infinity_norm,
        witness.l1_norm,
    )
    .checked_mul(num_claims as u128)
    .and_then(|t| t.checked_mul(1u128 << r_vars))
    .ok_or_else(|| AkitaError::InvalidSetup("fold_witness_beta: β overflows u128".to_string()))
}

// --- Legacy L2 helper (`l2_sq_from_linf`) ------------------------------------
//
// Production SIS pricing now uses coefficient-L∞ table keys directly. This
// helper remains for local arithmetic checks of the old sqrt(d) envelope.

/// Convert a coefficient-`L∞` collision bound to its Euclidean (L2) counterpart
/// via `||v||_2 <= sqrt(d)·||v||_inf`, kept squared and exact:
/// `||v||_2^2 <= d·linf^2`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on `u128` overflow of `d · linf^2`.
#[inline]
pub fn l2_sq_from_linf(d: u128, linf: u128) -> Result<u128, AkitaError> {
    linf.checked_mul(linf)
        .and_then(|sq| sq.checked_mul(d))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("l2_sq_from_linf: d · ||v||_inf^2 overflows u128".to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::super::ajtai_key::DEFAULT_SIS_SECURITY_BITS;
    use super::*;

    #[test]
    fn ring_product_picks_min_side() {
        assert_eq!(ring_product_infinity_norm_bound(2, 8, 4, 10), 20);
        assert_eq!(ring_product_infinity_norm_bound(8, 2, 5, 1), 8);
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
    fn l2_sq_from_linf_matches_sqrt_d_envelope() {
        // B/D-role digit collision 2^lb - 1 at lb=3 is 7; ||v||_2^2 <= d·49.
        assert_eq!(l2_sq_from_linf(64, 7).unwrap(), 64 * 49);
        assert!(l2_sq_from_linf(u128::MAX, u128::MAX).is_err());
    }

    #[test]
    fn committed_fold_collision_linf_bound_matches_lemma7_envelope() {
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
        let delta_fold = num_digits_fold(
            2,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            cap_config,
        )
        .unwrap();
        let z_bound = fold_witness_verifier_linf_bound(decomposition.log_basis, delta_fold);
        let collision_linf = 8u128 * challenge.l1_norm * z_bound;
        let envelope = ceil_supported_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q32,
            64,
            collision_linf,
        )
        .unwrap();
        assert_eq!(
            committed_fold_collision_linf_bound(
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
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_EXACT_SHELL_MAG1,
            D64_PRODUCTION_EXACT_SHELL_MAG2,
        };

        let stage1_config = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
        };
        let fold_shape = TensorChallengeShape::Flat;
        let challenge = fold_challenge_norms(&stage1_config, fold_shape);
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(&stage1_config, fold_shape, 64, 2).unwrap();
        let honest_cap = fold_witness_honest_prover_linf_cap(
            challenge,
            witness,
            4,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            &cap_config,
        )
        .unwrap();
        let delta_fold = num_digits_fold(
            4,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            cap_config,
        )
        .unwrap();
        let z_bound = fold_witness_verifier_linf_bound(decomposition.log_basis, delta_fold);
        assert!(
            z_bound >= honest_cap,
            "verifier envelope {z_bound} must cover honest cap {honest_cap}"
        );
        let digit_priced = committed_fold_collision_linf_bound(
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
    fn snap_num_digits_fold_down_matches_dense_l1_profile() {
        // Dense fp128_d64 L1 profile: δ 6→5, grind cap min(t*, z_ver(5)) = 682.
        let (delta, grind_cap) = snap_num_digits_fold_down(2, 6, 739, 739, 1, 2);
        assert_eq!(delta, 5);
        assert_eq!(grind_cap, 682);
        assert_eq!(grind_cap, fold_witness_verifier_linf_bound(2, delta));
    }

    #[test]
    fn fold_linf_digit_plan_applies_snap_for_tail_bound_levels() {
        use crate::DecompositionParams;
        use akita_challenges::{
            SparseChallengeConfig, TensorChallengeShape, D64_PRODUCTION_EXACT_SHELL_MAG1,
            D64_PRODUCTION_EXACT_SHELL_MAG2,
        };

        let stage1_config = SparseChallengeConfig::ExactShell {
            count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
        };
        let fold_shape = TensorChallengeShape::Flat;
        let challenge = fold_challenge_norms(&stage1_config, fold_shape);
        let witness = FoldWitnessNorms::new(3, 64, 1, false);
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        };
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(&stage1_config, fold_shape, 64, 2).unwrap();
        let plan = fold_witness_linf_digit_plan(
            5,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            &cap_config,
            1,
            2,
        )
        .unwrap();
        let t_star = plan.t_star.expect("tail-bound level should expose t*");
        let delta_unsnapped = num_digits_for_bound(
            (128 - plan.pre_snap_cap.leading_zeros()).saturating_add(1),
            decomposition.field_bits(),
            decomposition.log_basis,
        );
        if plan.delta_fold < delta_unsnapped {
            assert!(plan.grind_cap <= plan.pre_snap_cap);
            assert!(plan.grind_cap >= t_star / 2);
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
        let delta_fold = num_digits_fold(
            2,
            1,
            decomposition.field_bits(),
            decomposition.log_basis,
            challenge,
            witness,
            cap_config,
        )
        .unwrap();
        let z_bound = fold_witness_verifier_linf_bound(decomposition.log_basis, delta_fold);
        let priced = committed_fold_collision_linf_bound(
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
