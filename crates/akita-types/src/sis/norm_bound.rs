//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! `rounded_up_collision_norm_{s,t,w}` return the audited SIS collision *bucket*
//! ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{collision_l2_sq_for_linf_envelope, min_secure_rank, SisModulusFamily};
use super::decomposition_digits::{
    fold_witness_verifier_linf_bound, num_digits_fold, num_digits_for_bound,
};
use crate::DecompositionParams;

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
    cap_config: &FoldWitnessLinfCapConfig,
) -> Result<u128, AkitaError> {
    let beta = fold_witness_beta(r_vars, num_claims, challenge, witness)?;
    if beta == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_witness_honest_prover_linf_cap: folded-witness bound β = 0".to_string(),
        ));
    }
    let num_fold_blocks = (num_claims as u128)
        .checked_mul(1u128 << r_vars)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "fold_witness_honest_prover_linf_cap: num_fold_blocks overflows u128".to_string(),
            )
        })?;
    let witness_linf_sq = witness
        .infinity_norm()
        .saturating_mul(witness.infinity_norm());
    match cap_config.policy {
        FoldWitnessLinfCapPolicy::WorstCaseBetaOnly => Ok(beta),
        FoldWitnessLinfCapPolicy::TailBoundWithGrind => {
            let t_sq = fold_witness_linf_tail_bound_sq(
                num_fold_blocks,
                cap_config.challenge_l2_sq_max,
                witness_linf_sq,
                cap_config.grind_union_ln,
            )?;
            Ok(beta.min(isqrt_ceil(t_sq)))
        }
    }
}

/// A-role committed-level per-ring-row squared Euclidean collision bucket.
///
/// Prices the folded witness sum `z = Σ c_i·s_i` in the L2 MSIS table. Lemma 7
/// bounds the extracted kernel by challenge mass; stage-1 digit membership
/// accepts every balanced `δ_fold`-digit string, whose absolute coefficient
/// envelope is [`fold_witness_verifier_linf_bound`] at the `δ_fold` depth
/// induced by [`fold_witness_honest_prover_linf_cap`]. MSIS accounting converts
/// the resulting L∞ collision via `‖v‖_2^2 ≤ d · ‖v‖_inf^2`:
///
/// ```text
/// collision_A_inf = 8 · challenge_l1_mass · fold_witness_verifier_linf_bound · nu,
/// collision_l2_sq   = ceil_bucket(d · collision_A_inf^2),
///   challenge_l1_mass = ω (effective L1 mass per logical block),
///   nu     = ring_subfield_norm_bound.
/// ```
///
/// Returns `None` on overflow or when the collision exceeds every audited bucket
/// for `(sis_family, d)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn committed_fold_collision_l2_sq(
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
    collision_l2_sq_for_linf_envelope(sis_family, d, collision_linf)
}

/// A-role committed-fold collision bucket and audited secure rank at one geometry.
///
/// Prices with the effective challenge L1 mass `ω` from `fold_shape` and
/// [`SparseChallengeConfig::l1_norm`]. Returns `(collision_bucket, n_a)`.
#[allow(clippy::too_many_arguments)]
pub fn committed_fold_a_role_rank(
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
    let bucket = committed_fold_collision_l2_sq(
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
    let rank = min_secure_rank(sis_family, d as u32, bucket, inner_width)?;
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

/// B-role (`t̂`) rounded-up SIS collision bucket via `||v||_2^2 <= d·||v||_inf^2`.
///
/// The natural coefficient-`L∞` opening-digit collision is `2^lb − 1`.
pub fn rounded_up_collision_norm_t(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    let linf = 1u128.checked_shl(log_basis)?.checked_sub(1)?;
    collision_l2_sq_for_linf_envelope(sis_family, d as u32, linf)
}

/// D-role (`ŵ`) rounded-up SIS collision bucket. Identical bound to the B role.
pub fn rounded_up_collision_norm_w(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    rounded_up_collision_norm_t(sis_family, d, log_basis)
}

/// Tiered-commitment second-tier (`F`) rounded-up SIS collision bucket. The
/// matrix `F` commits the balanced base-`2^log_basis` digits of `u_1 ‖ … ‖ u_f`,
/// so its collision is the same digit-range difference as the B/D roles.
pub fn rounded_up_collision_norm_tiered_commitment(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u128> {
    rounded_up_collision_norm_t(sis_family, d, log_basis)
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

/// Maximum Fiat-Shamir rerolls per committed fold level under tail-bound-with-grind policy.
pub const MAX_FOLD_GRIND_ATTEMPTS: u32 = 4096;

/// Per-challenge **grind** acceptance target `p_grind = NUM / DEN` used in the union-bound
/// sizing for `t*` (`specs/fold-linf-rejection.md`).
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_NUM: u32 = 1;
pub const FOLD_LINF_GRIND_TARGET_ACCEPT_PROB_DEN: u32 = 8;

/// Whether [`crate::sis::num_digits_fold`] sizes `K` from the sub-Gaussian tail
/// `t*` (`min(β_inf, t*)`) or from the worst-case envelope `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoldWitnessLinfCapPolicy {
    /// Proved sub-Gaussian tail: flat `ExactShell` at `D = 64` or
    /// `Uniform{[-1,1]}` at `D ∈ {128, 256}`; `cap = min(β_inf, t*)` and grind allowed.
    TailBoundWithGrind,
    /// No tail certificate yet: tensor folds, `BoundedL1Norm`, uncertified flat presets;
    /// `cap = β_inf` only and grind nonce must be zero.
    WorstCaseBetaOnly,
}

/// Select the fold-linf threshold policy for a stage-1 sparse family at ring
/// degree `ring_dimension` with the given fold-challenge shape.
#[inline]
#[must_use]
pub fn fold_witness_linf_cap_policy(
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
) -> FoldWitnessLinfCapPolicy {
    if !matches!(fold_shape, TensorChallengeShape::Flat) {
        return FoldWitnessLinfCapPolicy::WorstCaseBetaOnly;
    }
    match stage1_config {
        SparseChallengeConfig::ExactShell { .. } if ring_dimension == 64 => {
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
        }
        SparseChallengeConfig::Uniform { nonzero_coeffs, .. }
            if (ring_dimension == 128 || ring_dimension == 256)
                && nonzero_coeffs.iter().all(|c| c.unsigned_abs() == 1) =>
        {
            FoldWitnessLinfCapPolicy::TailBoundWithGrind
        }
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
fn fold_witness_linf_grind_union_ln(
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

/// Level-static configuration for [`fold_witness_honest_prover_linf_cap`] inside
/// [`crate::sis::num_digits_fold`].
///
/// When the policy is [`WorstCaseBetaOnly`](FoldWitnessLinfCapPolicy::WorstCaseBetaOnly),
/// tail-bound fields are ignored and sizing uses `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FoldWitnessLinfCapConfig {
    pub policy: FoldWitnessLinfCapPolicy,
    /// Family worst-case `max ‖c‖_2²` (per logical block); see
    /// [`SparseChallengeConfig::challenge_l2_sq_max`].
    pub challenge_l2_sq_max: u128,
    pub num_fold_coeffs: u128,
    /// Grind reroll target `p_grind` (`NUM / DEN`); copied from
    /// [`crate::FoldLinfProtocolBinding`] at level construction time.
    pub grind_target_accept_num: u128,
    pub grind_target_accept_den: u128,
    /// Precomputed grind union ln term for [`FoldWitnessLinfCapPolicy::TailBoundWithGrind`].
    pub grind_union_ln: u128,
}

impl FoldWitnessLinfCapConfig {
    /// Worst-case `β_inf` sizing only (no tail certificate).
    #[inline]
    pub const fn worst_case_beta_only() -> Self {
        Self {
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            challenge_l2_sq_max: 0,
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
        stage1_config: &SparseChallengeConfig,
        fold_challenge_shape: TensorChallengeShape,
        ring_dimension: usize,
        inner_width: usize,
    ) -> Result<Self, AkitaError> {
        let (grind_target_accept_num, grind_target_accept_den) =
            crate::FoldLinfProtocolBinding::CURRENT.grind_target_accept_prob();
        let policy =
            fold_witness_linf_cap_policy(stage1_config, fold_challenge_shape, ring_dimension);
        Self::assemble(
            policy,
            stage1_config,
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
        stage1_config: &SparseChallengeConfig,
        fold_challenge_shape: TensorChallengeShape,
        ring_dimension: usize,
        inner_width: usize,
        grind_target_accept_num: u128,
        grind_target_accept_den: u128,
    ) -> Result<Self, AkitaError> {
        Self::assemble(
            policy,
            stage1_config,
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
        stage1_config: &SparseChallengeConfig,
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
        };
        Ok(Self {
            policy,
            challenge_l2_sq_max: fold_challenge_shape.effective_l2_sq_max(stage1_config),
            num_fold_coeffs,
            grind_target_accept_num,
            grind_target_accept_den,
            grind_union_ln,
        })
    }
}

// --- L2 MSIS accounting (`l2_sq_from_linf`) ---------------------------------
//
// A-role table lookup uses Lemma 7 plus [`l2_sq_from_linf`] (see
// [`committed_fold_collision_l2_sq`]). The same conversion prices B/D roles.

/// Convert a coefficient-`L∞` collision bound to its Euclidean (L2) counterpart
/// via `||v||_2 <= sqrt(d)·||v||_inf`, kept squared and exact:
/// `||v||_2^2 <= d·linf^2`.
///
/// This lets the B-role and D-role opening-digit collisions (natural bound
/// `2^lb - 1`, the difference of two balanced digits) be priced by the same
/// Euclidean MSIS floor as the A-role, rather than a separate `L∞` table.
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
    fn committed_fold_collision_l2_sq_matches_lemma7_conversion() {
        use super::super::ajtai_key::derived_collision_l2_sq_key;
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
        let envelope =
            collision_l2_sq_for_linf_envelope(SisModulusFamily::Q32, 64, collision_linf).unwrap();
        assert_eq!(
            committed_fold_collision_l2_sq(
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
        assert_eq!(
            envelope,
            derived_collision_l2_sq_key(SisModulusFamily::Q32, 64, collision_linf).unwrap(),
        );
        assert!(
            envelope >= l2_sq_from_linf(64, collision_linf).unwrap(),
            "derived bucket ceilings L∞ before squaring",
        );
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
        let honest_cap =
            fold_witness_honest_prover_linf_cap(challenge, witness, 4, 1, &cap_config).unwrap();
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
            z_bound > honest_cap,
            "verifier envelope {z_bound} must exceed honest cap {honest_cap}"
        );
        let digit_priced = committed_fold_collision_l2_sq(
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
        let cap_priced = collision_l2_sq_for_linf_envelope(
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
        let priced = committed_fold_collision_l2_sq(
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
            collision_l2_sq_for_linf_envelope(
                SisModulusFamily::Q32,
                64,
                8 * challenge.l1_norm * z_bound
            )
            .unwrap(),
        );
    }

    #[test]
    fn fold_witness_linf_tail_bound_sq_monotone_and_clamped_inputs() {
        let base = fold_witness_linf_tail_bound_sq(16, 71, 1, 24).unwrap();
        assert!(fold_witness_linf_tail_bound_sq(32, 71, 1, 24).unwrap() >= base);
        assert!(fold_witness_linf_tail_bound_sq(16, 71, 4, 24).unwrap() >= base);
        assert!(fold_witness_linf_tail_bound_sq(0, 71, 1, 24).is_err());
    }

    #[test]
    fn fold_witness_linf_cap_policy_certifies_production_flat_families() {
        use akita_challenges::TensorChallengeShape;

        let shell = SparseChallengeConfig::ExactShell {
            count_mag1: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG1,
            count_mag2: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG2,
        };
        assert_eq!(
            fold_witness_linf_cap_policy(&shell, TensorChallengeShape::Flat, 64),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        );
        let uni = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        assert_eq!(
            fold_witness_linf_cap_policy(&uni, TensorChallengeShape::Flat, 128),
            FoldWitnessLinfCapPolicy::TailBoundWithGrind,
        );
        assert_eq!(
            fold_witness_linf_cap_policy(&uni, TensorChallengeShape::Tensor, 128),
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
        );
        assert_eq!(
            fold_witness_linf_cap_policy(
                &SparseChallengeConfig::BoundedL1Norm,
                TensorChallengeShape::Flat,
                32
            ),
            FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
        );
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
        let challenge = FoldChallengeNorms {
            infinity_norm: 2,
            l1_norm: 51,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let tight_beta = fold_witness_beta(4, 1, challenge, witness).unwrap();
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
