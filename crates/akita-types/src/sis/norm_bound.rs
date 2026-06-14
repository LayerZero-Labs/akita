//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! `rounded_up_collision_norm_{s,t,w}` return the audited SIS collision *bucket*
//! ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{collision_l2_sq_for_linf_envelope, SisModulusFamily};
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

/// Effective fold-round challenge `(||c||_inf, ||c||_1)` for one level,
/// already accounting for the fold-challenge shape (flat vs tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldChallengeNorms {
    /// Effective challenge L∞ norm `||c||_inf`.
    pub infinity_norm: u128,
    /// Effective challenge L1 norm `||c||_1` (the paper's `ω`).
    pub l1_norm: u128,
}

/// Per-block committed-witness `(||s||_inf, ||s||_1)` for one fold level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A-role committed-level per-ring-row squared Euclidean collision bucket.
///
/// Prices the folded witness sum `z = Σ c_i·s_i` in the L2 MSIS table. Lemma 7
/// gives `‖z_A‖_2 ≤ 8 · op_norm(c) · ‖z‖_2 · ν` on the extracted kernel; until
/// a realized `‖z‖_2` certificate ships (S6+), the deterministic envelope is
/// `‖z‖_inf ≤ β_inf` with `β_inf =` [`fold_witness_beta`], then
/// `‖z‖_2 ≤ √d · β_inf`. MSIS accounting converts the resulting L∞ collision
/// via `‖v‖_2^2 ≤ d · ‖v‖_inf^2`:
///
/// ```text
/// collision_A_inf = 8 · ω · β_inf · ν,
/// collision_l2_sq   = ceil_bucket(d · collision_A_inf^2),
///   ω     = ||c||_1,
///   β_inf = fold_witness_beta(...),
///   ν     = ring_subfield_norm_bound.
/// ```
///
/// Operator-norm rejection (`gamma(c) <= Gamma`) is separate; sizing uses `ω`
/// from the accepted challenge distribution. `β_inf` is the same `‖z‖_inf`
/// envelope as [`fold_witness_beta`] / `num_digits_fold`, not `‖s‖_2`.
///
/// Returns `None` on overflow or when the collision exceeds every audited bucket
/// for `(sis_family, d)`.
#[must_use]
pub fn committed_fold_collision_l2_sq(
    sis_family: SisModulusFamily,
    d: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
) -> Option<u128> {
    let fold_beta = fold_witness_beta(r_vars, num_claims, challenge, witness).ok()?;
    // 2·κ̄·β̄·ν = 2·(2·ω)·(2·fold_beta)·ν = 8·ω·fold_beta·ν.
    let collision_linf = 8u128
        .checked_mul(challenge.l1_norm)?
        .checked_mul(fold_beta)?
        .checked_mul(u128::from(ring_subfield_norm_bound))?;
    collision_l2_sq_for_linf_envelope(sis_family, d, collision_linf)
}

/// A-role (committed witness `s`) rounded-up SIS collision bucket for one
/// committed fold level, per the corrected Hachi Lemma 7 weak-binding bound
/// priced in the L2 MSIS table.
///
/// Builds the level's effective challenge `(||c||_inf, ||c||_1)` and witness
/// `(||s||_inf, ||s||_1)` norms, then converts
/// `collision_A_inf = 8 · ω · fold_witness_beta · ν` into
/// [`committed_fold_collision_l2_sq`]. `r_vars` is the level's fold-arity
/// exponent (`num_blocks = 2^r_vars`); `num_claims` is the batch factor (`> 1`
/// only at a batched root).
///
/// Returns `None` on norm overflow or when the collision exceeds every audited
/// bucket for `(sis_family, d)`.
#[allow(clippy::too_many_arguments)]
pub fn rounded_up_collision_norm_s(
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
) -> Option<u128> {
    let is_onehot = is_root && decomposition.log_commit_bound == 1;
    let witness = FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge = FoldChallengeNorms {
        infinity_norm: fold_shape.effective_infinity_norm(stage1_config) as u128,
        l1_norm: fold_shape.effective_l1_mass(stage1_config) as u128,
    };
    committed_fold_collision_l2_sq(
        sis_family,
        d as u32,
        challenge,
        witness,
        r_vars,
        num_claims,
        ring_subfield_norm_bound,
    )
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

/// Whether [`crate::sis::num_digits_fold`] sizes `K` from the sub-Gaussian tail
/// `t*` (`min(β_inf, t*)`) or from the worst-case envelope `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldLinfThresholdPolicy {
    /// Flat `ExactShell` at `D = 64` or `Uniform{[-1,1]}` at `D ∈ {128, 256}`.
    CertifiedFlat,
    /// Tensor folds, `BoundedL1Norm`, and uncertified flat presets.
    DeterministicBetaInf,
}

/// Select the fold-linf threshold policy for a stage-1 sparse family at ring
/// degree `ring_dimension` with the given fold-challenge shape.
#[inline]
#[must_use]
pub fn fold_linf_threshold_policy(
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    ring_dimension: usize,
) -> FoldLinfThresholdPolicy {
    if !matches!(fold_shape, TensorChallengeShape::Flat) {
        return FoldLinfThresholdPolicy::DeterministicBetaInf;
    }
    match stage1_config {
        SparseChallengeConfig::ExactShell { .. } if ring_dimension == 64 => {
            FoldLinfThresholdPolicy::CertifiedFlat
        }
        SparseChallengeConfig::Uniform { nonzero_coeffs, .. }
            if (ring_dimension == 128 || ring_dimension == 256)
                && nonzero_coeffs.iter().all(|c| c.unsigned_abs() == 1) =>
        {
            FoldLinfThresholdPolicy::CertifiedFlat
        }
        _ => FoldLinfThresholdPolicy::DeterministicBetaInf,
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

/// Conservative integer for `ln(4·num_fold_coeffs) + num_fold_blocks·ln(1/p)` with
/// `p = p_num / p_den` the operator-norm acceptance probability (`p = 1` when
/// the cap does not bind).
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `num_fold_coeffs == 0`, `p_den == 0`,
/// `p_num == 0`, or `p_num > p_den`.
pub fn fold_linf_ln_term(
    num_fold_coeffs: u128,
    num_fold_blocks: u128,
    p_num: u128,
    p_den: u128,
) -> Result<u128, AkitaError> {
    if num_fold_coeffs == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_linf_ln_term: num_fold_coeffs must be positive".to_string(),
        ));
    }
    if p_den == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_linf_ln_term: p_den must be positive".to_string(),
        ));
    }
    if p_num == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_linf_ln_term: p_num must be positive".to_string(),
        ));
    }
    if p_num > p_den {
        return Err(AkitaError::InvalidSetup(
            "fold_linf_ln_term: acceptance probability exceeds 1".to_string(),
        ));
    }
    let ln_4n = ceil_natural_log(4u128.saturating_mul(num_fold_coeffs));
    let ln_inv_p = if p_num >= p_den {
        0
    } else {
        let ratio = p_den.div_ceil(p_num);
        num_fold_blocks.saturating_mul(ceil_natural_log(ratio))
    };
    Ok(ln_4n.saturating_add(ln_inv_p))
}

/// Squared `‖z‖_inf` tail bound `t*²` from the sub-Gaussian argument in
/// `specs/fold-linf-rejection.md`:
///
/// ```text
/// t*² = 2 · num_fold_blocks · challenge_l2_sq_max · witness_linf² · ln_term
/// ```
///
/// `ln_term` is a conservative integer for
/// `ln(4·num_fold_coeffs) + num_fold_blocks·ln(1/p)`. The real square root is
/// taken only at digit-sizing boundaries. Digit sizing uses `min(β_inf, t*)`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when any argument is zero or the product
/// overflows `u128`.
pub fn fold_linf_tail_bound_sq(
    num_fold_blocks: u128,
    challenge_l2_sq_max: u128,
    witness_linf_sq: u128,
    ln_term: u128,
) -> Result<u128, AkitaError> {
    if num_fold_blocks == 0 || challenge_l2_sq_max == 0 || witness_linf_sq == 0 || ln_term == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold_linf_tail_bound_sq: arguments must be positive".to_string(),
        ));
    }
    let two = 2u128;
    two.checked_mul(num_fold_blocks)
        .and_then(|v| v.checked_mul(challenge_l2_sq_max))
        .and_then(|v| v.checked_mul(witness_linf_sq))
        .and_then(|v| v.checked_mul(ln_term))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("fold_linf_tail_bound_sq: t*² overflows u128".to_string())
        })
}

/// Inputs for fold-linf-aware gadget digit sizing in [`crate::sis::num_digits_fold`].
///
/// When the policy is [`DeterministicBetaInf`](FoldLinfThresholdPolicy::DeterministicBetaInf),
/// tail-bound fields are ignored and sizing uses `β_inf` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldLinfDigitSizing {
    pub policy: FoldLinfThresholdPolicy,
    /// Family worst-case `max ‖c‖_2²` (per logical block); see
    /// [`SparseChallengeConfig::challenge_l2_sq_max`].
    pub challenge_l2_sq_max: u128,
    pub num_fold_coeffs: u128,
    pub op_norm_accept_p_num: u128,
    pub op_norm_accept_p_den: u128,
}

impl FoldLinfDigitSizing {
    /// Worst-case `β_inf` sizing (tensor folds, non-certified flat presets).
    #[inline]
    pub const fn deterministic() -> Self {
        Self {
            policy: FoldLinfThresholdPolicy::DeterministicBetaInf,
            challenge_l2_sq_max: 0,
            num_fold_coeffs: 0,
            op_norm_accept_p_num: 1,
            op_norm_accept_p_den: 1,
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
    ) -> Self {
        Self {
            policy: fold_linf_threshold_policy(
                stage1_config,
                fold_challenge_shape,
                ring_dimension,
            ),
            challenge_l2_sq_max: fold_challenge_shape.effective_l2_sq_max(stage1_config),
            num_fold_coeffs: (inner_width as u128).saturating_mul(ring_dimension as u128),
            op_norm_accept_p_num: 1,
            op_norm_accept_p_den: 1,
        }
    }
}

/// `‖z‖_inf` cap used for fold digit sizing: `β_inf` or `min(β_inf, ⌈√(t*²)⌉)`
/// under the certified-flat policy.
///
/// # Errors
///
/// Propagates [`fold_linf_ln_term`] / [`fold_linf_tail_bound_sq`] rejections.
pub fn fold_witness_linf_cap(
    beta: u128,
    num_fold_blocks: u128,
    witness_linf_sq: u128,
    sizing: &FoldLinfDigitSizing,
) -> Result<u128, AkitaError> {
    match sizing.policy {
        FoldLinfThresholdPolicy::DeterministicBetaInf => Ok(beta),
        FoldLinfThresholdPolicy::CertifiedFlat => {
            let ln_term = fold_linf_ln_term(
                sizing.num_fold_coeffs,
                num_fold_blocks,
                sizing.op_norm_accept_p_num,
                sizing.op_norm_accept_p_den,
            )?;
            let t_sq = fold_linf_tail_bound_sq(
                num_fold_blocks,
                sizing.challenge_l2_sq_max,
                witness_linf_sq,
                ln_term,
            )?;
            Ok(beta.min(isqrt_ceil(t_sq)))
        }
    }
}

/// Integer ceiling `⌈√x⌉` for `x > 0` without floating point.
#[inline]
pub fn isqrt_ceil(x: u128) -> u128 {
    if x <= 1 {
        return 1;
    }
    let root = integer_sqrt(x);
    if root.saturating_mul(root) < x {
        root.saturating_add(1)
    } else {
        root
    }
}

#[inline]
fn integer_sqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x0 = n / 2;
    let mut x1 = (x0 + n / x0) / 2;
    while x1 < x0 {
        x0 = x1;
        x1 = (x0 + n / x0) / 2;
    }
    x0
}

// --- L2 MSIS accounting (`l2_sq_from_linf`) ---------------------------------
//
// A-role table lookup uses Lemma 7 plus [`l2_sq_from_linf`] (see
// [`committed_fold_collision_l2_sq`]). The same conversion prices B/D roles.
// Realized `Z_SQUARED = Σ z[row][coeff]²` certificates (S6+) are proved in
// protocol code, not sized here.

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

        let challenge = FoldChallengeNorms {
            infinity_norm: 8,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let fold_beta = fold_witness_beta(2, 1, challenge, witness).unwrap();
        let collision_linf = 8u128 * challenge.l1_norm * fold_beta;
        let envelope =
            collision_l2_sq_for_linf_envelope(SisModulusFamily::Q32, 64, collision_linf).unwrap();
        assert_eq!(
            committed_fold_collision_l2_sq(SisModulusFamily::Q32, 64, challenge, witness, 2, 1, 1,)
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
    fn fold_linf_tail_bound_sq_monotone_and_clamped_inputs() {
        let base = fold_linf_tail_bound_sq(16, 78, 1, 24).unwrap();
        assert!(fold_linf_tail_bound_sq(32, 78, 1, 24).unwrap() >= base);
        assert!(fold_linf_tail_bound_sq(16, 78, 4, 24).unwrap() >= base);
        assert!(fold_linf_tail_bound_sq(0, 78, 1, 24).is_err());
    }

    #[test]
    fn fold_linf_threshold_policy_certifies_production_flat_families() {
        use akita_challenges::TensorChallengeShape;

        let shell = SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
            operator_norm_threshold: 54,
        };
        assert_eq!(
            fold_linf_threshold_policy(&shell, TensorChallengeShape::Flat, 64),
            FoldLinfThresholdPolicy::CertifiedFlat,
        );
        let uni = SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        };
        assert_eq!(
            fold_linf_threshold_policy(&uni, TensorChallengeShape::Flat, 128),
            FoldLinfThresholdPolicy::CertifiedFlat,
        );
        assert_eq!(
            fold_linf_threshold_policy(&uni, TensorChallengeShape::Tensor, 128),
            FoldLinfThresholdPolicy::DeterministicBetaInf,
        );
        assert_eq!(
            fold_linf_threshold_policy(
                &SparseChallengeConfig::BoundedL1Norm,
                TensorChallengeShape::Flat,
                32
            ),
            FoldLinfThresholdPolicy::DeterministicBetaInf,
        );
    }

    #[test]
    fn fold_linf_ln_term_rejects_zero_p_num() {
        assert!(fold_linf_ln_term(16, 16, 0, 1).is_err());
    }

    #[test]
    fn fold_linf_ln_term_p_one_matches_ln_4n() {
        let term_16 = fold_linf_ln_term(1 << 16, 16, 1, 1).unwrap();
        assert!((13..=15).contains(&term_16));
        let term_max = fold_linf_ln_term(1u128 << 32, 16, 1, 1).unwrap();
        assert!((24..=26).contains(&term_max));
    }

    #[test]
    fn threshold_t_star_below_pessimistic_linf_envelope_at_production_shell() {
        let challenge = FoldChallengeNorms {
            infinity_norm: 2,
            l1_norm: 54,
        };
        let witness = FoldWitnessNorms::new(3, 64, 64, true);
        let tight_beta = fold_witness_beta(4, 1, challenge, witness).unwrap();
        let pessimistic_linf_envelope = 16u128 * challenge.l1_norm * witness.infinity_norm();
        assert!(tight_beta < pessimistic_linf_envelope);
        let ln_term = fold_linf_ln_term(1u128 << 16, 16, 1, 1).unwrap();
        let t_sq = fold_linf_tail_bound_sq(16, 78, 1, ln_term).unwrap();
        let t = isqrt_ceil(t_sq);
        assert!(
            t < pessimistic_linf_envelope,
            "t* = {t} pessimistic envelope = {pessimistic_linf_envelope}"
        );
        // Digit sizing will use `min(tight_beta, t*)`; here `t*` exceeds the tight bound.
        assert_eq!(t.min(tight_beta), tight_beta);
    }
}
