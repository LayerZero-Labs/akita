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
    /// the fold-response envelope `β_inf` on `z = Σ c_i·s_i`.
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
/// Prices the **fold response** `z = Σ c_i·s_i` in the L2 MSIS table. Lemma 7
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
/// from the accepted challenge distribution. `β_inf` is the same fold-response
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

/// Deterministic coefficient-`L∞` envelope on the fold response
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
}
