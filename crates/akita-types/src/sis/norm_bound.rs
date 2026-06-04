//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! `rounded_up_collision_norm_{s,t,w}` return the audited SIS collision *bucket*
//! ready to feed [`super::ajtai_key::min_secure_rank`]. The folded witness `z`
//! is decomposed (not Ajtai-committed), so it has no SIS bucket.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use super::ajtai_key::{ceil_supported_collision, SisModulusFamily};
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

    /// Per-block committed-witness `(||s||_inf, ||s||_1)` for the folded witness.
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

/// A-role committed-level weak-binding SIS collision bucket, computed from
/// already-derived per-level fold norms.
///
/// Every Ajtai-committed level (the dense root and all recursive fold levels)
/// is priced at the *fold-response* norm — the only norm the weak-binding
/// extractor certifies for the kernel vector `z_A = c̄'(c̄·s) − c̄(c̄'·s')`:
///
/// ```text
/// collision_A = 2 · κ̄ · β̄ · ν
///   κ̄ = ||c − c'||_1 = 2·ω           (challenge difference; ω = ||c||_1)
///   β̄ = 2 · β^resp                   (extractor bound ||z^(ℓ,i) − z^0||_inf ≤ 2·β^resp)
///   β^resp = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)
/// ```
///
/// so `collision_A = 8 · ω · fold_witness_beta · ν`, where `fold_witness_beta`
/// is exactly [`fold_witness_beta`]. The `num_claims · 2^r_vars` factor is the
/// fold arity an *anchored* per-block price would unsoundly drop: dividing the
/// fold response by the ring unit `c̄` does not recover `||s||_inf`, and the
/// range / booleanity checks bind the honest committed table, not the extracted
/// quotient. One-hotness only shrinks `β^resp` (it sets `||s||_inf = 1`); it
/// does not remove the fold arity, so it is folded into `witness`, not into a
/// regime switch.
///
/// Returns `None` on norm overflow or when the collision exceeds every audited
/// bucket for `(sis_family, d)`.
#[must_use]
pub fn committed_fold_collision_s(
    sis_family: SisModulusFamily,
    d: u32,
    challenge: FoldChallengeNorms,
    witness: FoldWitnessNorms,
    r_vars: usize,
    num_claims: usize,
    ring_subfield_norm_bound: u32,
) -> Option<u32> {
    let fold_beta = fold_witness_beta(r_vars, num_claims, challenge, witness).ok()?;
    // 2·κ̄·β̄·ν = 2·(2·ω)·(2·fold_beta)·ν = 8·ω·fold_beta·ν.
    let collision = 8u128
        .checked_mul(challenge.l1_norm)?
        .checked_mul(fold_beta)?
        .checked_mul(u128::from(ring_subfield_norm_bound))?;
    ceil_supported_collision(sis_family, d, u32::try_from(collision).ok()?)
}

/// A-role (committed witness `s`) rounded-up SIS collision bucket for one
/// committed fold level, per the corrected Hachi Lemma 7 weak-binding bound.
///
/// Builds the level's effective challenge `(||c||_inf, ||c||_1)` and witness
/// `(||s||_inf, ||s||_1)` norms — one-hot roots commit a sparse witness
/// (`||s||_inf = 1`), dense roots and every recursive level commit balanced
/// digits (`||s||_inf = b/2`) — then prices the A collision at the fold response
/// via [`committed_fold_collision_s`]. `r_vars` is the level's fold-arity
/// exponent (`num_blocks = 2^r_vars`); `num_claims` is the batch factor, which
/// is `> 1` only at a batched root and `1` at a singleton root and every
/// recursive level.
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
) -> Option<u32> {
    let is_onehot = is_root && decomposition.log_commit_bound == 1;
    let witness = FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge = FoldChallengeNorms {
        infinity_norm: fold_shape.effective_infinity_norm(stage1_config) as u128,
        l1_norm: fold_shape.effective_l1_mass(stage1_config) as u128,
    };
    committed_fold_collision_s(
        sis_family,
        d as u32,
        challenge,
        witness,
        r_vars,
        num_claims,
        ring_subfield_norm_bound,
    )
}

/// B-role (`t̂`) rounded-up SIS collision bucket. The collision is the direct
/// difference of two balanced-digit openings (no challenge multiplication).
/// Each balanced digit lies in `[−b/2, b/2 − 1]` with `b = 2^lb`, so the
/// largest difference of two such digits is
/// `(b/2 − 1) − (−b/2) = b − 1 = 2^lb − 1`.
pub fn rounded_up_collision_norm_t(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u32> {
    let collision = 1u32.checked_shl(log_basis)?.checked_sub(1)?;
    ceil_supported_collision(sis_family, d as u32, collision)
}

/// D-role (`ŵ`) rounded-up SIS collision bucket. Identical bound to the B role.
pub fn rounded_up_collision_norm_w(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
) -> Option<u32> {
    rounded_up_collision_norm_t(sis_family, d, log_basis)
}

/// Folded-witness `z = Σ c_i·s_i` L∞ bound from precomputed per-level norms:
///
/// ```text
/// β = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf).
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

// --- L2 (Euclidean) folded-witness bound primitives ---------
//
// The L2 MSIS cutover prices the committed A-role against a Euclidean bound on
// the folded response `z = Σ c_i·s_i` instead of the coefficient-`L∞` envelope
// above. The protocol only ever consumes the *squared* bound (the certificate
// proves `Σ z[i]^2 = B_l2` and sizes against `B_l2`), so every primitive here
// stays in the squared, exact-integer domain: `sqrt(D)` is irrational for
// `D ∈ {32, 128}`, and squaring it away keeps the values exact `u128` integers.
// A real square root is taken only when the prover picks the bucket `B_l2` and
// the four-square slack (spec slice S8), never in these sizing helpers.
//
// None of these are wired into rank pricing yet; the L2 SIS table + planner
// cutover (S5, S11) and the prover certificate (S8) are the first consumers.

/// Squared per-block committed-witness Euclidean bound `s_l2_max^2`, the L2
/// analogue of the [`FoldWitnessNorms`] `(||s||_inf, ||s||_1)` pair:
///
/// ```text
/// s_l2_max^2 = D · (b/2)^2   dense balanced digits (||s||_inf = b/2 = 2^(lb-1)),
/// s_l2_max^2 = 1             a one-hot block (a single unit coefficient).
/// ```
///
/// The one-hot value is the spec's per-block contract (one unit coefficient).
/// Per-level policy for multi-chunk one-hot or tensor folds is decided by the
/// caller (S8/S11), not here.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on `u128` overflow of `D · (b/2)^2`.
#[inline]
pub fn s_l2_max_squared(
    is_onehot: bool,
    ring_dimension: u128,
    log_basis: u32,
) -> Result<u128, AkitaError> {
    if is_onehot {
        return Ok(1);
    }
    let half_basis = 1u128 << log_basis.saturating_sub(1); // b/2 = 2^(lb-1)
    half_basis
        .checked_mul(half_basis)
        .and_then(|sq| sq.checked_mul(ring_dimension))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("s_l2_max_squared: D · (b/2)^2 overflows u128".to_string())
        })
}

/// Squared deterministic per-level folded-response bound
/// `beta_l2^2 = (Gamma · B · s_l2_max)^2`, with fold arity
/// `B = num_claims · 2^r_vars` and `gamma` the operator-norm cap on accepted
/// challenges (`gamma(c_i) <= Gamma`):
///
/// ```text
/// ||Σ c_i·s_i||_2 <= Σ ||c_i·s_i||_2 <= Gamma · Σ ||s_i||_2 = Gamma · B · s_l2_max.
/// ```
///
/// Mirrors [`fold_witness_beta`]'s fold-arity guard so the L2 and L∞ betas share
/// the same overflow contract.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `r_vars >= 127` or the squared
/// product overflows `u128`.
#[inline]
pub fn beta_l2_squared(
    r_vars: usize,
    num_claims: usize,
    gamma: u128,
    s_l2_max_squared: u128,
) -> Result<u128, AkitaError> {
    if r_vars >= 127 {
        return Err(AkitaError::InvalidSetup(format!(
            "beta_l2_squared: r_vars = {r_vars} >= 127"
        )));
    }
    let fold_arity = (num_claims as u128)
        .checked_mul(1u128 << r_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("beta_l2_squared: B overflows u128".to_string()))?;
    // (Gamma · B)^2 · s_l2_max^2 = (Gamma · B · s_l2_max)^2 = beta_l2^2.
    gamma
        .checked_mul(fold_arity)
        .and_then(|gb| gb.checked_mul(gb))
        .and_then(|gb_sq| gb_sq.checked_mul(s_l2_max_squared))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("beta_l2_squared: beta_l2^2 overflows u128".to_string())
        })
}

/// Conservative squared Euclidean bound over a vector of `W` folded ring rows:
///
/// ```text
/// L2_BOUND_SQUARED = W · beta_l2^2.
/// ```
///
/// This is the deterministic A-role bound used directly when no realized
/// certificate is emitted (the field-capacity fallback); `B_l2` is the
/// certificate-tightened value in `Z_SQUARED <= B_l2 <= L2_BOUND_SQUARED`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on `u128` overflow of `W · beta_l2^2`.
#[inline]
pub fn l2_bound_squared(width_w: u128, beta_l2_squared: u128) -> Result<u128, AkitaError> {
    width_w.checked_mul(beta_l2_squared).ok_or_else(|| {
        AkitaError::InvalidSetup("l2_bound_squared: W · beta_l2^2 overflows u128".to_string())
    })
}

/// Convert a coefficient-`L∞` collision bound into the unified L2 table via
/// `||v||_2 <= sqrt(d)·||v||_inf`, kept squared and exact: `||v||_2^2 <= d·linf^2`.
///
/// This is how the B-role and D-role opening-digit collisions (natural bound
/// `2^lb - 1`, the difference of two balanced digits) price against the single
/// Euclidean MSIS floor.
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
    fn s_l2_max_squared_onehot_and_dense() {
        // One-hot: a single unit coefficient, ||s||_2^2 = 1, independent of D/lb.
        assert_eq!(s_l2_max_squared(true, 64, 11).unwrap(), 1);
        // Dense lb=3: b/2 = 2^2 = 4, so s_l2_max^2 = D · 16 = 64 · 16 = 1024.
        assert_eq!(s_l2_max_squared(false, 64, 3).unwrap(), 1024);
        // Dense lb=1 (b/2 = 2^0 = 1): s_l2_max^2 = D.
        assert_eq!(s_l2_max_squared(false, 32, 1).unwrap(), 32);
    }

    #[test]
    fn beta_l2_squared_is_exact_square() {
        // Gamma=16, B = num_claims·2^r_vars = 1·4 = 4, s_l2_max^2 = 1024
        // (so s_l2_max = 32): beta_l2 = 16·4·32 = 2048, beta_l2^2 = 4_194_304.
        assert_eq!(beta_l2_squared(2, 1, 16, 1024).unwrap(), 2048 * 2048);
        // num_claims scales B linearly: B = 2·4 = 8 doubles beta_l2, quadruples^2.
        assert_eq!(beta_l2_squared(2, 2, 16, 1024).unwrap(), 4096 * 4096);
    }

    #[test]
    fn beta_l2_squared_rejects_degenerate() {
        assert!(beta_l2_squared(127, 1, 16, 1).is_err());
        assert!(beta_l2_squared(0, 1, u128::MAX, u128::MAX).is_err());
    }

    #[test]
    fn l2_bound_squared_scales_with_width() {
        assert_eq!(l2_bound_squared(8, 4_194_304).unwrap(), 8 * 4_194_304);
        assert!(l2_bound_squared(u128::MAX, 2).is_err());
    }

    #[test]
    fn l2_sq_from_linf_matches_sqrt_d_envelope() {
        // B/D-role digit collision 2^lb - 1 at lb=3 is 7; ||v||_2^2 <= d·49.
        assert_eq!(l2_sq_from_linf(64, 7).unwrap(), 64 * 49);
        assert!(l2_sq_from_linf(u128::MAX, u128::MAX).is_err());
    }
}
