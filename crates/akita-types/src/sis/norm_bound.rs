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

/// A-role (committed witness `s`) rounded-up SIS collision bucket
/// `ceil(2·ω̄·β̄·ν)` per Hachi Lemma 7, with
/// `β̄ = min(||c||_inf·||s||_1, ||c||_1·||s||_inf)` and
/// `ω̄ = ||c||_1` and `ν = ring_subfield_norm_bound`.
///
/// # Precondition (inner-witness shortness)
///
/// The per-block `β̄ = ||c·s||_inf` is the *anchored* price: it is sound only
/// when the committed inner witness L∞ is independently enforced at the
/// `||s||_inf` recorded by [`FoldWitnessNorms`]. The weak-binding extractor only
/// ever sees `||c̄·s||_inf = ||z^(ℓ,i) − z^(0)||_inf`, bounded generically by the
/// *fold response* `2·β^resp` (which carries the fold arity `2^r` and the
/// batched-claim count); dividing by the unit `c̄` does not recover `||s||_inf`.
/// The anchored price replaces `2·β^resp` by `ω̄·||s||_inf`, and is justified at:
///   - every recursive level (`is_root == false`): the witness is committed at
///     `δ_commit = 1` ([`crate::sis::num_digits_s_commit`]), i.e. it *is* the
///     previous level's range-checked extended witness (`||s||_inf ≤ b/2`) with
///     no gadget gap — see the per-level inner-witness bound proposition;
///   - one-hot roots (`is_root == true` and `log_commit_bound == 1`): also
///     committed at `δ_commit = 1`, so `s = f` and `||s||_inf ≤ 1` *provided the
///     caller proves the committed vector is one-hot* (in Jolt, the booleanity +
///     Hamming-weight checks on the same commitment);
///   - cleartext-digit roots: range-bounded by construction.
///
/// For an UNCONSTRAINED dense root (`δ_commit > 1`, no structural guarantee) the
/// extracted `s` is bounded only by `2·β^resp`, so this per-block bucket is
/// UNSOUND there: such a root must be priced via the fold bound
/// ([`fold_witness_beta`]) instead. Callers select the regime via the schedule's
/// root-witness-bound policy; this function only computes the anchored bucket.
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
) -> Option<u32> {
    let is_onehot = is_root && decomposition.log_commit_bound == 1;
    let witness_norm =
        FoldWitnessNorms::new(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    // β̄ = ||c·s||_inf; collision_A = 2·ω̄·β̄·ν with ω̄ = ||c||_1.
    let beta = ring_product_infinity_norm_bound(
        fold_shape.effective_infinity_norm(stage1_config) as u128,
        fold_shape.effective_l1_mass(stage1_config) as u128,
        witness_norm.infinity_norm,
        witness_norm.l1_norm,
    );
    let collision = 2u128
        .checked_mul(fold_shape.effective_l1_mass(stage1_config) as u128)?
        .checked_mul(beta)?
        .checked_mul(u128::from(ring_subfield_norm_bound))?;
    ceil_supported_collision(sis_family, d as u32, u32::try_from(collision).ok()?)
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
}
