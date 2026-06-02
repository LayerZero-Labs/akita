//! Weak-binding collision norms (Hachi paper, Lemma 7) and the folded-witness
//! bound, per witness role.
//!
//! `rounded_up_norm_{s,t,w}` return the audited SIS collision *bucket* ready to
//! feed [`super::ajtai_key::min_secure_rank`]; `rounded_up_norm_z` returns the
//! folded-witness L∞ bound `β` (which `z` decomposes against — `z` is not
//! Ajtai-committed, so it has no SIS bucket).

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};

use super::ajtai_key::{ceil_supported_collision, SisModulusFamily};
use crate::DecompositionParams;

/// Worst-case `||c · s||_inf` of a negacyclic ring product, from the per-element
/// L1/L∞ bounds:
///
/// ```text
/// ||c · s||_inf  <=  min( ||c||_inf · ||s||_1 ,  ||c||_1 · ||s||_inf ).
/// ```
///
/// Saturating arithmetic keeps this panic-free on the verifier-reachable path.
#[inline]
#[must_use]
pub fn ring_product_infinity_norm_bound(
    challenge_infinity_norm: u128,
    challenge_l1_norm: u128,
    witness_infinity_norm: u128,
    witness_l1_norm: u128,
) -> u128 {
    challenge_infinity_norm
        .saturating_mul(witness_l1_norm)
        .min(challenge_l1_norm.saturating_mul(witness_infinity_norm))
}

/// Worst-case L1 mass of one committed witness ring element (block):
/// `||s||_1 <= nonzeros · ||s||_inf` with `nonzeros = ceil(D / K)`:
///
/// - dense / full-field        : `K = 1`     ⇒ `nonzeros = D`
/// - one-hot, chunk size `K ≥ D`: single-chunk ⇒ `nonzeros = 1`
/// - one-hot, chunk size `K < D`: multi-chunk  ⇒ `nonzeros = D / K`
#[inline]
#[must_use]
pub fn witness_block_l1_norm(
    witness_infinity_norm: u128,
    ring_dimension: usize,
    onehot_chunk_size: usize,
) -> u128 {
    let nonzeros = (ring_dimension as u128).div_ceil((onehot_chunk_size.max(1)) as u128);
    witness_infinity_norm.saturating_mul(nonzeros)
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
    pub infinity_norm: u128,
    /// Witness L1 norm `||s||_1 = nonzeros · ||s||_inf`.
    pub l1_norm: u128,
}

/// Per-block committed-witness `(||s||_inf, ||s||_1)` for the folded witness.
///
/// - **one-hot** (`is_onehot`): `||s||_inf = 1`, `||s||_1 = ceil(D / K)`.
/// - **dense**: `||s||_inf = b/2 = 2^(log_basis-1)`, `||s||_1 = D · b/2`.
#[inline]
#[must_use]
pub fn fold_witness_norms(
    log_basis: u32,
    ring_dimension: usize,
    onehot_chunk_size: usize,
    is_onehot: bool,
) -> FoldWitnessNorms {
    if is_onehot {
        let infinity_norm = 1u128;
        FoldWitnessNorms {
            infinity_norm,
            l1_norm: witness_block_l1_norm(infinity_norm, ring_dimension, onehot_chunk_size),
        }
    } else {
        let infinity_norm = 1u128 << (log_basis.saturating_sub(1));
        FoldWitnessNorms {
            infinity_norm,
            l1_norm: witness_block_l1_norm(infinity_norm, ring_dimension, 1),
        }
    }
}

/// Single-opening A-role witness infinity norm `||s||_inf` (un-doubled).
///
/// The Lemma-7 factor of 2 is applied explicitly by
/// [`a_role_collision_infinity_norm`]:
/// - root one-hot (`log_commit_bound == 1`): `1`
/// - root dense: `2^(lb−1) − 1` (balanced-digit half-range `β`)
/// - recursive: `2^(lb−1)` (balanced-digit max magnitude `b/2`)
fn a_role_witness_infinity_norm(
    log_basis: u32,
    log_commit_bound: u32,
    is_root: bool,
) -> Option<u32> {
    if is_root {
        if log_commit_bound == 1 {
            Some(1)
        } else {
            1u32.checked_shl(log_basis.checked_sub(1)?)?.checked_sub(1)
        }
    } else {
        1u32.checked_shl(log_basis.checked_sub(1)?)
    }
}

/// Hachi Lemma 7 A-role weak-binding collision infinity norm
/// `collision_A = 2 · ω̄ · β̄ · ν` with
/// `β̄ = min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`. The factor of 2 is the
/// lemma's two-term cross-multiplication factor (witness norms are un-doubled).
fn a_role_collision_infinity_norm(
    challenge_infinity_norm: u128,
    challenge_l1_norm: u128,
    witness_infinity_norm: u128,
    witness_l1_norm: u128,
    ring_subfield_norm_bound: u128,
) -> Option<u32> {
    let beta = ring_product_infinity_norm_bound(
        challenge_infinity_norm,
        challenge_l1_norm,
        witness_infinity_norm,
        witness_l1_norm,
    );
    let collision = 2u128
        .checked_mul(challenge_l1_norm)?
        .checked_mul(beta)?
        .checked_mul(ring_subfield_norm_bound)?;
    u32::try_from(collision).ok()
}

/// A-role (committed witness `s`) rounded-up SIS collision bucket
/// `ceil(2·ω̄·β̄·ν)`. `decomposition.log_basis` is the level's gadget base.
///
/// Returns `None` on norm overflow or when the collision exceeds every audited
/// bucket for `(sis_family, d)`.
#[allow(clippy::too_many_arguments)]
pub fn rounded_up_norm_s(
    sis_family: SisModulusFamily,
    d: usize,
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
) -> Option<u32> {
    let witness_inf = a_role_witness_infinity_norm(
        decomposition.log_basis,
        decomposition.log_commit_bound,
        is_root,
    )?;
    let challenge_inf = fold_shape.effective_infinity_norm(stage1_config) as u128;
    let challenge_l1 = fold_shape.effective_l1_mass(stage1_config) as u128;
    // One-hot sparsity applies only to a root level committing a one-hot
    // witness; recursive and dense levels use the full `nonzeros = D` (`K = 1`).
    let nonzeros_chunk = if is_root && decomposition.log_commit_bound == 1 {
        onehot_chunk_size
    } else {
        1
    };
    let witness_l1 = witness_block_l1_norm(u128::from(witness_inf), d, nonzeros_chunk);
    let collision = a_role_collision_infinity_norm(
        challenge_inf,
        challenge_l1,
        u128::from(witness_inf),
        witness_l1,
        u128::from(ring_subfield_norm_bound),
    )?;
    ceil_supported_collision(sis_family, d as u32, collision)
}

/// B-role (`t̂`) rounded-up SIS collision bucket. The collision is the direct
/// difference of two balanced-digit openings, `2γ̄ = 2^lb − 1` (no challenge
/// multiplication).
pub fn rounded_up_norm_t(sis_family: SisModulusFamily, d: usize, log_basis: u32) -> Option<u32> {
    let collision = 1u32.checked_shl(log_basis)?.checked_sub(1)?;
    ceil_supported_collision(sis_family, d as u32, collision)
}

/// D-role (`ŵ`) rounded-up SIS collision bucket. Identical bound to the B role.
pub fn rounded_up_norm_w(sis_family: SisModulusFamily, d: usize, log_basis: u32) -> Option<u32> {
    rounded_up_norm_t(sis_family, d, log_basis)
}

/// Folded witness `z = Σ c_i·s_i` L∞ bound
/// `β = num_claims · 2^r_vars · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`.
///
/// `z` is decomposed (not Ajtai-committed), so this returns the raw L∞ bound;
/// feed it to [`super::decomposition_digits::num_digits_fold`]. Saturates to
/// `u128::MAX` on overflow (which `num_digits_fold` maps to the field-width
/// ceiling).
#[allow(clippy::too_many_arguments)]
pub fn rounded_up_norm_z(
    decomposition: DecompositionParams,
    stage1_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    r_vars: usize,
    num_claims: usize,
    d: usize,
    onehot_chunk_size: usize,
    is_root: bool,
) -> u128 {
    if r_vars >= 127 {
        return u128::MAX;
    }
    let is_onehot = is_root && decomposition.log_commit_bound == 1;
    let witness = fold_witness_norms(decomposition.log_basis, d, onehot_chunk_size, is_onehot);
    let challenge_inf = fold_shape.effective_infinity_norm(stage1_config) as u128;
    let challenge_l1 = fold_shape.effective_l1_mass(stage1_config) as u128;
    let beta_block = ring_product_infinity_norm_bound(
        challenge_inf,
        challenge_l1,
        witness.infinity_norm,
        witness.l1_norm,
    );
    beta_block
        .saturating_mul(num_claims as u128)
        .saturating_mul(1u128 << r_vars)
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
        assert_eq!(witness_block_l1_norm(3, 64, 1), 3 * 64);
        assert_eq!(witness_block_l1_norm(1, 64, 64), 1);
        assert_eq!(witness_block_l1_norm(1, 64, 8), 8);
    }

    #[test]
    fn a_role_witness_norm_levels() {
        assert_eq!(a_role_witness_infinity_norm(3, 1, true), Some(1)); // root one-hot
        assert_eq!(a_role_witness_infinity_norm(3, 32, true), Some(3)); // root dense: 2^2 - 1
        assert_eq!(a_role_witness_infinity_norm(3, 3, false), Some(4)); // recursive: 2^2
    }

    #[test]
    fn rounded_up_norm_z_onehot_smaller_than_dense() {
        let stage1 = SparseChallengeConfig::Uniform {
            weight: 8,
            nonzero_coeffs: vec![-1, 1],
        };
        let dense = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(128),
        };
        let onehot = DecompositionParams {
            log_basis: 3,
            log_commit_bound: 1,
            log_open_bound: Some(128),
        };
        let beta_dense = rounded_up_norm_z(
            dense,
            &stage1,
            TensorChallengeShape::Flat,
            8,
            1,
            64,
            64,
            true,
        );
        let beta_onehot = rounded_up_norm_z(
            onehot,
            &stage1,
            TensorChallengeShape::Flat,
            8,
            1,
            64,
            64,
            true,
        );
        assert!(beta_onehot < beta_dense);
    }
}
