//! Tensor-shaped sparse-challenge sampling.
//!
//! For protocols that sample a length-`num_blocks` sparse-challenge vector per
//! claim, the tensor variant samples two factor vectors of length
//! `√num_blocks` and presents the logical fold challenge at block `(p, q)` as
//! the negacyclic tensor product `left[p] · right[q]`. This shrinks transcript
//! challenge sampling from `O(num_blocks)` to `O(√num_blocks)` per claim
//! while leaving the downstream fold semantics unchanged through structured
//! evaluation and factor-aware prover kernels.
//!
//! Sampling labels are taken as a [`ChallengeLabels`] parameter so callers can
//! choose stage-specific transcript wiring explicitly.
//!
//! The public types are split by protocol role:
//! [`ChallengeShape`] is only the flat-vs-tensor selector,
//! [`ChallengeLabels`] is the transcript metadata for one sampling round,
//! [`FoldingChallenges`] is the sampled runtime container, and
//! [`TensorChallenges`] is the factored tensor state whose left/right lengths
//! are part of the invariant.

use crate::{SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};
use akita_transcript::labels;
use sha3::{Digest, Sha3_256};

const TENSOR_LEFT_DIGEST_DOMAIN: &[u8] = b"akita/tensor-left-digest/v1";

/// Configuration selector for a tensor-vs-flat sparse-challenge round.
///
/// This type intentionally carries no sampled challenges; it is used when
/// choosing transcript labels, challenge counts, and L1 envelopes before a
/// runtime container exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChallengeShape {
    /// Sample one independent challenge for every logical block.
    #[default]
    Flat,
    /// Split each logical block index into two balanced dimensions and sample
    /// independent left/right challenge vectors.
    Tensor,
}

impl ChallengeShape {
    /// Effective per-logical-block integer L1 mass for this shape.
    ///
    /// Flat folds inherit the configured per-challenge L1 norm directly;
    /// tensor folds materialize `α_p · β_q` whose L1 envelope is bounded by
    /// the product of the two factors' L1 norms.
    #[inline]
    #[must_use]
    pub fn effective_l1_mass(&self, cfg: &SparseChallengeConfig) -> usize {
        match self {
            Self::Flat => cfg.l1_norm(),
            Self::Tensor => cfg.l1_norm().saturating_mul(cfg.l1_norm()),
        }
    }

    /// Effective per-logical-block integer L∞ norm for this shape.
    ///
    /// Flat folds inherit the configured per-challenge L∞ norm directly;
    /// tensor folds materialize the negacyclic product `α_p · β_q`, whose
    /// coefficients are bounded by `||α||_1 · ||β||_inf <= l1 · inf` (the
    /// ring-product L∞ inequality). This is the challenge `||c||_inf` used by
    /// the `min(||c||_inf·||s||_1, ||c||_1·||s||_inf)` folded-witness bound.
    #[inline]
    #[must_use]
    pub fn effective_infinity_norm(&self, cfg: &SparseChallengeConfig) -> usize {
        let inf = cfg.infinity_norm() as usize;
        match self {
            Self::Flat => inf,
            Self::Tensor => cfg.l1_norm().saturating_mul(inf),
        }
    }

    /// Effective per-logical-block deterministic `max ‖c‖_2²` for this
    /// fold-challenge shape.
    ///
    /// Flat folds use [`SparseChallengeConfig::challenge_l2_sq_max`] directly.
    /// Tensor folds materialize the negacyclic product `α_p · β_q`, whose
    /// deterministic L2 envelope is bounded by
    /// `||α||_1² · ||β||_2²` for the symmetric factor families used by Akita.
    ///
    /// This is not the tensor fold-grind chaos scale. The tensor tail formula
    /// uses the product of factor L2 variances as a proof artifact, but that
    /// quantity is not a deterministic L2 bound for the materialized product.
    #[inline]
    #[must_use]
    pub fn effective_l2_sq_max(&self, cfg: &SparseChallengeConfig) -> u128 {
        let challenge_l2_sq_max = cfg.challenge_l2_sq_max();
        match self {
            Self::Flat => challenge_l2_sq_max,
            Self::Tensor => (cfg.l1_norm() as u128)
                .saturating_mul(cfg.l1_norm() as u128)
                .saturating_mul(challenge_l2_sq_max),
        }
    }
}

/// Factored tensor sparse challenges for one folding round.
///
/// `left` and `right` are laid out per claim: claim `c`'s left factor occupies
/// `left[c * left_len .. (c + 1) * left_len]` and similarly for `right`. The
/// logical challenge at block `(p, q)` of claim `c` is the tensor product
/// `left[c, p] · right[c, q]`. Keeping this as the tensor-only payload makes
/// the factorization invariant explicit for callers that can evaluate weighted
/// aggregates without expanding every logical block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorChallenges {
    /// Left vector entries, grouped by claim.
    pub left: Vec<SparseChallenge>,
    /// Right vector entries, grouped by claim.
    pub right: Vec<SparseChallenge>,
    /// Number of left entries per claim.
    pub left_len: usize,
    /// Number of right entries per claim.
    pub right_len: usize,
    /// Number of claims represented by this tensor challenge family.
    pub num_claims: usize,
}

/// Stage-1 fold challenges — the single representation seen by prover and
/// verifier protocol code, with all per-variant logic encapsulated behind
/// challenge-domain methods such as [`Self::evals_at_pows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Challenges {
    /// Flat challenge vector indexed as `claim * num_blocks + block`.
    Sparse {
        /// Per-(claim, block) sparse challenges.
        challenges: Vec<SparseChallenge>,
        /// Number of (claim, block) entries packed into one claim.
        num_blocks_per_claim: usize,
        /// Number of claims represented by this vector.
        num_claims: usize,
    },
    /// Tensor-factored challenges.
    Tensor {
        /// Factored left/right sparse challenges (one factor pair per claim).
        factored: TensorChallenges,
    },
}

/// Transcript labels consumed by [`crate::sample_folding_challenges`].
///
/// Bundling them in a struct keeps the call sites self-describing and prevents
/// accidental left/right swaps. Callers pick label byte strings appropriate
/// for their protocol stage.
#[derive(Debug, Clone, Copy)]
pub struct ChallengeLabels<'a> {
    /// Label used for the flat shape's single sampling step.
    pub flat: &'a [u8],
    /// Label used for sampling the tensor shape's left factor.
    pub tensor_left: &'a [u8],
    /// Label under which the canonical digest of the sampled left factor is
    /// absorbed back into the transcript before sampling the right factor.
    pub tensor_left_digest: &'a [u8],
    /// Label used for sampling the tensor shape's right factor.
    pub tensor_right: &'a [u8],
}

/// Canonical stage-1 fold challenge transcript labels.
#[inline]
#[must_use]
pub fn stage1_fold_challenge_labels() -> ChallengeLabels<'static> {
    ChallengeLabels {
        flat: labels::CHALLENGE_STAGE1_FOLD,
        tensor_left: labels::CHALLENGE_TENSOR_FOLD_LEFT,
        tensor_left_digest: labels::ABSORB_TENSOR_FOLD_LEFT,
        tensor_right: labels::CHALLENGE_TENSOR_FOLD_RIGHT,
    }
}

impl Challenges {
    /// Construct flat sparse challenges from a pre-sampled vector and the
    /// claim/block shape used to interpret it.
    ///
    /// # Errors
    ///
    /// Returns an error if `challenges.len()` does not match
    /// `num_blocks_per_claim * num_claims`.
    pub fn from_sparse(
        challenges: Vec<SparseChallenge>,
        num_blocks_per_claim: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let expected = num_blocks_per_claim
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        Ok(Self::Sparse {
            challenges,
            num_blocks_per_claim,
            num_claims,
        })
    }

    /// Construct tensor challenges from factored left/right vectors,
    /// eagerly materializing the per-block product cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the factored input is malformed or if any tensor
    /// product overflows its integer coefficient representation.
    pub fn from_tensor<const D: usize>(factored: TensorChallenges) -> Result<Self, AkitaError> {
        Self::from_tensor_dyn(factored, D)
    }

    /// Runtime ring-dimension form of [`Self::from_tensor`].
    ///
    /// # Errors
    ///
    /// Returns an error if the factored input is malformed or if any tensor
    /// product overflows its integer coefficient representation.
    pub fn from_tensor_dyn(factored: TensorChallenges, ring_d: usize) -> Result<Self, AkitaError> {
        factored.validate_dyn(ring_d)?;
        Ok(Self::Tensor { factored })
    }

    /// Number of logical block challenges represented by this value.
    #[inline]
    #[must_use]
    pub fn logical_len(&self) -> usize {
        self.num_claims() * self.num_blocks_per_claim()
    }

    /// Number of claims represented by this challenge set.
    #[inline]
    #[must_use]
    pub fn num_claims(&self) -> usize {
        match self {
            Self::Sparse { num_claims, .. } => *num_claims,
            Self::Tensor { factored, .. } => factored.num_claims,
        }
    }

    /// Number of logical block challenges per claim.
    #[inline]
    #[must_use]
    pub fn num_blocks_per_claim(&self) -> usize {
        match self {
            Self::Sparse {
                num_blocks_per_claim,
                ..
            } => *num_blocks_per_claim,
            Self::Tensor { factored, .. } => factored.left_len * factored.right_len,
        }
    }

    /// Evaluate every logical challenge at the precomputed `alpha`-powers,
    /// in claim-major flat order. This is the boundary used by the prover's
    /// dense `compute_grouped_m_evals_x` path.
    ///
    /// For the sparse variant this is the canonical per-block evaluation
    /// loop; for the tensor variant it uses the factored aggregate
    /// formulation that avoids materializing every logical block.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha_pows` has the wrong length or if any
    /// per-challenge evaluation rejects its input.
    pub fn evals_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        match self {
            Self::Sparse { challenges, .. } => challenges
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect(),
            Self::Tensor { factored, .. } => factored.evals_at_pows::<F, E>(alpha_pows),
        }
    }

    /// Slice this challenge set down to the subset whose claim indices
    /// appear in `claim_indices`, returning a fresh `Challenges` of length
    /// `claim_indices.len() * num_blocks_per_claim`.
    ///
    /// Used by `RingRelationProver::new` to chunk the global challenge
    /// vector by opening point before handing each chunk to the poly
    /// backend.
    ///
    /// # Errors
    ///
    /// Returns an error if any claim index is out of range or if the tensor
    /// expansion of the selected sub-block fails.
    pub fn select_claims<const D: usize>(
        &self,
        claim_indices: &[usize],
    ) -> Result<Self, AkitaError> {
        match self {
            Self::Sparse {
                challenges,
                num_blocks_per_claim,
                ..
            } => {
                let mut selected = Vec::with_capacity(claim_indices.len() * num_blocks_per_claim);
                for &claim_idx in claim_indices {
                    let start = claim_idx
                        .checked_mul(*num_blocks_per_claim)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("challenge offset overflow".to_string())
                        })?;
                    let end = start.checked_add(*num_blocks_per_claim).ok_or_else(|| {
                        AkitaError::InvalidSetup("challenge offset overflow".to_string())
                    })?;
                    selected.extend_from_slice(challenges.get(start..end).ok_or(
                        AkitaError::InvalidSize {
                            expected: end,
                            actual: challenges.len(),
                        },
                    )?);
                }
                Self::from_sparse(selected, *num_blocks_per_claim, claim_indices.len())
            }
            Self::Tensor { factored, .. } => {
                factored.validate_lengths()?;
                let mut left = Vec::with_capacity(claim_indices.len() * factored.left_len);
                let mut right = Vec::with_capacity(claim_indices.len() * factored.right_len);
                for &claim_idx in claim_indices {
                    if claim_idx >= factored.num_claims {
                        return Err(AkitaError::InvalidInput(format!(
                            "tensor claim index {claim_idx} out of range for {} claims",
                            factored.num_claims
                        )));
                    }
                    let left_start = claim_idx * factored.left_len;
                    let right_start = claim_idx * factored.right_len;
                    left.extend_from_slice(
                        &factored.left[left_start..left_start + factored.left_len],
                    );
                    right.extend_from_slice(
                        &factored.right[right_start..right_start + factored.right_len],
                    );
                }
                Self::from_tensor::<D>(TensorChallenges {
                    left,
                    right,
                    left_len: factored.left_len,
                    right_len: factored.right_len,
                    num_claims: claim_indices.len(),
                })
            }
        }
    }
}

impl TensorChallenges {
    /// Validate the tensor challenge shape and all sparse challenge factors.
    ///
    /// # Errors
    ///
    /// Returns an error if left/right vector lengths do not match
    /// `num_claims * len`, dimensions are not powers of two, the total block
    /// count overflows, or any sparse factor is malformed for ring dimension
    /// `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), AkitaError> {
        self.validate_dyn(D)
    }

    /// Runtime ring-dimension form of [`Self::validate`].
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::validate`] with
    /// ring dimension `ring_d`.
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), AkitaError> {
        if !self.left_len.is_power_of_two() || !self.right_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "tensor challenge dimensions must be powers of two".to_string(),
            ));
        }
        self.total_blocks()?;
        self.validate_lengths()?;
        for challenge in self.left.iter().chain(self.right.iter()) {
            challenge.validate_dyn(ring_d)?;
        }
        Ok(())
    }

    /// Number of logical block challenges represented by one claim.
    ///
    /// # Errors
    ///
    /// Returns an error if `left_len * right_len` overflows.
    pub fn blocks_per_claim(&self) -> Result<usize, AkitaError> {
        self.left_len
            .checked_mul(self.right_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))
    }

    /// Total logical block challenges across all claims.
    ///
    /// # Errors
    ///
    /// Returns an error if count arithmetic overflows.
    pub fn total_blocks(&self) -> Result<usize, AkitaError> {
        self.num_claims
            .checked_mul(self.blocks_per_claim()?)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))
    }

    /// Return the factored challenges for one claim-major logical block.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor shape is malformed or `block_idx` is out
    /// of range.
    pub fn factors_for_logical_block(
        &self,
        block_idx: usize,
    ) -> Result<(usize, usize, &SparseChallenge, &SparseChallenge), AkitaError> {
        if !self.left_len.is_power_of_two() || !self.right_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "tensor challenge dimensions must be powers of two".to_string(),
            ));
        }
        self.validate_lengths()?;
        let blocks_per_claim = self.blocks_per_claim()?;
        let total_blocks = self
            .num_claims
            .checked_mul(blocks_per_claim)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("tensor challenge count overflow".to_string())
            })?;
        if block_idx >= total_blocks {
            return Err(AkitaError::InvalidInput(format!(
                "tensor block index {block_idx} out of range for {total_blocks} blocks"
            )));
        }

        let claim_idx = block_idx / blocks_per_claim;
        let local_idx = block_idx % blocks_per_claim;
        let left_idx = claim_idx * self.left_len + (local_idx / self.right_len);
        let right_idx = claim_idx * self.right_len + (local_idx % self.right_len);
        let left = self.left.get(left_idx).ok_or(AkitaError::InvalidSize {
            expected: left_idx + 1,
            actual: self.left.len(),
        })?;
        let right = self.right.get(right_idx).ok_or(AkitaError::InvalidSize {
            expected: right_idx + 1,
            actual: self.right.len(),
        })?;
        Ok((claim_idx, local_idx, left, right))
    }

    /// Evaluate reduced tensor products in logical block order.
    ///
    /// This mirrors [`Challenges::evals_at_pows`] for the tensor payload:
    /// it produces one field element per logical block without returning the
    /// intermediate tensor-product polynomials.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge shape validation or evaluation fails.
    pub fn evals_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        let ring_d = alpha_pows.len();
        if ring_d < 2 {
            return Err(AkitaError::InvalidInput(
                "tensor evaluation requires D >= 2".to_string(),
            ));
        }
        self.validate_lengths()?;

        // For `α ∈ E`, `α^D + 1 ∈ E` is the negacyclic-reduction scalar.
        // Tensor products commute with this reduction up to subtraction of a
        // quotient contribution; we precompute the scalar once and reuse it.
        let alpha_pow_d_plus_one = alpha_pows[ring_d - 1] * alpha_pows[1] + E::one();
        let mut out = Vec::with_capacity(self.num_claims * self.left_len * self.right_len);
        for claim_idx in 0..self.num_claims {
            let left_start = claim_idx * self.left_len;
            let right_start = claim_idx * self.right_len;
            let left = &self.left[left_start..left_start + self.left_len];
            let right = &self.right[right_start..right_start + self.right_len];
            let left_evals = left
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            let right_evals = right
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;

            for (p, left_challenge) in left.iter().enumerate() {
                for (q, right_challenge) in right.iter().enumerate() {
                    let quotient_eval = tensor_product_quotient_eval::<F, E>(
                        left_challenge,
                        right_challenge,
                        alpha_pows,
                    )?;
                    out.push(left_evals[p] * right_evals[q] - alpha_pow_d_plus_one * quotient_eval);
                }
            }
        }
        Ok(out)
    }

    /// Evaluate one weighted tensor aggregate exactly at a ring-switch point.
    ///
    /// This is the factored aggregate API. Unlike
    /// [`Self::evals_at_pows`], it consumes separable left/right weights and
    /// returns a single aggregate for one claim.
    ///
    /// Computes
    ///
    /// ```text
    /// Σ_{p,q} u[p] · v[q] · eval(reduce(L_p · R_q), α)
    /// ```
    ///
    /// without materializing every reduced tensor product. The negacyclic
    /// correction term is derived from `alpha_pows`, so the result is exact at
    /// every ring-switch point where `α^D + 1` is non-zero.
    ///
    /// # Errors
    ///
    /// Returns an error if weights, powers, claim routing, or sparse challenge
    /// representations are inconsistent with these tensor challenges.
    pub fn eval_factored_aggregate_at_pows<F, E, const D: usize>(
        &self,
        claim_idx: usize,
        u_weights: &[E],
        v_weights: &[E],
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        let ring_d = alpha_pows.len();
        if ring_d < 2 {
            return Err(AkitaError::InvalidInput(
                "tensor evaluation requires D >= 2".to_string(),
            ));
        }
        if ring_d != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: ring_d,
            });
        }
        if u_weights.len() != self.left_len {
            return Err(AkitaError::InvalidSize {
                expected: self.left_len,
                actual: u_weights.len(),
            });
        }
        if v_weights.len() != self.right_len {
            return Err(AkitaError::InvalidSize {
                expected: self.right_len,
                actual: v_weights.len(),
            });
        }
        if claim_idx >= self.num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "tensor claim index {claim_idx} out of range for {} claims",
                self.num_claims
            )));
        }
        self.validate_lengths()?;

        let alpha_pow_d_plus_one = alpha_pows[ring_d - 1] * alpha_pows[1] + E::one();
        let left_start = claim_idx * self.left_len;
        let right_start = claim_idx * self.right_len;

        // Build the weighted dense factors in E directly so the product
        // evaluation never materializes a length-O(left_len · right_len) buffer.
        let mut left_bar = [E::zero(); D];
        let mut right_bar = [E::zero(); D];

        for (p, &weight) in u_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(
                    &mut left_bar,
                    &self.left[left_start + p],
                    weight,
                )?;
            }
        }
        for (q, &weight) in v_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(
                    &mut right_bar,
                    &self.right[right_start + q],
                    weight,
                )?;
            }
        }

        let product_eval =
            eval_dense_at_pows(&left_bar, alpha_pows) * eval_dense_at_pows(&right_bar, alpha_pows);
        let mut quotient_eval = E::zero();
        for (i, &left_coeff) in left_bar.iter().enumerate() {
            if left_coeff.is_zero() {
                continue;
            }
            for (j, &right_coeff) in right_bar.iter().enumerate() {
                if right_coeff.is_zero() {
                    continue;
                }
                if i + j >= D {
                    quotient_eval += left_coeff * right_coeff * alpha_pows[i + j - D];
                }
            }
        }

        Ok(product_eval - alpha_pow_d_plus_one * quotient_eval)
    }

    fn validate_lengths(&self) -> Result<(), AkitaError> {
        let expected_left = self
            .num_claims
            .checked_mul(self.left_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor-left count overflow".to_string()))?;
        if self.left.len() != expected_left {
            return Err(AkitaError::InvalidSize {
                expected: expected_left,
                actual: self.left.len(),
            });
        }
        let expected_right = self
            .num_claims
            .checked_mul(self.right_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor-right count overflow".to_string()))?;
        if self.right.len() != expected_right {
            return Err(AkitaError::InvalidSize {
                expected: expected_right,
                actual: self.right.len(),
            });
        }
        Ok(())
    }
}

// Helper for `TensorChallenges::evals_at_pows`. This computes only the
// negacyclic wrap correction for one left/right pair; the caller combines it
// with `eval(left) * eval(right)` to produce one logical block evaluation.
fn tensor_product_quotient_eval<F, E>(
    left: &SparseChallenge,
    right: &SparseChallenge,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    let ring_d = alpha_pows.len();
    left.validate_dyn(ring_d)?;
    right.validate_dyn(ring_d)?;
    let mut quotient_eval = E::zero();
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        let left_idx = left_pos as usize;
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let right_idx = right_pos as usize;
            let degree = left_idx + right_idx;
            if degree >= ring_d {
                let term = i64::from(left_coeff) * i64::from(right_coeff);
                quotient_eval += alpha_pows[degree - ring_d].mul_base(F::from_i64(term));
            }
        }
    }
    Ok(quotient_eval)
}

// Helpers for `eval_factored_aggregate_at_pows`, which evaluates one weighted
// claim aggregate directly from the left/right tensor factors.
fn accumulate_sparse_scaled<F, E, const D: usize>(
    out: &mut [E],
    challenge: &SparseChallenge,
    scale: E,
) -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    if out.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: out.len(),
        });
    }
    challenge.validate::<D>()?;

    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        out[idx] += scale.mul_base(F::from_i64(coeff as i64));
    }
    Ok(())
}

fn eval_dense_at_pows<E: FieldCore>(coeffs: &[E], alpha_pows: &[E]) -> E {
    coeffs
        .iter()
        .zip(alpha_pows.iter())
        .fold(E::zero(), |acc, (&coeff, &power)| acc + coeff * power)
}

/// Split `num_blocks = 2^r` into balanced tensor dimensions.
///
/// # Errors
///
/// Returns an error if `num_blocks` is not a power of two.
#[inline]
pub fn tensor_split(num_blocks: usize) -> Result<(usize, usize), AkitaError> {
    if !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor challenges require a power-of-two block count".to_string(),
        ));
    }
    let r = num_blocks.trailing_zeros() as usize;
    let left_bits = r / 2;
    let right_bits = r - left_bits;
    Ok((1usize << left_bits, 1usize << right_bits))
}

/// Total sparse challenges drawn in one fold round.
///
/// Flat: `num_blocks · num_claims` with `num_blocks = 2^{r_vars}`.
/// Tensor: `num_claims · (left_len + right_len)` after [`tensor_split`].
#[inline]
pub fn fold_sparse_challenge_sample_count(
    shape: ChallengeShape,
    r_vars: usize,
    num_claims: usize,
) -> Option<usize> {
    let num_blocks = 1usize.checked_shl(r_vars as u32)?;
    match shape {
        ChallengeShape::Flat => num_blocks.checked_mul(num_claims),
        ChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_split(num_blocks).ok()?;
            left_len.checked_add(right_len)?.checked_mul(num_claims)
        }
    }
}

/// Compute the canonical digest absorbed between tensor-left and tensor-right
/// challenge sampling.
///
/// Binding the right vector's transcript challenge to the exact left vector
/// blocks any adaptive ground-out attempt on the right factor.
///
/// # Errors
///
/// Returns an error if the tensor-left vector length is inconsistent with the
/// supplied shape or if any sparse challenge violates structural invariants.
pub fn tensor_left_digest(
    left: &[SparseChallenge],
    left_len: usize,
    num_claims: usize,
    ring_d: usize,
) -> Result<[u8; 32], AkitaError> {
    let expected = left_len
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor-left digest count overflow".to_string()))?;
    if left.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: left.len(),
        });
    }

    let mut hasher = Sha3_256::new();
    hasher.update(TENSOR_LEFT_DIGEST_DOMAIN);
    // Byte-critical: same little-endian u64 encoding of the ring dimension as
    // the former `(D as u64)`; identical bytes for equal values.
    hasher.update((ring_d as u64).to_le_bytes());
    hasher.update((num_claims as u64).to_le_bytes());
    hasher.update((left_len as u64).to_le_bytes());
    hasher.update((left.len() as u64).to_le_bytes());

    for challenge in left {
        challenge.validate_dyn(ring_d)?;
        hasher.update((challenge.positions.len() as u64).to_le_bytes());

        let mut terms: Vec<(u32, i8)> = challenge
            .positions
            .iter()
            .copied()
            .zip(challenge.coeffs.iter().copied())
            .collect();
        terms.sort_by_key(|&(pos, _)| pos);
        for (pos, coeff) in terms {
            hasher.update(pos.to_le_bytes());
            hasher.update(coeff.to_le_bytes());
        }
    }

    Ok(hasher.finalize().into())
}
