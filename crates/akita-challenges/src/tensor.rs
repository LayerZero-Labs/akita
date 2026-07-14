//! Tensor-shaped sparse-challenge sampling.
//!
//! For protocols that sample a length-`live_fold_count` sparse-challenge vector per
//! claim, the tensor variant samples two factor vectors of length
//! `√live_fold_count` and presents the logical fold challenge at block `(p, q)` as
//! the negacyclic tensor product `fold_high[p] · fold_low[q]`. This shrinks transcript
//! challenge sampling from `O(live_fold_count)` to `O(√live_fold_count)` per claim
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
//! [`TensorChallenges`] is the factored tensor state whose fold-high/fold-low lengths
//! are part of the invariant.

use crate::{SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};
use akita_transcript::labels;
use sha3::{Digest, Sha3_256};

const FOLD_HIGH_DIGEST_DOMAIN: &[u8] = b"akita/fold-high-digest/v1";

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
    /// Factor each live fold index as `fold_low_len * high + low`.
    Tensor {
        /// Number of low-factor challenges per claim. Must be a power of two.
        fold_low_len: usize,
    },
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
            Self::Tensor { .. } => cfg.l1_norm().saturating_mul(cfg.l1_norm()),
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
            Self::Tensor { .. } => cfg.l1_norm().saturating_mul(inf),
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
            Self::Tensor { .. } => (cfg.l1_norm() as u128)
                .saturating_mul(cfg.l1_norm() as u128)
                .saturating_mul(challenge_l2_sq_max),
        }
    }
}

/// Factored tensor sparse challenges for one folding round.
///
/// `fold_high` and `fold_low` are laid out per claim. The logical challenge at
/// live fold `f = fold_low_len * h + q` is the tensor product
/// `fold_high[c, h] · fold_low[c, q]`. Keeping this as the tensor-only payload makes
/// the factorization invariant explicit for callers that can evaluate weighted
/// aggregates without expanding every logical block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorChallenges {
    /// High-factor entries, grouped by claim.
    pub fold_high: Vec<SparseChallenge>,
    /// Low-factor entries, grouped by claim.
    pub fold_low: Vec<SparseChallenge>,
    /// Exact number of live folds per claim.
    pub live_folds_per_claim: usize,
    /// Number of low-factor entries per claim.
    pub fold_low_len: usize,
    /// Number of claims represented by this tensor challenge family.
    pub num_claims: usize,
}

/// Stage-1 fold challenges — the single representation seen by prover and
/// verifier protocol code, with all per-variant logic encapsulated behind
/// challenge-domain methods such as [`Self::evals_at_pows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Challenges {
    /// Flat challenge vector indexed as `claim * live_fold_count + block`.
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
        /// Factored fold-high/fold-low sparse challenges.
        factored: TensorChallenges,
    },
}

/// Transcript labels consumed by [`crate::FoldDraw::draw_folding_challenges`].
///
/// Bundling them in a struct keeps the call sites self-describing and prevents
/// accidental fold-high/fold-low swaps. Callers pick label byte strings appropriate
/// for their protocol stage.
#[derive(Debug, Clone, Copy)]
pub struct ChallengeLabels<'a> {
    /// Label used for the flat shape's single sampling step.
    pub flat: &'a [u8],
    /// Label used for sampling the tensor shape's fold-high factor.
    pub fold_high: &'a [u8],
    /// Label under which the canonical digest of the sampled fold-high factor is
    /// absorbed back into the transcript before sampling the fold-low factor.
    pub fold_high_digest: &'a [u8],
    /// Label used for sampling the tensor shape's fold-low factor.
    pub fold_low: &'a [u8],
}

/// Canonical witness-fold challenge transcript labels.
#[inline]
#[must_use]
pub fn witness_fold_challenge_labels() -> ChallengeLabels<'static> {
    ChallengeLabels {
        flat: labels::CHALLENGE_WITNESS_FOLD,
        fold_high: labels::CHALLENGE_FOLD_HIGH,
        fold_high_digest: labels::ABSORB_FOLD_HIGH,
        fold_low: labels::CHALLENGE_FOLD_LOW,
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

    /// Construct tensor challenges from factored fold-high/fold-low vectors.
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
            Self::Tensor { factored, .. } => factored.live_folds_per_claim,
        }
    }

    /// Evaluate every logical challenge at the precomputed `alpha`-powers,
    /// in claim-major flat order. This is the boundary used by the prover's
    /// dense `compute_relation_matrix_col_evals` path.
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
                let fold_high_len = factored.fold_high_len();
                let mut fold_high = Vec::with_capacity(claim_indices.len() * fold_high_len);
                let mut fold_low = Vec::with_capacity(claim_indices.len() * factored.fold_low_len);
                for &claim_idx in claim_indices {
                    if claim_idx >= factored.num_claims {
                        return Err(AkitaError::InvalidInput(format!(
                            "tensor claim index {claim_idx} out of range for {} claims",
                            factored.num_claims
                        )));
                    }
                    let high_start = claim_idx * fold_high_len;
                    let low_start = claim_idx * factored.fold_low_len;
                    fold_high.extend_from_slice(
                        &factored.fold_high[high_start..high_start + fold_high_len],
                    );
                    fold_low.extend_from_slice(
                        &factored.fold_low[low_start..low_start + factored.fold_low_len],
                    );
                }
                Self::from_tensor::<D>(TensorChallenges {
                    fold_high,
                    fold_low,
                    live_folds_per_claim: factored.live_folds_per_claim,
                    fold_low_len: factored.fold_low_len,
                    num_claims: claim_indices.len(),
                })
            }
        }
    }
}

impl TensorChallenges {
    /// Exact number of high-factor entries per claim.
    #[inline]
    #[must_use]
    pub fn fold_high_len(&self) -> usize {
        if self.fold_low_len == 0 {
            0
        } else {
            self.live_folds_per_claim.div_ceil(self.fold_low_len)
        }
    }

    /// Validate the tensor challenge shape and all sparse challenge factors.
    ///
    /// # Errors
    ///
    /// Returns an error if fold-high/fold-low vector lengths do not match
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
        if self.live_folds_per_claim == 0 || !self.fold_low_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "tensor challenges require positive live folds and a power-of-two low length"
                    .to_string(),
            ));
        }
        self.total_blocks()?;
        self.validate_lengths()?;
        for challenge in self.fold_high.iter().chain(self.fold_low.iter()) {
            challenge.validate_dyn(ring_d)?;
        }
        Ok(())
    }

    /// Number of logical block challenges represented by one claim.
    ///
    /// # Errors
    ///
    /// This is the exact live prefix, not the padded tensor capacity.
    pub fn blocks_per_claim(&self) -> Result<usize, AkitaError> {
        Ok(self.live_folds_per_claim)
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
        if self.live_folds_per_claim == 0 || !self.fold_low_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "tensor challenges require positive live folds and a power-of-two low length"
                    .to_string(),
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
        let fold_high_len = self.fold_high_len();
        let high_idx = claim_idx * fold_high_len + (local_idx / self.fold_low_len);
        let low_idx = claim_idx * self.fold_low_len + (local_idx % self.fold_low_len);
        let high = self
            .fold_high
            .get(high_idx)
            .ok_or(AkitaError::InvalidSize {
                expected: high_idx + 1,
                actual: self.fold_high.len(),
            })?;
        let low = self.fold_low.get(low_idx).ok_or(AkitaError::InvalidSize {
            expected: low_idx + 1,
            actual: self.fold_low.len(),
        })?;
        Ok((claim_idx, local_idx, high, low))
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
        let fold_high_len = self.fold_high_len();
        let mut out = Vec::with_capacity(self.num_claims * self.live_folds_per_claim);
        for claim_idx in 0..self.num_claims {
            let high_start = claim_idx * fold_high_len;
            let low_start = claim_idx * self.fold_low_len;
            let high = &self.fold_high[high_start..high_start + fold_high_len];
            let low = &self.fold_low[low_start..low_start + self.fold_low_len];
            let high_evals = high
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            let low_evals = low
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;

            for local_idx in 0..self.live_folds_per_claim {
                let h = local_idx / self.fold_low_len;
                let q = local_idx % self.fold_low_len;
                let quotient_eval =
                    tensor_product_quotient_eval::<F, E>(&high[h], &low[q], alpha_pows)?;
                out.push(high_evals[h] * low_evals[q] - alpha_pow_d_plus_one * quotient_eval);
            }
        }
        Ok(out)
    }

    /// Evaluate one weighted tensor aggregate exactly at a ring-switch point.
    ///
    /// This is the factored aggregate API. Unlike
    /// [`Self::evals_at_pows`], it consumes separable fold-high/fold-low weights and
    /// returns a single aggregate for one claim.
    ///
    /// Computes
    ///
    /// ```text
    /// Σ_{p,q : p · Q + q < F} u[p] · v[q] · eval(reduce(H_p · L_q), α)
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
        let fold_high_len = self.fold_high_len();
        if u_weights.len() != fold_high_len {
            return Err(AkitaError::InvalidSize {
                expected: fold_high_len,
                actual: u_weights.len(),
            });
        }
        if v_weights.len() != self.fold_low_len {
            return Err(AkitaError::InvalidSize {
                expected: self.fold_low_len,
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
        let high_start = claim_idx * fold_high_len;
        let low_start = claim_idx * self.fold_low_len;

        // Build the weighted dense factors in E directly so the product
        // evaluation never materializes a length-O(fold_high_len · fold_low_len) buffer.
        let mut high_bar = [E::zero(); D];
        let mut low_bar = [E::zero(); D];

        for (p, &weight) in u_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(
                    &mut high_bar,
                    &self.fold_high[high_start + p],
                    weight,
                )?;
            }
        }
        for (q, &weight) in v_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(
                    &mut low_bar,
                    &self.fold_low[low_start + q],
                    weight,
                )?;
            }
        }

        let aggregate = eval_dense_negacyclic_product_at_pows(
            &high_bar,
            &low_bar,
            alpha_pows,
            alpha_pow_d_plus_one,
        );

        let final_low_len = self.live_folds_per_claim % self.fold_low_len;
        if final_low_len == 0 {
            return Ok(aggregate);
        }

        // The full factored product includes the padded suffix of the final
        // high row. Subtract that single separable rectangle to retain exactly
        // the live prefix without expanding one product per fold.
        let final_high_idx = fold_high_len - 1;
        let mut final_high_bar = [E::zero(); D];
        let final_high_weight = u_weights[final_high_idx];
        if !final_high_weight.is_zero() {
            accumulate_sparse_scaled::<F, E, D>(
                &mut final_high_bar,
                &self.fold_high[high_start + final_high_idx],
                final_high_weight,
            )?;
        }
        let mut padded_low_bar = [E::zero(); D];
        for (q, &weight) in v_weights.iter().enumerate().skip(final_low_len) {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(
                    &mut padded_low_bar,
                    &self.fold_low[low_start + q],
                    weight,
                )?;
            }
        }
        let padded_suffix = eval_dense_negacyclic_product_at_pows(
            &final_high_bar,
            &padded_low_bar,
            alpha_pows,
            alpha_pow_d_plus_one,
        );

        Ok(aggregate - padded_suffix)
    }

    fn validate_lengths(&self) -> Result<(), AkitaError> {
        let expected_high = self
            .num_claims
            .checked_mul(self.fold_high_len())
            .ok_or_else(|| AkitaError::InvalidSetup("fold-high count overflow".to_string()))?;
        if self.fold_high.len() != expected_high {
            return Err(AkitaError::InvalidSize {
                expected: expected_high,
                actual: self.fold_high.len(),
            });
        }
        let expected_low = self
            .num_claims
            .checked_mul(self.fold_low_len)
            .ok_or_else(|| AkitaError::InvalidSetup("fold-low count overflow".to_string()))?;
        if self.fold_low.len() != expected_low {
            return Err(AkitaError::InvalidSize {
                expected: expected_low,
                actual: self.fold_low.len(),
            });
        }
        Ok(())
    }
}

// Helper for `TensorChallenges::evals_at_pows`. This computes only the
// negacyclic wrap correction for one fold-high/fold-low pair; the caller combines it
// with `eval(high) * eval(low)` to produce one logical block evaluation.
fn tensor_product_quotient_eval<F, E>(
    high: &SparseChallenge,
    low: &SparseChallenge,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    let ring_d = alpha_pows.len();
    high.validate_dyn(ring_d)?;
    low.validate_dyn(ring_d)?;
    let mut quotient_eval = E::zero();
    for (&high_pos, &high_coeff) in high.positions.iter().zip(high.coeffs.iter()) {
        let high_idx = high_pos as usize;
        for (&low_pos, &low_coeff) in low.positions.iter().zip(low.coeffs.iter()) {
            let low_idx = low_pos as usize;
            let degree = high_idx + low_idx;
            if degree >= ring_d {
                let term = i64::from(high_coeff) * i64::from(low_coeff);
                quotient_eval += alpha_pows[degree - ring_d].mul_base(F::from_i64(term));
            }
        }
    }
    Ok(quotient_eval)
}

// Helpers for `eval_factored_aggregate_at_pows`, which evaluates one weighted
// claim aggregate directly from the fold-high/fold-low tensor factors.
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

fn eval_dense_negacyclic_product_at_pows<E: FieldCore, const D: usize>(
    high: &[E; D],
    low: &[E; D],
    alpha_pows: &[E],
    alpha_pow_d_plus_one: E,
) -> E {
    let product_eval = eval_dense_at_pows(high, alpha_pows) * eval_dense_at_pows(low, alpha_pows);
    let mut quotient_eval = E::zero();
    for (i, &high_coeff) in high.iter().enumerate() {
        if high_coeff.is_zero() {
            continue;
        }
        for (j, &low_coeff) in low.iter().enumerate() {
            if low_coeff.is_zero() || i + j < D {
                continue;
            }
            quotient_eval += high_coeff * low_coeff * alpha_pows[i + j - D];
        }
    }
    product_eval - alpha_pow_d_plus_one * quotient_eval
}

/// Total sparse challenges drawn in one fold round.
///
/// Flat: `live_fold_count · num_claims`.
/// Tensor: `num_claims · (ceil(live_fold_count / fold_low_len) + fold_low_len)`.
#[inline]
pub fn fold_sparse_challenge_sample_count(
    shape: ChallengeShape,
    live_fold_count: usize,
    num_claims: usize,
) -> Option<usize> {
    match shape {
        ChallengeShape::Flat => live_fold_count.checked_mul(num_claims),
        ChallengeShape::Tensor { fold_low_len } => {
            if live_fold_count == 0 || !fold_low_len.is_power_of_two() {
                return None;
            }
            let fold_high_len = live_fold_count.div_ceil(fold_low_len);
            fold_high_len
                .checked_add(fold_low_len)?
                .checked_mul(num_claims)
        }
    }
}

/// Compute the canonical digest absorbed between fold-high and fold-low
/// challenge sampling.
///
/// Binding the fold-low vector's transcript challenge to the exact fold-high vector
/// blocks any adaptive ground-out attempt on the fold-low factor.
///
/// # Errors
///
/// Returns an error if the fold-high vector length is inconsistent with the
/// supplied shape or if any sparse challenge violates structural invariants.
pub fn fold_high_digest(
    fold_high: &[SparseChallenge],
    fold_high_len: usize,
    num_claims: usize,
    ring_d: usize,
) -> Result<[u8; 32], AkitaError> {
    let expected = fold_high_len
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("fold-high digest count overflow".to_string()))?;
    if fold_high.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: fold_high.len(),
        });
    }

    let mut hasher = Sha3_256::new();
    hasher.update(FOLD_HIGH_DIGEST_DOMAIN);
    // Byte-critical: same little-endian u64 encoding of the ring dimension as
    // the former `(D as u64)`; identical bytes for equal values.
    hasher.update((ring_d as u64).to_le_bytes());
    hasher.update((num_claims as u64).to_le_bytes());
    hasher.update((fold_high_len as u64).to_le_bytes());
    hasher.update((fold_high.len() as u64).to_le_bytes());

    for challenge in fold_high {
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
