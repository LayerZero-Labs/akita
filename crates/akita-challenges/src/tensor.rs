//! Tensor-shaped sparse-challenge sampling.
//!
//! For protocols that sample a length-`num_blocks` sparse-challenge vector per
//! claim, the tensor variant samples two factor vectors of length
//! `√num_blocks` and presents the logical fold challenge at block `(p, q)` as
//! the negacyclic tensor product `left[p] · right[q]`. This shrinks transcript
//! challenge sampling from `O(num_blocks)` to `O(√num_blocks)` per claim
//! while leaving the downstream fold semantics unchanged: callers see a
//! uniform flat view through [`FoldingChallenges::expand_integer`] /
//! [`FoldingChallenges::evals_at_pows`].
//!
//! Sampling labels are taken as a [`ChallengeLabels`] parameter so this
//! module is not coupled to any specific protocol stage.
//!
//! The public types are split by protocol role:
//! [`ChallengeShape`] is only the flat-vs-tensor selector,
//! [`FoldingChallenges`] is the sampled runtime container, and
//! [`TensorChallenges`] is the factored tensor state whose left/right lengths
//! are part of the invariant. Materialized logical challenges use
//! [`IntegerChallenge`], because tensor products can widen coefficients beyond
//! the sampled [`SparseChallenge`] range.

use crate::{sample_sparse_challenges, IntegerChallenge, SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase};
use akita_transcript::Transcript;
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

/// Sampled folding challenges, either already flat or tensor-structured.
///
/// This enum preserves the runtime representation chosen by
/// [`ChallengeShape`]. Callers that need ordinary per-block polynomials
/// can use [`FoldingChallenges::expand_integer`]; callers that can exploit the
/// factorization can match the tensor variant directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldingChallenges {
    /// Flat challenge vector indexed as `claim * num_blocks + block`.
    Flat(Vec<SparseChallenge>),
    /// Tensor-structured vectors indexed as `(claim, p, q)`.
    Tensor(TensorChallenges),
}

/// Transcript labels consumed by [`sample_tensor_challenges`].
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

impl FoldingChallenges {
    /// Number of logical flat challenges represented by this value.
    #[inline]
    #[must_use]
    pub fn logical_len(&self) -> usize {
        match self {
            Self::Flat(challenges) => challenges.len(),
            Self::Tensor(tensor) => tensor.num_claims * tensor.left_len * tensor.right_len,
        }
    }

    /// Materialize logical ring challenges for prover-side fold kernels.
    ///
    /// Flat challenges widen coefficients without changing the distribution;
    /// tensor challenges materialize `left[p] · right[q]` per logical block.
    /// This is the boundary where a sampled challenge container becomes an
    /// [`IntegerChallenge`].
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor product has malformed inputs or overflows
    /// its integer coefficient representation.
    pub fn expand_integer<const D: usize>(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
        match self {
            Self::Flat(challenges) => Ok(challenges
                .iter()
                .map(IntegerChallenge::from_sparse)
                .collect()),
            Self::Tensor(tensor) => tensor.expand_integer::<D>(),
        }
    }

    /// Evaluate all logical challenges at a ring-switch point, in flat order.
    ///
    /// This is the logical flat-view API: it returns one evaluation per block
    /// regardless of whether the challenges were sampled flat or factored.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge expansion or evaluation fails.
    pub fn evals_at_pows<F, E, const D: usize>(
        &self,
        alpha_pows: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        match self {
            Self::Flat(challenges) => challenges
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect(),
            Self::Tensor(tensor) => tensor.evals_at_pows::<F, E, D>(alpha_pows),
        }
    }
}

impl TensorChallenges {
    /// Materialize tensor products into logical flat order.
    ///
    /// This expansion is intentionally separate from the evaluation helpers
    /// below: it produces the widened integer polynomials consumed by fold
    /// kernels, while the evaluation helpers produce field evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor product has malformed inputs or overflows
    /// its integer coefficient representation.
    pub fn expand_integer<const D: usize>(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
        self.validate_lengths()?;
        let mut out = Vec::with_capacity(self.num_claims * self.left_len * self.right_len);
        for claim_idx in 0..self.num_claims {
            let left_start = claim_idx * self.left_len;
            let right_start = claim_idx * self.right_len;
            for p in 0..self.left_len {
                let left = &self.left[left_start + p];
                for q in 0..self.right_len {
                    let right = &self.right[right_start + q];
                    out.push(IntegerChallenge::tensor_product::<D>(left, right)?);
                }
            }
        }
        Ok(out)
    }

    /// Evaluate reduced tensor products in logical flat order.
    ///
    /// This mirrors [`FoldingChallenges::evals_at_pows`] for the tensor payload:
    /// it produces one field element per logical block without returning the
    /// intermediate [`IntegerChallenge`] values.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge shape validation or evaluation fails.
    pub fn evals_at_pows<F, E, const D: usize>(
        &self,
        alpha_pows: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if D < 2 {
            return Err(AkitaError::InvalidInput(
                "tensor evaluation requires D >= 2".to_string(),
            ));
        }
        self.validate_lengths()?;

        // For `α ∈ E`, `α^D + 1 ∈ E` is the negacyclic-reduction scalar.
        // Tensor products commute with this reduction up to subtraction of a
        // quotient contribution; we precompute the scalar once and reuse it.
        let alpha_pow_d_plus_one = alpha_pows[D - 1] * alpha_pows[1] + E::one();
        let mut out = Vec::with_capacity(self.num_claims * self.left_len * self.right_len);
        for claim_idx in 0..self.num_claims {
            let left_start = claim_idx * self.left_len;
            let right_start = claim_idx * self.right_len;
            let left = &self.left[left_start..left_start + self.left_len];
            let right = &self.right[right_start..right_start + self.right_len];
            let left_evals = left
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            let right_evals = right
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;

            for (p, left_challenge) in left.iter().enumerate() {
                for (q, right_challenge) in right.iter().enumerate() {
                    let quotient_eval = tensor_product_quotient_eval::<F, E, D>(
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
    /// correction term is explicit, so the result is exact at every
    /// ring-switch point where `α^D + 1` is non-zero.
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
        alpha_pow_d_plus_one: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
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

// Evaluation helpers used by the factored aggregate path.
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
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "sparse challenge positions/coeffs length mismatch".to_string(),
        ));
    }

    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        if idx >= D {
            return Err(AkitaError::InvalidInput(format!(
                "sparse challenge position {idx} out of range for D={D}"
            )));
        }
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "sparse challenge coefficients must be non-zero".to_string(),
            ));
        }
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

// Evaluation helper used by the logical flat-view tensor path.
fn tensor_product_quotient_eval<F, E, const D: usize>(
    left: &SparseChallenge,
    right: &SparseChallenge,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    if left.positions.len() != left.coeffs.len() || right.positions.len() != right.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "tensor challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    let mut quotient_eval = E::zero();
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        let left_idx = left_pos as usize;
        if left_idx >= D {
            return Err(AkitaError::InvalidInput(format!(
                "tensor-left challenge position {left_idx} out of range for D={D}"
            )));
        }
        if left_coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "tensor-left challenge coefficients must be non-zero".to_string(),
            ));
        }
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let right_idx = right_pos as usize;
            if right_idx >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "tensor-right challenge position {right_idx} out of range for D={D}"
                )));
            }
            if right_coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "tensor-right challenge coefficients must be non-zero".to_string(),
                ));
            }
            let degree = left_idx + right_idx;
            if degree >= D {
                let term = i64::from(left_coeff) * i64::from(right_coeff);
                quotient_eval += alpha_pows[degree - D].mul_base(F::from_i64(term));
            }
        }
    }
    Ok(quotient_eval)
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
pub fn tensor_left_digest<const D: usize>(
    left: &[SparseChallenge],
    left_len: usize,
    num_claims: usize,
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
    hasher.update((D as u64).to_le_bytes());
    hasher.update((num_claims as u64).to_le_bytes());
    hasher.update((left_len as u64).to_le_bytes());
    hasher.update((left.len() as u64).to_le_bytes());

    for challenge in left {
        if challenge.positions.len() != challenge.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "tensor-left digest positions/coeffs length mismatch".to_string(),
            ));
        }
        hasher.update((challenge.positions.len() as u64).to_le_bytes());

        let mut terms: Vec<(u32, i8)> = challenge
            .positions
            .iter()
            .copied()
            .zip(challenge.coeffs.iter().copied())
            .collect();
        terms.sort_by_key(|&(pos, _)| pos);
        let mut previous_pos = None;
        for (pos, coeff) in terms {
            if pos as usize >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "tensor-left digest position {pos} out of range for D={D}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "tensor-left digest coefficients must be non-zero".to_string(),
                ));
            }
            if previous_pos == Some(pos) {
                return Err(AkitaError::InvalidInput(
                    "tensor-left digest positions must be unique".to_string(),
                ));
            }
            previous_pos = Some(pos);
            hasher.update(pos.to_le_bytes());
            hasher.update(coeff.to_le_bytes());
        }
    }

    Ok(hasher.finalize().into())
}

/// Sample folding challenges using the configured shape.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, if tensor splitting is
/// invalid, or if sparse challenge sampling fails.
pub fn sample_tensor_challenges<F, T, const D: usize>(
    transcript: &mut T,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &ChallengeShape,
    labels: ChallengeLabels<'_>,
) -> Result<FoldingChallenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    match shape {
        ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor challenge count overflow".to_string())
            })?;
            sample_sparse_challenges::<F, T, D>(transcript, labels.flat, total, cfg)
                .map(FoldingChallenges::Flat)
        }
        ChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_split(num_blocks)?;
            let left_total = left_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-left challenge count overflow".to_string())
            })?;
            let right_total = right_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-right challenge count overflow".to_string())
            })?;
            let left = sample_sparse_challenges::<F, T, D>(
                transcript,
                labels.tensor_left,
                left_total,
                cfg,
            )?;
            let left_digest = tensor_left_digest::<D>(&left, left_len, num_claims)?;
            transcript.append_bytes(labels.tensor_left_digest, &left_digest);
            let right = sample_sparse_challenges::<F, T, D>(
                transcript,
                labels.tensor_right,
                right_total,
                cfg,
            )?;
            Ok(FoldingChallenges::Tensor(TensorChallenges {
                left,
                right,
                left_len,
                right_len,
                num_claims,
            }))
        }
    }
}
