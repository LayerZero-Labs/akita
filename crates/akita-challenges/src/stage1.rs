//! Stage-1 folding challenge shapes.

use crate::{sample_sparse_challenges, IntegerChallenge, SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{
    CHALLENGE_STAGE1_FOLD, CHALLENGE_STAGE1_FOLD_TENSOR_LEFT, CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT,
};
use akita_transcript::Transcript;

/// Transcript-derived challenge shape used by stage-1 folding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage1ChallengeShape {
    /// Sample one independent challenge for every logical fold block.
    Flat,
    /// Split each fold block index into two balanced dimensions and sample
    /// independent left/right challenge vectors.
    Tensor,
}

impl Default for Stage1ChallengeShape {
    fn default() -> Self {
        Self::Flat
    }
}

impl Stage1ChallengeShape {
    /// Effective per-block integer L1 mass for this shape.
    #[inline]
    pub fn effective_l1_mass(&self, cfg: &SparseChallengeConfig) -> usize {
        match self {
            Self::Flat => cfg.l1_norm(),
            Self::Tensor => cfg.l1_norm().saturating_mul(cfg.l1_norm()),
        }
    }
}

/// Tensor-structured sparse challenges for one stage-1 fold round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStage1Challenges {
    /// Left vector entries, grouped by claim.
    pub left: Vec<SparseChallenge>,
    /// Right vector entries, grouped by claim.
    pub right: Vec<SparseChallenge>,
    /// Number of left entries per claim.
    pub left_len: usize,
    /// Number of right entries per claim.
    pub right_len: usize,
    /// Number of claims represented by this tensor challenge set.
    pub num_claims: usize,
}

/// Stage-1 folding challenges, either flat or tensor-structured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage1Challenges {
    /// Flat challenge vector indexed as `claim * num_blocks + block`.
    Flat(Vec<SparseChallenge>),
    /// Tensor-structured vectors indexed as `claim, p, q`.
    Tensor(TensorStage1Challenges),
}

impl Stage1Challenges {
    /// Number of logical flat challenges represented by this value.
    #[inline]
    pub fn logical_len(&self) -> usize {
        match self {
            Self::Flat(challenges) => challenges.len(),
            Self::Tensor(tensor) => tensor.num_claims * tensor.left_len * tensor.right_len,
        }
    }

    /// Expand to integer ring challenges for prover-side fold kernels.
    ///
    /// Flat challenges widen coefficients without changing the distribution;
    /// tensor challenges materialize `left[p] * right[q]` per logical block.
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

    /// Evaluate all logical challenges at a ring-switch point.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge expansion or evaluation fails.
    pub fn evals_at_pows<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        alpha_pows: &[F],
    ) -> Result<Vec<F>, AkitaError> {
        match self {
            Self::Flat(challenges) => challenges
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, D>(alpha_pows))
                .collect(),
            Self::Tensor(tensor) => tensor.evals_at_pows::<F, D>(alpha_pows),
        }
    }
}

impl TensorStage1Challenges {
    /// Expand tensor products into logical flat order.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor product has malformed inputs or overflows
    /// its integer coefficient representation.
    pub fn expand_integer<const D: usize>(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
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
    /// # Errors
    ///
    /// Returns an error if challenge expansion or evaluation fails.
    pub fn evals_at_pows<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        alpha_pows: &[F],
    ) -> Result<Vec<F>, AkitaError> {
        self.expand_integer::<D>()?
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, D>(alpha_pows))
            .collect()
    }

    /// Evaluate one factored weighted tensor aggregate exactly at a ring-switch
    /// point.
    ///
    /// Computes:
    ///
    /// ```text
    /// sum_{p,q} u[p] * v[q] * eval(reduce(L_p * R_q), alpha)
    /// ```
    ///
    /// without materializing every reduced tensor product. The correction term
    /// for negacyclic reduction is explicit, so this remains exact for generic
    /// ring-switch points where `alpha^D + 1` is non-zero.
    ///
    /// # Errors
    ///
    /// Returns an error if weights, powers, claim routing, or sparse challenge
    /// representations are inconsistent with this tensor challenge set.
    pub fn eval_factored_aggregate_at_pows<F: FieldCore + CanonicalField, const D: usize>(
        &self,
        claim_idx: usize,
        u_weights: &[F],
        v_weights: &[F],
        alpha_pows: &[F],
        alpha_pow_d_plus_one: F,
    ) -> Result<F, AkitaError> {
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

        let left_start = claim_idx * self.left_len;
        let right_start = claim_idx * self.right_len;
        let mut left_bar = vec![F::zero(); D];
        let mut right_bar = vec![F::zero(); D];

        for (p, &weight) in u_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, D>(
                    &mut left_bar,
                    &self.left[left_start + p],
                    weight,
                )?;
            }
        }
        for (q, &weight) in v_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, D>(
                    &mut right_bar,
                    &self.right[right_start + q],
                    weight,
                )?;
            }
        }

        let product_eval =
            eval_dense_at_pows(&left_bar, alpha_pows) * eval_dense_at_pows(&right_bar, alpha_pows);
        let mut quotient_eval = F::zero();
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
}

fn accumulate_sparse_scaled<F: FieldCore + CanonicalField, const D: usize>(
    out: &mut [F],
    challenge: &SparseChallenge,
    scale: F,
) -> Result<(), AkitaError> {
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
        out[idx] += scale * F::from_i64(coeff as i64);
    }
    Ok(())
}

fn eval_dense_at_pows<F: FieldCore>(coeffs: &[F], alpha_pows: &[F]) -> F {
    coeffs
        .iter()
        .zip(alpha_pows.iter())
        .fold(F::zero(), |acc, (&coeff, &power)| acc + coeff * power)
}

/// Split `num_blocks = 2^r` into balanced tensor dimensions.
///
/// # Errors
///
/// Returns an error if `num_blocks` is not a power of two.
#[inline]
pub fn tensor_stage1_split(num_blocks: usize) -> Result<(usize, usize), AkitaError> {
    if !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor stage-1 challenges require a power-of-two block count".to_string(),
        ));
    }
    let r = num_blocks.trailing_zeros() as usize;
    let left_bits = r / 2;
    let right_bits = r - left_bits;
    Ok((1usize << left_bits, 1usize << right_bits))
}

/// Sample stage-1 folding challenges using the configured shape.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, if tensor splitting is
/// invalid, or if sparse challenge sampling fails.
pub fn sample_stage1_challenges<F, T, const D: usize>(
    transcript: &mut T,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &Stage1ChallengeShape,
) -> Result<Stage1Challenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    match shape {
        Stage1ChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("stage-1 challenge count overflow".to_string())
            })?;
            sample_sparse_challenges::<F, T, D>(transcript, CHALLENGE_STAGE1_FOLD, total, cfg)
                .map(Stage1Challenges::Flat)
        }
        Stage1ChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_stage1_split(num_blocks)?;
            let left_total = left_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("stage-1 tensor-left challenge count overflow".to_string())
            })?;
            let right_total = right_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "stage-1 tensor-right challenge count overflow".to_string(),
                )
            })?;
            let left = sample_sparse_challenges::<F, T, D>(
                transcript,
                CHALLENGE_STAGE1_FOLD_TENSOR_LEFT,
                left_total,
                cfg,
            )?;
            let right = sample_sparse_challenges::<F, T, D>(
                transcript,
                CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT,
                right_total,
                cfg,
            )?;
            Ok(Stage1Challenges::Tensor(TensorStage1Challenges {
                left,
                right,
                left_len,
                right_len,
                num_claims,
            }))
        }
    }
}
