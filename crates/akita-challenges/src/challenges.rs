//! Sampled sparse challenges for one folding round.

use crate::SparseChallenge;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

/// Stage-1 fold challenges in claim-major block order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Challenges {
    /// Per-(claim, block) sparse challenges.
    challenges: Vec<SparseChallenge>,
    /// Exact number of live blocks packed into one claim.
    num_live_blocks_per_claim: usize,
    /// Number of claims represented by this vector.
    num_claims: usize,
}

impl Challenges {
    /// Construct challenges from a sampled vector and its claim/block layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector length does not match the layout.
    pub fn from_sparse(
        challenges: Vec<SparseChallenge>,
        num_live_blocks_per_claim: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let expected = num_live_blocks_per_claim
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        Ok(Self {
            challenges,
            num_live_blocks_per_claim,
            num_claims,
        })
    }

    /// Return the sparse challenges in claim-major block order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[SparseChallenge] {
        &self.challenges
    }

    /// Number of block challenges represented by this value.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.challenges.len()
    }

    /// Whether this challenge set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }

    /// Number of claims represented by this challenge set.
    #[inline]
    #[must_use]
    pub fn num_claims(&self) -> usize {
        self.num_claims
    }

    /// Number of logical block challenges per claim.
    #[inline]
    #[must_use]
    pub fn num_live_blocks_per_claim(&self) -> usize {
        self.num_live_blocks_per_claim
    }

    /// Evaluate one challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the challenge is invalid.
    pub fn eval_at_pows<F, E>(&self, index: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        self.challenges
            .get(index)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "challenge index {index} out of range for {} challenges",
                    self.challenges.len()
                ))
            })?
            .eval_at_pows::<F, E>(alpha_pows)
    }

    /// Evaluate every challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the powers or any challenge are invalid.
    pub fn evals_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        self.challenges
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
            .collect()
    }

    /// Select complete claims in the requested order.
    ///
    /// # Errors
    ///
    /// Returns an error if any claim index is out of range.
    pub fn select_claims(&self, claim_indices: &[usize]) -> Result<Self, AkitaError> {
        let capacity = claim_indices
            .len()
            .checked_mul(self.num_live_blocks_per_claim)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        let mut selected = Vec::with_capacity(capacity);
        for &claim_index in claim_indices {
            let start = claim_index
                .checked_mul(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            let end = start
                .checked_add(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            selected.extend_from_slice(self.challenges.get(start..end).ok_or(
                AkitaError::InvalidSize {
                    expected: end,
                    actual: self.challenges.len(),
                },
            )?);
        }
        Self::from_sparse(
            selected,
            self.num_live_blocks_per_claim,
            claim_indices.len(),
        )
    }
}
