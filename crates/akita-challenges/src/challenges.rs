//! Sampled sparse challenges for one folding round.

use crate::SparseChallenge;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

#[cfg(feature = "parallel")]
fn evals_at_pows_parallel<F, E>(
    challenges: &[SparseChallenge],
    evaluations: &mut [E],
    alpha_pows: &[E],
    sparse_terms: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    if evaluations.len() != challenges.len() {
        return Err(AkitaError::InvalidInput(
            "parallel challenge evaluation length mismatch".into(),
        ));
    }
    const PARALLEL_LEAF_TERMS: usize = 1 << 15;
    if sparse_terms <= PARALLEL_LEAF_TERMS || challenges.len() < 2 {
        for (evaluation, challenge) in evaluations.iter_mut().zip(challenges) {
            *evaluation = challenge.eval_at_pows::<F, E>(alpha_pows)?;
        }
        return Ok(());
    }

    let midpoint = challenges.len() / 2;
    let (left_challenges, right_challenges) = challenges.split_at(midpoint);
    let (left_evaluations, right_evaluations) = evaluations.split_at_mut(midpoint);
    let left_terms = left_challenges.iter().try_fold(0usize, |sum, challenge| {
        sum.checked_add(challenge.positions.len())
            .ok_or(AkitaError::InvalidProof)
    })?;
    let right_terms = sparse_terms
        .checked_sub(left_terms)
        .ok_or(AkitaError::InvalidProof)?;
    let (left_result, right_result) = cfg_join!(
        || evals_at_pows_parallel::<F, E>(
            left_challenges,
            left_evaluations,
            alpha_pows,
            left_terms,
        ),
        || evals_at_pows_parallel::<F, E>(
            right_challenges,
            right_evaluations,
            alpha_pows,
            right_terms,
        )
    );
    left_result?;
    right_result
}

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
        let evaluate = |challenge: &SparseChallenge| challenge.eval_at_pows::<F, E>(alpha_pows);
        #[cfg(feature = "parallel")]
        let parallel_candidate = {
            const MIN_PARALLEL_CHALLENGES: usize = 1 << 11;
            const MIN_LIGHTWEIGHT_CHALLENGES: usize = 1 << 12;
            let challenge_count = self.challenges.len();
            challenge_count >= MIN_LIGHTWEIGHT_CHALLENGES
                || (challenge_count >= MIN_PARALLEL_CHALLENGES && alpha_pows.len() <= 128)
        };
        #[cfg(feature = "parallel")]
        if parallel_candidate {
            const MIN_PARALLEL_TERMS: usize = 1 << 15;
            let sparse_terms = self.challenges.iter().try_fold(0usize, |sum, challenge| {
                sum.checked_add(challenge.positions.len()).ok_or_else(|| {
                    AkitaError::InvalidInput("sparse challenge term count overflow".into())
                })
            })?;
            if sparse_terms < MIN_PARALLEL_TERMS {
                return self.challenges.iter().map(evaluate).collect();
            }
            let mut evaluations = Vec::new();
            evaluations
                .try_reserve_exact(self.challenges.len())
                .map_err(|_| {
                    AkitaError::InvalidInput("challenge evaluation allocation failed".into())
                })?;
            evaluations.resize(self.challenges.len(), E::zero());
            evals_at_pows_parallel::<F, E>(
                &self.challenges,
                &mut evaluations,
                alpha_pows,
                sparse_terms,
            )?;
            return Ok(evaluations);
        }

        self.challenges.iter().map(evaluate).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn batch_evaluation_preserves_challenge_order() {
        let challenges = (0..4096)
            .map(|index| SparseChallenge {
                positions: (0..32).collect::<Vec<_>>().into(),
                coeffs: (0..32)
                    .map(|position| if (index + position) % 2 == 0 { 1 } else { -1 })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>();
        let batch = Challenges::from_sparse(challenges, 4096, 1).unwrap();
        let powers = (0..32)
            .map(|index| F::from_u64(index + 1))
            .collect::<Vec<_>>();
        let actual = batch.evals_at_pows::<F, F>(&powers).unwrap();
        let expected = batch
            .as_slice()
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, F>(&powers).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
