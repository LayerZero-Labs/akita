use akita_challenges::TensorChallenges as TensorChallengeSet;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

/// Challenge evaluations used by relation-matrix challenge replay.
#[derive(Clone)]
pub(crate) enum PreparedChallengeEvals<F: FieldCore> {
    Flat(Vec<F>),
    Tensor {
        challenges: TensorChallengeSet,
        alpha_pows: Vec<F>,
    },
}

/// One claim's factorized outer weights for the affine interval kernel.
pub(crate) struct PreparedAffineFactors<F> {
    pub(crate) high: Vec<F>,
    pub(crate) low: Vec<F>,
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    /// Evaluate one claim's outer fold weights as separate high/low factors.
    ///
    /// Flat challenges use one high factor and an exact live prefix in a
    /// power-of-two low vector. Tensor challenges evaluate only their sampled
    /// high and low factors, never the Cartesian logical fold product.
    pub(crate) fn affine_factors<Base>(
        &self,
        claim: usize,
        live_fold_count: usize,
    ) -> Result<PreparedAffineFactors<F>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => {
                if live_fold_count == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "flat challenge factors require a live fold".into(),
                    ));
                }
                let start = claim.checked_mul(live_fold_count).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor offset overflow".into())
                })?;
                let end = start.checked_add(live_fold_count).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor end overflow".into())
                })?;
                let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                    expected: end,
                    actual: c_alphas.len(),
                })?;
                let low_len = live_fold_count.checked_next_power_of_two().ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor length overflow".into())
                })?;
                let mut low = vec![F::zero(); low_len];
                low[..live_fold_count].copy_from_slice(values);
                Ok(PreparedAffineFactors {
                    high: vec![F::one()],
                    low,
                })
            }
            Self::Tensor {
                challenges,
                alpha_pows,
            } => {
                if claim >= challenges.num_claims
                    || challenges.live_folds_per_claim != live_fold_count
                    || challenges.fold_low_len == 0
                {
                    return Err(AkitaError::InvalidSetup(
                        "tensor challenge factors do not match witness blocks".into(),
                    ));
                }
                let high_len = challenges.fold_high_len();
                let high_start = claim.checked_mul(high_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor high factor offset overflow".into())
                })?;
                let high_end = high_start.checked_add(high_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor high factor end overflow".into())
                })?;
                let low_start = claim.checked_mul(challenges.fold_low_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor low factor offset overflow".into())
                })?;
                let low_end = low_start
                    .checked_add(challenges.fold_low_len)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("tensor low factor end overflow".into())
                    })?;
                let high = challenges
                    .fold_high
                    .get(high_start..high_end)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .map(|challenge| challenge.eval_at_pows::<Base, F>(alpha_pows))
                    .collect::<Result<Vec<_>, _>>()?;
                let low = challenges
                    .fold_low
                    .get(low_start..low_end)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .map(|challenge| challenge.eval_at_pows::<Base, F>(alpha_pows))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PreparedAffineFactors { high, low })
            }
        }
    }
}
