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
                // The affine kernel multiplies its two outer factors as a bare
                // product `high[i / Q] · low[i % Q]`. That separable shape cannot
                // represent the negacyclic wrap correction
                // `− (alpha^D + 1) · quotient(H_h, L_q)` that the reduced
                // tensor-product fold challenge carries, so feeding the raw
                // fold-high/fold-low evaluations would drop the wrap term and
                // disagree with the prover (which uses the wrap-corrected
                // reduced product). Materialize the per-fold wrap-corrected
                // logical evaluations here and return them in the same
                // single-high-factor shape as the flat branch, so the kernel
                // computes `1 · low[f]` and matches the prover exactly.
                let low_len = live_fold_count.checked_next_power_of_two().ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor challenge factor length overflow".into())
                })?;
                let base = claim.checked_mul(live_fold_count).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor challenge factor offset overflow".into())
                })?;
                let mut low = vec![F::zero(); low_len];
                for (fold, slot) in low.iter_mut().take(live_fold_count).enumerate() {
                    let block_idx = base.checked_add(fold).ok_or_else(|| {
                        AkitaError::InvalidSetup("tensor challenge block index overflow".into())
                    })?;
                    *slot = challenges.eval_logical_at_pows::<Base, F>(block_idx, alpha_pows)?;
                }
                Ok(PreparedAffineFactors {
                    high: vec![F::one()],
                    low,
                })
            }
        }
    }
}
