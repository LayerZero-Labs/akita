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

/// One claim's logical fold weights.
pub(crate) struct PreparedAffineFactors<F> {
    pub(crate) low: Vec<F>,
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    /// Evaluate one claim's outer fold weights as separate high/low factors.
    ///
    /// Flat challenges use an exact live prefix in a power-of-two low vector.
    /// Tensor challenges materialize their wrap-corrected logical evaluations,
    /// never the raw Cartesian high/low product.
    pub(crate) fn affine_factors<Base>(
        &self,
        claim: usize,
        num_live_blocks: usize,
    ) -> Result<PreparedAffineFactors<F>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => {
                if num_live_blocks == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "flat challenge factors require num_live_blocks > 0".into(),
                    ));
                }
                let start = claim.checked_mul(num_live_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor offset overflow".into())
                })?;
                let end = start.checked_add(num_live_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor end overflow".into())
                })?;
                let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                    expected: end,
                    actual: c_alphas.len(),
                })?;
                let low_len = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
                    AkitaError::InvalidSetup("flat challenge factor length overflow".into())
                })?;
                let mut low = vec![F::zero(); low_len];
                low[..num_live_blocks].copy_from_slice(values);
                Ok(PreparedAffineFactors { low })
            }
            Self::Tensor {
                challenges,
                alpha_pows,
            } => {
                if claim >= challenges.num_claims
                    || challenges.num_live_blocks_per_claim != num_live_blocks
                    || challenges.fold_low_len == 0
                {
                    return Err(AkitaError::InvalidSetup(
                        "tensor challenge factors do not match witness blocks".into(),
                    ));
                }
                // A separable high/low product cannot represent the negacyclic
                // wrap correction `− (alpha^D + 1) · quotient(H_h, L_q)` that
                // the reduced tensor-product fold challenge carries. Materialize
                // the per-fold wrap-corrected logical evaluations so every
                // caller can consume the same flat low vector.
                let low_len = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor challenge factor length overflow".into())
                })?;
                let base = claim.checked_mul(num_live_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor challenge factor offset overflow".into())
                })?;
                let mut low = vec![F::zero(); low_len];
                for (fold, slot) in low.iter_mut().take(num_live_blocks).enumerate() {
                    let block_idx = base.checked_add(fold).ok_or_else(|| {
                        AkitaError::InvalidSetup("tensor challenge block index overflow".into())
                    })?;
                    *slot = challenges.eval_logical_at_pows::<Base, F>(block_idx, alpha_pows)?;
                }
                Ok(PreparedAffineFactors { low })
            }
        }
    }
}
