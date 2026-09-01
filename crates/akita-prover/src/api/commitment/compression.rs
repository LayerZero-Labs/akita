//! Dimension-erased compression for root commitments.

use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_error::AkitaError;
use akita_types::{CompressionChainWitness, RingVec};
use jolt_field::{CanonicalEncoding, Field};

use super::CheckedCommitmentPlan;

/// Dimension-erased output of one complete commitment compression chain.
pub(super) struct CommitmentCompressionOutput<F: Field> {
    pub(super) payload: RingVec<F>,
    pub(super) witness: CompressionChainWitness,
    pub(super) quotients: Vec<RingVec<F>>,
}

/// Compute the complete compression chain for one outer commitment image.
pub(super) fn compute_commitment_compression<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    plan: &CheckedCommitmentPlan,
    source: RingVec<F>,
) -> Result<CommitmentCompressionOutput<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    let compression = plan.compression();
    let terminal_ring_dim = compression
        .maps()
        .last()
        .ok_or(AkitaError::InvalidSetup(
            "commitment compression plan has no maps".into(),
        ))?
        .ring_dimension();
    let (mut outputs, _) = execute_compression_chains(
        ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan: compression.clone(),
            coefficients: source.into_coeffs(),
        }],
    )?;
    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
    if output.witness.plan() != compression {
        return Err(AkitaError::InvalidProof);
    }
    let payload =
        RingVec::from_coeffs_with_ring_dim(output.terminal.into_coefficients(), terminal_ring_dim)?;
    Ok(CommitmentCompressionOutput {
        payload,
        witness: output.witness,
        quotients: output.quotients,
    })
}
