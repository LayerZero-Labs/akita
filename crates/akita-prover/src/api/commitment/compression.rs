//! Dimension-erased compression for root commitments.

use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{CompressionComputeBackend, OperationCtx};
use akita_error::AkitaError;
use akita_types::{
    CompressionChainPlan, CompressionChainWitness, RingRelationMode, RingVec, SisModulusProfileId,
};
use jolt_field::{CanonicalEncoding, Field};

/// Dimension-erased output of one complete commitment compression chain.
pub(super) struct CommitmentCompressionOutput<F: Field> {
    pub(super) payload: RingVec<F>,
    pub(super) witness: CompressionChainWitness,
    pub(super) quotients: Vec<RingVec<F>>,
}

/// Compute the complete compression chain for one outer commitment image.
pub(super) fn compute_commitment_compression<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    modulus_profile: SisModulusProfileId,
    source: RingVec<F>,
) -> Result<CommitmentCompressionOutput<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    let plan = CompressionChainPlan::for_complete_source(modulus_profile, source.coeff_len())?;
    let (mut outputs, _) = execute_compression_chains(
        ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan,
            coefficients: source.into_coeffs(),
            relation_mode: RingRelationMode::QuotientLift,
        }],
    )?;
    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
    let terminal_ring_dim = output
        .witness
        .plan()
        .maps()
        .last()
        .ok_or(AkitaError::InvalidProof)?
        .ring_dimension();
    let payload =
        RingVec::from_coeffs_with_ring_dim(output.terminal.into_coefficients(), terminal_ring_dim)?;
    let quotients = output.relation.into_quotient_lift()?;
    Ok(CommitmentCompressionOutput {
        payload,
        witness: output.witness,
        quotients,
    })
}
