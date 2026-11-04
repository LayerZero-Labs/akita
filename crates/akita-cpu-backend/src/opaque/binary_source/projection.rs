//! Private exact residual replay after the outgoing witness is fixed.

use super::*;
use akita_types::binary_source::BinaryProjectionRequest;

pub(super) fn prepare<F, E>(
    backend: &CpuBackend<F, E>,
    binding: &OperationBinding,
    context: &ProofContext,
    handle: &CommitmentHandle<F, E>,
    request: &BinaryProjectionRequest,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    let source = handle.committed.source.binary_source().ok_or_else(|| {
        AkitaError::InvalidInput("projection requires a binary source handle".into())
    })?;
    source
        .manifest
        .validate_carrier(&handle.committed.parameters)?;
    let (schedule, root_layout) = binding.scope_lease().proof_plan()?;
    if handle.owner != backend.owner_id()
        || context.fold_level() != 0
        || context
            .group_index()
            .and_then(|group| u64::try_from(group).ok())
            != Some(source.manifest.setup_group())
        || schedule.root.params.group_params(&root_layout, 0)?.profile != source.carrier_profile
        || handle.committed.parameters != source.carrier_profile
        || handle.committed.retained.is_some()
    {
        return Err(AkitaError::InvalidInput(
            "projection source owner mismatch".into(),
        ));
    }
    // Rebuild from authenticated private source material and the accepted fold.
    // No retained public residual or projected value is trusted by this boundary.
    let relation = crate::binary_root::BinaryRootRelation::build(
        source.manifest.descriptor(),
        &source.matrix,
        &source.sources.expanded(source.manifest.descriptor())?,
        &source.images,
        &request.challenges,
        akita_challenges::FoldChallengeDistribution::BinaryRoot,
    )?;
    let residuals = relation
        .rows
        .into_iter()
        .map(|row| row.remainder)
        .collect::<Vec<_>>();
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((Vec::new(), Box::new(residuals)))
}
