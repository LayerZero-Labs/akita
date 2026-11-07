//! CPU assembly of one research witness using the canonical local builders.

use super::*;
use akita_types::binary_source::{
    BinaryRootWitnessRequest, ResearchBatchPlan, ResearchBatchResponse, ResearchBatchWitnessRequest,
};
use akita_types::opening_claims::GroupClaimKind;

struct Preparation {
    binding: OperationBinding,
    plan: ResearchBatchPlan,
    binary: Vec<BinaryRootPreparation>,
    ordinary: Vec<i8>,
}

pub(super) fn prepare<F, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    request: &ResearchBatchWitnessRequest<CommitmentHandle<F, E>>,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    let binding = backend.binding(session, context)?;
    let (schedule, layout) = binding.scope_lease().proof_plan()?;
    let next = schedule
        .recursive_folds
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if context.fold_level() != 0
        || context.group_index().is_some()
        || request.handles.len() != 8
        || request.kinds.len() != 8
        || layout.num_groups() != 8
        || request.plan.physical_cells() != schedule.root.output_witness_len
        || request
            .plan
            .ordinary_repacking()
            .destination_ring_dimension()
            != next.params.d_a()
        || next.params.setup_prefix().is_some()
    {
        return Err(AkitaError::InvalidInput(
            "research witness admission mismatch".into(),
        ));
    }
    let mut binary = Vec::with_capacity(4);
    let mut payload = Vec::new();
    for (index, ((kind, handle), local)) in request
        .kinds
        .iter()
        .zip(&request.handles)
        .zip(request.plan.binary())
        .enumerate()
    {
        let GroupClaimKind::Binary {
            manifest,
            claim,
            prime,
        } = kind
        else {
            return Err(AkitaError::InvalidProof);
        };
        if claim.is_some() != local.terminal().is_some() || (index != 0 && claim.is_some()) {
            return Err(AkitaError::InvalidProof);
        }
        let group_context = context.for_group(index);
        let group_binding = backend.binding(session, &group_context)?;
        let local_request = BinaryRootWitnessRequest {
            manifest: manifest.as_ref().clone(),
            claim: local.terminal().cloned(),
            prime: prime.clone(),
            plan: local.plan().clone(),
            ordinary: None,
        };
        let (d, prepared) = prepare_binary(
            backend,
            session,
            group_binding,
            &group_context,
            handle,
            &local_request,
            true,
        )?;
        payload
            .try_reserve_exact(d.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        payload.extend(d);
        binary.push(
            *prepared
                .downcast::<BinaryRootPreparation>()
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    if binary.len() != 4 {
        return Err(AkitaError::InvalidProof);
    }
    let mut segments = Vec::new();
    for (index, (plan, handle)) in request
        .plan
        .ordinary()
        .plans()
        .iter()
        .zip(request.handles.iter().skip(4))
        .enumerate()
    {
        let group = checked::sum([index, 4]).ok_or(AkitaError::InvalidProof)?;
        let GroupClaimKind::ResearchOrdinary { owners, width, .. } =
            request.kinds.get(group).ok_or(AkitaError::InvalidProof)?
        else {
            return Err(AkitaError::InvalidProof);
        };
        if schedule.root.params.group_params(&layout, group)?.profile != *plan.profile()
            || Some(*width) != plan.source_width()
            || owners.len() != plan.profile().group.num_polynomials()
        {
            return Err(AkitaError::InvalidProof);
        }
        let group_context = context.for_group(group);
        backend.validate_context(session, &group_context)?;
        segments.extend(backend.ordinary_root_preimage(handle, plan)?);
        binding
            .scope_lease()
            .admit_commitment(&group_context, handle.committed.commitment_id)?;
    }
    let ordinary = request
        .plan
        .ordinary_repacking()
        .repack(&segments.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    Ok((
        payload,
        Box::new(Preparation {
            binding,
            plan: request.plan.clone(),
            binary,
            ordinary,
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build<F, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    preparation: &dyn std::any::Any,
    response: &ResearchBatchResponse,
    logical_len: usize,
    destination_ring: usize,
) -> Result<Option<CpuWitnessHandle>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    let preparation = preparation
        .downcast_ref::<Preparation>()
        .ok_or(AkitaError::InvalidProof)?;
    let binding = backend.binding(session, context)?;
    backend.validate_binding(&preparation.binding)?;
    preparation.binding.validate_lineage(&binding)?;
    if response.challenges.len() != 4
        || logical_len != preparation.plan.physical_cells()
        || context.fold_level() != 0
        || context.group_index().is_some()
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut local = Vec::with_capacity(4);
    for (prepared, challenges) in preparation.binary.iter().zip(&response.challenges) {
        let len = prepared.plan.repacking().domain_len();
        let Some(digits) = build_digits(prepared, challenges, len)? else {
            return Ok(None);
        };
        local.push(digits);
    }
    let mut digits = Vec::new();
    digits
        .try_reserve_exact(logical_len)
        .map_err(|_| AkitaError::InvalidProof)?;
    digits.resize(logical_len, 0);
    for (block, &offset) in local
        .iter()
        .chain([&preparation.ordinary])
        .zip(preparation.plan.offsets())
    {
        let range = checked::range(offset, block.len()).ok_or(AkitaError::InvalidProof)?;
        digits
            .get_mut(range)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(block);
    }
    super::super::binary_root_sessions::import_root_witness(
        backend,
        session,
        context,
        digits,
        destination_ring,
    )
    .map(Some)
}
