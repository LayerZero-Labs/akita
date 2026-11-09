//! Private assembly of the complete grouped native root inventory.

use super::*;
use akita_error::checked;
use akita_types::binary_source::{GroupedRootWitnessRequest, GroupedTypedPrefix};
use akita_types::root_repacking::RootCoefficientRepacking;

pub(super) struct Preparation {
    binding: OperationBinding,
    typed: GroupedTypedPrefix,
    binary: Vec<BinaryRootPreparation>,
    ordinary: Vec<Vec<i8>>,
}

pub(super) fn prepare<F, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    request: &GroupedRootWitnessRequest<CommitmentHandle<F, E>>,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    let binding = backend.binding(session, context)?;
    let (schedule, layout) = binding.scope_lease().proof_plan()?;
    let plan = request.binding.plan();
    let signed = plan.ordinary().ok_or(AkitaError::InvalidProof)?;
    let next = schedule.recursive_folds.first().ok_or(AkitaError::InvalidProof)?;
    if context.fold_level() != 0 || context.group_index() != Some(0)
        || request.handles.len() != 8 || request.terminals.len() != 4
        || plan.binary().len() != 4 || signed.plans().len() != 4
        || layout.num_groups() != 8
        || request.binding.successor() != &next.params.own_group().profile
        || next.params.setup_prefix().and_then(|prefix| prefix.slot_id()).as_ref()
            != Some(request.binding.first())
        || request.binding.setup() != backend.prepared()?.expanded.descriptor()
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut payload = Vec::new();
    let mut binary = Vec::new();
    for (index, (local, terminal)) in plan.binary().iter().zip(&request.terminals).enumerate() {
        let source = local.source();
        let handle = request.handles.get(index).ok_or(AkitaError::InvalidProof)?;
        if source.setup_group != index as u64 || source.word != terminal.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        let request = akita_types::binary_source::BinaryRootWitnessRequest {
            manifest: source.manifest.clone(), claim: terminal.clone(), prime: source.prime.clone(),
            plan: local.local().clone(), ordinary: None,
        };
        let (local_payload, preparation) = prepare_binary_root(
            backend, session, &context.for_group(index), handle, &request, Some(plan),
        )?;
        payload.try_reserve_exact(local_payload.len()).map_err(|_| AkitaError::InvalidProof)?;
        payload.extend(local_payload);
        binary.push(*preparation.downcast::<BinaryRootPreparation>()
            .map_err(|_| AkitaError::InvalidProof)?);
    }
    let mut ordinary = Vec::new();
    for (index, source) in signed.plans().iter().enumerate() {
        let group = index + 4;
        let handle = request.handles.get(group).ok_or(AkitaError::InvalidProof)?;
        if &schedule.root.params.group_params(&layout, group)?.profile != source.profile() {
            return Err(AkitaError::InvalidProof);
        }
        let context = context.for_group(group);
        backend.validate_context(session, &context)?;
        ordinary.extend(backend.ordinary_root_preimage(handle, source)?);
        binding.scope_lease().admit_commitment(&context, handle.committed.commitment_id)?;
    }
    Ok((payload, Box::new(Preparation {
        binding, typed: request.binding.clone(), binary, ordinary,
    })))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build<F: Field + CanonicalEncoding, E: Field>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    preparation: &Preparation,
    challenges: &[akita_challenges::Challenges],
    logical_len: usize,
    destination: usize,
) -> Result<Option<CpuWitnessHandle>, AkitaError> {
    let binding = backend.binding(session, context)?;
    backend.validate_binding(&preparation.binding)?;
    preparation.binding.validate_lineage(&binding)?;
    let (schedule, _) = binding.scope_lease().proof_plan()?;
    let plan = preparation.typed.plan();
    let repacking = plan.repacking();
    if context.fold_level() != 0 || context.group_index().is_some() || challenges.len() != 4
        || logical_len != schedule.root.output_witness_len
        || destination != repacking.destination_ring_dimension()
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut digits = Vec::new();
    digits.try_reserve_exact(logical_len).map_err(|_| AkitaError::InvalidProof)?;
    digits.resize(logical_len, 0);
    for ((local, prepared), challenges) in plan.binary().iter().zip(&preparation.binary)
        .zip(challenges)
    {
        backend.validate_binding(&prepared.binding)?;
        prepared.binding.validate_lineage(&binding)?;
        let Some(local_digits) = build_binary_digits(prepared, challenges)? else {
            return Ok(None);
        };
        let last = local.local().repacking().segments().len().checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?;
        let len = local.local().repacking().segment_range(last)?.end;
        let range = checked::range(local.coefficient_offset(), len)
            .ok_or(AkitaError::InvalidProof)?;
        digits.get_mut(range).ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(local_digits.get(..len).ok_or(AkitaError::InvalidProof)?);
    }
    let binary_segments = checked::sum(plan.binary().iter()
        .map(|local| local.local().repacking().segments().len())).ok_or(AkitaError::InvalidProof)?;
    let ordinary = RootCoefficientRepacking::new(
        repacking.segments().get(binary_segments..).ok_or(AkitaError::InvalidProof)?.to_vec(),
        destination,
    )?;
    let slices = preparation.ordinary.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let local = ordinary.repack(&slices)?;
    let first = repacking.segment_range(binary_segments)?.start;
    let len = logical_len.checked_sub(first).ok_or(AkitaError::InvalidProof)?;
    digits.get_mut(first..).ok_or(AkitaError::InvalidProof)?
        .copy_from_slice(local.get(..len).ok_or(AkitaError::InvalidProof)?);
    super::super::binary_root_sessions::import_root_witness(
        backend, session, context, digits, destination,
    ).map(Some)
}
