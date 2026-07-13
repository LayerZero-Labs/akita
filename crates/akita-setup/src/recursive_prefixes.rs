use akita_config::CommitmentConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitmentComputeBackend, ComputeBackendSetup,
    CpuBackend,
};
use akita_types::{dispatch_for_field, SetupPrefixSlotId, SETUP_OFFLOAD_D_SETUP};
use std::collections::BTreeSet;

fn commit_setup_prefix_slot<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    id: &SetupPrefixSlotId,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if id.d_setup != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(
            "setup prefix slot must use the recursive offload dimension".to_string(),
        ));
    }
    if setup.prefix_slots.get(id).is_some() {
        return Ok(());
    }
    let n_prefix = id.n_prefix()?;
    let slot = dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        SETUP_OFFLOAD_D_SETUP,
        |D| {
            commit_setup_prefix::<F, D, B>(
                &setup.expanded,
                backend,
                prepared,
                &id.commitment_params,
                n_prefix,
                id.natural_len,
            )
        }
    )?;
    setup.prefix_slots.insert(slot)?;
    Ok(())
}

pub(crate) fn materialize_setup_prefix_slots<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    slot_ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    for slot_id in slot_ids {
        commit_setup_prefix_slot(setup, backend, prepared, slot_id)?;
    }
    Ok(())
}

pub(crate) fn validate_prefix_registry_complete<F: FieldCore>(
    registry: &akita_types::SetupPrefixProverRegistry<F>,
    required_ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError> {
    let required: BTreeSet<_> = required_ids.iter().cloned().collect();
    let present: BTreeSet<_> = registry.iter().map(|(id, _)| id.clone()).collect();
    if required != present {
        return Err(AkitaError::InvalidSetup(format!(
            "setup-prefix registry mismatch: required {} slots, have {}",
            required.len(),
            present.len()
        )));
    }
    Ok(())
}

pub(crate) fn populate_required_setup_prefix_slots<F, Cfg>(
    setup: &mut AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    if !Cfg::recursive_setup_planning() {
        return Ok(());
    }
    let gen_ring_dim = setup.expanded.seed().gen_ring_dim;
    if gen_ring_dim != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires setup generation at D64".to_string(),
        ));
    }

    let required_ids =
        akita_config::setup_prefix_slot_ids_for_capacity::<Cfg>(max_num_vars, max_num_batched_polys)?;
    let backend = CpuBackend;
    let prepared = backend.prepare_setup(setup)?;
    materialize_setup_prefix_slots(setup, &backend, &prepared, &required_ids)?;
    validate_prefix_registry_complete(&setup.prefix_slots, &required_ids)?;

    tracing::info!(
        slots = setup.prefix_slots.len(),
        "materialized exact setup-prefix commitments"
    );
    Ok(())
}
