use akita_error::AkitaError;
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, ComputeBackendSetup, CpuBackend, DensePoly,
    NttExecutionRequirements, RuntimeCommitBackendFor,
};
use akita_serialization::Valid;
use akita_types::{dispatch_for_field, SetupPrefixSlotId};
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeSet;

fn commit_setup_prefix_slot<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    id: &SetupPrefixSlotId,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + Valid + 'static,
    B: RuntimeCommitBackendFor<F, DensePoly<F>>,
{
    if setup.prefix_slots.get(id).is_some() {
        return Ok(());
    }
    let n_prefix = id.n_prefix()?;
    let slot = dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        id.d_setup(),
        |D| {
            commit_setup_prefix::<F, D, B>(
                &setup.expanded,
                backend,
                prepared,
                &id.commitment_profile,
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
    F: Field + CanonicalEncoding + Valid + 'static,
    B: RuntimeCommitBackendFor<F, DensePoly<F>>,
{
    let mut requirements = NttExecutionRequirements::default();
    for slot_id in slot_ids {
        if setup.prefix_slots.get(slot_id).is_none() {
            requirements.add_setup_prefix_commitment(0, slot_id)?;
        }
    }
    for requirement in requirements.entries() {
        backend.ensure_ntt_slot(prepared, requirement.key)?;
    }
    for slot_id in slot_ids {
        commit_setup_prefix_slot(setup, backend, prepared, slot_id)?;
    }
    Ok(())
}

pub(crate) fn validate_prefix_registry_complete<F: Field>(
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

pub(crate) fn populate_required_setup_prefix_slots<F>(
    setup: &mut AkitaProverSetup<F>,
    required_ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + Valid + 'static,
{
    if required_ids.is_empty() {
        return Ok(());
    }
    let backend = CpuBackend::DEFAULT;
    let prepared = backend.prepare_setup(setup)?;
    materialize_setup_prefix_slots(setup, &backend, &prepared, required_ids)?;
    validate_prefix_registry_complete(&setup.prefix_slots, required_ids)?;

    tracing::info!(
        slots = setup.prefix_slots.len(),
        "materialized exact setup-prefix commitments"
    );
    Ok(())
}
