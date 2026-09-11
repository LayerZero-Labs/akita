use akita_error::AkitaError;
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitmentExecutor, ComputeBackendSetup, CpuBackend,
    DenseType, IntoPortableCommitmentState, PolynomialType, ResidentStatePolicy,
};
use akita_serialization::Valid;
use akita_types::{AkitaExpandedSetup, SetupPrefixProverRegistry, SetupPrefixSlotId};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::BTreeSet;

fn commit_setup_prefix_slot<F, SP>(
    expanded: &AkitaExpandedSetup<F>,
    registry: &mut SetupPrefixProverRegistry<F>,
    executor: &CommitmentExecutor<'_, F, SP>,
    id: &SetupPrefixSlotId,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + Valid + 'static,
    SP: akita_prover::CommitmentStatePolicy<F>,
    SP::State: IntoPortableCommitmentState<F>,
{
    if registry.get(id).is_some() {
        return Ok(());
    }
    let slot = commit_setup_prefix(expanded, executor, id)?;
    registry.insert(slot)?;
    Ok(())
}

pub(crate) fn materialize_setup_prefix_slots<F, SP>(
    expanded: &AkitaExpandedSetup<F>,
    registry: &mut SetupPrefixProverRegistry<F>,
    executor: &CommitmentExecutor<'_, F, SP>,
    slot_ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + Valid + 'static,
    SP: akita_prover::CommitmentStatePolicy<F>,
    SP::State: IntoPortableCommitmentState<F>,
{
    for slot_id in slot_ids {
        commit_setup_prefix_slot(expanded, registry, executor, slot_id)?;
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
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + Valid + 'static,
{
    if required_ids.is_empty() {
        return Ok(());
    }
    let backend = CpuBackend::DEFAULT;
    let prepared = backend.prepare_setup(setup)?;
    let expanded = &setup.expanded;
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        expanded,
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )?;
    materialize_setup_prefix_slots(expanded, &mut setup.prefix_slots, &executor, required_ids)?;
    validate_prefix_registry_complete(&setup.prefix_slots, required_ids)?;

    tracing::info!(
        slots = setup.prefix_slots.len(),
        "materialized exact setup-prefix commitments"
    );
    Ok(())
}
