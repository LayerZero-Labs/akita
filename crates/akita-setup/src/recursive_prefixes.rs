use akita_cpu_backend::{AkitaProverSetup, CpuBackend};
use akita_error::AkitaError;
use akita_serialization::Valid;
use akita_types::SetupPrefixSlotId;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::BTreeSet;

pub(crate) fn validate_prefix_registry_complete<F: Field>(
    registry: &akita_cpu_backend::SetupPrefixProverRegistry<F>,
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

/// Export the required setup-prefix commitments into `setup`.
///
/// Prefix commitments depend only on the setup and the slot ids. The exporting
/// backend's extension parameter never enters them, so the base field fills it.
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
    let backend = CpuBackend::<F, F>::new(setup.expanded.clone())?;
    setup.prefix_slots = backend.export_setup_prefixes(required_ids)?;
    validate_prefix_registry_complete(&setup.prefix_slots, required_ids)?;
    tracing::info!(
        slots = setup.prefix_slots.len(),
        "materialized exact setup-prefix commitments"
    );
    Ok(())
}
