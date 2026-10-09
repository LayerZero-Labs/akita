use akita_cpu_backend::{AkitaProverSetup, CpuBackend};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaVerifierSetup,
};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::collections::BTreeSet;
use std::sync::Arc;

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

/// Recompute every setup-prefix commitment of `setup` and authenticate its registry.
///
/// Decoding a verifier setup checks the registry's structure only. This
/// derives each slot's prefix of the public stream from the setup seed,
/// commits it with the slot's profile, and requires the stored commitment to
/// match. The cost is that of committing the longest prefix, the same work the
/// setup builder performed.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when a stored commitment differs from
/// its recomputation or a slot cannot be committed.
pub fn authenticate_verifier_setup_prefixes<F>(
    setup: AkitaVerifierSetup<F>,
) -> Result<AkitaVerifierSetup<F>, AkitaError>
where
    F: Field
        + CanonicalEncoding
        + Unreduced
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + 'static,
{
    let ids: Vec<SetupPrefixSlotId> = setup
        .prefix_slots()
        .iter()
        .map(|(id, _)| id.clone())
        .collect();
    let num_field_elements = ids
        .iter()
        .map(SetupPrefixSlotId::n_prefix)
        .try_fold(0, |longest, n_prefix| n_prefix.map(|n| longest.max(n)))?;
    if num_field_elements == 0 {
        return Ok(setup.assume_prefix_registry_authenticated());
    }
    let seed = &setup.expanded().descriptor().setup_seed;
    let stream = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            num_field_elements,
            ..setup.expanded().descriptor().clone()
        },
        derive_public_matrix_prefix::<F>(num_field_elements, seed),
    );
    let recomputed = CpuBackend::<F, F>::new(Arc::new(stream))?.export_setup_prefixes(&ids)?;
    for (id, slot) in setup.prefix_slots().iter() {
        let expected = recomputed
            .get(id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix slot was not recomputed".into()))?
            .verifier_slot();
        if expected.commitment.rows.len() != slot.commitment.rows.len()
            || expected
                .commitment
                .rows
                .iter()
                .zip(&slot.commitment.rows)
                .any(|(expected, stored)| expected.coeffs() != stored.coeffs())
        {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix commitment does not match its public stream prefix".into(),
            ));
        }
    }
    Ok(setup.assume_prefix_registry_authenticated())
}
