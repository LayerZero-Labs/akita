use akita_config::{opening_schedule_key, CommitmentConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitmentComputeBackend, ComputeBackendSetup,
    CpuBackend,
};
#[cfg(feature = "disk-persistence")]
use akita_serialization::Valid;
#[allow(dead_code)]
use akita_types::PolynomialGroupLayout;
#[cfg(feature = "disk-persistence")]
use akita_types::Schedule;
use akita_types::{
    dispatch_for_field, OpeningClaimsLayout, SetupPrefixSlotId, SETUP_OFFLOAD_D_SETUP,
};
#[cfg(feature = "disk-persistence")]
use std::collections::BTreeSet;

#[allow(dead_code)]
fn setup_prefix_scan_poly_counts(max_num_batched_polys: usize) -> Vec<usize> {
    if max_num_batched_polys <= 1 {
        vec![1]
    } else {
        vec![1, max_num_batched_polys]
    }
}

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

#[cfg(feature = "disk-persistence")]
pub(crate) fn collect_setup_prefix_slot_ids(
    schedule: &Schedule,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    let mut ids = BTreeSet::new();
    for fold in schedule.fold_steps() {
        if let Some(slot_id) = &fold.params.setup_prefix {
            slot_id.check().map_err(|err| {
                AkitaError::InvalidSetup(format!(
                    "runtime schedule contains invalid setup-prefix slot id: {err}"
                ))
            })?;
            ids.insert(slot_id.clone());
        }
    }
    Ok(ids.into_iter().collect())
}

#[cfg(feature = "disk-persistence")]
pub(crate) fn commit_setup_prefix_slots<F, B>(
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

#[allow(dead_code)]
fn populate_prefixes_for_layout<F, Cfg, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    layout: &OpeningClaimsLayout,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
    B: CommitmentComputeBackend<F>,
{
    let Ok(key) = opening_schedule_key::<Cfg>(layout) else {
        return Ok(());
    };
    let Ok(schedule) = Cfg::runtime_schedule(key) else {
        return Ok(());
    };
    for fold in schedule.fold_steps() {
        if let Some(slot_id) = &fold.params.setup_prefix {
            commit_setup_prefix_slot(setup, backend, prepared, slot_id)?;
        }
    }
    Ok(())
}

/// Populate all setup-prefix slots requested by recursive runtime schedules.
///
/// This mirrors the proof-optimized setup-envelope scan: every supported root
/// sub-shape may pick a different recursive schedule and therefore a different
/// setup-prefix slot. The selected `Cfg` is already the recursive adapter.
#[allow(dead_code)]
pub(crate) fn populate_recursive_setup_prefixes<F, Cfg>(
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
        return Ok(());
    }

    let backend = CpuBackend;
    let prepared = backend.prepare_setup(setup)?;
    for main_num_vars in 1..=max_num_vars {
        let precommitted = PolynomialGroupLayout::new(main_num_vars, 1);
        for main_num_polys in setup_prefix_scan_poly_counts(max_num_batched_polys) {
            let main_group = PolynomialGroupLayout::new(main_num_vars, main_num_polys);
            let layout = OpeningClaimsLayout::from_root_groups(&[], main_group)?;
            populate_prefixes_for_layout::<F, Cfg, CpuBackend>(
                setup, &backend, &prepared, &layout,
            )?;

            for num_precommitted in 1..=2 {
                let precommitteds = vec![precommitted; num_precommitted];
                let layout = OpeningClaimsLayout::from_root_groups(&precommitteds, main_group)?;
                populate_prefixes_for_layout::<F, Cfg, CpuBackend>(
                    setup, &backend, &prepared, &layout,
                )?;
            }
        }
    }

    tracing::info!(
        slots = setup.prefix_slots.len(),
        "populated setup-prefix commitments for recursive setup config"
    );
    Ok(())
}
