use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend, CpuPreparedSetup};
use akita_types::{CommittedGroupParams, FoldSchedule, NttCacheKey};
use std::collections::BTreeSet;

/// Register every full-envelope NTT dimension selected by a benchmark schedule.
///
/// `ComputeBackendSetup::prepare_setup` promises only the setup-generation
/// dimension. A mixed schedule legitimately consumes divisor dimensions later,
/// so the profile harness makes those cache slots part of preparation rather
/// than letting the first prove operation build them lazily.
pub(crate) fn register_schedule_ntt_contract<FF>(
    setup: &AkitaProverSetup<FF>,
    prepared: &CpuPreparedSetup<FF>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    FF: FieldCore + CanonicalField,
{
    let mut ring_dimensions = BTreeSet::new();
    let mut add_level_dimensions = |params: &CommittedGroupParams| {
        let dims = params.role_dims();
        ring_dimensions.extend([dims.d_a(), dims.d_b(), dims.d_d()]);
        ring_dimensions.extend(params.precommitted_group_iter().flat_map(|group| {
            let group_dims = group.role_dims(dims.d_d());
            [group_dims.d_a(), group_dims.d_b(), group_dims.d_d()]
        }));
    };
    add_level_dimensions(&schedule.root.params.final_group.commitment);
    for fold in &schedule.recursive_folds {
        add_level_dimensions(&fold.params.witness);
    }
    ring_dimensions.insert(schedule.terminal.params.witness.d_a());

    for ring_d in ring_dimensions {
        let key = NttCacheKey::from_envelope(setup.expanded.as_ref(), ring_d)?;
        CpuBackend.register_setup_contract_ntt_slot(prepared, key)?;
    }
    Ok(())
}
