//! Materialize schedule-required setup-prefix slots for synthetic mixed-D
//! recursive profiles.
//!
//! Catalog recursive configs populate D64 slots during `setup_prover`. The
//! mixed recursive experiment uses a dynamic D128 prefix that is outside that
//! registry contract, so tests and the profile harness share this helper.

use akita_field::{CanonicalField, FieldCore, RandomSampling};
use akita_prover::{commit_setup_prefix, AkitaProverSetup, CommitmentComputeBackend};
use akita_types::{dispatch_for_field, FoldSchedule, NttCacheKey};

/// Commit every missing `incoming_setup_prefix` slot referenced by `schedule`.
///
/// Already-present registry entries are left untouched. NTT material is warmed
/// for both the prefix source dimension and the outer commitment dimension
/// before the slot is committed.
pub fn materialize_schedule_setup_prefix_slots<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    schedule: &FoldSchedule,
) -> Result<(), akita_field::AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    for slot_id in schedule
        .recursive_folds
        .iter()
        .filter_map(|fold| fold.params.incoming_setup_prefix.as_ref())
    {
        if setup.prefix_slots.get(slot_id).is_some() {
            continue;
        }
        for ring_d in [
            slot_id.d_setup,
            slot_id
                .commitment_params
                .layout
                .outer_commit_matrix
                .ring_dimension(),
        ] {
            let ntt_key = NttCacheKey::from_envelope(&setup.expanded, ring_d)?;
            backend.ensure_ntt_slot(prepared, ntt_key)?;
        }
        let n_prefix = slot_id.n_prefix()?;
        let slot = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            slot_id.d_setup,
            |D_SETUP| {
                commit_setup_prefix::<F, D_SETUP, B>(
                    &setup.expanded,
                    backend,
                    prepared,
                    &slot_id.commitment_params,
                    n_prefix,
                    slot_id.natural_len,
                )
            }
        )?;
        setup.prefix_slots.insert(slot)?;
    }
    Ok(())
}
