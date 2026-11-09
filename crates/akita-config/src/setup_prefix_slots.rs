//! Exact setup-prefix slot requirements for recursive setup planning.

use std::collections::BTreeSet;

use akita_error::AkitaError;
use akita_params::{
    active_setup_field_len, padded_setup_prefix_len, FoldSchedule, SetupPrefixSlotId,
};
use akita_schedules::suffix_opening_layout;

fn setup_prefix_slot_matches(
    slot: &SetupPrefixSlotId,
    natural_len: usize,
    n_prefix: usize,
) -> Result<(), AkitaError> {
    let slot_n_prefix = slot.n_prefix()?;
    if slot.natural_len != natural_len {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot natural_len does not match recomputed active setup footprint"
                .to_string(),
        ));
    }
    if slot_n_prefix != n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot padded length does not match recomputed prefix domain".to_string(),
        ));
    }
    Ok(())
}

/// Return the canonical setup-prefix registry keys consumed by one schedule.
///
/// The supplied root layout is also used to revalidate every frozen prefix
/// length against the producer whose setup contribution it replaces.
pub fn required_setup_prefix_slot_ids_for_schedule(
    schedule: &FoldSchedule,
    root_layout: &akita_params::OpeningClaimsLayout,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    enumerate_setup_prefix_slot_ids(schedule, || {
        Ok(active_setup_field_len(&schedule.root.params, root_layout)?
            .max(schedule.root_accumulated_setup_prefix_len))
    })
}

/// Research-only prefix inventory for an exactly resolved native row.
/// The first edge uses grouped geometry; every later edge uses the ordinary audit.
#[cfg(feature = "labinius-trinomial")]
pub fn required_native_setup_prefix_slot_ids_for_row(
    row: &akita_schedules::ResolvedScheduleRow,
    capacity: &akita_types::binary_source::GroupedPrefixCapacity,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    if let Some(plan) = row.grouped_root() {
        if plan.identity() != capacity.plan().identity() {
            return Err(AkitaError::InvalidSetup(
                "native source plan mismatch".into(),
            ));
        }
    }
    let schedule = row.schedule();
    let successor = schedule.recursive_folds.first().ok_or_else(|| {
        AkitaError::InvalidSetup("native setup requires a first successor".into())
    })?;
    let repacking = capacity.plan().repacking();
    let last = repacking.segments().len().checked_sub(1).ok_or_else(|| {
        AkitaError::InvalidSetup("native setup requires a nonempty grouped plan".into())
    })?;
    let physical_cells = repacking.segment_range(last)?.end;
    if !row.is_native_root()
        || successor.params.own_group().profile != *capacity.successor()
        || successor
            .params
            .setup_prefix()
            .and_then(|prefix| prefix.slot_id())
            .as_ref()
            != Some(capacity.first())
        || schedule.root.output_witness_len != physical_cells
        || successor.input_witness_len != physical_cells
    {
        return Err(AkitaError::InvalidSetup(
            "native row disagrees with grouped capacity binding".into(),
        ));
    }
    if schedule.recursive_folds.iter().skip(1).any(|fold| {
        fold.params
            .setup_prefix()
            .and_then(|prefix| prefix.slot_id())
            .as_ref()
            == Some(capacity.first())
    }) {
        return Err(AkitaError::InvalidSetup(
            "ordinary suffix aliases the native setup prefix".into(),
        ));
    }
    enumerate_setup_prefix_slot_ids(schedule, || Ok(capacity.first().natural_len))
}

fn enumerate_setup_prefix_slot_ids(
    schedule: &FoldSchedule,
    first_natural_len: impl Fn() -> Result<usize, AkitaError>,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    schedule.validate_structure()?;

    let mut ids = BTreeSet::new();
    for producer_index in 0..=schedule.recursive_folds.len() {
        let successor_prefix = schedule
            .recursive_folds
            .get(producer_index)
            .and_then(|fold| fold.params.setup_prefix());
        let Some(slot_id) = successor_prefix else {
            continue;
        };
        let natural_len = if producer_index == 0 {
            first_natural_len()?
        } else {
            let producer = &schedule.recursive_folds[producer_index - 1];
            let incoming_len = producer
                .params
                .setup_prefix()
                .as_ref()
                .map(|slot| slot.setup_natural_len.expect("setup prefix group"));
            let opening_layout = suffix_opening_layout(producer.input_witness_len, incoming_len)?;
            active_setup_field_len(&producer.params, &opening_layout)?
        };
        let n_prefix = padded_setup_prefix_len(natural_len);
        let commitment_id = slot_id.slot_id().expect("setup prefix group");
        setup_prefix_slot_matches(&commitment_id, natural_len, n_prefix)?;
        if !ids.insert(commitment_id) {
            continue;
        }
    }

    Ok(ids.into_iter().collect())
}
