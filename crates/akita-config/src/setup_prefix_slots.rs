//! Exact setup-prefix slot requirements for recursive setup planning.

use std::collections::BTreeSet;

use akita_error::AkitaError;
use akita_schedules::suffix_opening_layout;
use akita_types::{
    active_setup_field_len, padded_setup_prefix_len, FoldSchedule, SetupPrefixSlotId,
};

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

pub(crate) fn extract_setup_prefix_slot_ids_from_schedule(
    schedule: &FoldSchedule,
    root_layout: &akita_types::OpeningClaimsLayout,
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
        let (params, opening_layout) = if producer_index == 0 {
            (&schedule.root.params, root_layout.clone())
        } else {
            let producer = &schedule.recursive_folds[producer_index - 1];
            let incoming_len = producer
                .params
                .setup_prefix()
                .as_ref()
                .map(|slot| slot.setup_natural_len.expect("setup prefix group"));
            (
                &producer.params,
                suffix_opening_layout(producer.input_witness_len, incoming_len)?,
            )
        };
        let natural_len = active_setup_field_len(params, &opening_layout)?;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let commitment_id = slot_id.slot_id().expect("setup prefix group");
        setup_prefix_slot_matches(&commitment_id, natural_len, n_prefix)?;
        if !ids.insert(commitment_id) {
            continue;
        }
    }

    Ok(ids.into_iter().collect())
}
