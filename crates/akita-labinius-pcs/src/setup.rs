//! Selected-row setup-prefix coverage for image preflight.

use akita_config::{required_setup_prefix_slot_ids_for_schedule, ResolvedScheduleRow};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;

pub(crate) fn ensure_setup_prefix_coverage(
    row: &ResolvedScheduleRow,
    contains: impl Fn(&SetupPrefixSlotId) -> bool,
) -> Result<(), AkitaError> {
    let layout = row.profiles().opening_layout()?;
    for id in required_setup_prefix_slot_ids_for_schedule(row.schedule(), &layout)? {
        if !contains(&id) {
            return Err(AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from setup".into(),
            ));
        }
    }
    Ok(())
}
