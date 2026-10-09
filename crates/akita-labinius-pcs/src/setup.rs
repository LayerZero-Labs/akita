//! Selected-row setup-prefix coverage shared by image and root preflight.

use akita_config::{required_setup_prefix_slot_ids_for_schedule, ResolvedScheduleRow};
use akita_error::AkitaError;
use akita_params::SetupPrefixSlotId;

/// Admit only profiles supported by the image and grouped root catalogs.
pub(crate) fn ensure_supported_root(admitted: &crate::RootSetup) -> Result<(), AkitaError> {
    if admitted
        .shape()
        .commitment_modulus()
        .small_modulus()
        .is_some()
    {
        return Err(AkitaError::InvalidSetup(
            "small-modulus LaBinius profile is not supported by the image or root PCS".into(),
        ));
    }
    Ok(())
}

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
