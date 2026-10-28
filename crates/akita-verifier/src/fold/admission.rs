use akita_error::AkitaError;
use akita_types::CommittedGroup;
use jolt_field::Field;

/// Extension seam: validate the payload according to its committed group kind.
/// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
pub(super) fn validate_group_payload<F: Field>(
    committed: &CommittedGroup<F>,
    field_bits: u32,
) -> Result<(), AkitaError> {
    let descriptor = committed.profile();
    descriptor.validate_frozen_precommit(field_bits)?;
    let plan = descriptor.outer_compression_plan()?;
    if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
        return Err(AkitaError::InvalidInput(
            "commitment length does not match its compression chain".to_string(),
        ));
    }
    Ok(())
}
