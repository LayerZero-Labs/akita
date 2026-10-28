use akita_error::AkitaError;
use akita_types::opening_claims::GroupClaimKind;
use akita_types::proof::commitment::CommittedGroupKind;
use akita_types::CommittedGroup;
use jolt_field::Field;

/// Extension seam: validate the payload according to its committed group kind.
/// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
pub(super) fn validate_group_payload<F: Field>(
    committed: &CommittedGroup<F>,
    kind: &GroupClaimKind,
    field_bits: u32,
) -> Result<(), AkitaError> {
    let descriptor = committed.profile();
    descriptor.validate_frozen_precommit(field_bits)?;
    match (committed.kind(), kind) {
        (CommittedGroupKind::Ordinary, GroupClaimKind::Ordinary) => {}
        #[cfg(feature = "labinius-trinomial")]
        (
            CommittedGroupKind::Binary(manifest),
            GroupClaimKind::Binary {
                manifest: claim_manifest,
                claim,
            },
        ) => {
            if manifest != claim_manifest.as_ref() {
                return Err(AkitaError::InvalidInput(
                    "binary claim manifest differs from its commitment".into(),
                ));
            }
            manifest.validate_carrier(descriptor)?;
            claim.validate(manifest.descriptor())?;
            if committed.rows().coeff_len()
                != akita_types::binary_source::BINARY_COMMITTED_PAYLOAD_COEFFICIENTS
            {
                return Err(AkitaError::InvalidInput(
                    "binary commitment length does not match its manifest".into(),
                ));
            }
            return Ok(());
        }
        // Dependency features can expose kinds whose verifier handler is disabled.
        #[allow(unreachable_patterns)]
        _ => {
            return Err(AkitaError::InvalidInput(
                "opening claim kind differs from its commitment".into(),
            ));
        }
    }
    let plan = descriptor.outer_compression_plan()?;
    if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
        return Err(AkitaError::InvalidInput(
            "commitment length does not match its compression chain".to_string(),
        ));
    }
    Ok(())
}
