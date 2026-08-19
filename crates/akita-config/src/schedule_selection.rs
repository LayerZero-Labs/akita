//! Shared batched schedule selection for prove and verify entry points.

use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::OpeningClaimsLayout;

/// Select the effective folded runtime schedule for a batched opening.
///
/// Prove and verify must call this helper so fold-vs-direct decisions dispatch
/// on the schedule root `ring_dimension`, not a caller-supplied stack `D`.
/// `final_group_point` is the final group's complete opening point; root
/// commitment geometry remains final/source-local even when a precommitted
/// group determines the batch's maximum opening arity.
///
/// # Errors
///
/// Returns an error when schedule lookup fails or an unsupported ring dimension
/// is encountered during dispatch.
pub fn effective_batched_schedule<Cfg>(
    resolved: akita_schedules::ResolvedScheduleRow,
    opening_batch: &OpeningClaimsLayout,
    _final_group_point: &[Cfg::ExtField],
) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if resolved.profiles().opening_layout()? != *opening_batch {
        return Err(AkitaError::InvalidInput(
            "committed-group descriptors do not match the opening layout".to_string(),
        ));
    }
    resolved.schedule().validate_structure()?;

    Ok(resolved)
}
