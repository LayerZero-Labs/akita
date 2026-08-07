//! Shared batched schedule selection for prove and verify entry points.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, folded_root_supports_opening_shape, root_tensor_projection_enabled,
    FpExtEncoding, OpeningClaimsLayout,
};
use jolt_field::Field;

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
    final_group_point: &[Cfg::ExtField],
) -> Result<akita_schedules::ResolvedScheduleRow, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
{
    let num_vars = opening_batch.max_num_vars();
    if resolved.profiles().opening_layout()? != *opening_batch {
        return Err(AkitaError::InvalidInput(
            "committed-group descriptors do not match the opening layout".to_string(),
        ));
    }
    let schedule = resolved.schedule();
    schedule.validate_structure()?;
    let root_step = &schedule.root;
    let root_params = &root_step.params.final_group.commitment;
    let alpha_bits = root_params.d_a().trailing_zeros() as usize;
    let supports_opening_shape = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        root_params.d_a(),
        |D| Ok(folded_root_supports_opening_shape::<
            Cfg::Field,
            Cfg::ExtField,
            D,
        >(
            std::slice::from_ref(&final_group_point),
            root_params,
            alpha_bits,
        ))
    )?;
    let tensor_projection_enabled =
        root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(root_params.d_a(), num_vars);

    if !supports_opening_shape && !tensor_projection_enabled {
        return Err(AkitaError::UnsupportedSchedule(
            "folded-root opening geometry is unsupported".to_string(),
        ));
    }

    Ok(resolved)
}
