//! Shared batched schedule selection for prove and verify entry points.

use crate::CommitmentConfig;
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    dispatch_for_field, folded_root_supports_opening_shape, root_tensor_projection_enabled,
    FoldSchedule, FpExtEncoding, OpeningClaimsLayout,
};

/// Select the effective folded runtime schedule for a batched opening.
///
/// Prove and verify must call this helper so fold-vs-direct decisions dispatch
/// on the schedule root `ring_dimension`, not a caller-supplied stack `D`.
///
/// # Errors
///
/// Returns an error when schedule lookup fails or an unsupported ring dimension
/// is encountered during dispatch.
pub fn effective_batched_schedule<Cfg>(
    opening_batch: &OpeningClaimsLayout,
    opening_point: &[Cfg::ExtField],
) -> Result<FoldSchedule, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
{
    let num_vars = opening_batch.max_num_vars();
    let schedule = Cfg::get_params_for_prove(opening_batch)?;
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
            std::slice::from_ref(&opening_point),
            root_params,
            alpha_bits,
        ))
    )?;
    let tensor_projection_enabled =
        root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(root_params.d_a(), num_vars);

    if opening_batch.num_groups() > 1 && Cfg::EXT_DEGREE != 1 {
        return Err(AkitaError::UnsupportedSchedule(
            "multi-group extension openings are not supported".to_string(),
        ));
    }
    if !supports_opening_shape && !tensor_projection_enabled {
        return Err(AkitaError::UnsupportedSchedule(
            "folded-root opening geometry is unsupported".to_string(),
        ));
    }

    Ok(schedule)
}
