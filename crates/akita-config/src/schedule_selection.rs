//! Shared batched schedule selection for prove and verify entry points.

use crate::CommitmentConfig;
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    dispatch_ring_dim_result, folded_root_supports_opening_shape, root_direct_schedule,
    root_tensor_projection_enabled, schedule_root_fold_step, FpExtEncoding, OpeningClaimsLayout,
    Schedule,
};

/// Select the effective runtime schedule for a batched opening, including the
/// root-direct rewrite when the folded-root opening geometry is unsupported.
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
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
{
    let num_vars = opening_batch.max_num_vars();
    let root_direct_witness_len = opening_batch.root_direct_witness_len()?;
    let mut schedule = Cfg::get_params_for_prove(opening_batch)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        let supports_opening_shape =
            dispatch_ring_dim_result!(root_step.params.ring_dimension, |D| Ok(
                folded_root_supports_opening_shape::<Cfg::Field, Cfg::ExtField, D>(
                    std::slice::from_ref(&opening_point),
                    &root_step.params,
                    alpha_bits,
                )
            ))?;
        let tensor_projection_enabled = root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(
            root_step.params.ring_dimension,
            num_vars,
        );
        if !supports_opening_shape && !tensor_projection_enabled {
            let commit_params = Cfg::get_params_for_batched_commitment(opening_batch)?;
            schedule = root_direct_schedule(root_direct_witness_len, commit_params)?;
        }
    }

    Ok(schedule)
}
