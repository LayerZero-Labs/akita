//! Verifier replay for batched, recursive, and ring-switch proof steps.

use akita_field::AkitaError;
use akita_types::LevelParams;

pub(crate) mod core;
pub(crate) mod ring_switch;
mod slice_mle;

pub use core::{batched_verify, batched_verify_shaped, batched_verify_shaped_root_direct};
pub use ring_switch::{prepare_ring_switch_row_eval, RingSwitchDeferredRowEval, RingSwitchReplay};
pub(crate) use slice_mle::{SetupEvalPlan, SetupEvaluator};

#[inline]
pub(crate) fn validate_ring_dispatch<const D: usize>() -> Result<usize, AkitaError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "ring dimension must be a non-zero power of two".to_string(),
        ));
    }
    Ok(D.trailing_zeros() as usize)
}

#[inline]
pub(crate) fn validate_level_dispatch<const D: usize>(
    lp: &LevelParams,
) -> Result<usize, AkitaError> {
    let ring_bits = validate_ring_dispatch::<D>()?;
    if lp.ring_dimension != D {
        return Err(AkitaError::InvalidSetup(
            "LevelParams ring dimension does not match verifier dispatch".to_string(),
        ));
    }
    Ok(ring_bits)
}

#[inline]
pub(crate) fn validate_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..128 for verifier gadget evaluation".to_string(),
        ));
    }
    Ok(())
}
