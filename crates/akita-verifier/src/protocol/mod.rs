//! Verifier replay for batched, recursive, and ring-switch proof steps.

use akita_field::AkitaError;

pub(crate) mod core;
pub(crate) mod ring_switch;
mod slice_mle;

pub use core::batched_verify;
pub use ring_switch::{
    prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay,
};
pub(crate) use slice_mle::SetupEvaluator;

#[inline]
pub(crate) fn validate_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..128 for verifier gadget evaluation".to_string(),
        ));
    }
    Ok(())
}
