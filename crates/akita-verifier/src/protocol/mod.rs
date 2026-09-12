//! Verifier replay for batched, recursive, and ring-switch proof steps.

use akita_error::AkitaError;

pub(crate) mod core;
pub(crate) mod evaluation_trace;
pub(crate) mod jl;
pub(crate) mod ring_switch;

pub use core::batched_verify;
#[cfg(any(test, feature = "benchmark-support"))]
pub use evaluation_trace::{evaluation_trace_benchmark_case, EvaluationTraceBenchmarkCase};
pub use jl::{
    absorb_jl_aligned_et_verification_images, absorb_jl_projection_verification_images,
    prepare_jl_aligned_et_verification, prepare_jl_projection_verification,
    verify_jl_aligned_et_reduction, verify_jl_projection_reduction_batch, JlAlignedEtSourceClaims,
    JlAlignedEtVerifierReduction, JlVerifierReductionBatch, PreparedJlAlignedEtVerification,
    PreparedJlProjectionVerification,
};
pub use ring_switch::{
    prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay,
};
#[cfg(feature = "benchmark-support")]
pub use ring_switch::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
    RelationEvaluatorBenchmarkCase,
};

#[inline]
pub(crate) fn validate_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..128 for verifier gadget evaluation".to_string(),
        ));
    }
    Ok(())
}
