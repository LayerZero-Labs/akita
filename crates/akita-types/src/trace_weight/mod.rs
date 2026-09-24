//! Public multilinear weights for the fold opening-digit trace term.
//!
//! Stage-2 stores the committed witness as a Boolean table `w[col, ring]`:
//! column index `col` runs over `col_bits` variables and `ring` over `ring_bits`
//! ring coefficients. Tables are laid out as `idx = col · 2^{ring_bits} + ring`.

mod build;
mod eval;
mod evaluation_trace;
mod layout;
mod stage2;
mod trace_table;

#[cfg(test)]
mod stage2_compact;
#[cfg(test)]
mod tests;

pub use build::{
    build_trace_weight_table_field_live_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_live_block_weights, build_trace_weight_table_ring_terms,
};
pub use eval::{
    eval_trace_terms_closed, eval_trace_weight_at_point, TraceFieldBlockOpening,
    TraceOpeningAtPoint, TraceRingBlockOpening, TraceTerm,
};
pub use evaluation_trace::{
    ensure_trace_stage2_supported, prepare_evaluation_trace_group_parameters,
    EvaluationTraceGroupParameters, EvaluationTraceInputs,
};
pub use layout::TraceWeightLayout;
pub use stage2::{build_trace_table_scaled, TraceClaim, TracePublicWeights, TraceTermBatch};
pub use trace_table::{TraceSparseColumn, TraceTable};

#[cfg(test)]
pub(crate) use test_only::trace_weight_mle_eval;

#[cfg(test)]
mod test_only {
    use akita_algebra::poly::multilinear_eval;
    use akita_error::AkitaError;
    use jolt_field::Field;

    use super::layout::TraceWeightLayout;

    pub(crate) fn trace_weight_mle_eval<E: Field>(
        layout: &TraceWeightLayout,
        table: &[E],
        col_point: &[E],
        ring_point: &[E],
    ) -> Result<E, AkitaError> {
        let expected = layout.table_len()?;
        if table.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: table.len(),
            });
        }
        let point: Vec<E> = ring_point.iter().chain(col_point.iter()).copied().collect();
        multilinear_eval(table, &point)
    }
}
