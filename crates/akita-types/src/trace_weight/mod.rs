//! Public multilinear weights for the fold opening-digit trace term.
//!
//! Stage-2 stores the committed witness as a Boolean table `w[col, ring]`:
//! column index `col` runs over `col_bits` variables and `ring` over `ring_bits`
//! ring coefficients. Tables are laid out as `idx = col · 2^{ring_bits} + ring`.

mod build;
mod eval;
mod layout;
mod stage2;
mod trace_table;

#[cfg(test)]
mod tests;

pub use build::{
    build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_block_weights, build_trace_weight_table_ring_terms,
};
pub use eval::{
    eval_trace_terms_closed, eval_trace_weight_at_point, TraceFieldBlockOpening,
    TraceOpeningAtPoint, TraceRingBlockOpening, TraceTerm,
};
pub use layout::TraceWeightLayout;
pub use stage2::{
    build_trace_claim_root, build_trace_table_scaled, ensure_trace_stage2_supported,
    root_trace_block_opening, trace_public_weights_recursive, trace_public_weights_root_terms,
    trace_terms_recursive, trace_terms_root, trace_weight_layout_from_segment, TraceClaim,
    TracePublicWeights,
};
pub use trace_table::{TraceSparseColumn, TraceTable};

#[cfg(test)]
pub(crate) use test_only::trace_weight_mle_eval;

#[cfg(test)]
mod test_only {
    use akita_algebra::poly::multilinear_eval;
    use akita_field::{AkitaError, FieldCore};

    use super::TraceWeightLayout;

    pub(crate) fn trace_weight_mle_eval<E: FieldCore>(
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
