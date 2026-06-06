//! Public multilinear weights for the fold opening-digit trace term.
//!
//! Stage-2 stores the committed witness as a Boolean table `w[col, ring]`:
//! column index `col` runs over `col_bits` variables and `ring` over `ring_bits`
//! ring coefficients. Tables are laid out as `idx = col · 2^{ring_bits} + ring`.

mod build;
mod eval;
mod layout;
mod stage2;

#[cfg(test)]
mod tests;

pub use build::{
    build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
    build_trace_weight_table_ring_block_weights, build_trace_weight_table_ring_terms,
};
pub use eval::{
    eval_trace_weight_at_point, TraceFieldBlockOpening, TraceOpeningAtPoint, TraceRingBlockOpening,
};
pub use layout::TraceWeightLayout;
pub use stage2::{
    build_trace_stage2_compact, eval_trace_stage2_wire_for_degree, trace_block_weights_k1,
    trace_input_claim, trace_opening_from_incidence, trace_stage2_enabled,
    trace_stage2_opening_owned_field_terms, trace_stage2_opening_owned_k1,
    trace_stage2_opening_owned_ring, trace_stage2_opening_owned_ring_terms, trace_stage2_supported,
    trace_weight_evals_for_witness, trace_weight_layout_from_segment, TraceStage2OpeningOwned,
    TraceStage2Wire,
};

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
