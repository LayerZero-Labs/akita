//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

use super::build::{
    build_trace_weight_compact_field_sparse_scaled, build_trace_weight_compact_ring_terms_scaled,
};
use super::trace_table::TraceTable;
use super::{TraceFieldBlockOpening, TraceRingBlockOpening, TraceTerm, TraceWeightLayout};
use crate::FpExtEncoding;

/// Owned public trace-weight factors used by the fused stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TracePublicWeights<F: Field, E: Field, const D: usize> {
    /// Degree-one path: scalar block-weight terms with their packed inner openings.
    Field {
        terms: Vec<TraceFieldBlockOpening<F, D>>,
    },
    /// Extension path: ring block weights and psi-packed inner point.
    Ring {
        terms: Vec<TraceRingBlockOpening<F, D>>,
        _ext: PhantomData<E>,
    },
}

/// One closed-form trace batch evaluated with its own column geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceTermBatch<F: Field, E: Field, const D: usize> {
    pub layout: TraceWeightLayout,
    pub terms: Vec<TraceTerm<F, E, D>>,
}

/// Verifier-side trace claim inputs for the stage-2 sumcheck final check.
///
/// The verifier reconstructs the fused trace term in its short closed form
/// ([`TraceTerm`]): one term per claim opening carrying the block-axis opening
/// `b_open`, the ψ-packed inner point, and a public coefficient. This is the
/// succinct counterpart of the prover's materialized [`TracePublicWeights`]
/// table; the two are kept distinct because the prover folds every block while
/// the verifier collapses each claim to a single `Tr_H`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceClaim<F: Field, E: Field, const D: usize> {
    pub layout: TraceWeightLayout,
    pub trace_terms: Vec<TraceTerm<F, E, D>>,
    /// Batching weight applied to the fused trace term. This is the `γ²` power
    /// of the stage-2 batching challenge (`CHALLENGE_SUMCHECK_BATCH`); the trace
    /// term reuses that challenge rather than sampling a dedicated one, so it is
    /// sampled after the next-level witness is bound to the transcript.
    pub trace_coeff: E,
    pub trace_opening_claim: E,
    /// Dense multi-group-root trace-weight table (`col ⊗ ring`, `output_scale = 1`).
    /// When present the stage-2 verifier evaluates its multilinear extension at
    /// the witness point instead of the closed-form [`Self::trace_terms`]. This
    /// is the multi-group-root counterpart of the succinct per-claim terms: multi-group
    /// roots decompose each group with per-group `num_live_blocks` and
    /// `num_digits_open` over chunk-major physical E ranges, which the
    /// single-layout closed form cannot express.
    pub dense_evals: Option<Vec<E>>,
    /// Optional closed-form batches with independent layouts.
    pub trace_term_batches: Vec<TraceTermBatch<F, E, D>>,
}

/// Build degree-one public trace weights from explicit block-offset terms.
#[cfg(test)]
pub(crate) fn trace_public_weights_field_terms<F, E, const D: usize>(
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: Field,
    E: Field,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TracePublicWeights::Field {
        terms: terms.to_vec(),
    })
}

/// Build extension-valued public trace weights from explicit block-offset terms.
#[cfg(test)]
pub(crate) fn trace_public_weights_ring_terms<F, E, const D: usize>(
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: Field,
    E: Field,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TracePublicWeights::Ring {
        terms: terms.to_vec(),
        _ext: PhantomData,
    })
}

/// Materialize the trace-weight table and keep only live witness columns.
#[cfg(test)]
pub(crate) fn trace_weight_evals_for_witness<E: Field>(
    layout: &TraceWeightLayout,
    table: &[E],
    live_x_cols: usize,
) -> Result<Vec<E>, AkitaError> {
    let x_len = 1usize
        .checked_shl(layout.col_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("trace-weight x length overflow".to_string()))?;
    if live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: x_len,
            actual: live_x_cols,
        });
    }
    let expected = layout.table_len()?;
    if table.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: table.len(),
        });
    }

    let ring_len = layout.ring_len();
    let out_len = live_x_cols.checked_mul(ring_len).ok_or_else(|| {
        AkitaError::InvalidInput("trace-weight compact table length overflow".to_string())
    })?;
    let mut out = Vec::with_capacity(out_len);
    for col in 0..live_x_cols {
        for ring_coord in 0..ring_len {
            out.push(table[layout.witness_index(col, ring_coord)]);
        }
    }
    Ok(out)
}

/// Build the typed trace table used by the stage-2 prover.
///
/// `K = 1` field weights use sparse active columns; `K > 1` ring weights use a dense flat table.
pub fn build_trace_table_scaled<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    public_weights: &TracePublicWeights<F, E, D>,
    live_x_cols: usize,
    output_scale: E,
) -> Result<TraceTable<E>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring,
{
    match public_weights {
        TracePublicWeights::Field { terms } => {
            let ring_len = layout.ring_len();
            let columns = build_trace_weight_compact_field_sparse_scaled::<F, E, D>(
                layout,
                terms,
                live_x_cols,
                output_scale,
            )?;
            Ok(TraceTable::field_sparse(columns, live_x_cols, ring_len))
        }
        TracePublicWeights::Ring { terms, .. } => Ok(TraceTable::ring_dense(
            build_trace_weight_compact_ring_terms_scaled::<F, E, D>(
                layout,
                terms,
                live_x_cols,
                output_scale,
            )?,
        )),
    }
}
