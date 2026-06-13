//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible, MulBase,
};

use super::build::{
    build_trace_weight_compact_field_sparse_scaled, build_trace_weight_compact_ring_terms_scaled,
};
use super::trace_table::TraceTable;
use crate::{
    embed_ring_subfield_scalar, BasisMode, ClaimIncidenceSummary, LevelParams,
    PreparedOpeningPoint, RingRelationSegmentLayout, RingSubfieldEncoding, TraceFieldBlockOpening,
    TraceRingBlockOpening, TraceTerm, TraceWeightLayout,
};

/// Owned public trace-weight factors used by the fused stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TracePublicWeights<F: FieldCore, E: FieldCore, const D: usize> {
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

/// Verifier-side trace claim inputs for the stage-2 sumcheck final check.
///
/// The verifier reconstructs the fused trace term in its short closed form
/// ([`TraceTerm`]): one term per claim opening carrying the block-axis opening
/// `b_open`, the ψ-packed inner point, and a public coefficient. This is the
/// succinct counterpart of the prover's materialized [`TracePublicWeights`]
/// table; the two are kept distinct because the prover folds every block while
/// the verifier collapses each claim to a single `Tr_H`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceClaim<F: FieldCore, E: FieldCore, const D: usize> {
    pub layout: TraceWeightLayout,
    pub trace_terms: Vec<TraceTerm<F, E, D>>,
    /// Batching weight applied to the fused trace term. This is the `γ²` power
    /// of the stage-2 batching challenge (`CHALLENGE_SUMCHECK_BATCH`); the trace
    /// term reuses that challenge rather than sampling a dedicated one, so it is
    /// sampled after the next-level witness is bound to the transcript.
    pub trace_coeff: E,
    pub trace_opening_claim: E,
}

/// Whether the trace-weight dispatcher has an algebraic implementation for this
/// claim-field extension degree.
#[inline]
fn trace_stage2_supported(extension_degree: usize) -> bool {
    matches!(extension_degree, 1 | 2 | 4 | 8)
}

/// Reject extension degrees with no trace-weight implementation.
///
/// The fused trace term is mandatory: it is what binds the fold opening to the
/// committed witness in place of the dropped on-wire `y_ring`. A degree with no
/// algebraic implementation must therefore be rejected rather than silently
/// skipped, which would leave the opening unbound (a soundness footgun). This
/// is verifier-reachable, so it returns an error instead of panicking.
#[inline]
pub fn ensure_trace_stage2_supported(extension_degree: usize) -> Result<(), AkitaError> {
    if trace_stage2_supported(extension_degree) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "fused stage-2 trace term has no implementation for claim-field extension degree {extension_degree}; cannot bind the fold opening"
        )))
    }
}

/// Derive the trace-weight layout for the `e_hat` digit segment.
///
/// `num_trace_blocks` is the logical number of folded opening blocks addressed
/// by the trace term. Recursive singleton folds use `lp.num_blocks`; batched
/// root folds can use a wider claim-weighted block row.
pub fn trace_weight_layout_from_segment(
    lp: &LevelParams,
    segment: &RingRelationSegmentLayout,
    col_bits: usize,
    ring_bits: usize,
    num_trace_blocks: usize,
) -> Result<TraceWeightLayout, AkitaError> {
    if num_trace_blocks == 0 {
        return Err(AkitaError::InvalidInput(
            "trace-weight block count must be non-zero".to_string(),
        ));
    }
    if ring_bits != lp.ring_dimension.trailing_zeros() as usize {
        return Err(AkitaError::InvalidInput(
            "trace-weight ring bits do not match level ring dimension".to_string(),
        ));
    }
    let r_vars = num_trace_blocks.next_power_of_two().trailing_zeros() as usize;
    let layout = TraceWeightLayout {
        ring_bits,
        col_bits,
        opening_digit_offset: segment.offset_e,
        num_blocks: num_trace_blocks,
        num_digits_open: lp.num_digits_open,
        r_vars,
        log_basis: lp.log_basis,
    };
    layout.validate_opening_digit_segment()?;
    Ok(layout)
}

/// Build degree-one public trace weights from explicit block-offset terms.
pub fn trace_public_weights_field_terms<F, E, const D: usize>(
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
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
pub fn trace_public_weights_ring_terms<F, E, const D: usize>(
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
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

fn scaled_base_weights<F, E>(weights: &[F], scale: E) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
    E: RingSubfieldEncoding<F> + FieldCore,
{
    let scale = scale.degree_one_base().ok_or_else(|| {
        AkitaError::InvalidInput("trace field scale had no base coordinate".to_string())
    })?;
    Ok(weights.iter().map(|&weight| scale * weight).collect())
}

fn scaled_ring_weights<F, E, const D: usize>(
    weights: &[CyclotomicRing<F, D>],
    scale: E,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FieldCore,
{
    let scale = embed_ring_subfield_scalar::<F, E, D>(
        scale,
        AkitaError::InvalidInput("trace ring scale had no ring-subfield encoding".to_string()),
    )?;
    Ok(weights.iter().map(|&weight| weight * scale).collect())
}

/// Build public trace weights for a root incidence, optionally scaling each
/// claim term by an extra public factor such as the EOR final tensor factor.
pub fn trace_public_weights_root_terms<F, E, const D: usize>(
    lp: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    prepared_points: &[PreparedOpeningPoint<F, E, D>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if row_coefficients.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if let Some(scales) = claim_scales {
        if scales.len() != incidence.num_claims() {
            return Err(AkitaError::InvalidSize {
                expected: incidence.num_claims(),
                actual: scales.len(),
            });
        }
    }

    if E::EXT_DEGREE == 1 {
        let mut terms = Vec::with_capacity(incidence.num_claims());
        for (claim_idx, &coefficient) in row_coefficients.iter().enumerate() {
            let scale = claim_scales
                .and_then(|scales| scales.get(claim_idx).copied())
                .unwrap_or_else(E::one);
            let coefficient = coefficient * scale;
            let point_idx = *incidence
                .claim_to_point()
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let prepared = prepared_points
                .get(point_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let block_offset = claim_idx.checked_mul(lp.num_blocks).ok_or_else(|| {
                AkitaError::InvalidSetup("trace block offset overflow".to_string())
            })?;
            terms.push(TraceFieldBlockOpening {
                block_offset,
                block_weights: scaled_base_weights(&prepared.ring_opening_point.b, coefficient)?,
                inner_opening_ring: prepared.packed_inner_point,
            });
        }
        trace_public_weights_field_terms(&terms)
    } else {
        let mut terms = Vec::with_capacity(incidence.num_claims());
        for (claim_idx, &coefficient) in row_coefficients.iter().enumerate() {
            let scale = claim_scales
                .and_then(|scales| scales.get(claim_idx).copied())
                .unwrap_or_else(E::one);
            let coefficient = coefficient * scale;
            let point_idx = *incidence
                .claim_to_point()
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let prepared = prepared_points
                .get(point_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let block_rings = prepared.ring_multiplier_point.b_rings().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension trace opening point is missing ring block weights".to_string(),
                )
            })?;
            let block_offset = claim_idx.checked_mul(lp.num_blocks).ok_or_else(|| {
                AkitaError::InvalidSetup("trace block offset overflow".to_string())
            })?;
            terms.push(TraceRingBlockOpening {
                block_offset,
                block_rings: scaled_ring_weights(block_rings, coefficient)?,
                packed_inner_point: prepared.packed_inner_point,
            });
        }
        trace_public_weights_ring_terms(&terms)
    }
}

/// Build public trace weights for a recursive singleton fold, optionally
/// scaling it by an EOR final tensor factor.
pub fn trace_public_weights_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E, D>,
    scale: E,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if E::EXT_DEGREE == 1 {
        trace_public_weights_field_terms(&[TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: scaled_base_weights(&prepared.ring_opening_point.b, scale)?,
            inner_opening_ring: prepared.packed_inner_point,
        }])
    } else {
        let block_rings = prepared.ring_multiplier_point.b_rings().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension trace opening point is missing ring block weights".to_string(),
            )
        })?;
        trace_public_weights_ring_terms(&[TraceRingBlockOpening {
            block_offset: 0,
            block_rings: scaled_ring_weights(block_rings, scale)?,
            packed_inner_point: prepared.packed_inner_point,
        }])
    }
}

/// Slice the block-axis opening `b_open` out of a root opening point.
///
/// The root uses [`crate::BlockOrder::RowMajor`]: after padding `opening_point`
/// to `m_vars + r_vars + alpha_bits` coordinates, the outer coordinates are
/// `padded[alpha_bits..]` and the block coordinates are their tail
/// `outer[m_vars..m_vars + r_vars]`. This reproduces the `b_open` that
/// `prepare_opening_point` consumes (and then discards) when it
/// materializes the ring block multipliers.
pub fn root_trace_block_opening<X: FieldCore>(
    opening_point: &[X],
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<Vec<X>, AkitaError> {
    let target = lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target {
        return Err(AkitaError::InvalidPointDimension {
            expected: target,
            actual: opening_point.len(),
        });
    }
    let mut padded = opening_point.to_vec();
    padded.resize(target, X::zero());
    let start = alpha_bits + lp.m_vars;
    Ok(padded[start..start + lp.r_vars].to_vec())
}

/// Build the verifier's short closed-form trace terms for a root incidence.
///
/// `b_opens_per_point[point_idx]` must hold the block-axis opening (in the
/// evaluation field `E`) for that opening point; the verifier obtains it via
/// [`root_trace_block_opening`] (lifting claim-field coordinates into `E` as
/// needed). Mirrors [`trace_public_weights_root_terms`] but emits the succinct
/// per-claim terms instead of materialized block weights.
#[allow(clippy::too_many_arguments)]
pub fn trace_terms_root<F, E, const D: usize>(
    lp: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    prepared_points: &[PreparedOpeningPoint<F, E, D>],
    b_opens_per_point: &[Vec<E>],
    basis: BasisMode,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<Vec<TraceTerm<F, E, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if row_coefficients.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if b_opens_per_point.len() != incidence.num_points() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_points(),
            actual: b_opens_per_point.len(),
        });
    }
    if let Some(scales) = claim_scales {
        if scales.len() != incidence.num_claims() {
            return Err(AkitaError::InvalidSize {
                expected: incidence.num_claims(),
                actual: scales.len(),
            });
        }
    }

    let mut terms = Vec::with_capacity(incidence.num_claims());
    for (claim_idx, &coefficient) in row_coefficients.iter().enumerate() {
        let scale = claim_scales
            .and_then(|scales| scales.get(claim_idx).copied())
            .unwrap_or_else(E::one);
        let point_idx = *incidence
            .claim_to_point()
            .get(claim_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let prepared = prepared_points
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let b_open = b_opens_per_point
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?
            .clone();
        let block_offset = claim_idx
            .checked_mul(lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block offset overflow".to_string()))?;
        terms.push(TraceTerm {
            block_offset,
            b_open,
            basis,
            packed_inner_point: prepared.packed_inner_point,
            coefficient: coefficient * scale,
        });
    }
    Ok(terms)
}

/// Build the verifier's short closed-form trace term for a recursive singleton
/// fold. The block-axis opening is sliced from the retained padded point under
/// the [`crate::BlockOrder::ColumnMajor`] convention.
pub fn trace_terms_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E, D>,
    lp: &LevelParams,
    basis: BasisMode,
    scale: E,
) -> Result<Vec<TraceTerm<F, E, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let outer_len = lp
        .m_vars
        .checked_add(lp.r_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let alpha_bits = prepared
        .padded_point
        .len()
        .checked_sub(outer_len)
        .ok_or(AkitaError::InvalidProof)?;
    // Column-major: the block coordinates are the head of the outer point.
    let b_open = prepared.padded_point[alpha_bits..alpha_bits + lp.r_vars].to_vec();
    Ok(vec![TraceTerm {
        block_offset: 0,
        b_open,
        basis,
        packed_inner_point: prepared.packed_inner_point,
        coefficient: scale,
    }])
}

/// Materialize the trace-weight table and keep only live witness columns.
pub fn trace_weight_evals_for_witness<E: FieldCore>(
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

/// Build the prover-side compact trace table and scale each live entry.
pub fn build_trace_stage2_compact_scaled<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    public_weights: &TracePublicWeights<F, E, D>,
    live_x_cols: usize,
    output_scale: E,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    Ok(
        build_trace_table_scaled(layout, public_weights, live_x_cols, output_scale)?
            .materialize_dense(live_x_cols, layout.ring_len()),
    )
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
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
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

/// Evaluate the fused trace term at the verifier's final stage-2 point.
///
/// Uses the short closed form: one `Tr_H` per claim term, with no dependence on
/// Sum batched public opening claims under per-claim row coefficients.
pub fn batched_eval_target_from_incidence<E, L>(
    incidence: &ClaimIncidenceSummary,
    row_coefficients: &[L],
    openings: &[E],
) -> Result<L, AkitaError>
where
    E: FieldCore,
    L: ExtField<E> + MulBase<E> + FieldCore,
{
    if row_coefficients.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: row_coefficients.len(),
        });
    }
    if openings.len() != incidence.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: incidence.num_claims(),
            actual: openings.len(),
        });
    }
    incidence
        .public_rows()
        .iter()
        .flat_map(|row| row.claim_indices())
        .try_fold(L::zero(), |acc, &claim_idx| {
            let coefficient = *row_coefficients
                .get(claim_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let opening = *openings.get(claim_idx).ok_or(AkitaError::InvalidProof)?;
            Ok(acc + coefficient.mul_base(opening))
        })
}

#[cfg(test)]
mod tests {
    use super::super::build::{
        build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
        build_trace_weight_table_ring_terms,
    };
    use super::*;
    use crate::{
        eval_trace_terms_closed, lagrange_weights, reduce_inner_opening_to_ring_element, BasisMode,
    };
    use akita_field::{Ext2, Prime128OffsetA7F7};

    type F = Prime128OffsetA7F7;
    const D: usize = 8;

    fn layout() -> TraceWeightLayout {
        TraceWeightLayout {
            ring_bits: 3,
            col_bits: 3,
            opening_digit_offset: 2,
            num_blocks: 2,
            num_digits_open: 2,
            r_vars: 1,
            log_basis: 3,
        }
    }

    #[test]
    fn compact_trace_table_keeps_live_columns_in_witness_order() {
        let layout = layout();
        let table = (0..layout.table_len().unwrap())
            .map(|idx| F::from_u64(idx as u64))
            .collect::<Vec<_>>();
        let compact = trace_weight_evals_for_witness(&layout, &table, 3).unwrap();

        let mut expected = Vec::new();
        for col in 0..3 {
            for ring in 0..layout.ring_len() {
                expected.push(table[layout.witness_index(col, ring)]);
            }
        }
        assert_eq!(compact, expected);
    }

    fn test_ring(seed: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            F::from_u64(seed + 3 * i as u64 + 1)
        }))
    }

    #[test]
    fn trace_table_field_sparse_matches_materialized_dense() {
        let layout = layout();
        let terms = vec![
            TraceFieldBlockOpening {
                block_offset: 0,
                block_weights: vec![F::from_u64(2), F::from_u64(5)],
                inner_opening_ring: test_ring(10),
            },
            TraceFieldBlockOpening {
                block_offset: 1,
                block_weights: vec![F::from_u64(7)],
                inner_opening_ring: test_ring(40),
            },
        ];
        let public_weights = trace_public_weights_field_terms::<F, F, D>(&terms).unwrap();
        let live_x_cols = 5;
        let dense =
            build_trace_stage2_compact_scaled(&layout, &public_weights, live_x_cols, F::one())
                .unwrap();
        let sparse = build_trace_table_scaled(&layout, &public_weights, live_x_cols, F::one())
            .unwrap()
            .materialize_dense(live_x_cols, layout.ring_len());
        assert_eq!(sparse, dense);
    }

    #[test]
    fn stage2_compact_field_matches_dense_slice_for_partial_live_columns() {
        let layout = layout();
        let terms = vec![
            TraceFieldBlockOpening {
                block_offset: 0,
                block_weights: vec![F::from_u64(2), F::from_u64(5)],
                inner_opening_ring: test_ring(10),
            },
            TraceFieldBlockOpening {
                block_offset: 1,
                block_weights: vec![F::from_u64(7)],
                inner_opening_ring: test_ring(40),
            },
        ];
        let public_weights = trace_public_weights_field_terms::<F, F, D>(&terms).unwrap();
        let dense = build_trace_weight_table_field_terms::<F, F, D>(&layout, &terms).unwrap();
        let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
        let actual =
            build_trace_stage2_compact_scaled(&layout, &public_weights, 5, F::one()).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn stage2_compact_ring_matches_dense_slice_for_partial_live_columns() {
        type E = Ext2<F>;

        let layout = layout();
        let terms = vec![
            TraceRingBlockOpening {
                block_offset: 0,
                block_rings: vec![test_ring(3), test_ring(17)],
                packed_inner_point: test_ring(31),
            },
            TraceRingBlockOpening {
                block_offset: 1,
                block_rings: vec![test_ring(53)],
                packed_inner_point: test_ring(71),
            },
        ];
        let public_weights = trace_public_weights_ring_terms::<F, E, D>(&terms).unwrap();
        let dense = build_trace_weight_table_ring_terms::<F, E, D>(&layout, &terms).unwrap();
        let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
        let actual =
            build_trace_stage2_compact_scaled(&layout, &public_weights, 5, E::one()).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn stage2_compact_scaled_matches_scaled_dense_slice() {
        type E = Ext2<F>;

        let layout = layout();
        let terms = vec![
            TraceRingBlockOpening {
                block_offset: 0,
                block_rings: vec![test_ring(3), test_ring(17)],
                packed_inner_point: test_ring(31),
            },
            TraceRingBlockOpening {
                block_offset: 1,
                block_rings: vec![test_ring(53)],
                packed_inner_point: test_ring(71),
            },
        ];
        let public_weights = trace_public_weights_ring_terms::<F, E, D>(&terms).unwrap();
        let output_scale = E::new(F::from_u64(11), F::from_u64(19));
        let dense = build_trace_weight_table_ring_terms::<F, E, D>(&layout, &terms).unwrap();
        let expected = trace_weight_evals_for_witness(&layout, &dense, 5)
            .unwrap()
            .into_iter()
            .map(|value| output_scale * value)
            .collect::<Vec<_>>();
        let actual =
            build_trace_stage2_compact_scaled(&layout, &public_weights, 5, output_scale).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn trace_claim_eval_matches_dense_table_for_k1() {
        let layout = layout();
        let inner_open = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let b_open = vec![F::from_u64(11)];
        let block_weights = lagrange_weights(&b_open).unwrap();
        let inner_ring =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open, BasisMode::Lagrange).unwrap();
        let claim = TraceClaim {
            layout,
            trace_terms: vec![TraceTerm {
                block_offset: 0,
                b_open: b_open.clone(),
                basis: BasisMode::Lagrange,
                packed_inner_point: inner_ring,
                coefficient: F::one(),
            }],
            trace_coeff: F::from_u64(13),
            trace_opening_claim: F::from_u64(17),
        };

        let table = build_trace_weight_table_field_block_weights::<F, F, D>(
            &layout,
            &block_weights,
            &inner_ring,
        )
        .unwrap();
        let ring_point = vec![F::from_u64(19), F::from_u64(23), F::from_u64(29)];
        let col_point = vec![F::from_u64(31), F::from_u64(37), F::from_u64(41)];
        let dense =
            crate::trace_weight::trace_weight_mle_eval(&layout, &table, &col_point, &ring_point)
                .unwrap();
        let closed = eval_trace_terms_closed::<F, F, D>(
            &claim.layout,
            &ring_point,
            &col_point,
            &claim.trace_terms,
        )
        .unwrap();

        assert_eq!(closed, dense);
    }
}
