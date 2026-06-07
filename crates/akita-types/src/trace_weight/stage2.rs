//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible, MulBase,
};

use super::build::{build_trace_weight_compact_field_terms, build_trace_weight_compact_ring_terms};
use crate::{
    block_rings_at_opening, embed_ring_subfield_scalar, eval_trace_weight_at_point,
    lagrange_weights, ClaimIncidenceSummary, LevelParams, PreparedRecursiveOpeningPoint,
    PreparedRootOpeningPoint, RingRelationSegmentLayout, RingSubfieldEncoding,
    TraceFieldBlockOpening, TraceOpeningAtPoint, TraceRingBlockOpening, TraceWeightLayout,
};

/// Owned public trace-opening data used by the fused stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceStage2OpeningOwned<F: FieldCore, E: FieldCore, const D: usize> {
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

impl<F: FieldCore, E: FieldCore, const D: usize> TraceStage2OpeningOwned<F, E, D> {
    fn as_trace_opening(&self) -> TraceOpeningAtPoint<'_, F, E, D> {
        match self {
            Self::Field { terms } => TraceOpeningAtPoint::Field { terms },
            Self::Ring { terms, .. } => TraceOpeningAtPoint::Ring {
                terms,
                _ext: PhantomData,
            },
        }
    }
}

/// Public trace payload consumed by a stage-2 verifier final-point check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStage2Wire<F: FieldCore, E: FieldCore, const D: usize> {
    pub layout: TraceWeightLayout,
    pub opening: TraceStage2OpeningOwned<F, E, D>,
    pub gamma_tr: E,
    pub trace_opening_claim: E,
}

/// Scalar contribution added to the stage-2 input claim.
#[inline]
pub fn trace_input_claim<E: FieldCore>(gamma_tr: E, opening: E) -> E {
    gamma_tr * opening
}

/// Whether the trace-weight dispatcher has an algebraic implementation for this
/// claim-field extension degree.
#[inline]
pub fn trace_stage2_supported(extension_degree: usize) -> bool {
    matches!(extension_degree, 1 | 2 | 4 | 8)
}

/// True when the fused trace term can be enabled for this build.
#[inline]
pub fn trace_stage2_enabled(
    lp: &LevelParams,
    extension_degree: usize,
    has_extension_opening_reduction: bool,
) -> bool {
    let _ = (lp, has_extension_opening_reduction);
    trace_stage2_supported(extension_degree)
}

/// Lagrange block weights for the degree-one closed-form trace evaluator.
pub fn trace_block_weights_k1<F: FieldCore>(block_open: &[F]) -> Result<Vec<F>, AkitaError> {
    lagrange_weights(block_open)
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

/// Build a degree-one owned trace opening from explicit factors.
pub fn trace_stage2_opening_owned_k1<F, E, const D: usize>(
    block_weights: &[F],
    inner_opening_ring: &CyclotomicRing<F, D>,
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    trace_stage2_opening_owned_field_terms(&[TraceFieldBlockOpening {
        block_offset: 0,
        block_weights: block_weights.to_vec(),
        inner_opening_ring: *inner_opening_ring,
    }])
}

/// Build a degree-one owned trace opening from explicit block-offset terms.
pub fn trace_stage2_opening_owned_field_terms<F, E, const D: usize>(
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TraceStage2OpeningOwned::Field {
        terms: terms.to_vec(),
    })
}

/// Build an extension-valued owned trace opening from explicit factors.
pub fn trace_stage2_opening_owned_ring<F, E, const D: usize>(
    packed_inner_point: &CyclotomicRing<F, D>,
    block_open: &[E],
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + FieldCore,
{
    trace_stage2_opening_owned_ring_terms(&[TraceRingBlockOpening {
        block_offset: 0,
        block_rings: block_rings_at_opening::<F, E, D>(block_open)?,
        packed_inner_point: *packed_inner_point,
    }])
}

/// Build an extension-valued owned trace opening from explicit block-offset terms.
pub fn trace_stage2_opening_owned_ring_terms<F, E, const D: usize>(
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TraceStage2OpeningOwned::Ring {
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

/// Build the owned trace opening for a root incidence, optionally scaling each
/// claim term by an extra public factor such as the EOR final tensor factor.
pub fn trace_stage2_opening_owned_root_terms<F, E, const D: usize>(
    lp: &LevelParams,
    incidence: &ClaimIncidenceSummary,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
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
        trace_stage2_opening_owned_field_terms(&terms)
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
        trace_stage2_opening_owned_ring_terms(&terms)
    }
}

/// Build the owned trace opening for a recursive singleton fold, optionally
/// scaling it by an EOR final tensor factor.
pub fn trace_stage2_opening_owned_recursive<F, E, const D: usize>(
    prepared: &PreparedRecursiveOpeningPoint<F, E, D>,
    scale: E,
) -> Result<TraceStage2OpeningOwned<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if E::EXT_DEGREE == 1 {
        trace_stage2_opening_owned_field_terms(&[TraceFieldBlockOpening {
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
        trace_stage2_opening_owned_ring_terms(&[TraceRingBlockOpening {
            block_offset: 0,
            block_rings: scaled_ring_weights(block_rings, scale)?,
            packed_inner_point: prepared.packed_inner_point,
        }])
    }
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

/// Build the prover-side compact trace table for a stage-2 witness.
pub fn build_trace_stage2_compact<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    opening: &TraceStage2OpeningOwned<F, E, D>,
    live_x_cols: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    match opening {
        TraceStage2OpeningOwned::Field { terms } => {
            build_trace_weight_compact_field_terms::<F, E, D>(layout, terms, live_x_cols)
        }
        TraceStage2OpeningOwned::Ring { terms, .. } => {
            build_trace_weight_compact_ring_terms::<F, E, D>(layout, terms, live_x_cols)
        }
    }
}

/// Evaluate the trace-weight table at the verifier's final stage-2 point.
pub fn eval_trace_stage2_wire_for_degree<F, E, const D: usize>(
    wire: &TraceStage2Wire<F, E, D>,
    ring_point: &[E],
    col_point: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    macro_rules! eval {
        ($k:expr) => {
            eval_trace_weight_at_point::<F, E, D, $k>(
                &wire.layout,
                ring_point,
                col_point,
                wire.opening.as_trace_opening(),
            )
        };
    }

    match E::EXT_DEGREE {
        1 => eval!(1),
        2 => eval!(2),
        4 => eval!(4),
        8 => eval!(8),
        _ => Err(AkitaError::InvalidInput(
            "unsupported trace-stage2 extension degree".to_string(),
        )),
    }
}

/// Sum batched public opening claims under per-claim row coefficients.
pub fn trace_opening_from_incidence<E, L>(
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
    use super::*;
    use crate::{
        build_trace_weight_table_field_block_weights, build_trace_weight_table_field_terms,
        build_trace_weight_table_ring_terms, reduce_inner_opening_to_ring_element, BasisMode,
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
        let opening = trace_stage2_opening_owned_field_terms::<F, F, D>(&terms).unwrap();
        let dense = build_trace_weight_table_field_terms::<F, F, D>(&layout, &terms).unwrap();
        let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
        let actual = build_trace_stage2_compact(&layout, &opening, 5).unwrap();

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
        let opening = trace_stage2_opening_owned_ring_terms::<F, E, D>(&terms).unwrap();
        let dense = build_trace_weight_table_ring_terms::<F, E, D>(&layout, &terms).unwrap();
        let expected = trace_weight_evals_for_witness(&layout, &dense, 5).unwrap();
        let actual = build_trace_stage2_compact(&layout, &opening, 5).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn stage2_wire_eval_matches_dense_table_for_k1() {
        let layout = layout();
        let inner_open = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let b_open = vec![F::from_u64(11)];
        let block_weights = lagrange_weights(&b_open).unwrap();
        let inner_ring =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_open, BasisMode::Lagrange).unwrap();
        let opening =
            trace_stage2_opening_owned_k1::<F, F, D>(&block_weights, &inner_ring).unwrap();
        let wire = TraceStage2Wire {
            layout,
            opening,
            gamma_tr: F::from_u64(13),
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
        let closed = eval_trace_stage2_wire_for_degree(&wire, &ring_point, &col_point).unwrap();

        assert_eq!(closed, dense);
    }
}
