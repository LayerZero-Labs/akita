use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use std::marker::PhantomData;

use crate::field_reduction::trace_open_ring_mle_dot;
use crate::{gadget_row_scalars, lagrange_weights, RingSubfieldEncoding};

use super::layout::TraceWeightLayout;

/// Opening weights consumed by [`eval_trace_weight_at_point`].
pub enum TraceOpeningAtPoint<'a, F: FieldCore, E: FieldCore, const D: usize> {
    /// `K = 1`: scalar block weights and the basis-correct packed inner opening.
    Field {
        block_weights: &'a [F],
        inner_opening_ring: &'a CyclotomicRing<F, D>,
    },
    /// `K > 1`: embedded block rings and ψ-packed inner point.
    Ring {
        block_rings: &'a [CyclotomicRing<F, D>],
        packed_inner_point: &'a CyclotomicRing<F, D>,
        _ext: PhantomData<E>,
    },
}

fn lift_gadget_row<F, E>(gadget_scalars: &[F]) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    gadget_scalars.iter().copied().map(E::lift_base).collect()
}

fn validate_eval_point(
    layout: &TraceWeightLayout,
    ring_point_len: usize,
    col_point_len: usize,
) -> Result<(), AkitaError> {
    if ring_point_len != layout.ring_bits || col_point_len != layout.col_bits {
        return Err(AkitaError::InvalidSize {
            expected: layout.col_bits + layout.ring_bits,
            actual: col_point_len + ring_point_len,
        });
    }
    layout.validate_opening_digit_segment()
}

#[inline]
fn eq_weight_at_index<E: FieldCore>(point: &[E], index: usize) -> E {
    let mut weight = E::one();
    for (bit, &coord) in point.iter().enumerate() {
        if ((index >> bit) & 1) == 1 {
            weight *= coord;
        } else {
            weight *= E::one() - coord;
        }
    }
    weight
}

/// Evaluate the trace-weight MLE at `(ring_point, col_point)`.
///
/// `K` must match the claim-field extension degree (`1` for base-field claims,
/// `2`/`4`/`8` for ring-subfield extension claims). The compiler monomorphizes
/// the `K = 1` tensor path separately from the extension trace path.
pub fn eval_trace_weight_at_point<F, E, const D: usize, const K: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    opening: TraceOpeningAtPoint<'_, F, E, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt + FieldCore,
{
    match opening {
        TraceOpeningAtPoint::Field {
            block_weights,
            inner_opening_ring,
        } => {
            if K != 1 {
                return Err(AkitaError::InvalidInput(
                    "field opening weights require K = 1".to_string(),
                ));
            }
            eval_at_point_k1::<F, E, D>(
                layout,
                ring_point,
                col_point,
                block_weights,
                inner_opening_ring,
            )
        }
        TraceOpeningAtPoint::Ring {
            block_rings,
            packed_inner_point,
            ..
        } => {
            if K == 1 {
                return Err(AkitaError::InvalidInput(
                    "ring opening weights require K > 1".to_string(),
                ));
            }
            eval_at_point_k_extension::<F, E, D>(
                layout,
                ring_point,
                col_point,
                block_rings,
                packed_inner_point,
            )
        }
    }
}

#[inline]
fn eval_at_point_k1<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    block_weights: &[F],
    inner_opening_ring: &CyclotomicRing<F, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if block_weights.len() != layout.num_blocks {
        return Err(AkitaError::InvalidInput(
            "field opening weights do not match layout".to_string(),
        ));
    }
    validate_eval_point(layout, ring_point.len(), col_point.len())?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars);
    let mut col_factor = E::zero();
    for (block, &block_weight) in block_weights.iter().enumerate() {
        for (plane, &gadget) in gadget_row.iter().enumerate() {
            let col = layout.opening_digit_col_index(block, plane);
            col_factor += eq_weight_at_index(col_point, col) * E::lift_base(block_weight) * gadget;
        }
    }
    let ring_eq = lagrange_weights(ring_point)?;
    let inner_factor = inner_opening_ring
        .coefficients()
        .iter()
        .zip(ring_eq.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + E::lift_base(coeff) * weight
        });
    Ok(col_factor * inner_factor)
}

fn eval_at_point_k_extension<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    ring_point: &[E],
    col_point: &[E],
    block_rings: &[CyclotomicRing<F, D>],
    packed_inner_point: &CyclotomicRing<F, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if block_rings.len() != layout.num_blocks {
        return Err(AkitaError::InvalidInput(
            "ring opening weights do not match layout".to_string(),
        ));
    }
    validate_eval_point(layout, ring_point.len(), col_point.len())?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars);

    let ring_eq = lagrange_weights(ring_point)?;
    let mut out = E::zero();
    for (block, block_ring) in block_rings.iter().enumerate().take(layout.num_blocks) {
        let block_inner = trace_open_ring_mle_dot::<F, E, D>(
            block_ring,
            &ring_eq,
            packed_inner_point,
            layout.ring_bits,
        )?;
        for (plane, &gadget) in gadget_row.iter().enumerate() {
            let col = layout.opening_digit_col_index(block, plane);
            out += eq_weight_at_index(col_point, col) * gadget * block_inner;
        }
    }
    Ok(out)
}
