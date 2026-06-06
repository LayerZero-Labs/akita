use akita_algebra::offset_eq::eval_offset_eq_tensor;
use akita_algebra::CyclotomicRing;
use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use std::marker::PhantomData;

use crate::field_reduction::trace_open_ring_mle_dot;
use crate::{gadget_row_scalars, lagrange_weights, RingSubfieldEncoding};

use super::layout::TraceWeightLayout;

/// Opening weights consumed by [`eval_trace_weight_at_point`].
pub enum TraceOpeningAtPoint<'a, F: FieldCore, E: FieldCore, const D: usize> {
    /// `K = 1`: scalar block weights and inner opening coordinates in the base field.
    Field {
        block_weights: &'a [F],
        inner_open: &'a [F],
    },
    /// `K > 1`: embedded block rings and ψ-packed inner point.
    Ring {
        block_rings: &'a [CyclotomicRing<F, D>],
        packed_inner_point: &'a CyclotomicRing<F, D>,
        _ext: PhantomData<E>,
    },
}

const MAX_COORDS: usize = 32;
const MAX_FACTOR_WIDTH: usize = 32;

#[inline]
fn eq_mle_base_point<F, E>(point: &[E], base_coords: &[F]) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + FieldCore,
{
    if base_coords.len() > MAX_COORDS {
        return Err(AkitaError::InvalidInput(
            "opening coordinate count exceeds stack bound".to_string(),
        ));
    }
    let mut lifted = [E::zero(); MAX_COORDS];
    for (dst, &src) in lifted[..base_coords.len()]
        .iter_mut()
        .zip(base_coords.iter())
    {
        *dst = E::lift_base(src);
    }
    EqPolynomial::mle(point, &lifted[..base_coords.len()])
}

fn lift_gadget_row<F, E>(gadget_scalars: &[F]) -> Result<[E; MAX_FACTOR_WIDTH], AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if gadget_scalars.len() > MAX_FACTOR_WIDTH {
        return Err(AkitaError::InvalidInput(
            "trace-weight gadget width exceeds stack bound".to_string(),
        ));
    }
    let mut out = [E::zero(); MAX_FACTOR_WIDTH];
    for (dst, &src) in out[..gadget_scalars.len()]
        .iter_mut()
        .zip(gadget_scalars.iter())
    {
        *dst = E::lift_base(src);
    }
    Ok(out)
}

/// Column factor `eq_seg · gadget` for the opening-digit segment (`K > 1`).
///
/// Block Lagrange weights live in the inner sum; a neutral all-ones block row keeps
/// [`eval_offset_eq_tensor`] aligned on `r_vars` without contributing `eq_block`.
fn opening_digit_gadget_factor<E: FieldCore>(
    layout: &TraceWeightLayout,
    col_point: &[E],
    gadget_row: &[E],
) -> Result<E, AkitaError> {
    if layout.num_blocks > MAX_FACTOR_WIDTH {
        return Err(AkitaError::InvalidInput(
            "block count exceeds stack bound".to_string(),
        ));
    }
    let neutral_block = [E::one(); MAX_FACTOR_WIDTH];
    let factors = [&neutral_block[..layout.num_blocks], gadget_row];
    eval_offset_eq_tensor(col_point, layout.opening_digit_offset, E::one(), &factors)
}

/// Column factor `eq_seg · eq_block · gadget` for the opening-digit segment (`K = 1`).
fn opening_digit_col_factor_k1<E: FieldCore>(
    layout: &TraceWeightLayout,
    col_point: &[E],
    block_row: &[E],
    gadget_row: &[E],
) -> Result<E, AkitaError> {
    eval_offset_eq_tensor(
        col_point,
        layout.opening_digit_offset,
        E::one(),
        &[block_row, gadget_row],
    )
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
            inner_open,
        } => {
            if K != 1 {
                return Err(AkitaError::InvalidInput(
                    "field opening weights require K = 1".to_string(),
                ));
            }
            eval_at_point_k1::<F, E, D>(layout, ring_point, col_point, block_weights, inner_open)
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
    inner_open: &[F],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt + FieldCore,
{
    if inner_open.len() != layout.ring_bits || block_weights.len() != layout.num_blocks {
        return Err(AkitaError::InvalidInput(
            "field opening weights do not match layout".to_string(),
        ));
    }
    layout.validate_closed_form_eval_point(ring_point.len(), col_point.len())?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars)?;
    let mut block_row = [E::zero(); MAX_FACTOR_WIDTH];
    if block_weights.len() > MAX_FACTOR_WIDTH {
        return Err(AkitaError::InvalidInput(
            "block weight count exceeds stack bound".to_string(),
        ));
    }
    for (dst, &src) in block_row[..block_weights.len()]
        .iter_mut()
        .zip(block_weights.iter())
    {
        *dst = E::lift_base(src);
    }
    let col_factor = opening_digit_col_factor_k1(
        layout,
        col_point,
        &block_row[..block_weights.len()],
        &gadget_row[..gadget_scalars.len()],
    )?;
    let inner_factor = eq_mle_base_point::<F, E>(ring_point, inner_open)?;
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
    layout.validate_closed_form_eval_point(ring_point.len(), col_point.len())?;

    let col_block = &col_point[..layout.r_vars];
    let block_eq = lagrange_weights(col_block)?;
    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let gadget_row = lift_gadget_row::<F, E>(&gadget_scalars)?;
    let col_factor =
        opening_digit_gadget_factor(layout, col_point, &gadget_row[..gadget_scalars.len()])?;

    let ring_eq = lagrange_weights(ring_point)?;
    let mut block_inner = E::zero();
    for (block, block_ring) in block_rings.iter().enumerate().take(layout.num_blocks) {
        block_inner += block_eq[block]
            * trace_open_ring_mle_dot::<F, E, D>(
                block_ring,
                &ring_eq,
                packed_inner_point,
                layout.ring_bits,
            )?;
    }
    Ok(col_factor * block_inner)
}
