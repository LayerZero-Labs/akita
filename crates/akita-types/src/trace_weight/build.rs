use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use crate::field_reduction::trace_open_ring_row;
use crate::{gadget_row_scalars, RingSubfieldEncoding};

use super::eval::{TraceFieldBlockOpening, TraceRingBlockOpening};
use super::layout::TraceWeightLayout;
use super::trace_table::TraceSparseColumn;

/// Write `gadget[plane] · block_rows[block][ring_coord]` into the witness table.
fn fill_opening_digit_table<F, E>(
    layout: &TraceWeightLayout,
    gadget_scalars: &[F],
    block_rows: &[E],
    table: &mut [E],
) where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt,
{
    let ring_len = layout.ring_len();
    debug_assert_eq!(block_rows.len(), layout.num_blocks * ring_len);
    for (plane, gadget_scalar) in gadget_scalars.iter().enumerate() {
        let gadget = E::lift_base(*gadget_scalar);
        for block in 0..layout.num_blocks {
            let col = layout.opening_digit_col_index(block, plane);
            let row_base = block * ring_len;
            for ring_coord in 0..ring_len {
                let idx = layout.witness_index(col, ring_coord);
                table[idx] = gadget * block_rows[row_base + ring_coord];
            }
        }
    }
}

fn compact_table_len(layout: &TraceWeightLayout, live_x_cols: usize) -> Result<usize, AkitaError> {
    let x_len = 1usize
        .checked_shl(layout.col_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("trace-weight x length overflow".to_string()))?;
    if live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: x_len,
            actual: live_x_cols,
        });
    }
    live_x_cols.checked_mul(layout.ring_len()).ok_or_else(|| {
        AkitaError::InvalidInput("trace-weight compact table length overflow".to_string())
    })
}

fn block_has_live_opening_digit(
    layout: &TraceWeightLayout,
    block: usize,
    live_x_cols: usize,
) -> bool {
    (0..layout.num_digits_open)
        .any(|plane| layout.opening_digit_col_index(block, plane) < live_x_cols)
}

fn add_ring_row_to_compact<F, E>(
    layout: &TraceWeightLayout,
    gadget_scalars: &[F],
    block: usize,
    row: &[E],
    live_x_cols: usize,
    output_scale: E,
    compact: &mut [E],
) where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt,
{
    let ring_len = layout.ring_len();
    debug_assert_eq!(row.len(), ring_len);
    for (plane, gadget_scalar) in gadget_scalars.iter().enumerate() {
        let col = layout.opening_digit_col_index(block, plane);
        if col >= live_x_cols {
            continue;
        }
        let gadget = output_scale * E::lift_base(*gadget_scalar);
        let dst_base = col * ring_len;
        for (dst, value) in compact[dst_base..dst_base + ring_len].iter_mut().zip(row) {
            *dst += gadget * *value;
        }
    }
}

/// Build the full Boolean trace-weight table for scalar (`K = 1`) block weights.
///
/// `block_weights` should be `lagrange_weights(b_open)`.
pub fn build_trace_weight_table_field_block_weights<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    block_weights: &[F],
    inner_opening_ring: &CyclotomicRing<F, D>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt,
{
    let term = TraceFieldBlockOpening {
        block_offset: 0,
        block_weights: block_weights.to_vec(),
        inner_opening_ring: *inner_opening_ring,
    };
    build_trace_weight_table_field_terms(layout, &[term])
}

/// Build the full Boolean trace-weight table for scalar (`K = 1`) terms.
pub fn build_trace_weight_table_field_terms<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field trace terms must be non-empty".to_string(),
        ));
    }
    layout.validate_ring_dimension::<D>()?;
    layout.validate_opening_digit_segment()?;

    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let ring_len = layout.ring_len();
    let mut block_rows = vec![E::zero(); layout.num_blocks * ring_len];

    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_weights.len())?;
        let inner_coeffs = term.inner_opening_ring.coefficients();
        for (local_block, block_weight) in term.block_weights.iter().enumerate() {
            let block_weight_e = E::lift_base(*block_weight);
            let row_base = (term.block_offset + local_block) * ring_len;
            for (ring_coord, coeff) in inner_coeffs.iter().enumerate().take(ring_len) {
                block_rows[row_base + ring_coord] += block_weight_e * E::lift_base(*coeff);
            }
        }
    }

    let mut table = vec![E::zero(); layout.table_len()?];
    fill_opening_digit_table(layout, &gadget_scalars, &block_rows, &mut table);
    Ok(table)
}

/// Build sparse opening-digit columns for the live `K = 1` stage-2 trace table.
pub(crate) fn build_trace_weight_compact_field_sparse_scaled<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    terms: &[TraceFieldBlockOpening<F, D>],
    live_x_cols: usize,
    output_scale: E,
) -> Result<Vec<TraceSparseColumn<E>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field trace terms must be non-empty".to_string(),
        ));
    }
    layout.validate_ring_dimension::<D>()?;
    layout.validate_opening_digit_segment()?;
    let _ = compact_table_len(layout, live_x_cols)?;

    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let ring_len = layout.ring_len();
    let mut columns = Vec::new();

    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_weights.len())?;
        let inner_coeffs = term.inner_opening_ring.coefficients();
        for (local_block, block_weight) in term.block_weights.iter().enumerate() {
            let block = term.block_offset + local_block;
            let block_weight_e = output_scale * E::lift_base(*block_weight);
            for (plane, gadget_scalar) in gadget_scalars.iter().enumerate() {
                let col = layout.opening_digit_col_index(block, plane);
                if col >= live_x_cols {
                    continue;
                }
                let scale = block_weight_e * E::lift_base(*gadget_scalar);
                let values = inner_coeffs
                    .iter()
                    .take(ring_len)
                    .map(|&coeff| scale * E::lift_base(coeff))
                    .collect();
                columns.push(TraceSparseColumn { col, values });
            }
        }
    }

    Ok(columns)
}

/// Build the full Boolean trace-weight table for ring (`K > 1`) block weights.
///
/// `block_rings` should come from [`crate::block_rings_at_opening`].
pub fn build_trace_weight_table_ring_block_weights<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    block_rings: &[CyclotomicRing<F, D>],
    packed_inner_point: &CyclotomicRing<F, D>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let term = TraceRingBlockOpening {
        block_offset: 0,
        block_rings: block_rings.to_vec(),
        packed_inner_point: *packed_inner_point,
    };
    build_trace_weight_table_ring_terms(layout, &[term])
}

/// Build the full Boolean trace-weight table for ring (`K > 1`) terms.
pub fn build_trace_weight_table_ring_terms<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring trace terms must be non-empty".to_string(),
        ));
    }
    layout.validate_ring_dimension::<D>()?;
    layout.validate_opening_digit_segment()?;

    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let ring_len = layout.ring_len();
    let mut block_rows = vec![E::zero(); layout.num_blocks * ring_len];

    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_rings.len())?;
        for (local_block, block_ring) in term.block_rings.iter().enumerate() {
            let row = trace_open_ring_row::<F, E, D>(
                block_ring,
                &term.packed_inner_point,
                layout.ring_bits,
            )?;
            let row_base = (term.block_offset + local_block) * ring_len;
            for (dst, value) in block_rows[row_base..row_base + ring_len]
                .iter_mut()
                .zip(row)
            {
                *dst += value;
            }
        }
    }

    let mut table = vec![E::zero(); layout.table_len()?];
    fill_opening_digit_table(layout, &gadget_scalars, &block_rows, &mut table);
    Ok(table)
}

/// Build the live compact ring (`K > 1`) witness slice and scale every output.
pub(crate) fn build_trace_weight_compact_ring_terms_scaled<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    terms: &[TraceRingBlockOpening<F, D>],
    live_x_cols: usize,
    output_scale: E,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring trace terms must be non-empty".to_string(),
        ));
    }
    layout.validate_ring_dimension::<D>()?;
    layout.validate_opening_digit_segment()?;
    let out_len = compact_table_len(layout, live_x_cols)?;

    let gadget_scalars = gadget_row_scalars::<F>(layout.num_digits_open, layout.log_basis);
    let mut compact = vec![E::zero(); out_len];

    for term in terms {
        layout.validate_trace_term_block_range(term.block_offset, term.block_rings.len())?;
        for (local_block, block_ring) in term.block_rings.iter().enumerate() {
            let block = term.block_offset + local_block;
            if !block_has_live_opening_digit(layout, block, live_x_cols) {
                continue;
            }
            let row = trace_open_ring_row::<F, E, D>(
                block_ring,
                &term.packed_inner_point,
                layout.ring_bits,
            )?;
            add_ring_row_to_compact(
                layout,
                &gadget_scalars,
                block,
                &row,
                live_x_cols,
                output_scale,
                &mut compact,
            );
        }
    }

    Ok(compact)
}
