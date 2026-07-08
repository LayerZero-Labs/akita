use super::super::{checked_add, checked_mul};
use super::types::SetupContributionGroupInputs;
use crate::layout::flat_matrix::FlatRingMatrixView;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_field::parallel::*;
use akita_field::{AkitaError, ExtField, FieldCore, MulBaseUnreduced};

pub(super) struct GroupSetupSegment<E> {
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) has_d: bool,
    pub(super) d_start_abs: usize,
    pub(super) d_weight: E,
    pub(super) has_b: bool,
    pub(super) b_start_abs: usize,
    pub(super) b_weight: E,
    pub(super) has_a: bool,
    pub(super) a_start_abs: usize,
    pub(super) a_weight: E,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_packed_scan_access<F, E>(
    d_rows: usize,
    d_physical_cols: usize,
    d_view: Option<&FlatRingMatrixView<'_, F>>,
    n_b: usize,
    t_cols: usize,
    b_view: &FlatRingMatrixView<'_, F>,
    n_a: usize,
    z_cols: usize,
    a_view: &FlatRingMatrixView<'_, F>,
    segments: &[GroupSetupSegment<E>],
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    for segment in segments {
        if segment.has_d && d_view.is_none() {
            return Err(AkitaError::InvalidSetup(
                "setup packed D scan missing D view".into(),
            ));
        }
    }
    let d_required = checked_mul(d_rows, d_physical_cols, "setup D footprint")?;
    if d_required > 0 {
        if let Some(d_view) = d_view {
            let probe = d_required - 1;
            d_view.elem(probe / d_physical_cols, probe % d_physical_cols)?;
        }
    }
    let b_required = checked_mul(n_b, t_cols, "setup B footprint")?;
    if b_required > 0 {
        let probe = b_required - 1;
        b_view.elem(probe / t_cols, probe % t_cols)?;
    }
    let a_required = checked_mul(n_a, z_cols, "setup A footprint")?;
    if a_required > 0 {
        let probe = a_required - 1;
        a_view.elem(probe / z_cols, probe % z_cols)?;
    }
    Ok(())
}

pub(super) struct AlphaChunkScales<E> {
    pub(super) scales: Vec<E>,
    pub(super) shift: usize,
    pub(super) mask: usize,
}

pub(super) fn alpha_chunk_scales<E: FieldCore>(
    alpha_pows: &[E],
    base_pows: &[E],
) -> Option<AlphaChunkScales<E>> {
    let base_d = base_pows.len();
    if base_d == 0 || !alpha_pows.len().is_multiple_of(base_d) {
        return None;
    }
    let ratio = alpha_pows.len() / base_d;
    if ratio == 0 || !ratio.is_power_of_two() {
        return None;
    }
    let mut scales = Vec::with_capacity(ratio);
    for chunk in 0..ratio {
        let offset = chunk * base_d;
        let scale = alpha_pows[offset];
        for idx in 0..base_d {
            if alpha_pows[offset + idx] != scale * base_pows[idx] {
                return None;
            }
        }
        scales.push(scale);
    }
    Some(AlphaChunkScales {
        scales,
        shift: ratio.trailing_zeros() as usize,
        mask: ratio - 1,
    })
}

pub(super) fn scaled_row_weights<E: FieldCore>(row_weights: &[E], scales: &[E]) -> Vec<E> {
    let mut scaled = Vec::with_capacity(row_weights.len() * scales.len());
    for &row_weight in row_weights {
        scaled.extend(scales.iter().map(|&scale| row_weight * scale));
    }
    scaled
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn group_bar_omega_segment_eval<
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    eq_lambda: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    E: FieldCore,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * e_eq[lambda - d_start];
            }
            if HAS_B {
                weight += b_weight * t_eq[lambda - b_start];
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if !weight.is_zero() {
                acc += eq_lambda[lambda] * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_uniform_group_slice_inner_sum<
    F,
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_view: &FlatRingMatrixView<'_, F>,
    alpha_pows: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * e_eq[lambda - d_start];
            }
            if HAS_B {
                weight += b_weight * t_eq[lambda - b_start];
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if !weight.is_zero() {
                let coeffs = setup_view.elem_in_band(0, lambda);
                acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_group_slice_inner_sum<
    F,
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    d_view: Option<&FlatRingMatrixView<'_, F>>,
    d_physical_cols: usize,
    b_view: &FlatRingMatrixView<'_, F>,
    t_cols: usize,
    a_view: &FlatRingMatrixView<'_, F>,
    z_cols: usize,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            if HAS_D {
                let eq_w = d_weight * e_eq[lambda - d_start];
                if !eq_w.is_zero() {
                    if let Some(d_view) = d_view {
                        let d_row = lambda / d_physical_cols;
                        let d_col = lambda % d_physical_cols;
                        let coeffs = d_view.elem_in_band(d_row, d_col);
                        acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_d) * eq_w;
                    }
                }
            }
            if HAS_B {
                let eq_w = b_weight * t_eq[lambda - b_start];
                if !eq_w.is_zero() {
                    let b_row = lambda / t_cols;
                    let b_col = lambda % t_cols;
                    let coeffs = b_view.elem_in_band(b_row, b_col);
                    acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_b) * eq_w;
                }
            }
            if HAS_A {
                let eq_w = a_weight * z_eq[lambda - a_start];
                if !eq_w.is_zero() {
                    let a_row = lambda / z_cols;
                    let a_col = lambda % z_cols;
                    let coeffs = a_view.elem_in_band(a_row, a_col);
                    acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_a) * eq_w;
                }
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

pub(super) fn validate_group_chunk_layout(
    group: &SetupContributionGroupInputs,
    num_groups: usize,
) -> Result<(), AkitaError> {
    if group.chunks.is_empty()
        || group.blocks_per_chunk == 0
        || !group.blocks_per_chunk.is_power_of_two()
    {
        return Err(AkitaError::InvalidSetup(
            "malformed setup witness chunk layout".into(),
        ));
    }
    if checked_mul(
        group.chunks.len(),
        group.blocks_per_chunk,
        "setup chunk block coverage",
    )? != group.num_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "setup witness chunk windows do not tile num_blocks".into(),
        ));
    }
    if group.chunks.len() > 1 && num_groups != 1 {
        return Err(AkitaError::InvalidSetup(
            "multi-chunk setup contribution requires exactly one commitment group".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn evaluate_weighted_setup_row<Base, E>(
    row: &[Base],
    col_offset: usize,
    col_weights: &[E],
    row_weight: E,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    Base: FieldCore,
    E: ExtField<Base> + MulBaseUnreduced<Base>,
{
    use super::super::checked_slice;

    let ring_d = alpha_pows.len();
    let mut acc = E::zero();
    for (col, &col_weight) in col_weights.iter().enumerate() {
        if col_weight.is_zero() {
            continue;
        }
        let setup_col = checked_add(col_offset, col, "weighted setup column")?;
        let coeff_start = checked_mul(setup_col, ring_d, "weighted setup coeff start")?;
        let coeffs = checked_slice(row, coeff_start, ring_d, "weighted setup coeffs")?;
        acc += row_weight * col_weight * eval_flat_ring_at_pows_fast::<Base, E>(coeffs, alpha_pows);
    }
    Ok(acc)
}

#[inline(always)]
pub(super) fn push_group_d_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    active_col_start: usize,
    active_cols: usize,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let active_col_end = checked_add(active_col_start, active_cols, "setup D active columns")?;
    let mut row_start = 0usize;
    for _ in 0..rows {
        let row_end = checked_add(row_start, stride, "packed D boundary")?;
        endpoints.push(row_end);
        if active_cols != 0 {
            endpoints.push(checked_add(
                row_start,
                active_col_start,
                "setup D active boundary",
            )?);
            endpoints.push(checked_add(
                row_start,
                active_col_end,
                "setup D active boundary",
            )?);
        }
        row_start = row_end;
    }
    Ok(())
}
