//! Dense left-weight materialization from paired equality tensor families.

use super::{charge_work, checked_axis_offset, EqPairTensorFamily, EqPairTensorWeights};
use crate::offset_eq::OffsetEqWindow;
use crate::FieldCore;
use akita_error::AkitaError;
use akita_field::parallel::*;

/// Materialize the left-address weights induced by tensor families and one
/// equality point on the right.
///
/// # Errors
///
/// Returns an error for malformed geometry, an out-of-range left address, or
/// recurrence work above [`crate::offset_eq::MAX_COMPACT_STRIDE_TERMS`].
pub fn materialize_eq_tensor_left<F: FieldCore>(
    equality: &OffsetEqWindow<F>,
    families: &[EqPairTensorFamily<F>],
    output_len: usize,
) -> Result<Vec<F>, AkitaError> {
    if let Some(output) = materialize_disjoint_unit_intervals(equality, families, output_len)? {
        return Ok(output);
    }
    if let Some(output) = materialize_dense_left_overlap(equality, families, output_len)? {
        return Ok(output);
    }
    let mut output = vec![F::zero(); output_len];
    let mut work = 0usize;
    for family in families {
        if family.scalar == F::one() {
            if let [axis] = family.axes.as_slice() {
                if matches!(axis.weights, EqPairTensorWeights::Unit)
                    && axis.left_stride == 1
                    && axis.right_stride == 1
                {
                    let end = family.left_offset.checked_add(axis.len).ok_or_else(|| {
                        AkitaError::InvalidInput("paired tensor left span overflow".into())
                    })?;
                    let destination = output.get_mut(family.left_offset..end).ok_or_else(|| {
                        AkitaError::InvalidInput("paired tensor left address out of range".into())
                    })?;
                    if destination.iter().all(|value| value.is_zero()) {
                        charge_work(&mut work, axis.len)?;
                        equality.fill_interval(family.right_offset, destination)?;
                        continue;
                    }
                }
            }
        }
        visit_tensor_coordinates(
            family,
            0,
            family.left_offset,
            family.right_offset,
            family.scalar,
            &mut work,
            &mut |left, right, weight| {
                let destination = output.get_mut(left).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor left address out of range".into())
                })?;
                let equality = equality.eval(right);
                *destination += if weight == F::one() {
                    equality
                } else {
                    weight * equality
                };
                Ok(())
            },
        )?;
    }
    Ok(output)
}

fn materialize_disjoint_unit_intervals<F: FieldCore>(
    equality: &OffsetEqWindow<F>,
    families: &[EqPairTensorFamily<F>],
    output_len: usize,
) -> Result<Option<Vec<F>>, AkitaError> {
    let mut intervals = Vec::new();
    intervals
        .try_reserve(families.len())
        .map_err(|_| AkitaError::InvalidInput("paired tensor interval allocation failed".into()))?;
    let mut work = 0usize;
    for family in families {
        if family.scalar.is_zero() {
            continue;
        }
        let [axis] = family.axes.as_slice() else {
            return Ok(None);
        };
        if family.scalar != F::one()
            || !matches!(axis.weights, EqPairTensorWeights::Unit)
            || axis.left_stride != 1
            || axis.right_stride != 1
        {
            return Ok(None);
        }
        let left_end = family
            .left_offset
            .checked_add(axis.len)
            .ok_or_else(|| AkitaError::InvalidInput("paired tensor left span overflow".into()))?;
        if left_end > output_len {
            return Err(AkitaError::InvalidInput(
                "paired tensor left address out of range".into(),
            ));
        }
        charge_work(&mut work, axis.len)?;
        intervals.push((family.left_offset, left_end, family.right_offset));
    }
    intervals.sort_unstable_by_key(|&(left_start, _, _)| left_start);

    let mut merged_len = 0usize;
    for read in 0..intervals.len() {
        let (left_start, left_end, right_start) = intervals[read];
        if merged_len != 0 {
            let (previous_left_start, previous_left_end, previous_right_start) =
                &mut intervals[merged_len - 1];
            if left_start < *previous_left_end {
                return Ok(None);
            }
            let previous_len = previous_left_end
                .checked_sub(*previous_left_start)
                .ok_or(AkitaError::InvalidProof)?;
            let previous_right_end =
                previous_right_start
                    .checked_add(previous_len)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("paired tensor right span overflow".into())
                    })?;
            if left_start == *previous_left_end && right_start == previous_right_end {
                *previous_left_end = left_end;
                continue;
            }
        }
        intervals[merged_len] = (left_start, left_end, right_start);
        merged_len += 1;
    }
    intervals.truncate(merged_len);

    let mut output = vec![F::zero(); output_len];
    for (left_start, left_end, right_start) in intervals {
        let destination = output
            .get_mut(left_start..left_end)
            .ok_or(AkitaError::InvalidProof)?;
        equality.fill_interval(right_start, destination)?;
    }
    Ok(Some(output))
}

struct DenseLeftTensorView<'a, F: FieldCore> {
    family: &'a EqPairTensorFamily<F>,
    destination_axis: usize,
}

fn materialize_dense_left_overlap<F: FieldCore>(
    equality: &OffsetEqWindow<F>,
    families: &[EqPairTensorFamily<F>],
    output_len: usize,
) -> Result<Option<Vec<F>>, AkitaError> {
    let Some(first) = families.first() else {
        return Ok(Some(vec![F::zero(); output_len]));
    };
    let Some(first_destination_axis) = dense_left_destination_axis(first) else {
        return Ok(None);
    };
    let first_axis = first
        .axes
        .get(first_destination_axis)
        .ok_or(AkitaError::InvalidProof)?;
    let destination_start = first.left_offset;
    let destination_end = destination_start
        .checked_add(first_axis.len)
        .ok_or_else(|| AkitaError::InvalidInput("paired tensor left span overflow".into()))?;
    if destination_end > output_len {
        return Err(AkitaError::InvalidInput(
            "paired tensor left address out of range".into(),
        ));
    }

    let mut views = Vec::with_capacity(families.len());
    let mut residual_coordinates = 0usize;
    for family in families {
        let Some(destination_axis) = dense_left_destination_axis(family) else {
            return Ok(None);
        };
        let axis = family
            .axes
            .get(destination_axis)
            .ok_or(AkitaError::InvalidProof)?;
        if family.left_offset != destination_start || axis.len != first_axis.len {
            return Ok(None);
        }
        let family_residual = family
            .axes
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != destination_axis)
            .try_fold(1usize, |product, (_, axis)| {
                product.checked_mul(axis.len).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor coordinate count overflow".into())
                })
            })?;
        residual_coordinates = residual_coordinates
            .checked_add(family_residual)
            .ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor coordinate count overflow".into())
            })?;
        views.push(DenseLeftTensorView {
            family,
            destination_axis,
        });
    }
    let work = first_axis
        .len
        .checked_mul(residual_coordinates)
        .ok_or_else(|| AkitaError::InvalidInput("paired tensor work overflow".into()))?;
    if work > crate::offset_eq::MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: crate::offset_eq::MAX_COMPACT_STRIDE_TERMS,
            actual: work,
        });
    }

    let evaluate_coordinate = |(coordinate, destination): (usize, &mut F)| {
        for view in &views {
            let axis = view
                .family
                .axes
                .get(view.destination_axis)
                .ok_or(AkitaError::InvalidProof)?;
            let right_offset = checked_axis_offset(
                view.family.right_offset,
                axis.right_stride,
                coordinate,
                "right",
            )?;
            *destination += contract_residual_tensor_axes(
                equality,
                view.family,
                view.destination_axis,
                0,
                right_offset,
                view.family.scalar,
            )?;
        }
        Ok::<_, AkitaError>(())
    };
    let mut output = vec![F::zero(); output_len];
    let destination = output
        .get_mut(destination_start..destination_end)
        .ok_or(AkitaError::InvalidProof)?;
    // Tuned with `benches/offset_eq_window.rs::bench_materialize_disjoint_intervals`
    // on an Apple M4 Max (16 cores, 64 GiB).
    const PARALLEL_THRESHOLD: usize = 1 << 14;
    if work >= PARALLEL_THRESHOLD {
        cfg_iter_mut!(destination)
            .enumerate()
            .try_for_each(evaluate_coordinate)?;
    } else {
        destination
            .iter_mut()
            .enumerate()
            .try_for_each(evaluate_coordinate)?;
    }
    Ok(Some(output))
}

fn dense_left_destination_axis<F: FieldCore>(family: &EqPairTensorFamily<F>) -> Option<usize> {
    let mut destination = None;
    for (index, axis) in family.axes.iter().enumerate() {
        if axis.left_stride == 1 && matches!(axis.weights, EqPairTensorWeights::Unit) {
            if destination.replace(index).is_some() {
                return None;
            }
        } else if axis.left_stride != 0 {
            return None;
        }
    }
    destination
}

fn contract_residual_tensor_axes<F: FieldCore>(
    equality: &OffsetEqWindow<F>,
    family: &EqPairTensorFamily<F>,
    destination_axis: usize,
    axis_index: usize,
    right_offset: usize,
    weight: F,
) -> Result<F, AkitaError> {
    if weight.is_zero() {
        return Ok(F::zero());
    }
    if axis_index == family.axes.len() {
        let equality = equality.eval(right_offset);
        return Ok(if weight == F::one() {
            equality
        } else {
            weight * equality
        });
    }
    if axis_index == destination_axis {
        return contract_residual_tensor_axes(
            equality,
            family,
            destination_axis,
            axis_index + 1,
            right_offset,
            weight,
        );
    }
    let axis = family
        .axes
        .get(axis_index)
        .ok_or(AkitaError::InvalidProof)?;
    let mut acc = F::zero();
    for coordinate in 0..axis.len {
        let axis_weight = axis
            .coordinate_weight(coordinate)
            .ok_or(AkitaError::InvalidProof)?;
        if axis_weight.is_zero() {
            continue;
        }
        let next_weight = if axis_weight == F::one() {
            weight
        } else if weight == F::one() {
            axis_weight
        } else {
            weight * axis_weight
        };
        acc += contract_residual_tensor_axes(
            equality,
            family,
            destination_axis,
            axis_index + 1,
            checked_axis_offset(right_offset, axis.right_stride, coordinate, "right")?,
            next_weight,
        )?;
    }
    Ok(acc)
}
fn visit_tensor_coordinates<F: FieldCore>(
    family: &EqPairTensorFamily<F>,
    axis_index: usize,
    left_offset: usize,
    right_offset: usize,
    weight: F,
    work: &mut usize,
    visit: &mut impl FnMut(usize, usize, F) -> Result<(), AkitaError>,
) -> Result<(), AkitaError> {
    if weight.is_zero() {
        return Ok(());
    }
    if axis_index == family.axes.len() {
        charge_work(work, 1)?;
        return visit(left_offset, right_offset, weight);
    }
    let axis = family
        .axes
        .get(axis_index)
        .ok_or(AkitaError::InvalidProof)?;
    for coordinate in 0..axis.len {
        let axis_weight = axis
            .coordinate_weight(coordinate)
            .ok_or(AkitaError::InvalidProof)?;
        if axis_weight.is_zero() {
            continue;
        }
        let next_weight = if axis_weight == F::one() {
            weight
        } else if weight == F::one() {
            axis_weight
        } else {
            weight * axis_weight
        };
        visit_tensor_coordinates(
            family,
            axis_index + 1,
            checked_axis_offset(left_offset, axis.left_stride, coordinate, "left")?,
            checked_axis_offset(right_offset, axis.right_stride, coordinate, "right")?,
            next_weight,
            work,
            visit,
        )?;
    }
    Ok(())
}
