//! Dense left-weight materialization from paired equality tensor families.

use super::{
    charge_work, checked_axis_offset, EqPairTensorAxis, EqPairTensorFamily, EqPairTensorWeights,
};
use crate::offset_eq::OffsetEqWindow;
use crate::Field;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use std::ops::Range;

/// Materialize the left-address weights induced by tensor families and one
/// equality point on the right.
///
/// # Errors
///
/// Returns an error for malformed geometry, an out-of-range left address, or
/// recurrence work above [`crate::offset_eq::MAX_COMPACT_STRIDE_TERMS`].
pub fn materialize_eq_tensor_left<F: Field>(
    equality: &OffsetEqWindow<F>,
    families: &[EqPairTensorFamily<F>],
    output_len: usize,
) -> Result<Vec<F>, AkitaError> {
    // Every path below sizes work and storage from `axis.len`, which the
    // public fields do not tie to the weight tables.
    families
        .iter()
        .flat_map(|family| &family.axes)
        .try_for_each(EqPairTensorAxis::validate_weights)?;
    if let Some(output) = materialize_disjoint_unit_intervals(equality, families, output_len)? {
        return Ok(output);
    }
    if let Some(overlap) = DenseLeftOverlap::plan(families, output_len, MAX_RESIDUAL_TABLE_BYTES)? {
        return overlap.materialize(equality, output_len);
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

fn materialize_disjoint_unit_intervals<F: Field>(
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
                .ok_or_else(|| {
                    AkitaError::Internal("merged tensor interval has reversed endpoints".into())
                })?;
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
        let destination = output.get_mut(left_start..left_end).ok_or_else(|| {
            AkitaError::Internal("merged tensor interval exceeds the output".into())
        })?;
        equality.fill_interval(right_start, destination)?;
    }
    Ok(Some(output))
}

/// Byte budget for the residual tables of one dense-left materialization.
///
/// The work bound alone does not bound memory: a two-coordinate destination
/// axis against a `2^27`-coordinate residual is within the work limit yet would
/// need gigabytes of tables for two outputs. Residuals above this budget are
/// contracted by streaming instead, which recomputes the residual walk for
/// every destination coordinate but needs only recursion-depth scratch. The
/// PCS end-to-end suites use under 2 KiB of residual tables per call.
pub(in crate::offset_eq) const MAX_RESIDUAL_TABLE_BYTES: usize = 1 << 20;

/// How one view contracts the axes other than its destination axis.
pub(in crate::offset_eq) enum ResidualTerms<F: Field> {
    /// Precomputed nonzero `(right offset, weight)` residual terms.
    Table(Vec<(usize, F)>),
    /// Walk the residual axes again for every destination coordinate.
    Streamed,
}

pub(in crate::offset_eq) struct DenseLeftTensorView<'a, F: Field> {
    family: &'a EqPairTensorFamily<F>,
    destination_axis: usize,
    pub(in crate::offset_eq) residual: ResidualTerms<F>,
}

impl<F: Field> DenseLeftTensorView<'_, F> {
    /// Number of nonzero-weight residual terms. Each axis count is at most
    /// its validated `len`, so this never exceeds the checked coordinate
    /// count charged to the work bound.
    fn live_residual_len(&self) -> Result<usize, AkitaError> {
        if self.family.scalar.is_zero() {
            return Ok(0);
        }
        self.family
            .axes
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != self.destination_axis)
            .try_fold(1usize, |count, (_, axis)| {
                count.checked_mul(axis.nonzero_len()?).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor residual count overflow".into())
                })
            })
    }

    fn tabulate_residual(&mut self, live: usize) -> Result<(), AkitaError> {
        let mut table = Vec::new();
        table.try_reserve_exact(live).map_err(|_| {
            AkitaError::InvalidInput("paired tensor residual table allocation failed".into())
        })?;
        collect_residual(
            self.family,
            self.destination_axis,
            &mut table,
            0,
            0,
            F::one(),
        )?;
        self.residual = ResidualTerms::Table(table);
        Ok(())
    }

    /// Contract the residual axes at one destination coordinate, including
    /// the family scalar.
    fn evaluate_residual(
        &self,
        equality: &OffsetEqWindow<F>,
        base: usize,
    ) -> Result<F, AkitaError> {
        let scalar = self.family.scalar;
        match &self.residual {
            ResidualTerms::Table(table) => {
                let mut value = F::zero();
                for &(offset, weight) in table {
                    let index = base.checked_add(offset).ok_or_else(|| {
                        AkitaError::InvalidInput("paired tensor right offset overflow".into())
                    })?;
                    let term = equality.eval(index);
                    value += if weight == F::one() {
                        term
                    } else {
                        weight * term
                    };
                }
                Ok(if scalar == F::one() {
                    value
                } else {
                    scalar * value
                })
            }
            ResidualTerms::Streamed => contract_residual_tensor_axes(
                equality,
                self.family,
                self.destination_axis,
                0,
                base,
                scalar,
            ),
        }
    }
}

fn collect_residual<F: Field>(
    family: &EqPairTensorFamily<F>,
    destination_axis: usize,
    table: &mut Vec<(usize, F)>,
    axis_index: usize,
    offset: usize,
    weight: F,
) -> Result<(), AkitaError> {
    if axis_index == family.axes.len() {
        table.push((offset, weight));
        return Ok(());
    }
    if axis_index == destination_axis {
        return collect_residual(
            family,
            destination_axis,
            table,
            axis_index + 1,
            offset,
            weight,
        );
    }
    let axis = family
        .axes
        .get(axis_index)
        .ok_or_else(|| AkitaError::Internal("residual tensor recursion axis is absent".into()))?;
    for coordinate in 0..axis.len {
        let factor = axis.coordinate_weight(coordinate).ok_or_else(|| {
            AkitaError::InvalidInput("residual tensor axis coordinate weight is absent".into())
        })?;
        if factor.is_zero() {
            continue;
        }
        let next = checked_axis_offset(offset, axis.right_stride, coordinate, "right")?;
        let weight = if factor == F::one() {
            weight
        } else {
            weight * factor
        };
        collect_residual(
            family,
            destination_axis,
            table,
            axis_index + 1,
            next,
            weight,
        )?;
    }
    Ok(())
}

/// Families that all write one contiguous left interval through a unit
/// destination axis, with every other axis acting only on the right domain.
pub(in crate::offset_eq) struct DenseLeftOverlap<'a, F: Field> {
    pub(in crate::offset_eq) views: Vec<DenseLeftTensorView<'a, F>>,
    destination: Range<usize>,
    work: usize,
}

impl<'a, F: Field> DenseLeftOverlap<'a, F> {
    /// Recognize the dense-left shape, charge its work, and tabulate the
    /// residuals when they fit `residual_budget_bytes`. Returns `None` when
    /// the families do not share one dense destination interval.
    ///
    /// Callers must have validated every axis's weight geometry.
    pub(in crate::offset_eq) fn plan(
        families: &'a [EqPairTensorFamily<F>],
        output_len: usize,
        residual_budget_bytes: usize,
    ) -> Result<Option<Self>, AkitaError> {
        let Some(first) = families.first() else {
            return Ok(None);
        };
        let Some(first_destination_axis) = dense_left_destination_axis(first) else {
            return Ok(None);
        };
        let first_axis = first.axes.get(first_destination_axis).ok_or_else(|| {
            AkitaError::Internal("selected first tensor destination axis is absent".into())
        })?;
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
            let axis = family.axes.get(destination_axis).ok_or_else(|| {
                AkitaError::Internal("selected tensor destination axis is absent".into())
            })?;
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
                residual: ResidualTerms::Streamed,
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

        // Tables pay off only when the destination axis revisits them; each
        // holds only nonzero-weight residuals, so zero weights cost nothing.
        if first_axis.len > 1 {
            let mut live = Vec::with_capacity(views.len());
            let mut live_total = 0usize;
            for view in &views {
                let count = view.live_residual_len()?;
                live_total = live_total.checked_add(count).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor residual count overflow".into())
                })?;
                live.push(count);
            }
            let fits = live_total
                .checked_mul(size_of::<(usize, F)>())
                .is_some_and(|bytes| bytes <= residual_budget_bytes);
            for (view, count) in views.iter_mut().zip(live) {
                if count == 0 {
                    view.residual = ResidualTerms::Table(Vec::new());
                } else if fits {
                    view.tabulate_residual(count)?;
                }
            }
        }
        Ok(Some(Self {
            views,
            destination: destination_start..destination_end,
            work,
        }))
    }

    pub(in crate::offset_eq) fn materialize(
        &self,
        equality: &OffsetEqWindow<F>,
        output_len: usize,
    ) -> Result<Vec<F>, AkitaError> {
        let evaluate_coordinate = |(coordinate, destination): (usize, &mut F)| {
            for view in &self.views {
                if view.family.scalar.is_zero() {
                    continue;
                }
                let axis = view.family.axes.get(view.destination_axis).ok_or_else(|| {
                    AkitaError::Internal("tensor view destination axis is absent".into())
                })?;
                let right_offset = checked_axis_offset(
                    view.family.right_offset,
                    axis.right_stride,
                    coordinate,
                    "right",
                )?;
                *destination += view.evaluate_residual(equality, right_offset)?;
            }
            Ok::<_, AkitaError>(())
        };
        let mut output = vec![F::zero(); output_len];
        let destination = output.get_mut(self.destination.clone()).ok_or_else(|| {
            AkitaError::Internal("validated tensor destination span exceeds the output".into())
        })?;
        // Tuned with `benches/offset_eq_window.rs::bench_materialize_disjoint_intervals`
        // on an Apple M4 Max (16 cores, 64 GiB).
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        if self.work >= PARALLEL_THRESHOLD {
            cfg_iter_mut!(destination)
                .enumerate()
                .try_for_each(evaluate_coordinate)?;
        } else {
            destination
                .iter_mut()
                .enumerate()
                .try_for_each(evaluate_coordinate)?;
        }
        Ok(output)
    }
}

fn dense_left_destination_axis<F: Field>(family: &EqPairTensorFamily<F>) -> Option<usize> {
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

fn contract_residual_tensor_axes<F: Field>(
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
        .ok_or_else(|| AkitaError::Internal("residual tensor recursion axis is absent".into()))?;
    let mut acc = F::zero();
    for coordinate in 0..axis.len {
        let axis_weight = axis.coordinate_weight(coordinate).ok_or_else(|| {
            AkitaError::InvalidInput("residual tensor axis coordinate weight is absent".into())
        })?;
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

fn visit_tensor_coordinates<F: Field>(
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
        .ok_or_else(|| AkitaError::Internal("tensor coordinate recursion axis is absent".into()))?;
    for coordinate in 0..axis.len {
        let axis_weight = axis.coordinate_weight(coordinate).ok_or_else(|| {
            AkitaError::InvalidInput("visited tensor axis coordinate weight is absent".into())
        })?;
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
