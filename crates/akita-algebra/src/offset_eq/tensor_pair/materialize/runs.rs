use super::{DenseLeftTensorView, ResidualTerms};
use crate::offset_eq::OffsetEqWindow;
use crate::Field;
use akita_error::AkitaError;

/// Reorder the residual-by-coordinate sum into runs within one low table.
/// The high factor and residual coefficient are constant within each run.
/// Carries end a run; no inversion or nonzero challenge assumption is used.
pub(super) fn try_accumulate<F: Field>(
    equality: &OffsetEqWindow<F>,
    views: &[DenseLeftTensorView<'_, F>],
    separated: bool,
    destination: &mut [F],
) -> Result<bool, AkitaError> {
    let Some(high_values) = &equality.eq_high else {
        return Ok(false);
    };
    if destination.len() < 16 {
        return Ok(false);
    }
    // Finish admission before writing anything, so declining is transactional.
    // A view that takes the no-carry separated path keeps that dispatch.
    if separated {
        return Ok(false);
    }
    for view in views {
        // Runs need the tabulated residual; a streamed residual declines.
        let ResidualTerms::Table(residual) = &view.residual else {
            return Ok(false);
        };
        let axis =
            view.family.axes.get(view.destination_axis).ok_or_else(|| {
                AkitaError::Internal("tensor run destination axis is absent".into())
            })?;
        if !axis.right_stride.is_power_of_two()
            || axis.right_stride > equality.eq_low.len() / 8
            || axis.len != destination.len()
        {
            return Ok(false);
        }
        let span = axis
            .right_stride
            .checked_mul(axis.len - 1)
            .ok_or_else(|| AkitaError::InvalidInput("tensor run right span overflow".into()))?;
        let last =
            view.family.right_offset.checked_add(span).ok_or_else(|| {
                AkitaError::InvalidInput("tensor run right offset overflow".into())
            })?;
        if !view.family.scalar.is_zero() {
            for &(offset, _) in residual {
                last.checked_add(offset).ok_or_else(|| {
                    AkitaError::InvalidInput("tensor run residual offset overflow".into())
                })?;
            }
        }
    }
    for view in views {
        if view.family.scalar.is_zero() {
            continue;
        }
        let axis = view.family.axes.get(view.destination_axis).ok_or_else(|| {
            AkitaError::Internal("validated tensor run destination axis is absent".into())
        })?;
        let stride_bits = axis.right_stride.trailing_zeros();
        let ResidualTerms::Table(residual) = &view.residual else {
            return Err(AkitaError::Internal(
                "validated tensor run residual is not tabulated".into(),
            ));
        };
        for &(offset, weight) in residual {
            let coefficient = if view.family.scalar == F::one() {
                weight
            } else {
                view.family.scalar * weight
            };
            let mut index = view
                .family
                .right_offset
                .checked_add(offset)
                .ok_or_else(|| {
                    AkitaError::Internal("validated tensor run offset overflow".into())
                })?;
            let mut remaining = &mut destination[..];
            while !remaining.is_empty() {
                let low = index & equality.low_mask;
                let high = index >> equality.low_bits;
                let Some(&high_value) = high_values.get(high) else {
                    // Monotone physical addresses cannot reenter the domain.
                    break;
                };
                let available = ((equality.low_mask - low) >> stride_bits) + 1;
                let take = available.min(remaining.len());
                let (run, tail) = remaining.split_at_mut_checked(take).ok_or_else(|| {
                    AkitaError::Internal("validated tensor run split exceeds output".into())
                })?;
                let scale = if coefficient == F::one() {
                    high_value
                } else {
                    coefficient * high_value
                };
                if !scale.is_zero() {
                    let low_values = equality.eq_low.get(low..).ok_or_else(|| {
                        AkitaError::Internal("tensor run low index exceeds table".into())
                    })?;
                    for (value, &low_value) in run
                        .iter_mut()
                        .zip(low_values.iter().step_by(axis.right_stride))
                    {
                        *value += low_value * scale;
                    }
                }
                if tail.is_empty() {
                    break;
                }
                index = axis
                    .right_stride
                    .checked_mul(take)
                    .and_then(|span| index.checked_add(span))
                    .ok_or_else(|| {
                        AkitaError::Internal("validated tensor run advance overflow".into())
                    })?;
                remaining = tail;
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
