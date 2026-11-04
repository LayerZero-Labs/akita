use super::{DenseLeftTensorView, ResidualTerms};
use crate::offset_eq::{eq_eval_at_index, OffsetEqWindow};
use crate::Field;
use akita_error::{checked, AkitaError};

pub(super) struct SeparatedResidual<F: Field> {
    origin: usize,
    low_origin: usize,
    high_sum: F,
}

impl<F: Field> SeparatedResidual<F> {
    /// A shared low residue and a no-carry span make the high weighted sum
    /// independent of the destination coordinate. No field division is used.
    pub(super) fn prepare(
        view: &DenseLeftTensorView<'_, F>,
        equality: &OffsetEqWindow<F>,
    ) -> Result<Option<Self>, AkitaError> {
        // Only a tabulated residual is separated; a streamed residual keeps
        // the original walk.
        let ResidualTerms::Table(residual) = &view.residual else {
            return Ok(None);
        };
        if residual.len() < 2 {
            return Ok(None);
        }
        let axis =
            view.family.axes.get(view.destination_axis).ok_or_else(|| {
                AkitaError::Internal("tensor view destination axis is absent".into())
            })?;
        let Some(last) = axis.len.checked_sub(1) else {
            return Ok(None);
        };
        let Some(span) = axis.right_stride.checked_mul(last) else {
            return Ok(None);
        };
        let Some(&(first_offset, _)) = residual.first() else {
            return Ok(None);
        };
        let origin = view.family.right_offset;
        let Some(first_index) = origin.checked_add(first_offset) else {
            return Ok(None);
        };
        let low_origin = first_index & equality.low_mask;
        if low_origin
            .checked_add(span)
            .is_none_or(|end| end > equality.low_mask)
        {
            return Ok(None);
        }
        // Validate every residual before doing field arithmetic. The original
        // path remains responsible for errors when this optimization declines.
        for &(offset, _) in residual {
            let Some(index) = origin.checked_add(offset) else {
                return Ok(None);
            };
            if index & equality.low_mask != low_origin || index.checked_add(span).is_none() {
                return Ok(None);
            }
        }
        let mut high_sum = F::zero();
        for &(offset, weight) in residual {
            // The validation above bounds the sum for every residual and coordinate.
            let high = (origin + offset) >> equality.low_bits;
            let factor = match &equality.eq_high {
                Some(table) => table.get(high).copied().unwrap_or_else(F::zero),
                None => eq_eval_at_index(&equality.high_challenges, high),
            };
            high_sum += if weight == F::one() {
                factor
            } else {
                weight * factor
            };
        }
        Ok(Some(Self {
            origin,
            low_origin,
            high_sum,
        }))
    }

    pub(super) fn evaluate(
        &self,
        equality: &OffsetEqWindow<F>,
        base: usize,
    ) -> Result<F, AkitaError> {
        let index = base
            .checked_sub(self.origin)
            .and_then(|delta| checked::sum([self.low_origin, delta]))
            .ok_or_else(|| AkitaError::Internal("separated tensor low index overflow".into()))?;
        let low = equality.eq_low.get(index).copied().ok_or_else(|| {
            AkitaError::Internal("separated tensor low index is out of range".into())
        })?;
        Ok(low * self.high_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};
    use jolt_field::{Fp64, One, Ring, Zero};

    type F = Fp64<4294967197>;

    #[test]
    fn separated_residual_matches_boolean_equality_with_zero_and_one_challenges() {
        for challenges in [
            vec![F::from_u64(2); 6],
            vec![
                F::zero(),
                F::one(),
                F::from_u64(3),
                F::one(),
                F::zero(),
                F::from_u64(5),
            ],
        ] {
            let equality = OffsetEqWindow::with_low_bits(&challenges, 3).unwrap();
            let family =
                EqPairTensorFamily::new(0, 2, F::one(), vec![EqPairTensorAxis::unit(4, 1, 1)])
                    .unwrap();
            let residual = vec![
                (0, F::one()),
                (16, F::from_u64(3)),
                (32, -F::one()),
                (64, F::from_u64(7)),
            ];
            let view = DenseLeftTensorView {
                family: &family,
                destination_axis: 0,
                residual: ResidualTerms::Table(residual.clone()),
            };
            let prepared = SeparatedResidual::prepare(&view, &equality)
                .unwrap()
                .unwrap();
            for coordinate in 0..4 {
                let expected = residual
                    .iter()
                    .map(|&(offset, weight)| {
                        weight * eq_eval_at_index(&challenges, 2 + coordinate + offset)
                    })
                    .sum::<F>();
                assert_eq!(
                    prepared.evaluate(&equality, 2 + coordinate).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn separated_residual_declines_carry_mixed_residue_and_overflow() {
        let equality = OffsetEqWindow::with_low_bits(&[F::from_u64(2); 6], 3).unwrap();
        for (origin, len, offsets) in [
            (2, 8, [0, 16]),
            (0, 4, [0, 17]),
            (usize::MAX - 1, 4, [0, 16]),
        ] {
            let family = EqPairTensorFamily::new(
                0,
                origin,
                F::one(),
                vec![EqPairTensorAxis::unit(len, 1, 1)],
            )
            .unwrap();
            let view = DenseLeftTensorView {
                family: &family,
                destination_axis: 0,
                residual: ResidualTerms::Table(
                    offsets
                        .into_iter()
                        .map(|offset| (offset, F::one()))
                        .collect(),
                ),
            };
            assert!(SeparatedResidual::prepare(&view, &equality)
                .unwrap()
                .is_none());
        }
    }
}
