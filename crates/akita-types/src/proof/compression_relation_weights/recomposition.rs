//! Keep negative-binary recomposition as two weighted affine axes until a
//! prover requests physical entries. Verifiers contract the axes directly.

use super::CompressionRelationEvent;
use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};
use akita_error::{checked, AkitaError};
use jolt_field::Field;

/// Weight at `start + bit*source_coefficients + row*ring_dimension + coefficient`:
/// `-row_weights[row] * 2^bit * alpha^coefficient`.
#[derive(Clone, Debug)]
pub(super) struct Recomposition<E: Field> {
    pub(super) start: usize,
    pub(super) source_coefficients: usize,
    pub(super) ring_dimension: usize,
    pub(super) digit_ring_dimension: usize,
    pub(super) row_weights: Vec<E>,
    pub(super) bit_axis: EqPairTensorAxis<E>,
}

impl<E: Field> Recomposition<E> {
    pub(super) fn append_families(
        &self,
        alpha_powers: &[E],
        families: &mut Vec<EqPairTensorFamily<E>>,
    ) -> Result<(), AkitaError> {
        let factors = (0..self.ring_dimension.trailing_zeros() as usize)
            .map(|bit| {
                let power = checked::pow2(bit)
                    .and_then(|index| alpha_powers.get(index))
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                Ok([E::one(), power])
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let coefficient_axis = EqPairTensorAxis::bit_product(1, 0, factors)?;
        families
            .try_reserve(self.row_weights.len())
            .map_err(|_| AkitaError::InvalidInput("compression tensor allocation failed".into()))?;
        for (row, &weight) in self.row_weights.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let base = row
                .checked_mul(self.ring_dimension)
                .and_then(|offset| self.start.checked_add(offset))
                .ok_or(AkitaError::InvalidProof)?;
            families.push(EqPairTensorFamily::new(
                base,
                0,
                -weight,
                vec![coefficient_axis.clone(), self.bit_axis.clone()],
            )?);
        }
        Ok(())
    }

    pub(super) fn for_each_event(
        &self,
        mut emit: impl FnMut(CompressionRelationEvent<E>) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for bit in 0..self.bit_axis.len {
            let power = self
                .bit_axis
                .coordinate_weight(bit)
                .ok_or(AkitaError::InvalidProof)?;
            for (row, &weight) in self.row_weights.iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                let row_start = bit
                    .checked_mul(self.source_coefficients)
                    .and_then(|offset| self.start.checked_add(offset))
                    .and_then(|offset| {
                        row.checked_mul(self.ring_dimension)
                            .and_then(|row_offset| offset.checked_add(row_offset))
                    })
                    .ok_or(AkitaError::InvalidProof)?;
                let scalar = -(weight * power);
                for coefficient in (0..self.ring_dimension).step_by(self.digit_ring_dimension) {
                    emit(CompressionRelationEvent {
                        physical_start: row_start
                            .checked_add(coefficient)
                            .ok_or(AkitaError::InvalidProof)?,
                        coefficient_count: self.digit_ring_dimension,
                        alpha_exponent_start: coefficient,
                        scalar,
                    })?;
                }
            }
        }
        Ok(())
    }
}
