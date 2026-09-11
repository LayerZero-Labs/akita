//! Ring-level opening multipliers and terminal-functional preparation.

use super::subfield::{subfield_basis_pairs, SubfieldMultiplierOpeningPoint};
use crate::RingOpeningPoint;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

/// Ring-level opening point whose outer weights act by ring multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingMultiplierOpeningPoint<F: Field> {
    /// Degree-one openings, where multipliers are ordinary base scalars.
    Base(RingOpeningPoint<F>),
    /// Validated ring-subfield coordinates used by extension-valued openings.
    Subfield(SubfieldMultiplierOpeningPoint<F>),
}

/// Position multipliers prepared for contraction against an arbitrary
/// terminal ring functional.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PreparedRingMultiplierKind<E: Field> {
    Base(Vec<E>),
    Subfield {
        position_coordinates: Vec<E>,
        extension_degree: usize,
        ring_dim: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRingMultiplier<E: Field> {
    kind: PreparedRingMultiplierKind<E>,
}

impl<E: Field> PreparedRingMultiplier<E> {
    /// Evaluate one position multiplier against a complete terminal
    /// coefficient functional for `F[X]/(X^D + 1)`.
    pub fn evaluate_position_functional(
        &self,
        position: usize,
        functional: &[E],
    ) -> Result<E, AkitaError> {
        match &self.kind {
            PreparedRingMultiplierKind::Base(weights) => {
                let multiplier = *weights.get(position).ok_or(AkitaError::InvalidProof)?;
                let constant_weight = *functional.first().ok_or(AkitaError::InvalidProof)?;
                Ok(constant_weight * multiplier)
            }
            PreparedRingMultiplierKind::Subfield {
                position_coordinates,
                extension_degree,
                ring_dim,
            } => {
                if functional.len() != *ring_dim || *extension_degree == 0 {
                    return Err(AkitaError::InvalidProof);
                }
                let start = position
                    .checked_mul(*extension_degree)
                    .ok_or(AkitaError::InvalidProof)?;
                let end = start
                    .checked_add(*extension_degree)
                    .ok_or(AkitaError::InvalidProof)?;
                let coordinates = position_coordinates
                    .get(start..end)
                    .ok_or(AkitaError::InvalidProof)?;
                let (&constant, nonconstant) =
                    coordinates.split_first().ok_or(AkitaError::InvalidProof)?;
                let basis_pairs = subfield_basis_pairs(*ring_dim, *extension_degree)?;
                let mut evaluation = functional[0] * constant;
                for (&coordinate, &(basis_index, inverse_index)) in
                    nonconstant.iter().zip(&basis_pairs)
                {
                    let positive = *functional
                        .get(basis_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    let negative = *functional
                        .get(inverse_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    evaluation += (positive - negative) * coordinate;
                }
                Ok(evaluation)
            }
        }
    }
}

impl<F: Field> RingMultiplierOpeningPoint<F> {
    /// Lift compact multiplier coordinates once for terminal-functional evaluation.
    pub fn prepare_functional_multiplier<E>(&self) -> PreparedRingMultiplier<E>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Base(point) => PreparedRingMultiplier {
                kind: PreparedRingMultiplierKind::Base(
                    point
                        .position_weights
                        .iter()
                        .copied()
                        .map(E::lift_base)
                        .collect(),
                ),
            },
            Self::Subfield(point) => PreparedRingMultiplier {
                kind: PreparedRingMultiplierKind::Subfield {
                    position_coordinates: point
                        .position_coordinates_flat()
                        .iter()
                        .copied()
                        .map(E::lift_base)
                        .collect(),
                    extension_degree: point.extension_degree(),
                    ring_dim: point.ring_dim(),
                },
            },
        }
    }

    /// Keep base-field scalar weights in their compact scalar form.
    pub fn from_base(point: &RingOpeningPoint<F>) -> Self {
        Self::Base(point.clone())
    }

    /// Stored ring dimension for the [`Self::Subfield`] variant, or zero for [`Self::Base`].
    pub fn ring_dim(&self) -> usize {
        match self {
            Self::Base(_) => 0,
            Self::Subfield(point) => point.ring_dim(),
        }
    }

    /// Check that the stored multiplier uses ring dimension `D`.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        match self {
            Self::Base(_) => Ok(()),
            Self::Subfield(point) => point.ensure_ring_dim::<D>(),
        }
    }

    pub fn as_base(&self) -> Option<&RingOpeningPoint<F>> {
        match self {
            Self::Base(point) => Some(point),
            Self::Subfield(_) => None,
        }
    }

    pub fn as_subfield(&self) -> Option<&SubfieldMultiplierOpeningPoint<F>> {
        match self {
            Self::Base(_) => None,
            Self::Subfield(point) => Some(point),
        }
    }

    pub fn materialize_position_rings<const D: usize>(
        &self,
    ) -> Result<Option<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match self {
            Self::Base(_) => Ok(None),
            Self::Subfield(point) => point.materialize_position_rings::<D>().map(Some),
        }
    }

    pub fn materialize_fold_rings<const D: usize>(
        &self,
    ) -> Result<Option<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match self {
            Self::Base(_) => Ok(None),
            Self::Subfield(point) => point.materialize_fold_rings::<D>().map(Some),
        }
    }

    pub fn position_len(&self) -> usize {
        match self {
            Self::Base(point) => point.position_weights.len(),
            Self::Subfield(point) => point.position_len(),
        }
    }

    pub fn fold_len(&self) -> usize {
        match self {
            Self::Base(point) => point.live_block_weights.len(),
            Self::Subfield(point) => point.fold_len(),
        }
    }

    pub fn is_constant(&self) -> bool {
        match self {
            Self::Base(_) => true,
            Self::Subfield(point) => point.is_constant(),
        }
    }

    pub fn eval_position_at<E>(&self, idx: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Base(point) => point
                .position_weights
                .get(idx)
                .copied()
                .map(E::lift_base)
                .ok_or(AkitaError::InvalidProof),
            Self::Subfield(point) => point.eval_position_at(idx, alpha_pows),
        }
    }

    pub fn fold_subfield_value<E>(&self, idx: usize) -> Result<Option<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Base(_) => Ok(None),
            Self::Subfield(point) => point.fold_subfield_value(idx).map(Some),
        }
    }

    pub fn accumulate_position_product<const D: usize>(
        &self,
        idx: usize,
        rhs: &CyclotomicRing<F, D>,
        output: &mut CyclotomicRing<F, D>,
    ) -> Result<(), AkitaError> {
        match self {
            Self::Base(point) => {
                let scalar = point
                    .position_weights
                    .get(idx)
                    .ok_or(AkitaError::InvalidProof)?;
                *output += rhs.scale(scalar);
                Ok(())
            }
            Self::Subfield(point) => point.accumulate_position_product(idx, rhs, output),
        }
    }

    pub fn accumulate_position_product_high_half<const D: usize>(
        &self,
        idx: usize,
        rhs: &CyclotomicRing<F, D>,
        output: &mut [F],
    ) -> Result<(), AkitaError> {
        if output.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: output.len(),
            });
        }
        match self {
            Self::Base(point) => point
                .position_weights
                .get(idx)
                .map(|_| ())
                .ok_or(AkitaError::InvalidProof),
            Self::Subfield(point) => point.accumulate_position_product_high_half(idx, rhs, output),
        }
    }

    pub fn fold_constant_coeff(&self, idx: usize) -> Option<F> {
        match self {
            Self::Base(point) => point.live_block_weights.get(idx).copied(),
            Self::Subfield(point) => point.fold_constant_coeff(idx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fp32, FpExt4, One, Ring, Zero};

    type F = Fp32<251>;
    type E = FpExt4<F>;

    #[test]
    fn prepared_subfield_multiplier_uses_full_terminal_functional() {
        let prepared = PreparedRingMultiplier {
            kind: PreparedRingMultiplierKind::Subfield {
                position_coordinates: vec![E::zero(), E::one()],
                extension_degree: 2,
                ring_dim: 4,
            },
        };
        let functional = [
            E::from_u64(2),
            E::from_u64(3),
            E::from_u64(5),
            E::from_u64(7),
        ];
        assert_eq!(
            prepared
                .evaluate_position_functional(0, &functional)
                .unwrap(),
            functional[1] - functional[3]
        );
    }
}
