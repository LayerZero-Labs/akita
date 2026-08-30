//! Exact terminal coefficient functionals for reduced ring relations.

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, terminal_residue_kernel};
use akita_error::{checked, AkitaError};
use jolt_field::{ExtField, Field, MulBaseUnreduced};
use std::{ops::Range, sync::Arc};

/// Checked terminal residue functional for one physical native window.
///
/// The weights already include the exact multilinear equality contraction for
/// `physical_range`. Callers must therefore use them as the complete native
/// coefficient functional; there is no additional common-alpha factor.
#[derive(Clone, Debug)]
pub struct ReducedCoefficientFunctional<E: Field> {
    physical_range: Range<usize>,
    weights: Arc<[E]>,
}

impl<E: Field> ReducedCoefficientFunctional<E> {
    /// Prepare the terminal residue kernel for an exact physical window.
    pub fn prepare(
        equality: &OffsetEqWindow<E>,
        physical_field_len: usize,
        physical_start: usize,
        ring_dimension: usize,
        alpha: E,
    ) -> Result<Self, AkitaError> {
        if !physical_field_len.is_power_of_two()
            || !ring_dimension.is_power_of_two()
            || ring_dimension == 0
        {
            return Err(AkitaError::InvalidSetup(
                "reduced coefficient functional requires power-of-two domains".into(),
            ));
        }
        let expected_variables = physical_field_len.trailing_zeros() as usize;
        if equality.variable_count() != expected_variables {
            return Err(AkitaError::InvalidSize {
                expected: expected_variables,
                actual: equality.variable_count(),
            });
        }
        let physical_range = checked::range(physical_start, ring_dimension)
            .filter(|range| range.end <= physical_field_len)
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "reduced coefficient window exceeds the physical relation domain".into(),
                )
            })?;
        let mut equality_weights = Vec::new();
        equality_weights
            .try_reserve_exact(ring_dimension)
            .map_err(|_| AkitaError::InvalidInput("coefficient window is too large".into()))?;
        equality_weights.resize(ring_dimension, E::zero());
        equality.fill_interval(physical_start, &mut equality_weights)?;
        Ok(Self {
            physical_range,
            weights: terminal_residue_kernel(&equality_weights, alpha)?.into(),
        })
    }

    /// Evaluate one public native multiplier against this functional.
    pub fn evaluate_multiplier<F>(&self, coefficients: &[F]) -> Result<E, AkitaError>
    where
        F: Field,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        if coefficients.len() != self.weights.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.weights.len(),
                actual: coefficients.len(),
            });
        }
        Ok(eval_flat_ring_at_pows_fast(coefficients, &self.weights))
    }

    /// Evaluate a canonical sparse public multiplier in `O(h)` work after the
    /// shared terminal kernel has been prepared.
    pub fn evaluate_sparse_multiplier<F>(
        &self,
        coefficients: &[(usize, F)],
    ) -> Result<E, AkitaError>
    where
        F: Field,
        E: ExtField<F>,
    {
        let mut previous = None;
        coefficients
            .iter()
            .try_fold(E::zero(), |evaluation, &(index, coefficient)| {
                if previous.is_some_and(|prior| index <= prior) {
                    return Err(AkitaError::InvalidInput(
                        "sparse multiplier positions must be strictly increasing".into(),
                    ));
                }
                previous = Some(index);
                let weight = self
                    .weights
                    .get(index)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(evaluation + weight.mul_base(coefficient))
            })
    }

    #[must_use]
    pub fn physical_range(&self) -> Range<usize> {
        self.physical_range.clone()
    }

    #[must_use]
    pub fn weights(&self) -> &[E] {
        &self.weights
    }

    pub(crate) fn into_weights(self) -> Arc<[E]> {
        self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use jolt_field::{Fp32, FpExt2, NegOneNr, One, Prime128OffsetA7F7 as F, Ring, Zero};

    fn quadratic_reduced_evaluation(
        multiplier: &[F],
        alpha: F,
        point: &[F],
        physical_start: usize,
    ) -> F {
        let dimension = multiplier.len();
        let powers = akita_algebra::ring::scalar_powers(alpha, dimension);
        (0..dimension).fold(F::zero(), |sum, witness_coefficient| {
            let residue_weight = multiplier.iter().enumerate().fold(
                F::zero(),
                |weight, (multiplier_coefficient, &value)| {
                    let exponent = multiplier_coefficient + witness_coefficient;
                    if exponent < dimension {
                        weight + value * powers[exponent]
                    } else {
                        weight - value * powers[exponent - dimension]
                    }
                },
            );
            sum + eq_eval_at_index(point, physical_start + witness_coefficient) * residue_weight
        })
    }

    #[test]
    fn unaligned_dense_and_sparse_multipliers_match_quadratic_oracle() {
        let point = (0..6)
            .map(|index| F::from_u64(101 + index as u64))
            .collect::<Vec<_>>();
        let equality = OffsetEqWindow::new(&point).unwrap();
        let alpha = F::from_u64(17);
        for physical_start in [0, 7, 19, 43] {
            let functional =
                ReducedCoefficientFunctional::prepare(&equality, 64, physical_start, 8, alpha)
                    .unwrap();
            let dense = (0..8)
                .map(|index| F::from_u64(211 + 3 * index as u64))
                .collect::<Vec<_>>();
            assert_eq!(
                functional.evaluate_multiplier(&dense).unwrap(),
                quadratic_reduced_evaluation(&dense, alpha, &point, physical_start)
            );

            let sparse = (0..8)
                .map(|index| {
                    if [1, 6].contains(&index) {
                        F::from_u64(307 + index as u64)
                    } else {
                        F::zero()
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                functional.evaluate_multiplier(&sparse).unwrap(),
                quadratic_reduced_evaluation(&sparse, alpha, &point, physical_start)
            );
            let sparse_entries = [(1, sparse[1]), (6, sparse[6])];
            assert_eq!(
                functional
                    .evaluate_sparse_multiplier(&sparse_entries)
                    .unwrap(),
                functional.evaluate_multiplier(&sparse).unwrap()
            );
        }
    }

    #[test]
    fn exact_window_identity_and_malformed_inputs_are_checked() {
        let point = vec![F::from_u64(7); 5];
        let equality = OffsetEqWindow::new(&point).unwrap();
        let first =
            ReducedCoefficientFunctional::prepare(&equality, 32, 3, 8, F::from_u64(11)).unwrap();
        let second =
            ReducedCoefficientFunctional::prepare(&equality, 32, 11, 8, F::from_u64(11)).unwrap();
        assert_ne!(first.weights(), second.weights());
        assert_eq!(first.physical_range(), 3..11);
        let short_equality = OffsetEqWindow::new(&point[..4]).unwrap();
        assert!(
            ReducedCoefficientFunctional::prepare(&short_equality, 32, 3, 8, F::one()).is_err()
        );
        assert!(ReducedCoefficientFunctional::prepare(&equality, 32, 27, 8, F::one()).is_err());
        assert!(ReducedCoefficientFunctional::prepare(&equality, 32, 0, 6, F::one()).is_err());
        assert!(first.evaluate_multiplier::<F>(&[F::one(); 4]).is_err());
        assert!(first
            .evaluate_sparse_multiplier(&[(2, F::one()), (2, F::one())])
            .is_err());
        assert!(first.evaluate_sparse_multiplier(&[(8, F::one())]).is_err());
    }

    #[test]
    fn base_multiplier_matches_literal_oracle_at_genuine_extension_point() {
        type B = Fp32<251>;
        type X = FpExt2<B, NegOneNr>;
        let extension = |lo, hi| X::from_base_slice(&[B::from_u64(lo), B::from_u64(hi)]);
        let point = [
            extension(3, 5),
            extension(7, 11),
            extension(13, 17),
            extension(19, 23),
        ];
        let equality = OffsetEqWindow::new(&point).unwrap();
        let alpha = extension(29, 31);
        let multiplier = (0..8)
            .map(|index| B::from_u64(37 + index as u64))
            .collect::<Vec<_>>();
        let functional = ReducedCoefficientFunctional::prepare(&equality, 16, 5, 8, alpha).unwrap();
        let powers = akita_algebra::ring::scalar_powers(alpha, 8);
        let expected = (0..8).fold(X::zero(), |evaluation, witness_coefficient| {
            let residue = multiplier.iter().enumerate().fold(
                X::zero(),
                |sum, (multiplier_coefficient, &coefficient)| {
                    let exponent = multiplier_coefficient + witness_coefficient;
                    let product = powers[exponent % 8].mul_base(coefficient);
                    if exponent < 8 {
                        sum + product
                    } else {
                        sum - product
                    }
                },
            );
            evaluation + eq_eval_at_index(&point, 5 + witness_coefficient) * residue
        });
        assert_eq!(
            functional.evaluate_multiplier(&multiplier).unwrap(),
            expected
        );
    }
}
