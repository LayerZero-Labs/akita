//! Exact terminal coefficient functionals for reduced ring relations.

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_algebra::ring::terminal_residue_kernel;
use akita_error::{checked, AkitaError};
use jolt_field::Field;
use std::sync::Arc;

/// Checked terminal residue functional for one physical native window.
///
/// The weights already include the exact multilinear equality contraction for
/// the physical window. Callers must therefore use them as the complete native
/// coefficient functional; there is no additional common-alpha factor.
#[derive(Clone, Debug)]
pub struct ReducedCoefficientFunctional<E: Field> {
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
        checked::range(physical_start, ring_dimension)
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
            weights: terminal_residue_kernel(&equality_weights, alpha)?.into(),
        })
    }

    #[must_use]
    pub fn weights(&self) -> &[E] {
        &self.weights
    }

    #[must_use]
    pub fn into_weights(self) -> Arc<[E]> {
        self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use akita_algebra::ring::eval_flat_ring_at_pows_fast;
    use jolt_field::{ExtField, Fp32, FpExt2, NegOneNr, One, Prime128OffsetA7F7 as F, Ring, Zero};

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
                eval_flat_ring_at_pows_fast(&dense, functional.weights()),
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
                eval_flat_ring_at_pows_fast(&sparse, functional.weights()),
                quadratic_reduced_evaluation(&sparse, alpha, &point, physical_start)
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
        let short_equality = OffsetEqWindow::new(&point[..4]).unwrap();
        assert!(
            ReducedCoefficientFunctional::prepare(&short_equality, 32, 3, 8, F::one()).is_err()
        );
        assert!(ReducedCoefficientFunctional::prepare(&equality, 32, 27, 8, F::one()).is_err());
        assert!(ReducedCoefficientFunctional::prepare(&equality, 32, 0, 6, F::one()).is_err());
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
            eval_flat_ring_at_pows_fast(&multiplier, functional.weights()),
            expected
        );
    }
}
