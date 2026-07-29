//! Shared batched Stage-3 point geometry.
//!
//! This module is the single source of truth for projecting the batched
//! Stage-3 challenge into witness and setup points.

use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

/// Geometry for one batched Stage-3 setup-product plus carried-witness sumcheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedStage3Geometry {
    witness_rounds: usize,
    setup_rounds: usize,
    batched_rounds: usize,
}

impl BatchedStage3Geometry {
    /// Build the shared Stage-3 geometry.
    ///
    /// `batched_rounds` is the common padded cube dimension. Native witness and
    /// setup coordinates occupy the suffix of the batched challenge vector.
    pub fn new(witness_rounds: usize, setup_rounds: usize) -> Result<Self, AkitaError> {
        if witness_rounds == 0 || setup_rounds == 0 {
            return Err(AkitaError::InvalidSetup(
                "batched stage-3 native round counts must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            witness_rounds,
            setup_rounds,
            batched_rounds: witness_rounds.max(setup_rounds),
        })
    }

    /// Native witness round count.
    #[must_use]
    pub fn witness_rounds(&self) -> usize {
        self.witness_rounds
    }

    /// Native setup round count.
    #[must_use]
    pub fn setup_rounds(&self) -> usize {
        self.setup_rounds
    }

    /// Common padded Stage-3 round count.
    #[must_use]
    pub fn batched_rounds(&self) -> usize {
        self.batched_rounds
    }

    /// Project the batched challenge onto the native witness point.
    pub fn witness_point<E: Clone>(&self, rho: &[E]) -> Result<Vec<E>, AkitaError> {
        self.project_native_point(rho, self.witness_rounds)
    }

    /// Project the batched challenge onto the native setup point.
    pub fn setup_point<E: Clone>(&self, rho: &[E]) -> Result<Vec<E>, AkitaError> {
        self.project_native_point(rho, self.setup_rounds)
    }

    /// Split `rho_setup` into ring-coordinate `rho_y` and setup-index tail.
    pub fn setup_y_and_index<'a, E>(
        &self,
        rho_setup: &'a [E],
        ring_bits: usize,
    ) -> Result<(&'a [E], &'a [E]), AkitaError> {
        if rho_setup.len() != self.setup_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.setup_rounds,
                actual: rho_setup.len(),
            });
        }
        if ring_bits > rho_setup.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: rho_setup.len(),
                actual: ring_bits,
            });
        }
        Ok(rho_setup.split_at(ring_bits))
    }

    /// Lifting scale for the witness term embedded into the common cube.
    pub fn witness_lift_scale<E: FieldCore + FromPrimitiveInt>(&self) -> Result<E, AkitaError> {
        lift_scale(self.batched_rounds - self.witness_rounds)
    }

    /// Lifting scale for the setup term embedded into the common cube.
    pub fn setup_lift_scale<E: FieldCore + FromPrimitiveInt>(&self) -> Result<E, AkitaError> {
        lift_scale(self.batched_rounds - self.setup_rounds)
    }

    fn project_native_point<E: Clone>(
        &self,
        rho: &[E],
        native_rounds: usize,
    ) -> Result<Vec<E>, AkitaError> {
        if rho.len() != self.batched_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.batched_rounds,
                actual: rho.len(),
            });
        }
        Ok(rho[self.batched_rounds - native_rounds..].to_vec())
    }
}

fn lift_scale<E: FieldCore + FromPrimitiveInt>(extra_rounds: usize) -> Result<E, AkitaError> {
    let inv_two = E::from_u64(2)
        .inverse()
        .ok_or_else(|| AkitaError::InvalidSetup("two is not invertible in Akita fields".into()))?;
    Ok((0..extra_rounds).fold(E::one(), |acc, _| acc * inv_two))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime32Offset99 as F;

    #[test]
    fn projects_suffix_points_for_unequal_domains() {
        let geometry = BatchedStage3Geometry::new(3, 5).expect("geometry");
        let rho = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
        ];
        assert_eq!(
            geometry.witness_point(&rho).expect("witness"),
            vec![F::from_u64(3), F::from_u64(4), F::from_u64(5)]
        );
        assert_eq!(geometry.setup_point(&rho).expect("setup"), rho);
    }

    #[test]
    fn computes_lift_scales() {
        let geometry = BatchedStage3Geometry::new(3, 5).expect("geometry");
        let inv_four = F::from_u64(4).inverse().expect("inverse");
        assert_eq!(
            geometry.witness_lift_scale::<F>().expect("witness"),
            inv_four
        );
        assert_eq!(geometry.setup_lift_scale::<F>().expect("setup"), F::one());
    }
}
