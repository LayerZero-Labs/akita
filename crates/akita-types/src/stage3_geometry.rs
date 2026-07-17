//! Shared batched Stage-3 point geometry.
//!
//! This module is the single source of truth for projecting the batched
//! Stage-3 challenge into witness/setup points and for routing those projected
//! points into the next recursive suffix opening batch.

use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

use crate::{PointVariableSelection, SetupPrefixSlotId};

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

    /// Merge Stage-3 setup/witness points into the shared point used by the
    /// next suffix opening batch.
    ///
    /// Returns `(shared_point, setup_offset)`, where `setup_offset` is the first
    /// coordinate of `setup_prefix_point` inside `shared_point`.
    pub fn shared_suffix_point<E: FieldCore>(
        setup_prefix_point: &[E],
        witness_point: &[E],
    ) -> Result<(Vec<E>, usize), AkitaError> {
        if setup_prefix_point.len() >= witness_point.len() {
            if &setup_prefix_point[setup_prefix_point.len() - witness_point.len()..]
                != witness_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((setup_prefix_point.to_vec(), 0))
        } else {
            if &witness_point[witness_point.len() - setup_prefix_point.len()..]
                != setup_prefix_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((
                witness_point.to_vec(),
                witness_point.len() - setup_prefix_point.len(),
            ))
        }
    }

    /// Coordinate routing for a setup-prefix group in the next suffix opening batch.
    pub fn setup_prefix_point_vars(
        setup_prefix_point_len: usize,
        setup_prefix_id: &SetupPrefixSlotId,
        offset: usize,
        shared_point_len: usize,
    ) -> Result<PointVariableSelection, AkitaError> {
        if setup_prefix_id.d_setup == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix d_setup must be nonzero".to_string(),
            ));
        }
        let ring_bits = setup_prefix_id.d_setup.trailing_zeros() as usize;
        let params = &setup_prefix_id.commitment_params;
        let position_index_bits = params.layout.num_positions_per_block.trailing_zeros() as usize;
        let block_index_bits = params
            .layout
            .num_live_blocks
            .checked_next_power_of_two()
            .map(|capacity| capacity.trailing_zeros() as usize)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix block-index domain size overflow".into())
            })?;
        let expected = ring_bits
            .checked_add(position_index_bits)
            .and_then(|n| n.checked_add(block_index_bits))
            .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix point length overflow".into()))?;
        if setup_prefix_point_len != expected {
            return Err(AkitaError::InvalidPointDimension {
                expected,
                actual: setup_prefix_point_len,
            });
        }
        let end = offset
            .checked_add(expected)
            .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix point range overflow".into()))?;
        let indices = (offset..end).collect();
        PointVariableSelection::new(indices, shared_point_len)
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
    use crate::{
        AjtaiKeyParams, PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
        SisModulusProfileId,
    };
    use akita_field::Prime32Offset99 as F;

    fn test_prefix_id() -> SetupPrefixSlotId {
        let a_key = AjtaiKeyParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            crate::sis::SisMatrixRole::A,
            1,
            1,
            1,
            32,
        );
        let b_key = AjtaiKeyParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            crate::sis::SisMatrixRole::B,
            1,
            1,
            1,
            32,
        );
        SetupPrefixSlotId {
            d_setup: 32,
            natural_len: 1,
            commitment_params: PrecommittedLevelParams {
                layout: PrecommittedGroupParams {
                    group: PolynomialGroupLayout::singleton(8),
                    num_live_ring_elements_per_claim: 8,
                    num_positions_per_block: 4,
                    num_live_blocks: 2,
                    fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
                    log_basis_inner: 1,
                    log_basis_outer: 3,
                    n_a: 1,
                    n_b: 1,
                },
                a_key,
                b_key,
                log_basis_open: 3,
                num_digits_inner: 1,
                num_digits_outer: 1,
                num_digits_open: 1,
                num_digits_fold_one: 1,
            },
        }
    }

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

    #[test]
    fn shared_suffix_point_accepts_suffix_consistency() {
        let setup = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(2), F::from_u64(3)];
        let (shared, offset) =
            BatchedStage3Geometry::shared_suffix_point(&setup, &witness).expect("shared");
        assert_eq!(shared, setup);
        assert_eq!(offset, 0);

        let setup = vec![F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let (shared, offset) =
            BatchedStage3Geometry::shared_suffix_point(&setup, &witness).expect("shared");
        assert_eq!(shared, witness);
        assert_eq!(offset, 1);
    }

    #[test]
    fn shared_suffix_point_rejects_inconsistent_suffix() {
        let setup = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(4), F::from_u64(3)];
        assert!(BatchedStage3Geometry::shared_suffix_point(&setup, &witness).is_err());
    }

    #[test]
    fn setup_prefix_point_vars_follow_exact_physical_order() {
        let id = test_prefix_id();
        let selection =
            BatchedStage3Geometry::setup_prefix_point_vars(8, &id, 1, 10).expect("selection");
        assert_eq!(selection.indices(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
