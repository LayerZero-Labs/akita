//! Checked address geometry for the flat relation witness.
//!
//! This module owns the split between the low coefficient block shared by
//! every current relation role and the remaining relation-lane coordinates.
//! The outgoing witness representation determines only the exact flat live
//! length and its Boolean padding; it does not change that relation split.

use akita_field::AkitaError;

use super::stage1::FlatBooleanDomain;
use crate::layout::{opening_domain_len, validate_role_dims, CommitmentRingDims, RingRole};

/// Checked address geometry for the flat relation witness.
///
/// The low coefficient block is the largest block shared by every current
/// relation role. The remaining Boolean coordinates address relation lanes in
/// the padded flat digit-witness domain. Repacking the same flat witness into a
/// different outgoing ring dimension leaves this split unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationAddressGeometry {
    role_dims: CommitmentRingDims,
    carrier_ring_dimension: usize,
    digit_witness_domain: FlatBooleanDomain,
    relation_coefficient_block_len: usize,
    outgoing_witness_ring_dimension: usize,
    outgoing_witness_source_len: usize,
}

impl RelationAddressGeometry {
    /// Validate and construct the flat relation-witness address geometry.
    ///
    /// `outgoing_witness_source_len` is the exact number of outgoing ring
    /// elements before Boolean-domain padding.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if any role or outgoing ring
    /// dimension is malformed, no common coefficient block exists, or the
    /// flat witness domain overflows.
    pub fn new(
        role_dims: CommitmentRingDims,
        outgoing_witness_ring_dimension: usize,
        outgoing_witness_source_len: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_for_groups(
            role_dims,
            &[],
            outgoing_witness_ring_dimension,
            outgoing_witness_source_len,
        )
    }

    /// Construct geometry whose common coefficient block covers the final
    /// group and every precommitted group's current role dimensions.
    pub fn new_for_groups(
        role_dims: CommitmentRingDims,
        group_role_dims: &[CommitmentRingDims],
        outgoing_witness_ring_dimension: usize,
        outgoing_witness_source_len: usize,
    ) -> Result<Self, AkitaError> {
        validate_role_dims(role_dims)?;
        for &dims in group_role_dims {
            validate_role_dims(dims)?;
        }
        let carrier_ring_dimension = group_role_dims
            .iter()
            .map(|dims| dims.d_a())
            .chain(std::iter::once(role_dims.d_a()))
            .max()
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "relation geometry requires a witness carrier dimension".into(),
                )
            })?;
        if outgoing_witness_ring_dimension == 0
            || !outgoing_witness_ring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "outgoing witness ring dimension must be a non-zero power of two".into(),
            ));
        }

        let relation_coefficient_block_len = group_role_dims
            .iter()
            .fold(role_dims.common_relation_coeff_count(), |common, dims| {
                common.min(dims.common_relation_coeff_count())
            });
        if relation_coefficient_block_len == 0
            || !relation_coefficient_block_len.is_power_of_two()
            || !role_dims
                .d_a()
                .is_multiple_of(relation_coefficient_block_len)
            || !role_dims
                .d_b()
                .is_multiple_of(relation_coefficient_block_len)
            || !role_dims
                .d_d()
                .is_multiple_of(relation_coefficient_block_len)
            || group_role_dims.iter().any(|dims| {
                !dims.d_a().is_multiple_of(relation_coefficient_block_len)
                    || !dims.d_b().is_multiple_of(relation_coefficient_block_len)
                    || !dims.d_d().is_multiple_of(relation_coefficient_block_len)
            })
        {
            return Err(AkitaError::InvalidSetup(
                "current relation roles do not admit a common coefficient block".into(),
            ));
        }

        let live_field_len = outgoing_witness_source_len
            .checked_mul(carrier_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("relation witness length overflow".into()))?;
        let padded_field_len = opening_domain_len(outgoing_witness_source_len)?
            .checked_mul(carrier_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("relation witness domain overflow".into()))?;
        if !padded_field_len.is_power_of_two()
            || !live_field_len.is_multiple_of(relation_coefficient_block_len)
            || !padded_field_len.is_multiple_of(relation_coefficient_block_len)
        {
            return Err(AkitaError::InvalidSetup(
                "flat relation witness domain is not aligned to its common coefficient block"
                    .into(),
            ));
        }
        let digit_witness_domain =
            FlatBooleanDomain::new(live_field_len, padded_field_len.trailing_zeros() as usize)?;

        Ok(Self {
            role_dims,
            carrier_ring_dimension,
            digit_witness_domain,
            relation_coefficient_block_len,
            outgoing_witness_ring_dimension,
            outgoing_witness_source_len,
        })
    }

    /// Per-role ring dimensions used by the relation.
    #[must_use]
    pub const fn role_dims(self) -> CommitmentRingDims {
        self.role_dims
    }

    /// Batch-owned ring dimension used as the physical stride of every
    /// recursive-witness slot.
    ///
    /// This is the largest A-role dimension in the opening batch. It is
    /// independent of group order and of which group happens to be final.
    #[must_use]
    pub const fn carrier_ring_dimension(self) -> usize {
        self.carrier_ring_dimension
    }

    /// Complete checked flat digit-witness domain.
    #[must_use]
    pub const fn digit_witness_domain(self) -> FlatBooleanDomain {
        self.digit_witness_domain
    }

    /// Low coefficient block shared by every current relation role.
    #[must_use]
    pub const fn relation_coefficient_block_len(self) -> usize {
        self.relation_coefficient_block_len
    }

    /// Ring dimension used by the outgoing witness representation.
    #[must_use]
    pub const fn outgoing_witness_ring_dimension(self) -> usize {
        self.outgoing_witness_ring_dimension
    }

    /// Exact outgoing ring-element count before Boolean-domain padding.
    #[must_use]
    pub const fn outgoing_witness_source_len(self) -> usize {
        self.outgoing_witness_source_len
    }

    /// Boolean-coordinate count for the common coefficient block.
    #[must_use]
    pub const fn relation_coefficient_variable_count(self) -> usize {
        self.relation_coefficient_block_len.trailing_zeros() as usize
    }

    /// Number of common-coefficient relation lanes carried by one ring element
    /// of `role`.
    #[must_use]
    pub const fn role_relation_lane_count(self, role: RingRole) -> usize {
        let role_dim = match role {
            RingRole::Inner => self.role_dims.d_a(),
            RingRole::Outer => self.role_dims.d_b(),
            RingRole::Opening => self.role_dims.d_d(),
        };
        role_dim / self.relation_coefficient_block_len
    }

    /// Exact number of live relation lanes before Boolean-domain padding.
    #[must_use]
    pub fn live_relation_lane_count(self) -> usize {
        self.digit_witness_domain.live_len() / self.relation_coefficient_block_len
    }

    /// Padded number of relation lanes above the common coefficient block.
    #[must_use]
    pub fn relation_lane_capacity(self) -> usize {
        self.digit_witness_domain.domain_len() / self.relation_coefficient_block_len
    }

    /// Boolean-coordinate count above the common coefficient block.
    #[must_use]
    pub fn relation_lane_variable_count(self) -> usize {
        self.relation_lane_capacity().trailing_zeros() as usize
    }

    /// Total number of Boolean coordinates in the flat relation domain.
    #[must_use]
    pub fn relation_point_variable_count(self) -> usize {
        self.digit_witness_domain.num_vars()
    }

    /// Check that a multilinear point addresses the complete padded domain.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSize`] for a point of the wrong length.
    pub fn validate_relation_point_len(self, actual: usize) -> Result<(), AkitaError> {
        let expected = self.relation_point_variable_count();
        if actual != expected {
            return Err(AkitaError::InvalidSize { expected, actual });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outgoing_repacking_preserves_mixed_relation_geometry() {
        let dims = CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 32,
        };
        let flat_live_len = 1024;
        let geometries = [16, 32, 64].map(|outgoing_dim| {
            RelationAddressGeometry::new(dims, outgoing_dim, flat_live_len / dims.d_a()).unwrap()
        });
        for geometry in geometries {
            assert_eq!(geometry.digit_witness_domain().live_len(), flat_live_len);
            assert_eq!(geometry.digit_witness_domain().domain_len(), flat_live_len);
            assert_eq!(geometry.relation_coefficient_block_len(), 32);
            assert_eq!(geometry.relation_coefficient_variable_count(), 5);
            assert_eq!(geometry.live_relation_lane_count(), 32);
            assert_eq!(geometry.relation_lane_capacity(), 32);
            assert_eq!(geometry.relation_lane_variable_count(), 5);
            assert_eq!(geometry.relation_point_variable_count(), 10);
        }
        assert_eq!(
            geometries[0].digit_witness_domain(),
            geometries[1].digit_witness_domain()
        );
        assert_eq!(
            geometries[1].digit_witness_domain(),
            geometries[2].digit_witness_domain()
        );
    }

    #[test]
    fn supports_mixed_roles_at_the_role_common_dimension() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let geometry = RelationAddressGeometry::new(dims, 64, 9).unwrap();
        assert_eq!(geometry.role_dims(), dims);
        assert_eq!(geometry.relation_coefficient_block_len(), 32);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 4);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Outer), 2);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Opening), 1);
        assert_eq!(geometry.live_relation_lane_count(), 36);
        assert_eq!(geometry.relation_lane_capacity(), 64);
        assert_eq!(geometry.relation_coefficient_variable_count(), 5);
        assert_eq!(geometry.relation_lane_variable_count(), 6);
        geometry.validate_relation_point_len(11).unwrap();
        assert!(matches!(
            geometry.validate_relation_point_len(10),
            Err(AkitaError::InvalidSize {
                expected: 11,
                actual: 10
            })
        ));
    }

    #[test]
    fn carrier_uses_largest_group_a_not_final_group_a() {
        let final_dims = CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 32,
        };
        let precommitted_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let geometry =
            RelationAddressGeometry::new_for_groups(final_dims, &[precommitted_dims], 64, 9)
                .expect("larger precommitted A");
        assert_eq!(geometry.role_dims(), final_dims);
        assert_eq!(geometry.carrier_ring_dimension(), 128);
        assert_eq!(geometry.relation_coefficient_block_len(), 32);
    }

    #[test]
    fn rejects_malformed_outgoing_ring() {
        assert!(matches!(
            RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 0, 9),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 48, 9),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
