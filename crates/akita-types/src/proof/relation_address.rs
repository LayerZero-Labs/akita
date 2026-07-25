//! Checked address geometry for the flat relation witness.
//!
//! This module owns the storage-dependent split between the low coefficient
//! block shared by every relation role and the outgoing witness representation,
//! and the remaining relation-lane coordinates. It does not define the
//! relation algebra or Stage-3 setup projection geometry.

use akita_field::AkitaError;

use super::stage1::FlatBooleanDomain;
use crate::layout::{opening_domain_len, validate_role_dims, CommitmentRingDims, RingRole};

/// Checked address geometry for the flat relation witness.
///
/// The low coefficient block has width
/// `min(d_a, d_b, d_d, outgoing_witness_ring_dimension)`. The remaining
/// Boolean coordinates address relation lanes in the padded flat digit-witness
/// domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationAddressGeometry {
    role_dims: CommitmentRingDims,
    digit_witness_domain: FlatBooleanDomain,
    common_relation_witness_coeff_count: usize,
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
        validate_role_dims(role_dims)?;
        if outgoing_witness_ring_dimension == 0
            || !outgoing_witness_ring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "outgoing witness ring dimension must be a non-zero power of two".into(),
            ));
        }

        let common_relation_witness_coeff_count =
            role_dims.common_relation_witness_coeff_count(outgoing_witness_ring_dimension);
        if common_relation_witness_coeff_count == 0
            || !common_relation_witness_coeff_count.is_power_of_two()
            || !role_dims
                .d_a()
                .is_multiple_of(common_relation_witness_coeff_count)
            || !role_dims
                .d_b()
                .is_multiple_of(common_relation_witness_coeff_count)
            || !role_dims
                .d_d()
                .is_multiple_of(common_relation_witness_coeff_count)
            || !outgoing_witness_ring_dimension.is_multiple_of(common_relation_witness_coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "relation and outgoing witness do not admit a common coefficient block".into(),
            ));
        }

        let live_field_len = outgoing_witness_source_len
            .checked_mul(outgoing_witness_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("relation witness length overflow".into()))?;
        let padded_field_len = opening_domain_len(outgoing_witness_source_len)?
            .checked_mul(outgoing_witness_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("relation witness domain overflow".into()))?;
        if !padded_field_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation witness domain must be a non-zero power of two".into(),
            ));
        }
        let digit_witness_domain =
            FlatBooleanDomain::new(live_field_len, padded_field_len.trailing_zeros() as usize)?;

        Ok(Self {
            role_dims,
            digit_witness_domain,
            common_relation_witness_coeff_count,
        })
    }

    /// Per-role ring dimensions used by the relation.
    #[must_use]
    pub const fn role_dims(self) -> CommitmentRingDims {
        self.role_dims
    }

    /// Complete checked flat digit-witness domain.
    #[must_use]
    pub const fn digit_witness_domain(self) -> FlatBooleanDomain {
        self.digit_witness_domain
    }

    /// Low coefficient block shared by every role and the outgoing witness.
    #[must_use]
    pub const fn common_relation_witness_coeff_count(self) -> usize {
        self.common_relation_witness_coeff_count
    }

    /// Boolean-coordinate count for the common coefficient block.
    #[must_use]
    pub const fn common_relation_witness_variable_count(self) -> usize {
        self.common_relation_witness_coeff_count.trailing_zeros() as usize
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
        role_dim / self.common_relation_witness_coeff_count
    }

    /// Exact number of live relation lanes before Boolean-domain padding.
    #[must_use]
    pub fn live_relation_lane_count(self) -> usize {
        self.digit_witness_domain.live_len() / self.common_relation_witness_coeff_count
    }

    /// Padded number of relation lanes above the common coefficient block.
    #[must_use]
    pub fn relation_lane_capacity(self) -> usize {
        self.digit_witness_domain.domain_len() / self.common_relation_witness_coeff_count
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
    fn preserves_uniform_outgoing_fast_path() {
        let same_dimension =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 128, 9).unwrap();
        assert_eq!(same_dimension.common_relation_witness_coeff_count(), 128);
        assert_eq!(same_dimension.live_relation_lane_count(), 9);
        assert_eq!(same_dimension.relation_lane_capacity(), 16);
        assert_eq!(same_dimension.relation_point_variable_count(), 11);

        let smaller_outgoing =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 64, 9).unwrap();
        assert_eq!(smaller_outgoing.digit_witness_domain().live_len(), 576);
        assert_eq!(smaller_outgoing.digit_witness_domain().domain_len(), 1024);
        assert_eq!(smaller_outgoing.common_relation_witness_coeff_count(), 64);
        assert_eq!(smaller_outgoing.live_relation_lane_count(), 9);
        assert_eq!(smaller_outgoing.relation_lane_capacity(), 16);
        assert_eq!(smaller_outgoing.common_relation_witness_variable_count(), 6);
        assert_eq!(smaller_outgoing.relation_lane_variable_count(), 4);
        assert_eq!(smaller_outgoing.relation_point_variable_count(), 10);
    }

    #[test]
    fn supports_mixed_roles_at_the_outgoing_dimension() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let geometry = RelationAddressGeometry::new(dims, 64, 9).unwrap();
        assert_eq!(geometry.role_dims(), dims);
        assert_eq!(geometry.common_relation_witness_coeff_count(), 32);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 4);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Outer), 2);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Opening), 1);
        assert_eq!(geometry.live_relation_lane_count(), 18);
        assert_eq!(geometry.relation_lane_capacity(), 32);
        assert_eq!(geometry.common_relation_witness_variable_count(), 5);
        assert_eq!(geometry.relation_lane_variable_count(), 5);
        geometry.validate_relation_point_len(10).unwrap();
        assert!(matches!(
            geometry.validate_relation_point_len(9),
            Err(AkitaError::InvalidSize {
                expected: 10,
                actual: 9
            })
        ));
    }

    #[test]
    fn supports_mixed_roles_at_the_common_dimension() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let geometry = RelationAddressGeometry::new(dims, 32, 9).unwrap();
        assert_eq!(geometry.common_relation_witness_coeff_count(), 32);
        assert_eq!(geometry.live_relation_lane_count(), 9);
        assert_eq!(geometry.relation_lane_capacity(), 16);
        assert_eq!(geometry.relation_point_variable_count(), 9);
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
