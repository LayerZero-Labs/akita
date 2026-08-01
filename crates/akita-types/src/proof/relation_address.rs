//! Checked coefficient geometry for the compact relation witness.

use akita_field::AkitaError;

use super::stage1::FlatBooleanDomain;
use crate::layout::{
    validate_role_dims, witness_commitment_domain_len, CommitmentRingDims, RingRole,
};

/// Checked address geometry for one compact relation witness.
///
/// The exact live prefix contains native Z/E/T segments and native quotient
/// rows. The committed domain adds one zero suffix that aligns the prefix to a
/// power-of-two number of successor-ring elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationAddressGeometry {
    role_dims: CommitmentRingDims,
    digit_witness_domain: FlatBooleanDomain,
    relation_coefficient_block_len: usize,
    outgoing_witness_ring_dimension: usize,
    live_witness_coeff_len: usize,
    committed_witness_coeff_len: usize,
}

impl RelationAddressGeometry {
    /// Validate scalar compact relation geometry.
    pub fn new(
        role_dims: CommitmentRingDims,
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_for_groups(
            role_dims,
            &[],
            outgoing_witness_ring_dimension,
            live_witness_coeff_len,
        )
    }

    /// Validate compact geometry across the final and precommitted groups.
    pub fn new_for_groups(
        role_dims: CommitmentRingDims,
        group_role_dims: &[CommitmentRingDims],
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<Self, AkitaError> {
        validate_role_dims(role_dims)?;
        for &dims in group_role_dims {
            validate_role_dims(dims)?;
        }
        if outgoing_witness_ring_dimension == 0
            || !outgoing_witness_ring_dimension.is_power_of_two()
            || live_witness_coeff_len == 0
        {
            return Err(AkitaError::InvalidSetup(
                "relation witness requires nonzero power-of-two successor geometry".into(),
            ));
        }

        let relation_coefficient_block_len = group_role_dims
            .iter()
            .fold(role_dims.common_relation_coeff_count(), |common, dims| {
                common.min(dims.common_relation_coeff_count())
            });
        if relation_coefficient_block_len == 0
            || !relation_coefficient_block_len.is_power_of_two()
            || !all_dims_divisible(role_dims, relation_coefficient_block_len)
            || group_role_dims
                .iter()
                .any(|&dims| !all_dims_divisible(dims, relation_coefficient_block_len))
            || !live_witness_coeff_len.is_multiple_of(relation_coefficient_block_len)
        {
            return Err(AkitaError::InvalidSetup(
                "relation witness does not admit one aligned coefficient block".into(),
            ));
        }

        let committed_witness_coeff_len =
            witness_commitment_domain_len(live_witness_coeff_len, outgoing_witness_ring_dimension)?;
        if !committed_witness_coeff_len.is_power_of_two()
            || !committed_witness_coeff_len.is_multiple_of(relation_coefficient_block_len)
        {
            return Err(AkitaError::InvalidSetup(
                "relation witness domain is not aligned to its common coefficient block".into(),
            ));
        }
        let digit_witness_domain = FlatBooleanDomain::new(
            live_witness_coeff_len,
            committed_witness_coeff_len.trailing_zeros() as usize,
        )?;

        Ok(Self {
            role_dims,
            digit_witness_domain,
            relation_coefficient_block_len,
            outgoing_witness_ring_dimension,
            live_witness_coeff_len,
            committed_witness_coeff_len,
        })
    }

    #[must_use]
    pub const fn role_dims(self) -> CommitmentRingDims {
        self.role_dims
    }

    #[must_use]
    pub const fn digit_witness_domain(self) -> FlatBooleanDomain {
        self.digit_witness_domain
    }

    #[must_use]
    pub const fn relation_coefficient_block_len(self) -> usize {
        self.relation_coefficient_block_len
    }

    #[must_use]
    pub const fn outgoing_witness_ring_dimension(self) -> usize {
        self.outgoing_witness_ring_dimension
    }

    #[must_use]
    pub const fn live_witness_coeff_len(self) -> usize {
        self.live_witness_coeff_len
    }

    #[must_use]
    pub const fn committed_witness_coeff_len(self) -> usize {
        self.committed_witness_coeff_len
    }

    #[must_use]
    pub const fn successor_live_ring_len(self) -> usize {
        self.live_witness_coeff_len
            .div_ceil(self.outgoing_witness_ring_dimension)
    }

    #[must_use]
    pub const fn relation_coefficient_variable_count(self) -> usize {
        self.relation_coefficient_block_len.trailing_zeros() as usize
    }

    #[must_use]
    pub const fn role_relation_lane_count(self, role: RingRole) -> usize {
        let role_dim = match role {
            RingRole::Inner => self.role_dims.d_a(),
            RingRole::Outer => self.role_dims.d_b(),
            RingRole::Opening => self.role_dims.d_d(),
        };
        role_dim / self.relation_coefficient_block_len
    }

    #[must_use]
    pub const fn live_relation_lane_count(self) -> usize {
        self.live_witness_coeff_len / self.relation_coefficient_block_len
    }

    #[must_use]
    pub const fn relation_lane_capacity(self) -> usize {
        self.committed_witness_coeff_len / self.relation_coefficient_block_len
    }

    #[must_use]
    pub const fn relation_lane_variable_count(self) -> usize {
        self.relation_lane_capacity().trailing_zeros() as usize
    }

    #[must_use]
    pub fn relation_point_variable_count(self) -> usize {
        self.digit_witness_domain.num_vars()
    }

    pub fn validate_relation_point_len(self, actual: usize) -> Result<(), AkitaError> {
        let expected = self.relation_point_variable_count();
        if actual != expected {
            return Err(AkitaError::InvalidSize { expected, actual });
        }
        Ok(())
    }
}

/// Independent compact address geometry for F/H compression rows.
///
/// Compression dimensions never reduce the coefficient block used by the
/// existing A/B/D relation roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionRelationAddressGeometry {
    digit_witness_domain: FlatBooleanDomain,
    coefficient_block_len: usize,
    live_witness_coeff_len: usize,
    committed_witness_coeff_len: usize,
}

impl CompressionRelationAddressGeometry {
    /// Derive compact geometry from the native dimensions of only F/H rows.
    pub fn new(
        compression_row_ring_dims: &[usize],
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<Self, AkitaError> {
        let coefficient_block_len = compression_row_ring_dims
            .iter()
            .copied()
            .try_fold(None, |minimum, ring_dim| {
                if ring_dim == 0 || !ring_dim.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(
                        "compression relation row has a malformed ring dimension".into(),
                    ));
                }
                Ok(Some(
                    minimum.map_or(ring_dim, |value: usize| value.min(ring_dim)),
                ))
            })?
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression relation requires F/H rows".into())
            })?;
        if outgoing_witness_ring_dimension == 0
            || !outgoing_witness_ring_dimension.is_power_of_two()
            || live_witness_coeff_len == 0
            || !live_witness_coeff_len.is_multiple_of(coefficient_block_len)
        {
            return Err(AkitaError::InvalidSetup(
                "compression relation witness geometry is malformed".into(),
            ));
        }
        let committed_witness_coeff_len =
            witness_commitment_domain_len(live_witness_coeff_len, outgoing_witness_ring_dimension)?;
        if !committed_witness_coeff_len.is_power_of_two()
            || !committed_witness_coeff_len.is_multiple_of(coefficient_block_len)
        {
            return Err(AkitaError::InvalidSetup(
                "compression relation domain is not coefficient aligned".into(),
            ));
        }
        let digit_witness_domain = FlatBooleanDomain::new(
            live_witness_coeff_len,
            committed_witness_coeff_len.trailing_zeros() as usize,
        )?;
        Ok(Self {
            digit_witness_domain,
            coefficient_block_len,
            live_witness_coeff_len,
            committed_witness_coeff_len,
        })
    }

    #[must_use]
    pub const fn digit_witness_domain(self) -> FlatBooleanDomain {
        self.digit_witness_domain
    }

    #[must_use]
    pub const fn coefficient_block_len(self) -> usize {
        self.coefficient_block_len
    }

    #[must_use]
    pub const fn live_lane_count(self) -> usize {
        self.live_witness_coeff_len / self.coefficient_block_len
    }

    #[must_use]
    pub const fn lane_capacity(self) -> usize {
        self.committed_witness_coeff_len / self.coefficient_block_len
    }

    #[must_use]
    pub const fn coefficient_variable_count(self) -> usize {
        self.coefficient_block_len.trailing_zeros() as usize
    }
}

const fn all_dims_divisible(dims: CommitmentRingDims, block: usize) -> bool {
    dims.d_a().is_multiple_of(block)
        && dims.d_b().is_multiple_of(block)
        && dims.d_d().is_multiple_of(block)
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
        let live_coeff_len = 1024;
        let geometries = [16, 32, 64].map(|outgoing_dim| {
            RelationAddressGeometry::new(dims, outgoing_dim, live_coeff_len).unwrap()
        });
        for geometry in geometries {
            assert_eq!(geometry.digit_witness_domain().live_len(), live_coeff_len);
            assert_eq!(geometry.digit_witness_domain().domain_len(), live_coeff_len);
            assert_eq!(geometry.relation_coefficient_block_len(), 32);
            assert_eq!(geometry.live_relation_lane_count(), 32);
            assert_eq!(geometry.relation_lane_capacity(), 32);
            assert_eq!(geometry.relation_point_variable_count(), 10);
        }
    }

    #[test]
    fn supports_mixed_groups_without_max_a_padding() {
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
            RelationAddressGeometry::new_for_groups(final_dims, &[precommitted_dims], 64, 288)
                .unwrap();
        assert_eq!(geometry.live_witness_coeff_len(), 288);
        assert_eq!(geometry.committed_witness_coeff_len(), 512);
        assert_eq!(geometry.successor_live_ring_len(), 5);
        assert_eq!(geometry.relation_coefficient_block_len(), 32);
        assert_eq!(geometry.live_relation_lane_count(), 9);
        assert_eq!(geometry.relation_lane_capacity(), 16);
    }

    #[test]
    fn compression_rows_use_an_independent_compact_coefficient_block() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let relation = RelationAddressGeometry::new(dims, 64, 1024).expect("relation geometry");
        let compression = CompressionRelationAddressGeometry::new(&[16, 8], 64, 1024)
            .expect("compression geometry");
        assert_eq!(relation.relation_coefficient_block_len(), 32);
        assert_eq!(relation.live_relation_lane_count(), 32);
        assert_eq!(compression.coefficient_block_len(), 8);
        assert_eq!(compression.live_lane_count(), 128);
    }

    #[test]
    fn successor_alignment_is_one_zero_suffix() {
        let geometry =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(64), 128, 64).unwrap();
        assert_eq!(geometry.live_witness_coeff_len(), 64);
        assert_eq!(geometry.successor_live_ring_len(), 1);
        assert_eq!(geometry.committed_witness_coeff_len(), 128);
        assert_eq!(geometry.digit_witness_domain().domain_len(), 128);
    }

    #[test]
    fn rejects_malformed_geometry() {
        assert!(RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 0, 128).is_err());
        assert!(RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 48, 128).is_err());
        assert!(RelationAddressGeometry::new(CommitmentRingDims::uniform(128), 64, 0).is_err());
        assert!(RelationAddressGeometry::new(
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 32,
            },
            64,
            80,
        )
        .is_err());
    }
}
