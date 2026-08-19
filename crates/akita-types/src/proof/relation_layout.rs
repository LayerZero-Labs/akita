//! Relation row identities and shared layout data.

use akita_field::AkitaError;

use crate::{CommitmentRingDims, CommitmentSliceCount, CompressionChainPlan, OpeningMethod};

/// Checked coefficient geometry of one logical relation row.
///
/// A row contains `coordinate_plane_count` polynomials modulo
/// `X^polynomial_modulus_dimension + 1`. The physical witness concatenates
/// those planes and therefore has `physical_coefficient_width` base-field
/// coordinates. Keeping these values distinct prevents an extension-coordinate
/// width from being mistaken for a polynomial modulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationRowGeometry {
    polynomial_modulus_dimension: usize,
    coordinate_plane_count: usize,
    physical_coefficient_width: usize,
}

impl RelationRowGeometry {
    /// Construct checked row geometry.
    pub fn new(
        polynomial_modulus_dimension: usize,
        coordinate_plane_count: usize,
    ) -> Result<Self, AkitaError> {
        if !polynomial_modulus_dimension.is_power_of_two()
            || !coordinate_plane_count.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "relation row modulus and coordinate-plane count must be nonzero powers of two"
                    .into(),
            ));
        }
        let physical_coefficient_width = polynomial_modulus_dimension
            .checked_mul(coordinate_plane_count)
            .ok_or_else(|| AkitaError::InvalidSetup("relation row width overflow".into()))?;
        Ok(Self {
            polynomial_modulus_dimension,
            coordinate_plane_count,
            physical_coefficient_width,
        })
    }

    /// Construct one ordinary native-ring row.
    pub fn native(ring_dimension: usize) -> Result<Self, AkitaError> {
        Self::new(ring_dimension, 1)
    }

    #[must_use]
    pub const fn polynomial_modulus_dimension(self) -> usize {
        self.polynomial_modulus_dimension
    }

    #[must_use]
    pub const fn coordinate_plane_count(self) -> usize {
        self.coordinate_plane_count
    }

    #[must_use]
    pub const fn physical_coefficient_width(self) -> usize {
        self.physical_coefficient_width
    }

    /// Flatten one `(coordinate plane, modulus coefficient)` pair.
    pub fn physical_coefficient_index(
        self,
        coordinate_plane: usize,
        modulus_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if coordinate_plane >= self.coordinate_plane_count
            || modulus_coefficient >= self.polynomial_modulus_dimension
        {
            return Err(AkitaError::InvalidInput(
                "relation row coefficient lies outside its geometry".into(),
            ));
        }
        coordinate_plane
            .checked_mul(self.polynomial_modulus_dimension)
            .and_then(|base| base.checked_add(modulus_coefficient))
            .filter(|&index| index < self.physical_coefficient_width)
            .ok_or_else(|| AkitaError::InvalidSetup("relation row index overflow".into()))
    }
}

/// Per-group row-count inputs for assembling the relation rhs vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationGroupRows {
    /// Opening-batch group index represented by this relation group.
    pub group_index: usize,
    /// This group's A/B dimensions completed by the level-shared D dimension.
    pub role_dims: CommitmentRingDims,
    /// Geometry of this group's one logical opening-consistency row and E data.
    pub opening_geometry: RelationRowGeometry,
    /// Algebraic procedure represented by the opening-consistency row.
    pub opening_method: OpeningMethod,
    pub n_a: usize,
    /// Rows in the one physical B matrix reused by every slice.
    pub physical_b_rows: usize,
    /// Number of logical B images stacked in canonical slice order.
    pub outer_slice_count: CommitmentSliceCount,
}

impl RelationGroupRows {
    /// Complete logical B row count for this relation group.
    pub fn logical_b_rows(&self) -> Result<usize, AkitaError> {
        self.outer_slice_count
            .logical_output_rows(self.physical_b_rows)
    }
}

/// Row-count inputs for assembling the relation rhs vector.
///
/// relation-matrix row order: `[final, precommitted_0, .., precommitted_{G-2}]`.
/// `groups.len() == 1` reproduces the historical scalar layout byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRhsLayout {
    /// D dimension owned by the consuming level and shared by every group.
    pub d_ring_dimension: usize,
    pub n_d: usize,
    pub groups: Vec<RelationGroupRows>,
    pub(super) compression: Option<RelationCompressionLayout>,
}

#[cfg(test)]
impl RelationRhsLayout {
    pub(crate) fn new_for_test(
        d_ring_dimension: usize,
        n_d: usize,
        groups: Vec<RelationGroupRows>,
    ) -> Self {
        Self {
            d_ring_dimension,
            n_d,
            groups,
            compression: None,
        }
    }
}

/// Joint checked relation and witness geometry for one opening level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationWitnessGeometry {
    extension_degree: usize,
    rhs_layout: RelationRhsLayout,
}

impl RelationWitnessGeometry {
    pub(crate) const fn from_parts(extension_degree: usize, rhs_layout: RelationRhsLayout) -> Self {
        Self {
            extension_degree,
            rhs_layout,
        }
    }

    #[must_use]
    pub const fn extension_degree(&self) -> usize {
        self.extension_degree
    }

    #[must_use]
    pub const fn rhs_layout(&self) -> &RelationRhsLayout {
        &self.rhs_layout
    }

    /// Opening coefficient geometry for one opening-batch group.
    pub fn group_opening_geometry(
        &self,
        group_index: usize,
    ) -> Result<RelationRowGeometry, AkitaError> {
        self.rhs_layout
            .groups
            .iter()
            .find_map(|group| (group.group_index == group_index).then_some(group.opening_geometry))
            .ok_or_else(|| AkitaError::InvalidInput("relation group index is invalid".into()))
    }

    /// Opening procedure represented by one group's consistency row.
    pub fn group_opening_method(&self, group_index: usize) -> Result<OpeningMethod, AkitaError> {
        self.rhs_layout
            .groups
            .iter()
            .find_map(|group| (group.group_index == group_index).then_some(group.opening_method))
            .ok_or_else(|| AkitaError::InvalidInput("relation group index is invalid".into()))
    }
}

/// Semantic identity and native dimension of one relation row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationRowFamily {
    /// Per-group consistency row at the A dimension.
    Consistency {
        group_index: usize,
        opening_method: OpeningMethod,
        geometry: RelationRowGeometry,
    },
    /// A matrix row.
    Inner {
        group_index: usize,
        row: usize,
        geometry: RelationRowGeometry,
    },
    /// B matrix row.
    Outer {
        group_index: usize,
        slice_index: usize,
        physical_row: usize,
        geometry: RelationRowGeometry,
    },
    /// Level-shared D matrix row.
    Opening {
        row: usize,
        geometry: RelationRowGeometry,
    },
    /// F compression row for one group and layer.
    CompressionF {
        group_index: usize,
        map_index: usize,
        geometry: RelationRowGeometry,
    },
    /// Level-shared H compression row for one layer.
    CompressionH {
        map_index: usize,
        geometry: RelationRowGeometry,
    },
}

impl RelationRowFamily {
    /// Checked modulus, coordinate-plane, and physical width of this row.
    #[must_use]
    pub const fn geometry(self) -> RelationRowGeometry {
        match self {
            Self::Consistency { geometry, .. }
            | Self::Inner { geometry, .. }
            | Self::Outer { geometry, .. }
            | Self::Opening { geometry, .. }
            | Self::CompressionF { geometry, .. }
            | Self::CompressionH { geometry, .. } => geometry,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelationCompressionLayout {
    pub(super) group_indices: Vec<usize>,
    pub(super) group_plans: Vec<CompressionChainPlan>,
    pub(super) opening_plan: CompressionChainPlan,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_geometry_keeps_modulus_planes_and_width_distinct() {
        let geometry = RelationRowGeometry::new(64, 2).expect("two-plane geometry");
        assert_eq!(geometry.polynomial_modulus_dimension(), 64);
        assert_eq!(geometry.coordinate_plane_count(), 2);
        assert_eq!(geometry.physical_coefficient_width(), 128);
        assert_eq!(geometry.physical_coefficient_index(1, 63).unwrap(), 127);
        assert!(geometry.physical_coefficient_index(2, 0).is_err());
        assert!(geometry.physical_coefficient_index(0, 64).is_err());
    }

    #[test]
    fn row_geometry_rejects_malformed_or_overflowing_shapes() {
        assert!(RelationRowGeometry::new(0, 1).is_err());
        assert!(RelationRowGeometry::new(3, 1).is_err());
        assert!(RelationRowGeometry::new(64, 3).is_err());
        let largest_power = 1usize << (usize::BITS - 1);
        assert!(RelationRowGeometry::new(largest_power, 2).is_err());
    }
}
