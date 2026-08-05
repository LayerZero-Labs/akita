//! Relation row identities and shared layout data.

use crate::{CommitmentRingDims, CompressionChainPlan};

/// Per-group row-count inputs for assembling the relation rhs vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationGroupRows {
    /// This group's A/B dimensions completed by the level-shared D dimension.
    pub role_dims: CommitmentRingDims,
    pub n_a: usize,
    pub commit_rows: usize,
    pub b_inner_rows: usize,
}

/// Row-count inputs for assembling the relation rhs vector.
///
/// relation-matrix row order: `[final, precommitted_0, .., precommitted_{G-2}]`.
/// `groups.len() == 1` reproduces the historical scalar layout byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRhsLayout {
    /// D dimension owned by the consuming level and shared by every group.
    pub opening_ring_dim: usize,
    pub n_d: usize,
    pub groups: Vec<RelationGroupRows>,
    pub(super) compression: Option<RelationCompressionLayout>,
}

/// Semantic identity and native dimension of one relation row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationRowFamily {
    /// Per-group consistency row at the A dimension.
    Consistency { group_index: usize, ring_dim: usize },
    /// A matrix row.
    Inner {
        group_index: usize,
        row: usize,
        ring_dim: usize,
    },
    /// B matrix row.
    Outer {
        group_index: usize,
        row: usize,
        ring_dim: usize,
    },
    /// Level-shared D matrix row.
    Opening { row: usize, ring_dim: usize },
    /// F compression row for one group and layer.
    CompressionF {
        group_index: usize,
        map_index: usize,
        ring_dim: usize,
    },
    /// Level-shared H compression row for one layer.
    CompressionH { map_index: usize, ring_dim: usize },
}

impl RelationRowFamily {
    /// Native coefficient count of this row.
    #[must_use]
    pub const fn ring_dim(self) -> usize {
        match self {
            Self::Consistency { ring_dim, .. }
            | Self::Inner { ring_dim, .. }
            | Self::Outer { ring_dim, .. }
            | Self::Opening { ring_dim, .. }
            | Self::CompressionF { ring_dim, .. }
            | Self::CompressionH { ring_dim, .. } => ring_dim,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelationCompressionLayout {
    pub(super) group_indices: Vec<usize>,
    pub(super) group_plans: Vec<CompressionChainPlan>,
    pub(super) opening_plan: CompressionChainPlan,
}
