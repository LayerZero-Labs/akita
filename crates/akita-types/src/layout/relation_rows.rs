//! Semantic relation row families and quotient-tail layout.
//!
//! `RelationRowLayout` is the single source of truth for logical row order,
//! per-family ring dimensions, and quotient witness slices.

use super::{CommitmentRingDims, LevelParams, MRowLayout};
use crate::proof::OpeningClaimsLayout;
use crate::r_decomp_levels;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use std::fmt;

/// Compression layer metadata for outer/opening consistency families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsistencyLayer {
    Base,
    Compression { index: usize },
}

/// Logical relation row families in canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationRowFamily {
    EvaluationTrace,
    FoldEvaluation,
    FoldConsistency,
    OuterConsistency { layer: ConsistencyLayer },
    OpeningConsistency { layer: ConsistencyLayer },
}

impl fmt::Display for RelationRowFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvaluationTrace => write!(f, "EvaluationTrace"),
            Self::FoldEvaluation => write!(f, "FoldEvaluation"),
            Self::FoldConsistency => write!(f, "FoldConsistency"),
            Self::OuterConsistency { layer } => match layer {
                ConsistencyLayer::Base => write!(f, "OuterConsistency(Base)"),
                ConsistencyLayer::Compression { index } => {
                    write!(f, "OuterConsistency(Compression({index}))")
                }
            },
            Self::OpeningConsistency { layer } => match layer {
                ConsistencyLayer::Base => write!(f, "OpeningConsistency(Base)"),
                ConsistencyLayer::Compression { index } => {
                    write!(f, "OpeningConsistency(Compression({index}))")
                }
            },
        }
    }
}

/// One quotient-bearing slice inside the `r_hat` witness tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationQuotientSlice {
    pub witness_offset: usize,
    pub row_count: usize,
    pub ring_dim: usize,
    pub digit_depth: usize,
    pub log_basis: u32,
}

/// Derived quotient tail layout (concatenation of family slices).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationQuotientLayout {
    pub slices: Vec<RelationQuotientSlice>,
}

impl RelationQuotientLayout {
    /// Total number of quotient witness coefficients across all slices.
    pub fn total_coeffs(&self) -> usize {
        self.slices
            .iter()
            .map(|s| s.row_count.saturating_mul(s.digit_depth))
            .sum()
    }

    /// Build quotient slices from quotient-bearing row families.
    pub fn from_row_layout(layout: &RelationRowLayout, digit_depth: usize) -> Self {
        let mut witness_offset = 0usize;
        let mut slices = Vec::new();
        for family in &layout.families {
            let Some(quotient) = family.quotient else {
                continue;
            };
            let slice = RelationQuotientSlice {
                witness_offset,
                row_count: family.row_count,
                ring_dim: quotient.ring_dim,
                digit_depth,
                log_basis: quotient.log_basis,
            };
            witness_offset = witness_offset
                .saturating_add(family.row_count.saturating_mul(digit_depth));
            slices.push(slice);
        }
        Self { slices }
    }

    /// Validate non-overlapping, monotonic witness offsets.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let mut expected_offset = 0usize;
        for slice in &self.slices {
            if slice.row_count == 0 || slice.digit_depth == 0 || slice.ring_dim == 0 {
                return Err(AkitaError::InvalidSetup(
                    "quotient slice has zero row_count, digit_depth, or ring_dim".to_string(),
                ));
            }
            if slice.witness_offset != expected_offset {
                return Err(AkitaError::InvalidSetup(format!(
                    "quotient slice witness_offset {} != expected {}",
                    slice.witness_offset, expected_offset
                )));
            }
            let slice_len = slice
                .row_count
                .checked_mul(slice.digit_depth)
                .ok_or_else(|| AkitaError::InvalidSetup("quotient slice length overflow".into()))?;
            expected_offset = expected_offset
                .checked_add(slice_len)
                .ok_or_else(|| AkitaError::InvalidSetup("quotient layout length overflow".into()))?;
        }
        Ok(())
    }
}

/// Per-family layout entry inside [`RelationRowLayout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRowFamilyLayout {
    pub kind: RelationRowFamily,
    pub row_start: usize,
    pub row_count: usize,
    pub ring_dim: Option<usize>,
    pub quotient: Option<RelationQuotientSlice>,
}

/// Canonical logical row order for the ring-switched relation matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRowLayout {
    pub families: Vec<RelationRowFamilyLayout>,
}

impl RelationRowLayout {
    /// Total logical M-row count including `EvaluationTrace`.
    pub fn total_row_count(&self) -> usize {
        self.families
            .iter()
            .map(|f| f.row_count)
            .fold(0usize, |acc, n| acc.saturating_add(n))
    }

    /// Locate a family by kind (first match).
    pub fn family(&self, kind: RelationRowFamily) -> Option<&RelationRowFamilyLayout> {
        self.families.iter().find(|f| f.kind == kind)
    }

    /// Build the uniform scalar-level layout used by current uncompressed schedules.
    ///
    /// Logical order:
    /// `EvaluationTrace | FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency`
    ///
    /// Quotient-bearing families map to the historical row-major `r` tail:
    /// consistency | A | B | D.
    pub fn for_scalar_level<F: FieldCore + CanonicalField>(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: MRowLayout,
        opening_batch: &OpeningClaimsLayout,
        num_commitments: usize,
    ) -> Result<Self, AkitaError> {
        if lp.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(
                "RelationRowLayout::for_scalar_level does not support grouped-root layouts yet"
                    .to_string(),
            ));
        }
        opening_batch.check()?;
        lp.require_scalar_level("RelationRowLayout::for_scalar_level")?;

        let d_a = role_dims.d_a();
        let d_b = role_dims.d_b();
        let d_d = role_dims.d_d();
        let n_a = lp.a_key.row_len();
        let n_b = lp.b_key.row_len();
        let n_d = lp.n_d_active_for(m_row_layout);
        let b_rows = n_b
            .checked_mul(num_commitments)
            .ok_or_else(|| AkitaError::InvalidSetup("B row count overflow".into()))?;

        let digit_depth = r_decomp_levels::<F>(lp.log_basis);
        let log_basis = lp.log_basis;

        let mut row_start = 0usize;
        let mut families = Vec::with_capacity(5);

        let push_family = |families: &mut Vec<RelationRowFamilyLayout>,
                           kind: RelationRowFamily,
                           row_start: &mut usize,
                           row_count: usize,
                           ring_dim: Option<usize>,
                           quotient_ring_dim: Option<usize>| {
            families.push(RelationRowFamilyLayout {
                kind,
                row_start: *row_start,
                row_count,
                ring_dim,
                quotient: quotient_ring_dim.map(|ring_dim| RelationQuotientSlice {
                    witness_offset: 0,
                    row_count,
                    ring_dim,
                    digit_depth,
                    log_basis,
                }),
            });
            *row_start = row_start.saturating_add(row_count);
        };

        push_family(
            &mut families,
            RelationRowFamily::EvaluationTrace,
            &mut row_start,
            1,
            None,
            None,
        );
        push_family(
            &mut families,
            RelationRowFamily::FoldEvaluation,
            &mut row_start,
            1,
            Some(d_a),
            Some(d_a),
        );
        push_family(
            &mut families,
            RelationRowFamily::FoldConsistency,
            &mut row_start,
            n_a,
            Some(d_a),
            Some(d_a),
        );
        push_family(
            &mut families,
            RelationRowFamily::OuterConsistency {
                layer: ConsistencyLayer::Base,
            },
            &mut row_start,
            b_rows,
            Some(d_b),
            Some(d_b),
        );
        if n_d > 0 {
            push_family(
                &mut families,
                RelationRowFamily::OpeningConsistency {
                    layer: ConsistencyLayer::Base,
                },
                &mut row_start,
                n_d,
                Some(d_d),
                Some(d_d),
            );
        }

        let layout = Self { families };
        layout.validate()?;
        Ok(layout)
    }

    /// Validate row indices are contiguous and quotient metadata is consistent.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let mut expected_start = 0usize;
        for family in &self.families {
            if family.row_count == 0 {
                return Err(AkitaError::InvalidSetup(format!(
                    "relation row family {:?} has zero row_count",
                    family.kind
                )));
            }
            if family.row_start != expected_start {
                return Err(AkitaError::InvalidSetup(format!(
                    "relation row family {:?} starts at {} (expected {expected_start})",
                    family.kind, family.row_start
                )));
            }
            match family.kind {
                RelationRowFamily::EvaluationTrace => {
                    if family.ring_dim.is_some() || family.quotient.is_some() {
                        return Err(AkitaError::InvalidSetup(
                            "EvaluationTrace must have ring_dim=None and quotient=None".to_string(),
                        ));
                    }
                }
                _ => {
                    if family.ring_dim.is_none() {
                        return Err(AkitaError::InvalidSetup(format!(
                            "relation row family {:?} missing ring_dim",
                            family.kind
                        )));
                    }
                    if let Some(q) = family.quotient {
                        if Some(q.ring_dim) != family.ring_dim {
                            return Err(AkitaError::InvalidSetup(format!(
                                "relation row family {:?} quotient ring_dim mismatch",
                                family.kind
                            )));
                        }
                    }
                }
            }
            expected_start = expected_start.saturating_add(family.row_count);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LevelParams, SisModulusFamily};
    use crate::proof::OpeningClaimsLayout;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;

    fn test_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            4,
            4,
            2,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 1, 2, 2, 4)
        .expect("valid test params")
    }

    #[test]
    fn evaluation_trace_is_first_without_quotient() {
        let lp = test_level_params();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            MRowLayout::WithDBlock,
            &opening,
            1,
        )
        .expect("layout");
        let trace = layout
            .family(RelationRowFamily::EvaluationTrace)
            .expect("trace row");
        assert_eq!(trace.row_start, 0);
        assert_eq!(trace.row_count, 1);
        assert!(trace.ring_dim.is_none());
        assert!(trace.quotient.is_none());
    }

    #[test]
    fn uniform_quotient_len_matches_legacy_row_times_levels() {
        let lp = test_level_params();
        let num_commitments = 1;
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            MRowLayout::WithDBlock,
            &opening,
            num_commitments,
        )
        .expect("layout");
        let quotient = RelationQuotientLayout::from_row_layout(&layout, r_decomp_levels::<F>(lp.log_basis));
        quotient.validate().expect("valid quotient layout");
        let legacy_rows = lp
            .m_row_count_for(num_commitments, MRowLayout::WithDBlock)
            .expect("row count");
        let legacy_levels = r_decomp_levels::<F>(lp.log_basis);
        assert_eq!(quotient.total_coeffs(), legacy_rows * legacy_levels);
    }
}
