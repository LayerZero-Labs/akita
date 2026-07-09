//! Semantic relation row families and quotient-tail layout.
//!
//! `RelationRowLayout` is the single source of truth for logical row order,
//! per-family ring dimensions, and quotient witness slices.
//!
//! Canonical order puts ring-switch families first (matching today's physical
//! `eq_tau1` indices) and appends the field-level [`RelationRowFamily::EvaluationTrace`]
//! last, so quotient witness offsets need no stagger.

use super::{CommitmentRingDims, LevelParams, RelationMatrixRowLayout};
use crate::gadget_row_scalars;
use crate::proof::OpeningClaimsLayout;
use crate::r_decomp_levels;
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase};
use std::fmt;

/// Logical row index of [`RelationRowFamily::FoldEvaluation`] (historical consistency row).
///
/// Matches today's `eq_tau1[0]` / physical consistency row.
pub const FOLD_EVALUATION_ROW: usize = 0;

/// First logical row index of [`RelationRowFamily::FoldConsistency`] (A-role rows).
///
/// Matches today's `a_start()`.
pub const FOLD_CONSISTENCY_ROW: usize = 1;

/// First logical row index of [`RelationRowFamily::OuterConsistency`] B/`u` rows.
///
/// Matches today's `b_start()` for a scalar level.
#[inline]
#[must_use]
pub fn outer_consistency_row_start(n_a: usize) -> usize {
    FOLD_CONSISTENCY_ROW.saturating_add(n_a)
}

/// Compression layer metadata for outer/opening consistency families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsistencyLayer {
    Base,
    Compression { index: usize },
}

/// Logical relation row families in canonical order.
///
/// Ring-switch families come first; [`Self::EvaluationTrace`] is last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationRowFamily {
    FoldEvaluation,
    FoldConsistency,
    OuterConsistency { layer: ConsistencyLayer },
    OpeningConsistency { layer: ConsistencyLayer },
    EvaluationTrace,
}

impl fmt::Display for RelationRowFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::EvaluationTrace => write!(f, "EvaluationTrace"),
        }
    }
}

/// One quotient-bearing slice inside the `r_hat` witness tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationQuotientSlice {
    /// Offset inside the flattened `r_hat` witness tail.
    pub witness_offset: usize,
    /// First logical M-row index for this slice.
    pub row_start: usize,
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
            .map(|s| {
                s.witness_offset
                    .saturating_add(s.row_count.saturating_mul(s.digit_depth))
            })
            .max()
            .unwrap_or(0)
    }

    /// Build quotient slices from quotient-bearing row families.
    ///
    /// `witness_offset` is `row_start * digit_depth`: ring-switch families are
    /// contiguous from logical row 0, and [`RelationRowFamily::EvaluationTrace`]
    /// (last, no quotient) is skipped. Uniform schedules keep the historical
    /// row-major `r_hat` byte layout with no stagger.
    pub fn from_row_layout(layout: &RelationRowLayout, digit_depth: usize) -> Self {
        let mut slices = Vec::new();
        for family in &layout.families {
            let Some(quotient) = family.quotient else {
                continue;
            };
            let witness_offset = family.row_start.saturating_mul(digit_depth);
            let slice = RelationQuotientSlice {
                witness_offset,
                row_start: family.row_start,
                row_count: family.row_count,
                ring_dim: quotient.ring_dim,
                digit_depth,
                log_basis: quotient.log_basis,
            };
            slices.push(slice);
        }
        Self { slices }
    }

    /// Validate non-overlapping, monotonic witness offsets.
    pub fn validate(&self) -> Result<(), AkitaError> {
        for slice in &self.slices {
            if slice.row_count == 0 || slice.digit_depth == 0 || slice.ring_dim == 0 {
                return Err(AkitaError::InvalidSetup(
                    "quotient slice has zero row_count, digit_depth, or ring_dim".to_string(),
                ));
            }
            let slice_end =
                slice
                    .witness_offset
                    .checked_add(slice.row_count.checked_mul(slice.digit_depth).ok_or_else(
                        || AkitaError::InvalidSetup("quotient slice length overflow".into()),
                    )?)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("quotient layout length overflow".into())
                    })?;
            if slice_end == 0 {
                return Err(AkitaError::InvalidSetup(
                    "quotient slice end must be positive".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Materialize flattened `r_hat` MLE weights:
    /// `-(alpha^{ring_dim} + 1) * eq_tau1[row] * gadget[level]` per slice.
    pub fn materialize_tail_weights<F, E>(
        &self,
        eq_tau1: &[E],
        alpha: E,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBase<F>,
    {
        let mut out = vec![E::zero(); self.total_coeffs()];
        for slice in &self.slices {
            let alpha_pows = scalar_powers(alpha, slice.ring_dim);
            let denom = alpha_pows[slice.ring_dim - 1] * alpha + E::one();
            let gadget: Vec<E> = gadget_row_scalars::<F>(slice.digit_depth, slice.log_basis)
                .into_iter()
                .map(E::lift_base)
                .collect();
            for local_row in 0..slice.row_count {
                let row_idx = slice.row_start.checked_add(local_row).ok_or_else(|| {
                    AkitaError::InvalidSetup("quotient row index overflow".into())
                })?;
                let eq = eq_tau1.get(row_idx).copied().unwrap_or(E::zero());
                for (level_idx, gadget_weight) in gadget.iter().enumerate() {
                    let idx = slice
                        .witness_offset
                        .checked_add(local_row.checked_mul(slice.digit_depth).ok_or_else(|| {
                            AkitaError::InvalidSetup("quotient witness offset overflow".into())
                        })?)
                        .and_then(|base| base.checked_add(level_idx))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("quotient witness offset overflow".into())
                        })?;
                    out[idx] = -eq * denom * *gadget_weight;
                }
            }
        }
        Ok(out)
    }
}

/// Quotient witness coefficient count for a scalar-level or multi-group-root layout.
pub fn quotient_witness_coeff_count_for_scalar_level<F: FieldCore + CanonicalField>(
    lp: &LevelParams,
    m_row_layout: RelationMatrixRowLayout,
    num_commitments: usize,
) -> Result<usize, AkitaError> {
    quotient_witness_coeff_count_for_scalar_level_bits(
        lp,
        m_row_layout,
        num_commitments,
        r_decomp_levels::<F>(lp.log_basis),
    )
}

/// Non-generic variant using an explicit gadget digit depth.
pub fn quotient_witness_coeff_count_for_scalar_level_bits(
    lp: &LevelParams,
    m_row_layout: RelationMatrixRowLayout,
    num_commitments: usize,
    digit_depth: usize,
) -> Result<usize, AkitaError> {
    let opening_batch = if lp.has_precommitted_groups() {
        let group_sizes: Vec<usize> = std::iter::repeat_n(1, num_commitments).collect();
        OpeningClaimsLayout::from_group_sizes(8, &group_sizes)?
    } else {
        OpeningClaimsLayout::new(8, num_commitments)?
    };
    let row_layout = if lp.has_precommitted_groups() {
        RelationRowLayout::for_multi_group_root_with_digit_depth(
            lp,
            lp.role_dims(),
            m_row_layout,
            &opening_batch,
            digit_depth,
        )?
    } else {
        RelationRowLayout::for_scalar_level_with_digit_depth(
            lp,
            lp.role_dims(),
            m_row_layout,
            &opening_batch,
            num_commitments,
            digit_depth,
        )?
    };
    let quotient = RelationQuotientLayout::from_row_layout(&row_layout, digit_depth);
    quotient.validate()?;
    Ok(quotient.total_coeffs())
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
    /// Total logical τ₁ row count including [`RelationRowFamily::EvaluationTrace`].
    pub fn total_row_count(&self) -> usize {
        self.families
            .iter()
            .map(|f| f.row_count)
            .fold(0usize, |acc, n| acc.saturating_add(n))
    }

    /// Quotient-bearing row count (excludes [`RelationRowFamily::EvaluationTrace`]).
    pub fn quotient_row_count(&self) -> usize {
        self.families
            .iter()
            .filter(|f| f.quotient.is_some())
            .map(|f| f.row_count)
            .fold(0usize, |acc, n| acc.saturating_add(n))
    }

    /// Locate a family by kind (first match).
    pub fn family(&self, kind: RelationRowFamily) -> Option<&RelationRowFamilyLayout> {
        self.families.iter().find(|f| f.kind == kind)
    }

    /// Logical row index of the shared [`RelationRowFamily::EvaluationTrace`] row.
    pub fn evaluation_trace_row(&self) -> Result<usize, AkitaError> {
        self.family(RelationRowFamily::EvaluationTrace)
            .map(|f| f.row_start)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("RelationRowLayout missing EvaluationTrace".to_string())
            })
    }

    /// Build the layout for a scalar or multi-group root level.
    pub fn for_level<F: FieldCore + CanonicalField>(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        if lp.has_precommitted_groups() {
            Self::for_multi_group_root::<F>(lp, role_dims, m_row_layout, opening_batch)
        } else {
            Self::for_scalar_level::<F>(
                lp,
                role_dims,
                m_row_layout,
                opening_batch,
                opening_batch.num_groups(),
            )
        }
    }

    /// τ₁ Boolean-table width (next power of two of [`Self::total_row_count`]).
    pub fn tau1_num_vars(&self) -> Result<usize, AkitaError> {
        let rows = self.total_row_count();
        let padded = rows.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("relation-row tau1 width overflow".to_string())
        })?;
        Ok(padded.trailing_zeros() as usize)
    }

    /// Build the uniform scalar-level layout used by current uncompressed schedules.
    ///
    /// Logical order:
    /// `FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency | EvaluationTrace`
    ///
    /// Quotient-bearing families map to the historical row-major `r` tail with
    /// `witness_offset = row_start * digit_depth` (no stagger).
    pub fn for_scalar_level<F: FieldCore + CanonicalField>(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
        num_commitments: usize,
    ) -> Result<Self, AkitaError> {
        Self::for_scalar_level_with_digit_depth(
            lp,
            role_dims,
            m_row_layout,
            opening_batch,
            num_commitments,
            r_decomp_levels::<F>(lp.log_basis),
        )
    }

    /// Build the uniform scalar-level layout with an explicit quotient digit depth.
    pub fn for_scalar_level_with_digit_depth(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
        num_commitments: usize,
        digit_depth: usize,
    ) -> Result<Self, AkitaError> {
        if lp.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(
                "RelationRowLayout::for_scalar_level does not support grouped-root layouts; use for_multi_group_root"
                    .to_string(),
            ));
        }
        opening_batch.check()?;
        lp.require_scalar_level("RelationRowLayout::for_scalar_level")?;
        if opening_batch.num_groups() != num_commitments {
            return Err(AkitaError::InvalidSetup(
                "scalar RelationRowLayout num_commitments must match opening_batch.num_groups()"
                    .to_string(),
            ));
        }

        let d_a = role_dims.d_a();
        let d_b = role_dims.d_b();
        let d_d = role_dims.d_d();
        let n_a = lp.a_key.row_len();
        let n_b = lp.b_key.row_len();
        let n_d = lp.n_d_active_for(m_row_layout);
        let b_rows = n_b
            .checked_mul(num_commitments)
            .ok_or_else(|| AkitaError::InvalidSetup("B row count overflow".into()))?;

        let q = QuotientBuildParams {
            digit_depth,
            log_basis: lp.log_basis,
        };
        let mut row_start = 0usize;
        let mut families = Vec::with_capacity(5);

        push_family(
            &mut families,
            RelationRowFamily::FoldEvaluation,
            &mut row_start,
            1,
            Some(d_a),
            Some(d_a),
            q,
        );
        push_family(
            &mut families,
            RelationRowFamily::FoldConsistency,
            &mut row_start,
            n_a,
            Some(d_a),
            Some(d_a),
            q,
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
            q,
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
                q,
            );
        }
        push_family(
            &mut families,
            RelationRowFamily::EvaluationTrace,
            &mut row_start,
            1,
            None,
            None,
            q,
        );

        let layout = Self { families };
        layout.validate()?;
        Ok(layout)
    }

    /// Build the multi-group root layout.
    ///
    /// Logical order:
    /// ```text
    /// FoldEvaluation
    /// | for g in root_group_order(): FoldConsistency_g | OuterConsistency_g
    /// | OpeningConsistency?
    /// | EvaluationTrace
    /// ```
    ///
    /// Per-group FoldConsistency / OuterConsistency starts match
    /// [`LevelParams::root_a_row_range`] / [`LevelParams::root_commitment_row_range`]
    /// exactly (no +1 shift). EvaluationTrace is appended after the quotient rows.
    pub fn for_multi_group_root<F: FieldCore + CanonicalField>(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        Self::for_multi_group_root_with_digit_depth(
            lp,
            role_dims,
            m_row_layout,
            opening_batch,
            r_decomp_levels::<F>(lp.log_basis),
        )
    }

    /// Multi-group root layout with an explicit quotient digit depth.
    pub fn for_multi_group_root_with_digit_depth(
        lp: &LevelParams,
        role_dims: CommitmentRingDims,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
        digit_depth: usize,
    ) -> Result<Self, AkitaError> {
        if !lp.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(
                "RelationRowLayout::for_multi_group_root requires precommitted groups".to_string(),
            ));
        }
        lp.reject_multi_group_multi_chunk("RelationRowLayout::for_multi_group_root")?;
        opening_batch.check()?;
        let num_groups = opening_batch.num_groups();
        if num_groups != lp.root_group_count() {
            return Err(AkitaError::InvalidSetup(
                "multi-group RelationRowLayout requires opening_batch.num_groups() == root_group_count()"
                    .to_string(),
            ));
        }

        let d_a = role_dims.d_a();
        let d_b = role_dims.d_b();
        let d_d = role_dims.d_d();
        let n_d = lp.n_d_active_for(m_row_layout);
        let q = QuotientBuildParams {
            digit_depth,
            log_basis: lp.log_basis,
        };
        let order = opening_batch.root_group_order()?;
        let final_group_index = opening_batch.root_final_group_index()?;

        let mut row_start = 0usize;
        let mut families = Vec::with_capacity(2 + 2 * num_groups);

        push_family(
            &mut families,
            RelationRowFamily::FoldEvaluation,
            &mut row_start,
            1,
            Some(d_a),
            Some(d_a),
            q,
        );

        for &group_index in &order {
            let (n_a, n_b) = if group_index == final_group_index {
                (lp.a_key.row_len(), lp.b_key.row_len())
            } else {
                let group = lp
                    .precommitted_groups
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?;
                (group.a_key.row_len(), group.b_key.row_len())
            };
            push_family(
                &mut families,
                RelationRowFamily::FoldConsistency,
                &mut row_start,
                n_a,
                Some(d_a),
                Some(d_a),
                q,
            );
            push_family(
                &mut families,
                RelationRowFamily::OuterConsistency {
                    layer: ConsistencyLayer::Base,
                },
                &mut row_start,
                n_b,
                Some(d_b),
                Some(d_b),
                q,
            );
        }

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
                q,
            );
        }

        push_family(
            &mut families,
            RelationRowFamily::EvaluationTrace,
            &mut row_start,
            1,
            None,
            None,
            q,
        );

        let layout = Self { families };
        layout.validate()?;
        Ok(layout)
    }

    /// Validate row indices are contiguous and quotient metadata is consistent.
    pub fn validate(&self) -> Result<(), AkitaError> {
        if self.families.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "RelationRowLayout has no families".to_string(),
            ));
        }
        let mut expected_start = 0usize;
        let last = self.families.len() - 1;
        for (idx, family) in self.families.iter().enumerate() {
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
                    if idx != last {
                        return Err(AkitaError::InvalidSetup(
                            "EvaluationTrace must be the last relation row family".to_string(),
                        ));
                    }
                    if family.row_count != 1 {
                        return Err(AkitaError::InvalidSetup(
                            "EvaluationTrace must have row_count=1".to_string(),
                        ));
                    }
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
                        if q.row_start != family.row_start || q.row_count != family.row_count {
                            return Err(AkitaError::InvalidSetup(format!(
                                "relation row family {:?} quotient row range mismatch",
                                family.kind
                            )));
                        }
                        let expected_offset = family.row_start.saturating_mul(q.digit_depth);
                        if q.witness_offset != expected_offset {
                            return Err(AkitaError::InvalidSetup(format!(
                                "relation row family {:?} quotient witness_offset {} != row_start * digit_depth ({expected_offset})",
                                family.kind, q.witness_offset
                            )));
                        }
                    }
                }
            }
            expected_start = expected_start.saturating_add(family.row_count);
        }
        if !matches!(
            self.families[last].kind,
            RelationRowFamily::EvaluationTrace
        ) {
            return Err(AkitaError::InvalidSetup(
                "RelationRowLayout must end with EvaluationTrace".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct QuotientBuildParams {
    digit_depth: usize,
    log_basis: u32,
}

fn push_family(
    families: &mut Vec<RelationRowFamilyLayout>,
    kind: RelationRowFamily,
    row_start: &mut usize,
    row_count: usize,
    ring_dim: Option<usize>,
    quotient_ring_dim: Option<usize>,
    q: QuotientBuildParams,
) {
    families.push(RelationRowFamilyLayout {
        kind,
        row_start: *row_start,
        row_count,
        ring_dim,
        quotient: quotient_ring_dim.map(|ring_dim| RelationQuotientSlice {
            witness_offset: row_start.saturating_mul(q.digit_depth),
            row_start: *row_start,
            row_count,
            ring_dim,
            digit_depth: q.digit_depth,
            log_basis: q.log_basis,
        }),
    });
    *row_start = row_start.saturating_add(row_count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{
        AjtaiKeyParams, LevelParams, PrecommittedLevelParams, SisModulusFamily,
    };
    use crate::proof::OpeningClaimsLayout;
    use crate::schedule::PrecommittedGroupParams;
    use crate::PolynomialGroupLayout;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    fn test_level_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            4,
            4,
            2,
            1,
            1,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(2, 1, 2, 2, 4)
        .expect("valid test params")
    }

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::pm1_only(3),
        )
    }

    fn sample_layout_lp() -> LevelParams {
        sample_params_only().with_decomp(4, 2, 2, 2, 0).unwrap()
    }

    fn sample_multi_group_root_params() -> (LevelParams, OpeningClaimsLayout) {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let precommit_lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let precommit = PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(
                PolynomialGroupLayout::new(4, 1),
                &precommit_lp,
            ),
            a_key: precommit_lp.a_key.clone(),
            b_key: AjtaiKeyParams::new_unchecked(
                precommit_lp.b_key.min_security_bits(),
                precommit_lp.b_key.sis_family(),
                5,
                precommit_lp.b_key.col_len(),
                precommit_lp.b_key.coeff_linf_bound(),
                precommit_lp.ring_dimension,
            ),
            num_blocks: precommit_lp.num_blocks,
            block_len: precommit_lp.block_len,
            num_digits_commit: precommit_lp.num_digits_commit,
            num_digits_open: precommit_lp.num_digits_open,
            num_digits_fold_one: precommit_lp.num_digits_fold_one,
        };
        let mut grouped = lp;
        grouped.precommitted_groups = vec![precommit];
        let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
        (grouped, batch)
    }

    #[test]
    fn evaluation_trace_is_last_without_quotient() {
        let lp = test_level_params();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            RelationMatrixRowLayout::WithDBlock,
            &opening,
            1,
        )
        .expect("layout");
        assert_eq!(layout.families.last().map(|f| f.kind), Some(RelationRowFamily::EvaluationTrace));
        let trace = layout
            .family(RelationRowFamily::EvaluationTrace)
            .expect("trace row");
        assert_eq!(trace.row_start, layout.total_row_count() - 1);
        assert_eq!(trace.row_count, 1);
        assert!(trace.ring_dim.is_none());
        assert!(trace.quotient.is_none());
        assert_eq!(
            layout.evaluation_trace_row().expect("trace index"),
            layout.total_row_count() - 1
        );
    }

    #[test]
    fn scalar_ring_family_indices_match_today() {
        let lp = test_level_params();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            RelationMatrixRowLayout::WithDBlock,
            &opening,
            1,
        )
        .expect("layout");
        let fold_eval = layout
            .family(RelationRowFamily::FoldEvaluation)
            .expect("fold evaluation");
        let fold_cons = layout
            .family(RelationRowFamily::FoldConsistency)
            .expect("fold consistency");
        let outer = layout
            .family(RelationRowFamily::OuterConsistency {
                layer: ConsistencyLayer::Base,
            })
            .expect("outer");
        assert_eq!(fold_eval.row_start, FOLD_EVALUATION_ROW);
        assert_eq!(fold_eval.row_start, 0);
        assert_eq!(fold_cons.row_start, FOLD_CONSISTENCY_ROW);
        assert_eq!(fold_cons.row_start, lp.a_start());
        assert_eq!(outer.row_start, outer_consistency_row_start(lp.a_key.row_len()));
        assert_eq!(outer.row_start, lp.b_start().expect("b_start"));
        let legacy_rows = lp
            .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .expect("row count");
        assert_eq!(layout.quotient_row_count(), legacy_rows);
        assert_eq!(layout.total_row_count(), legacy_rows + 1);
    }

    #[test]
    fn quotient_tail_weights_use_logical_row_tau1_index() {
        let lp = test_level_params();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            RelationMatrixRowLayout::WithDBlock,
            &opening,
            1,
        )
        .expect("layout");
        let quotient =
            RelationQuotientLayout::from_row_layout(&layout, r_decomp_levels::<F>(lp.log_basis));
        let fold_eval = layout
            .family(RelationRowFamily::FoldEvaluation)
            .expect("fold evaluation row");
        assert_eq!(fold_eval.row_start, 0);
        assert_eq!(quotient.slices[0].witness_offset, 0);
        assert_eq!(quotient.slices[0].row_start, 0);

        let mut eq_tau1 = vec![F::zero(); layout.total_row_count()];
        for (idx, weight) in eq_tau1.iter_mut().enumerate() {
            *weight = F::from_u64((idx + 1) as u64);
        }
        let alpha = F::from_u64(2);
        let weights = quotient
            .materialize_tail_weights::<F, F>(&eq_tau1, alpha)
            .expect("quotient weights");
        assert!(!weights.is_empty());

        let slice = &quotient.slices[0];
        let alpha_pows = scalar_powers(alpha, slice.ring_dim);
        let denom = alpha_pows[slice.ring_dim - 1] * alpha + F::one();
        let gadget0 = gadget_row_scalars::<F>(slice.digit_depth, slice.log_basis)[0];
        // FoldEvaluation at row 0: first quotient coeff uses eq_tau1[0].
        let expected0 = -eq_tau1[0] * denom * gadget0;
        assert_eq!(weights[0], expected0);
        let trace_row = layout.evaluation_trace_row().expect("trace");
        assert_ne!(weights[0], -eq_tau1[trace_row] * denom * gadget0);
    }

    #[test]
    fn mixed_role_dims_quotient_slices_use_per_family_ring_dim() {
        let lp = test_level_params();
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        assert!(dims.nests());
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            dims,
            RelationMatrixRowLayout::WithDBlock,
            &opening,
            1,
        )
        .expect("mixed role layout");
        let fold_eval = layout
            .family(RelationRowFamily::FoldEvaluation)
            .expect("fold evaluation");
        let outer = layout
            .family(RelationRowFamily::OuterConsistency {
                layer: ConsistencyLayer::Base,
            })
            .expect("outer consistency");
        let opening_family = layout
            .family(RelationRowFamily::OpeningConsistency {
                layer: ConsistencyLayer::Base,
            })
            .expect("opening consistency");
        assert_eq!(fold_eval.ring_dim, Some(128));
        assert_eq!(outer.ring_dim, Some(64));
        assert_eq!(opening_family.ring_dim, Some(32));
        assert_eq!(fold_eval.quotient.as_ref().map(|q| q.ring_dim), Some(128));
        assert_eq!(outer.quotient.as_ref().map(|q| q.ring_dim), Some(64));
        assert_eq!(
            opening_family.quotient.as_ref().map(|q| q.ring_dim),
            Some(32)
        );
    }

    #[test]
    fn uniform_quotient_len_matches_legacy_row_times_levels() {
        let lp = test_level_params();
        let num_commitments = 1;
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let layout = RelationRowLayout::for_scalar_level::<F>(
            &lp,
            CommitmentRingDims::uniform(4),
            RelationMatrixRowLayout::WithDBlock,
            &opening,
            num_commitments,
        )
        .expect("layout");
        let quotient =
            RelationQuotientLayout::from_row_layout(&layout, r_decomp_levels::<F>(lp.log_basis));
        quotient.validate().expect("valid quotient layout");
        let legacy_rows = lp
            .relation_matrix_row_count_for(num_commitments, RelationMatrixRowLayout::WithDBlock)
            .expect("row count");
        let legacy_levels = r_decomp_levels::<F>(lp.log_basis);
        assert_eq!(quotient.total_coeffs(), legacy_rows * legacy_levels);
        assert_eq!(layout.quotient_row_count() * legacy_levels, quotient.total_coeffs());
    }

    #[test]
    fn multi_group_layout_matches_root_row_ranges_without_shift() {
        let (lp, batch) = sample_multi_group_root_params();
        let m_layout = RelationMatrixRowLayout::WithDBlock;
        let layout = RelationRowLayout::for_multi_group_root::<F>(
            &lp,
            lp.role_dims(),
            m_layout,
            &batch,
        )
        .expect("multi-group layout");

        let legacy_rows = lp
            .relation_matrix_row_count_for(batch.num_groups(), m_layout)
            .expect("legacy rows");
        assert_eq!(layout.quotient_row_count(), legacy_rows);
        assert_eq!(layout.total_row_count(), legacy_rows + 1);
        assert_eq!(
            layout.evaluation_trace_row().expect("trace"),
            legacy_rows
        );

        let fold_eval = layout
            .family(RelationRowFamily::FoldEvaluation)
            .expect("fold evaluation");
        assert_eq!(fold_eval.row_start, 0);

        let order = batch.root_group_order().expect("order");
        let mut fold_consistency_starts = layout
            .families
            .iter()
            .filter(|f| f.kind == RelationRowFamily::FoldConsistency)
            .map(|f| f.row_start..f.row_start + f.row_count);
        let mut outer_starts = layout
            .families
            .iter()
            .filter(|f| {
                matches!(
                    f.kind,
                    RelationRowFamily::OuterConsistency {
                        layer: ConsistencyLayer::Base
                    }
                )
            })
            .map(|f| f.row_start..f.row_start + f.row_count);

        for &group_index in &order {
            let a_range = lp
                .root_a_row_range(&batch, group_index, m_layout)
                .expect("a range");
            let b_range = lp
                .root_commitment_row_range(&batch, group_index, m_layout)
                .expect("b range");
            assert_eq!(
                fold_consistency_starts.next().expect("fold consistency"),
                a_range,
                "FoldConsistency must match root_a_row_range (no +1)"
            );
            assert_eq!(
                outer_starts.next().expect("outer consistency"),
                b_range,
                "OuterConsistency must match root_commitment_row_range (no +1)"
            );
        }
        assert!(fold_consistency_starts.next().is_none());
        assert!(outer_starts.next().is_none());
    }

    #[test]
    fn multi_group_quotient_len_matches_legacy_row_times_levels() {
        let (lp, batch) = sample_multi_group_root_params();
        let m_layout = RelationMatrixRowLayout::WithDBlock;
        let layout = RelationRowLayout::for_multi_group_root::<F>(
            &lp,
            lp.role_dims(),
            m_layout,
            &batch,
        )
        .expect("multi-group layout");
        let digit_depth = r_decomp_levels::<F>(lp.log_basis);
        let quotient = RelationQuotientLayout::from_row_layout(&layout, digit_depth);
        quotient.validate().expect("valid quotient");
        let legacy_rows = lp
            .relation_matrix_row_count_for(batch.num_groups(), m_layout)
            .expect("legacy rows");
        assert_eq!(quotient.total_coeffs(), legacy_rows * digit_depth);
        assert_eq!(
            quotient_witness_coeff_count_for_scalar_level::<F>(
                &lp,
                m_layout,
                batch.num_groups(),
            )
            .expect("coeff count"),
            legacy_rows * digit_depth
        );
    }
}
