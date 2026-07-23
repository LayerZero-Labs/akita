//! Challenge-free setup product geometry: projection sizing and envelope guards.

use akita_algebra::offset_eq::MAX_COMPACT_STRIDE_TERMS;
use akita_field::{AkitaError, FieldCore};

use crate::layout::{validate_role_dims, CommitmentRingDims};
use crate::proof::AkitaExpandedSetup;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SetupProjectionGroupGeometry {
    pub(crate) a_rows: usize,
    pub(crate) a_cols: usize,
    pub(crate) b_rows: usize,
    pub(crate) b_cols: usize,
    pub(crate) d_active_cols: usize,
    pub(crate) ownership_units: usize,
    pub(crate) depth_fold: usize,
}

/// Checked common-base geometry for the Stage 3 setup projection.
///
/// Physical A, B, and D matrices retain their native role dimensions. Stage 3
/// views their flat coefficients as rings over `base_ring_dim = min(d_a,d_b,d_d)`.
/// The projection ratios expand each native role footprint into that common
/// base without changing its flat coefficient count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupProjectionGeometry {
    role_dims: CommitmentRingDims,
    base_ring_dim: usize,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
    a_projection_width: usize,
    b_projection_width: usize,
    d_projection_width: usize,
    required: usize,
    setup_index_len: usize,
    ring_bits: usize,
    rounds: usize,
    natural_field_len: usize,
    evaluation_terms: usize,
}

impl SetupProjectionGeometry {
    pub(crate) fn from_groups(
        role_dims: CommitmentRingDims,
        d_rows: usize,
        d_physical_cols: usize,
        groups: &[SetupProjectionGroupGeometry],
    ) -> Result<Self, AkitaError> {
        if groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup projection requires at least one group".into(),
            ));
        }
        let d_footprint = d_rows
            .checked_mul(d_physical_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
        let mut a_footprint = 0usize;
        let mut b_footprint = 0usize;
        for group in groups {
            a_footprint =
                a_footprint.max(group.a_rows.checked_mul(group.a_cols).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A footprint overflow".into())
                })?);
            b_footprint =
                b_footprint.max(group.b_rows.checked_mul(group.b_cols).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup B footprint overflow".into())
                })?);
        }
        let mut geometry =
            Self::from_role_footprints(role_dims, a_footprint, b_footprint, d_footprint)?;
        let mut evaluation_terms = 0usize;
        for group in groups {
            let d_terms = group
                .d_active_cols
                .checked_mul(d_rows)
                .and_then(|terms| terms.checked_mul(geometry.d_ratio))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup D evaluation work overflow".into())
                })?;
            let b_terms = group
                .b_cols
                .checked_mul(group.b_rows)
                .and_then(|terms| terms.checked_mul(geometry.b_ratio))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup B evaluation work overflow".into())
                })?;
            let a_terms = group
                .a_cols
                .checked_mul(group.a_rows)
                .and_then(|terms| terms.checked_mul(geometry.a_ratio))
                .and_then(|terms| terms.checked_mul(group.ownership_units))
                .and_then(|terms| terms.checked_mul(group.depth_fold))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A evaluation work overflow".into())
                })?;
            evaluation_terms = evaluation_terms
                .checked_add(d_terms)
                .and_then(|terms| terms.checked_add(b_terms))
                .and_then(|terms| terms.checked_add(a_terms))
                .ok_or_else(|| AkitaError::InvalidSetup("setup evaluation work overflow".into()))?;
        }
        geometry.evaluation_terms = evaluation_terms;
        Ok(geometry)
    }

    pub(crate) fn from_role_footprints(
        role_dims: CommitmentRingDims,
        a_footprint: usize,
        b_footprint: usize,
        d_footprint: usize,
    ) -> Result<Self, AkitaError> {
        let (base_ring_dim, a_ratio, b_ratio, d_ratio) = checked_role_ratios(role_dims)?;
        let a_projection_width = a_footprint
            .checked_mul(a_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A projection width overflow".into()))?;
        let b_projection_width = b_footprint
            .checked_mul(b_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B projection width overflow".into()))?;
        let d_projection_width = d_footprint
            .checked_mul(d_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D projection width overflow".into()))?;
        let required = a_projection_width
            .max(b_projection_width)
            .max(d_projection_width);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup projection requires a non-empty footprint".into(),
            ));
        }
        let setup_index_len = required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup index domain overflow".into()))?;
        let ring_bits = base_ring_dim.trailing_zeros() as usize;
        let rounds = ring_bits
            .checked_add(setup_index_len.trailing_zeros() as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("setup round count overflow".into()))?;
        let natural_field_len = required.checked_mul(base_ring_dim).ok_or_else(|| {
            AkitaError::InvalidSetup("setup product natural field length overflow".into())
        })?;
        Ok(Self {
            role_dims,
            base_ring_dim,
            a_ratio,
            b_ratio,
            d_ratio,
            a_projection_width,
            b_projection_width,
            d_projection_width,
            required,
            setup_index_len,
            ring_bits,
            rounds,
            natural_field_len,
            evaluation_terms: 0,
        })
    }

    /// Number of native B- and D-role subcolumns in one A-role witness column.
    pub fn witness_subcolumn_ratios(
        role_dims: CommitmentRingDims,
    ) -> Result<(usize, usize), AkitaError> {
        let (_, a_ratio, b_ratio, d_ratio) = checked_role_ratios(role_dims)?;
        let b_subcolumns = a_ratio
            .checked_div(b_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("B role does not divide the A-role witness width".into())
            })?;
        let d_subcolumns = a_ratio
            .checked_div(d_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("D role does not divide the A-role witness width".into())
            })?;
        if !b_subcolumns.is_power_of_two() || !d_subcolumns.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation role projection ratios must be powers of two".into(),
            ));
        }
        Ok((b_subcolumns, d_subcolumns))
    }

    #[must_use]
    pub const fn role_dims(self) -> CommitmentRingDims {
        self.role_dims
    }

    #[must_use]
    pub const fn base_ring_dim(self) -> usize {
        self.base_ring_dim
    }

    #[must_use]
    pub const fn a_ratio(self) -> usize {
        self.a_ratio
    }

    #[must_use]
    pub const fn b_ratio(self) -> usize {
        self.b_ratio
    }

    #[must_use]
    pub const fn d_ratio(self) -> usize {
        self.d_ratio
    }

    #[must_use]
    pub const fn a_projection_width(self) -> usize {
        self.a_projection_width
    }

    #[must_use]
    pub const fn b_projection_width(self) -> usize {
        self.b_projection_width
    }

    #[must_use]
    pub const fn d_projection_width(self) -> usize {
        self.d_projection_width
    }

    #[must_use]
    pub const fn required(self) -> usize {
        self.required
    }

    #[must_use]
    pub const fn setup_index_len(self) -> usize {
        self.setup_index_len
    }

    #[must_use]
    pub const fn ring_bits(self) -> usize {
        self.ring_bits
    }

    #[must_use]
    pub const fn rounds(self) -> usize {
        self.rounds
    }

    #[must_use]
    pub const fn alpha_power_len(self) -> usize {
        self.base_ring_dim
    }

    #[must_use]
    pub const fn natural_field_len(self) -> usize {
        self.natural_field_len
    }

    #[must_use]
    pub const fn evaluation_terms(self) -> usize {
        self.evaluation_terms
    }

    pub fn ensure_evaluation_budget(self) -> Result<(), AkitaError> {
        if self.evaluation_terms > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: self.evaluation_terms,
            });
        }
        Ok(())
    }
}

fn checked_role_ratios(
    role_dims: CommitmentRingDims,
) -> Result<(usize, usize, usize, usize), AkitaError> {
    validate_role_dims(role_dims)?;
    let base_ring_dim = role_dims.d_a().min(role_dims.d_b()).min(role_dims.d_d());
    let ratio = |role: &'static str, dimension: usize| {
        if !dimension.is_multiple_of(base_ring_dim) {
            return Err(AkitaError::InvalidSetup(format!(
                "{role} ring dimension does not decompose over the Stage 3 base"
            )));
        }
        let ratio = dimension / base_ring_dim;
        if ratio == 0 || !ratio.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "{role} Stage 3 projection ratio must be a non-zero power of two"
            )));
        }
        Ok(ratio)
    };
    Ok((
        base_ring_dim,
        ratio("A", role_dims.d_a())?,
        ratio("B", role_dims.d_b())?,
        ratio("D", role_dims.d_d())?,
    ))
}

/// Fail-closed envelope guard: `required` inner (`d_a`) rows must fit the shared
/// matrix prefix at `fold_ring_d`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the envelope.
pub fn ensure_setup_envelope<F: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    required: usize,
    fold_ring_d: usize,
) -> Result<(), AkitaError> {
    let setup_len = expanded
        .shared_matrix()
        .total_ring_elements_at_dyn(fold_ring_d)?;
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    #[test]
    fn ensure_setup_envelope_rejects_undersized_matrix() {
        let seed = crate::AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            gen_ring_dim: 32,
            max_setup_len: 1,
            public_matrix_seed: [1u8; 32],
        };
        let shared = crate::derive_public_matrix_flat::<F, 32>(1, &seed.public_matrix_seed);
        let expanded =
            crate::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared);
        let err = ensure_setup_envelope(&expanded, 2, 32).expect_err("undersized");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn projection_geometry_uses_nested_common_base() {
        let geometry = SetupProjectionGeometry::from_role_footprints(
            CommitmentRingDims {
                inner: 64,
                outer: 32,
                opening: 32,
            },
            7,
            11,
            13,
        )
        .expect("nested geometry");
        assert_eq!(geometry.base_ring_dim(), 32);
        assert_eq!(geometry.a_ratio(), 2);
        assert_eq!(geometry.b_ratio(), 1);
        assert_eq!(geometry.d_ratio(), 1);
        assert_eq!(geometry.required(), 14);
        assert_eq!(geometry.alpha_power_len(), 32);
        assert_eq!(geometry.natural_field_len(), 14 * 32);
    }

    #[test]
    fn projection_geometry_rejects_non_nested_roles() {
        let err = SetupProjectionGeometry::from_role_footprints(
            CommitmentRingDims {
                inner: 64,
                outer: 16,
                opening: 32,
            },
            1,
            1,
            1,
        )
        .expect_err("non-nested roles");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn evaluation_budget_accepts_cap_and_rejects_next_term() {
        let geometry_at_cap = SetupProjectionGeometry::from_groups(
            CommitmentRingDims::uniform(64),
            0,
            0,
            &[SetupProjectionGroupGeometry {
                a_rows: 1,
                a_cols: MAX_COMPACT_STRIDE_TERMS,
                b_rows: 0,
                b_cols: 0,
                d_active_cols: 0,
                ownership_units: 1,
                depth_fold: 1,
            }],
        )
        .expect("geometry at cap");
        assert_eq!(geometry_at_cap.evaluation_terms(), MAX_COMPACT_STRIDE_TERMS);
        geometry_at_cap
            .ensure_evaluation_budget()
            .expect("cap accepted");

        let geometry_above_cap = SetupProjectionGeometry::from_groups(
            CommitmentRingDims::uniform(64),
            0,
            0,
            &[SetupProjectionGroupGeometry {
                a_rows: 1,
                a_cols: MAX_COMPACT_STRIDE_TERMS + 1,
                b_rows: 0,
                b_cols: 0,
                d_active_cols: 0,
                ownership_units: 1,
                depth_fold: 1,
            }],
        )
        .expect("geometry above cap");
        assert!(matches!(
            geometry_above_cap.ensure_evaluation_budget(),
            Err(AkitaError::InvalidSize { .. })
        ));
    }
}
