//! Challenge-free setup product geometry: projection sizing and envelope guards.

use akita_error::AkitaError;
use jolt_field::Field;

use crate::layout::{validate_role_dims, CommitmentRingDims};
use crate::proof::AkitaExpandedSetup;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SetupProjectionGroupGeometry {
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) a_rows: usize,
    pub(crate) a_cols: usize,
    pub(crate) b_rows: usize,
    pub(crate) b_cols: usize,
    pub(crate) d_active_cols: usize,
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
        validate_role_dims(role_dims)?;
        let base_ring_dim =
            groups
                .iter()
                .try_fold(role_dims.common_relation_coeff_count(), |base, group| {
                    validate_role_dims(group.role_dims)?;
                    if group.role_dims.d_d() != role_dims.d_d() {
                        return Err(AkitaError::InvalidSetup(
                            "setup projection groups disagree on the shared D dimension".into(),
                        ));
                    }
                    Ok(base.min(group.role_dims.common_relation_coeff_count()))
                })?;
        let ratio = |role: &'static str, dimension: usize| {
            checked_projection_ratio(role, dimension, base_ring_dim)
        };
        let d_ratio = ratio("D", role_dims.d_d())?;
        let d_footprint = d_rows
            .checked_mul(d_physical_cols)
            .and_then(|footprint| footprint.checked_mul(d_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
        let mut a_footprint = 0usize;
        let mut b_footprint = 0usize;
        for group in groups {
            let a_ratio = ratio("A", group.role_dims.d_a())?;
            let b_ratio = ratio("B", group.role_dims.d_b())?;
            a_footprint = a_footprint.max(
                group
                    .a_rows
                    .checked_mul(group.a_cols)
                    .and_then(|footprint| footprint.checked_mul(a_ratio))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?,
            );
            b_footprint = b_footprint.max(
                group
                    .b_rows
                    .checked_mul(group.b_cols)
                    .and_then(|footprint| footprint.checked_mul(b_ratio))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?,
            );
        }
        let (a_ratio, b_ratio) = (ratio("A", role_dims.d_a())?, ratio("B", role_dims.d_b())?);
        let required = a_footprint.max(b_footprint).max(d_footprint);
        Self::from_projected_footprints(
            role_dims,
            base_ring_dim,
            a_ratio,
            b_ratio,
            d_ratio,
            a_footprint,
            b_footprint,
            d_footprint,
            required,
        )
    }

    #[cfg(test)]
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
        Self::from_projected_footprints(
            role_dims,
            base_ring_dim,
            a_ratio,
            b_ratio,
            d_ratio,
            a_projection_width,
            b_projection_width,
            d_projection_width,
            required,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_projected_footprints(
        role_dims: CommitmentRingDims,
        base_ring_dim: usize,
        a_ratio: usize,
        b_ratio: usize,
        d_ratio: usize,
        a_projection_width: usize,
        b_projection_width: usize,
        d_projection_width: usize,
        required: usize,
    ) -> Result<Self, AkitaError> {
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
        })
    }

    /// Number of B- and D-native subcolumns in one A-native source ring.
    ///
    /// B and D are not ordered relative to each other. Both dimensions divide
    /// the A-native source dimension under the canonical projection invariant.
    pub fn native_role_subcolumn_counts(
        role_dims: CommitmentRingDims,
    ) -> Result<(usize, usize), AkitaError> {
        let (_, a_ratio, b_ratio, d_ratio) = checked_role_ratios(role_dims)?;
        let b_subcolumns = a_ratio
            .checked_div(b_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A-native source rings do not decompose into B-native subcolumns".into(),
                )
            })?;
        let d_subcolumns = a_ratio
            .checked_div(d_ratio)
            .filter(|ratio| *ratio != 0)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A-native source rings do not decompose into D-native subcolumns".into(),
                )
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

    pub(crate) fn validate_alpha_power_lengths(
        self,
        a_len: usize,
        b_len: usize,
        d_len: usize,
    ) -> Result<(), AkitaError> {
        for (role, expected, actual) in [
            ("A", self.role_dims.d_a(), a_len),
            ("B", self.role_dims.d_b(), b_len),
            ("D", self.role_dims.d_d(), d_len),
        ] {
            if actual != expected {
                return Err(AkitaError::InvalidSize { expected, actual });
            }
            if actual < self.base_ring_dim {
                return Err(AkitaError::InvalidSetup(format!(
                    "{role} alpha powers are shorter than the Stage 3 base"
                )));
            }
        }
        Ok(())
    }
}

fn checked_role_ratios(
    role_dims: CommitmentRingDims,
) -> Result<(usize, usize, usize, usize), AkitaError> {
    validate_role_dims(role_dims)?;
    let base_ring_dim = role_dims.d_a().min(role_dims.d_b()).min(role_dims.d_d());
    Ok((
        base_ring_dim,
        checked_projection_ratio("A", role_dims.d_a(), base_ring_dim)?,
        checked_projection_ratio("B", role_dims.d_b(), base_ring_dim)?,
        checked_projection_ratio("D", role_dims.d_d(), base_ring_dim)?,
    ))
}

fn checked_projection_ratio(
    role: &'static str,
    dimension: usize,
    base_ring_dim: usize,
) -> Result<usize, AkitaError> {
    if base_ring_dim == 0 || !dimension.is_multiple_of(base_ring_dim) {
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
}

/// Fail-closed envelope guard: `required` inner (`d_a`) rows must fit the shared
/// matrix prefix at `fold_ring_d`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the envelope.
pub fn ensure_setup_envelope<F: Field>(
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
    use jolt_field::Prime128OffsetA7F7;

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
    fn projection_geometry_uses_common_base() {
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
        .expect("common-base geometry");
        assert_eq!(geometry.base_ring_dim(), 32);
        assert_eq!(geometry.a_ratio(), 2);
        assert_eq!(geometry.b_ratio(), 1);
        assert_eq!(geometry.d_ratio(), 1);
        assert_eq!(geometry.required(), 14);
        assert_eq!(geometry.alpha_power_len(), 32);
        assert_eq!(geometry.natural_field_len(), 14 * 32);
    }

    #[test]
    fn projection_geometry_accepts_reversed_b_d_order() {
        let geometry = SetupProjectionGeometry::from_role_footprints(
            CommitmentRingDims {
                inner: 128,
                outer: 32,
                opening: 64,
            },
            1,
            1,
            1,
        )
        .expect("role ordering is irrelevant to common-base projection");
        assert_eq!(geometry.base_ring_dim(), 32);
        assert_eq!(geometry.a_ratio(), 4);
        assert_eq!(geometry.b_ratio(), 1);
        assert_eq!(geometry.d_ratio(), 2);
    }

    #[test]
    fn projection_geometry_uses_every_groups_native_dimensions() {
        let geometry = SetupProjectionGeometry::from_groups(
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            1,
            3,
            &[
                SetupProjectionGroupGeometry {
                    role_dims: CommitmentRingDims::uniform(64),
                    a_rows: 2,
                    a_cols: 5,
                    b_rows: 3,
                    b_cols: 7,
                    d_active_cols: 1,
                },
                SetupProjectionGroupGeometry {
                    role_dims: CommitmentRingDims {
                        inner: 128,
                        outer: 32,
                        opening: 64,
                    },
                    a_rows: 2,
                    a_cols: 5,
                    b_rows: 3,
                    b_cols: 7,
                    d_active_cols: 2,
                },
            ],
        )
        .expect("mixed-group geometry");

        assert_eq!(geometry.base_ring_dim(), 32);
        assert_eq!(geometry.a_projection_width(), 40);
        assert_eq!(geometry.b_projection_width(), 42);
        assert_eq!(geometry.d_projection_width(), 6);
        assert_eq!(geometry.required(), 42);
    }
}
