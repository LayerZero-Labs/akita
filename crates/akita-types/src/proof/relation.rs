//! Shared protocol relation helpers.

use crate::dispatch_for_field;
use crate::layout::{CommitmentRingDims, CommittedGroupParams};
use crate::opening_claims::OpeningClaimsLayout;
use crate::proof::RingVec;
use crate::{
    CommitmentSliceCount, CommittedSourceEncoding, CompressionChainPlan, OpeningMethod,
    SisModulusProfileId, SubringCoefficientPackingGeometry,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{
    eval_flat_ring_at_pows_fast, eval_ring_at, eval_ring_at_pows_fast, scalar_powers,
};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field, MulBaseUnreduced};
use std::iter::repeat_n;

#[path = "relation_layout.rs"]
mod layout_types;
use layout_types::RelationCompressionLayout;
pub use layout_types::{
    RelationGroupRows, RelationRhsLayout, RelationRowFamily, RelationRowGeometry,
    RelationWitnessGeometry,
};

/// Row geometry for one group's opening.
///
/// `source_encoding` is passed in rather than read off the group. It is a
/// property of the **fold** — `CommittedSourceEncoding` is commitment identity
/// owned by the level that commits the witness, and a precommitted group has
/// nowhere to store it, which is why the group's own accessor returns a
/// hard-coded canonical value. Reading it from the group would make the
/// tensor-projection mismatch below unreachable.
pub(crate) fn opening_row_geometry(
    params: &crate::GroupOpenPhaseParams,
    source_encoding: CommittedSourceEncoding,
    extension_degree: usize,
) -> Result<RelationRowGeometry, AkitaError> {
    let d_a = params.inner_commit_matrix_params().ring_dimension();
    match (params.opening_method(), source_encoding) {
        (
            OpeningMethod::EvaluationTrace,
            CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: encoded_degree,
            },
        ) if encoded_degree != extension_degree => Err(AkitaError::InvalidSetup(
            "tensor source encoding does not match the protocol extension degree".into(),
        )),
        (OpeningMethod::EvaluationTrace, _) => RelationRowGeometry::native(d_a),
        (
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            },
            CommittedSourceEncoding::CanonicalCoefficientTable,
        ) => {
            let geometry = SubringCoefficientPackingGeometry::try_new(
                extension_degree,
                d_a,
                challenge_subring_dimension,
            )?;
            RelationRowGeometry::new(
                geometry.challenge_subring_dimension(),
                geometry.extension_degree(),
            )
        }
        (OpeningMethod::SubringCoefficientPacking { .. }, _) => Err(AkitaError::InvalidSetup(
            "coefficient packing requires the canonical coefficient source encoding".into(),
        )),
    }
}

impl RelationRhsLayout {
    pub fn uniform(
        role_dims: CommitmentRingDims,
        n_d: usize,
        n_a: usize,
        physical_b_rows_per_group: usize,
        outer_slice_count: CommitmentSliceCount,
        num_groups: usize,
    ) -> Result<Self, AkitaError> {
        let opening_geometry = RelationRowGeometry::native(role_dims.d_a())?;
        let layout = Self {
            d_ring_dimension: role_dims.d_d(),
            n_d,
            groups: (0..num_groups)
                .map(|group_index| RelationGroupRows {
                    role_dims,
                    opening_geometry,
                    opening_method: OpeningMethod::EvaluationTrace,
                    n_a,
                    physical_b_rows: physical_b_rows_per_group,
                    outer_slice_count,
                    group_index,
                })
                .collect(),
            compression: None,
        };
        layout.validate()?;
        Ok(layout)
    }

    fn validate(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.d_ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "relation rhs layout requires non-empty group and ring geometry".into(),
            ));
        }
        for group in &self.groups {
            group.role_dims.validate_role_projection()?;
            if group.role_dims.d_d() != self.d_ring_dimension {
                return Err(AkitaError::InvalidSetup(
                    "relation rhs groups disagree with the level-shared D dimension".into(),
                ));
            }
            if !group
                .opening_geometry
                .physical_coefficient_width()
                .is_multiple_of(self.d_ring_dimension)
            {
                return Err(AkitaError::InvalidSetup(
                    "relation opening width is not divisible by the D dimension".into(),
                ));
            }
        }
        if let Some(compression) = &self.compression {
            if compression.group_indices.len() != self.groups.len()
                || compression.group_plans.len() != self.groups.len()
            {
                return Err(AkitaError::InvalidSetup(
                    "relation compression groups disagree with relation groups".into(),
                ));
            }
            for (group, plan) in self.groups.iter().zip(&compression.group_plans) {
                let expected = group
                    .logical_b_rows()?
                    .checked_mul(group.role_dims.d_b())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation B compression shape overflow".into())
                    })?;
                if plan.source_coefficients() != expected {
                    return Err(AkitaError::InvalidSetup(
                        "relation B compression plan has the wrong source shape".into(),
                    ));
                }
            }
            let expected = self.n_d.checked_mul(self.d_ring_dimension).ok_or_else(|| {
                AkitaError::InvalidSetup("relation D compression shape overflow".into())
            })?;
            if compression.opening_plan.source_coefficients() != expected {
                return Err(AkitaError::InvalidSetup(
                    "relation D compression plan has the wrong source shape".into(),
                ));
            }
        }
        Ok(())
    }

    /// Ring dimension of every physical relation-quotient row, in canonical
    /// relation-matrix order.
    ///
    /// Each group contributes one native-A consistency row, its native A rows,
    /// and its native B rows. The trailing D rows use the level-shared opening
    /// dimension.
    pub fn row_geometries(&self) -> Result<Vec<RelationRowGeometry>, AkitaError> {
        Ok(self
            .row_families()?
            .into_iter()
            .map(RelationRowFamily::geometry)
            .collect())
    }

    /// Semantic row families in canonical relation and quotient order.
    pub fn row_families(&self) -> Result<Vec<RelationRowFamily>, AkitaError> {
        self.validate()?;
        let row_count = self.groups.iter().try_fold(0usize, |rows, group| {
            rows.checked_add(1)
                .and_then(|rows| rows.checked_add(group.n_a))
                .and_then(|rows| rows.checked_add(group.logical_b_rows().ok()?))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation quotient row count overflow".into())
                })
        })?;
        let row_count = row_count.checked_add(self.n_d).ok_or_else(|| {
            AkitaError::InvalidSetup("relation quotient row count overflow".into())
        })?;
        let mut rows = Vec::with_capacity(row_count);
        for group in &self.groups {
            let group_index = group.group_index;
            rows.push(RelationRowFamily::Consistency {
                group_index,
                opening_method: group.opening_method,
                geometry: group.opening_geometry,
            });
            let inner_geometry = RelationRowGeometry::native(group.role_dims.d_a())?;
            for row in 0..group.n_a {
                rows.push(RelationRowFamily::Inner {
                    group_index,
                    row,
                    geometry: inner_geometry,
                });
            }
            let outer_geometry = RelationRowGeometry::native(group.role_dims.d_b())?;
            for slice_index in 0..group.outer_slice_count.get() {
                for physical_row in 0..group.physical_b_rows {
                    rows.push(RelationRowFamily::Outer {
                        group_index,
                        slice_index,
                        physical_row,
                        geometry: outer_geometry,
                    });
                }
            }
        }
        let opening_geometry = RelationRowGeometry::native(self.d_ring_dimension)?;
        for row in 0..self.n_d {
            rows.push(RelationRowFamily::Opening {
                row,
                geometry: opening_geometry,
            });
        }
        if let Some(compression) = &self.compression {
            for map_index in 0..crate::COMPRESSION_MAP_COUNT {
                for (&group_index, plan) in compression
                    .group_indices
                    .iter()
                    .zip(&compression.group_plans)
                {
                    let geometry =
                        RelationRowGeometry::native(plan.maps()[map_index].ring_dimension())?;
                    rows.push(RelationRowFamily::CompressionF {
                        group_index,
                        map_index,
                        geometry,
                    });
                }
                let geometry = RelationRowGeometry::native(
                    compression.opening_plan.maps()[map_index].ring_dimension(),
                )?;
                rows.push(RelationRowFamily::CompressionH {
                    map_index,
                    geometry,
                });
            }
        }
        Ok(rows)
    }

    /// Canonical compression plan for one relation-ordered B group.
    pub fn group_compression_plan(
        &self,
        relation_group_index: usize,
    ) -> Result<(usize, &CompressionChainPlan), AkitaError> {
        let compression = self.compression.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("relation layout has no compression geometry".into())
        })?;
        let group_index = *compression
            .group_indices
            .get(relation_group_index)
            .ok_or_else(|| AkitaError::InvalidInput("relation group index is invalid".into()))?;
        let plan = compression
            .group_plans
            .get(relation_group_index)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation compression group is missing".into())
            })?;
        Ok((group_index, plan))
    }

    /// Canonical B-compression plan for one opening-batch group index.
    pub fn compression_plan_for_group(
        &self,
        group_index: usize,
    ) -> Result<&CompressionChainPlan, AkitaError> {
        let compression = self.compression.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("relation layout has no compression geometry".into())
        })?;
        let relation_index = compression
            .group_indices
            .iter()
            .position(|&candidate| candidate == group_index)
            .ok_or_else(|| AkitaError::InvalidInput("opening group index is invalid".into()))?;
        compression
            .group_plans
            .get(relation_index)
            .ok_or_else(|| AkitaError::InvalidSetup("relation compression group is missing".into()))
    }

    /// Canonical compression plan for the shared D image.
    pub fn opening_compression_plan(&self) -> Result<&CompressionChainPlan, AkitaError> {
        self.compression
            .as_ref()
            .map(|compression| &compression.opening_plan)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation layout has no compression geometry".into())
            })
    }

    /// Checked wire geometry for one relation-ordered B payload.
    pub fn group_payload_geometry(
        &self,
        relation_group_index: usize,
    ) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        self.validate()?;
        let group = self
            .groups
            .get(relation_group_index)
            .ok_or_else(|| AkitaError::InvalidInput("relation group index is invalid".into()))?;
        let plan = self
            .compression
            .as_ref()
            .and_then(|compression| compression.group_plans.get(relation_group_index));
        crate::CommitmentPayloadGeometry::new(group.logical_b_rows()?, group.role_dims.d_b(), plan)
    }

    /// Checked wire geometry for the shared D payload.
    pub fn opening_payload_geometry(&self) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        self.validate()?;
        crate::CommitmentPayloadGeometry::new(
            self.n_d,
            self.d_ring_dimension,
            self.compression
                .as_ref()
                .map(|compression| &compression.opening_plan),
        )
    }
}

fn compression_plan(
    profile: SisModulusProfileId,
    rows: usize,
    ring_dim: usize,
) -> Result<CompressionChainPlan, AkitaError> {
    let source_coefficients = rows
        .checked_mul(ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("compression source shape overflow".into()))?;
    CompressionChainPlan::for_complete_source(profile, source_coefficients)
}

/// Single source of truth for the relation rhs row layout at one level.
///
/// # Errors
///
/// Returns an error if the opening batch is malformed for multi-group root params.
fn build_relation_rhs_layout(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    extension_degree: usize,
) -> Result<RelationRhsLayout, AkitaError> {
    let final_group_index = lp.validate_opening_batch_geometry(opening_batch)?;
    let n_d = lp.open().matrix.output_rank();
    let opening_plan = lp
        .payload_mode
        .is_compressed()
        .then(|| {
            compression_plan(
                lp.open().matrix.sis_modulus_profile(),
                n_d,
                lp.open().matrix.ring_dimension(),
            )
        })
        .transpose()?;
    if !lp.has_preceding_groups() {
        let role_dims = lp.role_dims();
        role_dims.validate_role_projection()?;
        let group_indices = opening_batch.root_group_order()?;
        // Use the layout the opening batch already carries rather than
        // deriving one: it is the authority, and it is correct even for a
        // fixture whose geometry has not been through validate_root_geometry.
        let opening_geometry =
            opening_row_geometry(&lp.final_group(), lp.source_encoding, extension_degree)?;
        let groups = group_indices
            .iter()
            .map(|&group_index| RelationGroupRows {
                group_index,
                role_dims,
                opening_geometry,
                opening_method: lp.opening_method(),
                n_a: lp.inner().matrix.output_rank(),
                physical_b_rows: lp.outer().matrix.output_rank(),
                outer_slice_count: lp.outer_slice_count(),
            })
            .collect::<Vec<_>>();
        let compression = if let Some(opening_plan) = opening_plan {
            let group_plan = compression_plan(
                lp.outer().matrix.sis_modulus_profile(),
                lp.outer_slice_count()
                    .logical_output_rows(lp.outer().matrix.output_rank())?,
                role_dims.d_b(),
            )?;
            Some(RelationCompressionLayout {
                group_plans: repeat_n(group_plan, group_indices.len()).collect(),
                group_indices,
                opening_plan,
            })
        } else {
            None
        };
        let layout = RelationRhsLayout {
            d_ring_dimension: role_dims.d_d(),
            n_d,
            groups,
            compression,
        };
        layout.validate()?;
        return Ok(layout);
    }
    // `validate_opening_batch_geometry` above already checked every group.
    // Resolve the native A/B dimensions directly so relation construction is
    // linear, rather than revalidating the entire batch once per group.
    let final_role_dims = lp.role_dims();
    final_role_dims.validate_role_projection()?;
    let mut groups = Vec::with_capacity(lp.preceding_group_count() + 1);
    let mut group_indices = Vec::with_capacity(lp.preceding_group_count() + 1);
    let mut group_plans = Vec::with_capacity(lp.preceding_group_count() + 1);
    groups.push(RelationGroupRows {
        group_index: final_group_index,
        role_dims: final_role_dims,
        opening_geometry: opening_row_geometry(
            &lp.final_group(),
            lp.source_encoding,
            extension_degree,
        )?,
        opening_method: lp.opening_method(),
        n_a: lp.inner().matrix.output_rank(),
        physical_b_rows: lp.outer().matrix.output_rank(),
        outer_slice_count: lp.outer_slice_count(),
    });
    group_indices.push(final_group_index);
    if opening_plan.is_some() {
        group_plans.push(compression_plan(
            lp.outer().matrix.sis_modulus_profile(),
            lp.outer_slice_count()
                .logical_output_rows(lp.outer().matrix.output_rank())?,
            final_role_dims.d_b(),
        )?);
    }
    for (group_index, group) in lp.preceding_group_iter().enumerate() {
        let role_dims = group.role_dims(lp.open().matrix.ring_dimension());
        role_dims.validate_role_projection()?;
        groups.push(RelationGroupRows {
            group_index,
            role_dims,
            opening_geometry: opening_row_geometry(
                group,
                crate::CommittedSourceEncoding::CanonicalCoefficientTable,
                extension_degree,
            )?,
            opening_method: group.opening.opening_method,
            n_a: group.profile.inner.matrix.output_rank(),
            physical_b_rows: group.profile.outer.matrix.output_rank(),
            outer_slice_count: group.profile.outer_slice_count,
        });
        group_indices.push(group_index);
        if opening_plan.is_some() {
            group_plans.push(compression_plan(
                group.profile.outer.matrix.sis_modulus_profile(),
                group
                    .profile
                    .outer_slice_count
                    .logical_output_rows(group.profile.outer.matrix.output_rank())?,
                role_dims.d_b(),
            )?);
        }
    }
    let layout = RelationRhsLayout {
        d_ring_dimension: final_role_dims.d_d(),
        n_d,
        groups,
        compression: opening_plan.map(|opening_plan| RelationCompressionLayout {
            group_indices,
            group_plans,
            opening_plan,
        }),
    };
    layout.validate()?;
    Ok(layout)
}

impl RelationWitnessGeometry {
    /// Resolve the single checked relation and witness geometry for one level.
    pub fn for_level(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        extension_degree: usize,
    ) -> Result<Self, AkitaError> {
        if !extension_degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "relation witness extension degree must be a nonzero power of two".into(),
            ));
        }
        let rhs_layout = build_relation_rhs_layout(lp, opening_batch, extension_degree)?;
        Ok(Self::from_parts(extension_degree, rhs_layout))
    }

    /// Resolve the current EvaluationTrace execution geometry and reject any
    /// scheduled coefficient-packing group before legacy ring-only code runs.
    pub fn for_evaluation_trace_execution(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        lp.validate_opening_batch(opening_batch)?;
        let geometry = Self::for_level(lp, opening_batch, 1)?;
        if geometry
            .rhs_layout()
            .groups
            .iter()
            .any(|group| !matches!(group.opening_method, OpeningMethod::EvaluationTrace))
        {
            return Err(AkitaError::InvalidSetup(
                "EvaluationTrace execution received a coefficient-packing group".into(),
            ));
        }
        Ok(geometry)
    }

    /// Common Stage-2 coefficient block derived from row polynomial moduli.
    pub fn relation_coefficient_block_len(&self) -> Result<usize, AkitaError> {
        let row_geometries = self
            .rhs_layout()
            .row_families()?
            .into_iter()
            .filter(|row| {
                !matches!(
                    row,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
            })
            .map(RelationRowFamily::geometry)
            .collect::<Vec<_>>();
        let coefficient_block = row_geometries
            .iter()
            .map(|geometry| geometry.polynomial_modulus_dimension())
            .min()
            .ok_or_else(|| AkitaError::InvalidSetup("relation rows are empty".into()))?;
        if row_geometries.iter().any(|geometry| {
            !geometry
                .physical_coefficient_width()
                .is_multiple_of(coefficient_block)
        }) {
            return Err(AkitaError::InvalidSetup(
                "relation row width is not aligned to its common modulus block".into(),
            ));
        }
        Ok(coefficient_block)
    }
}

/// Logical relation-matrix row count encoded in assembled relation rhs.
///
/// Layout: [consistency_g | A_g | B_g]_g | D (`n_d`), with each B group
/// expanded in slice-major then physical-row order.
#[must_use]
pub fn relation_rhs_row_count(layout: &RelationRhsLayout) -> usize {
    let group_rows = layout.groups.iter().fold(0usize, |acc, group| {
        acc.saturating_add(group.n_a).saturating_add(
            group
                .physical_b_rows
                .saturating_mul(group.outer_slice_count.get()),
        )
    });
    let base = layout
        .groups
        .len()
        .saturating_add(group_rows)
        .saturating_add(layout.n_d);
    layout.compression.as_ref().map_or(base, |compression| {
        base.saturating_add(
            crate::COMPRESSION_MAP_COUNT.saturating_mul(compression.group_plans.len() + 1),
        )
    })
}

/// Expected flat coefficient length of assembled `y` under per-role dimensions.
///
/// # Errors
///
/// Returns an error if any segment length arithmetic overflows.
pub fn relation_rhs_coeff_len(layout: &RelationRhsLayout) -> Result<usize, AkitaError> {
    layout.validate()?;
    let mut group_segment = 0usize;
    for group in &layout.groups {
        let a_segment = group
            .n_a
            .checked_mul(group.role_dims.d_a())
            .ok_or_else(|| AkitaError::InvalidSetup("relation y A segment overflow".into()))?;
        let b_rows = group.logical_b_rows()?;
        let b_segment = b_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| AkitaError::InvalidSetup("relation y B segment overflow".into()))?;
        group_segment = group_segment
            .checked_add(group.opening_geometry.physical_coefficient_width())
            .and_then(|len| len.checked_add(a_segment))
            .and_then(|len| len.checked_add(b_segment))
            .ok_or_else(|| AkitaError::InvalidSetup("relation y group segment overflow".into()))?;
    }
    let d_segment = layout
        .n_d
        .checked_mul(layout.d_ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y D segment overflow".into()))?;
    let base = d_segment
        .checked_add(group_segment)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y coefficient length overflow".into()))?;
    let compression_len = compression_rhs_coeff_len(layout)?;
    base.checked_add(compression_len)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y coefficient length overflow".into()))
}

fn compression_rhs_coeff_len(layout: &RelationRhsLayout) -> Result<usize, AkitaError> {
    layout
        .compression
        .as_ref()
        .map_or(Ok(0usize), |compression| {
            compression
                .group_plans
                .iter()
                .chain(core::iter::once(&compression.opening_plan))
                .try_fold(0usize, |total, plan| {
                    plan.maps().iter().try_fold(total, |total, map| {
                        total.checked_add(map.output_coefficients()).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "compression relation rhs length overflow".into(),
                            )
                        })
                    })
                })
        })
}

/// Number of ring rows decodable at role dimension `d` (compact or tagged storage).
fn ring_row_count_at<F: Field>(vec: &RingVec<F>, d: usize) -> Result<usize, AkitaError> {
    if vec.coeff_len() == 0 {
        return Ok(0);
    }
    if !vec.can_decode_vec(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: vec.coeff_len(),
        });
    }
    Ok(vec.coeff_len() / d)
}

/// Build the RHS vector `y` matching the scalar M row layout:
/// consistency (zero) | A (zeros) | B (`commitment_rows`) | D (`v`).
///
/// Public-output rows bind through the fused trace term, not `y`.
///
/// `commit_rows_per_group` is the B row count per commitment bundle
/// (`outer_commit_matrix.output_rank()`). The number of commitment bundles is inferred from
/// `commitment_rows.len() / commit_rows_per_group`.
///
/// # Errors
///
/// Returns an error if the supplied row slices do not match the expected row
/// counts for the level layout.
pub fn generate_relation_rhs<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    n_d: usize,
    commit_rows_per_group: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: Field,
{
    if v.len() != n_d {
        return Err(AkitaError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if commit_rows_per_group == 0
        || commitment_rows.is_empty()
        || !commitment_rows.len().is_multiple_of(commit_rows_per_group)
    {
        return Err(AkitaError::InvalidSize {
            expected: commit_rows_per_group,
            actual: commitment_rows.len(),
        });
    }
    let mut out = Vec::with_capacity(1 + n_a + commitment_rows.len() + n_d);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    out.extend_from_slice(commitment_rows);
    out.extend_from_slice(v);
    Ok(out)
}

/// D-free assembly of `y` from per-role flat carriers (`v` at `d_d`, commitments at `d_b`).
///
/// Each segment is validated under its role dimension before concatenation.
/// The returned [`RingVec`] uses compact mode (`ring_dim = 0`); interpret segments
/// through [`CommitmentRingDims`] when borrowing typed rows.
///
/// # Errors
///
/// Returns an error if segment lengths or role dimensions do not match `layout`.
pub fn assemble_relation_rhs<F: Field>(
    layout: &RelationRhsLayout,
    v: &RingVec<F>,
    commitment_rows: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    layout.validate()?;
    let v_rows = ring_row_count_at(v, layout.d_ring_dimension)?;
    if v_rows != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v_rows,
        });
    }
    let expected_commit_coeffs = layout.groups.iter().try_fold(0usize, |acc, group| {
        let group_coeffs = group
            .logical_b_rows()?
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs commit width overflow".into())
            })?;
        acc.checked_add(group_coeffs).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs commit length overflow".into())
        })
    })?;
    if commitment_rows.coeff_len() != expected_commit_coeffs {
        return Err(AkitaError::InvalidSize {
            expected: expected_commit_coeffs,
            actual: commitment_rows.coeff_len(),
        });
    }
    let coeff_len = relation_rhs_coeff_len(layout)?;
    let mut coeffs = Vec::with_capacity(coeff_len);
    let mut commit_offset = 0usize;
    for group in &layout.groups {
        coeffs.extend(repeat_n(
            F::zero(),
            group.opening_geometry.physical_coefficient_width(),
        ));
        let a_coeff_len = group
            .n_a
            .checked_mul(group.role_dims.d_a())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs A segment overflow".into())
            })?;
        coeffs.extend(repeat_n(F::zero(), a_coeff_len));
        let commit_coeff_len = group
            .logical_b_rows()?
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs B segment overflow".into())
            })?;
        let commit_end = commit_offset.checked_add(commit_coeff_len).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs B offset overflow".into())
        })?;
        let rows = commitment_rows
            .coeffs()
            .get(commit_offset..commit_end)
            .ok_or(AkitaError::InvalidProof)?;
        coeffs.extend_from_slice(rows);
        commit_offset = commit_end;
    }
    coeffs.extend_from_slice(v.coeffs());
    coeffs.extend(repeat_n(F::zero(), compression_rhs_coeff_len(layout)?));
    if coeffs.len() != coeff_len {
        return Err(AkitaError::InvalidSetup(
            "assembled relation rhs disagrees with its layout".into(),
        ));
    }
    Ok(RingVec::from_coeffs(coeffs))
}

/// Assemble the mandatory compressed relation RHS.
///
/// B, D, and first-map rows are zero. Only each chain's terminal map row
/// carries its 128-byte public payload.
pub fn assemble_compressed_relation_rhs<F: Field>(
    layout: &RelationRhsLayout,
    group_terminal_payloads: &[&[F]],
    opening_terminal_payload: &[F],
) -> Result<RingVec<F>, AkitaError> {
    layout.validate()?;
    let compression = layout.compression.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("relation layout has no compression geometry".into())
    })?;
    if group_terminal_payloads.len() != layout.groups.len() {
        return Err(AkitaError::InvalidSize {
            expected: layout.groups.len(),
            actual: group_terminal_payloads.len(),
        });
    }
    for (payload, plan) in group_terminal_payloads.iter().zip(&compression.group_plans) {
        if payload.len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidSize {
                expected: plan.terminal_coefficients(),
                actual: payload.len(),
            });
        }
    }
    if opening_terminal_payload.len() != compression.opening_plan.terminal_coefficients() {
        return Err(AkitaError::InvalidSize {
            expected: compression.opening_plan.terminal_coefficients(),
            actual: opening_terminal_payload.len(),
        });
    }

    let mut coefficients = Vec::with_capacity(relation_rhs_coeff_len(layout)?);
    for group in &layout.groups {
        let b_rows = group.logical_b_rows()?;
        let ordinary_coefficients = group
            .role_dims
            .d_a()
            .checked_mul(group.n_a)
            .and_then(|a| {
                group
                    .role_dims
                    .d_b()
                    .checked_mul(b_rows)
                    .and_then(|b| a.checked_add(b))
            })
            .and_then(|native| {
                native.checked_add(group.opening_geometry.physical_coefficient_width())
            })
            .ok_or_else(|| AkitaError::InvalidSetup("relation RHS width overflow".into()))?;
        coefficients.extend(repeat_n(F::zero(), ordinary_coefficients));
    }
    coefficients.extend(repeat_n(
        F::zero(),
        layout
            .n_d
            .checked_mul(layout.d_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("relation D width overflow".into()))?,
    ));
    for map_index in 0..crate::COMPRESSION_MAP_COUNT {
        for (payload, plan) in group_terminal_payloads.iter().zip(&compression.group_plans) {
            let map = plan.maps()[map_index];
            if map_index + 1 == crate::COMPRESSION_MAP_COUNT {
                coefficients.extend_from_slice(payload);
            } else {
                coefficients.extend(repeat_n(F::zero(), map.output_coefficients()));
            }
        }
        let opening_map = compression.opening_plan.maps()[map_index];
        if map_index + 1 == crate::COMPRESSION_MAP_COUNT {
            coefficients.extend_from_slice(opening_terminal_payload);
        } else {
            coefficients.extend(repeat_n(F::zero(), opening_map.output_coefficients()));
        }
    }
    let expected = relation_rhs_coeff_len(layout)?;
    if coefficients.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: coefficients.len(),
        });
    }
    Ok(RingVec::from_coeffs(coefficients))
}

fn accumulate_extension_rows<F, E, const D: usize>(
    eq_tau1: &[E],
    alpha: E,
    rows: &[CyclotomicRing<F, D>],
    row_idx: &mut usize,
    acc: &mut E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    for r in rows {
        if *row_idx >= eq_tau1.len() {
            return Ok(());
        }
        *acc += eq_tau1[*row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        *row_idx += 1;
    }
    Ok(())
}

fn accumulate_extension_flat_rows<F, E, const D: usize>(
    eq_tau1: &[E],
    alpha: E,
    coeffs: &[F],
    row_idx: &mut usize,
    acc: &mut E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    if !coeffs.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: coeffs.len(),
        });
    }
    let alpha_pows = scalar_powers(alpha, D);
    for row in coeffs.chunks_exact(D) {
        if *row_idx >= eq_tau1.len() {
            return Ok(());
        }
        let coefficients: [F; D] = row.try_into().map_err(|_| AkitaError::InvalidProof)?;
        let ring = CyclotomicRing::from_coefficients(coefficients);
        *acc += eq_tau1[*row_idx] * eval_ring_at_pows_fast(&ring, &alpha_pows);
        *row_idx += 1;
    }
    Ok(())
}

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: per-group consistency/A zero rows, B rows `u`, then D
/// rows `v`. Public openings bind through the fused trace term, not M rows.
///
/// # Errors
///
/// Returns an error if the equality table implied by `tau1` would overflow or
/// exceed the verifier sequence bound.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows")]
pub fn relation_claim_from_rows<F: Field + CanonicalEncoding, const D: usize>(
    tau1: &[F],
    alpha: F,
    n_a: usize,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError> {
    let row_count = 1usize
        .checked_add(n_a)
        .and_then(|count| count.checked_add(u.len()))
        .and_then(|count| count.checked_add(v.len()))
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
    let mut acc = F::zero();
    let mut row_idx = 1usize + n_a;

    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    Ok(acc)
}

/// Compute the stage-2 relation claim with an extension-field evaluation point.
///
/// Ring rows remain over `F`; their coefficients are multiplied into `E`
/// with mixed base-field scaling while evaluating at `alpha`.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows_extension")]
pub fn relation_claim_from_rows_extension<F, E, const D: usize>(
    tau1: &[E],
    alpha: E,
    n_a: usize,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    let row_count = 1usize
        .checked_add(n_a)
        .and_then(|count| count.checked_add(u.len()))
        .and_then(|count| count.checked_add(v.len()))
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    let mut row_idx = 1usize + n_a;

    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        row_idx += 1;
    }
    Ok(acc)
}

/// Per-role relation claim: `v` at `d_d`, commitment rows `u` at `d_b`.
///
/// Skips each group's native consistency and A rows (all zero in `y`) and
/// dispatches each public segment under its role dimension.
#[tracing::instrument(skip_all, name = "relation_claim_from_layout_extension")]
pub fn relation_claim_from_layout_extension<F, E>(
    layout: &RelationRhsLayout,
    tau1: &[E],
    alpha: E,
    v: &RingVec<F>,
    u: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    layout.validate()?;
    if !v.can_decode_vec(layout.d_ring_dimension) {
        return Err(AkitaError::InvalidSize {
            expected: layout.d_ring_dimension,
            actual: v.coeff_len(),
        });
    }
    let expected_u_coeffs = layout.groups.iter().try_fold(0usize, |acc, group| {
        let group_coeffs = group
            .logical_b_rows()?
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation claim commit width overflow".into())
            })?;
        acc.checked_add(group_coeffs)
            .ok_or_else(|| AkitaError::InvalidSetup("relation claim commit length overflow".into()))
    })?;
    if u.coeff_len() != expected_u_coeffs {
        return Err(AkitaError::InvalidSize {
            expected: expected_u_coeffs,
            actual: u.coeff_len(),
        });
    }
    if v.coeff_len() / layout.d_ring_dimension != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v.coeff_len() / layout.d_ring_dimension,
        });
    }
    let row_count = layout
        .groups
        .len()
        .checked_add(layout.n_d)
        .and_then(|count| {
            layout.groups.iter().try_fold(count, |count, group| {
                count
                    .checked_add(group.n_a)
                    .and_then(|count| count.checked_add(group.logical_b_rows().ok()?))
            })
        })
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
    let mut acc = E::zero();
    let mut row_idx = 0usize;
    let uniform_outer_dim = layout.groups.first().and_then(|first| {
        layout
            .groups
            .iter()
            .all(|group| group.role_dims.d_b() == first.role_dims.d_b())
            .then_some(first.role_dims.d_b())
    });
    if let Some(outer_dim) = uniform_outer_dim {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            F,
            outer_dim,
            |D_B| {
                let u_typed = u.as_ring_slice::<D_B>()?;
                let mut commit_offset = 0usize;
                for group in &layout.groups {
                    row_idx = row_idx
                        .checked_add(1)
                        .and_then(|row| row.checked_add(group.n_a))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("relation claim row index overflow".into())
                        })?;
                    let commit_end = commit_offset
                        .checked_add(group.logical_b_rows()?)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("relation claim commit offset overflow".into())
                        })?;
                    let rows = u_typed
                        .get(commit_offset..commit_end)
                        .ok_or(AkitaError::InvalidProof)?;
                    accumulate_extension_rows::<F, E, D_B>(
                        &eq_tau1,
                        alpha,
                        rows,
                        &mut row_idx,
                        &mut acc,
                    )?;
                    commit_offset = commit_end;
                }
                Ok::<(), AkitaError>(())
            }
        )?;
    } else {
        let mut commit_coeff_offset = 0usize;
        for group in &layout.groups {
            row_idx = row_idx
                .checked_add(1)
                .and_then(|row| row.checked_add(group.n_a))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim row index overflow".into())
                })?;
            let commit_coeff_len = group
                .logical_b_rows()?
                .checked_mul(group.role_dims.d_b())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim commit width overflow".into())
                })?;
            let commit_coeff_end = commit_coeff_offset
                .checked_add(commit_coeff_len)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim commit offset overflow".into())
                })?;
            let coeffs = u
                .coeffs()
                .get(commit_coeff_offset..commit_coeff_end)
                .ok_or(AkitaError::InvalidProof)?;
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                F,
                group.role_dims.d_b(),
                |D_B| {
                    accumulate_extension_flat_rows::<F, E, D_B>(
                        &eq_tau1,
                        alpha,
                        coeffs,
                        &mut row_idx,
                        &mut acc,
                    )
                }
            )?;
            commit_coeff_offset = commit_coeff_end;
        }
    }
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        F,
        layout.d_ring_dimension,
        |D_D| {
            let v_typed = v.as_ring_slice::<D_D>()?;
            accumulate_extension_rows::<F, E, D_D>(&eq_tau1, alpha, v_typed, &mut row_idx, &mut acc)
        }
    )?;
    Ok(acc)
}

/// Evaluate a heterogeneous canonical relation RHS at `(tau1, alpha)`.
pub fn relation_claim_from_compressed_rhs_extension<F, E>(
    layout: &RelationRhsLayout,
    tau1: &[E],
    alpha: E,
    rhs: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    relation_claim_from_rhs_matching(layout, tau1, alpha, rhs, |_| true)
}

/// Evaluate only the F/H rows of a compressed relation RHS.
pub fn compression_relation_claim_from_rhs_extension<F, E>(
    layout: &RelationRhsLayout,
    tau1: &[E],
    alpha: E,
    rhs: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    relation_claim_from_rhs_matching(layout, tau1, alpha, rhs, |family| {
        matches!(
            family,
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        )
    })
}

fn relation_claim_from_rhs_matching<F, E>(
    layout: &RelationRhsLayout,
    tau1: &[E],
    alpha: E,
    rhs: &RingVec<F>,
    include: impl Fn(RelationRowFamily) -> bool,
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field + MulBaseUnreduced<F>,
{
    let row_families = layout.row_families()?;
    if rhs.coeff_len() != relation_rhs_coeff_len(layout)? {
        return Err(AkitaError::InvalidSize {
            expected: relation_rhs_coeff_len(layout)?,
            actual: rhs.coeff_len(),
        });
    }
    let row_weights = EqPolynomial::evals_prefix(tau1, row_families.len())?;
    let mut alpha_powers = Vec::<(usize, Vec<E>)>::new();
    let mut offset = 0usize;
    let mut claim = E::zero();
    for (row_index, family) in row_families.into_iter().enumerate() {
        let geometry = family.geometry();
        let ring_dim = geometry.physical_coefficient_width();
        let end = offset
            .checked_add(ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation RHS offset overflow".into()))?;
        let row = rhs
            .coeffs()
            .get(offset..end)
            .ok_or(AkitaError::InvalidProof)?;
        if matches!(
            family,
            RelationRowFamily::Consistency {
                opening_method: crate::OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ) {
            if row.iter().any(|coefficient| !coefficient.is_zero()) {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing consistency RHS must be zero".into(),
                ));
            }
            offset = end;
            continue;
        }
        if geometry.coordinate_plane_count() != 1 {
            return Err(AkitaError::InvalidSetup(
                "non-packing relation RHS cannot use coordinate planes".into(),
            ));
        }
        let modulus_dimension = geometry.polynomial_modulus_dimension();
        let power_index = alpha_powers
            .iter()
            .position(|(dimension, _)| *dimension == modulus_dimension)
            .unwrap_or_else(|| {
                let index = alpha_powers.len();
                alpha_powers.push((modulus_dimension, scalar_powers(alpha, modulus_dimension)));
                index
            });
        let powers = &alpha_powers
            .get(power_index)
            .ok_or(AkitaError::InvalidProof)?
            .1;
        if include(family) {
            let row_evaluation = eval_flat_ring_at_pows_fast(row, powers);
            claim += *row_weights.get(row_index).ok_or(AkitaError::InvalidProof)? * row_evaluation;
        }
        offset = end;
    }
    Ok(claim)
}

/// Equality weight for one authenticated relation-row index.
pub fn relation_row_weight<E: Field>(relation_row: usize, tau1: &[E]) -> Result<E, AkitaError> {
    let num_vars = tau1.len();
    if num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidSize {
            expected: (usize::BITS as usize).saturating_sub(1),
            actual: num_vars,
        });
    }
    let domain_size = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("tau1 row-index domain overflow".to_string()))?;
    if relation_row >= domain_size {
        return Err(AkitaError::InvalidSize {
            expected: domain_size,
            actual: relation_row.saturating_add(1),
        });
    }
    Ok(eq_eval_at_index(tau1, relation_row))
}

#[cfg(test)]
#[path = "relation_tests.rs"]
mod tests;
