//! Checked construction of the relation rhs layout and witness geometry.

use akita_error::AkitaError;
use std::iter::repeat_n;

use crate::layout::opening_layout::OpeningClaimsLayout;
use crate::layout::relation_layout::{
    RelationCompressionLayout, RelationGroupRows, RelationRhsLayout, RelationRowFamily,
    RelationRowGeometry, RelationWitnessGeometry,
};
use crate::layout::CommittedGroupParams;
use crate::{
    CommitmentRingDims, CommitmentSliceCount, CommittedSourceEncoding, CompressionChainPlan,
    OpeningMethod, SisModulusProfileId, SubringCoefficientPackingGeometry,
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

    pub(crate) fn validate(&self) -> Result<(), AkitaError> {
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

pub(crate) fn compression_plan(
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
}
