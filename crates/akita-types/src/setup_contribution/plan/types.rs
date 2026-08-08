use super::kernels::GroupSetupSegment;
use crate::{
    CommitmentRingDims, CommittedGroupParams, LevelParamsLike, OpeningClaimsLayout,
    SetupProjectionGeometry, WitnessLayout,
};
use akita_algebra::offset_eq::{EqPairTensorFamily, OffsetEqWindow};
use akita_field::{AkitaError, FieldCore};
use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub group_id: usize,
    pub num_claims: usize,
    pub depth_fold: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
}

pub(crate) fn validate_setup_inputs(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_layout: &WitnessLayout,
    groups: &[SetupContributionGroupInputs],
) -> Result<(), AkitaError> {
    if groups.is_empty() || groups.len() != witness_layout.num_groups() {
        return Err(AkitaError::InvalidSetup(
            "setup groups disagree with witness layout".into(),
        ));
    }
    let group_count = groups.len();
    if witness_layout
        .units()
        .iter()
        .enumerate()
        .any(|(index, unit)| {
            let expected_group = groups[index % group_count].group_id;
            unit.group_index() != expected_group || unit.chunk_index() != index / group_count
        })
    {
        return Err(AkitaError::InvalidSetup(
            "setup witness units do not follow chunk-major relation order".into(),
        ));
    }
    for group in groups {
        group.validate_against(level_params, opening_batch)?;
        validate_group_witness_layout(
            witness_layout,
            group.group_id,
            group.num_live_blocks_for(level_params, opening_batch)?,
        )?;
    }
    validate_setup_group_ids(groups, witness_layout.num_groups())
}

fn validate_setup_group_ids(
    groups: &[SetupContributionGroupInputs],
    num_groups: usize,
) -> Result<(), AkitaError> {
    let mut seen = vec![false; num_groups];
    for group in groups {
        let slot = seen
            .get_mut(group.group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(())
}

fn validate_group_witness_layout(
    layout: &WitnessLayout,
    group_id: usize,
    num_live_blocks: usize,
) -> Result<(), AkitaError> {
    let units = layout.units_for_group(group_id)?;
    let mut next_fold = 0usize;
    for unit in units {
        if unit.global_block_start() != next_fold {
            return Err(AkitaError::InvalidSetup(
                "setup witness units do not form a contiguous fold tiling".into(),
            ));
        }
        next_fold = next_fold
            .checked_add(unit.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("setup fold coverage overflow".into()))?;
    }
    if next_fold != num_live_blocks {
        return Err(AkitaError::InvalidSetup(
            "setup group dimensions disagree with witness layout".into(),
        ));
    }
    Ok(())
}

impl SetupContributionGroupInputs {
    fn group_params_for<'a>(
        &self,
        level_params: &'a CommittedGroupParams,
        opening_batch: &'a OpeningClaimsLayout,
    ) -> Result<&'a dyn LevelParamsLike, AkitaError> {
        level_params.group_params(opening_batch, self.group_id)
    }

    fn validate_against(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        let group_layout = opening_batch.group_layout(self.group_id)?;
        if self.num_claims != group_layout.num_polynomials() {
            return Err(AkitaError::InvalidSetup(
                "setup group claim count disagrees with opening batch".into(),
            ));
        }
        let n_a = self.n_a_for(level_params, opening_batch)?;
        let n_b = self.n_b(level_params, opening_batch)?;
        let a_range = level_params.a_row_range(opening_batch, self.group_id)?;
        let b_range = level_params.commitment_row_range(opening_batch, self.group_id)?;
        if a_range.start != self.a_row_start || a_range.len() != n_a {
            return Err(AkitaError::InvalidSetup(
                "setup group A row range disagrees with level params".into(),
            ));
        }
        if b_range.start != self.b_row_start || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "setup group B row range disagrees with level params".into(),
            ));
        }
        Ok(())
    }

    fn num_live_blocks_for(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_live_blocks())
    }

    fn n_a_for(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .a_rows_len())
    }

    pub(crate) fn num_live_blocks(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_live_blocks())
    }

    pub(crate) fn num_positions_per_block(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_positions_per_block())
    }

    pub(crate) fn depth_witness(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_inner())
    }

    pub(crate) fn depth_commit(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_outer())
    }

    pub(crate) fn depth_open(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_open())
    }

    pub(crate) fn log_basis_open(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<u32, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .log_basis_open())
    }

    pub(crate) fn n_a(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .a_rows_len())
    }

    pub(crate) fn n_b(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .b_rows_len())
    }

    pub(crate) fn t_vector_width(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        let n_a = self.n_a(level_params, opening_batch)?;
        let depth_commit = self.depth_commit(level_params, opening_batch)?;
        let num_live_blocks = self.num_live_blocks(level_params, opening_batch)?;
        n_a.checked_mul(depth_commit)
            .and_then(|n| n.checked_mul(num_live_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("setup B vector width overflow".into()))
    }

    pub(crate) fn d_active_cols(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        let num_live_blocks = self.num_live_blocks(level_params, opening_batch)?;
        let depth_open = self.depth_open(level_params, opening_batch)?;
        self.num_claims
            .checked_mul(num_live_blocks)
            .and_then(|cols| cols.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D active width overflow".into()))
    }
}

/// Checked relation-address point and its bounded equality window.
///
/// Clones share both allocations. Keeping the point and window in one value
/// prevents setup planning from consuming a window prepared for different
/// challenges.
#[derive(Clone)]
pub struct PreparedRelationAddress<E: FieldCore> {
    pub(crate) point: Arc<[E]>,
    pub(crate) equality_window: Arc<OffsetEqWindow<E>>,
}

impl<E: FieldCore> PreparedRelationAddress<E> {
    /// Prepare the reusable equality state for one relation-address point.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded equality window cannot be constructed.
    pub fn new(point: &[E]) -> Result<Self, AkitaError> {
        Ok(Self {
            point: point.to_vec().into(),
            equality_window: Arc::new(OffsetEqWindow::new(point)?),
        })
    }

    /// Relation lane-and-column challenges in LSB-first order.
    #[must_use]
    pub fn point(&self) -> &[E] {
        &self.point
    }

    /// Equality window prepared from [`Self::point`].
    #[must_use]
    pub fn equality_window(&self) -> &OffsetEqWindow<E> {
        &self.equality_window
    }
}

pub struct SetupContributionPlan<E: FieldCore> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) d_weights: Arc<[E]>,
    pub(crate) setup_index_tensors: Vec<ProjectedEqPairTensor<E>>,
    pub(crate) relation_address: PreparedRelationAddress<E>,
    pub(crate) relation_address_geometry: crate::RelationAddressGeometry,
    pub(crate) projection_geometry: SetupProjectionGeometry,
    pub(crate) direct_scan_alpha: Option<E>,
}

pub(crate) struct ProjectedEqPairTensorBatch<E: FieldCore> {
    pub(crate) ratio: usize,
    pub(crate) families: Vec<EqPairTensorFamily<E>>,
}

pub(crate) enum ProjectedEqPairTensor<E: FieldCore> {
    Native(ProjectedEqPairTensorBatch<E>),
    RelationFactored(ProjectedEqPairTensorBatch<E>),
}

impl<E: FieldCore> ProjectedEqPairTensor<E> {
    pub(crate) fn ratio(&self) -> usize {
        match self {
            Self::Native(batch) | Self::RelationFactored(batch) => batch.ratio,
        }
    }

    pub(crate) fn families(&self) -> &[EqPairTensorFamily<E>] {
        match self {
            Self::Native(batch) | Self::RelationFactored(batch) => &batch.families,
        }
    }
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Equality window shared by every direct contribution over this opening point.
    #[must_use]
    pub fn eq_window(&self) -> &OffsetEqWindow<E> {
        self.relation_address.equality_window()
    }

    /// Prepared D/B/A column equality slices for `group_id`.
    ///
    /// The D-role slice is laid out
    /// `(claim, block, opening_subcolumn, opening_digit)`, the B-role slice
    /// `(claim, block, A_row, outer_subcolumn, commit_digit)`, and the A-role
    /// slice `(position, witness_digit)` after contraction over units and fold
    /// digits. Subcolumn axes have length one for uniform roles.
    /// The direct ring-switch verifier reuses all three instead of evaluating
    /// the same opening equality addresses a second time.
    #[must_use]
    pub fn group_column_eq_slices(&self, group_id: usize) -> Option<(&[E], &[E], &[E])> {
        self.groups
            .iter()
            .find(|group| group.group_id == group_id)
            .and_then(SetupContributionGroupPlan::column_eq_slices)
    }
}

pub(crate) struct DirectScanWeights<E> {
    pub(crate) e: Vec<E>,
    pub(crate) t: Vec<E>,
    pub(crate) z: Vec<E>,
}

type ColumnEqSlices<'a, E> = (&'a [E], &'a [E], &'a [E]);

pub(crate) struct SetupContributionGroupPlan<E: FieldCore> {
    pub(crate) group_id: usize,
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) a_ratio: usize,
    pub(crate) b_ratio: usize,
    pub(crate) d_ratio: usize,
    pub(crate) consistency_weight: E,
    pub(crate) num_claims: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) num_positions_per_block: usize,
    pub(crate) depth_witness: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_open: usize,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_outer: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) d_col_range: Range<usize>,
    pub(crate) t_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) required: usize,
    pub(crate) segments: Arc<[GroupSetupSegment<E>]>,
    pub(crate) a_row_weights: Arc<[E]>,
    pub(crate) b_weights: Arc<[E]>,
    pub(crate) fold_gadget: Arc<[E]>,
    pub(crate) direct_scan_weights: Option<DirectScanWeights<E>>,
    pub(crate) unit_partition: Arc<[SetupUnitRange]>,
    pub(crate) d_tensors: Vec<EqPairTensorFamily<E>>,
    pub(crate) b_tensors: Vec<EqPairTensorFamily<E>>,
    pub(crate) a_tensors: Vec<EqPairTensorFamily<E>>,
}

#[derive(Clone, Copy)]
pub(crate) struct SetupUnitRange {
    pub(crate) global_block_start: usize,
    pub(crate) num_live_blocks: usize,
}

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    pub(crate) fn column_eq_slices(&self) -> Option<(&[E], &[E], &[E])> {
        self.direct_scan_weights
            .as_ref()
            .map(|weights| (&weights.e[..], &weights.t[..], &weights.z[..]))
    }

    pub(crate) fn require_column_eq_slices(&self) -> Result<ColumnEqSlices<'_, E>, AkitaError> {
        self.column_eq_slices().ok_or_else(|| {
            AkitaError::InvalidSetup(
                "direct setup operation requires prepared column weights".into(),
            )
        })
    }

    pub(crate) fn set_projection_ratios(&mut self, base_ring_dim: usize) -> Result<(), AkitaError> {
        let ratio = |name: &'static str, dimension: usize| {
            dimension
                .checked_div(base_ring_dim)
                .filter(|ratio| {
                    base_ring_dim != 0
                        && dimension.is_multiple_of(base_ring_dim)
                        && ratio.is_power_of_two()
                })
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "setup {name} dimension does not project to the shared base"
                    ))
                })
        };
        self.a_ratio = ratio("A", self.role_dims.d_a())?;
        self.b_ratio = ratio("B", self.role_dims.d_b())?;
        self.d_ratio = ratio("D", self.role_dims.d_d())?;
        Ok(())
    }
}
