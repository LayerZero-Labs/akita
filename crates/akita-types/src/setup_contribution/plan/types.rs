use super::kernels::GroupSetupSegment;
use crate::{
    CommitmentRingDims, CommittedGroupParams, LevelParamsLike, OpeningClaimsLayout,
    SetupProjectionGeometry, WitnessLayout,
};
use akita_algebra::offset_eq::OffsetEqWindow;
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
    let witness_group_order = witness_layout
        .units()
        .iter()
        .map(|unit| unit.group_index())
        .fold(Vec::new(), |mut order, group_id| {
            if order.last() != Some(&group_id) {
                order.push(group_id);
            }
            order
        });
    if witness_group_order
        != groups
            .iter()
            .map(|group| group.group_id)
            .collect::<Vec<_>>()
    {
        return Err(AkitaError::InvalidSetup(
            "setup groups do not follow witness relation order".into(),
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
        if unit.num_live_blocks() == 0 || unit.global_block_start() != next_fold {
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

pub struct SetupContributionPlan<E: FieldCore> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) d_weights: Arc<[E]>,
    pub(crate) address_point: Arc<[E]>,
    pub(crate) relation_address_geometry: crate::RelationAddressGeometry,
    pub(crate) projection_geometry: SetupProjectionGeometry,
    pub(crate) eq_window: OffsetEqWindow<E>,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Equality window shared by every direct contribution over this opening point.
    #[must_use]
    pub fn eq_window(&self) -> &OffsetEqWindow<E> {
        &self.eq_window
    }

    /// Prepared D/B/A column equality slices for `group_id`.
    ///
    /// The D-role slice is laid out
    /// `(claim, block, opening_subcolumn, opening_digit)`, the B-role slice
    /// `(claim, block, A_row, commit_digit, outer_subcolumn)`, and the A-role
    /// slice `(position, witness_digit)` after contraction over units and fold
    /// digits. Subcolumn axes have length one for uniform roles.
    /// The direct ring-switch verifier reuses all three instead of evaluating
    /// the same opening equality addresses a second time.
    #[must_use]
    pub fn group_column_eq_slices(&self, group_id: usize) -> Option<(&[E], &[E], &[E])> {
        let group = self
            .groups
            .iter()
            .find(|group| group.group_id == group_id)
            .filter(|group| {
                (group.e_eq_slice.is_empty() || !group.d_spans.is_empty())
                    && (group.t_eq_slice.is_empty() || !group.b_spans.is_empty())
                    && (group.z_eq_slice.is_empty() || !group.a_families.is_empty())
            })?;
        Some((
            group.e_eq_slice.as_slice(),
            group.t_eq_slice.as_slice(),
            group.z_eq_slice.as_slice(),
        ))
    }
}

/// A strided family of setup columns and their corresponding relation lanes.
///
/// Every occurrence maps one physical setup column to
/// `relation_lane_count` consecutive lanes. The lanes are folded by the
/// ring-switch challenge when column equality weights are materialized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SetupContributionSpan {
    pub(crate) setup_column_start: usize,
    pub(crate) setup_column_stride: usize,
    pub(crate) relation_lane_start: usize,
    pub(crate) relation_lane_stride: usize,
    pub(crate) occurrence_count: usize,
    pub(crate) relation_lane_count: usize,
    pub(crate) fold_count: usize,
    pub(crate) fold_lane_stride: usize,
}

impl SetupContributionSpan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        setup_column_start: usize,
        setup_column_stride: usize,
        relation_lane_start: usize,
        relation_lane_stride: usize,
        occurrence_count: usize,
        relation_lane_count: usize,
        setup_column_count: usize,
        relation_lane_capacity: usize,
    ) -> Result<Self, AkitaError> {
        if setup_column_stride == 0 || relation_lane_stride == 0 || relation_lane_count == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup contribution span strides and lane count must be positive".into(),
            ));
        }
        if occurrence_count != 0 {
            let last_setup_column =
                checked_span_index(setup_column_start, setup_column_stride, occurrence_count)?;
            if last_setup_column >= setup_column_count {
                return Err(AkitaError::InvalidSetup(
                    "setup contribution span exceeds role columns".into(),
                ));
            }
            let last_relation_lane =
                checked_span_index(relation_lane_start, relation_lane_stride, occurrence_count)?;
            let relation_lane_end = last_relation_lane
                .checked_add(relation_lane_count)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup contribution relation span overflow".into())
                })?;
            if relation_lane_end > relation_lane_capacity {
                return Err(AkitaError::InvalidSetup(
                    "setup contribution span exceeds relation lane domain".into(),
                ));
            }
        }
        Ok(Self {
            setup_column_start,
            setup_column_stride,
            relation_lane_start,
            relation_lane_stride,
            occurrence_count,
            relation_lane_count,
            fold_count: 1,
            fold_lane_stride: relation_lane_count,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_fold_family(
        setup_column_start: usize,
        setup_column_stride: usize,
        relation_lane_start: usize,
        relation_lane_stride: usize,
        occurrence_count: usize,
        relation_lane_count: usize,
        fold_count: usize,
        fold_lane_stride: usize,
        setup_column_count: usize,
        relation_lane_capacity: usize,
    ) -> Result<Self, AkitaError> {
        if fold_count == 0 || fold_lane_stride < relation_lane_count {
            return Err(AkitaError::InvalidSetup(
                "setup A fold family geometry must be non-empty and disjoint".into(),
            ));
        }
        let mut family = Self::new(
            setup_column_start,
            setup_column_stride,
            relation_lane_start,
            relation_lane_stride,
            occurrence_count,
            relation_lane_count,
            setup_column_count,
            relation_lane_capacity,
        )?;
        let fold_end = fold_count
            .checked_sub(1)
            .and_then(|last| last.checked_mul(fold_lane_stride))
            .and_then(|offset| offset.checked_add(relation_lane_count))
            .ok_or_else(|| AkitaError::InvalidSetup("setup A fold family overflow".into()))?;
        if fold_end > relation_lane_stride {
            return Err(AkitaError::InvalidSetup(
                "setup A fold family exceeds its occurrence stride".into(),
            ));
        }
        if occurrence_count != 0 {
            let last_start =
                checked_span_index(relation_lane_start, relation_lane_stride, occurrence_count)?;
            let last_end = last_start
                .checked_add(fold_end)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A fold family overflow".into()))?;
            if last_end > relation_lane_capacity {
                return Err(AkitaError::InvalidSetup(
                    "setup A fold family exceeds relation lane domain".into(),
                ));
            }
        }
        family.fold_count = fold_count;
        family.fold_lane_stride = fold_lane_stride;
        Ok(family)
    }

    pub(crate) fn occurrences(
        &self,
    ) -> impl Iterator<Item = Result<(usize, usize), AkitaError>> + '_ {
        (0..self.occurrence_count).map(|occurrence| {
            let setup_column = self
                .setup_column_stride
                .checked_mul(occurrence)
                .and_then(|offset| self.setup_column_start.checked_add(offset))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup contribution span overflow".into())
                })?;
            let relation_lane = self
                .relation_lane_stride
                .checked_mul(occurrence)
                .and_then(|offset| self.relation_lane_start.checked_add(offset))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup contribution relation span overflow".into())
                })?;
            Ok((setup_column, relation_lane))
        })
    }

    pub(crate) fn relation_lane_start_for_setup_column(
        &self,
        setup_column: usize,
    ) -> Result<Option<usize>, AkitaError> {
        let Some(delta) = setup_column.checked_sub(self.setup_column_start) else {
            return Ok(None);
        };
        let occurrence = if self.setup_column_stride == 1 {
            delta
        } else if delta.is_multiple_of(self.setup_column_stride) {
            delta / self.setup_column_stride
        } else {
            return Ok(None);
        };
        if occurrence >= self.occurrence_count {
            return Ok(None);
        }
        self.relation_lane_stride
            .checked_mul(occurrence)
            .and_then(|offset| self.relation_lane_start.checked_add(offset))
            .map(Some)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup contribution relation span overflow".into())
            })
    }
}

fn checked_span_index(start: usize, stride: usize, len: usize) -> Result<usize, AkitaError> {
    let last = len.checked_sub(1).ok_or_else(|| {
        AkitaError::InvalidSetup("setup contribution span length must be positive".into())
    })?;
    stride
        .checked_mul(last)
        .and_then(|offset| start.checked_add(offset))
        .ok_or_else(|| AkitaError::InvalidSetup("setup contribution span overflow".into()))
}

pub(crate) struct SetupContributionGroupPlan<E> {
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
    pub(crate) e_eq_slice: Vec<E>,
    pub(crate) t_eq_slice: Vec<E>,
    pub(crate) z_eq_slice: Vec<E>,
    pub(crate) d_spans: Vec<SetupContributionSpan>,
    pub(crate) b_spans: Vec<SetupContributionSpan>,
    pub(crate) a_families: Vec<SetupContributionSpan>,
}

impl<E: FieldCore> SetupContributionGroupPlan<E> {
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
