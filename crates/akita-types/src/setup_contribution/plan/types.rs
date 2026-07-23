use crate::{
    CommittedGroupParams, LevelParamsLike, OpeningClaimsLayout, SetupProjectionGeometry,
    WitnessLayout,
};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::{AkitaError, FieldCore};
use std::{ops::Range, sync::Arc};

use super::GroupSetupSegment;

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

pub(crate) fn get_d_col_range(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[SetupContributionGroupInputs],
    group_id: usize,
) -> Result<Range<usize>, AkitaError> {
    let mut cursor = 0usize;
    for group in groups {
        let width = group.d_active_cols(level_params, opening_batch)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        if group.group_id == group_id {
            return Ok(cursor..end);
        }
        cursor = end;
    }
    Err(AkitaError::InvalidSetup("setup D group is missing".into()))
}

pub(crate) fn get_total_d(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[SetupContributionGroupInputs],
) -> Result<usize, AkitaError> {
    groups.iter().try_fold(0usize, |cursor, group| {
        let width = group.d_active_cols(level_params, opening_batch)?;
        cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))
    })
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
    pub(crate) eq_tau1: Arc<[E]>,
    pub(crate) x_challenges: Arc<[E]>,
    pub(crate) fold_gadget: Arc<[E]>,
    pub(crate) outgoing_ring_dim: usize,
    pub(crate) common_coeff_count: usize,
    pub(crate) inner_lane_count: usize,
    pub(crate) outer_lane_count: usize,
    pub(crate) opening_lane_count: usize,
    pub(crate) d_row_start: usize,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) d_weights: Arc<[E]>,
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
    #[must_use]
    pub fn group_column_eq_slices(&self, group_id: usize) -> Option<(&[E], &[E], &[E])> {
        self.groups
            .iter()
            .find(|group| group.group_id == group_id)
            .map(|group| {
                (
                    group.e_eq_slice.as_slice(),
                    group.t_eq_slice.as_slice(),
                    group.z_eq_slice.as_slice(),
                )
            })
    }
}

#[derive(Clone)]
pub(crate) struct SetupContributionSpan {
    pub(crate) setup_start: usize,
    pub(crate) setup_stride: usize,
    pub(crate) witness_start: usize,
    pub(crate) witness_stride: usize,
    pub(crate) len: usize,
    pub(crate) fold_digit: Option<usize>,
}

impl SetupContributionSpan {
    pub(crate) fn new(
        setup_start: usize,
        setup_stride: usize,
        witness_start: usize,
        witness_stride: usize,
        len: usize,
        fold_digit: Option<usize>,
        setup_len: usize,
        witness_len: usize,
        witness_width: usize,
    ) -> Result<Self, AkitaError> {
        if setup_stride == 0 || witness_stride == 0 || witness_width == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup contribution span stride and witness width must be positive".into(),
            ));
        }
        if len != 0 {
            let last_setup = checked_span_index(setup_start, setup_stride, len)?;
            if last_setup >= setup_len {
                return Err(AkitaError::InvalidSetup(
                    "setup contribution span exceeds role columns".into(),
                ));
            }
            let last_witness = checked_span_index(witness_start, witness_stride, len)?;
            let witness_end = last_witness
                .checked_add(witness_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup contribution witness span overflow".into())
                })?;
            if witness_end > witness_len {
                return Err(AkitaError::InvalidSetup(
                    "setup contribution span exceeds relation address domain".into(),
                ));
            }
        }
        Ok(Self {
            setup_start,
            setup_stride,
            witness_start,
            witness_stride,
            len,
            fold_digit,
        })
    }

    #[inline]
    fn witness_index_for_setup(&self, setup_index: usize) -> Result<Option<usize>, AkitaError> {
        let Some(delta) = setup_index.checked_sub(self.setup_start) else {
            return Ok(None);
        };
        let offset = if self.setup_stride == 1 {
            delta
        } else if delta.is_multiple_of(self.setup_stride) {
            delta / self.setup_stride
        } else {
            return Ok(None);
        };
        if offset >= self.len {
            return Ok(None);
        }
        let witness_offset = offset.checked_mul(self.witness_stride).ok_or_else(|| {
            AkitaError::InvalidSetup("setup contribution witness span overflow".into())
        })?;
        self.witness_start
            .checked_add(witness_offset)
            .map(Some)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup contribution witness span overflow".into())
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

#[derive(Clone)]
pub(crate) struct SetupContributionGroupPlan<E> {
    pub(crate) group_id: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
    pub(crate) d_col_range: Range<usize>,
    pub(crate) d_native_col_range: Range<usize>,
    pub(crate) t_cols: usize,
    pub(crate) b_native_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) required: usize,
    pub(crate) segments: Arc<[GroupSetupSegment<E>]>,
    pub(crate) a_row_weights: Arc<[E]>,
    pub(crate) b_weights: Arc<[E]>,
    pub(crate) e_eq_slice: Vec<E>,
    pub(crate) t_eq_slice: Vec<E>,
    pub(crate) z_eq_slice: Vec<E>,
    pub(crate) d_spans: Vec<SetupContributionSpan>,
    pub(crate) b_spans: Vec<SetupContributionSpan>,
    pub(crate) a_spans: Vec<SetupContributionSpan>,
}

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    pub(crate) fn d_eq_at(
        &self,
        column: usize,
        eq_window: &OffsetEqWindow<E>,
    ) -> Result<E, AkitaError> {
        if column >= self.d_col_range.len() {
            return Err(AkitaError::InvalidProof);
        }
        for span in &self.d_spans {
            if let Some(witness_index) = span.witness_index_for_setup(column)? {
                return Ok(eq_window.eval(witness_index));
            }
        }
        Ok(E::zero())
    }

    pub(crate) fn b_eq_at(
        &self,
        column: usize,
        eq_window: &OffsetEqWindow<E>,
    ) -> Result<E, AkitaError> {
        if column >= self.t_cols {
            return Err(AkitaError::InvalidProof);
        }
        for span in &self.b_spans {
            if let Some(witness_index) = span.witness_index_for_setup(column)? {
                return Ok(eq_window.eval(witness_index));
            }
        }
        Ok(E::zero())
    }

    pub(crate) fn a_eq_at(
        &self,
        column: usize,
        eq_window: &OffsetEqWindow<E>,
        fold_gadget: &[E],
    ) -> Result<E, AkitaError> {
        if column >= self.z_cols {
            return Err(AkitaError::InvalidProof);
        }
        let mut weight = E::zero();
        for span in &self.a_spans {
            if let Some(witness_index) = span.witness_index_for_setup(column)? {
                let fold_digit = span.fold_digit.ok_or(AkitaError::InvalidProof)?;
                let fold = *fold_gadget
                    .get(fold_digit)
                    .ok_or(AkitaError::InvalidProof)?;
                weight -= eq_window.eval(witness_index) * fold;
            }
        }
        Ok(weight)
    }
}
