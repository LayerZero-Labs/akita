use crate::{CommittedGroupParams, OpeningClaimsLayout, SetupProjectionGeometry, WitnessLayout};
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_field::{AkitaError, FieldCore};
use std::{ops::Range, sync::Arc};

#[cfg(test)]
type TestColumnEqSlices<E> = (Vec<E>, Vec<E>, Vec<E>);
pub(crate) fn validate_setup_inputs(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_layout: &WitnessLayout,
) -> Result<Vec<usize>, AkitaError> {
    let groups = opening_batch.root_group_order()?;
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
    if witness_group_order != groups {
        return Err(AkitaError::InvalidSetup(
            "setup groups do not follow witness relation order".into(),
        ));
    }
    let mut seen = vec![false; witness_layout.num_groups()];
    for &group_id in &groups {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let num_live_blocks = level_params
            .group_params(opening_batch, group_id)?
            .num_live_blocks();
        validate_group_witness_layout(witness_layout, group_id, num_live_blocks)?;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(groups)
}

pub(crate) fn get_d_col_range(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[usize],
    group_id: usize,
) -> Result<Range<usize>, AkitaError> {
    let mut cursor = 0usize;
    for &candidate in groups {
        let width = d_active_cols(level_params, opening_batch, candidate)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        if candidate == group_id {
            return Ok(cursor..end);
        }
        cursor = end;
    }
    Err(AkitaError::InvalidSetup("setup D group is missing".into()))
}

pub(crate) fn get_total_d(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[usize],
) -> Result<usize, AkitaError> {
    groups.iter().try_fold(0usize, |cursor, &group_id| {
        let width = d_active_cols(level_params, opening_batch, group_id)?;
        cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))
    })
}

pub(crate) fn d_active_cols(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    group_id: usize,
) -> Result<usize, AkitaError> {
    let group_layout = opening_batch.group_layout(group_id)?;
    let group_params = level_params.group_params(opening_batch, group_id)?;
    group_layout
        .num_polynomials()
        .checked_mul(group_params.num_live_blocks())
        .and_then(|cols| cols.checked_mul(group_params.num_digits_open()))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D active width overflow".into()))
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
pub struct SetupContributionPlan<E: FieldCore> {
    pub(crate) groups: Vec<SetupContributionGroupPlan>,
    pub(crate) eq_tau1: Arc<[E]>,
    pub(crate) x_challenges: Arc<[E]>,
    pub(crate) fold_gadget: Arc<[E]>,
    pub(crate) common_coeff_count: usize,
    pub(crate) inner_lane_count: usize,
    pub(crate) outer_lane_count: usize,
    pub(crate) opening_lane_count: usize,
    pub(crate) d_row_start: usize,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) projection_geometry: SetupProjectionGeometry,
    pub(crate) eq_window: OffsetEqWindow<E>,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Equality window shared by every direct contribution over this opening point.
    #[must_use]
    pub fn eq_window(&self) -> &OffsetEqWindow<E> {
        &self.eq_window
    }

    #[cfg(test)]
    pub(crate) fn group_column_eq_slices_for_test(
        &self,
        group_id: usize,
        alpha: E,
    ) -> Result<TestColumnEqSlices<E>, AkitaError> {
        let group = self
            .groups
            .iter()
            .find(|group| group.group_id == group_id)
            .ok_or(AkitaError::InvalidProof)?;
        let inner = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.inner_lane_count,
        )?;
        let outer = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.outer_lane_count,
        )?;
        let opening = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.opening_lane_count,
        )?;
        Ok((
            super::scan::materialize_role_columns(
                &group.d_spans,
                group.d_col_range.len(),
                &self.eq_window,
                &opening,
                None,
            )?,
            super::scan::materialize_role_columns(
                &group.b_spans,
                group.t_cols,
                &self.eq_window,
                &outer,
                None,
            )?,
            super::scan::materialize_role_columns(
                &group.a_spans,
                group.z_cols,
                &self.eq_window,
                &inner,
                Some(&self.fold_gadget),
            )?,
        ))
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
    #[allow(clippy::too_many_arguments)]
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
            let witness_end = last_witness.checked_add(witness_width).ok_or_else(|| {
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
    pub(crate) fn witness_index_for_setup(
        &self,
        setup_index: usize,
    ) -> Result<Option<usize>, AkitaError> {
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
pub(crate) struct SetupContributionGroupPlan {
    pub(crate) group_id: usize,
    pub(crate) num_claims: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) num_positions_per_block: usize,
    pub(crate) depth_witness: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_fold: usize,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_outer: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
    pub(crate) d_col_range: Range<usize>,
    pub(crate) t_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) d_spans: Vec<SetupContributionSpan>,
    pub(crate) b_spans: Vec<SetupContributionSpan>,
    pub(crate) a_spans: Vec<SetupContributionSpan>,
}
