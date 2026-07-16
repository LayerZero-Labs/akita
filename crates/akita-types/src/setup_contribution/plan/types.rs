use super::kernels::GroupSetupSegment;
use crate::{
    LevelParams, LevelParamsLike, OpeningClaimsLayout, RelationMatrixRowLayout,
    SetupProjectionGeometry, WitnessLayout,
};
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

/// Canonical setup-side view of witness ownership and D column placement.
///
/// Group descriptors carry only relation-order anchors. This shared layout is
/// the sole owner of the root parameters, opening-batch geometry, witness
/// ranges, and opening-source length used to derive setup contribution
/// geometry.
#[derive(Clone)]
pub struct SetupContributionLayout {
    level_params: Arc<LevelParams>,
    opening_batch: Arc<OpeningClaimsLayout>,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    witness_layout: Arc<WitnessLayout>,
    opening_source_len: usize,
    groups: Vec<SetupContributionGroupInputs>,
}

impl SetupContributionLayout {
    pub fn new(
        level_params: Arc<LevelParams>,
        opening_batch: Arc<OpeningClaimsLayout>,
        relation_matrix_row_layout: RelationMatrixRowLayout,
        witness_layout: Arc<WitnessLayout>,
        opening_source_len: usize,
        groups: Vec<SetupContributionGroupInputs>,
    ) -> Result<Self, AkitaError> {
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
        for group in &groups {
            group.validate_against(&level_params, &opening_batch, relation_matrix_row_layout)?;
            validate_group_witness_layout(
                &witness_layout,
                group.group_id,
                group.num_blocks_for(&level_params, &opening_batch)?,
            )?;
        }
        validate_setup_group_ids(&groups, witness_layout.num_groups())?;
        Ok(Self {
            level_params,
            opening_batch,
            relation_matrix_row_layout,
            witness_layout,
            opening_source_len,
            groups,
        })
    }

    #[must_use]
    pub fn level_params(&self) -> &LevelParams {
        &self.level_params
    }

    #[must_use]
    pub fn opening_batch(&self) -> &OpeningClaimsLayout {
        &self.opening_batch
    }

    #[must_use]
    pub const fn relation_matrix_row_layout(&self) -> RelationMatrixRowLayout {
        self.relation_matrix_row_layout
    }

    #[must_use]
    pub fn witness_layout(&self) -> &WitnessLayout {
        &self.witness_layout
    }

    #[must_use]
    pub fn opening_source_len(&self) -> usize {
        self.opening_source_len
    }

    #[must_use]
    pub fn groups(&self) -> &[SetupContributionGroupInputs] {
        &self.groups
    }

    pub fn get_d_col_range(&self, group_id: usize) -> Result<Range<usize>, AkitaError> {
        let mut cursor = 0usize;
        for group in &self.groups {
            let width = group.d_active_cols(self)?;
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

    pub fn get_total_d(&self) -> Result<usize, AkitaError> {
        self.groups.iter().try_fold(0usize, |cursor, group| {
            let width = group.d_active_cols(self)?;
            cursor
                .checked_add(width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))
        })
    }
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
    num_blocks: usize,
) -> Result<(), AkitaError> {
    let units = layout.units_for_group(group_id)?;
    let mut next_fold = 0usize;
    for unit in units {
        if unit.live_block_count() == 0 || unit.global_block_start() != next_fold {
            return Err(AkitaError::InvalidSetup(
                "setup witness units do not form a contiguous fold tiling".into(),
            ));
        }
        next_fold = next_fold
            .checked_add(unit.live_block_count())
            .ok_or_else(|| AkitaError::InvalidSetup("setup fold coverage overflow".into()))?;
    }
    if next_fold != num_blocks {
        return Err(AkitaError::InvalidSetup(
            "setup group dimensions disagree with witness layout".into(),
        ));
    }
    Ok(())
}

impl SetupContributionGroupInputs {
    fn group_params_for<'a>(
        &self,
        level_params: &'a LevelParams,
        opening_batch: &'a OpeningClaimsLayout,
    ) -> Result<&'a dyn LevelParamsLike, AkitaError> {
        level_params.group_params(opening_batch, self.group_id)
    }

    fn validate_against(
        &self,
        level_params: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
        relation_matrix_row_layout: RelationMatrixRowLayout,
    ) -> Result<(), AkitaError> {
        let group_layout = opening_batch.group_layout(self.group_id)?;
        if self.num_claims != group_layout.num_polynomials() {
            return Err(AkitaError::InvalidSetup(
                "setup group claim count disagrees with opening batch".into(),
            ));
        }
        let n_a = self.n_a_for(level_params, opening_batch)?;
        let n_b = self.n_b_for(level_params, opening_batch)?;
        let a_range =
            level_params.a_row_range(opening_batch, self.group_id, relation_matrix_row_layout)?;
        let b_range = level_params.commitment_row_range(
            opening_batch,
            self.group_id,
            relation_matrix_row_layout,
        )?;
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

    fn num_blocks_for(
        &self,
        level_params: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_blocks())
    }

    fn n_a_for(
        &self,
        level_params: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .a_rows_len())
    }

    fn n_b_for(
        &self,
        level_params: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .b_rows_len())
    }

    fn group_params<'a>(
        &self,
        layout: &'a SetupContributionLayout,
    ) -> Result<&'a dyn LevelParamsLike, AkitaError> {
        self.group_params_for(layout.level_params(), layout.opening_batch())
    }

    pub(crate) fn num_blocks(&self, layout: &SetupContributionLayout) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.num_blocks())
    }

    pub(crate) fn block_len(&self, layout: &SetupContributionLayout) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.block_len())
    }

    pub(crate) fn depth_open(&self, layout: &SetupContributionLayout) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.num_digits_open())
    }

    pub(crate) fn depth_commit(
        &self,
        layout: &SetupContributionLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.num_digits_commit())
    }

    pub(crate) fn log_basis(&self, layout: &SetupContributionLayout) -> Result<u32, AkitaError> {
        Ok(self.group_params(layout)?.log_basis())
    }

    pub(crate) fn n_a(&self, layout: &SetupContributionLayout) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.a_rows_len())
    }

    pub(crate) fn n_b(&self, layout: &SetupContributionLayout) -> Result<usize, AkitaError> {
        Ok(self.group_params(layout)?.b_rows_len())
    }

    pub(crate) fn t_cols_per_vector(
        &self,
        layout: &SetupContributionLayout,
    ) -> Result<usize, AkitaError> {
        let n_a = self.n_a(layout)?;
        let depth_open = self.depth_open(layout)?;
        let num_blocks = self.num_blocks(layout)?;
        n_a.checked_mul(depth_open)
            .and_then(|n| n.checked_mul(num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("setup B vector width overflow".into()))
    }

    pub(crate) fn d_active_cols(
        &self,
        layout: &SetupContributionLayout,
    ) -> Result<usize, AkitaError> {
        let num_blocks = self.num_blocks(layout)?;
        let depth_open = self.depth_open(layout)?;
        self.num_claims
            .checked_mul(num_blocks)
            .and_then(|cols| cols.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D active width overflow".into()))
    }
}

pub struct SetupContributionPlan<E> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) projection_geometry: SetupProjectionGeometry,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Prepared A-role (Z) column equality slice for the group at `index` in
    /// plan (witness relation) order, laid out as
    /// `z_eq_slice[position * depth_commit + commit_digit]` and already
    /// contracted over units and fold digits.
    ///
    /// Exposed so the ring-switch verifier can reuse this slice for the
    /// structured Z relation contribution instead of recomputing the same
    /// equality evaluations, per the setup-contribution reuse in Fix 6.
    #[must_use]
    pub fn group_z_eq_slice(&self, index: usize) -> Option<&[E]> {
        self.groups
            .get(index)
            .map(|group| group.z_eq_slice.as_slice())
    }
}

/// Tau1-derived setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct SetupContributionStatic<E> {
    pub(super) level_params: LevelParams,
    pub(super) opening_batch: OpeningClaimsLayout,
    pub(super) relation_matrix_row_layout: RelationMatrixRowLayout,
    pub(super) rows: usize,
    pub(super) eq_tau1: Arc<[E]>,
    pub(super) groups: Vec<SetupContributionGroupStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Arc<[E]>,
}

impl<E> SetupContributionStatic<E> {
    /// Relation-matrix row count covered by the cached tau1 equality table.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Expanded tau1 equality table used for setup row weights.
    #[must_use]
    pub fn eq_tau1(&self) -> &[E] {
        &self.eq_tau1
    }

    /// Level parameters used to prepare this static setup contribution.
    #[must_use]
    pub fn level_params(&self) -> &LevelParams {
        &self.level_params
    }

    /// Opening batch layout used to prepare this static setup contribution.
    #[must_use]
    pub fn opening_batch(&self) -> &OpeningClaimsLayout {
        &self.opening_batch
    }

    /// Relation row layout used to prepare this static setup contribution.
    #[must_use]
    pub fn relation_matrix_row_layout(&self) -> RelationMatrixRowLayout {
        self.relation_matrix_row_layout
    }

    /// Number of shared D rows in the packed setup contribution.
    #[must_use]
    pub fn d_rows(&self) -> usize {
        self.d_rows
    }

    /// Physical D-row width, including inactive columns between groups.
    #[must_use]
    pub fn d_physical_cols(&self) -> usize {
        self.d_physical_cols
    }
}

#[derive(Clone)]
pub(crate) struct SetupContributionGroupStatic<E> {
    pub(super) d_col_range: Range<usize>,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) required: usize,
    pub(super) segments: Arc<[GroupSetupSegment<E>]>,
    pub(super) a_row_weights: Arc<[E]>,
    pub(super) b_weights: Arc<[E]>,
}

pub(crate) struct SetupContributionGroupPlan<E> {
    pub(crate) d_col_range: Range<usize>,
    pub(crate) t_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) required: usize,
    pub(crate) segments: Arc<[GroupSetupSegment<E>]>,
    pub(crate) e_eq_slice: Vec<E>,
    pub(crate) t_eq_slice: Vec<E>,
    pub(crate) z_eq_slice: Vec<E>,
    pub(crate) a_row_weights: Arc<[E]>,
    pub(crate) b_weights: Arc<[E]>,
    pub(crate) d_weights: Arc<[E]>,
}
