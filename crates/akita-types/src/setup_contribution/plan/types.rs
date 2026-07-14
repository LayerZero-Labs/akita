use super::kernels::GroupSetupSegment;
use crate::{
    RelationMatrixRowLayout, SetupContributionPlanInputs, SetupProjectionGeometry, WitnessLayout,
};
use akita_field::{AkitaError, FieldCore};
use std::sync::Arc;

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub group_id: usize,
    pub e_col_offset: usize,
    pub num_claims: usize,
    pub live_fold_count: usize,
    pub fold_position_count: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_b: usize,
    pub t_cols_per_vector: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
    pub layout: Arc<WitnessLayout>,
    pub opening_source_len: usize,
}

pub struct SingleGroupSetupContributionLayout {
    pub group: SetupContributionGroupInputs,
    pub d_row_start: usize,
    pub d_rows: usize,
    pub d_physical_cols: usize,
}

impl SetupContributionGroupInputs {
    /// Derive the legacy single-commitment-group setup layout.
    ///
    /// This only computes the group descriptor and shared D footprint. Callers
    /// still go through [`SetupContributionPlan::prepare_static`] and
    /// [`SetupContributionPlan::finish_plan`] so single- and multi-group setup
    /// contributions share the same planning pipeline.
    pub fn single_group_layout<E: FieldCore>(
        inputs: &SetupContributionPlanInputs<E>,
        layout: &WitnessLayout,
        opening_source_len: usize,
        fold_log_basis: u32,
    ) -> Result<SingleGroupSetupContributionLayout, AkitaError> {
        if inputs.num_groups != 1 || inputs.num_polys_per_group.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "single-group setup contribution requires exactly one commitment group".into(),
            ));
        }
        let d_rows = match inputs.relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => inputs.n_d,
            RelationMatrixRowLayout::WithoutDBlock => 0,
        };
        let a_row_start = 1usize;
        let b_row_start = a_row_start
            .checked_add(inputs.n_a)
            .ok_or_else(|| AkitaError::InvalidSetup("B row start overflow".into()))?;
        let d_row_start = b_row_start
            .checked_add(inputs.n_b)
            .ok_or_else(|| AkitaError::InvalidSetup("D row start overflow".into()))?;
        let b_per_claim_e = inputs
            .live_fold_count
            .checked_mul(inputs.depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("e-hat claim width overflow".into()))?;
        let d_physical_cols = inputs
            .num_claims
            .checked_mul(b_per_claim_e)
            .ok_or_else(|| AkitaError::InvalidSetup("e-hat column width overflow".into()))?;
        let t_cols_per_vector = inputs
            .n_a
            .checked_mul(inputs.depth_open)
            .and_then(|width| width.checked_mul(inputs.live_fold_count))
            .ok_or_else(|| AkitaError::InvalidSetup("T polynomial width overflow".into()))?;
        Ok(SingleGroupSetupContributionLayout {
            group: SetupContributionGroupInputs {
                group_id: 0,
                e_col_offset: 0,
                num_claims: inputs.num_claims,
                live_fold_count: inputs.live_fold_count,
                fold_position_count: inputs.fold_position_count,
                depth_open: inputs.depth_open,
                depth_commit: inputs.depth_commit,
                depth_fold: inputs.depth_fold,
                log_basis: fold_log_basis,
                n_a: inputs.n_a,
                n_b: inputs.n_b,
                t_cols_per_vector,
                a_row_start,
                b_row_start,
                layout: Arc::new(layout.clone()),
                opening_source_len,
            },
            d_row_start,
            d_rows,
            d_physical_cols,
        })
    }
}

pub struct SetupContributionPlan<E> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) projection_geometry: SetupProjectionGeometry,
}

/// Tau1-derived setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct SetupContributionStatic<E> {
    pub(super) groups: Vec<SetupContributionGroupStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Arc<[E]>,
}

impl<E> SetupContributionStatic<E> {
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
    pub(super) e_col_offset: usize,
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
    pub(crate) e_col_offset: usize,
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
