use crate::WitnessChunkLayout;

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub e_col_offset: usize,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub block_len: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_b: usize,
    pub t_cols_per_vector: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkLayout>,
}

pub struct SetupContributionPlan<E> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
}

/// Tau1-derived setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct SetupContributionStatic<E> {
    pub(super) groups: Vec<SetupContributionGroupStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Vec<E>,
}

#[derive(Clone)]
pub(crate) struct SetupContributionGroupStatic<E> {
    pub(super) e_col_offset: usize,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) a_weights: Vec<E>,
    pub(super) b_weights: Vec<E>,
}

pub(crate) struct SetupContributionGroupPlan<E> {
    pub(crate) e_col_offset: usize,
    pub(crate) t_cols: usize,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) n_b: usize,
    pub(crate) e_eq_slice: Vec<E>,
    pub(crate) t_eq_slice: Vec<E>,
    pub(crate) z_eq_slice: Vec<E>,
    pub(crate) a_weights: Vec<E>,
    pub(crate) b_weights: Vec<E>,
    pub(crate) d_weights: Vec<E>,
}
