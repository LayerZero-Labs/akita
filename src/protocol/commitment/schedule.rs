use super::config::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split_with_params, CommitmentConfig,
    DecompositionParams, HachiCommitmentLayout,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;

/// Public inputs that deterministically select one level's active Hachi params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiScheduleInputs {
    /// Root polynomial variable count.
    pub max_num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Runtime source of truth for one Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelParams {
    /// Ring dimension at this level.
    pub d: usize,
    /// Gadget base exponent.
    pub log_basis: u32,
    /// Active inner Ajtai rank.
    pub n_a: usize,
    /// Active outer commitment rank.
    pub n_b: usize,
    /// Active D-matrix rank.
    pub n_d: usize,
    /// Conservative sparse-challenge L1 mass used by folded-norm bounds.
    pub challenge_l1_mass: usize,
    /// Stage-1 challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
}

impl HachiLevelParams {
    /// Total number of quotient / relation rows in `M`.
    pub fn m_row_count(&self) -> usize {
        self.n_d + self.n_b + 2 + self.n_a
    }
}

fn with_log_basis(mut decomp: DecompositionParams, log_basis: u32) -> DecompositionParams {
    decomp.log_basis = log_basis;
    decomp
}

fn main_level_decomposition<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
) -> DecompositionParams {
    with_log_basis(Cfg::decomposition(), params.log_basis)
}

fn recursive_level_decomposition<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
) -> DecompositionParams {
    let parent = Cfg::decomposition();
    let parent_open = parent.log_open_bound.unwrap_or(parent.log_commit_bound);
    DecompositionParams {
        log_basis: params.log_basis,
        log_commit_bound: params.log_basis,
        log_open_bound: Some(parent_open),
    }
}

fn layout_from_params(
    m_vars: usize,
    r_vars: usize,
    params: &HachiLevelParams,
    decomp: DecompositionParams,
) -> Result<HachiCommitmentLayout, HachiError> {
    let depth_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = compute_num_digits(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold(r_vars, params.challenge_l1_mass, decomp.log_basis);
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        params.n_a,
        depth_commit,
        depth_open,
        depth_fold,
        decomp.log_basis,
    )
}

/// Derive the root level's active params and layout.
///
/// # Errors
///
/// Returns an error if the root variable split is invalid or overflows.
pub fn hachi_root_level_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    });
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
        HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
    })?;
    if reduced_vars == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_vars must leave at least one outer variable".to_string(),
        ));
    }
    let decomp = main_level_decomposition::<Cfg>(&params);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars);
    let layout = layout_from_params(m_vars, r_vars, &params, decomp)?;
    Ok((params, layout))
}

/// Derive a recursive `w`-opening level's active params and layout.
///
/// # Errors
///
/// Returns an error if the recursive layout derivation overflows.
pub fn hachi_level_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(inputs);
    let num_ring_elems = inputs.current_w_len / params.d;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    let decomp = recursive_level_decomposition::<Cfg>(&params);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars);
    let layout = layout_from_params(m_vars, r_vars, &params, decomp)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok((params, layout))
}
