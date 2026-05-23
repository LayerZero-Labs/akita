//! Pure layout helpers shared by config, scheme, and planner code.
//!
//! The verifier-reachable layout helpers
//! (`level_layout_from_params`, `recursive_level_layout_from_params`,
//! `decomp_depths`) live here.
//! Search and SIS-derivation loops moved to `akita_planner::derivation`.

use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::{DecompositionParams, LevelParams};
use akita_field::AkitaError;

/// Compute `(depth_commit, depth_open)` for one decomposition.
pub fn decomp_depths(decomp: DecompositionParams) -> (usize, usize) {
    let field_bits = decomp.field_bits();
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, field_bits, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, field_bits, decomp.log_basis);
    (depth_commit, depth_open)
}

/// Apply layout coordinates and decomposition depths to a parameter-only level.
///
/// # Errors
///
/// Returns an error when the resulting layout is internally inconsistent.
pub fn level_layout_from_params(
    m_vars: usize,
    r_vars: usize,
    lp: &LevelParams,
    decomp: DecompositionParams,
    num_ring: usize,
) -> Result<LevelParams, AkitaError> {
    let (depth_commit, depth_open) = decomp_depths(decomp);
    let depth_fold = compute_num_digits_fold_with_claims(
        r_vars,
        lp.challenge_l1_mass(),
        decomp.log_basis,
        1,
        decomp.field_bits(),
    );
    lp.with_decomp(
        m_vars,
        r_vars,
        depth_commit,
        depth_open,
        depth_fold,
        num_ring,
    )
}

/// Derive a recursive `w`-opening layout from the active level params.
///
/// # Errors
///
/// Returns an error if the witness length is incompatible with `params.d` or if
/// the recursive layout derivation overflows.
pub fn recursive_level_layout_from_params(
    lp: &LevelParams,
    current_w_len: usize,
    root_decomp: DecompositionParams,
) -> Result<LevelParams, AkitaError> {
    if !current_w_len.is_multiple_of(lp.ring_dimension) {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {current_w_len} is not divisible by D={}",
            lp.ring_dimension
        )));
    }
    let num_ring_elems = current_w_len / lp.ring_dimension;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = lp.ring_dimension.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    // Recursive level: balanced-digit `w` entries collapse `log_commit_bound`
    // to `log_basis`; opening folds inherit the parent's open bound.
    let decomp = DecompositionParams {
        log_basis: lp.log_basis,
        log_commit_bound: lp.log_basis,
        log_open_bound: Some(
            root_decomp
                .log_open_bound
                .unwrap_or(root_decomp.log_commit_bound),
        ),
    };
    let (m_vars, r_vars) = optimal_m_r_split(
        lp.a_key.row_len() as u32,
        lp.challenge_l1_mass(),
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        num_ring_elems,
        decomp.field_bits(),
    );
    let layout = level_layout_from_params(m_vars, r_vars, lp, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}
