//! SIS-derivation primitives for config and schedule policy code.

use crate::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use crate::{
    AjtaiKeyParams, CommitmentEnvelope, DecompositionParams, HachiScheduleInputs, LevelParams,
};
use akita_algebra::SparseChallengeConfig;
use akita_field::HachiError;

/// Compute `(depth_commit, depth_open)` for one decomposition.
pub fn decomp_depths(decomp: DecompositionParams) -> (usize, usize) {
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
    (depth_commit, depth_open)
}

/// SIS-secure rank derivation inputs, bundled to keep
/// [`sis_secure_level_params`] under clippy's argument-count cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SisRoleWidths {
    /// Inner A-matrix width.
    pub inner: usize,
    /// Outer B-matrix width.
    pub outer: usize,
    /// Prover D-matrix width.
    pub d_matrix: usize,
}

/// Build a SIS-secure `LevelParams` from the explicit width budget.
///
/// Looks up the minimum module-SIS rank for each of `(a, b, d)` against the
/// 128-bit security tables; falls back to `fallback` when the table does not
/// cover the requested width.
///
/// # Errors
///
/// Returns an error when no generated SIS-security row covers one of the
/// requested role widths and no fallback envelope supplies the rank.
pub fn sis_secure_level_params(
    d: usize,
    log_basis: u32,
    a_collision: u32,
    bd_collision: u32,
    widths: SisRoleWidths,
    fallback: Option<&CommitmentEnvelope>,
    stage1_config: SparseChallengeConfig,
) -> Result<LevelParams, HachiError> {
    let resolve = |role: &str, collision: u32, width: u64, fallback_rank: Option<usize>| {
        min_rank_for_secure_width(d as u32, collision, width)
            .or(fallback_rank)
            .ok_or_else(|| {
                HachiError::InvalidSetup(format!(
                    "missing secure root {role}-row rank for D={d} lb={log_basis} width={width}"
                ))
            })
    };

    let n_a = resolve(
        "A",
        a_collision,
        widths.inner as u64,
        fallback.map(|e| e.max_n_a),
    )?;
    let n_b = resolve(
        "B",
        bd_collision,
        widths.outer as u64,
        fallback.map(|e| e.max_n_b),
    )?;
    let n_d = resolve(
        "D",
        bd_collision,
        widths.d_matrix as u64,
        fallback.map(|e| e.max_n_d),
    )?;

    let mut result = LevelParams::params_only(d, log_basis, n_a, n_b, n_d, stage1_config);
    result.a_key = AjtaiKeyParams::new(n_a, 0, a_collision, d);
    result.b_key = AjtaiKeyParams::new(n_b, 0, bd_collision, d);
    result.d_key = AjtaiKeyParams::new(n_d, 0, bd_collision, d);
    Ok(result)
}

/// Derive SIS-secure recursive params for a concrete recursive layout.
pub fn sis_derived_recursive_params_for_layout(
    d: usize,
    log_basis: u32,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
    layout: &LevelParams,
) -> Option<LevelParams> {
    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = bd_collision;
    let a_collision = ceil_supported_collision(d as u32, a_raw * stage1_config.max_abs_coeff())?;

    let exact_outer_width = {
        let n_a = min_rank_for_secure_width(d as u32, a_collision, layout.inner_width() as u64)
            .unwrap_or(envelope.max_n_a);
        n_a * layout.num_digits_open * layout.num_blocks
    };
    sis_secure_level_params(
        d,
        log_basis,
        a_collision,
        bd_collision,
        SisRoleWidths {
            inner: layout.inner_width(),
            outer: exact_outer_width,
            d_matrix: layout.d_matrix_width(),
        },
        Some(envelope),
        stage1_config.clone(),
    )
    .ok()
}

/// Derive SIS-secure root params for a concrete root layout.
///
/// # Errors
///
/// Returns an error when the root layout does not fit a supported SIS
/// collision bucket or rank table entry.
pub fn sis_derived_root_params_for_layout(
    d: usize,
    decomp: DecompositionParams,
    stage1_config: SparseChallengeConfig,
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    let bd_collision = (1u32 << lp.log_basis) - 1;
    let a_raw = if inputs.level == 0 && decomp.log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_collision = ceil_supported_collision(d as u32, a_raw * stage1_config.max_abs_coeff())
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing supported root A-role collision bucket for D={} and raw collision {}",
                d,
                a_raw * stage1_config.max_abs_coeff()
            ))
        })?;
    sis_secure_level_params(
        d,
        lp.log_basis,
        a_collision,
        bd_collision,
        SisRoleWidths {
            inner: lp.inner_width(),
            outer: lp.outer_width(),
            d_matrix: lp.d_matrix_width(),
        },
        None,
        stage1_config,
    )
}

/// Build a root `LevelParams` from a candidate parameter set by splitting
/// `max_num_vars` into outer (`m`) and inner (`r`) variables.
///
/// # Errors
///
/// Returns an error when the root arity is too small for the ring dimension.
pub fn derived_root_commitment_layout_from_params(
    inputs: HachiScheduleInputs,
    decomp: DecompositionParams,
    params: &LevelParams,
    allow_zero_outer: bool,
) -> Result<LevelParams, HachiError> {
    let alpha = params.ring_dimension.trailing_zeros() as usize;
    let reduced_vars = if allow_zero_outer {
        inputs.max_num_vars.saturating_sub(alpha)
    } else {
        inputs.max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?
    };
    if reduced_vars == 0 && !allow_zero_outer {
        return Err(HachiError::InvalidSetup(
            "max_num_vars must leave at least one outer variable".to_string(),
        ));
    }

    let mut decomp = decomp;
    decomp.log_basis = params.log_basis;
    let (m_vars, r_vars) = optimal_m_r_split(
        params.a_key.row_len() as u32,
        params.challenge_l1_mass(),
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        0,
    );
    let (depth_commit, depth_open) = decomp_depths(decomp);
    let depth_fold = compute_num_digits_fold_with_claims(
        r_vars,
        params.challenge_l1_mass(),
        decomp.log_basis,
        1,
    );
    params.with_decomp(m_vars, r_vars, depth_commit, depth_open, depth_fold, 0)
}
