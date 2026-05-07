//! SIS-derivation primitives for config and schedule policy code.

use crate::generated::sis_floor::min_rank_for_secure_width;
use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::{
    AjtaiKeyParams, AkitaScheduleInputs, CommitmentEnvelope, DecompositionParams, LevelParams,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

/// Compute `(depth_commit, depth_open)` for one decomposition.
pub fn decomp_depths(decomp: DecompositionParams) -> (usize, usize) {
    let field_bits = decomp.field_bits();
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, field_bits, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, field_bits, decomp.log_basis);
    (depth_commit, depth_open)
}

/// Derive recursive-level decomposition bounds from the root decomposition.
pub fn recursive_level_decomposition_from_root(
    root_decomp: DecompositionParams,
    log_basis: u32,
) -> DecompositionParams {
    let parent_open = root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound);
    DecompositionParams {
        log_basis,
        log_commit_bound: log_basis,
        log_open_bound: Some(parent_open),
    }
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
) -> Result<LevelParams, AkitaError> {
    let resolve = |role: &str, collision: u32, width: u64, fallback_rank: Option<usize>| {
        min_rank_for_secure_width(d as u32, collision, width)
            .or(fallback_rank)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
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
    let a_report = layout.stage1_sis_extraction_report(a_raw).ok()?;
    let a_collision = a_report.a_role_supported_collision_bucket;

    let exact_outer_width = {
        let n_a = min_rank_for_secure_width(d as u32, a_collision, layout.inner_width() as u64)
            .unwrap_or(envelope.max_n_a);
        n_a * layout.num_digits_open * layout.num_blocks
    };
    let mut params = sis_secure_level_params(
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
    .ok()?;
    params.stage1_challenge_shape = layout.stage1_challenge_shape.clone();
    Some(params)
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
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let bd_collision = (1u32 << lp.log_basis) - 1;
    let a_raw = if inputs.level == 0 && decomp.log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_report = lp.stage1_sis_extraction_report(a_raw).map_err(|err| {
        AkitaError::InvalidSetup(format!("root A-role extraction report failed: {err}"))
    })?;
    let a_collision = a_report.a_role_supported_collision_bucket;
    let mut params = sis_secure_level_params(
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
    )?;
    params.stage1_challenge_shape = lp.stage1_challenge_shape.clone();
    Ok(params)
}

/// Build a root `LevelParams` from a candidate parameter set by splitting
/// `max_num_vars` into outer (`m`) and inner (`r`) variables.
///
/// # Errors
///
/// Returns an error when the root arity is too small for the ring dimension.
pub fn derived_root_commitment_layout_from_params(
    inputs: AkitaScheduleInputs,
    decomp: DecompositionParams,
    params: &LevelParams,
    allow_zero_outer: bool,
) -> Result<LevelParams, AkitaError> {
    let alpha = params.ring_dimension.trailing_zeros() as usize;
    let reduced_vars = if allow_zero_outer {
        inputs.max_num_vars.saturating_sub(alpha)
    } else {
        inputs.max_num_vars.checked_sub(alpha).ok_or_else(|| {
            AkitaError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?
    };
    if reduced_vars == 0 && !allow_zero_outer {
        return Err(AkitaError::InvalidSetup(
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
        decomp.field_bits(),
    );
    let (depth_commit, depth_open) = decomp_depths(decomp);
    let depth_fold = compute_num_digits_fold_with_claims(
        r_vars,
        params.challenge_l1_mass(),
        decomp.log_basis,
        1,
        decomp.field_bits(),
    );
    params.with_decomp(m_vars, r_vars, depth_commit, depth_open, depth_fold, 0)
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
    let decomp = recursive_level_decomposition_from_root(root_decomp, lp.log_basis);
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::Stage1ChallengeShape;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn sample_root_layout(shape: Stage1ChallengeShape) -> LevelParams {
        let mut params = LevelParams::params_only(64, 2, 1, 1, 1, stage1_config());
        params.stage1_challenge_shape = shape;
        params.with_decomp(2, 1, 1, 1, 1, 0).unwrap()
    }

    #[test]
    fn sis_root_derivation_uses_tensor_extraction_collision_bucket() {
        let flat = sample_root_layout(Stage1ChallengeShape::Flat);
        let tensor = sample_root_layout(Stage1ChallengeShape::Tensor);
        let decomp = DecompositionParams {
            log_basis: 2,
            log_commit_bound: 128,
            log_open_bound: Some(128),
        };
        let inputs = AkitaScheduleInputs {
            max_num_vars: 8,
            level: 0,
            current_w_len: 256,
        };

        let flat_params =
            sis_derived_root_params_for_layout(64, decomp, stage1_config(), inputs, &flat).unwrap();
        let tensor_params =
            sis_derived_root_params_for_layout(64, decomp, stage1_config(), inputs, &tensor)
                .unwrap();

        assert_eq!(flat_params.a_key.collision_inf(), 3);
        assert_eq!(tensor_params.a_key.collision_inf(), 63);
        assert_eq!(
            tensor_params.stage1_challenge_shape,
            Stage1ChallengeShape::Tensor
        );
    }

    #[test]
    fn sis_root_derivation_rejects_tensor_collision_beyond_generated_buckets() {
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 700,
            nonzero_coeffs: vec![-1, 1],
        };
        let mut params = LevelParams::params_only(128, 2, 1, 1, 1, stage1_config.clone());
        params.stage1_challenge_shape = Stage1ChallengeShape::Tensor;
        let layout = params.with_decomp(2, 1, 1, 1, 1, 0).unwrap();
        let decomp = DecompositionParams {
            log_basis: 2,
            log_commit_bound: 128,
            log_open_bound: Some(128),
        };
        let inputs = AkitaScheduleInputs {
            max_num_vars: 9,
            level: 0,
            current_w_len: 512,
        };

        let err = sis_derived_root_params_for_layout(128, decomp, stage1_config, inputs, &layout)
            .unwrap_err();
        assert!(format!("{err:?}").contains("missing supported stage-1 A-role collision bucket"));
    }
}
