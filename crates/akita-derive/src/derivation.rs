//! SIS-derivation search loops moved out of `akita-types`.
//!
//! These functions invoke `optimal_m_r_split` and the generated SIS-floor
//! tables to derive secure level parameters for the planner and config-policy
//! adapters. They are intentionally not on the verifier replay path: the
//! verifier consumes already-derived `LevelParams` from materialized plans.
//!
//! Pure layout helpers (`level_layout_from_params`,
//! `recursive_level_layout_from_params`, `recursive_level_decomposition_from_root`,
//! `decomp_depths`) stay in `akita-types::layout::sis_derivation` since the
//! verifier reaches them via plan-from-table materialization and recursive
//! suffix wiring.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::layout::digit_math::{compute_num_digits_fold_with_claims, optimal_m_r_split};
use akita_types::{
    decomp_depths, AjtaiKeyParams, AkitaScheduleInputs, CommitmentEnvelope, DecompositionParams,
    LevelParams, SisModulusFamily,
};

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

/// Collision bounds for the A role versus the B/D roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SisCollisionBounds {
    /// Collision infinity norm used for the A role.
    pub a: u32,
    /// Collision infinity norm shared by the B and D roles.
    pub bd: u32,
}

fn a_role_collision_raw(
    a_raw: u32,
    stage1_config: &SparseChallengeConfig,
    ring_subfield_embedding_norm_bound: u32,
) -> Option<u32> {
    a_raw
        .checked_mul(stage1_config.infinity_norm())?
        .checked_mul(ring_subfield_embedding_norm_bound)
}

/// Build a SIS-secure `LevelParams` from the explicit width budget.
///
/// Looks up the minimum module-SIS rank for each of `(a, b, d)` against the
/// generated 128-bit security tables. The optional fallback envelope is only
/// a setup-sizing floor: when present, the selected rank is
/// `max(generated_floor, envelope_floor)`. Missing generated SIS coverage is
/// always an error.
///
/// # Errors
///
/// Returns an error when no generated SIS-security row covers one of the
/// requested role widths.
pub fn sis_secure_level_params(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
    collisions: SisCollisionBounds,
    widths: SisRoleWidths,
    fallback: Option<&CommitmentEnvelope>,
    stage1_config: SparseChallengeConfig,
) -> Result<LevelParams, AkitaError> {
    let resolve = |role: &str, collision: u32, width: u64, fallback_rank: Option<usize>| {
        min_rank_for_secure_width(sis_family, d as u32, collision, width)
            .map(|floor| fallback_rank.map_or(floor, |rank| floor.max(rank)))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "missing secure root {role}-row rank for family={sis_family:?} \
                     D={d} lb={log_basis} width={width}"
                ))
            })
    };

    let n_a = resolve(
        "A",
        collisions.a,
        widths.inner as u64,
        fallback.map(|e| e.max_n_a),
    )?;
    let n_b = resolve(
        "B",
        collisions.bd,
        widths.outer as u64,
        fallback.map(|e| e.max_n_b),
    )?;
    let n_d = resolve(
        "D",
        collisions.bd,
        widths.d_matrix as u64,
        fallback.map(|e| e.max_n_d),
    )?;

    let mut result =
        LevelParams::params_only(sis_family, d, log_basis, n_a, n_b, n_d, stage1_config);
    result.a_key = AjtaiKeyParams::new(sis_family, n_a, 0, collisions.a, d);
    result.b_key = AjtaiKeyParams::new(sis_family, n_b, 0, collisions.bd, d);
    result.d_key = AjtaiKeyParams::new(sis_family, n_d, 0, collisions.bd, d);
    Ok(result)
}

/// Derive SIS-secure recursive params for a concrete recursive layout.
pub fn sis_derived_recursive_params_for_layout(
    sis_family: SisModulusFamily,
    d: usize,
    log_basis: u32,
    stage1_config: &SparseChallengeConfig,
    ring_subfield_embedding_norm_bound: u32,
    envelope: &CommitmentEnvelope,
    layout: &LevelParams,
) -> Option<LevelParams> {
    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = bd_collision;
    let a_collision_raw =
        a_role_collision_raw(a_raw, stage1_config, ring_subfield_embedding_norm_bound)?;
    let a_collision = ceil_supported_collision(sis_family, d as u32, a_collision_raw)?;

    let exact_outer_width = {
        let n_a = min_rank_for_secure_width(
            sis_family,
            d as u32,
            a_collision,
            layout.inner_width() as u64,
        )?
        .max(envelope.max_n_a);
        n_a * layout.num_digits_open * layout.num_blocks
    };
    sis_secure_level_params(
        sis_family,
        d,
        log_basis,
        SisCollisionBounds {
            a: a_collision,
            bd: bd_collision,
        },
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
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1_config: SparseChallengeConfig,
    ring_subfield_embedding_norm_bound: u32,
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let bd_collision = (1u32 << lp.log_basis) - 1;
    let a_raw = if inputs.level == 0 && decomp.log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_collision_raw =
        a_role_collision_raw(a_raw, &stage1_config, ring_subfield_embedding_norm_bound)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "root A-role collision overflow for family={sis_family:?}, D={d}"
                ))
            })?;
    let a_collision =
        ceil_supported_collision(sis_family, d as u32, a_collision_raw).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "missing supported root A-role collision bucket for family={sis_family:?}, D={d} \
                 and raw collision {a_collision_raw}"
            ))
        })?;
    sis_secure_level_params(
        sis_family,
        d,
        lp.log_basis,
        SisCollisionBounds {
            a: a_collision,
            bd: bd_collision,
        },
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
/// the root variable count into outer (`m`) and inner (`r`) variables.
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
        inputs.num_vars.saturating_sub(alpha)
    } else {
        inputs
            .num_vars
            .checked_sub(alpha)
            .ok_or_else(|| AkitaError::InvalidSetup("num_vars is smaller than alpha".to_string()))?
    };
    if reduced_vars == 0 && !allow_zero_outer {
        return Err(AkitaError::InvalidSetup(
            "num_vars must leave at least one outer variable".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sis_floor_miss_is_not_rescued_by_envelope() {
        let envelope = CommitmentEnvelope {
            max_n_a: 4,
            max_n_b: 4,
            max_n_d: 4,
        };
        let err = sis_secure_level_params(
            SisModulusFamily::Q32,
            32,
            3,
            SisCollisionBounds { a: 2047, bd: 2047 },
            SisRoleWidths {
                inner: 557_704,
                outer: 1,
                d_matrix: 1,
            },
            Some(&envelope),
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .expect_err("width beyond generated Q32/D32 rank-20 floor must fail");
        assert!(
            err.to_string().contains("missing secure root A-row rank"),
            "unexpected error: {err}"
        );
    }
}
