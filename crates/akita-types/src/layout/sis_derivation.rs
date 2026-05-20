//! SIS-derivation primitives for config and schedule policy code.

use crate::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use crate::layout::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::{
    AjtaiKeyParams, AkitaScheduleInputs, CommitmentEnvelope, DecompositionParams, LevelParams,
    SisModulusFamily,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

/// Inclusive upper bound on `outer_log_basis` for the tiered root
/// commitment. Today's balanced i8 `FlatDigitBlocks` storage only
/// supports `log_basis ≤ 6`; the tiered outer digit basis must obey the
/// same bound until a non-i8 digit type is introduced.
pub const MAX_TIERED_OUTER_LOG_BASIS: u32 = 6;
/// Inclusive lower bound on `outer_log_basis`. Bases below 2 would not
/// actually compress and are rejected by the planner.
pub const MIN_TIERED_OUTER_LOG_BASIS: u32 = 2;

/// Collision-`inf` bound used by the SIS floor table for a balanced
/// i8 gadget decomposition with basis `2^log_basis`.
///
/// Two valid digit vectors `x, x'` with `‖x‖_∞, ‖x'‖_∞ < 2^log_basis`
/// have `‖x − x'‖_∞ ≤ 2^log_basis − 1`. Existing code already adopts
/// this convention (see `sis_derived_recursive_params_for_layout`'s
/// `bd_collision = (1u32 << log_basis) - 1`); the tiered helpers below
/// follow it so all SIS sizing speaks one bound convention.
#[inline]
pub fn balanced_digit_delta_bound(log_basis: u32) -> u32 {
    (1u32 << log_basis).saturating_sub(1)
}

/// Validate that `outer_log_basis` falls in the supported range. Returns
/// `Err` if it does not.
///
/// # Errors
///
/// Returns an error when `outer_log_basis` is outside
/// `[MIN_TIERED_OUTER_LOG_BASIS, MAX_TIERED_OUTER_LOG_BASIS]`.
pub fn validate_tiered_outer_log_basis(outer_log_basis: u32) -> Result<(), AkitaError> {
    if !(MIN_TIERED_OUTER_LOG_BASIS..=MAX_TIERED_OUTER_LOG_BASIS).contains(&outer_log_basis) {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered outer_log_basis = {outer_log_basis} is outside the supported \
             range [{MIN_TIERED_OUTER_LOG_BASIS}, {MAX_TIERED_OUTER_LOG_BASIS}] \
             (current balanced i8 storage limits log_basis to 6)"
        )));
    }
    Ok(())
}

/// Compute the SIS rank for the tier-1 `B'` matrix.
///
/// Implements the case-2 branch of the two-case binding proof in
/// `specs/tiered_commit.md` §6: a `B'` collision is exhibited by a
/// non-zero `Δt_i` with `‖Δt_i‖_∞ ≤ 2 · t_inf_bound`. The planner sizes
/// `n_b'` so that no such collision exists within the chunk width
/// `outer_width / split_factor`.
///
/// `t_inf_bound` is the existing per-cell `t̂` infinity-norm bound (the
/// inner gadget bound used today for the legacy `B` key).
///
/// # Errors
///
/// Returns `Err` when the generated SIS floor table has no entry for the
/// requested `(family, d, collision, width)` tuple.
pub fn tiered_b_prime_rank(
    family: SisModulusFamily,
    d: u32,
    t_inf_bound: u32,
    outer_width: usize,
    split_factor: usize,
) -> Result<u32, AkitaError> {
    if split_factor < 1 {
        return Err(AkitaError::InvalidSetup(
            "tiered_b_prime_rank: split_factor must be ≥ 1".to_string(),
        ));
    }
    if outer_width % split_factor != 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "tiered_b_prime_rank: outer_width = {outer_width} not divisible by \
             split_factor = {split_factor}"
        )));
    }
    let chunk_width = outer_width / split_factor;
    let collision_raw = t_inf_bound
        .checked_mul(2)
        .ok_or_else(|| AkitaError::InvalidSetup("Δt collision bound overflow".to_string()))?;
    let collision = ceil_supported_collision(family, d, collision_raw).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_b_prime_rank: no supported SIS collision bucket covers Δt bound \
             {collision_raw} for family={family:?}, D={d}"
        ))
    })?;
    let rank =
        min_rank_for_secure_width(family, d, collision, chunk_width as u64).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "tiered_b_prime_rank: no generated SIS row covers \
                 family={family:?}, D={d}, collision={collision}, width={chunk_width}"
            ))
        })?;
    u32::try_from(rank).map_err(|_| {
        AkitaError::InvalidSetup("tiered_b_prime_rank: SIS rank exceeds u32".to_string())
    })
}

/// Compute the SIS rank for the tier-1 `F` matrix.
///
/// Implements the case-1 branch of the two-case binding proof in
/// `specs/tiered_commit.md` §6: an `F` collision is exhibited by a
/// non-zero `Δû_concat` with `‖Δû_concat‖_∞ ≤ 2 · floor(2^outer_log_basis / 2)`.
/// We use `2^outer_log_basis − 1` here for consistency with the existing
/// balanced-digit collision convention (see `balanced_digit_delta_bound`).
///
/// `F` has width `n_b' · split_factor · num_digits_outer`.
///
/// # Errors
///
/// Returns `Err` when `outer_log_basis` is outside the supported range or
/// when the generated SIS floor table has no entry for the requested
/// `(family, d, collision, width)` tuple.
pub fn tiered_f_rank(
    family: SisModulusFamily,
    d: u32,
    outer_log_basis: u32,
    n_b_prime: u32,
    split_factor: usize,
    num_digits_outer: usize,
) -> Result<u32, AkitaError> {
    validate_tiered_outer_log_basis(outer_log_basis)?;
    if split_factor < 2 {
        return Err(AkitaError::InvalidSetup(
            "tiered_f_rank: split_factor must be ≥ 2 for the tiered path".to_string(),
        ));
    }
    if num_digits_outer == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered_f_rank: num_digits_outer must be ≥ 1".to_string(),
        ));
    }
    let width = (n_b_prime as usize)
        .checked_mul(split_factor)
        .and_then(|x| x.checked_mul(num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("F width overflow".to_string()))?;
    let collision = balanced_digit_delta_bound(outer_log_basis);
    let collision = ceil_supported_collision(family, d, collision).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_f_rank: no supported SIS collision bucket covers Δû bound \
             {collision} for family={family:?}, D={d}"
        ))
    })?;
    let rank = min_rank_for_secure_width(family, d, collision, width as u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "tiered_f_rank: no generated SIS row covers family={family:?}, D={d}, \
             collision={collision}, width={width}"
        ))
    })?;
    u32::try_from(rank)
        .map_err(|_| AkitaError::InvalidSetup("tiered_f_rank: SIS rank exceeds u32".to_string()))
}

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
