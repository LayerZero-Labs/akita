//! SIS-secure level-parameter orchestration.
//!
//! These builders assemble a `LevelParams` from the [`crate::sis`] leaf
//! primitives (collision norms, secure ranks, per-role widths). They contain
//! no SIS formula of their own. Verifier-reachable through the runtime schedule
//! expansion, so every public function returns `Result<_, AkitaError>` and
//! never panics on malformed input.

use crate::sis::{
    decomposed_t_ring_count, min_secure_rank, rounded_up_norm_s, rounded_up_norm_t, AjtaiKeyParams,
    SisModulusFamily,
};
use crate::{AkitaScheduleInputs, DecompositionParams, LevelParams};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

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

/// Collision bounds (already rounded up to audited SIS buckets) for the A role
/// versus the B/D roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SisCollisionBounds {
    /// Collision infinity norm used for the A role.
    pub a: u32,
    /// Collision infinity norm shared by the B and D roles.
    pub bd: u32,
}

/// Build a SIS-secure `LevelParams` from explicit (rounded-up) collision
/// buckets and widths, looking up the tight minimum module rank per role.
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
    stage1_config: SparseChallengeConfig,
) -> Result<LevelParams, AkitaError> {
    let resolve = |role: &str, collision: u32, width: u64| {
        min_secure_rank(sis_family, d as u32, collision, width).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "missing secure root {role}-row rank for family={sis_family:?} \
                 D={d} lb={log_basis} width={width}"
            ))
        })
    };

    let n_a = resolve("A", collisions.a, widths.inner as u64)?;
    let n_b = resolve("B", collisions.bd, widths.outer as u64)?;
    let n_d = resolve("D", collisions.bd, widths.d_matrix as u64)?;

    let mut result =
        LevelParams::params_only(sis_family, d, log_basis, n_a, n_b, n_d, stage1_config);
    // Carry the audited SIS-floor bucket on each role; `col_len` stays at the
    // `params_only` placeholder `0` and is filled by a later
    // `with_layout`/`with_decomp`. `new_unchecked` is required because
    // `col_len = 0` is an intentional placeholder the strict audit rejects.
    result.a_key = AjtaiKeyParams::new_unchecked(sis_family, n_a, 0, collisions.a, d);
    result.b_key = AjtaiKeyParams::new_unchecked(sis_family, n_b, 0, collisions.bd, d);
    result.d_key = AjtaiKeyParams::new_unchecked(sis_family, n_d, 0, collisions.bd, d);
    Ok(result)
}

/// Derive SIS-secure root params for a concrete root layout, sizing each role
/// against the `crate::sis` norms.
///
/// # Errors
///
/// Returns an error when the root layout does not fit a supported SIS
/// collision bucket or rank table entry.
#[allow(clippy::too_many_arguments)]
pub fn sis_derived_root_params_for_layout(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1_config: SparseChallengeConfig,
    ring_subfield_embedding_norm_bound: u32,
    onehot_chunk_size: usize,
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let is_root = inputs.level == 0;
    // The level's gadget base may differ from the config default; the `sis`
    // norm/digit helpers read it from `decomposition.log_basis`.
    let level_decomp = DecompositionParams {
        log_basis: lp.log_basis,
        ..decomp
    };
    let bd_collision = rounded_up_norm_t(sis_family, d, lp.log_basis).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "root B/D collision bucket miss for family={sis_family:?} D={d} lb={}",
            lp.log_basis
        ))
    })?;
    let a_collision = rounded_up_norm_s(
        sis_family,
        d,
        level_decomp,
        &stage1_config,
        lp.fold_challenge_shape,
        is_root,
        onehot_chunk_size,
        ring_subfield_embedding_norm_bound,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "missing supported root A-role collision bucket for family={sis_family:?}, D={d}"
        ))
    })?;
    // Size the outer B-matrix width against the *secure* A-rank for this
    // layout's inner width, matching what the runtime expansion reconstructs
    // from the stored `n_a`.
    let exact_outer_width = {
        let n_a = min_secure_rank(sis_family, d as u32, a_collision, lp.inner_width() as u64)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "missing secure root A-row rank for family={sis_family:?} D={d} \
                 lb={} inner_width={}",
                    lp.log_basis,
                    lp.inner_width()
                ))
            })?;
        decomposed_t_ring_count(n_a, lp.num_digits_open, lp.num_blocks, 1)
            .ok_or_else(|| AkitaError::InvalidSetup("root outer width overflow".to_string()))?
    };
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
            outer: exact_outer_width,
            d_matrix: lp.d_matrix_width(),
        },
        stage1_config,
    )
}

/// Apply [`sis_derived_root_params_for_layout`] to an explicit root layout and
/// re-attach the layout to the resulting params.
///
/// # Errors
///
/// Propagates the SIS-floor lookup error from
/// [`sis_derived_root_params_for_layout`].
#[allow(clippy::too_many_arguments)]
pub fn root_level_params_for_layout_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1: SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    onehot_chunk_size: usize,
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let params = sis_derived_root_params_for_layout(
        sis_family,
        d,
        decomp,
        stage1,
        ring_subfield_norm_bound,
        onehot_chunk_size,
        inputs,
        lp,
    )?;
    Ok(params.with_layout(lp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sis_floor_miss_surfaces_as_error() {
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
