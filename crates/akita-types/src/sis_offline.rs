//! SIS-secure level-parameter derivation.
//!
//! These functions invoke `optimal_m_r_split` and the generated SIS-floor
//! tables to derive secure level parameters for the planner DP search and the
//! Cfg-driven runtime schedule expansion (`akita_planner::schedule_from_entry`).
//! They are verifier-reachable transitively through that expansion, so every
//! public function returns `Result<_, AkitaError>` on malformed inputs and
//! never panics on the verifier replay path.
//!
//! Pure layout helpers (`level_layout_from_params`,
//! `recursive_level_layout_from_params`, `recursive_level_decomposition_from_root`,
//! `decomp_depths`) live in [`crate::layout::sis_derivation`].

use crate::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
use crate::{
    AjtaiKeyParams, AkitaScheduleInputs, DecompositionParams, LevelParams, SisModulusFamily,
};
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

/// Canonical A-role base infinity-norm bound (per coefficient), before
/// scaling by the stage-1 challenge norm and the ring-subfield embedding
/// norm.
///
/// This is the single source of truth for the A-role `a_raw` used by the
/// planner DP (`akita_planner`'s `WitnessType::S::binding_norm`), the
/// offline SIS derivation in this module, and the runtime table-hit
/// expansion (`akita_planner::generated::GeneratedFoldStep::expand_to_level_params`).
/// Keeping one formula guarantees the rank the planner sizes against and
/// the `collision_inf` the runtime reconstructs can never drift apart.
///
/// The root commits the balanced-decomposed witness, bounded per
/// coefficient by `2·β` with `β = 2^(lb−1) − 1` (or `1` when
/// `log_commit_bound == 1`, the one-hot fast path). A recursive level
/// commits the full digit-range witness, bounded by `2^lb − 1`.
///
/// Returns `None` when `log_basis` overflows the bound (verifier-reachable
/// callers surface that as an `AkitaError`).
pub fn a_role_base_norm(log_basis: u32, log_commit_bound: u32, is_root: bool) -> Option<u32> {
    if is_root {
        let beta = if log_commit_bound == 1 {
            1
        } else {
            1u32.checked_shl(log_basis.checked_sub(1)?)?
                .checked_sub(1)?
        };
        beta.checked_mul(2)
    } else {
        1u32.checked_shl(log_basis)?.checked_sub(1)
    }
}

/// Build a SIS-secure `LevelParams` from the explicit width budget.
///
/// Looks up the minimum module-SIS rank for each of `(a, b, d)` against the
/// generated 128-bit security tables and returns the resulting layout-free
/// `LevelParams`. There is no rank floor beyond what the SIS-floor tables
/// require — anything above is secure but unnecessary, so the planner /
/// materializer always uses the tight minimum.
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
        min_rank_for_secure_width(sis_family, d as u32, collision, width).ok_or_else(|| {
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
    // Carry the audited SIS-floor bucket on each role. `col_len` stays
    // at the `params_only` placeholder value of `0` — the layout is
    // filled in by a subsequent `with_layout`/`with_decomp`, which now
    // preserves `collision_inf` from `self`, so the audit at the next
    // `AjtaiKeyParams::try_new` boundary sees the right bucket.
    //
    // `new_unchecked` is required here because `col_len = 0` is an
    // intentional placeholder; the strict audit on `AjtaiKeyParams::new`
    // rejects it (and would otherwise re-introduce the silent-permissive
    // bypass that this entire path exists to close).
    result.a_key = AjtaiKeyParams::new_unchecked(sis_family, n_a, 0, collisions.a, d);
    result.b_key = AjtaiKeyParams::new_unchecked(sis_family, n_b, 0, collisions.bd, d);
    result.d_key = AjtaiKeyParams::new_unchecked(sis_family, n_d, 0, collisions.bd, d);
    Ok(result)
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
    // Checked: malformed verifier-reachable inputs (e.g. `log_basis >= 32`)
    // must surface as `AkitaError`, not a panic.
    let bd_collision = 1u32
        .checked_shl(lp.log_basis)
        .and_then(|bound| bound.checked_sub(1))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root collision bound overflow for D={d} lb={}",
                lp.log_basis
            ))
        })?;
    let a_raw = a_role_base_norm(lp.log_basis, decomp.log_commit_bound, inputs.level == 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root A-role base norm overflow for family={sis_family:?}, D={d} lb={}",
                lp.log_basis
            ))
        })?;
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
    // Size the outer B-matrix width against the *secure* A-rank for this
    // layout's inner width, not the (possibly smaller) provisional rank the
    // layout was built with. Otherwise `n_b` would be sized against a
    // narrower `outer_width` than `n_a · δ_open · num_blocks`, which the
    // runtime expansion reconstructs from the stored `n_a`.
    let exact_outer_width = {
        let n_a =
            min_rank_for_secure_width(sis_family, d as u32, a_collision, lp.inner_width() as u64)
                .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "missing secure root A-row rank for family={sis_family:?} D={d} \
                 lb={} inner_width={}",
                    lp.log_basis,
                    lp.inner_width()
                ))
            })?;
        n_a.checked_mul(lp.num_digits_open)
            .and_then(|w| w.checked_mul(lp.num_blocks))
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

/// Apply [`sis_derived_root_params_for_layout`] to an explicit root layout
/// and re-attach the layout to the resulting params.
///
/// Used by the planner DP's candidate evaluator and by tests that
/// pre-compute a root layout.
///
/// # Errors
///
/// Propagates the SIS-floor lookup error from
/// [`sis_derived_root_params_for_layout`].
pub fn root_level_params_for_layout_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1: SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, AkitaError> {
    let params = sis_derived_root_params_for_layout(
        sis_family,
        d,
        decomp,
        stage1,
        ring_subfield_norm_bound,
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
