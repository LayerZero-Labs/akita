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
use akita_types::generated::sis_floor::{
    ceil_supported_collision, min_rank_for_secure_width, sis_max_widths,
};
use akita_types::layout::digit_math::{compute_num_digits_fold_with_claims, optimal_m_r_split};
use akita_types::{
    decomp_depths, exact_planned_level_execution, level_layout_from_params,
    recursive_level_layout_from_params, AjtaiKeyParams, AkitaScheduleInputs, AkitaSchedulePlan,
    CommitmentEnvelope, DecompositionParams, LevelParams, SisModulusFamily,
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

/// Pick level-params for one level + log-basis.
///
/// Prefers the exact entry from a pre-materialized
/// [`AkitaSchedulePlan`] (`schedule_plan = Cfg::schedule_plan(singleton_key(num_vars))?`
/// — fixed throughout a search), otherwise derives SIS-secure recursive
/// params from the envelope (level > 0) or returns the envelope-row-floor
/// params-only `LevelParams` (level 0 / final fallback).
///
/// `stage1_chooser` resolves the sparse-challenge config for a ring
/// dimension; it's the same hook `Cfg::stage1_challenge_config` plays at
/// runtime, threaded through as a value to keep this function free of
/// `<Cfg>` plumbing.
///
/// # Errors
///
/// Returns an error if `stage1_chooser` rejects `d`, if exact-plan
/// resolution fails, or if envelope-driven recursive derivation
/// over/underflows.
#[allow(clippy::too_many_arguments)]
pub fn level_params_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    ring_subfield_norm_bound: u32,
    schedule_plan: Option<&AkitaSchedulePlan>,
    envelope: &CommitmentEnvelope,
    stage1_chooser: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    if let Some(plan) = schedule_plan {
        if let Some(planned_level) =
            exact_planned_level_execution(plan, inputs, log_basis, stage1_chooser)?
        {
            return Ok(planned_level.level.lp.clone());
        }
    }
    let stage1_config = stage1_chooser(d)?;

    if inputs.level > 0 {
        if let Some(params) = sis_derived_recursive_params(
            sis_family,
            d,
            decomp,
            log_basis,
            inputs.current_w_len,
            &stage1_config,
            ring_subfield_norm_bound,
            envelope,
        ) {
            if let Ok(lp) =
                recursive_level_layout_from_params(&params, inputs.current_w_len, decomp)
            {
                return Ok(lp);
            }
            return Ok(params);
        }
    }

    Ok(LevelParams::params_only(
        sis_family,
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    ))
}

/// Current-level commitment-layout hook used by the planner DP at
/// recursive (level > 0) candidate evaluation.
///
/// Level 0 delegates to [`root_level_layout_with_log_basis`]. Level > 0
/// reads params via [`level_params_with_log_basis`] (which prefers the
/// plan's exact entry) and re-applies the recursive layout.
///
/// # Errors
///
/// Returns an error if the params lookup or the recursive layout
/// derivation fails.
#[allow(clippy::too_many_arguments)]
pub fn current_level_layout_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    ring_subfield_norm_bound: u32,
    schedule_plan: Option<&AkitaSchedulePlan>,
    envelope: &CommitmentEnvelope,
    stage1_chooser: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    if inputs.level == 0 {
        return root_level_layout_with_log_basis(
            sis_family,
            d,
            decomp,
            stage1_chooser(d)?,
            ring_subfield_norm_bound,
            inputs,
            log_basis,
        );
    }
    let params = level_params_with_log_basis(
        sis_family,
        d,
        decomp,
        ring_subfield_norm_bound,
        schedule_plan,
        envelope,
        stage1_chooser,
        inputs,
        log_basis,
    )?;
    let layout = recursive_level_layout_from_params(&params, inputs.current_w_len, decomp)?;
    Ok(params.with_layout(&layout))
}

/// Direct-step level-params hook used by the planner DP and the schedule
/// materializer.
///
/// Level 0 delegates to [`root_level_layout_with_log_basis`]. Level > 0
/// derives recursive params straight from the envelope (no
/// `Cfg::schedule_plan` consultation) and applies the recursive layout —
/// this is the "ship the witness directly at level N" hypothesis that
/// the planner evaluates as one alternative.
///
/// `envelope` is fixed throughout a search (it only depends on
/// `inputs.num_vars`, which is the polynomial size); callers compute it
/// once at `SearchOptions` / `PlanPolicy` construction time.
///
/// # Errors
///
/// Returns an error if the derivation cannot satisfy SIS-secure widths
/// for the requested level/basis combination.
#[allow(clippy::too_many_arguments)]
pub fn direct_level_params_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    root_decomp: DecompositionParams,
    stage1_config: SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    envelope: &CommitmentEnvelope,
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    if inputs.level == 0 {
        return root_level_layout_with_log_basis(
            sis_family,
            d,
            root_decomp,
            stage1_config,
            ring_subfield_norm_bound,
            inputs,
            log_basis,
        );
    }
    let params = sis_derived_recursive_params(
        sis_family,
        d,
        root_decomp,
        log_basis,
        inputs.current_w_len,
        &stage1_config,
        ring_subfield_norm_bound,
        envelope,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "failed to derive direct terminal params for level {} at num_vars={}",
            inputs.level, inputs.num_vars
        ))
    })?;
    akita_types::recursive_level_layout_from_params(&params, inputs.current_w_len, root_decomp)
}

/// Derive SIS-secure recursive (level > 0) params from the active envelope.
///
/// Builds the recursive layout for the envelope's `(n_a, n_b=1, n_d=1)`
/// row floors and then runs the layout-flavoured SIS derivation. Returns
/// `None` when either step rejects the candidate.
///
/// Sibling of [`sis_derived_recursive_params_for_layout`]: this is the
/// "I have an envelope, give me both a layout and SIS-secure params"
/// entry point; the `_for_layout` form is the "I already have a layout"
/// entry point.
#[allow(clippy::too_many_arguments)]
pub fn sis_derived_recursive_params(
    sis_family: SisModulusFamily,
    d: usize,
    root_decomp: DecompositionParams,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    // The envelope-shaped params-only `LevelParams` is just plumbing for
    // `recursive_level_layout_from_params`, so it stays an inline expression
    // with no named binding.
    let layout = recursive_level_layout_from_params(
        &LevelParams::params_only(
            sis_family,
            d,
            log_basis,
            envelope.max_n_a,
            1,
            1,
            stage1_config.clone(),
        ),
        current_w_len,
        root_decomp,
    )
    .ok()?;
    sis_derived_recursive_params_for_layout(
        sis_family,
        d,
        log_basis,
        stage1_config,
        ring_subfield_norm_bound,
        envelope,
        &layout,
    )
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
    // Checked: malformed inputs (e.g. `log_basis >= 32`) reach this
    // function transitively from `schedule_plan_from_table_entry` on the
    // verifier replay path, so the shift must not panic.
    let bd_collision = 1u32.checked_shl(log_basis).and_then(|b| b.checked_sub(1))?;
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
    // Checked: malformed verifier-reachable inputs (e.g. `log_basis >= 32`)
    // must surface as `AkitaError`, not a panic. Matches `root_a_rank_cap`.
    let bd_collision = 1u32
        .checked_shl(lp.log_basis)
        .and_then(|bound| bound.checked_sub(1))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root collision bound overflow for D={d} lb={}",
                lp.log_basis
            ))
        })?;
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

/// Derive the root commit layout for a root-direct schedule at `num_vars`.
///
/// Used by both the planner DP and the schedule-table materializer to fill
/// `DirectStep.commit_params` when the schedule emits a root-direct step.
/// Consumers (`Cfg::get_params_for_batched_commitment`, prover/verifier
/// commit paths) then read commit params straight off the schedule, with
/// no out-of-band fallback derivation.
///
/// Handles two regimes:
///
/// - `num_vars > trailing_zeros(d)` (normal root): iterates root A-row rank
///   against the audited SIS-floor table, computing layout via
///   [`derived_root_commitment_layout_from_params`] and reproving via
///   [`sis_derived_root_params_for_layout`].
/// - `num_vars <= trailing_zeros(d)` (tiny root): fixed-point convergence
///   over the SIS-derived params, allowing a zero-outer layout that fits
///   inside one padded ring element.
///
/// # Errors
///
/// Returns an error if no SIS-floor row covers the candidate widths, the
/// rank-cap iteration does not converge, or the layout arithmetic
/// overflows.
pub fn root_direct_commit_layout(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1: SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    num_vars: usize,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
    };
    let alpha = (d as u32).trailing_zeros() as usize;

    if num_vars > alpha {
        // Normal root: iterate A-row rank against the SIS-floor table.
        let rank_cap = root_a_rank_cap(
            sis_family,
            d,
            &decomp,
            log_basis,
            &stage1,
            ring_subfield_norm_bound,
        )?;
        let mut candidate_n_a = 1usize;
        for _ in 0..rank_cap {
            let candidate_params = LevelParams::params_only(
                sis_family,
                d,
                log_basis,
                candidate_n_a,
                1,
                1,
                stage1.clone(),
            );
            let root_lp = derived_root_commitment_layout_from_params(
                inputs,
                decomp,
                &candidate_params,
                false,
            )?;
            let derived_params = sis_derived_root_params_for_layout(
                sis_family,
                d,
                decomp,
                stage1.clone(),
                ring_subfield_norm_bound,
                inputs,
                &root_lp,
            )?;
            if derived_params.a_key.row_len() == candidate_n_a {
                return Ok(derived_params.with_layout(&root_lp));
            }
            candidate_n_a = derived_params.a_key.row_len();
        }
        return Err(AkitaError::InvalidSetup(format!(
            "failed to converge on self-consistent root A-row rank for D={d} lb={log_basis}"
        )));
    }

    // Tiny-root: fits in one padded ring element, allow zero-outer layout.
    let mut params = LevelParams::params_only(sis_family, d, log_basis, 1, 1, 1, stage1.clone());
    let layout_decomp = DecompositionParams {
        log_basis,
        ..decomp
    };
    for _ in 0..4 {
        let layout = level_layout_from_params(0, 0, &params, layout_decomp, 0)?;
        let derived_params = sis_derived_root_params_for_layout(
            sis_family,
            d,
            decomp,
            stage1.clone(),
            ring_subfield_norm_bound,
            inputs,
            &layout,
        )?
        .with_layout(&layout);
        if (
            derived_params.a_key.row_len(),
            derived_params.b_key.row_len(),
            derived_params.d_key.row_len(),
        ) == (
            params.a_key.row_len(),
            params.b_key.row_len(),
            params.d_key.row_len(),
        ) {
            return Ok(derived_params);
        }
        params = derived_params;
    }
    Err(AkitaError::InvalidSetup(format!(
        "failed to converge on tiny-root params for D={d} at num_vars={num_vars}"
    )))
}

/// Audited root-rank-cap iteration: iterate the root A-row rank against
/// the SIS-floor table until self-consistent, then return the converged
/// `LevelParams`.
///
/// This is the "normal root" variant used by the planner DP, the table
/// materializer, and the config's `level_params_with_log_basis` fast-path.
/// For root-direct (tiny-root) layouts use
/// [`root_direct_commit_layout`] instead.
///
/// # Errors
///
/// Returns an error when the rank-cap iteration does not converge, when
/// the SIS-floor table does not cover the candidate widths, or when the
/// layout arithmetic overflows.
pub fn root_level_layout_with_log_basis(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: DecompositionParams,
    stage1: SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    let rank_cap = root_a_rank_cap(
        sis_family,
        d,
        &decomp,
        log_basis,
        &stage1,
        ring_subfield_norm_bound,
    )?;
    let mut candidate_n_a = 1usize;
    for _ in 0..rank_cap {
        let candidate_params = LevelParams::params_only(
            sis_family,
            d,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1.clone(),
        );
        let root_lp =
            derived_root_commitment_layout_from_params(inputs, decomp, &candidate_params, false)?;
        let derived_params = sis_derived_root_params_for_layout(
            sis_family,
            d,
            decomp,
            stage1.clone(),
            ring_subfield_norm_bound,
            inputs,
            &root_lp,
        )?;
        if derived_params.a_key.row_len() == candidate_n_a {
            return Ok(derived_params.with_layout(&root_lp));
        }
        candidate_n_a = derived_params.a_key.row_len();
    }
    Err(AkitaError::InvalidSetup(format!(
        "failed to converge on self-consistent root A-row rank for D={d} lb={log_basis}"
    )))
}

/// Apply [`sis_derived_root_params_for_layout`] to an explicit root layout
/// and re-attach the layout to the resulting params.
///
/// This is the one-line "post-process the layout we already have" sister
/// of [`root_level_layout_with_log_basis`]. Used by the planner DP's
/// candidate evaluator and by tests that pre-compute a root layout.
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

/// Number of audited A-row SIS-rank buckets available for the root A role
/// at this `(sis_family, d, log_basis, decomp.log_commit_bound, stage1,
/// ring_subfield_norm_bound)`. Used as the iteration cap when probing
/// self-consistent root A-row ranks.
///
/// `log_basis` is the candidate basis being evaluated (which the planner
/// DP varies independently of `decomp.log_basis`), NOT `decomp.log_basis`
/// — the rank cap depends on the candidate's collision bound, so reading
/// it off the default decomposition would understate the cap and produce
/// spurious "failed to converge" errors at non-default bases.
fn root_a_rank_cap(
    sis_family: SisModulusFamily,
    d: usize,
    decomp: &DecompositionParams,
    log_basis: u32,
    stage1: &SparseChallengeConfig,
    ring_subfield_norm_bound: u32,
) -> Result<usize, AkitaError> {
    let bd_collision = 1u32
        .checked_shl(log_basis)
        .and_then(|bound| bound.checked_sub(1))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "root collision bound overflow for D={d} lb={log_basis}"
            ))
        })?;
    // Root level (level == 0); for `log_commit_bound == 1` the A-role bound
    // is the tight constant `2`, otherwise it equals the B/D collision bound.
    let a_raw = if decomp.log_commit_bound == 1 {
        2
    } else {
        bd_collision
    };
    let a_collision_raw = a_raw
        .checked_mul(stage1.infinity_norm())
        .and_then(|collision| collision.checked_mul(ring_subfield_norm_bound))
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
    sis_max_widths(sis_family, d as u32, a_collision)
        .map(<[u64]>::len)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "missing root A-role SIS rank table for family={sis_family:?}, D={d}, \
                 collision_inf={a_collision}"
            ))
        })
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
