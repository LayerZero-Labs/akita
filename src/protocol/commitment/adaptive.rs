//! Shared planner-backed helpers used by every concrete commitment config.
//!
//! Concrete presets only override [`CommitmentConfig::schedule_table`] (and
//! the audited rank floors); every other trait method has a default body
//! that calls into one of the helpers below. The helpers themselves prefer
//! the pre-computed schedule tables in `super::generated/*` and fall back
//! to [`crate::planner::schedule_params::find_optimal_schedule`] only on
//! cache misses.

use super::config::{AjtaiRole, CommitmentConfig, CommitmentEnvelope};
use super::generated::table_entry_envelope_for_max_num_vars;
use super::schedule::{
    exact_planned_level_execution, fallback_batched_root_split, generated_schedule_plan_from_table,
    hachi_recursive_level_layout_from_params, planned_log_basis_at_level_from_schedule,
    planned_schedule_key_from_schedule, HachiRootBatchSummary, HachiScheduleInputs,
    HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::planner::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};
use crate::protocol::params::{AjtaiKeyParams, LevelParams};

/// Inclusive minimum of the adaptive log-basis search range.
pub(crate) const ADAPTIVE_LOG_BASIS_MIN: u32 = 2;
/// Inclusive maximum of the adaptive log-basis search range.
pub(crate) const ADAPTIVE_LOG_BASIS_MAX: u32 = 6;

/// Inclusive `(min, max)` log-basis search range used by every adaptive preset.
pub(crate) fn adaptive_log_basis_search_range() -> (u32, u32) {
    (ADAPTIVE_LOG_BASIS_MIN, ADAPTIVE_LOG_BASIS_MAX)
}

/// Read the planned schedule for `key` from the config's generated table.
///
/// Returns:
/// - `Ok(Some(plan))` when the config has a table containing this key,
/// - `Ok(None)` when the config has no table or the table has no entry for
///   `key` (callers fall back to the runtime planner),
/// - `Err(_)` only on genuine table-decoding failures.
fn lookup_planned_schedule<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    let Some(table) = Cfg::schedule_table() else {
        return Ok(None);
    };
    generated_schedule_plan_from_table::<Cfg>(key, table)
}

/// Adaptive `schedule_plan` impl.
pub(crate) fn adaptive_schedule_plan<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    lookup_planned_schedule::<Cfg>(key)
}

/// Adaptive `schedule_key` impl: derive a stable identifier from the planned
/// schedule (or from the lookup key when no entry exists).
pub(crate) fn adaptive_schedule_key<Cfg: CommitmentConfig>(key: HachiScheduleLookupKey) -> String {
    match lookup_planned_schedule::<Cfg>(key) {
        Ok(Some(plan)) => planned_schedule_key_from_schedule(key, &plan),
        _ => format!(
            "generated-miss/d{}/max{}/num{}/claims{}/batch{}g{}p{}",
            Cfg::D,
            key.max_num_vars,
            key.num_vars,
            key.layout_num_claims,
            key.batch.num_claims,
            key.batch.num_commitment_groups,
            key.batch.num_points,
        ),
    }
}

/// Adaptive `log_basis_at_level` impl: read from the planned schedule when
/// available; otherwise fall back to the root decomposition's basis.
pub(crate) fn adaptive_log_basis_at_level<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
) -> u32 {
    let key = HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    match lookup_planned_schedule::<Cfg>(key) {
        Ok(Some(plan)) => planned_log_basis_at_level_from_schedule(&plan, inputs)
            .expect("generated adaptive schedule must be derivable from public inputs"),
        _ => Cfg::decomposition().log_basis,
    }
}

/// Adaptive `level_params_with_log_basis` impl: prefer the exact planned
/// level when the public inputs match; otherwise derive SIS-secure recursive
/// params (or fall back to the envelope for level 0).
pub(crate) fn adaptive_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let singleton_key =
        HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    if let Ok(Some(plan)) = lookup_planned_schedule::<Cfg>(singleton_key) {
        if let Ok(Some(planned_level)) =
            exact_planned_level_execution::<Cfg>(&plan, inputs, log_basis)
        {
            return planned_level.level.lp.clone();
        }
    }
    let envelope = Cfg::envelope(inputs.max_num_vars);
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);

    if inputs.level > 0 {
        if let Some(params) = sis_derived_recursive_params::<Cfg>(
            d,
            log_basis,
            inputs.current_w_len,
            &stage1_config,
            &envelope,
        ) {
            if let Ok(lp) =
                hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)
            {
                return lp;
            }
            return params;
        }
    }

    LevelParams::params_only(
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    )
}

/// Adaptive `root_level_params_for_layout_with_log_basis` impl.
pub(crate) fn adaptive_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    let params = sis_derived_root_params_for_layout::<Cfg>(inputs, lp)?;
    Ok(params.with_layout(lp))
}

/// Adaptive `root_level_layout_with_log_basis` impl.
pub(crate) fn adaptive_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, HachiError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    for _ in 0..crate::planner::sis_security::MAX_RANK {
        let candidate_params = LevelParams::params_only(
            Cfg::D,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1_config.clone(),
        );
        let root_lp =
            derived_root_commitment_layout_from_params::<Cfg>(inputs, &candidate_params, false)?;
        let derived_params = sis_derived_root_params_for_layout::<Cfg>(inputs, &root_lp)?;
        if derived_params.a_key.row_len() == candidate_n_a {
            return Ok(derived_params.with_layout(&root_lp));
        }
        candidate_n_a = derived_params.a_key.row_len();
    }
    Err(HachiError::InvalidSetup(format!(
        "failed to converge on self-consistent root A-row rank for D={} lb={log_basis}",
        Cfg::D
    )))
}

/// Adaptive `envelope` impl: combine the audited rank floor with the maximum
/// rank reached by any planned level for `max_num_vars`.
///
/// When the config ships a generated schedule table that records fold-level
/// ranks for `max_num_vars`, those ranks bound the envelope from above. When
/// the table is missing the entry — or every step is direct (no fold ranks
/// to read) — only the audited floor applies.
pub(crate) fn adaptive_envelope<Cfg: CommitmentConfig>(max_num_vars: usize) -> CommitmentEnvelope {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, max_num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, max_num_vars);
    let mut envelope = CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    };
    if let Some(table) = Cfg::schedule_table() {
        if let Some((gen_n_a, gen_n_b, gen_n_d)) =
            table_entry_envelope_for_max_num_vars(table, max_num_vars)
        {
            envelope.max_n_a = envelope.max_n_a.max(gen_n_a);
            envelope.max_n_b = envelope.max_n_b.max(gen_n_b);
            envelope.max_n_d = envelope.max_n_d.max(gen_n_d);
        }
    }
    envelope
}

/// Size the shared setup matrix from the planned schedule.
pub(crate) fn adaptive_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), HachiError> {
    if max_num_batched_polys == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(HachiError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys ({max_num_batched_polys})"
        )));
    }

    let batch_summary =
        HachiRootBatchSummary::new(max_num_batched_polys, max_num_batched_polys, max_num_points)?;
    let cached_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        max_num_batched_polys,
        batch_summary,
    );

    let fallback = fallback_batched_root_split::<Cfg>(max_num_vars, max_num_batched_polys)?;

    let fold_levels: Vec<LevelParams> = if let Some(plan) = Cfg::schedule_plan(cached_key)? {
        plan.fold_levels().map(|level| level.lp.clone()).collect()
    } else {
        use crate::planner::schedule_params::{find_optimal_schedule, Step, WitnessShape};
        let shape = WitnessShape {
            num_claims: max_num_batched_polys,
            num_commitment_groups: max_num_batched_polys,
            num_points: max_num_points,
        };
        let schedule = find_optimal_schedule::<Cfg>(max_num_vars, shape)?;
        schedule
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Fold(fold) => Some(fold.params.clone()),
                Step::Direct(_) => None,
            })
            .collect()
    };

    Ok(reduce_level_params_to_matrix_size(
        std::iter::once(&fallback).chain(fold_levels.iter()),
    ))
}

fn reduce_level_params_to_matrix_size<'a, I>(level_params: I) -> (usize, usize)
where
    I: IntoIterator<Item = &'a LevelParams>,
{
    let mut max_rows: usize = 1;
    let mut max_stride: usize = 1;
    for lp in level_params {
        max_rows = max_rows
            .max(lp.a_key.row_len())
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        max_stride = max_stride
            .max(lp.inner_width())
            .max(lp.outer_width())
            .max(lp.d_matrix_width());
    }
    (max_rows, max_stride)
}

// -----------------------------------------------------------------------
// SIS-derived recursive / root params, sharing one secure-rank core.
// -----------------------------------------------------------------------

/// Compute (depth_commit, depth_open) for one decomposition.
pub(crate) fn decomp_depths(decomp: super::config::DecompositionParams) -> (usize, usize) {
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
    (depth_commit, depth_open)
}

/// SIS-secure rank derivation inputs, bundled to keep
/// [`sis_secure_level_params`] under clippy's argument-count cap.
struct SisRoleWidths {
    inner: usize,
    outer: usize,
    d_matrix: usize,
}

/// Build a SIS-secure `LevelParams` from the explicit width budget.
///
/// Looks up the minimum module-SIS rank for each of `(a, b, d)` against the
/// 128-bit security tables; falls back to `fallback` when the table does not
/// cover the requested width.
fn sis_secure_level_params(
    d: usize,
    log_basis: u32,
    a_collision: u32,
    bd_collision: u32,
    widths: SisRoleWidths,
    fallback: Option<&CommitmentEnvelope>,
    stage1_config: SparseChallengeConfig,
) -> Result<LevelParams, HachiError> {
    use super::generated::sis_floor::min_rank_for_secure_width;

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

fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    use super::generated::sis_floor::ceil_supported_collision;

    let tentative =
        LevelParams::params_only(d, log_basis, envelope.max_n_a, 1, 1, stage1_config.clone());
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&tentative, current_w_len).ok()?;

    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = bd_collision;
    let a_collision = ceil_supported_collision(d as u32, a_raw * stage1_config.max_abs_coeff())?;

    // Compute the exact widths from the real layout, then size each role
    // for SIS security with the envelope as the fallback.
    let exact_outer_width = {
        use super::generated::sis_floor::min_rank_for_secure_width;
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

fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    use super::generated::sis_floor::ceil_supported_collision;

    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let bd_collision = (1u32 << lp.log_basis) - 1;
    let a_raw = if inputs.level == 0 && Cfg::decomposition().log_commit_bound == 1 {
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

pub(crate) fn derived_root_commitment_layout_from_params<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
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

    let mut decomp = Cfg::decomposition();
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
