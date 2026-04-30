//! Shared planner-backed helpers used by every concrete commitment config.
//!
//! Each fp128 preset is a unit struct that delegates layout/schedule decisions
//! to these helpers. The helpers themselves consult the pre-computed schedule
//! tables in `super::generated/*` first; on a cache miss they fall back to
//! [`crate::planner::schedule_params::find_optimal_schedule`].

use super::config::{CommitmentConfig, CommitmentEnvelope};
use super::generated::{table_entry_envelope_for_max_num_vars, GeneratedScheduleTable};
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

fn missing_generated_schedule(err: &HachiError) -> bool {
    matches!(err, HachiError::InvalidSetup(msg) if msg.starts_with("missing generated schedule for "))
}

fn schedule_source<Cfg: CommitmentConfig>(
    table: impl Fn() -> GeneratedScheduleTable,
    key: HachiScheduleLookupKey,
) -> Result<HachiSchedulePlan, HachiError> {
    generated_schedule_plan_from_table::<Cfg>(key, table())?.ok_or_else(|| {
        HachiError::InvalidSetup(format!(
            "missing generated schedule for {} at key={key:?}",
            std::any::type_name::<Cfg>()
        ))
    })
}

/// Adaptive `schedule_plan` impl: read from the pre-computed table when an
/// entry exists; otherwise return `None` so the caller can fall back to the
/// runtime planner.
pub(crate) fn adaptive_schedule_plan<Cfg: CommitmentConfig>(
    table: impl Fn() -> GeneratedScheduleTable,
    key: HachiScheduleLookupKey,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    match schedule_source::<Cfg>(&table, key) {
        Ok(plan) => Ok(Some(plan)),
        Err(err) if missing_generated_schedule(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Adaptive `schedule_key` impl: derive a stable identifier from the planned
/// schedule (or from the lookup key when no entry exists).
pub(crate) fn adaptive_schedule_key<Cfg: CommitmentConfig>(
    table: impl Fn() -> GeneratedScheduleTable,
    key: HachiScheduleLookupKey,
) -> String {
    match schedule_source::<Cfg>(&table, key) {
        Ok(plan) => planned_schedule_key_from_schedule(key, &plan),
        Err(_) => format!(
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
    table: impl Fn() -> GeneratedScheduleTable,
    inputs: HachiScheduleInputs,
) -> u32 {
    let key = HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    match schedule_source::<Cfg>(&table, key) {
        Ok(plan) => planned_log_basis_at_level_from_schedule(&plan, inputs)
            .expect("generated adaptive schedule must be derivable from public inputs"),
        Err(_) => Cfg::decomposition().log_basis,
    }
}

/// Adaptive `level_params_with_log_basis` impl: prefer the exact planned
/// level when the public inputs match; otherwise derive SIS-secure recursive
/// params (or fall back to the envelope for level 0).
pub(crate) fn adaptive_level_params_with_log_basis<Cfg: CommitmentConfig>(
    table: impl Fn() -> GeneratedScheduleTable,
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let key = HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
    if let Ok(Some(plan)) = generated_schedule_plan_from_table::<Cfg>(key, table()) {
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
pub(crate) fn adaptive_envelope<Cfg: CommitmentConfig>(
    table: impl Fn() -> GeneratedScheduleTable,
    audited_root_a_rank: usize,
    audited_root_rank: usize,
    max_num_vars: usize,
) -> CommitmentEnvelope {
    let mut envelope = CommitmentEnvelope {
        max_n_a: audited_root_a_rank,
        max_n_b: audited_root_rank,
        max_n_d: audited_root_rank,
    };
    if let Some((gen_n_a, gen_n_b, gen_n_d)) =
        table_entry_envelope_for_max_num_vars(table(), max_num_vars)
    {
        envelope.max_n_a = envelope.max_n_a.max(gen_n_a);
        envelope.max_n_b = envelope.max_n_b.max(gen_n_b);
        envelope.max_n_d = envelope.max_n_d.max(gen_n_d);
    }
    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    };
    let alpha = Cfg::D.trailing_zeros() as usize;
    for log_basis in ADAPTIVE_LOG_BASIS_MIN..=ADAPTIVE_LOG_BASIS_MAX {
        let root_params = if max_num_vars > alpha {
            adaptive_root_level_layout_with_log_basis::<Cfg>(root_inputs, log_basis).ok()
        } else {
            converge_zero_outer_root::<Cfg>(root_inputs, log_basis)
        };
        if let Some(root_params) = root_params {
            envelope.max_n_a = envelope.max_n_a.max(root_params.a_key.row_len());
            envelope.max_n_b = envelope.max_n_b.max(root_params.b_key.row_len());
            envelope.max_n_d = envelope.max_n_d.max(root_params.d_key.row_len());
        }
    }
    envelope
}

fn converge_zero_outer_root<Cfg: CommitmentConfig>(
    root_inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Option<LevelParams> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut params = LevelParams::params_only(Cfg::D, log_basis, 1, 1, 1, stage1_config);
    let mut converged = None;
    for _ in 0..4 {
        let Ok(root_lp) =
            derived_root_commitment_layout_from_params::<Cfg>(root_inputs, &params, true)
        else {
            break;
        };
        let Ok(derived_lp) =
            adaptive_root_level_params_for_layout_with_log_basis::<Cfg>(root_inputs, &root_lp)
        else {
            break;
        };
        if (
            derived_lp.a_key.row_len(),
            derived_lp.b_key.row_len(),
            derived_lp.d_key.row_len(),
        ) == (
            params.a_key.row_len(),
            params.b_key.row_len(),
            params.d_key.row_len(),
        ) {
            converged = Some(derived_lp);
            break;
        }
        params = derived_lp;
    }
    converged
}

/// Size the shared setup matrix from the planned schedule.
pub(crate) fn adaptive_max_setup_matrix_size<Cfg: CommitmentConfig, const D: usize>(
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

    let fallback = fallback_batched_root_split::<Cfg, D>(max_num_vars, max_num_batched_polys)?;

    let fold_levels: Vec<LevelParams> = if let Some(plan) = Cfg::schedule_plan(cached_key)? {
        plan.fold_levels().map(|level| level.lp.clone()).collect()
    } else {
        use crate::planner::schedule_params::{find_optimal_schedule, Step, WitnessShape};
        let shape = WitnessShape {
            num_claims: max_num_batched_polys,
            num_commitment_groups: max_num_batched_polys,
            num_points: max_num_points,
        };
        let schedule = find_optimal_schedule::<Cfg, D>(max_num_vars, shape)?;
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
        std::iter::once(&fallback.params).chain(fold_levels.iter()),
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
// SIS-derived recursive / root params (previously in `config.rs`)
// -----------------------------------------------------------------------

fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    use super::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};

    let tentative =
        LevelParams::params_only(d, log_basis, envelope.max_n_a, 1, 1, stage1_config.clone());
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&tentative, current_w_len).ok()?;

    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = bd_collision;
    let a_collision = ceil_supported_collision(d as u32, a_raw * stage1_config.max_abs_coeff())?;

    let n_a = min_rank_for_secure_width(d as u32, a_collision, layout.inner_width() as u64)
        .unwrap_or(envelope.max_n_a);
    let exact_outer_width = n_a * layout.num_digits_open * layout.num_blocks;
    let n_b = min_rank_for_secure_width(d as u32, bd_collision, exact_outer_width as u64)
        .unwrap_or(envelope.max_n_b);
    let n_d = min_rank_for_secure_width(d as u32, bd_collision, layout.d_matrix_width() as u64)
        .unwrap_or(envelope.max_n_d);

    let mut result = LevelParams::params_only(d, log_basis, n_a, n_b, n_d, stage1_config.clone());
    result.a_key = AjtaiKeyParams::new(n_a, 0, a_collision, d);
    result.b_key = AjtaiKeyParams::new(n_b, 0, bd_collision, d);
    result.d_key = AjtaiKeyParams::new(n_d, 0, bd_collision, d);
    Some(result)
}

fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    use super::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};

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
    let n_a = min_rank_for_secure_width(d as u32, a_collision, lp.inner_width() as u64)
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing secure root A-row rank for D={} lb={} inner_width={}",
                d,
                lp.log_basis,
                lp.inner_width()
            ))
        })?;
    let n_b = min_rank_for_secure_width(d as u32, bd_collision, lp.outer_width() as u64)
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing secure root B-row rank for D={} lb={} outer_width={}",
                d,
                lp.log_basis,
                lp.outer_width()
            ))
        })?;
    let n_d = min_rank_for_secure_width(d as u32, bd_collision, lp.d_matrix_width() as u64)
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing secure root D-row rank for D={} lb={} d_matrix_width={}",
                d,
                lp.log_basis,
                lp.d_matrix_width()
            ))
        })?;
    let mut result = LevelParams::params_only(
        d,
        lp.log_basis,
        n_a as usize,
        n_b as usize,
        n_d as usize,
        stage1_config,
    );
    result.a_key = AjtaiKeyParams::new(n_a, 0, a_collision, d);
    result.b_key = AjtaiKeyParams::new(n_b, 0, bd_collision, d);
    result.d_key = AjtaiKeyParams::new(n_d, 0, bd_collision, d);
    Ok(result)
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
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold_with_claims(
        r_vars,
        params.challenge_l1_mass(),
        decomp.log_basis,
        1,
    );
    params.with_decomp(m_vars, r_vars, depth_commit, depth_open, depth_fold, 0)
}
