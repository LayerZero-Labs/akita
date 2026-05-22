//! Test-only utilities. Gated behind the `test-utils` Cargo feature so
//! production builds never link this module.
//!
//! [`PlannerCfg<Cfg>`][PlannerCfg] is a `CommitmentConfig` wrapper that adds
//! DP fallback on schedule-table miss. All `Cfg` trait hooks delegate to
//! the inner config; the three parameter accessors (`commitment_layout`,
//! `get_params_for_commitment`, `get_params_for_prove`) consult the inner
//! config's generated table first, and on miss fall through to
//! [`crate::find_optimal_schedule`] — the offline DP search restricted to
//! `<Cfg>`.
//!
//! Use this from cross-crate test fixtures that exercise table-miss
//! incidences (multipoint, non-singleton `num_t_vectors`, presets with
//! `table = None`, full setup-matrix sizing iteration, etc.). Production
//! presets cover their expected workloads via generated tables and do not
//! need this wrapper.

use std::marker::PhantomData;

use akita_challenges::SparseChallengeConfig;
use akita_config::{
    fallback_batched_root_split, matrix_envelope_for_levels,
    setup_level_params_from_runtime_schedule, CommitmentConfig,
};
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::{
    schedule_from_plan, schedule_root_fold_params, AjtaiRole, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, ClaimIncidenceSummary, CommitmentEnvelope,
    DecompositionParams, LevelParams, Schedule, SisModulusFamily,
};

use crate::find_optimal_schedule;

/// `Cfg` wrapper that routes schedule-table misses through the planner DP.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerCfg<Cfg>(PhantomData<Cfg>);

#[allow(clippy::expl_impl_clone_on_copy)]
impl<Cfg> PlannerCfg<Cfg> {
    /// Construct the marker value. `PlannerCfg<Cfg>` carries no state; all
    /// trait methods are associated functions, so this is rarely needed.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Cfg: CommitmentConfig> CommitmentConfig for PlannerCfg<Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;

    const D: usize = Cfg::D;

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn schedule_table() -> Option<GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("planner-cfg/{}", Cfg::schedule_key(key))
    }

    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        Cfg::schedule_plan(key)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Cfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        // The inner cfg's table-only sizing is a lower bound — it can miss
        // batched shapes that aren't in the generated table. We must iterate
        // every `(nv, polys, points)` shape ourselves and consult DP on
        // table miss to compute the true envelope.
        if max_num_batched_polys == 0 || max_num_points == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup matrix sizing requires nonzero poly/point counts".to_string(),
            ));
        }
        if max_num_points > max_num_batched_polys {
            return Err(AkitaError::InvalidSetup(format!(
                "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys \
                 ({max_num_batched_polys})"
            )));
        }
        let mut max_rows: usize = 1;
        let mut max_stride: usize = 1;
        for num_vars in 1..=max_num_vars {
            for num_polys in 1..=max_num_batched_polys {
                let upper_pts = num_polys.min(max_num_points);
                for num_points in 1..=upper_pts {
                    let incidence =
                        ClaimIncidenceSummary::from_counts(num_vars, num_polys, num_points)?;
                    let schedule = <Self as CommitmentConfig>::get_params_for_prove(&incidence)?;
                    let setup_levels = setup_level_params_from_runtime_schedule(&schedule.steps);
                    let fallback = fallback_batched_root_split::<Self>(num_vars, num_polys)?;
                    let (rows, stride) =
                        matrix_envelope_for_levels::<Self>(&fallback, &setup_levels)?;
                    max_rows = max_rows.max(rows);
                    max_stride = max_stride.max(stride);
                }
            }
        }
        Ok((max_rows, max_stride))
    }

    fn level_params_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Cfg::level_params_with_log_basis(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> Result<u32, AkitaError> {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
    }

    // ---- DP-aware parameter accessors ----------------------------------

    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        let key = AkitaScheduleLookupKey::singleton(max_num_vars);
        if let Some(plan) = Cfg::schedule_plan(key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
        }
        let schedule = find_optimal_schedule::<Self>(key, true)?;
        if let Some(root_params) = schedule_root_fold_params(&schedule) {
            return Ok(root_params.clone());
        }
        // Tiny-root fallback: defer to the inner cfg's table-only path.
        Cfg::commitment_layout(max_num_vars)
    }

    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
        max_num_points: usize,
    ) -> Result<LevelParams, AkitaError> {
        if num_polys_per_point == 0 || max_num_points == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment shape counts must be nonzero".to_string(),
            ));
        }
        let num_claims = num_polys_per_point
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("commitment claim count overflow".into()))?;
        if num_claims == 1 {
            return Self::commitment_layout(num_vars);
        }
        let lookup_key = AkitaScheduleLookupKey::new_with_points(
            num_vars,
            1,
            num_polys_per_point,
            num_claims,
            max_num_points,
        );
        if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
        }
        let schedule = find_optimal_schedule::<Self>(lookup_key, true)?;
        if let Some(root_params) = schedule_root_fold_params(&schedule) {
            return Ok(root_params.clone());
        }
        // Fall back to the inner cfg's same-method (singleton-derived split).
        Cfg::get_params_for_commitment(num_vars, num_polys_per_point, max_num_points)
    }

    fn get_params_for_prove(incidence: &ClaimIncidenceSummary) -> Result<Schedule, AkitaError> {
        let key = AkitaScheduleLookupKey::new_from_incidence(incidence)?;
        if let Some(plan) = Cfg::schedule_plan(key)? {
            return Ok(schedule_from_plan(&plan, Cfg::decomposition().field_bits()));
        }
        find_optimal_schedule::<Self>(key, true)
    }
}
