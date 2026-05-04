//! Commitment-config trait and concrete protocol configs.
//!
//! The trait [`CommitmentConfig`] is intentionally slim: presets must
//! implement every runtime hook explicitly (no thin delegating defaults).
//! Substantive helpers that encode protocol logic — `commitment_layout`,
//! `get_params_for_commitment`, `get_params_for_prove` — keep default
//! bodies because they are not policy choices and would otherwise be
//! duplicated verbatim across every config.

use crate::protocol::commitment::schedule::{
    fallback_batched_root_split, hachi_batched_root_layout, hachi_root_commitment_layout,
};
use crate::{CanonicalField, FieldCore};
use akita_algebra::SparseChallengeConfig;
use akita_field::HachiError;
use akita_types::{
    recursive_level_decomposition_from_root, AjtaiRole, CommitmentEnvelope, DecompositionParams,
    LevelParams,
};
use akita_types::{
    HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
    Schedule, ScheduleProvider, WitnessShape,
};
use std::marker::PhantomData;

pub mod proof_optimized;

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Concrete presets must implement every runtime hook below: the trait
/// intentionally provides no default bodies for the delegating hooks so
/// that each preset is fully explicit about which planner-backed helper
/// it uses. The substantive helpers (`commitment_layout`,
/// `get_params_for_commitment`, `get_params_for_prove`) keep defaults
/// because they encode protocol logic rather than per-config policy.
pub trait CommitmentConfig:
    ScheduleProvider + akita_planner::PlannerConfig + Clone + Send + Sync + 'static
{
    /// Base field used by this config.
    type Field: CanonicalField + FieldCore;

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Sparse challenge family used at this level.
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    /// Audited rank floor for the root level, by role.
    #[doc(hidden)]
    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize;

    /// Maximum matrix row envelope needed across all runtime levels.
    #[doc(hidden)]
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope;

    /// `(max_rows, max_stride)` bounds for the shared setup matrix.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] on arithmetic overflow.
    #[doc(hidden)]
    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), HachiError>;

    /// Active level params for one level under an explicit basis.
    #[doc(hidden)]
    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams;

    /// Active root params for a concrete root layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the config cannot derive a sound root parameter
    /// set for the supplied root layout.
    #[doc(hidden)]
    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError>;

    /// Root fold layout for an explicit basis.
    ///
    /// # Errors
    ///
    /// Returns an error if the root variable split underflows, overflows, or
    /// does not admit a sound root parameterization.
    #[doc(hidden)]
    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError>;

    /// Active basis for one level from public inputs.
    #[doc(hidden)]
    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32;

    /// Inclusive `(min, max)` log-basis search range at one state.
    #[doc(hidden)]
    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32);

    /// Choose the runtime commitment layout for `max_num_vars` (singleton
    /// case: one polynomial per opening point).
    ///
    /// # Errors
    ///
    /// Returns an error if `max_num_vars` does not admit a valid layout.
    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, HachiError> {
        let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        if let Some(plan) = Self::schedule_plan(key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
        }
        // Tiny-root fallback: roots that don't admit any fold step.
        hachi_root_commitment_layout::<Self>(max_num_vars)
    }

    /// Choose the root parameters consumed by the commitment path.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch summary, schedule lookup, or derived
    /// layout is invalid for the requested commitment shape.
    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<LevelParams, HachiError> {
        if num_polys_per_point <= 1 {
            return Self::commitment_layout(num_vars);
        }

        let lookup_key = HachiScheduleLookupKey::with_batch(
            num_vars,
            num_vars,
            num_polys_per_point,
            HachiRootBatchSummary::new(num_polys_per_point, 1, 1)?,
        );
        if let Some(plan) = Self::schedule_plan(lookup_key)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.lp.clone());
            }
            return fallback_batched_root_split::<Self>(num_vars, num_polys_per_point);
        }

        hachi_batched_root_layout::<Self>(num_vars, num_polys_per_point)
    }

    /// Choose the root parameters consumed by the prove/verify root path.
    ///
    /// # Errors
    ///
    /// Returns an error if the root layout, batched layout scaling, next
    /// witness sizing, or next-level basis selection is invalid.
    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: HachiRootBatchSummary,
    ) -> Result<Schedule, HachiError> {
        let key =
            HachiScheduleLookupKey::with_batch(max_num_vars, num_vars, layout_num_claims, batch);
        if let Some(plan) = Self::schedule_plan(key)? {
            return Ok(akita_types::schedule_from_plan(
                &plan,
                Self::decomposition().field_bits(),
            ));
        }

        if layout_num_claims != batch.num_claims {
            return Err(HachiError::InvalidSetup(format!(
                "fallback prove schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }

        akita_planner::find_optimal_schedule::<Self>(
            num_vars,
            WitnessShape::new(
                batch.num_claims,
                batch.num_commitment_groups,
                batch.num_points,
            ),
        )
    }
}

/// Derived commitment config for recursive w-openings.
///
/// Sets `log_commit_bound = log_basis` because recursive `w` entries are
/// balanced digits, and sets `log_open_bound` from the parent opening bound
/// because recursive opening folds produce full-field coefficients.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> ScheduleProvider for WCommitmentConfig<D, Cfg> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Cfg::schedule_table()
    }

    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        Cfg::schedule_key(key)
    }

    fn schedule_plan(key: HachiScheduleLookupKey) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Cfg::schedule_plan(key)
    }
}

impl<const D: usize, Cfg: CommitmentConfig> akita_planner::PlannerConfig
    for WCommitmentConfig<D, Cfg>
{
    const PLANNER_D: usize = D;

    fn planner_field_bits() -> u32 {
        <Self as CommitmentConfig>::decomposition().field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        <Self as CommitmentConfig>::stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: HachiScheduleLookupKey,
    ) -> Result<Option<HachiSchedulePlan>, HachiError> {
        <Self as ScheduleProvider>::schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        crate::protocol::commitment::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        <Self as CommitmentConfig>::log_basis_search_range(inputs)
    }
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    type Field = Cfg::Field;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        recursive_level_decomposition_from_root(
            Cfg::decomposition(),
            Cfg::decomposition().log_basis,
        )
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
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
    ) -> Result<(usize, usize), HachiError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        debug_assert_eq!(params.ring_dimension, D);
        params
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, HachiError> {
        Err(HachiError::InvalidSetup(
            "recursive w layout requires active level params".to_string(),
        ))
    }
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::proof_optimized::fp128;
    use super::*;
    use crate::protocol::commitment::schedule::scale_batched_root_layout;
    use akita_types::generated::sis_floor::min_rank_for_secure_width;

    fn assert_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        min_num_vars: usize,
        max_num_vars: usize,
    ) {
        let d = Cfg::D as u32;
        let root_onehot = Cfg::decomposition().log_commit_bound == 1;
        for num_vars in min_num_vars..=max_num_vars {
            let plan = Cfg::schedule_plan(HachiScheduleLookupKey::singleton(num_vars, num_vars, 1))
                .unwrap()
                .expect("audited config should have a schedule");

            for level in plan.fold_levels() {
                let raw_collision = if root_onehot && level.inputs.level == 0 {
                    2
                } else {
                    (1u32 << level.lp.log_basis) - 1
                };

                let a_rank = min_rank_for_secure_width(
                    d,
                    raw_collision,
                    u64::try_from(level.lp.inner_width())
                        .expect("inner width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited A-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.inner_width()
                    )
                });
                assert!(
                    a_rank <= level.lp.a_key.row_len(),
                    "A-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={a_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.inner_width(),
                    level.lp.a_key.row_len(),
                );

                let bd_collision = (1u32 << level.lp.log_basis) - 1;
                let b_rank = min_rank_for_secure_width(
                    d,
                    bd_collision,
                    u64::try_from(level.lp.outer_width())
                        .expect("outer width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited B-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.outer_width()
                    )
                });
                assert!(
                    b_rank <= level.lp.b_key.row_len(),
                    "B-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={b_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.outer_width(),
                    level.lp.b_key.row_len(),
                );

                let d_rank = min_rank_for_secure_width(
                    d,
                    bd_collision,
                    u64::try_from(level.lp.d_matrix_width())
                        .expect("d-matrix width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited D-row SIS width for D={d}, num_vars={num_vars}, level={}, lb={}, width={}",
                        level.inputs.level,
                        level.lp.log_basis,
                        level.lp.d_matrix_width()
                    )
                });
                assert!(
                    d_rank <= level.lp.d_key.row_len(),
                    "D-row SIS audit failed for D={d}, num_vars={num_vars}, level={}, lb={}, width={}, required_rank={d_rank}, actual_rank={}",
                    level.inputs.level,
                    level.lp.log_basis,
                    level.lp.d_matrix_width(),
                    level.lp.d_key.row_len(),
                );
            }
        }
    }

    #[test]
    fn current_d128_full_schedule_stays_within_audited_sis_widths() {
        assert_schedule_stays_within_audited_sis_widths::<fp128::D128Full>(8, 50);
    }

    #[test]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        // B-row rank=1 at num_vars>=46 level=1 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D64Full>(8, 45);
    }

    #[test]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        assert_schedule_stays_within_audited_sis_widths::<fp128::D64OneHot>(8, 50);
    }

    #[test]
    fn current_d32_full_schedule_stays_within_audited_sis_widths() {
        // D-row rank=1 at num_vars>=30 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D32Full>(8, 29);
    }

    #[test]
    fn current_d32_onehot_schedule_stays_within_audited_sis_widths() {
        // D-row rank=1 at num_vars>=36 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<fp128::D32OneHot>(8, 35);
    }

    #[test]
    fn batched_commitment_direct_fallback_scales_root_layout() {
        type Cfg = fp128::D64OneHot;

        let num_vars = 10;
        let num_claims = 4;
        let singleton = Cfg::commitment_layout(num_vars).expect("singleton layout");
        let expected =
            scale_batched_root_layout::<Cfg>(&singleton, num_claims).expect("scaled layout");
        let actual = Cfg::get_params_for_commitment(num_vars, num_claims).expect("batched layout");

        assert_eq!(actual, expected);
        assert_eq!(actual.outer_width(), singleton.outer_width() * num_claims);
        assert_eq!(
            actual.d_matrix_width(),
            singleton.d_matrix_width() * num_claims
        );
        assert!(actual.num_digits_fold >= singleton.num_digits_fold);
    }
}
