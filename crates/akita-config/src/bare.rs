//! Inverse wrapper for the setup-side claim-reduction sumcheck.
//!
//! [`BareCfg`] forces a base [`CommitmentConfig`] back to the bare
//! (CR-off) path regardless of whether the base preset opts into CR
//! by default. The wrapper exists so post-audit-B-1 tests can still
//! exercise the bare baseline against the production presets without
//! reaching for the underlying SIS / table machinery directly.
//!
//! All [`CommitmentConfig`] and [`akita_planner::PlannerConfig`] hooks
//! delegate to `Base`; the only differences are:
//!
//! 1. [`use_setup_claim_reduction`](CommitmentConfig::use_setup_claim_reduction)
//!    returns `false`.
//! 2. [`schedule_plan`](akita_types::ScheduleProvider::schedule_plan)
//!    consults the base preset's generated schedule table directly,
//!    bypassing the base config's CR-on guard that returns `Ok(None)`
//!    (see `proof_optimized::proof_optimized_schedule_plan`). This
//!    keeps the bare-baseline path on the cached generated schedule
//!    rather than re-running the planner on every commit.
//! 3. Every LP-returning hook strips the `use_setup_claim_reduction`
//!    flag from the layout it returns, so the prover/verifier dispatch
//!    against `lp.use_setup_claim_reduction` lands on the bare path
//!    even when the base preset's helpers would inject the flag.

use crate::proof_optimized::{
    proof_optimized_envelope, proof_optimized_level_params_with_log_basis,
    proof_optimized_log_basis_at_level, proof_optimized_max_setup_matrix_size,
    proof_optimized_root_level_layout_with_log_basis,
    proof_optimized_root_level_params_for_layout_with_log_basis, proof_optimized_schedule_key,
    proof_optimized_schedule_plan,
};
use crate::CommitmentConfig;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::{
    AjtaiRole, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, CommitmentEnvelope,
    DecompositionParams, LevelParams, ScheduleProvider,
};
use std::marker::PhantomData;

/// Force `Base` onto the bare (CR-off) path.
///
/// Useful in tests that need a CR-off baseline against production
/// presets (`fp128::D128Full`, `fp128::D64OneHot`) which default to
/// CR on per audit B-1.
#[derive(Clone, Copy, Debug, Default)]
pub struct BareCfg<Base>(PhantomData<Base>);

#[inline]
fn strip_setup_claim_reduction(mut lp: LevelParams) -> LevelParams {
    lp.use_setup_claim_reduction = false;
    lp
}

impl<Base> ScheduleProvider for BareCfg<Base>
where
    Base: CommitmentConfig,
{
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Base::schedule_table()
    }

    fn allow_tensor_stage1_schedules() -> bool {
        Base::allow_tensor_stage1_schedules()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("bare/{}", proof_optimized_schedule_key::<Self>(key))
    }

    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        // `Self::use_setup_claim_reduction()` is `false`, so the shared
        // proof-optimized helper falls through to the generated table
        // lookup against `Self::schedule_table()` (delegated to `Base`).
        proof_optimized_schedule_plan::<Self>(key)
    }
}

impl<Base> CommitmentConfig for BareCfg<Base>
where
    Base: CommitmentConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn use_setup_claim_reduction() -> bool {
        false
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        proof_optimized_envelope::<Self>(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        proof_optimized_max_setup_matrix_size::<Self>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )
    }

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        strip_setup_claim_reduction(proof_optimized_level_params_with_log_basis::<Self>(
            inputs, log_basis,
        ))
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        proof_optimized_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
            .map(strip_setup_claim_reduction)
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        proof_optimized_root_level_layout_with_log_basis::<Self>(inputs, log_basis)
            .map(strip_setup_claim_reduction)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        proof_optimized_log_basis_at_level::<Self>(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }
}

#[cfg(feature = "planner")]
impl<Base> akita_planner::PlannerConfig for BareCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
        <Self as ScheduleProvider>::schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        crate::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }

    fn planner_setup_polynomial_size(_max_num_vars: usize) -> usize {
        // BareCfg explicitly turns the cascade off, so the planner cost
        // model should not factor in the `|S|/f` penalty.
        0
    }

    fn planner_setup_shrink_factor() -> usize {
        1
    }

    fn planner_setup_shrink_factor_at_level(_level: usize) -> usize {
        1
    }
}
