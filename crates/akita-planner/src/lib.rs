//! Proof-size parameter planner for the Akita polynomial commitment scheme.
//!
//! Implements a security-aware dynamic programming search over multi-D ring
//! configurations (D=128, 64, 32) to find globally optimal proof schedules
//! with 128-bit SIS security.
//!
//! Five complementary optimizations:
//! 1. Ring dimension reduction across the supported ring ladder
//! 2. Eq-compressed sumcheck (1 fewer element/round)
//! 3. Fully 4-ary GKR tree for Stage 1
//! 4. Column-major block layout (tight z_pre)
//! 5. Serialization header stripping

pub mod baseline;
pub mod proof_size;
pub mod schedule_params;
pub mod search;
pub mod sis_security;

use akita_algebra::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField};
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, LevelParams};

/// Minimal config surface needed by the offline schedule search.
///
/// The planner intentionally depends only on shared data types and this
/// trait. Runtime crates decide which concrete config implements the trait;
/// the verifier/prover role crates do not need to know about planner search.
pub trait PlannerConfig: Clone + Send + Sync + 'static {
    /// Base field used by this planner config.
    type PlannerField: CanonicalField;

    /// Ring degree used at this level.
    const PLANNER_D: usize;

    /// Effective field-element bit width used when sizing proofs.
    fn planner_field_bits() -> u32;

    /// Sparse challenge family used for the given ring dimension.
    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    /// Optional precomputed schedule-table lookup.
    ///
    /// # Errors
    ///
    /// Returns an error when a generated table entry is malformed or
    /// inconsistent with the config.
    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, AkitaError>;

    /// Root fold layout for an explicit basis.
    ///
    /// # Errors
    ///
    /// Returns an error when the root variable split or SIS derivation is
    /// invalid.
    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError>;

    /// Recursive level layout for an explicit basis.
    ///
    /// # Errors
    ///
    /// Returns an error when the recursive layout is invalid.
    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, AkitaError>;

    /// Active root params for a concrete root layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the layout has no SIS-secure parameterization.
    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError>;

    /// Inclusive `(min, max)` log-basis search range at one planner state.
    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32);
}

pub use akita_types::WitnessShape;
pub use baseline::{
    baseline_params_for, run_baseline_planner, BaselineParams, BaselineResult, BASELINE_CASES,
};
pub use schedule_params::{find_optimal_schedule, find_optimal_schedule_with_max};
pub use search::{
    run_universal_planner, DirectWitnessShape, PlannedDirectStep, PlannedFoldStep, PlannedStep,
    PlannerOptions, Schedule,
};
