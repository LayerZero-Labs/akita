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

use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::AkitaError;
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, LevelParams};

/// Minimal config surface needed by the offline schedule search.
///
/// The planner intentionally depends only on shared data types and this
/// trait. Runtime crates decide which concrete config implements the trait;
/// the verifier/prover role crates do not need to know about planner search.
pub trait PlannerConfig: Clone + Send + Sync + 'static {
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

    /// Extra planner weight for prover-heavy stage-1 sumcheck bytes.
    ///
    /// The proof byte count remains the reported schedule size and all SIS
    /// constraints remain hard filters. This weight only breaks proof-size
    /// ties/trade-offs in favour of schedules with cheaper prover sumchecks.
    fn planner_stage1_prover_weight() -> usize {
        0
    }

    /// Stage-1 challenge shapes the planner is allowed to consider per fold
    /// level. Defaults to "use whatever shape
    /// [`Self::planner_root_level_layout_with_log_basis`] (or the recursive
    /// equivalent) returns", i.e. a single choice per level.
    ///
    /// Configs that opt into hybrid per-level search return both `Flat` and
    /// `Tensor` (typically with a config-wide `allow_hybrid_stage1_shapes`
    /// flag, security-audited). The planner then iterates over both shapes
    /// at each level and the DP picks the best per-level mix.
    ///
    /// SIS safety: switching shape after the base helper has derived `n_a`,
    /// `n_b`, `n_d` is only safe Tensor → Flat (mass shrinks → derived rank
    /// is over-secured). Flat → Tensor is unsafe and must not be returned
    /// by configs whose base helper produces Flat layouts.
    fn planner_stage1_shapes_to_search() -> Vec<Stage1ChallengeShape> {
        Vec::new()
    }
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
