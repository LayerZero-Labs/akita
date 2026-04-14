//! Proof-size parameter planner for the Hachi polynomial commitment scheme.
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
pub mod digit_math;
pub mod proof_size;
pub mod schedule_params;
pub mod search;
pub mod sis_security;

pub use baseline::{
    baseline_params_for, run_baseline_planner, BaselineParams, BaselineResult, BASELINE_CASES,
};
pub use search::{
    run_universal_planner, DirectWitnessShape, PlannedDirectStep, PlannedFoldStep, PlannedStep,
    PlannerOptions, Schedule,
};
