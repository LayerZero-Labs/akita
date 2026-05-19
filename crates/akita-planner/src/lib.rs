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

use akita_challenges::SparseChallengeConfig;
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

    /// Size of the shared setup polynomial `S` in ring elements at this
    /// config's `D`.
    ///
    /// Phase D-full v2 cascade constraint per book §5.3 lines 627-642
    /// and §5.4 Table 1 (line 762): when a level emits a
    /// setup-claim-reduction payload, the next level's multi-claim
    /// joint open of `(w_L, S)` adds `|S|/f` ring elements to the
    /// next-level witness (where `f` is the tiered shrink factor;
    /// `f = 1` for un-tiered). The planner consults this hook together
    /// with [`Self::planner_setup_shrink_factor`] to compute the
    /// additive cascade penalty `w_fold_L + |S|/f`.
    ///
    /// Returns 0 to disable the cascade penalty (default). Configs that
    /// enable `use_setup_claim_reduction` should override this so the
    /// planner search yields a schedule whose level-L+1 LP fits `S`.
    fn planner_setup_polynomial_size(_max_num_vars: usize) -> usize {
        0
    }

    /// Tiered shrink factor `f` for the shared setup polynomial `S`
    /// per book §5.4. Per the book sweet spot, production configs use
    /// `f = 8` (yielding `k = f² = 64` chunks). The un-tiered case is
    /// `f = 1` (single chunk). Returns 1 by default; configs that
    /// override [`Self::planner_setup_polynomial_size`] should also
    /// override this if they use the tiered design.
    ///
    /// **Uniform-tier shorthand.** Configs that want a different
    /// `f` per recursion level (book §5.8 line 1170, e.g.
    /// `f_{L0} = 8`, `f_{L1} = 4`) should additionally override
    /// [`Self::planner_setup_shrink_factor_at_level`].
    fn planner_setup_shrink_factor() -> usize {
        1
    }

    /// Per-level tiered shrink factor `f` per book §5.4. The book's
    /// headline cascade (§5.8 line 1170) uses `f_{L0} = 8` and
    /// `f_{L1} = 4` to keep the T2 cascade ratio `≲ 1` across two
    /// levels — neither uniform `f = 8` nor uniform `f = 4` matches
    /// the book's prescription, so per-level selection is required to
    /// realise the headline `T1+T2 @ L0+L1` row of Table at line
    /// 1133–1158.
    ///
    /// Default delegates to [`Self::planner_setup_shrink_factor`] for
    /// backward compatibility with uniform-tier configs. Override only
    /// when the planner should select different tiers per level.
    fn planner_setup_shrink_factor_at_level(_level: usize) -> usize {
        Self::planner_setup_shrink_factor()
    }

    /// Relative weight for verifier setup-precompute storage in the planner
    /// objective.
    ///
    /// The reported schedule size remains proof bytes. This weight only affects
    /// DP comparisons by adding
    /// `ceil(setup_storage_bytes * weight / amortization_proofs)` to each
    /// setup-claim-reduction level. The default models Jolt-style amortization:
    /// storage is paid once per setup and spread over many proofs. Set this to
    /// `0` to disable the storage component, or raise it to approximate a
    /// per-deployment policy without changing planner code.
    fn planner_setup_storage_weight() -> usize {
        1
    }

    /// Expected number of proofs sharing one verifier setup precompute.
    ///
    /// The default `1000` is intentionally conservative for production
    /// integrations where setup material is reused many times. Configs with a
    /// one-shot deployment model can lower this value; `0` is treated as `1`.
    fn planner_setup_storage_amortization_proofs() -> usize {
        1000
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
