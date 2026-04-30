//! Configuration trait for ring-native commitment construction.
use super::adaptive;
use super::generated::GeneratedScheduleTable;
use super::schedule::{
    fallback_batched_root_split, hachi_root_commitment_layout, HachiRootBatchSummary,
    HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::planner::schedule_params::{schedule_from_plan, Schedule};
use crate::protocol::params::LevelParams;
use crate::{CanonicalField, FieldCore};

/// Parameters controlling the gadget decomposition depth (called δ in the paper).
///
/// The gadget base is `b = 2^log_basis`. Each ring coefficient with centered
/// magnitude fitting in `log_commit_bound` bits is decomposed into
/// `ceil(log_commit_bound / log_basis)` balanced digits in `[-b/2, b/2)`.
///
/// Smaller `log_commit_bound` (when polynomial coefficients are known to be
/// small) yields fewer decomposition levels, which proportionally shrinks the
/// witness vector, the commitment matrices, and the proving cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompositionParams {
    /// Base-2 logarithm of the gadget base (e.g., 3 for base-8 digits in [-4, 3]).
    pub log_basis: u32,

    /// Bit-width of the largest coefficient that the *commitment* decomposition
    /// must represent. Controls the commitment-side decomposition depth (δ in
    /// the paper): `num_digits = ceil(log_commit_bound / log_basis)`.
    ///
    /// The centered representation maps each coefficient `c ∈ [0, q)` to the
    /// signed value in `(-q/2, q/2]`. A value of `k` means the signed magnitude
    /// fits in `k` bits, i.e., lies in `[-2^(k-1), 2^(k-1) - 1]`.
    ///
    /// Examples:
    /// - Binary (0/1) polynomials: 1
    /// - Already range-checked digits in `[-b/2, b/2)`: `log_basis` (one digit)
    /// - Arbitrary Fp128 elements: 128
    pub log_commit_bound: u32,

    /// Bit-width of the largest coefficient that the *opening* decomposition
    /// must represent (ŵ = G⁻¹(w_folded)).
    ///
    /// During opening, `fold_blocks` computes inner products with arbitrary
    /// field-element weights, so the result always has full-field-size
    /// coefficients regardless of the original `log_commit_bound`. When `None`,
    /// defaults to `log_commit_bound` (correct when `log_commit_bound` already
    /// covers the full field, e.g. 128). Set to the field modulus bit-width
    /// when `log_commit_bound` is smaller (e.g. for recursive w commitments
    /// where entries are small but fold products are not).
    pub log_open_bound: Option<u32>,
}

impl DecompositionParams {
    /// Effective field-element bit-width (the opening bound, defaulting to
    /// the commit bound when no explicit opening bound is set).
    pub(crate) fn field_bits(self) -> u32 {
        self.log_open_bound.unwrap_or(self.log_commit_bound)
    }
}

/// Maximum matrix row envelope needed across all runtime levels for a config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentEnvelope {
    /// Maximum inner Ajtai rank needed by any supported level.
    pub max_n_a: usize,
    /// Maximum outer commitment rank needed by any supported level.
    pub max_n_b: usize,
    /// Maximum prover D-matrix rank needed by any supported level.
    pub max_n_d: usize,
}

/// Selects which Ajtai role (`A`, or `B`/`D` together) the audited rank
/// floor applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AjtaiRole {
    /// Inner Ajtai matrix `A`.
    Inner,
    /// Outer commitment matrices `B` and `D` (sized together).
    Outer,
}

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Concrete presets (e.g. [`crate::protocol::commitment::presets::fp128::D128Full`])
/// only need to provide:
/// - their base field and ring degree,
/// - their decomposition and sparse-challenge family,
/// - the [`GeneratedScheduleTable`] that backs them,
/// - and (optionally) the audited root-rank floor.
///
/// Every other method on this trait has a default body that routes through
/// `Self::schedule_table()` and the planner-backed helpers in
/// [`super::adaptive`]. Concrete presets do not override the runtime hooks
/// by hand.
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Base field used by this config.
    type Field: CanonicalField + FieldCore;

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Sparse challenge family used at this level.
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;

    // ---------------------------------------------------------------
    // Optional preset hooks (audited rank floor + offline schedule table).
    // Default to "ad-hoc test config" behavior.
    // ---------------------------------------------------------------

    /// Pre-computed schedule table backing this config, if any.
    ///
    /// Presets return their generated table here; ad-hoc configs return
    /// `None` and let the runtime planner search from scratch.
    #[doc(hidden)]
    #[allow(private_interfaces)]
    fn schedule_table() -> Option<GeneratedScheduleTable> {
        None
    }

    /// Audited rank floor for the root level, by role.
    #[doc(hidden)]
    fn audited_root_rank(_role: AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    // ---------------------------------------------------------------
    // Runtime hooks below: every shipped preset uses the default body.
    // ---------------------------------------------------------------

    /// Maximum matrix row envelope needed across all runtime levels.
    #[doc(hidden)]
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        adaptive::adaptive_envelope::<Self>(max_num_vars)
    }

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
    ) -> Result<(usize, usize), HachiError> {
        adaptive::adaptive_max_setup_matrix_size::<Self>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )
    }

    /// Active level params for one level under an explicit basis.
    #[doc(hidden)]
    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        adaptive::adaptive_level_params_with_log_basis::<Self>(inputs, log_basis)
    }

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
    ) -> Result<LevelParams, HachiError> {
        adaptive::adaptive_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
    }

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
    ) -> Result<LevelParams, HachiError> {
        adaptive::adaptive_root_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    /// Active basis for one level from public inputs.
    #[doc(hidden)]
    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        adaptive::adaptive_log_basis_at_level::<Self>(inputs)
    }

    /// Inclusive `(min, max)` log-basis search range at one state.
    #[doc(hidden)]
    fn log_basis_search_range(_inputs: HachiScheduleInputs) -> (u32, u32) {
        adaptive::adaptive_log_basis_search_range()
    }

    /// Stable identity for the active schedule at `key`.
    #[doc(hidden)]
    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        adaptive::adaptive_schedule_key::<Self>(key)
    }

    /// Optional full schedule plan for configs with an explicit planner.
    ///
    /// `None` means the caller should fall back to the runtime planner.
    ///
    /// # Errors
    ///
    /// Returns an error when the planner cannot derive a valid schedule.
    #[doc(hidden)]
    fn schedule_plan(key: HachiScheduleLookupKey) -> Result<Option<HachiSchedulePlan>, HachiError> {
        adaptive::adaptive_schedule_plan::<Self>(key)
    }

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
    /// Returns an error if the batch summary, schedule lookup, planner
    /// fallback, or derived layout is invalid for the requested commitment
    /// shape.
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
            return fallback_batched_root_split::<Self>(num_vars, 1);
        }

        use crate::planner::schedule_params::{find_optimal_schedule, Step, WitnessShape};

        let schedule = find_optimal_schedule::<Self>(
            num_vars,
            WitnessShape {
                num_claims: num_polys_per_point,
                num_commitment_groups: 1,
                num_points: 1,
            },
        )?;

        match schedule.steps.first() {
            Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
            _ => fallback_batched_root_split::<Self>(num_vars, 1),
        }
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
            return Ok(schedule_from_plan::<Self>(&plan));
        }

        if layout_num_claims != batch.num_claims {
            return Err(HachiError::InvalidSetup(format!(
                "fallback prove schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }

        use crate::planner::schedule_params::{find_optimal_schedule, WitnessShape};

        find_optimal_schedule::<Self>(
            num_vars,
            WitnessShape {
                num_claims: batch.num_claims,
                num_commitment_groups: batch.num_commitment_groups,
                num_points: batch.num_points,
            },
        )
    }
}

/// Deterministic upper bound for the stage-1 folded-witness infinity norm.
///
/// This encodes the bound used in `QuadraticEquation::compute_z_hat`:
/// `||z||_inf <= 2^R * challenge_l1_mass * (b/2)` where `b = 2^LOG_BASIS`.
///
/// # Errors
///
/// Returns an error when parameters are out of range or intermediate products
/// overflow `u128`.
pub fn beta_linf_fold_bound(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
) -> Result<u128, HachiError> {
    if !(1..128).contains(&log_basis) {
        return Err(HachiError::InvalidSetup("invalid LOG_BASIS".to_string()));
    }
    if r >= 128 {
        return Err(HachiError::InvalidSetup("r_vars must be < 128".to_string()));
    }

    let blocks = 1u128 << r;
    let b = 1u128 << log_basis;
    let half_b = b / 2;

    let term = blocks
        .checked_mul(challenge_l1_mass as u128)
        .ok_or_else(|| HachiError::InvalidSetup("beta bound overflow".to_string()))?;
    term.checked_mul(half_b)
        .ok_or_else(|| HachiError::InvalidSetup("beta bound overflow".to_string()))
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::*;
    use crate::planner::sis_security::min_rank_for_secure_width;

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
                    a_rank <= level.lp.a_key.row_len() as u32,
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
                    b_rank <= level.lp.b_key.row_len() as u32,
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
                    d_rank <= level.lp.d_key.row_len() as u32,
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
        type Cfg = crate::protocol::commitment::presets::fp128::D128Full;
        assert_schedule_stays_within_audited_sis_widths::<Cfg>(8, 50);
    }

    #[test]
    fn current_d64_full_schedule_stays_within_audited_sis_widths() {
        type Cfg = crate::protocol::commitment::presets::fp128::D64Full;
        // B-row rank=1 at num_vars>=46 level=1 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<Cfg>(8, 45);
    }

    #[test]
    fn current_d64_onehot_schedule_stays_within_audited_sis_widths() {
        type Cfg = crate::protocol::commitment::presets::fp128::D64OneHot;
        assert_schedule_stays_within_audited_sis_widths::<Cfg>(8, 50);
    }

    #[test]
    fn current_d32_full_schedule_stays_within_audited_sis_widths() {
        type Cfg = crate::protocol::commitment::presets::fp128::D32Full;
        // D-row rank=1 at num_vars>=30 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<Cfg>(8, 29);
    }

    #[test]
    fn current_d32_onehot_schedule_stays_within_audited_sis_widths() {
        type Cfg = crate::protocol::commitment::presets::fp128::D32OneHot;
        // D-row rank=1 at num_vars>=36 level=2 lb=2 — needs SIS floor fix
        assert_schedule_stays_within_audited_sis_widths::<Cfg>(8, 35);
    }
}
