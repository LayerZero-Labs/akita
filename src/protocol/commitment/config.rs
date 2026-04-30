//! Configuration trait for ring-native commitment construction.
use super::schedule::{
    fallback_batched_root_split, hachi_root_commitment_layout, HachiRootBatchSummary,
    HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
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

/// Compute the gadget decomposition depth (δ in the paper) from a
/// coefficient bit-width bound.
///
/// Returns `ceil(log_bound / log_basis)`, with an extra level when the
/// balanced-digit range would not cover the full bound.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_num_digits(log_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if log_bound == 0 {
        return 1;
    }
    let mut levels = (log_bound as usize).div_ceil(log_basis as usize);

    // When levels * log_basis > log_bound (i.e., not exactly aligned), the
    // balanced digit range (b/2-1) * (b^levels - 1)/(b-1) always exceeds
    // 2^(log_bound-1) for b >= 4 (log_basis >= 2). Only check when aligned.
    let total_bits = (levels as u32).saturating_mul(log_basis);
    if total_bits <= log_bound {
        let b: u128 = 1u128 << log_basis;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul(b_pow.saturating_sub(1) / b_minus_1);
        let required = if log_bound > 128 {
            u128::MAX / 2
        } else if log_bound == 0 {
            0
        } else {
            (1u128 << (log_bound - 1)).saturating_sub(1)
        };
        if max_positive < required {
            levels += 1;
        }
    }
    levels.max(1)
}

/// Compute the decomposition depth for full-field values using asymmetric
/// centering: `ceil(field_bits / log_basis)` with no +1 correction.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_num_digits_full_field(field_bits: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if field_bits == 0 {
        return 1;
    }
    (field_bits as usize).div_ceil(log_basis as usize).max(1)
}

/// Choose the correct digit-count function for a given bit-width bound.
///
/// Full-field bounds (>=128 bits) use asymmetric centering (no +1 correction).
/// Smaller bounds use symmetric centering (possible +1 correction).
pub fn num_digits_for_bound(log_bound: u32, log_basis: u32) -> usize {
    if log_bound >= 128 {
        compute_num_digits_full_field(log_bound, log_basis)
    } else {
        compute_num_digits(log_bound, log_basis)
    }
}

/// Compute the decomposition depth for the folded witness `z_pre`
/// (τ in the paper).
///
/// The folded witness satisfies `||z_pre||_inf <= β` where
/// `β = 2^r_vars * challenge_l1_mass * 2^(log_basis - 1)`.
/// Returns enough gadget levels to represent values up to `β`.
pub fn compute_num_digits_fold(r_vars: usize, challenge_l1_mass: usize, log_basis: u32) -> usize {
    compute_num_digits_fold_with_claims(r_vars, challenge_l1_mass, log_basis, 1)
}

fn recursive_tight_z_pre_rows(num_ring: usize, r_vars: usize) -> u64 {
    debug_assert!(
        r_vars < 53,
        "recursive_tight_z_pre_rows expects r_vars < 53, got {r_vars}"
    );
    (num_ring as u64).div_ceil(1u64 << r_vars)
}

/// Find the `(m_vars, r_vars)` split that minimizes the level-0
/// witness-to-polynomial ratio for a given config.
///
/// The witness ring element count is dominated by:
/// ```text
/// w ≈ 2^r · (δ_open + N_A · δ_commit) + 2^m · δ_commit · δ_fold(r)
/// ```
/// Multiplying the ratio by `2^(m+r)` (constant for fixed `reduced_vars`)
/// gives an equivalent integer cost:
/// ```text
/// C1 · 2^r  +  δ_commit · δ_fold(r) · 2^m
/// ```
/// where `C1 = δ_open + N_A · δ_commit`. This function searches all valid
/// `(m, r)` pairs for the minimum using pure integer arithmetic (no
/// floating-point), so it is safe to run inside a zkVM guest.
///
/// For the full-field config (`δ_commit = 43`), z_pre dominates and the
/// result is near-balanced (`m ≈ r`). For narrow configs (`δ_commit = 1`),
/// the w_hat/t_hat term matters more and the result skews to `m ≈ r + 4`.
/// Find the cheapest `(m, r)` split for a given `reduced_vars`.
///
/// When `num_ring > 0` (recursive levels), the z_pre cost uses the tight
/// block length `ceil(num_ring / 2^r)` instead of `2^m`.  Pass `0` for
/// root-level splits where the polynomial fills the full `2^(m+r)` domain.
pub(super) fn optimal_m_r_split_with_params(
    params: &LevelParams,
    decomp: DecompositionParams,
    reduced_vars: usize,
    num_ring: usize,
) -> (usize, usize) {
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let delta_open = num_digits_for_bound(open_bound, decomp.log_basis) as u64;
    let delta_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis) as u64;
    let c1 = delta_open + params.a_key.row_len() as u64 * delta_open;

    let mut best_r = reduced_vars / 2;
    let mut best_cost = u64::MAX;

    for r in 1..reduced_vars {
        let m = reduced_vars - r;
        let delta_fold =
            compute_num_digits_fold(r, params.challenge_l1_mass(), decomp.log_basis) as u64;
        let m_eff = if num_ring > 0 {
            recursive_tight_z_pre_rows(num_ring, r)
        } else {
            1u64 << m
        };
        let cost = c1 * (1u64 << r) + delta_commit * delta_fold * m_eff;
        if cost < best_cost {
            best_cost = cost;
            best_r = r;
        }
    }

    (reduced_vars - best_r, best_r)
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

/// Commitment-config trait for the ring-native commitment core (§4.1–§4.2).
///
/// Concrete presets (e.g. [`crate::protocol::commitment::presets::fp128::D128Full`])
/// implement this trait via the `impl_fp128_preset!` macro, which routes
/// every layout / schedule / log-basis decision through the planner-backed
/// helpers in [`super::adaptive`]. The trait surface is therefore minimal: a
/// concrete config only needs to declare its field, ring degree,
/// decomposition, sparse challenge family, and (optionally) which generated
/// schedule table backs it.
///
/// All other methods on this trait are runtime-protocol hooks that the
/// commit/prove/verify pipeline calls; they should not be overridden by
/// hand. Their default bodies fall back to the simple "no offline plan"
/// behavior used by ad-hoc test configs, but every shipped preset replaces
/// them with planner-backed implementations through the macro.
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
    // The methods below are runtime hooks. Concrete presets override
    // them via `impl_fp128_preset!`; no preset implements them by hand.
    // ---------------------------------------------------------------

    /// Maximum matrix row envelope needed across all runtime levels.
    #[doc(hidden)]
    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        let max_rank = crate::planner::sis_security::MAX_RANK as usize;
        CommitmentEnvelope {
            max_n_a: max_rank,
            max_n_b: max_rank,
            max_n_d: max_rank,
        }
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
        let max_rows = crate::planner::sis_security::MAX_RANK as usize;
        let alpha = Self::D.trailing_zeros() as usize;
        let outer_vars = max_num_vars.saturating_sub(alpha);
        let max_stride = 1usize
            .checked_shl(outer_vars as u32)
            .and_then(|x| x.checked_mul(128))
            .and_then(|x| x.checked_mul(max_rows))
            .and_then(|x| x.checked_mul(max_num_batched_polys))
            .and_then(|x| x.checked_mul(max_num_points))
            .ok_or_else(|| {
                HachiError::InvalidSetup("max_setup_matrix_size overflow".to_string())
            })?;
        Ok((max_rows, max_stride))
    }

    /// Active level params for one level under an explicit basis.
    #[doc(hidden)]
    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let d = Self::D;
        let stage1_config = Self::stage1_challenge_config(d);
        let envelope = Self::envelope(inputs.max_num_vars);
        LevelParams::params_only(
            d,
            log_basis,
            envelope.max_n_a,
            envelope.max_n_b,
            envelope.max_n_d,
            stage1_config,
        )
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
        let params = Self::level_params_with_log_basis(inputs, lp.log_basis);
        Ok(params.with_layout(lp))
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
        let params = Self::level_params_with_log_basis(inputs, log_basis);
        super::adaptive::derived_root_commitment_layout_from_params::<Self>(inputs, &params, false)
    }

    /// Active basis for one level from public inputs.
    #[doc(hidden)]
    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        let _ = inputs;
        Self::decomposition().log_basis
    }

    /// Inclusive `(min, max)` log-basis search range at one state.
    #[doc(hidden)]
    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        let basis = Self::log_basis_at_level(inputs);
        (basis, basis)
    }

    /// Stable identity for the active schedule at `key`.
    #[doc(hidden)]
    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        format!(
            "static_v1_b{}_nv{}_poly{}_layout{}_claims{}_groups{}_points{}",
            Self::decomposition().log_basis,
            key.max_num_vars,
            key.num_vars,
            key.layout_num_claims,
            key.batch.num_claims,
            key.batch.num_commitment_groups,
            key.batch.num_points
        )
    }

    /// Optional full schedule plan for configs with an explicit planner.
    ///
    /// `None` means the caller should fall back to the runtime planner.
    ///
    /// # Errors
    ///
    /// Returns an error when the planner cannot derive a valid schedule.
    #[doc(hidden)]
    fn schedule_plan(
        _key: HachiScheduleLookupKey,
    ) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Ok(None)
    }

    /// Choose the runtime commitment layout for `max_num_vars`.
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
        hachi_root_commitment_layout::<Self>(max_num_vars)
    }

    /// Choose the root parameters consumed by the commitment path.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch summary, schedule lookup, planner
    /// fallback, or derived layout is invalid for the requested commitment
    /// shape.
    fn get_params_for_commitment<const D: usize>(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<LevelParams, HachiError> {
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
            let split = fallback_batched_root_split::<Self, D>(num_vars, 1)?;
            return Ok(split.params);
        }

        use crate::planner::schedule_params::{find_optimal_schedule, Step, WitnessShape};

        let schedule = find_optimal_schedule::<Self, D>(
            num_vars,
            WitnessShape {
                num_claims: num_polys_per_point,
                num_commitment_groups: 1,
                num_points: 1,
            },
        )?;

        match schedule.steps.first() {
            Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
            _ => {
                let split = fallback_batched_root_split::<Self, D>(num_vars, 1)?;
                Ok(split.params)
            }
        }
    }

    /// Choose the root parameters consumed by the prove/verify root path.
    ///
    /// # Errors
    ///
    /// Returns an error if the root layout, batched layout scaling, next
    /// witness sizing, or next-level basis selection is invalid.
    fn get_params_for_prove<const D: usize>(
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

        find_optimal_schedule::<Self, D>(
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

// Internal SIS-derived recursive/root params and the blanket
// `GeneratedAdaptivePolicy` impl have moved to `super::adaptive`. Each concrete
// preset implements `CommitmentConfig` directly via the `impl_fp128_preset!`
// macro defined in `super::presets`.

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
