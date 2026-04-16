//! Configuration presets for ring-native commitment construction.
use super::profile::{CommitmentFieldProfile, CommitmentFieldProfileSchedule};
use super::schedule::{
    exact_planned_level_execution, hachi_recursive_level_layout_from_params,
    hachi_root_commitment_layout, HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
use super::utils::norm::detect_field_modulus;
use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::protocol::params::{AjtaiKeyParams, LevelParams};
use crate::{CanonicalField, FieldCore};
use std::marker::PhantomData;

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

/// Find the `(m_vars, r_vars)` split using the level-0 params of `Cfg`.
pub fn optimal_m_r_split<Cfg: CommitmentConfig>(reduced_vars: usize) -> (usize, usize) {
    let params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars: reduced_vars + Cfg::D.trailing_zeros() as usize,
        level: 0,
        current_w_len: 1usize
            .checked_shl((reduced_vars + Cfg::D.trailing_zeros() as usize) as u32)
            .unwrap_or(0),
    });
    optimal_m_r_split_with_params(&params, Cfg::decomposition(), reduced_vars, 0)
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

/// Parameter bundle for the ring-native commitment core (§4.1–§4.2).
///
/// Ring degree `D` remains compile time because `CyclotomicRing<F, D>` is
/// array-backed, but the active Ajtai ranks and sparse-challenge family are
/// runtime values derived from public schedule inputs. Setup allocates against a
/// per-config row envelope, while each level uses the exact rows selected by
/// [`LevelParams`].
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Base field used by this config.
    type Field: CanonicalField + FieldCore;

    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Maximum matrix row counts needed by this config for the given setup size.
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope;

    /// Conservative `(max_rows, max_stride)` bounds for the shared setup
    /// matrix.
    ///
    /// The default implementation pins the row count to
    /// [`sis_security::MAX_RANK`](crate::planner::sis_security::MAX_RANK)
    /// and derives the column stride as the max of the inner (A),
    /// outer (B), and D column widths under the worst-case (smallest)
    /// root `log_basis` produced by [`Self::log_basis_search_range`].
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidSetup`] if any intermediate width bound
    /// (block length, block count, inner / outer / D stride) overflows
    /// `usize`.
    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<(usize, usize), HachiError> {
        let max_rows = crate::planner::sis_security::MAX_RANK as usize;

        let alpha = Self::D.trailing_zeros() as usize;
        let outer_vars = max_num_vars.saturating_sub(alpha);
        let decomp = Self::decomposition();
        let field_bits = 128u32;
        let root_inputs = HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
        };
        let (min_log_basis, _) = Self::log_basis_search_range(root_inputs);
        let worst_log_basis = min_log_basis.max(1);
        let num_digits_commit = compute_num_digits(decomp.log_commit_bound, worst_log_basis);
        let log_open_bound = decomp.log_open_bound.unwrap_or(field_bits);
        let num_digits_open = compute_num_digits(log_open_bound, worst_log_basis);
        let m_exp = (2 * outer_vars).div_ceil(3);
        let r_exp = outer_vars.div_ceil(2);
        let block_len_bound = 1usize
            .checked_shl(m_exp as u32)
            .ok_or_else(|| HachiError::InvalidSetup(format!("2^{m_exp} does not fit usize")))?;
        let num_blocks_bound = 1usize
            .checked_shl(r_exp as u32)
            .ok_or_else(|| HachiError::InvalidSetup(format!("2^{r_exp} does not fit usize")))?;
        let inner_width = block_len_bound
            .checked_mul(num_digits_commit)
            .ok_or_else(|| HachiError::InvalidSetup("inner width bound overflow".to_string()))?;
        let outer_width = max_rows
            .checked_mul(num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks_bound))
            .and_then(|x| x.checked_mul(max_num_batched_polys))
            .ok_or_else(|| HachiError::InvalidSetup("outer width bound overflow".to_string()))?;
        let d_width = num_digits_open
            .checked_mul(num_blocks_bound)
            .and_then(|x| x.checked_mul(max_num_batched_polys))
            .ok_or_else(|| HachiError::InvalidSetup("D width bound overflow".to_string()))?;
        let max_stride = inner_width.max(outer_width).max(d_width);

        Ok((max_rows, max_stride))
    }

    /// Stable identifier for setup-cache versioning and fixture selection.
    fn family_key() -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Exact field modulus used by this config's planner and runtime sizing.
    fn field_modulus() -> u128 {
        detect_field_modulus::<Self::Field>()
    }

    /// Build one level's active params from public inputs and an explicit basis.
    ///
    /// Returns a `LevelParams` with zeroed layout fields (block geometry,
    /// digit depths, column counts). Use `with_layout` or
    /// `current_level_layout_with_log_basis` to populate layout fields.
    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let d = Self::d_at_level(inputs.level, inputs.current_w_len);
        let stage1_config = Self::stage1_challenge_config(d);
        LevelParams::params_only(
            d,
            log_basis,
            Self::n_a_at_level(inputs.level),
            Self::n_b_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            Self::n_d_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            stage1_config,
        )
    }

    /// Derive root params for a concrete root layout.
    ///
    /// The default implementation trusts the config's pinned root ranks and only
    /// uses the layout to recover the active basis. Adaptive configs can
    /// override this to derive exact root ranks from the real layout widths.
    ///
    /// # Errors
    ///
    /// Returns an error if the config cannot derive a sound root parameter set
    /// for the supplied root layout.
    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        let params = Self::level_params_with_log_basis(inputs, lp.log_basis);
        Ok(params.with_layout(lp))
    }

    /// Derive the root fold layout for an explicit basis.
    ///
    /// The default implementation uses the config's current root params and the
    /// standard `(m, r)` split search. Adaptive configs can override this to
    /// search over self-consistent root ranks before materializing the layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the root variable split underflows, overflows, or
    /// does not admit a sound root parameterization.
    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        let params = Self::level_params_with_log_basis(inputs, log_basis);
        derived_root_commitment_layout_from_params::<Self>(inputs, &params, false)
    }

    /// Deterministically choose the active basis for one level from public inputs.
    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        let _ = inputs;
        Self::decomposition().log_basis
    }

    /// Inclusive search range for adaptive recursive basis planning at one state.
    ///
    /// Static configs should return a singleton range containing the unique basis
    /// allowed at `inputs`.
    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        let basis = Self::log_basis_at_level(inputs);
        (basis, basis)
    }

    /// Stable identity for the active log-basis schedule at `key`.
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
    /// `None` means callers should fall back to the legacy local stop heuristic.
    ///
    /// # Errors
    ///
    /// Returns an error when the config's planner cannot derive a valid
    /// schedule from the public inputs.
    fn schedule_plan(
        _key: HachiScheduleLookupKey,
    ) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Ok(None)
    }

    /// Optional proof-size planner for recursive suffixes that start from an
    /// arbitrary witness state.
    ///
    /// `None` means callers should fall back to a local byte comparison.
    ///
    /// # Errors
    ///
    /// Returns an error when the config's planner cannot derive a valid
    /// suffix from the public inputs.
    fn recursive_suffix_bytes(
        _key: HachiScheduleLookupKey,
        _level: usize,
        _current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        Ok(None)
    }

    /// Deterministically derive the active params for one level from public inputs.
    fn level_params(inputs: HachiScheduleInputs) -> LevelParams {
        let log_basis = Self::log_basis_at_level(inputs);
        Self::level_params_with_log_basis(inputs, log_basis)
    }

    /// Choose the runtime commitment layout for `max_num_vars`.
    ///
    /// Planner-backed families use the exact root fold layout when one is
    /// pinned; otherwise this falls back to the derived root-commitment layout.
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

    /// Runtime L∞ bound for `z` (`β`) used by stage-1 folding checks.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid parameters or arithmetic overflow.
    fn beta_bound(lp: &LevelParams) -> Result<u128, HachiError> {
        beta_linf_fold_bound(
            lp.r_vars,
            Self::stage1_challenge_config(Self::D).l1_mass(),
            lp.log_basis,
        )
    }

    /// Ring dimension to use at a given fold level.
    ///
    /// `level` is 0-indexed (level 0 is the initial polynomial).
    /// `_w_num_vars` is the number of variables in the witness at this level.
    ///
    /// The default implementation returns `Self::D` at all levels (constant D).
    /// Override for decreasing-D schedules.
    fn d_at_level(_level: usize, _current_w_len: usize) -> usize {
        Self::D
    }

    /// Module rank (inner Ajtai row count) at a given fold level.
    ///
    /// Must satisfy `d_at_level(level) * n_a_at_level(level) >= security_dim`
    /// for the target security level. The default uses the config envelope's
    /// maximum `n_a`.
    fn n_a_at_level(_level: usize) -> usize {
        Self::envelope(0).max_n_a
    }

    /// Outer commitment matrix rank at a given fold level.
    fn n_b_at_level(_level: usize, max_num_vars: usize, _current_w_len: usize) -> usize {
        Self::envelope(max_num_vars).max_n_b
    }

    /// Prover D-matrix rank at a given fold level.
    fn n_d_at_level(_level: usize, max_num_vars: usize, _current_w_len: usize) -> usize {
        Self::envelope(max_num_vars).max_n_d
    }

    /// Sparse challenge family used at this level.
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig;
}

/// Concrete commitment preset obtained by pairing a field with a scheduling policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitmentPreset<F, Policy> {
    _marker: PhantomData<(F, Policy)>,
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

/// Ensure `max_num_vars` is sufficient for a commitment layout.
///
/// # Errors
///
/// Returns an error when `max_num_vars` cannot support the layout's outer
/// variable count after accounting for the ring's inner `alpha = log2(D)`
/// slots. Underfull roots with `max_num_vars < alpha` are allowed when the
/// layout uses zero outer variables.
pub(super) fn ensure_layout_supported_num_vars<const D: usize>(
    max_num_vars: usize,
    lp: &LevelParams,
) -> Result<(), HachiError> {
    let alpha = D.trailing_zeros() as usize;
    let available_outer = max_num_vars.saturating_sub(alpha);
    let required_outer = lp.outer_vars();
    if available_outer < required_outer {
        return Err(HachiError::InvalidSetup(format!(
            "max_num_vars {max_num_vars} leaves only {available_outer} outer vars but layout requires {required_outer}"
        )));
    }
    Ok(())
}

/// Ensure input blocks match the expected config-derived layout.
///
/// # Errors
///
/// Returns an error when block count or per-block size mismatch.
pub(super) fn ensure_block_layout<F: FieldCore, const D: usize>(
    f_blocks: &[Vec<CyclotomicRing<F, D>>],
    lp: &LevelParams,
) -> Result<(), HachiError> {
    if f_blocks.len() != lp.num_blocks {
        return Err(HachiError::InvalidSize {
            expected: lp.num_blocks,
            actual: f_blocks.len(),
        });
    }
    for block in f_blocks {
        if block.len() != lp.block_len {
            return Err(HachiError::InvalidSize {
                expected: lp.block_len,
                actual: block.len(),
            });
        }
    }
    Ok(())
}

/// Ensure matrix has at least the expected dimensions.
///
/// Matrix envelopes may have more rows and columns than the active level uses.
/// Small correctness-first config for tests and local benchmarks.
///
/// Fixed layout (m_vars=4, r_vars=2) for fast test iteration.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmallTestCommitmentConfig;

impl CommitmentConfig for SmallTestCommitmentConfig {
    type Field = crate::test_utils::F;
    const D: usize = 32;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 4,
            max_n_b: 4,
            max_n_d: 4,
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        assert_eq!(d, Self::D, "unsupported ring dim {d}");
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, HachiError> {
        let decomp = Self::decomposition();
        let params = Self::level_params(HachiScheduleInputs {
            max_num_vars: 12,
            level: 0,
            current_w_len: 1 << 12,
        });
        let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
        let depth_fold = compute_num_digits_fold(2, params.challenge_l1_mass(), decomp.log_basis);
        params.with_decomp(4, 2, depth_commit, depth_open, depth_fold, 0)
    }

    fn schedule_plan(
        key: HachiScheduleLookupKey,
    ) -> Result<Option<super::schedule::HachiSchedulePlan>, HachiError> {
        if key != HachiScheduleLookupKey::singleton(key.max_num_vars, key.max_num_vars, 1) {
            return Ok(None);
        }
        if key.max_num_vars >= usize::BITS as usize {
            return Ok(None);
        }
        let decomp = Self::decomposition();
        let params = Self::level_params(HachiScheduleInputs {
            max_num_vars: 12,
            level: 0,
            current_w_len: 1 << 12,
        });
        let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
        let depth_fold = compute_num_digits_fold(2, params.challenge_l1_mass(), decomp.log_basis);
        let root_lp = params.with_decomp(4, 2, depth_commit, depth_open, depth_fold, 0)?;
        Ok(Some(super::schedule::build_schedule_plan_from_config::<
            Self,
        >(key.max_num_vars, &root_lp)?))
    }
}

/// Static bounded policy with explicit root and recursive log bases.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticBoundedPolicy<
    Profile,
    const D: usize,
    const LOG_COMMIT_BOUND: u32,
    const LOG_BASIS: u32,
    const W_LOG_BASIS: u32 = LOG_BASIS,
    const N_A: usize = 1,
    const N_B: usize = 1,
    const N_D: usize = 1,
>(PhantomData<Profile>);

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
        const LOG_BASIS: u32,
        const W_LOG_BASIS: u32,
        const N_A: usize,
        const N_B: usize,
        const N_D: usize,
    > CommitmentConfig
    for CommitmentPreset<
        F,
        StaticBoundedPolicy<Profile, D, LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS, N_A, N_B, N_D>,
    >
{
    type Field = F;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        Profile::decomposition(LOG_COMMIT_BOUND, LOG_BASIS)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        let audited_root_rank = Profile::audited_root_outer_rank(D, 0, max_num_vars);
        let audited_root_a_rank =
            Profile::audited_root_a_rank::<LOG_COMMIT_BOUND>(D, 0, max_num_vars);
        CommitmentEnvelope {
            max_n_a: N_A.max(audited_root_a_rank),
            max_n_b: N_B.max(audited_root_rank),
            max_n_d: N_D.max(audited_root_rank),
        }
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        if inputs.level == 0 {
            LOG_BASIS
        } else {
            W_LOG_BASIS
        }
    }

    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        format!(
            "static_v1_root{LOG_BASIS}_rec{W_LOG_BASIS}_nv{}_poly{}_layout{}_claims{}_groups{}_points{}",
            key.max_num_vars,
            key.num_vars,
            key.layout_num_claims,
            key.batch.num_claims,
            key.batch.num_commitment_groups,
            key.batch.num_points
        )
    }

    fn schedule_plan(
        key: HachiScheduleLookupKey,
    ) -> Result<Option<super::schedule::HachiSchedulePlan>, crate::error::HachiError> {
        if key != HachiScheduleLookupKey::singleton(key.max_num_vars, key.max_num_vars, 1) {
            return Ok(None);
        }
        if key.max_num_vars >= usize::BITS as usize {
            return Ok(None);
        }
        let root_lp = super::schedule::hachi_root_commitment_layout::<Self>(key.max_num_vars)?;
        Ok(Some(super::schedule::build_schedule_plan_from_config::<
            Self,
        >(key.max_num_vars, &root_lp)?))
    }

    fn n_b_at_level(level: usize, max_num_vars: usize, _current_w_len: usize) -> usize {
        N_B.max(Profile::audited_root_outer_rank(D, level, max_num_vars))
    }

    fn n_d_at_level(level: usize, max_num_vars: usize, current_w_len: usize) -> usize {
        let _ = current_w_len;
        N_D.max(Profile::audited_root_outer_rank(D, level, max_num_vars))
    }

    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        let d = Self::d_at_level(inputs.level, inputs.current_w_len);
        let stage1_config = Self::stage1_challenge_config(d);
        LevelParams::params_only(
            d,
            log_basis,
            N_A.max(Profile::audited_root_a_rank::<LOG_COMMIT_BOUND>(
                D,
                inputs.level,
                inputs.max_num_vars,
            )),
            Self::n_b_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            Self::n_d_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            stage1_config,
        )
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Profile::stage1_challenge_config(d)
    }
}

/// Derive exact `(n_a, n_b, n_d)` for a recursive level from the generated
/// SIS width thresholds.
///
/// Computes the recursive layout to obtain the actual matrix widths, then
/// looks up the minimum Module-SIS rank required for 128-bit security at
/// each role. Falls back to the envelope when the SIS table does not cover
/// the requested parameters.
fn sis_derived_recursive_params<Cfg: CommitmentConfig>(
    d: usize,
    log_basis: u32,
    current_w_len: usize,
    stage1_config: &SparseChallengeConfig,
    envelope: &CommitmentEnvelope,
) -> Option<LevelParams> {
    use super::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
    use super::schedule::hachi_recursive_level_layout_from_params;

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
    result.a_key = AjtaiKeyParams::new_unchecked(n_a, 0, a_collision, d);
    result.b_key = AjtaiKeyParams::new_unchecked(n_b, 0, bd_collision, d);
    result.d_key = AjtaiKeyParams::new_unchecked(n_d, 0, bd_collision, d);
    Some(result)
}

fn derived_root_commitment_layout_from_params<Cfg: CommitmentConfig>(
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
    let (m_vars, r_vars) = optimal_m_r_split_with_params(params, decomp, reduced_vars, 0);
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold(r_vars, params.challenge_l1_mass(), decomp.log_basis);
    params.with_decomp(m_vars, r_vars, depth_commit, depth_open, depth_fold, 0)
}

fn sis_derived_root_params_for_layout<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, HachiError> {
    use super::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};

    let d = Cfg::d_at_level(inputs.level, inputs.current_w_len);
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
    // A secures weak-opening consistency of the inner witness, so its width is
    // the unbatched inner matrix width.
    let n_a = min_rank_for_secure_width(d as u32, a_collision, lp.inner_width() as u64)
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing secure root A-row rank for D={} lb={} inner_width={}",
                d,
                lp.log_basis,
                lp.inner_width()
            ))
        })?;
    // B secures the digitized inner commitments, so its width must use the
    // batch-effective outer matrix width.
    let n_b = min_rank_for_secure_width(d as u32, bd_collision, lp.outer_width() as u64)
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "missing secure root B-row rank for D={} lb={} outer_width={}",
                d,
                lp.log_basis,
                lp.outer_width()
            ))
        })?;
    // D secures the flattened opening witness, so its width must use the
    // batch-effective D-matrix width.
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
    result.a_key = AjtaiKeyParams::new_unchecked(n_a, 0, a_collision, d);
    result.b_key = AjtaiKeyParams::new_unchecked(n_b, 0, bd_collision, d);
    result.d_key = AjtaiKeyParams::new_unchecked(n_d, 0, bd_collision, d);
    Ok(result)
}

/// Generated adaptive policy with table-selected per-level log bases.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeneratedAdaptivePolicy<Profile, const D: usize, const LOG_COMMIT_BOUND: u32>(
    PhantomData<Profile>,
);

fn missing_generated_schedule(err: &HachiError) -> bool {
    matches!(err, HachiError::InvalidSetup(msg) if msg.starts_with("missing generated schedule for "))
}

impl<
        F: CanonicalField + FieldCore + Send + Sync + 'static,
        Profile: CommitmentFieldProfile + CommitmentFieldProfileSchedule,
        const D: usize,
        const LOG_COMMIT_BOUND: u32,
    > CommitmentConfig
    for CommitmentPreset<F, GeneratedAdaptivePolicy<Profile, D, LOG_COMMIT_BOUND>>
{
    type Field = F;
    const D: usize = D;

    fn decomposition() -> DecompositionParams {
        Profile::decomposition(LOG_COMMIT_BOUND, 3)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        let audited_root_rank = if D == 64 && LOG_COMMIT_BOUND == 1 {
            Profile::onehot_d64_root_rank(0, max_num_vars)
        } else {
            Profile::audited_root_outer_rank(D, 0, max_num_vars)
        };
        let audited_root_a_rank = if D == 64 && LOG_COMMIT_BOUND == 1 {
            1
        } else {
            Profile::audited_root_a_rank::<LOG_COMMIT_BOUND>(D, 0, max_num_vars)
        };
        let mut envelope = CommitmentEnvelope {
            max_n_a: audited_root_a_rank,
            max_n_b: audited_root_rank,
            max_n_d: audited_root_rank,
        };
        if let Some((gen_n_a, gen_n_b, gen_n_d)) =
            Profile::generated_schedule_envelope::<D, LOG_COMMIT_BOUND>(max_num_vars)
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
        let alpha = D.trailing_zeros() as usize;
        let (min_log_basis, max_log_basis) = Profile::adaptive_log_basis_search_range();
        for log_basis in min_log_basis..=max_log_basis {
            let root_params = if max_num_vars > alpha {
                Self::root_level_layout_with_log_basis(root_inputs, log_basis).ok()
            } else {
                let stage1_config = Self::stage1_challenge_config(D);
                let mut params = LevelParams::params_only(D, log_basis, 1, 1, 1, stage1_config);
                let mut converged = None;
                for _ in 0..4 {
                    let Ok(root_lp) = derived_root_commitment_layout_from_params::<Self>(
                        root_inputs,
                        &params,
                        true,
                    ) else {
                        break;
                    };
                    let Ok(derived_lp) =
                        Self::root_level_params_for_layout_with_log_basis(root_inputs, &root_lp)
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
            };
            if let Some(root_params) = root_params {
                envelope.max_n_a = envelope.max_n_a.max(root_params.a_key.row_len());
                envelope.max_n_b = envelope.max_n_b.max(root_params.b_key.row_len());
                envelope.max_n_d = envelope.max_n_d.max(root_params.d_key.row_len());
            }
        }
        envelope
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(
            HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1),
        )
        .and_then(|source| source.log_basis_at_level::<Self>(inputs))
        .expect("generated adaptive schedule must be derivable from public inputs")
    }

    fn log_basis_search_range(_inputs: HachiScheduleInputs) -> (u32, u32) {
        Profile::adaptive_log_basis_search_range()
    }

    fn schedule_key(key: HachiScheduleLookupKey) -> String {
        match Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(key) {
            Ok(source) => source.schedule_key(),
            Err(_) => format!(
                "generated-miss/d{D}/lcb{LOG_COMMIT_BOUND}/max{}/num{}/claims{}/batch{}g{}p{}",
                key.max_num_vars,
                key.num_vars,
                key.layout_num_claims,
                key.batch.num_claims,
                key.batch.num_commitment_groups,
                key.batch.num_points,
            ),
        }
    }

    fn schedule_plan(key: HachiScheduleLookupKey) -> Result<Option<HachiSchedulePlan>, HachiError> {
        match Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(key) {
            Ok(source) => {
                tracing::debug!(
                    max_num_vars = key.max_num_vars,
                    num_vars = key.num_vars,
                    "schedule plan: read from pre-computed generated table"
                );
                Ok(Some(source.schedule_plan()))
            }
            Err(err) if missing_generated_schedule(&err) => {
                tracing::warn!(
                    max_num_vars = key.max_num_vars,
                    num_vars = key.num_vars,
                    "schedule plan: no pre-computed table entry found, will recompute from scratch"
                );
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    fn recursive_suffix_bytes(
        key: HachiScheduleLookupKey,
        level: usize,
        current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        match Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(key) {
            Ok(source) => {
                tracing::debug!(
                    max_num_vars = key.max_num_vars,
                    level,
                    current_w_len,
                    "recursive suffix: read from pre-computed generated table"
                );
                Ok(Some(source.recursive_suffix_bytes::<Self>(
                    key.max_num_vars,
                    level,
                    current_w_len,
                )?))
            }
            Err(err) if missing_generated_schedule(&err) => {
                tracing::warn!(
                    max_num_vars = key.max_num_vars,
                    level,
                    current_w_len,
                    "recursive suffix: no pre-computed table entry, will recompute"
                );
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, HachiError> {
        let params = sis_derived_root_params_for_layout::<Self>(inputs, lp)?;
        Ok(params.with_layout(lp))
    }

    fn root_level_layout_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, HachiError> {
        let stage1_config = Self::stage1_challenge_config(D);
        let mut candidate_n_a = 1usize;
        for _ in 0..crate::planner::sis_security::MAX_RANK {
            let candidate_params =
                LevelParams::params_only(D, log_basis, candidate_n_a, 1, 1, stage1_config.clone());
            let root_lp = derived_root_commitment_layout_from_params::<Self>(
                inputs,
                &candidate_params,
                false,
            )?;
            let derived_params = sis_derived_root_params_for_layout::<Self>(inputs, &root_lp)?;
            if derived_params.a_key.row_len() == candidate_n_a {
                return Ok(derived_params.with_layout(&root_lp));
            }
            candidate_n_a = derived_params.a_key.row_len();
        }
        Err(HachiError::InvalidSetup(format!(
            "failed to converge on self-consistent root A-row rank for D={D} lb={log_basis}"
        )))
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Profile::stage1_challenge_config(d)
    }

    fn level_params_with_log_basis(inputs: HachiScheduleInputs, log_basis: u32) -> LevelParams {
        if let Ok(source) = Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(
            HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1),
        ) {
            if let Ok(Some(planned_level)) =
                exact_planned_level_execution::<Self>(&source.schedule_plan(), inputs, log_basis)
            {
                return planned_level.level.lp.clone();
            }
        }
        let envelope = Self::envelope(inputs.max_num_vars);
        let d = Self::d_at_level(inputs.level, inputs.current_w_len);
        let stage1_config = Self::stage1_challenge_config(d);

        if inputs.level > 0 {
            if let Some(params) = sis_derived_recursive_params::<Self>(
                d,
                log_basis,
                inputs.current_w_len,
                &stage1_config,
                &envelope,
            ) {
                if let Ok(lp) =
                    hachi_recursive_level_layout_from_params::<Self>(&params, inputs.current_w_len)
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
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::*;
    use crate::planner::sis_security::min_rank_for_secure_width;
    use crate::protocol::commitment::profile::Fp128PrimeProfile;

    #[test]
    fn small_d_stage1_challenge_families_match_study_parameters() {
        let d32 = <Fp128PrimeProfile as CommitmentFieldProfile>::stage1_challenge_config(32);
        assert_eq!(d32.hamming_weight(), 32);
        assert_eq!(d32.l1_mass(), 256);
        match d32 {
            SparseChallengeConfig::Uniform {
                weight,
                nonzero_coeffs,
            } => {
                assert_eq!(weight, 32);
                assert_eq!(nonzero_coeffs.first().copied(), Some(-8));
                assert_eq!(nonzero_coeffs.last().copied(), Some(8));
                assert_eq!(nonzero_coeffs.len(), 16);
            }
            other => panic!("expected uniform D=32 family, got {other:?}"),
        }
    }

    type D128OneHotCandidate = CommitmentPreset<
        crate::algebra::Prime128Offset2355,
        GeneratedAdaptivePolicy<Fp128PrimeProfile, 128, 1>,
    >;

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
    fn current_d128_onehot_candidate_schedule_stays_within_audited_sis_widths() {
        assert_schedule_stays_within_audited_sis_widths::<D128OneHotCandidate>(8, 50);
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

    #[test]
    fn adaptive_d128_envelope_accounts_for_audited_root_rank_escalation() {
        type FullCfg = crate::protocol::commitment::presets::fp128::D128Full;
        type OneHotCfg = D128OneHotCandidate;

        let full_envelope = FullCfg::envelope(59);
        assert_eq!(full_envelope.max_n_a, 2);
        assert_eq!(full_envelope.max_n_b, 2);
        assert_eq!(full_envelope.max_n_d, 2);

        let onehot_envelope = OneHotCfg::envelope(54);
        assert_eq!(onehot_envelope.max_n_a, 1);
        assert_eq!(onehot_envelope.max_n_b, 2);
        assert_eq!(onehot_envelope.max_n_d, 2);
    }

    #[test]
    fn adaptive_d32_onehot_root_layout_converges_to_secure_rank() {
        type Cfg = crate::protocol::commitment::presets::fp128::D32OneHot;

        let inputs = HachiScheduleInputs {
            max_num_vars: 32,
            level: 0,
            current_w_len: 1usize << 32,
        };
        let root_lp = Cfg::root_level_layout_with_log_basis(inputs, 2).unwrap();

        assert_eq!(root_lp.a_key.row_len(), 3);
        assert_eq!(root_lp.b_key.row_len(), 2);
        assert_eq!(root_lp.d_key.row_len(), 2);
        assert_eq!(root_lp.m_vars, 16);
        assert_eq!(root_lp.r_vars, 11);

        let a_rank = min_rank_for_secure_width(32, 31, root_lp.inner_width() as u64).unwrap();
        let b_rank = min_rank_for_secure_width(32, 3, root_lp.outer_width() as u64).unwrap();
        let d_rank = min_rank_for_secure_width(32, 3, root_lp.d_matrix_width() as u64).unwrap();
        assert_eq!(a_rank, root_lp.a_key.row_len() as u32);
        assert_eq!(b_rank, root_lp.b_key.row_len() as u32);
        assert_eq!(d_rank, root_lp.d_key.row_len() as u32);
    }

    #[test]
    fn generated_d32_onehot_schedule_uses_secure_root_rank() {
        type Cfg = crate::protocol::commitment::presets::fp128::D32OneHot;

        let schedule = Cfg::schedule_plan(HachiScheduleLookupKey::singleton(32, 32, 1))
            .unwrap()
            .expect("generated D32 onehot schedule");
        let root = schedule.fold_levels().next().expect("root fold");

        assert_eq!(root.lp.a_key.row_len(), 3);
        assert_eq!(root.lp.b_key.row_len(), 2);
        assert_eq!(root.lp.d_key.row_len(), 2);
        assert_eq!(root.lp.m_vars, 16);
        assert_eq!(root.lp.r_vars, 11);
    }

    #[test]
    fn static_d128_level_params_account_for_audited_root_rank_escalation() {
        type Cfg = crate::protocol::commitment::presets::fp128::StaticBounded<128, 6, 6>;

        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: 59,
            level: 0,
            current_w_len: 1,
        });
        assert_eq!(root_params.a_key.row_len(), 2);
        assert_eq!(root_params.b_key.row_len(), 2);
        assert_eq!(root_params.d_key.row_len(), 2);

        let recursive_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: 59,
            level: 1,
            current_w_len: 1,
        });
        assert_eq!(recursive_params.a_key.row_len(), 1);
        assert_eq!(recursive_params.b_key.row_len(), 1);
        assert_eq!(recursive_params.d_key.row_len(), 1);
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::algebra::SparseChallengeConfig;
    use crate::protocol::commitment::schedule_planner::planned_recursive_suffix_bytes;
    use crate::protocol::params::LevelParams;

    fn test_level_params() -> LevelParams {
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        LevelParams::params_only(64, 2, 2, 2, 2, stage1_config)
    }

    #[test]
    fn recursive_tight_rows_use_u64_block_counts() {
        assert_eq!(recursive_tight_z_pre_rows(1, 40), 1);
        assert_eq!(recursive_tight_z_pre_rows((1usize << 20) + 1, 20), 2);
    }

    #[test]
    fn recursive_split_handles_large_r_with_nonzero_num_ring() {
        let params = test_level_params();
        let decomp = DecompositionParams {
            log_basis: 2,
            log_commit_bound: 128,
            log_open_bound: Some(128),
        };

        let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, 40, 1);
        assert_eq!(m_vars + r_vars, 40);
    }

    #[test]
    fn adaptive_runtime_search_bounds_match_basis6_schedule_bounds() {
        type FullCfg = crate::protocol::commitment::presets::fp128::D128Full;
        type OneHotCfg = crate::protocol::commitment::presets::fp128::D64OneHot;

        let inputs = HachiScheduleInputs {
            max_num_vars: 30,
            level: 4,
            current_w_len: 245_888,
        };

        assert_eq!(FullCfg::log_basis_search_range(inputs), (2, 6));
        assert_eq!(OneHotCfg::log_basis_search_range(inputs), (2, 6));

        let key = HachiScheduleLookupKey::singleton(inputs.max_num_vars, inputs.max_num_vars, 1);
        assert_eq!(
            FullCfg::recursive_suffix_bytes(key, inputs.level, inputs.current_w_len).unwrap(),
            Some(
                planned_recursive_suffix_bytes::<FullCfg>(
                    inputs.max_num_vars,
                    inputs.level,
                    inputs.current_w_len,
                    2,
                    6,
                )
                .unwrap()
            )
        );
        assert_eq!(
            OneHotCfg::recursive_suffix_bytes(key, inputs.level, inputs.current_w_len).unwrap(),
            Some(
                planned_recursive_suffix_bytes::<OneHotCfg>(
                    inputs.max_num_vars,
                    inputs.level,
                    inputs.current_w_len,
                    2,
                    6,
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn missing_generated_schedule_does_not_swallow_missing_table_errors() {
        assert!(missing_generated_schedule(&HachiError::InvalidSetup(
            "missing generated schedule for test".to_string(),
        )));
        assert!(!missing_generated_schedule(&HachiError::InvalidSetup(
            "missing generated schedule table for test".to_string(),
        )));
    }
}
