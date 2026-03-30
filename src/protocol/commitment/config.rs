//! Configuration presets for ring-native commitment construction.

use super::schedule::{
    hachi_root_level_layout, planned_log_basis_at_level, planned_recursive_suffix_bytes,
    planned_schedule_key, HachiLevelParams, HachiScheduleInputs, HachiSchedulePlan,
};
use super::utils::math::checked_pow2;
use super::utils::norm::detect_field_modulus;
use crate::algebra::ring::CyclotomicRing;
use crate::algebra::{Prime128Offset5823, SparseChallengeConfig};
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

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

/// Compute the decomposition depth for the folded witness `z_pre`
/// (τ in the paper).
///
/// The folded witness satisfies `||z_pre||_inf <= β` where
/// `β = 2^r_vars * challenge_l1_mass * 2^(log_basis - 1)`.
/// Returns enough gadget levels to represent values up to `β`.
pub fn compute_num_digits_fold(r_vars: usize, challenge_l1_mass: usize, log_basis: u32) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_l1_mass == 0 {
        return compute_num_digits(128, log_basis);
    }
    let beta = (challenge_l1_mass as u128).saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    compute_num_digits(log_beta, log_basis)
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
pub(super) fn optimal_m_r_split_with_params(
    params: &HachiLevelParams,
    decomp: DecompositionParams,
    reduced_vars: usize,
) -> (usize, usize) {
    // Guard: for S >= 53, shifts could overflow u64. Fall back to balanced
    // split (this threshold is far beyond any practical polynomial size).
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let delta_open = compute_num_digits(open_bound, decomp.log_basis) as u64;
    let delta_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis) as u64;
    let c1 = delta_open + params.n_a as u64 * delta_open;

    let mut best_r = reduced_vars / 2;
    let mut best_cost = u64::MAX;

    for r in 1..reduced_vars {
        let m = reduced_vars - r;
        let delta_fold =
            compute_num_digits_fold(r, params.challenge_l1_mass, decomp.log_basis) as u64;
        let cost = c1 * (1u64 << r) + delta_commit * delta_fold * (1u64 << m);
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
    optimal_m_r_split_with_params(&params, Cfg::decomposition(), reduced_vars)
}

fn uniform_pm1_stage1_challenge(weight: usize) -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight,
        nonzero_coeffs: vec![-1, 1],
    }
}

fn d64_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    assert_eq!(d, 64, "d64_stage1_challenge_config requires d=64, got {d}");
    SparseChallengeConfig::SplitRing {
        half_weight: 21,
        max_mag2_per_half: 6,
    }
}

fn d128_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    assert_eq!(
        d, 128,
        "d128_stage1_challenge_config requires d=128, got {d}"
    );
    uniform_pm1_stage1_challenge(31)
}

fn fp128_half_field_bound() -> u128 {
    detect_field_modulus::<Prime128Offset5823>() / 2
}

/// Runtime commitment layout authority for ring-native commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiCommitmentLayout {
    /// Number of variables inside each committed block (`2^m_vars` entries).
    pub m_vars: usize,
    /// Number of block-select variables (`2^r_vars` blocks).
    pub r_vars: usize,
    /// Number of committed blocks (`2^r_vars`).
    pub num_blocks: usize,
    /// Number of ring elements per block (`2^m_vars`).
    pub block_len: usize,
    /// Width of inner matrix `A` (`block_len * num_digits_commit`).
    pub inner_width: usize,
    /// Width of outer matrix `B` (`n_a * num_digits_open * num_blocks`).
    pub outer_width: usize,
    /// Width of prover matrix `D` (`num_digits_open * num_blocks`).
    pub d_matrix_width: usize,
    /// Number of gadget decomposition levels for commitment-time coefficients
    /// (δ_commit in the paper). Controls how the original polynomial
    /// coefficients are decomposed into balanced base-b digits for the Ajtai
    /// commitment.
    pub num_digits_commit: usize,
    /// Number of gadget decomposition levels for opening-time folded
    /// evaluations (δ_open in the paper). Folding inner-products with
    /// arbitrary field-element weights produces full-field-size coefficients,
    /// so this equals `num_digits_commit` when `log_commit_bound` covers
    /// the full field, and is larger otherwise (e.g. recursive w witnesses).
    pub num_digits_open: usize,
    /// Number of gadget decomposition levels for the folded witness `z_pre`
    /// (τ in the paper). Derived from the L∞ bound on `z_pre`.
    pub num_digits_fold: usize,
    /// Base-2 logarithm of gadget decomposition base.
    pub log_basis: u32,
}

impl HachiCommitmentLayout {
    /// Build a layout from `(m_vars, r_vars)` and a config's runtime-derived
    /// level parameters.
    ///
    /// `num_digits_fold` (τ) is auto-derived from the folded-witness beta bound
    /// (`r_vars`, `challenge_l1_mass`, `log_basis`).
    ///
    /// # Errors
    ///
    /// Returns an error when powers or derived widths overflow.
    pub fn new<Cfg: CommitmentConfig>(
        m_vars: usize,
        r_vars: usize,
        decomp: &DecompositionParams,
    ) -> Result<Self, HachiError> {
        let alpha = Cfg::D.trailing_zeros() as usize;
        let max_num_vars = m_vars
            .checked_add(r_vars)
            .and_then(|vars| vars.checked_add(alpha))
            .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))?;
        let current_w_len = 1usize.checked_shl(max_num_vars as u32).unwrap_or(0);
        let params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len,
        });
        let depth_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = compute_num_digits(open_bound, decomp.log_basis);
        let depth_fold =
            compute_num_digits_fold(r_vars, params.challenge_l1_mass, decomp.log_basis);
        Self::new_with_decomp(
            m_vars,
            r_vars,
            params.n_a,
            depth_commit,
            depth_open,
            depth_fold,
            decomp.log_basis,
        )
    }

    /// Build a layout from explicit decomposition parameters (no config trait needed).
    ///
    /// # Errors
    ///
    /// Returns an error when parameters are invalid or derived widths overflow.
    pub fn new_with_decomp(
        m_vars: usize,
        r_vars: usize,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
        log_basis: u32,
    ) -> Result<Self, HachiError> {
        if log_basis == 0 || log_basis >= 128 {
            return Err(HachiError::InvalidSetup("invalid log_basis".to_string()));
        }
        let num_blocks = checked_pow2(r_vars)?;
        let block_len = checked_pow2(m_vars)?;
        let inner_width = block_len
            .checked_mul(num_digits_commit)
            .ok_or_else(|| HachiError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = n_a
            .checked_mul(num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks))
            .ok_or_else(|| HachiError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = num_digits_open
            .checked_mul(num_blocks)
            .ok_or_else(|| HachiError::InvalidSetup("D-matrix width overflow".to_string()))?;
        Ok(Self {
            m_vars,
            r_vars,
            num_blocks,
            block_len,
            inner_width,
            outer_width,
            d_matrix_width,
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
            log_basis,
        })
    }

    /// Total number of outer variables consumed by ring coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error if the variable count overflows.
    pub fn outer_vars(&self) -> Result<usize, HachiError> {
        self.m_vars
            .checked_add(self.r_vars)
            .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))
    }

    /// Required polynomial variable count for this layout (`outer + alpha`).
    ///
    /// # Errors
    ///
    /// Returns an error if the variable count overflows.
    pub fn required_num_vars<const D: usize>(&self) -> Result<usize, HachiError> {
        let alpha = D.trailing_zeros() as usize;
        self.outer_vars()?
            .checked_add(alpha)
            .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))
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

impl Valid for HachiCommitmentLayout {
    fn check(&self) -> Result<(), SerializationError> {
        if self.num_blocks == 0 || self.block_len == 0 {
            return Err(SerializationError::InvalidData(
                "invalid zero block layout".to_string(),
            ));
        }
        Ok(())
    }
}

impl HachiSerialize for HachiCommitmentLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.m_vars.serialize_with_mode(&mut writer, compress)?;
        self.r_vars.serialize_with_mode(&mut writer, compress)?;
        self.num_blocks.serialize_with_mode(&mut writer, compress)?;
        self.block_len.serialize_with_mode(&mut writer, compress)?;
        self.inner_width
            .serialize_with_mode(&mut writer, compress)?;
        self.outer_width
            .serialize_with_mode(&mut writer, compress)?;
        self.d_matrix_width
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_commit
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_open
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_fold
            .serialize_with_mode(&mut writer, compress)?;
        (self.log_basis as usize).serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.m_vars.serialized_size(compress)
            + self.r_vars.serialized_size(compress)
            + self.num_blocks.serialized_size(compress)
            + self.block_len.serialized_size(compress)
            + self.inner_width.serialized_size(compress)
            + self.outer_width.serialized_size(compress)
            + self.d_matrix_width.serialized_size(compress)
            + self.num_digits_commit.serialized_size(compress)
            + self.num_digits_open.serialized_size(compress)
            + self.num_digits_fold.serialized_size(compress)
            + (self.log_basis as usize).serialized_size(compress)
    }
}

impl HachiDeserialize for HachiCommitmentLayout {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            m_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            r_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_blocks: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            block_len: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            inner_width: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            outer_width: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            d_matrix_width: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_digits_commit: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_digits_open: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_digits_fold: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            log_basis: usize::deserialize_with_mode(&mut reader, compress, validate)? as u32,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Parameter bundle for the ring-native commitment core (§4.1–§4.2).
///
/// Ring degree `D` remains compile time because `CyclotomicRing<F, D>` is
/// array-backed, but the active Ajtai ranks and sparse-challenge family are
/// runtime values derived from public schedule inputs. Setup allocates against a
/// per-config row envelope, while each level uses the exact rows selected by
/// [`HachiLevelParams`].
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Maximum matrix row counts needed by this config for the given setup size.
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope;

    /// Stable identifier for setup-cache versioning and fixture selection.
    fn family_key() -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Build one level's active params from public inputs and an explicit basis.
    fn level_params_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> HachiLevelParams {
        let d = Self::d_at_level(inputs.level, inputs.current_w_len);
        let stage1_config = Self::stage1_challenge_config(d);
        HachiLevelParams {
            d,
            log_basis,
            n_a: Self::n_a_at_level(inputs.level),
            n_b: Self::n_b_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            n_d: Self::n_d_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config,
        }
    }

    /// Deterministically choose the active basis for one level from public inputs.
    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        let _ = inputs;
        Self::decomposition().log_basis
    }

    /// Stable identity for the active log-basis schedule at `max_num_vars`.
    fn schedule_key(_max_num_vars: usize) -> String {
        format!("static_v1_b{}", Self::decomposition().log_basis)
    }

    /// Optional full schedule plan for configs with an explicit planner.
    ///
    /// `None` means callers should fall back to the legacy local stop heuristic.
    ///
    /// # Errors
    ///
    /// Returns an error when the config's planner cannot derive a valid
    /// schedule from the public inputs.
    fn schedule_plan(_max_num_vars: usize) -> Result<Option<HachiSchedulePlan>, HachiError> {
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
        _max_num_vars: usize,
        _level: usize,
        _current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        Ok(None)
    }

    /// Half-range bound used by the planner when sizing recursive `r`.
    ///
    /// By default this uses the decomposition bit bound as a conservative
    /// power-of-two proxy. Configs with a known concrete field modulus should
    /// override this with the exact centered range used at runtime.
    fn planner_half_field_bound() -> u128 {
        let decomp = Self::decomposition();
        let field_bits = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        1u128 << field_bits.saturating_sub(1)
    }

    /// Deterministically derive the active params for one level from public inputs.
    fn level_params(inputs: HachiScheduleInputs) -> HachiLevelParams {
        let log_basis = Self::log_basis_at_level(inputs);
        Self::level_params_with_log_basis(inputs, log_basis)
    }

    /// Choose the runtime commitment layout for `max_num_vars`.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_num_vars` does not admit a valid layout.
    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let (_, layout) = hachi_root_level_layout::<Self>(max_num_vars)?;
        Ok(layout)
    }

    /// Runtime L∞ bound for `z` (`β`) used by stage-1 folding checks.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid parameters or arithmetic overflow.
    fn beta_bound(layout: HachiCommitmentLayout) -> Result<u128, HachiError> {
        beta_linf_fold_bound(
            layout.r_vars,
            Self::stage1_challenge_config(Self::D).l1_mass(),
            layout.log_basis,
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

/// Validate static config invariants and derive runtime dimensions.
///
/// # Errors
///
/// Returns an error when config constants are inconsistent or overflow.
pub(super) fn validate_and_derive_layout<Cfg: CommitmentConfig, const D: usize>(
    max_num_vars: usize,
) -> Result<HachiCommitmentLayout, HachiError> {
    if D != Cfg::D {
        return Err(HachiError::InvalidSetup(format!(
            "const D={D} mismatches config D={}",
            Cfg::D
        )));
    }
    Cfg::commitment_layout(max_num_vars)
}

/// Ensure `max_num_vars` is sufficient for config dimensions.
///
/// # Errors
///
/// Returns an error when `max_num_vars < required_vars`.
pub(super) fn ensure_supported_num_vars(
    max_num_vars: usize,
    required_vars: usize,
) -> Result<(), HachiError> {
    if max_num_vars < required_vars {
        return Err(HachiError::InvalidSetup(format!(
            "max_num_vars {max_num_vars} is smaller than required {required_vars}"
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
    layout: HachiCommitmentLayout,
) -> Result<(), HachiError> {
    if f_blocks.len() != layout.num_blocks {
        return Err(HachiError::InvalidSize {
            expected: layout.num_blocks,
            actual: f_blocks.len(),
        });
    }
    for block in f_blocks {
        if block.len() != layout.block_len {
            return Err(HachiError::InvalidSize {
                expected: layout.block_len,
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
/// Fixed layout (m_vars=4, r_vars=2) for fast test iteration. For larger
/// polynomials, use [`DynamicSmallTestCommitmentConfig`] instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmallTestCommitmentConfig;

impl CommitmentConfig for SmallTestCommitmentConfig {
    const D: usize = 16;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 8,
            max_n_b: 4,
            max_n_d: 4,
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        assert_eq!(d, Self::D, "unsupported ring dim {d}");
        uniform_pm1_stage1_challenge(3)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2, &Self::decomposition())
    }
}

/// D=16 config with dynamic layout that adapts to polynomial size.
///
/// Uses the same D=16 ring dimension as [`SmallTestCommitmentConfig`] but
/// derives `m_vars`/`r_vars` from `max_num_vars`, so it can commit
/// polynomials with an arbitrary number of variables.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicSmallTestCommitmentConfig;

impl CommitmentConfig for DynamicSmallTestCommitmentConfig {
    const D: usize = 16;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 8,
            max_n_b: 4,
            max_n_d: 4,
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        assert_eq!(d, Self::D, "unsupported ring dim {d}");
        uniform_pm1_stage1_challenge(3)
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let (_, layout) = hachi_root_level_layout::<Self>(max_num_vars)?;
        Ok(layout)
    }
}

/// Production-oriented profile for 128-bit base fields (`Fp128<P>`),
/// parameterized by the coefficient bound and gadget basis.
///
/// This profile targets the `D = 128`, `n_A = n_B = n_D = 1` regime with
/// balanced decomposition over ~128-bit moduli.
///
/// - `LOG_COMMIT_BOUND`: bit-width of the largest polynomial coefficient the
///   commitment decomposition must represent.
/// - `LOG_BASIS`: base-2 log of the gadget base at level 0.
/// # Aliases
///
/// - [`Fp128FullCommitmentConfig`] = adaptive full-field preset over `D = 128`
/// - [`Fp128LogBasisCommitmentConfig`] = adaptive log-bounded preset over `D = 128`
/// - [`Fp128OneHotCommitmentConfig`] = adaptive `D = 64` onehot preset
/// - [`Fp128CommitmentConfig`] — alias for `Fp128FullCommitmentConfig`
///
/// # β derivation (stage-1 folded witness `z`)
///
/// `||z||_inf <= 2^R * ω * (b/2)` where `b = 2^LOG_BASIS`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128BoundedCommitmentConfig<
    const LOG_COMMIT_BOUND: u32,
    const LOG_BASIS: u32,
    const W_LOG_BASIS: u32 = LOG_BASIS,
>;

impl<const LOG_COMMIT_BOUND: u32, const LOG_BASIS: u32, const W_LOG_BASIS: u32> CommitmentConfig
    for Fp128BoundedCommitmentConfig<LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS>
{
    const D: usize = 128;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: LOG_BASIS,
            log_commit_bound: LOG_COMMIT_BOUND,
            log_open_bound: if LOG_COMMIT_BOUND < 128 {
                Some(128)
            } else {
                None
            },
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        d128_stage1_challenge_config(d)
    }

    fn planner_half_field_bound() -> u128 {
        fp128_half_field_bound()
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        if inputs.level == 0 {
            LOG_BASIS
        } else {
            W_LOG_BASIS
        }
    }

    fn schedule_key(_max_num_vars: usize) -> String {
        format!("static_v1_root{LOG_BASIS}_rec{W_LOG_BASIS}")
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let (_, layout) = hachi_root_level_layout::<Self>(max_num_vars)?;
        Ok(layout)
    }
}

/// D=64, rank-1 everywhere.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128D64BoundedCommitmentConfig<
    const LOG_COMMIT_BOUND: u32,
    const LOG_BASIS: u32,
    const W_LOG_BASIS: u32 = LOG_BASIS,
>;

impl<const LOG_COMMIT_BOUND: u32, const LOG_BASIS: u32, const W_LOG_BASIS: u32> CommitmentConfig
    for Fp128D64BoundedCommitmentConfig<LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS>
{
    const D: usize = 64;

    fn decomposition() -> DecompositionParams {
        Fp128BoundedCommitmentConfig::<LOG_COMMIT_BOUND, LOG_BASIS, W_LOG_BASIS>::decomposition()
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let (_, layout) = hachi_root_level_layout::<Self>(max_num_vars)?;
        Ok(layout)
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        if inputs.level == 0 {
            LOG_BASIS
        } else {
            W_LOG_BASIS
        }
    }

    fn schedule_key(_max_num_vars: usize) -> String {
        format!("static_v1_root{LOG_BASIS}_rec{W_LOG_BASIS}")
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        d64_stage1_challenge_config(d)
    }

    fn planner_half_field_bound() -> u128 {
        fp128_half_field_bound()
    }
}

/// Adaptive `D=128`, rank-1 family that chooses the level basis by proof bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128AdaptiveBoundedCommitmentConfig<const LOG_COMMIT_BOUND: u32>;

impl<const LOG_COMMIT_BOUND: u32> CommitmentConfig
    for Fp128AdaptiveBoundedCommitmentConfig<LOG_COMMIT_BOUND>
{
    const D: usize = 128;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: LOG_COMMIT_BOUND,
            log_open_bound: if LOG_COMMIT_BOUND < 128 {
                Some(128)
            } else {
                None
            },
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        d128_stage1_challenge_config(d)
    }

    fn planner_half_field_bound() -> u128 {
        fp128_half_field_bound()
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        planned_log_basis_at_level::<Self>(inputs, 2, 5)
            .expect("adaptive schedule must be derivable from public inputs")
    }

    fn schedule_key(max_num_vars: usize) -> String {
        planned_schedule_key::<Self>(max_num_vars, 2, 5)
            .expect("adaptive schedule key must be derivable from public inputs")
    }

    fn schedule_plan(max_num_vars: usize) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Ok(Some(super::schedule::planned_schedule::<Self>(
            max_num_vars,
            2,
            5,
        )?))
    }

    fn recursive_suffix_bytes(
        max_num_vars: usize,
        level: usize,
        current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        Ok(Some(planned_recursive_suffix_bytes::<Self>(
            max_num_vars,
            level,
            current_w_len,
            2,
            5,
        )?))
    }
}

/// Full-field (128-bit) coefficient family with adaptive per-level basis.
pub type Fp128FullCommitmentConfig = Fp128AdaptiveBoundedCommitmentConfig<128>;

/// Binary (1-bit) D=64 onehot preset with the coarse adaptive outer-rank
/// schedule.
pub type Fp128OneHotCommitmentConfig = Fp128AdaptiveOneHotCommitmentConfig;

/// Log-bounded coefficient family with adaptive per-level basis.
pub type Fp128LogBasisCommitmentConfig = Fp128AdaptiveBoundedCommitmentConfig<3>;

/// Alias for [`Fp128FullCommitmentConfig`].
pub type Fp128CommitmentConfig = Fp128FullCommitmentConfig;

/// D=64 onehot preset with the coarse adaptive outer-rank schedule from the
/// current local planning note: rank-2 only in the short early window.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128AdaptiveOneHotCommitmentConfig;

impl CommitmentConfig for Fp128AdaptiveOneHotCommitmentConfig {
    const D: usize = 64;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 1,
            log_open_bound: Some(128),
        }
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        let root_rank = usize::from(max_num_vars >= 38) + 1;
        CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: root_rank,
            max_n_d: root_rank,
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let (_, layout) = hachi_root_level_layout::<Self>(max_num_vars)?;
        Ok(layout)
    }

    fn n_b_at_level(level: usize, max_num_vars: usize, _current_w_len: usize) -> usize {
        if max_num_vars >= 38 {
            if level == 0 {
                2
            } else {
                1
            }
        } else {
            1
        }
    }

    fn n_d_at_level(level: usize, max_num_vars: usize, current_w_len: usize) -> usize {
        Self::n_b_at_level(level, max_num_vars, current_w_len)
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        d64_stage1_challenge_config(d)
    }

    fn planner_half_field_bound() -> u128 {
        fp128_half_field_bound()
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        planned_log_basis_at_level::<Self>(inputs, 2, 5)
            .expect("adaptive schedule must be derivable from public inputs")
    }

    fn schedule_key(max_num_vars: usize) -> String {
        planned_schedule_key::<Self>(max_num_vars, 2, 5)
            .expect("adaptive schedule key must be derivable from public inputs")
    }

    fn schedule_plan(max_num_vars: usize) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Ok(Some(super::schedule::planned_schedule::<Self>(
            max_num_vars,
            2,
            5,
        )?))
    }

    fn recursive_suffix_bytes(
        max_num_vars: usize,
        level: usize,
        current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        Ok(Some(planned_recursive_suffix_bytes::<Self>(
            max_num_vars,
            level,
            current_w_len,
            2,
            5,
        )?))
    }
}
