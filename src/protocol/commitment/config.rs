//! Configuration presets for ring-native commitment construction.

use super::profile::{CommitmentFieldProfile, CommitmentFieldProfileSchedule};
use super::schedule::{
    exact_planned_level_execution, hachi_root_commitment_layout, HachiLevelParams,
    HachiScheduleInputs, HachiSchedulePlan,
};
use super::utils::math::checked_pow2;
use super::utils::norm::detect_field_modulus;
use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore};
use std::io::{Read, Write};
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
    params: &HachiLevelParams,
    decomp: DecompositionParams,
    reduced_vars: usize,
    num_ring: usize,
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
            0,
        )
    }

    /// Build a layout from explicit decomposition parameters (no config trait needed).
    ///
    /// When `num_ring > 0` (recursive levels), `block_len` is set to
    /// `ceil(num_ring / num_blocks)` instead of `2^m_vars`, giving tight
    /// z_pre sizing.  Pass `0` for root-level layouts where the polynomial
    /// fills the full `2^(m+r)` domain.
    ///
    /// # Errors
    ///
    /// Returns an error when parameters are invalid or derived widths overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_decomp(
        m_vars: usize,
        r_vars: usize,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
        log_basis: u32,
        num_ring: usize,
    ) -> Result<Self, HachiError> {
        if log_basis == 0 || log_basis >= 128 {
            return Err(HachiError::InvalidSetup("invalid log_basis".to_string()));
        }
        let num_blocks = checked_pow2(r_vars)?;
        let block_len = if num_ring > 0 {
            num_ring.div_ceil(num_blocks)
        } else {
            checked_pow2(m_vars)?
        };
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
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            m_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            r_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_blocks: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            block_len: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            inner_width: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            outer_width: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            d_matrix_width: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_digits_commit: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_digits_open: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_digits_fold: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            log_basis: usize::deserialize_with_mode(&mut reader, compress, validate, &())? as u32,
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
    /// Base field used by this config.
    type Field: CanonicalField + FieldCore;

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

    /// Exact field modulus used by this config's planner and runtime sizing.
    fn field_modulus() -> u128 {
        detect_field_modulus::<Self::Field>()
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

    /// Inclusive search range for adaptive recursive basis planning at one state.
    ///
    /// Static configs should return a singleton range containing the unique basis
    /// allowed at `inputs`.
    fn log_basis_search_range(inputs: HachiScheduleInputs) -> (u32, u32) {
        let basis = Self::log_basis_at_level(inputs);
        (basis, basis)
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
    fn planner_half_field_bound() -> u128 {
        Self::field_modulus() / 2
    }

    /// Deterministically derive the active params for one level from public inputs.
    fn level_params(inputs: HachiScheduleInputs) -> HachiLevelParams {
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
    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        if let Some(plan) = Self::schedule_plan(max_num_vars)? {
            if let Some(root_fold) = plan.fold_levels().next() {
                return Ok(root_fold.layout);
            }
        }
        let (_, layout) = hachi_root_commitment_layout::<Self>(max_num_vars)?;
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

/// Validate static config invariants and derive runtime dimensions.
///
/// # Errors
///
/// Returns an error when config constants are inconsistent or overflow.
pub(super) fn validate_and_derive_layout<
    F: CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
    const D: usize,
>(
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
    layout: HachiCommitmentLayout,
) -> Result<(), HachiError> {
    let alpha = D.trailing_zeros() as usize;
    let available_outer = max_num_vars.saturating_sub(alpha);
    let required_outer = layout.outer_vars()?;
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
            max_n_a: 8,
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

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2, &Self::decomposition())
    }

    fn schedule_plan(
        max_num_vars: usize,
    ) -> Result<Option<super::schedule::HachiSchedulePlan>, HachiError> {
        let root_layout = HachiCommitmentLayout::new::<Self>(4, 2, &Self::decomposition())?;
        Ok(Some(super::schedule::build_schedule_plan_from_config::<
            Self,
        >(max_num_vars, root_layout)?))
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

    fn schedule_key(_max_num_vars: usize) -> String {
        format!("static_v1_root{LOG_BASIS}_rec{W_LOG_BASIS}")
    }

    fn schedule_plan(
        max_num_vars: usize,
    ) -> Result<Option<super::schedule::HachiSchedulePlan>, crate::error::HachiError> {
        let (_, root_layout) = super::schedule::hachi_root_commitment_layout::<Self>(max_num_vars)?;
        Ok(Some(super::schedule::build_schedule_plan_from_config::<
            Self,
        >(max_num_vars, root_layout)?))
    }

    fn n_b_at_level(level: usize, max_num_vars: usize, _current_w_len: usize) -> usize {
        N_B.max(Profile::audited_root_outer_rank(D, level, max_num_vars))
    }

    fn n_d_at_level(level: usize, max_num_vars: usize, current_w_len: usize) -> usize {
        let _ = current_w_len;
        N_D.max(Profile::audited_root_outer_rank(D, level, max_num_vars))
    }

    fn level_params_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> HachiLevelParams {
        let d = Self::d_at_level(inputs.level, inputs.current_w_len);
        let stage1_config = Self::stage1_challenge_config(d);
        HachiLevelParams {
            d,
            log_basis,
            n_a: N_A.max(Profile::audited_root_a_rank::<LOG_COMMIT_BOUND>(
                D,
                inputs.level,
                inputs.max_num_vars,
            )),
            n_b: Self::n_b_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            n_d: Self::n_d_at_level(inputs.level, inputs.max_num_vars, inputs.current_w_len),
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config,
        }
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
) -> Option<HachiLevelParams> {
    use super::generated::sis_floor::{ceil_supported_collision, min_rank_for_secure_width};
    use super::schedule::hachi_recursive_level_layout_from_params;

    let tentative = HachiLevelParams {
        d,
        log_basis,
        n_a: envelope.max_n_a,
        n_b: 1,
        n_d: 1,
        challenge_l1_mass: stage1_config.l1_mass(),
        stage1_config: stage1_config.clone(),
    };
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&tentative, current_w_len).ok()?;

    let bd_collision = (1u32 << log_basis) - 1;
    let a_raw = bd_collision;
    let a_collision = ceil_supported_collision(d as u32, a_raw * stage1_config.max_abs_coeff())?;

    let n_a = min_rank_for_secure_width(d as u32, a_collision, layout.inner_width as u64)
        .unwrap_or(envelope.max_n_a);
    let exact_outer_width = n_a * layout.num_digits_open * layout.num_blocks;
    let n_b = min_rank_for_secure_width(d as u32, bd_collision, exact_outer_width as u64)
        .unwrap_or(envelope.max_n_b);
    let n_d = min_rank_for_secure_width(d as u32, bd_collision, layout.d_matrix_width as u64)
        .unwrap_or(envelope.max_n_d);

    Some(HachiLevelParams {
        d,
        log_basis,
        n_a,
        n_b,
        n_d,
        challenge_l1_mass: stage1_config.l1_mass(),
        stage1_config: stage1_config.clone(),
    })
}

/// Generated adaptive policy with table-selected per-level log bases.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeneratedAdaptivePolicy<Profile, const D: usize, const LOG_COMMIT_BOUND: u32>(
    PhantomData<Profile>,
);

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
        envelope
    }

    fn log_basis_at_level(inputs: HachiScheduleInputs) -> u32 {
        Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(inputs.max_num_vars)
            .and_then(|source| source.log_basis_at_level::<Self>(inputs))
            .expect("generated adaptive schedule must be derivable from public inputs")
    }

    fn log_basis_search_range(_inputs: HachiScheduleInputs) -> (u32, u32) {
        Profile::adaptive_log_basis_search_range()
    }

    fn schedule_key(max_num_vars: usize) -> String {
        Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(max_num_vars)
            .map(|source| source.schedule_key())
            .expect("generated adaptive schedule key must be derivable from public inputs")
    }

    fn schedule_plan(max_num_vars: usize) -> Result<Option<HachiSchedulePlan>, HachiError> {
        Ok(Some(
            Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(max_num_vars)?
                .schedule_plan(),
        ))
    }

    fn recursive_suffix_bytes(
        max_num_vars: usize,
        level: usize,
        current_w_len: usize,
    ) -> Result<Option<usize>, HachiError> {
        Ok(Some(
            Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(max_num_vars)?
                .recursive_suffix_bytes::<Self>(max_num_vars, level, current_w_len)?,
        ))
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Profile::stage1_challenge_config(d)
    }

    fn level_params_with_log_basis(
        inputs: HachiScheduleInputs,
        log_basis: u32,
    ) -> HachiLevelParams {
        if let Ok(source) =
            Profile::generated_schedule_source::<Self, D, LOG_COMMIT_BOUND>(inputs.max_num_vars)
        {
            if let Ok(Some(planned_level)) =
                exact_planned_level_execution::<Self>(&source.schedule_plan(), inputs, log_basis)
            {
                return planned_level.level.params;
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
                return params;
            }
        }

        HachiLevelParams {
            d,
            log_basis,
            n_a: envelope.max_n_a,
            n_b: envelope.max_n_b,
            n_d: envelope.max_n_d,
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config,
        }
    }
}

#[cfg(test)]
mod fp128_policy_tests {
    use super::*;
    use crate::protocol::commitment::profile::Fp128PrimeProfile;
    use hachi_planner::sis_security::min_rank_for_secure_width;

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
        crate::algebra::Prime128Offset275,
        GeneratedAdaptivePolicy<Fp128PrimeProfile, 128, 1>,
    >;

    fn assert_d128_schedule_stays_within_audited_sis_widths<Cfg: CommitmentConfig>(
        min_num_vars: usize,
        max_num_vars: usize,
    ) {
        assert_eq!(Cfg::D, 128, "helper only audits D=128 configs");

        let root_onehot = Cfg::decomposition().log_commit_bound == 1;
        for num_vars in min_num_vars..=max_num_vars {
            let plan = Cfg::schedule_plan(num_vars)
                .unwrap()
                .expect("audited D=128 config should have a schedule");

            for level in plan.fold_levels() {
                let raw_collision = if root_onehot && level.inputs.level == 0 {
                    2
                } else {
                    (1u32 << level.params.log_basis) - 1
                };

                let a_rank = min_rank_for_secure_width(
                    128,
                    raw_collision,
                    u64::try_from(level.layout.inner_width)
                        .expect("inner width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited A-row SIS width for num_vars={}, level={}, lb={}, width={}",
                        num_vars,
                        level.inputs.level,
                        level.params.log_basis,
                        level.layout.inner_width
                    )
                });
                assert!(
                    a_rank <= level.params.n_a as u32,
                    "A-row SIS audit failed for num_vars={}, level={}, lb={}, width={}, required_rank={}, actual_rank={}",
                    num_vars,
                    level.inputs.level,
                    level.params.log_basis,
                    level.layout.inner_width,
                    a_rank,
                    level.params.n_a,
                );

                let bd_collision = (1u32 << level.params.log_basis) - 1;
                let b_rank = min_rank_for_secure_width(
                    128,
                    bd_collision,
                    u64::try_from(level.layout.outer_width)
                        .expect("outer width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited B-row SIS width for num_vars={}, level={}, lb={}, width={}",
                        num_vars,
                        level.inputs.level,
                        level.params.log_basis,
                        level.layout.outer_width
                    )
                });
                assert!(
                    b_rank <= level.params.n_b as u32,
                    "B-row SIS audit failed for num_vars={}, level={}, lb={}, width={}, required_rank={}, actual_rank={}",
                    num_vars,
                    level.inputs.level,
                    level.params.log_basis,
                    level.layout.outer_width,
                    b_rank,
                    level.params.n_b,
                );

                let d_rank = min_rank_for_secure_width(
                    128,
                    bd_collision,
                    u64::try_from(level.layout.d_matrix_width)
                        .expect("d-matrix width should fit in u64"),
                )
                .unwrap_or_else(|| {
                    panic!(
                        "missing audited D-row SIS width for num_vars={}, level={}, lb={}, width={}",
                        num_vars,
                        level.inputs.level,
                        level.params.log_basis,
                        level.layout.d_matrix_width
                    )
                });
                assert!(
                    d_rank <= level.params.n_d as u32,
                    "D-row SIS audit failed for num_vars={}, level={}, lb={}, width={}, required_rank={}, actual_rank={}",
                    num_vars,
                    level.inputs.level,
                    level.params.log_basis,
                    level.layout.d_matrix_width,
                    d_rank,
                    level.params.n_d,
                );
            }
        }
    }

    #[test]
    fn current_d128_full_schedule_stays_within_audited_sis_widths() {
        type Cfg = crate::protocol::commitment::presets::fp128::D128Full;
        assert_d128_schedule_stays_within_audited_sis_widths::<Cfg>(8, 50);
    }

    #[test]
    fn current_d128_onehot_candidate_schedule_stays_within_audited_sis_widths() {
        assert_d128_schedule_stays_within_audited_sis_widths::<D128OneHotCandidate>(8, 50);
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
    fn static_d128_level_params_account_for_audited_root_rank_escalation() {
        type Cfg = crate::protocol::commitment::presets::fp128::StaticBounded<128, 6, 6>;

        let root_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: 59,
            level: 0,
            current_w_len: 1,
        });
        assert_eq!(root_params.n_a, 2);
        assert_eq!(root_params.n_b, 2);
        assert_eq!(root_params.n_d, 2);

        let recursive_params = Cfg::level_params(HachiScheduleInputs {
            max_num_vars: 59,
            level: 1,
            current_w_len: 1,
        });
        assert_eq!(recursive_params.n_a, 1);
        assert_eq!(recursive_params.n_b, 1);
        assert_eq!(recursive_params.n_d, 1);
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::algebra::SparseChallengeConfig;
    use crate::protocol::commitment::schedule_planner::planned_recursive_suffix_bytes;
    use crate::protocol::commitment::HachiLevelParams;

    fn test_level_params() -> HachiLevelParams {
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        HachiLevelParams {
            d: 64,
            log_basis: 2,
            n_a: 2,
            n_b: 2,
            n_d: 2,
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config,
        }
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

        assert_eq!(
            FullCfg::recursive_suffix_bytes(
                inputs.max_num_vars,
                inputs.level,
                inputs.current_w_len
            )
            .unwrap(),
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
            OneHotCfg::recursive_suffix_bytes(
                inputs.max_num_vars,
                inputs.level,
                inputs.current_w_len
            )
            .unwrap(),
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
}
