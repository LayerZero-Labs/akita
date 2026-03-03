//! Configuration presets for ring-native commitment construction.

use super::utils::math::checked_pow2;
use crate::algebra::ring::CyclotomicRing;
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
/// `β = 2^r_vars * challenge_weight * 2^(log_basis - 1)`.
/// Returns enough gadget levels to represent values up to `β`.
pub fn compute_num_digits_fold(r_vars: usize, challenge_weight: usize, log_basis: u32) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_weight == 0 {
        return compute_num_digits(128, log_basis);
    }
    let beta = (challenge_weight as u128).saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    compute_num_digits(log_beta, log_basis)
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
    /// Width of outer matrix `B` (`n_a * num_digits_commit * num_blocks`).
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
    /// Build a layout from `(m_vars, r_vars)`, config constants, and decomposition
    /// parameters.
    ///
    /// `num_digits_fold` (τ) is auto-derived from the beta bound
    /// (`r_vars`, `challenge_weight`, `log_basis`).
    ///
    /// # Errors
    ///
    /// Returns an error when powers or derived widths overflow.
    pub fn new<Cfg: CommitmentConfig>(
        m_vars: usize,
        r_vars: usize,
        decomp: &DecompositionParams,
    ) -> Result<Self, HachiError> {
        let depth_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis);
        let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
        let depth_open = compute_num_digits(open_bound, decomp.log_basis);
        let depth_fold = compute_num_digits_fold(r_vars, Cfg::CHALLENGE_WEIGHT, decomp.log_basis);
        Self::new_with_decomp(
            m_vars,
            r_vars,
            Cfg::N_A,
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
            .checked_mul(num_digits_commit)
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
/// Security parameters (`N_A`, `N_B`, `N_D`, `CHALLENGE_WEIGHT`) are
/// compile-time constants fixed for a given security level. Decomposition
/// parameters (gadget depths, `log_basis`) are runtime values derived from
/// [`DecompositionParams`] and live in [`HachiCommitmentLayout`].
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;
    /// Inner Ajtai matrix row count.
    const N_A: usize;
    /// Outer commitment matrix row count.
    const N_B: usize;
    /// Prover commitment matrix `D` row count (§4.2).
    const N_D: usize;
    /// Hamming weight of sparse challenges (`ω` in the paper).
    const CHALLENGE_WEIGHT: usize;

    /// Decomposition parameters (gadget base and coefficient bounds).
    fn decomposition() -> DecompositionParams;

    /// Choose the runtime commitment layout for `max_num_vars`.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_num_vars` does not admit a valid layout.
    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError>;

    /// Runtime L∞ bound for `z` (`β`) used by stage-1 folding checks.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid parameters or arithmetic overflow.
    fn beta_bound(layout: HachiCommitmentLayout) -> Result<u128, HachiError> {
        beta_linf_fold_bound(layout.r_vars, Self::CHALLENGE_WEIGHT, layout.log_basis)
    }
}

/// Deterministic upper bound for the stage-1 folded-witness infinity norm.
///
/// This encodes the bound used in `QuadraticEquation::compute_z_hat`:
/// `||z||_inf <= 2^R * ω * (b/2)` where `b = 2^LOG_BASIS`.
///
/// # Errors
///
/// Returns an error when parameters are out of range or intermediate products
/// overflow `u128`.
pub(crate) fn beta_linf_fold_bound(
    r: usize,
    challenge_weight: usize,
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
        .checked_mul(challenge_weight as u128)
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
/// Matrices may be wider than the main layout requires when widened to
/// accommodate the w-commitment's column counts.
///
/// # Errors
///
/// Returns an error if row count mismatches or any row is too narrow.
pub(super) fn ensure_matrix_shape_ge<T>(
    mat: &[Vec<T>],
    expected_rows: usize,
    min_cols: usize,
    name: &str,
) -> Result<(), HachiError> {
    if mat.len() != expected_rows {
        return Err(HachiError::InvalidSize {
            expected: expected_rows,
            actual: mat.len(),
        });
    }
    for (row_idx, row) in mat.iter().enumerate() {
        if row.len() < min_cols {
            return Err(HachiError::InvalidSetup(format!(
                "{name} row {row_idx} has width {}, expected >= {min_cols}",
                row.len()
            )));
        }
    }
    Ok(())
}

/// Small correctness-first config for tests and local benchmarks.
///
/// Fixed layout (m_vars=4, r_vars=2) for fast test iteration. For larger
/// polynomials, use [`DynamicSmallTestCommitmentConfig`] instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmallTestCommitmentConfig;

impl CommitmentConfig for SmallTestCommitmentConfig {
    const D: usize = 16;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
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
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = Self::D.trailing_zeros() as usize;
        let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let r_vars = reduced_vars / 2;
        let m_vars = reduced_vars - r_vars;
        HachiCommitmentLayout::new::<Self>(m_vars, r_vars, &Self::decomposition())
    }
}

/// Production-oriented profile for 128-bit base fields (`Fp128<P>`).
///
/// This profile targets the `D = 512`, `n_A = n_B = n_D = 1` regime with
/// base-16 decomposition over ~128-bit moduli.
///
/// Rigorous β derivation for the stage-1 folded witness `z`:
/// - In `compute_z_hat`, each coordinate is `z[j] = Σ_i s_i[j].mul_by_sparse(c_i)`.
/// - `balanced_decompose_pow2` yields per-coefficient digits in `[-b/2, b/2)` where
///   `b = 2^LOG_BASIS`, so each input coefficient has `|·| <= b/2`.
/// - Challenges use exactly `ω = CHALLENGE_WEIGHT` nonzeros in `{±1}`.
/// - Therefore each `mul_by_sparse` output coefficient is a signed sum of `ω`
///   shifted digits, hence bounded by `ω * (b/2)`.
/// - Summing over `2^R` blocks (R = r_vars) gives:
///   `||z||_inf <= 2^R * ω * (b/2)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128CommitmentConfig;

impl CommitmentConfig for Fp128CommitmentConfig {
    const D: usize = 512;
    const N_A: usize = 1;
    const N_B: usize = 1;
    const N_D: usize = 1;
    const CHALLENGE_WEIGHT: usize = 19;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: None,
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = Self::D.trailing_zeros() as usize;
        let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let r_vars = reduced_vars / 2;
        let m_vars = reduced_vars - r_vars;
        HachiCommitmentLayout::new::<Self>(m_vars, r_vars, &Self::decomposition())
    }
}
