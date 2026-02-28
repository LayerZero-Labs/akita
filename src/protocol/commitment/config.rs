//! Configuration presets for ring-native commitment construction.

use super::utils::math::checked_pow2;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

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
    /// Width of inner matrix `A`.
    pub inner_width: usize,
    /// Width of outer matrix `B`.
    pub outer_width: usize,
    /// Width of prover matrix `D` (`delta * 2^r_vars`).
    pub d_matrix_width: usize,
}

impl HachiCommitmentLayout {
    /// Build a layout from `(m_vars, r_vars)` and static config constants.
    ///
    /// # Errors
    ///
    /// Returns an error when powers or derived widths overflow.
    pub fn new<Cfg: CommitmentConfig>(m_vars: usize, r_vars: usize) -> Result<Self, HachiError> {
        let num_blocks = checked_pow2(r_vars)?;
        let block_len = checked_pow2(m_vars)?;
        let inner_width = block_len
            .checked_mul(Cfg::DELTA)
            .ok_or_else(|| HachiError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = Cfg::N_A
            .checked_mul(Cfg::DELTA)
            .and_then(|x| x.checked_mul(num_blocks))
            .ok_or_else(|| HachiError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = Cfg::DELTA
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
        })
    }

    /// Total number of outer variables consumed by ring coefficients.
    pub fn outer_vars(&self) -> Result<usize, HachiError> {
        self.m_vars
            .checked_add(self.r_vars)
            .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))
    }

    /// Required polynomial variable count for this layout (`outer + alpha`).
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
        self.inner_width.serialize_with_mode(&mut writer, compress)?;
        self.outer_width.serialize_with_mode(&mut writer, compress)?;
        self.d_matrix_width.serialize_with_mode(&mut writer, compress)?;
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
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Parameter bundle for the ring-native commitment core (§4.1–§4.2).
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;
    /// Inner Ajtai matrix row count.
    const N_A: usize;
    /// Outer commitment matrix row count.
    const N_B: usize;
    /// Prover commitment matrix `D` row count (§4.2).
    const N_D: usize;
    /// Base-2 logarithm of gadget decomposition base.
    const LOG_BASIS: u32;
    /// Decomposition levels `delta`.
    const DELTA: usize;
    /// Decomposition levels for the folded witness `z` (`τ` in the paper).
    const TAU: usize;
    /// Hamming weight of sparse challenges (`ω` in the paper).
    const CHALLENGE_WEIGHT: usize;

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
        beta_linf_fold_bound(layout.r_vars, Self::CHALLENGE_WEIGHT, Self::LOG_BASIS)
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
    if Cfg::LOG_BASIS == 0 || Cfg::LOG_BASIS >= 128 {
        return Err(HachiError::InvalidSetup("invalid LOG_BASIS".to_string()));
    }
    if (Cfg::DELTA as u32).saturating_mul(Cfg::LOG_BASIS) > 128 {
        return Err(HachiError::InvalidSetup(
            "DELTA * LOG_BASIS must be <= 128".to_string(),
        ));
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

/// Ensure matrix shape matches expected dimensions.
///
/// # Errors
///
/// Returns an error if row count or row width mismatch.
pub(super) fn ensure_matrix_shape<T>(
    mat: &[Vec<T>],
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
) -> Result<(), HachiError> {
    if mat.len() != expected_rows {
        return Err(HachiError::InvalidSize {
            expected: expected_rows,
            actual: mat.len(),
        });
    }
    for (row_idx, row) in mat.iter().enumerate() {
        if row.len() != expected_cols {
            return Err(HachiError::InvalidSetup(format!(
                "{name} row {row_idx} has width {}, expected {expected_cols}",
                row.len()
            )));
        }
    }
    Ok(())
}

/// Small correctness-first config for tests and local benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmallTestCommitmentConfig;

impl CommitmentConfig for SmallTestCommitmentConfig {
    const D: usize = 16;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 9;
    const TAU: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2)
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
/// - Summing over `2^R` blocks gives:
///   `||z||_inf <= 2^R * ω * (b/2)`.
///
/// For this profile: `R=11`, `ω=19`, `b=16`, so
/// `β = 2^11 * 19 * 8 = 311_296`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductionFp128CommitmentConfig;

impl CommitmentConfig for ProductionFp128CommitmentConfig {
    const D: usize = 512;
    const N_A: usize = 1;
    const N_B: usize = 1;
    const N_D: usize = 1;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 32;
    const TAU: usize = 5;
    const CHALLENGE_WEIGHT: usize = 19;

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
        let m_vars = reduced_vars.min(11);
        let r_vars = reduced_vars - m_vars;
        HachiCommitmentLayout::new::<Self>(m_vars, r_vars)
    }
}
