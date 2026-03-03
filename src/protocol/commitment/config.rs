//! Configuration presets for ring-native commitment construction.

use super::utils::math::checked_pow2;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

/// Parameters controlling the gadget decomposition depth.
///
/// The gadget base is `b = 2^log_basis`. Each ring coefficient with centered
/// magnitude fitting in `log_coeff_bound` bits is decomposed into
/// `delta = ceil(log_coeff_bound / log_basis)` balanced digits in `[-b/2, b/2)`.
///
/// Smaller `log_coeff_bound` (when polynomial coefficients are known to be
/// small) yields a smaller `delta`, which proportionally shrinks the witness
/// vector, the commitment matrices, and the proving cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompositionParams {
    /// Base-2 logarithm of the gadget base (e.g., 4 for base-16 digits in [-8, 7]).
    pub log_basis: u32,

    /// Base-2 logarithm of the maximum centered coefficient magnitude.
    ///
    /// Determines the decomposition depth: `delta = ceil(log_coeff_bound / log_basis)`.
    /// The centered representation maps each coefficient `c ∈ [0, q)` to the
    /// signed value in `(-q/2, q/2]`. A value of `k` means the signed magnitude
    /// fits in `k` bits, i.e., lies in `[-2^(k-1), 2^(k-1) - 1]`.
    ///
    /// Examples:
    /// - Binary (0/1) polynomials: 1
    /// - Already range-checked digits in `[-8, 7]`: 4  (= `log_basis` for one digit)
    /// - Arbitrary Fp128 elements: 128
    pub log_coeff_bound: u32,
}

/// Compute the gadget decomposition depth from a coefficient bound.
///
/// Returns `delta = ceil(log_coeff_bound / log_basis)`, with an extra level
/// when the balanced-digit range would not cover the full bound.
///
/// # Panics
///
/// Panics if `log_basis` is 0 or >= 128.
pub fn compute_delta(log_coeff_bound: u32, log_basis: u32) -> usize {
    assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
    if log_coeff_bound == 0 {
        return 1;
    }
    let mut delta = (log_coeff_bound as usize).div_ceil(log_basis as usize);

    // Verify the balanced-digit range covers the bound.
    // Max positive representable by delta digits in base b: (b/2-1)*(b^delta-1)/(b-1).
    let b: u128 = 1u128 << log_basis;
    let half_b_minus_1 = b / 2 - 1;
    let b_minus_1 = b - 1;
    let mut b_pow = 1u128;
    for _ in 0..delta {
        b_pow = b_pow.saturating_mul(b);
    }
    let max_positive = half_b_minus_1.saturating_mul(b_pow.saturating_sub(1) / b_minus_1);
    let required = if log_coeff_bound > 128 {
        u128::MAX / 2
    } else if log_coeff_bound == 0 {
        0
    } else {
        (1u128 << (log_coeff_bound - 1)).saturating_sub(1)
    };
    if max_positive < required {
        delta += 1;
    }
    delta.max(1)
}

/// Compute the decomposition depth `tau` for the folded witness `z_pre`.
///
/// The folded witness satisfies `||z_pre||_inf <= beta` where
/// `beta = 2^r_vars * challenge_weight * 2^(log_basis - 1)`.
/// Returns enough levels to represent values up to `beta`.
pub fn compute_tau(r_vars: usize, challenge_weight: usize, log_basis: u32) -> usize {
    let shift = r_vars + (log_basis as usize) - 1;
    if shift >= 127 || challenge_weight == 0 {
        return compute_delta(128, log_basis);
    }
    let beta = (challenge_weight as u128).saturating_mul(1u128 << shift);
    if beta == 0 {
        return 1;
    }
    let log_beta = 128 - beta.leading_zeros();
    compute_delta(log_beta, log_basis)
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
    /// Width of inner matrix `A`.
    pub inner_width: usize,
    /// Width of outer matrix `B`.
    pub outer_width: usize,
    /// Width of prover matrix `D` (`delta * 2^r_vars`).
    pub d_matrix_width: usize,
    /// Decomposition levels `delta` (gadget decomposition depth).
    pub delta: usize,
    /// Decomposition levels for the folded witness `z` (`tau` in the paper).
    pub tau: usize,
    /// Base-2 logarithm of gadget decomposition base.
    pub log_basis: u32,
}

impl HachiCommitmentLayout {
    /// Build a layout from `(m_vars, r_vars)`, config constants, and decomposition
    /// parameters.
    ///
    /// `tau` is auto-derived from the beta bound (`r_vars`, `challenge_weight`,
    /// `log_basis`).
    ///
    /// # Errors
    ///
    /// Returns an error when powers or derived widths overflow.
    pub fn new<Cfg: CommitmentConfig>(
        m_vars: usize,
        r_vars: usize,
        decomp: &DecompositionParams,
    ) -> Result<Self, HachiError> {
        let delta = compute_delta(decomp.log_coeff_bound, decomp.log_basis);
        let tau = compute_tau(r_vars, Cfg::CHALLENGE_WEIGHT, decomp.log_basis);
        Self::new_with_decomp(m_vars, r_vars, Cfg::N_A, delta, tau, decomp.log_basis)
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
        delta: usize,
        tau: usize,
        log_basis: u32,
    ) -> Result<Self, HachiError> {
        if log_basis == 0 || log_basis >= 128 {
            return Err(HachiError::InvalidSetup("invalid log_basis".to_string()));
        }
        let num_blocks = checked_pow2(r_vars)?;
        let block_len = checked_pow2(m_vars)?;
        let inner_width = block_len
            .checked_mul(delta)
            .ok_or_else(|| HachiError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = n_a
            .checked_mul(delta)
            .and_then(|x| x.checked_mul(num_blocks))
            .ok_or_else(|| HachiError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = delta
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
            delta,
            tau,
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
        self.delta.serialize_with_mode(&mut writer, compress)?;
        self.tau.serialize_with_mode(&mut writer, compress)?;
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
            + self.delta.serialized_size(compress)
            + self.tau.serialized_size(compress)
            + self.delta.serialized_size(compress) // log_basis as usize
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
            delta: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            tau: usize::deserialize_with_mode(&mut reader, compress, validate)?,
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
/// parameters (`delta`, `tau`, `log_basis`) are runtime values derived from
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

    /// Decomposition parameters (gadget base and coefficient bound).
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
            log_basis: 4,
            log_coeff_bound: 32,
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
            log_basis: 4,
            log_coeff_bound: 32,
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
pub struct ProductionFp128CommitmentConfig;

impl CommitmentConfig for ProductionFp128CommitmentConfig {
    const D: usize = 512;
    const N_A: usize = 1;
    const N_B: usize = 1;
    const N_D: usize = 1;
    const CHALLENGE_WEIGHT: usize = 19;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 4,
            log_coeff_bound: 128,
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
