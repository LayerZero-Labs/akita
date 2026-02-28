//! Configuration presets for ring-native commitment construction.

use super::utils::math::checked_pow2;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::FieldCore;

/// Parameter bundle for the ring-native commitment core (§4.1–§4.2).
pub trait CommitmentConfig: Clone + Send + Sync + 'static {
    /// Ring degree used by `CyclotomicRing<F, D>`.
    const D: usize;
    /// Number of variables inside each committed block (`2^M` entries).
    const M: usize;
    /// Number of block-select variables (`2^R` blocks).
    const R: usize;
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
    /// L∞ norm bound for `z` (`β` in the paper). Prover aborts if exceeded.
    const BETA: u128;
    /// Hamming weight of sparse challenges (`ω` in the paper).
    const CHALLENGE_WEIGHT: usize;
}

/// Deterministic upper bound for the stage-1 folded-witness infinity norm.
///
/// This encodes the bound used in `QuadraticEquation::compute_z_hat`:
/// `||z||_inf <= 2^R * ω * (b/2)` where `b = 2^LOG_BASIS`.
///
/// # Panics
///
/// Panics when `log_basis` or `r` are out of range, or when intermediate
/// products overflow `u128`.
pub(super) const fn beta_linf_fold_bound(
    r: usize,
    challenge_weight: usize,
    log_basis: u32,
) -> u128 {
    assert!(log_basis > 0 && log_basis < 128, "invalid LOG_BASIS");
    assert!(r < 128, "R must be < 128");

    let blocks = 1u128 << r;
    let b = 1u128 << log_basis;
    let half_b = b / 2;

    let term = match blocks.checked_mul(challenge_weight as u128) {
        Some(v) => v,
        None => panic!("beta bound overflow (blocks * challenge_weight)"),
    };
    match term.checked_mul(half_b) {
        Some(v) => v,
        None => panic!("beta bound overflow (term * half_b)"),
    }
}

/// Runtime-derived dimensions from a `CommitmentConfig`.
#[derive(Debug, Clone, Copy)]
pub(super) struct CommitmentLayout {
    /// Number of committed blocks (`2^R`).
    pub(super) num_blocks: usize,
    /// Number of ring elements per block (`2^M`).
    pub(super) block_len: usize,
    /// Width of inner matrix `A`.
    pub(super) inner_width: usize,
    /// Width of outer matrix `B`.
    pub(super) outer_width: usize,
    /// Width of prover matrix `D` (`δ · 2^R`).
    pub(super) d_matrix_width: usize,
    /// Minimum variable count supported by this config.
    pub(super) required_vars: usize,
}

/// Validate static config invariants and derive runtime dimensions.
///
/// # Errors
///
/// Returns an error when config constants are inconsistent or overflow.
pub(super) fn validate_and_derive_layout<Cfg: CommitmentConfig, const D: usize>(
) -> Result<CommitmentLayout, HachiError> {
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

    let num_blocks = checked_pow2(Cfg::R)?;
    let block_len = checked_pow2(Cfg::M)?;
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
    let required_vars = Cfg::M
        .checked_add(Cfg::R)
        .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))?;

    Ok(CommitmentLayout {
        num_blocks,
        block_len,
        inner_width,
        outer_width,
        d_matrix_width,
        required_vars,
    })
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
    layout: CommitmentLayout,
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
    const M: usize = 4;
    const R: usize = 2;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 9;
    const TAU: usize = 4;
    const BETA: u128 = 1_000_000;
    const CHALLENGE_WEIGHT: usize = 3;
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
    const M: usize = 11;
    const R: usize = 11;
    const N_A: usize = 1;
    const N_B: usize = 1;
    const N_D: usize = 1;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 32;
    const TAU: usize = 5;
    const BETA: u128 = beta_linf_fold_bound(Self::R, Self::CHALLENGE_WEIGHT, Self::LOG_BASIS);
    const CHALLENGE_WEIGHT: usize = 19;
}
