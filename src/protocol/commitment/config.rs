//! Configuration presets for ring-native commitment construction.

use super::utils::math::checked_pow2;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::FieldCore;

/// Parameter bundle for the ring-native §4.1 commitment core.
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
    /// Base-2 logarithm of gadget decomposition base.
    const LOG_BASIS: u32;
    /// Decomposition levels `delta`.
    const DELTA: usize;
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
    let required_vars = Cfg::M
        .checked_add(Cfg::R)
        .ok_or_else(|| HachiError::InvalidSetup("variable count overflow".to_string()))?;

    Ok(CommitmentLayout {
        num_blocks,
        block_len,
        inner_width,
        outer_width,
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

/// Default correctness-first config for early protocol integration.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultCommitmentConfig;

impl CommitmentConfig for DefaultCommitmentConfig {
    const D: usize = 64;
    const M: usize = 4;
    const R: usize = 2;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 8;
}
