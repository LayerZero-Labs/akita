//! Unified per-level parameters for the Hachi protocol.
//!
//! `LevelParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use crate::algebra::ring::sparse_challenge::SparseChallengeConfig;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};

/// Parameters for a single Ajtai commitment matrix.
///
/// Each matrix in the protocol (A, B, D) is characterised by its row count
/// (security rank), column count (message width), and the worst-case L∞
/// collision bound used for SIS security sizing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    row_len: usize,
    col_len: usize,
    collision_inf: u32,
}

impl Valid for AjtaiKeyParams {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiSerialize for AjtaiKeyParams {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.row_len.serialize_with_mode(&mut writer, compress)?;
        self.col_len.serialize_with_mode(&mut writer, compress)?;
        self.collision_inf
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.row_len.serialized_size(compress)
            + self.col_len.serialized_size(compress)
            + self.collision_inf.serialized_size(compress)
    }
}

impl HachiDeserialize for AjtaiKeyParams {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            row_len: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            col_len: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            collision_inf: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl AjtaiKeyParams {
    /// Create a new `AjtaiKeyParams` with SIS security enforcement.
    ///
    /// # Panics
    ///
    /// Panics if `row_len` is below the 128-bit SIS security floor for the
    /// given `(ring_dimension, collision_inf, col_len)` triple.
    pub fn new(row_len: usize, col_len: usize, collision_inf: u32, ring_dimension: usize) -> Self {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::protocol::commitment::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) =
                min_rank_for_secure_width(ring_dimension as u32, collision_inf, col_len as u64)
            {
                assert!(
                    row_len >= floor,
                    "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                     (d={ring_dimension}, collision_inf={collision_inf}, col_len={col_len})"
                );
            }
        }
        Self {
            row_len,
            col_len,
            collision_inf,
        }
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Logs a warning if `row_len` is below the SIS floor but does not
    /// panic. Use this for intermediate construction steps where ranks
    /// have not yet converged (e.g., batched scaling, iterative SIS
    /// fixed-point loops).
    pub fn new_unchecked(
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Self {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::protocol::commitment::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) =
                min_rank_for_secure_width(ring_dimension as u32, collision_inf, col_len as u64)
            {
                if row_len < floor {
                    tracing::warn!(
                        row_len,
                        floor,
                        ring_dimension,
                        collision_inf,
                        col_len,
                        "AjtaiKeyParams::new_unchecked: row_len below SIS floor"
                    );
                }
            }
        }
        Self {
            row_len,
            col_len,
            collision_inf,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Number of columns.
    #[inline]
    pub fn col_len(&self) -> usize {
        self.col_len
    }

    /// Worst-case L∞ collision bound for SIS security sizing.
    #[inline]
    pub fn collision_inf(&self) -> u32 {
        self.collision_inf
    }
}

/// Unified per-level parameters for one Hachi recursion level.
///
/// Combines ring dimension, Ajtai matrix descriptions, block geometry,
/// sparse-challenge configuration, and digit decomposition depths into a
/// single authoritative struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelParams {
    /// Ring dimension (`d` in the protocol).
    pub ring_dimension: usize,
    /// Base-2 logarithm of the gadget decomposition base.
    pub log_basis: u32,
    /// Inner Ajtai matrix (A): `row_len = n_a`, `col_len = inner_width`.
    pub a_key: AjtaiKeyParams,
    /// Outer commitment matrix (B): `row_len = n_b`, `col_len = outer_width`.
    pub b_key: AjtaiKeyParams,
    /// Prover matrix (D): `row_len = n_d`, `col_len = d_matrix_width`.
    pub d_key: AjtaiKeyParams,
    /// Number of committed blocks (`2^r_vars`).
    pub num_blocks: usize,
    /// Number of ring elements per block. Equals `2^m_vars` at the root level
    /// but may differ at recursive levels (`ceil(num_ring / num_blocks)`).
    pub block_len: usize,
    /// Block-select variable count (log₂ `num_blocks`). Stored explicitly
    /// because `num_blocks.trailing_zeros()` suffices only when `num_blocks`
    /// is a power of two, which is always true by construction.
    pub m_vars: usize,
    /// Per-block variable count. Stored explicitly because at recursive
    /// levels `block_len` is not necessarily `2^r_vars`.
    pub r_vars: usize,
    /// Stage-1 sparse challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
    /// Gadget decomposition depth for commitment coefficients (δ_commit).
    pub num_digits_commit: usize,
    /// Gadget decomposition depth for opening evaluations (δ_open).
    pub num_digits_open: usize,
    /// Gadget decomposition depth for the folded witness (δ_fold / τ).
    pub num_digits_fold: usize,
}

impl Valid for LevelParams {
    fn check(&self) -> Result<(), SerializationError> {
        self.a_key.check()?;
        self.b_key.check()?;
        self.d_key.check()?;
        self.stage1_config.check()?;
        if self.ring_dimension == 0 {
            return Err(SerializationError::InvalidData(
                "LevelParams::ring_dimension must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl HachiSerialize for LevelParams {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.ring_dimension
            .serialize_with_mode(&mut writer, compress)?;
        self.log_basis.serialize_with_mode(&mut writer, compress)?;
        self.a_key.serialize_with_mode(&mut writer, compress)?;
        self.b_key.serialize_with_mode(&mut writer, compress)?;
        self.d_key.serialize_with_mode(&mut writer, compress)?;
        self.num_blocks.serialize_with_mode(&mut writer, compress)?;
        self.block_len.serialize_with_mode(&mut writer, compress)?;
        self.m_vars.serialize_with_mode(&mut writer, compress)?;
        self.r_vars.serialize_with_mode(&mut writer, compress)?;
        self.stage1_config
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_commit
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_open
            .serialize_with_mode(&mut writer, compress)?;
        self.num_digits_fold
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ring_dimension.serialized_size(compress)
            + self.log_basis.serialized_size(compress)
            + self.a_key.serialized_size(compress)
            + self.b_key.serialized_size(compress)
            + self.d_key.serialized_size(compress)
            + self.num_blocks.serialized_size(compress)
            + self.block_len.serialized_size(compress)
            + self.m_vars.serialized_size(compress)
            + self.r_vars.serialized_size(compress)
            + self.stage1_config.serialized_size(compress)
            + self.num_digits_commit.serialized_size(compress)
            + self.num_digits_open.serialized_size(compress)
            + self.num_digits_fold.serialized_size(compress)
    }
}

impl HachiDeserialize for LevelParams {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            ring_dimension: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            log_basis: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
            a_key: AjtaiKeyParams::deserialize_with_mode(&mut reader, compress, validate, &())?,
            b_key: AjtaiKeyParams::deserialize_with_mode(&mut reader, compress, validate, &())?,
            d_key: AjtaiKeyParams::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_blocks: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            block_len: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            m_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            r_vars: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            stage1_config: SparseChallengeConfig::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
            num_digits_commit: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_digits_open: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            num_digits_fold: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl LevelParams {
    /// Build a params-only `LevelParams` with zeroed layout fields.
    ///
    /// Only ring dimension, matrix row counts, log_basis, and stage1_config
    /// are populated. Column counts, block geometry, and digit depths are
    /// zeroed. Call `with_layout` to fill them from a derived layout.
    pub fn params_only(
        ring_dimension: usize,
        log_basis: u32,
        n_a: usize,
        n_b: usize,
        n_d: usize,
        stage1_config: SparseChallengeConfig,
    ) -> Self {
        Self {
            ring_dimension,
            log_basis,
            a_key: AjtaiKeyParams {
                row_len: n_a,
                ..Default::default()
            },
            b_key: AjtaiKeyParams {
                row_len: n_b,
                ..Default::default()
            },
            d_key: AjtaiKeyParams {
                row_len: n_d,
                ..Default::default()
            },
            num_blocks: 0,
            block_len: 0,
            m_vars: 0,
            r_vars: 0,
            stage1_config,
            num_digits_commit: 0,
            num_digits_open: 0,
            num_digits_fold: 0,
        }
    }

    /// Worst-case L1 mass of the sparse challenge, derived from `stage1_config`.
    #[inline]
    pub fn challenge_l1_mass(&self) -> usize {
        self.stage1_config.l1_mass()
    }

    /// Block-select variable count (the `r_vars` of the legacy layout).
    #[inline]
    pub fn log_num_blocks(&self) -> usize {
        self.r_vars
    }

    /// Per-block variable count (the `m_vars` of the legacy layout).
    #[inline]
    pub fn log_block_len(&self) -> usize {
        self.m_vars
    }

    /// Width of inner matrix A (column count of the A-key).
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.a_key.col_len()
    }

    /// Width of outer matrix B (column count of the B-key).
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.b_key.col_len()
    }

    /// Width of prover matrix D (column count of the D-key).
    #[inline]
    pub fn d_matrix_width(&self) -> usize {
        self.d_key.col_len()
    }

    /// Total outer variable count (`log_num_blocks + log_block_len`).
    #[inline]
    pub fn outer_vars(&self) -> usize {
        self.log_num_blocks() + self.log_block_len()
    }

    /// Total quotient / relation rows in M (single-claim, single-group).
    #[inline]
    pub fn m_row_count(&self) -> usize {
        self.m_row_count_with_public_outputs(1)
    }

    /// Row count when the root carries `num_public_outputs` public y-rows.
    #[inline]
    pub fn m_row_count_with_public_outputs(&self, num_public_outputs: usize) -> usize {
        self.d_key.row_len() + self.b_key.row_len() + num_public_outputs + 1 + self.a_key.row_len()
    }

    /// Row count with `num_commitments` explicit commitment vectors and
    /// `num_public_outputs` public y-rows.
    #[inline]
    pub fn m_row_count_with_commitments_and_public_outputs(
        &self,
        num_commitments: usize,
        num_public_outputs: usize,
    ) -> usize {
        self.d_key.row_len()
            + self.b_key.row_len() * num_commitments
            + num_public_outputs
            + 1
            + self.a_key.row_len()
    }

    /// Root-batched row count where each claim keeps its own commitment vector.
    #[inline]
    pub fn batched_root_m_row_count(&self, num_claims: usize) -> usize {
        self.m_row_count_with_commitments_and_public_outputs(num_claims, num_claims)
    }

    /// Fill in the layout-derived fields from explicit decomposition parameters.
    ///
    /// Takes a params-only `LevelParams` (with zeroed layout fields) and
    /// computes block geometry, matrix column counts, and digit depths.
    ///
    /// When `num_ring > 0` (recursive levels), `block_len` is set to
    /// `ceil(num_ring / num_blocks)` instead of `2^m_vars`, giving tight
    /// z_pre sizing. Pass `0` for root-level layouts.
    ///
    /// # Errors
    ///
    /// Returns an error when parameters are invalid or derived widths overflow.
    pub fn with_decomp(
        &self,
        m_vars: usize,
        r_vars: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        num_digits_fold: usize,
        num_ring: usize,
    ) -> Result<Self, HachiError> {
        let num_blocks = 1usize
            .checked_shl(r_vars as u32)
            .ok_or_else(|| HachiError::InvalidSetup("2^r_vars does not fit usize".to_string()))?;
        let block_len = if num_ring > 0 {
            num_ring.div_ceil(num_blocks)
        } else {
            1usize.checked_shl(m_vars as u32).ok_or_else(|| {
                HachiError::InvalidSetup("2^m_vars does not fit usize".to_string())
            })?
        };
        let inner_width = block_len
            .checked_mul(num_digits_commit)
            .ok_or_else(|| HachiError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = self
            .a_key
            .row_len()
            .checked_mul(num_digits_open)
            .and_then(|x| x.checked_mul(num_blocks))
            .ok_or_else(|| HachiError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = num_digits_open
            .checked_mul(num_blocks)
            .ok_or_else(|| HachiError::InvalidSetup("D-matrix width overflow".to_string()))?;
        let d = self.ring_dimension;
        Ok(Self {
            ring_dimension: d,
            log_basis: self.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.row_len,
                inner_width,
                self.a_key.collision_inf,
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.row_len,
                outer_width,
                self.b_key.collision_inf,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.row_len,
                d_matrix_width,
                self.d_key.collision_inf,
                d,
            ),
            num_blocks,
            block_len,
            m_vars,
            r_vars,
            stage1_config: self.stage1_config.clone(),
            num_digits_commit,
            num_digits_open,
            num_digits_fold,
        })
    }

    /// Build a new `LevelParams` that keeps rank/ring info from `self` but
    /// replaces all layout-derived fields with those from `other`.
    pub fn with_layout(&self, other: &LevelParams) -> Self {
        let d = self.ring_dimension;
        Self {
            ring_dimension: d,
            log_basis: other.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.row_len,
                other.a_key.col_len,
                other.a_key.collision_inf,
                d,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                self.b_key.row_len,
                other.b_key.col_len,
                other.b_key.collision_inf,
                d,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                self.d_key.row_len,
                other.d_key.col_len,
                other.d_key.collision_inf,
                d,
            ),
            num_blocks: other.num_blocks,
            block_len: other.block_len,
            m_vars: other.m_vars,
            r_vars: other.r_vars,
            stage1_config: self.stage1_config.clone(),
            num_digits_commit: other.num_digits_commit,
            num_digits_open: other.num_digits_open,
            num_digits_fold: other.num_digits_fold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
    }

    fn sample_layout_lp() -> LevelParams {
        sample_params_only().with_decomp(4, 2, 2, 2, 3, 0).unwrap()
    }

    #[test]
    fn with_layout_keeps_self_ranks() {
        let params = sample_params_only();
        let layout_lp = sample_layout_lp();

        let lp = params.with_layout(&layout_lp);

        assert_eq!(lp.ring_dimension, 64);
        assert_eq!(lp.log_basis, layout_lp.log_basis);
        assert_eq!(lp.a_key.row_len(), 2);
        assert_eq!(lp.b_key.row_len(), 4);
        assert_eq!(lp.d_key.row_len(), 3);
        assert_eq!(lp.num_blocks, layout_lp.num_blocks);
        assert_eq!(lp.block_len, layout_lp.block_len);
        assert_eq!(lp.challenge_l1_mass(), 3);
        assert_eq!(lp.num_digits_commit, layout_lp.num_digits_commit);
        assert_eq!(lp.num_digits_open, layout_lp.num_digits_open);
        assert_eq!(lp.num_digits_fold, layout_lp.num_digits_fold);
    }

    #[test]
    fn derived_widths_match_ajtai_col_len() {
        let lp = sample_params_only().with_layout(&sample_layout_lp());

        assert_eq!(lp.inner_width(), lp.a_key.col_len());
        assert_eq!(lp.outer_width(), lp.b_key.col_len());
        assert_eq!(lp.d_matrix_width(), lp.d_key.col_len());
    }

    #[test]
    fn derived_log_values() {
        let layout_lp = sample_layout_lp();
        let lp = sample_params_only().with_layout(&layout_lp);

        assert_eq!(lp.log_num_blocks(), layout_lp.r_vars);
        assert_eq!(lp.log_block_len(), layout_lp.m_vars);
        assert_eq!(lp.outer_vars(), layout_lp.m_vars + layout_lp.r_vars);
    }

    #[test]
    fn m_row_count_values() {
        let lp = sample_params_only().with_layout(&sample_layout_lp());

        assert_eq!(lp.m_row_count(), 3 + 4 + 1 + 1 + 2);
        assert_eq!(lp.m_row_count_with_public_outputs(3), 3 + 4 + 3 + 1 + 2);
        assert_eq!(
            lp.m_row_count_with_commitments_and_public_outputs(2, 5),
            3 + 4 * 2 + 5 + 1 + 2
        );
        assert_eq!(
            lp.batched_root_m_row_count(4),
            lp.m_row_count_with_commitments_and_public_outputs(4, 4)
        );
    }
}
